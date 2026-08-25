from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import isaaclab.utils.math as math_utils

@dataclass
class FossenForceModels:
    num_envs: int
    device: torch.device
    dt: float | torch.Tensor
    debug: bool = False

    def __post_init__(self):
        self.added_mass = torch.tensor(
            [5.5, 12.7, 14.57, 0.12, 0.12, 0.12], device=self.device
        )
        self.linear_damping = torch.tensor(
            [4.03, 6.22, 5.18, 0.07, 0.07, 0.07], device=self.device
        )
        self.quadratic_damping = torch.tensor(
            [18.18, 21.66, 36.99, 1.55, 1.55, 1.55], device=self.device
        )
        
        self._prev_linvel_w = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float
        )
        self._prev_angvel_w = torch.zeros(
            self.num_envs, 3, device=self.device, dtype=torch.float
        )
        self._reset_mask = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._nu_dot_filt = torch.zeros(
            self.num_envs, 6, device=self.device, dtype=torch.float
        )

        self._has_reset = False
        self._has_prev = False
        self.dt = torch.as_tensor(self.dt, device=self.device, dtype=torch.float)
        self.dt = self.dt if self.dt.numel() == 1 else self.dt.flatten()[0]
        print("FossenForceModels initialized: ")
        print(f"  added_mass: {self.added_mass}")
        print(f"  linear_damping: {self.linear_damping}")
        print(f"  quadratic_damping: {self.quadratic_damping}")
        print(f"  dt: {self.dt}")
        self._step = 0

    def calculate_buoyancy_forces(
        self,
        root_quats_w: torch.tensor,  # robot orientations in world frame
        fluid_density: float,  # fluid density
        volumes: torch.tensor,  # rigid body volume
        g_mag: float,  # magnitude of gravity
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Compute wrenches (forces and torques) due to buoyancy on fully-submerged rigid body in fluid.
        Returned forces and torques are in the body root frame COB with FRD coordinates.
        Note that gravity is applied by Isaac Sim by default.
        """
        buoyancy_directions_w = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        # opposing gravity vector in the world frame
        buoyancy_directions_w[..., 2] = 1.0
        # FRD Body Frame
        buoyancy_directions_b = math_utils.quat_apply_inverse(
            root_quats_w, buoyancy_directions_w
        )
        # print("B: ", buoyancy_directions_b)
        
        buoyancy_forces_b = (
            buoyancy_directions_b * fluid_density * volumes.repeat(1, 3) * g_mag
        )
        # torque = r x F
        buoyancy_torques_b = torch.zeros_like(buoyancy_forces_b)

        if self.debug and False:
            print(f"Buoyancy F: {buoyancy_forces_b}, T: {buoyancy_torques_b}")

        return (buoyancy_forces_b, buoyancy_torques_b)

    def _coriolis_added(self, nu: torch.Tensor) -> torch.Tensor:
        # C_A(nu) for diagonal added-mass matrix.
        m = self.added_mass
        u, v, w, p, q, r = [nu[:, i] for i in range(6)]
        N = nu.shape[0]
        c = torch.zeros((N, 6, 6), device=self.device, dtype=nu.dtype)

        c[:, 0, 4] = m[2] * w
        c[:, 0, 5] = -m[1] * v
        c[:, 1, 3] = -m[2] * w
        c[:, 1, 5] = m[0] * u
        c[:, 2, 3] = m[1] * v
        c[:, 2, 4] = -m[0] * u
        c[:, 3, 1] = m[2] * w
        c[:, 3, 2] = -m[1] * v
        c[:, 3, 4] = m[5] * r
        c[:, 3, 5] = -m[4] * q
        c[:, 4, 0] = -m[2] * w
        c[:, 4, 2] = m[0] * u
        c[:, 4, 3] = -m[5] * r
        c[:, 4, 5] = m[3] * p
        c[:, 5, 0] = m[1] * v
        c[:, 5, 1] = -m[0] * u
        c[:, 5, 3] = m[4] * q
        c[:, 5, 4] = -m[3] * p
        return c

    def calculate_density_and_viscosity_forces(
        self,
        root_quats_w: torch.Tensor,
        root_linvels_b: torch.Tensor,
        root_angvels_b: torch.Tensor,
        inertias: torch.Tensor,
        water_beta: float,
        fluid_density: float,
        masses: torch.tensor,
        com_positions_b: torch.Tensor,
        current_velocities_w: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        'Compute forces for body-frame velocity ``nu`` with shape ``(num_envs, 6)``.'
        if current_velocities_w is not None:
            current_velocities_w = current_velocities_w.reshape(root_linvels_b.shape[0], 3)
            current_velocities_b = math_utils.quat_apply_inverse(
                root_quats_w, current_velocities_w
            )
            rel_linvels_b = root_linvels_b - current_velocities_b
        else:
            rel_linvels_b = root_linvels_b

        # n,6
        nu = torch.cat([rel_linvels_b, root_angvels_b], dim=-1)
        nu_dot = self._calculate_acc(
            root_quats_w,
            root_linvels_b,
            root_angvels_b,
            current_velocities_w=current_velocities_w,
            alpha=0.2,
        )

        lin_accel_limit = 2.0
        ang_accel_limit = 20.0
        nu_dot[:, :3].clamp_(-lin_accel_limit, lin_accel_limit)
        nu_dot[:, 3:].clamp_(-ang_accel_limit, ang_accel_limit)
        
        lin = -self.linear_damping * nu
        quad = -self.quadratic_damping * torch.abs(nu) * nu
        
        damping = lin + quad
        
        added_mass_term = -self.added_mass * nu_dot
        
        c_a = self._coriolis_added(nu)

        test_ca = torch.bmm(
            nu.unsqueeze(1), torch.bmm(c_a, nu.unsqueeze(-1))  # (N,1,6)  # (N,6,1)
        ).squeeze()
        if self.debug and (self._step % 200 == 0):
            print("nu:", nu)
            print("nu_dot:", nu_dot)
            print("damping:", damping)
            print("added_mass_term:", added_mass_term)
            print(
                "TEST CA mean/max:", test_ca.mean().item(), test_ca.abs().max().item()
            )
        
        coriolis_term = -torch.bmm(c_a, nu.unsqueeze(-1)).squeeze(-1)

        hydro_dynamic = damping + added_mass_term + coriolis_term
        self._step += 1
        return (hydro_dynamic[:, :3], hydro_dynamic[:, 3:])

    def _calculate_acc(
        self,
        root_quats_w: torch.Tensor,  # (N,4) Body->World
        root_linvels_b: torch.Tensor,  # (N,3) current body
        root_angvels_b: torch.Tensor,  # (N,3) current body
        current_velocities_w: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        # current body -> world
        linvel_w = math_utils.quat_apply(root_quats_w, root_linvels_b)  # (N,3)
        angvel_w = math_utils.quat_apply(root_quats_w, root_angvels_b)  # (N,3)
        if current_velocities_w is not None:
            linvel_w = linvel_w - current_velocities_w.reshape(linvel_w.shape[0], 3)
        if not self._has_prev:
            self._prev_linvel_w = linvel_w.clone()
            self._prev_angvel_w = angvel_w.clone()
            self._nu_dot_filt.zero_()
            self._has_prev = True
            return torch.zeros(
                (root_linvels_b.shape[0], 6),
                device=self.device,
                dtype=root_linvels_b.dtype,
            )

        # Handle resets: set prev = current and output zero accel for those envs
        if self._has_reset:
            reset_mask = self._reset_mask
            # For reset envs, sync previous state to current state
            self._prev_linvel_w[reset_mask] = linvel_w[reset_mask]
            self._prev_angvel_w[reset_mask] = angvel_w[reset_mask]
            # clear mask
            self._reset_mask[reset_mask] = False
            self._has_reset = False

        # diff in world frame
        lin_acc_w = (linvel_w - self._prev_linvel_w) / self.dt
        ang_acc_w = (angvel_w - self._prev_angvel_w) / self.dt
        # convert to current body frame
        lin_acc_b = math_utils.quat_apply_inverse(root_quats_w, lin_acc_w)
        ang_acc_b = math_utils.quat_apply_inverse(root_quats_w, ang_acc_w)

        nu_dot_raw = torch.cat([lin_acc_b, ang_acc_b], dim=-1)
        self._nu_dot_filt = (1 - alpha) * self._nu_dot_filt + alpha * nu_dot_raw
        nu_dot = self._nu_dot_filt
        # update prev caches (world)
        self._prev_linvel_w = linvel_w.clone()
        self._prev_angvel_w = angvel_w.clone()
        return nu_dot

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> None:
        """Reset internal states for selected environments."""
        # print("Resetting FossenForceModels for env_ids:", env_ids)
        if env_ids is None:
            self._prev_angvel_w.zero_()
            self._prev_linvel_w.zero_()
            self._nu_dot_filt.zero_()
            self._reset_mask.fill_(True)
        else:
            self._prev_angvel_w[env_ids] = 0.0
            self._prev_linvel_w[env_ids] = 0.0
            self._reset_mask[env_ids] = True
            self._nu_dot_filt[env_ids] = 0.0
        self._has_reset = True


def _coriolis_added_direct(nu: torch.Tensor, added_mass: torch.Tensor) -> torch.Tensor:
    # Closed-form -C_A(nu) * nu for diagonal added-mass (Fossen 6DOF).
    m = added_mass
    u, v, w, p, q, r = [nu[:, i] for i in range(6)]
    tau = torch.zeros_like(nu)
    tau[:, 0] = -m[2] * w * q + m[1] * v * r
    tau[:, 1] = m[2] * w * p - m[0] * u * r
    tau[:, 2] = -m[1] * v * p + m[0] * u * q
    tau[:, 3] = -(m[2] - m[1]) * v * w - (m[5] - m[4]) * q * r
    tau[:, 4] = (m[2] - m[0]) * u * w + (m[5] - m[3]) * p * r
    tau[:, 5] = (m[0] - m[1]) * u * v + (m[3] - m[4]) * p * q
    return tau


def _test_coriolis_term() -> None:
    device = torch.device("cpu")
    model = FossenForceModels(num_envs=8, device=device, dt=0.01)
    nu = torch.randn(model.num_envs, 6, device=device)
    c_a = model._coriolis_added(nu)
    coriolis_term = -torch.bmm(c_a, nu.unsqueeze(-1)).squeeze(-1)
    coriolis_direct = _coriolis_added_direct(nu, model.added_mass)
    if not torch.allclose(coriolis_term, coriolis_direct, atol=1e-6, rtol=1e-6):
        max_err = (coriolis_term - coriolis_direct).abs().max().item()
        raise AssertionError(f"coriolis_term mismatch, max_err={max_err:.3e}")
    skew_sym = c_a + c_a.transpose(1, 2)
    if not torch.allclose(skew_sym, torch.zeros_like(skew_sym), atol=1e-6, rtol=1e-6):
        max_err = skew_sym.abs().max().item()
        raise AssertionError(f"C_A not skew-symmetric, max_err={max_err:.3e}")
    print("coriolis_term test passed")


if __name__ == "__main__":
    device = torch.device("cpu")
    model = FossenForceModels(num_envs=8, device=device, dt=0.01)
    nu = torch.randn(model.num_envs, 6, device=device)
    c_a = model._coriolis_added(nu)
    cnu = torch.bmm(c_a, nu.unsqueeze(-1)).squeeze(-1)
    power = (nu * cnu).sum(dim=1)
    if not torch.allclose(power, torch.zeros_like(power), atol=1e-5, rtol=1e-5):
        max_err = power.abs().max().item()
        raise AssertionError(f"nu^T C(nu) nu != 0, max_err={max_err:.3e}")
    print("nu^T C(nu) nu test passed")
    _test_coriolis_term()
