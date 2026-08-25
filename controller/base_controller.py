'Base Fossen dynamics model for controllers, including physical parameters and matrix computations.'
import torch.nn as nn
from dataclasses import dataclass
import torch

import isaaclab.utils.math as math_utils

@dataclass
class FossenParam:
    """Data container for physical parameters."""

    m: torch.Tensor  # (n, 1)
    I: torch.Tensor  # (n, 3)
    cg: torch.Tensor  # (n, 3)
    volume: torch.Tensor  # (n, 1)
    cb: torch.Tensor = None  # (n, 3) Will be init in post_init if None
    rho: torch.Tensor | float | None = None  # kg/m^3 for fresh water
    g_const: torch.Tensor | float | None = None  # m/s^2
    dt: torch.Tensor | float | None = None  # controller step

    def __post_init__(self):
        if self.cb is None:
            self.cb = torch.zeros_like(self.cg)
        tensor_kwargs = {"device": self.cg.device, "dtype": self.cg.dtype}
        self.rho = torch.as_tensor(997.0 if self.rho is None else self.rho, **tensor_kwargs)
        self.g_const = torch.as_tensor(
            9.81 if self.g_const is None else self.g_const, **tensor_kwargs
        )
        self.dt = torch.as_tensor(0.02 if self.dt is None else self.dt, **tensor_kwargs)


class Fossen_Model(nn.Module):
    'Base controller using the SNAME/Fossen FRD body frame and ZYX Euler convention.'

    def __init__(self, device: torch.device, num_envs: int, param: FossenParam):
        super().__init__()
        self.device = device
        self.num_envs = num_envs
        self.p = param

        
        M_RB = self._calculate_M_RB()
        self.register_buffer("M_RB", M_RB)
        # n,6,6
        M_A_diag = torch.tensor([5.5, 12.7, 14.57, 0.12, 0.12, 0.12], device=device)
        self.register_buffer("M_A", torch.diag(M_A_diag).expand(num_envs, 6, 6))
        # n,6
        self.register_buffer(
            "linear_damping",
            torch.tensor([4.03, 6.22, 5.18, 0.07, 0.07, 0.07], device=device).repeat(
                num_envs, 1
            ),
        )
        self.register_buffer(
            "quad_damping",
            torch.tensor([18.18, 21.66, 36.99, 1.55, 1.55, 1.55], device=device).repeat(
                num_envs, 1
            ),
        )
        thruster_min, thruster_max = -40.0, 40.0
        wrench_range = thruster_max - thruster_min
        wrench_max = torch.tensor(
            [
                torch.sqrt(torch.tensor(2.0, device=device)) * wrench_range,
                torch.sqrt(torch.tensor(2.0, device=device)) * wrench_range,
                4 * thruster_max,
                0.0417 * wrench_range * 2 + 0.222 * wrench_range * 2,
                0.0417 * wrench_range * 2
                - 0.1214 * thruster_min * 2
                + 0.1186 * thruster_max * 2,
                0.1853 * wrench_range * 2,
            ],
            device=device,
        )
        wrench_min = torch.tensor(
            [
                -torch.sqrt(torch.tensor(2.0, device=device)) * wrench_range,
                -torch.sqrt(torch.tensor(2.0, device=device)) * wrench_range,
                4 * thruster_min,
                -0.0417 * wrench_range * 2 - 0.222 * wrench_range * 2,
                -0.0417 * wrench_range * 2
                - 0.1214 * thruster_max * 2
                + 0.1186 * thruster_min * 2,
                -0.1853 * wrench_range * 2,
            ],
            device=device,
        )
        self.register_buffer("wrench_scale", 0.5 * (wrench_max - wrench_min))
        self.register_buffer("wrench_bias", 0.5 * (wrench_max + wrench_min))

    def update_parameters(self, param: FossenParam):
        """Refresh stored parameters and derived matrices."""
        self.p = param
        with torch.no_grad():
            self.M_RB.copy_(self._calculate_M_RB())

    def _calculate_M_RB(self) -> torch.Tensor:
        m = self.p.m.view(self.num_envs, 1, 1).to(device=self.device)  # (n,1,1)
        I_mat = torch.diag_embed(self.p.I).to(device=self.device)  # (n,3,3)
        # n,3,3
        S_rg = math_utils.skew_symmetric_matrix(self.p.cg).to(
            device=self.device
        )  # (n,3,3)
        # n,6,6
        M_RB = torch.zeros((self.num_envs, 6, 6), device=self.device)
        M_RB[:, 0:3, 0:3] = m * torch.eye(3, device=self.device)
        M_RB[:, 0:3, 3:6] = -m * S_rg
        M_RB[:, 3:6, 0:3] = m * S_rg
        M_RB[:, 3:6, 3:6] = I_mat
        return M_RB

    def calculate_M(self):
        """Calculate the total mass matrix."""
        return self.M_RB + self.M_A

    def calculate_B(self):
        M = self.calculate_M()
        I = torch.eye(6, device=self.device, dtype=M.dtype).expand(self.num_envs, 6, 6)
        return torch.linalg.solve(M, I)

    def calculate_C(self, nu: torch.Tensor) -> torch.Tensor:
        """Calculate Coriolis matrix C(nu) = C_RB(nu) + C_A(nu)."""
        # n,3,3
        M_RB = self.M_RB
        m11, m12 = M_RB[:, 0:3, 0:3], M_RB[:, 0:3, 3:6]
        m21, m22 = M_RB[:, 3:6, 0:3], M_RB[:, 3:6, 3:6]
        nu1, nu2 = nu[:, 0:3].unsqueeze(-1), nu[:, 3:6].unsqueeze(-1)
        # Proper Fossen parameterization for C_RB
        h = torch.cat([m11 @ nu1 + m12 @ nu2, m21 @ nu1 + m22 @ nu2], dim=1).squeeze(
            -1
        )  # (n, 6)
        C_RB = torch.zeros_like(M_RB)
        C_RB[:, 0:3, 3:6] = -math_utils.skew_symmetric_matrix(h[:, 0:3])
        C_RB[:, 3:6, 0:3] = -math_utils.skew_symmetric_matrix(h[:, 0:3])
        C_RB[:, 3:6, 3:6] = -math_utils.skew_symmetric_matrix(h[:, 3:6])

        # Added Mass Coriolis (Simpler approximation using M_A)
        # Note: accurate C_A calculation depends on M_A structure.
        # For diagonal M_A, C_A is simpler. Using full skew formula:
        M_A = self.M_A
        ma11, ma12 = M_A[:, 0:3, 0:3], M_A[:, 0:3, 3:6]
        ma21, ma22 = M_A[:, 3:6, 0:3], M_A[:, 3:6, 3:6]
        ha = torch.cat(
            [ma11 @ nu1 + ma12 @ nu2, ma21 @ nu1 + ma22 @ nu2], dim=1
        ).squeeze(-1)
        C_A = torch.zeros_like(M_A)
        C_A[:, 0:3, 3:6] = -math_utils.skew_symmetric_matrix(ha[:, 0:3])
        C_A[:, 3:6, 0:3] = -math_utils.skew_symmetric_matrix(ha[:, 0:3])
        C_A[:, 3:6, 3:6] = -math_utils.skew_symmetric_matrix(ha[:, 3:6])

        return C_RB + C_A

    def calculate_D(self, nu: torch.Tensor) -> torch.Tensor:
        """Damping Matrix D(nu) = D_lin + D_quad * |nu|"""
        d_diag = self.linear_damping + self.quad_damping * torch.abs(nu)
        return torch.diag_embed(d_diag)

    def calculate_g(self, angles: torch.Tensor, from_quat=False) -> torch.Tensor:
        'Compute restoring forces from either gravity-vector projections or quaternions.'
        W = self.p.m.squeeze() * self.p.g_const
        B = self.p.rho * self.p.g_const * self.p.volume.squeeze()
        g = torch.zeros((self.num_envs, 6), device=self.device)
        if from_quat:
            gravity_dir_w = angles.new_tensor([0.0, 0.0, -1.0]).expand(
                angles.shape[0], -1
            )
            gamma = math_utils.quat_apply_inverse(angles, gravity_dir_w)
        else:
            gamma = angles  # should be R_bn e3 (Down in body)
        # print("device:", self.device, W.device, B.device, gamma.device)
        f = (W - B).unsqueeze(-1) * gamma
        m = torch.cross(self.p.cg, (W.unsqueeze(-1) * gamma), dim=-1) - torch.cross(
            self.p.cb, (B.unsqueeze(-1) * gamma), dim=-1
        )

        g[:, 0:3] = f
        g[:, 3:6] = m
        return g

    def calculate_J_eta(self, euler: torch.Tensor) -> torch.Tensor:
        """
        Calculate the transformation matrix from body to inertial frame.
        euler: n,3 roll pitch yaw
        J_eta:n,6,6; eta_dot = J_eta*nu
        """
        phi, theta, psi = euler[:, 0], euler[:, 1], euler[:, 2]
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        cth, sth = torch.cos(theta), torch.sin(theta)
        cpsi, spsi = torch.cos(psi), torch.sin(psi)

        R_nb = torch.zeros((self.num_envs, 3, 3), device=self.device)
        R_nb[:, 0, 0] = cpsi * cth
        R_nb[:, 0, 1] = -spsi * cphi + cpsi * sth * sphi
        R_nb[:, 0, 2] = spsi * sphi + cpsi * sth * cphi
        R_nb[:, 1, 0] = spsi * cth
        R_nb[:, 1, 1] = cpsi * cphi + spsi * sth * sphi
        R_nb[:, 1, 2] = -cpsi * sphi + spsi * sth * cphi
        R_nb[:, 2, 0] = -sth
        R_nb[:, 2, 1] = cth * sphi
        R_nb[:, 2, 2] = cth * cphi

        # Angular Velocity Transform T (nu_omega -> eta_dot_angle)
        # [dphi; dtheta; dpsi] = T * [p; q; r]
        T = torch.zeros((self.num_envs, 3, 3), device=self.device)
        # Singularity protection
        sign_cth = torch.where(cth >= 0, torch.ones_like(cth), -torch.ones_like(cth))
        cth_safe = torch.where(torch.abs(cth) < 0.05, 0.05 * sign_cth, cth)

        T[:, 0, 0] = 1.0
        T[:, 0, 1] = sphi * sth / cth_safe
        T[:, 0, 2] = cphi * sth / cth_safe
        T[:, 1, 1] = cphi
        T[:, 1, 2] = -sphi
        T[:, 2, 1] = sphi / cth_safe
        T[:, 2, 2] = cphi / cth_safe

        J = torch.zeros((self.num_envs, 6, 6), device=self.device)
        J[:, 0:3, 0:3] = R_nb
        J[:, 3:6, 3:6] = T
        return J, R_nb

    def calculate_Jinv_eta(self, euler: torch.Tensor) -> torch.Tensor:
        phi, theta, psi = euler[:, 0], euler[:, 1], euler[:, 2]
        cphi, sphi = torch.cos(phi), torch.sin(phi)
        cth, sth = torch.cos(theta), torch.sin(theta)
        cpsi, spsi = torch.cos(psi), torch.sin(psi)

        R_nb = torch.zeros((self.num_envs, 3, 3), device=self.device)
        R_nb[:, 0, 0] = cpsi * cth
        R_nb[:, 0, 1] = -spsi * cphi + cpsi * sth * sphi
        R_nb[:, 0, 2] = spsi * sphi + cpsi * sth * cphi
        R_nb[:, 1, 0] = spsi * cth
        R_nb[:, 1, 1] = cpsi * cphi + spsi * sth * sphi
        R_nb[:, 1, 2] = -cpsi * sphi + spsi * sth * cphi
        R_nb[:, 2, 0] = -sth
        R_nb[:, 2, 1] = cth * sphi
        R_nb[:, 2, 2] = cth * cphi

        R_bn = R_nb.transpose(1, 2)

        sign_cth = torch.where(cth >= 0, torch.ones_like(cth), -torch.ones_like(cth))
        cth_safe = torch.where(torch.abs(cth) < 0.05, 0.05 * sign_cth, cth)
        T_inv = torch.zeros((self.num_envs, 3, 3), device=self.device)
        T_inv[:, 0, 0] = 1.0
        T_inv[:, 0, 2] = -sth
        T_inv[:, 1, 1] = cphi
        T_inv[:, 1, 2] = sphi * cth_safe
        T_inv[:, 2, 1] = -sphi
        T_inv[:, 2, 2] = cphi * cth_safe

        J_inv = torch.zeros((self.num_envs, 6, 6), device=self.device)
        J_inv[:, 0:3, 0:3] = R_bn
        J_inv[:, 3:6, 3:6] = T_inv
        return J_inv

    def compute_euler_from_gravity(
        self, gravity_b: torch.Tensor, yaw: torch.Tensor
    ) -> torch.Tensor:
        # --- normalize ---
        gx, gy, gz = gravity_b[:, 0], gravity_b[:, 1], gravity_b[:, 2]

        # --- roll/pitch in FRD from "down" vector ---
        # Using standard relations for FRD with gravity = +Down axis in body.
        pitch_frd = torch.atan2(-gx, torch.sqrt(gy * gy + gz * gz))
        roll_frd = torch.atan2(gy, gz)

        # --- yaw: FLU(z-up) -> FRD(z-down) ---
        yaw_flu = yaw.squeeze(-1)
        yaw_frd = math_utils.wrap_to_pi(-yaw_flu)

        euler_frd = torch.stack([roll_frd, pitch_frd, yaw_frd], dim=-1)

  
        return euler_frd
