from __future__ import annotations
import torch
import torch.nn as nn
import isaacsim 
from isaaclab.utils import math as math_utils
try:
    from .obac_paraller import OBACmodule
    from .base_controller import FossenParam, Fossen_Model
except ImportError:
    from obac_paraller import OBACmodule
    from base_controller import FossenParam, Fossen_Model
    
    
class OBACStepModel(nn.Module):
    """Single-step OBAC dynamics model on the compact 17D predictor state."""

    STATE_DIM = 17
    GAIN_DIM = 14
    ACTION_DIM = 6

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        fossen_params: FossenParam,
        obac_gains: torch.Tensor,
        num_centers: int = 50,
        center_range: tuple[float, float, float, float] = (-8.0, 8.0, -10.0, 10.0),
    ):
        super().__init__()
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = torch.get_default_dtype()
        self.dynamic_model = Fossen_Model(device, self.num_envs, fossen_params)
        self.num_centers = int(num_centers)
        self.center_range = tuple(center_range)
        self.dt = torch.as_tensor(fossen_params.dt, device=device, dtype=self.dtype)
        self.down_axis = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=self.dtype)
        self.world_down = torch.tensor([0.0, 0.0, -1.0], device=device, dtype=self.dtype)

        gains = self._expand_gain(obac_gains)
        self.obac_module = OBACmodule(
            input_dim=6,
            num_centers=self.num_centers,
            output_dim=6,
            ka=gains[:, 12],
            kc=gains[:, 13],
            dt=float(self.dt.item()),
            num_envs=self.num_envs,
            device=device,
            dtype=self.dtype,
            center_range=list(self.center_range),
            mode=0
        ).to(device)

        self.wrench_scale = self.dynamic_model.wrench_scale
        self.wrench_bias = self.dynamic_model.wrench_bias

        self.reset(fossen_params, obac_gains)

    def reset(self, dynamic_param: FossenParam, obac_gains: torch.Tensor | None = None) -> None:
        self.dynamic_model.update_parameters(dynamic_param)
        self.M = self.dynamic_model.calculate_M()
        self.B = self.dynamic_model.calculate_B()
        self.update_gain(obac_gains)

    def update_gain(self, gain: torch.Tensor) -> None:
        gain = self._expand_gain(gain)
        self.k1 = gain[:, 0:6]
        self.k2 = gain[:, 6:12]
        self.obac_module.update_k(gain[:, 12], gain[:, 13])

    def _expand_gain(self, gain: torch.Tensor) -> torch.Tensor:
        gain = torch.as_tensor(gain, device=self.device, dtype=self.dtype)
        if gain.dim() == 1:
            gain = gain.unsqueeze(0)
        if gain.shape[0] == 1 and self.num_envs > 1:
            gain = gain.expand(self.num_envs, -1)
        if gain.shape[0] != self.num_envs:
            raise ValueError(f"Expected gain batch {self.num_envs}, got {gain.shape[0]}")
        if gain.shape[1] != self.GAIN_DIM:
            raise ValueError(f"Expected gain dim {self.GAIN_DIM}, got {gain.shape[1]}")
        return gain.contiguous()

    @staticmethod
    def quat_to_rotvec_wxyz(q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        qw = torch.clamp(q[:, 0], -1.0, 1.0)
        qv = q[:, 1:4]
        sin_half = torch.norm(qv, dim=-1)
        angle = 2.0 * torch.atan2(sin_half, qw.abs() + eps)
        axis = qv / (sin_half.unsqueeze(-1) + eps)
        rotvec = axis * angle.unsqueeze(-1)
        return torch.where((qw < 0.0).unsqueeze(-1), -rotvec, rotvec)

    @staticmethod
    def rotmat_to_quat_wxyz(rot: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        trace = rot[:, 0, 0] + rot[:, 1, 1] + rot[:, 2, 2]
        quat = torch.zeros((rot.shape[0], 4), device=rot.device, dtype=rot.dtype)

        mask = trace > 0.0
        if torch.any(mask):
            s = torch.sqrt(trace[mask] + 1.0) * 2.0
            quat[mask, 0] = 0.25 * s
            quat[mask, 1] = (rot[mask, 2, 1] - rot[mask, 1, 2]) / s
            quat[mask, 2] = (rot[mask, 0, 2] - rot[mask, 2, 0]) / s
            quat[mask, 3] = (rot[mask, 1, 0] - rot[mask, 0, 1]) / s

        mask_x = (~mask) & (rot[:, 0, 0] > rot[:, 1, 1]) & (rot[:, 0, 0] > rot[:, 2, 2])
        if torch.any(mask_x):
            s = torch.sqrt(1.0 + rot[mask_x, 0, 0] - rot[mask_x, 1, 1] - rot[mask_x, 2, 2] + eps) * 2.0
            quat[mask_x, 0] = (rot[mask_x, 2, 1] - rot[mask_x, 1, 2]) / s
            quat[mask_x, 1] = 0.25 * s
            quat[mask_x, 2] = (rot[mask_x, 0, 1] + rot[mask_x, 1, 0]) / s
            quat[mask_x, 3] = (rot[mask_x, 0, 2] + rot[mask_x, 2, 0]) / s

        mask_y = (~mask) & (~mask_x) & (rot[:, 1, 1] > rot[:, 2, 2])
        if torch.any(mask_y):
            s = torch.sqrt(1.0 + rot[mask_y, 1, 1] - rot[mask_y, 0, 0] - rot[mask_y, 2, 2] + eps) * 2.0
            quat[mask_y, 0] = (rot[mask_y, 0, 2] - rot[mask_y, 2, 0]) / s
            quat[mask_y, 1] = (rot[mask_y, 0, 1] + rot[mask_y, 1, 0]) / s
            quat[mask_y, 2] = 0.25 * s
            quat[mask_y, 3] = (rot[mask_y, 1, 2] + rot[mask_y, 2, 1]) / s

        mask_z = (~mask) & (~mask_x) & (~mask_y)
        if torch.any(mask_z):
            s = torch.sqrt(1.0 + rot[mask_z, 2, 2] - rot[mask_z, 0, 0] - rot[mask_z, 1, 1] + eps) * 2.0
            quat[mask_z, 0] = (rot[mask_z, 1, 0] - rot[mask_z, 0, 1]) / s
            quat[mask_z, 1] = (rot[mask_z, 0, 2] + rot[mask_z, 2, 0]) / s
            quat[mask_z, 2] = (rot[mask_z, 1, 2] + rot[mask_z, 2, 1]) / s
            quat[mask_z, 3] = 0.25 * s

        return nn.functional.normalize(quat, dim=-1)

    def _compute_control(
        self,
        desire_quat_b: torch.Tensor,
        desire_pose_b: torch.Tensor,
        nu: torch.Tensor,
        C: torch.Tensor,
        D: torch.Tensor,
        g: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        e1_pos = -desire_pose_b
        q_err = desire_quat_b.clone()
        q_err = q_err * torch.where(q_err[:, 0:1] >= 0.0, 1.0, -1.0)
        e_rot = -2.0 * q_err[:, 1:4]
        e1 = torch.cat([e1_pos, e_rot], dim=-1)

        nu_r = -self.k1 * e1
        nu_r_dot = -self.k1 * nu
        e2 = nu - nu_r
        feedforward = torch.bmm(self.M, nu_r_dot.unsqueeze(-1)).squeeze(-1)
        feedforward = feedforward + torch.bmm(C, nu_r.unsqueeze(-1)).squeeze(-1)
        feedforward = feedforward + torch.bmm(D, nu_r.unsqueeze(-1)).squeeze(-1)
        feedforward = feedforward + g

        actor_out = self.obac_module(e2)
        actor_zero = self.obac_module(torch.zeros_like(e2))
        actor_out = actor_out - actor_zero
        tau_input = e2 * self.k2 + 0.5 * actor_out
        tau = feedforward - torch.bmm(self.M, tau_input.unsqueeze(-1)).squeeze(-1)
        self.obac_module.update_weights()
        tau_norm = torch.clamp((tau - self.wrench_bias) / self.wrench_scale, -1.0, 1.0)
        # print("tau: ", tau, "e1: ", e1,"e2: ", e2, "actor_out: ", actor_out,"feedforward: ", feedforward)
        return tau, tau_norm

    def _jinv_dot(
        self,
        euler: torch.Tensor,
        nu: torch.Tensor,
        R_nb: torch.Tensor,
        T: torch.Tensor,
    ) -> torch.Tensor:
        phi = euler[:, 0]
        theta = euler[:, 1]
        omega = nu[:, 3:6]
        rt = R_nb.transpose(1, 2)
        rt_dot = -torch.bmm(math_utils.skew_symmetric_matrix(omega), rt)

        theta_dot = torch.bmm(T, omega.unsqueeze(-1)).squeeze(-1)
        phi_dot = theta_dot[:, 0]
        pitch_dot = theta_dot[:, 1]

        sphi = torch.sin(phi)
        cphi = torch.cos(phi)
        st = torch.sin(theta)
        ct = torch.clamp(torch.cos(theta), min=1e-6)

        tinv_dot = torch.zeros((euler.shape[0], 3, 3), device=self.device, dtype=euler.dtype)
        tinv_dot[:, 0, 2] = -ct * pitch_dot
        tinv_dot[:, 1, 1] = -sphi * phi_dot
        tinv_dot[:, 1, 2] = cphi * phi_dot * ct - sphi * st * pitch_dot
        tinv_dot[:, 2, 1] = -cphi * phi_dot
        tinv_dot[:, 2, 2] = -sphi * phi_dot * ct - cphi * st * pitch_dot

        zeros = torch.zeros((euler.shape[0], 3, 3), device=self.device, dtype=euler.dtype)
        return torch.cat([torch.cat([rt_dot, zeros], dim=2), torch.cat([zeros, tinv_dot], dim=2)], dim=1)

    @torch.no_grad()
    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        state = torch.as_tensor(state, device=self.device, dtype=self.dtype)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.shape[0] == 1 and self.num_envs > 1:
            state = state.expand(self.num_envs, -1)
        if state.shape[0] != self.num_envs:
            raise ValueError(f"Expected state batch {self.num_envs}, got {state.shape[0]}")
        if state.shape[1] < self.STATE_DIM:
            raise ValueError(f"Expected state dim >= {self.STATE_DIM}, got {state.shape[1]}")

        core_state = state[:, : self.STATE_DIM]
        desire_quat_b = core_state[:, 0:4]
        desire_pose_b = core_state[:, 4:7]
        gravity_b = core_state[:, 7:10]
        yaw = core_state[:, 10:11]
        nu = core_state[:, 11:17]

        C = self.dynamic_model.calculate_C(nu)
        D = self.dynamic_model.calculate_D(nu)
        g = self.dynamic_model.calculate_g(gravity_b, from_quat=False)
        tau, tau_norm = self._compute_control(
            desire_quat_b,
            desire_pose_b,
            nu,
            C,
            D,
            g
        )

        euler_now = self.dynamic_model.compute_euler_from_gravity(gravity_b, yaw)
        J, R_nb = self.dynamic_model.calculate_J_eta(euler_now)
        eta_dot = torch.bmm(J, nu.unsqueeze(-1)).squeeze(-1)
        pos_dot = eta_dot[:, 0:3]
        euler_dot = eta_dot[:, 3:6]

        hydrodynamic = torch.bmm(C, nu.unsqueeze(-1)).squeeze(-1)
        hydrodynamic = hydrodynamic + torch.bmm(D, nu.unsqueeze(-1)).squeeze(-1)
        hydrodynamic = hydrodynamic + g
        nu_next = nu + torch.bmm(self.B, (tau - hydrodynamic).unsqueeze(-1)).squeeze(-1) * self.dt

        pos_des = torch.bmm(R_nb, desire_pose_b.unsqueeze(-1)).squeeze(-1)
        pos_next = pos_dot * self.dt
        euler_next = euler_now + euler_dot * self.dt
        euler_next[:, 2] = math_utils.wrap_to_pi(euler_next[:, 2])

        _, R_nb_next = self.dynamic_model.calculate_J_eta(euler_next)
        R_bn_next = R_nb_next.transpose(1, 2)
        t_bd_next = torch.bmm(R_bn_next, (pos_des - pos_next).unsqueeze(-1)).squeeze(-1)
        gravity_b_next = torch.bmm(R_bn_next, self.down_axis.expand(self.num_envs, -1).unsqueeze(-1)).squeeze(-1)
        yaw_next = math_utils.wrap_to_pi(-euler_next[:, 2]).unsqueeze(-1)

        q_nb = self.rotmat_to_quat_wxyz(R_nb)
        q_nb_next = self.rotmat_to_quat_wxyz(R_nb_next)
        q_nd = math_utils.quat_mul(q_nb, desire_quat_b)
        q_bd_next = math_utils.quat_mul(math_utils.quat_inv(q_nb_next), q_nd)
        q_bd_next = nn.functional.normalize(q_bd_next, dim=-1)

        next_core = torch.cat([q_bd_next, t_bd_next, gravity_b_next, yaw_next, nu_next], dim=-1)

        if state.shape[1] == self.STATE_DIM:
            return next_core

        tail = state[:, self.STATE_DIM :].clone()
        if tail.shape[1] >= self.ACTION_DIM:
            tail[:, 0 : self.ACTION_DIM] = tau_norm
        return torch.cat([next_core, tail], dim=-1)


class OBACPredictor(nn.Module):
    """Fast rollout model used to build privileged critic observations."""
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        fossenParams: FossenParam,
        obac_gains: torch.Tensor,
        predict_steps: int,
    ):
        super().__init__()
        self.predict_steps = int(predict_steps)
        self.step_model = OBACStepModel(
            num_envs=num_envs,
            device=device,
            fossen_params=fossenParams,
            obac_gains=obac_gains,
        )

    def reset(self, dynamic_param: FossenParam, obac_gains: torch.Tensor | None = None) -> None:
        self.step_model.reset(dynamic_param, obac_gains)

    def update_gain(self, gain: torch.Tensor) -> None:
        self.step_model.update_gain(gain)

    @staticmethod
    def _quat_to_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat.unbind(dim=-1)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    @torch.no_grad()
    def observation_to_state(self, observation: torch.Tensor) -> torch.Tensor:
        observation = torch.as_tensor(
            observation, device=self.step_model.device, dtype=self.step_model.dtype
        )
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        if observation.shape[1] < 17:
            raise ValueError(f"Expected observation dim >= 17, got {observation.shape[1]}")

        desire_quat_b = observation[:, 0:4]
        desire_pose_b = observation[:, 4:7]
        body_quat_w = observation[:, 7:11]
        nu = observation[:, 11:17]
        gravity_b = math_utils.quat_apply_inverse(
            body_quat_w,
            self.step_model.world_down.expand(body_quat_w.shape[0], -1),
        )
        yaw = self._quat_to_yaw_wxyz(body_quat_w).unsqueeze(-1)
        compact = torch.cat([desire_quat_b, desire_pose_b, gravity_b, yaw, nu], dim=-1)
        if observation.shape[1] == 17:
            return compact
        return torch.cat([compact, observation[:, 17:]], dim=-1)

    @torch.no_grad()
    def predict(self,state: torch.Tensor) -> torch.Tensor:
        current_state = state
        rollout = None
        for _ in range(self.predict_steps):
            current_state = self.step_model(current_state)
            if rollout is None:
                rollout = current_state.new_empty(
                    (current_state.shape[0], self.predict_steps, current_state.shape[1])
                )
            rollout[:, _, :] = current_state
        return rollout

    @torch.no_grad()
    def predict_from_observation(self,observation: torch.Tensor) -> torch.Tensor:
        compact_state = self.observation_to_state(observation)
        return self.predict(compact_state)
    

def _test_obac_predictor() -> None:
    torch.manual_seed(0)
    device = torch.device("cpu")
    dtype = torch.get_default_dtype()
    num_envs = 3
    predict_steps = 40

    mass = torch.full((num_envs, 1), 12.4, device=device, dtype=dtype)
    inertia_const = torch.tensor(
        [
            (0.4**2 + 0.2**2) / 12.0,
            (0.7**2 + 0.2**2) / 12.0,
            (0.7**2 + 0.4**2) / 12.0,
        ],
        device=device,
        dtype=dtype,
    )
    params = FossenParam(
        m=mass,
        I=mass * inertia_const,
        cg=torch.tensor(
            [[0.0, 0.0, 0.02]],
            device=device,
            dtype=dtype,
        ).repeat(num_envs, 1),
        volume=torch.full((num_envs, 1), 0.0126, device=device, dtype=dtype),
        dt=torch.tensor(0.02, device=device, dtype=dtype),
    )
    gains = torch.tensor(
        [1.5, 1.6, 1.7, 5.0, 5.1, 5.2, 4.0, 4.1, 4.2, 15.0, 15.1, 15.2, 3.0, 5.0],
        device=device,
        dtype=dtype,
    ).repeat(num_envs, 1)
    predictor = OBACPredictor(
        num_envs=num_envs,
        device=device,
        fossenParams=params,
        obac_gains=gains,
        predict_steps=predict_steps,
    )
    # print("Test Parameters:")
    # print("Mass:", params.m)
    # print("Inertia:", params.I)
    # print("Center of Gravity:", params.cg)
    # print("Volume:", params.volume)
    # print("Gains:", gains)
    # print("Predict Steps:", predict_steps)

    state = torch.zeros((num_envs, OBACStepModel.STATE_DIM), device=device, dtype=dtype)
    state[:, 0:4] = nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.00, 0.00, 0.00],
                [0.997, 0.04, -0.03, 0.02],
                [0.990, -0.06, 0.04, -0.03],
            ],
            device=device,
            dtype=dtype,
        ),
        dim=-1,
    )
    state[:, 4:7] = torch.tensor(
        [
            [0.00, 0.00, 0.00],
            [0.15, -0.10, 0.05],
            [-0.20, 0.08, -0.04],
        ],
        device=device,
        dtype=dtype,
    )
    state[:, 7:10] = torch.tensor(
        [
            [0.00, 0.00, 1.00],
            [0.05, -0.03, 1.00],
            [-0.04, 0.06, 1.00],
        ],
        device=device,
        dtype=dtype,
    )
    state[:, 7:10] = nn.functional.normalize(state[:, 7:10], dim=-1)
    state[:, 10] = torch.tensor([0.0, 0.12, -0.10], device=device, dtype=dtype)
    state[:, 11:17] = torch.tensor(
        [
            [0.00, -0.00, 0.0, 0.0, 0.0, -0.0],
            [0.03, -0.02, 0.01, 0.02, -0.01, 0.015],
            [-0.04, 0.025, -0.015, -0.015, 0.02, -0.01],
        ],
        device=device,
        dtype=dtype,
    )
    # print("*"*20)
    # print("Initial state: quat: ", state[:, 0:4], "pos: ", state[:, 4:7], "gravity: ", state[:, 7:10], "yaw: ", state[:, 10:11], "nu: ", state[:, 11:17])
    rollout = predictor.predict(state)
    # print("Predicted rollout shape:", rollout.shape)
    # for t in range(predict_steps):
    #     print("*"*20)
    #     print(f"Step {t+1} quat: ", rollout[:, t, 0:4], "pos: ", rollout[:, t, 4:7], "gravity: ", rollout[:, t, 7:10], "yaw: ", rollout[:, t, 10:11], "nu: ", rollout[:, t, 11:17])

if __name__ == '__main__':
    _test_obac_predictor()
