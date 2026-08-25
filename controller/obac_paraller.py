from __future__ import annotations
import csv
import math
import os
from dataclasses import dataclass, field, MISSING
import torch
import torch.nn as nn

from isaaclab.utils import math as math_utils

try:
    from .base_controller import FossenParam, Fossen_Model
except ImportError:
    from base_controller import FossenParam, Fossen_Model
from tensordict import TensorDict

from isaaclab.utils import configclass

import yaml

import datetime
FILE_NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
RUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../", "runs", "mhobppo"))
CENTERS_LOAD_DIR = os.path.join(RUNS_DIR, FILE_NAME)
CENTERS_SAVE_PATH = os.path.join(CENTERS_LOAD_DIR, "centers.yaml")
PLAY_LOAD_DIR = os.path.join(RUNS_DIR, "2026-06-05_20-10")


def _load_yaml_tensor(file_name: str, key: str, device: torch.device, dtype: torch.dtype, mode=0) -> torch.Tensor:
    if mode == 2:
        with open(os.path.join(PLAY_LOAD_DIR, file_name), "r", encoding="utf-8") as f:
            return torch.as_tensor(yaml.safe_load(f)[key], device=device, dtype=dtype)
    elif mode == 0:
        try:
            with open(os.path.join(CENTERS_LOAD_DIR, file_name), "r", encoding="utf-8") as f:
                return torch.as_tensor(yaml.safe_load(f)[key], device=device, dtype=dtype)
        except FileNotFoundError:
            
            print("[Warning]: File not found, using default values.")
            with open(os.path.join(PLAY_LOAD_DIR, file_name), "r", encoding="utf-8") as f:
                return torch.as_tensor(yaml.safe_load(f)[key], device=device, dtype=dtype)
        


# ---------------- RBF ----------------
class RBFLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_centers: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
        center_range=[-8.0, 8.0, -10.0, 10.0],
        mode = 1, # 0 predictor, 1 learning 2 playing
    ):
        
        super().__init__()
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.input_dim = input_dim
        self.num_centers = num_centers
        # m,input_dim
        centers = torch.empty(num_centers, input_dim,
                              device=device, dtype=dtype)
        base_low = torch.tensor(
            [center_range[0]] * 3 + [center_range[2]] * 3,
            device=device,
            dtype=dtype,
        )
        base_high = torch.tensor(
            [center_range[1]] * 3 + [center_range[3]] * 3,
            device=device,
            dtype=dtype,
        )
        repeat_count = (input_dim + 5) // 6
        low = base_low.repeat(repeat_count)[:input_dim]
        high = base_high.repeat(repeat_count)[:input_dim]
        if mode == 2 or mode == 0:  # predicting or playing
            # print("[Debug]: Loading centers from yaml: ",mode)
            centers.copy_(_load_yaml_tensor("centers.yaml", "centers", device, dtype,mode=mode))
        else:
            centers.uniform_(0.0, 1.0).mul_(high - low).add_(low)
        self.register_buffer("centers", centers)
        # m,
        self.register_buffer(
            "beta", torch.full((num_centers,), 0.05,
                               device=device, dtype=dtype)
        )
        self.register_buffer(
            "x_scale",
            torch.tensor([1.0, 5.0, 5.0, 8.0, 8.0, 3.0],
                         device=device, dtype=dtype).repeat(repeat_count)[:input_dim],
        )
        
        print("[Debug]: RBFLayer initialized with centers: ", centers.shape)
        if mode == 1:  # only save when in learning mode
            os.makedirs(os.path.dirname(CENTERS_SAVE_PATH), exist_ok=True)
            with open(CENTERS_SAVE_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump({"centers": centers.cpu().tolist()}, f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # n,6
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x_n = x / self.x_scale
        c_n = self.centers / self.x_scale
        # n,m,6 = n,1,6 - 1,m,6
        diff = x_n.unsqueeze(1) - c_n.unsqueeze(0)
        # n,m
        sq_dist = (diff * diff).sum(dim=2)
        # n,m
        out = torch.exp(-self.beta.unsqueeze(0) * sq_dist)
        return out


class OBACmodule(nn.Module):
    def __init__(
        self,
        input_dim,
        num_centers,
        output_dim,
        ka,
        kc,
        dt,
        num_envs: int,
        device: torch.device,
        dtype: torch.dtype | None = None,
        center_range=[-8.0, 8.0, -10, 10.0],
        mode=1, # 0 predictor, 1 learning 2 playing
    ):
        super(OBACmodule, self).__init__()
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.mode = mode
        self.num_envs = num_envs
        self.output_dim = output_dim
        self.num_centers = num_centers
        self.dtype = dtype
        self.device = device
        
        self.rbf_layer = RBFLayer(
            input_dim,
            num_centers,
            device=device,
            dtype=dtype,
            center_range=center_range,
            mode=mode,
        )
        # n,m
        self.S = torch.zeros(self.num_envs, num_centers,
                             device=device, dtype=dtype)
        # base_linear = nn.Linear(num_centers, output_dim,
        #                         bias=False, dtype=dtype)
        # base_weight = base_linear.weight.detach().to(device=device, dtype=dtype)
        actor_tensor = torch.empty(
            output_dim,
            num_centers,
            device=device,
            dtype=dtype,
        )

        nn.init.normal_(
            actor_tensor,
            mean=2.0,
            std=0.5,
        )

        critic_tensor = actor_tensor.clone()

        # critic_tensor += (
        #     1e-2 *
        #     torch.randn_like(critic_tensor)
        # )

        if mode == 2:  # playing
            actor_tensor.copy_(
                _load_yaml_tensor("actor_weights.yaml", "actor_weights", device, dtype,mode=mode)
            )
            critic_tensor.copy_(
                _load_yaml_tensor("critic_weights.yaml", "critic_weights", device, dtype,mode=mode)
            )

        self.actor_weights = nn.Parameter(
            actor_tensor,
            requires_grad=False,
        )

        self.critic_weights = nn.Parameter(
            critic_tensor,
            requires_grad=False,
        )
        self.dt = dt
        self.update_k(ka, kc)

    def forward(self, x):
        # n,6
        if x.dim() == 1:
            x = x.unsqueeze(0)
        # n,m
        self.S = self.rbf_layer(x).detach()
        # n,6
        out = self.S.matmul(self.actor_weights.t())
        return out

    def update_weights(self):
        if self.mode == 2:  # not in learning mode
            return
        with torch.no_grad():
            if self.S.numel() == 0:
                return
            inv_n = 1.0 / self.S.shape[0]
            ka = self.ka.reshape(-1, 1)
            kc = self.kc.reshape(-1, 1)

            critic_out = self.S.matmul(self.critic_weights.t())
            actor_critic_out = self.S.matmul(
                (self.actor_weights - self.critic_weights).t()
            )
            Wc_dot = -(kc * critic_out).t().matmul(self.S) * inv_n
            Wa_dot = -(
                ka * actor_critic_out + kc * critic_out
            ).t().matmul(self.S) * inv_n

            self.critic_weights += Wc_dot * self.dt
            self.actor_weights += Wa_dot * self.dt

    def update_weights_e(self, e2: torch.Tensor):
        if self.mode == 2:  # not in learning mode
            return
        with torch.no_grad():
            if self.S.numel() == 0:
                return
            inv_n = 1.0 / self.S.shape[0]
            ka = self.ka.reshape(-1, 1)
            kc_mean = self.kc.mean()
            grad = (ka * e2).t().matmul(self.S) * inv_n
            Wa_dot = grad - kc_mean * self.actor_weights
            Wc_dot = grad - kc_mean * self.critic_weights
            self.actor_weights.copy_(self.actor_weights + Wa_dot * self.dt)
            self.critic_weights.copy_(self.critic_weights + Wc_dot * self.dt)

    def update_k(self, ka, kc):
        # n,
        ka_t = torch.as_tensor(ka, device=self.device, dtype=self.dtype)
        kc_t = torch.as_tensor(kc, device=self.device, dtype=self.dtype)
        if ka_t.dim() == 0 or ka_t.numel() == 1:
            ka_t = ka_t.expand(self.num_envs)
        if kc_t.dim() == 0 or kc_t.numel() == 1:
            kc_t = kc_t.expand(self.num_envs)
        self.ka = ka_t
        self.kc = kc_t
class OptimizedBacksteppingACController(torch.nn.Module):
    def __init__(self, num_envs: int, device: torch.device, fossenParams: FossenParam, ObppoGains: torch.Tensor, mode):
        super().__init__()
        self.num_envs = num_envs
        self.device = device
        
        self.dt = fossenParams.dt
        self.dtype = torch.get_default_dtype()
        
        self.num_centers = 50
        self.center_range = [-8.0, 8.0, -10.0, 10.0]
        
        self.params = fossenParams
        
        self.dynamic_model = Fossen_Model(device, num_envs, fossenParams)
        
        self.obac_module = OBACmodule(
            input_dim=6,
            num_centers=self.num_centers,
            output_dim=6,
            ka=ObppoGains[:, 12],  # n,1
            kc=ObppoGains[:, 13],  # n,1
            dt=self.dt,
            num_envs=self.num_envs,
            device=device,
            center_range=self.center_range,
            dtype=self.dtype,
            mode=mode,
        ).to(device)
        self.k1 = ObppoGains[:, 0:6].to(device)
        self.k2 = ObppoGains[:, 6:12].to(device)

        self.wrench_scale = self.dynamic_model.wrench_scale
        self.wrench_bias = self.dynamic_model.wrench_bias
        self.reset(fossenParams)

    def reset(self, dynamic_param: FossenParam, env_ids=None) -> None:
        'Reset target-dependent state while retaining gains to support exploration across environment resets.'
        self.dynamic_model.update_parameters(dynamic_param)
        self.M = self.dynamic_model.calculate_M()  # n,6,6
        self.B = self.dynamic_model.calculate_B()  # n,6,6,Matrix M^-1
        self.B_norm=self.B/torch.linalg.norm(
                    self.B,
                    dim=(1,2),
                    keepdim=True
                )
    def forward(self, obs: TensorDict, debug=False) -> tuple[torch.Tensor, dict]:
        state = obs["obppo"]
        if debug:
            print("-")
            print("[Debug]: OBAC forward called: ", state)
            print("[Debug]: K1: ", self.k1)
            print("[Debug]: K2: ", self.k2)
            print("[Debug]: Ka: ", self.obac_module.ka)
            print("[Debug]: Kc: ", self.obac_module.kc)
        # -------- Parse observation --------
        desire_quat_b = state[:, 0:4]
        desire_pose_b = state[:, 4:7]
        body_quat_w = state[:, 7:11]  # world FRD
        lin_vel_b = state[:, 11:14]  # body FRD
        ang_vel_b = state[:, 14:17]  # body FRD
        nu = torch.cat([lin_vel_b, ang_vel_b], dim=-1)  # (n,6)

        e1_pos = -desire_pose_b
        q_err = desire_quat_b.clone()  
        sign = torch.where(q_err[:, 0:1] >= 0.0, 1.0, -1.0)
        q_err = q_err * sign
        e_rot = -2.0 * q_err[:, 1:4]

        e1 = torch.cat([e1_pos, e_rot], dim=-1)
        L1 = 0.5 * torch.sum(e1 * e1, dim=1)

        nu_r = -self.k1 * e1
        nu_r_dot = -self.k1 * nu
        e2 = nu - nu_r
        # print("q_err scalar:", q_err[:, 0])
        # print("||e_rot||:", torch.norm(e_rot, dim=1))
        # print("omega_ref:", nu_r[:, 3:])
        # print("[Debug]: e1", e1)
        # print("[Debug]: nu_r", nu_r)
        # print("[Debug]: nu_r_dot", nu_r_dot)
        # print("[Debug]: e2", e2)

        # Dynamic feedforward
        C = self.dynamic_model.calculate_C(nu)
        D = self.dynamic_model.calculate_D(nu)
        g = self.dynamic_model.calculate_g(body_quat_w, from_quat=True)

        M_nu_r_dot = torch.bmm(self.M, nu_r_dot.unsqueeze(-1)).squeeze(-1)
        C_nu = torch.bmm(C, nu_r.unsqueeze(-1)).squeeze(-1)
        D_nu = torch.bmm(D, nu_r.unsqueeze(-1)).squeeze(-1)
        feedforward = M_nu_r_dot + C_nu + D_nu + g
        # Network output n,6
        actor_out = self.obac_module(e2)
        # n,6
        tau_input = e2 * self.k2 + 0.5 * actor_out
        tau_net = torch.bmm(self.M, tau_input.unsqueeze(-1)).squeeze(-1)
        tau = (feedforward - tau_net)

        # self.obac_module.update_weights_e(e2)
        self.obac_module.update_weights()
        # print("[Debug]: feedforward", feedforward)
        # print("[Debug]: M_nu_r_dot", M_nu_r_dot)
        # print("[Debug]: C_nu", C_nu)
        # print("[Debug]: D_nu", D_nu)
        # print("[Debug]: g", g)
        # print("[Debug]: tau_net", tau_net)
        # Stage cost
        e2_cost = torch.sum(e2 * e2, dim=1)
        B_tau = torch.bmm(self.B_norm, tau.unsqueeze(-1)).squeeze(-1)
        BTB_tau = torch.bmm(self.B_norm.transpose(
            1, 2), B_tau.unsqueeze(-1)).squeeze(-1)
        # print("[Debug]: BTB_tau", BTB_tau)
        tau_cost = torch.sum(tau * BTB_tau, dim=1)
        V2 = e2_cost + 1e-3*tau_cost

        info = {
            "e1": e1,
            "e2": e2,
            "L1": L1,
            "nu_r": nu_r,
            "tau_cost": tau_cost,
            "V2": V2,
            "e2_cost": e2_cost,
            "actor_weights": self.obac_module.actor_weights,
            "critic_weights": self.obac_module.critic_weights,

        }
        tau = torch.clamp((tau - self.wrench_bias) /
                          self.wrench_scale, -1.0, 1.0)
        if debug:
            print("[Debug]: Desired Tau: ", tau)
        return tau, info

    def update_gain(self, gain: torch.Tensor):
        self.k1 = gain[:, 0:6]
        self.k2 = gain[:, 6:12]
        self.obac_module.update_k(gain[:, 12], gain[:, 13])
