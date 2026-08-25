"""
BlueROV environment for IsaacLabs
MHOBPPO for BlueROV control

Author: Airporal Chen
"""
from __future__ import annotations

import torch

import os
from collections.abc import Sequence
from .assets.bluerov import BLUEROV_ARTICULATION_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.markers import (
    CUBOID_MARKER_CFG,
    SPHERE_MARKER_CFG,
    VisualizationMarkers,
    RED_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    BLUE_ARROW_X_MARKER_CFG,
)
import isaaclab.utils.math as math_utils
from isaaclab.utils.timer import Timer
import gymnasium as gym
import numpy as np
import yaml


from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
)
from .hydrodynamics import InertiaForceModels, FossenForceModels

from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, TerrainGenerator
from .config.mhobppo_config import DefaultArgsConfig, EventCfg
from .thrusterModels.thruster import Thruster
from .controller.obac_paraller import OptimizedBacksteppingACController

from .controller.base_controller import FossenParam

from .controller import OBACPredictor
import datetime
FILE_NAME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
ACTOR_WEIGHTS_SAVE_PATH = os.path.join(os.path.dirname(__file__), "runs","mhobppo", FILE_NAME, "actor_weights.yaml")
CRITIC_WEIGHTS_SAVE_PATH = os.path.join(os.path.dirname(__file__), "runs","mhobppo", FILE_NAME, "critic_weights.yaml")


_OBS_CLIP = 1.0e6
_WRENCH_CLIP = 1.0e6


class BlueROVEnvWindow(BaseEnvWindow):
    """Window manager for the BlueROV environment."""

    def __init__(self, env: BlueROVEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class BlueROVEnvCfg(DirectRLEnvCfg):
    params: DefaultArgsConfig = DefaultArgsConfig()
    # ui
    ui_window_class_type = BlueROVEnvWindow
    # simulation， 100hz
    sim: SimulationCfg = SimulationCfg(
    render_interval=1,
    render=sim_utils.RenderCfg(
        rendering_mode="quality",
        antialiasing_mode="DLAA",
        enable_shadows=True,
        enable_ambient_occlusion=True,
        enable_reflections=True,
        enable_dl_denoiser=True,
        samples_per_pixel=4,
    ),
)

    # terrain = TerrainCfg()
    decimation = 20  # Number of control action updates @ sim dt per policy dt.
    episode_length_s = 4.0  # episode time limit in seconds
    # env
    observation_space: gym.spaces.Space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(23,), dtype=np.float64
    )
    action_space: gym.spaces.Space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(14,), dtype=np.float64
    )
    # policy(23) + predictor rollout
    state_space: gym.spaces.Space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(26 + params.predictor_out_steps * 17,), dtype=np.float64
    )
    # Domain randomization
    events: EventCfg = EventCfg()
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = (
        NoiseModelWithAdditiveBiasCfg(
            noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
            bias_noise_cfg=GaussianNoiseCfg(
                mean=0.0, std=0.0001, operation="abs"),
        )
    )

    # scene , clone_in_fabric=True
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4, env_spacing=4.0, replicate_physics=True
    )
    # robot
    robot_cfg: ArticulationCfg = BLUEROV_ARTICULATION_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )


class BlueROVEnv(DirectRLEnv):
    cfg: BlueROVEnvCfg

    def __init__(self, cfg: BlueROVEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Debug mode
        self._debug = self.cfg.params.debug_out
        self.idx = 0
        # Initialize buffers
        self._actions = torch.zeros(
            self.num_envs, 6, device=self.device
        )  
        self._prev_actions = torch.zeros(
            self.num_envs, 6, device=self.device
        )  
        self._thrust = torch.zeros(
            self.num_envs, 1, 3, device=self.device)  # n,1,3
        self._moment = torch.zeros(
            self.num_envs, 1, 3, device=self.device)  # n,1,3
        self.first_actions = torch.zeros(
            self.num_envs, 6, device=self.device
        )  
        self.old_actions = torch.zeros(
            self.num_envs, 6, device=self.device
        )  
        # n,13 (pos,ori,linvel,angvel)
        self.robot_spawn_state = self._robot.data.default_root_state.clone()
        # n,3 (x,y,z)self._robot.data.default_root_state[:, :3]
        # print("env_origins: ", self.scene.env_origins)
        self._default_env_origins = self.scene.env_origins.clone().to(self.device)
        # n, used for stay alive in rewards
        self._completed_envs = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # n,3 (x,y,z)
        self._desired_pos_w = (
            self._robot.data.default_root_state[:,
                                                :3] + self._default_env_origins
        )
        self.robot_default_state = self._robot.data.default_root_state.clone()
        self.robot_default_state[:, :3] += self._default_env_origins
        # print("Default desired positions: ", self._desired_pos_w)
        # desired goal quat_w in world frame [w,x,y,z]
        self._desired_quat_w = torch.zeros(
            self.num_envs, self.cfg.params.goal_dims, device=self.device
        )  # n,4

        torch.manual_seed(0)

        if self.cfg.params.eval_mode:
            print("Setting manual seed")
            torch.manual_seed(0)

        # Debug visualization
        self.set_debug_vis(self.cfg.params.debug_vis)

        if self._debug:
            print("mass: ", self._robot.root_physx_view.get_masses()
                  [0].sum().item())
            print(
                "com: ",
                self._robot.root_physx_view.get_coms()[:, 0, :3].cpu().numpy(),
            )

        # Get specific information about the AUV
        self._gravity_magnitude = torch.tensor(
            self.sim.cfg.gravity, device=self.device
        ).norm()
        self.masses = torch.full(
            (self.num_envs, 1), self.cfg.params.mass, device=self.device
        )
        self.inertia_const = torch.tensor(
            self.cfg.params.inertia_const_list, device=self.device
        )
        self.inertia_tensors = self.masses * self.inertia_const
        print("masses: ", self.masses)
        print("inertia_tensors: ", self.inertia_tensors)
        self.com_to_cob_offsets = torch.tensor(
            self.cfg.params.com_to_cob_offset, device=self.device
        ).repeat(self.num_envs, 1)
        self.volumes = torch.full(
            (self.num_envs, 1), self.cfg.params.volume, device=self.device
        )
        
        self._body_id = self._robot.find_bodies("base_link")[0]
        self.timer = Timer(msg="[Debug Timer]")
        
        self.dt = torch.tensor(self.sim.cfg.dt, device=self.device)
        
        controller_dt = getattr(
            self.cfg.params,
            "controller_dt",
            self.sim.cfg.dt * self.cfg.params.controller_interval,
        )
        self.controller_dt = torch.tensor(controller_dt, device=self.device)
        # thruster and hydrodynamic models
        self.thruster = Thruster(
            numEnvs=self.num_envs, cfg=self.cfg.params.thrusterCfg, device=self.device
        )
        if self.cfg.params.hydrodynamics_model == "FossenModel":
            print("[Info] Using Fossen Hydrodynamic Model")
            self.hydrodynamic = FossenForceModels(
                self.num_envs, self.device, self.dt, self._debug
            )
        if self.cfg.params.hydrodynamics_model == "InertiaModel":
            print("[Info] Using Inertia Hydrodynamic Model")
            self.hydrodynamic = InertiaForceModels(
                self.num_envs, self.device, self.dt, self._debug
            )
        
        fossen_param = self._build_fossen_param()
        # n,6
        k1_init = torch.tensor(self.cfg.params.k1_init, device=self.device).repeat(
            self.num_envs, 1
        )  # n,6
        # n,6
        k2_init = torch.tensor(self.cfg.params.k2_init, device=self.device).repeat(
            self.num_envs, 1
        )  # n,6
        # n,1
        ka_init = torch.tensor(self.cfg.params.ka_init, device=self.device).repeat(
            self.num_envs, 1
        )  # n,1
        # n,1
        kc_init = torch.tensor(self.cfg.params.kc_init, device=self.device).repeat(
            self.num_envs, 1
        )  # n,1
        
        self.obppo_gains_init = torch.cat(
            [k1_init, k2_init, ka_init, kc_init], dim=-1
        )  # n,14
        
        self._prev_gains = self.obppo_gains_init.clone()
        self.obac_module = OptimizedBacksteppingACController(
            num_envs=self.num_envs,
            device=self.device,
            fossenParams=fossen_param,
            ObppoGains=self.obppo_gains_init,
            mode=self.cfg.params.mode,
        ).to(self.device)
        self._latest_actor_weights = None
        self._latest_critic_weights = None
        self._obac_weights_saved = False
        
        self.predictor_out_steps = int(self.cfg.params.predictor_out_steps)
        self.obac_predictor = None
        if self.predictor_out_steps > 0:
            self.obac_predictor = OBACPredictor(
                num_envs=self.num_envs,
                device=self.device,
                fossenParams=fossen_param,
                obac_gains=self.obppo_gains_init,
                predict_steps=self.predictor_out_steps,
            ).to(self.device)
            
        
        self.disturb_thrust = torch.zeros(
            self.num_envs, 1, 3, device=self.device)  # n,1,3
        self.disturb_moment = torch.zeros(
            self.num_envs, 1, 3, device=self.device)  # n,1,3
        
        self.current_velocity = torch.zeros(
            self.num_envs, 1, 3, device=self.device)  # n,1,3
        
        self.current_direction = torch.zeros(
            self.num_envs, 1, 3, device=self.device)  # n,1,3
        
        self.current_velocity_c = torch.zeros(
            self.num_envs, 1, 1, device=self.device)  # n,1,1
        
        self.current_mean_velocity = torch.zeros(
            self.num_envs, 1, 1, device=self.device)  # n,1,1
        
        self.gust_current = torch.zeros(
            self.num_envs, 1, 3, device=self.device)  # n,1,3
        self.gust_duration = torch.zeros(
            self.num_envs, 1, 1, device=self.device)  # n,1,1
        
        self.have_current_mask = torch.ones(
            self.num_envs, device=self.device, dtype=torch.bool) # n,
        
        
        self._cumulate_rewards = torch.zeros(
            self.num_envs, device=self.device)  # n,


    def _build_fossen_param(self) -> FossenParam:
        """Read randomized physical parameters from the env and pack into FossenParam."""
        return FossenParam(
            m=self.masses.clone(),
            I=self.inertia_tensors.clone(),
            cg=self.com_to_cob_offsets.clone(),
            volume=self.volumes.clone(),
            dt=self.controller_dt,
        )

    def _setup_scene(self):
        self.cfg.robot_cfg.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, self.cfg.params.starting_depth), rot=(0, 1, 0, 0)
        )
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        seabed_usd_path = os.path.join(
            os.path.dirname(__file__), "assets", "bluerov", "seabed.usd"
        )
        seabed_scale = (0.01, 0.01, 0.006)
        seabed_quat = math_utils.quat_from_euler_xyz(
            torch.tensor(0.5 * np.pi),
            torch.tensor(0.0),
            torch.tensor(0.0),
        ).tolist()
        seabed_cfg = sim_utils.UsdFileCfg(
            usd_path=seabed_usd_path,
            scale=seabed_scale,
        )
        seabed_cfg.func("/World/Seabed", seabed_cfg, orientation=seabed_quat)
        # spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0, color=(0.15, 0.35, 0.55))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:        
        # actions: n,14, a[0:6] k1; a[6:12] k2; a[12] kappa_a; a[13] kappa_c
        gains = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(
            -1.0, 1.0
        )
        
        gains = self._prev_gains * (1 - self.cfg.params.gamma) + self.cfg.params.gamma * \
            self.obppo_gains_init * (1 + self.cfg.params.alpha * gains)
        
        gains[:, 0:6].clamp_(min=self.cfg.params.k1_min)
        gains[:, 6:12].clamp_(min=self.cfg.params.k2_min)
        kappa_a = gains[:, 12].clamp(min=1e-6)
        lower = 0.5 * kappa_a
        kappa_c = torch.clamp(gains[:, 13], min=lower, max=kappa_a)
        gains[:, 12] = kappa_a
        gains[:, 13] = kappa_c
        gains.nan_to_num_(nan=1.0, posinf=1.0e6, neginf=1.0e-6)
        self._prev_gains = gains.clone()
        self.obac_module.update_gain(gains)
        if self.obac_predictor is not None:
            self.obac_predictor.update_gain(gains)
        if self._debug and False:
            print("original actions vec: ", actions)
            print("concatenated actions shape: ", self._actions)
        self.first_actions = None
        
        self._cumulate_rewards.zero_()
        # print("[Debug] Updated gains: ", gains)

    def _apply_action(self) -> None:
        'Run one controller update per physics step and one policy update per decimation window.'
        if self._sim_step_counter % self.cfg.params.controller_interval == 0:
            
            if self.cfg.events:
                if "interval" in self.event_manager.available_modes:
                    self.event_manager.apply(mode="interval", dt=self.step_dt)
            # self.sim.render()
            obs = self._get_obppo_observations()
            
            self._actions, info = self.obac_module(obs)
            # actor_weights = info.get("actor_weights")
            # critic_weights = info.get("critic_weights")
            
            # v2 = info.get("V2")
            # tau_cost = info.get("tau_cost")
            # e2_cost = info.get("e2_cost")
            # print("[Debug] actor_weights: ", actor_weights)
            # print("[Debug] critic_weights: ", critic_weights)
            # print("[Debug] V2: ", v2)
            # print("[Debug] tau_cost: ", tau_cost)
            # print("[Debug] e2_cost: ", e2_cost)
            # print("[Debug] V2_data: ", v2.mean(), v2.max(), v2.min())
            self._record_obac_info(info)
            self._actions.nan_to_num_(nan=0.0, posinf=1.1, neginf=-1.1).clamp_(
                -1.0, 1.0
            )
            # print("[DEBUG] cumulate rewards before: ", self._cumulate_rewards)
            self._cumulate_rewards += self._get_obppo_rewards()
            # print("[DEBUG] cumulate rewards: ", self._cumulate_rewards)
            
            if self.first_actions is None:
                self.first_actions = self._actions.clone()
                self._cumulate_rewards += self._get_obppo_rewards()
        
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(
            self._actions
        )
        # FRD Frame forces and torques
        self._body_id = self._robot.find_bodies("base_link")[0]
        # print("[DEBUG] self._body_id: ", self._body_id)
        
        # print("[DEBUG] Random disturbance thrust: ", self.disturb_thrust, " moment: ", self.disturb_moment)
        self._robot.set_external_force_and_torque(
            self._thrust+self.disturb_thrust,
            self._moment+self.disturb_moment,
            positions=torch.zeros_like(self._thrust),
            body_ids=self._body_id,
            is_global=False,
        )
        self.disturb_thrust.zero_()
        self.disturb_moment.zero_()

    def _record_obac_info(self, info: dict) -> None:
        log = self.extras.setdefault("log", {})
        actor_weights = info.get("actor_weights")
        critic_weights = info.get("critic_weights")
        v2 = info.get("V2")

        if isinstance(actor_weights, torch.Tensor):
            actor_weights = actor_weights.detach()
            self._latest_actor_weights = actor_weights
            log["obac/actor_weights_mean"] = actor_weights.mean()
            log["obac/actor_weights_norm"] = actor_weights.norm()
        if isinstance(critic_weights, torch.Tensor):
            critic_weights = critic_weights.detach()
            self._latest_critic_weights = critic_weights
            log["obac/critic_weights_mean"] = critic_weights.mean()
            log["obac/critic_weights_norm"] = critic_weights.norm()
        if isinstance(actor_weights, torch.Tensor) and isinstance(critic_weights, torch.Tensor):
            log["obac/weights_gap_norm"] = (actor_weights - critic_weights).norm()
        if isinstance(v2, torch.Tensor):
            v2 = v2.detach()
            log["obac/V2_mean"] = v2.mean()
            log["obac/V2_max"] = v2.max()

    @staticmethod
    def _weights_to_yaml_data(weights):
        if isinstance(weights, torch.Tensor):
            return weights.detach().cpu().tolist()
        return weights

    def _save_obac_weights(self) -> None:
        if self._obac_weights_saved:
            return
        self._obac_weights_saved = True

        if self._latest_actor_weights is not None:
            os.makedirs(os.path.dirname(ACTOR_WEIGHTS_SAVE_PATH), exist_ok=True)
            with open(ACTOR_WEIGHTS_SAVE_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {"actor_weights": self._weights_to_yaml_data(self._latest_actor_weights)},
                    f,
                    sort_keys=False,
                )

        if self._latest_critic_weights is not None:
            os.makedirs(os.path.dirname(CRITIC_WEIGHTS_SAVE_PATH), exist_ok=True)
            with open(CRITIC_WEIGHTS_SAVE_PATH, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {"critic_weights": self._weights_to_yaml_data(self._latest_critic_weights)},
                    f,
                    sort_keys=False,
                )

    def close(self) -> None:
        # try:
        self._save_obac_weights()
        # finally:
        super().close()

    def _get_obppo_observations(self) -> dict:
        desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
            self._desired_quat_w,
        )
        # 17
        obs = torch.cat(
            [
                desired_quat_b,
                desired_pos_b,
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
            ],
            dim=-1,
        )
        return {"obppo": obs}
    def _get_observations(self) -> dict:
        desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
            self._desired_quat_w,
        )
        # 23
        obs = torch.cat(
            [
                desired_quat_b,  
                desired_pos_b,  
                self._robot.data.root_quat_w,  
                self._robot.data.root_lin_vel_b,  
                self._robot.data.root_ang_vel_b,  
                self.old_actions,  
            ],
            dim=-1,
        )
        obs.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        
        observations = {"policy": obs}
        if self.obac_predictor is not None:
            predictor_obs = self.obac_predictor.predict_from_observation(obs[:, :17]).flatten(1)
            predictor_obs.nan_to_num_(nan=0.0, posinf=_OBS_CLIP, neginf=-_OBS_CLIP)
            predictor_obs.clamp_(-_OBS_CLIP, _OBS_CLIP)
            observations["predictor"] = predictor_obs
        else:
            observations["predictor"] = obs[:, :0]
        observations["others"] = self.current_velocity.view(self.num_envs, 3)
        return observations


    def _get_obppo_rewards(self) -> torch.Tensor:
        total_reward = _compute_rewards(
            self.cfg.params.rew_scale_pos,
            self.cfg.params.rew_scale_ang,
            self.cfg.params.rew_scale_action_smooth,
            self.cfg.params.rew_scale_action_mag,
            self.cfg.params.rew_scale_lin_vel,
            self.cfg.params.rew_scale_ang_vel,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._desired_pos_w,
            self._desired_quat_w,
            self._actions,
            self._prev_actions,
        )
        self._prev_actions = self._actions.clone()
        return total_reward
        
    def _get_rewards(self) -> torch.Tensor:
        total_reward = _compute_rewards(
            self.cfg.params.rew_scale_pos,
            self.cfg.params.rew_scale_ang,
            self.cfg.params.rew_scale_action_smooth,
            self.cfg.params.rew_scale_action_mag,
            self.cfg.params.rew_scale_lin_vel,
            self.cfg.params.rew_scale_ang_vel,
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._robot.data.root_lin_vel_b,
            self._robot.data.root_ang_vel_b,
            self._desired_pos_w,
            self._desired_quat_w,
            self.first_actions,
            self.old_actions,
        )

        if self._debug and False:
            print("=" * 40)
            print("Min reward:", total_reward.min().item())

            self.idx = torch.where(mask)[0][0]  

            print("Details:")
            print("Lin Vel reward:", rew_lin_vel[self.idx].item())
            print("Ang Vel reward:", rew_ang_vel[self.idx].item())
            print("Lin Vel:",
                  self._robot.data.root_lin_vel_b[self.idx].cpu().numpy())
            print("Ang Vel:",
                  self._robot.data.root_ang_vel_b[self.idx].cpu().numpy())
            print("Mass:", self._robot.root_physx_view.get_masses()
                  [self.idx].sum())
            print("Volume:", self.volumes[self.idx].item())
            print("Inertia Tensor:",
                  self.inertia_tensors[self.idx].cpu().numpy())
            print(
                "COM to COB offset:",
                self._robot.data.com_pos_b[self.idx, 0].cpu().numpy(),
            )
            print("Action:", self._actions[self.idx].cpu().numpy())
            print("Old Action:", self.old_actions[self.idx].cpu().numpy())
            print(
                "Set Wrench (thrust & moment):",
                self._thrust[self.idx, 0, :].cpu().numpy(),
                self._moment[self.idx, 0, :].cpu().numpy(),
            )
            print("=" * 40)
            self.print_force = True
        
        self.old_actions = self._actions.clone()
        # print("Rot error (deg): ", rot_error * 180.0 / 3.1415926)
        # print("[DEBUG] ",total_reward)
        # print("[DEBUG] self._cumulate_rewards: ", self._cumulate_rewards)
        total_reward += self._cumulate_rewards
        # print("[DEBUG] total_reward + cumulate_rewards: ", total_reward)
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.params.cap_episode_length:
            time_out = self.episode_length_buf >= self.max_episode_length - 1
        else:
            time_out = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self.cfg.params.use_boundaries:
            pos = self._robot.data.root_pos_w
            origin = self.scene.env_origins
            out_of_bounds = (
                (torch.abs(pos[:, 0] - origin[:, 0])
                 > self.cfg.params.max_bound_x)
                | (torch.abs(pos[:, 1] - origin[:, 1]) > self.cfg.params.max_bound_y)
                | (
                    torch.abs(pos[:, 2] - self.cfg.params.starting_depth)
                    > self.cfg.params.max_bound_z
                )
            )
        else:
            out_of_bounds = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        root_state = self._robot.data.root_state_w
        invalid_state = (
            ~torch.isfinite(root_state).all(dim=-1)
            | ~torch.isfinite(self._robot.data.root_lin_vel_b).all(dim=-1)
            | ~torch.isfinite(self._robot.data.root_ang_vel_b).all(dim=-1)
        )

        terminated = out_of_bounds | invalid_state
        truncated = time_out & (~terminated)
        return terminated, truncated

    def _random_volumes_inertial(
        self, env_ids: Sequence[int] | None, ratio_range: tuple[float, float]
    ):
        buoyancy_ratios = math_utils.sample_uniform(
            *ratio_range, (len(env_ids), 1), device=self.device
        )
        self.volumes[env_ids] = (
            buoyancy_ratios * self.masses[env_ids] /
            self.cfg.params.fluid_density
        )
        self.inertia_tensors[env_ids] = self.masses[env_ids] * \
            self.inertia_const

    def _reset_idx(self, env_ids: Sequence[int] | None):
        'Reset randomized dynamics, targets, reached states, propellers, and controller parameters.'
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        env_ids_cpu = env_ids.cpu()
        self._robot.reset(env_ids)

        # self._robot.write_root_state_to_sim(self.robot_default_state[env_ids], env_ids)
        super()._reset_idx(env_ids)
        
        self.com_to_cob_offsets[env_ids] = self._robot.data.com_pos_b[env_ids, 0]
        self.masses[env_ids] = (
            self._robot.root_physx_view.get_masses()[env_ids_cpu]
            .sum(dim=1, keepdim=True)
            .to(device=self.device)
        )
        
        self._random_volumes_inertial(
            env_ids_cpu, self.cfg.params.volume_ratio_range)

        if self._debug and False:
            # self.timer.start()
            # print(
            #     "mass after randomization: ",
            #     self._robot.root_physx_view.get_masses()[env_ids_cpu].sum(
            #         dim=1, keepdim=True
            #     ),
            # )
            # self.timer.stop()
            # print("[Timer] get_masses time: ", self.timer.total_run_time)
            self.timer.start()
            print(
                "com after randomization: ",
                self._robot.data.com_pos_b[env_ids, 0],
            )
            self.timer.stop()
            print("[Timer] get_coms time: ", self.timer.total_run_time)
            # self.timer.start()
            # print(self.robot_default_state[env_ids])
            # print(
            #     "state after randomization: ",
            #     self._robot.data.root_state_w[env_ids],
            # )

            # self.timer.stop()
            # print("[Timer] get_state time: ", self.timer.total_run_time)
            # self.timer.start()
            # print(
            #     "Volumes after randomization: ",
            #     self.volumes[env_ids_cpu],
            # )
            # self.timer.stop()
            # print("[Timer] get_volumes time: ", self.timer.total_run_time)
            # self.timer.start()
            # print(
            #     "Inertial after randomization: ",
            #     self.inertia_tensors[env_ids_cpu],
            # )
            # self.timer.stop()
            # print("[Timer] get_Inertial time: ", self.timer.total_run_time)
        
        self._desired_quat_w[env_ids, 0:4] = math_utils.random_orientation(
            len(env_ids), device=self.device
        )
        
        if not self.cfg.params.eval_mode:
            envs_to_guide = (
                math_utils.sample_uniform(0, 1, len(env_ids), self.device)
                < self.cfg.params.init_guidance_rate
            )
            env_ids_to_guide = env_ids[envs_to_guide]
            self.robot_spawn_state[env_ids_to_guide, :3] = self._desired_pos_w[
                env_ids_to_guide
            ]
            self.robot_spawn_state[env_ids_to_guide, 3:7] = self._desired_quat_w[
                env_ids_to_guide, 0:4
            ]
            self._robot.write_root_state_to_sim(
                self.robot_spawn_state[env_ids_to_guide], env_ids_to_guide
            )
            
            self.have_current_mask[env_ids] = True
            envs_no_current= (
                    math_utils.sample_uniform(0, 1, len(env_ids), self.device)
                    < self.cfg.params.no_ocean_current
                )
            env_ids_no_current = env_ids[envs_no_current]
            self.have_current_mask[env_ids_no_current] = False
            # print(f"[Info] {len(env_ids_no_current)} envs have no ocean current.")
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(
            joint_pos, joint_vel, None, env_ids)
        self.hydrodynamic.reset(env_ids_cpu)

        
        self.obac_module.reset(self._build_fossen_param(),env_ids)
        if self.obac_predictor is not None:
            self.obac_predictor.reset(self._build_fossen_param(), self._prev_gains)
        
        self.old_actions[env_ids] = torch.zeros(
            len(env_ids), 6, device=self.device)
        self.first_actions[env_ids] = torch.zeros(
            len(env_ids), 6, device=self.device)
        self._actions[env_ids] = torch.zeros(
            len(env_ids), 6, device=self.device)
        self._prev_actions[env_ids] = torch.zeros(
            len(env_ids), 6, device=self.device)
        self.disturb_thrust[env_ids] = torch.zeros(len(env_ids), 1, 3, device=self.device)
        self.disturb_moment[env_ids] = torch.zeros(len(env_ids), 1, 3, device=self.device)
        
        
    def _compute_dynamics(self, actions):
        """
        Args:
            actions (torch.Tensor): Actions shape (num_envs, num_actions)

        Returns:
            [torch.Tensor]: Forces sent to the simulation frame 0 points n,n_thrusters,3
            [torch.Tensor]: Torques sent to the simulation
        """

        # if self._debug and True:
        if False:
            print("-" * 30)
            print("actions: ", actions)

        wrench_b = torch.zeros(
            (self.num_envs, 6), device=self.device, dtype=torch.float
        )
        thrustForce = torch.zeros(
            (self.num_envs, 8), device=self.device, dtype=torch.float
        )
        # at this point these are PWM commands between -1 and 1
        _inputCommand = actions.clone()
        _inputCommand.nan_to_num_(nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)

        # get the current motor velocities using thruster dynamics
        
        
        
        # thrustForce: n,8; wrench_b: n,6
        thrustForce, wrench_b = self.thruster.update(
            _inputCommand, self.dt, debug=False, type="wrench"
        )
        # Calculate hydrodynamics
        buoyancy_forces, buoyancy_torques = self.hydrodynamic.calculate_buoyancy_forces(
            self._robot.data.root_quat_w,
            self.cfg.params.fluid_density,
            self.volumes.to(device=self.device),
            abs(self._gravity_magnitude),
        )
        
        gust_mask = (self.gust_duration > 0.0).to(self.gust_current.dtype)
        current_velocity = self.current_velocity + self.gust_current * gust_mask
        self.gust_duration.sub_(self.dt).clamp_(min=0.0)
        current_velocity[~self.have_current_mask]=0
        # print("[Info] current_velocity: ", current_velocity)
        hydro_dynamic_F, hydro_dynamic_T = (
            self.hydrodynamic.calculate_density_and_viscosity_forces(
                self._robot.data.root_quat_w,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self.inertia_tensors,
                self.cfg.params.water_beta,
                self.cfg.params.fluid_density,
                self.masses,
                self.com_to_cob_offsets,
                current_velocity,
            )
        )
        # if True:
        #     hydro_dynamic_f, hydro_dynamic_t = (
        #     self.hydrodynamic.calculate_density_and_viscosity_forces(
        #         self._robot.data.root_quat_w,
        #         self._robot.data.root_lin_vel_b,
        #         self._robot.data.root_ang_vel_b,
        #         self.inertia_tensors,
        #         self.cfg.params.water_beta,
        #         self.cfg.params.fluid_density,
        #         self.masses,
        #         self.com_to_cob_offsets,
        #         torch.zeros_like(current_velocity),
        #     )
        # )
        #     print("=" * 60)
        #     print("current_velocity: ", current_velocity)
        #     print("gust_current: ", self.gust_current)
        #     print("gust_mask: ", gust_mask)
        #     print("current direction: ", self.current_direction)
        #     print("current_velocity_c: ", self.current_velocity_c)
        #     print("hydro_dynamic_F: ", hydro_dynamic_F)
        #     print("hydro_dynamic_T: ", hydro_dynamic_T)
        #     print("hydro_dynamic_f: ", hydro_dynamic_f)
        #     print("hydro_dynamic_t: ", hydro_dynamic_t)

        # n,3, base_link frame cob as the origin
        forces = hydro_dynamic_F + buoyancy_forces + wrench_b[:, :3]
        torques = hydro_dynamic_T + buoyancy_torques + wrench_b[:, 3:]
        forces.nan_to_num_(nan=0.0, posinf=_WRENCH_CLIP, neginf=-_WRENCH_CLIP).clamp_(
            -_WRENCH_CLIP, _WRENCH_CLIP
        )
        torques.nan_to_num_(nan=0.0, posinf=_WRENCH_CLIP, neginf=-_WRENCH_CLIP).clamp_(
            -_WRENCH_CLIP, _WRENCH_CLIP
        )
        # important All forces and torques are in the FRD body frame
        if self._debug and True:
            print("=-" * 20)
            print("v: ", self._robot.data.root_lin_vel_b)
            print("omega: ", self._robot.data.root_ang_vel_b)
            # print("damping: ", damping)
            # print("added mass term: ", added_mass_term)
            # print("coriolis term: ", coriolis_term)
            # print("nu_dot: ", nu)
            print("buoyancy forces: ", buoyancy_forces)
            print("buoyancy torques: ", buoyancy_torques)
            print("hydrodynamic forces: ", hydro_dynamic_F)
            print("hydrodynamic torques: ", hydro_dynamic_T)
            print("FRD thruster forces: ", wrench_b[:, :3])
            print("FRD thruster torques: ", wrench_b[:, 3:])
            print("final forces", forces)
            print("final torques", torques)

        return forces, torques

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = SPHERE_MARKER_CFG.copy()
                marker_cfg.markers["sphere"].radius = 0.05
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_x_ang_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_x_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_x_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "goal_z_ang_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/goal_z_ang"
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                self.goal_z_ang_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "x_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/x_b"
                self.x_b_visualizer = VisualizationMarkers(marker_cfg)

            if not hasattr(self, "z_b_visualizer"):
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (0.125, 0.125, 1)
                marker_cfg.prim_path = "/Visuals/Command/z_b"
                self.z_b_visualizer = VisualizationMarkers(marker_cfg)

            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.goal_x_ang_visualizer.set_visibility(True)
            self.goal_z_ang_visualizer.set_visibility(True)
            self.x_b_visualizer.set_visibility(True)
            self.z_b_visualizer.set_visibility(True)

        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

            if hasattr(self, "goal_x_ang_visualizer"):
                self.goal_x_ang_visualizer.set_visibility(False)

            if hasattr(self, "goal_z_ang_visualizer"):
                self.goal_z_ang_visualizer.set_visibility(False)

            if hasattr(self, "x_b_visualizer"):
                self.x_b_visualizer.set_visibility(False)

            if hasattr(self, "z_b_visualizer"):
                self.z_b_visualizer.set_visibility(False)

    def _rotate_quat_by_euler_xyz(
        self,
        q: torch.tensor,
        x: float | torch.tensor,
        y: float | torch.tensor,
        z: float | torch.tensor,
        device=None,
    ):
        # Assumes q has shape [num_envs, 4]
        num_envs = q.shape[0]
        if device == None:
            device = self.device

        if type(x) == float:
            x = torch.zeros(num_envs, device=device) + x

        if type(y) == float:
            y = torch.zeros(num_envs, device=device) + y

        if type(z) == float:
            z = torch.zeros(num_envs, device=device) + z

        iq = math_utils.quat_from_euler_xyz(x, y, z)
        return math_utils.quat_mul(q, iq)

    def _debug_vis_callback(self, event):
        # Visualize the goal positions
        self.goal_pos_visualizer.visualize(translations=self._desired_pos_w)

        ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        ang_marker_scales[:, 0] = 1
        self.goal_x_ang_visualizer.visualize(
            translations=self._desired_pos_w,
            orientations=self._desired_quat_w,
            scales=ang_marker_scales,
        )

        # Visualize goal orientations via another axis
        goal_z_quat = self._rotate_quat_by_euler_xyz(
            self._desired_quat_w, 0.0, -torch.pi / 2, 0.0
        )
        ang_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        ang_marker_scales[:, 0] = 1
        self.goal_z_ang_visualizer.visualize(
            translations=self._desired_pos_w,
            orientations=goal_z_quat,
            scales=ang_marker_scales,
        )

        # Visualize current X-direction
        x_w = self._robot.data.root_quat_w
        x_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        x_w_marker_scales[:, 0] = 1
        self.x_b_visualizer.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=x_w,
            scales=x_w_marker_scales,
        )

        # Visualize current Z-direction
        z_w_quat = self._rotate_quat_by_euler_xyz(
            self._robot.data.root_quat_w, 0.0, -torch.pi / 2, 0.0
        )
        z_w_marker_scales = torch.tensor([1, 1, 1]).repeat(self.num_envs, 1)
        z_w_marker_scales[:, 0] = 1
        self.z_b_visualizer.visualize(
            translations=self._robot.data.root_pos_w,
            orientations=z_w_quat,
            scales=z_w_marker_scales,
        )
    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
            """Execute one time-step of the environment's dynamics.

            The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
            lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
            independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
            and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
            time-step is computed as the product of the two.

            This function performs the following steps:

            1. Pre-process the actions before stepping through the physics.
            2. Apply the actions to the simulator and step through the physics in a decimated manner.
            3. Compute the reward and done signals.
            4. Reset environments that have terminated or reached the maximum episode length.
            5. Apply interval events if they are enabled.
            6. Compute observations.

            Args:
                action: The actions to apply on the environment. Shape is (num_envs, action_dim).

            Returns:
                A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
            """
            action = action.to(self.device)
            # add action noise
            if self.cfg.action_noise_model:
                action = self._action_noise_model(action)

            # process actions
            self._pre_physics_step(action)

            # check if we need to do rendering within the physics loop
            # note: checked here once to avoid multiple checks within the loop
            is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

            # perform physics stepping
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                # set actions into buffers
                self._apply_action()
                # set actions into simulator
                self.scene.write_data_to_sim()
                # simulate
                self.sim.step(render=False)
                # render between steps only if the GUI or an RTX sensor needs it
                # note: we assume the render interval to be the shortest accepted rendering interval.
                #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()
                # update buffers at sim dt
                self.scene.update(dt=self.physics_dt)

            # post-step:
            # -- update env counters (used for curriculum generation)
            self.episode_length_buf += 1  # step in current episode (per env)
            self.common_step_counter += 1  # total step (common for all envs)

            self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
            self.reset_buf = self.reset_terminated | self.reset_time_outs
            self.reward_buf = self._get_rewards()

            # -- reset envs that terminated/timed-out and log the episode information
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self._reset_idx(reset_env_ids)
                # if sensors are added to the scene, make sure we render to reflect changes in reset
                if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                    self.sim.render()

            # post-step: step interval event
            # if self.cfg.events:
            #     if "interval" in self.event_manager.available_modes:
            #         self.event_manager.apply(mode="interval", dt=self.step_dt)

            # update observations
            self.obs_buf = self._get_observations()

            # add observation noise
            # note: we apply no noise to the state space (since it is used for critic networks)
            if self.cfg.observation_noise_model:
                self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

            # return observations, rewards, resets and extras
            return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


@torch.jit.script
def quat_dist(q1, q2):
    return 1 - torch.sum(q1 * q2, dim=-1) ** 2


# @torch.jit.script
def _compute_rewards(
    rew_scale_pos: float,
    rew_scale_ang: float,
    rew_scale_action_smooth: float,
    rew_scale_action_mag: float,
    rew_scale_ang_vel: float,
    rew_scale_lin_vel: float,
    robot_pos_w: torch.Tensor,
    robot_quat_w: torch.Tensor,
    lin_vel_b: torch.Tensor,
    ang_vel_b: torch.Tensor,
    target_pos_w: torch.Tensor,
    target_quat_w: torch.Tensor,
    actions: torch.Tensor,
    old_actions: torch.Tensor,
) -> torch.Tensor:

    distance = torch.norm(target_pos_w - robot_pos_w, p=2, dim=-1)
    rew_pos = rew_scale_pos * torch.exp(-distance / 0.5)  # sigma=0.5
    
    rot_error = math_utils.quat_error_magnitude(robot_quat_w, target_quat_w)
    rew_orient = rew_scale_ang * torch.exp(-rot_error / 0.2)  # sigma=0.2
    
    delta_action = torch.norm(actions - old_actions, p=2, dim=-1)
    rew_action_rate = -(rew_scale_action_smooth * delta_action**2)

    action_mag = torch.norm(actions, p=2, dim=-1)
    rew_action_mag = -(rew_scale_action_mag * action_mag**2)
    
    rew_ang_vel = -rew_scale_ang_vel * \
        torch.sum(torch.square(ang_vel_b), dim=-1)
    rew_lin_vel = -rew_scale_lin_vel * \
        torch.sum(torch.square(lin_vel_b), dim=-1)

    
    total_rew = (
        rew_pos
        + rew_orient
        + rew_action_rate
        + rew_action_mag
        + rew_ang_vel
        + rew_lin_vel
    )

    return total_rew

