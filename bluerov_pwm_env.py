'Isaac Lab BlueROV environment with direct eight-thruster control.'

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


from isaaclab.utils.noise import (
    GaussianNoiseCfg,
    NoiseModelWithAdditiveBiasCfg,
)
from .hydrodynamics import InertiaForceModels, FossenForceModels

from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg, TerrainGenerator
from .config.pwm_config import DefaultArgsConfig, EventCfg
from .thrusterModels.thruster import Thruster


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
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=2)
    # terrain = TerrainCfg()
    # general
    decimation = 2  # Number of control action updates @ sim dt per policy dt.
    episode_length_s = 4.0  # episode time limit in seconds
    # env
    observation_space: gym.spaces.Space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64
    )
    action_space: gym.spaces.Space = gym.spaces.Box(
        low=-1.0, high=1.0, shape=(8,), dtype=np.float64
    )
    state_space: gym.spaces.Space = gym.spaces.Box(
        low=-np.inf, high=np.inf, shape=(25,), dtype=np.float64
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
            bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
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
        self._actions = torch.zeros(self.num_envs, 8, device=self.device)  # n,8
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)  # n,1,3
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)  # n,1,3
        self.old_actions = torch.zeros(self.num_envs, 8, device=self.device)
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
            self._robot.data.default_root_state[:, :3] + self._default_env_origins
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
            print("mass: ", self._robot.root_physx_view.get_masses()[0].sum().item())
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
        print("running in: ", self.device)
        print("masses: ", self.masses)
        print("inertia_tensors: ", self.inertia_tensors)
        self.com_to_cob_offsets = torch.tensor(
            self.cfg.params.com_to_cob_offset, device=self.device
        ).repeat(self.num_envs, 1)
        self.volumes = torch.full((self.num_envs, 1), self.cfg.params.volume, device=self.device)
        
        self._body_id = self._robot.find_bodies("base_link")[0]
        self.timer = Timer(msg="[Debug Timer]")
        self.dt = torch.tensor(self.sim.cfg.dt, device=self.device)
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
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.15, 0.35, 0.55))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions[:] = torch.clip(actions, -1, 1).to(self.device)
        if self._debug and False:
            print("original actions vec: ", actions)
            print("concatenated actions shape: ", self._actions)

    def _apply_action(self) -> None:
        'Apply the action once before the decimated physics steps.'
        self._thrust[:, 0, :], self._moment[:, 0, :] = self._compute_dynamics(
            self._actions
        )
        # FRD Frame forces and torques
        self._body_id = self._robot.find_bodies("base_link")[0]
        self._robot.set_external_force_and_torque(
            self._thrust+self.disturb_thrust,
            self._moment+self.disturb_moment,
            positions=torch.zeros_like(self._thrust),
            body_ids=self._body_id,
            is_global=False,
        )
        self.disturb_thrust.zero_()
        self.disturb_moment.zero_()

    def _get_observations(self) -> dict:
        desired_pos_b, desired_quat_b = math_utils.subtract_frame_transforms(
            self._robot.data.root_pos_w,
            self._robot.data.root_quat_w,
            self._desired_pos_w,
            self._desired_quat_w,
        )
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
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        observations = {"policy": obs}
        return observations

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
            self._actions,
            self.old_actions,
        )

        if self._debug and False:
            print("=" * 40)
            print("Min reward:", total_reward.min().item())

            self.idx = torch.where(mask)[0][0]  

            print("Details:")
            print("Lin Vel reward:", rew_lin_vel[self.idx].item())
            print("Ang Vel reward:", rew_ang_vel[self.idx].item())
            print("Lin Vel:", self._robot.data.root_lin_vel_b[self.idx].cpu().numpy())
            print("Ang Vel:", self._robot.data.root_ang_vel_b[self.idx].cpu().numpy())
            print("Mass:", self._robot.root_physx_view.get_masses()[self.idx].sum())
            print("Volume:", self.volumes[self.idx].item())
            print("Inertia Tensor:", self.inertia_tensors[self.idx].cpu().numpy())
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
                (torch.abs(pos[:, 0] - origin[:, 0]) > self.cfg.params.max_bound_x)
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
            buoyancy_ratios * self.masses[env_ids] / self.cfg.params.fluid_density
        )
        self.inertia_tensors[env_ids] = self.masses[env_ids] * self.inertia_const

    def _reset_idx(self, env_ids: Sequence[int] | None):
        'Reset randomized dynamics, targets, reached states, and propellers.'
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        
        env_ids_cpu = env_ids.cpu()
        self._robot.reset(env_ids)
        self.old_actions[env_ids] = torch.zeros(len(env_ids), 8, device=self.device)
        # self._robot.write_root_state_to_sim(self.robot_default_state[env_ids], env_ids)
        super()._reset_idx(env_ids)
        
        self.com_to_cob_offsets[env_ids] = self._robot.data.com_pos_b[env_ids, 0]
        self.masses[env_ids] = (
            self._robot.root_physx_view.get_masses()[env_ids_cpu]
            .sum(dim=1, keepdim=True)
            .to(self.device)
        )
        
        self._random_volumes_inertial(env_ids, self.cfg.params.volume_ratio_range)

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
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.hydrodynamic.reset(env_ids_cpu)
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

        if self._debug and False:
            print("-" * 30)
            print("actions: ", actions)

        wrench_b = torch.zeros(
            (self.num_envs, 6), device=self.device, dtype=torch.float
        )
        thrustForce = torch.zeros(
            (self.num_envs, 8), device=self.device, dtype=torch.float
        )
        # at this point these are PWM commands between -1 and 1
        _inputCommand = torch.clone(actions)
        
        # thrustForce: n,8; wrench_b: n,6
        thrustForce, wrench_b = self.thruster.update(
            _inputCommand, self.dt, debug=False
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

        # n,3, base_link frame cob as the origin
        forces = hydro_dynamic_F + buoyancy_forces + wrench_b[:, :3]
        torques = hydro_dynamic_T + buoyancy_torques + wrench_b[:, 3:]
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
    rew_pos = rew_scale_pos * torch.exp(-distance / 0.5) 
    
    rot_error = math_utils.quat_error_magnitude(robot_quat_w, target_quat_w)
    rew_orient = rew_scale_ang * torch.exp(-rot_error / 0.2) 

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
