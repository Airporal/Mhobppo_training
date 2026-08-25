from __future__ import annotations
import math
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
from isaaclab.managers import SceneEntityCfg

try:
    from thrusterModels.thruster import ThrusterCfg, Thruster
except ImportError:
    from ..thrusterModels.thruster import ThrusterCfg, Thruster

import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import DirectRLEnv
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, DeformableObject, RigidObject

@configclass
class BlueROVThrusterCfg(ThrusterCfg):
    dynamic_type = "FirstOrder"
    tau = 0.05
    conversion_type = "Basic"
    rotor_constant = 3.9e-6  # N/(rpm^2)
    thruster_configure = "BLUEROV_THRUSTER_CONFIG"


@configclass
class DefaultArgsConfig:
    # super arguments setting here
    # rewards scales
    mode = 2 # 1 training 2 playing
    rew_scale_pos = 4.5
    rew_scale_ang = 2.5
    rew_scale_action_smooth = 0.5
    rew_scale_action_mag = 0.3
    rew_scale_ang_vel = 0.05
    rew_scale_lin_vel = 0.05
    # motion limits
    max_bound_x = 7.0
    max_bound_y = 7.0
    max_bound_z = 7.0
    starting_depth = 8.0
    goal_pos_tolerance = 0.1
    goal_euler_tolerance = 0.174  # ~10 degrees in radians
    # dynamics
    com_to_cob_offset = [
        0.0,
        0.0,
        0.02,
    ]  # in meters, add this (xyz) to COM to get COB location
    fluid_density = 997.0  # kg/m^3
    water_beta = 0.001306  # Pa s, dynamic viscosity of water @ 50 deg F
    # rotor constant used in Gazebo, note /10 because 0.04 is "10x bigger than it should be"
    volume = 0.022747843530591776
    mass = 12.4  # kg
    len_i = 0.7  # m
    len_j = 0.4  # m
    len_k = 0.2  # m
    # I_ii = (1 / 12) * mass * (len_j**2 + len_k**2)
    inertia_const_list = [
        1 / 12 * (len_j**2 + len_k**2),
        1 / 12 * (len_i**2 + len_k**2),
        1 / 12 * (len_i**2 + len_j**2),
    ]
    # inertia_ijk = mass * inertia_const
    # Flags
    debug_vis = True
    debug_out = False
    cap_episode_length = True
    use_boundaries = True
    eval_mode = False
    # dynamics
    thrusterCfg: ThrusterCfg = BlueROVThrusterCfg()
    hydrodynamics_model = "FossenModel"  # InertiaModel, FossenModel
    # others
    init_guidance_rate = 0.1
    no_ocean_current = 0.5
    goal_dims = 4
    volume_ratio_range = (0.95, 1.05)
    
    k1_init = [7.5, 7.5, 7.5, 7.5, 7.5, 7.5]
    k1_min = 0.5
    k2_init = [25.0, 25.0, 25.0, 25.0, 25.0, 25.0]
    k2_min = 2.0
    ka_init = 8.0
    ka_min = 1e-6
    kc_init = 6.0
    kc_min = 1e-6
    
    
    alpha = 0.9
    gamma = 0.9
    
    controller_interval = 2
    controller_dt = 0.02
    
    predictor_out_steps = 6


def randomize_rigid_body_com_at_reset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(
            asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(
            asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu"
    ).unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()
    # print(torch.tensor([0.0, 0.0, 0.02]).repeat(len(env_ids), 1).unsqueeze(1).shape)
    # print(coms[env_ids[:, None], body_ids, :3])
    # print(torch.tensor([0.0, 0.0, 0.02]).repeat(len(env_ids), 1).unsqueeze(1))
    coms[env_ids[:, None], body_ids, :3] = (
        torch.tensor([0.0, 0.0, 0.02]).repeat(len(env_ids), 1).unsqueeze(1)
        + rand_samples
    )

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)

def apply_external_force_torque_propeller(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_propellers: int = 1,
):
    'Apply random disturbances to a subset of thrusters.'
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # prop_body_ids = asset.find_bodies("L.*")[0]
    # print(f"[INFO 1] Found propeller body ids: {prop_body_ids} for asset: {asset_cfg.name}")
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # 
    num_propellers = torch.randint(0, max_propellers + 1, (1,), device=asset.device).item()
    selected_mask = torch.zeros(8, dtype=torch.bool, device=asset.device)
    if num_propellers > 0:
        selected_mask[torch.randperm(8, device=asset.device)[:num_propellers]] = True
    asset_cfg.body_ids = [1,2,3,4,5,6,7,8]

    # sample random forces and torques
    size = (len(env_ids), 8, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    forces *= selected_mask.view(1, -1, 1)
    torques *= selected_mask.view(1, -1, 1)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    # print(f"[INFO 0] Env: {env_ids}, +{num_propellers}+Applying random external forces: {forces} and torques: {torques} to asset: {asset_cfg.name}'s bodies: {asset_cfg.body_ids}")
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)



def apply_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    size = (len(env_ids), 1,3)
    # print(f"[INFO 0] Env: {env_ids}, +1+Applying random external forces and torques to asset")
    env.disturb_thrust[env_ids] = math_utils.sample_uniform(*force_range, size, asset.device)
    env.disturb_moment[env_ids] = math_utils.sample_uniform(*torque_range, size, asset.device)


def _random_unit_vectors(size: tuple[int, ...], device: torch.device | str) -> torch.Tensor:
    vectors = torch.randn(size, device=device)
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp_min_(1.0e-6)


def randomize_constant_current(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    speed_range: tuple[float, float] = (0.0, 1.0),
):
    
    device = env.current_velocity.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    
    size = (len(env_ids), 1, 1)
    
    direction = _random_unit_vectors((len(env_ids), 1, 3), device) # n,1,3
    speed = math_utils.sample_uniform(*speed_range, size, device) # n,1,1

    
    env.current_direction[env_ids] = direction
    
    env.current_mean_velocity[env_ids] = speed
    
    env.current_velocity_c[env_ids] = speed
    
    env.current_velocity[env_ids] = direction * speed
    
    env.gust_current[env_ids] = torch.zeros(len(env_ids), 1, 3, device=device)
    env.gust_duration[env_ids] = torch.zeros(len(env_ids), 1, 1, device=device)
    # print(f"[INFO] Randomizing constant current for envs: {env_ids}x{speed}x{env.current_velocity[env_ids]}")

def update_ou_current(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    theta: float = 0.15,
    sigma: float = 0.05,
):
    device = env.current_velocity.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)
    
    dt = env.sim.cfg.dt
    # n,1
    velocity = env.current_velocity_c[env_ids]
    # n,1
    mean_velocity = env.current_mean_velocity[env_ids]
    
    noise = torch.randn_like(velocity)
    # print("current_velocity_c: ", env.current_velocity_c)
    env.current_velocity_c[env_ids] +=-theta*(velocity - mean_velocity)*dt + sigma*math.sqrt(dt)*noise    
    env.current_velocity[env_ids] = env.current_direction[env_ids] * env.current_velocity_c[env_ids]
    # print("current_velocity: ", env.current_velocity)
    
def add_random_gust_current(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    speed_range: tuple[float, float] = (0.0, 0.5),
    duration_range: tuple[float, float] = (0.5, 2.0),
):
    device = env.current_velocity.device
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=device)

    size = (len(env_ids), 1, 1)
    direction = _random_unit_vectors((len(env_ids), 1, 3), device)
    speed = math_utils.sample_uniform(*speed_range, size, device)
    duration = math_utils.sample_uniform(*duration_range, size, device)
    
    env.gust_current[env_ids] = direction * speed
    env.gust_duration[env_ids] = duration




# domain randomization
@configclass
class EventCfg:
    """
    Randomization configuration for BlueROV environment,including:
    com,mass,init_state,wrench
    """

    # randomize at reset
    robot_com = EventTerm(
        func=randomize_rigid_body_com_at_reset,
        mode="reset",
        params={
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
            
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )
    robot_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    random_init_state_with_SO3 = EventTerm(
        func=mdp.reset_root_state_with_random_orientation,
        mode="reset",
        params={
            "pose_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
                "z": (-2.0, 2.0),
            },
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )
    randomize_constant_current = EventTerm(
        func=randomize_constant_current,
        mode="reset",
        params={
            "speed_range": (0.2, 1.2),
        },
    )
    update_ou_current = EventTerm(
        func=update_ou_current,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "theta": 0.15,
            "sigma": 0.05,
        },
    )
    add_random_gust_current = EventTerm(
        func=add_random_gust_current,
        mode="interval",
        interval_range_s=(2.0, 3.0),
        params={
            "speed_range": (0.0, 0.5),
            "duration_range": (0.1, 1.0),
        },
    )
    
    add_random_propeller_wrenchs = EventTerm(
        func=apply_external_force_torque_propeller,
        mode="interval",
        interval_range_s=(0.5, 1.5),
        params={
            "force_range": (-10.0, 10.0),  # N
            "torque_range": (-3.0, 3.0),  # Nm
            "asset_cfg": SceneEntityCfg("robot"),
            "max_propellers": 4,
        },
    )

    add_random_body_wrenchs = EventTerm(
        func=apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.5, 1.5),
        params={
            "force_range": (-10.0, 10.0),  # N
            "torque_range": (-3.0, 3.0),  # Nm
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )
    
    push_by_current_velocity = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 2.0),
        params={
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.05, 0.05),
            },
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
        },
    )
