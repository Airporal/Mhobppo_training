import isaaclab.sim as sim_utils

from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import torch
from isaaclab.utils.math import quat_from_euler_xyz
import os

USD_PATH = os.path.join(os.path.dirname(__file__), "../assets/bluerov", "bluerov.usd")
INIT_ROT = quat_from_euler_xyz(
    torch.tensor(0.0),
    torch.tensor(0.0),
    torch.tensor(0.0),
).tolist()
# just rigid object
BLUEROV_RIGID_CFG = RigidObjectCfg(
    # {ENV_REGEX_NS}/Robot`` will be replaced with ``/World/envs/env_.*/Robot``
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=False,
        ),
        copy_from_source=False,
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.0, 0.0, 5),
    ),
)


USD_PATH = os.path.join(os.path.dirname(__file__), "../assets/bluerov", "bluerov.usd")
BLUEROV_ARTICULATION_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0, 1, 0, 0),
        joint_pos={
            ".*": 0.0,
        },
        joint_vel={
            "Joint1": 20.0,
            "Joint2": 20.0,
            "Joint3": 20.0,
            "Joint4": 20.0,
            "Joint5": 20.0,
            "Joint6": 20.0,
            "Joint7": 20.0,
            "Joint8": 20.0,
        },
    ),
    actuators={
        "thruster": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=0.0,
            damping=0.0,
        ),
    },
)
