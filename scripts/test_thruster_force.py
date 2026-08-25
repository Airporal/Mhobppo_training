"""
This script is used to test the set_external_force_and_torque function on the BlueROV model.
this function is used to apply external forces and torques to the BlueROV model.
force and torque are added in the local frame origin point of the thruster links.
x,y,z,are fixed in FLU (front left up) direction.
"""

import argparse
import torch
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script Open the Bluerov.usd file for simulation testing."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab.markers import (
    VisualizationMarkers,
    GREEN_ARROW_X_MARKER_CFG,
)

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
import isaaclab.utils.math as math_utils

try:
    from assets.bluerov import BLUEROV_ARTICULATION_CFG
except ImportError:
    import os, sys

    ASSETS_PATH = os.path.join(os.path.dirname(__file__), "../assets")
    sys.path.append(ASSETS_PATH)
    from bluerov import BLUEROV_ARTICULATION_CFG  # type: ignore


def _get_visual_orientation(robot: Articulation, link_names: list, device: str):
    'Compute the world-frame orientation of an arrow.'
    thruster_indices = robot.find_bodies(link_names)[0]
    thruster_links = robot.data.body_state_w[0, thruster_indices, :7]
    thruster_pos_world = thruster_links[..., :3]
    thruster_quat_world = thruster_links[..., 3:]

    num_markers = len(link_names)
    q_rotate_x_to_z = torch.tensor(
        [0.7071068, 0.0, -0.7071068, 0.0], device=device
    ).repeat(num_markers, 1)

    marker_quats_world = math_utils.quat_mul(thruster_quat_world, q_rotate_x_to_z)
    return thruster_pos_world, marker_quats_world


def main():
    """Main function."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)

    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(0.5, 0.5, 1.0), target=(0.0, 0.0, 0.5))

    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)

    robot_cfg = BLUEROV_ARTICULATION_CFG.replace(prim_path="/World/bluerov")  # type: ignore
    # robot_cfg.spawn.func(
    #     "/World/Bluerov",
    #     robot_cfg.spawn,
    #     translation=robot_cfg.init_state.pos,
    # )

    robot = Articulation(robot_cfg)
    sim.reset()

    # Thrusters
    link_names = ["Link1", "Link2", "Link3", "Link4"]
    prop_body_ids = robot.find_bodies("L.*")[0]
    # prop_body_ids = robot.find_bodies(["Link*"])[0]
    base_link_id = robot.find_bodies(["base_link"])[0]
    print(prop_body_ids)    
    num_thrusters = len(prop_body_ids)
    print(robot.data.body_names)
    print("com:", robot.root_physx_view.get_coms())

    # Markers
    # marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
    # marker_cfg.markers["arrow"].scale = (0.05, 0.05, 0.3)
    # marker_cfg.prim_path = "/World/Thrusters/Markers"
    # thrust_markers = VisualizationMarkers(marker_cfg)

    
    sim.step()
    robot.update(sim.get_physics_dt())

    
    # thruster_pos = robot.data.body_state_w[0, prop_body_ids, :3]  # [4, 3]
    # pos_x = thruster_pos[:, 0]
    # pos_y = thruster_pos[:, 1]

    # x_mean = torch.mean(pos_x)
    # y_mean = torch.mean(pos_y)

    
    # front_mask = pos_x > x_mean
    # rear_mask = pos_x < x_mean

    
    # left_mask = pos_y > y_mean
    # right_mask = pos_y < y_mean

    
    # print(f"   Front Indices: {torch.nonzero(front_mask).flatten().tolist()}")
    # print(f"   Rear Indices:  {torch.nonzero(rear_mask).flatten().tolist()}")
    # print(f"   Left Indices:  {torch.nonzero(left_mask).flatten().tolist()}")
    # print(f"   Right Indices: {torch.nonzero(right_mask).flatten().tolist()}")

    
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity_magnitude = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    total_hover_force = robot_mass * gravity_magnitude
    thrust_base = total_hover_force / num_thrusters

    
    # PITCH_TRIM = 0.0265
    # ROLL_TRIM = 0.00025
    # PITCH_TRIM = 0.0
    # ROLL_TRIM = 0.0
    sim_dt = sim.get_physics_dt()
    count = 0
    print(f"[INFO] Robot Mass: {robot_mass:.4f} kg")
    print(f"baselink com: {robot.root_physx_view.get_coms()[0][0][:3]}")
    print(f"[INFO] Base Thrust: {thrust_base:.4f} N")
    # print(f"[INFO] Pitch Trim: {PITCH_TRIM} (Front + / Rear -)")
    # print(f"[INFO] Roll Trim:  {ROLL_TRIM} (Right + / Left -)")

    ranges = torch.tensor([[0.02, 0.02], [0.02, 0.02], [0.02, 0.02]], device="cpu")
    indices = torch.tensor([0], dtype=torch.int32, device="cpu")
    while simulation_app.is_running():
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (1, 3), device="cpu"
        ).unsqueeze(1)
        # coms = robot.root_physx_view.get_coms().clone()
        # default_coms_p = robot.data.com_pos_b.clone().to("cpu")
        # default_coms_q = robot.data.body_com_quat_b.clone().to("cpu")
        # default_coms = torch.cat([default_coms_p, default_coms_q], dim=-1)
        # print("Original com:", coms[0, 0, :3], rand_samples)
        # print("Default com:", default_coms[0, 0, :3], rand_samples)
        # coms[0, 0, :3] += rand_samples.reshape(3)
        # default_coms[0, 0, :3] += rand_samples.reshape(3)
        # robot.root_physx_view.set_coms(default_coms, indices)
        # print("com:", robot.root_physx_view.get_coms())
        if count % 2000 == 0:
            count = 0
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos, robot.data.default_joint_vel
            )
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()
            print(">>> Reset Robot State")

        # ---------------- Forces Calculation ----------------
        forces_local = torch.zeros(
            robot.num_instances, num_thrusters, 3, device=sim.device
        )

        
        thrust_values = torch.ones(num_thrusters, device=sim.device) * thrust_base

        
        # thrust_values[front_mask] *= 1.0 + PITCH_TRIM
        # thrust_values[rear_mask] *= 1.0 - PITCH_TRIM

        
        
        # thrust_values[right_mask] *= 1.0 + ROLL_TRIM
        # thrust_values[left_mask] *= 1.0 - ROLL_TRIM

        forces_local[..., 0] = thrust_values
        forces_local[..., 0] *= torch.tensor(
            [0.0, 0.0, 0.0, 2.0], device=sim.device
        )  
        print(f"[INFO] Thrust Values: {forces_local.tolist()}")
        
        # important set_external_force_and_torque api set forces in local frame of the link origin，which defined in urdf
        base_link_f_test = torch.tensor([[0.0, 200.0, 0.0]], device=sim.device)
        # print(robot.data.projected_gravity_b)
        # print(math_utils.euler_xyz_from_quat(robot.data.root_quat_w))
        robot.set_external_force_and_torque(
            base_link_f_test,
            torch.zeros_like(base_link_f_test),
            body_ids=base_link_id,
            is_global=False,
        )
        # root_state = robot.data.default_root_state.clone()
        # root_state[..., 7:] = torch.tensor(
        #     [20.0, 20.0, 20.0, 20.0, 20.0, 20.0], device=sim.device
        # )
        # important write_root_state api set state based on world frame
        # robot.write_root_velocity_to_sim(
        #     torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], device=sim.device)
        # )
        print(
            "Set root lin vel to sim:",
            robot.data.root_lin_vel_w,
            robot.data.root_ang_vel_w,
        )
        # This api get FRD frame state
        print("Get state Body", robot.data.root_lin_vel_b, robot.data.root_ang_vel_b)

        # robot.write_root_state_to_sim(root_state)
        # ---------------- Visualization ----------------
        # marker_pos, marker_rot = _get_visual_orientation(robot, link_names, sim.device)
        # thrust_markers.visualize(translations=marker_pos, orientations=marker_rot)

        # Step
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        count += 1


if __name__ == "__main__":

    main()
    simulation_app.close()
