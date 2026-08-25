'Load the robot from USD and test automatic thrust-allocation computation.'

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script Open the Bluerov.usd file for simulation testing."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext


try:
    from assets.bluerov import BLUEROV_ARTICULATION_CFG
    from thrusterModels.thruster_configure import (
        calculate_thrust_configure_from_usd_vec,
    )
except ImportError:
    import os, sys

    ASSETS_PATH = os.path.join(os.path.dirname(__file__), "../")

    sys.path.append(ASSETS_PATH)
    from assets.bluerov import BLUEROV_ARTICULATION_CFG
    from thrusterModels.thruster_configure import (
        calculate_thrust_configure_from_usd_vec,
    )

import time


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg_light = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg_light.func("/World/Light", cfg_light)

    # Robots
    robot_cfg = BLUEROV_ARTICULATION_CFG.replace(prim_path="/World/Bluerov")
    # robot_cfg.spawn.func(
    #     "/World/Bluerov", robot_cfg.spawn, translation=robot_cfg.init_state.pos
    # )

    # create handles for the robots
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()

    
    print(
        "[INFO]: Updating robot to fetch initial states...",
        robot.data.root_state_w[0, :7],
    )
    sim.step()
    robot.update(sim.get_physics_dt())
    
    thrust_dirs_local = torch.tensor(
        [
            [-1, 0, 0],  # Thruster 1 if + outside
            [-1, 0, 0],  # Thruster 2 if + outside
            [-1, 0, 0],  # Thruster 3 if + outside
            [-1, 0, 0],  # Thruster 4 if + outside
            [0, 0, -1],  # Thruster 5 if + outside
            [0, 0, -1],  # Thruster 6 if + outside
            [0, 0, -1],  # Thruster 7 if + outside
            [0, 0, -1],  # Thruster 8 if + outside
        ],
        device=sim.device,
    ).float()
    count = 0
    while simulation_app.is_running:
        if count % 100 == 0:
            t1 = time.time()
            tam_matrix = calculate_thrust_configure_from_usd_vec(
                robot, ".*Link.*", thrust_dirs_local, sim.device, True
            )
            print(f"[INFO]: Time elapsed: {time.time() - t1} seconds")
            count = 0
        count += 1
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

    # Now we are ready!
    print("[INFO]: Setup complete...")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
