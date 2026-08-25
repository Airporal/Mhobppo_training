'Compute the thrust-allocation matrix from a USD asset.'
import torch
import isaaclab.utils.math as math_utils

# from isaaclab.assets import Articulation
import numpy as np

BLUEROV_THRUSTER_CONFIG = torch.tensor(
    [
        [-0.7071, -0.7071, 0.7071, 0.7071, 0, 0, 0, 0],
        [0.7071, -0.7071, 0.7071, -0.7071, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [-0.0417, 0.0417, -0.0417, 0.0417, 0.2226, -0.2204, 0.2226, -0.2204],
        [-0.0417, -0.0417, 0.0417, 0.0417, -0.1214, -0.1214, 0.1186, 0.1186],
        [0.1853, -0.1838, -0.1853, 0.1838, 0, 0, 0, 0],
    ]
)

ThrusterConfigureMap = {
    "BLUEROV_THRUSTER_CONFIG": BLUEROV_THRUSTER_CONFIG,
}


def calculate_thrust_configure_from_usd_vec(
    robot,  # Articulation object of the robot
    link_names: str,
    thrust_dirs_local: torch.tensor,
    device: torch.device,
    debug: bool = False,
):
    """
       Vectorized calculation of Thrust Allocation Matrix (TAM)
       root: Articulation object of the robot
       link_names:
           list of thruster link names, for example: ".*Link.*"
           Please check the :meth:`isaaclab.utils.string_utils.resolve_matching_names` function for more
           information on the name matching.
       thrust_dirs_local: each thruster directions in thrusters' local frame,Nx3,for example:
           torch.tensor([[1,0,0],[1,0,0],[1,0,0],[1,0,0],[0,0,1],[0,0,-1],[0,0,-1],[0,0,1]])
       device: torch device
       return: Thrust Allocation Matrix (TAM) [6, N]
               However,TAM directions are defined in FLU, not FRD

    [[-0.7431 -0.7431  0.6691  0.6691 -0.     -0.     -0.     -0.    ]
    [ 0.6691 -0.6691  0.7432 -0.7432  0.      0.      0.      0.    ]
    [ 0.      0.     -0.      0.      1.      1.      1.      1.    ]
    [ 0.0806 -0.0725  0.0813 -0.0813  0.2634 -0.1796  0.2634 -0.1796]
    [ 0.0895  0.0806 -0.0732 -0.0732 -0.1425 -0.1425  0.0975  0.0975]
    [ 0.2094 -0.1471 -0.189   0.1329  0.     -0.      0.     -0.    ]]

    """

    # ------------------------------------------------------------
    # 1. Root & Thruster poses (world frame)
    # ------------------------------------------------------------
    # robot.update(0.0)
    root_pose = robot.data.root_state_w[0, :7]  # base_link[7]
    root_pos = root_pose[:3]  # [3]

    root_quat = root_pose[3:]  # [4]
    thruster_indices = robot.find_bodies(link_names)[0]
    # joints positions and quaternions
    thruster_poses = robot.data.body_state_w[0, thruster_indices, :7]

    # thruster_poses: [N, 7]

    t_pos = thruster_poses[:, :3]  # [N, 3]
    t_quat = thruster_poses[:, 3:]  # [N, 4]
    N = t_pos.shape[0]
    # ------------------------------------------------------------
    # 2. r_body = R_root^{-1} (p_t - p_root)
    # ------------------------------------------------------------
    r_world = t_pos - root_pos.unsqueeze(0)  # [N, 3]
    r_body = math_utils.quat_apply_inverse(root_quat.unsqueeze(0), r_world)  # [N, 3]

    # ------------------------------------------------------------
    # 3. transform thrust directions to body frame
    #    n_body = R_root^{-1} * (R_thruster * n_local)
    # ------------------------------------------------------------
    n_world = math_utils.quat_apply(t_quat, thrust_dirs_local)  # [N, 3]
    n_body = math_utils.quat_apply_inverse(root_quat.unsqueeze(0), n_world)  # [N, 3]

    # ------------------------------------------------------------
    # 4. Torque = r × n
    # ------------------------------------------------------------
    torque = torch.cross(r_body, n_body, dim=1)  # [N, 3]

    # ------------------------------------------------------------
    # 5. Assemble TAM: [6, N]
    # ------------------------------------------------------------
    tam = torch.zeros((6, N), device=device)
    tam[0:3, :] = n_body.T
    tam[3:6, :] = torque.T

    # ------------------------------------------------------------
    # If the base frame is FRD, convert TAM from FLU to FRD
    # ------------------------------------------------------------

    tam_FRD = tam.clone() * torch.tensor([1, 1, 1, 1, 1, 1], device=device).unsqueeze(1)

    if debug:
        print("-" * 30)
        print("Root Position:", root_pose)
        print("Thruster Positions:", r_body)
        print(f"Thrust Allocation Matrix ({tam.shape}):")
        print(np.round(tam.cpu().numpy(), 4))
        print(np.round(tam_FRD.cpu().numpy(), 4))
        print("-" * 30)

    return tam
