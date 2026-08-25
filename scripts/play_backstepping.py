'Run the integral backstepping controller in the BlueROV task.'
import argparse
import gymnasium as gym
import torch
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# ------------------- CLI -------------------
parser = argparse.ArgumentParser(description="Play controller in simulation.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument("--task", type=str, default="Isaac-BlueROV-Direct-v1", help="Name of the task.")
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--controller", type=str, default="ibs", help="Type of controller to use."
)
parser.add_argument(
    "--print_every",
    type=int,
    default=1,
    help="Print actions every N steps (1=every step).",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------- IsaacLab / RSL-RL -------------------
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)
    print("Added controller path:", PKG_ROOT)
from controller import (
    FossenParam,
    BacksteppingController,
    
)


def _extract_done_mask(dones, device: torch.device) -> torch.Tensor | None:
    """Convert different done formats to a boolean tensor."""
    if isinstance(dones, torch.Tensor):
        return dones.bool()
    if isinstance(dones, (tuple, list)) and len(dones) > 0:
        done_mask = dones[0]
        for extra in dones[1:]:
            done_mask = done_mask | extra
        return torch.as_tensor(done_mask, device=device).bool()
    if dones is None:
        return None
    return torch.as_tensor(dones, device=device).bool()


def _build_fossen_param(task_env, device: torch.device) -> FossenParam:
    """Read randomized physical parameters from the env and pack into FossenParam."""
    cg = task_env._robot.data.com_pos_b[:, 0, :3].to(device)
    cob_offset = getattr(task_env, "com_to_cob_offsets", None)
    cb = torch.zeros_like(cg)
    dt_val = task_env.dt
    dt = float(dt_val.item()) if torch.is_tensor(dt_val) else float(dt_val)
    return FossenParam(
        m=task_env._robot.root_physx_view.get_masses()
        .sum(dim=1, keepdim=True)
        .to(device),
        I=task_env.inertia_tensors.to(device),
        cg=cg,
        volume=task_env.volumes.to(device),
        cb=cb,
    )


def main():
    device_str = "cpu" if args_cli.cpu else "cuda:0"
    device = torch.device(device_str)
    print(f"[INFO] Using device: {device}")
    # parse env config
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device_str,
        num_envs=1 if args_cli.num_envs is None else args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # create env
    env_gym = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env_gym)
    
    obs, _ = env.reset()

    base_env = env_gym.unwrapped
    fossen_param = _build_fossen_param(base_env, device)
    num_envs = getattr(env, "num_envs", getattr(base_env, "num_envs", 1))
    policy = BacksteppingController(
        num_envs=num_envs, device=device, param=fossen_param
    ).to(device)
    policy.eval()

    with torch.inference_mode():
        while simulation_app.is_running():
            actions = policy(obs)  # (num_envs, 6) thruster forces
            # actions = torch.zeros_like(actions)
            obs, rews, dones, infos = env.step(actions)
            done_mask = _extract_done_mask(dones, device)
            if done_mask is not None and torch.any(done_mask):
                reset_ids = torch.nonzero(done_mask, as_tuple=False).flatten()
                policy.reset_buffers(reset_ids)
                updated_param = _build_fossen_param(base_env, device)
                if reset_ids.numel() > 0:
                    ids = reset_ids.tolist()
                    print(
                        "[INFO] Updated params",
                        {
                            "env_id": ids,
                            "m": updated_param.m[reset_ids]
                            .flatten()
                            .detach()
                            .cpu()
                            .tolist(),
                            "I": updated_param.I[reset_ids].detach().cpu().tolist(),
                            "cg": updated_param.cg[reset_ids].detach().cpu().tolist(),
                            "cb": updated_param.cb[reset_ids].detach().cpu().tolist(),
                            "volume": updated_param.volume[reset_ids]
                            .flatten()
                            .detach()
                            .cpu()
                            .tolist(),
                            "dt": updated_param.dt,
                        },
                    )
                policy.update_parameters(updated_param)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
