"""Play an OBPPO controller checkpoint with graded domain-randomization difficulty."""
from __future__ import annotations
import argparse

import random
import importlib
import csv
import datetime
import math
from rsl_rl.runners import OnPolicyRunner
import gymnasium as gym
import os
import sys
import torch
import time
from isaaclab.app import AppLauncher
import isaacsim 
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PKG_PARENT = os.path.dirname(PKG_ROOT)
PKG_NAME = os.path.basename(PKG_ROOT)

if PKG_PARENT not in sys.path:
    sys.path.insert(0, PKG_PARENT)

SIM_DT = 1.0 / 100.0
CONTROLLER_INTERVAL = 2
POLICY_DECIMATION = 20
CONTROLLER_DT = SIM_DT * CONTROLLER_INTERVAL
DIFFICULTY_LEVEL_COUNT = 20
import isaaclab.utils.math as math_utils


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.managers import SceneEntityCfg

    from isaaclab.assets import Articulation, RigidObject


class playObppoController:
    def __init__(self, description: str = "Play Model Arguments"):
        parser = argparse.ArgumentParser(description=description)
        self.add_rsl_rl_args(parser)
        self.add_other_args(parser)
        AppLauncher.add_app_launcher_args(parser)
        self.args_cli = parser.parse_args()
        self.app_launcher = AppLauncher(self.args_cli)
        self.simulation_app = self.app_launcher.app
        
    def add_other_args(self, parser: argparse.ArgumentParser):
        'Add subclass-specific command-line arguments.'
        parser.add_argument(
            "--disable_fabric",
            action="store_true",
            default=False,
            help="Disable fabric and use USD I/O operations.",
        )
        parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
        parser.add_argument("--export_onnx", action="store_true", default=True, help="Whether to export the model to ONNX format.")
        parser.add_argument(
            "--num_envs", type=int, default=None, help="Number of environments to simulate."
        )
        parser.add_argument("--task", type=str, default=None, help="Name of the task.")
        parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
        parser.add_argument(
            "--difficulty_level",
            type=int,
            default=DIFFICULTY_LEVEL_COUNT,
            choices=range(1, DIFFICULTY_LEVEL_COUNT + 1),
            metavar=f"[1-{DIFFICULTY_LEVEL_COUNT}]",
            help="Domain-randomization difficulty level. 1 is easiest, 20 matches the default ranges.",
        )
        parser.add_argument(
            "--action_log_dir",
            type=str,
            default=None,
            help="Directory for per-reset action Excel logs.",
        )
        parser.add_argument(
            "--disable_action_log",
            action="store_true",
            default=False,
            help="Disable action logging on environment resets.",
        )
        
        parser.set_defaults(
            task="Isaac-BlueROV-Direct-v2",
            seed=40,
            real_time=True,
            load_run="obppo_controller",
            checkpoint="model_3999.pt",
            export_onnx=True,
            num_envs=4,
        )

    def add_rsl_rl_args(self,parser: argparse.ArgumentParser):
        """Add RSL-RL arguments to the parser.
        Args:
            parser: The parser to add the arguments to.
        """
        # create a new argument group
        arg_group = parser.add_argument_group("rsl_rl", description="Arguments for RSL-RL agent.")
        # -- experiment arguments
        arg_group.add_argument(
            "--experiment_name", type=str, default=None, help="Name of the experiment folder where logs will be stored."
        )
        arg_group.add_argument("--run_name", type=str, default=None, help="Run name suffix to the log directory.")
        # -- load arguments
        arg_group.add_argument("--resume", action="store_true", default=False, help="Whether to resume from a checkpoint.")
        arg_group.add_argument("--load_run", type=str, default=None, help="Name of the run folder to resume from.")
        arg_group.add_argument("--checkpoint", type=str, default=None, help="Checkpoint file to resume from.")
        # -- logger arguments
        arg_group.add_argument(
            "--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"}, help="Logger module to use."
        )
        arg_group.add_argument(
            "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
        )
        
    def parse_rsl_rl_cfg(self, task_name: str, args_cli: argparse.Namespace):
        """Parse configuration for RSL-RL agent based on inputs.

        Args:
            task_name: The name of the environment.
            args_cli: The command line arguments.

        Returns:
            The parsed configuration for RSL-RL agent based on inputs.
        """
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        # load the default configuration
        rslrl_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(task_name, "rsl_rl_cfg_entry_point")
        rslrl_cfg = self.update_rsl_rl_cfg(rslrl_cfg, args_cli)
        return rslrl_cfg

    def update_rsl_rl_cfg(self, agent_cfg: RslRlOnPolicyRunnerCfg, args_cli: argparse.Namespace):
        """Update configuration for RSL-RL agent based on inputs.

        Args:
            agent_cfg: The configuration for RSL-RL agent.
            args_cli: The command line arguments.

        Returns:
            The updated configuration for RSL-RL agent based on inputs.
        """
        # override the default configuration with CLI arguments
        if hasattr(args_cli, "seed") and args_cli.seed is not None:
            # randomly sample a seed if seed = -1
            if args_cli.seed == -1:
                args_cli.seed = random.randint(0, 10000)
            agent_cfg.seed = args_cli.seed
        if args_cli.resume is not None:
            agent_cfg.resume = args_cli.resume
        if args_cli.load_run is not None:
            agent_cfg.load_run = args_cli.load_run
        if args_cli.checkpoint is not None:
            agent_cfg.load_checkpoint = args_cli.checkpoint
        if args_cli.run_name is not None:
            agent_cfg.run_name = args_cli.run_name
        if args_cli.logger is not None:
            agent_cfg.logger = args_cli.logger
        # set the project name for wandb and neptune
        if agent_cfg.logger in {"wandb", "neptune"} and args_cli.log_project_name:
            agent_cfg.wandb_project = args_cli.log_project_name
            agent_cfg.neptune_project = args_cli.log_project_name

        return agent_cfg

    def run(self):
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg
        from isaaclab_rl.rsl_rl import (
            RslRlOnPolicyRunnerCfg,
            RslRlVecEnvWrapper,
            export_policy_as_jit,
            export_policy_as_onnx,
        )
        importlib.import_module(PKG_NAME)

        device = "cpu" if self.args_cli.cpu else "cuda:0"
        env_cfg:DirectRLEnvCfg = parse_env_cfg(
            self.args_cli.task,
            device=device,
            num_envs=self.args_cli.num_envs,
            use_fabric=not self.args_cli.disable_fabric,)
        agent_cfg: RslRlOnPolicyRunnerCfg = self.parse_rsl_rl_cfg(self.args_cli.task, self.args_cli)
        env_cfg = self.modify_env_cfg(env_cfg)
        env = gym.make(self.args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env)
        log_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
        # load_run = os.path.join(agent_cfg.load_run, "2026-04-29_20-14-19")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(),log_dir=None,device=agent_cfg.device)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        if self.args_cli.export_onnx:
            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
            file_name = self.args_cli.checkpoint.split(".")[0]
            policy_nn = runner.alg.policy
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None
            export_policy_as_jit(
                policy_nn,
                normalizer=normalizer,
                path=export_model_dir,
                filename=f"{file_name}.pt",
            )
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename=f"{file_name}.onnx")
        # step_dt=sim_dt*decimation=
        dt = env.unwrapped.step_dt
        # reset environment
        reset_data = env.reset()
        obs = reset_data[0] if isinstance(reset_data, tuple) else reset_data
        self.setup_play(env.unwrapped, resume_path)
        last_controller_update_time = None
        controller_update_count = 0
        base_env = env.unwrapped
        while self.simulation_app.is_running():
            start_time = time.time()
            self.action_before_inference() 
            
            with torch.inference_mode():
                actions = policy(obs)
                if not self.args_cli.disable_action_log:
                    gains = self._compute_obppo_gains(base_env, actions)
                    difficulty = self._compute_env_difficulty(base_env)
                    self._record_action_batch(gains, difficulty, controller_update_count, last_controller_update_time)
                obs, rewards, dones, _ = env.step(actions)
                self.action_with_inference()
                self._save_reset_records(dones)
                controller_update_time = time.perf_counter()
                last_controller_update_time = controller_update_time
                controller_update_count += 1
            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if self.args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        env.close()
        self.simulation_app.close()


    def modify_env_cfg(self, env_cfg: DirectRLEnvCfg)->DirectRLEnvCfg:
        'Load standard and task-specific parameters into the environment.'
        self._difficulty_normalizers = self._read_difficulty_normalizers(env_cfg)
        self._apply_difficulty_level(env_cfg, self.args_cli.difficulty_level)
        env_cfg.episode_length_s = 4
        env_cfg.sim.dt = SIM_DT
        env_cfg.sim.render_interval = CONTROLLER_INTERVAL
        env_cfg.decimation = POLICY_DECIMATION
        env_cfg.params.controller_interval = CONTROLLER_INTERVAL
        env_cfg.params.controller_dt = CONTROLLER_DT
        env_cfg.params.eval_mode = True
        env_cfg.params.debug_out = False
        return env_cfg

    def _difficulty_ratio(self, level: int) -> float:
        return (level - 1) / (DIFFICULTY_LEVEL_COUNT - 1)

    def _read_difficulty_normalizers(self, env_cfg: DirectRLEnvCfg) -> dict[str, float]:
        events = env_cfg.events
        com_range = events.robot_com.params["com_range"]
        com_max_sq = sum(max(abs(v) for v in com_range[axis]) ** 2 for axis in ("x", "y", "z"))
        mass_range = events.robot_mass.params["mass_distribution_params"]
        current_range = events.randomize_constant_current.params["speed_range"]
        return {
            "com": max(math.sqrt(com_max_sq), 1.0e-6),
            "mass": max(abs(mass_range[0] - 1.0), abs(mass_range[1] - 1.0), 1.0e-6),
            "current": max(abs(current_range[0]), abs(current_range[1]), 1.0e-6),
        }

    def _scale_tuple(self, values: tuple[float, float], ratio: float) -> tuple[float, float]:
        return (values[0] * ratio, values[1] * ratio)

    def _scale_ratio_tuple(self, values: tuple[float, float], ratio: float) -> tuple[float, float]:
        return (1.0 + (values[0] - 1.0) * ratio, 1.0 + (values[1] - 1.0) * ratio)

    def _scale_range_dict(self, ranges: dict[str, tuple[float, float]], ratio: float) -> dict[str, tuple[float, float]]:
        return {key: self._scale_tuple(value, ratio) for key, value in ranges.items()}

    def _scale_interval(self, values: tuple[float, float], ratio: float) -> tuple[float, float]:
        interval_factor = 1.0 + 4.0 * (1.0 - ratio)
        return (values[0] * interval_factor, values[1] * interval_factor)

    def _apply_difficulty_level(self, env_cfg: DirectRLEnvCfg, level: int) -> None:
        ratio = self._difficulty_ratio(level)
        events = env_cfg.events

        events.robot_com.params["com_range"] = self._scale_range_dict(
            events.robot_com.params["com_range"], ratio
        )
        events.robot_mass.params["mass_distribution_params"] = self._scale_ratio_tuple(
            events.robot_mass.params["mass_distribution_params"], ratio
        )
        env_cfg.params.volume_ratio_range = self._scale_ratio_tuple(env_cfg.params.volume_ratio_range, ratio)

        init_params = events.random_init_state_with_SO3.params
        init_params["pose_range"] = self._scale_range_dict(init_params["pose_range"], ratio)
        init_params["velocity_range"] = self._scale_range_dict(init_params["velocity_range"], ratio)

        events.randomize_constant_current.params["speed_range"] = self._scale_tuple(
            events.randomize_constant_current.params["speed_range"], ratio
        )
        events.update_ou_current.interval_range_s = self._scale_interval(
            events.update_ou_current.interval_range_s, ratio
        )
        events.update_ou_current.params["sigma"] *= ratio

        events.add_random_gust_current.interval_range_s = self._scale_interval(
            events.add_random_gust_current.interval_range_s, ratio
        )
        events.add_random_gust_current.params["speed_range"] = self._scale_tuple(
            events.add_random_gust_current.params["speed_range"], ratio
        )
        events.add_random_gust_current.params["duration_range"] = self._scale_tuple(
            events.add_random_gust_current.params["duration_range"], max(ratio, 0.1)
        )

        for term_name in ("add_random_propeller_wrenchs", "add_random_body_wrenchs"):
            term = getattr(events, term_name)
            term.interval_range_s = self._scale_interval(term.interval_range_s, ratio)
            term.params["force_range"] = self._scale_tuple(term.params["force_range"], ratio)
            term.params["torque_range"] = self._scale_tuple(term.params["torque_range"], ratio)
        events.add_random_propeller_wrenchs.params["max_propellers"] = int(
            round(events.add_random_propeller_wrenchs.params["max_propellers"] * ratio)
        )

        events.push_by_current_velocity.interval_range_s = self._scale_interval(
            events.push_by_current_velocity.interval_range_s, ratio
        )
        push_params = events.push_by_current_velocity.params
        push_params["velocity_range"] = self._scale_range_dict(push_params["velocity_range"], ratio)
        self.difficulty_ratio = ratio
    
    def setup_play(self, base_env=None, resume_path: str | None = None):
        'Initialize parameters before playback starts.'
        self.step_count = 0
        self._save_count = 0
        self._pd = None
        self._checkpoint_name = (
            os.path.splitext(os.path.basename(resume_path))[0]
            if resume_path
            else os.path.splitext(str(self.args_cli.checkpoint))[0]
        )
        log_dir = self.args_cli.action_log_dir or os.path.join(
            os.path.dirname(__file__), "..", "logs", "obppo_action_records"
        )
        self._action_log_dir = os.path.abspath(log_dir)
        if not self.args_cli.disable_action_log:
            os.makedirs(self._action_log_dir, exist_ok=True)
        num_envs = getattr(base_env, "num_envs", self.args_cli.num_envs or 0)
        self._records_by_env = [[] for _ in range(num_envs)]

    def action_before_inference(self):
        'Run per-iteration work before model inference.'
        pass
    def action_with_inference(self):
        'Run model inference.'
        pass

    def _compute_env_difficulty(self, base_env) -> dict[str, torch.Tensor]:
        current_speed = base_env.current_mean_velocity.reshape(base_env.num_envs, -1).norm(dim=1)
        default_com = torch.as_tensor(
            base_env.cfg.params.com_to_cob_offset, device=base_env.device, dtype=base_env.com_to_cob_offsets.dtype
        )
        com_offset = (base_env.com_to_cob_offsets - default_com).norm(dim=1)
        mass_scale = base_env.masses.squeeze(-1) / float(base_env.cfg.params.mass)
        mass_delta = (mass_scale - 1.0).abs()

        current_score = (current_speed / self._difficulty_normalizers["current"]).clamp(0.0, 1.0)
        com_score = (com_offset / self._difficulty_normalizers["com"]).clamp(0.0, 1.0)
        mass_score = (mass_delta / self._difficulty_normalizers["mass"]).clamp(0.0, 1.0)
        level_score = torch.full_like(current_score, float(self.difficulty_ratio))
        difficulty_score = (
            0.35 * current_score + 0.20 * com_score + 0.20 * mass_score + 0.25 * level_score
        ).clamp(0.0, 1.0)

        return {
            "difficulty_score": difficulty_score,
            "current_speed": current_speed,
            "com_offset": com_offset,
            "mass_scale": mass_scale,
            "mass_delta": mass_delta,
        }

    def _compute_obppo_gains(self, base_env, actions: torch.Tensor) -> torch.Tensor:
        gains = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
        gains = (
            base_env._prev_gains * (1 - base_env.cfg.params.gamma)
            + base_env.cfg.params.gamma
            * base_env.obppo_gains_init
            * (1 + base_env.cfg.params.alpha * gains)
        )
        gains[:, 0:6].clamp_(min=base_env.cfg.params.k1_min)
        gains[:, 6:12].clamp_(min=base_env.cfg.params.k2_min)
        kappa_a = gains[:, 12].clamp(min=1e-6)
        kappa_c = torch.clamp(gains[:, 13], min=0.5 * kappa_a, max=kappa_a)
        gains[:, 12] = kappa_a
        gains[:, 13] = kappa_c
        gains.nan_to_num_(nan=1.0, posinf=1.0e6, neginf=1.0e-6)
        return gains

    def _record_action_batch(
        self,
        gains: torch.Tensor,
        difficulty: dict[str, torch.Tensor],
        controller_update_count: int,
        last_controller_update_time: float | None,
    ) -> None:
        if self.args_cli.disable_action_log:
            return
        now = time.time()
        update_interval = 0.0 if last_controller_update_time is None else time.perf_counter() - last_controller_update_time
        gain_rows = gains.detach().cpu().tolist()
        metrics = {key: value.detach().cpu().tolist() for key, value in difficulty.items()}
        for env_id, gain_values in enumerate(gain_rows):
            row = {
                "time": now,
                "step": self.step_count,
                "controller_update": controller_update_count,
                "controller_update_interval_s": update_interval,
                "env_id": env_id,
                "difficulty_level": self.args_cli.difficulty_level,
                "difficulty_ratio": self.difficulty_ratio,
                "difficulty_score": metrics["difficulty_score"][env_id],
                "current_speed": metrics["current_speed"][env_id],
                "com_offset": metrics["com_offset"][env_id],
                "mass_scale": metrics["mass_scale"][env_id],
                "mass_delta": metrics["mass_delta"][env_id],
            }
            for gain_id, value in enumerate(gain_values):
                row[f"gain_{gain_id:02d}"] = value
            self._records_by_env[env_id].append(row)
        self.step_count += 1

    def _save_reset_records(self, dones) -> None:
        if self.args_cli.disable_action_log or dones is None:
            return
        done_mask = dones.bool() if isinstance(dones, torch.Tensor) else torch.as_tensor(dones).bool()
        if not torch.any(done_mask):
            return
        env_ids = done_mask.nonzero(as_tuple=False).squeeze(-1).detach().cpu().tolist()
        rows = []
        for env_id in env_ids:
            rows.extend(self._records_by_env[env_id])
            self._records_by_env[env_id].clear()
        if not rows:
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._save_count += 1
        base_name = (
            f"obppo_gains_level_{self.args_cli.difficulty_level:02d}_"
            f"{self._checkpoint_name}_{timestamp}_{self._save_count:04d}"
        )
        xlsx_path = os.path.join(self._action_log_dir, f"{base_name}.xlsx")
        try:
            if self._pd is None:
                import pandas as pd

                self._pd = pd
            self._pd.DataFrame.from_records(rows).to_excel(xlsx_path, index=False)
        except Exception as exc:
            csv_path = os.path.join(self._action_log_dir, f"{base_name}.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[WARN] Failed to write Excel log ({exc}); wrote CSV instead: {csv_path}")
        
        
    
    

if __name__ == "__main__":
    playObppoController().run()
