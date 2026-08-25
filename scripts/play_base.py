'Base class for controller playback scripts.'

from __future__ import annotations

import argparse

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg
    from isaaclab.envs import DirectRLEnvCfg


import random
from rsl_rl.runners import OnPolicyRunner
import gymnasium as gym
import os
import torch
import time
from isaaclab.app import AppLauncher

class playBase:
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
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        runner = OnPolicyRunner(env, agent_cfg.to_dict(),log_dir=None,device=agent_cfg.device)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
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
        # sim_dt*decimation
        dt = env.unwrapped.step_dt
        # reset environment
        obs_data = env.get_observations()
        obs = obs_data[0] if isinstance(obs_data, tuple) else obs_data
        self.setup_play()
        while self.simulation_app.is_running():
            start_time = time.time()
            self.action_before_inference()
            
            with torch.inference_mode():
                actions = policy(obs)
                obs, rewards, _, _ = env.step(actions)
                self.action_with_inference()

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if self.args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        env.close()
        self.simulation_app.close()


    def modify_env_cfg(self, env_cfg: DirectRLEnvCfg)->DirectRLEnvCfg:
        'Load standard and task-specific parameters into the environment.'
        raise NotImplementedError
    
    def setup_play(self):
        'Initialize parameters before playback starts.'
        raise NotImplementedError
    def action_before_inference(self):
        'Run per-iteration work before model inference.'
        raise NotImplementedError
    def action_with_inference(self):
        'Run model inference.'
        raise NotImplementedError
        
        
    
    

        
        
