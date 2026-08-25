'Load the OBPPO environment and evaluate the OBAC controller without an RL policy.'

from __future__ import annotations
from torch.utils.tensorboard import SummaryWriter

import importlib
import os
import sys
import time
import types

import gymnasium as gym
import torch

from play_base import playBase


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
PKG_PARENT = os.path.dirname(PKG_ROOT)
PKG_NAME = os.path.basename(PKG_ROOT)

if PKG_PARENT not in sys.path:
    sys.path.insert(0, PKG_PARENT)

#
writer = SummaryWriter(log_dir="runs/debug_obac")


def _extract_done_mask(*done_terms) -> torch.Tensor | None:
    done_mask = None
    for term in done_terms:
        if term is None:
            continue
        mask = term.bool() if isinstance(
            term, torch.Tensor) else torch.as_tensor(term).bool()
        done_mask = mask if done_mask is None else (done_mask | mask)
    return done_mask


class playObacController(playBase):
    def add_other_args(self, parser):
        super().add_other_args(parser)
        parser.add_argument(
            "--print_every",
            type=int,
            default=50,
            help="Print controller status every N simulation steps.",
        )
        parser.set_defaults(
            task="Isaac-BlueROV-Direct-v2",
            num_envs=1,
            export_onnx=False,
        )

    def modify_env_cfg(self, env_cfg):
        env_cfg.episode_length_s = 400
        env_cfg.params.eval_mode = True
        env_cfg.params.debug_out = False
        env_cfg.decimation = 2
        return env_cfg

    def setup_play(self):
        self.step_count = 0

    def action_before_inference(self):
        pass

    def action_with_inference(self):
        pass

    def _install_direct_action_adapter(self, base_env) -> None:
        base_env._manual_obac_actions = torch.zeros(
            base_env.num_envs, 6, device=base_env.device
        )

        def _pre_physics_step(env_self, actions: torch.Tensor) -> None:
            print("[Debug]: Pre-physics step called:", actions)
            manual_actions = torch.clamp(
                actions, -1.0, 1.0).to(env_self.device)
            env_self._manual_obac_actions.copy_(manual_actions)
            env_self._actions.copy_(manual_actions)
            env_self.first_actions = manual_actions.clone()

        def _apply_action(env_self) -> None:
            print("[Debug]: Applying OBAC actions to the environment")
            env_self._thrust[:, 0, :], env_self._moment[:, 0, :] = env_self._compute_dynamics(
                env_self._manual_obac_actions
            )
            env_self._body_id = env_self._robot.find_bodies("base_link")[0]
            env_self._robot.set_external_force_and_torque(
                env_self._thrust,
                env_self._moment,
                positions=torch.zeros_like(env_self._thrust),
                body_ids=env_self._body_id,
                is_global=False,
            )

        base_env._pre_physics_step = types.MethodType(
            _pre_physics_step, base_env)
        base_env._apply_action = types.MethodType(_apply_action, base_env)

    def run(self):
        import isaaclab_tasks  # noqa: F401
        from isaaclab_tasks.utils import parse_env_cfg

        importlib.import_module(PKG_NAME)

        device = "cpu" if self.args_cli.cpu else "cuda:0"
        env_cfg = parse_env_cfg(
            self.args_cli.task,
            device=device,
            num_envs=self.args_cli.num_envs,
            use_fabric=not self.args_cli.disable_fabric,
        )
        env_cfg = self.modify_env_cfg(env_cfg)
        env = gym.make(self.args_cli.task, cfg=env_cfg)
        base_env = env.unwrapped
        self._install_direct_action_adapter(base_env)
        self.setup_play()

        env.reset()
        controller = base_env.obac_module
        step_dt = getattr(
            base_env, "step_dt", float(
                base_env.cfg.sim.dt * base_env.cfg.decimation)
        )

        while self.simulation_app.is_running():
            start_time = time.time()
            self.action_before_inference()
            obs = base_env._get_obppo_observations()

            with torch.inference_mode():
                actions, controller_info = controller(obs)
                actor_weights = controller_info.get("actor_weights")
                critic_weights = controller_info.get("critic_weights")
                v2 = controller_info.get("V2")

                if isinstance(actor_weights, torch.Tensor):
                    writer.add_scalar(
                        "obac/actor_weights_mean",
                        actor_weights.mean().item(),
                        self.step_count,
                    )
                    writer.add_scalar(
                        "obac/actor_weights_norm",
                        actor_weights.norm().item(),
                        self.step_count,
                    )
                if isinstance(critic_weights, torch.Tensor):
                    writer.add_scalar(
                        "obac/critic_weights_mean",
                        critic_weights.mean().item(),
                        self.step_count,
                    )
                    writer.add_scalar(
                        "obac/critic_weights_norm",
                        critic_weights.norm().item(),
                        self.step_count,
                    )
                if isinstance(actor_weights, torch.Tensor) and isinstance(critic_weights, torch.Tensor):
                    writer.add_scalar(
                        "obac/weights_gap_norm",
                        (actor_weights - critic_weights).norm().item(),
                        self.step_count,
                    )
                if isinstance(v2, torch.Tensor):
                    writer.add_scalar(
                        "obac/V2_mean",
                        v2.mean().item(),
                        self.step_count,
                    )
                    writer.add_scalar(
                        "obac/V2_max",
                        v2.max().item(),
                        self.step_count,
                    )
                print("[info] Outer actions: ", actions)
                _, rewards, terminated, truncated, _ = env.step(actions)
                self.action_with_inference()

            self.step_count += 1
            if self.args_cli.print_every > 0 and self.step_count % self.args_cli.print_every == 0:
                pos_err = controller_info["e1"][:, :3].norm(
                    dim=-1).mean().item()
                ang_err = controller_info["e1"][:, 3:].norm(
                    dim=-1).mean().item()
                reward_mean = rewards.mean().item() if isinstance(
                    rewards, torch.Tensor) else float(rewards)
                print(
                    f"[OBAC] step={self.step_count} pos_err={pos_err:.4f} "
                    f"ang_err={ang_err:.4f} reward={reward_mean:.4f}"
                )

            done_mask = _extract_done_mask(terminated, truncated)
            if done_mask is not None and torch.any(done_mask):
                controller.reset(base_env._build_fossen_param())

            sleep_time = step_dt - (time.time() - start_time)
            if self.args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        env.close()
        self.simulation_app.close()


if __name__ == "__main__":
    playObacController().run()
