# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class BlueROVPWMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 
    max_iterations = 4000 
    empirical_normalization = False 
    save_interval = 20 
    experiment_name = "bluerov_pwm_control"
    
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }
    resume = False 
    load_run = ".*" 
    load_checkpoint = "model_.*.pt" 
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", 
        init_noise_std=1.0,  
        noise_std_type="log", 
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5, 
        num_mini_batches=4, 
        learning_rate=1.0e-4,
        schedule="adaptive", 
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )


@configclass
class BlueROVObppoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24 
    max_iterations = 4000 
    empirical_normalization = False 
    save_interval = 50 
    experiment_name = "bluerov_obppo_control"
    
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "predictor", "others"],
    }
    resume = False 
    load_run = ".*" 
    load_checkpoint = "model_.*.pt" 
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic", 
        init_noise_std=1.0,  
        noise_std_type="log", 
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5, 
        num_mini_batches=4, 
        learning_rate=1.0e-4,
        schedule="adaptive", 
        gamma=0.99,
        lam=0.95,
        entropy_coef=0.0,
        desired_kl=0.01,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
    )
