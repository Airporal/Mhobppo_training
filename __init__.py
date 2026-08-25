# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# 
"""
Quacopter environment.
"""

import gymnasium as gym

from . import agents


##
# Register Gym environments.
##
# End-to-end thruster controller.
gym.register(
    id="Isaac-BlueROV-Direct-v0",
    entry_point=f"{__name__}.bluerov_pwm_env:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bluerov_pwm_env:BlueROVEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVPWMPPORunnerCfg",
    },
)

# OBPPO controller.
gym.register(
    id="Isaac-BlueROV-Direct-v1",
    entry_point=f"{__name__}.bluerov_mhobppo_env:BlueROVEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bluerov_mhobppo_env:BlueROVEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BlueROVObppoRunnerCfg",
    },
)
