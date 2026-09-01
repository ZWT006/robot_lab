# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Modified in 2026 for robot_lab and Isaac Lab 2.2 compatibility.

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-ReLIC-Spot-Interlimb-Phase-1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:SpotInterlimbEnvCfg_Phase_1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SpotInterlimbPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-ReLIC-Spot-Interlimb-Phase-2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:SpotInterlimbEnvCfg_Phase_2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SpotInterlimbPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-ReLIC-Spot-Interlimb-Phase-3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:SpotInterlimbEnvCfg_Phase_3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SpotInterlimbPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-ReLIC-Spot-Interlimb-Phase-4-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:SpotInterlimbEnvCfg_Phase_4",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SpotInterlimbPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-ReLIC-Spot-Interlimb-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:SpotInterlimbEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:SpotInterlimbPPORunnerCfg",
    },
)
