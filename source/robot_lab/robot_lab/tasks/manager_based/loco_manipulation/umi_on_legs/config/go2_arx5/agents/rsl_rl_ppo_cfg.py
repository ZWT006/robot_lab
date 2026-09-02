# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UmiOnLegsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO settings matching the released UMI-on-Legs whole-body controller."""

    num_steps_per_env = 24
    max_iterations = 4001
    save_interval = 200
    experiment_name = "go2_arx5"
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 64, 32],
        critic_hidden_dims=[128, 64, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=32,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.9,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
