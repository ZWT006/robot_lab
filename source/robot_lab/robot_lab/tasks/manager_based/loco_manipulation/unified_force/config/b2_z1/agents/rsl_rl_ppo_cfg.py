# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnifiedForceActorCriticCfg(RslRlPpoActorCriticCfg):
    class_name: str = "UnifiedForceActorCritic"
    single_frame_dim: int = 73
    estimator_target_group: str = "state_estimation_target"
    estimator_output_dim: int = 12
    encoder_hidden_dims: list[int] = [512, 256, 128]
    decoder_hidden_dims: list[int] = [128, 64]


@configclass
class UnifiedForceAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "UnifiedForcePPO"
    adaptation_learning_rate: float = 1.0e-5
    adaptation_weights: tuple[float, float, float, float] = (0.2, 0.2, 1.0, 1.0)
    adaptation_dims: tuple[int, int, int, int] = (3, 3, 3, 3)
    adaptation_substeps: int = 1


@configclass
class UnifiedForcePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "UnifiedForceRunner"
    num_steps_per_env = 24
    max_iterations = 60000
    save_interval = 200
    experiment_name = "unified_force"
    clip_actions = 100.0
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    policy = UnifiedForceActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = UnifiedForceAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
