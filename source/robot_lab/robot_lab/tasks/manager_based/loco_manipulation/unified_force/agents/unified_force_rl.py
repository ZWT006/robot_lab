# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Concurrent state-estimation policy and PPO integration for UniFP."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.networks import MLP
from rsl_rl.runners import OnPolicyRunner


class UnifiedForceHistoryActor(nn.Module):
    """Deployment module mapping the full actor history directly to actions."""

    def __init__(self, encoder: nn.Module, actor: nn.Module, single_frame_dim: int):
        super().__init__()
        self.encoder = encoder
        self.actor = actor
        self.single_frame_dim = single_frame_dim

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(observation)
        return self.actor(torch.cat((observation[:, -self.single_frame_dim :], latent), dim=-1))

    def __getitem__(self, index: int):
        """Expose the input layer for Isaac Lab's ONNX shape discovery."""
        return self.encoder[index]


class UnifiedForceExportPolicy(nn.Module):
    """Adapter understood by Isaac Lab's standard RSL-RL exporters."""

    is_recurrent = False

    def __init__(self, encoder: nn.Module, actor: nn.Module, single_frame_dim: int):
        super().__init__()
        self.actor = UnifiedForceHistoryActor(encoder, actor, single_frame_dim)


class UnifiedForceActorCritic(nn.Module):
    """UniFP actor with a history encoder and concurrent state-estimation decoder."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        single_frame_dim: int = 73,
        estimator_target_group: str = "state_estimation_target",
        estimator_output_dim: int = 12,
        encoder_hidden_dims: list[int] = [512, 256, 128],
        decoder_hidden_dims: list[int] = [128, 64],
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            print(f"UnifiedForceActorCritic ignored arguments: {sorted(kwargs)}")
        if actor_obs_normalization or critic_obs_normalization:
            raise ValueError("The UniFP reproduction expects observation normalization to be disabled.")

        self.obs_groups = obs_groups
        self.single_frame_dim = single_frame_dim
        self.estimator_target_group = estimator_target_group
        actor_obs_dim = sum(obs[name].shape[-1] for name in obs_groups["policy"])
        critic_obs_dim = sum(obs[name].shape[-1] for name in obs_groups["critic"])
        if actor_obs_dim % single_frame_dim != 0:
            raise ValueError(f"Actor observation size {actor_obs_dim} is not divisible by {single_frame_dim}")
        self.history_length = actor_obs_dim // single_frame_dim
        self.latent_dim = self.history_length * 2

        self.adaptation_encoder_module = MLP(actor_obs_dim, self.latent_dim, encoder_hidden_dims, activation)
        self.adaptation_decoder_module = MLP(self.latent_dim, estimator_output_dim, decoder_hidden_dims, activation)
        self.actor = MLP(
            single_frame_dim + self.latent_dim,
            num_actions,
            actor_hidden_dims,
            activation,
        )
        self.critic = MLP(critic_obs_dim, 1, critic_hidden_dims, activation)
        self.actor_obs_normalization = False
        self.critic_obs_normalization = False
        self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unsupported noise_std_type: {noise_std_type}")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        print(f"Adaptation encoder: {self.adaptation_encoder_module}")
        print(f"Adaptation decoder: {self.adaptation_decoder_module}")
        print(f"Actor MLP: {self.actor}")
        print(f"Critic MLP: {self.critic}")

    def get_actor_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs) -> torch.Tensor:
        return torch.cat([obs[name] for name in self.obs_groups["critic"]], dim=-1)

    def get_estimator_target(self, obs) -> torch.Tensor:
        return obs[self.estimator_target_group]

    def encode(self, obs) -> torch.Tensor:
        return self.adaptation_encoder_module(self.get_actor_obs(obs))

    def predict_state(self, obs) -> torch.Tensor:
        return self.adaptation_decoder_module(self.encode(obs))

    def update_distribution(self, obs):
        actor_obs = self.get_actor_obs(obs)
        latent = self.adaptation_encoder_module(actor_obs)
        mean = self.actor(torch.cat((actor_obs[:, -self.single_frame_dim :], latent), dim=-1))
        std = self.std.expand_as(mean) if self.noise_std_type == "scalar" else self.log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.get_actor_obs(obs)
        latent = self.adaptation_encoder_module(actor_obs)
        return self.actor(torch.cat((actor_obs[:, -self.single_frame_dim :], latent), dim=-1))

    def evaluate(self, obs, **kwargs):
        return self.critic(self.get_critic_obs(obs))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_normalization(self, obs):
        pass

    def reset(self, dones=None):
        pass

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True

    def exportable_policy(self) -> UnifiedForceExportPolicy:
        return UnifiedForceExportPolicy(
            self.adaptation_encoder_module,
            self.actor,
            self.single_frame_dim,
        )


class UnifiedForcePPO(PPO):
    """PPO plus UniFP's separately optimized state-estimation objective."""

    def __init__(
        self,
        policy: UnifiedForceActorCritic,
        adaptation_learning_rate: float = 1.0e-5,
        adaptation_weights: tuple[float, float, float, float] = (0.2, 0.2, 1.0, 1.0),
        adaptation_dims: tuple[int, int, int, int] = (3, 3, 3, 3),
        adaptation_substeps: int = 1,
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.adaptation_optimizer = optim.Adam(policy.parameters(), lr=adaptation_learning_rate)
        self.adaptation_weights = adaptation_weights
        self.adaptation_dims = adaptation_dims
        self.adaptation_substeps = adaptation_substeps

    def update(self):
        loss_dict = super().update()
        mean_total = 0.0
        component_sums = [0.0 for _ in self.adaptation_dims]
        update_count = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            obs_batch = batch[0]
            target = self.policy.get_estimator_target(obs_batch).detach()
            for _ in range(self.adaptation_substeps):
                prediction = self.policy.predict_state(obs_batch)
                component_losses = []
                start = 0
                for dim, weight in zip(self.adaptation_dims, self.adaptation_weights):
                    end = start + dim
                    component_losses.append(
                        F.mse_loss(prediction[:, start:end] * weight, target[:, start:end] * weight)
                    )
                    start = end
                adaptation_loss = torch.stack(component_losses).sum()
                self.adaptation_optimizer.zero_grad()
                adaptation_loss.backward()
                if self.is_multi_gpu:
                    self.reduce_parameters()
                self.adaptation_optimizer.step()
                mean_total += adaptation_loss.item()
                for index, component_loss in enumerate(component_losses):
                    component_sums[index] += component_loss.item()
                update_count += 1

        denominator = max(update_count, 1)
        loss_dict["state_estimation"] = mean_total / denominator
        labels = ("base_velocity", "ee_position", "ee_force", "base_force")
        for label, total in zip(labels, component_sums):
            loss_dict[f"state_estimation/{label}"] = total / denominator
        return loss_dict


class UnifiedForceRunner(OnPolicyRunner):
    """RSL-RL runner that constructs the UniFP-specific policy and PPO classes."""

    def _construct_algorithm(self, obs) -> UnifiedForcePPO:
        policy_cfg = self.policy_cfg.copy()
        policy_cfg.pop("class_name", None)
        policy = UnifiedForceActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        algorithm_cfg = self.alg_cfg.copy()
        algorithm_cfg.pop("class_name", None)
        algorithm = UnifiedForcePPO(
            policy,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **algorithm_cfg,
        )
        algorithm.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return algorithm

    def save(self, path: str, infos=None):
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "adaptation_optimizer_state_dict": self.alg.adaptation_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        if self.logger_type in ["neptune", "wandb"] and not self.disable_logs:
            self.writer.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None):
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed_training = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer and resumed_training:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "adaptation_optimizer_state_dict" in loaded_dict:
                self.alg.adaptation_optimizer.load_state_dict(loaded_dict["adaptation_optimizer_state_dict"])
        if resumed_training:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")
