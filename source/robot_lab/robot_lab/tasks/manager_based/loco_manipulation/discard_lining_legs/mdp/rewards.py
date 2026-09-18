# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward terms for discard lining whole-body control."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PoseTrackingReward(ManagerTermBase):
    """Unified end-effector position/orientation reward with sigma curriculum."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["asset_cfg"].name]
        self.body_idx = self.robot.find_bodies(params["asset_cfg"].body_names)[0][0]
        self.command_name: str = params["command_name"]
        self.position_curriculum: tuple[tuple[float, float], ...] = params["position_curriculum"]
        self.orientation_curriculum: tuple[tuple[float, float], ...] = params["orientation_curriculum"]
        self.position_level: int = params["initial_position_level"]
        self.orientation_level: int = params["initial_orientation_level"]
        self.smoothing_multiplier: float = params["smoothing_multiplier"]
        self.past_position_error = torch.ones(self.num_envs, device=self.device)
        self.past_orientation_error = torch.ones(self.num_envs, device=self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        position_curriculum: tuple[tuple[float, float], ...],
        orientation_curriculum: tuple[tuple[float, float], ...],
        initial_position_level: int,
        initial_orientation_level: int,
        smoothing_multiplier: float,
    ) -> torch.Tensor:
        command = env.command_manager.get_term(command_name)
        pos_error, rot_error = math_utils.compute_pose_error(
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
            command.current_target_pos_w,
            command.current_target_quat_w,
        )
        pos_error = torch.linalg.norm(pos_error, dim=-1)
        orientation_error = torch.linalg.norm(rot_error, dim=-1)
        smoothing = env.step_dt * self.smoothing_multiplier
        self.past_position_error.lerp_(pos_error, smoothing)
        self.past_orientation_error.lerp_(orientation_error, smoothing)
        position_sigma = self.position_curriculum[self.position_level][1]
        orientation_sigma = self.orientation_curriculum[self.orientation_level][1]
        position_reward = torch.exp(-(pos_error.square()) / position_sigma)
        orientation_reward = torch.exp(-orientation_error / orientation_sigma)
        return position_reward # commented out orientation reward for now

    def reset(self, env_ids: Sequence[int] | None = None):
        mean_position_error = self.past_position_error.mean().item()
        mean_orientation_error = self.past_orientation_error.mean().item()
        for level, (threshold, _) in enumerate(self.position_curriculum):
            if mean_position_error < threshold:
                self.position_level = level
        for level, (threshold, _) in enumerate(self.orientation_curriculum):
            if mean_orientation_error < threshold:
                self.orientation_level = level


def root_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize root height error on flat ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize squared deviation from the configured default joint pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(error), dim=-1)


def even_mass_distribution(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    flying_penalty: float = 100.0,
) -> torch.Tensor:
    """Penalize variance in normalized vertical load over the four feet."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    vertical_force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].clamp_min(0.0)
    total_force = vertical_force.sum(dim=-1, keepdim=True)
    distribution = vertical_force / (total_force + 1.0e-8)
    penalty = torch.var(distribution, dim=-1, correction=1)
    return torch.where(total_force.squeeze(-1) > 1.0e-8, penalty, flying_penalty)


class FeetUnderHips(ManagerTermBase):
    """Penalize planar displacement of each foot from its hip."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["asset_cfg"].name]
        pairs = params["body_pairs"]
        self.hip_ids = torch.tensor(
            [self.robot.find_bodies(hip_name)[0][0] for hip_name, _ in pairs],
            device=self.device,
            dtype=torch.long,
        )
        self.foot_ids = torch.tensor(
            [self.robot.find_bodies(foot_name)[0][0] for _, foot_name in pairs],
            device=self.device,
            dtype=torch.long,
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        body_pairs: tuple[tuple[str, str], ...],
        distance_sigma: float,
    ) -> torch.Tensor:
        hip_pos = self.robot.data.body_pos_w[:, self.hip_ids, :2]
        foot_pos = self.robot.data.body_pos_w[:, self.foot_ids, :2]
        distance = torch.linalg.norm(hip_pos - foot_pos, dim=-1)
        return torch.sum(1.0 - torch.exp(-distance / distance_sigma), dim=-1)
