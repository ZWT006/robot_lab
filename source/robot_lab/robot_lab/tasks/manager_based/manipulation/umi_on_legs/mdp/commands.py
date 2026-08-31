# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""World-frame end-effector trajectory commands used by UMI-on-Legs."""

from __future__ import annotations

import math
import os
import pickle
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class EndEffectorTrajectoryCommand(CommandTerm):
    """Sample and track a world-frame trajectory from the released UMI data format.

    The policy command is the target pose at every configured preview time expressed
    relative to the measured end-effector pose.  As in the release, all position
    values are concatenated first, followed by the first two rows of every rotation
    matrix (the 6D rotation representation used by the paper).
    """

    cfg: EndEffectorTrajectoryCommandCfg

    def __init__(self, cfg: EndEffectorTrajectoryCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        body_ids, _ = self.robot.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one body matching {cfg.body_name!r}, found {len(body_ids)}")
        self.body_idx = body_ids[0]

        self._command = torch.zeros(
            self.num_envs,
            len(cfg.preview_times) * 9,
            device=self.device,
        )
        self.current_target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.current_target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.current_target_quat_w[:, 0] = 1.0
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # A controller update is 20 ms.  The released 10 ms estimator latency is
        # consequently represented by the nearest complete policy observation tick.
        self._pose_history_steps = max(1, math.ceil(cfg.pose_latency_s / env.step_dt) + 1)
        self._eef_pos_history = torch.zeros(self.num_envs, self._pose_history_steps, 3, device=self.device)
        self._eef_quat_history = torch.zeros(self.num_envs, self._pose_history_steps, 4, device=self.device)
        self._eef_quat_history[..., 0] = 1.0

        self._trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long)
        self._generator = torch.Generator(device="cpu")
        # ``ManagerBasedRLEnvCfg.seed`` is optional.  Keep trajectory selection
        # reproducible when it is supplied, while still allowing the stock
        # RobotLab launch path (whose default is ``None``) to construct the task.
        trajectory_seed = env.cfg.seed if env.cfg.seed is not None else torch.initial_seed()
        self._generator.manual_seed(trajectory_seed)
        self._procedural_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._procedural_anchor_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._procedural_anchor_quat_w[:, 0] = 1.0
        self._dataset_pos: torch.Tensor | None = None
        self._dataset_quat: torch.Tensor | None = None
        self._dataset_lengths: torch.Tensor | None = None
        if cfg.trajectory_file:
            self._load_trajectory_file(cfg.trajectory_file)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _load_trajectory_file(self, file_path: str):
        path = Path(os.path.expandvars(file_path)).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"UMI trajectory file not found: {path}. See docs/umi_on_legs.md for the official data download."
            )
        with path.open("rb") as stream:
            episodes = pickle.load(stream)  # noqa: S301 - the official UMI dataset is a pickle release
        if not isinstance(episodes, (list, tuple)) or not episodes:
            raise ValueError("Trajectory pickle must contain a non-empty list of episode dictionaries")

        positions: list[torch.Tensor] = []
        quaternions: list[torch.Tensor] = []
        for episode_idx, episode in enumerate(episodes):
            if not isinstance(episode, dict) or "ee_pos" not in episode or "ee_axis_angle" not in episode:
                raise ValueError(f"Episode {episode_idx} must contain 'ee_pos' and 'ee_axis_angle' arrays")
            pos = torch.as_tensor(np.asarray(episode["ee_pos"]), dtype=torch.float32)
            axis_angle = torch.as_tensor(np.asarray(episode["ee_axis_angle"]), dtype=torch.float32)
            if pos.ndim != 2 or pos.shape[-1] != 3 or axis_angle.shape != pos.shape or len(pos) < 4:
                raise ValueError(
                    f"Episode {episode_idx} must have matching (T, 3) arrays with T >= 4; "
                    f"got {tuple(pos.shape)} and {tuple(axis_angle.shape)}"
                )
            # Match the released loader: center x/y using the first three valid samples.
            pos = pos.clone()
            pos[:, :2] -= pos[1:4, :2].mean(dim=0, keepdim=True)
            angle = torch.linalg.norm(axis_angle, dim=-1)
            axis = axis_angle / angle.clamp_min(1.0e-8).unsqueeze(-1)
            quat = math_utils.quat_from_angle_axis(angle, axis)
            positions.append(pos)
            quaternions.append(quat)

        max_length = max(len(pos) for pos in positions)
        dataset_pos = torch.empty(len(positions), max_length, 3, dtype=torch.float32)
        dataset_quat = torch.empty(len(positions), max_length, 4, dtype=torch.float32)
        for episode_idx, (pos, quat) in enumerate(zip(positions, quaternions)):
            length = len(pos)
            dataset_pos[episode_idx, :length] = pos
            dataset_pos[episode_idx, length:] = pos[-1]
            dataset_quat[episode_idx, :length] = quat
            dataset_quat[episode_idx, length:] = quat[-1]
        self._dataset_pos = dataset_pos
        self._dataset_quat = dataset_quat
        self._dataset_lengths = torch.tensor([len(pos) for pos in positions], dtype=torch.long)

    def _update_metrics(self):
        pos_error, rot_error = math_utils.compute_pose_error(
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
            self.current_target_pos_w,
            self.current_target_quat_w,
        )
        self.metrics["position_error"][:] = torch.linalg.norm(pos_error, dim=-1)
        self.metrics["orientation_error"][:] = torch.linalg.norm(rot_error, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids_tensor = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids_cpu = env_ids_tensor.cpu()
        eef_pos = self.robot.data.body_pos_w[env_ids_tensor, self.body_idx]
        eef_quat = self.robot.data.body_quat_w[env_ids_tensor, self.body_idx]
        self._eef_pos_history[env_ids_tensor] = eef_pos[:, None, :]
        self._eef_quat_history[env_ids_tensor] = eef_quat[:, None, :]
        self._procedural_anchor_pos_w[env_ids_tensor] = eef_pos
        self._procedural_anchor_quat_w[env_ids_tensor] = eef_quat
        if self._dataset_pos is not None:
            self._trajectory_ids[env_ids_cpu] = torch.randint(
                len(self._dataset_pos),
                (len(env_ids_cpu),),
                device="cpu",
                generator=self._generator,
            )
        self._update_command_for_envs(env_ids_tensor, torch.zeros(len(env_ids_cpu)))

    def _update_command(self):
        eef_pos = self.robot.data.body_pos_w[:, self.body_idx]
        eef_quat = self.robot.data.body_quat_w[:, self.body_idx]
        if self._pose_history_steps > 1:
            self._eef_pos_history = torch.roll(self._eef_pos_history, shifts=-1, dims=1)
            self._eef_quat_history = torch.roll(self._eef_quat_history, shifts=-1, dims=1)
        self._eef_pos_history[:, -1] = eef_pos
        self._eef_quat_history[:, -1] = eef_quat
        env_ids = torch.arange(self.num_envs, device=self.device)
        episode_time = self._env.episode_length_buf.cpu() * self._env.step_dt
        self._update_command_for_envs(env_ids, episode_time)

    def _update_command_for_envs(self, env_ids: torch.Tensor, episode_time_cpu: torch.Tensor):
        target_pos_w, target_quat_w = self._targets_at_times(env_ids, episode_time_cpu, (0.0,))
        self.current_target_pos_w[env_ids] = target_pos_w[:, 0]
        self.current_target_quat_w[env_ids] = target_quat_w[:, 0]

        preview_pos_w, preview_quat_w = self._targets_at_times(env_ids, episode_time_cpu, self.cfg.preview_times)
        measured_pos_w = self._eef_pos_history[env_ids, 0]
        measured_quat_w = self._eef_quat_history[env_ids, 0]
        measured_pos_w = measured_pos_w[:, None, :].expand_as(preview_pos_w).reshape(-1, 3)
        measured_quat_w = measured_quat_w[:, None, :].expand_as(preview_quat_w).reshape(-1, 4)
        relative_pos, relative_quat = math_utils.subtract_frame_transforms(
            measured_pos_w,
            measured_quat_w,
            preview_pos_w.reshape(-1, 3),
            preview_quat_w.reshape(-1, 4),
        )
        relative_rot_6d = math_utils.matrix_from_quat(relative_quat)[..., :2, :].reshape(-1, 6)
        relative_pose = torch.cat(
            (
                (relative_pos * self.cfg.position_scale).reshape(len(env_ids), -1),
                (relative_rot_6d * self.cfg.orientation_scale).reshape(len(env_ids), -1),
            ),
            dim=-1,
        )
        self._command[env_ids] = relative_pose

    def _targets_at_times(
        self,
        env_ids: torch.Tensor,
        episode_time_cpu: torch.Tensor,
        offsets: tuple[float, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dataset_pos is None or self._dataset_quat is None or self._dataset_lengths is None:
            return self._procedural_targets(env_ids, episode_time_cpu, offsets)

        env_ids_cpu = env_ids.cpu()
        trajectory_ids = self._trajectory_ids[env_ids_cpu]
        times = episode_time_cpu[:, None] + torch.tensor(offsets, dtype=torch.float32)[None, :]
        frames = torch.clamp((times / self.cfg.source_dt).long(), min=0)
        max_frames = self._dataset_lengths[trajectory_ids, None] - 1
        frames = torch.minimum(frames, max_frames)
        expanded_ids = trajectory_ids[:, None].expand_as(frames)
        target_pos = self._dataset_pos[expanded_ids, frames].to(self.device)
        target_quat = self._dataset_quat[expanded_ids, frames].to(self.device)
        target_pos += self._env.scene.env_origins[env_ids, None, :]
        return target_pos, target_quat

    def _procedural_targets(
        self,
        env_ids: torch.Tensor,
        episode_time_cpu: torch.Tensor,
        offsets: tuple[float, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        times = episode_time_cpu.to(self.device)[:, None] + torch.tensor(offsets, device=self.device)[None, :]
        delta = torch.stack(
            (
                0.35 * torch.sin(0.55 * times),
                0.18 * torch.sin(0.83 * times),
                0.12 * torch.sin(1.10 * times),
            ),
            dim=-1,
        )
        target_pos = self._procedural_anchor_pos_w[env_ids, None, :] + delta
        pitch = 0.25 * torch.sin(0.70 * times)
        yaw = 0.35 * torch.sin(0.45 * times)
        zero = torch.zeros_like(times)
        delta_quat = math_utils.quat_from_euler_xyz(zero, pitch, yaw)
        anchor_quat = self._procedural_anchor_quat_w[env_ids, None, :].expand_as(delta_quat)
        target_quat = math_utils.quat_mul(anchor_quat, delta_quat)
        return target_pos, target_quat


@configclass
class EndEffectorTrajectoryCommandCfg(CommandTermCfg):
    """Configuration for :class:`EndEffectorTrajectoryCommand`."""

    class_type: type = EndEffectorTrajectoryCommand
    asset_name: str = cast(str, MISSING)
    body_name: str = cast(str, MISSING)
    trajectory_file: str | None = None
    source_dt: float = 0.005
    preview_times: tuple[float, ...] = (-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 1.0)
    pose_latency_s: float = 0.01
    position_scale: float = 10.0
    orientation_scale: float = 1.5
