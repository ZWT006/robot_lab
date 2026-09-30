# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reset events specific to the UniFP reproduction."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def reset_unified_force_joints(
    env,
    env_ids: torch.Tensor,
    leg_cfg: SceneEntityCfg,
    arm_cfg: SceneEntityCfg,
):
    """Use UniFP's multiplicative leg and additive arm reset distributions."""
    robot: Articulation = env.scene[leg_cfg.name]
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)
    leg_ids = leg_cfg.joint_ids
    arm_ids = arm_cfg.joint_ids
    joint_pos[:, leg_ids] *= torch.empty(len(env_ids), len(leg_ids), device=robot.device).uniform_(0.5, 1.5)
    joint_pos[:, arm_ids] += torch.empty(len(env_ids), len(arm_ids), device=robot.device).uniform_(-0.5, 0.5)
    joint_pos.clamp_(robot.data.soft_joint_pos_limits[env_ids, :, 0], robot.data.soft_joint_pos_limits[env_ids, :, 1])
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
