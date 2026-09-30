# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Action terms for the B2 + Z1 unified-force controller."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class UnifiedForceJointPositionAction(JointPositionAction):
    """Drive the 17 policy joints and hold the two tool joints at their defaults."""

    cfg: UnifiedForceJointPositionActionCfg

    def __init__(self, cfg: UnifiedForceJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._fixed_joint_ids, fixed_names = self._asset.find_joints(cfg.fixed_joint_names, preserve_order=True)
        if len(self._fixed_joint_ids) != len(cfg.fixed_joint_names):
            raise ValueError(
                f"Could not resolve all fixed tool joints: requested={cfg.fixed_joint_names}, found={fixed_names}"
            )
        self._fixed_targets = self._asset.data.default_joint_pos[:, self._fixed_joint_ids].clone()
        self.motor_strength = torch.empty(self.num_envs, self.action_dim, device=self.device).uniform_(
            *cfg.motor_strength_range
        )
        self._scale = self.motor_strength * float(cfg.scale)

    def apply_actions(self):
        super().apply_actions()
        self._asset.set_joint_position_target(self._fixed_targets, joint_ids=self._fixed_joint_ids)


@configclass
class UnifiedForceJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`UnifiedForceJointPositionAction`."""

    class_type: type[ActionTerm] = UnifiedForceJointPositionAction
    fixed_joint_names: list[str] = MISSING
    motor_strength_range: tuple[float, float] = (0.85, 1.15)
