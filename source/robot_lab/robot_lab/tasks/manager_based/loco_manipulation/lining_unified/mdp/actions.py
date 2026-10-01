# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Action terms for the Spot + Kortex unified-force controller."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class LiningUnifiedJointPositionAction(JointPositionAction):
    """Drive the 12 Spot leg joints and 7 Kortex arm joints with randomized motor strength.

    Unlike the B2 + Z1 reproduction, Kortex has no tool or gripper joint, so every
    actuated joint belongs to the policy and no joint is held at its default.
    """

    cfg: LiningUnifiedJointPositionActionCfg

    def __init__(self, cfg: LiningUnifiedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self.motor_strength = torch.empty(self.num_envs, self.action_dim, device=self.device).uniform_(
            *cfg.motor_strength_range
        )
        self._scale = self.motor_strength * float(cfg.scale)


@configclass
class LiningUnifiedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for :class:`LiningUnifiedJointPositionAction`."""

    class_type: type[ActionTerm] = LiningUnifiedJointPositionAction
    motor_strength_range: tuple[float, float] = (0.85, 1.15)
