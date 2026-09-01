# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Modified in 2026 for robot_lab and Isaac Lab 2.2 compatibility.

"""Custom actuator definitions for Spot"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.actuators.actuator_cfg import RemotizedPDActuatorCfg
from isaaclab.actuators.actuator_pd import RemotizedPDActuator
from isaaclab.utils import LinearInterpolation, configclass
from isaaclab.utils.types import ArticulationActions
from torch._tensor import Tensor

from robot_lab.assets.relic_spot_constants import NEG_TORQUE_SPEED_LIMIT, POS_TORQUE_SPEED_LIMIT


class SpotKneeActuator(RemotizedPDActuator):
    """Spot knee actuator."""

    def __init__(
        self,
        cfg: RemotizedPDActuatorCfg,
        joint_names: list[str],
        joint_ids: Sequence[int],
        num_envs: int,
        device: str,
        stiffness: Tensor | float = 0,
        damping: Tensor | float = 0,
        armature: Tensor | float = 0,
        friction: Tensor | float = 0,
        dynamic_friction: Tensor | float = 0,
        viscous_friction: Tensor | float = 0,
        effort_limit: Tensor | float = torch.inf,
        velocity_limit: Tensor | float = torch.inf,
    ):

        super().__init__(
            cfg,
            joint_names,
            joint_ids,
            num_envs,
            device,
            stiffness,
            damping,
            armature,
            friction,
            dynamic_friction,
            viscous_friction,
            effort_limit,
            velocity_limit,
        )

        self._pos_torque_speed_data = torch.tensor(cfg.pos_torque_speed_limit, device=self._device)
        self._neg_torque_speed_data = torch.tensor(cfg.neg_torque_speed_limit, device=self._device)
        self._enable_torque_speed_limit = cfg.enable_torque_speed_limit

        # define remotized joint torque limit
        self._pos_torque_speed_limit = LinearInterpolation(
            self._pos_torque_speed_data[:, 0],
            self._pos_torque_speed_data[:, 1],
            device=self._device,
        )
        self._neg_torque_speed_limit = LinearInterpolation(
            self._neg_torque_speed_data[:, 0],
            self._neg_torque_speed_data[:, 1],
            device=self._device,
        )

    def compute(
        self,
        control_action: ArticulationActions,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> ArticulationActions:
        """Compute the control action for the Spot robot with positional torque speed limits."""
        control_action = super().compute(control_action, joint_pos, joint_vel)

        # compute torque-speed limits
        if self._enable_torque_speed_limit:
            pos_torque_limits = self._pos_torque_speed_limit.compute(joint_vel)
            neg_torque_limits = self._neg_torque_speed_limit.compute(joint_vel)
            control_action.joint_efforts = torch.clamp(
                control_action.joint_efforts,
                min=neg_torque_limits,
                max=pos_torque_limits,
            )
        self.applied_effort = control_action.joint_efforts
        return control_action


@configclass
class SpotKneeActuatorCfg(RemotizedPDActuatorCfg):
    """Configuration for the ReLIC Spot knee actuator."""

    class_type: type = SpotKneeActuator

    enable_torque_speed_limit: bool = False
    pos_torque_speed_limit: list[list[float]] = POS_TORQUE_SPEED_LIMIT
    neg_torque_speed_limit: list[list[float]] = NEG_TORQUE_SPEED_LIMIT
