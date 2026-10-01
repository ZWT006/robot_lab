# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""UniFP-style actor, critic, and state-estimator observations for Spot + Kortex.

Frame sizes follow the number of policy joints ``N`` (19 = 12 Spot legs + 7 Kortex
arm joints): the actor frame is ``22 + 3N`` = 79-D and the critic frame is
``64 + 5N`` = 159-D, compared with 73-D and 149-D for UniFP's 17 B2 + Z1 joints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .commands import LiningUnifiedCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


NUM_LEG_JOINTS = 12
COMMAND_SCALE = torch.tensor([2.0, 2.0, 0.25, 0.5, 1.0, 1.3, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])


def _command(env: ManagerBasedRLEnv, name: str) -> LiningUnifiedCommand:
    return env.command_manager.get_term(name)


def policy_frame_dim(num_joints: int) -> int:
    """Return the actor frame size for ``num_joints`` policy joints."""
    return 2 + 3 + 3 * num_joints + 2 + COMMAND_SCALE.numel()


def _policy_joint_state(robot: Articulation, joint_ids: list[int] | slice) -> tuple[torch.Tensor, torch.Tensor]:
    pos = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
    vel = robot.data.joint_vel[:, joint_ids]
    return pos, vel


class LiningUnifiedPolicyObservation(ManagerTermBase):
    """Build the ``22 + 3N``-D proprioceptive observation frame (79-D for Spot + Kortex)."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["asset_cfg"].name]
        self.joint_ids, names = self.robot.find_joints(params["joint_names"], preserve_order=True)
        num_joints = len(params["joint_names"])
        if len(self.joint_ids) != num_joints:
            raise ValueError(f"Could not resolve all policy joints: requested={params['joint_names']}, found={names}")
        self.command_scale = COMMAND_SCALE.to(self.device)
        self.noise_scale = torch.zeros(policy_frame_dim(num_joints), device=self.device)
        self.noise_scale[:2] = 0.05
        self.noise_scale[2:5] = 0.05
        self.noise_scale[5 : 5 + num_joints] = 0.01
        self.noise_scale[5 + num_joints : 5 + 2 * num_joints] = 0.075

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        joint_names: list[str],
        add_noise: bool,
    ) -> torch.Tensor:
        command = _command(env, command_name)
        roll, pitch, _ = math_utils.euler_xyz_from_quat(self.robot.data.root_link_quat_w)
        joint_pos, joint_vel = _policy_joint_state(self.robot, self.joint_ids)
        phase = 2.0 * torch.pi * command.gait_phase
        observation = torch.cat(
            (
                torch.stack((roll, pitch), dim=-1),
                self.robot.data.root_link_ang_vel_b * 0.25,
                joint_pos,
                joint_vel * 0.05,
                env.action_manager.action,
                torch.sin(phase).unsqueeze(-1),
                torch.cos(phase).unsqueeze(-1),
                command.command * self.command_scale,
            ),
            dim=-1,
        )
        if add_noise:
            observation = observation + torch.randn_like(observation) * self.noise_scale
        return observation


class LiningUnifiedCriticObservation(ManagerTermBase):
    """Build the ``64 + 5N``-D asymmetric critic frame (159-D for Spot + Kortex)."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["asset_cfg"].name]
        self.sensor: ContactSensor = env.scene.sensors[params["sensor_cfg"].name]
        self.joint_ids, names = self.robot.find_joints(params["joint_names"], preserve_order=True)
        self.leg_joint_ids, _ = self.robot.find_joints(params["leg_joint_names"], preserve_order=True)
        self.foot_ids, _ = self.sensor.find_bodies(params["foot_body_names"], preserve_order=True)
        self.base_id = self.robot.find_bodies(params["base_body_name"])[0][0]
        self.payload_id = self.robot.find_bodies(params["payload_body_name"])[0][0]
        self.num_joints = len(params["joint_names"])
        if (
            len(self.joint_ids) != self.num_joints
            or len(self.leg_joint_ids) != NUM_LEG_JOINTS
            or len(self.foot_ids) != 4
        ):
            raise ValueError(
                f"Unexpected unified-force dimensions: policy={names}, legs={self.leg_joint_ids}, feet={self.foot_ids}"
            )
        self.command_scale = COMMAND_SCALE.to(self.device)

        # Observation terms are constructed before startup randomization.  Keep
        # these nominal values so the privileged parameters are true deltas.
        self._default_masses = self.robot.root_physx_view.get_masses().clone()
        self._default_coms = self.robot.root_physx_view.get_coms()[..., :3].clone()
        self._physics_params: torch.Tensor | None = None

    def _read_physics_params(self) -> torch.Tensor:
        masses = self.robot.root_physx_view.get_masses()
        coms = self.robot.root_physx_view.get_coms()[..., :3]
        mass_delta = masses - self._default_masses
        com_delta = coms - self._default_coms
        # UniFP reserves one link-mass slot per policy joint (17 for B2 + Z1, 19
        # here). They remain zero because link-mass DR is disabled.
        link_mass_delta = torch.zeros(self.num_envs, self.num_joints)
        mass_params = torch.cat(
            (
                mass_delta[:, self.base_id : self.base_id + 1],
                com_delta[:, self.base_id],
                mass_delta[:, self.payload_id : self.payload_id + 1],
                link_mass_delta,
            ),
            dim=-1,
        )
        friction = self.robot.root_physx_view.get_material_properties()[:, 0, 0:1]
        return torch.cat((mass_params, friction), dim=-1).to(self.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        self._physics_params = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        action_name: str,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        joint_names: list[str],
        leg_joint_names: list[str],
        foot_body_names: list[str],
        base_body_name: str,
        payload_body_name: str,
    ) -> torch.Tensor:
        command = _command(env, command_name)
        if self._physics_params is None:
            self._physics_params = self._read_physics_params()
        ee_force_b, base_force_b = command.external_forces_b()
        joint_pos, joint_vel = _policy_joint_state(self.robot, self.joint_ids)
        leg_diff = self.robot.data.joint_pos[:, self.leg_joint_ids] - command.reference_leg_joint_pos
        stance = command.stance_mask.float()
        contact = (self.sensor.data.net_forces_w[:, self.foot_ids, 2] > 5.0).float()
        action_term = env.action_manager.get_term(action_name)
        strength_delta = action_term.motor_strength - 1.0
        phase = 2.0 * torch.pi * command.gait_phase
        compliant_sphere = command.compliant_target_sphere()
        compliant_sphere = compliant_sphere * self.command_scale[3:6]
        return torch.cat(
            (
                self.robot.data.root_link_lin_vel_b * 2.0,
                command.measured_ee_sphere() * self.command_scale[3:6],
                ee_force_b * 0.01,
                base_force_b * 0.01,
                leg_diff,
                self._physics_params,
                strength_delta,
                stance,
                contact,
                self.robot.data.projected_gravity_b,
                self.robot.data.root_link_ang_vel_b * 0.25,
                joint_pos,
                joint_vel * 0.05,
                env.action_manager.action,
                torch.sin(phase).unsqueeze(-1),
                torch.cos(phase).unsqueeze(-1),
                command.command * self.command_scale,
                compliant_sphere,
            ),
            dim=-1,
        )


class LiningUnifiedEstimatorTarget(ManagerTermBase):
    """Return the 12-D concurrent state-estimation supervision target."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.ee_scale = COMMAND_SCALE[3:6].to(self.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        command = _command(env, command_name)
        ee_force_b, base_force_b = command.external_forces_b()
        return torch.cat(
            (
                self.robot.data.root_link_lin_vel_b * 2.0,
                command.measured_ee_sphere() * self.ee_scale,
                ee_force_b * 0.01,
                base_force_b * 0.01,
            ),
            dim=-1,
        )
