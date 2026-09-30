# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""UniFP-compatible actor, critic, and state-estimator observations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .commands import UnifiedForceCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


COMMAND_SCALE = torch.tensor([2.0, 2.0, 0.25, 0.5, 1.0, 1.3, 0.5, 0.5, 0.5, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])


def _command(env: ManagerBasedRLEnv, name: str) -> UnifiedForceCommand:
    return env.command_manager.get_term(name)


def _policy_joint_state(robot: Articulation, joint_ids: list[int] | slice) -> tuple[torch.Tensor, torch.Tensor]:
    pos = robot.data.joint_pos[:, joint_ids] - robot.data.default_joint_pos[:, joint_ids]
    vel = robot.data.joint_vel[:, joint_ids]
    return pos, vel


class UnifiedForcePolicyObservation(ManagerTermBase):
    """Build the released 73-D proprioceptive observation frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["asset_cfg"].name]
        self.joint_ids, names = self.robot.find_joints(params["joint_names"], preserve_order=True)
        if len(self.joint_ids) != 17:
            raise ValueError(f"Unified-force policy expects 17 joints, found {names}")
        self.command_scale = COMMAND_SCALE.to(self.device)
        self.noise_scale = torch.zeros(73, device=self.device)
        self.noise_scale[:2] = 0.05
        self.noise_scale[2:5] = 0.05
        self.noise_scale[5:22] = 0.01
        self.noise_scale[22:39] = 0.075

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


class UnifiedForceCriticObservation(ManagerTermBase):
    """Build the released 149-D asymmetric critic observation frame."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        self.robot: Articulation = env.scene[params["asset_cfg"].name]
        self.sensor: ContactSensor = env.scene.sensors[params["sensor_cfg"].name]
        self.joint_ids, names = self.robot.find_joints(params["joint_names"], preserve_order=True)
        self.leg_joint_ids, _ = self.robot.find_joints(params["leg_joint_names"], preserve_order=True)
        self.foot_ids, _ = self.sensor.find_bodies(params["foot_body_names"], preserve_order=True)
        self.base_id = self.robot.find_bodies(params["base_body_name"])[0][0]
        self.ee_id = self.robot.find_bodies(params["ee_body_name"])[0][0]
        if len(self.joint_ids) != 17 or len(self.leg_joint_ids) != 12 or len(self.foot_ids) != 4:
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
        # UniFP reserves 17 slots for link-mass perturbations. They remain zero
        # with the released configuration because link-mass DR is disabled.
        link_mass_delta = torch.zeros(self.num_envs, 17)
        mass_params = torch.cat(
            (
                mass_delta[:, self.base_id : self.base_id + 1],
                com_delta[:, self.base_id],
                mass_delta[:, self.ee_id : self.ee_id + 1],
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
        ee_body_name: str,
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


class UnifiedForceEstimatorTarget(ManagerTermBase):
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
