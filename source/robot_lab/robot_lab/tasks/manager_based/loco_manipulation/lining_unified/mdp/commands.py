# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Unified locomotion, end-effector pose, and force commands for Spot + Kortex.

Adapted from the UniFP B2 + Z1 reproduction in ``unified_force``. The command
layout and force-pulse logic are unchanged; the spherical workspace and the
end-effector frame convention are re-derived for the Kortex Gen3 arm mounted at
the center of Spot's body.
"""

from __future__ import annotations

import math
import weakref
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, cast

import omni.kit.app
import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def sphere_to_cartesian(sphere: torch.Tensor) -> torch.Tensor:
    """Convert ``(radius, pitch, yaw)`` coordinates to Cartesian coordinates."""
    radius, pitch, yaw = sphere.unbind(dim=-1)
    return torch.stack(
        (
            radius * torch.cos(pitch) * torch.cos(yaw),
            radius * torch.cos(pitch) * torch.sin(yaw),
            radius * torch.sin(pitch),
        ),
        dim=-1,
    )


def cartesian_to_sphere(cartesian: torch.Tensor) -> torch.Tensor:
    """Convert Cartesian coordinates to ``(radius, pitch, yaw)`` coordinates."""
    radius = torch.linalg.norm(cartesian, dim=-1).clamp_min(1.0e-6)
    pitch = torch.asin(torch.clamp(cartesian[..., 2] / radius, -1.0, 1.0))
    yaw = torch.atan2(cartesian[..., 1], cartesian[..., 0])
    return torch.stack((radius, pitch, yaw), dim=-1)


class LiningUnifiedCommand(CommandTerm):
    """Generate the 15-D UniFP-style command and its commanded/external force pulses.

    The command layout matches the released UniFP implementation:

    ``[vx, vy, wz, ee_radius, ee_pitch, ee_yaw, ee_dr, ee_dp, ee_dy,
    ee_fx, ee_fy, ee_fz, base_fx, base_fy, base_fz]``.

    Force-command pulses do not directly push the robot. They offset the target
    through the configured virtual stiffness/damping. External-force pulses are
    written to the articulation and provide the disturbance that the concurrent
    state-estimation module must infer from proprioceptive history.

    The end-effector orientation target is built in a canonical "pointing" frame
    whose +x axis is the tool axis, then mapped to the robot's EE link through
    ``ee_frame_offset_euler``. Z1's tool axis is its EE +x axis, whereas the Kortex
    ``arm_end_effector_link`` points its tool along +z, so the offset is
    ``Ry(pi/2)`` here instead of UniFP's ``Rx(pi/2)``.
    """

    cfg: LiningUnifiedCommandCfg

    def __init__(self, cfg: LiningUnifiedCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if cfg.external_force_arrow_scale <= 0.0:
            raise ValueError("external_force_arrow_scale must be positive")
        self.robot: Articulation = env.scene[cfg.asset_name]
        ee_ids, _ = self.robot.find_bodies(cfg.ee_body_name)
        base_ids, _ = self.robot.find_bodies(cfg.base_body_name)
        if len(ee_ids) != 1 or len(base_ids) != 1:
            raise ValueError(f"Expected one EE body and one base body, found ee={len(ee_ids)}, base={len(base_ids)}")
        self.ee_body_id = ee_ids[0]
        self.base_body_id = base_ids[0]
        self._wrench_body_ids = [self.base_body_id, self.ee_body_id]
        leg_joint_ids, leg_joint_names = self.robot.find_joints(cfg.leg_joint_names, preserve_order=True)
        if len(leg_joint_ids) != 12:
            raise ValueError(f"Expected 12 leg joints, found {leg_joint_names}")
        self.leg_joint_ids = leg_joint_ids

        self._command = torch.zeros(self.num_envs, 15, device=self.device)
        self.velocity_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self._velocity_time_left = torch.zeros(self.num_envs, device=self.device)

        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros_like(self.ee_start_sphere)
        self.ee_current_sphere = torch.zeros_like(self.ee_start_sphere)
        self.ee_orientation_delta = torch.zeros_like(self.ee_start_sphere)
        self.ee_target_pos_w = torch.zeros_like(self.ee_start_sphere)
        self.ee_target_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self.ee_target_quat_w[:, 0] = 1.0
        offset = torch.tensor(cfg.ee_frame_offset_euler, device=self.device)
        self._ee_frame_offset_quat = math_utils.quat_from_euler_xyz(offset[0], offset[1], offset[2]).expand(
            self.num_envs, -1
        )
        self._goal_elapsed = torch.zeros(self.num_envs, device=self.device)
        self._goal_travel_time = torch.ones(self.num_envs, device=self.device)
        self._goal_total_time = torch.ones(self.num_envs, device=self.device)
        self._is_first_goal = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.gait_phase = torch.zeros(self.num_envs, device=self.device)
        self.ee_stiffness = torch.full((self.num_envs, 3), cfg.ee_stiffness, device=self.device)
        self.base_damping = torch.full((self.num_envs, 3), cfg.base_damping, device=self.device)

        self._pulses: dict[str, dict[str, torch.Tensor]] = {}
        for name in ("ee_command", "ee_external", "base_command", "base_external"):
            self._pulses[name] = {
                "active": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "time_to_start": torch.zeros(self.num_envs, device=self.device),
                "phase": torch.zeros(self.num_envs, device=self.device),
                "duration": torch.ones(self.num_envs, device=self.device),
                "target": torch.zeros(self.num_envs, 3, device=self.device),
                "current": torch.zeros(self.num_envs, 3, device=self.device),
            }

        self.metrics["ee_position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["ee_orientation_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["base_velocity_error"] = torch.zeros(self.num_envs, device=self.device)

        self._external_force_debug_vis_handle = None
        if cfg.external_force_debug_vis:
            self.set_external_force_debug_vis(True)

    def __del__(self):
        """Unsubscribe the external-force callback before the command is destroyed."""
        if getattr(self, "_external_force_debug_vis_handle", None):
            self._external_force_debug_vis_handle.unsubscribe()
            self._external_force_debug_vis_handle = None
        super().__del__()

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def ee_force_command_b(self) -> torch.Tensor:
        return self._pulses["ee_command"]["current"]

    @property
    def base_force_command_b(self) -> torch.Tensor:
        return self._pulses["base_command"]["current"]

    @property
    def ee_external_force_w(self) -> torch.Tensor:
        return self._pulses["ee_external"]["current"]

    @property
    def base_external_force_w(self) -> torch.Tensor:
        return self._pulses["base_external"]["current"]

    @property
    def yaw_quat_w(self) -> torch.Tensor:
        return math_utils.yaw_quat(self.robot.data.root_link_quat_w)

    @property
    def walking_mask(self) -> torch.Tensor:
        command = self.velocity_command_b
        return (
            (torch.abs(command[:, 0]) > self.cfg.lin_vel_x_deadband)
            | (torch.abs(command[:, 1]) > self.cfg.lin_vel_y_deadband)
            | (torch.abs(command[:, 2]) > self.cfg.ang_vel_z_deadband)
        )

    @property
    def stance_mask(self) -> torch.Tensor:
        sine = torch.sin(2.0 * torch.pi * self.gait_phase)
        left = sine + self.cfg.gait_double_support_threshold
        right = sine - self.cfg.gait_double_support_threshold
        stance = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        stance[:, 0] = left >= 0.0
        stance[:, 3] = left >= 0.0
        stance[:, 1] = right < 0.0
        stance[:, 2] = right < 0.0
        return stance

    @property
    def reference_leg_joint_pos(self) -> torch.Tensor:
        """Return the released diagonal-trot reference in FL, FR, RL, RR order."""
        default = self.robot.data.default_joint_pos[:, self.leg_joint_ids].clone()
        sine = torch.sin(2.0 * torch.pi * self.gait_phase)
        left = sine + self.cfg.gait_double_support_threshold
        right = sine - self.cfg.gait_double_support_threshold
        left = torch.where(left > 0.0, torch.zeros_like(left), left)
        right = torch.where(right < 0.0, torch.zeros_like(right), right)
        scale_1 = self.cfg.gait_joint_scale / (1.0 - self.cfg.gait_double_support_threshold)
        scale_2 = 2.0 * scale_1
        default[:, 1] -= left * scale_1
        default[:, 2] += left * scale_2
        default[:, 10] -= left * scale_1
        default[:, 11] += left * scale_2
        default[:, 4] += right * scale_1
        default[:, 5] -= right * scale_2
        default[:, 7] += right * scale_1
        default[:, 8] -= right * scale_2
        return default

    @property
    def ee_force_command_w(self) -> torch.Tensor:
        return math_utils.quat_apply(self.yaw_quat_w, self.ee_force_command_b)

    @property
    def ee_compliant_target_pos_w(self) -> torch.Tensor:
        force_offset = self.ee_external_force_w + self.ee_force_command_w
        return self.ee_target_pos_w + force_offset / self.ee_stiffness

    @property
    def base_velocity_target_b(self) -> torch.Tensor:
        external_force_b = math_utils.quat_apply_inverse(self.yaw_quat_w, self.base_external_force_w)
        target = self.velocity_command_b.clone()
        target[:, :2] += ((external_force_b + self.base_force_command_b) / self.base_damping)[:, :2]
        moving = (
            (torch.abs(target[:, 0]) > self.cfg.lin_vel_x_deadband)
            | (torch.abs(target[:, 1]) > self.cfg.lin_vel_y_deadband)
            | (torch.abs(target[:, 2]) > self.cfg.ang_vel_z_deadband)
        )
        target[:, :2] *= moving.unsqueeze(-1)
        return target

    def measured_ee_sphere(self) -> torch.Tensor:
        center_w = self._spherical_center_w()
        ee_local = math_utils.quat_apply_inverse(
            self.yaw_quat_w,
            self.robot.data.body_link_pos_w[:, self.ee_body_id] - center_w,
        )
        return cartesian_to_sphere(ee_local)

    def compliant_target_sphere(self) -> torch.Tensor:
        target_local = math_utils.quat_apply_inverse(
            self.yaw_quat_w,
            self.ee_compliant_target_pos_w - self._spherical_center_w(),
        )
        return cartesian_to_sphere(target_local)

    def external_forces_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        yaw = self.yaw_quat_w
        return (
            math_utils.quat_apply_inverse(yaw, self.ee_external_force_w),
            math_utils.quat_apply_inverse(yaw, self.base_external_force_w),
        )

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Toggle the world-frame force-compliant target EE pose marker."""
        if debug_vis:
            if not hasattr(self, "target_pose_visualizer"):
                self.target_pose_visualizer = VisualizationMarkers(self.cfg.target_pose_visualizer_cfg)
            self.target_pose_visualizer.set_visibility(True)
        elif hasattr(self, "target_pose_visualizer"):
            self.target_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        self.target_pose_visualizer.visualize(self.ee_compliant_target_pos_w, self.ee_target_quat_w)

    def set_external_force_debug_vis(self, debug_vis: bool):
        """Toggle arrows for the external forces applied at the base and EE."""
        if debug_vis:
            if not hasattr(self, "external_force_visualizer"):
                self.external_force_visualizer = VisualizationMarkers(self.cfg.external_force_visualizer_cfg)
            self.external_force_visualizer.set_visibility(True)
            if self._external_force_debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._external_force_debug_vis_handle = (
                    app_interface.get_post_update_event_stream().create_subscription_to_pop(
                        lambda event, obj=weakref.proxy(self): obj._external_force_debug_vis_callback(event)
                    )
                )
        else:
            if hasattr(self, "external_force_visualizer"):
                self.external_force_visualizer.set_visibility(False)
            if self._external_force_debug_vis_handle is not None:
                self._external_force_debug_vis_handle.unsubscribe()
                self._external_force_debug_vis_handle = None

    def _external_force_debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        forces_w = torch.stack((self.base_external_force_w, self.ee_external_force_w), dim=1).reshape(-1, 3)
        positions_w = self.robot.data.body_com_pos_w[:, self._wrench_body_ids].reshape(-1, 3)
        magnitudes = torch.linalg.norm(forces_w, dim=-1)
        orientations = self._vector_to_arrow_orientation(forces_w, magnitudes)

        scales = torch.ones_like(forces_w)
        scales[:, 0] = magnitudes * self.cfg.external_force_arrow_scale
        scales[magnitudes <= 1.0e-6] = 0.0
        self.external_force_visualizer.visualize(positions_w, orientations, scales)

    def _vector_to_arrow_orientation(
        self, vectors: torch.Tensor, magnitudes: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Rotate the marker's +X axis onto each world-frame vector."""
        if magnitudes is None:
            magnitudes = torch.linalg.norm(vectors, dim=-1)
        default_direction = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=vectors.dtype).expand_as(vectors)
        unit_direction = torch.where(
            magnitudes.unsqueeze(-1) > 1.0e-6,
            vectors / magnitudes.unsqueeze(-1).clamp_min(1.0e-6),
            default_direction,
        )

        rotation_axis = torch.linalg.cross(default_direction, unit_direction, dim=-1)
        axis_norm = torch.linalg.norm(rotation_axis, dim=-1)
        fallback_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=vectors.dtype).expand_as(vectors)
        rotation_axis = torch.where(
            axis_norm.unsqueeze(-1) > 1.0e-6,
            rotation_axis / axis_norm.unsqueeze(-1).clamp_min(1.0e-6),
            fallback_axis,
        )

        angle = torch.acos(torch.clamp(torch.sum(default_direction * unit_direction, dim=-1), -1.0, 1.0))
        angle = torch.where(magnitudes > 1.0e-6, angle, torch.zeros_like(angle))
        return math_utils.quat_from_angle_axis(angle, rotation_axis)

    def _update_metrics(self):
        ee_position = self.robot.data.body_link_pos_w[:, self.ee_body_id]
        self.metrics["ee_position_error"] += torch.linalg.norm(ee_position - self.ee_compliant_target_pos_w, dim=-1)
        self.metrics["ee_orientation_error"] += math_utils.quat_error_magnitude(
            self.robot.data.body_link_quat_w[:, self.ee_body_id], self.ee_target_quat_w
        )
        self.metrics["base_velocity_error"] += torch.linalg.norm(
            self.robot.data.root_link_lin_vel_b[:, :2] - self.base_velocity_target_b[:, :2], dim=-1
        )

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = self._as_tensor(env_ids)
        self._resample_velocity(env_ids)

        initial_start = torch.tensor(self.cfg.initial_ee_start, device=self.device)
        initial_end = torch.tensor(self.cfg.initial_ee_end, device=self.device)
        self.ee_start_sphere[env_ids] = initial_start
        self.ee_goal_sphere[env_ids] = initial_end
        self.ee_current_sphere[env_ids] = initial_start
        self.ee_orientation_delta[env_ids] = 0.0
        self._goal_elapsed[env_ids] = 0.0
        self._sample_goal_times(env_ids)
        self._is_first_goal[env_ids] = True
        self.gait_phase[env_ids] = 0.0

        self._reset_pulse("ee_command", env_ids, self.cfg.ee_force_interval_range_s)
        self._reset_pulse("ee_external", env_ids, self.cfg.ee_force_interval_range_s)
        self._reset_pulse("base_command", env_ids, self.cfg.base_force_interval_range_s)
        self._reset_pulse("base_external", env_ids, self.cfg.base_force_interval_range_s)
        self._refresh_ee_target()
        self._assemble_command()
        self._write_external_wrenches()

    def _update_command(self):
        dt = self._env.step_dt
        self._velocity_time_left -= dt
        velocity_ids = (self._velocity_time_left <= 0.0).nonzero().flatten()
        if len(velocity_ids) > 0:
            self._resample_velocity(velocity_ids)

        self._advance_ee_goal(dt)
        self.gait_phase = torch.remainder(self.gait_phase + dt / self.cfg.gait_cycle_time, 1.0)
        self.gait_phase[~self.walking_mask] = 0.0

        if self._env.common_step_counter >= self.cfg.force_start_step:
            self._update_pulse(
                "ee_command",
                dt,
                self.cfg.ee_force_interval_range_s,
                self.cfg.ee_force_duration_range_s,
                self.cfg.ee_force_settling_time_s,
                self.cfg.ee_force_range,
                self.cfg.ee_force_active_probability,
            )
            self._update_pulse(
                "ee_external",
                dt,
                self.cfg.ee_force_interval_range_s,
                self.cfg.ee_force_duration_range_s,
                self.cfg.ee_force_settling_time_s,
                self.cfg.ee_force_range,
                self.cfg.ee_force_active_probability,
            )
            self._update_pulse(
                "base_command",
                dt,
                self.cfg.base_force_interval_range_s,
                self.cfg.base_force_duration_range_s,
                self.cfg.base_force_settling_time_s,
                self.cfg.base_force_range,
                self.cfg.base_force_active_probability,
            )
            if self.cfg.apply_base_external_force:
                self._update_pulse(
                    "base_external",
                    dt,
                    self.cfg.base_force_interval_range_s,
                    self.cfg.base_force_duration_range_s,
                    self.cfg.base_force_settling_time_s,
                    self.cfg.base_force_range,
                    self.cfg.base_force_active_probability,
                )

        self._assemble_command()
        self._write_external_wrenches()

    def _resample_velocity(self, env_ids: torch.Tensor):
        count = len(env_ids)
        if count == 0:
            return
        r = torch.empty(count, device=self.device)
        self.velocity_command_b[env_ids, 0] = r.uniform_(*self.cfg.lin_vel_x_range)
        self.velocity_command_b[env_ids, 1] = r.uniform_(*self.cfg.lin_vel_y_range)
        self.velocity_command_b[env_ids, 2] = r.uniform_(*self.cfg.ang_vel_z_range)
        standing = torch.rand(count, device=self.device) < self.cfg.zero_velocity_probability
        self.velocity_command_b[env_ids[standing]] = 0.0
        command = self.velocity_command_b[env_ids]
        moving = (
            (torch.abs(command[:, 0]) > self.cfg.lin_vel_x_deadband)
            | (torch.abs(command[:, 1]) > self.cfg.lin_vel_y_deadband)
            | (torch.abs(command[:, 2]) > self.cfg.ang_vel_z_deadband)
        )
        self.velocity_command_b[env_ids] *= moving.unsqueeze(-1)
        self._velocity_time_left[env_ids] = r.uniform_(*self.cfg.velocity_resampling_time_range_s)

    def _advance_ee_goal(self, dt: float):
        self._goal_elapsed += dt
        completed = (self._goal_elapsed >= self._goal_total_time).nonzero().flatten()
        if len(completed) > 0:
            self.ee_start_sphere[completed] = self.ee_goal_sphere[completed]
            self._sample_random_goal(completed)
            self._goal_elapsed[completed] = 0.0
            self._sample_goal_times(completed)
            self._is_first_goal[completed] = False
        interpolation = torch.clamp(self._goal_elapsed / self._goal_travel_time, 0.0, 1.0)
        self.ee_current_sphere = torch.lerp(
            self.ee_start_sphere,
            self.ee_goal_sphere,
            interpolation.unsqueeze(-1),
        )
        self._refresh_ee_target()

    def _sample_random_goal(self, env_ids: torch.Tensor):
        unresolved = env_ids
        for _ in range(self.cfg.collision_check_attempts):
            candidate = self._sample_sphere(len(unresolved))
            self.ee_goal_sphere[unresolved] = candidate
            collision = self._trajectory_collides(
                self.ee_start_sphere[unresolved],
                candidate,
            )
            unresolved = unresolved[collision]
            if len(unresolved) == 0:
                break
        r = torch.empty(len(env_ids), device=self.device)
        self.ee_orientation_delta[env_ids, 0] = r.uniform_(*self.cfg.ee_delta_roll_range)
        self.ee_orientation_delta[env_ids, 1] = r.uniform_(*self.cfg.ee_delta_pitch_range)
        self.ee_orientation_delta[env_ids, 2] = r.uniform_(*self.cfg.ee_delta_yaw_range)

    def _sample_sphere(self, count: int) -> torch.Tensor:
        sample = torch.empty(count, 3, device=self.device)
        sample[:, 0].uniform_(*self.cfg.ee_radius_range)
        sample[:, 1].uniform_(*self.cfg.ee_pitch_range)
        sample[:, 2].uniform_(*self.cfg.ee_yaw_range)
        return sample

    def _trajectory_collides(self, start: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        alpha = torch.linspace(0.0, 1.0, self.cfg.collision_check_samples, device=self.device)
        trajectory = torch.lerp(start[:, None, :], goal[:, None, :], alpha[None, :, None])
        cartesian = sphere_to_cartesian(trajectory)
        lower = torch.tensor(self.cfg.collision_box_lower, device=self.device)
        upper = torch.tensor(self.cfg.collision_box_upper, device=self.device)
        inside_box = torch.all((cartesian > lower) & (cartesian < upper), dim=-1)
        underground = cartesian[..., 2] < self.cfg.underground_limit
        return torch.any(inside_box | underground, dim=1)

    def _sample_goal_times(self, env_ids: torch.Tensor):
        travel = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ee_trajectory_time_range_s)
        hold = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.ee_hold_time_range_s)
        self._goal_travel_time[env_ids] = travel
        self._goal_total_time[env_ids] = self._goal_travel_time[env_ids] + hold

    def _refresh_ee_target(self):
        local_target = sphere_to_cartesian(self.ee_current_sphere)
        target_offset_w = math_utils.quat_apply(self.yaw_quat_w, local_target)
        self.ee_target_pos_w = self._spherical_center_w() + target_offset_w
        default_yaw = torch.atan2(target_offset_w[:, 1], target_offset_w[:, 0])
        default_pitch = -self.ee_current_sphere[:, 1] + self.cfg.arm_induced_pitch
        # Tool axis points along the target ray (pitch is positive downward), then
        # the arm-specific offset maps the canonical pointing frame onto the EE link.
        pointing_quat = math_utils.quat_from_euler_xyz(
            self.ee_orientation_delta[:, 0],
            default_pitch + self.ee_orientation_delta[:, 1],
            default_yaw + self.ee_orientation_delta[:, 2],
        )
        self.ee_target_quat_w = math_utils.quat_mul(pointing_quat, self._ee_frame_offset_quat)

    def _spherical_center_w(self) -> torch.Tensor:
        center = torch.cat(
            (self.robot.data.root_link_pos_w[:, :2], torch.zeros(self.num_envs, 1, device=self.device)),
            dim=-1,
        )
        offset = torch.tensor(self.cfg.sphere_center_offset, device=self.device).expand(self.num_envs, -1)
        return center + math_utils.quat_apply(self.yaw_quat_w, offset)

    def _reset_pulse(self, name: str, env_ids: torch.Tensor, interval_range: tuple[float, float]):
        pulse = self._pulses[name]
        pulse["active"][env_ids] = False
        pulse["phase"][env_ids] = 0.0
        pulse["target"][env_ids] = 0.0
        pulse["current"][env_ids] = 0.0
        pulse["time_to_start"][env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(*interval_range)

    def _update_pulse(
        self,
        name: str,
        dt: float,
        interval_range: tuple[float, float],
        duration_range: tuple[float, float],
        settling_time: float,
        force_range: tuple[float, float],
        active_probability: float,
    ):
        pulse = self._pulses[name]
        inactive = ~pulse["active"]
        pulse["time_to_start"][inactive] -= dt
        candidates = (inactive & (pulse["time_to_start"] <= 0.0)).nonzero().flatten()
        if len(candidates) > 0:
            selected_mask = torch.rand(len(candidates), device=self.device) < active_probability
            selected = candidates[selected_mask]
            skipped = candidates[~selected_mask]
            if len(selected) > 0:
                pulse["active"][selected] = True
                pulse["phase"][selected] = 0.0
                pulse["duration"][selected] = torch.empty(len(selected), device=self.device).uniform_(*duration_range)
                pulse["target"][selected] = torch.empty(len(selected), 3, device=self.device).uniform_(*force_range)
            if len(skipped) > 0:
                pulse["time_to_start"][skipped] = torch.empty(len(skipped), device=self.device).uniform_(*interval_range)

        active_ids = pulse["active"].nonzero().flatten()
        if len(active_ids) == 0:
            return
        pulse["phase"][active_ids] += dt
        phase = pulse["phase"][active_ids]
        duration = pulse["duration"][active_ids]
        amplitude = torch.where(
            phase < duration,
            phase / duration,
            torch.where(
                phase < duration + settling_time,
                torch.ones_like(phase),
                1.0 - (phase - duration - settling_time) / duration,
            ),
        ).clamp(0.0, 1.0)
        pulse["current"][active_ids] = pulse["target"][active_ids] * amplitude.unsqueeze(-1)
        finished = active_ids[phase >= (2.0 * duration + settling_time)]
        if len(finished) > 0:
            pulse["active"][finished] = False
            pulse["phase"][finished] = 0.0
            pulse["target"][finished] = 0.0
            pulse["current"][finished] = 0.0
            pulse["time_to_start"][finished] = torch.empty(len(finished), device=self.device).uniform_(*interval_range)

    def _assemble_command(self):
        self._command[:, :3] = self.velocity_command_b
        self._command[:, 3:6] = self.ee_current_sphere
        self._command[:, 6:9] = self.ee_orientation_delta
        self._command[:, 9:12] = self.ee_force_command_b
        self._command[:, 12:15] = self.base_force_command_b

    def _write_external_wrenches(self):
        forces = torch.stack((self.base_external_force_w, self.ee_external_force_w), dim=1)
        torques = torch.zeros_like(forces)
        self.robot.set_external_force_and_torque(
            forces=forces,
            torques=torques,
            body_ids=self._wrench_body_ids,
            is_global=True,
        )

    def _as_tensor(self, env_ids: Sequence[int]) -> torch.Tensor:
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)


@configclass
class LiningUnifiedCommandCfg(CommandTermCfg):
    """Configuration for :class:`LiningUnifiedCommand`."""

    class_type: type = LiningUnifiedCommand
    asset_name: str = cast(str, MISSING)
    ee_body_name: str = cast(str, MISSING)
    base_body_name: str = cast(str, MISSING)
    leg_joint_names: list[str] = MISSING

    lin_vel_x_range: tuple[float, float] = (-0.6, 0.6)
    lin_vel_y_range: tuple[float, float] = (-0.4, 0.4)
    ang_vel_z_range: tuple[float, float] = (-0.6, 0.6)
    velocity_resampling_time_range_s: tuple[float, float] = (5.0, 5.0)
    zero_velocity_probability: float = 0.3
    lin_vel_x_deadband: float = 0.1
    lin_vel_y_deadband: float = 0.1
    ang_vel_z_deadband: float = 0.2

    # Kortex is mounted at Spot's body center; arm_joint_2 sits about 0.87 m above
    # the ground when Spot stands with its feet on the ground (body at ~0.46 m).
    sphere_center_offset: tuple[float, float, float] = (0.0, 0.0, 0.87)
    # FK of Lining_CFG's default arm pose in (radius, pitch, yaw) about the center.
    initial_ee_start: tuple[float, float, float] = (0.705, -0.276, -0.037)
    initial_ee_end: tuple[float, float, float] = (0.66, 0.0, 0.0)
    # Kortex Gen3 reaches about 0.90 m from arm_joint_2 to the EE frame.
    ee_radius_range: tuple[float, float] = (0.35, 0.85)
    ee_pitch_range: tuple[float, float] = (-2.0 * math.pi / 5.0, 2.0 * math.pi / 5.0)
    ee_yaw_range: tuple[float, float] = (-3.0 * math.pi / 5.0, 3.0 * math.pi / 5.0)
    ee_delta_roll_range: tuple[float, float] = (-0.5, 0.5)
    ee_delta_pitch_range: tuple[float, float] = (-0.5, 0.5)
    ee_delta_yaw_range: tuple[float, float] = (-0.5, 0.5)
    ee_trajectory_time_range_s: tuple[float, float] = (1.0, 3.0)
    ee_hold_time_range_s: tuple[float, float] = (0.5, 2.0)
    # Extra downward tool pitch relative to the target ray. UniFP uses 0.38 for Z1;
    # zero keeps the Kortex tool axis aligned with the ray from the shoulder.
    arm_induced_pitch: float = 0.0
    # Canonical pointing frame (+x = tool axis) -> EE link. Kortex tool axis is +z.
    ee_frame_offset_euler: tuple[float, float, float] = (0.0, math.pi / 2.0, 0.0)
    # Spot body collision mesh spans x in [-0.44, 0.45], y in [-0.13, 0.13] and the
    # mount top is at ~0.61 m; the box keeps a ~0.1 m margin around it.
    collision_box_lower: tuple[float, float, float] = (-0.55, -0.2, -0.95)
    collision_box_upper: tuple[float, float, float] = (0.55, 0.2, -0.2)
    underground_limit: float = -0.77
    collision_check_samples: int = 10
    collision_check_attempts: int = 10

    ee_force_interval_range_s: tuple[float, float] = (3.5, 9.0)
    ee_force_duration_range_s: tuple[float, float] = (1.0, 3.0)
    ee_force_settling_time_s: float = 1.0
    ee_force_active_probability: float = 0.8
    # UniFP uses +/-60 N for Z1. Kortex wrist joints are limited to 9 Nm and the
    # shoulder keeps only ~15 Nm after gravity compensation at full reach.
    ee_force_range: tuple[float, float] = (-20.0, 20.0)
    ee_stiffness: float = 200.0

    base_force_interval_range_s: tuple[float, float] = (3.5, 9.0)
    base_force_duration_range_s: tuple[float, float] = (1.0, 3.0)
    base_force_settling_time_s: float = 3.0
    base_force_active_probability: float = 0.8
    # UniFP's +/-50 N scaled by the robot mass ratio (Spot + Kortex ~40 kg vs.
    # B2 + Z1 ~80 kg).
    base_force_range: tuple[float, float] = (-25.0, 25.0)
    base_damping: float = 200.0
    apply_base_external_force: bool = False
    force_start_step: int = 8000 * 24

    gait_cycle_time: float = 0.64
    gait_joint_scale: float = 0.17
    gait_double_support_threshold: float = 0.5

    external_force_debug_vis: bool = False
    # Arrow length in meters per newton; marker thickness is fixed below.
    external_force_arrow_scale: float = 0.01
    target_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/lining_unified/target_ee_pose"
    )
    external_force_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/lining_unified/external_force"
    )
    target_pose_visualizer_cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
    external_force_visualizer_cfg.markers["arrow"].scale = (1.0, 0.05, 0.05)
