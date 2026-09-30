# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Reward terms matching the released UniFP B2 + Z1 task."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .commands import UnifiedForceCommand


def _robot(env, cfg: SceneEntityCfg) -> Articulation:
    return env.scene[cfg.name]


def _command(env, name: str) -> UnifiedForceCommand:
    return env.command_manager.get_term(name)


def feet_contact_number(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2] > 5.0
    stance = _command(env, command_name).stance_mask
    return torch.where(contact == stance, 1.0, -0.3).mean(dim=-1)


def track_base_velocity_force_exp(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    tracking_sigma: float,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    target = _command(env, command_name).base_velocity_target_b[:, :2]
    error = torch.sum(torch.square(target - robot.data.root_link_lin_vel_b[:, :2]), dim=-1)
    return torch.exp(-error / tracking_sigma)


def track_yaw_velocity_exp(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    tracking_sigma: float,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    target = _command(env, command_name).velocity_command_b[:, 2]
    error = torch.square(target - robot.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / tracking_sigma)


def track_ee_force_exp(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    body_name: str,
    tracking_sigma: float,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    body_id = robot.find_bodies(body_name)[0][0]
    target = _command(env, command_name).ee_compliant_target_pos_w
    error = torch.sum(torch.abs(robot.data.body_link_pos_w[:, body_id] - target), dim=-1)
    return torch.exp(-2.0 * error / tracking_sigma)


def stand_still(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    error = torch.sum(
        torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=-1,
    )
    reward = torch.exp(-0.05 * error)
    return reward * (~_command(env, command_name).walking_mask)


def reference_leg_pose(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    error = torch.sum(
        torch.abs(robot.data.joint_pos[:, asset_cfg.joint_ids] - _command(env, command_name).reference_leg_joint_pos),
        dim=-1,
    )
    return torch.exp(-0.1 * error)


def alive(env) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def feet_air_time(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=-1)
    return reward * _command(env, command_name).walking_mask


def front_feet_height(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    height = torch.max(robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2], dim=-1).values
    reward = torch.clamp(height - 0.10, max=0.0)
    return reward * _command(env, command_name).walking_mask


def feet_height_high(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    height = torch.max(robot.data.body_link_pos_w[:, asset_cfg.body_ids, 2], dim=-1).values
    reward = torch.clamp(height - 0.20, min=0.0)
    return reward * _command(env, command_name).walking_mask


def action_rate_subset(env, action_slice: tuple[int, int]) -> torch.Tensor:
    start, stop = action_slice
    delta = env.action_manager.action[:, start:stop] - env.action_manager.prev_action[:, start:stop]
    return torch.sum(torch.square(delta), dim=-1)


def applied_torque_limits_soft(
    env,
    asset_cfg: SceneEntityCfg,
    soft_ratio: float,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    torques = torch.abs(robot.data.applied_torque[:, asset_cfg.joint_ids])
    limits = robot.data.joint_effort_limits[:, asset_cfg.joint_ids]
    return torch.sum((torques - limits * soft_ratio).clip(min=0.0), dim=-1)


def joint_deviation_l2(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    error = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(error), dim=-1)


def current_contact_count(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > threshold
    return contact.float().sum(dim=-1)


def current_contact_force_excess(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    magnitude = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return torch.sum((magnitude - threshold).clip(min=0.0), dim=-1)


def root_height_l2(env, target_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return torch.square(robot.data.root_link_pos_w[:, 2] - target_height)


def feet_planar_distance(env, asset_cfg: SceneEntityCfg, thigh_body_names: list[str]) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    thigh_ids, _ = robot.find_bodies(thigh_body_names, preserve_order=True)
    difference = robot.data.body_link_pos_w[:, asset_cfg.body_ids, :2] - robot.data.body_link_pos_w[:, thigh_ids, :2]
    return torch.linalg.norm(difference, dim=-1).mean(dim=-1)


def feet_drag(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot = _robot(env, asset_cfg)
    velocity = torch.abs(robot.data.body_link_lin_vel_w[:, asset_cfg.body_ids]).sum(dim=-1)
    force = torch.linalg.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1)
    return torch.sum(force * velocity, dim=-1)
