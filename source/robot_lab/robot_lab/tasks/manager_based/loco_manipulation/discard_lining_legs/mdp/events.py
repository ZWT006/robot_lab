# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Domain-randomization events for discard lining training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: str = "add",
    recompute_inertia: bool = True,
):
    """Randomize body mass while retaining a 10 g lower bound."""
    if operation != "add":
        raise ValueError("Discard-lining mass randomization supports only operation='add'")
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_cpu = torch.arange(env.num_envs, device="cpu")
    else:
        env_ids_cpu = env_ids.cpu()
    if asset_cfg.body_ids is None or asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, device="cpu")

    masses = asset.root_physx_view.get_masses()
    default_masses = asset.data.default_mass[env_ids_cpu[:, None], body_ids].cpu()
    offsets = torch.empty_like(default_masses).uniform_(*mass_distribution_params)
    randomized_masses = (default_masses + offsets).clamp_min(0.01)
    masses[env_ids_cpu[:, None], body_ids] = randomized_masses
    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    if recompute_inertia:
        inertias = asset.root_physx_view.get_inertias()
        ratios = randomized_masses / default_masses.clamp_min(1.0e-8)
        default_inertias = asset.data.default_inertia[env_ids_cpu[:, None], body_ids].cpu()
        inertias[env_ids_cpu[:, None], body_ids] = default_inertias * ratios[..., None]
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)


def randomize_joint_damping(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    damping_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Set physical joint damping uniformly in the configured range."""
    asset: Articulation = env.scene[asset_cfg.name]
    resolved_env_ids = torch.arange(env.num_envs, device=asset.device) if env_ids is None else env_ids
    joint_ids = slice(None) if asset_cfg.joint_ids is None else asset_cfg.joint_ids
    num_joints = asset.num_joints if isinstance(joint_ids, slice) else len(cast(Sequence[int], joint_ids))
    damping = torch.empty(len(resolved_env_ids), num_joints, device=asset.device).uniform_(*damping_range)
    asset.write_joint_damping_to_sim(damping, joint_ids=joint_ids, env_ids=resolved_env_ids)


def reset_joints_by_range(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range_ratio: float,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset joints within a ratio of their full range around the default pose."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = slice(None) if asset_cfg.joint_ids is None else asset_cfg.joint_ids
    indexed_env_ids = env_ids if isinstance(joint_ids, slice) else env_ids[:, None]
    default_pos = asset.data.default_joint_pos[indexed_env_ids, joint_ids]
    limits = asset.data.default_joint_pos_limits[indexed_env_ids, joint_ids]
    half_width = position_range_ratio * (limits[..., 1] - limits[..., 0])
    joint_pos = default_pos + (2.0 * torch.rand_like(default_pos) - 1.0) * half_width
    joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
    joint_vel = torch.empty_like(joint_pos).uniform_(*velocity_range)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=joint_ids, env_ids=env_ids)


def perturb_root_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position_std: tuple[float, float, float],
    euler_std: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply occasional root-pose jumps while keeping the world-frame target fixed."""
    asset: Articulation = env.scene[asset_cfg.name]
    resolved_env_ids = torch.arange(env.num_envs, device=asset.device) if env_ids is None else env_ids
    root_pose = asset.data.root_pose_w[resolved_env_ids].clone()
    pos_std = torch.tensor(position_std, device=asset.device)
    angle_std = torch.tensor(euler_std, device=asset.device)
    root_pose[:, :3] += torch.randn(len(resolved_env_ids), 3, device=asset.device) * pos_std
    euler = torch.randn(len(resolved_env_ids), 3, device=asset.device) * angle_std
    delta_quat = math_utils.quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
    root_pose[:, 3:7] = math_utils.quat_mul(root_pose[:, 3:7], delta_quat)
    asset.write_root_pose_to_sim(root_pose, env_ids=resolved_env_ids)
