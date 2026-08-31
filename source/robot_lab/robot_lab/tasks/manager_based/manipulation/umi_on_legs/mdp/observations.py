# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Privileged observations for the UMI-on-Legs asymmetric critic."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _selected_joint_values(values: torch.Tensor, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    joint_ids = slice(None) if asset_cfg.joint_ids is None else asset_cfg.joint_ids
    return values[:, joint_ids]


def _actuator_gains(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    gain_name: Literal["stiffness", "damping"],
) -> torch.Tensor:
    """Gather explicit-actuator gains in articulation joint order."""
    asset: Articulation = env.scene[asset_cfg.name]
    gains = torch.zeros(env.num_envs, asset.num_joints, device=asset.device)
    for actuator in asset.actuators.values():
        gains[:, actuator.joint_indices] = getattr(actuator, gain_name)
    return _selected_joint_values(gains, asset_cfg)


def actuator_stiffness(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the randomized controller stiffness gains."""
    return _actuator_gains(env, asset_cfg, "stiffness")


def actuator_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the randomized controller damping gains."""
    return _actuator_gains(env, asset_cfg, "damping")


def joint_friction(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return randomized physical joint friction coefficients."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _selected_joint_values(asset.data.joint_friction_coeff, asset_cfg)


def joint_damping(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return randomized physical joint damping coefficients."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _selected_joint_values(asset.data.joint_damping, asset_cfg)


class StartupPhysicsProperty(ManagerTermBase):
    """Cache CPU-only startup physics properties on the simulation device.

    Isaac Lab builds the observation manager before applying startup events, then
    resets stateful terms.  Clearing this cache in :meth:`reset` ensures the first
    policy observation captures the randomized values without a CPU transfer on
    every control step.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._value: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        asset_cfg: SceneEntityCfg,
        property_name: Literal["mass", "com", "shape_friction"],
    ) -> torch.Tensor:
        if self._value is None:
            if property_name == "mass":
                value = self.asset.root_physx_view.get_masses()[:, asset_cfg.body_ids]
            elif property_name == "com":
                value = self.asset.root_physx_view.get_coms()[:, asset_cfg.body_ids, :3].flatten(1)
            elif property_name == "shape_friction":
                value = self.asset.root_physx_view.get_material_properties()[..., 0]
            else:
                raise ValueError(f"Unknown startup physics property: {property_name}")
            self._value = value.to(self.device)
        return self._value

    def reset(self, env_ids: torch.Tensor | None = None):
        self._value = None
