# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
# Modified in 2026 for robot_lab and Isaac Lab 2.2 compatibility.

"""Functions specific to the interlimb loco-manipulation environments."""

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def illegal_ground_contact(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces_with_ground = contact_sensor.data.force_matrix_w[:, sensor_cfg.body_ids, ...]
    contact_forces_with_ground = contact_forces_with_ground.reshape(env.num_envs, -1, 3)
    # check if any contact force with the ground exceeds the threshold
    return torch.any(torch.norm(contact_forces_with_ground, dim=-1) > threshold, dim=1)
