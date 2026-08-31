# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""MDP terms for the UMI-on-Legs whole-body controller."""

from isaaclab.envs.mdp import (
    JointPositionActionCfg,
    action_rate_l2,
    base_ang_vel,
    base_lin_vel,
    generated_commands,
    illegal_contact,
    joint_acc_l2,
    joint_pos_limits,
    joint_pos_rel,
    joint_torques_l2,
    joint_vel_rel,
    last_action,
    projected_gravity,
    push_by_setting_velocity,
    randomize_actuator_gains,
    randomize_joint_parameters,
    randomize_rigid_body_com,
    randomize_rigid_body_material,
    reset_root_state_uniform,
    time_out,
    undesired_contacts,
)

from .commands import EndEffectorTrajectoryCommand, EndEffectorTrajectoryCommandCfg
from .events import perturb_root_pose, randomize_joint_damping, randomize_rigid_body_mass, reset_joints_by_range
from .observations import (
    StartupPhysicsProperty,
    actuator_damping,
    actuator_stiffness,
    joint_damping,
    joint_friction,
)
from .rewards import (
    FeetUnderHips,
    PoseTrackingReward,
    even_mass_distribution,
    joint_deviation_l2,
    root_height_l2,
)
