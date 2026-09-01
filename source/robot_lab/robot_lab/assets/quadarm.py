# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Asset configurations for quadruped robots with mounted manipulators.

The calibrated Go2 + ARX5 URDF and meshes are derived from the MIT-licensed
``real-stanford/umi-on-legs`` release. See the NOTICE file beside the asset.

The ReLIC Spot + Arm configuration is kept in a separately licensed module and
re-exported here so downstream tasks have one stable quadruped-manipulator asset
interface.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR

GO2_ARX5_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=(f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/go2_arx5/urdf/go2_arx5_finray_x85_z94.urdf"),
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.5, 0.0, 0.3),
        joint_pos={
            "FR_hip_joint": 0.1,
            "FL_hip_joint": -0.1,
            "RR_hip_joint": 0.1,
            "RL_hip_joint": -0.1,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 1.0,
            ".*_calf_joint": -1.5,
            "joint1": 0.0,
            "joint2": 0.3,
            "joint3": 0.5,
            "joint[4-6]": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # Four physics steps at 200 Hz reproduce the 20 ms motor/command latency
        # used by the released controller.
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_(hip|thigh|calf)_joint"],
            min_delay=4,
            max_delay=4,
            effort_limit={
                ".*_hip_joint": 35.278,
                ".*_thigh_joint": 35.278,
                ".*_calf_joint": 44.4,
            },
            velocity_limit=30.1,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        ),
        "arm": DelayedPDActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            min_delay=4,
            max_delay=4,
            effort_limit={
                "joint[12]": 20.0,
                "joint3": 15.0,
                "joint4": 7.0,
                "joint[56]": 5.0,
            },
            stiffness={"joint[1-3]": 100.0, "joint[45]": 20.0, "joint6": 5.0},
            damping={"joint[1-3]": 3.0, "joint4": 2.0, "joint5": 1.0, "joint6": 0.5},
            friction=0.0,
        ),
    },
)

# Imported last to keep the RAI Research License implementation isolated in its
# own module while exposing a single asset catalog to task configurations.
from robot_lab.assets.relic_spot import SPOT_ARM_CFG  # noqa: E402

__all__ = ["GO2_ARX5_CFG", "SPOT_ARM_CFG"]
"""Calibrated 18-DoF Go2 + ARX5 used by UMI-on-Legs."""
