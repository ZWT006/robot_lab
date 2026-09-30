# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Manager-based reproduction of UniFP's B2 + Z1 position/force task."""

from __future__ import annotations

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.loco_manipulation.unified_force.mdp as mdp
from robot_lab.assets.quadarm import B2_Z1_CFG

LEG_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]
ARM_JOINT_NAMES = ["z1_waist", "z1_shoulder", "z1_elbow", "z1_wrist_angle", "z1_forearm_roll"]
POLICY_JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES
FIXED_TOOL_JOINT_NAMES = ["z1_wrist_rotate", "z1_jointGripper"]
FOOT_BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
THIGH_BODY_NAMES = ["FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh"]


ROUGH_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=None,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "rough_flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.0, 0.05),
            noise_step=0.005,
            border_width=0.15,
        )
    },
)


@configclass
class UnifiedForceSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_FLAT_TERRAIN_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        update_period=0.005,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=1000.0),
    )


@configclass
class CommandsCfg:
    unified_force = mdp.UnifiedForceCommandCfg(
        asset_name="robot",
        base_body_name="base_link",
        ee_body_name="ee_gripper_link",
        leg_joint_names=LEG_JOINT_NAMES,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.UnifiedForceJointPositionActionCfg(
        asset_name="robot",
        joint_names=POLICY_JOINT_NAMES,
        fixed_joint_names=FIXED_TOOL_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
        motor_strength_range=(0.85, 1.15),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        frame = ObsTerm(
            func=mdp.UnifiedForcePolicyObservation,
            params={
                "command_name": "unified_force",
                "asset_cfg": SceneEntityCfg("robot"),
                "joint_names": POLICY_JOINT_NAMES,
                "add_noise": True,
            },
            history_length=32,
            flatten_history_dim=True,
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        frame = ObsTerm(
            func=mdp.UnifiedForceCriticObservation,
            params={
                "command_name": "unified_force",
                "action_name": "joint_pos",
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "joint_names": POLICY_JOINT_NAMES,
                "leg_joint_names": LEG_JOINT_NAMES,
                "foot_body_names": FOOT_BODY_NAMES,
                "base_body_name": "base_link",
                "ee_body_name": "ee_gripper_link",
            },
            history_length=3,
            flatten_history_dim=True,
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class StateEstimationTargetCfg(ObsGroup):
        target = ObsTerm(
            func=mdp.UnifiedForceEstimatorTarget,
            params={"command_name": "unified_force", "asset_cfg": SceneEntityCfg("robot")},
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    state_estimation_target: StateEstimationTargetCfg = StateEstimationTargetCfg()


@configclass
class EventCfg:
    randomize_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 2.0),
            "dynamic_friction_range": (0.3, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "mass_distribution_params": (0.0, 15.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_gripper_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["ee_gripper_link"]),
            "mass_distribution_params": (0.0, 0.2),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "com_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "z": (-0.15, 0.15)},
        },
    )
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0), "yaw": (-math.pi / 2, math.pi / 2)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_unified_force_joints,
        mode="reset",
        params={
            "leg_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
            "arm_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 8.0),
        params={"velocity_range": {"x": (-0.8, 0.8), "y": (-0.8, 0.8)}},
    )


@configclass
class RewardsCfg:
    feet_contact_number = RewTerm(
        func=mdp.feet_contact_number,
        weight=2.0,
        params={
            "command_name": "unified_force",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
        },
    )
    tracking_lin_vel_force_world = RewTerm(
        func=mdp.track_base_velocity_force_exp,
        weight=2.0,
        params={"command_name": "unified_force", "asset_cfg": SceneEntityCfg("robot"), "tracking_sigma": 0.25},
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_yaw_velocity_exp,
        weight=1.0,
        params={"command_name": "unified_force", "asset_cfg": SceneEntityCfg("robot"), "tracking_sigma": 0.25},
    )
    leg_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-5.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=0.5,
        params={
            "command_name": "unified_force",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
        },
    )
    reference_leg_pose = RewTerm(
        func=mdp.reference_leg_pose,
        weight=1.0,
        params={
            "command_name": "unified_force",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
        },
    )
    alive = RewTerm(func=mdp.alive, weight=1.5)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5, params={"asset_cfg": SceneEntityCfg("robot")})
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "unified_force",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "threshold": 0.5,
        },
    )
    front_feet_height = RewTerm(
        func=mdp.front_feet_height,
        weight=1.0,
        params={
            "command_name": "unified_force",
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES[:2], preserve_order=True),
        },
    )
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02, params={"asset_cfg": SceneEntityCfg("robot")})
    leg_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    leg_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-8.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True)},
    )
    arm_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-4.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True)},
    )
    arm_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES, preserve_order=True)},
    )
    collision = RewTerm(
        func=mdp.current_contact_count,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_thigh", ".*_calf", "base_link"]),
            "threshold": 0.1,
        },
    )
    leg_action_rate = RewTerm(func=mdp.action_rate_subset, weight=-0.02, params={"action_slice": (0, 12)})
    arm_action_rate = RewTerm(func=mdp.action_rate_subset, weight=-0.045, params={"action_slice": (12, 17)})
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True)},
    )
    torque_limits = RewTerm(
        func=mdp.applied_torque_limits_soft,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=POLICY_JOINT_NAMES, preserve_order=True),
            "soft_ratio": 0.9,
        },
    )
    hip_position = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                preserve_order=True,
            )
        },
    )
    feet_drag = RewTerm(
        func=mdp.feet_drag,
        weight=-8.0e-4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.current_contact_force_excess,
        weight=-0.001,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "threshold": 200.0,
        },
    )
    base_height = RewTerm(
        func=mdp.root_height_l2,
        weight=-2.0,
        params={"target_height": 0.5, "asset_cfg": SceneEntityCfg("robot")},
    )
    feet_position_xy = RewTerm(
        func=mdp.feet_planar_distance,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "thigh_body_names": THIGH_BODY_NAMES,
        },
    )
    feet_height_high = RewTerm(
        func=mdp.feet_height_high,
        weight=-15.0,
        params={
            "command_name": "unified_force",
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
        },
    )
    tracking_ee_force_world = RewTerm(
        func=mdp.track_ee_force_exp,
        weight=2.0,
        params={
            "command_name": "unified_force",
            "asset_cfg": SceneEntityCfg("robot"),
            "body_name": "ee_gripper_link",
            "tracking_sigma": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class UnifiedForceEnvCfg(ManagerBasedRLEnvCfg):
    scene: UnifiedForceSceneCfg = UnifiedForceSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.scene.robot = B2_Z1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class UnifiedForceEnvCfg_PLAY(UnifiedForceEnvCfg):
    """Small deterministic configuration for visualization and smoke tests."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
        self.observations.policy.frame.params["add_noise"] = False
        self.actions.joint_pos.motor_strength_range = (1.0, 1.0)
        self.commands.unified_force.force_start_step = 0
        self.events.randomize_material = None
        self.events.randomize_base_mass = None
        self.events.randomize_gripper_mass = None
        self.events.randomize_base_com = None
        self.events.push_robot = None
