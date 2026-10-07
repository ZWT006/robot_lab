# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""UniFP-style unified position/force whole-body control on Spot + Kortex.

Ported from ``unified_force/config/b2_z1``. Kinematic differences handled here:

- Spot legs share B2's ``hip_x -> hip_y -> knee`` (x, y, y) structure, sign
  conventions, and FL/FR/RL/RR order, so the gait reference indices carry over.
- Kortex Gen3 has 7 revolute joints (all actuated by the policy) instead of Z1's
  5 policy joints plus 2 fixed tool joints, giving a 19-D action.
- ``arm_joint_1`` turns about world -z (its URDF origin flips the frame by pi),
  opposite to ``z1_waist``. No term here depends on per-arm-joint signs.
- The Kortex tool axis is EE +z (Z1: +x); see ``ee_frame_offset_euler`` in the
  command configuration.
- Force and mass ranges are scaled from UniFP by the robot mass ratio
  (~40 kg vs. ~80 kg); the EE force is further limited by Kortex joint torques.
"""

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

import robot_lab.tasks.manager_based.loco_manipulation.lining_unified.mdp as mdp
from robot_lab.assets.quadarm import LINING_ACTION_SCALE, Lining_CFG

from .ui import LiningUnifiedEnvWindow

LEG_JOINT_NAMES = [
    "front_left_hip_x",
    "front_left_hip_y",
    "front_left_knee",
    "front_right_hip_x",
    "front_right_hip_y",
    "front_right_knee",
    "rear_left_hip_x",
    "rear_left_hip_y",
    "rear_left_knee",
    "rear_right_hip_x",
    "rear_right_hip_y",
    "rear_right_knee",
]
HIP_X_JOINT_NAMES = ["front_left_hip_x", "front_right_hip_x", "rear_left_hip_x", "rear_right_hip_x"]
ARM_JOINT_NAMES = [f"arm_joint_{joint_index}" for joint_index in range(1, 8)]
POLICY_JOINT_NAMES = LEG_JOINT_NAMES + ARM_JOINT_NAMES
NUM_LEG_ACTIONS = len(LEG_JOINT_NAMES)
NUM_POLICY_ACTIONS = len(POLICY_JOINT_NAMES)

BASE_BODY_NAME = "body"
END_EFFECTOR_BODY = "arm_end_effector_link"
# arm_end_effector_link is a 1 g frame; payload mass is randomized on the last arm link.
PAYLOAD_BODY_NAME = "arm_bracelet_link"
# Spot's foot collision spheres live on the *_ee links.
FOOT_BODY_NAMES = ["front_left_ee", "front_right_ee", "rear_left_ee", "rear_right_ee"]
THIGH_BODY_NAMES = ["front_left_upper_leg", "front_right_upper_leg", "rear_left_upper_leg", "rear_right_upper_leg"]
# Spot's lower-leg mesh ends ~4 cm above the foot-sphere bottom and grazes rough
# terrain, so (unlike UniFP's calf) it is not penalized. Matches discard_lining_legs.
CONTACT_PENALTY_BODIES = [
    BASE_BODY_NAME,
    ".*_hip",
    ".*_upper_leg",
    "spot_kortex_mount",
    "arm_base_link",
    "arm_shoulder_link",
    "arm_half_arm_[12]_link",
    "arm_forearm_link",
    "arm_spherical_wrist_[12]_link",
    "arm_bracelet_link",
]


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
class LiningUnifiedSceneCfg(InteractiveSceneCfg):
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
    # Spherical workspace, EE frame offset, and force ranges default to Spot + Kortex
    # values in LiningUnifiedCommandCfg.
    lining_unified = mdp.LiningUnifiedCommandCfg(
        asset_name="robot",
        base_body_name=BASE_BODY_NAME,
        ee_body_name=END_EFFECTOR_BODY,
        leg_joint_names=LEG_JOINT_NAMES,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.LiningUnifiedJointPositionActionCfg(
        asset_name="robot",
        joint_names=POLICY_JOINT_NAMES,
        scale=LINING_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
        motor_strength_range=(0.85, 1.15),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        frame = ObsTerm(
            func=mdp.LiningUnifiedPolicyObservation,
            params={
                "command_name": "lining_unified",
                "action_name": "joint_pos",
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
            func=mdp.LiningUnifiedCriticObservation,
            params={
                "command_name": "lining_unified",
                "action_name": "joint_pos",
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "joint_names": POLICY_JOINT_NAMES,
                "leg_joint_names": LEG_JOINT_NAMES,
                "foot_body_names": FOOT_BODY_NAMES,
                "base_body_name": BASE_BODY_NAME,
                "payload_body_name": PAYLOAD_BODY_NAME,
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
            func=mdp.LiningUnifiedEstimatorTarget,
            params={"command_name": "lining_unified", "asset_cfg": SceneEntityCfg("robot")},
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
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_BODY_NAME]),
            # UniFP adds 0-15 kg to B2; scaled by the ~0.5 robot mass ratio.
            "mass_distribution_params": (0.0, 7.5),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_payload_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[PAYLOAD_BODY_NAME]),
            "mass_distribution_params": (0.0, 0.2),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_BODY_NAME]),
            # Spot's body is smaller than B2's; matches discard_lining_legs.
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
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
        func=mdp.reset_lining_unified_joints,
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
            "command_name": "lining_unified",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
        },
    )
    tracking_lin_vel_force_world = RewTerm(
        func=mdp.track_base_velocity_force_exp,
        weight=2.0,
        params={"command_name": "lining_unified", "asset_cfg": SceneEntityCfg("robot"), "tracking_sigma": 0.25},
    )
    tracking_ang_vel = RewTerm(
        func=mdp.track_yaw_velocity_exp,
        weight=1.0,
        params={"command_name": "lining_unified", "asset_cfg": SceneEntityCfg("robot"), "tracking_sigma": 0.25},
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
            "command_name": "lining_unified",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
        },
    )
    reference_leg_pose = RewTerm(
        func=mdp.reference_leg_pose,
        weight=1.0,
        params={
            "command_name": "lining_unified",
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES, preserve_order=True),
        },
    )
    alive = RewTerm(func=mdp.alive, weight=1.5)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.5, params={"asset_cfg": SceneEntityCfg("robot")})
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "lining_unified",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            "threshold": 0.5,
        },
    )
    front_feet_height = RewTerm(
        func=mdp.front_feet_height,
        weight=1.0,
        params={
            "command_name": "lining_unified",
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
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=CONTACT_PENALTY_BODIES),
            "threshold": 0.1,
        },
    )
    leg_action_rate = RewTerm(func=mdp.action_rate_subset, weight=-0.02, params={"action_slice": (0, NUM_LEG_ACTIONS)})
    arm_action_rate = RewTerm(
        func=mdp.action_rate_subset, weight=-0.045, params={"action_slice": (NUM_LEG_ACTIONS, NUM_POLICY_ACTIONS)}
    )
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
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=HIP_X_JOINT_NAMES, preserve_order=True)},
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
            # UniFP's 200 N is about B2's static per-foot load; Spot's is ~100 N.
            "threshold": 100.0,
        },
    )
    base_height = RewTerm(
        func=mdp.root_height_l2,
        weight=-2.0,
        params={"target_height": 0.48, "asset_cfg": SceneEntityCfg("robot")},
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
            "command_name": "lining_unified",
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
        },
    )
    tracking_ee_pose_force_world = RewTerm(
        func=mdp.track_ee_pose_force_exp,
        weight=2.0,
        params={
            "command_name": "lining_unified",
            "asset_cfg": SceneEntityCfg("robot"),
            "tracking_sigma": 1.0,
            "orientation_weight": 0.5,
            "orientation_sigma": 0.5,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class LiningUnifiedEnvCfg(ManagerBasedRLEnvCfg):
    scene: LiningUnifiedSceneCfg = LiningUnifiedSceneCfg(num_envs=4096, env_spacing=3.0)
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
        self.scene.robot = Lining_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class LiningUnifiedEnvCfg_PLAY(LiningUnifiedEnvCfg):
    """Small deterministic configuration for visualization and smoke tests."""

    def __post_init__(self):
        super().__post_init__()
        self.ui_window_class_type = LiningUnifiedEnvWindow
        self.scene.num_envs = 1
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 1
            self.scene.terrain.terrain_generator.num_cols = 1
        self.observations.policy.frame.params["add_noise"] = False
        self.actions.joint_pos.motor_strength_range = (1.0, 1.0)
        self.commands.lining_unified.force_start_step = 0
        # Play exposes only physical EE disturbances; virtual force commands would
        # otherwise shift the compliant target independently of the three UI flags.
        self.commands.lining_unified.apply_ee_force_command = False
        self.commands.lining_unified.apply_base_external_force = False
        self.events.randomize_material = None
        self.events.randomize_base_mass = None
        self.events.randomize_payload_mass = None
        self.events.randomize_base_com = None
        self.events.push_robot = None
