# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab reproduction of the UMI-on-Legs whole-body RL environment."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.loco_manipulation.umi_on_legs.mdp as mdp
from robot_lab.assets.quadarm import GO2_ARX5_CFG

JOINT_NAMES = [
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
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]

TERMINATION_BODIES = ["base", "Head_.*", ".*_hip", ".*_thigh", "link[1-5]"]
FOOT_BODIES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]


@configclass
class UmiOnLegsSceneCfg(InteractiveSceneCfg):
    """Flat-ground Go2 + ARX5 scene."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        update_period=0.005,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=1000.0),
    )


@configclass
class CommandsCfg:
    ee_trajectory = mdp.EndEffectorTrajectoryCommandCfg(
        asset_name="robot",
        body_name="end_effector",
        trajectory_file="",
        source_dt=0.005,
        preview_times=(-0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 1.0),
        pose_latency_s=0.01,
        position_scale=10.0,
        orientation_scale=1.5,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.075, n_max=0.075),
            scale=0.05,
        )
        ee_trajectory = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_trajectory"},
        )
        previous_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        ee_trajectory = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "ee_trajectory"},
        )
        previous_action = ObsTerm(func=mdp.last_action)
        actuator_stiffness = ObsTerm(
            func=mdp.actuator_stiffness,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            scale=0.1,
        )
        actuator_damping = ObsTerm(
            func=mdp.actuator_damping,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            scale=10.0,
        )
        body_mass = ObsTerm(
            func=mdp.StartupPhysicsProperty,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["base", "link1", "link3", "link5", "link6"],
                    preserve_order=True,
                ),
                "property_name": "mass",
            },
        )
        body_com = ObsTerm(
            func=mdp.StartupPhysicsProperty,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["base", "link1", "link3", "link5", "link6"],
                    preserve_order=True,
                ),
                "property_name": "com",
            },
            scale=10.0,
        )
        shape_friction = ObsTerm(
            func=mdp.StartupPhysicsProperty,
            params={"asset_cfg": SceneEntityCfg("robot"), "property_name": "shape_friction"},
        )
        joint_friction = ObsTerm(
            func=mdp.joint_friction,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
            scale=10.0,
        )
        joint_damping = ObsTerm(
            func=mdp.joint_damping,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 8.0),
            "dynamic_friction_range": (0.1, 8.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    randomize_body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base", "link1", "link3", "link5", "link6"]),
            "mass_distribution_params": (-0.25, 0.25),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base", "link1", "link3", "link5", "link6"]),
            "com_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1)},
        },
    )
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
            "friction_distribution_params": (0.0, 0.05),
            "operation": "abs",
        },
    )
    randomize_joint_damping = EventTerm(
        func=mdp.randomize_joint_damping,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
            "damping_range": (0.01, 0.5),
        },
    )
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (-0.5, 0.5),
            },
            "velocity_range": {},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_range,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
            "position_range_ratio": 0.05,
            "velocity_range": (0.0, 0.0),
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True),
            "stiffness_distribution_params": (0.5, 1.5),
            "damping_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "roll": (-1.0, 1.0),
                "pitch": (-1.0, 1.0),
                "yaw": (-1.0, 1.0),
            }
        },
    )
    perturb_pose = EventTerm(
        func=mdp.perturb_root_pose,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "position_std": (0.1, 0.1, 0.01),
            "euler_std": (0.01, 0.01, 0.5),
        },
    )


@configclass
class RewardsCfg:
    pose_tracking = RewTerm(
        func=mdp.PoseTrackingReward,
        weight=4.0,
        params={
            "command_name": "ee_trajectory",
            "asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "position_curriculum": (
                (100.0, 2.0),
                (1.0, 1.0),
                (0.8, 0.5),
                (0.5, 0.1),
                (0.4, 0.05),
                (0.2, 0.01),
                (0.1, 0.005),
            ),
            "orientation_curriculum": (
                (100.0, 8.0),
                (1.0, 4.0),
                (0.8, 2.0),
                (0.6, 1.0),
                (0.2, 0.5),
            ),
            "initial_position_level": 1,
            "initial_orientation_level": 1,
            "smoothing_multiplier": 0.25,
        },
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
    )
    joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
    )
    joint_torque = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)},
    )
    root_height = RewTerm(func=mdp.root_height_l2, weight=-1.0, params={"target_height": 0.35})
    collision = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=TERMINATION_BODIES),
            "threshold": 1.0,
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    body_ee_alignment = RewTerm(
        func=mdp.joint_deviation_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint5"], preserve_order=True)},
    )
    even_mass_distribution = RewTerm(
        func=mdp.even_mass_distribution,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES),
            "flying_penalty": 100.0,
        },
    )
    feet_under_hips = RewTerm(
        func=mdp.FeetUnderHips,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "body_pairs": (
                ("FL_thigh", "FL_foot"),
                ("FR_thigh", "FR_foot"),
                ("RL_thigh", "RL_foot"),
                ("RR_thigh", "RR_foot"),
            ),
            "distance_sigma": 0.5,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=TERMINATION_BODIES),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """The pose reward owns the error-threshold curriculum from the release."""

    pass


@configclass
class UmiOnLegsEnvCfg(ManagerBasedRLEnvCfg):
    scene: UmiOnLegsSceneCfg = UmiOnLegsSceneCfg(num_envs=4096, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 7.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.scene.robot = GO2_ARX5_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class UmiOnLegsEnvCfg_PLAY(UmiOnLegsEnvCfg):
    """Deterministic, lightweight configuration for policy visualization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.observations.policy.enable_corruption = False

        # Training-only domain randomization belongs to the task configuration,
        # not to the shared play script.
        self.events.randomize_rigid_body_material = None
        self.events.randomize_body_mass = None
        self.events.randomize_body_com = None
        self.events.randomize_joint_friction = None
        self.events.randomize_joint_damping = None
        self.events.randomize_actuator_gains = None
        self.events.push_robot = None
        self.events.perturb_pose = None
