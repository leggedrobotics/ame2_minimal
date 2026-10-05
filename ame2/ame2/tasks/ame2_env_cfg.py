# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp
from ame2.assets.unitree import G1_CFG  # isort: skip
from ame2.sensors.mid360_lidar import Mid360RayCasterCfg  # isort: skip
from ame2.sensors.livox_neural_map_sensor import LivoxNeuralMapSensorCfg  # isort: skip

##
# Pre-defined configs
##
from .terrains import AME2_TERRAINS_CFG_G1  # isort: skip


##
# Scene definition
##


@configclass
class Ame2SceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a G1 legged robot."""


    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=AME2_TERRAINS_CFG_G1,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # robots
    robot: ArticulationCfg = MISSING

    # sensors
    # Grid covers base-relative x in [-0.5, 1.5], y in [-0.56, 0.56] at 0.08 m
    # (offset 0.5 + size 2.0/1.12 -> 26 x 15 cells).
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=[2.0, 1.12], ordering="yx"),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
        drift_range=(-0.04, 0.04),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=4, track_air_time=True
    )

    # Critic height scan: matches the actor grid coverage x in [-0.5, 1.5],
    # y in [-0.56, 0.56] at 0.08 m (26 x 15 cells).
    height_scanner_critic = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.5, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.08, size=[2.0, 1.12], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # sensor for self-collision
    self_col_knee = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_knee_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/right_knee_link",
        ],
        history_length=4,
    )

    self_col_ankle = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
        ],
        history_length=4,
    )

    self_col_torso = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/Robot/left_shoulder_roll_link",
            "{ENV_REGEX_NS}/Robot/left_shoulder_yaw_link",
            "{ENV_REGEX_NS}/Robot/left_elbow_link",
            "{ENV_REGEX_NS}/Robot/left_wrist_pitch_link",
            "{ENV_REGEX_NS}/Robot/right_shoulder_roll_link",
            "{ENV_REGEX_NS}/Robot/right_shoulder_yaw_link",
            "{ENV_REGEX_NS}/Robot/right_elbow_link",
            "{ENV_REGEX_NS}/Robot/right_wrist_pitch_link",
        ],
        history_length=4,
    )

    body_accel = mdp.BodyAccelerationSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_pose = mdp.SafeTerrainBasedPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(16.01, 16.01),
        simple_heading=False,
        debug_vis=True,
        cone_angle_deg=45.0,
        ranges=mdp.SafeTerrainBasedPose2dCommandCfg.Ranges(
            heading=(-math.pi/3, math.pi/3),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP"""

    joint_pos = mdp.Ame2DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        delay_range=(0, 1),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class NaiveObsCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        # 1. Base state
        base_lin_vel = ObsTerm(func=mdp.ame2_base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.ame2_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.ame2_projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 2. Joint state
        joint_pos = ObsTerm(func=mdp.ame2_joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.ame2_joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # 3. Previous actions
        actions = ObsTerm(func=mdp.ame2_last_action)
        # 4. Commands (pose + time left)
        pose_commands = ObsTerm(
            func=mdp.ame2_commands_obs, params={"command_name": "base_pose"}
        )
        time_left = ObsTerm(func=mdp.ame2_time_left)
        # 5. Contact states (all links with mass > 0.1)
        contact_states = ObsTerm(
            func=mdp.ame2_contact_states,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces"),
                "asset_cfg": SceneEntityCfg("robot"),
                "mass_threshold": 0.1,
                "force_threshold": 1.0,
                "ignore_bodies": ["imu_in_pelvis", "d435_link", "imu_in_torso", "mid360_link"],
            },
        )
        # 6. Height scan
        height_scan = ObsTerm(
            func=mdp.ame2_height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner_critic"), "offset": 0.74},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.5, 1.5),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class TeacherMappingObsCfg(ObsGroup):
        """Observations for teacher mapping group."""
        
        teacher_mapping = ObsTerm(
            func=mdp.ame2_teacher_mapping,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "custom_noise": [0.001, 0.001, 0.05], "offset": 0.74},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherPropCmdObsCfg(ObsGroup):
        """Observations for teacher prop group."""

        # observation terms (order preserved)
        # 1. Base state
        base_lin_vel = ObsTerm(func=mdp.ame2_base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.ame2_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.ame2_projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # 2. Joint state
        joint_pos = ObsTerm(func=mdp.ame2_joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.ame2_joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # 3. Previous actions
        actions = ObsTerm(func=mdp.ame2_last_action)
        # 4. Commands (pose + time left)
        pose_commands = ObsTerm(
            func=mdp.ame2_commands_obs, params={"command_name": "base_pose", "dist_thr": 2.0}
        )
        # no time_left term: the teacher actor does not observe remaining episode time

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StudentPropHistObsCfg(ObsGroup):
        """Observations for student proprioception group (20-step history).

        No base_lin_vel (not measurable on the real robot), everything else
        with the teacher's noise levels.
        """

        base_ang_vel = ObsTerm(
            func=mdp.ame2_base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=20, flatten_history_dim=False
        )
        projected_gravity = ObsTerm(
            func=mdp.ame2_projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=20, flatten_history_dim=False
        )
        joint_pos = ObsTerm(
            func=mdp.ame2_joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=20, flatten_history_dim=False
        )
        joint_vel = ObsTerm(
            func=mdp.ame2_joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=20, flatten_history_dim=False
        )
        actions = ObsTerm(
            func=mdp.ame2_last_action,
            history_length=20, flatten_history_dim=False
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class StudentCmdsObsCfg(ObsGroup):
        """Observations for student commands (no history)."""

        pose_commands = ObsTerm(
            func=mdp.ame2_commands_obs, params={"command_name": "base_pose", "dist_thr": 2.0}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    teacher_mapping = TeacherMappingObsCfg()
    teacher_prop = TeacherPropCmdObsCfg()

    critic = NaiveObsCfg()
    critic.enable_corruption = False


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.ame2_randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.ame2_randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0), # kg; skewed positive to cover battery and backpack payload
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.ame2_randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "z": (-0.04, 0.04)},
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.ame2_randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "armature_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # Collider offsets keep their defaults (mdp.ame2_optional_randomize_rigid_body_collider_offsets is available).


    # reset
    reset_base = EventTerm(
        func=mdp.ame2_reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-3.14/4, 3.14/4)},
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

    reset_robot_joints = EventTerm(
        func=mdp.ame2_reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    assistive_force = EventTerm(
        func=mdp.ame2_apply_assistive_force,
        mode="reset", # updates every episode
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    ame2_position_tracking = RewTerm(
        func=mdp.ame2_position_tracking, weight=100.0, params={"command_name": "base_pose", "T": 4.0, "std": 2.0}
    )
    ame2_heading_tracking = RewTerm(
        func=mdp.ame2_heading_tracking, weight=50.0, params={"command_name": "base_pose", "T": 2.0, "dist_thr": 0.5}
    )
    ame2_move2goal = RewTerm(
        func=mdp.ame2_move2goal,
        weight=5.0,
        params={
            "command_name": "base_pose",
            "cos_thr": 0.5,
            "dist_thr": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )
    ame2_standatgoal = RewTerm(
        func=mdp.ame2_standatgoal,
        weight=5.0,
        params={
            "command_name": "base_pose",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
        },
    )

    # ame2 penalties
    early_termination_penalty = RewTerm(
        func=mdp.ame2_is_terminated_term,
        params={"term_keys": "early_.*"},
        weight=-500.0,
    )
    # undesired events separated
    undesired_spinning = RewTerm(
        func=mdp.ame2_undesired_spinning,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="head_link"), "yaw_rate_thr": 3.0},
    )
    undesired_leaping = RewTerm(
        func=mdp.ame2_undesired_leaping,
        weight=-1.0,
        params={
            "feet_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "height_scanner_cfg": SceneEntityCfg("height_scanner_critic"),
            "elevation_thr": 0.3,
            "sensor_threshold": 1.0,
            "start_time": 0.2,
        },
    )
    undesired_non_foot_contacts = RewTerm(
        func=mdp.ame2_undesired_non_foot_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*ankle|d435_link|mid360_link).*_link"),
            "sensor_threshold": 1.0,
        },
    )
    undesired_stumbling = RewTerm(
        func=mdp.ame2_undesired_stumbling,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!d435_link|mid360_link).*_link")},
    )
    undesired_slippage = RewTerm(
        func=mdp.ame2_undesired_slippage,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!d435_link|mid360_link).*_link"),
            "asset_cfg": SceneEntityCfg("robot"),
            "slip_vel_thr": 0.3,
            "sensor_threshold": 1.0,
        },
    )
    undesired_self_col_knee = RewTerm(
        func=mdp.ame2_undesired_self_collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("self_col_knee"),
            "sensor_threshold": 1.0,
        },
    )
    undesired_self_col_torso = RewTerm(
        func=mdp.ame2_undesired_self_collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("self_col_torso"),
            "sensor_threshold": 1.0,
        },
    )
    undesired_self_col_ankle = RewTerm(
        func=mdp.ame2_undesired_self_collision,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("self_col_ankle"),
            "sensor_threshold": 1.0,
        },
    )

    # ame2 regularization
    base_roll_rate = RewTerm(
        func=mdp.ame2_base_roll_rate_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link")},
    )
    joint_regularization = RewTerm(
        func=mdp.ame2_joint_regularization_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    action_smoothness = RewTerm(
        func=mdp.ame2_action_rate_l2,
        weight=-0.01,
    )
    link_contact_forces = RewTerm(
        func=mdp.ame2_link_contact_forces_l2,
        weight=-1.0e-5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!d435_link|mid360_link).*_link"),
            "robot_weight": 400.0,
        },
    )
    body_lin_acc_l1 = RewTerm(
        func=mdp.ame2_body_lin_acc_l1,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="^(?!d435_link|mid360_link).*_link"), "sensor_cfg": SceneEntityCfg("body_accel")},
    )
    # simulation fidelity
    joint_pos_limits = RewTerm(
        func=mdp.ame2_joint_pos_limits,
        weight=-1000.0,
    )
    joint_vel_limits = RewTerm(
        func=mdp.ame2_joint_vel_limits,
        weight=-1.0,
    )
    joint_torque_limits = RewTerm(
        func=mdp.ame2_joint_torque_limits,
        weight=-1.0,
    )

    # ame2 optional upper body shaping
    ame2_optional_uppershaping = RewTerm(
        func=mdp.ame2_optional_uppershaping,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_(pitch|roll|yaw)_joint", ".*_elbow_joint"]),
            "height_scanner_cfg": SceneEntityCfg("height_scanner_critic"),
            "elevation_thr": 0.3,
        },
    )

    ame2_optional_foot_shaping = RewTerm(
        func=mdp.ame2_optional_foot_yaw,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll_link"),
            "torso_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "lin_vel_thr": 0.3,
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.ame2_time_out, time_out=True)
    early_gravity = DoneTerm(
        func=mdp.ame2_bad_orientation,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "start_time": 0.2,
            "limit_gx": 0.985,
            "limit_gy": 0.7,
            "limit_gz": 0.0,
        },
    )
    early_hip_acc = DoneTerm(
        func=mdp.ame2_body_lin_acc_out_of_limit,
        params={
            "threshold": 100.0,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*hip_pitch_link"),
            "sensor_cfg": SceneEntityCfg("body_accel"),
            "foot_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll_link"),
            "start_time": 1.0,
        },
    ) # m/s^2; threshold extrapolated from human kinematics data
    early_illegal_contact = DoneTerm(
        func=mdp.ame2_illegal_contact,
        params={
            "threshold": 400.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link", "pelvis_contour_link"]),
            "start_time": 1.0,
        },
    )
    early_stagnation = DoneTerm(
        func=mdp.ame2_stagnation,
        params={
            "distance_threshold": 0.5,
            "goal_distance_threshold": 1.0,
            "time_window": 5.0,
            "command_name": "base_pose",
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.ame2_terrain_levels_goalreaching,
        params={"command_name": "base_pose", "move_up_dist": 0.5, "move_down_dist": 4.0},
    )
    init_yaw_range = CurrTerm(
        func=mdp.ame2_modify_term_cfg,
        params={
            "address": "events.reset_base.params.pose_range.yaw",
            "modify_fn": mdp.ame2_modify_yaw_range,
            "modify_params": {"max_steps": 24 * 2000, "limit": math.pi},
        },
    )
    goal_cone_angle = CurrTerm(
        func=mdp.ame2_modify_term_cfg,
        params={
            "address": "commands.base_pose.cone_angle_deg",
            "modify_fn": mdp.ame2_modify_goal_cone_angle,
            "modify_params": {"max_steps": 24 * 2000, "start_angle": 10.0, "end_angle": 45.0},
        },
    )
    teacher_mapping_noise = CurrTerm(
        func=mdp.ame2_modify_term_cfg,
        params={
            "address": "observations.teacher_mapping.teacher_mapping.params.custom_noise",
            "modify_fn": mdp.ame2_modify_teacher_mapping_noise,
            "modify_params": {"max_steps": 24 * 2000, "limit": 0.05},
        },
    )


##
# Environment configuration
##


@configclass
class Ame2EnvCfg_G1(ManagerBasedRLEnvCfg):
    """Configuration for the G1 goal-reaching locomotion environment on rough terrain."""

    # Scene settings
    scene: Ame2SceneCfg = Ame2SceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15
        # the default 2**26 collision stack overflows at ~12k envs
        self.sim.physx.gpu_collision_stack_size = 2**27
        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # set robot
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # viewer settings
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"
        self.viewer.eye = (0.2, 2.7, 1.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)


# Backward-compatible alias
Ame2EnvCfg = Ame2EnvCfg_G1


@configclass
class Ame2EnvCfg_G1_PLAY(Ame2EnvCfg_G1):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 2
            self.scene.terrain.terrain_generator.num_cols = 2
            self.scene.terrain.terrain_generator.curriculum = False

            # single terrain type at a fixed difficulty
            terrain_type = "stones"  
            difficulty =  0.93
            self.scene.terrain.terrain_generator.difficulty_range = (difficulty, difficulty)
            selected_cfg = self.scene.terrain.terrain_generator.sub_terrains[terrain_type]
            selected_cfg.proportion = 1.0
            self.scene.terrain.terrain_generator.sub_terrains = {terrain_type: selected_cfg}

        # disable randomization for play
        self.observations.teacher_mapping.enable_corruption = False
        self.observations.teacher_prop.enable_corruption = False
        # remove randomizations
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.randomize_actuator_gains = None
        # set actuator delay for play
        self.actions.joint_pos.delay_range = (0, 0)
        # remove random pushing event
        self.events.push_robot = None

        # turn off height scanner drift
        self.scene.height_scanner.drift_range = (0, 0)

        # disable curriculums for play (terrain levels stay active)
        self.curriculum.goal_cone_angle = None
        self.curriculum.init_yaw_range = None
        self.curriculum.teacher_mapping_noise = None

        # The Mid360 lidar and neural-map sensors are added only in the student cfgs;
        # the teacher does not use them and the 20k-ray raycast slows simulation.


@configclass
class Ame2EnvCfg_G1_Gaze_Student(Ame2EnvCfg_G1):
    """G1 gaze student: the Mid360 livox neural map replaces the privileged height scan.

    Adds the student-only sensors and observation groups, map randomization (query drift,
    clean-source mixing, random pixel outliers), noiseless teacher observations as
    distillation targets, and zero drift on the privileged scanner.
    """

    def __post_init__(self):
        super().__post_init__()
        # students start across all terrain levels (0..9) instead of the teacher's 0..5
        self.scene.terrain.max_init_terrain_level = 9

        # --- Student-Only Sensors ---
        # 1. Mid360 lidar (10 Hz, one real 20k-ray scan-sequence window per update).
        self.scene.lidar_cam = Mid360RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/mid360_link",
            update_period=1.0 / 10.0,
            mesh_prim_paths=["/World/ground"],
        )
        # 2. Livox neural map (10 Hz accumulate, per-policy-step base query).
        self.scene.neural_map = LivoxNeuralMapSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=1.0 / 10.0,
            query_drift_range=0.03,
            clean_source_ratio=0.1,
            random_pixel_ratio=0.01,
        )

        # --- Student observation groups ---
        @configclass
        class NeuralMapObsCfg(ObsGroup):
            neural_map = ObsTerm(
                func=mdp.ame2_neural_map_obs_from_sensor,
                params={"sensor_cfg": SceneEntityCfg("neural_map"), "elevation_offset": 0.74},
            )

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        self.observations.neural_map = NeuralMapObsCfg()
        self.observations.student_prop_hist = ObservationsCfg.StudentPropHistObsCfg()
        self.observations.student_cmds = ObservationsCfg.StudentCmdsObsCfg()

        # --- Noiseless teacher observations for distillation / validation ---
        self.observations.teacher_mapping.teacher_mapping.params["custom_noise"] = [0.0, 0.0, 0.0]
        self.observations.teacher_prop.base_lin_vel.noise = None
        self.observations.teacher_prop.base_ang_vel.noise = None
        self.observations.teacher_prop.projected_gravity.noise = None
        self.observations.teacher_prop.joint_pos.noise = None
        self.observations.teacher_prop.joint_vel.noise = None
        # drop the mapping-noise curriculum so teacher targets stay noiseless
        self.curriculum.teacher_mapping_noise = None

        # --- Zero drift for student training (privileged scan stays clean) ---
        self.scene.height_scanner.drift_range = (0.0, 0.0)


@configclass
class Ame2EnvCfg_G1_Gaze_Student_PLAY(Ame2EnvCfg_G1_PLAY):
    """Play cfg for the gaze student.

    Small scene / no randomization from ``Ame2EnvCfg_G1_PLAY``, plus the Mid360
    lidar + livox neural map with debug/viz buffers on and all map DR off, and
    the student observation groups. play.py auto-detects ``neural_map`` and
    renders the lidar rays (--viz_lidar), estimated map, uncertainty, and
    local-scan markers.
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.lidar_cam = Mid360RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/mid360_link",
            update_period=1.0 / 10.0,  # one 20k-ray window per real Mid360 scan
            mesh_prim_paths=["/World/ground"],
            keep_world_ray_buffers=True,  # --viz_lidar buffers; play-scale env counts only
        )
        self.scene.neural_map = LivoxNeuralMapSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            update_period=1.0 / 10.0,
            keep_debug_buffers=True,  # EM-grid/raw-hit viz; play-scale env counts only
        )

        @configclass
        class NeuralMapObsCfg(ObsGroup):
            neural_map = ObsTerm(
                func=mdp.ame2_neural_map_obs_from_sensor,
                params={"sensor_cfg": SceneEntityCfg("neural_map"), "elevation_offset": 0.74},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.observations.neural_map = NeuralMapObsCfg()
        self.observations.student_prop_hist = ObservationsCfg.StudentPropHistObsCfg()
        self.observations.student_cmds = ObservationsCfg.StudentCmdsObsCfg()
