# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from ame2.assets import ISAAC_ASSET_DIR


G1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_ASSET_DIR}/g1_lidar/g1_lidar.usd",
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
            enabled_self_collisions=True, solver_position_iteration_count=6, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.78),
        joint_pos={
            ".*_hip_pitch_joint": -0.30,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_joint": 0.62,
            ".*_ankle_pitch_joint": -0.33,
            ".*_ankle_roll_joint": 0.0,
            ".*_elbow_joint": 0.30,
            "waist_.*_joint": 0.0,
            "left_shoulder_roll_joint": 0.16,
            "left_shoulder_pitch_joint": 0.35,
            "right_shoulder_roll_joint": -0.16,
            "right_shoulder_pitch_joint": 0.35,
            ".*_shoulder_yaw_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9, # unused
    actuators={
        "hips": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
            ],
            effort_limit=88,
            saturation_effort=88*1.25,
            velocity_limit=32.0,
            stiffness={
                ".*_hip_yaw_joint": 88.0*1,
                ".*_hip_roll_joint": 139.0*1,
                ".*_hip_pitch_joint": 88.0*1,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 3.0,
                ".*_hip_pitch_joint": 2.0
            },
            armature={
                ".*_hip_yaw_joint": 0.01017752004,
                ".*_hip_roll_joint": 0.025101925,
                ".*_hip_pitch_joint": 0.01017752004
            },
            friction=0.0,
        ),
        "knees": DCMotorCfg(
            joint_names_expr=[".*_knee_joint",],
            effort_limit=139,
            saturation_effort=139*1.25,
            velocity_limit=20.0,
            stiffness=139.0*1,
            damping=3.0,
            armature=0.025101925,
            friction=0.0,
        ),
        "waist_yaw": DCMotorCfg(
            joint_names_expr=["waist_yaw_joint",],
            effort_limit=88,
            saturation_effort=88*1.25,
            velocity_limit=32.0,
            stiffness=88.0*1,
            damping=2.0,
            armature=0.01017752004,
            friction=0.0,
        ),
        "waist_roll": DCMotorCfg(
            joint_names_expr=["waist_roll_joint"],
            effort_limit=50,
            saturation_effort=50*1.25,
            velocity_limit=37.0,
            stiffness=50.0*1,
            damping=1.0,
            armature=0.00721945,
            friction=0.0,
        ),
        "waist_pitch": DCMotorCfg(
            joint_names_expr=["waist_pitch_joint",],
            effort_limit=50,
            saturation_effort=50*1.25,
            velocity_limit=37.0,
            stiffness=50.0*1,
            damping=1.0,
            armature=0.00721945,
            friction=0.0,
        ),
        "ankle_pitch": DCMotorCfg(
            effort_limit=50,
            saturation_effort=50*1.25,
            velocity_limit=37,
            joint_names_expr=[".*_ankle_pitch_joint"],
            stiffness=35.0*1,
            damping=0.8,
            armature=0.00721945,
            friction=0.00,
        ),
        "ankle_roll": DCMotorCfg(
            effort_limit=50,
            saturation_effort=50*1.25,
            velocity_limit=37,
            joint_names_expr=[".*_ankle_roll_joint"],
            stiffness=35.0*1,
            damping=0.8,
            armature=0.00721945,
            friction=0.0,
        ),
        "shoulder_pitch": DCMotorCfg(
            joint_names_expr=[".*_shoulder_pitch_joint",],
            effort_limit=25,
            saturation_effort=25*1.25,
            velocity_limit=37.0,
            stiffness=35.0*1,
            damping=0.8,
            armature=0.003609725,
            friction=0.0,
        ),
        "shoulder_roll": DCMotorCfg(
            joint_names_expr=[".*_shoulder_roll_joint",],
            effort_limit=25,
            saturation_effort=25*1.25,
            velocity_limit=37.0,
            stiffness=40.0*1,
            damping=1.0,
            armature=0.003609725,
            friction=0.0,
        ),
        "shoulder_yaw": DCMotorCfg(
            joint_names_expr=[".*_shoulder_yaw_joint",],
            effort_limit=25,
            saturation_effort=25*1.25,
            velocity_limit=37.0,
            stiffness=40.0*1,
            damping=1.0,
            armature=0.003609725,
            friction=0.00,
        ),
        "elbow": DCMotorCfg(
            joint_names_expr=[".*_elbow_joint",],
            effort_limit=25,
            saturation_effort=25*1.25,
            velocity_limit=37.0,
            stiffness=40.0*1,
            damping=1.0,
            armature=0.003597,
            friction=0.0,
        ),
    },
)
