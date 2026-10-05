# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers.command_manager import CommandTerm

"""
MDP terminations.
"""


def ame2_time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate the episode when the episode length exceeds the maximum episode length."""
    return env.episode_length_buf >= env.max_episode_length



def ame2_bad_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    start_time: float = 0.0,
    limit_gx: float = 0.985,
    limit_gy: float = 0.7,
    limit_gz: float = 0.0,
) -> torch.Tensor:
    """Terminate when the asset's orientation is bad based on projected gravity.

    Criteria (default values from paper):
    - abs(gravity_b_x) > limit_gx
    - abs(gravity_b_y) > limit_gy
    - gravity_b_z > limit_gz
    """
    # check if the start time has been reached
    current_time = env.episode_length_buf * env.step_dt
    if torch.all(current_time < start_time):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    
    if hasattr(asset.data, "body_link_quat_w") and len(asset_cfg.body_ids) == 1:
        # compute the orientation of the specific body
        body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0]]
        gravity_b = math_utils.quat_apply_inverse(body_quat_w, asset.data.GRAVITY_VEC_W)
    else:
        # fallback to root orientation
        gravity_b = asset.data.projected_gravity_b

    # specific criteria using parameters
    bad_x = gravity_b[:, 0].abs() > limit_gx
    bad_y = gravity_b[:, 1].abs() > limit_gy
    bad_z = gravity_b[:, 2] > limit_gz
    
    termination = bad_x | bad_y | bad_z

    # only terminate after start_time
    return torch.logical_and(termination, current_time >= start_time)




def ame2_illegal_contact(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg, start_time: float = 0.0
) -> torch.Tensor:
    """Terminate when the contact force on the sensor exceeds the force threshold.

    If start_time is provided, the termination is only active after the episode has progressed for start_time seconds.
    """
    # check if the start time has been reached
    current_time = env.episode_length_buf * env.step_dt
    if torch.all(current_time < start_time):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]

    # average across history and then per-body norm
    # shape: (num_envs, num_bodies)
    avg_forces = torch.mean(torch.norm(net_contact_forces, dim=-1), dim=1)

    is_illegal = avg_forces > threshold
    active_after_start = current_time >= start_time
    is_illegal_and_active = torch.logical_and(is_illegal, active_after_start.unsqueeze(1))

    # check if any contact force exceeds the threshold after start_time
    return torch.any(is_illegal_and_active, dim=1)



def ame2_body_lin_acc_out_of_limit(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
    foot_sensor_cfg: SceneEntityCfg | None = None,
    sensor_threshold: float = 1.0,
    start_time: float = 0.0,
) -> torch.Tensor:
    """Terminate when the linear acceleration of any specified body exceeds the threshold.

    The termination is only triggered if the corresponding foot is in contact with the ground.
    Contact is determined by checking the maximum force in the contact sensor's history.

    If start_time is provided, the termination is only active after the episode has progressed for start_time seconds.
    """
    # check if the start time has been reached
    current_time = env.episode_length_buf * env.step_dt
    if torch.all(current_time < start_time):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids

    if sensor_cfg is not None:
        # finite-difference acceleration sensor, attached to the asset on first use
        sensor = env.scene.sensors[sensor_cfg.name]
        if getattr(sensor, "_asset", None) is None:
            sensor.set_asset(asset)
        body_accs = sensor.data.acc_w[:, body_ids, :]

    else:
        # fall back to the PhysX body acceleration
        body_accs = asset.data.body_lin_acc_w[:, body_ids, :]
        print('alert:using physx acceleration')

    # vertical acceleration magnitude
    acc_z = body_accs[..., 2].abs()

    # check if contact information is provided
    if foot_sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[foot_sensor_cfg.name]
        foot_ids = foot_sensor_cfg.body_ids

        # print the body/foot pairing once
        if not hasattr(env, "_ame2_body_lin_acc_debug_printed"):
            body_names = [asset.body_names[i] for i in body_ids]
            foot_names = [contact_sensor.body_names[i] for i in foot_ids]
            print(f"[DEBUG] ame2_body_lin_acc_out_of_limit - Body names: {body_names}")
            print(f"[DEBUG] ame2_body_lin_acc_out_of_limit - Foot names: {foot_names}")
            env._ame2_body_lin_acc_debug_printed = True

        assert len(body_ids) == len(foot_ids), f"Number of bodies ({len(body_ids)}) and feet ({len(foot_ids)}) must match!"

        # get contact forces history: (num_envs, history_length, num_feet, 3)
        net_contact_forces = contact_sensor.data.net_forces_w_history[:, :, foot_ids, :]
        # max force over history: (num_envs, num_feet)
        max_forces = torch.max(torch.norm(net_contact_forces, dim=-1), dim=1).values
        foot_in_contact = max_forces > sensor_threshold
        # terminate only if the corresponding foot is in contact
        termination = torch.any((acc_z > threshold) & foot_in_contact, dim=1)
    else:
        # without contact info, terminate on any |acc_z| above threshold
        termination = torch.any(acc_z > threshold, dim=1)

    # only terminate after start_time
    return torch.logical_and(termination, current_time >= start_time)


def ame2_stagnation(
    env: ManagerBasedRLEnv,
    distance_threshold: float,
    goal_distance_threshold: float,
    time_window: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the robot is stagnant for a given time window.

    Stagnation is defined as moving less than distance_threshold while being at least
    goal_distance_threshold far from the goal. The root position is compared against a snapshot
    refreshed every time_window seconds and stored on the env.
    """
    # extract assets
    asset: Articulation = env.scene[asset_cfg.name]
    current_time = env.episode_length_buf * env.step_dt

    # initialize buffers if not present on the env
    if not hasattr(env, "ame2_stagnation_last_pos"):
        env.ame2_stagnation_last_pos = asset.data.root_pos_w.clone()
        env.ame2_stagnation_last_check_time = current_time.clone()

    # reset buffers for new episodes
    reset_ids = (env.episode_length_buf <= 1).nonzero(as_tuple=False).flatten()
    if len(reset_ids) > 0:
        env.ame2_stagnation_last_pos[reset_ids] = asset.data.root_pos_w[reset_ids]
        env.ame2_stagnation_last_check_time[reset_ids] = current_time[reset_ids]

    # check if time window has passed for each environment
    check_ids = (current_time - env.ame2_stagnation_last_check_time >= time_window).nonzero(as_tuple=False).flatten()
    
    # default to False
    stagnant = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if len(check_ids) > 0:
        # compute distance moved since last check
        dist_moved = torch.norm(asset.data.root_pos_w[check_ids] - env.ame2_stagnation_last_pos[check_ids], dim=1)
        
        # check distance to goal
        command = env.command_manager.get_command(command_name)
        d_xy = torch.norm(command[check_ids, :2], dim=1)
        
        # determine stagnation
        stagnant[check_ids] = (dist_moved < distance_threshold) & (d_xy > goal_distance_threshold)
        
        # update snapshots
        env.ame2_stagnation_last_pos[check_ids] = asset.data.root_pos_w[check_ids]
        env.ame2_stagnation_last_check_time[check_ids] = current_time[check_ids]

    return stagnant
