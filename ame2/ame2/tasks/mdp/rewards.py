# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import wrap_to_pi, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

"""
General.
"""





class ame2_is_terminated_term(ManagerTermBase):
    """Penalize termination for specific terms that don't correspond to episodic timeouts.

    The parameters are as follows:

    * attr:`term_keys`: The termination terms to penalize. This can be a string, a list of strings
      or regular expressions. Default is ".*" which penalizes all terminations.

    The reward is computed as the sum of the termination terms that are not episodic timeouts.
    This means that the reward is 0 if the episode is terminated due to an episodic timeout. Otherwise,
    if two termination terms are active, the reward is 2.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        term_keys = cfg.params.get("term_keys", ".*")
        self._term_names = env.termination_manager.find_terms(term_keys)

    def __call__(self, env: ManagerBasedRLEnv, term_keys: str | list[str] = ".*") -> torch.Tensor:
        reset_buf = torch.zeros(env.num_envs, device=env.device)
        for term in self._term_names:
            # sum to count multiple terminations in the same step
            reset_buf += env.termination_manager.get_term(term)

        return (reset_buf * (~env.termination_manager.time_outs)).float()


"""
Root penalties.
"""
def ame2_body_lin_acc_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize the linear acceleration of the bodies using L1-kernel (sum of norms)."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    
    # cached mask that ignores light links (< 0.1 kg)
    if not hasattr(env, "_ame2_mass_masks"):
        env._ame2_mass_masks = {}
    
    key = (asset_cfg.name, tuple(body_ids) if isinstance(body_ids, list) else body_ids)
    if key not in env._ame2_mass_masks:
        body_masses = asset.data.default_mass[:, body_ids]
        mass_mask = (body_masses >= 0.1).float().to(env.device)
        env._ame2_mass_masks[key] = mass_mask

    mass_mask = env._ame2_mass_masks[key]

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

    return torch.sum(torch.norm(body_accs, dim=-1) * mass_mask, dim=1)

"""
Joint penalties.
"""


def ame2_joint_torques_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques applied on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def ame2_joint_vel_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation using an L1-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def ame2_joint_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint velocities contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def ame2_joint_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint accelerations on the articulation using L2 squared kernel.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint accelerations contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def ame2_joint_deviation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)




"""
Action penalties.
"""


def ame2_applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def ame2_action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)



def ame2_move2goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    v_min: float = 0.3,
    v_max: float = 2.0,
    cos_thr: float = 0.5,
    dist_thr: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for moving towards the goal or being close to it."""
    # extract the command
    command = env.command_manager.get_command(command_name)
    # the command is in base frame: (x, y, z, heading)
    goal_xy = command[:, :2]
    dist_xy = torch.norm(goal_xy, dim=1)

    # extract base velocity in base frame
    asset = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) > 0:
        body_id = asset_cfg.body_ids[0]
        vel_w = asset.data.body_lin_vel_w[:, body_id]
        quat_w = asset.data.root_quat_w
        vel_b = quat_apply_inverse(quat_w, vel_w)
        vel_xy = vel_b[:, :2]
    else:
        vel_xy = asset.data.root_lin_vel_b[:, :2]
    vel_norm = torch.norm(vel_xy, dim=1)

    # cos of the angle between velocity and goal direction
    eps = 1e-6
    cos_theta = torch.sum(vel_xy * goal_xy, dim=1) / (vel_norm * dist_xy + eps)

    # reward = 1 if (dist_xy < dist_thr) OR (cos_theta > cos_thr AND v_min <= vel_norm <= v_max)
    is_close = dist_xy < dist_thr
    is_moving_towards = (cos_theta > cos_thr) & (vel_norm >= v_min) & (vel_norm <= v_max)

    return (is_close | is_moving_towards).float()


def ame2_position_tracking(
    env: ManagerBasedRLEnv, T: float, command_name: str, std: float = 2.0
) -> torch.Tensor:
    """Reward for being close to the goal in the last T seconds of the episode."""
    # extract the command
    command = env.command_manager.get_command(command_name)
    # d_xy: horizontal distance from the robot to the goal position
    d_xy = torch.norm(command[:, :2], dim=1)

    # t_left: remaining time of the current episode (in seconds)
    t_left = (env.max_episode_length - env.episode_length_buf) * env.step_dt

    # t_mask(T) = 1/T * 1(t_left < T)
    mask = (t_left < T).float() / T

    # r_position_tracking = 1 / (1 + (d_xy/std)^2) * t_mask(T)
    reward = (1.0 / (1.0 + torch.square(d_xy / std))) * mask
    return reward


def ame2_heading_tracking(
    env: ManagerBasedRLEnv, T: float, command_name: str, dist_thr: float = 0.5
) -> torch.Tensor:
    """Reward for tracking the heading when close to the goal in the last T seconds of the episode."""
    # extract the command
    command = env.command_manager.get_command(command_name)
    # the command is (num_envs, 4) -> (pos_b_x, pos_b_y, pos_b_z, yaw_b)
    # d_yaw: yaw error
    d_yaw = command[:, -1]
    # d_xy: horizontal distance from the robot to the goal position
    d_xy = torch.norm(command[:, :2], dim=1)

    # t_left: remaining time of the current episode (in seconds)
    t_left = (env.max_episode_length - env.episode_length_buf) * env.step_dt

    # t_mask(T) = 1/T * 1(t_left < T)
    mask = (t_left < T).float() / T

    # r_heading_tracking = 1 / (1 + d_yaw^2) * t_mask(T) * 1(d_xy < dist_thr)
    reward = (1.0 / (1.0 + torch.square(d_yaw))) * mask * (d_xy < dist_thr).float()
    return reward


def ame2_standatgoal(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    dist_thr: float = 0.5,
    yaw_thr: float = 0.5,
    sensor_threshold: float = 1.0,
) -> torch.Tensor:
    """Reward for stable standing at the goal."""
    # extract the command
    command = env.command_manager.get_command(command_name)
    # the command is (num_envs, 4) -> (pos_b_x, pos_b_y, pos_b_z, yaw_b)
    d_xy = torch.norm(command[:, :2], dim=1)
    d_yaw = command[:, -1].abs()

    # check if at goal
    is_at_goal = (d_xy < dist_thr) & (d_yaw < yaw_thr)

    # extract physics quantities
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # d_foot: proportion of feet not in contact
    # net_forces_w_history has shape (num_envs, history_length, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    contact_forces = torch.norm(net_forces, dim=-1).max(dim=1)[0]
    num_feet_in_contact = torch.sum((contact_forces > sensor_threshold).float(), dim=1)
    total_feet = len(sensor_cfg.body_ids)
    d_foot = (total_feet - num_feet_in_contact) / total_feet

    # d_g: base tilt, 1 - g_z^2 with g the projected gravity in the base frame
    d_g = 1.0 - torch.square(asset.data.projected_gravity_b[:, 2])

    # d_q: mean deviation from standing reference (default joint positions)
    diff_q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    d_q = torch.mean(torch.abs(diff_q), dim=1)

    # r_stand = 1(at_goal) * exp(-(d_foot + d_g + d_q + d_xy) / 4)
    exp_term = torch.exp(-(d_foot + d_g + d_q + d_xy) / 4.0)
    reward = is_at_goal.float() * exp_term

    return reward


def ame2_base_roll_rate_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize base roll rate using L2 squared kernel."""
    # extract the used quantities
    asset: RigidObject = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) > 0:
        body_id = asset_cfg.body_ids[0]
        ang_vel_w = asset.data.body_ang_vel_w[:, body_id]
        quat_w = asset.data.body_link_quat_w[:, body_id]
        ang_vel_b = quat_apply_inverse(quat_w, ang_vel_w)
        return torch.square(ang_vel_b[:, 0])
    else:
        return torch.square(asset.data.root_ang_vel_b[:, 0])


def ame2_joint_regularization_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Joint regularization term: ||q_dot||^2 + 0.01||tau||^2 + 0.001||q_ddot||^2.

    NOTE: Only the joints configured in :attr:`asset_cfg.joint_ids` will contribute to the term.
    """
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    # ||q_dot||^2
    vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)
    # ||tau||^2
    torque_sq = torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)
    # ||q_ddot||^2
    acc_sq = torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)

    return vel_sq + 0.01 * torque_sq + 0.001 * acc_sq


def ame2_link_contact_forces_l2(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, robot_weight: float) -> torch.Tensor:
    """Penalize link contact forces exceeding robot weight: ||max(F_con - G, 0)||^2."""
    # extract the used quantities
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
    net_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    # mean across history and then per-body norm
    contact_forces = torch.mean(torch.norm(net_forces, dim=-1), dim=1)
    # cap at 2 * robot_weight to limit outliers
    contact_forces = torch.clamp(contact_forces, max=2.0 * robot_weight)
    violation = torch.clamp(contact_forces - robot_weight, min=0.0)

    return torch.sum(torch.square(violation), dim=1)

def ame2_undesired_spinning(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    yaw_rate_thr: float = 2.0,
) -> torch.Tensor:
    """Indicator for spinning too fast: |yaw_rate| > yaw_rate_thr."""
    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) == 1:
        # world-frame yaw rate of the selected body
        yaw_rate = asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0], 2].abs()
    else:
        # fallback to root angular velocity in base frame
        yaw_rate = asset.data.root_ang_vel_b[:, 2].abs()
    return (yaw_rate > yaw_rate_thr).float()


def ame2_undesired_leaping(
    env: ManagerBasedRLEnv,
    feet_sensor_cfg: SceneEntityCfg,
    height_scanner_cfg: SceneEntityCfg,
    elevation_thr: float = 0.3,
    sensor_threshold: float = 1.0,
    start_time: float = 0.0,
) -> torch.Tensor:
    """Indicator for leaping on flat terrain: all feet off the ground and scan elevation range < elevation_thr."""
    # check if the start time has been reached
    current_time = env.episode_length_buf * env.step_dt
    if torch.all(current_time < start_time):
        return torch.zeros(env.num_envs, device=env.device)
    feet_sensor: ContactSensor = env.scene.sensors[feet_sensor_cfg.name]
    # check if all feet were off the ground at any physics step in the history window
    # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
    history_forces = torch.norm(feet_sensor.data.net_forces_w_history[:, :, feet_sensor_cfg.body_ids, :], dim=-1)
    # any foot in contact per physics step: (num_envs, history_length)
    any_foot_in_contact = torch.any(history_forces > sensor_threshold, dim=2)
    # leaped: at least one time step in history had NO feet in contact: (num_envs)
    leaped = torch.any(~any_foot_in_contact, dim=1)

    height_scanner: RayCaster = env.scene.sensors[height_scanner_cfg.name]
    # heights are in world frame (Z values)
    heights = height_scanner.data.ray_hits_w[..., 2]
    # elevation range of the scan
    elevation_diff = torch.max(heights, dim=1)[0] - torch.min(heights, dim=1)[0]

    return (leaped & (elevation_diff < elevation_thr) & (current_time >= start_time)).float()


def ame2_undesired_non_foot_contacts(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    sensor_threshold: float = 1.0,
) -> torch.Tensor:
    """Count non-foot links in contact, plus new contacts if air time is tracked."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = torch.max(torch.norm(sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :], dim=-1), dim=1)[0]
    count = torch.sum((forces > sensor_threshold).float(), dim=1)
    if sensor.cfg.track_air_time:
        first_contacts = sensor.compute_first_contact(env.step_dt)
        count += torch.sum(first_contacts[:, sensor_cfg.body_ids].float(), dim=1)
    return count


def ame2_undesired_stumbling(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Count links whose horizontal contact force exceeds 1.1 * vertical force + 1 N."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    f_xy = torch.max(torch.norm(sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :2], dim=-1), dim=1)[0]
    f_z = torch.max(sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2], dim=1)[0]
    stumbling = f_xy > (1.1 * f_z + 1.0)
    return torch.sum(stumbling.float(), dim=1)


def ame2_undesired_slippage(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    slip_vel_thr: float = 0.25,
    sensor_threshold: float = 1.0,
    contact_time_thr: float = 0.025,
) -> torch.Tensor:
    """Count links moving horizontally faster than slip_vel_thr while in contact for > contact_time_thr."""
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # sensor body order (USD traversal) differs from articulation order (PhysX BFS):
    # map sensor body names to asset indices once and cache them on the env
    if not hasattr(env, "_ame2_slip_asset_ids"):
        env._ame2_slip_asset_ids = {}
    key = (sensor_cfg.name, asset_cfg.name,
           tuple(sensor_cfg.body_ids) if isinstance(sensor_cfg.body_ids, list) else "all")
    if key not in env._ame2_slip_asset_ids:
        if isinstance(sensor_cfg.body_ids, list):
            names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        else:
            names = list(sensor.body_names)
        asset_ids = [asset.body_names.index(n) for n in names]
        env._ame2_slip_asset_ids[key] = torch.tensor(asset_ids, device=env.device, dtype=torch.long)
    asset_body_ids = env._ame2_slip_asset_ids[key]

    # ignore brief contacts (e.g. impact velocity at touchdown)
    # current_contact_time shape: (num_envs, num_bodies_in_sensor)
    contact_time = sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > contact_time_thr
    # horizontal velocity only; vertical motion is not slip
    body_vels = torch.norm(asset.data.body_lin_vel_w[:, asset_body_ids, :2], dim=-1)
    slipping = in_contact & (body_vels > slip_vel_thr)
    return torch.sum(slipping.float(), dim=1)


def ame2_undesired_self_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    sensor_threshold: float = 1.0,
) -> torch.Tensor:
    """Count robot links in contact with other robot links.

    Uses force_matrix_w_history which only contains forces between bodies matching
    the filter_prim_paths_expr (robot links), excluding floor/terrain contacts.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # force_matrix_w_history: (N, T, B, M, 3) with M filter bodies
    fm = sensor.data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :]
    # norm over xyz, sum over M, max over history T -> (N, B)
    forces = torch.norm(fm, dim=-1).sum(dim=-1).max(dim=1)[0]

    return torch.sum((forces > sensor_threshold).float(), dim=1)


def ame2_undesired_self_col_foot_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    thr: float = 0.07,
) -> torch.Tensor:
    """Indicator for feet being too close to each other: min pairwise distance < thr."""
    asset: Articulation = env.scene[asset_cfg.name]
    # feet positions: (num_envs, num_feet, 3)
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    # pairwise distances: (num_envs, num_feet, num_feet)
    dists = torch.cdist(feet_pos, feet_pos)
    # mask diagonal
    num_feet = dists.shape[1]
    eye = torch.eye(num_feet, device=env.device).bool()
    dists.masked_fill_(eye.unsqueeze(0), float('inf'))
    # min distance per env: (num_envs)
    min_dists, _ = dists.view(env.num_envs, -1).min(dim=1)
    return (min_dists < thr).float()


def ame2_joint_pos_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joint position limits.

    Formula: sum(max(0, q - 0.95 * q_max, 0.95 * q_min - q)) where q_min/q_max are hard limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    limits = asset.data.joint_pos_limits
    q_min, q_max = limits[..., 0], limits[..., 1]
    
    violation_max = asset.data.joint_pos - 0.95 * q_max
    violation_min = 0.95 * q_min - asset.data.joint_pos
    violation = torch.clamp(violation_max, min=0.0) + torch.clamp(violation_min, min=0.0)
    return torch.sum(violation, dim=1)


def ame2_joint_vel_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joint velocity limits.

    Formula: sum(max(0, |q_dot| - 0.9 * q_dot_max)) with q_dot_max taken from the actuator velocity limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # velocity limits cached on the env
    if not hasattr(env, "_ame2_vel_limits"):
        env._ame2_vel_limits = {}
    
    key = asset_cfg.name
    if key not in env._ame2_vel_limits:
        # limits from the actuator models: (num_envs, num_joints)
        q_vel_max = torch.zeros_like(asset.data.joint_vel)
        for actuator in asset.actuators.values():
            if hasattr(actuator, "velocity_limit"):
                # (num_envs, num_joints_in_actuator)
                q_vel_max[:, actuator.joint_indices] = actuator.velocity_limit.to(env.device)
        
        env._ame2_vel_limits[key] = q_vel_max

    q_vel_max = env._ame2_vel_limits[key]
    
    violation = asset.data.joint_vel.abs() - 0.9 * q_vel_max
    return torch.sum(torch.clamp(violation, min=0.0), dim=1)


def ame2_joint_torque_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joint torque limits.

    Formula: sum(max(0, |tau| - 0.8 * tau_max)) with tau_max taken from the actuator effort limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    
    # torque limits cached on the env
    if not hasattr(env, "_ame2_torque_limits"):
        env._ame2_torque_limits = {}
    
    key = asset_cfg.name
    if key not in env._ame2_torque_limits:
        # limits from the actuator models: (num_envs, num_joints)
        torque_limits = torch.zeros_like(asset.data.applied_torque)
        for actuator in asset.actuators.values():
            if hasattr(actuator, "effort_limit"):
                # (num_envs, num_joints_in_actuator)
                torque_limits[:, actuator.joint_indices] = actuator.effort_limit.to(env.device)
        
        env._ame2_torque_limits[key] = torque_limits

    torque_limits = env._ame2_torque_limits[key]
    
    violation = asset.data.applied_torque.abs() - 0.8 * torque_limits
    return torch.sum(torch.clamp(violation, min=0.0), dim=1)

def ame2_optional_uppershaping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_scanner_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    elevation_thr: float = 0.3,
) -> torch.Tensor:
    """Squared deviation of the selected joints from their defaults, gated by terrain roughness.

    The term is scaled by 1.0 when the scan elevation range is below elevation_thr and by 0.1 otherwise.
    """
    # scan elevation range, as in ame2_undesired_leaping
    height_scanner: RayCaster = env.scene.sensors[height_scanner_cfg.name]
    heights = height_scanner.data.ray_hits_w[..., 2]
    elevation_diff = torch.max(heights, dim=1)[0] - torch.min(heights, dim=1)[0]

    # joint deviation (sum of squares)
    asset: Articulation = env.scene[asset_cfg.name]
    diff_q = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    sse = torch.sum(torch.square(diff_q), dim=1)

    multiplier = torch.where(elevation_diff < elevation_thr, 1.0, 0.1)
    reward = sse * multiplier

    return reward


def ame2_optional_foot_yaw(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    torso_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="torso_link"),
    lin_vel_thr: float = 0.3,
) -> torch.Tensor:
    """Squared yaw difference between feet and torso; full weight above lin_vel_thr forward speed, 0.1 below."""
    asset: Articulation = env.scene[asset_cfg.name]

    # torso yaw (first matched body, else root)
    if isinstance(torso_cfg.body_ids, (list, tuple, torch.Tensor)) and len(torso_cfg.body_ids) > 0:
        torso_id = torso_cfg.body_ids[0]
        torso_quat = asset.data.body_link_quat_w[:, torso_id]
    else:
        torso_quat = asset.data.root_quat_w

    # yaw-only quaternion (w, 0, 0, z) -> yaw = 2 * atan2(z, w)
    t_yaw_q = yaw_quat(torso_quat)
    t_yaw = 2.0 * torch.atan2(t_yaw_q[:, 3], t_yaw_q[:, 0])

    # feet yaw
    feet_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids]
    f_yaw_q = yaw_quat(feet_quat)
    # f_yaw_q: (N, num_feet, 4)
    f_yaw = 2.0 * torch.atan2(f_yaw_q[..., 3], f_yaw_q[..., 0])

    yaw_diff = wrap_to_pi(f_yaw - t_yaw.unsqueeze(1))
    penalty = torch.sum(torch.square(yaw_diff), dim=1)

    lin_vel_x = asset.data.root_lin_vel_b[:, 0].abs()

    return penalty * ( 0.1 + 0.9 * (lin_vel_x > lin_vel_thr).float())


def ame2_optional_torsoshaping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    height_scanner_cfg: SceneEntityCfg,
    elevation_thr: float,
) -> torch.Tensor:
    """Penalize torso tilt based on terrain roughness.

    Returns 1 - g_z^2 of the torso frame, zeroed when the scan elevation range is >= elevation_thr.
    """
    height_scanner: RayCaster = env.scene.sensors[height_scanner_cfg.name]
    heights = height_scanner.data.ray_hits_w[..., 2]
    elevation_diff = torch.max(heights, dim=1)[0] - torch.min(heights, dim=1)[0]

    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) > 0:
        body_id = asset_cfg.body_ids[0]
        quat_w = asset.data.body_link_quat_w[:, body_id]
    else:
        quat_w = asset.data.root_quat_w

    gravity_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    gravity_b = quat_apply_inverse(quat_w, gravity_w)
    
    tilt_l2 = 1.0 - torch.square(gravity_b[:, 2])

    multiplier = torch.where(elevation_diff < elevation_thr, 1.0, 0.0)
    
    return tilt_l2 * multiplier

def ame2_optional_front_hind_reg(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    left_pair: list[str],
    right_pair: list[str],
    sensor_threshold: float = 1.0,
    start_time: float = 1.0,
    dist_thr: float = 0.1,
) -> torch.Tensor:
    """Penalize same-side flight phases and clustering for two body pairs.

    Counts: both bodies of ``left_pair`` or both of ``right_pair`` off the ground (one count),
    plus each pair closer than ``dist_thr``. Active after ``start_time`` seconds.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    
    # resolve indices separately for sensor and asset (orders differ)
    left_idx_sens = [sensor.body_names.index(f) for f in left_pair]
    right_idx_sens = [sensor.body_names.index(f) for f in right_pair]
    
    left_idx_asset = [asset.body_names.index(f) for f in left_pair]
    right_idx_asset = [asset.body_names.index(f) for f in right_pair]

    # max force over history
    forces = torch.max(torch.norm(sensor.data.net_forces_w_history, dim=-1), dim=1)[0]
    
    left_in_air = (forces[:, left_idx_sens[0]] < sensor_threshold) & (forces[:, left_idx_sens[1]] < sensor_threshold)
    right_in_air = (forces[:, right_idx_sens[0]] < sensor_threshold) & (forces[:, right_idx_sens[1]] < sensor_threshold)
    
    pacing = left_in_air | right_in_air

    left_dist = torch.norm(asset.data.body_pos_w[:, left_idx_asset[0]] - asset.data.body_pos_w[:, left_idx_asset[1]], dim=-1)
    right_dist = torch.norm(asset.data.body_pos_w[:, right_idx_asset[0]] - asset.data.body_pos_w[:, right_idx_asset[1]], dim=-1)
    
    left_close = left_dist < dist_thr
    right_close = right_dist < dist_thr

    current_time = env.episode_length_buf * env.step_dt
    mask = (current_time >= start_time).float()
    
    violations = (pacing.float() + left_close.float() + right_close.float()) * mask

    return violations
