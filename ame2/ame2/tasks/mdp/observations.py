# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define observations for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to specify
the observation function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import (
    quat_rotate_inverse,
    quat_unique,
    wrap_to_pi,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv


"""
Root state.
"""





def ame2_base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def ame2_base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def ame2_projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b





"""
Joint state.
"""





def ame2_joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions."""
    extract_asset: Articulation = env.scene[asset_cfg.name]
    return extract_asset.data.joint_pos[:, asset_cfg.joint_ids] - extract_asset.data.default_joint_pos[:, asset_cfg.joint_ids]





def ame2_joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint velocities of the asset w.r.t. the default joint velocities."""
    extract_asset: Articulation = env.scene[asset_cfg.name]
    return extract_asset.data.joint_vel[:, asset_cfg.joint_ids] - extract_asset.data.default_joint_vel[:, asset_cfg.joint_ids]


"""
Sensors.
"""


def ame2_height_scan(
    env: ManagerBasedEnv, 
    sensor_cfg: SceneEntityCfg, 
    offset: float = 0.8,
    void_bias_range: tuple[float, float] = (-10.0, -10.0)
) -> torch.Tensor:
    """Height scan from the ray-caster sensor."""
    sensor: RayCaster = env.scene[sensor_cfg.name]
    z = sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[:, 2].unsqueeze(1) + offset

    # fill voids (inf/nan) with the lowest valid height plus a random per-env bias
    is_valid = torch.isfinite(z)
    if not torch.all(is_valid):
        z_masked = torch.where(is_valid, z, torch.tensor(1e6, device=z.device))
        min_valid_z, _ = torch.min(z_masked, dim=1, keepdim=True)
        
        # envs without any valid ray fall back to -10
        any_valid = torch.any(is_valid, dim=1, keepdim=True)
        min_valid_z = torch.where(any_valid, min_valid_z, torch.tensor(-10.0, device=z.device))
        
        num_envs = z.shape[0]
        bias = torch.empty((num_envs, 1), device=z.device).uniform_(*void_bias_range)
        
        z = torch.where(is_valid, z, min_valid_z + bias)
        
    # processed world-frame heights, used for visualization
    sensor.data._processed_z_w = (z + sensor.data.pos_w[:, 2].unsqueeze(1) - offset).detach()
    
    return z


def ame2_teacher_mapping(env: ManagerBasedEnv, 
    sensor_cfg: SceneEntityCfg, 
    offset: float = 0.8, 
    clip_z: tuple[float, float] | None = (-1.5, 1.5),
    void_bias_range: tuple[float, float] = (-1.5, -0.65),
    custom_noise: list[float] | None = None) -> torch.Tensor:
    """Height map of shape (num_envs, nx, ny, 3) with channels (x, y, z).

    x and y are the grid coordinates in the base frame (scan grid shifted by the sensor offset);
    z is the hit height relative to the sensor plus ``offset``, optionally clipped to ``clip_z``.
    Voids are filled as in :func:`ame2_height_scan`. ``custom_noise`` = [x, y, z] adds uniform noise.
    """
    sensor: RayCaster = env.scene[sensor_cfg.name]
    # hit height relative to the sensor, plus offset
    z = sensor.data.ray_hits_w[..., 2] - sensor.data.pos_w[:, 2].unsqueeze(1) + offset
    
    # fill voids (inf/nan) with the lowest valid height plus a random per-env bias
    is_valid = torch.isfinite(z)
    if not torch.all(is_valid):
        z_masked = torch.where(is_valid, z, torch.tensor(1e6, device=z.device))
        min_valid_z, _ = torch.min(z_masked, dim=1, keepdim=True)
        
        # envs without any valid ray fall back to -10
        any_valid = torch.any(is_valid, dim=1, keepdim=True)
        min_valid_z = torch.where(any_valid, min_valid_z, torch.tensor(-10.0, device=z.device))
        
        num_envs = z.shape[0]
        bias = torch.empty((num_envs, 1), device=z.device).uniform_(*void_bias_range)
        z = torch.where(is_valid, z, min_valid_z + bias)

    # processed world-frame heights, used for visualization
    sensor.data._processed_z_w = (z + sensor.data.pos_w[:, 2].unsqueeze(1) - offset).detach()

    if clip_z is not None:
        z = torch.clamp(z, min=clip_z[0], max=clip_z[1])

    # grid x, y are constant in the yaw-aligned sensor frame, so they are rebuilt from the pattern cfg
    device = z.device
    num_envs = z.shape[0]

    x_size, y_size = sensor.cfg.pattern_cfg.size
    res = sensor.cfg.pattern_cfg.resolution
    
    sensor_offset_x = sensor.cfg.offset.pos[0]
    sensor_offset_y = sensor.cfg.offset.pos[1]
    
    # grid coordinates shifted by the sensor offset
    grid_x, grid_y = torch.meshgrid(
        torch.arange(-x_size / 2, x_size / 2 + 1e-5, res, device=device) + sensor_offset_x,
        torch.arange(-y_size / 2, y_size / 2 + 1e-5, res, device=device) + sensor_offset_y,
        indexing="ij"
    )
    
    nx = grid_x.shape[0]
    ny = grid_x.shape[1]

    # expand these to match (num_envs, nx * ny)
    local_x = grid_x.flatten().unsqueeze(0).expand(num_envs, -1)
    local_y = grid_y.flatten().unsqueeze(0).expand(num_envs, -1)

    if custom_noise is not None:
        # uniform noise with half-widths custom_noise = [x, y, z]
        noise_x = (torch.rand_like(local_x) * 2 - 1) * custom_noise[0]
        noise_y = (torch.rand_like(local_y) * 2 - 1) * custom_noise[1]
        noise_z = (torch.rand_like(z) * 2 - 1) * custom_noise[2]
        
        local_x = local_x + noise_x
        local_y = local_y + noise_y
        z = z + noise_z

    # (num_envs, nx * ny, 3)
    xyz = torch.stack([local_x, local_y, z], dim=-1)

    # (num_envs, nx, ny, 3)
    xyz = xyz.view(num_envs, nx, ny, 3)

    return xyz




def ame2_neural_map_obs_from_sensor(
    env: ManagerBasedEnv, 
    sensor_cfg: SceneEntityCfg,
    elevation_offset: float = 0.5
) -> torch.Tensor:
    """Neural map observation of shape (num_envs, nx, ny, 4) with channels (x, y, z_est, uncertainty).

    z_est is shifted by ``elevation_offset`` and clipped to [-2, 2]; the uncertainty is clipped to
    [0.01, 2] and mapped to ``0.5 * log(u) + 0.5``.
    """
    sensor = env.scene[sensor_cfg.name]
    
    # (num_envs, L_base * W_base, 2) with channels (z_est, unc)
    raw_data = sensor.data
    num_envs = raw_data.shape[0]
    
    nx, ny = sensor.cfg.L_base, sensor.cfg.W_base
    device = raw_data.device
    
    # static base-frame grid from the sensor, (L_base, W_base) each
    local_x = sensor.Xb.flatten().unsqueeze(0).expand(num_envs, -1)
    local_y = sensor.Yb.flatten().unsqueeze(0).expand(num_envs, -1)
    
    z_est = raw_data[..., 0].detach()
    unc = raw_data[..., 1].detach()
    
    # (num_envs, nx * ny, 4)
    xyzu = torch.stack([local_x, local_y, z_est, unc], dim=-1)
    
    # (num_envs, nx, ny, 4)
    xyzu = xyzu.view(num_envs, nx, ny, 4)
    
    # z channel: add offset and clip
    xyzu[:, :, :, 2] = torch.clip(xyzu[:, :, :, 2] + elevation_offset, min=-2.0, max=2.0)
    
    # uncertainty channel: clip and log-transform
    xyzu[:, :, :, 3] = torch.clip(xyzu[:, :, :, 3], min=0.01, max=2.0)
    xyzu[:, :, :, 3] = 0.5 * torch.log(xyzu[:, :, :, 3]) + 0.5
    
    return xyzu


"""
Actions.
"""


def ame2_last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The last action applied to the environment."""
    return torch.clip(env.action_manager.prev_action, min=-50., max=50.)


def ame2_time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    """The normalized time left in the episode."""
    return (1.0 - env.episode_length_buf / env.max_episode_length).unsqueeze(1)


"""
Commands.
"""


def ame2_commands_obs(
    env: ManagerBasedRLEnv, command_name: str, dist_thr: float | None = None
) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name.

    The observation is: [x, y, sin(yaw_error), cos(yaw_error)].
    If dist_thr is provided, the goal position is clipped to dist_thr and the yaw error is randomized
    if the actual goal is further than dist_thr.
    """
    command = env.command_manager.get_command(command_name)
    # the command is (num_envs, 4) -> (pos_b_x, pos_b_y, pos_b_z, yaw_b)
    pos_xy = command[:, :2]
    yaw_error = command[:, -1]

    if dist_thr is not None:
        dist = torch.norm(pos_xy, dim=1)
        is_far = dist > dist_thr
        # clip position
        scale = dist_thr / torch.clamp(dist, min=1e-6)
        scaled_pos_xy = pos_xy * scale.unsqueeze(1)
        obs_pos_xy = torch.where(is_far.unsqueeze(1), scaled_pos_xy, pos_xy)
        # randomize yaw error if far
        random_yaw = (torch.rand_like(yaw_error) * 2 * torch.pi) - torch.pi
        obs_yaw = torch.where(is_far, random_yaw, yaw_error)
    else:
        obs_pos_xy = pos_xy
        obs_yaw = yaw_error

    return torch.cat(
        [
            obs_pos_xy,
            torch.sin(obs_yaw).unsqueeze(1),
            torch.cos(obs_yaw).unsqueeze(1),
        ],
        dim=1,
    )


"""
Contact states.
"""


def ame2_contact_states(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mass_threshold: float = 0.1,
    force_threshold: float = 1.0,
    ignore_bodies: list[str] | None = None,
) -> torch.Tensor:
    """Binary contact states (0/1) for all bodies with mass > mass_threshold.

    Returns a tensor of shape (num_envs, num_heavy_bodies) with 1.0 where
    the net contact force norm exceeds force_threshold, 0.0 otherwise.

    On the first call per configuration, prints the selected bodies and their masses.
    Caches the sensor indices tensor on ``env._ame2_contact_sensor_indices``.
    """
    if ignore_bodies is None:
        ignore_bodies = []
    ignore_set = set(ignore_bodies)

    # sensor indices are cached on the env per configuration
    if not hasattr(env, "_ame2_contact_sensor_indices"):
        env._ame2_contact_sensor_indices = {}

    key = (sensor_cfg.name, asset_cfg.name, mass_threshold, tuple(sorted(ignore_set)))

    if key not in env._ame2_contact_sensor_indices:
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene[sensor_cfg.name]

        body_masses = asset.data.default_mass[0]  # (num_bodies,)
        body_names = asset.data.body_names
        sensor_body_names = sensor.body_names

        # print the body selection
        print("\n" + "=" * 60)
        print(f"[ame2_contact_states] DEBUG: mass_threshold = {mass_threshold}")
        print(f"  ignore_bodies = {ignore_bodies}")
        print(f"  Total bodies: {len(body_names)}")
        for i, (name, mass) in enumerate(zip(body_names, body_masses.tolist())):
            selected = mass > mass_threshold and name not in ignore_set
            marker = "V" if selected else "X"
            reason = ""
            if name in ignore_set:
                reason = " (IGNORED)"
            elif mass <= mass_threshold:
                reason = " (too light)"
            print(f"    [{marker}] {i:2d}: {name:30s}  mass = {mass:.4f} kg{reason}")

        # keep bodies with mass > threshold that are not ignored
        heavy_names = [
            n for n, m in zip(body_names, body_masses.tolist())
            if m > mass_threshold and n not in ignore_set
        ]
        print(f"  Selected {len(heavy_names)} bodies: {heavy_names}")
        print("=" * 60 + "\n")

        indices = []
        for hname in heavy_names:
            if hname in sensor_body_names:
                indices.append(sensor_body_names.index(hname))
        env._ame2_contact_sensor_indices[key] = torch.tensor(
            indices, device=sensor.data.net_forces_w.device, dtype=torch.long
        )

    sensor_indices = env._ame2_contact_sensor_indices[key]

    sensor: ContactSensor = env.scene[sensor_cfg.name]
    # net_forces_w_history shape: (num_envs, history_length, num_bodies, 3)
    selected_forces = sensor.data.net_forces_w_history[:, :, sensor_indices, :]
    # max force norm across history -> (num_envs, num_selected)
    contact_force_norms = torch.max(torch.norm(selected_forces, dim=-1), dim=1)[0]
    contact_states = (contact_force_norms > force_threshold).float()
    return contact_states
