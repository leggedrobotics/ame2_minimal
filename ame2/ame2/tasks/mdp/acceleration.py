# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import weakref
from dataclasses import MISSING
from typing import TYPE_CHECKING, Any, Sequence

from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import Articulation


class BodyAccelerationSensor(SensorBase):
    """Sensor that estimates body linear velocity and acceleration by finite differences.

    On each sensor update, velocity is the finite difference of body positions and acceleration
    the finite difference of those velocities, both over the time since the previous update.
    The PhysX body acceleration is stored alongside for reference.
    """

    def __init__(self, cfg: SensorBaseCfg):
        """Initializes the sensor."""
        super().__init__(cfg)
        
        self._data = BodyAccelerationSensorData()
        
        self._asset = None
        self._prev_pos = None
        self._prev_vel = None

    @property
    def data(self) -> BodyAccelerationSensorData:
        """Data from the sensor."""
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the sensor internals."""
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        
        # re-seed the finite-difference history from the current asset state
        if self._prev_pos is not None:
            self._prev_pos[env_ids] = self._asset.data.body_pos_w[env_ids]
            self._prev_vel[env_ids] = self._asset.data.body_lin_vel_w[env_ids]
            # asset data still holds the pre-reset pose here, so the next 2 FD accelerations are spurious
            self._blank_steps[env_ids] = 2

    """
    Implementation.
    """

    def _initialize_impl(self):
        """Initializes the sensor handles."""
        super()._initialize_impl()
        self._asset = None

    def set_asset(self, asset: Articulation):
        """Sets the asset to track and initializes buffers."""
        self._asset = asset
        
        num_bodies = self._asset.data.body_pos_w.shape[1]
        self._data.vel_w = torch.zeros(self._num_envs, num_bodies, 3, device=self._device)
        self._data.acc_w = torch.zeros(self._num_envs, num_bodies, 3, device=self._device)
        self._data.physx_acc_w = torch.zeros(self._num_envs, num_bodies, 3, device=self._device)
        
        self._prev_pos = torch.zeros(self._num_envs, num_bodies, 3, device=self._device)
        self._prev_vel = torch.zeros(self._num_envs, num_bodies, 3, device=self._device)
        
        self._prev_pos[:] = self._asset.data.body_pos_w
        self._prev_vel[:] = self._asset.data.body_lin_vel_w
        self._blank_steps = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the data buffer (called by the environment after decimation)."""
        if self._asset is None:
            return
            
        # per-env time since the last sensor update
        dt = (self._timestamp[env_ids] - self._timestamp_last_update[env_ids]).view(-1, 1, 1)
        
        # skip envs with dt ~ 0 (redundant updates)
        mask = (dt > 1e-6).flatten()
        if torch.any(mask):
            valid_env_ids = env_ids[mask] if isinstance(env_ids, torch.Tensor) else [env_ids[i] for i, m in enumerate(mask) if m]
            valid_dt = dt[mask]
            
            current_pos = self._asset.data.body_pos_w[valid_env_ids]
            
            calc_vel = (current_pos - self._prev_pos[valid_env_ids]) / valid_dt
            
            calc_acc = (calc_vel - self._prev_vel[valid_env_ids]) / valid_dt
            
            # zero the acceleration for the first 2 updates after a reset
            blank = self._blank_steps[valid_env_ids] > 0
            calc_acc = torch.where(blank.view(-1, 1, 1), torch.zeros_like(calc_acc), calc_acc)
            self._blank_steps[valid_env_ids] = (self._blank_steps[valid_env_ids] - 1).clamp(min=0)

            self._data.vel_w[valid_env_ids] = calc_vel
            self._data.acc_w[valid_env_ids] = calc_acc
            self._data.physx_acc_w[valid_env_ids] = self._asset.data.body_lin_acc_w[valid_env_ids]
            
            self._prev_pos[valid_env_ids] = current_pos
            self._prev_vel[valid_env_ids] = calc_vel


@configclass
class BodyAccelerationSensorCfg(SensorBaseCfg):
    """Configuration for the body acceleration sensor."""
    class_type: type = BodyAccelerationSensor

    asset_cfg: Any = None
    """The asset configuration to track. Usually a SceneEntityCfg."""


class BodyAccelerationSensorData:
    """Data container for the body acceleration sensor."""
    vel_w: torch.Tensor = None
    """Averaged finite-difference velocity in world frame (num_envs, num_bodies, 3)."""
    acc_w: torch.Tensor = None
    """Averaged finite-difference acceleration in world frame (num_envs, num_bodies, 3)."""
    physx_acc_w: torch.Tensor = None
    """Instantaneous PhysX acceleration in world frame (num_envs, num_bodies, 3)."""
