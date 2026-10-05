# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import math
import re
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from isaaclab.assets import Articulation
from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ame2_modify_reward_weight(ManagerTermBase):
    """Curriculum that modifies the reward weight based on a step-wise schedule."""

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # obtain term configuration
        term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(term_name)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        weight: float,
        num_steps: int,
    ) -> float:
        # update term settings
        if env.common_step_counter > num_steps:
            self._term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)

        return self._term_cfg.weight


class ame2_modify_env_param(ManagerTermBase):
    """Curriculum term for modifying an environment parameter at runtime."""

    NO_CHANGE: ClassVar = object()

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # resolve term configuration
        if "address" not in cfg.params:
            raise ValueError("The 'address' parameter must be specified in the curriculum term configuration.")

        # store current address
        self._address: str = cfg.params["address"]
        # store accessor functions
        self._get_fn: callable = None
        self._set_fn: callable = None

    def __del__(self):
        """Destructor to clean up the compiled functions."""
        # clear the getter and setter functions
        self._get_fn = None
        self._set_fn = None
        self._container = None
        self._last_path = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        address: str,
        modify_fn: callable,
        modify_params: dict | None = None,
    ):
        # fetch the getter and setter functions if not already compiled
        if not self._get_fn:
            self._get_fn, self._set_fn = self._process_accessors(self._env, self._address)

        # resolve none type
        modify_params = {} if modify_params is None else modify_params

        # get the current value of the target attribute
        data = self._get_fn()
        # modify the value using the provided function
        new_val = modify_fn(self._env, env_ids, data, **modify_params)
        # set the modified value back to the target attribute
        if new_val is not self.NO_CHANGE:
            self._set_fn(new_val)

    def _process_accessors(self, root: ManagerBasedRLEnv, path: str) -> tuple[callable, callable]:
        """Process and return the (getter, setter) functions for a dotted attribute path."""
        # Turn "a.b[2].c" into ["a", ("b", 2), "c"] and store in parts
        path_parts: list[str | tuple[str, int]] = []
        for part in path.split("."):
            m = re.compile(r"^(\w+)\[(\d+)\]$").match(part)
            if m:
                path_parts.append((m.group(1), int(m.group(2))))
            else:
                path_parts.append(part)

        # Traverse the parts to find the container
        container = root
        for container_path in path_parts[:-1]:
            if isinstance(container_path, tuple):
                # we are accessing a list element
                name, idx = container_path
                # find underlying attribute
                if isinstance(container, dict):
                    seq = container[name]
                else:
                    seq = getattr(container, name)
                # save the container for the next iteration
                container = seq[idx]
            else:
                # we are accessing a dictionary key or an attribute
                if isinstance(container, dict):
                    container = container[container_path]
                else:
                    container = getattr(container, container_path)

        # save the container and the last part of the path
        self._container = container
        self._last_path = path_parts[-1]

        # build the getter and setter
        if isinstance(self._container, tuple):
            get_value = lambda: self._container[self._last_path]

            def set_value(val):
                tuple_list = list(self._container)
                tuple_list[self._last_path] = val
                self._container = tuple(tuple_list)

        elif isinstance(self._container, (list, dict)):
            get_value = lambda: self._container[self._last_path]

            def set_value(val):
                self._container[self._last_path] = val

        elif isinstance(self._container, object):
            get_value = lambda: getattr(self._container, self._last_path)
            set_value = lambda val: setattr(self._container, self._last_path, val)
        else:
            raise TypeError(
                f"Unable to build accessors for address '{path}'. Unknown type found for access variable:"
                f" '{type(self._container)}'. Expected a list, dict, or object with attributes."
            )

        return get_value, set_value


class ame2_modify_term_cfg(ame2_modify_env_param):
    """Curriculum for modifying a manager term configuration at runtime."""

    def __init__(self, cfg, env):
        # initialize the parent
        super().__init__(cfg, env)
        # overwrite the simplified address with the full manager path
        self._address = self._address.replace("s.", "_manager.cfg.", 1)


def ame2_modify_yaw_range(env: ManagerBasedRLEnv, env_ids: Sequence[int], data: float, max_steps: int, limit: float) -> tuple[float, float] | object:
    """Modify the initial yaw range based on a step-wise schedule.
    
    Gradually increases the yaw range from 0 to +/- limit over `max_steps` steps.
    """
    progress = min(env.common_step_counter / max_steps, 1.0)
    new_limit = progress * limit
    
    # logging
    if "log" in env.extras:
        env.extras["log"].update({"Curriculum/YawLimit": new_limit})

    return (-new_limit, new_limit)


def ame2_modify_goal_cone_angle(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], data: float, max_steps: int, start_angle: float, end_angle: float
) -> float | object:
    """Modify the goal sampling cone angle based on a step-wise schedule.

    Gradually increases the angle from `start_angle` to `end_angle` over `max_steps` steps.
    """
    progress = min(env.common_step_counter / max_steps, 1.0)
    new_angle = start_angle + progress * (end_angle - start_angle)

    # logging
    if "log" in env.extras:
        env.extras["log"].update({"Curriculum/GoalConeAngle": new_angle})

    return new_angle


def ame2_modify_height_scan_noise_max(env: ManagerBasedRLEnv, env_ids: Sequence[int], data: float, max_steps: int, limit: float) -> float | object:
    """Modify the height scan noise max range based on a step-wise schedule.
    
    Gradually increases the noise max from 0 to limit over `max_steps` steps.
    """
    progress = min(env.common_step_counter / max_steps, 1.0)
    new_limit = progress * limit
    
    # logging
    if "log" in env.extras:
        env.extras["log"].update({"Curriculum/HeightScanNoiseMax": new_limit})
    
    if env.common_step_counter <= max_steps:
        return new_limit
    
    return ame2_modify_env_param.NO_CHANGE

def ame2_modify_height_scan_noise_min(env: ManagerBasedRLEnv, env_ids: Sequence[int], data: float, max_steps: int, limit: float) -> float | object:
    """Modify the height scan noise min range based on a step-wise schedule.
    
    Gradually decreases the noise min from 0 to limit over `max_steps` steps.
    """
    progress = min(env.common_step_counter / max_steps, 1.0)
    new_limit = progress * limit
    
    # logging
    if "log" in env.extras:
        env.extras["log"].update({"Curriculum/HeightScanNoiseMin": new_limit})
    
    if env.common_step_counter <= max_steps:
        return new_limit
    
    return ame2_modify_env_param.NO_CHANGE


def ame2_modify_teacher_mapping_noise(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], data: list[float], max_steps: int, limit: float
) -> list[float] | object:
    """Modify the teacher mapping noise (last element) based on a step-wise schedule.

    Gradually increases the noise from 0 to limit over `max_steps` steps.
    """
    progress = min(env.common_step_counter / max_steps, 1.0)
    new_val = progress * limit

    # logging
    if "log" in env.extras:
        env.extras["log"].update({"Curriculum/TeacherMappingNoise": new_val})

    # data is expected to be [noise_x, noise_y, noise_z]
    if isinstance(data, list) and len(data) > 0:
        data[-1] = new_val
        return data

    return new_val




"""
Terrain Curriculums.
"""


def ame2_terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> float:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float()).item()


def ame2_terrain_levels_goalreaching(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    move_up_dist: float = 0.5,
    move_down_dist: float = 4.0,
) -> float:
    """Curriculum based on the distance to the goal at the end of the episode.

    Promote if dist < move_up_dist and the success-rate EMA > 0.5, demote if dist > move_down_dist.
    Also logs the success-rate EMA and terrain level per sub-terrain type.
    """
    # initialize buffers if they don't exist
    if not hasattr(env, "success_rate_reach_ema"):
        env.success_rate_reach_ema = torch.zeros(env.num_envs, device=env.device)
    
    # extract the command
    command = env.command_manager.get_command(command_name)
    # distance to goal in 2D plane
    dist = torch.norm(command[env_ids, :2], dim=1)
    
    # time left in episode
    t_left = (env.max_episode_length - env.episode_length_buf[env_ids]) * env.step_dt
    
    # success: within move_up_dist of the goal with less than 4 s left
    success_ = (dist < move_up_dist) & (t_left < 4.0)
    
    # EMA update: 0.8 * ema + 0.2 * success
    env.success_rate_reach_ema[env_ids] = 0.8 * env.success_rate_reach_ema[env_ids] + 0.2 * success_.float()

    move_up = (dist < move_up_dist) & (env.success_rate_reach_ema[env_ids] > 0.5)
    
    move_down = dist > move_down_dist

    # update terrain levels
    env.scene.terrain.update_env_origins(env_ids, move_up, move_down)

    # logging success rate
    extras = dict()
    extras["Log/SuccessRateReachEMA"] = torch.mean(env.success_rate_reach_ema).item()

    # per terrain logging
    if hasattr(env.scene.terrain, "terrain_types"):
        if not hasattr(env, "_ame2_terrain_type_masks"):
            terrain_types = env.scene.terrain.terrain_types
            unique_terrains = torch.unique(terrain_types)
            
            # map terrain column indices to sub-terrain names
            terrain_names = {}
            if hasattr(env.scene.terrain.cfg, "terrain_generator") and env.scene.terrain.cfg.terrain_generator is not None:
                sub_terrains = env.scene.terrain.cfg.terrain_generator.sub_terrains
                num_cols = env.scene.terrain.cfg.terrain_generator.num_cols
                proportions = np.array([s.proportion for s in sub_terrains.values()])
                proportions /= np.sum(proportions)
                names = list(sub_terrains.keys())
                
                sub_indices = []
                for index in range(num_cols):
                    sub_index = np.min(np.where(index / num_cols + 0.001 < np.cumsum(proportions))[0])
                    sub_indices.append(sub_index)
                
                for t_type in unique_terrains:
                    t_idx = int(t_type.item())
                    if t_idx < len(sub_indices):
                        terrain_names[t_idx] = names[sub_indices[t_idx]]
                    else:
                        terrain_names[t_idx] = f"Type_{t_idx}"
            else:
                for t_type in unique_terrains:
                    terrain_names[int(t_type.item())] = f"Type_{int(t_type.item())}"

            # group masks by name
            env._ame2_terrain_type_masks = {}
            for t_type in unique_terrains:
                name = terrain_names[int(t_type.item())]
                mask = (terrain_types == t_type)
                if name not in env._ame2_terrain_type_masks:
                    env._ame2_terrain_type_masks[name] = mask
                else:
                    env._ame2_terrain_type_masks[name] |= mask
        
        for t_name, t_mask in env._ame2_terrain_type_masks.items():
            t_avg_success = torch.mean(env.success_rate_reach_ema[t_mask]).item()
            extras[f"Log/SuccessRate/Terrain_{t_name}"] = t_avg_success
            t_avg_level = torch.mean(env.scene.terrain.terrain_levels[t_mask].float()).item()
            extras[f"Log/TerrainLevel/Terrain_{t_name}"] = t_avg_level
    
    if "log" in env.extras: 
        env.extras["log"].update(extras)

    # return mean terrain level
    return torch.mean(env.scene.terrain.terrain_levels.float()).item()

def ame2_optional_assistive_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    starting_force: float = 300.0,
    end_force: float = 0.0,
    starting_thr: float = 0.2,
    end_thr: float = 0.8,
    is_play: bool = False,
    play_value: float = 0.0,
    enable_latch: bool = False,
    share_force: bool = False,
) -> float:
    """Curriculum that sets a per-env upward assistive force from the success-rate EMA.

    The force (``env.assistive_force``) decreases linearly from ``starting_force`` to ``end_force`` as
    the success rate goes from ``starting_thr`` to ``end_thr``. It is applied by
    :func:`ame2_apply_assistive_force`. With ``share_force`` the mean success rate is used for all envs;
    with ``enable_latch`` an env that reaches ``end_thr`` keeps ``end_force``.
    """
    if not hasattr(env, "assistive_force"):
        env.assistive_force = torch.zeros(env.num_envs, device=env.device)
    
    if enable_latch and not hasattr(env, "assistive_force_latched"):
        env.assistive_force_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    if is_play:
        env.assistive_force.fill_(play_value)
        return play_value

    # success-rate EMA from ame2_terrain_levels_goalreaching
    if hasattr(env, "success_rate_reach_ema"):
        success_rate = env.success_rate_reach_ema
        if share_force:
            success_rate = torch.mean(success_rate).expand_as(success_rate)
    else:
        print("warning: success_rate_reach_ema not found, using 1.0 for force calculation")
        success_rate = torch.ones(env.num_envs, device=env.device)

    # linear interpolation, clamped at both thresholds
    progress = (success_rate - starting_thr) / (end_thr - starting_thr)
    progress = torch.clamp(progress, 0.0, 1.0)
    
    force_values = starting_force + (end_force - starting_force) * progress

    if enable_latch:
        # latch envs once success_rate >= end_thr
        env.assistive_force_latched |= (success_rate >= end_thr)
        force_values = torch.where(env.assistive_force_latched, end_force, force_values)

    env.assistive_force[:] = force_values

    # the mean force is logged via the return value
    
    if enable_latch:
        latch_rate = torch.mean(env.assistive_force_latched.float()).item()
        if "log" in env.extras:
            env.extras["log"].update({"Curriculum/AssistiveForceLatchRate": latch_rate})

    return torch.mean(env.assistive_force).item()
