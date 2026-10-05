# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.envs.mdp.commands import TerrainBasedPose2dCommand, TerrainBasedPose2dCommandCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class SafeTerrainBasedPose2dCommand(TerrainBasedPose2dCommand):
    """Terrain-based pose command that samples goals from flat patches inside a forward cone.

    Patches with ``|y| <= x * tan(cone_angle_deg) + 0.1`` relative to the env origin are preferred.
    If no patch lies inside the cone, any valid patch is used.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain the terrain levels and types for the environments
        levels = self.terrain.terrain_levels[env_ids]
        types = self.terrain.terrain_types[env_ids]
        origins = self.terrain.env_origins[env_ids]

        # sample new position targets from the terrain
        for i, env_id in enumerate(env_ids):
            level = levels[i]
            ttype = types[i]
            origin = origins[i]

            # get all valid patches for this environment's level and type
            patches = self.valid_targets[level, ttype]  # (num_patches, 3)

            # compute relative position to environment origin
            rel_pos = patches[:, :2] - origin[:2]  # (num_patches, 2)

            # forward cone |y| <= x * tan(cone_angle) + 0.1 (x forward, y lateral)
            tan_cone_angle = torch.tan(torch.tensor(self.cfg.cone_angle_deg * torch.pi / 180.0, device=self.device))
            mask = (rel_pos[:, 1].abs() <= rel_pos[:, 0] * tan_cone_angle + 0.1)
            safe_patches = patches[mask]

            if len(safe_patches) > 0:
                # randomly pick one from the safe patches
                idx = torch.randint(0, len(safe_patches), (1,), device=self.device)
                self.pos_command_w[env_id] = safe_patches[idx[0]]
            else:
                # fallback to regular sampling if no patch satisfies the cone
                idx = torch.randint(0, len(patches), (1,), device=self.device)
                self.pos_command_w[env_id] = patches[idx[0]]

        # offset the position command by the current root height
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)


@configclass
class SafeTerrainBasedPose2dCommandCfg(TerrainBasedPose2dCommandCfg):
    """Configuration for the safe terrain-based position command generator."""

    class_type: type = SafeTerrainBasedPose2dCommand

    cone_angle_deg: float = 45.0
    """The cone angle in degrees for sampling goals. Goals are sampled within +/- cone_angle_deg from the forward direction."""
