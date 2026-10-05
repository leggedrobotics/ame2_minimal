# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING
from collections.abc import Sequence

from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class Ame2DelayedJointPositionAction(JointPositionAction):
    """Joint position action term with a per-environment action delay sampled at reset."""

    cfg: Ame2DelayedJointPositionActionCfg

    def __init__(self, cfg: Ame2DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self.max_delay = cfg.delay_range[1]
        # action history, index 0 is the newest: (num_envs, max_delay + 1, action_dim)
        self.action_history = torch.zeros(
            (self.num_envs, self.max_delay + 1, self.action_dim), 
            device=self.device
        )
        # per-env delay in steps
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # sample delays in [delay_range[0], delay_range[1]] (randint's upper bound is exclusive)
        self.delay_steps[env_ids] = torch.randint(
            self.cfg.delay_range[0], 
            self.cfg.delay_range[1] + 1, 
            (len(env_ids),),
            device=self.device
        )
        
        # fill the history with the current processed actions
        initial_actions = self.processed_actions[env_ids].unsqueeze(1).repeat(1, self.max_delay + 1, 1)
        self.action_history[env_ids] = initial_actions

    def apply_actions(self):
        # shift history so that index 0 holds the newest action
        if self.max_delay > 0:
            self.action_history = torch.roll(self.action_history, shifts=1, dims=1)
            
        self.action_history[:, 0, :] = self.processed_actions
        
        # per-env delayed action: (num_envs, action_dim)
        delayed_actions = self.action_history[torch.arange(self.num_envs, device=self.device), self.delay_steps]
        
        self._asset.set_joint_position_target(delayed_actions, joint_ids=self._joint_ids)


@configclass
class Ame2DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for delayed joint position action term."""
    class_type: type = Ame2DelayedJointPositionAction
    delay_range: tuple[int, int] = (0, 1)
    """Range of delay steps (inclusive) sampled at reset."""
