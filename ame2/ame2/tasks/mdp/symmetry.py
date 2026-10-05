# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import TYPE_CHECKING
from rsl_rl.env import VecEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def ame2_g1_symmetry_augmentation(
    env: ManagerBasedRLEnv,
    obs: TensorDict | torch.Tensor,
    actions: torch.Tensor | None = None,
    for_critic: bool = False
) -> tuple[TensorDict | torch.Tensor, torch.Tensor | None]:
    """Left-right symmetry augmentation for the G1 observations and actions.

    Only the critic path is supported.

    Args:
        env: The environment.
        obs: The observations (TensorDict or Tensor).
        actions: The actions (Tensor).
        for_critic: Whether this is for the critic (includes height scan).
        
    Returns:
        A tuple of (augmented_obs, augmented_actions).
    """
    if not for_critic:
        assert False, "Symmetry augmentation for policy is not supported/enabled yet."

    if isinstance(obs, TensorDict):
        return _aug_tensordict(env, obs, actions, for_critic)
    else:
        return _aug_tensor(env, obs, actions, for_critic)

def _aug_tensordict(env, obs: TensorDict, actions: torch.Tensor | None, for_critic: bool):
    batch_size = obs.batch_size[0]
    num_envs = batch_size
    
    obs_aug_dict = {}
    for key in obs.keys():
        tensor = obs[key].clone()
        # first half original, second half mirrored
        aug_tensor = torch.zeros(batch_size * 2, *tensor.shape[1:], device=tensor.device)
        aug_tensor[:batch_size] = tensor
        
        if key == "policy" or (for_critic and key == "critic"):
            aug_tensor[batch_size:] = _mirror_obs_tensor(env, tensor, for_critic)
        else:
            aug_tensor[batch_size:] = tensor
            
        obs_aug_dict[key] = aug_tensor
        
    obs_aug = TensorDict(obs_aug_dict, batch_size=[batch_size * 2], device=obs.device)
    
    actions_aug = None
    if actions is not None:
        actions_aug = torch.zeros(batch_size * 2, *actions.shape[1:], device=actions.device)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _switch_joints_lr_g1(actions)
        
    return obs_aug, actions_aug

def _aug_tensor(env, obs: torch.Tensor, actions: torch.Tensor | None, for_critic: bool):
    batch_size = obs.shape[0]
    obs_aug = torch.zeros(batch_size * 2, *obs.shape[1:], device=obs.device)
    obs_aug[:batch_size] = obs
    obs_aug[batch_size:] = _mirror_obs_tensor(env, obs, for_critic)
    
    actions_aug = None
    if actions is not None:
        actions_aug = torch.zeros(batch_size * 2, *actions.shape[1:], device=actions.device)
        actions_aug[:batch_size] = actions
        actions_aug[batch_size:] = _switch_joints_lr_g1(actions)
        
    return obs_aug, actions_aug

def _switch_joints_lr_g1(dof: torch.Tensor) -> torch.Tensor:
    """Switches segments of the DOF tensor to their left-right counterparts.
    
    Joint Order:
    0: left_hip_pitch       1: right_hip_pitch
    2: waist_yaw            3: left_hip_roll
    4: right_hip_roll       5: waist_roll
    6: left_hip_yaw         7: right_hip_yaw
    8: waist_pitch          9: left_knee
    10: right_knee          11: left_shoulder_pitch
    12: right_shoulder_pitch 13: left_ankle_pitch
    14: right_ankle_pitch   15: left_shoulder_roll
    16: right_shoulder_roll 17: left_ankle_roll
    18: right_ankle_roll    19: left_shoulder_yaw
    20: right_shoulder_yaw  21: left_elbow
    22: right_elbow
    """
    dof_switched = dof.clone()

    # (0, 1): Hip Pitch
    dof_switched[..., 0] = dof[..., 1]
    dof_switched[..., 1] = dof[..., 0]

    # (3, 4): Hip Roll (FLIP)
    dof_switched[..., 3] = -dof[..., 4]
    dof_switched[..., 4] = -dof[..., 3]

    # (6, 7): Hip Yaw (FLIP)
    dof_switched[..., 6] = -dof[..., 7]
    dof_switched[..., 7] = -dof[..., 6]

    # (9, 10): Knee
    dof_switched[..., 9] = dof[..., 10]
    dof_switched[..., 10] = dof[..., 9]

    # (11, 12): Shoulder Pitch
    dof_switched[..., 11] = dof[..., 12]
    dof_switched[..., 12] = dof[..., 11]

    # (13, 14): Ankle Pitch
    dof_switched[..., 13] = dof[..., 14]
    dof_switched[..., 14] = dof[..., 13]

    # (15, 16): Shoulder Roll (FLIP)
    dof_switched[..., 15] = -dof[..., 16]
    dof_switched[..., 16] = -dof[..., 15]

    # (17, 18): Ankle Roll (FLIP)
    dof_switched[..., 17] = -dof[..., 18]
    dof_switched[..., 18] = -dof[..., 17]

    # (19, 20): Shoulder Yaw (FLIP)
    dof_switched[..., 19] = -dof[..., 20]
    dof_switched[..., 20] = -dof[..., 19]

    # (21, 22): Elbow
    dof_switched[..., 21] = dof[..., 22]
    dof_switched[..., 22] = dof[..., 21]

    # Waist: yaw and roll flip, pitch unchanged
    dof_switched[..., 2] = -dof[..., 2]
    dof_switched[..., 5] = -dof[..., 5]
    dof_switched[..., 8] =  dof[..., 8]

    assert dof.shape[-1] == 23, f"Expected 23 joints, got {dof.shape[-1]}"

    return dof_switched

def _mirror_obs_tensor(env: ManagerBasedRLEnv, obs: torch.Tensor, for_critic: bool) -> torch.Tensor:
    """Mirrors a flat observation tensor."""
    obs = obs.clone()
    device = obs.device
    
    # layout follows the observation term order in the env cfg
    # 1. Base State [lin_vel(3), ang_vel(3), gravity(3)]
    # lin_vel: [x, y, z] -> [x, -y, z]
    obs[:, 0:3] *= torch.tensor([1, -1, 1], device=device)
    # ang_vel: [wx, wy, wz] -> [-wx, wy, -wz]
    obs[:, 3:6] *= torch.tensor([-1, 1, -1], device=device)
    # gravity: [gx, gy, gz] -> [gx, -gy, gz]
    obs[:, 6:9] *= torch.tensor([1, -1, 1], device=device)
    
    idx = 9
    num_actions = 23 # G1 joints
    
    # 2. Joint State [pos_rel(23), vel_rel(23)]
    obs[:, idx : idx + num_actions] = _switch_joints_lr_g1(obs[:, idx : idx + num_actions])
    idx += num_actions
    obs[:, idx : idx + num_actions] = _switch_joints_lr_g1(obs[:, idx : idx + num_actions])
    idx += num_actions
    
    # 3. Actions [23]
    obs[:, idx : idx + num_actions] = _switch_joints_lr_g1(obs[:, idx : idx + num_actions])
    idx += num_actions
    
    assert idx == 9 + 3 * 23, f"Joint/Action index mismatch: {idx}"
    
    # 4. Commands [pos_x, pos_y, sin(yaw), cos(yaw)]
    obs[:, idx:idx+2] *= torch.tensor([1, -1], device=device) # pos_x, pos_y
    obs[:, idx+2] *= -1.0 # sin(yaw) -> sin(-yaw) = -sin(yaw)
    # cos(yaw) at idx+3 is unchanged
    idx += 4
    
    # 5. Time left (Scalar)
    idx += 1
    
    # 6. Contact states (26 links)
    # ['base', 'l_hip_p', 'r_hip_p', 'waist_y', 'l_hip_r', 'r_hip_r', 'l_hip_y', 'r_hip_y', 
    #  'torso', 'l_knee', 'r_knee', 'head', 'l_sh_p', 'r_sh_p', 'l_sh_r', 'r_sh_r', 
    #  'l_ank_r', 'r_ank_r', 'l_sh_y', 'r_sh_y', 'l_elb', 'r_elb', 'l_wr_p', 'r_wr_p', 'l_hand', 'r_hand']
    contacts = obs[:, idx : idx + 26].clone()
    swap_indices = [
        (1, 2), (4, 5), (6, 7), (9, 10), (12, 13), (14, 15), 
        (16, 17), (18, 19), (20, 21), (22, 23), (24, 25)
    ]
    for i, j in swap_indices:
        obs[:, idx + i] = contacts[:, j]
        obs[:, idx + j] = contacts[:, i]
    # base, waist_y, torso, head are unchanged
    idx += 26
    
    assert idx == 9 + 3 * 23 + 4 + 1 + 26, f"Contact index mismatch: {idx}"
    
    # 7. Height scan
    if for_critic:
        sensor = env.unwrapped.scene["height_scanner_critic"]
        pattern_cfg = sensor.cfg.pattern_cfg
        # size = [length, width], resolution = grid_step
        nx = int(round(pattern_cfg.size[0] / pattern_cfg.resolution)) + 1
        ny = int(round(pattern_cfg.size[1] / pattern_cfg.resolution)) + 1
        
        height_scan = obs[:, idx : idx + nx * ny].view(-1, nx, ny)
        # mirror along y (dim 2 of the (batch, nx, ny) tensor)
        height_scan = height_scan.flip(dims=[2])
        obs[:, idx : idx + nx * ny] = height_scan.view(-1, nx * ny)
        idx += nx * ny

    assert idx == obs.shape[1], f"Observation dimension mismatch: expected {idx}, got {obs.shape[1]}"
    
    return obs
