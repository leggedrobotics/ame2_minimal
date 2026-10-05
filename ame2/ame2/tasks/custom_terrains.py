# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import trimesh
from dataclasses import MISSING

import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass

@configclass
class ClimbUpCfg(SubTerrainBaseCfg):
    """Configuration for a custom climb_up terrain."""
    min_height: float = 0.5
    max_height: float = 1.0
    rough: bool = True  # perturb box tops by up to +-rough_rise * difficulty
    rough_rise: float = 0.05
    box_height_diff_max: float = 0.05
    wall_x: float = 1.0

def climb_up(difficulty: float, cfg: ClimbUpCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Climb-up terrain: lower ground, a row of 4-5 rotated boxes at ``wall_x``, then a raised plateau.

    The robot starts at the tile center on the lower ground. Box yaw (up to +-45 deg) and
    roll/pitch (~+-5 deg) scale with difficulty. Boxes stay inside the tile along y.
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    
    # Plateau height scales linearly with difficulty
    wall_x = cfg.wall_x
    nominal_height = cfg.min_height + (cfg.max_height - cfg.min_height) * difficulty
    max_noise = cfg.rough_rise * difficulty
    
    meshes = []
    
    # sub-terrain center (origin for pose commands)
    sc = np.array([tw / 2, tl / 2])
    
    # 1. Lower Ground Plane (x < wall_x)
    ground_height = 0.1
    # Extends up to the wall transition
    lg_w = sc[0] + wall_x
    lg_center_x = lg_w / 2
    lg_pose = np.eye(4)
    lg_pose[:3, 3] = [lg_center_x, sc[1], -ground_height / 2]
    lg_box = trimesh.creation.box([lg_w, tl, ground_height], transform=lg_pose)
    meshes.append(lg_box)
    
    # 2. Raised Plateau (x > wall_x) - No Roughness
    p_w = tw - lg_w
    p_center_x = lg_w + p_w / 2
    p_pose = np.eye(4)
    p_pose[:3, 3] = [p_center_x, sc[1], nominal_height / 2]
    plateau = trimesh.creation.box([p_w, tl, nominal_height], transform=p_pose)
    meshes.append(plateau)
    
    # 3. Wall Boxes (Contained Jags)
    num_boxes = np.random.randint(4, 6)
    
    # Margin to prevent box corners from invading adjacent envs (~max_box_dim/2 + rotation buffer)
    margin = 1.2 
    y_range = tl - 2 * margin
    y_step = y_range / max(1, num_boxes - 1)
    
    for i in range(num_boxes):
        # Box height within +-5 cm of the plateau height
        bh = nominal_height + np.random.uniform(-0.05, 0.05)
        # depth and length
        bw = np.random.uniform(0.8, 1.6) 
        bl = y_step * 1.2 # neighbouring boxes overlap along y
        
        # Position centered on wall_x
        cy = margin + i * y_step + np.random.uniform(-0.2, 0.2)
        cx = lg_w
        cz = bh / 2
        
        # Yaw up to +-45 deg, roll/pitch up to ~+-5 deg (pitch x1.5), scaled by difficulty
        yaw = np.random.uniform(-np.pi/4, np.pi/4) * difficulty
        roll = np.random.uniform(-0.087, 0.087) * difficulty
        pitch = np.random.uniform(-0.087, 0.087) * difficulty * 1.5
        
        pose = trimesh.transformations.euler_matrix(roll, pitch, yaw)
        pose[:3, 3] = [cx, cy, cz]
        
        box = trimesh.creation.box((bw, bl, bh), transform=pose)
        if cfg.rough:
            box = box.subdivide().subdivide().subdivide()
            top_threshold = bh * 0.9
            top_indices = np.where(box.vertices[:, 2] > top_threshold)[0]
            if len(top_indices) > 0:
                box.vertices[top_indices, 2] += np.random.uniform(-max_noise, max_noise, size=len(top_indices))
        meshes.append(box)
        
    origin = np.array([sc[0], sc[1], 0.0])
    return meshes, origin
@configclass
class RoughTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a custom rough terrain."""
    # These are max values at difficulty 1.0
    max_slope_deg: float = 20.0
    max_roughness_m: float = 0.16
    undulation_res: float = 2.0  # Resolution of low-frequency undulations in meters

@height_field_to_mesh
def rough_terrain(difficulty: float, cfg: RoughTerrainCfg) -> np.ndarray:
    """Custom rough terrain with slopes and noise conditioned on difficulty."""
    # 1. Resolve parameters based on difficulty
    target_slope_deg = difficulty * cfg.max_slope_deg
    target_roughness = difficulty * cfg.max_roughness_m
    
    # Undulation height range: tan(slope) * undulation_res
    slope_tan = np.tan(np.deg2rad(target_slope_deg))
    undulation_height_range = slope_tan * cfg.undulation_res
    
    # 2. Dimensions
    num_rows = int(cfg.size[0] / cfg.horizontal_scale)
    num_cols = int(cfg.size[1] / cfg.horizontal_scale)
    
    # 3. Generate Low-Frequency Undulations (Slopes)
    if target_slope_deg > 1e-3:
        # Sample points every undulation_res meters
        res_rows = int(cfg.size[0] / cfg.undulation_res) + 1
        res_cols = int(cfg.size[1] / cfg.undulation_res) + 1
        
        # Random heights for undulations
        low_res_hf = np.random.uniform(-undulation_height_range/2, undulation_height_range/2, (res_rows, res_cols))
        
        # Interpolate to full resolution
        x = np.linspace(0, cfg.size[0], res_rows)
        y = np.linspace(0, cfg.size[1], res_cols)
        func = interpolate.RectBivariateSpline(x, y, low_res_hf)
        
        x_new = np.linspace(0, cfg.size[0], num_rows)
        y_new = np.linspace(0, cfg.size[1], num_cols)
        hf_undulations = func(x_new, y_new)
    else:
        hf_undulations = np.zeros((num_rows, num_cols))
        
    # 4. Generate High-Frequency Roughness (Noise)
    if target_roughness > 1e-4:
        hf_noise = np.random.uniform(-target_roughness/2, target_roughness/2, (num_rows, num_cols))
    else:
        hf_noise = np.zeros((num_rows, num_cols))
        
    # 5. Combine and Discretize
    hf_total = hf_undulations + hf_noise
    
    return np.rint(hf_total / cfg.vertical_scale).astype(np.int16)

@configclass
class PyramidStairsCfg(SubTerrainBaseCfg):
    """Configuration for a pyramid stair mesh terrain ported from another simulator."""
    min_step_height: float = 0.05
    max_step_height: float = 0.23
    step_width: tuple[float, float] = (0.27, 0.35)
    border_size: float = 1.4
    platform_size: float = 2.0
    passage_width: tuple[float, float] = (0.8, 1.8)
    ground_height: tuple[float, float] = (-1.5, -0.0)
    holes: bool = True
    rand_walls: bool = True
    wall_h: tuple[float, float] = (0.4, 1.8)
    wall_prob: float = 0.35
    wall_thick: tuple[float, float] = (0.05, 0.2)

@configclass
class PyramidStairsInvCfg(PyramidStairsCfg):
    """Configuration for an inverted pyramid stair mesh terrain ported from another simulator."""

def pyramid_stairs(difficulty: float, cfg: PyramidStairsCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a pyramid stair mesh terrain (ported)."""
    step_height = cfg.min_step_height + (cfg.max_step_height - cfg.min_step_height) * difficulty
    step_width = np.random.uniform(cfg.step_width[0], cfg.step_width[1])
    
    tl, tw = cfg.size
    num_steps = (
        min(tl - 2 * cfg.border_size - cfg.platform_size, tw - 2 * cfg.border_size - cfg.platform_size)
        // (2 * step_width)
        + 1
    )

    pos = [0.5 * tl, 0.5 * tw, 0.0]
    meshes = []
    
    # Border
    border_meshes = make_border(
        (tl, tw),
        (tl - 2 * cfg.border_size, tw - 2 * cfg.border_size),
        step_height,
        [pos[0], pos[1], -step_height / 2],
    )
    meshes += border_meshes

    # Ground
    ground_h = np.random.uniform(cfg.ground_height[0], cfg.ground_height[1])
    dim = [tl, tw, 0.1]
    pose = np.eye(4)
    pose[:3, -1] = [pos[0], pos[1], -0.05 + ground_h]

    env_length = tl - 2 * cfg.border_size
    env_width = tw - 2 * cfg.border_size
    
    for k in range(int(num_steps)):
        if cfg.holes:
            box_length = np.random.uniform(cfg.passage_width[0], cfg.passage_width[1])
            box_width = np.random.uniform(cfg.passage_width[0], cfg.passage_width[1])
        else:
            box_length = env_length - k * 2 * step_width
            box_width = env_width - k * 2 * step_width

        # North/South
        dims_ns = [box_length, step_width + np.random.uniform(-0.02, 0.02), 0.02]
        # North
        pose_n = np.eye(4)
        pose_n[:3, -1] = [pos[0], pos[1] + env_width / 2 - (k + 0.5) * step_width, pos[2] + (k + 1) * step_height - 0.01]
        meshes.append(trimesh.creation.box(dims_ns, pose_n))
        # South
        pose_s = np.eye(4)
        pose_s[:3, -1] = [pos[0], pos[1] - env_width / 2 + (k + 0.5) * step_width, pos[2] + (k + 1) * step_height - 0.01]
        meshes.append(trimesh.creation.box(dims_ns, pose_s))

        # East/West
        dims_ew = [step_width + np.random.uniform(-0.02, 0.02), box_width, 0.02]
        # East
        pose_e = np.eye(4)
        pose_e[:3, -1] = [pos[0] - env_length / 2 + (k + 0.5) * step_width, pos[1], pos[2] + (k + 1) * step_height - 0.01]
        meshes.append(trimesh.creation.box(dims_ew, pose_e))
        # West
        pose_w = np.eye(4)
        pose_w[:3, -1] = [pos[0] + env_length / 2 - (k + 0.5) * step_width, pos[1], pos[2] + (k + 1) * step_height - 0.01]
        meshes.append(trimesh.creation.box(dims_ew, pose_w))

        if cfg.rand_walls:
            # Random walls
            for side in [1, -1]: # Left/Right (relative to North/South direction)
                if np.random.uniform(0.0, 1.0) < cfg.wall_prob:
                    wh = np.random.uniform(cfg.wall_h[0], cfg.wall_h[1])
                    wt = np.random.uniform(cfg.wall_thick[0], cfg.wall_thick[1])
                    dims_wall = [step_width, wt, wh]
                    pose_wall = np.eye(4)
                    pose_wall[:3, -1] = [
                        pos[0] + env_length / 2 - (k + 0.5) * step_width,
                        pos[1] + side * (box_width / 2 + wt / 2),
                        pos[2] + (k + 1) * step_height + wh / 2
                    ]
                    meshes.append(trimesh.creation.box(dims_wall, pose_wall))

    # Middle platform
    curr_k = int(num_steps) - 1
    dims_mid = [env_length - 2 * (curr_k + 1) * step_width, env_width - 2 * (curr_k + 1) * step_width, (curr_k + 2) * step_height]
    pose_mid = np.eye(4)
    pose_mid[:3, -1] = [pos[0], pos[1], pos[2] + curr_k * step_height / 2]
    meshes.append(trimesh.creation.box(dims_mid, pose_mid))

    origin = np.array([pos[0], pos[1], (num_steps + 1) * step_height])
    return meshes, origin

def pyramid_stairs_inv(difficulty: float, cfg: PyramidStairsInvCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate an inverted pyramid stair mesh terrain (ported)."""
    step_height = cfg.min_step_height + (cfg.max_step_height - cfg.min_step_height) * difficulty
    step_width = np.random.uniform(cfg.step_width[0], cfg.step_width[1])
    
    tl, tw = cfg.size
    num_steps = (
        min(tl - 2 * cfg.border_size - cfg.platform_size, tw - 2 * cfg.border_size - cfg.platform_size)
        // (2 * step_width)
        + 2
    )

    pos = [0.5 * tl, 0.5 * tw, 0.0]
    tot_height = num_steps * step_height
    meshes = []
    
    # Border
    border_meshes = make_border(
        (tl, tw),
        (tl - 2 * cfg.border_size, tw - 2 * cfg.border_size),
        tot_height,
        [pos[0], pos[1], -tot_height / 2],
    )
    meshes += border_meshes

    # Ground
    ground_h = np.random.uniform(cfg.ground_height[0], cfg.ground_height[1])
    dim = [tl, tw, 0.1]
    pose = np.eye(4)
    pose[:3, -1] = [pos[0], pos[1], -0.05 + ground_h - tot_height]

    env_length = tl - 2 * cfg.border_size
    env_width = tw - 2 * cfg.border_size
    
    for k in range(int(num_steps) - 1):
        if cfg.holes:
            box_length = np.random.uniform(cfg.passage_width[0], cfg.passage_width[1])
            box_width = np.random.uniform(cfg.passage_width[0], cfg.passage_width[1])
        else:
            box_length = env_length - k * 2 * step_width
            box_width = env_width - k * 2 * step_width

        # Randomize thickness from 0.02 to step_height
        thickness = np.random.uniform(0.02, step_height)
        target_top = pos[2] - (k + 1) * step_height

        # North/South
        dims_ns = [box_length, step_width + np.random.uniform(-0.02, 0.02), thickness]
        # North
        pose_n = np.eye(4)
        pose_n[:3, -1] = [pos[0], pos[1] + env_width / 2 - (k + 0.5) * step_width, target_top - thickness / 2]
        meshes.append(trimesh.creation.box(dims_ns, pose_n))
        # South
        pose_s = np.eye(4)
        pose_s[:3, -1] = [pos[0], pos[1] - env_width / 2 + (k + 0.5) * step_width, target_top - thickness / 2]
        meshes.append(trimesh.creation.box(dims_ns, pose_s))

        # East/West
        dims_ew = [step_width + np.random.uniform(-0.02, 0.02), box_width, thickness]
        # East
        pose_e = np.eye(4)
        pose_e[:3, -1] = [pos[0] - env_length / 2 + (k + 0.5) * step_width, pos[1], target_top - thickness / 2]
        meshes.append(trimesh.creation.box(dims_ew, pose_e))
        # West
        pose_w = np.eye(4)
        pose_w[:3, -1] = [pos[0] + env_length / 2 - (k + 0.5) * step_width, pos[1], target_top - thickness / 2]
        meshes.append(trimesh.creation.box(dims_ew, pose_w))

        if cfg.rand_walls:
            # Random walls
            for side in [1, -1]:
                if np.random.uniform(0.0, 1.0) < cfg.wall_prob:
                    wh = np.random.uniform(cfg.wall_h[0], cfg.wall_h[1])
                    wt = np.random.uniform(cfg.wall_thick[0], cfg.wall_thick[1])
                    dims_wall = [step_width, wt, wh]
                    pose_wall = np.eye(4)
                    pose_wall[:3, -1] = [
                        pos[0] + env_length / 2 - (k + 0.5) * step_width,
                        pos[1] + side * (box_width / 2 + wt / 2),
                        pos[2] - (k + 1) * step_height + wh / 2
                    ]
                    meshes.append(trimesh.creation.box(dims_wall, pose_wall))

    # Middle platform
    curr_k = int(num_steps) - 1
    dims_mid = [cfg.platform_size, cfg.platform_size, tot_height - (curr_k + 1) * step_height]
    pose_mid = np.eye(4)
    pose_mid[:3, -1] = [pos[0], pos[1], pos[2] - tot_height / 2 - (curr_k + 1) * step_height / 2]
    meshes.append(trimesh.creation.box(dims_mid, pose_mid))

    origin = np.array([pos[0], pos[1], -(num_steps - 1) * step_height])
    return meshes, origin

@configclass
class BoxesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a custom boxes height field terrain."""
    num_boxes: int = 20
    box_size_range: tuple[float, float] = (0.4, 3.0)
    min_max_height: float = 0.05
    max_max_height: float = 0.3
    platform_size: float = 2.0  # Initial safe platform in the center
    dilation_size: float = 0.1  # Dilation to avoid narrow walls/gaps

@height_field_to_mesh
def boxes_terrain(difficulty: float, cfg: BoxesTerrainCfg) -> np.ndarray:
    """Generate a custom boxes height field terrain."""
    # 1. Dimensions
    num_rows = int(cfg.size[0] / cfg.horizontal_scale)
    num_cols = int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((num_rows, num_cols))
    
    # 2. Parameters
    max_h = cfg.min_max_height + (cfg.max_max_height - cfg.min_max_height) * difficulty
    
    # 3. Add/Subtract boxes
    for _ in range(cfg.num_boxes):
        # Random box size
        bw_m = np.random.uniform(cfg.box_size_range[0], cfg.box_size_range[1])
        bl_m = np.random.uniform(cfg.box_size_range[0], cfg.box_size_range[1])
        
        bw = max(1, int(bw_m / cfg.horizontal_scale))
        bl = max(1, int(bl_m / cfg.horizontal_scale))
        
        # Random position
        if num_rows - bw <= 0 or num_cols - bl <= 0:
            continue
        r = np.random.randint(0, num_rows - bw)
        c = np.random.randint(0, num_cols - bl)
        
        # Random height
        h = np.random.uniform(-max_h, max_h)
        
        # Apply box
        hf[r:r+bw, c:c+bl] += h
        
    # 4. Apply dilation to avoid narrow walls/gaps (0-cm walls)
    if cfg.dilation_size > 0:
        d_size = max(1, int(cfg.dilation_size / cfg.horizontal_scale))
        hf_pos = np.maximum(hf, 0)
        hf_neg = np.minimum(hf, 0)
        # Dilate positive parts, erode negative parts
        hf_pos = ndimage.grey_dilation(hf_pos, size=(d_size, d_size))
        hf_neg = ndimage.grey_erosion(hf_neg, size=(d_size, d_size))
        hf = hf_pos + hf_neg
        
    # 5. Clear platform (center) - This is the initial safe platform
    p_rows = int(cfg.platform_size / cfg.horizontal_scale)
    p_cols = int(cfg.platform_size / cfg.horizontal_scale)
    r_start = (num_rows - p_rows) // 2
    c_start = (num_cols - p_cols) // 2
    hf[r_start:r_start+p_rows, c_start:c_start+p_cols] = 0.0
    
    return np.rint(hf / cfg.vertical_scale).astype(np.int16)

@configclass
class ObstaclesTerrainCfg(HfTerrainBaseCfg):
    """Configuration for a custom obstacles height field terrain."""
    max_slope_deg: float = 20.0
    undulation_res: float = 3.0
    max_obstacle_density: float = 0.5  # m^-2 at difficulty 1.0
    obstacle_height_range: tuple[float, float] = (0.1, 0.5)
    obstacle_size_range: tuple[float, float] = (0.3, 0.8)
    platform_size: float = 2.0
    dilation_size: float = 0.1  # Dilation for obstacles

@height_field_to_mesh
def obstacles_terrain(difficulty: float, cfg: ObstaclesTerrainCfg) -> np.ndarray:
    """Generate a custom obstacles height field terrain."""
    # 1. Resolve parameters
    num_rows = int(cfg.size[0] / cfg.horizontal_scale)
    num_cols = int(cfg.size[1] / cfg.horizontal_scale)
    
    # 2. Base Slopes (Low-Frequency Undulations)
    target_slope_deg = difficulty * cfg.max_slope_deg
    slope_tan = np.tan(np.deg2rad(target_slope_deg))
    undulation_height_range = slope_tan * cfg.undulation_res
    
    if target_slope_deg > 1e-3:
        res_rows = int(cfg.size[0] / cfg.undulation_res) + 1
        res_cols = int(cfg.size[1] / cfg.undulation_res) + 1
        low_res_hf = np.random.uniform(-undulation_height_range/2, undulation_height_range/2, (res_rows, res_cols))
        x = np.linspace(0, cfg.size[0], res_rows)
        y = np.linspace(0, cfg.size[1], res_cols)
        func = interpolate.RectBivariateSpline(x, y, low_res_hf)
        x_new = np.linspace(0, cfg.size[0], num_rows)
        y_new = np.linspace(0, cfg.size[1], num_cols)
        hf = func(x_new, y_new)
    else:
        hf = np.zeros((num_rows, num_cols))
        
    # 3. Obstacles
    target_density = difficulty * cfg.max_obstacle_density
    area = cfg.size[0] * cfg.size[1]
    num_obstacles = int(target_density * area)
    hf_obs = np.zeros_like(hf)
    
    # Obstacle-free center bounds
    p_rows = int(cfg.platform_size / cfg.horizontal_scale)
    p_cols = int(cfg.platform_size / cfg.horizontal_scale)
    r_start = (num_rows - p_rows) // 2
    r_end = r_start + p_rows
    c_start = (num_cols - p_cols) // 2
    c_end = c_start + p_cols

    for _ in range(num_obstacles):
        # Random size
        bw_m = np.random.uniform(cfg.obstacle_size_range[0], cfg.obstacle_size_range[1])
        bl_m = np.random.uniform(cfg.obstacle_size_range[0], cfg.obstacle_size_range[1])
        bw = max(1, int(bw_m / cfg.horizontal_scale))
        bl = max(1, int(bl_m / cfg.horizontal_scale))
        
        # Random position
        if num_rows - bw <= 0 or num_cols - bl <= 0:
            continue
        r = np.random.randint(0, num_rows - bw)
        c = np.random.randint(0, num_cols - bl)
        
        # Check overlap with center obstacle-free zone
        if r < r_end and r + bw > r_start and c < c_end and c + bl > c_start:
            continue

        # Random height
        h = np.random.uniform(cfg.obstacle_height_range[0], cfg.obstacle_height_range[1])
        if np.random.rand() > 0.5:
            h *= -1
            
        hf_obs[r:r+bw, c:c+bl] += h
        
    # 4. Apply dilation to obstacles
    if cfg.dilation_size > 0:
        d_size = max(1, int(cfg.dilation_size / cfg.horizontal_scale))
        hf_obs_pos = np.maximum(hf_obs, 0)
        hf_obs_neg = np.minimum(hf_obs, 0)
        hf_obs_pos = ndimage.grey_dilation(hf_obs_pos, size=(d_size, d_size))
        hf_obs_neg = ndimage.grey_erosion(hf_obs_neg, size=(d_size, d_size))
        hf_obs = hf_obs_pos + hf_obs_neg

    # 5. Combine slopes and dilated obstacles
    hf += hf_obs
    
    return np.rint(hf / cfg.vertical_scale).astype(np.int16)

@configclass
class ClimbDownCfg(ClimbUpCfg):
    """Configuration for a custom climb_down terrain."""

def climb_down(difficulty: float, cfg: ClimbDownCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Climb-down terrain: raised plateau, a row of 4-5 rotated boxes at ``wall_x``, then lower ground.

    The robot starts at the tile center on the raised plateau.
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    
    # Plateau height scales linearly with difficulty
    wall_x = cfg.wall_x
    nominal_height = cfg.min_height + (cfg.max_height - cfg.min_height) * difficulty
    max_noise = cfg.rough_rise * difficulty
    
    meshes = []
    
    # sub-terrain center (origin for pose commands)
    sc = np.array([tw / 2, tl / 2])
    
    # 1. Higher Start Plateau (x < wall_x)
    hp_w = sc[0] + wall_x
    hp_center_x = hp_w / 2
    hp_pose = np.eye(4)
    hp_pose[:3, 3] = [hp_center_x, sc[1], nominal_height / 2]
    plateau = trimesh.creation.box([hp_w, tl, nominal_height], transform=hp_pose)
    meshes.append(plateau)
    
    # 2. Lower End Ground (x > wall_x)
    lg_height = 0.1
    lg_w = tw - hp_w
    lg_center_x = hp_w + lg_w / 2
    lg_pose = np.eye(4)
    lg_pose[:3, 3] = [lg_center_x, sc[1], -lg_height / 2] # underground box for base
    lg_box = trimesh.creation.box([lg_w, tl, lg_height], transform=lg_pose)
    meshes.append(lg_box)
    
    # 3. Wall Boxes (Contained Jags)
    num_boxes = np.random.randint(4, 6)
    
    margin = 1.2 
    y_range = tl - 2 * margin
    y_step = y_range / max(1, num_boxes - 1)
    
    for i in range(num_boxes):
        # Boxes at the transition height
        bh = nominal_height + np.random.uniform(-0.05, 0.05)
        # depth and length
        bw = np.random.uniform(0.8, 1.6) 
        bl = y_step * 1.2
        
        # Position centered on wall_x
        cy = margin + i * y_step + np.random.uniform(-0.2, 0.2)
        cx = hp_w
        cz = bh / 2
        
        yaw = np.random.uniform(-np.pi/4, np.pi/4) * difficulty
        roll = np.random.uniform(-0.087, 0.087) * difficulty
        pitch = np.random.uniform(-0.087, 0.087) * difficulty * 1.5
        
        pose = trimesh.transformations.euler_matrix(roll, pitch, yaw)
        pose[:3, 3] = [cx, cy, cz]
        
        box = trimesh.creation.box((bw, bl, bh), transform=pose)
        if cfg.rough:
            box = box.subdivide().subdivide().subdivide()
            top_threshold = bh * 0.9
            top_indices = np.where(box.vertices[:, 2] > top_threshold)[0]
            if len(top_indices) > 0:
                box.vertices[top_indices, 2] += np.random.uniform(-max_noise, max_noise, size=len(top_indices))
        meshes.append(box)
        
    origin = np.array([sc[0], sc[1], nominal_height])
    return meshes, origin

@configclass
class ClimbingConsecutiveCfg(SubTerrainBaseCfg):
    """Configuration for a climbing consecutive hill."""
    min_height: float = 0.05
    max_height_lower: float = 0.4
    max_height_upper: float = 0.5
    rough: bool = False
    wall_offsets: tuple[float, float, float, float] = (0.8, 1.7, 2.2, 3.1)

@configclass
class GapTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a custom gap terrain."""
    min_gap: float = 0.1
    max_gap: float = 0.5
    platform_height: float = 0.5
    gap_x: float = 0.0 # Relative to center sc[0]
    floor_ratio: float = 0.5
    floor_height: tuple[float, float] = (-1.5, -0.65)

@configclass
class BalanceBeamCfg(SubTerrainBaseCfg):
    """Configuration for a custom balance beam terrain."""
    gap_size: float = 2.0
    gap_x: float = 0.0 # Relative to center sc[0]
    beam_width: list[float] = [0.5, 0.17]
    platform_height: float = 2.5 # platform thickness (m)
    max_rpy: float = 0.087
    floor_ratio: float = 0.5
    floor_height: tuple[float, float] = (-1.5, -0.65)

@configclass
class StonesCfg(SubTerrainBaseCfg):
    """Configuration for a custom stones terrain."""
    min_box_size: float = 0.35
    max_box_size: float = 0.85
    min_box_height: float = 0.01
    max_box_height: float = 0.06
    min_gap_size: float = 0.04
    max_gap_size: float = 0.12
    platform_size: float = 1.4
    boundary_size: float = 2.0
    higher_flat: bool = True
    flat_rise: tuple[float, float] = (-0.05, 0.15)
    floor_ratio: float = 0.5
    floor_height: tuple[float, float] = (-1.5, -0.65)
    sink_ratio: float = 0.06
    rise_ratio: float = 0.04

@configclass
class PalletZcCfg(SubTerrainBaseCfg):
    """Configuration for a custom pallets terrain."""
    platform_size: float = 2.0
    border_size: float = 1.5
    step_height: list[float] = [0.0, 0.3]
    step_width: list[float] = [0.4, 0.16]
    gap_size: list[float] = [0.08, 0.35]
    floor_ratio: float = 0.5
    floor_height: tuple[float, float] = (-1.5, -0.65)
    target_range_x: tuple[float, float] = (4.0, 4.01)
    target_range_y: tuple[float, float] = (0.0, 0.01)

@configclass
class SteppingStonesCylindersCfg(SubTerrainBaseCfg):
    """Configuration for a custom stepping stones cylinders terrain."""
    radius: float = 0.15
    min_init_pos_height: float = 0.5
    max_rand_height: float = 0.1
    @configclass
    class CurriculumParams:
        height: float = 2.0
        num: int = 1000
        rot: float = 10.0
    cp0: CurriculumParams = CurriculumParams(height=2.0, num=1000, rot=10.0)
    cp1: CurriculumParams = CurriculumParams(height=2.0, num=1000, rot=10.0)
    platform_size: float = 1.6
    border_size: float = 1.0
    target_range_x: tuple[float, float] = (5.65, 5.66)
    target_range_y: tuple[float, float] = (0.7, 0.71)

@configclass
class MixSparseClimbCfg(SubTerrainBaseCfg):
    """Configuration for a custom stone row terrain."""
    stone_size: float = 0.2
    gap_size: float = 0.2
    platform_size: float = 1.65
    border_size: float = 3.5 
    end_height: float = 0.7
    target_range_x: tuple[float, float] = (4.5, 4.51)
    target_range_y: tuple[float, float] = (-0.02, 0.02)

@configclass
class PitTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a custom pit terrain."""
    pit_width: float = 1.2
    pit_depth: float = 0.8

@configclass
class PeakTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a custom peak terrain (a single box; difficulty agnostic)."""
    peak_width: float = 1.2
    peak_height: float = 1.0

def climbing_consecutive(difficulty: float, cfg: ClimbingConsecutiveCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom climbing_consecutive terrain function.
    
    Creates four sequential transitions (Up-Up-Down-Down) relative to the center.
    Wall 1: Climb UP to h1 (lower layer max)
    Wall 2: Climb UP to h1+h2 (upper layer max)
    Wall 3: Climb DOWN to h1 (upper layer max)
    Wall 4: Climb DOWN to Ground (lower layer max)
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    sc = np.array(cfg.size) / 2 # center offset
    
    meshes = []
    
    # Scale heights by difficulty
    h_min = cfg.min_height
    h_lower = h_min + (cfg.max_height_lower - h_min) * difficulty
    h_upper = h_min + (cfg.max_height_upper - h_min) * difficulty
    
    # Wall positions (relative to origin)
    x1, x2, x3, x4 = [sc[0] + dx for dx in cfg.wall_offsets]
    
    # Heights
    h1 = h_lower
    h_total = h_upper + h_lower
    
    # 1. Base Plateaus
    # Region 1 (Start to x1): Ground 0.0 (Already handled by underground base)
    
    # Region 2 (x1 to x2): Height h1
    p1_w = x2 - x1
    p1_pose = np.eye(4)
    p1_pose[:3, 3] = [x1 + p1_w / 2, sc[1], h1 / 2]
    meshes.append(trimesh.creation.box([p1_w, tl, h1], transform=p1_pose))
    
    # Region 3 (x2 to x3): Peak Height h_total
    peak_w = x3 - x2
    peak_pose = np.eye(4)
    peak_pose[:3, 3] = [x2 + peak_w / 2, sc[1], h_total / 2]
    meshes.append(trimesh.creation.box([peak_w, tl, h_total], transform=peak_pose))
    
    # Region 4 (x3 to x4): Height h1
    p2_w = x4 - x3
    p2_pose = np.eye(4)
    p2_pose[:3, 3] = [x3 + p2_w / 2, sc[1], h1 / 2]
    meshes.append(trimesh.creation.box([p2_w, tl, h1], transform=p2_pose))
    
    # Underground base for start/end
    base_pose = np.eye(4)
    base_pose[:3, 3] = [tw / 2, tl / 2, -0.05]
    meshes.append(trimesh.creation.box([tw, tl, 0.1], transform=base_pose))
    
    # 2. Transition Walls (Jags)
    walls = [
        (x1, h1),      # Wall 1: Up to h1
        (x2, h_total), # Wall 2: Up to h_total
        (x3, h_total), # Wall 3: Down from h_total
        (x4, h1)       # Wall 4: Down from h1
    ]
    
    margin = 1.2
    y_range = tl - 2 * margin
    
    for wall_idx, (wall_x, wall_h) in enumerate(walls):
        num_boxes = np.random.randint(8, 12)
        y_step = y_range / max(1, num_boxes - 1)
        
        for i in range(num_boxes):
            # Randomly skip ~30% of boxes
            if np.random.rand() < 0.3:
                continue
                
            # Box height within +-5 cm of the wall height
            bh = wall_h + np.random.uniform(-0.05, 0.05)
            
            bw = 0.5
            bl = 0.5
            
            # Independent shift in Y
            cy = margin + i * y_step + np.random.uniform(-0.35, 0.35)
            cz = bh / 2
            
            # Independent shift in X
            cx = wall_x + np.random.uniform(-0.1, 0.1)
            
            yaw = np.pi / 4 # 45 degrees for diamond pattern
            roll = 0.0
            pitch = 0.0
            
            pose = trimesh.transformations.euler_matrix(roll, pitch, yaw)
            pose[:3, 3] = [cx, cy, cz]
            
            box = trimesh.creation.box((bw, bl, bh), transform=pose)
            meshes.append(box)
            
    # Robot starts in the center
    origin = np.array([sc[0], sc[1], 0.0])
    return meshes, origin

def gap_terrain(difficulty: float, cfg: GapTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom gap terrain function.
    
    Creates two platforms separated by a gap.
    Gap width scales from min_gap to max_gap with difficulty.
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    sc = np.array(cfg.size) / 2
    
    meshes = []
    
    # Calculate gap width
    gap_width = cfg.min_gap + (cfg.max_gap - cfg.min_gap) * difficulty
    
    ph = cfg.platform_height
    
    # 1. First Platform (Start side)
    p1_w = sc[0] + cfg.gap_x
    p1_pose = np.eye(4)
    p1_pose[:3, 3] = [p1_w / 2, sc[1], -ph / 2]
    meshes.append(trimesh.creation.box([p1_w, tl, ph], transform=p1_pose))
    
    # 2. Second Platform (Goal side)
    delta_h = np.random.uniform(-0.25 * gap_width, 0.25 * gap_width)
    p2_start_x = p1_w + gap_width
    p2_w = tw - p2_start_x
    p2_pose = np.eye(4)
    p2_pose[:3, 3] = [p2_start_x + p2_w / 2, sc[1], delta_h - ph / 2]
    meshes.append(trimesh.creation.box([p2_w, tl, ph], transform=p2_pose))
    
    # 3. Optional Floor below the gap
    if np.random.uniform(0, 1) < cfg.floor_ratio:
        z_floor = np.random.uniform(cfg.floor_height[0], cfg.floor_height[1])
        floor_thick = 0.1
        floor_pose = np.eye(4)
        floor_pose[:3, 3] = [tw / 2, tl / 2, z_floor - floor_thick / 2]
        meshes.append(trimesh.creation.box([tw, tl, floor_thick], transform=floor_pose))

    # Spawn in the center of the first platform
    origin = np.array([sc[0], sc[1], 0])
    return meshes, origin

def balance_beam_terrain(difficulty: float, cfg: BalanceBeamCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom balance beam terrain function.
    
    Creates two platforms separated by a gap, bridged by a beam.
    Beam width scales with difficulty.
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    sc = np.array(cfg.size) / 2
    
    meshes = []
    
    gap_width = cfg.gap_size
    bw = cfg.beam_width[0] + (cfg.beam_width[1] - cfg.beam_width[0]) * difficulty
    ph = cfg.platform_height
    
    # 1. First Platform (Start side)
    p1_w = sc[0] + cfg.gap_x
    p1_pose = np.eye(4)
    p1_pose[:3, 3] = [p1_w / 2, sc[1], -ph / 2]
    meshes.append(trimesh.creation.box([p1_w, tl, ph], transform=p1_pose))
    
    # 2. Second Platform (Goal side)
    delta_h = np.random.uniform(-0.05 * difficulty, 0.05 * difficulty)
    p2_start_x = p1_w + gap_width
    p2_w = tw - p2_start_x
    p2_pose = np.eye(4)
    p2_pose[:3, 3] = [p2_start_x + p2_w / 2, sc[1], delta_h - ph / 2]
    meshes.append(trimesh.creation.box([p2_w, tl, ph], transform=p2_pose))
    
    # 3. Balance Beam (bridging the gap)
    max_rpy = cfg.max_rpy * difficulty
    roll = np.random.uniform(-max_rpy, max_rpy)
    pitch = np.random.uniform(-max_rpy, max_rpy)
    yaw = np.random.uniform(-max_rpy, max_rpy)

    beam_pose = trimesh.transformations.euler_matrix(roll, pitch, yaw)
    beam_pose[:3, 3] = [p1_w + gap_width / 2, sc[1], -ph / 2]
    meshes.append(trimesh.creation.box([gap_width, bw, ph], transform=beam_pose))
    
    # 4. Optional Floor below the beam
    if np.random.uniform(0, 1) < cfg.floor_ratio:
        z_floor = np.random.uniform(cfg.floor_height[0], cfg.floor_height[1])
        floor_thick = 0.1
        floor_pose = np.eye(4)
        floor_pose[:3, 3] = [tw / 2, tl / 2, z_floor - floor_thick / 2]
        meshes.append(trimesh.creation.box([tw, tl, floor_thick], transform=floor_pose))

    # Spawn in the center of the first platform
    origin = np.array([sc[0], sc[1], 0])
    return meshes, origin

def pallets_up_down_zc(difficulty: float, cfg: PalletZcCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Pallet terrain: rotated parallel slats with random height offsets and gaps.

    Slats are clipped to the tile bounds.
    """
    tw, tl = cfg.size[0], cfg.size[1]
    sc = np.array(cfg.size) / 2
    meshes = []
    
    # 1. Platform (Base Support)
    platform_pose = np.eye(4)
    platform_pose[:3, 3] = [sc[0], sc[1], -0.1]
    platform = trimesh.creation.box([cfg.platform_size, cfg.platform_size, 0.2], transform=platform_pose)
    meshes.append(platform)

    # 2. Border Mesh
    border_meshes = make_border(
        (tw, tl),
        (tw - 2 * cfg.border_size, tl - 2 * cfg.border_size),
        1.0,
        [sc[0], sc[1], -0.5],
    )
    meshes += border_meshes

    # Difficulty-scaled parameters
    terrain_ori = np.random.uniform(-np.pi/2, np.pi/2)
    step_height = cfg.step_height[0] + (cfg.step_height[1] - cfg.step_height[0]) * difficulty
    step_width = (cfg.step_width[0] + (cfg.step_width[1] - cfg.step_width[0]) * difficulty)
    gap_size = (cfg.gap_size[0] + (cfg.gap_size[1] - cfg.gap_size[0]) * difficulty) 
    
    # Rotation
    R_full = trimesh.transformations.rotation_matrix(terrain_ori, [0, 0, 1])
    R = R_full[:3, :3]
    
    # Coverage
    period = (step_width + gap_size)
    scale_coef = 1.0 / max(np.abs(R[0, 0]), np.abs(R[1, 0]), 1e-6)
    num_steps = int((max(tw, tl) * scale_coef) / period)
    
    # 3. Pallet Slats
    for i in range(-num_steps, num_steps + 1):
        x_off = i * period
        cx = sc[0] + x_off * R[0, 0]
        cy = sc[1] + x_off * R[1, 0]
        
        # Slat direction (Local Y)
        ux, uy = R[0, 1], R[1, 1]
        
        # Clip to tile bounds [0, tw] x [0, tl]
        s_min, s_max = -max(tw, tl), max(tw, tl)
        valid = True
        for b_val, dir_val, limit in [(cx, ux, tw), (cy, uy, tl)]:
            if np.abs(dir_val) > 1e-6:
                l1, l2 = -b_val / dir_val, (limit - b_val) / dir_val
                s_min = max(s_min, min(l1, l2))
                s_max = min(s_max, max(l1, l2))
            else:
                if b_val < -1e-3 or b_val > limit + 1e-3:
                    valid = False
                    break
        
        if valid and s_max > s_min + 0.05:
            l_slat = s_max - s_min
            h_rand = step_height * np.random.uniform(-0.5, 0.5) if np.abs(i) > 0 else 0.0
            s_pose = np.eye(4)
            s_pose[:3, 3] = [cx + (s_max + s_min)/2 * ux, cy + (s_max + s_min)/2 * uy, -0.1 + h_rand]
            s_pose[:3, :3] = R
            slat = trimesh.creation.box([step_width, l_slat, 0.2])
            slat.apply_transform(s_pose)
            meshes.append(slat)

    # 4. Optional Floor below the pallets
    if np.random.uniform(0, 1) < cfg.floor_ratio:
        z_floor = np.random.uniform(cfg.floor_height[0], cfg.floor_height[1])
        floor_thick = 0.1
        floor_pose = np.eye(4)
        floor_pose[:3, 3] = [tw / 2, tl / 2, z_floor - floor_thick / 2]
        meshes.append(trimesh.creation.box([tw, tl, floor_thick], transform=floor_pose))

    origin = np.array([sc[0], sc[1], step_height])
    return meshes, origin

def stones_terrain(difficulty: float, cfg: StonesCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom stones terrain function.
    
    Creates a grid of stones with randomized shifts, gaps, and heights.
    """
    tw, tl = cfg.size[0], cfg.size[1]
    sc = np.array(cfg.size) / 2
    
    box_height = cfg.min_box_height + (cfg.max_box_height - cfg.min_box_height) * difficulty
    gap_size = cfg.min_gap_size + (cfg.max_gap_size - cfg.min_gap_size) * difficulty
    box_size = cfg.max_box_size - (cfg.max_box_size - cfg.min_box_size) * difficulty

    ph_rise = np.random.uniform(cfg.flat_rise[0], cfg.flat_rise[1]) if cfg.higher_flat else 0.0
    flat_rise = [ph_rise, ph_rise]

    meshes = []
    
    # 1. Border
    num_boxes_x = int((tw - cfg.boundary_size) / box_size)
    num_boxes_y = int((tl - cfg.boundary_size) / box_size)
    border_x = tw - num_boxes_x * box_size
    border_y = tl - num_boxes_y * box_size
    
    border_meshes = make_border(
        (tw, tl), 
        (tw - border_x, tl - border_y), 
        1.0, 
        [sc[0], sc[1], -0.5]
    )
    meshes += border_meshes

    # 2. Stones Generation
    box_dim = [box_size, box_size, 1.0]
    base_box = trimesh.creation.box(box_dim)
    base_box.apply_translation([box_size * 0.5, box_size * 0.5, -0.5])
    
    vertices = base_box.vertices
    faces = base_box.faces
    
    num_total_boxes = num_boxes_x * num_boxes_y
    all_indices = np.arange(num_total_boxes)
    sink_indices = np.random.choice(all_indices, int(num_total_boxes * cfg.sink_ratio), replace=False)

    remaining_indices = np.setdiff1d(all_indices, sink_indices)
    rise_indices = np.random.choice(remaining_indices, int(num_total_boxes * cfg.rise_ratio), replace=False)
    all_vertices = []
    all_faces = []
    
    # Grid of offsets
    x_indices = np.arange(num_boxes_x)
    y_indices = np.arange(num_boxes_y)
    xx, yy = np.meshgrid(x_indices, y_indices, indexing='ij')
    xx = xx.flatten()
    yy = yy.flatten().astype(float)
    
    # Row shifts
    row_shifts = np.random.uniform(-0.5, 0.5, num_boxes_x)
    for i in range(num_boxes_x):
        yy[xx == i] += row_shifts[i]
        
    offsets = box_size * np.stack([xx, yy], axis=1)
    offsets[:, 0] += border_x / 2
    offsets[:, 1] += border_y / 2
    
    for i in range(num_total_boxes):
        v = vertices.copy()
        v[:, :2] += offsets[i]
        
        # Shrink each box by gap_size per side to create gaps
        local_center = box_size / 2
        v[v[:, 0] > (offsets[i, 0] + local_center), 0] -= gap_size
        v[v[:, 0] < (offsets[i, 0] + local_center), 0] += gap_size
        v[v[:, 1] > (offsets[i, 1] + local_center), 1] -= gap_size
        v[v[:, 1] < (offsets[i, 1] + local_center), 1] += gap_size
        
        # Noise
        xy_noise = np.random.uniform(-gap_size / 1.1, gap_size / 1.1, 2)
        z_noise = np.random.uniform(-box_height, box_height)
        top_verts = v[:, 2] > -0.1
        v[top_verts, :2] += xy_noise
        v[top_verts, 2] += z_noise
        if i in sink_indices:
            v[:, 2] -= 1.5
        if i in rise_indices:
            # No raised stones within platform_size/2 + box_size/2 of the origin
            dist_x = np.abs(offsets[i][0] + box_size / 2 - sc[0])
            dist_y = np.abs(offsets[i][1] + box_size / 2 - sc[1])
            exclusion_radius = cfg.platform_size / 2 + box_size / 2
            if dist_x >= exclusion_radius or dist_y >= exclusion_radius:
                v[top_verts, 2] += 1.5
        
        all_vertices.append(v)
        all_faces.append(faces + i * len(vertices))
        
    if num_total_boxes > 0:
        stones_mesh = trimesh.Trimesh(vertices=np.vstack(all_vertices), faces=np.vstack(all_faces))
        meshes.append(stones_mesh)

    # 3. Platform
    platform_dim = [cfg.platform_size, cfg.platform_size, 1.0 + box_height]
    platform_pose = np.eye(4)
    platform_pose[:3, 3] = [sc[0], sc[1], -0.5 + box_height / 2 + flat_rise[0]]
    platform = trimesh.creation.box(platform_dim, transform=platform_pose)
    platform = platform.subdivide().subdivide()
    top_mask = platform.vertices[:, 2] > (platform_pose[2, 3] + box_height / 2 - 0.01)
    platform.vertices[top_mask] += np.random.uniform(-0.02, 0.02, size=(np.sum(top_mask), 3))
    meshes.append(platform)

    # 4. Optional Floor below the stones
    if np.random.uniform(0, 1) < cfg.floor_ratio:
        z_floor = np.random.uniform(cfg.floor_height[0], cfg.floor_height[1])
        floor_thick = 0.1
        floor_pose = np.eye(4)
        floor_pose[:3, 3] = [tw / 2, tl / 2, z_floor - floor_thick / 2]
        meshes.append(trimesh.creation.box([tw, tl, floor_thick], transform=floor_pose))

    # 5. Extra Border
    outer_border_meshes = make_border(
        (tw, tl),
        (tw - 2 * cfg.boundary_size, tl - 2 * cfg.boundary_size),
        0.2,
        [sc[0], sc[1], -0.1 + flat_rise[1]]
    )
    meshes += outer_border_meshes

    origin = np.array([sc[0], sc[1], box_height + 0.05 + flat_rise[0]])
    return meshes, origin

def stepping_stones_cylinders(difficulty: float, cfg: SteppingStonesCylindersCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate cylindrical stepping stones around a central platform (fixed seed and difficulty)."""
    np.random.seed(42)
    difficulty = 0.9 # fixed difficulty
    tw, tl = cfg.size
    sc = np.array(cfg.size) / 2
    
    # Process parameters
    d = 0.5 * cfg.platform_size
    h = cfg.cp0.height + (cfg.cp1.height - cfg.cp0.height) * difficulty
    num = cfg.cp0.num + int((cfg.cp1.num - cfg.cp0.num) * difficulty)
    max_rot_deg = cfg.cp0.rot + (cfg.cp1.rot - cfg.cp0.rot) * difficulty
    
    meshes = []
    
    # 1. Ground plane (Top at z=0)
    ground_pose = np.eye(4)
    ground_pose[:3, 3] = [sc[0], sc[1], -0.005]
    meshes.append(trimesh.creation.box([tw, tl, 0.01], transform=ground_pose))
    
    # 2. Obstacles
    from scipy.spatial.transform import Rotation as Rot
    # Sample all positions at once (deterministic under the fixed seed)
    c_list = np.zeros((num, 3))
    c_list[:, 0] = np.random.uniform(cfg.border_size, tw - cfg.border_size, num)
    c_list[:, 1] = np.random.uniform(cfg.border_size, tl - cfg.border_size, num)

    for i in range(num):
        cx, cy = c_list[i, 0], c_list[i, 1]
        
        # Check platform exclusion
        is_platform = (
            sc[0] - d < cx < sc[0] + d and 
            sc[1] - d < cy < sc[1] + d
        )
        
        if not is_platform:
            delta_h = np.random.uniform(-cfg.max_rand_height, cfg.max_rand_height)
            new_h = h + delta_h
            if new_h > 0.0:
                pose = np.eye(4)
                pose[0:3, -1] = [cx, cy, 0.0] # Centered at z=0 -> top at new_h/2
                
                # Random yaw; roll/pitch scaled by max_rot_deg / 180
                R_euler = Rot.random().as_euler("zyx")
                R_euler[1:3] *= max_rot_deg / 180.0
                pose[0:3, 0:3] = Rot.from_euler("zyx", R_euler).as_matrix()
                
                stone = trimesh.creation.cylinder(radius=cfg.radius, height=new_h, sections=np.random.randint(4, 6), transform=pose)
                meshes.append(stone)
                
    # 3. Platform (Top at 0.5h)
    plat_dim = [cfg.platform_size, cfg.platform_size, 0.5 * h]
    plat_pose = np.eye(4)
    plat_pose[:3, 3] = [sc[0], sc[1], h * 0.25]
    meshes.append(trimesh.creation.box(plat_dim, transform=plat_pose))
    
    # 4. Border (Top at 0.5h)
    border_meshes = make_border(
        (tw, tl), (tw - 2 * cfg.border_size, tl - 2 * cfg.border_size),
        0.5 * h, [sc[0], sc[1], h * 0.25]
    )
    meshes += border_meshes
    
    origin = np.array([sc[0], sc[1], 0.5 * h]) 
    return meshes, origin

def mix_sparse_climb(difficulty: float, cfg: MixSparseClimbCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a stone-row terrain: two stone rows and two rotated pallets ending at a tall block."""
    stone_size = cfg.stone_size
    gap_size = cfg.gap_size
    platform_size = cfg.platform_size
    border_size = cfg.border_size
    tw, tl = cfg.size
    sc = np.array(cfg.size) / 2

    meshes = []

    # 1. Platform
    platform_pose = np.eye(4)
    platform_pose[:3, -1] = [sc[0], sc[1], -0.4]
    platform_dim = [cfg.platform_size, cfg.platform_size, 1.0]
    platform = trimesh.creation.box(platform_dim, platform_pose)
    meshes.append(platform)

    # 2. Stones (Boxes - Set 1)
    for box_id in range(2):
        for k in range(box_id + 1):
            box_x = 0.5 * platform_size + sc[0] + gap_size + stone_size / 2 + box_id * (stone_size + gap_size) - 0.1
            dim = [stone_size, stone_size, 0.4]
            pose = np.eye(4)
            pose[:3, -1] = [box_x, sc[1] - 0.2 * (box_id + 0.5) + 0.4 * k + 0.1, -0.2]
            thisbox = trimesh.creation.box(dim, pose)
            meshes.append(thisbox)

    # 3. rotated Pallets (Boxes - Set 2)
    for pallet_id in range(2):
        box_x = 0.5 * platform_size + sc[0] + gap_size + stone_size / 2 + (pallet_id + 2) * (stone_size + gap_size)
        dim = [stone_size, 1.0, 0.4]

        pose = np.eye(4)
        pose[:3, -1] = [box_x, sc[1], -0.2]

        theta = -np.pi / 18 * (-1)**pallet_id
        c, s = np.cos(theta), np.sin(theta)
        pose[0, 0] = c
        pose[0, 1] = -s
        pose[1, 0] = s
        pose[1, 1] = c
        thisbox = trimesh.creation.box(dim, pose)
        meshes.append(thisbox)

    # 4. Border (End Obstacle)
    border_dim = [cfg.border_size, cfg.platform_size * 2, cfg.end_height]
    border_pose = np.eye(4)
    border_pose[:3, -1] = [tw - cfg.border_size / 2, sc[1], 0.0]
    border_box = trimesh.creation.box(border_dim, border_pose)
    meshes.append(border_box)

    origin = np.array([sc[0], sc[1], 0.1])
    return meshes, origin

def pit_terrain(difficulty: float, cfg: PitTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom pit terrain function."""
    tw, tl = cfg.size
    sc = np.array(cfg.size) / 2
    meshes = []

    pw = cfg.pit_width
    ph = 2.0  # Thickness of the ground

    x1 = sc[0] - pw/2
    x2 = sc[0] + pw/2
    y1 = sc[1] - pw/2
    y2 = sc[1] + pw/2

    # 1. North Ground
    n_h = tl - y2
    n_pose = np.eye(4)
    n_pose[:3, 3] = [sc[0], y2 + n_h/2, -ph/2]
    meshes.append(trimesh.creation.box([tw, n_h, ph], transform=n_pose))

    # 2. South Ground
    s_h = y1
    s_pose = np.eye(4)
    s_pose[:3, 3] = [sc[0], s_h/2, -ph/2]
    meshes.append(trimesh.creation.box([tw, s_h, ph], transform=s_pose))

    # 3. East Ground
    e_w = tw - x2
    e_pose = np.eye(4)
    e_pose[:3, 3] = [x2 + e_w/2, sc[1], -ph/2]
    meshes.append(trimesh.creation.box([e_w, pw, ph], transform=e_pose))

    # 4. West Ground
    w_w = x1
    w_pose = np.eye(4)
    w_pose[:3, 3] = [w_w/2, sc[1], -ph/2]
    meshes.append(trimesh.creation.box([w_w, pw, ph], transform=w_pose))

    # 5. Pit Bottom
    pd = cfg.pit_depth
    b_pose = np.eye(4)
    b_pose[:3, 3] = [sc[0], sc[1], -pd - ph/2]
    meshes.append(trimesh.creation.box([pw, pw, ph], transform=b_pose))

    origin = np.array([sc[0], sc[1], -pd])
    return meshes, origin

def peak_terrain(difficulty: float, cfg: PeakTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom peak terrain function: a single centered box on flat ground.

    Difficulty agnostic — geometry is fully controlled by ``peak_width`` and ``peak_height``.
    """
    tw, tl = cfg.size
    sc = np.array(cfg.size) / 2
    meshes = []

    pw = cfg.peak_width
    ph_thick = 2.0  # Thickness of the ground slab (top surface at z=0)

    # 1. Ground slab covering the whole tile (top surface at z = 0)
    g_pose = np.eye(4)
    g_pose[:3, 3] = [sc[0], sc[1], -ph_thick / 2]
    meshes.append(trimesh.creation.box([tw, tl, ph_thick], transform=g_pose))

    # 2. Peak box centered on the tile, sitting on the ground
    pk_h = cfg.peak_height
    pk_pose = np.eye(4)
    pk_pose[:3, 3] = [sc[0], sc[1], pk_h / 2]
    meshes.append(trimesh.creation.box([pw, pw, pk_h], transform=pk_pose))

    # Robot origin on top of the peak (mirrors pit_terrain placing origin at pit bottom)
    origin = np.array([sc[0], sc[1], pk_h])
    return meshes, origin
