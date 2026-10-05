# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AME2 terrain configurations."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

from .custom_terrains import *


AME2_TERRAINS_CFG_G1 = TerrainGeneratorCfg(
    size=(12.0, 12.0),
    border_width=10.0,
    num_rows=10,
    num_cols=60,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "rough": RoughTerrainCfg(
            function=rough_terrain,
            proportion=0.05,
            max_slope_deg=20.0,
            max_roughness_m=0.16,
            undulation_res=3.0,
            border_width=0.25,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(0.0, 5.5),
                    y_range=(-5.5, 5.5),
                )
            },
        ),
        "stair_down": PyramidStairsCfg(
            function=pyramid_stairs,
            proportion=0.05,
            max_step_height=0.27,
            border_size=2.0,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(1.0, 6.0),
                    y_range=(-0.2, 0.2),
                )
            },
        ),
        "stair_up": PyramidStairsInvCfg(
            function=pyramid_stairs_inv,
            proportion=0.05,
            max_step_height=0.27,
            border_size=2.0,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(1.0, 6.0),
                    y_range=(-0.2, 0.2),
                )
            },
        ),
        "boxes": BoxesTerrainCfg(
            function=boxes_terrain,
            proportion=0.05,
            num_boxes=20,
            box_size_range=(0.8, 2.0),
            min_max_height=0.05,
            max_max_height=0.3,
            platform_size=2.0,
            dilation_size=0.1,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(0.0, 5.5),
                    y_range=(-5.5, 5.5),
                )
            },
        ),
        "obstacles": ObstaclesTerrainCfg(
            function=obstacles_terrain,
            proportion=0.05,
            max_slope_deg=20.0,
            undulation_res=3.0,
            max_obstacle_density=0.5,
            obstacle_height_range=(0.2, 2.0),
            obstacle_size_range=(0.3, 0.8),
            platform_size=2.0,
            dilation_size=0.1,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(2.0, 5.5),
                    y_range=(-5.5, 5.5),
                )
            },
        ),
        "climb_up": ClimbUpCfg(
            function=climb_up,
            proportion=0.2,
            min_height=0.1,
            max_height=0.65,
            rough=True,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.15,
                    x_range=(2.6, 5.5),
                    y_range=(-3.0, 3.0),
                )
            },
            wall_x=2.5,
        ),
        "climb_down": ClimbDownCfg(
            function=climb_down,
            proportion=0.05,
            min_height=0.2,
            max_height=0.88,
            rough=True,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.15,
                    x_range=(2.6, 5.5),
                    y_range=(-3.0, 3.0),
                )
            },
            wall_x=1.0,
        ),
        "consecutive": ClimbingConsecutiveCfg(
            function=climbing_consecutive,
            proportion=0.05,
            min_height=0.05,
            max_height_lower=0.35,
            max_height_upper=0.35,
            rough=False,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.15,
                    x_range=(4.0, 5.0),
                    y_range=(-4.0, 4.0),
                )
            },
            wall_offsets=(0.8, 1.7, 2.2, 3.1),
        ),
        "gap": GapTerrainCfg(
            function=gap_terrain,
            proportion=0.05,
            min_gap=0.18,
            max_gap=1.2,
            platform_height=2.0,
            gap_x=1.0,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.1,
                    x_range=(1.8, 5.0),
                    y_range=(-3.0, 3.0),
                )
            },
        ),
        "pallets": PalletZcCfg(
            function=pallets_up_down_zc,
            proportion=0.05,
            platform_size=2.0,
            border_size=2.0,
            step_height=[0.0, 0.2],
            step_width=[0.4, 0.16],
            gap_size=[0.16, 0.24],
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.3,
                    x_range=(3.5, 5.8),
                    y_range=(-5., 5.),
                )
            },
        ),
        "beam": BalanceBeamCfg(
            function=balance_beam_terrain,
            proportion=0.05,
            gap_x=1.0,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.3,
                    x_range=(4.0, 5.8),
                    y_range=(-0.5, 0.5),
                )
            },
        ),
        "stones": StonesCfg(
            function=stones_terrain,
            proportion=0.30,
            boundary_size=2.0,
            min_box_size=0.41,
            max_box_size=0.71,
            higher_flat=True,
            max_box_height=0.06,
            flat_rise=(-0.05, 0.15),
            min_gap_size=0.08,
            max_gap_size=0.12,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.3,
                    x_range=(1.0, 5.8),
                    y_range=(-5.0, 5.0),
                )
            },
        ),
        "pit": PitTerrainCfg(
            function=pit_terrain,
            proportion=0.0,
            pit_width=1.8,
            pit_depth=0.5,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.5,
                    max_height_diff=0.35,
                    x_range=(2.0, 5.5),
                    y_range=(-0.5, 0.5),
                )
            },
        ),
        "peak": PeakTerrainCfg(
            function=peak_terrain,
            proportion=0.0,
            peak_width=1.8,
            peak_height=1.0,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.5,
                    max_height_diff=0.35,
                    x_range=(2.4, 5.5),
                    y_range=(-0.5, 0.5),
                )
            },
        ),
        "test_1": SteppingStonesCylindersCfg(
            function=stepping_stones_cylinders,
            proportion=0.0,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=10,
                    patch_radius=0.1,
                    max_height_diff=0.1,
                    x_range=(5.65, 5.66),
                    y_range=(0.7, 0.71),
                )
            },
        ),
        "test_2": MixSparseClimbCfg(
            function=mix_sparse_climb,
            proportion=0.0,
            flat_patch_sampling={
                "target": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=10,
                    patch_radius=0.1,
                    max_height_diff=0.1,
                    x_range=(4.5, 4.51),
                    y_range=(-0.02, 0.02),
                )
            },
        ),
    },
)

# Backward-compatible alias
AME2_TERRAINS_CFG = AME2_TERRAINS_CFG_G1
