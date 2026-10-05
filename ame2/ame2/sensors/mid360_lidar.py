"""Livox Mid360 lidar simulation for the G1.

Reproduces the ray pattern used to train the lidar height-map model: the
bundled table holds the real Mid360 non-repetitive scan sequence (~800k unit
directions, sensor frame); each scan uses one random contiguous window of
``rays_per_scan`` rows (one 0.1 s scan at 10 Hz), shared across all envs
updated in the same call.

Mid360 sensor artifacts (mixed-pixel ghosts, edge dropout, driver filter,
near-field pull-ins, ...) are not simulated: :func:`add_noise` is an identity
placeholder gated by ``Mid360RayCasterCfg.enable_noise``.
"""

from __future__ import annotations

import math
import os
import numpy as np
import torch
from collections.abc import Sequence
from typing import Callable, Tuple

from isaaclab.sensors import RayCaster, RayCasterCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.sensors.ray_caster.ray_cast_utils import obtain_world_pose_from_view
from isaaclab.utils import configclass
from isaaclab.utils.math import combine_frame_transforms, quat_apply
from isaaclab.utils.warp import raycast_mesh

# Bundled Mid360 scan sequence: (N, 3) fp32 unit ray directions in the sensor
# frame, in real scan order.
_MID360_RAYDIR_PATH = os.path.join(os.path.dirname(__file__), "mid360_raydirs.npy")

# Mounting orientation: R_tot = R_base @ R_bias with R_bias = Ry(pitch) @ Rx(roll)
# (R = Rz@Ry@Rx); the Mid360 hangs upside down (roll pi) with a calibrated pitch.
_SENSOR_BIAS_ROLL = math.pi
_SENSOR_BIAS_PITCH = 0.05112069379091391
# In g1_lidar.usd the mid360_joint already pitches mid360_link w.r.t. the
# torso by Ry(2*atan2(0.02006994, 0.9997986)) (localRot0 of the fixed joint),
# so the sensor-frame offset applied on mid360_link only carries the flip and
# the residual calibrated-vs-nominal pitch. Assumes torso_link == base at the
# default waist pose (as during calibration).
_USD_MID360_JOINT_PITCH = 2.0 * math.atan2(0.02006994, 0.9997986)


def _mid360_mount_offset_rot() -> Tuple[float, float, float, float]:
    """Quaternion (w,x,y,z) of Ry(bias_pitch - usd_joint_pitch) @ Rx(pi)."""
    half_p = 0.5 * (_SENSOR_BIAS_PITCH - _USD_MID360_JOINT_PITCH)
    qy = (math.cos(half_p), 0.0, math.sin(half_p), 0.0)
    qx = (math.cos(0.5 * _SENSOR_BIAS_ROLL), math.sin(0.5 * _SENSOR_BIAS_ROLL), 0.0, 0.0)
    w1, x1, y1, z1 = qy
    w2, x2, y2, z2 = qx
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


MID360_MOUNT_OFFSET_ROT: Tuple[float, float, float, float] = _mid360_mount_offset_rot()
"""``RayCasterCfg.OffsetCfg.rot`` for a Mid360 sensor parented to ``mid360_link``."""


def add_noise(
    ray_hits_w: torch.Tensor,
    sensor_pos_w: torch.Tensor,
    ray_dirs_w: torch.Tensor,
    enable: bool = False,
) -> torch.Tensor:
    """Mid360 artifact model (identity placeholder; returns the hits unchanged).

    Args:
        ray_hits_w: (E, R, 3) world-frame hit points (misses = inf).
        sensor_pos_w: (E, 3) world sensor origins.
        ray_dirs_w: (E, R, 3) world ray directions.
        enable: master switch; False (default) skips everything.
    """
    if not enable:
        return ray_hits_w
    return ray_hits_w


def mid360_pattern(cfg: "Mid360PatternCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Static fallback pattern: the first window of the scan sequence.

    :class:`Mid360RayCaster` re-slices a random window per update; this function
    only fixes the ray count (and lets a plain ``RayCaster`` use a frozen window).
    """
    dirs = cfg.load_table(device)
    window = dirs[: cfg.rays_per_scan : cfg.ray_decimation]
    return torch.zeros_like(window), window.clone()


@configclass
class Mid360PatternCfg(PatternBaseCfg):
    """Rolling-window pattern over the real Mid360 scan sequence."""

    func: Callable = mid360_pattern

    raydirs_path: str = _MID360_RAYDIR_PATH
    """Path to the baked (N, 3) fp32 direction table (sensor frame, scan order)."""

    rays_per_scan: int = 20000
    """Contiguous rows consumed per scan (real Mid360: 200k pts/s @ 10 Hz = 20k)."""

    ray_decimation: int = 1
    """Keep every n-th ray inside the window (same spatial coverage, thinner
    density). 1 = the real ray count."""

    def load_table(self, device: str) -> torch.Tensor:
        """Load (cached) and return the full normalized direction table."""
        if not hasattr(self, "_all_dirs"):
            dirs = torch.from_numpy(np.load(self.raydirs_path).astype(np.float32))
            self._all_dirs = dirs / dirs.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            assert self._all_dirs.shape[0] >= self.rays_per_scan, (
                f"Mid360 table has {self._all_dirs.shape[0]} rows < rays_per_scan={self.rays_per_scan}"
            )
        if str(self._all_dirs.device) != str(device):
            self._all_dirs = self._all_dirs.to(device)
        return self._all_dirs


class Mid360RayCaster(RayCaster):
    """RayCaster casting a fresh random window of the Mid360 sequence per update.

    Deviations from the base class (for the 20k-rays x many-envs scale):

    * ray directions are stored once in the full table (no per-env ``repeat`` of
      starts/directions: at 8k envs those two buffers alone would cost ~3.6 GB);
    * rays always follow the full body orientation (``ray_alignment`` is
      ignored — a body-mounted lidar has no meaningful yaw-only mode);
    * the world-frame per-ray debug buffers (``_ray_starts_w`` /
      ``_ray_directions_w``, used by play.py --viz_lidar) are only allocated
      when ``cfg.keep_world_ray_buffers`` is set.
    """

    cfg: Mid360RayCasterCfg

    def _initialize_rays_impl(self):
        pattern: Mid360PatternCfg = self.cfg.pattern_cfg
        # full table with the mounting rotation baked in (world dirs then only need the body quat)
        all_dirs = pattern.load_table(self._device)
        offset_quat = torch.tensor(list(self.cfg.offset.rot), device=self._device)
        self._all_ray_dirs = quat_apply(offset_quat.expand(all_dirs.shape[0], 4), all_dirs)
        self._offset_pos = torch.tensor(list(self.cfg.offset.pos), device=self._device)
        self._n_total_dirs = self._all_ray_dirs.shape[0]
        self._window_len = pattern.rays_per_scan
        self._decimation = pattern.ray_decimation
        self.num_rays = len(range(0, self._window_len, self._decimation))
        # base-class buffers (drift kept for parity with RayCaster.reset)
        self.drift = torch.zeros(self._view.count, 3, device=self.device)
        self.ray_cast_drift = torch.zeros(self._view.count, 3, device=self.device)
        self._data.pos_w = torch.zeros(self._view.count, 3, device=self.device)
        self._data.quat_w = torch.zeros(self._view.count, 4, device=self.device)
        self._data.ray_hits_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
        if self.cfg.keep_world_ray_buffers:
            self._ray_starts_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)
            self._ray_directions_w = torch.zeros(self._view.count, self.num_rays, 3, device=self.device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # one random contiguous window per call, shared by the updated envs
        start = int(torch.randint(0, self._n_total_dirs - self._window_len + 1, (1,)).item())
        dirs = self._all_ray_dirs[start : start + self._window_len : self._decimation]  # (R, 3)

        pos_w, quat_w = obtain_world_pose_from_view(self._view, env_ids)
        pos_w, quat_w = combine_frame_transforms(
            pos_w, quat_w, self._offset[0][env_ids], self._offset[1][env_ids]
        )
        pos_w = pos_w + quat_apply(quat_w, self._offset_pos.expand(pos_w.shape[0], 3))
        pos_w = pos_w + self.drift[env_ids]
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        n = pos_w.shape[0]
        ray_dirs_w = quat_apply(
            quat_w.repeat(1, self.num_rays), dirs.unsqueeze(0).expand(n, self.num_rays, 3)
        )
        ray_starts_w = pos_w.unsqueeze(1).expand(n, self.num_rays, 3).contiguous()

        hits = raycast_mesh(
            ray_starts_w,
            ray_dirs_w,
            max_dist=self.cfg.max_distance,
            mesh=RayCaster.meshes[self.cfg.mesh_prim_paths[0]],
        )[0]
        hits[:, :, 2] += self.ray_cast_drift[env_ids, 2].unsqueeze(-1)
        hits = add_noise(hits, pos_w, ray_dirs_w, enable=self.cfg.enable_noise)
        self._data.ray_hits_w[env_ids] = hits

        if self.cfg.keep_world_ray_buffers:
            self._ray_starts_w[env_ids] = ray_starts_w
            self._ray_directions_w[env_ids] = ray_dirs_w


@configclass
class Mid360RayCasterCfg(RayCasterCfg):
    """Configuration for :class:`Mid360RayCaster`."""

    class_type: type = Mid360RayCaster

    pattern_cfg: Mid360PatternCfg = Mid360PatternCfg()

    offset: RayCasterCfg.OffsetCfg = RayCasterCfg.OffsetCfg(rot=MID360_MOUNT_OFFSET_ROT)
    """Sensor frame in the parent (``mid360_link``) frame: the upside-down
    mount flip plus the residual calibrated pitch (see module docstring)."""

    max_distance: float = 6.0
    """Max lidar range (m). The EM input grid only spans +-3 m around the
    sensor (slant ranges < ~5 m), so 6 m covers everything the map can use."""

    enable_noise: bool = False
    """Master switch for :func:`add_noise` (currently an identity placeholder)."""

    keep_world_ray_buffers: bool = False
    """Allocate per-ray world start/direction buffers (2 x E x R x 3 floats —
    ~3.6 GB at 8k envs x 20k rays). Needed only by play.py --viz_lidar."""
