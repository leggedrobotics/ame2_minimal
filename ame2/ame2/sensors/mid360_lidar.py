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
import warp as wp
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
        if self.cfg.self_occlusion or self.cfg.min_range > 0.0:
            rng = (hits - ray_starts_w).norm(dim=-1)                     # inf for misses
            drop = rng < self.cfg.min_range
            if self.cfg.self_occlusion:
                drop |= self._self_box_occluded(env_ids, pos_w, ray_dirs_w, rng)
            hits[drop] = float("inf")                                    # dropped = no return
        self._data.ray_hits_w[env_ids] = hits

        if self.cfg.keep_world_ray_buffers:
            self._ray_starts_w[env_ids] = ray_starts_w
            self._ray_directions_w[env_ids] = ray_dirs_w

    def _init_self_boxes(self):
        """One tight oriented box per link (from its visual mesh, link frame): the cheap whole-body occluder set."""
        import gc
        import trimesh
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene
        from isaaclab.utils.math import matrix_from_quat, quat_from_matrix
        from isaaclab.utils.mesh import PRIMITIVE_MESH_TYPES, create_trimesh_from_geom_mesh, create_trimesh_from_geom_shape

        scene = next(o for o in gc.get_objects() if isinstance(o, InteractiveScene) and "robot" in o.articulations)
        self._robot = scene["robot"]
        root = self._robot.cfg.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0").replace("env_.*", "env_0")
        ids, off_tf, half, sph = [], [], [], []
        for i, name in enumerate(self._robot.body_names):
            if name in self.cfg.self_occlusion_skip:
                continue
            link = f"{root}/{name}"
            prims = sim_utils.get_all_matching_child_prims(link, lambda p: p.GetTypeName() in PRIMITIVE_MESH_TYPES + ["Mesh"])
            vis = [p for p in prims if "/collisions" not in str(p.GetPath())]  # USD collisions duplicate the visuals
            parts = []
            for mp in vis or prims:
                m = create_trimesh_from_geom_mesh(mp) if mp.GetTypeName() == "Mesh" else create_trimesh_from_geom_shape(mp)
                m.apply_scale(sim_utils.resolve_prim_scale(mp))
                pos, quat = sim_utils.resolve_prim_pose(mp, sim_utils.find_first_matching_prim(link))
                tf = np.eye(4)
                tf[:3, :3] = matrix_from_quat(torch.tensor(quat, dtype=torch.float32)).numpy()
                tf[:3, 3] = pos
                m.apply_transform(tf)
                parts.append(m)
            if not parts:
                continue
            tm = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]
            boxes = _link_boxes(tm, self.cfg.self_occlusion_box_len, self.cfg.self_occlusion_max_boxes)
            corners = np.concatenate([(b_tf[:3, :3] @ (np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1) * b_h[:, None])).T
                                      + b_tf[:3, 3] for b_tf, b_h in boxes])
            ctr = 0.5 * (corners.min(0) + corners.max(0))
            sph.append((i, ctr, float(np.linalg.norm(corners - ctr, axis=1).max()), len(ids), len(ids) + len(boxes)))
            for box_tf, box_half in boxes:
                ids.append(i); off_tf.append(box_tf); half.append(box_half)
        for name, ctr, hx in self.cfg.self_occlusion_extra_boxes:   # hand-placed occluders (e.g. head-window struts)
            i = self._robot.body_names.index(name); b_tf = np.eye(4); b_tf[:3, 3] = ctr
            sph.append((i, np.asarray(ctr, dtype=float), float(np.linalg.norm(hx)), len(ids), len(ids) + 1))
            ids.append(i); off_tf.append(b_tf); half.append(np.asarray(hx, dtype=float))
        tf = torch.tensor(np.stack(off_tf), dtype=torch.float32, device=self.device)
        self._box_ids = torch.tensor(ids, device=self.device)
        # per-link bounding sphere over its boxes: rays that miss it skip that link's boxes
        self._sph_ids = torch.tensor([q[0] for q in sph], device=self.device)
        self._sph_off = torch.tensor(np.stack([q[1] for q in sph]), dtype=torch.float32, device=self.device)
        self._sph_r = torch.tensor([q[2] for q in sph], dtype=torch.float32, device=self.device)
        self._sph_range = torch.tensor([[q[3], q[4]] for q in sph], dtype=torch.int32, device=self.device)
        self._box_off_pos = tf[:, :3, 3]
        self._box_off_quat = quat_from_matrix(tf[:, :3, :3])
        self._box_half = torch.tensor(np.stack(half), dtype=torch.float32, device=self.device)
        print(f"[Mid360] self-occlusion boxes: {len(ids)} boxes on {len(sph)} links, skipped {list(self.cfg.self_occlusion_skip)}")

    def _self_box_occluded(self, env_ids, origin_w: torch.Tensor, dirs_w: torch.Tensor, rng: torch.Tensor) -> torch.Tensor:
        """(n, R) bool: the ray meets a link box before its terrain hit (slab test, one warp thread per ray)."""
        from isaaclab.utils.math import quat_mul

        if not hasattr(self, "_box_ids"):
            self._init_self_boxes()
        n, R, K = dirs_w.shape[0], dirs_w.shape[1], len(self._box_ids)
        lp = self._robot.data.body_pos_w[env_ids][:, self._box_ids]
        lq = self._robot.data.body_quat_w[env_ids][:, self._box_ids]
        bq = quat_mul(lq, self._box_off_quat.expand(n, K, 4))
        bp = lp + quat_apply(lq, self._box_off_pos.expand(n, K, 3))
        L = len(self._sph_ids)
        sq = self._robot.data.body_quat_w[env_ids][:, self._sph_ids]
        sc = self._robot.data.body_pos_w[env_ids][:, self._sph_ids] + quat_apply(sq, self._sph_off.expand(n, L, 3))
        r = torch.nan_to_num(rng, nan=self.cfg.max_distance, posinf=self.cfg.max_distance).contiguous()
        out = torch.zeros(n, R, dtype=torch.int32, device=self.device)
        wp.launch(_box_occlusion_kernel, dim=(n, R), device=self.device, inputs=[
            wp.from_torch(origin_w.contiguous(), dtype=wp.vec3), wp.from_torch(dirs_w.contiguous(), dtype=wp.vec3),
            wp.from_torch(r), wp.from_torch(sc.contiguous(), dtype=wp.vec3), wp.from_torch(self._sph_r),
            wp.from_torch(self._sph_range, dtype=wp.vec2i), wp.from_torch(bp.contiguous(), dtype=wp.vec3),
            wp.from_torch(bq[..., [1, 2, 3, 0]].contiguous(), dtype=wp.quat),  # warp quats are (x, y, z, w)
            wp.from_torch(self._box_half, dtype=wp.vec3), wp.from_torch(out)])
        return out.bool()


def _link_boxes(m, box_len: float, max_boxes: int) -> list:
    """Oriented boxes for one link mesh: slice it along its longest box axis into ceil(length / box_len) equal pieces
    (at most ``max_boxes``), one tight box per piece, fitted on hull corners + surface samples. Returns (tf, half) pairs."""
    import trimesh

    def fit(pts):   # PCA box: principal axes + extents (O(n); a bit looser than the minimum-volume box)
        c = pts.mean(0)
        _, vec = np.linalg.eigh(np.cov((pts - c).T))
        if np.linalg.det(vec) < 0:   # proper rotation
            vec[:, 0] *= -1.0
        q = (pts - c) @ vec
        lo, hi = q.min(0), q.max(0)
        tf = np.eye(4); tf[:3, :3] = vec; tf[:3, 3] = c + vec @ (0.5 * (lo + hi))
        return tf, 0.5 * (hi - lo)

    pts = np.concatenate([m.vertices, trimesh.sample.sample_surface(m, 1500, seed=0)[0]])
    tf, half = fit(pts)
    k = int(np.argmax(half)); n = int(min(max_boxes, max(1, np.ceil(2 * half[k] / box_len))))
    if n == 1:
        return [(tf, half)]
    s = (pts - tf[:3, 3]) @ tf[:3, k]
    edges = np.linspace(-half[k], half[k], n + 1)
    return [fit(pts[(s >= a) & (s <= b)]) for a, b in zip(edges[:-1], edges[1:]) if ((s >= a) & (s <= b)).sum() >= 8]


@wp.func
def _slab(o: float, d: float, h: float, t: wp.vec2) -> wp.vec2:
    """Clip the ray interval t against one box slab |x| <= h (box frame)."""
    if wp.abs(d) < 1.0e-9:
        if o < -h or o > h:
            return wp.vec2(1.0, 0.0)  # parallel and outside: empty interval
        return t
    t1 = (-h - o) / d
    t2 = (h - o) / d
    return wp.vec2(wp.max(t[0], wp.min(t1, t2)), wp.min(t[1], wp.max(t1, t2)))


@wp.kernel
def _box_occlusion_kernel(origins: wp.array(dtype=wp.vec3), dirs: wp.array2d(dtype=wp.vec3), rng: wp.array2d(dtype=float),
                          sph_pos: wp.array2d(dtype=wp.vec3), sph_r: wp.array(dtype=float), sph_range: wp.array(dtype=wp.vec2i),
                          box_pos: wp.array2d(dtype=wp.vec3), box_quat: wp.array2d(dtype=wp.quat), half: wp.array(dtype=wp.vec3),
                          out: wp.array2d(dtype=wp.int32)):
    """out[i, j] = 1 if ray j of env i meets any link box before its terrain range rng[i, j]; a link's boxes are only
    tested when the ray passes that link's bounding sphere."""
    i, j = wp.tid()
    o = origins[i]
    d = dirs[i, j]
    r = rng[i, j]
    for s in range(sph_r.shape[0]):
        oc = sph_pos[i, s] - o
        tca = wp.dot(oc, d)
        rad = sph_r[s]
        d2 = wp.dot(oc, oc) - tca * tca
        if d2 > rad * rad or tca + rad < 0.0 or tca - wp.sqrt(rad * rad - d2) > r:
            continue
        for k in range(sph_range[s][0], sph_range[s][1]):
            q = box_quat[i, k]
            lo = wp.quat_rotate_inv(q, o - box_pos[i, k])
            ld = wp.quat_rotate_inv(q, d)
            h = half[k]
            t = wp.vec2(0.0, r)
            t = _slab(lo[0], ld[0], h[0], t)
            t = _slab(lo[1], ld[1], h[1], t)
            t = _slab(lo[2], ld[2], h[2], t)
            if t[0] <= t[1]:
                out[i, j] = 1
                return


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

    self_occlusion: bool = False
    """Drop rays whose first hit is the robot's own body (self occlusion + the deploy self-body filter).
    Occluders are lidar-only boxes (physics colliders untouched): each link's visual mesh is cut into up to
    ``self_occlusion_max_boxes`` pieces along its longest axis, one tight oriented box per piece (warp slab test)."""

    self_occlusion_box_len: float = 0.1
    """Links longer than this get sliced along their longest axis, one box per piece [m]."""

    self_occlusion_max_boxes: int = 5
    """Max boxes per link."""

    self_occlusion_extra_boxes: tuple = (
        ("mid360_link", (0.0, 0.058, -0.03), (0.006, 0.0025, 0.045)),
        ("mid360_link", (0.0, -0.058, -0.03), (0.006, 0.0025, 0.045)),
    )
    """Occluders added by hand: (link, centre, half extents) in that link's frame [m]. Default: the two thin struts
    of the head window left and right of the Mid360 (guessed size; the head mesh is a closed shell there)."""

    self_occlusion_skip: tuple[str, ...] = ("mid360_link", "d435_link", "head_link")
    """Links never treated as occluders (sensor housings; the head shell encloses the lidar origin)."""

    min_range: float = 0.0
    """Drop hits closer than this to the sensor origin [m] (deploy near-range filter); 0 = off."""

    keep_world_ray_buffers: bool = False
    """Allocate per-ray world start/direction buffers (2 x E x R x 3 floats —
    ~3.6 GB at 8k envs x 20k rays). Needed only by play.py --viz_lidar."""
