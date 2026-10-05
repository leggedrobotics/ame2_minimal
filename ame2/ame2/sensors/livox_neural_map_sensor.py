from __future__ import annotations

import math
import os
import torch
from collections.abc import Sequence

from isaaclab.sensors.sensor_base import SensorBase
from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils

from .neural_mapping_utils import LIVOX_HOLE, LivoxGlobalProbWinnerPipeline


class LivoxNeuralMapSensor(SensorBase):
    """Neural height mapping from the Mid360 lidar.

    Per lidar scan (``update_period``, 10 Hz):

    1. scatter the Mid360 hits into the model input grid: centered at the
       lidar xy, axes = base yaw, heights relative to the lidar z (the frame
       the mapping model was trained in); empty cells = -3;
    2. run the per-frame mode-selection JIT (mean + log-variance);
    3. fuse into the per-env world-aligned global map, pose-anchored at the
       lidar (xy/z) with base yaw.

    Every :attr:`data` access (each policy step) re-queries the global map on
    the export grid at the current base pose -> base-relative heights + var.
    """

    cfg: LivoxNeuralMapSensorCfg

    def __init__(self, cfg: LivoxNeuralMapSensorCfg):
        super().__init__(cfg)

    def _initialize_impl(self):
        sim = sim_utils.SimulationContext.instance()
        self._device = sim.device
        env_prim_path_expr = self.cfg.prim_path.rsplit("/", 1)[0]
        self._parent_prims = sim_utils.find_matching_prims(env_prim_path_expr)
        self._num_envs = len(self._parent_prims)

        self._is_outdated = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self._timestamp = torch.zeros(self._num_envs, device=self._device)
        self._timestamp_last_update = torch.zeros_like(self._timestamp)
        self._query_drift = torch.zeros((self._num_envs, 3), device=self._device)
        self._is_clean_source = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)

        # pre-trained per-frame TorchScript model; path is relative to the dir holding the ame2 package
        model_path_abs = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", self.cfg.jit_model_path))
        if not os.path.isfile(model_path_abs):
            raise FileNotFoundError(f"Neural mapping model not found: {model_path_abs} (not shipped; set jit_model_path).")
        self.mapping_model = torch.jit.load(model_path_abs, map_location=self._device).eval()
        try:
            self.mapping_model = torch.jit.freeze(self.mapping_model)
        except RuntimeError:
            pass  # exported with optimize_for_inference -> already frozen

        # model input grid: arange over the ranges (inclusive)
        x0, x1 = self.cfg.gt_x_range
        y0, y1 = self.cfg.gt_y_range
        res = self.cfg.res_s
        xs_s = torch.arange(x0, x1 + 1e-9, res, dtype=torch.float32, device=self._device)
        ys_s = torch.arange(y0, y1 + 1e-9, res, dtype=torch.float32, device=self._device)
        self.L_sensor, self.W_sensor = xs_s.shape[0], ys_s.shape[0]
        self.Xs, self.Ys = torch.meshgrid(xs_s, ys_s, indexing="ij")
        # export grid (what the policy/visualization reads), centered on x_mid_b
        self.L_base, self.W_base = self.cfg.L_base, self.cfg.W_base
        xs_b = (torch.arange(self.L_base, dtype=torch.float32) - (self.L_base - 1) / 2) * self.cfg.res_b + self.cfg.x_mid_b
        ys_b = (torch.arange(self.W_base, dtype=torch.float32) - (self.W_base - 1) / 2) * self.cfg.res_b
        self.Xb, self.Yb = torch.meshgrid(xs_b.to(self._device), ys_b.to(self._device), indexing="ij")

        self.pipeline = LivoxGlobalProbWinnerPipeline(
            Xs_local=self.Xs, Ys_local=self.Ys,
            batch_size=self._num_envs,
            res=self.cfg.res_s, size_m=self.cfg.pipeline_size_m,
            init_var=self.cfg.init_var, tau_min=self.cfg.tau_min,
            overwrite_drop_m=self.cfg.overwrite_drop_m,
            overwrite_std_max=self.cfg.overwrite_std_max,
            ground_reject_slope=self.cfg.ground_reject_slope,
            device=self._device,
        )

        self.last_xyzyaw = torch.zeros((self._num_envs, 4), device=self._device)
        self._is_first_update = torch.ones(self._num_envs, dtype=torch.bool, device=self._device)
        self.data_output = torch.zeros((self._num_envs, self.L_base * self.W_base, 2), device=self._device)

        if self.cfg.keep_debug_buffers:
            # persistent play-viz buffers (updated per-env as scans come in)
            self._sensor_z_local = torch.full(
                (self._num_envs, self.L_sensor, self.W_sensor), LIVOX_HOLE, device=self._device)
            self._raw_hits_w = torch.zeros((0, 3), device=self._device)
            self._raw_hits_env = torch.zeros((0,), dtype=torch.long, device=self._device)

        self._scene = self._get_scene()
        body_name = self.cfg.prim_path.split("/")[-1]
        self._body_idx, _ = self._scene["robot"].find_bodies(body_name)
        self._body_idx = self._body_idx[0]

        if self.cfg.clean_source_ratio > 0:
            self._is_clean_source[:] = torch.rand(self._num_envs, device=self._device) < self.cfg.clean_source_ratio

    def _get_scene(self):
        """Helper to dynamically fetch the InteractiveScene."""
        import gc
        from isaaclab.scene import InteractiveScene

        for obj in gc.get_objects():
            if isinstance(obj, InteractiveScene):
                if "robot" in obj.articulations:
                    return obj
        raise RuntimeError("Could not locate active InteractiveScene!")

    # ------------------------------------------------------------------
    # high-frequency query (every policy step)
    # ------------------------------------------------------------------
    @property
    def data(self):
        # accumulate a new scan if the lidar interval lapsed
        self._update_outdated_buffers()

        robot = self._scene["robot"]
        body_pos_w = robot.data.body_pos_w[:, self._body_idx]
        body_quat_w = robot.data.body_quat_w[:, self._body_idx]
        body_yaw = self._yaw_from_q(body_quat_w)  # (E,)

        # query at the current base pose -> base-relative heights
        self.pipeline.set_pose(body_pos_w[:, 0], body_pos_w[:, 1], body_pos_w[:, 2], body_yaw)

        is_clean_any = self._is_clean_source.any()
        if is_clean_any:
            critic_cam = self._scene.sensors.get(self.cfg.critic_sensor_name, None)
            if critic_cam is not None:
                critic_hits_w = critic_cam.data.ray_hits_w              # (E, R, 3)
                est_z_clean = critic_hits_w[:, :, 2] - body_pos_w[:, 2].unsqueeze(1)
                if self.cfg.clean_source_noise > 0:
                    est_z_clean += torch.randn_like(est_z_clean) * self.cfg.clean_source_noise
            else:
                is_clean_any = False
                print(f"[WARNING] {self.cfg.critic_sensor_name} clean source sensor not found!")

        if self.cfg.soft_export:
            mean_t, var_t = self.pipeline.export_to_base_soft(
                self.Xb, self.Yb, drift_xy=self._query_drift[:, :2],
                std_ratio=self.cfg.soft_std_ratio, min_confident=self.cfg.soft_min_confident,
            )
        else:
            mean_t, var_t = self.pipeline.export_to_base(self.Xb, self.Yb, drift_xy=self._query_drift[:, :2])

        est_drifted = mean_t.reshape(self._num_envs, -1) - self._query_drift[:, 2].unsqueeze(1)
        unc_drifted = var_t.reshape(self._num_envs, -1)

        if is_clean_any:
            idx_clean = self._is_clean_source.nonzero(as_tuple=True)[0]
            est_c = est_z_clean[idx_clean]
            # critic ray misses (inf) would otherwise clip to a confident +2 wall in the
            # obs; encode them like map voids (-3 sentinel at unseen-level variance)
            void_c = ~torch.isfinite(est_c)
            est_drifted[idx_clean] = torch.where(void_c, torch.full_like(est_c, LIVOX_HOLE), est_c)
            unc_c = torch.full_like(est_c, self.cfg.clean_source_unc)
            unc_drifted[idx_clean] = torch.where(void_c, torch.full_like(unc_c, self.cfg.init_var), unc_c)

        if self.cfg.random_pixel_ratio > 0:
            mask_noisy = torch.rand_like(est_drifted) < self.cfg.random_pixel_ratio
            est_drifted[mask_noisy] = torch.rand_like(est_drifted[mask_noisy]) * 0.5 - 0.7
            unc_drifted[mask_noisy] = torch.rand_like(unc_drifted[mask_noisy]) * 1.0 + 0.3

        self.data_output = torch.stack([est_drifted, unc_drifted], dim=-1)
        return self.data_output

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = range(self._num_envs)
        env_ids_tensor = torch.as_tensor(env_ids, device=self._device)

        reset_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        reset_mask[env_ids_tensor] = True
        # provisional wipe at the (possibly stale) tracked pose; the
        # authoritative spawn-aligned seeding happens on the first update
        self.pipeline.reset(reset_mask)

        self._is_first_update[env_ids_tensor] = True
        self.last_xyzyaw[env_ids_tensor] = 0.0

        if self.cfg.query_drift_range > 0:
            self._query_drift[env_ids_tensor] = torch.zeros_like(self._query_drift[env_ids_tensor]).uniform_(
                -self.cfg.query_drift_range, self.cfg.query_drift_range
            )
        else:
            self._query_drift[env_ids_tensor] = 0.0

        if self.cfg.clean_source_ratio > 0:
            self._is_clean_source[env_ids_tensor] = (
                torch.rand(len(env_ids_tensor), device=self._device) < self.cfg.clean_source_ratio
            )
        else:
            self._is_clean_source[env_ids_tensor] = False

        super().reset(env_ids)

    # ------------------------------------------------------------------
    # low-frequency accumulation (lidar rate)
    # ------------------------------------------------------------------
    def _build_em_from_lidar(self, hits_w: torch.Tensor, sensor_pos_w: torch.Tensor,
                             base_yaw: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        """Scatter Mid360 hits into the model input grid (subset-sized batch).

        Grid: centered at the lidar xy, axes rotated to base yaw, heights
        relative to the lidar z. Empty
        cells -> -3, valid heights clamped to [-3, 2].

        Args:
            hits_w: (n, R, 3) world hit points (misses = inf).
            sensor_pos_w: (n, 3) lidar origins.
            base_yaw: (n,) base yaw.
            ids: (n,) global env indices (only used to maintain the persistent
                play-viz debug buffers; may be None when those are off).

        Returns (n, L_sensor, W_sensor).
        """
        n, R = hits_w.shape[0], hits_w.shape[1]
        rel = hits_w - sensor_pos_w.unsqueeze(1)                         # (n, R, 3)
        c = torch.cos(base_yaw).unsqueeze(1)                             # (n, 1)
        s = torch.sin(base_yaw).unsqueeze(1)
        lx = c * rel[..., 0] + s * rel[..., 1]                           # world -> base-yaw frame
        ly = -s * rel[..., 0] + c * rel[..., 1]
        zr = rel[..., 2]                                                 # lidar-z-relative height

        x0, x1 = self.cfg.gt_x_range
        y0, y1 = self.cfg.gt_y_range
        res = self.cfg.res_s
        L, W = self.L_sensor, self.W_sensor
        P = L * W
        valid = (
            (lx >= x0) & (lx <= x1) & (ly >= y0) & (ly <= y1) & torch.isfinite(zr)
        )

        ix = torch.round((lx - x0) / res).long().clamp(0, L - 1)
        iy = torch.round((ly - y0) / res).long().clamp(0, W - 1)
        cell = ix * W + iy
        benv = torch.arange(n, device=self._device).unsqueeze(1).expand(n, R)
        gidx = (benv * P + cell)[valid]
        grid = torch.full((n * P,), float("-inf"), device=self._device)
        grid.scatter_reduce_(0, gidx, zr[valid], reduce="amax", include_self=True)
        grid = grid.view(n, L, W)
        filled = grid > float("-inf")
        em = torch.where(filled, grid.clamp(LIVOX_HOLE, 2.0), torch.full_like(grid, LIVOX_HOLE))

        if self.cfg.keep_debug_buffers and ids is not None:
            # persistent per-env viz buffers: replace only the updated envs
            self._sensor_z_local[ids] = em.detach()                      # (E, L, W)
            keep = ~torch.isin(self._raw_hits_env, ids)
            new_env = ids.unsqueeze(1).expand(n, R)[valid]               # global env of each new hit
            self._raw_hits_w = torch.cat([self._raw_hits_w[keep], hits_w[valid]], dim=0)
            self._raw_hits_env = torch.cat([self._raw_hits_env[keep], new_env], dim=0)

        return em

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        # only the outdated envs (post-reset phases differ per env): fuse exactly
        # one fresh scan per env per update_period. The lidar shares the same
        # per-env phase (timestamps reset together), so these scans are fresh.
        ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self._device)
        ids = ids.to(device=self._device, dtype=torch.long).reshape(-1)
        n = ids.numel()
        if n == 0:
            return

        robot = self._scene["robot"]
        base_pos = robot.data.body_pos_w[ids, self._body_idx]            # (n, 3)
        base_quat = robot.data.body_quat_w[ids, self._body_idx]
        base_yaw = self._yaw_from_q(base_quat)                           # (n,)

        # fusion pose = lidar origin (xy/z) with base yaw (the model-grid frame)
        lidar = self._scene.sensors[self.cfg.lidar_sensor_name]
        lidar_data = lidar.data                                          # triggers the lidar's own update
        hits_w = lidar_data.ray_hits_w[ids]                              # (n, R, 3)
        sensor_pos_w = lidar_data.pos_w[ids]                             # (n, 3)
        lidar_xyzyaw = torch.cat([sensor_pos_w, base_yaw.unsqueeze(-1)], dim=-1)

        em = self._build_em_from_lidar(hits_w, sensor_pos_w, base_yaw, ids)  # (n, L, W)

        # Chunked inference bounds conv activation memory (several GB per 1k envs at 201x101).
        # Batches are padded to a power-of-two size: the TorchScript fuser re-specializes per
        # batch shape, and freely varying shapes caused CUDA misaligned-address crashes.
        chunk = self.cfg.inference_chunk if self.cfg.inference_chunk > 0 else n
        preds, logvs = [], []
        with torch.inference_mode():
            for i in range(0, n, chunk):
                batch = em[i:i + chunk]
                b = batch.shape[0]
                bucket = min(1 << (b - 1).bit_length(), chunk)
                if bucket > b:
                    batch = torch.cat(
                        [batch, batch.new_full((bucket - b, *batch.shape[1:]), LIVOX_HOLE)], dim=0)
                p, lv = self.mapping_model(batch.unsqueeze(1))
                preds.append(p.squeeze(1)[:b])
                logvs.append(lv.squeeze(1)[:b])
        batch_pred = torch.cat(preds, dim=0).clone()                     # (n, L, W); clone: leave
        batch_logv = torch.cat(logvs, dim=0).clone()                     # inference_mode tensors behind

        # void predictions carry no height: keep -3 but with a large variance so
        # the fuser never lets them win
        void_logv = math.log(self.cfg.void_std ** 2)
        is_void = batch_pred <= (LIVOX_HOLE + 0.1)
        batch_logv = torch.where(is_void, torch.full_like(batch_logv, void_logv), batch_logv)

        # optionally void out too-uncertain predictions entirely
        if self.cfg.filter_estimation_std is not None:
            too_unc = torch.exp(0.5 * batch_logv) > self.cfg.filter_estimation_std
            batch_pred = torch.where(too_unc, torch.full_like(batch_pred, LIVOX_HOLE), batch_pred)
            batch_logv = torch.where(too_unc, torch.full_like(batch_logv, void_logv), batch_logv)

        # spawn-aligned seeding for freshly reset envs: pose tracker + all-unseen
        # map + (optional) confident floor square around the base at ground height
        first = self._is_first_update[ids]                               # (n,) bool
        if first.any():
            seed_ids = ids[first]
            self.last_xyzyaw[seed_ids] = lidar_xyzyaw[first]
            self.pipeline.seed_episode(
                seed_ids,
                lidar_xyzyaw[first, 0], lidar_xyzyaw[first, 1], lidar_xyzyaw[first, 2], base_yaw[first],
                floor_xy=base_pos[first, 0:2],
                floor_z=base_pos[first, 2] - self.cfg.elevation_offset,
                floor_half_m=0.5 * self.cfg.reset_floor_square_size,
                floor_var=self.cfg.reset_floor_unc,
            )
            self._is_first_update[seed_ids] = False

        # advance by the measurement-to-measurement delta: re-anchor to the
        # previous measurement pose first (the high-frequency query in `data`
        # leaves pipeline.pose at the base), so step lands exactly on the
        # current lidar pose for both placement and the z reference
        last = self.last_xyzyaw[ids]
        d_tf = self._rel_to_last_in_current_frame(curr=lidar_xyzyaw, last=last)
        self.pipeline.set_pose(last[:, 0], last[:, 1], last[:, 2], last[:, 3], indices=ids)
        self.pipeline.step(d_tf[:, 0], d_tf[:, 1], d_tf[:, 2], d_tf[:, 3],
                           batch_pred, batch_logv, ids=ids)

        self.last_xyzyaw[ids] = lidar_xyzyaw

    @staticmethod
    def _yaw_from_q(q: torch.Tensor) -> torch.Tensor:
        """Extract yaw from quaternion(s) in (w,x,y,z) convention."""
        if q.dim() == 1:
            w, x, y, z = q[0], q[1], q[2], q[3]
        else:
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return torch.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _rel_to_last_in_current_frame(curr: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        """Delta (dx, dy, dz, dyaw) from last to curr, expressed in the current frame."""
        dx = curr[:, 0] - last[:, 0]
        dy = curr[:, 1] - last[:, 1]
        dz = curr[:, 2] - last[:, 2]
        dyaw = curr[:, 3] - last[:, 3]
        dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi

        yaw = curr[:, 3]
        c, s = torch.cos(-yaw), torch.sin(-yaw)
        dx_b = c * dx - s * dy
        dy_b = s * dx + c * dy
        return torch.stack([dx_b, dy_b, dz, dyaw], dim=-1)


@configclass
class LivoxNeuralMapSensorCfg(SensorBaseCfg):
    """Configuration for the livox (Mid360) neural map sensor."""

    class_type: type = LivoxNeuralMapSensor

    lidar_sensor_name: str = "lidar_cam"
    """Scene name of the :class:`Mid360RayCaster` providing the scans."""

    jit_model_path: str = "ame2/NN_models/g1LivoxMapping_perframe.pt"
    """Per-frame mode-selection mapping model (TorchScript)."""

    # model input grid (201 x 101 @ 0.03 m)
    gt_x_range: tuple[float, float] = (-3.0, 3.0)
    gt_y_range: tuple[float, float] = (-1.5, 1.5)
    res_s: float = 0.03

    # export grid (matches the teacher height-scan grid: x in [-0.5, 1.5],
    # y in [-0.56, 0.56] @ 0.08 -> 26 x 15)
    L_base: int = 26
    W_base: int = 15
    res_b: float = 0.08
    x_mid_b: float = 0.5

    # fuser (map res == model grid res)
    pipeline_size_m: float = 6.0
    init_var: float = 10.0
    tau_min: float = 0.5
    # anti-erase guard (protects a confident high surface from a low close-range
    # "ground" return); disabled by default, deployment uses overwrite_drop_m = 0.15
    overwrite_drop_m: float = 0.0
    overwrite_std_max: float = 0.2
    ground_reject_slope: float = 0.8

    void_std: float = 2.7
    """Std assigned to void (-3) model outputs before fusion."""

    filter_estimation_std: float | None = None
    """If set, predictions with std above this are voided entirely."""

    soft_export: bool = False
    soft_std_ratio: float = 0.5
    soft_min_confident: int = 3

    inference_chunk: int = 2048
    """Max envs per JIT forward (bounds transient activation memory). 0 = single batch."""

    elevation_offset: float = 0.74
    """Nominal base height over ground; used only to place the spawn floor square."""

    reset_floor_square_size: float = 0.0
    """Full side length (m) of the confident floor square stamped at spawn (0 = off)."""
    reset_floor_unc: float = 0.04

    query_drift_range: float = 0.0
    """Episode-consistent xyz drift applied to map queries (odometry drift DR)."""

    clean_source_ratio: float = 0.0
    clean_source_noise: float = 0.02
    clean_source_unc: float = 0.05
    critic_sensor_name: str = "height_scanner_critic"

    random_pixel_ratio: float = 0.0

    keep_debug_buffers: bool = False
    """Store the per-scan EM grid and raw hits for play.py visualization
    (E x 201 x 101 floats — enable only for play-scale env counts)."""
