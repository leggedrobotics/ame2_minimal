import math
from typing import Tuple, Optional
import torch
import torch.nn.functional as F

# -------------------- Pose utils --------------------

def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2*math.pi) - math.pi

def advance_pose_world_batch(x, y, z, yaw, dx_l, dy_l, dz, dyaw):
    """
    All tensors are shape (B,). Local-frame deltas to world-frame pose update.
    """
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx_w =  c * dx_l - s * dy_l
    dy_w =  s * dx_l + c * dy_l
    x = x + dx_w
    y = y + dy_w
    z = z + dz
    yaw = wrap_angle(yaw + dyaw)
    return x, y, z, yaw

def mesh_local_to_world_batched(X_local: torch.Tensor, Y_local: torch.Tensor,
                                x: torch.Tensor, y: torch.Tensor, yaw: torch.Tensor):
    """
    X_local, Y_local: (L, W) shared across batch; x,y,yaw: (B,)
    Returns: Xw, Yw of shape (B, L, W)
    """
    B = x.shape[0]
    c = torch.cos(yaw).view(B, 1, 1)
    s = torch.sin(yaw).view(B, 1, 1)
    Xl = X_local.unsqueeze(0)  # (1,L,W)
    Yl = Y_local.unsqueeze(0)  # (1,L,W)
    Xw =  c * Xl - s * Yl + x.view(B, 1, 1)
    Yw =  s * Xl + c * Yl + y.view(B, 1, 1)
    return Xw, Yw

# -------------------- Global grid (batched) --------------------

class GlobalGrid:
    """
    World-aligned grids centered at (0,0), one per batch.
    mean,var stored as (B,1,Ny,Nx). Each batch has its own origin x_min[b], y_min[b].
    """
    def __init__(self, *, batch_size: int,
                 size_m: float = 20.0, res: float = 0.08,
                 init_var: float = 100.0, device: torch.device = torch.device("cpu"),
                 h_bias: float = -0.5):
        self.device = device
        self.res = float(res)
        self.size_m = float(size_m)
        self.Nx = int(round(self.size_m / self.res))
        self.Ny = int(round(self.size_m / self.res))
        self.h_bias = float(h_bias)

        B = batch_size
        start_min = -0.5 * self.size_m + 0.5 * self.res
        # per-batch world origins of the (0,0) grid center:
        self.x_min = torch.full((B,), start_min, device=device)
        self.y_min = torch.full((B,), start_min, device=device)

        self.mean = torch.zeros(B, 1, self.Ny, self.Nx, device=self.device) + self.h_bias
        self.var  = torch.full((B, 1, self.Ny, self.Nx), float(init_var), device=self.device)
        self.init_var = float(init_var)

    # centers of each batch's map in world coords (B,)
    def centers(self) -> Tuple[torch.Tensor, torch.Tensor]:
        half_span = 0.5 * (self.Nx - 1) * self.res
        cx = self.x_min + half_span
        cy = self.y_min + half_span
        return cx, cy

    # ----- helpers: world <-> index / norm -----

    def world_to_index(self, Xw: torch.Tensor, Yw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ix = torch.round((Xw - self.x_min.view(-1,1,1)) / self.res).to(torch.long)
        iy = torch.round((Yw - self.y_min.view(-1,1,1)) / self.res).to(torch.long)
        return ix, iy

    def world_to_norm(self, Xw: torch.Tensor, Yw: torch.Tensor) -> torch.Tensor:
        lx = (Xw - self.x_min.view(-1,1,1)) / self.res
        ly = (Yw - self.y_min.view(-1,1,1)) / self.res
        x_norm = 2.0 * (lx / max(self.Nx - 1, 1)) - 1.0
        y_norm = 2.0 * (ly / max(self.Ny - 1, 1)) - 1.0
        return torch.stack([x_norm, y_norm], dim=-1).contiguous()

    # ----- lightweight recentring (per selected batches) -----
    
    @torch.no_grad()
    def recenter(self, mask: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
                 fill_world: torch.Tensor | None = None):
        """
        mask: (B,) bool — which batches to recenter.
        cx, cy: (B,) desired centers (e.g., robot poses).
        fill_world: (B,) per-env world height for the newly uncovered stripes; if
            None, the world-constant ``self.h_bias`` is used. Pass ``z + h_bias``
            for a base-relative prior so new stripes track the robot's height.
        Rolls map so that (cx,cy) becomes the map center for masked batches,
        filling newly uncovered stripes with the prior mean and var=init_var.
        """
        if not mask.any():
            return

        # Ensure dtypes/devices match internal buffers
        cx = cx.to(dtype=self.x_min.dtype, device=self.x_min.device)
        cy = cy.to(dtype=self.y_min.dtype, device=self.y_min.device)
        if fill_world is not None:
            fill_world = fill_world.to(dtype=self.mean.dtype, device=self.mean.device)
        res = torch.as_tensor(self.res, dtype=self.x_min.dtype, device=self.x_min.device)
        halfspan = 0.5 * self.size_m - 0.5 * self.res  # = (Nx-1)/2 * res

        # Target origins for each batch to put (cx,cy) at map center
        x_min_new_all = cx - halfspan
        y_min_new_all = cy - halfspan

        for b in torch.nonzero(mask, as_tuple=False).flatten().tolist():
            x_min_new = x_min_new_all[b]
            y_min_new = y_min_new_all[b]
            fb = self.h_bias if fill_world is None else float(fill_world[b].item())
    
            # integer cell shifts
            sx = int(((self.x_min[b] - x_min_new) / res).round().to(torch.int64).item())  # +sx → roll right
            sy = int(((self.y_min[b] - y_min_new) / res).round().to(torch.int64).item())  # +sy → roll down

            # Snap the origin to the cell lattice (x_min - sx*res), not to (cx - halfspan):
            # only whole cells are rolled, so a continuous origin would leave a sub-cell
            # offset that drifts the rolled map every recenter. The center stays within
            # half a cell of (cx,cy).
            self.x_min[b] = self.x_min[b] - sx * self.res
            self.y_min[b] = self.y_min[b] - sy * self.res

            if sx == 0 and sy == 0:
                continue

            # Roll this batch's map
            m = torch.roll(self.mean[b:b+1], shifts=(sy, sx), dims=(-2, -1))
            v = torch.roll(self.var[b:b+1],  shifts=(sy, sx), dims=(-2, -1))

            # Fill newly uncovered stripes
            if sx > 0:
                m[:, :, :, :sx] = fb; v[:, :, :, :sx] = self.init_var
            elif sx < 0:
                m[:, :, :, sx:] = fb; v[:, :, :, sx:] = self.init_var
            if sy > 0:
                m[:, :, :sy, :] = fb; v[:, :, :sy, :] = self.init_var
            elif sy < 0:
                m[:, :, sy:, :] = fb; v[:, :, sy:, :] = self.init_var

            self.mean[b:b+1] = m
            self.var[b:b+1]  = v
    
    @torch.no_grad()
    def reset(self, mask: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor,
              fill_world: torch.Tensor | None = None, yaw: torch.Tensor | None = None,
              floor_half_m: float = 0.0, floor_var: float | None = None):
        """
        Hard reset for selected batches:
          - recenter so that (cx,cy) is map center
          - set the whole map to the prior mean (fill_world or h_bias) at var=init_var
          - optionally stamp a confident floor square around (cx,cy), rotated to
            ``yaw`` (base frame), with the same mean but low variance.
        mask: (B,) bool ; cx, cy, yaw, fill_world: (B,) (e.g., current robot pose)
        floor_half_m: half side length of the floor square in metres (0 = none).
        floor_var: variance for the floor square cells (low = confident).
        """
        if not mask.any():
            return
        cx = cx.to(dtype=self.x_min.dtype, device=self.x_min.device)
        cy = cy.to(dtype=self.y_min.dtype, device=self.y_min.device)
        halfspan = 0.5 * self.size_m - 0.5 * self.res
        x_min_new_all = cx - halfspan
        y_min_new_all = cy - halfspan

        # Update origins for masked batches
        self.x_min[mask] = x_min_new_all[mask]
        self.y_min[mask] = y_min_new_all[mask]
        # Reset tiles to the prior
        if fill_world is None:
            self.mean[mask] = self.h_bias
        else:
            self.mean[mask] = fill_world.to(self.mean.dtype)[mask].view(-1, 1, 1, 1)
        self.var[mask]  = self.init_var

        # Confident floor square around the robot, rotated to base yaw so it
        # carries no world-yaw information into the (base-frame) export.
        if floor_half_m > 0.0 and floor_var is not None and yaw is not None and fill_world is not None:
            self._stamp_floor_square(mask, cx, cy, yaw, fill_world, float(floor_half_m), float(floor_var))

    @torch.no_grad()
    def _stamp_floor_square(self, mask, cx, cy, yaw, fill_world, half_m, floor_var):
        """Stamp a yaw-rotated square of confident floor (mean=fill_world, var=floor_var)."""
        ids = torch.nonzero(mask, as_tuple=False).flatten()  # (M,)
        if ids.numel() == 0:
            return
        res = self.res
        half_cells = int(round(half_m / res))
        if half_cells <= 0:
            return
        dt = self.mean.dtype
        offs = torch.arange(-half_cells, half_cells + 1, device=self.device, dtype=dt) * res  # (n,)
        ox, oy = torch.meshgrid(offs, offs, indexing="ij")
        ox = ox.reshape(-1)  # (P,)  base-frame x offsets
        oy = oy.reshape(-1)  # (P,)  base-frame y offsets
        yaw_m = yaw.to(dt)[ids].unsqueeze(1)        # (M,1)
        c, s = torch.cos(yaw_m), torch.sin(yaw_m)
        wx = c * ox.unsqueeze(0) - s * oy.unsqueeze(0)   # (M,P) world offsets
        wy = s * ox.unsqueeze(0) + c * oy.unsqueeze(0)
        world_x = cx.to(dt)[ids].unsqueeze(1) + wx       # (M,P)
        world_y = cy.to(dt)[ids].unsqueeze(1) + wy
        ix = torch.round((world_x - self.x_min[ids].unsqueeze(1)) / res).long().clamp_(0, self.Nx - 1)
        iy = torch.round((world_y - self.y_min[ids].unsqueeze(1)) / res).long().clamp_(0, self.Ny - 1)
        env_idx = ids.unsqueeze(1).expand_as(ix)         # (M,P)
        vals = fill_world.to(dt)[ids].unsqueeze(1).expand_as(ix)
        self.mean[env_idx.reshape(-1), 0, iy.reshape(-1), ix.reshape(-1)] = vals.reshape(-1)
        self.var[env_idx.reshape(-1), 0, iy.reshape(-1), ix.reshape(-1)] = floor_var

# -------------------- Batched pipeline with auto recentring + reset --------------------

class BatchedGlobalProbWinnerPipeline:
    """
    One global map per batch (B).
    - Auto recentre batches whose robot pose is within a margin to the map border.
    - Nearest write + probabilistic winner (only-if-better, capped by tau_min).
    - Export (bilinear mean, nearest var) to base mesh per batch.
    - Reset(mask_b): reset selected batches to zero-mean, init variance and recenter at current pose.
    """

    def __init__(self,
                 Xs_local: torch.Tensor, Ys_local: torch.Tensor,   # (Ls,Ws), sensor grid (base frame)
                 *,
                 batch_size: int,
                 res: float = 0.08,
                 size_m: float = 20.0,
                 init_var: float = 10.0,
                 tau_min: float = 0.5,                 # cap: R_eff = max(R, tau_min * P_prior)
                 recenter_margin_ratio: float = 0.3,    # fraction of map size; e.g., 0.3 → recenter near outer 30%
                 device: torch.device = torch.device("cpu"),
                 h_bias: float = -0.5):
        self.device = device
        self.grid = GlobalGrid(batch_size=batch_size, size_m=size_m, res=res, init_var=init_var, device=device, h_bias=h_bias)
        self.h_bias = float(h_bias)
        self.Xs_local = Xs_local.to(device)
        self.Ys_local = Ys_local.to(device)
        self.tau_min = float(tau_min)

        # per-batch base pose (world)
        B = batch_size
        self.x   = torch.zeros(B, device=device)
        self.y   = torch.zeros(B, device=device)
        self.z   = torch.zeros(B, device=device)
        self.yaw = torch.zeros(B, device=device)

        self.recenter_margin_ratio = float(recenter_margin_ratio)

    @torch.no_grad()
    def _auto_recentre_if_needed(self):
        cx_map, cy_map = self.grid.centers()            # (B,)
        half = 0.5 * self.grid.size_m
        margin = self.recenter_margin_ratio * self.grid.size_m
        dx = (self.x - cx_map).abs()
        dy = (self.y - cy_map).abs()
        need = (dx > (half - margin)) | (dy > (half - margin))  # (B,)
        if need.any():
            # base-relative prior: new stripes get (z + h_bias) at high uncertainty
            self.grid.recenter(need, self.x, self.y, fill_world=self.z + self.h_bias)

    @torch.no_grad()
    def set_pose(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, yaw: torch.Tensor, indices: Optional[torch.Tensor] = None):
        """Force the internal world pose tracker to match the provided values."""
        if indices is None:
            self.x[:] = x[:]
            self.y[:] = y[:]
            self.z[:] = z[:]
            self.yaw[:] = yaw[:]
        else:
            self.x[indices] = x
            self.y[indices] = y
            self.z[indices] = z
            self.yaw[indices] = yaw

    @torch.no_grad()
    def reset(self, mask_b: torch.Tensor):
        """
        Reset selected batch maps (mask_b: (B,) bool) to initial values,
        recentered on the *current pose* of each selected batch. This runs at
        sensor-reset time when the pose tracker may still be stale; the
        authoritative spawn-aligned seeding happens in :meth:`seed_episode`.
        """
        mask_b = mask_b.to(dtype=torch.bool, device=self.device)
        self.grid.reset(mask_b, self.x, self.y, fill_world=self.z + self.h_bias)

    @torch.no_grad()
    def seed_episode(self, idx: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                     z: torch.Tensor, yaw: torch.Tensor,
                     floor_half_m: float = 0.0, floor_var: float | None = None):
        """Spawn-aligned reset: set the pose for ``idx`` to the fresh spawn pose,
        then re-seed those maps with the base-relative prior (z + h_bias at high
        uncertainty) plus a confident floor square around the robot (low var,
        rotated to base yaw). Call from the sensor's first-update block where the
        true per-episode spawn pose is available."""
        self.set_pose(x, y, z, yaw, indices=idx)
        mask = torch.zeros(self.grid.mean.shape[0], dtype=torch.bool, device=self.device)
        mask[idx] = True
        self.grid.reset(
            mask, self.x, self.y, fill_world=self.z + self.h_bias, yaw=self.yaw,
            floor_half_m=floor_half_m, floor_var=floor_var,
        )

    @torch.no_grad()
    def step(self,
             dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor, dyaw: torch.Tensor,  # (B,)
             mu_s: torch.Tensor, logv_s: torch.Tensor                                   # (B,Ls,Ws) or (B,1,Ls,Ws)
             ):
        """
        Nearest update + probabilistic winner per impacted global cell (per batch).
        If multiple sensor pixels map to same cell this step, last write wins (undefined order on ties).
        """
        # normalize shapes
        if mu_s.ndim == 3:   mu_s = mu_s.unsqueeze(1)    # → (B,1,Ls,Ws)
        if logv_s.ndim == 3: logv_s = logv_s.unsqueeze(1)
        B, _, Ls, Ws = mu_s.shape
        Ny, Nx = self.grid.Ny, self.grid.Nx

        # integrate poses per batch
        self.x, self.y, self.z, self.yaw = advance_pose_world_batch(self.x, self.y, self.z, self.yaw,
                                                                    dx, dy, dz, dyaw)
        mu_s = mu_s + self.z.view(B, 1, 1, 1)
        # auto recentre selected batches
        self._auto_recentre_if_needed()

        var_s = torch.exp(logv_s)  # (B,1,Ls,Ws) interpret as log-variance

        # sensor mesh (world) per batch: (B,Ls,Ws)
        Xw, Yw = mesh_local_to_world_batched(self.Xs_local, self.Ys_local, self.x, self.y, self.yaw)

        # nearest indices on global grid, per batch
        ix, iy = self.grid.world_to_index(Xw, Yw)                       # (B,Ls,Ws) long
        inb = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny)             # (B,Ls,Ws)
        if not inb.any():
            return  # nothing overlaps global maps this step

        # Flatten selection over (B,Ls,Ws)
        base_offset = (torch.arange(B, device=self.device).view(B, 1, 1) * (Ny * Nx)).to(torch.long)
        lin_idx_all = base_offset + (iy * Nx + ix)  # (B,Ls,Ws)
        lin_idx = lin_idx_all[inb].reshape(-1)      # (N,)

        # Gather prior at those cells (flatten over B maps)
        mean_flat = self.grid.mean.view(-1)         # (B*Ny*Nx,)
        var_flat  = self.grid.var.view(-1)
        M_prior = mean_flat[lin_idx]                # (N,)
        P_prior = var_flat[lin_idx]                 # (N,)

        # Measurements at those sensor pixels (collapse channel=1)
        m_meas = mu_s[:, 0, :, :][inb].reshape(-1)     # (N,)
        R_meas = var_s[:, 0, :, :][inb].reshape(-1)    # (N,)

        # --- winner with guards (simple & robust) ---
        eps = torch.finfo(m_meas.dtype).eps
        # Cap measurement precision relative to prior
        R_eff = torch.maximum(R_meas, self.tau_min * P_prior)
        # Only allow overwrite if measurement is actually better
        better = torch.logical_or(R_eff < P_prior * (1+self.tau_min), R_eff<0.04)

        # probabilistic winner only on 'better' pixels
        p_meas = torch.zeros_like(R_eff)
        if better.any():
            prec_eff   = 1.0 / (R_eff[better]   + eps)
            prec_prior = 1.0 / (P_prior[better] + eps)
            p_meas_b = prec_eff / (prec_eff + prec_prior + eps)
            p_meas[better] = p_meas_b

        take = (torch.rand_like(p_meas) < p_meas)

        # write back
        M_new = torch.where(take, m_meas, M_prior)
        P_new = torch.where(take, R_eff,  P_prior)
        mean_flat[lin_idx] = M_new
        var_flat[lin_idx]  = P_new

    @torch.no_grad()
    def export_to_base(
        self,
        Xb_local: torch.Tensor,          # (Lb, Wb)
        Yb_local: torch.Tensor,          # (Lb, Wb)
        drift_xy: Optional[torch.Tensor] = None,  # (B, 2) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Xb_local, Yb_local: (Lb, Wb) base-frame mesh (shared across batch).
            drift_xy: (B, 2) per-env world drift (dx, dy) applied to query points; or None.
        Returns:
            mean_base: (B,1,Lb,Wb)
            var_base:  (B,1,Lb,Wb)
        """
        Xb_local = Xb_local.to(self.device)
        Yb_local = Yb_local.to(self.device)
        # world base meshes per batch: (B, Lb, Wb)
        Xw_b, Yw_b = mesh_local_to_world_batched(Xb_local, Yb_local, self.x, self.y, self.yaw)
        # per-env drift
        if drift_xy is not None:
            B = Xw_b.shape[0]
            drift_xy = drift_xy.to(device=self.device, dtype=Xw_b.dtype)
            assert drift_xy.shape == (B, 2), f"drift_xy must be (B,2), got {tuple(drift_xy.shape)}"
            dx_b, dy_b = drift_xy[:, 0], drift_xy[:, 1]
            Xw_b = Xw_b + dx_b.view(B, 1, 1)
            Yw_b = Yw_b + dy_b.view(B, 1, 1)

        grid = self.grid.world_to_norm(Xw_b, Yw_b)  # (B,Lb,Wb,2)

        mean_b = F.grid_sample(self.grid.mean, grid, mode="bilinear",
                               padding_mode="zeros", align_corners=True)  # (B,1,Lb,Wb)
        var_b  = F.grid_sample(self.grid.var,  grid, mode="nearest",
                               padding_mode="zeros", align_corners=True)  # (B,1,Lb,Wb)

        mean_b = mean_b - self.z.view(mean_b.shape[0], 1, 1, 1)
        mean_b = mean_b.clamp(-2.0, 2.0)
        return mean_b, var_b

    @torch.no_grad()
    def export_to_base_soft(
        self, 
        Xb_local: torch.Tensor, Yb_local: torch.Tensor,
        std_ratio: float = 0.5, min_confident: int = 3,
        drift_xy: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Drop-in replacement for export_to_base (same args incl. drift_xy, and same
        return), but first builds a confidence-infill copy of the stored map
        (self.grid.mean/var is left untouched) and exports from that copy.

        Filter (fully vectorized over the whole grid; 1-neighbourhood = the 8 surrounding cells):
        for every cell, count how many of its 8 neighbours have uncertainty std
        < std_ratio * (this cell's std). If that count >= min_confident (default >= 3 of 8),
        replace the cell with the precision(1/var)-weighted mean of those 'confident'
        neighbours, and set its variance to the fused 1 / sum(1/var) accordingly.
        Cells that don't meet the rule (and out-of-grid neighbours) are left unchanged.

        Returns (mean_base, var_base) exactly as export_to_base.
        """
        eps = 1e-8
        mean = self.grid.mean                       # (B,1,Ny,Nx)
        var  = self.grid.var
        std  = var.clamp_min(0.0).sqrt()
        Ny, Nx = self.grid.Ny, self.grid.Nx

        # pad by 1; out-of-grid neighbours get huge std/var so they never count as 'confident'
        BIG = 1e12
        mean_p = F.pad(mean, (1, 1, 1, 1), mode="constant", value=0.0)
        var_p  = F.pad(var,  (1, 1, 1, 1), mode="constant", value=BIG)
        std_p  = F.pad(std,  (1, 1, 1, 1), mode="constant", value=BIG)

        thresh     = std_ratio * std                # per-cell: 0.5 * (center std)
        conf_count = torch.zeros_like(mean)
        prec_sum   = torch.zeros_like(mean)         # sum of 1/var over confident neighbours
        mprec_sum  = torch.zeros_like(mean)         # sum of mean/var over confident neighbours
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue                        # exclude the centre cell itself
                nb_m = mean_p[:, :, 1 + dy:1 + dy + Ny, 1 + dx:1 + dx + Nx]
                nb_v = var_p [:, :, 1 + dy:1 + dy + Ny, 1 + dx:1 + dx + Nx]
                nb_s = std_p [:, :, 1 + dy:1 + dy + Ny, 1 + dx:1 + dx + Nx]
                confident = (nb_s < thresh).to(mean.dtype)
                conf_count += confident
                prec = confident / (nb_v + eps)
                prec_sum  += prec
                mprec_sum += prec * nb_m

        replace    = conf_count >= float(min_confident)            # >= 3 of 8 by default
        fused_mean = mprec_sum / prec_sum.clamp_min(eps)
        fused_var  = 1.0 / prec_sum.clamp_min(eps)
        new_mean = torch.where(replace, fused_mean, mean)
        new_var  = torch.where(replace, fused_var,  var)

        # export from the filtered copy (same math as export_to_base; self.grid not modified)
        Xb_local = Xb_local.to(self.device)
        Yb_local = Yb_local.to(self.device)
        Xw_b, Yw_b = mesh_local_to_world_batched(Xb_local, Yb_local, self.x, self.y, self.yaw)  # (B,Lb,Wb)
        # per-env drift (identical to export_to_base)
        if drift_xy is not None:
            B = Xw_b.shape[0]
            drift_xy = drift_xy.to(device=self.device, dtype=Xw_b.dtype)
            assert drift_xy.shape == (B, 2), f"drift_xy must be (B,2), got {tuple(drift_xy.shape)}"
            dx_b, dy_b = drift_xy[:, 0], drift_xy[:, 1]
            Xw_b = Xw_b + dx_b.view(B, 1, 1)
            Yw_b = Yw_b + dy_b.view(B, 1, 1)
        grid = self.grid.world_to_norm(Xw_b, Yw_b)                                              # (B,Lb,Wb,2)
        mean_b = F.grid_sample(new_mean, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        var_b  = F.grid_sample(new_var,  grid, mode="nearest",  padding_mode="zeros", align_corners=True)
        mean_b = mean_b - self.z.view(mean_b.shape[0], 1, 1, 1)
        mean_b = mean_b.clamp(-2.0, 2.0)
        return mean_b, var_b

    @torch.no_grad()
    def export_global(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full global tensors: (B,1,Ny,Nx)."""
        return self.grid.mean, self.grid.var


# -------------------- Livox (Mid360) pipeline --------------------

LIVOX_HOLE = -3.0
"""Hole / unobserved sentinel shared by the lidar mapping model, its fuser and
the model input grid."""


class LivoxGlobalProbWinnerPipeline:
    """Fuser for the Mid360 G1 mapping model.

    Differences from :class:`BatchedGlobalProbWinnerPipeline` (co-designed with
    the mode-selection model):

    * unseen cells are the HOLE sentinel (-3), not a base-relative floor prior;
    * prediction pixels at/below the hole sentinel are never fused (the model
      says "void", the map keeps its knowledge);
    * anti-erase guard: a low close-range "ground" return cannot overwrite an
      established confident higher surface (e.g. a ladder rung), as on the
      deployed system;
    * export samples the mean with nearest (bilinear blends real cells with the
      -3 background into ghost halos around thin structures), clamps to
      [-3, 2], and treats out-of-map queries as unseen (-3, init_var) instead
      of a confident phantom platform.

    Pose convention: :meth:`step` is anchored to the lidar pose (xy/z of the
    Mid360 origin, base yaw; the frame the model input grid is built in), while
    exports are queried after :meth:`set_pose` to the base pose, giving
    base-relative heights.
    """

    def __init__(self,
                 Xs_local: torch.Tensor, Ys_local: torch.Tensor,   # (Ls,Ws) model grid (base-yaw frame)
                 *,
                 batch_size: int,
                 res: float = 0.03,
                 size_m: float = 6.0,
                 init_var: float = 10.0,
                 tau_min: float = 0.5,                  # cap: R_eff = max(R, tau_min * P_prior)
                 recenter_margin_ratio: float = 0.3,
                 hole_value: float = LIVOX_HOLE,        # sensor pixels at/below this are not fused
                 # anti-erase guard (0 disables; deployment uses 0.15)
                 overwrite_drop_m: float = 0.0,         # stored surface must sit this far above the new pt
                 overwrite_std_max: float = 0.2,        # ... and be confident (std < this) to be protected
                 ground_reject_slope: float = 0.8,      # reject if vertical drop > slope * horizontal dist
                 device: torch.device = torch.device("cpu")):
        self.device = device
        # h_bias = hole sentinel: recenter/reset fills (fill_world=None) are void
        self.grid = GlobalGrid(batch_size=batch_size, size_m=size_m, res=res,
                               init_var=init_var, device=device, h_bias=hole_value)
        self.Xs_local = Xs_local.to(device)
        self.Ys_local = Ys_local.to(device)
        self.tau_min = float(tau_min)
        self.hole_value = float(hole_value)
        self.overwrite_drop_m = float(overwrite_drop_m)
        self.overwrite_std_max = float(overwrite_std_max)
        self.ground_reject_slope = float(ground_reject_slope)
        self.recenter_margin_ratio = float(recenter_margin_ratio)

        B = batch_size
        self.x   = torch.zeros(B, device=device)
        self.y   = torch.zeros(B, device=device)
        self.z   = torch.zeros(B, device=device)
        self.yaw = torch.zeros(B, device=device)

    @torch.no_grad()
    def _auto_recentre_if_needed(self):
        cx_map, cy_map = self.grid.centers()            # (B,)
        half = 0.5 * self.grid.size_m
        margin = self.recenter_margin_ratio * self.grid.size_m
        dx = (self.x - cx_map).abs()
        dy = (self.y - cy_map).abs()
        need = (dx > (half - margin)) | (dy > (half - margin))  # (B,)
        if need.any():
            self.grid.recenter(need, self.x, self.y)    # new stripes -> hole sentinel

    @torch.no_grad()
    def set_pose(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, yaw: torch.Tensor,
                 indices: Optional[torch.Tensor] = None):
        """Force the internal world pose tracker to match the provided values."""
        if indices is None:
            self.x[:] = x[:]
            self.y[:] = y[:]
            self.z[:] = z[:]
            self.yaw[:] = yaw[:]
        else:
            self.x[indices] = x
            self.y[indices] = y
            self.z[indices] = z
            self.yaw[indices] = yaw

    @torch.no_grad()
    def reset(self, mask_b: torch.Tensor):
        """Reset selected batch maps to all-unseen (-3), recentered on the
        current tracked pose. Runs at sensor-reset time when the tracker may be
        stale; the authoritative spawn seeding is :meth:`seed_episode`."""
        mask_b = mask_b.to(dtype=torch.bool, device=self.device)
        self.grid.reset(mask_b, self.x, self.y)         # fill_world=None -> hole sentinel

    @torch.no_grad()
    def seed_episode(self, idx: torch.Tensor,
                     x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, yaw: torch.Tensor,
                     floor_xy: Optional[torch.Tensor] = None,   # (n,2) base world xy for the stamp
                     floor_z: Optional[torch.Tensor] = None,    # (n,) ground world height under spawn
                     floor_half_m: float = 0.0, floor_var: float = 0.04):
        """Spawn-aligned reset: pose(idx) <- the fresh lidar spawn pose, map <-
        all-unseen recentered there, plus (optionally) a confident floor square
        stamped around the base spawn xy at the true ground height."""
        self.set_pose(x, y, z, yaw, indices=idx)
        mask = torch.zeros(self.grid.mean.shape[0], dtype=torch.bool, device=self.device)
        mask[idx] = True
        self.grid.reset(mask, self.x, self.y)           # all-unseen
        if floor_half_m > 0.0 and floor_xy is not None and floor_z is not None:
            cx = self.x.clone(); cy = self.y.clone()
            fz = torch.zeros_like(self.z)
            cx[idx] = floor_xy[:, 0]
            cy[idx] = floor_xy[:, 1]
            fz[idx] = floor_z
            self.grid._stamp_floor_square(mask, cx, cy, self.yaw, fz,
                                          float(floor_half_m), float(floor_var))

    @torch.no_grad()
    def step(self,
             dx: torch.Tensor, dy: torch.Tensor, dz: torch.Tensor, dyaw: torch.Tensor,  # (n,)
             mu_s: torch.Tensor, logv_s: torch.Tensor,                # (n,Ls,Ws) or (n,1,Ls,Ws)
             ids: Optional[torch.Tensor] = None,                      # (n,) env indices; None = all
             ):
        """Nearest write + probabilistic winner per impacted global cell.

        ``mu_s`` is the per-frame model output: sensor(z)-relative heights in
        the base-yaw grid, holes = -3. Hole pixels never update the map.

        ``ids`` selects which envs this step's data belongs to (deltas and
        mu/logv are subset-sized in the same order); other envs' maps and poses
        are untouched. Isaac Lab sensors update per env on their own post-reset
        phase, so partial updates must not re-fuse all envs.
        """
        if mu_s.ndim == 3:   mu_s = mu_s.unsqueeze(1)    # -> (n,1,Ls,Ws)
        if logv_s.ndim == 3: logv_s = logv_s.unsqueeze(1)
        n, _, Ls, Ws = mu_s.shape
        Ny, Nx = self.grid.Ny, self.grid.Nx
        if ids is None:
            ids = torch.arange(self.grid.mean.shape[0], device=self.device)
        assert ids.numel() == n, f"ids ({ids.numel()}) must match the batch of mu_s ({n})"

        # blind/void prediction pixels must not update the map; evaluate on the
        # raw (sensor-relative) prediction, before the world-z shift
        valid_pix = mu_s[:, 0] > (self.hole_value + 0.5)                # (n,Ls,Ws)

        # integrate poses for the updated envs
        xs, ys, zs, yaws = advance_pose_world_batch(
            self.x[ids], self.y[ids], self.z[ids], self.yaw[ids], dx, dy, dz, dyaw)
        self.x[ids], self.y[ids], self.z[ids], self.yaw[ids] = xs, ys, zs, yaws
        mu_s = mu_s + zs.view(n, 1, 1, 1)
        self._auto_recentre_if_needed()

        var_s = torch.exp(logv_s)  # (n,1,Ls,Ws) log-variance -> variance

        # sensor mesh (world) per updated env: (n,Ls,Ws)
        Xw, Yw = mesh_local_to_world_batched(self.Xs_local, self.Ys_local, xs, ys, yaws)

        # nearest indices on each env's global grid (per-env origins, read AFTER
        # the recenter above so freshly rolled maps line up)
        res = self.grid.res
        ix = torch.round((Xw - self.grid.x_min[ids].view(-1, 1, 1)) / res).to(torch.long)
        iy = torch.round((Yw - self.grid.y_min[ids].view(-1, 1, 1)) / res).to(torch.long)
        inb = (ix >= 0) & (ix < Nx) & (iy >= 0) & (iy < Ny) & valid_pix
        if not inb.any():
            return

        base_offset = (ids.view(n, 1, 1) * (Ny * Nx)).to(torch.long)
        lin_idx = (base_offset + (iy * Nx + ix))[inb].reshape(-1)       # (N,)

        mean_flat = self.grid.mean.view(-1)
        var_flat  = self.grid.var.view(-1)
        M_prior = mean_flat[lin_idx]                # (N,)
        P_prior = var_flat[lin_idx]                 # (N,)

        m_meas = mu_s[:, 0, :, :][inb].reshape(-1)     # (N,)
        R_meas = var_s[:, 0, :, :][inb].reshape(-1)    # (N,)

        # --- winner with guards ---
        eps = torch.finfo(m_meas.dtype).eps
        R_eff = torch.maximum(R_meas, self.tau_min * P_prior)
        better = torch.logical_or(R_eff < P_prior * 1.25, R_eff < 0.04)

        p_meas = torch.zeros_like(R_eff)
        if better.any():
            prec_eff   = 1.0 / (R_eff[better]   + eps)
            prec_prior = 1.0 / (P_prior[better] + eps)
            p_meas[better] = prec_eff / (prec_eff + prec_prior + eps)

        take = (torch.rand_like(p_meas) < p_meas)

        # --- anti-erase guard: protect an established higher surface from a low
        # close-range "ground" return (e.g. sensor near a ladder sees the floor,
        # not the rung). Reject when: (1) the cell holds a real confident surface,
        # (2) it sits >= overwrite_drop_m above the new pt, (3) the new pt drops
        # steeply below the sensor (vertical > slope * horizontal), i.e. it looks
        # like close ground rather than a genuine lower step.
        if self.overwrite_drop_m > 0.0:
            rx = xs.view(n, 1, 1).expand(n, Ls, Ws)[inb]                   # (N,) fusion-pose world xyz
            ry = ys.view(n, 1, 1).expand(n, Ls, Ws)[inb]
            rz = zs.view(n, 1, 1).expand(n, Ls, Ws)[inb]
            Xw_sel, Yw_sel = Xw[inb], Yw[inb]                              # (N,) cell world xy
            protected = (M_prior > self.hole_value + 0.1) \
                        & (P_prior < self.overwrite_std_max ** 2) \
                        & ((M_prior - m_meas) >= self.overwrite_drop_m)
            horiz = torch.sqrt((Xw_sel - rx) ** 2 + (Yw_sel - ry) ** 2)
            close_ground = (rz - m_meas) > (self.ground_reject_slope * horiz)
            take = take & ~(protected & close_ground)

        M_new = torch.where(take, m_meas, M_prior)
        P_new = torch.where(take, R_eff,  P_prior)
        mean_flat[lin_idx] = M_new
        var_flat[lin_idx]  = P_new

    @torch.no_grad()
    def _export(self, mean_src: torch.Tensor, var_src: torch.Tensor,
                Xb_local: torch.Tensor, Yb_local: torch.Tensor,
                drift_xy: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared export math: nearest sampling, -z, clamp [-3,2], OOB -> unseen."""
        Xb_local = Xb_local.to(self.device)
        Yb_local = Yb_local.to(self.device)
        Xw_b, Yw_b = mesh_local_to_world_batched(Xb_local, Yb_local, self.x, self.y, self.yaw)
        if drift_xy is not None:
            B = Xw_b.shape[0]
            drift_xy = drift_xy.to(device=self.device, dtype=Xw_b.dtype)
            assert drift_xy.shape == (B, 2), f"drift_xy must be (B,2), got {tuple(drift_xy.shape)}"
            Xw_b = Xw_b + drift_xy[:, 0].view(B, 1, 1)
            Yw_b = Yw_b + drift_xy[:, 1].view(B, 1, 1)

        grid = self.grid.world_to_norm(Xw_b, Yw_b)  # (B,Lb,Wb,2)

        # nearest (not bilinear): bilinear blends real cells with the -3 unseen
        # background and produces ghost heights (halos/lines around thin bars)
        mean_b = F.grid_sample(mean_src, grid, mode="nearest",
                               padding_mode="zeros", align_corners=True)  # (B,1,Lb,Wb)
        var_b  = F.grid_sample(var_src,  grid, mode="nearest",
                               padding_mode="zeros", align_corners=True)

        mean_b = mean_b - self.z.view(mean_b.shape[0], 1, 1, 1)
        mean_b = mean_b.clamp(self.hole_value, 2.0)   # keep unseen (-3) distinct from terrain [-2,2]
        # OOB queries return 0/0 (zeros padding), i.e. a confident phantom platform
        # at height -z; treat them as unseen (hole mean + init var)
        oob = (grid.abs() > 1.0).any(dim=-1, keepdim=False).unsqueeze(1)   # (B,1,Lb,Wb)
        mean_b = torch.where(oob, torch.full_like(mean_b, self.hole_value), mean_b)
        var_b  = torch.where(oob, torch.full_like(var_b, self.grid.init_var), var_b)
        return mean_b, var_b

    @torch.no_grad()
    def export_to_base(self, Xb_local: torch.Tensor, Yb_local: torch.Tensor,
                       drift_xy: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the map at the current tracked pose (call set_pose first).

        Returns (mean (B,1,Lb,Wb) pose-z-relative clamped [-3,2], var (B,1,Lb,Wb)).
        """
        return self._export(self.grid.mean, self.grid.var, Xb_local, Yb_local, drift_xy)

    @torch.no_grad()
    def export_to_base_soft(self, Xb_local: torch.Tensor, Yb_local: torch.Tensor,
                            std_ratio: float = 0.5, min_confident: int = 3,
                            drift_xy: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """export_to_base from a confidence-infilled copy of the map: a cell with
        >= min_confident of its 8 neighbours at std < std_ratio * (its own std)
        is replaced by their precision-weighted fusion. Stored map untouched."""
        eps = 1e-8
        mean = self.grid.mean                       # (B,1,Ny,Nx)
        var  = self.grid.var
        std  = var.clamp_min(0.0).sqrt()
        Ny, Nx = self.grid.Ny, self.grid.Nx

        BIG = 1e12
        mean_p = F.pad(mean, (1, 1, 1, 1), mode="constant", value=0.0)
        var_p  = F.pad(var,  (1, 1, 1, 1), mode="constant", value=BIG)
        std_p  = F.pad(std,  (1, 1, 1, 1), mode="constant", value=BIG)

        thresh     = std_ratio * std
        conf_count = torch.zeros_like(mean)
        prec_sum   = torch.zeros_like(mean)
        mprec_sum  = torch.zeros_like(mean)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nb_m = mean_p[:, :, 1 + dy:1 + dy + Ny, 1 + dx:1 + dx + Nx]
                nb_v = var_p [:, :, 1 + dy:1 + dy + Ny, 1 + dx:1 + dx + Nx]
                nb_s = std_p [:, :, 1 + dy:1 + dy + Ny, 1 + dx:1 + dx + Nx]
                confident = (nb_s < thresh).to(mean.dtype)
                conf_count += confident
                prec = confident / (nb_v + eps)
                prec_sum  += prec
                mprec_sum += prec * nb_m

        replace    = conf_count >= float(min_confident)
        fused_mean = mprec_sum / prec_sum.clamp_min(eps)
        fused_var  = 1.0 / prec_sum.clamp_min(eps)
        new_mean = torch.where(replace, fused_mean, mean)
        new_var  = torch.where(replace, fused_var,  var)
        return self._export(new_mean, new_var, Xb_local, Yb_local, drift_xy)

    @torch.no_grad()
    def export_global(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full global tensors: (B,1,Ny,Nx)."""
        return self.grid.mean, self.grid.var


# -------------------- Misc utils --------------------
@torch.inference_mode()
def morph2d_same(x, k, op="dilate"):
    k = (k, k) if isinstance(k, int) else k
    pt, pb = (k[0]-1)//2, (k[0]-1) - (k[0]-1)//2
    pl, pr = (k[1]-1)//2, (k[1]-1) - (k[1]-1)//2
    y, dt = (-x if op[0]=='e' else x), x.dtype
    y = F.pad(y.float(), (pl, pr, pt, pb), mode="constant", value=float("-inf"))
    y = F.max_pool2d(y, kernel_size=k, stride=1)
    return (-y if op[0]=='e' else y).to(dt)

def upsample(img, new_L, new_W):
    img_up = F.interpolate(
        img,               # N x C x H x W
        size=(new_L, new_W),            # target H, W
        mode='nearest',
    )
    return img_up


def add_occlusion_straight(xnew, L, W, ignore_first: int = 0, tie_break: float = 0.05):
    """
    Minimal straight-ray (left->right) occlusion for a heightmap.
    - xnew: (T, B, L*W) flattened heightmap
    Returns: (T, B, L*W) masked map where only straight-ray-visible points remain;
             occluded points are set to -2.
    """
    T, B = xnew.shape[0], xnew.shape[1]
    x = xnew.clone().reshape(T, B, L, W)
    device, dtype = x.device, x.dtype
    # Prepare slopes tensor (height / dx) and -inf padding
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    slopes = torch.full((T, B, L, W), neg_inf, dtype=dtype, device=device)
    if ignore_first < L:
        # Distance from the first checked column (strictly positive)
        dx = (torch.arange(L, device=device, dtype=dtype) - ignore_first + 1).clamp_min(1)
        dx_valid = dx[ignore_first:].view(1, 1, -1, 1)              # (1,1,L-ignore_first,1)
        h_valid = x[:, :, ignore_first:, :] - tie_break              # nudge to break ties
        slopes[:, :, ignore_first:, :] = h_valid / dx_valid          # per-row "slope to viewer"
    # Prefix max of slopes along x (left->right); exclude current position
    prefix_max = torch.cummax(slopes, dim=2).values                  # (T,B,L,W)
    pad = neg_inf.expand(T, B, 1, W)
    prefix_max_excl = torch.cat([pad, prefix_max[:, :, :-1, :]], dim=2)
    # Visible if it is not lower than any prior slope on the same row
    visible = slopes >= prefix_max_excl
    # Output: keep only visible samples; everything else set to -2
    out = torch.full_like(x, -2.0)
    out[visible] = x[visible]
    return out.reshape(T, B, L * W)

def add_occlusion_straight_batched(x, ignore_first: int = 0, tie_break: float = 0.05, length_dim: int = 2):
    """
    Minimal straight-ray (near->far) occlusion for a heightmap (batched).

    Args:
        x: Tensor of shape (B, 1, L, W) or (B, 1, W, L).
        ignore_first: Number of leading cells to ignore for visibility computation.
        tie_break: Small value subtracted from heights to break ties.
        length_dim: The dimension index representing the Length (Depth) axis. Default is 2.

    Returns:
        Tensor of same shape as x with occluded points set to -2.
    """
    assert x.dim() == 4 and x.shape[1] == 1, "x must have shape (B,1,H,W)"
    B, _, H, W = x.shape
    device, dtype = x.device, x.dtype
    
    L = x.shape[length_dim]
    
    # Prepare slopes tensor (height / dx) and -inf padding
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=dtype)
    slopes = torch.full_like(x, neg_inf)
    
    if ignore_first < L:
        # Distance from the first checked column (strictly positive)
        dx = (torch.arange(L, device=device, dtype=dtype) - ignore_first + 1).clamp_min(1)
        
        # Reshape dx for broadcasting based on length_dim
        if length_dim == 2:
            dx_view = dx.view(1, 1, -1, 1) # (1,1,L,1)
        else:
            dx_view = dx.view(1, 1, 1, -1) # (1,1,1,L)
            
        h_valid = x - tie_break
        
        # Subset selection based on length_dim
        if length_dim == 2:
            slopes[:, :, ignore_first:, :] = h_valid[:, :, ignore_first:, :] / dx_view[:, :, ignore_first:, :]
        else:
            slopes[:, :, :, ignore_first:] = h_valid[:, :, :, ignore_first:] / dx_view[:, :, :, ignore_first:]

    # Prefix max of slopes along length_dim; exclude current position
    prefix_max = torch.cummax(slopes, dim=length_dim).values
    
    # Pad to create exclude-self prefix max
    if length_dim == 2:
        pad = neg_inf.expand(B, 1, 1, W)
        prefix_max_excl = torch.cat([pad, prefix_max[:, :, :-1, :]], dim=2)
    else:
        pad = neg_inf.expand(B, 1, H, 1)
        prefix_max_excl = torch.cat([pad, prefix_max[:, :, :, :-1]], dim=3)
        
    # Visible if it is not lower than any prior slope on the same row
    visible = slopes >= prefix_max_excl
    
    # Output: keep only visible samples; everything else set to -2
    out = torch.full_like(x, -2.0)
    out[visible] = x[visible]
    return out

def world_yaw_from_xyzw(q: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    q: (..., 4) quaternion in (x, y, z, w) order.
       Interpreted as body->world orientation.
    Returns yaw (rotation about world Z) in radians in [-pi, pi].
    """
    assert q.shape[-1] == 4, "q must be (..., 4) in xyzw order"
    q = q.to(torch.float32)

    if normalize:
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    x, y, z, w = q.unbind(-1)

    # Z-up: yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny, cosy)

def realsense_grid_pattern(cfg: any, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pattern logic for standard RealSense depth cameras.
    
    This ensures identical FOV (approx 80x50.5), rectilinear projection,
    and ray ordering (Width outer, Height inner).
    """
    width = 48
    height = 27
    horizontal_fov = 80.0
    far_plane = 4.0
    
    # Calculate ranges
    y_range = math.tan(math.radians(horizontal_fov) / 2.0) * far_plane
    y_edges = torch.linspace(y_range, -y_range, width + 1, device=device)
    y = 0.5 * (y_edges[:-1] + y_edges[1:])  # center of each bin (Left to Right)
    
    z_range = y_range * height / width
    z_edges = torch.linspace(z_range, -z_range, height + 1, device=device)
    z = 0.5 * (z_edges[:-1] + z_edges[1:])  # center of each bin (Top to Bottom)
    
    # indexing="xy": inputs y(48), z(27) -> output (27, 48)
    y_grid, z_grid = torch.meshgrid(y, z, indexing="xy")
    x_grid = torch.full_like(y_grid, far_plane)
    
    # (27, 48, 3): inner (fastest) loop is width
    ray_directions = torch.stack([x_grid, y_grid, z_grid], dim=-1)
    ray_directions = F.normalize(ray_directions, p=2, dim=-1)
    
    ray_directions = ray_directions.reshape(-1, 3)
    ray_starts = torch.zeros_like(ray_directions)
    
    return ray_starts, ray_directions


def zedx_mini_grid_pattern(cfg: any, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pattern logic for the Stereolabs ZED X Mini (wide, f=2.2 mm).

    Mirrors :func:`realsense_grid_pattern` in structure but uses the ZED X Mini's
    *measured* horizontal and vertical fields of view independently, instead of
    deriving VFOV from the pixel aspect ratio. Necessary here because the
    ZED X Mini's lens is not angularly-square: HFOV=110° with VFOV=80°
    (sensor aspect 1920×1200 = 16:10).

    Grid is 64×40 (W×H) so the per-pixel angular resolution stays close to
    the realsense pattern's (~1.7°/px horizontal, ~2.0°/px vertical here).
    Consumers of this pattern must expect an image shape of ``(40, 64)``.

    Far plane is 4.0 m.
    """
    width = 64
    height = 40
    horizontal_fov = 110.0
    vertical_fov = 80.0
    far_plane = 4.0

    # Horizontal extent at far plane derived from HFOV
    y_range = math.tan(math.radians(horizontal_fov) / 2.0) * far_plane
    y_edges = torch.linspace(y_range, -y_range, width + 1, device=device)
    y = 0.5 * (y_edges[:-1] + y_edges[1:])  # center of each bin (Left to Right)

    # Vertical extent at far plane derived from VFOV (independent of pixel aspect)
    z_range = math.tan(math.radians(vertical_fov) / 2.0) * far_plane
    z_edges = torch.linspace(z_range, -z_range, height + 1, device=device)
    z = 0.5 * (z_edges[:-1] + z_edges[1:])  # center of each bin (Top to Bottom)

    # Same indexing convention as realsense_grid_pattern: indexing="xy" so
    # y(64), z(40) -> output (40, 64). Inner (fastest) loop is Width.
    y_grid, z_grid = torch.meshgrid(y, z, indexing="xy")
    x_grid = torch.full_like(y_grid, far_plane)

    # (40, 64, 3)
    ray_directions = torch.stack([x_grid, y_grid, z_grid], dim=-1)
    ray_directions = F.normalize(ray_directions, p=2, dim=-1)

    ray_directions = ray_directions.reshape(-1, 3)
    ray_starts = torch.zeros_like(ray_directions)

    return ray_starts, ray_directions
