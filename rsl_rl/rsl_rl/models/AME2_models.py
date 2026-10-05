from __future__ import annotations

import warnings
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules.ame2_modules import MLP_512, MLP_GlobalPool, mha_math
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories


class AME2Model(nn.Module):
    """AME2 attention-based model.

    Uses CNN+FC map encoding, a max-pooled global context and dense MHA over all map cells.
    Expects dictionary observations with "teacher_prop" and "teacher_mapping".
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_heads: int = 32,
        stochastic: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        self.obs_groups = obs_groups[obs_set]

        # Infer dimensions from observation TensorDict
        # Teacher Mapping: (B, L, W, dmap) or (B, L*W, dmap)
        self.dmap = obs["teacher_mapping"].shape[-1]
        # Teacher Prop: (B, prop_dim)
        teacher_prop = obs.get("teacher_prop", obs.get("prop_cmd"))
        self.prop_dim = teacher_prop.shape[-1]

        # -----------------------------------------------------------------
        # 1. Map Encoding (CNN + FC -> 96d Local Features)
        # -----------------------------------------------------------------
        # FC processes all channels (including explicit x and y coordinates)
        self.fc_part = nn.Sequential(
            nn.Linear(self.dmap, 16),
            nn.ELU(),
        )

        self.cnn_part = nn.Sequential(
            nn.Conv2d(self.dmap - 2, 8, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
            nn.Conv2d(8, 48, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
        )

        # Projection of concatenated FC(16) + CNN(48) to exact 96d d_model
        self.local_proj = nn.Sequential(
            nn.Linear(16 + 48, 96),
            nn.ELU(),
        )

        # -----------------------------------------------------------------
        # 2. Proprioceptive Encoding
        # -----------------------------------------------------------------
        self.prop_encoder = MLP_512(self.prop_dim, 128)

        # -----------------------------------------------------------------
        # 3. Terrain encoder: max-pooled global context + dense MHA
        # -----------------------------------------------------------------
        self.global_pool_mlp = MLP_GlobalPool(96, 64, 64)
        self.query_mlp = MLP_512(128 + 64, 96)
        self.mha = nn.MultiheadAttention(embed_dim=96, num_heads=num_heads, batch_first=True)

        # -----------------------------------------------------------------
        # 4. Final Output MLP (attn + prop + global -> output)
        # -----------------------------------------------------------------
        self.output_mlp = MLP_512(96 + 128 + 64, output_dim)

        # Stochasticity
        self.stochastic = stochastic
        self.noise_std_type = noise_std_type
        if stochastic:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(output_dim))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Note: Populated in _update_distribution
        self.distribution = None
        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        return_attention: bool = False,
        return_latent: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # If observations are padded for recurrent training but the model is non-recurrent, unpad
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs

        # Get latent representations
        res = self.get_latent(obs, masks, hidden_state, return_attention=return_attention)
        if return_attention:
            prop, attn_out, global_feat, weights = res
        else:
            prop, attn_out, global_feat = res

        # Final output: concat attn + prop + global -> MLP
        final_input = torch.cat([attn_out, prop, global_feat], dim=-1)
        output = self.output_mlp(final_input)

        if self.stochastic and stochastic_output:
            self._update_distribution(output)
            output = self.distribution.sample()

        if return_latent:
            return output, attn_out, global_feat

        if return_attention:
            return output, weights

        return output

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (prop, attn_out, global_feat) latent vectors."""
        prop = obs["teacher_prop"].detach()
        teacher_mapping = obs["teacher_mapping"].detach()
        
        B = prop.shape[0]
        L, W = teacher_mapping.shape[1:3]

        # --- A. Map Encoding ---
        map_4d = teacher_mapping.view(B, L, W, self.dmap)

        # 1. FC: (B, L, W, 16)
        fc_out = self.fc_part(map_4d)

        # 2. CNN over the non-coordinate channels: (B, C, L, W)
        cnn_in = map_4d[..., 2:].permute(0, 3, 1, 2)
        cnn_out = self.cnn_part(cnn_in)  # (B, 48, L, W)
        cnn_out = cnn_out.permute(0, 2, 3, 1) # back to NHWC for concat

        # 3. Fuse & Project -> (B, L*W, 96)
        fused_map = torch.cat([fc_out, cnn_out], dim=-1)
        local_feat = self.local_proj(fused_map).view(B, L * W, 96)

        # --- B. Terrain encoding ---
        prop_embedded = self.prop_encoder(prop)
        global_pre_pool = self.global_pool_mlp(local_feat)  # (B, L*W, 64)
        global_feat, global_indices = torch.max(global_pre_pool, dim=1)  # (B, 64), (B, 64)
        prop_global = torch.cat([prop_embedded, global_feat], dim=-1)
        query = self.query_mlp(prop_global).unsqueeze(1)  # (B, 1, 96)
        attn_out, weights = mha_math(self.mha, query, local_feat, local_feat, need_weights=return_attention)
        attn_out = attn_out.squeeze(1)  # (B, 96)
        weights_mixed = None
        if return_attention:  # local (MHA) + global one-hot of max-indices across features
            weights_global = F.one_hot(global_indices, num_classes=L * W).float().mean(dim=1)  # (B, L*W)
            weights_mixed = weights.squeeze(1) + weights_global

        if return_attention:
            return prop_embedded, attn_out, global_feat, weights_mixed

        return prop_embedded, attn_out, global_feat

    def get_hidden_state(self) -> HiddenState:
        return None

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return (self.output_mean, self.output_std)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs).sum(dim=-1)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchAME2Model(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        raise NotImplementedError("AME2Model does not support ONNX export yet.")

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def _update_distribution(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

class AME2LSIOModel(AME2Model):
    """AME2 LSIO (long-short IO) student model.

    Uses the AME2Model map encoder on "neural_map" and a proprioceptive hub
    (history CNN + 3-step recent MLP + commands) on "student_prop_hist" and "student_cmds".
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_heads: int = 32,
        stochastic: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # super().__init__ is skipped: the inputs differ from "teacher_prop"
        nn.Module.__init__(self)
        self.obs_groups = obs_groups[obs_set]

        # 1. Dimensions (student observation groups)
        # Map: (B, L, W, 4)
        self.dmap = obs["neural_map"].shape[-1]
        
        # Prop History: (B, T, prop_dim) -> T=20
        self.T = obs["student_prop_hist"].shape[1]
        self.prop_dim = obs["student_prop_hist"].shape[2]
        
        # Commands: (B, cmd_dim)
        self.cmd_dim = obs["student_cmds"].shape[-1]

        # -----------------------------------------------------------------
        # 2. Map Encoding (same as AME2Model)
        # -----------------------------------------------------------------
        self.fc_part = nn.Sequential(
            nn.Linear(self.dmap, 16),
            nn.ELU(),
        )
        self.cnn_part = nn.Sequential(
            nn.Conv2d(self.dmap - 2, 8, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
            nn.Conv2d(8, 48, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
        )
        self.local_proj = nn.Sequential(
            nn.Linear(16 + 48, 96),
            nn.ELU(),
        )
        self.global_pool_mlp = MLP_GlobalPool(96, 64, 64)

        # -----------------------------------------------------------------
        # 3. Proprioceptive Hub (128d total)
        # -----------------------------------------------------------------
        # Path A: CNN over the full history
        # Conv1d(prop_dim, 16, k5) -> Flatten(16 * (T - 4)) -> Linear(64)
        self.hist_encoder = nn.Sequential(
            nn.Conv1d(self.prop_dim, 16, kernel_size=5, padding=0),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16 * (self.T - 5 + 1), 64),
            nn.ELU(),
        )

        # Path B: recent IO window (latest 3 steps)
        # Linear(3 * prop_dim, 60)
        self.recent_encoder = nn.Sequential(
            nn.Linear(3 * self.prop_dim, 60),
            nn.ELU(),
        )

        # -----------------------------------------------------------------
        # 4. Prop Fusion MLP: raw concat (128) -> learned embedding (128)
        # -----------------------------------------------------------------
        self.prop_encoder = MLP_512(128, 128)

        # -----------------------------------------------------------------
        # 5. Fusion & Output (P=128, Q=96)
        # -----------------------------------------------------------------
        self.query_mlp = MLP_512(128 + 64, 96)
        self.mha = nn.MultiheadAttention(embed_dim=96, num_heads=num_heads, batch_first=True)
        
        # Final Output: attn(96) + prop(128) + global(64) = 288
        self.output_mlp = MLP_512(96 + 128 + 64, output_dim)

        # Stochasticity
        self.stochastic = stochastic
        self.noise_std_type = noise_std_type
        if stochastic:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(output_dim))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)))
        self.distribution = None
        Normal.set_default_validate_args(False)

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prop_hist = obs["student_prop_hist"]
        neural_map = obs["neural_map"]
        cmds = obs["student_cmds"]
        
        B = prop_hist.shape[0]
        L, W = neural_map.shape[1:3]

        # --- A. Spatial Encoding ---
        map_4d = neural_map.view(B, L, W, self.dmap)
        fc_out = self.fc_part(map_4d)
        cnn_in = map_4d[..., 2:].permute(0, 3, 1, 2)
        cnn_out = self.cnn_part(cnn_in).permute(0, 2, 3, 1)
        fused_map = torch.cat([fc_out, cnn_out], dim=-1)
        local_feat = self.local_proj(fused_map).view(B, L * W, 96)
        
        global_pre_pool = self.global_pool_mlp(local_feat)
        global_feat, _ = torch.max(global_pre_pool, dim=1)

        # --- B. Proprioceptive Encoding ---
        # 1. 1D-CNN Path
        hist_in = prop_hist.permute(0, 2, 1) # (B, prop_dim, T)
        emb_hist = self.hist_encoder(hist_in) # (B, 64)
        
        # 2. Recent Window Path (Latest 3 steps)
        recent_in = prop_hist[:, -3:, :].reshape(B, 3 * self.prop_dim)
        emb_recent = self.recent_encoder(recent_in) # (B, 60)
        
        # 3. Fuse with commands (4) -> 128, then encode
        prop_raw = torch.cat([emb_hist, emb_recent, cmds], dim=-1)
        prop_latent = self.prop_encoder(prop_raw)

        # --- C. Attention ---
        query_in = torch.cat([prop_latent, global_feat], dim=-1)
        query = self.query_mlp(query_in).unsqueeze(1) # (B, 1, 96)
        
        attn_out, weights = mha_math(self.mha, query, local_feat, local_feat, need_weights=return_attention)
        attn_out = attn_out.squeeze(1) # (B, 96)

        if return_attention:
            return prop_latent, attn_out, global_feat, weights

        return prop_latent, attn_out, global_feat


class AME2GazeModel(AME2Model):
    """TAGA-style active gaze (arXiv:2606.05880) with a dense-attention crop encoder.

    Stage 1: a shallow dilated CNN (small channels, no padding) over the whole
    map, max-pooled and projected by an MLP into the global context (64d). A
    gaze head on [global; prop_emb] predicts a continuous crop center; a
    crop_x x crop_y patch is extracted with a straight-through crop:
    forward = integer-aligned original height values (no interpolation), backward
    = bilinear grid-sample gradients w.r.t. the center, so the gaze head trains
    end-to-end from the PPO loss.
    Stage 2: the shape-constant AME2 encoder runs on the patch only, followed by
    dense MHA over all crop cells with a query from [prop_emb; stage-1 context].
    Per the TAGA authors, even a sigmoid-constrained gaze seeks the map edge, so
    a small boundary penalty is exposed via auxiliary_loss() (picked up by PPO).
    Attention visualization returns the crop-local weights scattered into
    full-map zeros; the nonzero rectangle doubles as the gaze visualization.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_heads: int = 32,
        crop_x: int = 12,
        crop_y: int = 10,
        roi_coef: float = 0.01,
        roi_margin: float = 0.35,
        stochastic: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # super().__init__ is skipped: the terrain encoder wiring differs
        nn.Module.__init__(self)
        self.obs_groups = obs_groups[obs_set]

        # Infer dimensions from observation TensorDict
        self.dmap = obs["teacher_mapping"].shape[-1]
        teacher_prop = obs.get("teacher_prop", obs.get("prop_cmd"))
        self.prop_dim = teacher_prop.shape[-1]
        L, W = obs["teacher_mapping"].shape[1:3]
        if crop_x > L or crop_y > W:
            raise ValueError(f"Gaze crop ({crop_x}x{crop_y}) exceeds the map grid ({L}x{W})")
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.roi_coef = roi_coef
        self.roi_margin = roi_margin

        # -----------------------------------------------------------------
        # 1. Crop Encoding (same shape-constant CNN + FC as AME2Model,
        #    but applied to the gaze crop only)
        # -----------------------------------------------------------------
        self.fc_part = nn.Sequential(
            nn.Linear(self.dmap, 16),
            nn.ELU(),
        )
        self.cnn_part = nn.Sequential(
            nn.Conv2d(self.dmap - 2, 8, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
            nn.Conv2d(8, 48, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
        )
        self.local_proj = nn.Sequential(
            nn.Linear(16 + 48, 96),
            nn.ELU(),
        )

        # -----------------------------------------------------------------
        # 2. Proprioceptive Encoding
        # -----------------------------------------------------------------
        self.prop_encoder = MLP_512(self.prop_dim, 128)

        # -----------------------------------------------------------------
        # 3. Stage-1 global context: shallow dilated CNN over the whole map
        #    -> channel-wise spatial max pool -> MLP to 64d. Small conv channels
        #    and no padding, since the max pool aggregates globally.
        # -----------------------------------------------------------------
        # all channels incl. x/y (CoordConv-style), so the spatial max-pool keeps where features are
        self.global_cnn = nn.Sequential(
            nn.Conv2d(self.dmap, 8, kernel_size=3, dilation=1),
            nn.ELU(),
            nn.Conv2d(8, 32, kernel_size=3, dilation=2),
            nn.ELU(),
        )
        context_dim = 64
        self.global_proj = nn.Sequential(
            nn.Linear(32, context_dim),
            nn.ELU(),
        )

        # -----------------------------------------------------------------
        # 4. Gaze head: [global; prop_emb(128)] -> r in (0,1)^2. Default init
        #    is already centered in expectation (E[logit] ~ 0 -> r ~ 0.5), and
        #    the sub-cell init scatter rounds to the same integer crop for every
        #    env, so no special centering init is needed.
        # -----------------------------------------------------------------
        self.gaze_head = nn.Sequential(
            nn.Linear(context_dim + 128, 64),
            nn.ELU(),
            nn.Linear(64, 2),
        )

        # -----------------------------------------------------------------
        # 5. Dense MHA over the crop, query from [prop_emb; stage-1 context]
        # -----------------------------------------------------------------
        self.query_mlp = MLP_512(128 + context_dim, 96)
        self.mha = nn.MultiheadAttention(embed_dim=96, num_heads=num_heads, batch_first=True)

        # -----------------------------------------------------------------
        # 6. Final Output MLP (attn + prop + global -> output)
        # -----------------------------------------------------------------
        self.output_mlp = MLP_512(96 + 128 + context_dim, output_dim)

        # Stochasticity
        self.stochastic = stochastic
        self.noise_std_type = noise_std_type
        if stochastic:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(output_dim))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

        # Last gaze location (B, 2) in (0,1)^2, kept with graph for auxiliary_loss()
        self._last_gaze_r: torch.Tensor | None = None

    def auxiliary_loss(self) -> torch.Tensor:
        """TAGA L_roi gaze boundary penalty (margin hinge on the normalized gaze).

        Zero inside |r - 0.5| <= roi_margin, quadratic outside; engages before
        sigmoid saturation would kill its own gradient. Valid after a forward.
        """
        excess = (self._last_gaze_r - 0.5).abs() - self.roi_margin
        return self.roi_coef * excess.clamp(min=0.0).square().mean()

    def _crop_window(self, cx: torch.Tensor, cy: torch.Tensor, L: int, W: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Integer top-left corner of the crop window, always fully inside the map."""
        half_x = (self.crop_x - 1) / 2.0
        half_y = (self.crop_y - 1) / 2.0
        ox = torch.clamp(torch.round(cx - half_x).long(), 0, L - self.crop_x)
        oy = torch.clamp(torch.round(cy - half_y).long(), 0, W - self.crop_y)
        return ox, oy

    def _hard_crop(self, map_4d: torch.Tensor, ox: torch.Tensor, oy: torch.Tensor) -> torch.Tensor:
        """Gather the integer-aligned window: original, uninterpolated map values."""
        B, L, W, C = map_4d.shape
        u = torch.arange(self.crop_x, device=map_4d.device)
        v = torch.arange(self.crop_y, device=map_4d.device)
        rows = ox.unsqueeze(1) + u.unsqueeze(0)                       # (B, crop_x)
        cols = oy.unsqueeze(1) + v.unsqueeze(0)                       # (B, crop_y)
        idx = (rows.unsqueeze(2) * W + cols.unsqueeze(1)).reshape(B, -1)  # (B, crop_x*crop_y)
        flat = map_4d.reshape(B, L * W, C)
        patch = flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, C))
        return patch.view(B, self.crop_x, self.crop_y, C)

    def _ste_crop(self, map_4d: torch.Tensor, cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
        """Straight-through crop: forward = original values at the integer-aligned
        window; backward = bilinear grid-sample gradients w.r.t. the continuous
        center (interpolate gradients, not heights)."""
        B, L, W, _ = map_4d.shape
        ox, oy = self._crop_window(cx, cy, L, W)
        hard = self._hard_crop(map_4d, ox, oy)

        # soft bilinear sample centered at the continuous (cx, cy)
        src = map_4d.permute(0, 3, 1, 2)                              # (B, C, L, W)
        u = torch.arange(self.crop_x, device=map_4d.device, dtype=map_4d.dtype)
        v = torch.arange(self.crop_y, device=map_4d.device, dtype=map_4d.dtype)
        px = cx.view(B, 1, 1) + (u - (self.crop_x - 1) / 2.0).view(1, -1, 1)  # rows (B, crop_x, 1)
        py = cy.view(B, 1, 1) + (v - (self.crop_y - 1) / 2.0).view(1, 1, -1)  # cols (B, 1, crop_y)
        # normalize to [-1, 1] (align_corners=True); grid[..., 0] indexes width (cols)
        gx = 2.0 * py / (W - 1) - 1.0
        gy = 2.0 * px / (L - 1) - 1.0
        grid = torch.stack(
            [gx.expand(B, self.crop_x, self.crop_y), gy.expand(B, self.crop_x, self.crop_y)], dim=-1
        )
        # Native CUDA kernel, not cuDNN: torch routes 4D bilinear grid_sample to cuDNN's
        # spatial transformer, which raises CUDNN_STATUS_NOT_SUPPORTED once the batch
        # exceeds ~1e5 rows (large PPO minibatches). The native kernel has no such limit.
        with torch.backends.cudnn.flags(enabled=False):
            soft = F.grid_sample(src, grid, mode="bilinear", align_corners=True)
        soft = soft.permute(0, 2, 3, 1)                               # (B, crop_x, crop_y, C)

        # (soft - soft.detach()) is exactly zero in forward, so the patch is
        # bitwise the hard crop; backward still flows through the soft path
        return hard + (soft - soft.detach())

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (prop, attn_out, global_feat) latent vectors."""
        prop = obs["teacher_prop"].detach()
        teacher_mapping = obs["teacher_mapping"].detach()

        B = prop.shape[0]
        L, W = teacher_mapping.shape[1:3]
        map_4d = teacher_mapping.view(B, L, W, self.dmap)

        # --- A. Stage 1: global context + gaze ---
        gfeat = self.global_cnn(map_4d.permute(0, 3, 1, 2))          # (B, 32, L', W')
        pooled, _ = torch.max(gfeat.flatten(2), dim=2)                # (B, 32)
        global_feat = self.global_proj(pooled)                       # (B, 64)
        prop_embedded = self.prop_encoder(prop)

        r = torch.sigmoid(self.gaze_head(torch.cat([global_feat, prop_embedded], dim=-1)))  # (B, 2)
        self._last_gaze_r = r
        # continuous center mapped into the exact valid box: window always inside the map
        cx = (self.crop_x - 1) / 2.0 + r[:, 0] * float(L - self.crop_x)
        cy = (self.crop_y - 1) / 2.0 + r[:, 1] * float(W - self.crop_y)

        patch = self._ste_crop(map_4d, cx, cy)                        # (B, crop_x, crop_y, dmap)

        # --- B. Stage 2: crop encoding (same ops as AME2Model, on the patch) ---
        fc_out = self.fc_part(patch)
        cnn_in = patch[..., 2:].permute(0, 3, 1, 2)
        cnn_out = self.cnn_part(cnn_in).permute(0, 2, 3, 1)
        fused = torch.cat([fc_out, cnn_out], dim=-1)
        local_feat = self.local_proj(fused).view(B, self.crop_x * self.crop_y, 96)

        query = self.query_mlp(torch.cat([prop_embedded, global_feat], dim=-1)).unsqueeze(1)
        attn_out, weights_crop = mha_math(self.mha, query, local_feat, local_feat, need_weights=return_attention)
        attn_out = attn_out.squeeze(1)

        if return_attention:
            # crop-local attention scattered into the full map at the crop offsets;
            # the nonzero rectangle also visualizes the gaze region (no global viz)
            ox, oy = self._crop_window(cx, cy, L, W)
            u = torch.arange(self.crop_x, device=prop.device)
            v = torch.arange(self.crop_y, device=prop.device)
            rows = ox.unsqueeze(1) + u.unsqueeze(0)
            cols = oy.unsqueeze(1) + v.unsqueeze(0)
            idx = (rows.unsqueeze(2) * W + cols.unsqueeze(1)).reshape(B, -1)
            weights_full = torch.zeros(B, L * W, device=prop.device, dtype=weights_crop.dtype)
            weights_full.scatter_(1, idx, weights_crop.squeeze(1))
            return prop_embedded, attn_out, global_feat, weights_full

        return prop_embedded, attn_out, global_feat

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchAME2GazeModel(self)


class AME2GazeLSIOModel(AME2GazeModel):
    """TAGA-gaze student: the AME2GazeModel map path on the LSIO inputs.

    Consumes the student observation groups (``neural_map`` with 4 channels
    x/y/z/unc, ``student_prop_hist`` 20-step history, ``student_cmds``) like
    :class:`AME2LSIOModel`, but replaces its dense full-map MHA with the gaze
    teacher's two-stage encoder: a shallow global CNN over the whole map -> 64d
    context, a gaze head on [context; prop_emb] -> straight-through crop, the
    shape-constant crop encoder, and dense MHA over the crop cells with the
    external context. The proprioceptive hub is the LSIO one (history CNN + recent
    window + commands -> 128d), so the latent interfaces (prop 128 / attn 96 /
    global 64) match the gaze teacher's for PPO_IL representation distillation.

    Inherits the STE crop machinery and the TAGA L_roi ``auxiliary_loss`` from
    :class:`AME2GazeModel`.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_heads: int = 32,
        crop_x: int = 12,
        crop_y: int = 10,
        roi_coef: float = 0.01,
        roi_margin: float = 0.35,
        stochastic: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        # No super().__init__: the input wiring differs from both parents
        nn.Module.__init__(self)
        self.obs_groups = obs_groups[obs_set]

        # Infer dimensions from the student observation TensorDict
        self.dmap = obs["neural_map"].shape[-1]
        self.T = obs["student_prop_hist"].shape[1]
        self.prop_dim = obs["student_prop_hist"].shape[2]
        self.cmd_dim = obs["student_cmds"].shape[-1]
        L, W = obs["neural_map"].shape[1:3]
        if crop_x > L or crop_y > W:
            raise ValueError(f"Gaze crop ({crop_x}x{crop_y}) exceeds the map grid ({L}x{W})")
        self.crop_x = crop_x
        self.crop_y = crop_y
        self.roi_coef = roi_coef
        self.roi_margin = roi_margin

        # -----------------------------------------------------------------
        # 1. Crop encoding (shape-constant, same as AME2GazeModel)
        # -----------------------------------------------------------------
        self.fc_part = nn.Sequential(
            nn.Linear(self.dmap, 16),
            nn.ELU(),
        )
        self.cnn_part = nn.Sequential(
            nn.Conv2d(self.dmap - 2, 8, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
            nn.Conv2d(8, 48, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
        )
        self.local_proj = nn.Sequential(
            nn.Linear(16 + 48, 96),
            nn.ELU(),
        )

        # -----------------------------------------------------------------
        # 2. Proprioceptive hub (LSIO: history CNN + recent window + commands)
        # -----------------------------------------------------------------
        self.hist_encoder = nn.Sequential(
            nn.Conv1d(self.prop_dim, 16, kernel_size=5, padding=0),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(16 * (self.T - 5 + 1), 64),
            nn.ELU(),
        )
        self.recent_encoder = nn.Sequential(
            nn.Linear(3 * self.prop_dim, 60),
            nn.ELU(),
        )
        self.prop_encoder = MLP_512(64 + 60 + self.cmd_dim, 128)

        # -----------------------------------------------------------------
        # 3. Stage-1 global context over the whole map (as in AME2GazeModel)
        # -----------------------------------------------------------------
        # all channels incl. x/y (CoordConv-style), so the spatial max-pool keeps where features are
        self.global_cnn = nn.Sequential(
            nn.Conv2d(self.dmap, 8, kernel_size=3, dilation=1),
            nn.ELU(),
            nn.Conv2d(8, 32, kernel_size=3, dilation=2),
            nn.ELU(),
        )
        context_dim = 64
        self.global_proj = nn.Sequential(
            nn.Linear(32, context_dim),
            nn.ELU(),
        )

        # -----------------------------------------------------------------
        # 4. Gaze head: [global; prop_emb(128)] -> r in (0,1)^2
        # -----------------------------------------------------------------
        self.gaze_head = nn.Sequential(
            nn.Linear(context_dim + 128, 64),
            nn.ELU(),
            nn.Linear(64, 2),
        )

        # -----------------------------------------------------------------
        # 5. Dense MHA over the crop, query from [prop_emb; stage-1 context]
        # -----------------------------------------------------------------
        self.query_mlp = MLP_512(128 + context_dim, 96)
        self.mha = nn.MultiheadAttention(embed_dim=96, num_heads=num_heads, batch_first=True)

        # -----------------------------------------------------------------
        # 6. Final output MLP (attn + prop + global -> output)
        # -----------------------------------------------------------------
        self.output_mlp = MLP_512(96 + 128 + context_dim, output_dim)

        # Stochasticity
        self.stochastic = stochastic
        self.noise_std_type = noise_std_type
        if stochastic:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(output_dim))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

        self._last_gaze_r: torch.Tensor | None = None

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (prop, attn_out, global_feat) latent vectors."""
        prop_hist = obs["student_prop_hist"]
        neural_map = obs["neural_map"]
        cmds = obs["student_cmds"]

        B = prop_hist.shape[0]
        L, W = neural_map.shape[1:3]
        map_4d = neural_map.view(B, L, W, self.dmap)

        # --- A. Proprioceptive hub (LSIO) ---
        hist_in = prop_hist.permute(0, 2, 1)                          # (B, prop_dim, T)
        emb_hist = self.hist_encoder(hist_in)                         # (B, 64)
        recent_in = prop_hist[:, -3:, :].reshape(B, 3 * self.prop_dim)
        emb_recent = self.recent_encoder(recent_in)                   # (B, 60)
        prop_embedded = self.prop_encoder(torch.cat([emb_hist, emb_recent, cmds], dim=-1))

        # --- B. Stage 1: global context + gaze ---
        gfeat = self.global_cnn(map_4d.permute(0, 3, 1, 2))          # (B, 32, L', W')
        pooled, _ = torch.max(gfeat.flatten(2), dim=2)                # (B, 32)
        global_feat = self.global_proj(pooled)                        # (B, 64)

        r = torch.sigmoid(self.gaze_head(torch.cat([global_feat, prop_embedded], dim=-1)))  # (B, 2)
        self._last_gaze_r = r
        cx = (self.crop_x - 1) / 2.0 + r[:, 0] * float(L - self.crop_x)
        cy = (self.crop_y - 1) / 2.0 + r[:, 1] * float(W - self.crop_y)

        patch = self._ste_crop(map_4d, cx, cy)                        # (B, crop_x, crop_y, dmap)

        # --- C. Stage 2: crop encoding + dense MHA with the stage-1 context ---
        fc_out = self.fc_part(patch)
        cnn_in = patch[..., 2:].permute(0, 3, 1, 2)
        cnn_out = self.cnn_part(cnn_in).permute(0, 2, 3, 1)
        fused = torch.cat([fc_out, cnn_out], dim=-1)
        local_feat = self.local_proj(fused).view(B, self.crop_x * self.crop_y, 96)

        query = self.query_mlp(torch.cat([prop_embedded, global_feat], dim=-1)).unsqueeze(1)
        attn_out, weights_crop = mha_math(self.mha, query, local_feat, local_feat, need_weights=return_attention)
        attn_out = attn_out.squeeze(1)

        if return_attention:
            # crop-local attention scattered into the full map at the crop offsets
            ox, oy = self._crop_window(cx, cy, L, W)
            u = torch.arange(self.crop_x, device=prop_hist.device)
            v = torch.arange(self.crop_y, device=prop_hist.device)
            rows = ox.unsqueeze(1) + u.unsqueeze(0)
            cols = oy.unsqueeze(1) + v.unsqueeze(0)
            idx = (rows.unsqueeze(2) * W + cols.unsqueeze(1)).reshape(B, -1)
            weights_full = torch.zeros(B, L * W, device=prop_hist.device, dtype=weights_crop.dtype)
            weights_full.scatter_(1, idx, weights_crop.squeeze(1))
            return prop_embedded, attn_out, global_feat, weights_full

        return prop_embedded, attn_out, global_feat

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchAME2GazeLSIOModel(self)


class AME1Model(nn.Module):
    """AME1 attention-based model.

    Unlike AME2Model, there is no global pooling and the query comes only from the proprioceptive embedding.
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_heads: int = 32,
        stochastic: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        self.obs_groups = obs_groups[obs_set]

        # Infer dimensions from observation TensorDict
        self.dmap = obs["teacher_mapping"].shape[-1]
        teacher_prop = obs.get("teacher_prop", obs.get("prop_cmd"))
        self.prop_dim = teacher_prop.shape[-1]

        # -----------------------------------------------------------------
        # 1. Map Encoding (CNN + FC -> 96d Local Features)
        # -----------------------------------------------------------------
        self.fc_part = nn.Sequential(
            nn.Linear(self.dmap, 16),
            nn.ELU(),
        )

        self.cnn_part = nn.Sequential(
            nn.Conv2d(self.dmap - 2, 8, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
            nn.Conv2d(8, 48, kernel_size=5, padding=2, padding_mode="zeros"),
            nn.ELU(),
        )

        self.local_proj = nn.Sequential(
            nn.Linear(16 + 48, 96),
            nn.ELU(),
        )

        # -----------------------------------------------------------------
        # 2. Query Generation (prop -> 96)
        # -----------------------------------------------------------------
        self.prop_encoder = MLP_512(self.prop_dim, 128)
        self.query_mlp = MLP_512(128, 96)

        # -----------------------------------------------------------------
        # 3. Multi-Head Attention (96d, 32 heads)
        # -----------------------------------------------------------------
        self.mha = nn.MultiheadAttention(embed_dim=96, num_heads=num_heads, batch_first=True)

        # -----------------------------------------------------------------
        # 4. Final Output MLP (attn + prop -> output)
        # -----------------------------------------------------------------
        self.output_mlp = MLP_512(96 + 128, output_dim)

        # Stochasticity
        self.stochastic = stochastic
        self.noise_std_type = noise_std_type
        if stochastic:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(output_dim))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = None
        Normal.set_default_validate_args(False)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs

        res = self.get_latent(obs, masks, hidden_state, return_attention=return_attention)
        if return_attention:
            prop, attn_out, weights = res
        else:
            prop, attn_out = res

        # Final output: concat attn + prop -> MLP
        final_input = torch.cat([attn_out, prop], dim=-1)
        output = self.output_mlp(final_input)

        if self.stochastic and stochastic_output:
            self._update_distribution(output)
            output = self.distribution.sample()

        if return_attention:
            return output, weights

        return output

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (prop, attn_out) latent vectors."""
        prop = obs["teacher_prop"].detach()
        teacher_mapping = obs["teacher_mapping"].detach()
        
        B = prop.shape[0]
        L, W = teacher_mapping.shape[1:3]

        # --- A. Map Encoding ---
        map_4d = teacher_mapping.view(B, L, W, self.dmap)

        # 1. FC
        fc_out = self.fc_part(map_4d)

        # 2. CNN
        cnn_in = map_4d[..., 2:].permute(0, 3, 1, 2)
        cnn_out = self.cnn_part(cnn_in)
        cnn_out = cnn_out.permute(0, 2, 3, 1)

        # 3. Fuse & Project -> (B, L*W, 96)
        fused_map = torch.cat([fc_out, cnn_out], dim=-1)
        local_feat = self.local_proj(fused_map).view(B, L * W, 96)

        # --- B. Query Generation ---
        prop_embedded = self.prop_encoder(prop)
        query = self.query_mlp(prop_embedded).unsqueeze(1)  # (B, 1, 96)

        # --- C. Attention ---
        attn_out, weights = mha_math(self.mha, query, local_feat, local_feat, need_weights=return_attention)
        attn_out = attn_out.squeeze(1)  # (B, 96)

        if return_attention:
            weights_local = weights.squeeze(1)  # (B, L*W)
            return prop_embedded, attn_out, weights_local

        return prop_embedded, attn_out

    def get_hidden_state(self) -> HiddenState:
        return None

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return (self.output_mean, self.output_std)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs).sum(dim=-1)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    def as_jit(self) -> nn.Module:
        raise NotImplementedError("AME1Model does not support JIT export yet.")

    def as_onnx(self, verbose: bool) -> nn.Module:
        raise NotImplementedError("AME1Model does not support ONNX export yet.")

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def _update_distribution(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)


class MoEModel(nn.Module):
    """Soft mixture-of-experts MLP over the concatenated observation groups.

    Can be used as either actor (stochastic=True) or critic (stochastic=False, output_dim=1).
    """

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_experts: int = 16,
        stochastic: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__()
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        # Experts: Each is a 512-unit MLP
        self.experts = nn.ModuleList([
            MLP_512(self.obs_dim, output_dim) for _ in range(num_experts)
        ])

        # Router: 512-unit MLP outputting num_experts logits
        self.router = MLP_512(self.obs_dim, num_experts)

        self.output_dim = output_dim

        # Stochasticity
        self.stochastic = stochastic
        self.noise_std_type = noise_std_type
        if stochastic:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(output_dim))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(output_dim)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # Note: Populated in _update_distribution
        self.distribution = None
        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        # If observations are padded for recurrent training but the model is non-recurrent, unpad
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs

        # Get latent
        x = self.get_latent(obs, masks, hidden_state)

        # 1. Gate Weights via Softmax -> (B, num_experts)
        gate_weights = F.softmax(self.router(x), dim=-1)

        # 2. Evaluate all experts -> (B, num_experts, output_dim)
        expert_outs = torch.stack([expert(x) for expert in self.experts], dim=1)

        # 3. Soft Mixture -> (B, output_dim)
        output = torch.sum(gate_weights.unsqueeze(-1) * expert_outs, dim=1)

        if self.stochastic and stochastic_output:
            self._update_distribution(output)
            return self.distribution.sample()

        return output

    def get_latent(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
    ) -> torch.Tensor:
        """Concatenate observation groups into a flat latent vector."""
        obs_list = [obs[g].detach() for g in self.obs_groups]
        return torch.cat(obs_list, dim=-1)

    def get_hidden_state(self) -> HiddenState:
        return None

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        pass

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def output_entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        return (self.output_mean, self.output_std)

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(outputs).sum(dim=-1)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        raise NotImplementedError("MoEModel does not support JIT export yet.")

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        raise NotImplementedError("MoEModel does not support ONNX export yet.")

    def update_normalization(self, obs: TensorDict) -> None:
        pass

    def _update_distribution(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

class _TorchAME2Model(nn.Module):
    """JIT twin for the AME2 encoder."""

    def __init__(self, model: AME2Model) -> None:
        super().__init__()
        self.dmap = model.dmap
        self.prop_dim = model.prop_dim

        self.fc_part = copy.deepcopy(model.fc_part)
        self.cnn_part = copy.deepcopy(model.cnn_part)
        self.local_proj = copy.deepcopy(model.local_proj)
        self.global_pool_mlp = copy.deepcopy(model.global_pool_mlp)
        self.prop_encoder = copy.deepcopy(model.prop_encoder)
        self.query_mlp = copy.deepcopy(model.query_mlp)
        self.mha = copy.deepcopy(model.mha)
        self.output_mlp = copy.deepcopy(model.output_mlp)

        if hasattr(model, "distribution") and model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, prop: torch.Tensor, teacher_mapping: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = prop.shape[0]
        L = teacher_mapping.shape[1]
        W = teacher_mapping.shape[2]

        map_4d = teacher_mapping.view(B, L, W, self.dmap)

        fc_out = self.fc_part(map_4d)

        cnn_in = map_4d[..., 2:].permute(0, 3, 1, 2)
        cnn_out = self.cnn_part(cnn_in)
        cnn_out = cnn_out.permute(0, 2, 3, 1)

        fused_map = torch.cat([fc_out, cnn_out], dim=-1)
        local_feat = self.local_proj(fused_map).view(B, L * W, 96)

        global_pre_pool = self.global_pool_mlp(local_feat)
        global_feat, _ = torch.max(global_pre_pool, dim=1)

        prop_embedded = self.prop_encoder(prop)
        prop_global = torch.cat([prop_embedded, global_feat], dim=-1)
        query = self.query_mlp(prop_global).unsqueeze(1)

        if torch.jit.is_scripting():
            attn_out, _ = self.mha(query, local_feat, local_feat, need_weights=False)
        else:
            attn_out, _ = mha_math(self.mha, query, local_feat, local_feat, need_weights=False)
        attn_out = attn_out.squeeze(1)

        final_input = torch.cat([attn_out, prop_embedded, global_feat], dim=-1)
        output = self.output_mlp(final_input)
        actions = self.deterministic_output(output)

        return actions, attn_out, global_feat


class _TorchAME2GazeModel(nn.Module):
    """JIT twin for the AME2GazeModel (hard integer crop only: the STE forward path)."""

    def __init__(self, model: AME2GazeModel) -> None:
        super().__init__()
        self.dmap = model.dmap
        self.prop_dim = model.prop_dim
        self.crop_x = model.crop_x
        self.crop_y = model.crop_y

        self.fc_part = copy.deepcopy(model.fc_part)
        self.cnn_part = copy.deepcopy(model.cnn_part)
        self.local_proj = copy.deepcopy(model.local_proj)
        self.prop_encoder = copy.deepcopy(model.prop_encoder)
        self.global_cnn = copy.deepcopy(model.global_cnn)
        self.global_proj = copy.deepcopy(model.global_proj)
        self.gaze_head = copy.deepcopy(model.gaze_head)
        self.query_mlp = copy.deepcopy(model.query_mlp)
        self.mha = copy.deepcopy(model.mha)
        self.output_mlp = copy.deepcopy(model.output_mlp)

        if hasattr(model, "distribution") and model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, prop: torch.Tensor, teacher_mapping: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        actions, attn_out, global_feat, _mu = self.forward_with_gaze(prop, teacher_mapping)
        return actions, attn_out, global_feat

    def forward_with_gaze(
        self, prop: torch.Tensor, teacher_mapping: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (actions, attn_out, global_feat, gaze logits)."""
        B = prop.shape[0]
        L = teacher_mapping.shape[1]
        W = teacher_mapping.shape[2]

        map_4d = teacher_mapping.view(B, L, W, self.dmap)

        # Stage 1: global context + gaze
        gfeat = self.global_cnn(map_4d.permute(0, 3, 1, 2))
        pooled, _ = torch.max(gfeat.flatten(2), dim=2)
        global_feat = self.global_proj(pooled)
        prop_embedded = self.prop_encoder(prop)
        mu = self.gaze_head(torch.cat([global_feat, prop_embedded], dim=-1))
        r = torch.sigmoid(mu)
        half_x = float(self.crop_x - 1) / 2.0
        half_y = float(self.crop_y - 1) / 2.0
        cx = half_x + r[:, 0] * float(L - self.crop_x)
        cy = half_y + r[:, 1] * float(W - self.crop_y)

        # hard integer-aligned crop (original values; STE soft path is training-only)
        ox = torch.clamp(torch.round(cx - half_x).long(), 0, L - self.crop_x)
        oy = torch.clamp(torch.round(cy - half_y).long(), 0, W - self.crop_y)
        u = torch.arange(self.crop_x, device=prop.device)
        v = torch.arange(self.crop_y, device=prop.device)
        rows = ox.unsqueeze(1) + u.unsqueeze(0)
        cols = oy.unsqueeze(1) + v.unsqueeze(0)
        idx = (rows.unsqueeze(2) * W + cols.unsqueeze(1)).reshape(B, -1)
        flat = map_4d.reshape(B, L * W, self.dmap)
        patch = flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.dmap)).view(
            B, self.crop_x, self.crop_y, self.dmap
        )

        # Stage 2: crop encoding + dense MHA with the external context
        fc_out = self.fc_part(patch)
        cnn_in = patch[..., 2:].permute(0, 3, 1, 2)
        cnn_out = self.cnn_part(cnn_in).permute(0, 2, 3, 1)
        fused = torch.cat([fc_out, cnn_out], dim=-1)
        local_feat = self.local_proj(fused).view(B, self.crop_x * self.crop_y, 96)

        query = self.query_mlp(torch.cat([prop_embedded, global_feat], dim=-1)).unsqueeze(1)
        if torch.jit.is_scripting():
            attn_out, _ = self.mha(query, local_feat, local_feat, need_weights=False)
        else:
            attn_out, _ = mha_math(self.mha, query, local_feat, local_feat, need_weights=False)
        attn_out = attn_out.squeeze(1)

        final_input = torch.cat([attn_out, prop_embedded, global_feat], dim=-1)
        output = self.output_mlp(final_input)
        actions = self.deterministic_output(output)

        return actions, attn_out, global_feat, mu


class _TorchAME2GazeLSIOModel(nn.Module):
    """JIT twin for the AME2GazeLSIOModel (hard integer crop; deployment inputs).

    forward(neural_map (B,L,W,dmap), prop_hist (B,T,prop_dim), cmds (B,cmd_dim))
    -> (actions, attn_out, global_feat).
    """

    def __init__(self, model: AME2GazeLSIOModel) -> None:
        super().__init__()
        self.dmap = model.dmap
        self.prop_dim = model.prop_dim
        self.crop_x = model.crop_x
        self.crop_y = model.crop_y

        self.fc_part = copy.deepcopy(model.fc_part)
        self.cnn_part = copy.deepcopy(model.cnn_part)
        self.local_proj = copy.deepcopy(model.local_proj)
        self.hist_encoder = copy.deepcopy(model.hist_encoder)
        self.recent_encoder = copy.deepcopy(model.recent_encoder)
        self.prop_encoder = copy.deepcopy(model.prop_encoder)
        self.global_cnn = copy.deepcopy(model.global_cnn)
        self.global_proj = copy.deepcopy(model.global_proj)
        self.gaze_head = copy.deepcopy(model.gaze_head)
        self.query_mlp = copy.deepcopy(model.query_mlp)
        self.mha = copy.deepcopy(model.mha)
        self.output_mlp = copy.deepcopy(model.output_mlp)

        if hasattr(model, "distribution") and model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(
        self, neural_map: torch.Tensor, prop_hist: torch.Tensor, cmds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = prop_hist.shape[0]
        L = neural_map.shape[1]
        W = neural_map.shape[2]
        map_4d = neural_map.view(B, L, W, self.dmap)

        # LSIO proprioceptive hub
        hist_in = prop_hist.permute(0, 2, 1)
        emb_hist = self.hist_encoder(hist_in)
        recent_in = prop_hist[:, -3:, :].reshape(B, 3 * self.prop_dim)
        emb_recent = self.recent_encoder(recent_in)
        prop_embedded = self.prop_encoder(torch.cat([emb_hist, emb_recent, cmds], dim=-1))

        # Stage 1: global context + gaze
        gfeat = self.global_cnn(map_4d.permute(0, 3, 1, 2))
        pooled, _ = torch.max(gfeat.flatten(2), dim=2)
        global_feat = self.global_proj(pooled)
        r = torch.sigmoid(self.gaze_head(torch.cat([global_feat, prop_embedded], dim=-1)))
        half_x = float(self.crop_x - 1) / 2.0
        half_y = float(self.crop_y - 1) / 2.0
        cx = half_x + r[:, 0] * float(L - self.crop_x)
        cy = half_y + r[:, 1] * float(W - self.crop_y)

        # hard integer-aligned crop (original values; STE soft path is training-only)
        ox = torch.clamp(torch.round(cx - half_x).long(), 0, L - self.crop_x)
        oy = torch.clamp(torch.round(cy - half_y).long(), 0, W - self.crop_y)
        u = torch.arange(self.crop_x, device=prop_hist.device)
        v = torch.arange(self.crop_y, device=prop_hist.device)
        rows = ox.unsqueeze(1) + u.unsqueeze(0)
        cols = oy.unsqueeze(1) + v.unsqueeze(0)
        idx = (rows.unsqueeze(2) * W + cols.unsqueeze(1)).reshape(B, -1)
        flat = map_4d.reshape(B, L * W, self.dmap)
        patch = flat.gather(1, idx.unsqueeze(-1).expand(-1, -1, self.dmap)).view(
            B, self.crop_x, self.crop_y, self.dmap
        )

        # Stage 2: crop encoding + dense MHA with the external context
        fc_out = self.fc_part(patch)
        cnn_in = patch[..., 2:].permute(0, 3, 1, 2)
        cnn_out = self.cnn_part(cnn_in).permute(0, 2, 3, 1)
        fused = torch.cat([fc_out, cnn_out], dim=-1)
        local_feat = self.local_proj(fused).view(B, self.crop_x * self.crop_y, 96)

        query = self.query_mlp(torch.cat([prop_embedded, global_feat], dim=-1)).unsqueeze(1)
        if torch.jit.is_scripting():
            attn_out, _ = self.mha(query, local_feat, local_feat, need_weights=False)
        else:
            attn_out, _ = mha_math(self.mha, query, local_feat, local_feat, need_weights=False)
        attn_out = attn_out.squeeze(1)

        final_input = torch.cat([attn_out, prop_embedded, global_feat], dim=-1)
        output = self.output_mlp(final_input)
        actions = self.deterministic_output(output)

        return actions, attn_out, global_feat
