# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid Muon/AdamW optimizer that accepts a flat parameter list."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

# Newton-Schulz constants from Keller Jordan's Muon; identical to torch.optim.Muon (torch >= 2.10)
_EPS = 1e-7
_NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def _zeropower_via_newtonschulz(update: Tensor, ns_steps: int) -> Tensor:
    """Quintic Newton-Schulz orthogonalization in bfloat16 (matches torch.optim.Muon)."""
    a, b, c = _NS_COEFFICIENTS
    ortho = update.bfloat16()
    transposed = update.size(0) > update.size(1)
    if transposed:
        ortho = ortho.T
    ortho = ortho / ortho.norm().clamp(min=_EPS)
    for _ in range(ns_steps):
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.T
    return ortho


class Muon(Optimizer):
    """Hybrid Muon/AdamW optimizer for a flat parameter list.

    Muon (SGD-momentum orthogonalized via Newton-Schulz,
    https://kellerjordan.github.io/posts/muon/) is applied to parameters with
    ndim >= 2 — hidden weight matrices, with conv kernels flattened to
    (out_channels, -1) for the orthogonalization. Parameters with ndim < 2
    (biases, gains, log_std) are optimized with AdamW in a second param group,
    since orthogonalization is undefined for vectors and scalars.

    Muon updates are rescaled per matrix by 0.2 * sqrt(max(rows, cols))
    ("match_rms_adamw", Moonlight, arXiv:2502.16982) so both groups share one
    Adam-scale learning rate: external schedules that uniformly set
    param_group["lr"] (PPO adaptive KL schedule, PPO_IL lr switching) remain
    correct without modification.

    The Muon math matches torch.optim.Muon (torch >= 2.10) with
    adjust_lr_fn="match_rms_adamw"; this class exists because the torch
    implementation rejects non-2D parameters and is unavailable on torch 2.7.
    Checkpoint resume uses the standard Optimizer state dict; the param groups
    are constructed in a deterministic order (Muon first, AdamW second) so the
    saved per-parameter state maps back onto the same parameters. Checkpoints
    written by a different optimizer (e.g. plain Adam) cannot be resumed with
    Muon — resume those with the original optimizer or skip the optimizer
    state via the runner's load config.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.999),
        adamw_eps: float = 1e-8,
    ) -> None:
        params = list(params)
        if any(isinstance(p, dict) for p in params):
            raise TypeError("Muon expects a flat iterable of tensors; param-group dicts are not supported.")
        muon_params = [p for p in params if p.ndim >= 2]
        adamw_params = [p for p in params if p.ndim < 2]
        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
        }
        param_groups = []
        if muon_params:
            param_groups.append({"params": muon_params, "use_muon": True})
        if adamw_params:
            param_groups.append({"params": adamw_params, "use_muon": False})
        if not param_groups:
            raise ValueError("Muon got an empty parameter list.")
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group: dict) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        momentum = group["momentum"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad.reshape(p.grad.size(0), -1) if p.grad.ndim > 2 else p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)
            buf = state["momentum_buffer"]
            buf.lerp_(grad, 1 - momentum)
            update = grad.lerp(buf, momentum) if group["nesterov"] else buf
            update = _zeropower_via_newtonschulz(update, group["ns_steps"])
            # RMS-matched scaling: one Adam-scale lr drives all matrix shapes
            adjusted_lr = 0.2 * lr * math.sqrt(max(update.size(0), update.size(1)))
            p.mul_(1 - lr * weight_decay)
            p.add_(update.reshape(p.shape), alpha=-adjusted_lr)

    def _adamw_step(self, group: dict) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]
            if "exp_avg" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            state["step"] += 1
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            bias_correction1 = 1 - beta1 ** state["step"]
            bias_correction2 = 1 - beta2 ** state["step"]
            denom = (exp_avg_sq / bias_correction2).sqrt_().add_(eps)
            p.mul_(1 - lr * weight_decay)
            p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
