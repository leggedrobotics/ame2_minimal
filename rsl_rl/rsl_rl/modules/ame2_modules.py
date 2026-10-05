import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel


@torch.jit.unused  # context manager is not scriptable; scripted callers use self.mha directly
def mha_math(mha: nn.MultiheadAttention, query, key, value, need_weights: bool = False):
    """``mha(query, key, value)`` with scaled-dot-product attention pinned to the math kernel.

    With torch 2.7.0+cu128, bf16 attention dispatches to the fused (memory-efficient) kernel, whose
    backward faults on Blackwell GPUs (sm_120) for the shapes used here -- head_dim 3 (96 / 32 heads)
    and batch * heads above ~1e5: illegal memory access / CUBLAS_STATUS_INTERNAL_ERROR. The math
    kernel is correct at every size. fp32 never uses a fused kernel, so pinning does not affect fp32
    runs; with 64 keys and head_dim 3 the math path is not slower.
    """
    with sdpa_kernel(SDPBackend.MATH):
        return mha(query, key, value, need_weights=need_weights)


class MLP_512(nn.Module):
    """MLP with two hidden layers of 512 units."""
    def __init__(self, input_dim, output_dim, activation=nn.ELU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            activation,
            nn.Linear(512, 512),
            activation,
            nn.Linear(512, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class MLP_GlobalPool(nn.Module):
    """Two-layer MLP (input_dim -> hidden_dim -> output_dim), 64 units by default."""
    def __init__(self, input_dim, hidden_dim=64, output_dim=64, activation=nn.ELU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
