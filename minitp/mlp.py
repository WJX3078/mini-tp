"""Qwen2 SwiGLU MLP with Column(gate/up) + Row(down) parallelism."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.parallel.linear import ColumnParallelLinear, RowParallelLinear


class TPQwen2MLP(nn.Module):
    """gate/up ColumnParallel (intermediate stays sharded, zero comm in between),
    local SiLU gating, down RowParallel with one AllReduce."""

    def __init__(self, cfg: ModelConfig, ctx: ParallelContext) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, ctx, bias=False)
        self.up_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, ctx, bias=False)
        self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, ctx, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
