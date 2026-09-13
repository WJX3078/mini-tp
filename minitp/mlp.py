"""Qwen2 SwiGLU MLP: fused gate|up GEMM + Row(down) parallelism.

v0.2: gate/up shards are packed into one fused parameter at load time — a
single GEMM per forward instead of two (FusedGateUpColumnParallelLinear).
"""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from minitp.config import ModelConfig
from minitp.distributed.context import ParallelContext
from minitp.parallel.fused import FusedGateUpColumnParallelLinear
from minitp.parallel.linear import RowParallelLinear


class TPQwen2MLP(nn.Module):
    """gate/up fused ColumnParallel (intermediate stays sharded, zero comm in
    between), local SiLU gating, down RowParallel with one AllReduce."""

    def __init__(self, cfg: ModelConfig, ctx: ParallelContext) -> None:
        super().__init__()
        self.intermediate_local = cfg.intermediate_size // ctx.tp_size
        self.gate_up_proj = FusedGateUpColumnParallelLinear(
            cfg.hidden_size, self.intermediate_local, ctx, bias=False
        )
        self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, ctx, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x)
        return self.down_proj(F.silu(gate) * up)
