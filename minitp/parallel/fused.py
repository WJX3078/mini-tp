"""Fused per-rank packed projections: one GEMM instead of 2–3 thin ones.

The Qwen2 checkpoint stores q/k/v (and gate/up) as separate tensors, but the
TP runtime only ever uses each rank's local shard. Packing the shards into a
single parameter at *load time* turns 3 GEMMs + 3 bias adds into 1 per layer
per forward (see docs/PERFORMANCE_AUDIT.md §7). Packing happens here, never
in the hot path.

Packed row layout (PyTorch weight is [out_features, in_features]):
- FusedQKV:      [ q_shard ; k_shard ; v_shard ]  -> q_local + 2*kv_local rows
- FusedGateUp:   [ gate_shard ; up_shard ]        -> 2*inter_local rows

Sizes are NOT assumed equal: GQA gives kv_local != q_local, and KV
replication (num_kv_heads < tp) gives one KV head per rank.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minitp.distributed.context import ParallelContext


class FusedQKVColumnParallelLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        q_out_local: int,
        kv_out_local: int,
        ctx: ParallelContext,
        bias: bool = True,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % ctx.tp_size != 0:
            raise ValueError(
                f"hidden_size={hidden_size} not divisible by tp_size={ctx.tp_size}"
            )
        self.hidden_size = hidden_size
        self.q_out_local = q_out_local
        self.kv_out_local = kv_out_local
        self.ctx = ctx
        out_rows = q_out_local + 2 * kv_out_local
        self.weight = nn.Parameter(torch.empty(out_rows, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_rows))
        else:
            self.register_parameter("bias", None)
        if init_weights:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def out_features(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y = F.linear(x, self.weight, self.bias)
        # split along last dim returns zero-copy views
        q, k, v = torch.split(y, [self.q_out_local, self.kv_out_local, self.kv_out_local], dim=-1)
        return q, k, v

    def forward_unfused(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Reference v0.1 compute pattern: 3 GEMMs on row views of the same
        # packed parameter. Ablation only - mathematically identical.
        q_rows, kv_rows = self.q_out_local, self.kv_out_local
        w, b = self.weight, self.bias
        b1 = b[:q_rows] if b is not None else None
        b2 = b[q_rows : q_rows + kv_rows] if b is not None else None
        b3 = b[q_rows + kv_rows :] if b is not None else None
        q = F.linear(x, w[:q_rows], b1)
        k = F.linear(x, w[q_rows : q_rows + kv_rows], b2)
        v = F.linear(x, w[q_rows + kv_rows :], b3)
        return q, k, v


class FusedGateUpColumnParallelLinear(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_local: int,
        ctx: ParallelContext,
        bias: bool = False,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        if intermediate_local * ctx.tp_size == 0 or hidden_size % ctx.tp_size != 0:
            raise ValueError(
                f"hidden_size={hidden_size} not divisible by tp_size={ctx.tp_size}"
            )
        self.hidden_size = hidden_size
        self.intermediate_local = intermediate_local
        self.ctx = ctx
        self.weight = nn.Parameter(torch.empty(2 * intermediate_local, hidden_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(2 * intermediate_local))
        else:
            self.register_parameter("bias", None)
        if init_weights:
            nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = F.linear(x, self.weight, self.bias)
        return torch.chunk(y, 2, dim=-1)  # (gate, up) views

    def forward_unfused(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Reference v0.1 pattern: 2 GEMMs on row views of the same parameter.
        n = self.intermediate_local
        b = self.bias
        return (
            F.linear(x, self.weight[:n], b[:n] if b is not None else None),
            F.linear(x, self.weight[n:], b[n:] if b is not None else None),
        )
