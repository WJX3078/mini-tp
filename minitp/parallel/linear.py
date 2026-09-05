"""Column- and Row-parallel linear layers (Megatron-style, from scratch)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minitp.distributed import all_gather, all_reduce
from minitp.distributed.context import ParallelContext


def shard_range(size: int, tp_rank: int, tp_size: int) -> tuple[int, int]:
    """Contiguous [start, end) slice for a rank; supports size % tp_size != 0 (uneven)."""
    base, rem = divmod(size, tp_size)
    start = tp_rank * base + min(tp_rank, rem)
    return start, start + base + (1 if tp_rank < rem else 0)


class ColumnParallelLinear(nn.Module):
    """Y = XW with W's *output* dimension sharded across the TP group.

    PyTorch weight layout [out_features, in_features] means the shard is along
    dim 0. Output stays sharded along the last dim unless gather_output=True.
    Bias is sharded identically (each rank owns its output slice), so local
    adds sum exactly to the full bias — no double counting.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        ctx: ParallelContext,
        bias: bool = True,
        gather_output: bool = False,
    ) -> None:
        super().__init__()
        if output_size % ctx.tp_size != 0:
            raise ValueError(
                f"output_size={output_size} not divisible by tp_size={ctx.tp_size}"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.ctx = ctx
        self.gather_output = gather_output
        self.output_size_local = output_size // ctx.tp_size
        start, _ = shard_range(output_size, ctx.tp_rank, ctx.tp_size)
        self.output_start = start
        self.weight = nn.Parameter(torch.empty(self.output_size_local, input_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.output_size_local))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.linear(x, self.weight, self.bias)
        if not self.gather_output:
            return local
        if self.ctx.tp_size == 1:
            return local
        out = torch.empty(
            *local.shape[:-1],
            self.output_size,
            dtype=local.dtype,
            device=local.device,
        )
        return all_gather(out, local, self.ctx.process_group)


class RowParallelLinear(nn.Module):
    """Y = XW with W's *input* dimension sharded; partial sums AllReduced.

    With input_is_parallel=True the input's last dim is already the local
    shard (the usual post-ColumnParallel case). Bias is added exactly once,
    *after* the reduction — adding it per-rank before AllReduce would scale
    the bias by tp_size.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        ctx: ParallelContext,
        bias: bool = True,
        input_is_parallel: bool = True,
        reduce_output: bool = True,
    ) -> None:
        super().__init__()
        if input_size % ctx.tp_size != 0:
            raise ValueError(
                f"input_size={input_size} not divisible by tp_size={ctx.tp_size}"
            )
        self.input_size = input_size
        self.output_size = output_size
        self.ctx = ctx
        self.input_is_parallel = input_is_parallel
        self.reduce_output = reduce_output
        self.input_size_local = input_size // ctx.tp_size
        self.weight = nn.Parameter(torch.empty(output_size, self.input_size_local))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
        else:
            self.register_parameter("bias", None)
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.input_is_parallel:
            if self.ctx.tp_size > 1:
                x = x.split(self.input_size_local, dim=-1)[self.ctx.tp_rank]
            else:
                x = x
        local = F.linear(x, self.weight)  # partial sum, bias added post-reduce
        if self.reduce_output and self.ctx.tp_size > 1:
            local = all_reduce(local, self.ctx.process_group)
        if self.bias is not None:
            local = local + self.bias
        return local
