"""Vocab-parallel embedding and LM head, plus distributed greedy argmax."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from minitp.distributed import all_gather, all_reduce
from minitp.distributed.context import ParallelContext
from minitp.parallel.linear import shard_range


class VocabParallelEmbedding(nn.Module):
    """Embedding table row-sharded along vocab; per-rank out-of-shard lookups
    contribute zeros and an AllReduce restores the full embedding. Uneven
    vocab splits use contiguous ranges (no padding rows needed)."""

    def __init__(self, vocab_size: int, hidden_size: int, ctx: ParallelContext) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ctx = ctx
        start, end = shard_range(vocab_size, ctx.tp_rank, ctx.tp_size)
        self.vocab_start, self.vocab_end = start, end
        self.weight = nn.Parameter(torch.zeros(end - start, hidden_size))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids >= self.vocab_start) & (ids < self.vocab_end)
        local_ids = (ids - self.vocab_start).clamp(min=0) * mask
        out = F.embedding(local_ids, self.weight) * mask.unsqueeze(-1).to(self.weight.dtype)
        if self.ctx.tp_size > 1:
            out = all_reduce(out, self.ctx.process_group)
        return out


class VocabParallelLMHead(nn.Module):
    """hidden -> vocab, column-sharded over vocab. Default mode returns local
    logits [B, T, vocab_local]; gather_logits=True AllGathers the full vocab
    (debug/correctness). Use distributed_argmax for inference-time greedy
    sampling without gathering O(vocab) logits."""

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        ctx: ParallelContext,
        gather_logits: bool = False,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ctx = ctx
        self.gather_logits = gather_logits
        start, end = shard_range(vocab_size, ctx.tp_rank, ctx.tp_size)
        self.vocab_start, self.vocab_end = start, end
        self.weight = nn.Parameter(torch.zeros(end - start, hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.linear(x, self.weight)  # [..., vocab_local]
        if not self.gather_logits or self.ctx.tp_size == 1:
            return local
        out = torch.empty(
            *local.shape[:-1], self.vocab_size, dtype=local.dtype, device=local.device
        )
        return all_gather(out, local, self.ctx.process_group)

    @torch.no_grad()
    def distributed_argmax(self, local_logits: torch.Tensor) -> torch.Tensor:
        """Greedy next-token id over sharded vocab without gathering logits.

        Input: this rank's logits [..., vocab_local]. Per rank: (local max,
        local argmax + vocab_start); AllGather the pair per row (O(tp) bytes,
        not O(vocab)); global winner = max value, ties broken toward the
        smaller token id to match torch.argmax. Returns [..., 1] long tensor,
        identical on all ranks.
        """
        values, idx = local_logits.max(dim=-1)  # [...], int64
        global_ids = idx + self.vocab_start
        if self.ctx.tp_size == 1:
            return global_ids.unsqueeze(-1)
        # cast to fp32 BEFORE stacking: stack(bf16, int64) promotes to bf16,
        # which cannot represent vocab ids > 256 exactly (12095 -> 12096!)
        pair = torch.stack([values.float(), global_ids.float()], dim=-1)
        gathered = [
            torch.empty_like(pair) for _ in range(self.ctx.tp_size)
        ]
        dist.all_gather(gathered, pair, group=self.ctx.process_group)
        stack = torch.stack(gathered, dim=-2)  # [..., tp, 2]
        best_value = stack[..., 0].max(dim=-1).values  # [...]
        # smallest token id among ranks achieving the best value
        is_best = stack[..., 0] == best_value.unsqueeze(-1)
        token = torch.where(is_best, stack[..., 1], torch.full_like(stack[..., 1], float("inf")))
        return token.min(dim=-1).values.unsqueeze(-1).to(torch.long)
