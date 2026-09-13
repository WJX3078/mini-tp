"""Vocab-parallel embedding and LM head, plus distributed greedy argmax.

v0.3: TP=1 fast paths (no vocab masking / no max-reduce bookkeeping), the
TP>1 argmax uses a persistent contiguous gather buffer instead of per-token
list/stack allocations, and gather_logits handles uneven vocab shards by
padding (equal-size collective requirement).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from minitp.distributed import all_gather, all_gather_dim0, all_reduce
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
        self.tp1_fast = True  # ablation toggle

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if self.ctx.tp_size == 1 and self.tp1_fast:
            return F.embedding(ids, self.weight)  # fast path: no mask, no reduce
        mask = (ids >= self.vocab_start) & (ids < self.vocab_end)
        local_ids = (ids - self.vocab_start).clamp(min=0) * mask
        out = F.embedding(local_ids, self.weight) * mask.unsqueeze(-1).to(self.weight.dtype)
        return all_reduce(out, self.ctx.process_group)


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
        # persistent gather buffer for distributed_argmax (reused across tokens)
        self._gather_buf: torch.Tensor | None = None
        self._gather_buf_shape: tuple[int, ...] | None = None

    def _max_local_vocab(self) -> int:
        base, rem = divmod(self.vocab_size, self.ctx.tp_size)
        return base + (1 if rem else 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.linear(x, self.weight)  # [..., vocab_local]
        if not self.gather_logits or self.ctx.tp_size == 1:
            return local
        out = torch.empty(
            *local.shape[:-1], self.vocab_size, dtype=local.dtype, device=local.device
        )
        # NOTE: branch on the GLOBAL vocab/tp property, never on the local
        # shape — a local-shape branch diverges across ranks when the vocab
        # is uneven and aborts the collective.
        if self.vocab_size % self.ctx.tp_size == 0:
            all_gather(out, local, self.ctx.process_group)
            return out
        # uneven shards: pad to the max local vocab so the collective is
        # equal-size on every rank, then trim each rank's column range
        pad = self._max_local_vocab() - local.shape[-1]
        padded = F.pad(local, (0, pad))
        gathered = torch.empty(
            *local.shape[:-1],
            self.ctx.tp_size * self._max_local_vocab(),
            dtype=local.dtype,
            device=local.device,
        )
        all_gather(gathered, padded, self.ctx.process_group)
        for r in range(self.ctx.tp_size):
            s, e = shard_range(self.vocab_size, r, self.ctx.tp_size)
            out[..., s:e] = gathered[
                ..., r * self._max_local_vocab() : r * self._max_local_vocab() + (e - s)
            ]
        return out

    @torch.no_grad()
    def distributed_argmax(self, local_logits: torch.Tensor) -> torch.Tensor:
        """Greedy next-token id over sharded vocab without gathering logits.

        Input: this rank's logits [..., vocab_local]. Per rank: (local max
        value, local argmax + vocab_start); gather the pairs (O(tp) bytes, not
        O(vocab)); global winner = max value, ties broken toward the smaller
        token id to match torch.argmax. Returns [..., 1] long tensor,
        identical on all ranks.

        Numerical contract: values are cast to fp32 BEFORE pairing — the
        v0.1 bf16-stack bug rounded token ids > 256 (12095 -> 12096). Ids stay
        int64 end to end. The gather buffer is persistent across tokens
        (shape-keyed) instead of a per-token list/stack.
        """
        leading = local_logits.shape[:-1]
        if self.ctx.tp_size == 1:
            return local_logits.argmax(dim=-1, keepdim=True)  # fast path
        values, idx = local_logits.max(dim=-1)  # [...]
        global_ids = idx + self.vocab_start
        # fp32 values + int64->fp32 ids (ids < 2^24 stay exact)
        pair = torch.stack([values.float(), global_ids.float()], dim=-1)  # [..., 2]
        # all_gather_dim0 stacks rank blocks along dim 0: buffer must be
        # [tp, *leading, 2], then moved to [..., tp, 2] (zero-copy view).
        # (Writing [*leading, tp, 2] is silently wrong when B == tp.)
        shape = (self.ctx.tp_size, *leading, 2)
        if (
            self._gather_buf is None
            or self._gather_buf_shape != shape
            or self._gather_buf.device != pair.device
        ):
            self._gather_buf = torch.empty(shape, dtype=torch.float32, device=pair.device)
            self._gather_buf_shape = shape
        all_gather_dim0(self._gather_buf, pair, self.ctx.process_group)
        stack = self._gather_buf.movedim(0, -2)  # [..., tp, 2] view
        best_value = stack[..., 0].max(dim=-1).values  # [...]
        is_best = stack[..., 0] == best_value.unsqueeze(-1)
        token = torch.where(
            is_best, stack[..., 1], torch.full_like(stack[..., 1], float("inf"))
        )
        return token.min(dim=-1).values.unsqueeze(-1).to(torch.long)
