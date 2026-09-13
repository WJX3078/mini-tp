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

# Precision-safe default: bitpack keeps one int64 per candidate and is exact
# for any vocab < 2^32 (the fp32 legacy encoding rounds odd ids >= 2^24).
ARGMAX_ENCODING = "bitpack"


class VocabParallelEmbedding(nn.Module):
    """Embedding table row-sharded along vocab; per-rank out-of-shard lookups
    contribute zeros and an AllReduce restores the full embedding. Uneven
    vocab splits use contiguous ranges (no padding rows needed)."""

    def __init__(
        self, vocab_size: int, hidden_size: int, ctx: ParallelContext, init_weights: bool = True
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ctx = ctx
        start, end = shard_range(vocab_size, ctx.tp_rank, ctx.tp_size)
        self.vocab_start, self.vocab_end = start, end
        self.weight = nn.Parameter(
            torch.zeros(end - start, hidden_size) if init_weights
            else torch.empty(end - start, hidden_size)
        )
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
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ctx = ctx
        self.gather_logits = gather_logits
        start, end = shard_range(vocab_size, ctx.tp_rank, ctx.tp_size)
        self.vocab_start, self.vocab_end = start, end
        self.weight = nn.Parameter(
            torch.zeros(end - start, hidden_size) if init_weights
            else torch.empty(end - start, hidden_size)
        )
        # persistent gather buffer for distributed_argmax (reused across tokens)
        self._gather_buf: torch.Tensor | None = None
        self._gather_buf_shape: tuple[int, ...] | None = None
        self._gather_buf_dtype: torch.dtype | None = None

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
    def distributed_argmax(
        self, local_logits: torch.Tensor, encoding: str | None = None
    ) -> torch.Tensor:
        """Greedy next-token id over sharded vocab without gathering logits.

        Input: this rank's logits [..., vocab_local]. Per rank: (local max
        value, local argmax + vocab_start); ONE O(tp) gather (not O(vocab));
        global winner = max value, ties broken toward the smaller token id to
        match torch.argmax. Returns [..., 1] long tensor, identical on all
        ranks.

        Encodings (``encoding`` or module default ``ARGMAX_ENCODING``):

        - ``"bitpack"`` (default, precision-safe for any vocab < 2^32):
          monotone fp32->uint32 key transform (sign-flip trick, -0.0
          normalized to +0.0), packed ``(key << 32) | (2^32-1 - id)`` into one
          int64 per candidate. Max over ranks picks the largest key; equal
          keys resolve to the SMALLEST id by construction. NaN policy: a rank
          whose local max is NaN cannot win (NaN -> -inf before encoding).
        - ``"fp32"`` (legacy): values+ids in fp32 — exact only for
          ``vocab_size <= 2**24``; asserted, not silent (v0.1/v0.4 audit B7).
        - ``"fp64"``: [value, id] as one fp64 pair — ids exact < 2^53.
        - ``"split"``: two gathers (fp32 values, int64 ids) — reference.

        The gather buffer is persistent across tokens (shape-keyed); all
        layouts are gathered along dim 0 as [tp, *leading, C].
        """
        if self.ctx.tp_size == 1:
            return local_logits.argmax(dim=-1, keepdim=True)  # fast path
        enc = encoding or ARGMAX_ENCODING
        # NaN policy (documented): a NaN logit cannot win. Sanitize BEFORE the
        # local max so a NaN cannot shadow a real logit in the same shard; a
        # rank that ends up all-NaN simply loses to any finite value.
        clean = torch.where(
            torch.isnan(local_logits), torch.full_like(local_logits, float("-inf")), local_logits
        )
        values, idx = clean.max(dim=-1)  # [...]
        global_ids = idx + self.vocab_start

        if enc == "bitpack":
            vals = torch.where(values == 0, torch.zeros_like(values), values)  # normalize -0.0
            bits = vals.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
            # monotone map into signed 31 bits so (key << 32) stays a positive
            # int64: positive floats keep their bits, negatives fold to
            # ~(bits) - 2^31. Order-preserving across the whole float range.
            key = torch.where(
                bits < 0x80000000, bits, ((~bits) & 0xFFFFFFFF) - 0x80000000
            )
            packed = (key << 32) | (0xFFFFFFFF - global_ids.to(torch.int64))
            payload = packed.unsqueeze(-1)  # [..., 1]
            gathered = self._gather(payload, torch.int64)
            best = gathered.movedim(0, -2).squeeze(-1)  # [..., tp]
            winner = best.max(dim=-1).values
            return (0xFFFFFFFF - (winner & 0xFFFFFFFF)).unsqueeze(-1).to(torch.long)

        if enc == "fp32":
            if self.vocab_size > 2**24:
                raise ValueError(
                    f"fp32 argmax encoding cannot represent ids >= 2**24 exactly "
                    f"(vocab_size={self.vocab_size}); use the bitpack encoding"
                )
            pair = torch.stack([values.float(), global_ids.float()], dim=-1)
            gathered = self._gather(pair, torch.float32)
            stack = gathered.movedim(0, -2)  # [..., tp, 2]
            best_value = stack[..., 0].max(dim=-1).values
            is_best = stack[..., 0] == best_value.unsqueeze(-1)
            token = torch.where(
                is_best, stack[..., 1], torch.full_like(stack[..., 1], float("inf"))
            )
            return token.min(dim=-1).values.unsqueeze(-1).to(torch.long)

        if enc == "fp64":
            pair = torch.stack([values.double(), global_ids.double()], dim=-1)
            gathered = self._gather(pair, torch.float64)
            stack = gathered.movedim(0, -2)
            best_value = stack[..., 0].max(dim=-1).values
            is_best = stack[..., 0] == best_value.unsqueeze(-1)
            token = torch.where(
                is_best, stack[..., 1], torch.full_like(stack[..., 1], float("inf"))
            )
            return token.min(dim=-1).values.unsqueeze(-1).to(torch.long)

        if enc == "split":
            gvals = self._gather(values.float().unsqueeze(-1), torch.float32)
            gids = self._gather(global_ids.unsqueeze(-1), torch.int64)
            vstack = gvals.movedim(0, -2)  # [..., tp, 1]
            istack = gids.movedim(0, -2)
            best_value = vstack.max(dim=-2).values  # [..., 1]
            is_best = vstack == best_value
            token = torch.where(
                is_best, istack, torch.full_like(istack, torch.iinfo(torch.int64).max)
            )
            return token.amin(dim=-2).to(torch.long)

        raise ValueError(f"unknown argmax encoding {enc!r}")

    def _gather(self, payload: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Gather [tp, *leading, C] along dim 0 with a persistent buffer."""
        shape = (self.ctx.tp_size, *payload.shape)
        if (
            self._gather_buf is None
            or self._gather_buf_shape != shape
            or self._gather_buf.dtype != dtype
            or self._gather_buf.device != payload.device
        ):
            self._gather_buf = torch.empty(shape, dtype=dtype, device=payload.device)
            self._gather_buf_shape = shape
            self._gather_buf_dtype = dtype
        all_gather_dim0(self._gather_buf, payload, self.ctx.process_group)
        return self._gather_buf
