"""Per-rank contiguous KV cache: only local KV heads are stored."""

from __future__ import annotations

import torch


class KVCache:
    """One k/v buffer per layer, layout [batch, kv_heads_local, max_seq_len, head_dim].

    ``seq_len`` is the fill cursor; only ``[:, :, :seq_len]`` is ever read —
    the write-before-read invariant that makes ``init="empty"`` safe
    (NaN-poison tested in tests/test_kv_cache.py, K and V separately).

    ``init``: "empty" (default, skips the zero-fill kernels) or "zeros".
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        kv_heads_local: int,
        head_dim: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        init: str = "empty",
    ) -> None:
        if init not in ("empty", "zeros"):
            raise ValueError(f"init must be 'empty' or 'zeros', got {init!r}")
        self.max_seq_len = max_seq_len
        self.seq_len = 0
        alloc = torch.empty if init == "empty" else torch.zeros
        self.k = [
            alloc(batch_size, kv_heads_local, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v = [
            alloc(batch_size, kv_heads_local, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

    def update(
        self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append [B, H, T, D] at the cursor; returns the full valid view."""
        t = k_new.shape[2]
        end = self.seq_len + t
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: seq {self.seq_len}+{t} > max_seq_len={self.max_seq_len}"
            )
        self.k[layer_idx][:, :, self.seq_len : end] = k_new
        self.v[layer_idx][:, :, self.seq_len : end] = v_new
        return self.k[layer_idx][:, :, :end], self.v[layer_idx][:, :, :end]

    def advance(self, tokens: int) -> None:
        self.seq_len += tokens

    def poison(self, value: float = float("nan")) -> None:
        """Test helper: fill every slot (written or not) with a sentinel."""
        for buf in (*self.k, *self.v):
            buf.fill_(value)
