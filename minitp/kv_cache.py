"""Per-rank contiguous KV cache: only local KV heads are stored."""

from __future__ import annotations

import torch


class KVCache:
    """One k/v buffer per layer, layout [batch, kv_heads_local, max_seq_len, head_dim].

    `seq_len` is the fill cursor; slices [:, :, :seq_len] are valid.
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
    ) -> None:
        self.max_seq_len = max_seq_len
        self.seq_len = 0
        self.k = [
            torch.zeros(batch_size, kv_heads_local, max_seq_len, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v = [
            torch.zeros_like(kb) for kb in self.k
        ]

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
