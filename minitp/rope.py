"""RoPE (rotary position embedding), HF-Qwen2-compatible.

HF Qwen2 pairs dim i with i + head_dim/2 (rotate_half convention), NOT
interleaved (2i, 2i+1) — this must match the reference exactly.

v0.2: ``RotaryEmbedding`` precomputes cos/sin tables once (fp32) instead of
rebuilding inv_freq/freqs/cos/sin in every layer on every forward
(24 layers × per token — see docs/PERFORMANCE_AUDIT.md §8). Tables are fp32
for precision; outputs are cast back to the input dtype.
"""

from __future__ import annotations

import torch


class RotaryEmbedding:
    """Precomputed rotary tables for absolute positions [0, max_seq_len)."""

    def __init__(
        self,
        head_dim: int,
        theta: float,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
        )
        pos = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        freqs = pos[:, None] * inv_freq[None, :]  # [S, hd/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [S, hd]
        self.cos_cache = emb.cos()  # fp32 [max_seq_len, head_dim]
        self.sin_cache = emb.sin()
        self.max_seq_len = max_seq_len

    def cos_sin(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather [T, head_dim] fp32 cos/sin rows for absolute positions.

        No device->CPU bounds check here on purpose: ``int(positions.max())``
        would synchronize the GPU every layer every token. Out-of-range
        positions raise a device-side indexing assert; callers validate
        sequence length once (see ModelConfig.max_position_embeddings).
        """
        return self.cos_cache[positions], self.sin_cache[positions]

    def apply(
        self,
        q: torch.Tensor,  # [B, H, T, head_dim]
        k: torch.Tensor,
        positions: torch.Tensor,  # [T]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos, sin = self.cos_sin(positions)
        cos = cos[None, None]  # [1, 1, T, hd]
        sin = sin[None, None]

        def rotate_half(x: torch.Tensor) -> torch.Tensor:
            x1, x2 = x.float().chunk(2, dim=-1)
            return torch.cat([-x2, x1], dim=-1)

        def rotate(x: torch.Tensor) -> torch.Tensor:
            return (x.float() * cos + rotate_half(x) * sin).to(x.dtype)

        return rotate(q), rotate(k)


def apply_rope(
    q: torch.Tensor,  # [B, num_heads, T, head_dim]
    k: torch.Tensor,
    positions: torch.Tensor,  # [T] absolute positions
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference (uncached) implementation — kept for correctness tests and
    as the microbenchmark baseline; the model uses RotaryEmbedding.apply."""
    head_dim = q.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    freqs = positions.to(q.device).float()[:, None] * inv_freq[None, :]  # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, hd]
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        return (x.float() * cos + rotate_half(x) * sin).to(x.dtype)

    return rotate(q), rotate(k)
