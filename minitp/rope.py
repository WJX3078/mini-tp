"""RoPE (rotary position embedding), HF-Qwen2-compatible.

HF Qwen2 pairs dim i with i + head_dim/2 (rotate_half convention), NOT
interleaved (2i, 2i+1) — this must match the reference exactly.
"""

from __future__ import annotations

import torch


def _rope_inv_freq(head_dim: int, theta: float, device) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))


def apply_rope(
    q: torch.Tensor,  # [B, num_heads, T, head_dim]
    k: torch.Tensor,
    positions: torch.Tensor,  # [T] int64 absolute positions
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = q.shape[-1]
    inv_freq = _rope_inv_freq(head_dim, theta, q.device)  # [hd/2]
    freqs = positions.to(q.device).float()[:, None] * inv_freq[None, :]  # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)  # [T, hd]
    cos, sin = emb.cos()[None, None], emb.sin()[None, None]

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        return (x.float() * cos + rotate_half(x) * sin).to(x.dtype)

    return rotate(q), rotate(k)
