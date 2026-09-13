"""RoPE cache correctness: RotaryEmbedding (cached) vs apply_rope (reference)."""

import pytest
import torch

from minitp.rope import RotaryEmbedding, apply_rope

HEAD_DIM = 16
THETA = 10000.0
MAX_SEQ = 128


def _rotary(device) -> RotaryEmbedding:
    return RotaryEmbedding(HEAD_DIM, THETA, MAX_SEQ, device)


@pytest.mark.parametrize("t", [1, 2, 7, 33])
@pytest.mark.parametrize("offset", [0, 1, 95])
def test_cached_matches_reference(t, offset):
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rotary = _rotary(dev)
    q = torch.randn(2, 4, t, HEAD_DIM, device=dev)
    k = torch.randn(2, 2, t, HEAD_DIM, device=dev)
    positions = torch.arange(offset, offset + t, device=dev)
    q_ref, k_ref = apply_rope(q, k, positions, THETA)
    q_cached, k_cached = rotary.apply(q, k, positions)
    torch.testing.assert_close(q_cached, q_ref, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(k_cached, k_ref, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cached_bf16_output_dtype_preserved():
    dev = torch.device("cuda")
    rotary = RotaryEmbedding(HEAD_DIM, THETA, MAX_SEQ, dev)
    q = torch.randn(1, 2, 3, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    k = torch.randn(1, 2, 3, HEAD_DIM, device=dev, dtype=torch.bfloat16)
    q_out, k_out = rotary.apply(q, k, torch.arange(3, device=dev))
    assert q_out.dtype == torch.bfloat16 and k_out.dtype == torch.bfloat16
    q_ref, k_ref = apply_rope(q, k, torch.arange(3, device=dev), THETA)
    torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)


def test_out_of_range_positions_device_assert():
    """Out-of-cache positions must fail loudly, not silently wrap."""
    rotary = _rotary(torch.device("cpu"))
    q = torch.randn(1, 1, 1, HEAD_DIM)
    with pytest.raises(IndexError):
        rotary.apply(q, q, torch.tensor([MAX_SEQ + 5]))
