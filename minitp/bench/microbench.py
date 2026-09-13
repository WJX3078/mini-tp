"""Kernel-level microbenchmarks: old (v0.1) vs optimized (v0.2) hot-path ops.

Isolates the changes from docs/PERFORMANCE_AUDIT.md on realistic Qwen2.5-0.5B
TP shapes (bf16, batch=1, single GPU; CPU fallback just runs slower):

1. RoPE: reference (uncached, per-forward table construction) vs cached
   RotaryEmbedding.apply, decode (T=1) and prefill (T=512) shapes.
2. QKV: three separate GEMMs vs one fused GEMM + split (TP=1 and TP=2 shapes).
3. Gate/Up: two GEMMs vs one fused GEMM + chunk.
4. Sequence bookkeeping: torch.cat per token vs preallocated buffer writes.

Usage:  python -m minitp.bench.microbench [--dtype bf16] [--iters 200]
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from minitp.rope import RotaryEmbedding, apply_rope


def _cuda_time_ms(fn, iters: int, warmup: int = 20) -> float:
    use_cuda = torch.cuda.is_available()
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters
    import time

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3


def bench_rope(b: int, t: int, hd: int = 64, theta: float = 1e6) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    q = torch.randn(b, 14, t, hd, device=device, dtype=dtype)
    k = torch.randn(b, 2, t, hd, device=device, dtype=dtype)
    positions = torch.arange(t, device=device)
    cached = RotaryEmbedding(hd, theta, 32768, device)
    ref = lambda: apply_rope(q, k, positions, theta)  # noqa: E731
    fast = lambda: cached.apply(q, k, positions)  # noqa: E731
    torch.testing.assert_close(fast()[0], ref()[0], atol=1e-2, rtol=1e-2)
    return {
        "case": f"rope B={b} T={t}",
        "reference_ms": _cuda_time_ms(ref, 200),
        "optimized_ms": _cuda_time_ms(fast, 200),
    }


def _qkv_fns(b, hidden, q_rows, kv_rows, device, dtype):
    x = torch.randn(b, 1, hidden, device=device, dtype=dtype)
    wq = torch.randn(q_rows, hidden, device=device, dtype=dtype)
    wk = torch.randn(kv_rows, hidden, device=device, dtype=dtype)
    wv = torch.randn(kv_rows, hidden, device=device, dtype=dtype)
    bq = torch.randn(q_rows, device=device, dtype=dtype)
    bk = torch.randn(kv_rows, device=device, dtype=dtype)
    bv = torch.randn(kv_rows, device=device, dtype=dtype)
    w_fused = torch.cat([wq, wk, wv], dim=0).contiguous()
    b_fused = torch.cat([bq, bk, bv], dim=0).contiguous()

    def three():
        q = F.linear(x, wq, bq)
        k = F.linear(x, wk, bk)
        v = F.linear(x, wv, bv)
        return q, k, v

    def fused():
        y = F.linear(x, w_fused, b_fused)
        return torch.split(y, [q_rows, kv_rows, kv_rows], dim=-1)

    torch.testing.assert_close(fused()[0], three()[0], atol=1e-2, rtol=1e-2)
    return three, fused


def _gate_up_fns(b, hidden, inter_local, device, dtype):
    x = torch.randn(b, 1, hidden, device=device, dtype=dtype)
    wg = torch.randn(inter_local, hidden, device=device, dtype=dtype)
    wu = torch.randn(inter_local, hidden, device=device, dtype=dtype)
    w_fused = torch.cat([wg, wu], dim=0).contiguous()

    def two():
        return F.linear(x, wg), F.linear(x, wu)

    def fused():
        return torch.chunk(F.linear(x, w_fused), 2, dim=-1)

    torch.testing.assert_close(fused()[0], two()[0], atol=1e-2, rtol=1e-2)
    return two, fused


def bench_gemm(hidden, q_rows, kv_rows, inter_local) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    three, fused = _qkv_fns(1, hidden, q_rows, kv_rows, device, dtype)
    two, fused_gu = _gate_up_fns(1, hidden, inter_local, device, dtype)
    return [
        {
            "case": f"qkv B=1 (q{q_rows}+kv{kv_rows}x2)",
            "reference_ms": _cuda_time_ms(three, 200),
            "optimized_ms": _cuda_time_ms(fused, 200),
        },
        {
            "case": f"gate_up B=1 (inter {inter_local})",
            "reference_ms": _cuda_time_ms(two, 200),
            "optimized_ms": _cuda_time_ms(fused_gu, 200),
        },
    ]


def bench_bookkeeping(steps: int = 256) -> dict:
    """v0.1 (cat per token + finished.all() sync per token) vs v0.2 (prealloc,
    sync-free). The buffer write alone is launch-bound like cat; the real win
    is removing the per-token GPU->CPU synchronization and allocator churn."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = torch.tensor([[12095]], device=device)
    finished = torch.zeros(1, dtype=torch.bool, device=device)

    def old_cat_with_sync():
        out = torch.empty(1, 0, dtype=torch.long, device=device)
        for _ in range(steps):
            out = torch.cat([out, tok], dim=1)
            if bool(finished.all()):  # v0.1 did this every token: hard sync
                break
        return out

    def prealloc_nosync():
        out = torch.empty(1, steps, dtype=torch.long, device=device)
        for i in range(steps):
            out[:, i] = tok
        return out

    torch.testing.assert_close(old_cat_with_sync()[:, :steps], prealloc_nosync())
    return {
        "case": f"decode bookkeeping {steps} tok",
        "reference_ms": _cuda_time_ms(old_cat_with_sync, 20, warmup=5),
        "optimized_ms": _cuda_time_ms(prealloc_nosync, 20, warmup=5),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="mini-TP hot-path microbenchmarks")
    p.add_argument("--iters", type=int, default=200)
    p.parse_args(argv)

    rows = [
        bench_rope(1, 1),
        bench_rope(1, 512),
        *bench_gemm(896, 896, 128, 2432),
        bench_bookkeeping(),
    ]
    print(f"{'case':<28}{'reference ms':>14}{'optimized ms':>14}{'speedup':>9}")
    for r in rows:
        sp = r["reference_ms"] / r["optimized_ms"] if r["optimized_ms"] > 0 else 0
        print(f"{r['case']:<28}{r['reference_ms']:>14.4f}{r['optimized_ms']:>14.4f}{sp:>8.2f}x")


if __name__ == "__main__":
    main()
