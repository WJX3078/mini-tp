"""Kernel-level microbenchmarks: old (v0.1) vs optimized (v0.3) hot-path ops.

Isolates the changes from docs/PERFORMANCE_AUDIT.md on realistic Qwen2.5-0.5B
TP shapes (bf16, batch=1, single GPU; CPU fallback just runs slower):

1. RoPE: reference (uncached, per-forward table construction) vs cached
   RotaryEmbedding.apply, decode (T=1) and prefill (T=512) shapes.
2. QKV: three separate GEMMs vs one fused GEMM + split (TP=1 shapes).
3. Gate/Up: two GEMMs vs one fused GEMM + chunk.
4. Decode bookkeeping: v0.1 (cat per token + finished.all() sync) vs v0.2+
   (preallocated buffer, sync-free).
5. Embedding: v0.1 masked vocab path vs TP=1 fast path (T=1 / 128 / 2048).
6. Argmax: legacy max+index bookkeeping vs direct argmax (TP=1).

Usage:  python -m minitp.bench.microbench [--iters N] [--warmup N] [--json]
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from minitp.rope import RotaryEmbedding, apply_rope


def _cuda_time_ms(fn, iters: int, warmup: int) -> float:
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


def bench_rope(b, t, hd, theta, iters, warmup) -> dict:
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
        "reference_ms": _cuda_time_ms(ref, iters, warmup),
        "optimized_ms": _cuda_time_ms(fast, iters, warmup),
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
        return (F.linear(x, wq, bq), F.linear(x, wk, bk), F.linear(x, wv, bv))

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


def bench_gemm(hidden, q_rows, kv_rows, inter_local, iters, warmup) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    three, fused = _qkv_fns(1, hidden, q_rows, kv_rows, device, dtype)
    two, fused_gu = _gate_up_fns(1, hidden, inter_local, device, dtype)
    return [
        {
            "case": f"qkv B=1 (q{q_rows}+kv{kv_rows}x2)",
            "reference_ms": _cuda_time_ms(three, iters, warmup),
            "optimized_ms": _cuda_time_ms(fused, iters, warmup),
        },
        {
            "case": f"gate_up B=1 (inter {inter_local})",
            "reference_ms": _cuda_time_ms(two, iters, warmup),
            "optimized_ms": _cuda_time_ms(fused_gu, iters, warmup),
        },
    ]


def bench_bookkeeping(steps, iters, warmup) -> dict:
    """v0.1 (cat per token + finished.all() sync per token) vs v0.2+
    (prealloc, sync-free). The buffer write alone is launch-bound like cat;
    the real win is removing the per-token GPU->CPU synchronization."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = torch.tensor([[12095]], device=device)
    finished = torch.zeros(1, dtype=torch.bool, device=device)
    it, wu = max(1, iters // 10), max(1, warmup // 10)

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
        "reference_ms": _cuda_time_ms(old_cat_with_sync, it, wu),
        "optimized_ms": _cuda_time_ms(prealloc_nosync, it, wu),
    }


def bench_embedding(iters, warmup) -> list[dict]:
    """v0.1 masked vocab path vs TP=1 fast path (F.embedding)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    table = torch.randn(151936, 896, device=device, dtype=torch.bfloat16)

    for t in (1, 128, 2048):
        ids = torch.randint(0, 151936, (1, t), device=device)
        mask = (ids >= 0) & (ids < 151936)
        local_ids = (ids - 0).clamp(min=0) * mask

        def old_masked(local_ids=local_ids, mask=mask, table=table):
            return F.embedding(local_ids, table) * mask.unsqueeze(-1).to(table.dtype)

        def fast(ids=ids, table=table):
            return F.embedding(ids, table)

        torch.testing.assert_close(fast().float(), old_masked().float())
        rows.append({
            "case": f"embedding TP1 T={t}",
            "reference_ms": _cuda_time_ms(old_masked, iters, warmup),
            "optimized_ms": _cuda_time_ms(fast, iters, warmup),
        })
    return rows


def bench_argmax(iters, warmup) -> dict:
    """Legacy (values, idx) bookkeeping vs direct argmax at TP=1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.randn(1, 151936, device=device, dtype=torch.bfloat16)

    def legacy():
        values, idx = logits.max(dim=-1)
        global_ids = idx  # + vocab_start (0 at TP=1)
        return torch.stack([values.float(), global_ids.float()], dim=-1)

    def fast():
        return logits.argmax(dim=-1, keepdim=True)

    return {
        "case": "argmax TP1 vocab=151936",
        "reference_ms": _cuda_time_ms(legacy, iters, warmup),
        "optimized_ms": _cuda_time_ms(fast, iters, warmup),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="mini-TP hot-path microbenchmarks")
    p.add_argument("--iters", type=int, default=200,
                   help="measured iterations per case (bookkeeping uses iters/10)")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = p.parse_args(argv)

    rows = [
        bench_rope(1, 1, 64, 1e6, args.iters, args.warmup),
        bench_rope(1, 512, 64, 1e6, args.iters, args.warmup),
        *bench_gemm(896, 896, 128, 2432, args.iters, args.warmup),
        bench_bookkeeping(256, args.iters, args.warmup),
        *bench_embedding(args.iters, args.warmup),
        bench_argmax(args.iters, args.warmup),
    ]
    for r in rows:
        r["speedup"] = round(r["reference_ms"] / r["optimized_ms"], 2)

    if args.json:
        print(json.dumps({"schema": "minitp.microbench/1", "iters": args.iters,
                          "warmup": args.warmup, "cases": rows}, indent=2))
        return
    print(f"{'case':<30}{'reference ms':>14}{'optimized ms':>14}{'speedup':>9}")
    for r in rows:
        print(f"{r['case']:<30}{r['reference_ms']:>14.4f}{r['optimized_ms']:>14.4f}"
              f"{r['speedup']:>8.2f}x")


if __name__ == "__main__":
    main()
