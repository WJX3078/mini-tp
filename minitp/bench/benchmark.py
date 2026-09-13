"""mini-TP benchmark: phase-separated latency/throughput/memory/communication.

Methodology (docs/BENCHMARK.md):
- checkpoint load time measured separately and excluded from compute windows
- prefill timed alone (prompt forward, CUDA events)
- decode timed **per token** with fixed-length, sync-free stepping
  (early_stop=False) -> p50/p90/p99 of the true per-token latency
- e2e = prefill + mean(decode); never reported as "decode ms/token"
- memory: peak allocated/reserved, reset after load so decode peaks are clean
- communication: optional per-op host_launch/gpu_elapsed records

Compare TP=1 vs TP=2 by running under different torchrun --nproc-per-node.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoConfig, AutoTokenizer

from minitp.config import ModelConfig
from minitp.distributed import comm_summary, get_comm_stats, reset_comm_stats, set_profiling
from minitp.distributed.context import init_context
from minitp.generation import decode_step, make_kv_cache, prefill, select_token
from minitp.weight_loader import load_qwen2_tp


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="mini-TP phase-separated benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--new-tokens", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile-communication", action="store_true")
    p.add_argument(
        "--early-stop", action="store_true",
        help="stop at EOS (adds a GPU->CPU sync per token); benchmark default is fixed-length",
    )
    p.add_argument("--output-json", default=None, help="write the JSON result here (rank 0)")
    return p.parse_args(argv)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _percentiles(ms: list[float]) -> dict:
    xs = sorted(ms)

    def pct(q: float) -> float:
        idx = max(0, min(len(xs) - 1, round(q * len(xs)) - 1 if q > 0 else 0))
        return round(xs[idx], 3)

    return {
        "mean": round(sum(xs) / len(xs), 3),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "min": round(xs[0], 3),
        "max": round(xs[-1], 3),
    }


def bench_one_iter(model, input_ids, max_new_tokens, distributed_argmax=True):
    """One fixed-length generation; returns (prefill_s, per_token_ms list).

    Decode windows include token selection (argmax / distributed argmax),
    matching what an autoregressive loop actually pays per token. Fixed-length
    mode: no EOS check, hence zero GPU->CPU synchronization inside the loop.
    """
    device = input_ids.device
    use_cuda = device.type == "cuda"
    kv = make_kv_cache(model, input_ids.shape[0], input_ids.shape[1] + max_new_tokens)
    if use_cuda:
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
    else:
        t0 = time.perf_counter()
    logits, kv = prefill(model, input_ids, kv)
    if use_cuda:
        ev1.record()
        torch.cuda.synchronize()
        prefill_s = ev0.elapsed_time(ev1) / 1e3
    else:
        prefill_s = time.perf_counter() - t0

    token_ms: list[float] = []
    next_tok = select_token(model, logits, distributed_argmax)
    events = []
    for _ in range(max_new_tokens - 1):
        if use_cuda:
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
        else:
            t0 = time.perf_counter()
        logits = decode_step(model, next_tok.unsqueeze(-1), kv)
        next_tok = select_token(model, logits, distributed_argmax)
        if use_cuda:
            e.record()
            events.append((s, e))
        else:
            token_ms.append((time.perf_counter() - t0) * 1e3)
    if use_cuda:
        torch.cuda.synchronize()
        token_ms = [s.elapsed_time(e) for s, e in events]
    return prefill_s, token_ms


def main(argv=None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    ctx = init_context()
    device = ctx.device
    use_cuda = device.type == "cuda"
    set_profiling(args.profile_communication)

    dtype = DTYPES[args.dtype]
    hf_cfg = AutoConfig.from_pretrained(args.model)
    cfg = ModelConfig.from_hf(hf_cfg, tp_size=ctx.tp_size)
    cfg.validate(ctx.tp_size)
    AutoTokenizer.from_pretrained(args.model)  # outside all timed windows

    # ---- phase 1: checkpoint load (timed, excluded from compute windows) ----
    from huggingface_hub import snapshot_download

    ctx.barrier()
    t0 = time.perf_counter()
    model = load_qwen2_tp(
        snapshot_download(args.model), cfg, ctx, dtype=dtype, device=device
    ).eval()
    load_s = time.perf_counter() - t0
    peak_after_load = None
    if use_cuda:
        torch.cuda.synchronize()
        peak_after_load = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()  # decode-phase peaks only

    ids = torch.randint(
        1000, min(cfg.vocab_size - 1, 30000), (args.batch_size, args.prompt_len), device=device
    )

    # warmup (not reported)
    for _ in range(args.warmup):
        bench_one_iter(model, ids, args.new_tokens)
        ctx.barrier()

    # ---- phase 2/3: measured iterations ----
    reset_comm_stats()
    prefill_s_list, token_ms_lists = [], []
    for _ in range(args.iters):
        prefill_s, token_ms = bench_one_iter(model, ids, args.new_tokens)
        prefill_s_list.append(prefill_s)
        token_ms_lists.append(token_ms)
        ctx.barrier()

    flat_ms = [ms for lst in token_ms_lists for ms in lst]
    prefill_mean = sum(prefill_s_list) / len(prefill_s_list)
    decode_mean_ms = sum(flat_ms) / len(flat_ms)
    e2e_s = prefill_mean + sum(flat_ms) / 1e3 / args.iters

    result = {
        "schema": "minitp.bench/2",
        "model": args.model,
        "tp_size": ctx.tp_size,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "prompt_len": args.prompt_len,
        "new_tokens": args.new_tokens,
        "timed_iters": args.iters,
        "load": {"seconds": round(load_s, 3)},
        "prefill": {
            "seconds_mean": round(prefill_mean, 4),
            "prompt_tokens_per_s": round(args.batch_size * args.prompt_len / prefill_mean, 1),
        },
        "decode": {
            "ms_per_token": _percentiles(flat_ms),
            "tokens_per_s": round(1e3 / decode_mean_ms * args.batch_size, 1),
            "mode": "early_stop" if args.early_stop else "fixed_length_no_sync",
        },
        "e2e": {"seconds_mean": round(e2e_s, 4)},
        "memory": {
            "peak_allocated_gib_after_load": (
                round(peak_after_load / 2**30, 3) if use_cuda else None
            ),
            "peak_allocated_gib": (
                round(torch.cuda.max_memory_allocated() / 2**30, 3) if use_cuda else None
            ),
            "peak_reserved_gib": (
                round(torch.cuda.max_memory_reserved() / 2**30, 3) if use_cuda else None
            ),
        },
        "backend": "cuda_events" if use_cuda else "perf_counter",
    }

    if args.profile_communication:
        stats = get_comm_stats()
        summary = comm_summary(stats)
        by_op: dict[str, dict] = {}
        for s in stats:
            agg = by_op.setdefault(
                s["op"], {"calls": 0, "bytes": 0, "host_launch_ms": 0.0, "gpu_elapsed_ms": 0.0}
            )
            agg["calls"] += 1
            agg["bytes"] += s["bytes"]
            agg["host_launch_ms"] += s["host_launch_ms"]
            if s["gpu_elapsed_ms"] is not None:
                agg["gpu_elapsed_ms"] += s["gpu_elapsed_ms"]
        for agg in by_op.values():
            for k in ("host_launch_ms", "gpu_elapsed_ms"):
                agg[k] = round(agg[k], 2)
        result["communication"] = {
            "calls": summary.calls,
            "total_bytes": summary.total_bytes,
            "host_launch_ms": round(summary.host_launch_ms, 2),
            "gpu_elapsed_ms": (
                round(summary.gpu_elapsed_ms, 2) if summary.gpu_elapsed_ms is not None else None
            ),
            "by_op": by_op,
        }

    if ctx.is_rank_zero:
        line = json.dumps(result)
        print(line)
        if args.output_json:
            with open(args.output_json, "w") as f:
                f.write(line + "\n")
    ctx.barrier()
    ctx.destroy()


if __name__ == "__main__":
    main()
