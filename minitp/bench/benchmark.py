"""mini-TP benchmark: phase-separated, multi-rank-aggregated, reproducible.

Methodology (docs/BENCHMARK.md):
- snapshot resolve (network/cache) measured separately from checkpoint load
- prefill timed alone; TTFT = prefill + first token selection (distributed
  argmax included); TPOT = mean of the remaining decode steps (also includes
  selection); E2E = TTFT + TPOT*(new_tokens-1)
- decode is ALWAYS fixed-length sync-free (no EOS check, zero GPU->CPU syncs);
  interactive early-stop lives in generate.py only
- multi-rank: per-rank latencies aggregated with tensor collectives; the
  primary latency is max-across-ranks (slowest rank gates the step), skew
  reported
- JSON carries full environment metadata (git SHA, torch/CUDA/GPU, flags)
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import time

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from minitp import __version__
from minitp.config import ModelConfig
from minitp.distributed import comm_summary, get_comm_stats, reset_comm_stats, set_profiling
from minitp.distributed.context import ParallelContext, init_context
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
    p.add_argument("--output-json", default=None, help="write the JSON result here (rank 0)")
    return p.parse_args(argv)


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _percentiles(ms: list[float]) -> dict | None:
    if not ms:
        return None
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


def _nccl_version_safe():
    try:
        return tuple(torch.cuda.nccl.version())
    except Exception:
        return None


def _git_metadata() -> dict:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5
        ).stdout.strip())
    except Exception:
        sha, dirty = None, None
    return {"git_commit": sha, "git_dirty": dirty}


def bench_one_iter(model, input_ids, max_new_tokens, distributed_argmax=True):
    """One fixed-length generation.

    Windows (CUDA events on GPU, perf_counter on CPU):
      prefill            — prompt forward only
      first_selection    — greedy pick producing token #1 (part of TTFT)
      per decode step    — forward + selection for tokens #2..N (TPOT samples)

    No EOS check and no .item()/bool() inside the loop: zero GPU->CPU syncs.
    Returns (prefill_s, first_selection_ms, step_ms list len max_new_tokens-1).
    """
    device = input_ids.device
    use_cuda = device.type == "cuda"
    kv = make_kv_cache(model, input_ids.shape[0], input_ids.shape[1] + max_new_tokens)

    ev0, ev1 = (torch.cuda.Event(True), torch.cuda.Event(True)) if use_cuda else (None, None)
    if use_cuda:
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

    if use_cuda:
        s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
        s0.record()
    else:
        t0 = time.perf_counter()
    next_tok = select_token(model, logits, distributed_argmax)
    if use_cuda:
        e0.record()
        torch.cuda.synchronize()
        first_selection_ms = s0.elapsed_time(e0)
    else:
        first_selection_ms = (time.perf_counter() - t0) * 1e3

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
            events.append(time.perf_counter() - t0)
    if use_cuda:
        torch.cuda.synchronize()
        step_ms = [s.elapsed_time(e) for s, e in events]
    else:
        step_ms = [x * 1e3 for x in events]
    return prefill_s, first_selection_ms, step_ms


def _aggregate_across_ranks(ctx: ParallelContext, values: dict[str, float]) -> dict:
    """Tensor-collective aggregation across TP ranks (max gates the step).

    Runs once after benchmarking — never in the hot path. TP=1 returns the
    rank-local values unchanged.
    """
    if ctx.tp_size == 1 or not dist_is_initialized(ctx):
        return {}
    keys = list(values)
    local = torch.tensor([values[k] for k in keys], dtype=torch.float64)
    mx, mn, sm = local.clone(), local.clone(), local.clone()
    dist.all_reduce(mx, op=dist.ReduceOp.MAX, group=ctx.process_group)
    dist.all_reduce(mn, op=dist.ReduceOp.MIN, group=ctx.process_group)
    dist.all_reduce(sm, op=dist.ReduceOp.SUM, group=ctx.process_group)
    mean = sm / ctx.tp_size
    out = {}
    for i, k in enumerate(keys):
        out[k] = {
            "max": round(mx[i].item(), 4),
            "min": round(mn[i].item(), 4),
            "mean": round(mean[i].item(), 4),
            "skew": round((mx[i] - mn[i]).item(), 4),
        }
    return out


def dist_is_initialized(ctx: ParallelContext) -> bool:
    import torch.distributed as dist

    return ctx.process_group is not None and dist.is_initialized()


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

    # ---- phase 0: snapshot resolve (network/cache) — NOT checkpoint load ----
    from huggingface_hub import snapshot_download

    ctx.barrier()
    t0 = time.perf_counter()
    model_dir = snapshot_download(args.model)
    snapshot_resolve_s = time.perf_counter() - t0

    # ---- phase 1: checkpoint load (materialize + shard pack + H2D) ----
    ctx.barrier()
    t0 = time.perf_counter()
    model = load_qwen2_tp(model_dir, cfg, ctx, dtype=dtype, device=device).eval()
    checkpoint_load_s = time.perf_counter() - t0
    peak_after_load = None
    if use_cuda:
        torch.cuda.synchronize()
        peak_after_load = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()  # compute-phase peaks only

    ids = torch.randint(
        1000, min(cfg.vocab_size - 1, 30000), (args.batch_size, args.prompt_len), device=device
    )

    for _ in range(args.warmup):
        bench_one_iter(model, ids, args.new_tokens)
        ctx.barrier()

    # ---- measured iterations (fixed-length, sync-free) ----
    reset_comm_stats()
    prefill_list, ttft_list, step_lists = [], [], []
    for _ in range(args.iters):
        prefill_s, first_sel_ms, step_ms = bench_one_iter(model, ids, args.new_tokens)
        prefill_list.append(prefill_s)
        ttft_list.append(prefill_s + first_sel_ms / 1e3)
        step_lists.append(step_ms)
        ctx.barrier()

    flat_ms = [ms for lst in step_lists for ms in lst]
    prefill_mean = sum(prefill_list) / len(prefill_list)
    ttft_mean = sum(ttft_list) / len(ttft_list)
    tpot_mean = sum(flat_ms) / len(flat_ms) if flat_ms else None  # None if new_tokens == 1
    e2e_s = ttft_mean + (tpot_mean * (args.new_tokens - 1) / 1e3 if tpot_mean else 0.0)
    peak_mem_gib = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0

    # ---- multi-rank aggregation (slowest rank gates the step) ----
    rank_local = {
        "prefill_s": prefill_mean,
        "ttft_s": ttft_mean,
        "tpot_ms": tpot_mean or 0.0,
        "peak_mem_gib": peak_mem_gib,
    }
    aggregated = _aggregate_across_ranks(ctx, rank_local)

    result = {
        "schema": "minitp.bench/3",
        "metadata": {
            "schema_version": 3,
            **_git_metadata(),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": _nccl_version_safe() if use_cuda else None,
            "gpu": torch.cuda.get_device_name(0) if use_cuda else None,
            "gpu_count": torch.cuda.device_count() if use_cuda else 0,
            "minitp_version": __version__,
            "model": args.model,
            "tp_size": ctx.tp_size,
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "prompt_len": args.prompt_len,
            "new_tokens": args.new_tokens,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed,
            "feature_flags": {
                "fused_qkv": True,
                "fused_gate_up": True,
                "rope_cache": True,
                "selective_loader": False,  # flipped in v0.3 loader section
                "native_gqa": False,
                "optimized_rmsnorm": False,
                "inference_mode": False,
                "compile": False,
                "cuda_graph": False,
            },
        },
        "load": {
            "snapshot_resolve_s": round(snapshot_resolve_s, 3),
            "checkpoint_load_s": round(checkpoint_load_s, 3),
            "note": "checkpoint_load excludes network/cache resolve",
        },
        "prefill": {
            "seconds_mean": round(prefill_mean, 4),
            "prompt_tokens_per_s": round(args.batch_size * args.prompt_len / prefill_mean, 1),
        },
        "ttft": {"seconds_mean": round(ttft_mean, 4),
                 "definition": "prefill + first token selection (distributed argmax included)"},
        "decode": {
            "tpot_ms": _percentiles(flat_ms),
            "tokens_per_s": round(1e3 / tpot_mean * args.batch_size, 1) if tpot_mean else None,
            "definition": "TPOT = forward + selection per output token after the first",
            "mode": "fixed_length_no_sync",
        },
        "e2e": {"seconds_mean": round(e2e_s, 4),
                "definition": "TTFT + TPOT*(new_tokens-1)"},
        "memory": {
            "peak_allocated_gib_after_load": round(peak_after_load / 2**30, 3) if use_cuda else None,
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if use_cuda else None,
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3) if use_cuda else None,
        },
        "backend": "cuda_events" if use_cuda else "perf_counter",
    }

    if aggregated:
        result["multi_rank"] = {
            "aggregation": "max/min/mean/skew across TP ranks (max gates the step)",
            **aggregated,
        }

    if args.profile_communication:
        stats = get_comm_stats()
        summary = comm_summary(stats)
        by_op: dict[str, dict] = {}
        for s in stats:
            agg = by_op.setdefault(
                s["op"],
                {
                    "calls": 0, "bytes": 0, "host_launch_ms": 0.0,
                    "collective_gpu_ms": 0.0, "postprocess_host_ms": 0.0,
                },
            )
            agg["calls"] += 1
            agg["bytes"] += s["bytes"]
            agg["host_launch_ms"] += s["host_launch_ms"]
            if s["collective_gpu_ms"] is not None:
                agg["collective_gpu_ms"] += s["collective_gpu_ms"]
            agg["postprocess_host_ms"] += s["postprocess_host_ms"]
        for agg in by_op.values():
            for k in ("host_launch_ms", "collective_gpu_ms", "postprocess_host_ms"):
                agg[k] = round(agg[k], 3)
        result["communication"] = {
            "calls": summary.calls,
            "total_bytes": summary.total_bytes,
            "host_launch_ms": round(summary.host_launch_ms, 3),
            "collective_gpu_ms": (
                round(summary.collective_gpu_ms, 3) if summary.collective_gpu_ms is not None else None
            ),
            "postprocess_host_ms": round(summary.postprocess_host_ms, 3),
            "units": "milliseconds; bytes = input tensor (all_reduce/reduce_scatter) or output tensor (all_gather)",
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
