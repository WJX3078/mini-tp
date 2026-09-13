"""mini-TP benchmark: phase-separated, multi-rank-aggregated, reproducible.

Methodology v4 (docs/BENCHMARK.md):
- snapshot resolve measured separately from checkpoint load
- continuous TTFT: ONE outer CUDA event pair around (prefill -> first token
  selection), no intermediate synchronize; a nested inner pair breaks out
  prefill. Invariant: ttft >= prefill + selection - event noise.
- TPOT: per-step samples (forward + selection). Multi-rank semantics: the
  user-visible step latency is the per-step MAX across ranks, so the
  benchmark keeps per-step samples per rank and reduces them ELEMENT-WISE
  with a MAX collective after measurement (never in the hot path) —
  mean/p50/p90/p99 are computed on the global slowest-rank samples
  (max(mean(rank)) is a different, smaller-biased number, V04_AUDIT B5).
- decode ALWAYS fixed-length sync-free via GenerationState — the same hot
  path as generate_greedy (no per-token torch.arange, V04_AUDIT B3).
- feature flags derived from the live runtime objects, not hardcoded (B6).
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
from minitp.distributed import get_comm_stats, reset_comm_stats, set_profiling
from minitp.distributed.context import ParallelContext, init_context
from minitp.generation import GenerationState
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


def _percentiles(samples: list[float]) -> dict | None:
    if not samples:
        return None
    xs = sorted(samples)

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


def bench_one_iter(model, input_ids, max_new_tokens, distributed_argmax=True, kv_init="empty"):
    """One fixed-length generation through GenerationState (the shared hot path).

    Windows (CUDA events on GPU, perf_counter on CPU), NO intermediate
    synchronize inside the TTFT window:
      ttft        — outer pair: prefill start -> first token selected
      prefill     — inner pair: prompt forward only
      per step    — forward + selection for tokens #2..N (TPOT samples)

    Returns (prefill_s, ttft_s, first_selection_ms, step_ms list).
    """
    device = input_ids.device
    use_cuda = device.type == "cuda"
    state = GenerationState(
        model, input_ids, max_new_tokens, distributed_argmax=distributed_argmax, kv_init=kv_init
    )
    if use_cuda:
        ttft0 = torch.cuda.Event(True)
        prefill0, prefill1 = torch.cuda.Event(True), torch.cuda.Event(True)
        sel1 = torch.cuda.Event(True)
        ttft0.record()
        prefill0.record()
    else:
        t0 = time.perf_counter()
    logits = state.prefill()
    if use_cuda:
        prefill1.record()
    else:
        prefill_s = time.perf_counter() - t0
        t0 = time.perf_counter()
    tok = state.select_next(logits)
    if use_cuda:
        sel1.record()
        torch.cuda.synchronize()  # single resolve point for the whole TTFT window
        ttft_s = ttft0.elapsed_time(sel1) / 1e3
        prefill_s = prefill0.elapsed_time(prefill1) / 1e3
        first_selection_ms = prefill1.elapsed_time(sel1)
    else:
        first_selection_ms = (time.perf_counter() - t0) * 1e3
        ttft_s = prefill_s + first_selection_ms / 1e3

    events = []
    for _ in range(max_new_tokens - 1):
        if use_cuda:
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
        else:
            t0 = time.perf_counter()
        pos = state.append(tok)
        logits = state.decode_step(tok, pos)
        tok = state.select_next(logits)
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
    return prefill_s, ttft_s, first_selection_ms, step_ms


def _aggregate_max(ctx: ParallelContext, samples: list[float]) -> list[float]:
    """Element-wise MAX across ranks over per-step/per-iter samples.

    Aggregation runs ONCE after measurement, on tensors placed on the process
    group's device — NCCL requires CUDA tensors, Gloo accepts both
    (V04_AUDIT B1). Returns the global slowest-rank samples.
    """
    if ctx.tp_size == 1 or ctx.process_group is None or not dist.is_initialized():
        return samples
    local = torch.tensor(samples, dtype=torch.float64, device=ctx.device)
    dist.all_reduce(local, op=dist.ReduceOp.MAX, group=ctx.process_group)
    return local.tolist()


def _feature_flags(model, ctx: ParallelContext, loader: str, argmax_encoding: str) -> dict:
    """Flags derived from the LIVE runtime configuration (V04_AUDIT B6)."""
    rmsnorm_impls = sorted({layer.input_layernorm.implementation for layer in model.model.layers})
    return {
        "fused_qkv": bool(model.model.layers[0].self_attn.use_fused_qkv),
        "fused_gate_up": bool(model.model.layers[0].mlp.use_fused_gateup),
        "rope_cache": bool(model.use_rotary_cache),
        "rmsnorm_backend": rmsnorm_impls[0] if len(rmsnorm_impls) == 1 else rmsnorm_impls,
        "argmax_encoding": argmax_encoding,
        "loader": loader,
        "inference_mode": True,  # generation entry points are @torch.inference_mode
        "kv_init": "empty",
        "comm_backend": (
            dist.get_backend(ctx.process_group)
            if ctx.process_group is not None and dist.is_initialized() else "none"
        ),
        "nccl_available": dist.is_nccl_available(),
        "cuda_graph": False,
        "compile": False,
    }


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
    loader = "selective"
    model = load_qwen2_tp(model_dir, cfg, ctx, dtype=dtype, device=device, loader=loader).eval()
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
    prefill_iters, ttft_iters, step_lists = [], [], []
    for _ in range(args.iters):
        prefill_s, ttft_s, _sel_ms, step_ms = bench_one_iter(model, ids, args.new_tokens)
        prefill_iters.append(prefill_s)
        ttft_iters.append(ttft_s)
        step_lists.append(step_ms)
        ctx.barrier()

    rank_tpot_samples = [ms for lst in step_lists for ms in lst]
    prefill_mean = sum(prefill_iters) / len(prefill_iters)
    ttft_mean = sum(ttft_iters) / len(ttft_iters)
    tpot_mean = sum(rank_tpot_samples) / len(rank_tpot_samples) if rank_tpot_samples else None
    e2e_s = ttft_mean + (tpot_mean * (args.new_tokens - 1) / 1e3 if tpot_mean else 0.0)
    peak_mem_gib = torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0

    # ---- multi-rank aggregation: per-sample element-wise MAX (B5) ----
    global_prefill = _aggregate_max(ctx, prefill_iters)
    global_ttft = _aggregate_max(ctx, ttft_iters)
    global_tpot = _aggregate_max(ctx, rank_tpot_samples)

    result = {
        "schema": "minitp.bench/4",
        "metadata": {
            "schema_version": 4,
            **_git_metadata(),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "nccl": _nccl_version_safe(),
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
            "feature_flags": _feature_flags(model, ctx, loader, "bitpack"),
        },
        "load": {
            "snapshot_resolve_s": round(snapshot_resolve_s, 3),
            "checkpoint_load_s": round(checkpoint_load_s, 3),
            "note": "checkpoint_load excludes network/cache resolve",
        },
        "rank_local": {
            "prefill": {
                "seconds_mean": round(prefill_mean, 4),
                "prompt_tokens_per_s": round(args.batch_size * args.prompt_len / prefill_mean, 1),
            },
            "ttft": {
                "seconds_mean": round(ttft_mean, 4),
                "definition": "CONTINUOUS: prefill start -> first token selected (one event pair)",
            },
            "tpot_ms": _percentiles(rank_tpot_samples),
            "tokens_per_s": round(1e3 / tpot_mean * args.batch_size, 1) if tpot_mean else None,
            "peak_mem_gib": round(peak_mem_gib, 3) if use_cuda else None,
        },
        "global_slowest_rank": {
            "note": "element-wise per-sample MAX across ranks; the slowest rank gates the step",
            "prefill_seconds_mean": (
                round(sum(global_prefill) / len(global_prefill), 4)
                if global_prefill else round(prefill_mean, 4)
            ),
            "ttft_seconds_mean": (
                round(sum(global_ttft) / len(global_ttft), 4)
                if global_ttft else round(ttft_mean, 4)
            ),
            "tpot_ms": _percentiles(global_tpot) if global_tpot else None,
        },
        "e2e": {"seconds_mean": round(e2e_s, 4),
                "definition": "TTFT + TPOT*(new_tokens-1), rank-local"},
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

    if ctx.tp_size > 1:
        gp, gt = global_prefill, global_ttft
        result["rank_skew"] = {
            "prefill_s": round(max(gp) - min(gp), 4) if len(gp) > 1 else 0.0,
            "ttft_s": round(max(gt) - min(gt), 4) if len(gt) > 1 else 0.0,
        }

    if args.profile_communication:
        stats = get_comm_stats()
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
            "calls": len(stats),
            "total_bytes": sum(s["bytes"] for s in stats),
            "host_launch_ms": round(sum(s["host_launch_ms"] for s in stats), 3),
            "collective_gpu_ms": round(sum(
                s["collective_gpu_ms"] for s in stats if s["collective_gpu_ms"] is not None), 3),
            "postprocess_host_ms": round(sum(s["postprocess_host_ms"] for s in stats), 3),
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
