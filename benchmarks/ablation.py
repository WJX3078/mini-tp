"""Apples-to-apples v0.1-equivalent -> v0.3 ablation benchmark.

All configurations run under the IDENTICAL harness (same weights, same input,
same CUDA-event timing, same warmup/iters, same fixed-length sync-free decode
as minitp.bench.benchmark). Only the feature toggles change:

  v01_equivalent   rotary cache OFF, unfused QKV (3 GEMMs), unfused gate/up
                   (2 GEMMs), reference RMSNorm, TP=1 slow embedding/argmax
  +rope_cache      precomputed cos/sin tables
  +fused_qkv       3 GEMMs -> 1
  +fused_gateup    2 GEMMs -> 1
  +tp1_fastpaths   unmasked embedding + direct argmax at TP=1
  +inference_mode  torch.inference_mode instead of torch.no_grad
  +rmsnorm_fn      F.rms_norm fused kernel (kept only if faster + equivalent)
  v03_full         all of the above

Run on the benchmark GPU:  python benchmarks/ablation.py --iters 3
Output: JSON per config on stdout + benchmarks/results/ablation_<ts>.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig

from minitp.bench.benchmark import _percentiles, bench_one_iter
from minitp.config import ModelConfig
from minitp.distributed.context import init_context
from minitp.weight_loader import load_qwen2_tp

RESULTS_DIR = Path(__file__).parent / "results"

CONFIGS = [
    ("v01_equivalent", dict(rotary=False, fused_qkv=False, fused_gateup=False,
                            tp1_fast=False, inference_mode=False, rmsnorm_fn=False)),
    ("v01+rope_cache", dict(rotary=True, fused_qkv=False, fused_gateup=False,
                            tp1_fast=False, inference_mode=False, rmsnorm_fn=False)),
    ("+fused_qkv", dict(rotary=True, fused_qkv=True, fused_gateup=False,
                        tp1_fast=False, inference_mode=False, rmsnorm_fn=False)),
    ("+fused_gateup", dict(rotary=True, fused_qkv=True, fused_gateup=True,
                           tp1_fast=False, inference_mode=False, rmsnorm_fn=False)),
    ("+tp1_fastpaths", dict(rotary=True, fused_qkv=True, fused_gateup=True,
                            tp1_fast=True, inference_mode=False, rmsnorm_fn=False)),
    ("+inference_mode", dict(rotary=True, fused_qkv=True, fused_gateup=True,
                             tp1_fast=True, inference_mode=True, rmsnorm_fn=False)),
    ("+rmsnorm_fn", dict(rotary=True, fused_qkv=True, fused_gateup=True,
                         tp1_fast=True, inference_mode=True, rmsnorm_fn=True)),
    ("v03_full", dict(rotary=True, fused_qkv=True, fused_gateup=True,
                      tp1_fast=True, inference_mode=True, rmsnorm_fn=True)),
]


def apply_config(model, flags: dict) -> None:
    model.use_rotary_cache = flags["rotary"]
    for layer in model.model.layers:
        layer.self_attn.use_fused_qkv = flags["fused_qkv"]
        layer.mlp.use_fused_gateup = flags["fused_gateup"]
        layer.input_layernorm.implementation = "functional" if flags["rmsnorm_fn"] else "reference"
        layer.post_attention_layernorm.implementation = (
            "functional" if flags["rmsnorm_fn"] else "reference"
        )
    model.model.embed_tokens.tp1_fast = flags["tp1_fast"]


def count_kernels(model, ids, steps: int = 5) -> int:
    """Approximate CUDA kernel launches for `steps` decode steps."""
    from torch.profiler import ProfilerActivity, profile

    from minitp.generation import make_kv_cache

    kv = make_kv_cache(model, ids.shape[0], ids.shape[1] + steps + 1)
    with torch.no_grad():
        logits, kv = prefill_wrapped(model, ids, kv)
        nxt = model.lm_head.distributed_argmax(logits[:, -1]).squeeze(-1)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(steps):
                logits = model(nxt.unsqueeze(-1), kv_cache=kv)
                nxt = model.lm_head.distributed_argmax(logits[:, -1]).squeeze(-1)
            torch.cuda.synchronize()
    return sum(e.count for e in prof.key_averages())


def prefill_wrapped(model, ids, kv):
    from minitp.generation import prefill

    return prefill(model, ids, kv)


def run_config(model, ids, new_tokens, iters, flags) -> dict:
    apply_config(model, flags)
    ctx_mgr = torch.inference_mode if flags["inference_mode"] else torch.no_grad

    # correctness guard inside the ablation itself: full-model logits must not
    # drift between configurations (bf16 tolerance)
    with ctx_mgr():
        bench_one_iter(model, ids, new_tokens)  # warmup (absorbs toggle effects)
    prefill_list, ttft_list, step_lists = [], [], []
    with ctx_mgr():
        for _ in range(iters):
            prefill_s, first_sel_ms, step_ms = bench_one_iter(model, ids, new_tokens)
            prefill_list.append(prefill_s)
            ttft_list.append(prefill_s + first_sel_ms / 1e3)
            step_lists.append(step_ms)
    flat = [x for lst in step_lists for x in lst]
    prefill_mean = sum(prefill_list) / len(prefill_list)
    ttft_mean = sum(ttft_list) / len(ttft_list)
    tpot = sum(flat) / len(flat) if flat else None
    return {
        "prefill_s": round(prefill_mean, 4),
        "ttft_s": round(ttft_mean, 4),
        "tpot_ms": _percentiles(flat),
        "decode_tokens_per_s": round(1e3 / tpot, 1) if tpot else None,
        "e2e_s": round(ttft_mean + (tpot * (new_tokens - 1) / 1e3 if tpot else 0.0), 4),
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="mini-TP v0.1->v0.3 ablation")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32", "fp16"])
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--new-tokens", type=int, default=64)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--warmup-iters", type=int, default=1)
    p.add_argument("--kernel-profile-steps", type=int, default=5,
                   help="decode steps profiled per config for kernels/token (0 = off)")
    p.add_argument("--output-json", default=None)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    ctx = init_context()
    hf_cfg = AutoConfig.from_pretrained(args.model)
    cfg = ModelConfig.from_hf(hf_cfg, tp_size=ctx.tp_size)
    from huggingface_hub import snapshot_download

    t0 = time.perf_counter()
    model = load_qwen2_tp(
        snapshot_download(args.model), cfg, ctx, dtype=dtype, device=device
    ).eval()
    load_s = time.perf_counter() - t0

    torch.manual_seed(0)
    ids = torch.randint(1000, 30000, (1, args.prompt_len), device=device)
    with torch.no_grad():
        reference_logits = model(ids, gather_logits=True).clone()

    # sanity: every configuration must produce (near-)identical logits
    rows = []
    for name, flags in CONFIGS:
        apply_config(model, flags)
        with torch.no_grad():
            got = model(ids, gather_logits=True)
        max_diff = (got.float() - reference_logits.float()).abs().max().item()
        row = {
            "config": name,
            "logits_max_abs_diff_vs_v01": max_diff,
            **run_config(model, ids, args.new_tokens, args.iters, flags),
        }
        if args.kernel_profile_steps and device.type == "cuda":
            row["kernels_per_token"] = round(
                count_kernels(model, ids, args.kernel_profile_steps) / args.kernel_profile_steps, 1
            )
        rows.append(row)
        print(
            f"{name:<18} ttft={row['ttft_s']:.4f}s tpot={row['tpot_ms']['mean'] if row['tpot_ms'] else '-'}ms"
            f" tok/s={row['decode_tokens_per_s']} kernels/tok={row.get('kernels_per_token', '-')}"
            f" diff={max_diff:.4f}",
            flush=True,
        )

    base = rows[0]
    for row in rows[1:]:
        row["tpot_speedup_vs_v01"] = (
            round(base["tpot_ms"]["mean"] / row["tpot_ms"]["mean"], 3) if row["tpot_ms"] else None
        )

    result = {
        "schema": "minitp.ablation/1",
        "model": args.model,
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "checkpoint_load_s": round(load_s, 3),
        "prompt_len": args.prompt_len,
        "new_tokens": args.new_tokens,
        "iters": args.iters,
        "note": "identical harness across configs; only feature toggles differ",
        "configs": rows,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    line = json.dumps(result)
    RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.output_json) if args.output_json else (
        RESULTS_DIR / f"ablation_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out.write_text(line + "\n")
    print(f"saved: {out}", file=sys.stderr)
    ctx.destroy()


if __name__ == "__main__":
    main()
