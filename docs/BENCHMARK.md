# Benchmark Methodology

What `minitp.bench.benchmark` measures, how, and what it deliberately does not
mix together. Driver: v0.1 divided *whole-generation* time by `new_tokens` and
called it "decode ms/token" — that silently includes prefill and load. v0.2
reports phases separately.

## Phases (schema `minitp.bench/3`)

| phase | window | timing | reported as |
|---|---|---|---|
| snapshot resolve | network/cache resolution of the model dir | `perf_counter` | `load.snapshot_resolve_s` |
| checkpoint load | safetensors read + shard packing + H2D copy (download excluded) | `perf_counter`, barrier-delimited, `synchronize` after | `load.checkpoint_load_s` |
| prefill | prompt forward only (positions 0..T-1, KV write) | CUDA event pair, `synchronize` after | `prefill.seconds_mean`, `prefill.prompt_tokens_per_s` |
| first selection | greedy pick of token #1 (distributed argmax included) | CUDA event pair | folded into `ttft` |
| decode steps | forward + selection for tokens #2..N | one CUDA event pair **per token**, one `synchronize` at loop end | `decode.tpot_ms` percentiles, `decode.tokens_per_s` |

**Standard serving metrics:**

- `TTFT` = prefill + first token selection (time to first token).
- `TPOT` = mean latency per *subsequent* output token (forward + selection;
  the distributed argmax collective is part of inference latency and is
  included).
- `E2E` = `TTFT + TPOT × (new_tokens − 1)` — starts when prompt execution
  starts and ends when the last requested token is produced.
- `new_tokens == 1` is legal: the TPOT section is `null`, E2E == TTFT.

Model download, tokenizer construction, and CUDA init happen outside every
timed window. Warmup iterations run before measurement and are discarded
(they absorb kernel autotuning, allocator growth, and lazy RoPE-cache build).

## Multi-rank aggregation

For TP>1 the step latency a user observes is gated by the **slowest rank**.
After measurement (never in the hot path) each rank reduces its
`prefill_s / ttft_s / tpot_ms / peak_mem_gib` with MAX/MIN/SUM tensor
collectives; the JSON reports max/min/mean/**skew** per metric. Primary
latency = max across ranks; rank-0 raw values remain in the phase fields.

## Decode distribution

Decode uses **fixed-length mode** (`early_stop=False`): exactly
`new_tokens - 1` steps, no EOS check — because `bool(finished.all())` forces a
GPU→CPU synchronization every token, which both adds host latency (expensive
under Windows WDDM in particular) and breaks kernel-queue pipelining. The
fixed-length path contains zero `.item()`/`bool()` calls; tokens that would
have triggered EOS are still produced and simply ignored. Interactive
`generate.py` uses `early_stop=True` (per-token sync, stops at EOS).

Percentiles (`p50/p90/p99`) use nearest-rank over all timed steps of all
iterations. Each decode window *includes* token selection (argmax or
distributed argmax over the sharded vocabulary) — that is what an
autoregressive loop pays per token.

## Memory

`torch.cuda.max_memory_allocated/reserved` per rank. Peak after load is read
and then the stats are reset, so `peak_allocated_gib` reflects the
prefill+decode phase only. Load-phase peak is reported separately. Multi-rank
values are per-rank; rank 0 aggregates nothing in v0.2 (per-rank JSON per
torchrun process, one file per invocation).

## Communication

`--profile-communication` wraps every collective: paired CUDA events bracket
**only the distributed op** (cat/copy post-processing is reported separately
as `postprocess_host_ms` — see docs/COMMUNICATION_PROFILING.md). The JSON
reports per-op calls / bytes / `host_launch_ms` / `collective_gpu_ms`, all in
milliseconds (units converted exactly once — v0.2's summary double-scaled them,
see docs/V03_AUDIT.md B2). TP=1 runs do zero collectives, so the section is
empty by construction. `bytes` convention: input tensor for
all_reduce/reduce_scatter, output tensor for all_gather.

## Reproducibility metadata

Every JSON carries `schema`, git commit SHA + dirty flag, UTC timestamp, OS,
Python/PyTorch/CUDA/NCCL versions, GPU name and count, mini-TP version,
dtype, model, TP size, batch, prompt/new-token counts, warmup/iters, seed,
and a `feature_flags` block (fused QKV/gate-up, RoPE cache, selective loader,
native GQA, RMSNorm variant, inference_mode, compile, CUDA graph).

## Reproducing

```bash
# TP=1 (this machine's measured results)
python -m minitp.bench.benchmark --prompt-len 512 --new-tokens 128 --iters 3 \
    --profile-communication --output-json results/v02_tp1_len512_out128.json

# TP=2 — UNVERIFIED until run on >=2 CUDA GPUs
torchrun --standalone --nproc-per-node=2 -m minitp.bench.benchmark \
    --prompt-len 512 --new-tokens 128 --output-json results/tp2_len512_out128.json

# kernel-level old-vs-new comparisons
python -m minitp.bench.microbench
```

JSON schema `minitp.bench/2` (one line, rank 0): `load`, `prefill`, `decode`
(with `ms_per_token` percentiles), `e2e`, `memory`, optional
`communication`, plus model/tp/dtype/batch/seed metadata.

## Known caveats

- Single consumer-GPU laptop (Windows, WDDM): per-kernel launch overhead is
  much higher than on Linux/TCC or datacenter GPUs; absolute numbers are
  machine-specific, relative old-vs-new comparisons are the signal.
- Decode is kernel-launch-bound for a 0.5B model (see
  docs/PERFORMANCE_AUDIT.md): ~6.5k kernels/token at v0.1, which is why
  launch-reducing fusions dominate the improvement.
- All TP=2 / TP=4 GPU numbers remain **UNVERIFIED — requires real multi-GPU
  hardware** until executed there.
