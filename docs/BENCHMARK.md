# Benchmark Methodology

What `minitp.bench.benchmark` measures, how, and what it deliberately does not
mix together. Driver: v0.1 divided *whole-generation* time by `new_tokens` and
called it "decode ms/token" — that silently includes prefill and load. v0.2
reports phases separately.

## Phases

| phase | window | timing | reported as |
|---|---|---|---|
| checkpoint load | safetensors read + shard packing + H2D copy | `perf_counter` around the whole load, `synchronize` after | `load.seconds` |
| prefill | prompt forward only (positions 0..T-1, KV write) | CUDA event pair, `synchronize` after | `prefill.seconds_mean`, `prefill.prompt_tokens_per_s` |
| decode | each autoregressive step: model forward + greedy token selection | one CUDA event pair **per token**, one `synchronize` at loop end | `decode.ms_per_token` distribution, `decode.tokens_per_s` |
| e2e | prefill + mean decode | derived | `e2e.seconds_mean` |

Model download, tokenizer construction, and CUDA init happen outside every
timed window. Warmup iterations run before measurement and are discarded
(they absorb kernel autotuning, allocator growth, and lazy RoPE-cache build).

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

`--profile-communication` wraps every collective with paired CUDA events
(host launch vs device span — see docs/COMMUNICATION_PROFILING.md for why
wall-clock around an NCCL enqueue is not GPU time). The JSON reports calls /
bytes / `host_launch_ms` / `gpu_elapsed_ms` per op. TP=1 runs do zero
collectives, so the section is empty by construction.

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
