# Checkpoint Loader Pipeline (v0.4)

Loaders, measured. Same checkpoint (`Qwen/Qwen2.5-0.5B`, bf16, TP=1, same
process so later rows inherit earlier allocations — compare within a row, and
treat `host_rss_peak` as a sampled max over that loader's load window).

| loader | load_s | what it does |
|---|---|---|
| legacy | 0.88 | full safetensors state dict -> CPU TP model -> slice -> one H2D transfer |
| selective (default) | 0.61 | `safe_open` rank-local slices -> CPU TP model (no full state dict) -> one H2D transfer |
| direct_gpu (v0.4) | 0.84 | model constructed on GPU with uninitialized params; each rank-local slice copied straight into its GPU parameter |

GPU peak is identical (0.92 GiB = local shard footprint) for all three —
correct: TP memory benefit does not depend on the loader. Host behavior
differs: legacy materializes the full checkpoint on the host; selective
never does; direct_gpu additionally skips the CPU TP model.

Reproduce: `python -m minitp.bench.loader_benchmark`.

## Design notes

- **No random init, ever** (v0.4): all loaders construct with
  `init_weights=False` (uninitialized parameters). The loader owns every
  value; `tests/test_loader_coverage.py` poisons every parameter with NaN and
  asserts finiteness after load — a missed parameter fails loudly instead of
  silently shipping random weights.
- **direct_gpu** avoids the CPU TP model entirely (the win scales with model
  size and TP: the CPU footprint becomes local-shard-only). On the 0.5B
  model the per-slice H2D of many small tensors is *slower* than one bulk
  transfer (0.84 s vs 0.61 s) — recorded, not hidden. The breakeven is the
  point where the CPU TP model no longer fits comfortably in RAM.
- **Pinned double-buffer overlap**: measured H2D volume is 0.92 GiB and file
  reads are page-cached; the timeline shows file IO and H2D are each far
  shorter than the python/packing overhead at this scale, so a ping-pong
  staging scheme has no measurable room on this host — recorded as a
  negative result instead of shipping complexity (V04_REPORT Q9).

## Correctness contract

All loaders produce bit-identical local parameters
(`tests/test_loader_coverage.py`, `tests/test_selective_loader.py`), on CPU
and CUDA targets, TP=1/2, GQA sharding and KV replication, tied and untied.
