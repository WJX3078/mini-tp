# mini-TP v0.3 Adversarial Audit (over v0.2 @ cea27f2)

Stance: *v0.2 contains hidden correctness / benchmark / profiler bugs — find them.*
Each finding lists location, root cause, why the existing test suite missed it, and
disposition. Bugs **B1–B3 were reproduced empirically** on TP=2 CPU/Gloo
(`recorded probe output` below); the rest are verified by code inspection /
deterministic reasoning. All B-numbers map to regression tests added in v0.3.

```
probe (TP=2 Gloo, profiling ON):
  allreduce_off_returns_t      true
  allreduce_on_returns_t       false
  allreduce_on_returns_none    true        <- B1
  allreduce_on_value_ok        true        (data correct; contract broken)
  broadcast_on_returns_t       false       <- B1
  record_host_ms               0.6232
  summary_host_ms              1183.70     <- B2 (~1899x for 2 records)
  rowparallel_profiled_ok      false
  rowparallel_profiled_err     TypeError: NoneType + Parameter   <- B1 impact
```

## B1 (P0 BLOCKER) — profiled `all_reduce`/`broadcast` return `None`

- **Where**: `minitp/distributed/collectives.py`, `_profiled` returns
  `result = run()`, but `dist.all_reduce` / `dist.broadcast` return `None`.
  `all_reduce` non-profiled path returns `t`; profiled path returns `None`.
- **Impact**: `RowParallelLinear.forward` does `local = all_reduce(local, ...)` —
  with **profiling enabled + TP>1** every forward crashes
  (`None + bias`), i.e. the communication profiler is unusable on exactly the
  path it exists to measure (NCCL TP=2). Latent on v0.2 because all profiling
  tests ran TP=2 without a model forward, and model tests ran without profiling.
- **Fix**: contract — `all_reduce(t)→t`, `broadcast(t)→t`, `reduce_scatter→out`,
  `all_gather→out`, identity of the input/output tensor, on every path
  (profiling on/off, CPU/CUDA). Regression tests assert `is` identity under
  both profiling states, TP=2 Gloo.

## B2 (P0 BLOCKER) — `comm_summary` scales ms twice

- **Where**: records store `host_launch_ms` (already ms); `comm_summary` does
  `sum(...) * 1e3` → ~1000× inflation (probe: 0.623 ms → 1183.70).
- **Impact**: benchmark JSON `communication.host_launch_ms` and
  `communication %` were wrong by 3 orders of magnitude whenever
  `--profile-communication` was set. Nobody caught it: TP=1 has zero
  collectives, and the TP=2 GPU numbers that would surface it are UNVERIFIED.
- **Fix**: single conversion at the record boundary; unit test with a mock
  timer asserting `summary.host_launch_ms == sum(record.host_launch_ms)`.

## B3 (P0) — `all_gather` profiled device window includes `torch.cat` + copy

- **Where**: the CUDA event pair brackets `finish()` = `dist.all_gather` +
  `torch.cat` + `out.copy_`.
- **Impact**: "collective GPU time" for AllGather is contaminated by
  post-processing; communication % overstated.
- **Fix**: split the record into `collective_gpu_ms` (events around the
  distributed op only) and `postprocess_gpu_ms`/`wrapper_total_ms` (the rest);
  document both. See docs/COMMUNICATION_PROFILING.md.

## B4 (P0) — `--early-stop` is inert but reported

- **Where**: `benchmark.py` parses `--early-stop` and writes
  `"mode": "early_stop"` into JSON, but `bench_one_iter` is always fixed-length.
- **Impact**: benchmark metadata could claim a path it never ran.
- **Fix (chosen)**: remove the flag from the benchmark CLI entirely; the
  benchmark is *always* fixed-length sync-free (documented); interactive
  `generate.py` keeps EOS early-stop. CLI test asserts the schema reports
  `fixed_length_no_sync`.

## B5 (P0) — checkpoint load timer includes `snapshot_download`

- **Where**: `benchmark.py` times `load_qwen2_tp(snapshot_download(...), ...)`.
- **Impact**: first run after cache eviction measures network download as
  "checkpoint load"; numbers not reproducible.
- **Fix**: resolve the snapshot first, barrier, then time only
  materialization (safetensors read + shard packing + H2D). Report
  `snapshot_resolve_s` separately.

## B6 (P0) — decode stats illegal at `new_tokens == 1`

- **Where**: `flat_ms` empty → `sum/len` ZeroDivisionError; `_percentiles`
  indexes `xs[0]`.
- **Fix**: TTFT/TPOT refactor covers `new_tokens=1` (TPOT section empty →
  `null`, not a crash); CLI test included.

## B7 (P0) — no multi-rank aggregation; TTFT/TPOT missing

- **Where**: benchmark prints rank-0-local timings only. For TP>1 the user
  visible step latency is the **slowest rank**; rank skew is invisible.
- **Fix**: tensor-collective aggregation after benchmarking (MAX/MIN/SUM via
  `dist.all_reduce` on a small stats tensor — no Python object gather);
  report max/min/mean/skew for prefill and decode, per-rank memory, plus
  TTFT (prefill + first selection) and TPOT (subsequent steps), with
  distributed argmax cost included in both.

## B8 (P1) — dead stale module `minitp/bench/communication.py`

References removed record key `elapsed_ms` (KeyError if ever used); nothing
imports it. **Fix**: delete; `comm_summary` in `distributed.collectives` is
the single implementation.

## B9 (P1) — microbench `--iters` ignored

`microbench.py` parses `--iters` then calls benches with hardcoded 200/20.
**Fix**: thread `iters`/`warmup` through all bench functions; JSON+table
output; smoke test.

## B10 (P1) — tests share production shard helpers as their oracle

`tests/test_fused_projections.py` builds *both* the fused input and the
expected reference with `_shard`/`_shard_heads`/`pack_qkv`. A systematic
shard-mapping bug would be invisible. **Fix**: independent oracle tests with
hand-written `arange` tensors and hand-computed expected indices/slices
(kept alongside the convenience tests, which remain for TP wiring).

## B11 (credibility) — README "28 % faster decode" is apples-to-oranges

v0.1's number was `E2E/new_tokens` (includes prefill, early-stop syncs);
v0.2's is phase-measured fixed-length decode. Same hardware, but different
methodology — not a legal comparison. **Fix**: `benchmarks/ablation.py`
compares feature configurations under the *identical* v0.3 harness
(same warmup/iters/events/mode); README reports ablation-derived deltas only,
plus a machine-readable environment block (GPU name, torch/CUDA, git SHA).

## B12 (consistency) — version skew

`pyproject.toml` = 0.1.0, `README` says v0.2. **Fix**: 0.3.0 everywhere;
README architecture/layout updated (fused QKV/gate-up, RoPE cache, selective
loader, new docs).

## Missing capability checks (upgrades, not bugs)

- TP=1 `VocabParallelEmbedding` runs the full mask path (~6 wasted kernels) —
  fast path added.
- TP=1 distributed argmax computes values+stack — direct `argmax` fast path.
- TP>1 distributed argmax allocates list/stack per token — persistent
  contiguous `[tp, B, 2]` buffer + `all_gather_into_tensor` where supported
  (feature-detected; Gloo fallback verified), bf16-token-id bug impossible
  (ids stay int64; values cast fp32 before stacking — regression-tested with
  ids 12095/50000/150000 and exact ties).
- General `all_gather` wrapper: equal-size fast path with
  `all_gather_into_tensor` (dim-0-compatible), uneven fallback preserved.
- KV cache zero-fill: `torch.zeros` → `torch.empty` safe because attention
  only reads `[:end]` after writing; NaN-poison test added.
- `torch.inference_mode` vs `no_grad` — measured; adopted only if correct+fast.
- Native SDPA `enable_gqa` — measured vs manual expand/repeat; default picked
  by measurement, feature-detected for torch compatibility.

## Disposition

All B1–B12 fixed in v0.3 commits; see docs/REVIEW.md ("v0.3 bugs found") for
the bug→root-cause→why-tests-missed→regression-test mapping, and
docs/V03_REPORT.md for measured outcomes.
