# mini-TP v0.4 Adversarial Audit (over v0.3 @ cfae88a)

Stance: *v0.3 still hides distributed / benchmark / test bugs.* Evidence is
labeled: **[empirical]** = reproduced on this host, **[inspection]** = proven
by code reading against documented torch semantics. Environment facts that
shape this audit: this torch build (`2.6.0+cu124`, Windows) has **no NCCL
compiled** (`dist.is_nccl_available() == False`) and the torchrun elastic
agent's TCPStore is broken (`USE_LIBUV=0` works for direct
`init_process_group` but not for the torchrun agent) — so every NCCL-path fix
below ships with contract tests here and is marked UNVERIFIED for real NCCL
runs. A world-size-1 NCCL probe was attempted and failed at import: no NCCL.

## BLOCKER

### B1 — multi-rank aggregation builds a CPU tensor [inspection]
`benchmark._aggregate_across_ranks` creates
`torch.tensor([...], dtype=torch.float64)` with no device. Gloo accepts CPU
and CUDA, so the CPU Gloo test passes — but **NCCL collectives require CUDA
tensors**; a real TP=2 GPU run crashes in aggregation (after all measurement,
but still a crash and a missing metric). CPU-tensor-under-NCCL is documented
c10d behavior; not probed locally because NCCL is not compiled here.
**Fix**: aggregation tensor on `ctx.device` (correct for both backends).
Regression: contract test that captures the tensor device passed to
`all_reduce`, plus a real `@pytest.mark.multi_gpu` NCCL test (auto-skip).

### B2 — KV cache half-converted to `torch.empty` + false-confidence tests [inspection]
`KVCache.__init__`: K uses `torch.empty` but V uses `torch.zeros_like(k)` —
the zero-fill elimination was applied to exactly half the cache. Worse,
`tests/test_kv_cache.py` contains `assert ... or True` (an assertion that can
never fail) and the second poisoned cache (`kv2`) is **created and poisoned
but never passed to `generate_greedy`** — the poison test tests nothing about
generation. **Fix**: `empty_like` for V; rewrite the test file (K and V
poisoned separately, actually passed into prefill AND a multi-token
generation, overflow test, per-advance cursor bound check) and record the
zero-fill elimination benchmark at 2K/8K/32K contexts.

## P0

### B3 — benchmark decode path ≠ optimized generation hot path **[empirical]**
`bench_one_iter` calls `decode_step(model, token, kv)` without positions, so
`TPQwen2ForCausalLM.forward` builds `torch.arange(...)` **every token**.
Measured: one `bench_one_iter(ids, 8)` issues **10 `torch.arange` calls**
(1 prefill + 8 decode + 1) while `generate_greedy(fixed)` issues **2**
(positions built once). The benchmark therefore measures a slower path than
the runtime it claims to evaluate. **Fix**: a single `GenerationState`
(KV cache + positions buffer + output buffer + cursor + `prefill()` /
`decode_step()`) used by `generate_greedy`, the benchmark, and the ablation —
benchmarks time it, they do not re-implement it. Regression: monkeypatched
`torch.arange` counter must be equal between benchmark path and generation
path, and decode must issue zero aranges after prefill.

### B4 — TTFT measured as two isolated synchronized phases [inspection]
v0.3 records a prefill event pair, **synchronizes**, records a selection
event pair, **synchronizes**, then adds the two device times. That reports
the sum of two isolated device spans, not the continuous time from prompt
forward start to first token ready (host gaps and event/scheduling overhead
are excluded). **Fix**: one outer TTFT event pair around (prefill → first
selection) with **no intermediate synchronize**, a nested inner prefill pair
for the breakdown, single resolve at the end; report `ttft` (continuous),
`prefill_gpu_ms`, `first_selection_gpu_ms`, and validate
`ttft ≥ prefill + selection − noise` in a test. CPU path: continuous
`perf_counter` window.

### B5 — `max(rank mean)` ≠ `mean(step max)` **[empirical, synthetic]**
The aggregation reduces each rank to its mean step latency and then takes the
max across ranks. Under lockstep TP the user-visible step latency is the
per-step max across ranks; the two are not equal (demo: rank0=[10,10,10,100],
rank1=[12,12,12,12] → max(mean)=32.5 vs mean(step max)=34; skew grows when
one rank has tail spikes). **Fix**: keep per-step samples per rank; after
measurement, tensorize per-step TPOT samples (and per-iteration
prefill/TTFT), element-wise MAX collective → *global slowest-rank samples*;
compute mean/p50/p90/p99/min/max on those; keep rank-local stats and skew.
No collective in the hot path — aggregation runs once after measurement.

### B6 — benchmark feature flags are hardcoded lies [inspection]
JSON reports `"selective_loader": False, "inference_mode": False` while the
defaults are loader=`selective` and generation under `torch.inference_mode`.
**Fix**: flags derived from the live runtime objects (model attributes,
loader argument, dist backend, argmax encoding, rmsnorm implementation);
contract test compares JSON flags to the model's actual configuration.

### B7 — distributed argmax token-id precision contract **[empirical]**
The v0.3 comment claims "ids stay int64 end-to-end" but `global_ids.float()`
stores ids in fp32, which is exact only below 2^24. Probed: ids 16_777_217,
20_000_001, 25_165_823 lose precision (the charter's own test ids
12095/150000/16_777_215/16_777_216/20_000_000 all happen to be fp32-exact —
the danger zone is **odd ids above 2^24**, which the old test list misses).
**Fix**: precision-safe one-collective encoding by default —
**bitpack**: monotone fp32→uint32 key transform (sign-flip trick), packed
`(key << 32) | (2^32−1 − id)` into int64 (one 64-bit element per candidate,
any vocab < 2^32, ties resolve to the smallest id by construction); plus
`fp64` pair and `split` (two typed gathers) encodings for the A/B/C
benchmark; legacy `fp32` path kept behind an explicit
`vocab_size ≤ 2^24` guard. NaN policy documented: a rank whose local max is
NaN cannot win (NaN replaced by −inf before packing). Tests include the odd
ids above and equal-logit ties.

### B8 — stale documentation contradicts the shipped loader [inspection]
README limitations still say "P0 checkpoint loading materializes the full CPU
state_dict per rank (selective safetensors loading is a P1)" and DESIGN §14
says loading materializes the full state dict — both false since v0.3 made
the selective loader the default. **Fix**: sweep README/DESIGN/AUDIT/
V03_REPORT for completed-but-still-future and deprecated-but-still-current
claims.

## P1

- **B9** `SelectiveTensorReader.names()` returns `[]` for single-file
  checkpoints (the very case the loader default uses), and `close()` only
  clears a dict — no explicit `safe_open` release (Windows file-handle
  pressure). Fix: real key listing for both layouts; `contextlib.ExitStack`
  ownership + close test.
- **B10** Loader still pays random initialization for every parameter before
  overwriting them from the checkpoint. Fix: `init_weights=False`
  construction path; **parameter coverage audit** asserting every parameter
  was written by the loader (NaN-prefill + finiteness check) — also guards
  against silently-uninitialized params forever.
- **B11** No direct-to-device path: CPU TP model is fully materialized, then
  `model.to(cuda)`. Fix: `selective_direct_gpu` loader (target-device
  construction, per-slice H2D), plus `loader_benchmark.py` reporting
  load_s / host RSS start/peak/end (sampled, honesty-marked) / GPU peak /
  H2D bytes & time; pinned double-buffer overlap studied in
  docs/LOADER_PIPELINE.md (timeline first; negative result acceptable).

## P2 (new hunts beyond the charter)

- **B12** [inspection] `collective_benchmark` all_gather gloo path allocates
  `parts` per call inside the timed lambda closure chain — harmless but the
  benchmark should measure steady state (kept, list reuse added).
- **B13** [inspection] benchmark JSON `nccl` version reported `None` on this
  host via the safe helper — fine, but metadata should also record
  `dist.is_nccl_available()` and the process-group backend so NCCL-absent
  hosts are self-describing.
- **B14** [inspection] `scripts/run_multi_gpu_bench.sh` relies on torchrun,
  which is broken on this Windows build — documented; Linux multi-GPU hosts
  (the actual target) are unaffected.
- **B15** [inspection] `docs/TEST_AUDIT.md` did not exist; test-quality sweep
  (or True / unused fixtures / shape-only asserts / production-helper
  oracles) added as a standing artifact.

## Disposition

B1–B8 are fixed in v0.4 with regression tests before any performance work;
B9–B11 in the loader stage; B12–B15 recorded. Performance claims in README
are re-based on the corrected benchmark (schema `minitp.bench/4`).
