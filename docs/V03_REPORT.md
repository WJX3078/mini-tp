# mini-TP v0.3 Report — Trustworthy Benchmarking + Runtime Optimization

Answers the 17 questions of the v0.3 charter with measured evidence. Every
number below is from this host (NVIDIA GeForce RTX 4060 Laptop GPU, single
GPU, torch 2.6.0+cu124, bf16, Qwen2.5-0.5B) unless marked UNVERIFIED.

## 1. What the v0.2 architecture was

Megatron-style replicated-residual TP on raw torch.distributed: fused QKV and
fused gate/up GEMMs (packed at load), RowParallel o_proj/down_proj with one
AllReduce each, cached RoPE, vocab-parallel embedding/LM head with
distributed greedy argmax, per-rank sharded KV cache, preallocated
sync-free-fixed-length or early-stop decode, phase-separated benchmark and a
CUDA-event communication profiler. (v0.2 details: docs/PERFORMANCE_AUDIT.md.)

## 2. Bugs the v0.3 adversarial audit found

Eleven, all fixed with regression tests (docs/V03_AUDIT.md B1–B11):
profiled collectives returned `None` (crashing TP>1 + profiling), the
comm summary double-scaled milliseconds, AllGather profiling included
post-processing, `--early-stop` was inert but reported, the checkpoint load
timer included the HF download, `new_tokens == 1` crashed the benchmark,
uneven-vocab `gather_logits` aborted via cross-rank branch divergence, the
argmax gather buffer was silently mis-shaped for non-square batches, Gloo's
`all_gather_into_tensor` corrupts flat payloads (feature-gated to NCCL), the
selective loader skipped fused-QKV biases on single-file checkpoints, and
microbench `--iters` was ignored.

## 3. Why the original tests missed them

Each bug lived in a *contract between components* — return values, units,
rank agreement, timer windows — while the tests verified features in
isolation: profiler tests checked records, not return values; model tests
never enabled profiling; even/odd-vocab cases were split across different
features; the argmax buffer bug is silent exactly when batch == tp. v0.3's
regression suite now owns each contract explicitly.

## 4. Benchmark methodology corrections

Checkpoint load split from snapshot resolve; prefill / first-selection /
per-decode-step CUDA-event windows; TTFT = prefill + first selection,
TPOT = subsequent steps (argmax included), E2E = TTFT + TPOT·(N−1);
fixed-length zero-sync decode always; multi-rank max/min/mean/skew
aggregation via tensor collectives; percentiles nearest-rank; full
environment metadata in every JSON (schema `minitp.bench/3`).

## 5. TTFT / TPOT

TTFT: time from prompt-forward start to the first output token (prefill +
first greedy selection — the distributed argmax collective is part of
latency). TPOT: mean latency per subsequent token. E2E = TTFT +
TPOT·(new_tokens − 1). Legal at `new_tokens == 1` (TPOT null).

## 6. How selective loading works

`SelectiveTensorReader` maps tensor → shard file (index JSON or single file),
opens `safe_open` handles, and reads **only the rank's slices** — dim 0 for
column/vocab, dim 1 for row, KV-head blocks for k/v (replication included),
full reads only for tiny 1-D norms/biases (PySafeSlice 1-D indexing is
unreliable in safetensors 0.8 and returns garbage). Fused QKV / gate-up are
packed from rank-local slices at load. Legacy (full state dict) is kept as
`--loader legacy`; both produce bit-identical local parameters
(tests/test_selective_loader.py, synthetic single-file + index-sharded
checkpoints, TP=1/2, GQA + KV replication, tied and untied).
Measured on the real checkpoint (TP=1, bf16): **3.87 s → 2.97 s load, host
RSS +1.32 GiB → +0.01 GiB**, GPU peak unchanged (0.92 GiB).

## 7. Distributed argmax optimization

Persistent `[tp, *B, 2]` fp32 gather buffer (reused across tokens) + dim-0
`all_gather_into_tensor` on NCCL (Gloo takes the list path — its
`all_gather_into_tensor` silently corrupts flat payloads on this torch
build); values cast fp32 before pairing (the v0.1 bf16-token-id bug class),
ids int64 end-to-end; tie-break = smallest global id. TP=1 is a direct
`argmax` (3.1× faster microbench, ids 12095/50000/150000 + exact ties
regression-tested).

## 8. Native GQA experiment

`enable_gqa=True` is 10–30× slower than manual expand/repeat on this GPU
(math-backend fallback) with a 7.8e-3 bf16 diff — rejected; manual
expand/repeat stays default (scripts/gqa_experiment.py). A KV-head
batch-fold variant (one small q copy instead of 7×-expanded k/v) also
measured **slower (TPOT 53 → 65 ms) and was reverted** — recorded as a
negative result per the no-hidden-failures rule.

## 9. RMSNorm experiment

Reference (7 kernels) vs `F.rms_norm` (1 kernel): fp32 bit-exact, bf16
per-element diff ≤0.031 (rounding order) which amplifies to 1.17 on
30-magnitude logits — **faster (−3.4 ms/token) but kept opt-in**
(`RMSNorm.implementation="functional"`) because adopting it would require
relaxing our bf16 correctness gate.

## 10/11. Full profiler before/after & kernel count

docs/V02_PROFILE_RESULT.md. Measured: **6,565 → 5,662 kernels/token
(−13.8 %) on the same methodology**; ablation's deterministic count across
configurations: **6,952 → 5,569 (−19.9 %)**. Remaining budget: dtype casts
(1,018 calls), RMSNorm op kernels, view dispatch.

## 12. What actually improved (same-harness ablation)

- RoPE cache: −8.3 ms/token TPOT, −528 kernels (largest single win)
- inference_mode: −4.1 ms/token
- fused QKV: −624 kernels/token (TPOT contribution within noise on this
  launch-bound GPU; latency shows clearly on prefill)
- fused gate/up, TP=1 embedding/argmax fast paths: kernel-count reductions,
  small TPOT effect
- selective loader: −23 % load time, host memory ~0 (reported separately —
  never counted as decode improvement)
- Overall apples-to-apples: **TPOT −13 %, TTFT −24 %, kernels −19.9 %**
  (run variance ±5 % on this laptop; kernel counts are exact).

## 13. What did not

- KV-head batch-fold: regressed (reverted).
- native `enable_gqa`: regressed (rejected).
- fused gate/up latency: within noise on this GPU (kept: fewer launches,
  wins where dispatch is more expensive).
- `torch.compile` islands: deferred (P2) — first-pass experiments kept
  recompiling on decode shapes; revisit after CUDA-graph study
  (docs/CUDA_GRAPH_FEASIBILITY.md).

## 14. TP communication bottleneck

Decode pays 48 latency-bound AllReduces per token of 1792 B each
(B=1, hidden 896, bf16). The measured collective floor (Gloo TCP loopback
TP=2): ~563 µs per 4 KB AllReduce, 0.11 GB/s busbw at 64 MB — socket
transport is hopeless for TP decode; NCCL over real links is the only
viable path (docs/TP_SCALING.md).

## 15/16. Verified vs UNVERIFIED

Verified (this host): all TP=1 GPU results above; CPU Gloo TP=2
correctness (44 → 59 tests); Gloo TP=2 collective floor.
UNVERIFIED (needs ≥2 CUDA GPUs): NCCL TP=2/4 performance, rank-skew
behavior, zero-copy expand + dim-0 gather fast-path performance on NCCL,
graph/collective interaction.

## 17. Next highest-value work

1. Run `scripts/run_multi_gpu_bench.sh` on real hardware → fill the TP
   scaling matrix (the single biggest credibility upgrade).
2. Paged/bucketed KV to unblock CUDA-graph island capture (docs/CUDA_GRAPH_FEASIBILITY.md).
3. Adopt `F.rms_norm` default once bf16 tolerance policy is settled
   (fp32-exact today).
4. Selective loader P2: mmap + pinned-staging for overlap of H2D with file IO.
