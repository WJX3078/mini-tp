# mini-TP v0.4 Report — Multi-GPU Correctness + Low-Latency TP Runtime

Answers the 17 questions of the v0.4 charter. Environment unless stated:
RTX 4060 Laptop GPU (single), torch 2.6.0+cu124 (no NCCL compiled,
`dist.is_nccl_available() == False`), bf16, Qwen2.5-0.5B.

## 1. Bugs found in v0.3 (docs/V04_AUDIT.md)

Eleven findings, 2 BLOCKER + 6 P0 + 3 P1: aggregation collectives on CPU
tensors (crashes under NCCL), KV-cache V still zero-filled plus two
false-confidence poison tests, benchmark decode path diverging from the
generation hot path (10 vs 2 `torch.arange` per iteration — measured), TTFT
built from two isolated synchronized spans, `max(mean(rank))` aggregation
math, hardcoded-wrong feature flags, fp32 token-id precision hole in the
distributed argmax, stale loader documentation, `names()`/handle-cleanup
gaps, wasted random init in the loader, missing direct-to-device path.

## 2. Why the v0.3 tests missed them

Same pattern as v0.3's own audit: contracts between components had no owner.
The Gloo CPU suite cannot see a NCCL device bug; the benchmark asserted its
JSON fields but not that its decode loop equals `generate_greedy`'s; poison
tests were written but the poisoned object was never wired into the code
under test; the argmax test list (12095/150000/16777215/16777216/20000000)
happened to contain only fp32-exact values — the dangerous set is **odd ids
above 2^24** (16_777_217 rounds to 16_777_216, measured).

## 3. Why the KV poison test was false confidence

It asserted `... or True` (a tautology) and created a second, poisoned cache
that was never passed to `generate_greedy` — the generation path under test
used a fresh, clean cache. v0.4 rewrites the file: K and V poisoned
separately, the poisoned cache really passed in (prefill AND multi-token
generation), overflow, cursor bounds (`isnan` outside `[:seq_len]` after
every advance), and a zeros-vs-empty equivalence check.

## 4. Why a CPU tensor is illegal for NCCL aggregation

`ProcessGroupNCCL` maps collectives onto CUDA streams; it requires CUDA
tensors and raises on CPU input. Gloo accepts both, which is why the CPU
Gloo test passed. The fix places aggregation tensors on `ctx.device`
(backend-aware without inspecting `torch.cuda.is_available()`), covered by a
capturing-spy contract test and a real `@pytest.mark.multi_gpu` NCCL test
that auto-skips on hosts without hardware.

## 5. `max(mean(rank))` vs `mean(step max)`

Lockstep TP makes every rank pay the slowest rank **per step**. Reducing each
rank to its mean first hides tail spikes: rank0=[10,10,10,100],
rank1=[12,12,12,12] gives max(mean)=32.5 vs mean(step max)=34. v0.4 keeps
per-step samples per rank and applies one element-wise MAX collective after
measurement; all TPOT percentiles are computed on the *global slowest-rank*
sample vector; per-iteration prefill/TTFT get the same treatment; rank-local
stats and skew are retained.

## 6. TTFT measurement fix

v0.3: prefill events → synchronize → selection events → synchronize → sum of
two isolated device spans (host gaps excluded). v0.4: ONE outer event pair
spans prefill start → first token selected with **no intermediate
synchronize**; a nested inner pair breaks out prefill; single resolve at the
end. Reported: continuous `ttft`, `prefill_gpu_ms`, `first_selection_ms`;
invariant tested: ttft ≥ prefill + selection − event noise.

## 7. Unifying the benchmark and generation hot paths

`GenerationState` (KV cache + positions buffer + output buffer + cursor +
`prefill`/`select_next`/`append`/`decode_step`) is now the only decode loop.
`generate_greedy`, the benchmark, and the ablation all drive it; benchmarks
time it and never re-implement generation. Regression: monkeypatched
`torch.arange` counts must be equal (and equal 1 — the positions buffer)
between the benchmark path and fixed-length generation.

## 8. Precision-safe distributed argmax

Default encoding is now **bitpack**: monotone fp32→uint32 key transform
(sign-flip trick, −0.0 normalized, NaN→−inf before the local max so a NaN
cannot shadow a real logit), packed `(key << 32) | (2^32−1 − id)` into one
int64 per candidate — one collective, any vocab < 2^32, exact ids, ties
resolve to the smallest id by construction (a first implementation overflowed
int64 by shifting the full 32-bit key; corrected to a signed 31-bit monotone
map). Legacy `fp32` path kept behind an explicit `vocab_size ≤ 2^24` guard
(the audit probe shows odd ids ≥ 2^24 corrupt); `fp64` and `split` (two
gathers) implemented for the A/B/C comparison; correctness tested over the
real 2-rank Gloo path with ids 16_777_217, 20_000_001, 20_979_201, exact
cross-rank ties, all-negative logits and the NaN policy.

## 9. Loader: startup memory/latency

- No random init in any loader (`init_weights=False`); a coverage audit
  poisons every parameter with NaN and asserts finiteness after load, so a
  skipped parameter fails loudly.
- `direct_gpu`: model constructed on the target device, rank-local slices
  copied straight into GPU parameters — no CPU TP model at all.
- Measured (`minitp/bench/loader_benchmark.py`, real checkpoint): legacy
  0.88 s, selective 0.61 s, direct_gpu 0.84 s; GPU peak identical (0.92 GiB).
  direct_gpu is *slower* on a 0.5B model (many small H2D vs one bulk
  transfer) — recorded; its value is host footprint on larger models.
- Pinned double-buffer overlap: no measurable opportunity at this scale
  (H2D 0.92 GiB, page-cached reads) — negative result, docs/LOADER_PIPELINE.md.

## 10. RMSNorm backend decision

New experiment: `torch.compile(reference impl)` is **bit-exact to the
reference path** and 2.5–3.3× faster (0.067 vs 0.218 ms at [1,1,896]) — this
corrects the v0.3 conclusion that only the numerically-shifting `F.rms_norm`
was fast. Backends now: `reference` (default, oracle), `functional` (fastest,
bf16 rounding shift — not default), `compiled` (bit-exact fast path, used in
the v0.4 full ablation config), `triton` (not attempted: Triton does not
support this Windows host; recorded as unavailable, not as a benchmark).

## 11. 1792-byte AllReduce latency

The real decode message (B=1, T=1, hidden 896, bf16) on **Gloo TCP loopback
TP=2**: p50 ≈ 525 µs, p99 ≈ 868 µs (all_reduce; AllGather 301 µs). Purely
latency-bound — at 48 AllReduces/token this transport would add ~25 ms/token
of pure communication. A tight-loop variant of the same collective measured
0.6 µs/iter, demonstrating how harness-sensitive loopback socket timing is;
both numbers and their harnesses are documented. NCCL numbers: UNVERIFIED.

## 12. Symmetric communication benefit

Not measurable on this host: the torch build lacks both NCCL and
`torch.distributed._symmetric_memory`. Capability detection was added and the
experiment documented as SKIP, not FAIL; any integration is gated on a
microbenchmark win (charter §19/§20).

## 13. Does async AllReduce overlap?

No, and it cannot in this runtime: after o_proj/down_proj the AllReduce
result feeds the residual add and RMSNorm — the consumer depends on it.
Measured on Gloo TP=2 with a wait-at-first-consumer pattern: sync 0.6 µs/iter
vs async+wait 0.6 µs/iter — identical. Recorded as a negative result; the
useful direction is reduce-scatter + sequence parallel (prefill only, P3).

## 14. Verified multi-GPU results

CPU/Gloo TP=2: all correctness (59→79 tests) + collective floor. **Real
NCCL TP=2/4: still UNVERIFIED on this host** (no NCCL in the build, one GPU);
`tests/test_multi_gpu.py` now contains the real NCCL aggregation, bitpack
argmax and end-to-end TP=2-vs-HF greedy tests and runs automatically where
the hardware exists.

## 15. PCIe vs NVLink

docs/TP_TOPOLOGY.md records the host (single PCIe-attached laptop GPU, no
NCCL in the build). No interconnect comparison can be measured here; the
honest statement remains: decode's 48×1792 B AllReduces are latency-bound, so
interconnect *latency* (NVLink ≪ PCIe) dominates TP decode scaling, while
prefill's large messages are bandwidth-bound.

## 16. What failed

- KV-head batch-fold (v0.3): TPOT regression, reverted.
- Native SDPA `enable_gqa`: 10–30× slower here (v0.3).
- `direct_gpu` loader on a 0.5B model: slower than CPU-staged selective
  (kept for host-footprint scaling).
- Pinned double-buffer load overlap: no room at this scale (timeline).
- async AllReduce: no overlap possible (dependency analysis + measurement).

## 17. Next (v0.5)

1. Real multi-GPU run: the entire NCCL matrix is scripted and waiting
   (`scripts/run_multi_gpu_bench.sh`, `tests/test_multi_gpu.py`).
2. ReduceScatter+AllGather sequence-parallel prefill prototype (P3) — the
   only structurally promising communication overlap.
3. Re-evaluate `rmsnorm_fn` vs `compiled` defaults on hardware where launch
   overhead is lower (the bit-exact compiled path makes the numerics
   question moot for latency work).
4. Selective loader: mmap + pinned staging becomes worthwhile only for
   checkpoints larger than RAM — revisit with a bigger model on multi-GPU
   hardware.
