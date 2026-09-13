# mini-TP Review Notes (Adversarial Pass)

Reviewer stance: *assume a hidden TP correctness bug exists; find it.* This documents
the issues actually found during development and the checks performed on the final
code. All items are fixed and covered by tests unless marked otherwise.

## Bugs found and fixed during development

1. **bf16 token-id corruption in distributed argmax** — `torch.stack([bf16_values,
   int64_ids])` promotes to bf16, which cannot represent ids > 256 exactly; token
   12095 silently became 12096 and TP=2 generation produced garbage. Fix: cast both
   to fp32 before stacking. Found via real-model TP=2 generate test; impossible to
   hit at TP=1. (`minitp/parallel/embedding.py`)
2. **KV cache cursor advanced per layer** — each layer's attention advanced the shared
   cursor, so layer 1 wrote where layer 0 had just advanced past (with 24 layers,
   layer 23 wrote at position `prompt + 23`). Fix: layers write at the cursor; the
   model advances once per forward. (`minitp/attention.py`, `minitp/layer.py`)
3. **RowParallel bias semantics** — bias is added once *after* AllReduce. Regression
   test feeds zero input with all-ones bias and asserts output is exactly 1.0, not
   `tp_size`. (`tests/test_column_row_parallel.py`)
4. **GQA local expansion missing** — local Q heads (7) > local KV heads (1) requires
   `repeat_interleave` before SDPA; without it SDPA errors or (if shapes happened to
   match) would silently attend across wrong groups. (`minitp/attention.py`)
5. **Local Q/KV head contiguity** — KV-head slices must be contiguous blocks aligned
   with the rank's contiguous Q-head block; verified for shard and replication modes
   (`tests/test_weight_loader.py`).

## Checklist pass over final code

- **Shard dimensions**: Column → dim 0 of `[out, in]`; Row → dim 1; embedding/LM head →
  dim 0 (vocab). Unit tests build `arange` weights and compare exact contents per rank.
- **Collective ordering / deadlock**: every collective (AllReduce in RowParallel,
  AllGather in gather_output/distributed argmax, embedding AllReduce) is executed
  unconditionally on all ranks — no rank-dependent control flow in the model path.
  Decode selection (`distributed_argmax`) is itself a collective, keeping ranks in
  lockstep. Single-process TP=1 path calls no collectives at all.
- **Double/missing AllReduce**: AllReduce appears exactly twice per decoder layer
  (after o_proj, after down_proj) plus the embedding AllReduce once per model. The
  communication profiler cross-checks counts (`--profile-communication`).
- **Bias duplication**: ColumnParallel bias is sharded (sums exactly); RowParallel bias
  added post-reduce; o_proj/down_proj have no bias in Qwen2.
- **LM head vocab offsets**: `vocab_start` from the same `shard_range` arithmetic as
  the loader; tie-breaking matches `torch.argmax` (smallest id wins).
- **Embedding mask**: out-of-shard tokens clamp to index 0 and are zero-masked before
  the AllReduce, so no cross-shard contamination; uneven vocab (9 % 2) covered.
- **KV cache shapes**: `[B, kv_heads_local, S, head_dim]` per rank; TP=2 stores half
  the KV heads. Prefill-with-cache uses `tril(diagonal=cache_len)` masks (top-left
  causal misaligns with cache offsets — caught in review of the first draft).
- **Device/group consistency**: single process group (`WORLD`) used everywhere; the
  `ParallelContext` carries the group and device; tensors created on the context device.
- **Rank divergence**: only rank 0 prints; token selection results are identical on
  all ranks by construction (collective).
- **Weight loading**: loader writes into pre-allocated local parameters; GPU receives
  only local shards (CPU-side full state_dict is a documented P0 limitation).

## Known remaining gaps (documented, not bugs)

- `num_kv_heads < tp_size` replication path is unit-tested for shard contents but has
  no real-checkpoint end-to-end test (Qwen2.5-0.5B has 2 KV heads).
- P1 items not yet implemented: selective safetensors loading, async AllReduce /
  reduce-scatter experiments, sequence parallelism.
- bf16-vs-HF greedy equality is not asserted on CUDA (HF's own bf16 greedy is
  unstable); fp32 is the asserted path.

---

# v0.3 Adversarial Review — bugs found and fixed

Full analysis in docs/V03_AUDIT.md. The pattern across all of them: **the
v0.2 test suite tested features and collectives in isolation, so
cross-cutting contracts (return values, units, cross-rank branch agreement,
timer windows) had no owner.**

| bug | root cause | why v0.2 tests missed it | v0.3 regression test |
|---|---|---|---|
| Profiled `all_reduce`/`broadcast` returned `None` | `dist.all_reduce` returns None; wrapper passed `run()` through | profiler tests checked recorded stats, never the return value; model tests never enabled profiling | `tests/test_comm_profiler.py` asserts `is t` identity under BOTH profiling states, TP=2 Gloo |
| `comm_summary.host_launch_ms` ~1000× | summed ms values then multiplied by 1e3 again | TP=1 benchmarks have zero collectives; no TP=2 GPU run existed to expose it | deterministic mock-record test: summary == plain sum of records |
| `all_gather` CUDA-event window included `torch.cat`+copy | events bracketed the wrapper, not the op | no test compared collective vs postprocess timing | record now has `collective_gpu_ms` vs `postprocess_host_ms`; doc updated |
| Benchmark `--early-stop` inert but reported in JSON | flag parsed, never plumbed into the loop | CLI had no test; JSON field was asserted nowhere | flag removed; benchmark is fixed-length by contract; CLI smoke test |
| Checkpoint load timer included `snapshot_download` | download + materialize timed together | only visible when HF cache is cold | resolve untimed → barrier → timed load; `load.snapshot_resolve_s` separate |
| `new_tokens == 1` crashed the benchmark (empty percentile list) | mean over empty steps, `xs[0]` | no CLI test with new_tokens=1 | TTFT/TPOT refactor; TPOT null at 1 token; tested |
| Uneven-vocab `gather_logits` abort (rank divergence) | branch chosen from *local* shape — rank0 padded, rank1 didn't | uneven tests covered embedding only, not gather_logits | branch on the global `vocab % tp` property; TP=2 uneven gather test |
| Distributed argmax gather buffer mis-shaped `[*B, tp, 2]` | `all_gather_dim0` stacks along dim 0, buffer read as if `[..., tp, 2]` | only square B==tp cases were tested, where the misread is silent | non-square batch tests; shape fixed to `[tp, *B, 2]` + `movedim` |
| Gloo `all_gather_into_tensor` silently corrupts flat payloads | backend limitation in torch 2.6.0 Windows build | fast path was never exercised on Gloo with the probe shapes | dim-0 fast path feature-gated to NCCL; Gloo uses list path; values asserted |
| Selective loader skipped fused QKV **bias** (uninitialized memory) | `names()` empty for single-file checkpoints → bias check false | synthetic tests compare legacy vs selective — bias diff was caught immediately by the new test | `reader.has()` uses handle keys; single-file + index both covered |
| README claimed "28 % faster decode" | v0.1 e2e-derived decode vs v0.2 phase-measured — different methodology | — | apples-to-apples ablation harness (`benchmarks/ablation.py`); README now reports −19.9 % kernels, −24 % TTFT, −13 % TPOT under one harness |

Also verified-and-rejected on measurement (kept documented): native SDPA
`enable_gqa` (10–30× slower here), KV-head batch-fold (TPOT 53→65 ms,
reverted), RMSNorm `F.rms_norm` as default (faster but bf16 rounding-order
shift exceeds our logits gate; fp32 is bit-exact; kept opt-in).
