# Test Quality Audit (v0.4)

Full-repo sweep for tests that look real but cannot fail, ordered by the
v0.4 charter. Result: **2 false-confidence findings (fixed), everything else
has a real oracle.**

## Findings

| test | contract | oracle | failure mode before v0.4 | status |
|---|---|---|---|---|
| `tests/test_kv_cache.py::test_cache_creation_*` (v0.3) | K/V unwritten slots never read | NaN poison actually passed into prefill/generation | **`assert ... or True` — unconditional pass**; second poisoned cache created but never passed to `generate_greedy` | **fixed**: file rewritten (K/V poisoned separately, cache really passed in, overflow, cursor bounds, zeros==empty) |
| shard-mapping tests (`test_fused_projections`) | shard ranges correct | production `_shard`/`pack_qkv` (convenience) | correlated-bug blindness | kept, plus **independent oracle** `tests/test_shard_oracle.py` (hand-computed arange indices) |
| `test_qwen_real` fp32 greedy | token equality vs HF | HF generate | — | real oracle (32 tokens, CUDA + CPU Gloo TP=2 16 tokens) |
| `test_bench_aggregation` (v0.3) | multi-rank aggregation semantics | max(mean(rank)) — **the wrong math** | passed while semantics were wrong (V04_AUDIT B5) | **replaced** by `test_bench_consistency.py` (per-sample element-wise MAX + synthetic skew proof 34 != 32.5) |
| profiler tests | records + return contract | synthetic records / identity `is` checks | return contract untested in v0.2 | `test_comm_profiler.py` covers both profiling states |
| loader tests | every parameter written | NaN-prefill + finiteness (no random init to hide behind) | v0.3 had random init masking missed params | `test_loader_coverage.py` |
| multi-GPU | real NCCL paths | HF greedy on TP=2 | impossible on this host | `test_multi_gpu.py` with honest auto-skip (`multi_gpu` marker, needs ≥2 GPUs + NCCL) |

## Sweep rules applied

- No `or True`, no unconditional pass, no `xfail`.
- `pytest.skip` only for environment gates (`SKIP_DISTRIBUTED_TESTS`,
  hardware/NCCL absence) — never for known failures.
- Tolerances: bf16 logits gate at atol 0.5 unchanged since v0.2; fp32 gates
  at 1e-4/1e-5; nothing was relaxed to make an optimization pass.
- Value assertions everywhere; shape-only asserts appear only as
  pre-conditions (`out.shape == ...`) before value checks.
- Production helpers may build test *inputs*, but expected values come from
  HF reference, hand-computed oracles, or cross-loader bitwise equality.
