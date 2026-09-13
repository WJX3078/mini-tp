# v0.2/v0.3 Hot-Path Profile: before vs after

Measured with `torch.profiler` (CPU+CUDA activities), Qwen2.5-0.5B bf16,
TP=1, batch=1, decode (T=1, KV-cache present), 10 profiled steps after
warmup. Same GPU for both rows (NVIDIA GeForce RTX 4060 Laptop, torch 2.6.0+cu124).

> Caveat: profiler overhead inflates absolute `self CUDA` times (the v0.3 row
> sums to 198 ms/token against a ~50 ms wall clock). **Counts and ratios are
> the comparable signal**, never the absolute CUDA-ms.

## Kernel count per decode token

| metric | v0.1 (baseline) | v0.3 (measured) | delta |
|---|---|---|---|
| CUDA kernels / token | **6,565** | **5,662** | −903 (−13.8 %) |
| GEMM calls (`linear`+`mm`+`addmm`) | 169 `linear` | 267 (`mm`+`addmm` counted separately; 169 → 121 `linear` calls after fusions) | fewer, larger GEMMs |
| SDPA calls | 48 | 48 | — (one per layer, expected) |
| RMSNorm op kernels (`pow`/`mean`/`rsqrt`/`mul`) | ≈ 336 | 365 calls (incl. residual adds) | F.rms_norm would collapse these (opt-in) |
| dtype-cast kernels (`to`/`_to_copy`/`copy_`) | ≈ 700 (11 % of GPU time) | 1,018 calls / 39.4 ms self-CUDA | dominant remaining group (RoPE fp32 rotate + RMSNorm casts) |
| GQA `repeat_interleave` | 48 calls | 96 calls / 5.4 ms (TP=1: 2 local KV heads × 2 tensors × 24 layers) | TP≥2 uses zero-copy expand (UNVERIFIED) |
| RoPE table construction kernels | per-layer per-forward | 0 (cached); `cat`/`chunk` 121 calls remain from rotate_half itself | −(arange/outer/cat/cos/sin)×24 |
| view ops (`slice`/`transpose`/`as_strided`) | 433 slice + 1,037 as_strided + 361 transpose | 2,969 (all view ops) | mostly host-side dispatch |
| embedding | masked path | 1 call (TP=1 fast path) | −6 mask kernels |

v0.1 numbers: docs/PERFORMANCE_AUDIT.md (same GPU/methodology).

## What the after-profile says is left on the table

1. **dtype casts (1,018 calls, ~39 ms self-CUDA)** — the RoPE `rotate_half`
   fp32 round-trip and RMSNorm's fp32 upcast dominate. F.rms_norm (opt-in
   `implementation="functional"`) removes ~365 calls; it is **faster
   (−3.4 ms/token measured) but changes bf16 rounding order** (per-element
   diff ≤0.031; fp32 bit-exact) — kept opt-in, not default, per the
   no-tolerance-relaxation rule.
2. **GQA copies at TP=1 (5.4 ms)** — an attempt to fold the KV-head dim into
   the SDPA batch dim (one small q copy instead of 7×-expanded k/v copies)
   was implemented and **measured slower (TPOT 53→65 ms)** — the folded
   layout selects a slower SDPA path on this GPU. Reverted; negative result
   recorded (docs/V03_REPORT.md). At TP=2 the zero-copy expand path already
   avoids the copies.
3. **Native SDPA `enable_gqa`** measured 10–30× slower than manual
   expand/repeat on this GPU (math-backend fallback) — not adopted
   (scripts/gqa_experiment.py).

## Where to look next (P2, out of v0.3 scope)

CUDA graphs (docs/CUDA_GRAPH_FEASIBILITY.md) and torch.compile compute
islands target exactly the cast/view churn that remains.
