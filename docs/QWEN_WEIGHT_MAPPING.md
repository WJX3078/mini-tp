# Qwen2.5-0.5B TP Weight Mapping

Tensor names and shapes below were read from the actual safetensors header of
`Qwen/Qwen2.5-0.5B` (BF16). PyTorch stores Linear weights as `[out_features,
in_features]`, so "Column" means slicing dim 0 and "Row" means dim 1 of the
stored tensor. Local shapes shown for TP=2.

| HF tensor name | shape | TP destination | partition dim | local shape (TP=2) |
|---|---|---|---|---|
| `model.embed_tokens.weight` | [151936, 896] | `model.embed_tokens.weight` | 0 (vocab) | [75968, 896] |
| `model.layers.{i}.input_layernorm.weight` | [896] | same | replicated | [896] |
| `model.layers.{i}.post_attention_layernorm.weight` | [896] | same | replicated | [896] |
| `model.layers.{i}.self_attn.q_proj.weight` | [896, 896] | `self_attn.q_proj.weight` | 0 (out) | [448, 896] |
| `model.layers.{i}.self_attn.q_proj.bias` | [896] | `self_attn.q_proj.bias` | 0 | [448] |
| `model.layers.{i}.self_attn.k_proj.weight` | [128, 896] | `self_attn.k_proj.weight` | 0 (kv-head blocks) | [64, 896] |
| `model.layers.{i}.self_attn.k_proj.bias` | [128] | `self_attn.k_proj.bias` | 0 | [64] |
| `model.layers.{i}.self_attn.v_proj.weight` | [128, 896] | `self_attn.v_proj.weight` | 0 (kv-head blocks) | [64, 896] |
| `model.layers.{i}.self_attn.v_proj.bias` | [128] | `self_attn.v_proj.bias` | 0 | [64] |
| `model.layers.{i}.self_attn.o_proj.weight` | [896, 896] | `self_attn.o_proj.weight` | 1 (in) | [896, 448] |
| `model.layers.{i}.mlp.gate_proj.weight` | [4864, 896] | `mlp.gate_proj.weight` | 0 (out) | [2432, 896] |
| `model.layers.{i}.mlp.up_proj.weight` | [4864, 896] | `mlp.up_proj.weight` | 0 (out) | [2432, 896] |
| `model.layers.{i}.mlp.down_proj.weight` | [896, 4864] | `mlp.down_proj.weight` | 1 (in) | [896, 2432] |
| `model.norm.weight` | [896] | `model.norm.weight` | replicated | [896] |
| (`lm_head` absent — tied to `embed_tokens`; the LM head reuses the embedding shard) | | `lm_head.weight` (shared param) | 0 (vocab) | shared |

Notes:
- k/v are sliced as contiguous KV-head blocks: with 2 KV heads and TP=2, rank r
  gets exactly head r (rows [r*64, (r+1)*64)), matching its contiguous block of
  7 Q heads (GQA group size 7).
- q/k/v carry bias in Qwen2; o_proj/gate/up/down do not.
- Verified by tests: shard-content unit tests (`tests/test_weight_loader.py`)
  and end-to-end token equality against HF greedy generation.
