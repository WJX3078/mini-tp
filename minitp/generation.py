"""Greedy autoregressive generation over the per-rank KV cache.

v0.2 hot-path rules (see docs/PERFORMANCE_AUDIT.md):

- Output ids are written into one preallocated ``[B, prompt+max_new]`` buffer —
  no per-token ``torch.cat`` (which was O(T^2) copies over a generation).
- Positions come from one precomputed arange, sliced per step (a view).
- ``early_stop=True``  — interactive mode: stops at EOS; costs one GPU→CPU
  sync per token (``bool(finished.all())``).
- ``early_stop=False`` — fixed-length benchmark mode: runs exactly
  ``max_new_tokens`` steps with **zero GPU→CPU synchronizations** (no
  ``.item()``/``bool()``); tokens after EOS are still written and must be
  ignored/trimmed by the caller.

``prefill``/``decode_step`` are exposed separately so benchmarks can time the
two phases (and each decode token) without re-implementing the loop.

All entry points run under ``torch.inference_mode`` (measured ~8 % faster than
``no_grad`` on the v0.3 ablation harness, docs/V03_REPORT.md); inference
tensors are returned, which is safe for greedy decoding.
"""

from __future__ import annotations

import torch

from minitp.distributed import all_gather
from minitp.kv_cache import KVCache
from minitp.layer import TPQwen2ForCausalLM


def _kv_heads_local(model: TPQwen2ForCausalLM) -> int:
    return model.model.layers[0].self_attn.kv_heads_local


def make_kv_cache(model: TPQwen2ForCausalLM, batch_size: int, max_seq_len: int) -> KVCache:
    return KVCache(
        num_layers=model.cfg.num_hidden_layers,
        batch_size=batch_size,
        kv_heads_local=_kv_heads_local(model),
        head_dim=model.cfg.head_dim,
        max_seq_len=max_seq_len,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
    )


@torch.inference_mode()
def prefill(
    model: TPQwen2ForCausalLM,
    input_ids: torch.Tensor,  # [B, T] identical on every rank
    kv: KVCache | None = None,
) -> tuple[torch.Tensor, KVCache]:
    """Prompt forward. Returns (local logits [B, T, vocab_local], kv cache)."""
    if kv is None:
        kv = make_kv_cache(model, input_ids.shape[0], input_ids.shape[1])
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    logits = model(input_ids, positions=positions, kv_cache=kv)
    return logits, kv


@torch.inference_mode()
def decode_step(
    model: TPQwen2ForCausalLM,
    token: torch.Tensor,  # [B, 1]
    kv: KVCache,
    position: torch.Tensor | None = None,  # [1] absolute position of the new token
) -> torch.Tensor:
    """One autoregressive step. Returns local logits [B, 1, vocab_local]."""
    return model(token, positions=position, kv_cache=kv)


def select_token(
    model: TPQwen2ForCausalLM, logits: torch.Tensor, distributed_argmax: bool
) -> torch.Tensor:
    """Greedy pick from local logits; identical result on every rank."""
    if distributed_argmax:
        return model.lm_head.distributed_argmax(logits[:, -1]).squeeze(-1)
    if model.ctx.tp_size == 1:
        return logits[:, -1].argmax(dim=-1)
    full = torch.empty(
        *logits.shape[:-1], model.cfg.vocab_size,
        dtype=logits.dtype, device=logits.device,
    )
    all_gather(full, logits[:, -1], model.ctx.process_group)
    return full.argmax(dim=-1)


@torch.inference_mode()
def generate_greedy(
    model: TPQwen2ForCausalLM,
    input_ids: torch.Tensor,  # [B, T_prompt] identical on every rank
    max_new_tokens: int,
    eos_token_id: int | None = None,
    distributed_argmax: bool = True,
    early_stop: bool = True,
    kv: KVCache | None = None,
) -> torch.Tensor:
    """Greedy decode into a preallocated buffer.

    Returns [B, prompt + generated]: trimmed at the first step where all rows
    are EOS-finished when ``early_stop=True``; exactly prompt+max_new_tokens
    columns when ``early_stop=False``.
    """
    b, t = input_ids.shape
    if kv is None:
        kv = make_kv_cache(model, b, t + max_new_tokens)
    total = t + max_new_tokens
    out_ids = torch.empty(b, total, dtype=torch.long, device=input_ids.device)
    out_ids[:, :t] = input_ids
    positions_buf = torch.arange(total, device=input_ids.device)

    logits, kv = prefill(model, input_ids, kv)
    cursor = t
    eos = torch.tensor(eos_token_id if eos_token_id is not None else -1, device=input_ids.device)
    finished = torch.zeros(b, dtype=torch.bool, device=input_ids.device) if early_stop else None

    for step in range(max_new_tokens):
        next_tok = select_token(model, logits, distributed_argmax)
        if early_stop and eos_token_id is not None:
            next_tok = torch.where(finished, eos, next_tok)
            finished = finished | (next_tok == eos)
        out_ids[:, cursor] = next_tok
        cursor += 1
        if early_stop and eos_token_id is not None and bool(finished.all()):
            break
        if step + 1 < max_new_tokens:
            logits = decode_step(
                model, next_tok.unsqueeze(-1), kv, positions_buf[cursor - 1 : cursor]
            )
    return out_ids[:, :cursor] if early_stop else out_ids
