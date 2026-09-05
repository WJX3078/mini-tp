"""Greedy autoregressive generation: prefill + decode over the per-rank KV cache."""

from __future__ import annotations

import torch

from minitp.distributed import all_gather
from minitp.kv_cache import KVCache
from minitp.layer import TPQwen2ForCausalLM


def _kv_heads_local(model: TPQwen2ForCausalLM) -> int:
    return model.model.layers[0].self_attn.kv_heads_local


@torch.no_grad()
def generate_greedy(
    model: TPQwen2ForCausalLM,
    input_ids: torch.Tensor,  # [B, T_prompt] identical on every rank
    max_new_tokens: int,
    eos_token_id: int | None = None,
    distributed_argmax: bool = True,
) -> torch.Tensor:
    """Greedy decode. With distributed_argmax and TP>1, token selection uses an
    O(tp) AllGather of (max value, token id) per row instead of gathering the
    full vocabulary logits. Returns [B, T_prompt + generated] identical on all
    ranks (single-token collectives keep ranks in lockstep)."""
    b, t = input_ids.shape
    kv = KVCache(
        num_layers=model.cfg.num_hidden_layers,
        batch_size=b,
        kv_heads_local=_kv_heads_local(model),
        head_dim=model.cfg.head_dim,
        max_seq_len=t + max_new_tokens,
        dtype=next(model.parameters()).dtype,
        device=input_ids.device,
    )
    eos = torch.tensor(eos_token_id if eos_token_id is not None else -1, device=input_ids.device)
    finished = torch.zeros(b, dtype=torch.bool, device=input_ids.device)

    logits = model(input_ids, kv_cache=kv)  # prefill -> local logits [B, T, vocab_local]
    out = input_ids
    for step in range(max_new_tokens):
        if distributed_argmax:
            next_tok = model.lm_head.distributed_argmax(logits[:, -1]).squeeze(-1)
        elif model.ctx.tp_size == 1:
            next_tok = logits[:, -1].argmax(dim=-1)
        else:
            full = torch.empty(
                *logits.shape[:-1], model.cfg.vocab_size,
                dtype=logits.dtype, device=logits.device,
            )
            all_gather(full, logits[:, -1], model.ctx.process_group)
            next_tok = full.argmax(dim=-1)
        next_tok = torch.where(finished, eos.clamp(min=0), next_tok)
        out = torch.cat([out, next_tok.unsqueeze(-1)], dim=1)
        finished = finished | (next_tok == eos) if eos_token_id is not None else finished
        if eos_token_id is not None and bool(finished.all()):
            break
        if step + 1 < max_new_tokens:
            logits = model(next_tok.unsqueeze(-1), kv_cache=kv)  # decode step
    return out
