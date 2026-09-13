"""Greedy autoregressive generation over the per-rank KV cache.

v0.4: ``GenerationState`` is the SINGLE decode hot path — ``generate_greedy``,
the benchmark, and the ablation all drive the same object, so the measured
path is the shipped path (v0.3's benchmark re-implemented the loop and quietly
re-built ``torch.arange`` positions every token: 10 aranges per benchmark
iteration vs 2 on the generation path, docs/V04_AUDIT.md B3).

Rules carried over from v0.2/v0.3:

- Output ids go into one preallocated ``[B, prompt+max_new]`` buffer; positions
  come from one precomputed arange sliced per step (a view).
- ``early_stop=True``  — interactive: stops at EOS; one GPU→CPU sync per token.
- ``early_stop=False`` — benchmark: exactly ``max_new_tokens`` steps, zero
  GPU→CPU synchronizations; tokens after EOS are written but must be ignored
  by the caller.

All entry points run under ``torch.inference_mode`` (measured faster on the
v0.3 ablation harness); inference tensors are returned, which is safe for
greedy decoding.
"""

from __future__ import annotations

import torch

from minitp.distributed import all_gather
from minitp.kv_cache import KVCache
from minitp.layer import TPQwen2ForCausalLM


def _kv_heads_local(model: TPQwen2ForCausalLM) -> int:
    return model.model.layers[0].self_attn.kv_heads_local


def make_kv_cache(
    model: TPQwen2ForCausalLM,
    batch_size: int,
    max_seq_len: int,
    init: str = "empty",
) -> KVCache:
    return KVCache(
        num_layers=model.cfg.num_hidden_layers,
        batch_size=batch_size,
        kv_heads_local=_kv_heads_local(model),
        head_dim=model.cfg.head_dim,
        max_seq_len=max_seq_len,
        dtype=next(model.parameters()).dtype,
        device=next(model.parameters()).device,
        init=init,
    )


def select_token(
    model: TPQwen2ForCausalLM, logits: torch.Tensor, distributed_argmax: bool = True
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


class GenerationState:
    """The one decode hot path: KV cache + positions buffer + output buffer.

    Usage::

        state = GenerationState(model, input_ids, max_new_tokens)
        logits = state.prefill()                 # prompt forward
        tok = state.select_next(logits)          # first token (TTFT end)
        pos = state.append(tok)
        logits = state.decode_step(tok, pos)     # subsequent steps
        tok = state.select_next(logits)
        ...
        state.out_ids  # [B, prompt + written tokens]

    Positions come exclusively from the precomputed buffer — no per-step
    ``torch.arange`` anywhere after construction.
    """

    def __init__(
        self,
        model: TPQwen2ForCausalLM,
        input_ids: torch.Tensor,  # [B, T] identical on every rank
        max_new_tokens: int,
        kv: KVCache | None = None,
        distributed_argmax: bool = True,
        kv_init: str = "empty",
    ) -> None:
        self.model = model
        self.distributed_argmax = distributed_argmax
        b, t = input_ids.shape
        self.prompt_len = t
        self.max_new_tokens = max_new_tokens
        self.total = t + max_new_tokens
        self.kv = kv if kv is not None else make_kv_cache(model, b, self.total, init=kv_init)
        device = input_ids.device
        self.out_ids = torch.empty(b, self.total, dtype=torch.long, device=device)
        self.out_ids[:, :t] = input_ids
        self.positions_buf = torch.arange(self.total, device=device)
        self.cursor = t  # tokens written into out_ids so far
        self.finished: torch.Tensor | None = None
        self.eos = -1

    def prefill(self) -> torch.Tensor:
        """Prompt forward; positions come from the buffer (0..T-1)."""
        positions = self.positions_buf[: self.prompt_len]
        return self.model(
            self.out_ids[:, : self.prompt_len], positions=positions, kv_cache=self.kv
        )

    def select_next(self, logits: torch.Tensor) -> torch.Tensor:
        return select_token(self.model, logits, self.distributed_argmax)

    def append(self, tok: torch.Tensor) -> torch.Tensor:
        """Write token at the cursor; returns its absolute position tensor."""
        self.out_ids[:, self.cursor] = tok
        pos = self.positions_buf[self.cursor]
        self.cursor += 1
        return pos

    def decode_step(self, tok: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return self.model(tok.unsqueeze(-1), positions=pos.unsqueeze(-1), kv_cache=self.kv)


@torch.inference_mode()
def generate_greedy(
    model: TPQwen2ForCausalLM,
    input_ids: torch.Tensor,  # [B, T_prompt] identical on every rank
    max_new_tokens: int,
    eos_token_id: int | None = None,
    distributed_argmax: bool = True,
    early_stop: bool = True,
    kv: KVCache | None = None,
    kv_init: str = "empty",
) -> torch.Tensor:
    """Greedy decode through ``GenerationState``.

    Returns [B, prompt + generated]: trimmed at the first step where all rows
    are EOS-finished when ``early_stop=True``; exactly prompt+max_new_tokens
    columns when ``early_stop=False``.
    """
    state = GenerationState(
        model, input_ids, max_new_tokens, kv=kv,
        distributed_argmax=distributed_argmax, kv_init=kv_init,
    )
    b = input_ids.shape[0]
    if early_stop and eos_token_id is not None:
        state.eos = eos_token_id
        state.finished = torch.zeros(b, dtype=torch.bool, device=input_ids.device)

    logits = state.prefill()
    eos_tok = torch.tensor(state.eos, device=input_ids.device)
    for step in range(max_new_tokens):
        tok = state.select_next(logits)
        if state.finished is not None:
            tok = torch.where(state.finished, eos_tok, tok)
            state.finished = state.finished | (tok == state.eos)
        pos = state.append(tok)
        if state.finished is not None and bool(state.finished.all()):
            break
        if step + 1 < max_new_tokens:
            logits = state.decode_step(tok, pos)
    return state.out_ids[:, : state.cursor] if early_stop else state.out_ids
