"""Formatting helpers for benchmark/communication results."""

from __future__ import annotations


def comm_percent(runtime_s: float, comm_ms: float) -> float:
    if runtime_s <= 0:
        return 0.0
    return round(min(100.0, comm_ms / (runtime_s * 1e3) * 100), 2)


def markdown_table(results: list[dict]) -> str:
    header = (
        "| Workload | TP | dtype | E2E (s) | ms/token | tok/s | Mem/GPU (GiB) | Comm % |"
        "\n|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for r in results:
        rows.append(
            f"| {r['prompt_len']}+{r['new_tokens']} | {r['tp_size']} | {r['dtype']} "
            f"| {r['e2e_latency_s']:.2f} | {sum(r['decode_ms_per_token'])/len(r['decode_ms_per_token']):.1f} "
            f"| {r['output_tokens_per_s']} "
            f"| {r.get('peak_mem_gib', 'n/a')} | {r.get('comm_percent', 'n/a')} |"
        )
    return "\n".join([header, *rows])
