#!/usr/bin/env bash
# Multi-GPU TP benchmark matrix. Requires >=2 CUDA GPUs (nvidia-smi).
# Results marked UNVERIFIED in README until executed on a real multi-GPU host.
set -euo pipefail
mkdir -p results
for TP in 1 2 4; do
  for PL in 128 512 2048; do
    for NT in 32 128; do
      torchrun --standalone --nproc-per-node="$TP" -m minitp.bench.benchmark \
        --model "${MODEL:-Qwen/Qwen2.5-0.5B}" \
        --dtype bf16 --prompt-len "$PL" --new-tokens "$NT" --iters 3 \
        --profile-communication \
        --output "results/tp${TP}_len${PL}_out${NT}.json"
    done
  done
done
