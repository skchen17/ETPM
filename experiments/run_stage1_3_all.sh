#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
models=(B0_no_persistent B1_gru B2_single_persistent B3_joint B4_m_only B5_f_only B6_arbitration B7_nonconserving_self_replay)
seeds=(5301 5302 5303 5304 5305 5306 5307 5308)
mkdir -p results/stage1_3/raw/stage1_3-formal-v1/logs
index=0
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    gpu=$((index % 2))
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/run_stage1_3.py --mode formal --model "$model" --seed "$seed" --device cuda > "results/stage1_3/raw/stage1_3-formal-v1/logs/${model}__seed${seed}.log" 2>&1 &
    index=$((index+1))
    if (( index % 2 == 0 )); then wait; fi
  done
done
wait
