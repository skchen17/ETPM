#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
models=(B0_no_persistent B1_gru B2_single_persistent B3_joint B4_m_only B5_f_only B6_arbitration B7_nonconserving_self_replay)
lrs=(0.001 0.0003 0.0001)
seeds=(4301 4302)
index=0
for model in "${models[@]}"; do
  for lr in "${lrs[@]}"; do
    for seed in "${seeds[@]}"; do
      gpu=$((index % 2))
      CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/run_stage1_3.py --mode development --model "$model" --lr "$lr" --seed "$seed" --device cuda > "results/stage1_3_dev_${model}_${lr}_${seed}.log" 2>&1 &
      index=$((index+1))
      if (( index % 2 == 0 )); then wait; fi
    done
  done
done
wait
.venv/bin/python experiments/run_stage1_3.py --mode select-lr
index=0
for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    gpu=$((index % 2))
    CUDA_VISIBLE_DEVICES=$gpu .venv/bin/python experiments/run_stage1_3.py --mode threshold --model "$model" --seed "$seed" --device cuda > "results/stage1_3_threshold_${model}_${seed}.log" 2>&1 &
    index=$((index+1))
    if (( index % 2 == 0 )); then wait; fi
  done
done
wait
.venv/bin/python experiments/run_stage1_3.py --mode freeze-selection
