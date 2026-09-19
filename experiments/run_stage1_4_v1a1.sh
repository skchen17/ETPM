#!/usr/bin/env bash
set -euo pipefail

phase="${1:?phase: development/formal/evaluate}"
gpu="${2:?GPU index required}"
partition="${3:?partition 0 or 1 required}"
if [[ "$partition" == "0" ]]; then
  models=(B0_no_memory B2_single_memory B4_shared B6_gamma_zero)
elif [[ "$partition" == "1" ]]; then
  models=(B1_gru B3_joint B5_separate B7_random_query)
else
  echo "Invalid partition" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
config=configs/stage1_4_v1a1.yaml
if [[ "$phase" == "development" ]]; then
  run_id=stage1_4-development-v1a1
  seeds=(6501 6502)
  lrs=(0.001 0.0003)
elif [[ "$phase" == "formal" || "$phase" == "evaluate" ]]; then
  run_id=stage1_4-formal-v1a1
  seeds=(7501 7502 7503 7504 7505 7506 7507 7508)
  lrs=(0.001)
else
  echo "Invalid phase" >&2
  exit 2
fi

for model in "${models[@]}"; do
  for seed in "${seeds[@]}"; do
    for lr in "${lrs[@]}"; do
      if [[ "$phase" == "evaluate" ]]; then
        shard="${model}_seed${seed}"
        if [[ -s "results/stage1_4/$run_id/$shard/summary.json" ]]; then
          echo "SKIP $shard"
          continue
        fi
        echo "EVAL $shard GPU $gpu"
        .venv/bin/python experiments/evaluate_stage1_4.py --phase formal \
          --model "$model" --seed "$seed" --device "cuda:$gpu" --config "$config"
      else
        shard="${model}_lr${lr}_seed${seed}"
        if [[ -s "results/stage1_4/$run_id/$shard/summary.json" ]]; then
          echo "SKIP $shard"
          continue
        fi
        echo "RUN $shard GPU $gpu"
        .venv/bin/python experiments/run_stage1_4.py --mode "$phase" \
          --model "$model" --seed "$seed" --lr "$lr" --run-id "$run_id" \
          --device "cuda:$gpu" --config "$config"
      fi
    done
  done
done
