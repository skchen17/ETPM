#!/usr/bin/env bash
set -euo pipefail

gpu="${1:?GPU index required}"
partition="${2:?partition 0 or 1 required}"
if [[ "$partition" == "0" ]]; then
  models=(B0_no_memory B2_single_memory B4_shared B6_gamma_zero)
elif [[ "$partition" == "1" ]]; then
  models=(B1_gru B3_joint B5_separate B7_random_query)
else
  echo "Invalid partition: $partition" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
run_id=stage1_4-formal-v1
for model in "${models[@]}"; do
  for seed in 7401 7402 7403 7404 7405 7406 7407 7408; do
    shard="${model}_lr0.001_seed${seed}"
    if [[ -s "results/stage1_4/$run_id/$shard/summary.json" ]]; then
      echo "SKIP $shard"
      continue
    fi
    echo "RUN $shard GPU $gpu"
    .venv/bin/python experiments/run_stage1_4.py \
      --mode formal --model "$model" --seed "$seed" --lr 0.001 \
      --run-id "$run_id" --device "cuda:$gpu"
  done
done
