#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
mkdir -p results/stage1_5/logs

run_pair() {
  local model="$1" hidden="$2" memory="$3" lr="$4"
  local pid=() seed device shard
  for seed in 8401 8402; do
    device="cuda:$((seed-8401))"
    shard="${model}_h${hidden}_m${memory}_lr${lr}_seed${seed}"
    if [[ "$hidden" == 64 && "$memory" == 16 ]]; then
      if [[ -f "results/stage1_5/stage1_5-development-v1/${model}_lr${lr}_seed${seed}/summary.json" ]]; then
        continue
      fi
    elif [[ -f "results/stage1_5/stage1_5-development-v1/${model}_h${hidden}_m${memory}_lr${lr}_seed${seed}/summary.json" ]]; then
      continue
    fi
    .venv/bin/python experiments/run_stage1_5.py --mode development --model "$model" \
      --seed "$seed" --lr "$lr" --hidden-dim "$hidden" --memory-dim "$memory" \
      --device "$device" > "results/stage1_5/logs/${shard}.log" 2>&1 &
    pid+=("$!")
  done
  for job in "${pid[@]}"; do wait "$job"; done
  printf '%s %s %s %s done\n' "$model" "$hidden" "$memory" "$lr"
}

for model in B0_no_memory B1_gru B2_single_memory B3_joint B4_shared B5_separate B6_gamma_zero B7_random_query; do
  for lr in 0.001 0.0003; do run_pair "$model" 64 16 "$lr"; done
done
for cell in '32 8' '128 32' '256 64' '256 128'; do
  read -r hidden memory <<< "$cell"
  for lr in 0.001 0.0003; do run_pair B5_separate "$hidden" "$memory" "$lr"; done
done
