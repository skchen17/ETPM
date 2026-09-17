#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
RUN_ID="${1:-stage1_1-formal-v1}"
LR="${2:-0.001}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="results/stage1_1/raw/${RUN_ID}/logs"
mkdir -p "$LOG_DIR"

jobs=(
  B0_no_memory_mlp:2101 B0_no_memory_mlp:2102 B0_no_memory_mlp:2103
  B1_gru:2101 B1_gru:2102 B1_gru:2103
  B2_single_persistent:2101 B2_single_persistent:2102 B2_single_persistent:2103
  B3_uniform:2101 B3_uniform:2102 B3_uniform:2103
  B5_no_idle:2101 B5_no_idle:2102 B5_no_idle:2103
  B6_full:2101 B6_full:2102 B6_full:2103
)

worker() {
  local gpu="$1"
  local offset="$2"
  local index model seed log
  for ((index=offset; index<${#jobs[@]}; index+=2)); do
    model="${jobs[$index]%%:*}"
    seed="${jobs[$index]##*:}"
    log="$LOG_DIR/${model}__seed${seed}.log"
    if [[ -f "results/stage1_1/raw/${RUN_ID}/${model}__seed${seed}__manifest.json" ]]; then
      echo "skip completed ${model} ${seed}" | tee -a "$log"
      continue
    fi
    echo "start $(date --iso-8601=seconds) gpu=${gpu} ${model} ${seed}" | tee "$log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" experiments/run_stage1_1.py \
      --mode formal --model "$model" --seed "$seed" --learning-rate "$LR" \
      --device cuda --run-id "$RUN_ID" >>"$log" 2>&1
    echo "done $(date --iso-8601=seconds) gpu=${gpu} ${model} ${seed}" | tee -a "$log"
  done
}

worker 0 0 &
pid0=$!
worker 1 1 &
pid1=$!
wait "$pid0"
wait "$pid1"
