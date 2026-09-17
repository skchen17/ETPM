#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN_ID="${1:-stage1_2-formal-v1}"
PYTHON="${PYTHON:-.venv/bin/python}"
LOG_DIR="results/stage1_2/raw/${RUN_ID}/logs"
mkdir -p "$LOG_DIR"
models=(B1_gru B2_single_persistent B3_uniform B5_no_idle B6_full)
seeds=(3201 3202 3203 3204 3205 3206 3207 3208)
jobs=()
for seed in "${seeds[@]}"; do
  for model in "${models[@]}"; do jobs+=("${model}:${seed}"); done
done

worker() {
  local gpu="$1" offset="$2" index model seed log
  for ((index=offset; index<${#jobs[@]}; index+=2)); do
    model="${jobs[$index]%%:*}"; seed="${jobs[$index]##*:}"
    log="$LOG_DIR/${model}__seed${seed}.log"
    if [[ -f "results/stage1_2/raw/${RUN_ID}/${model}__seed${seed}__manifest.json" ]]; then
      echo "skip completed ${model} ${seed}" | tee -a "$log"; continue
    fi
    echo "start $(date --iso-8601=seconds) gpu=${gpu} ${model} ${seed}" | tee "$log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" experiments/run_stage1_2.py \
      --mode formal --model "$model" --seed "$seed" --device cuda --run-id "$RUN_ID" >>"$log" 2>&1
    echo "done $(date --iso-8601=seconds) gpu=${gpu} ${model} ${seed}" | tee -a "$log"
  done
}

worker 0 0 & pid0=$!
worker 1 1 & pid1=$!
wait "$pid0"; wait "$pid1"

