#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN_ID="${1:-stage1_2-development-v1a1}"
STEPS="${2:-600}"
PYTHON="${PYTHON:-.venv/bin/python}"
mkdir -p "results/stage1_2/raw/${RUN_ID}/logs"
models=(B1_gru B2_single_persistent B3_uniform B5_no_idle B6_full)
lrs=(0.001 0.0003 0.0001)
seeds=(1301 1302)
jobs=()
for model in "${models[@]}"; do for lr in "${lrs[@]}"; do for seed in "${seeds[@]}"; do jobs+=("${model}:${lr}:${seed}"); done; done; done
worker() {
  local gpu="$1" offset="$2" i item model rest lr seed tag log out
  for ((i=offset; i<${#jobs[@]}; i+=2)); do
    item="${jobs[$i]}"; model="${item%%:*}"; rest="${item#*:}"; lr="${rest%%:*}"; seed="${rest##*:}"
    tag="${lr//./p}"; out="results/stage1_2/raw/${RUN_ID}/dev__${model}__lr${tag}__seed${seed}.parquet"
    log="results/stage1_2/raw/${RUN_ID}/logs/${model}__lr${tag}__seed${seed}.log"
    [[ -f "$out" ]] && { echo "skip $item"; continue; }
    echo "start $(date --iso-8601=seconds) gpu=${gpu} ${item}" | tee "$log"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" experiments/run_stage1_2.py --mode development-job \
      --model "$model" --learning-rate "$lr" --seed "$seed" --development-steps "$STEPS" \
      --device cuda --run-id "$RUN_ID" >>"$log" 2>&1
    echo "done $(date --iso-8601=seconds) ${item}" | tee -a "$log"
  done
}
worker 0 0 & p0=$!; worker 1 1 & p1=$!; wait "$p0"; wait "$p1"
CUDA_VISIBLE_DEVICES=0 "$PYTHON" experiments/run_stage1_2.py --mode development-collect \
  --development-steps "$STEPS" --device cuda --run-id "$RUN_ID"

