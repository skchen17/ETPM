#!/usr/bin/env bash
set -euo pipefail
cd /data/CSK/ETPM/et-rcm
root=results/stage2d3/curriculum_dev/checkpoints
mkdir -p "$root"
run_one() {
  local init="$1"
  local data="$2"
  local window="$3"
  local out="$root/C1_w${window}_i${init}_d${data}"
  mkdir -p "$out"
  PYTHONPATH=src:. .venv/bin/python experiments/stage2d3_train.py \
    --init-seed "$init" --data-seed "$data" --eval-seed 15601 \
    --arm C1 --warmup-steps "$window" --out "$out" > "$out.log" 2>&1
}
export -f run_one
export root
for window in 50 100 200 300; do
  run_one 9601 12601 "$window" &
  run_one 9602 12602 "$window" &
done
wait
printf 'CURRICULUM_DEV_COMPLETE\n'
