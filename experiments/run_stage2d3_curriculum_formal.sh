#!/usr/bin/env bash
set -euo pipefail
cd /data/CSK/ETPM/et-rcm
window="${1:?selected warmup window required}"
root=results/stage2d3/curriculum_formal/checkpoints
mkdir -p "$root"
run_one() {
  local arm="$1"
  local init="$2"
  local data="$3"
  local window="$4"
  local out="$root/${arm}_w${window}_i${init}_d${data}"
  mkdir -p "$out"
  PYTHONPATH=src:. .venv/bin/python experiments/stage2d3_train.py \
    --init-seed "$init" --data-seed "$data" --eval-seed 15701 \
    --arm "$arm" --warmup-steps "$window" --out "$out" > "$out.log" 2>&1
}
export -f run_one
export root
for arm in C0 C1 C2 C3; do
  for offset in $(seq 0 7); do
    run_one "$arm" "$((9701+offset))" "$((12701+offset))" "$window" &
  done
done
wait
printf 'CURRICULUM_FORMAL_COMPLETE\n'
