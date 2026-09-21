#!/usr/bin/env bash
set -euo pipefail
cd /data/CSK/ETPM/et-rcm
root=results/stage2d3/confirmatory_baseline/checkpoints
mkdir -p "$root"
run_one() {
  local init="$1"
  local data="$2"
  local out="$root/i${init}_d${data}"
  mkdir -p "$out"
  PYTHONPATH=src:. .venv/bin/python experiments/stage2d1_train.py \
    --init-seed "$init" --data-seed "$data" --eval-seed 15501 \
    --gamma .50 --rho-fast .97 --rho-slow .9995 --curriculum C0 \
    --out "$out" > "$out.log" 2>&1
}
export -f run_one
export root
for init in $(seq 9501 9508); do
  for data in $(seq 12501 12503); do
    printf '%s %s\n' "$init" "$data"
  done
done | xargs -n2 -P24 bash -c 'run_one "$0" "$1"'
printf 'TRAINING_COMPLETE\n'
