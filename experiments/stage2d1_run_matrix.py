"""Launch the preregistered 8x4 factorial or curriculum matrices."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from etrcm.stage2d1.protocol import DATA_SEEDS, INIT_SEEDS


def main(args):
    pairs = [(i, d) for i in args.init_seeds for d in args.data_seeds]
    def one(pair):
        i, d = pair; out = args.out / f"i{i}_d{d}"
        cmd = [sys.executable, "experiments/stage2d1_train.py", "--init-seed", str(i),
               "--data-seed", str(d), "--eval-seed", str(args.eval_seed), "--curriculum", args.curriculum,
               "--gate-floor", str(args.gate_floor), "--gate-window", str(args.gate_window), "--out", str(out)]
        log = out.with_suffix(".log"); log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as handle: subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)
        return pair
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, pair) for pair in pairs]
        for future in as_completed(futures): print("COMPLETED", future.result(), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--init-seeds", type=int, nargs="+", default=list(INIT_SEEDS))
    p.add_argument("--data-seeds", type=int, nargs="+", default=list(DATA_SEEDS)); p.add_argument("--eval-seed", type=int, default=15101)
    p.add_argument("--curriculum", choices=("C0","C1","C2","C3","C4"), default="C0")
    p.add_argument("--gate-floor", type=float, default=.65); p.add_argument("--gate-window", type=int, default=100)
    p.add_argument("--jobs", type=int, default=16); p.add_argument("--out", type=Path, required=True); main(p.parse_args())
