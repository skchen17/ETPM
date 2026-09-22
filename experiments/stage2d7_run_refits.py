"""Parallel preregistered failed-checkpoint R0–R4 ceilings."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from stage2d7_refit import one


def main(args):
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one,
                    args.checkpoints / run / "checkpoint_1500.pt",
                    args.out / f"{run}.json", episodes=args.episodes,
                    steps=args.steps, reps=args.reps, seed=args.train_seed)
                   for run in args.runs]
        for future in as_completed(futures):
            print("COMPLETED", future.result()["run"], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--train-seed", type=int, default=37101)
    p.add_argument("--jobs", type=int, default=4)
    main(p.parse_args())
