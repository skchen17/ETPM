"""Parallel held-out health audits for development/formal/expanded runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from stage2d7_eval_training import one


def main(args):
    checkpoints = sorted(args.checkpoints.rglob("checkpoint_1500.pt"))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, path,
                    args.out / path.parent.relative_to(args.checkpoints) / "evaluation.json",
                    args.reps) for path in checkpoints]
        for f in as_completed(futures):
            print("COMPLETED", f.result()["checkpoint"], flush=True)
    print("COMPLETE_EVALUATION", len(checkpoints), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--reps", type=int, default=16)
    main(p.parse_args())
