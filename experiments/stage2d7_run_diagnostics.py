"""Parallel read-only Stage 2D.7 audits for one prespecified cohort."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from stage2d7_diagnostic import one


def main(args):
    runs = sorted(args.checkpoints.glob("*/checkpoint_1500.pt"))
    template = json.loads(args.template.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, path, args.out / f"{path.parent.name}.json",
                               template, args.reps, args.trajectory) for path in runs]
        for f in as_completed(futures):
            print("COMPLETED", f.result()["run"], flush=True)
    print("COMPLETE_COHORT", args.cohort, len(runs), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--trajectory", action="store_true")
    main(p.parse_args())
