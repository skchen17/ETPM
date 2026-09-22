"""Independent, preregistered Stage 2D.7 C0 replication; frozen D.3 recipe."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)
INIT_SEEDS = tuple(range(24101, 24109))
STREAM_SEEDS = tuple(range(25101, 25104))
EVAL_SEED = 26601


def one(root: Path, init: int, stream: int) -> dict:
    name = f"i{init}_d{stream}"
    out = root / "checkpoints" / name
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "experiments/stage2d3_train.py", "--arm", "C0",
               "--init-seed", str(init), "--data-seed", str(stream),
               "--eval-seed", str(EVAL_SEED), "--steps", "1500", "--batch", "16",
               "--evaluator-steps", "1000", "--train-p", ".70",
               "--gamma", ".50", "--rho-fast", ".97", "--rho-slow", ".9995",
               "--checkpoints", *map(str, STEPS), "--out", str(out)]
    with (out / "train.log").open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"{name} training failed: {out / 'train.log'}")
    summary = json.loads((out / "summary.json").read_text())
    return {"run": name, "checkpoint": str(out / "checkpoint_1500.pt"),
            "summary": summary, "recipe": command}


def main(args):
    args.root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(one, args.root, init, stream)
                   for init in INIT_SEEDS for stream in STREAM_SEEDS]
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            print("COMPLETED", row["run"], flush=True)
    rows.sort(key=lambda x: x["run"])
    (args.root / "training_summaries.json").write_text(json.dumps(rows, indent=2))
    print("COMPLETE_BASELINE", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    main(parser.parse_args())
