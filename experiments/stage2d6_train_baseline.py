"""Independent 8-initialization x 3-stream frozen C0 baseline replication."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)


def one(root: Path, init: int, stream: int) -> dict:
    name = f"i{init}_d{stream}"
    out = root / "checkpoints" / name
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "experiments/stage2d3_train.py", "--arm", "C0",
               "--init-seed", str(init), "--data-seed", str(stream),
               "--eval-seed", "23601", "--steps", "1500", "--batch", "16",
               "--evaluator-steps", "1000", "--train-p", ".70",
               "--gamma", ".50", "--rho-fast", ".97", "--rho-slow", ".9995",
               "--checkpoints", *map(str, STEPS), "--out", str(out)]
    with (out / "train.log").open("w") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"{name} training failed, see {out / 'train.log'}")
    summary = json.loads((out / "summary.json").read_text())
    return {"run": name, "checkpoint": str(out / "checkpoint_1500.pt"),
            "summary": summary, "recipe": command}


def main(args):
    args.root.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = [pool.submit(one, args.root, init, stream)
                   for init in range(21101, 21109)
                   for stream in range(22101, 22104)]
        rows = []
        for future in as_completed(pending):
            row = future.result(); rows.append(row)
            print("COMPLETED", row["run"], flush=True)
    rows.sort(key=lambda x: x["run"])
    (args.root / "training_summaries.json").write_text(json.dumps(rows, indent=2))
    print("COMPLETE_BASELINE", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    main(parser.parse_args())
