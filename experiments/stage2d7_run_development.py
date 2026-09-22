"""Two-seed-per-configuration C0–C5 development; fixed grid, no per-seed tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SEEDS = ((26101, 27101), (26102, 27102))
CONFIGS = (
    ("C0", 4, .1, 300),
    ("C1", 4, .01, 300), ("C1", 4, .1, 300), ("C1", 4, .25, 300),
    ("C2", 4, .1, 100), ("C2", 4, .1, 300), ("C2", 4, .1, 500),
    ("C3", 2, .1, 300), ("C3", 4, .1, 300),
    ("C4", 2, .1, 300), ("C4", 4, .1, 300),
    ("C5", 4, .1, 300),
)


def name(arm, rank, ratio, freeze):
    return f"{arm}_r{rank}_lr{ratio:g}_freeze{freeze}"


def one(root, config, init, stream):
    arm, rank, ratio, freeze = config
    tag = name(*config)
    out = root / tag / f"i{init}_d{stream}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "experiments/stage2d7_train_compat.py",
           "--arm", arm, "--rank", str(rank), "--lr-ratio", str(ratio),
           "--freeze-step", str(freeze), "--init-seed", str(init),
           "--data-seed", str(stream), "--eval-seed", "28601",
           "--out", str(out)]
    with (out / "train.log").open("w") as log:
        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"failed {tag} {init} {stream}: {out / 'train.log'}")
    return {"config": tag, "arm": arm, "rank": rank, "lr_ratio": ratio,
            "freeze_step": freeze, "init_seed": init, "data_seed": stream,
            "checkpoint": str(out / "checkpoint_1500.pt"),
            "summary": json.loads((out / "summary.json").read_text())}


def main(args):
    args.root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, args.root, config, init, stream)
                   for config in CONFIGS for init, stream in SEEDS]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print("COMPLETED", row["config"], row["init_seed"], flush=True)
    rows.sort(key=lambda x: (x["config"], x["init_seed"]))
    (args.root / "training_summaries.json").write_text(json.dumps(rows, indent=2))
    print("COMPLETE_DEVELOPMENT", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=12)
    main(p.parse_args())
