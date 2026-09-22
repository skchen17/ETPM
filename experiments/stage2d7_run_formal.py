"""Eight independent formal seeds for frozen selected arm and matched controls."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SEEDS = tuple((29101 + i, 30101 + i) for i in range(8))


def parse_config(name):
    parts = name.split("_")
    arm = parts[0]
    rank = int(parts[1][1:])
    ratio = float(parts[2][2:])
    freeze = int(parts[3][6:])
    return arm, rank, ratio, freeze


def tag(config):
    arm, rank, ratio, freeze = config
    return f"{arm}_r{rank}_lr{ratio:g}_freeze{freeze}"


def one(root, config, init, stream):
    arm, rank, ratio, freeze = config
    out = root / tag(config) / f"i{init}_d{stream}"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "experiments/stage2d7_train_compat.py",
           "--arm", arm, "--rank", str(rank), "--lr-ratio", str(ratio),
           "--freeze-step", str(freeze), "--init-seed", str(init),
           "--data-seed", str(stream), "--eval-seed", "31601", "--out", str(out)]
    with (out / "train.log").open("w") as log:
        process = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"failed {tag(config)} {init}: {out / 'train.log'}")
    return {"config": tag(config), "run": out.name,
            "checkpoint": str(out / "checkpoint_1500.pt"),
            "summary": json.loads((out / "summary.json").read_text())}


def main(args):
    selection = json.loads(args.selection.read_text())
    if not selection["selected"]:
        raise RuntimeError("development selected no state-specific arm")
    selected = parse_config(selection["selected"])
    control_rank = 0 if selected[0] in {"C1", "C2"} else selected[1]
    configs = (("C0", 4, .1, 300), selected,
               ("C4", control_rank, .1, 300), ("C5", 4, .1, 300))
    args.root.mkdir(parents=True, exist_ok=True)
    rows = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, args.root, config, init, stream)
                   for config in configs for init, stream in SEEDS]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print("COMPLETED", row["config"], row["run"], flush=True)
    rows.sort(key=lambda x: (x["config"], x["run"]))
    (args.root / "training_summaries.json").write_text(json.dumps(rows, indent=2))
    (args.root / "formal_protocol.json").write_text(json.dumps({
        "selection_file": str(args.selection), "selected": selection["selected"],
        "control_rank": control_rank, "configs": [tag(x) for x in configs],
        "seeds": SEEDS, "eval_seed": 31601, "jobs": args.jobs,
        "objective": "original observed-only consequence CE"}, indent=2))
    print("COMPLETE_FORMAL", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=12)
    main(p.parse_args())
