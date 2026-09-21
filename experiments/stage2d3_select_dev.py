"""Select the shortest effective paired-action warmup using development runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch

from stage2d_evaluate import load_model
from etrcm.stage2d1.engine import classify, health_audit


def main(args):
    rows = []
    for run in sorted(args.checkpoints.iterdir()):
        if not (run / "checkpoint_1500.pt").exists():
            continue
        match = re.search(r"C1_w(\d+)_", run.name)
        if not match:
            continue
        torch.set_num_threads(1)
        model, _ = load_model(run / "checkpoint_1500.pt", "cpu")
        health = health_audit(model, args.eval_seed, reps=args.reps)
        rows.append({"run": run.name, "window": int(match.group(1)),
                     "class": classify(health), "health": health})
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["window"]].append(row)
    counts = {str(w): dict(Counter(x["class"] for x in items)) for w, items in sorted(grouped.items())}
    # Development only: maximize healthy count, then choose the shortest window.
    selected = min(grouped, key=lambda w: (-sum(x["class"] == "healthy" for x in grouped[w]), w))
    payload = {"rows": rows, "counts": counts, "selected_window": selected,
               "selection_rule": "highest development healthy count, shortest tie"}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"counts": counts, "selected_window": selected}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--eval-seed", type=int, required=True); p.add_argument("--reps", type=int, default=16)
    p.add_argument("--out", type=Path, required=True); main(p.parse_args())
