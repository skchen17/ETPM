#!/usr/bin/env python3
"""Select one equal-budget LR from completed two-seed development cells."""

from __future__ import annotations

import json
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT/"configs/stage1_6.yaml").read_text())
DEV = ROOT/"results/stage1_6/raw"/CONFIG["protocol"]["development_run_id"]


def main() -> None:
    scores = {}
    for lr in CONFIG["training"]["learning_rate_candidates"]:
        items = []
        for seed in CONFIG["training"]["development_seeds"]:
            for arm in CONFIG["training"]["arms"]:
                path = DEV/f"{arm}-seed{seed}-lr{lr:g}"/"summary.json"
                if not path.exists():
                    raise RuntimeError(f"missing development cell: {path}")
                summary = json.loads(path.read_text())
                condition = "oracle" if arm == "oracle" else "learned"
                items.append({"arm":arm,"seed":seed,"CE":summary["final_CE"][condition]})
        scores[str(lr)]={"mean_CE":sum(row["CE"] for row in items)/len(items),"cells":items}
    selected=min(CONFIG["training"]["learning_rate_candidates"],
                 key=lambda rate:(scores[str(rate)]["mean_CE"],rate))
    output={"selected_learning_rate":selected,"rule":"minimum equal-arm equal-seed heldout CE",
            "development_scores":scores}
    target=ROOT/"results/stage1_6/processed"/CONFIG["protocol"]["development_run_id"]
    target.mkdir(parents=True,exist_ok=True)
    (target/"lr_selection.json").write_text(json.dumps(output,indent=2))
    print(json.dumps(output))


if __name__=="__main__":
    main()
