"""Freeze one state-specific compatibility arm from a two-seed fixed grid."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


ORDER = ("C1", "C2", "C3")


def simplicity(config):
    arm = config.split("_")[0]
    if arm == "C1":
        ratio = float(next(x[2:] for x in config.split("_") if x.startswith("lr")))
        return (0, ratio)
    if arm == "C2":
        freeze = int(next(x[6:] for x in config.split("_") if x.startswith("freeze")))
        return (1, freeze)
    if arm == "C3":
        rank = int(next(x[1:] for x in config.split("_") if x.startswith("r")))
        return (2, rank)
    return (9, 0)


def main(args):
    groups = defaultdict(list)
    for path in sorted(args.evaluations.rglob("evaluation.json")):
        row = json.loads(path.read_text())
        groups[path.parent.parent.name].append(row)
    if len(groups) != 13 or any(len(rows) != 2 for rows in groups.values()):
        raise AssertionError("fixed 13-configuration x 2-seed grid incomplete")
    table = {}
    for config, rows in groups.items():
        table[config] = {"healthy": sum(r["class"] == "healthy" for r in rows),
                         "mean_IHA": statistics.mean(r["health"]["I_HA"] for r in rows),
                         "mean_abs_BS": statistics.mean(abs(r["health"]["BS"]) for r in rows),
                         "classes": [r["class"] for r in rows],
                         "runs": [r["run"] for r in rows]}
    eligible = [name for name, row in table.items()
                if name.split("_")[0] in ORDER and row["healthy"] == 2]
    if eligible:
        selected = min(eligible, key=simplicity)
        basis = "two-of-two healthy; simplest state-specific configuration"
    else:
        candidates = [name for name, row in table.items()
                      if name.split("_")[0] in ORDER and row["healthy"] >= 1]
        if candidates:
            selected = max(candidates, key=lambda name: (
                table[name]["healthy"], table[name]["mean_IHA"],
                table[name]["mean_abs_BS"], -simplicity(name)[0], -simplicity(name)[1]))
            basis = "no two-of-two; highest development healthy count, then IHA/BS"
        else:
            selected = None
            basis = "no state-specific development configuration healthy; stop before formal"
    payload = {"selected": selected, "selection_basis": basis,
               "frozen_rule": "2/2 healthy -> simplest C1 then C2 then C3; else 1/2 best; else stop",
               "configurations": table,
               "state_specific_only": True,
               "same_fixed_seeds_per_configuration": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"selected": selected, "basis": basis,
                      "healthy_counts": {k: v["healthy"] for k, v in table.items()}}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--evaluations", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
