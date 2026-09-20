"""Freeze per-variant development learning rates before formal training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    config_path = ROOT / "configs/stage1_5.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_id = config["protocol"]["development_run_id"]
    seeds = config["training"]["development_seeds"]
    rates = config["training"]["candidate_learning_rates"]
    variants = list(config["training"]["base_models"])
    variants += [
        f"B5_separate_h{cell['hidden_dim']}_m{cell['memory_dim']}"
        for cell in config["evaluation"]["capacity_grid"]
        if (cell["hidden_dim"], cell["memory_dim"]) != (64, 16)
    ]
    selected: dict[str, float] = {}
    details: dict[str, object] = {}
    source_hashes: dict[str, str] = {}
    for variant in variants:
        grid = []
        for lr in rates:
            losses = []
            family_losses = []
            for seed in seeds:
                summary_path = ROOT / "results/stage1_5" / run_id / f"{variant}_lr{lr:g}_seed{seed}/summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if summary["config_sha256"] != digest(config_path):
                    raise ValueError(f"config drift: {summary_path}")
                if summary["seed"] != seed or summary["variant"] != variant:
                    raise ValueError(f"shard identity mismatch: {summary_path}")
                source_hashes[str(summary_path.relative_to(ROOT))] = digest(summary_path)
                losses.append(float(summary["validation_loss"]))
                family_losses.append(summary["validation_by_family"])
            grid.append({"learning_rate": lr, "mean_equal_family_CE": float(np.mean(losses)),
                         "seed_CE": losses, "family_CE": family_losses})
        chosen = min(grid, key=lambda row: (row["mean_equal_family_CE"], row["learning_rate"]))
        selected[variant] = chosen["learning_rate"]
        details[variant] = grid
    out = ROOT / "configs/stage1_5_selected.json"
    if out.exists():
        raise FileExistsError(f"selection is immutable: {out}")
    payload = {
        "run_id": run_id, "protocol_sha256": digest(config_path),
        "development_seeds": seeds, "selection_rule": config["training"]["selection_rule"],
        "learning_rates": selected, "development_grid": details,
        "source_summary_sha256": source_hashes,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(selected, sort_keys=True))


if __name__ == "__main__":
    main()
