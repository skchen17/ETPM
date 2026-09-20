"""Read-only early G31 authorization check from all eight immutable shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = "stage1_5-formal-v1"


def main() -> None:
    shards = []
    for seed in range(8501, 8509):
        folder = ROOT / "results/stage1_5" / RUN / "evaluation/oracle" / f"B5_separate_seed{seed}"
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        path = folder / "records.parquet"
        if summary["record_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"oracle shard hash mismatch: {seed}")
        shards.append(pd.read_parquet(path))
    frame = pd.concat(shards, ignore_index=True)
    primary = frame[frame.internal_tick.eq(4) & frame.distractor_count.isin([512, 2048])]
    mean = primary.groupby(["seed", "intervention_condition"]).future_CE.mean().unstack()
    results = {}
    for comparator in ("zero", "random", "shuffled", "learned"):
        difference = mean[comparator] - mean["oracle_static"]
        results[comparator] = {
            "mean_CE_advantage": float(difference.mean()),
            "positive_seeds": int((difference > 0).sum()),
            "per_seed": {str(seed): float(value) for seed, value in difference.items()},
        }
    authorized = all(
        item["mean_CE_advantage"] >= 0.01 and item["positive_seeds"] >= 6
        for item in results.values()
    )
    print(json.dumps({"G31": "PASS" if authorized else "FAIL",
                      "B5_finite_advantage_authorized": authorized,
                      "comparators": results}, indent=2))


if __name__ == "__main__":
    main()
