"""Exploratory frequency audit; never changes frozen Stage 1.4 gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from etrcm.stage1_4.model import PredictiveETRCM, Stage14Config
from etrcm.stage1_4.world import generate_world


ROOT = Path(__file__).resolve().parents[1]
RUN = "stage1_4-formal-v1a1"


def incremental_rank_r2(data: pd.DataFrame) -> float:
    columns = ["exposure_count", "read_usage", "hard_access_count", "causal_usage", "retention"]
    ranked = data[columns].rank(pct=True).to_numpy(dtype=float)
    base = np.column_stack([np.ones(len(data)), ranked[:, :3]])
    full = np.column_stack([base, ranked[:, 3]])
    y = ranked[:, 4]
    pred_base, pred_full = np.zeros_like(y), np.zeros_like(y)
    for fold in range(4):
        test = np.arange(len(y)) % 4 == fold
        train = ~test
        pred_base[test] = base[test] @ np.linalg.lstsq(base[train], y[train], rcond=None)[0]
        pred_full[test] = full[test] @ np.linalg.lstsq(full[train], y[train], rcond=None)[0]
    sst = ((y - y.mean()) ** 2).sum()
    return float((((y - pred_base) ** 2).sum() - ((y - pred_full) ** 2).sum()) / sst)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/stage1_4_v1a1.yaml").read_text())
    selected = json.loads((ROOT / "configs/stage1_4_selected_v1a1.json").read_text())
    rows: list[dict] = []
    summaries: list[dict] = []
    with torch.no_grad():
        for seed in config["training"]["formal_seeds"]:
            torch.manual_seed(seed)
            model = PredictiveETRCM(Stage14Config.from_mapping(config), "B5_separate").eval()
            lr = selected["learning_rates"]["B5_separate"]
            checkpoint = ROOT / "results/stage1_4" / RUN / f"B5_separate_lr{lr:g}_seed{seed}/checkpoint.pt"
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            world = generate_world("long_gap_relation", batch=32, length=16,
                                   seed=seed * 100_000 + 40_001, gap=128)
            state = model.initial_state(32)
            key = model.target_key(world.target_key)
            hard = torch.zeros(32)
            soft = torch.zeros(32)
            for step in range(world.bridge_index):
                state, output = model.step(state, world.events[step])
                cosine = (output["q_M"] * key).sum(-1).abs()
                hard += cosine.ge(0.5).float()
                soft += cosine
            records = pd.read_parquet(
                ROOT / "results/stage1_4" / RUN / f"B5_separate_seed{seed}/records.parquet"
            )
            items = records.loc[records.experiment.eq("E_F")].sort_values("episode").reset_index(drop=True)
            if len(items) != 32 or not np.array_equal(items.episode.to_numpy(), np.arange(32)):
                raise ValueError("E/F item identities do not match replay")
            for index in range(32):
                rows.append({
                    "seed": seed, "episode": index,
                    "hard_access_count": float(hard[index]),
                    "soft_access_count": float(soft[index]),
                    "exposure_count": float(items.exposure_count.iloc[index]),
                    "read_usage": float(items.read_usage.iloc[index]),
                    "causal_usage": float(items.causal_usage.iloc[index]),
                    "retention": float(items.retention.iloc[index]),
                    "access_threshold": 0.5,
                    "analysis_status": "EXPLORATORY_NOT_GATE",
                })
            one = pd.DataFrame(rows[-32:])
            summaries.append({
                "seed": seed,
                "hard_count_retention_spearman": float(one.hard_access_count.corr(one.retention, method="spearman")),
                "soft_count_retention_spearman": float(one.soft_access_count.corr(one.retention, method="spearman")),
                "causal_retention_spearman": float(one.causal_usage.corr(one.retention, method="spearman")),
                "CU_incremental_R2_beyond_exposure_magnitude_count": incremental_rank_r2(one),
            })
    out = ROOT / "results/stage1_4/processed" / RUN
    if not out.is_dir() or (out / "read_frequency.parquet").exists():
        raise FileExistsError("processed output missing or frequency audit already exists")
    path = out / "read_frequency.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    summary = {
        "status": "EXPLORATORY_NOT_GATE",
        "hard_threshold": 0.5,
        "seed_count": 8, "item_count": len(rows),
        "seed_results": summaries,
        "mean_CU_incremental_R2": float(np.mean([x["CU_incremental_R2_beyond_exposure_magnitude_count"] for x in summaries])),
        "parquet_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "does_not_change_G26": True,
    }
    (out / "read_frequency.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
