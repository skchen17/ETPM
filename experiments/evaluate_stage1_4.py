"""Development causal-selection and immutable per-seed formal evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from etrcm.stage1_4.evaluation import (
    evaluate_A, evaluate_B, evaluate_CD, evaluate_EF, evaluate_G, evaluate_safety,
)
from etrcm.stage1_4.model import PredictiveETRCM, Stage14Config


ROOT = Path(__file__).resolve().parents[1]


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank_regression_incremental(rows: list[dict[str, object]]) -> dict[str, float]:
    data = pd.DataFrame(rows)
    data = data.loc[data.experiment.eq("E_F")].reset_index(drop=True)
    if len(data) < 16:
        raise ValueError("insufficient causal-usage items")
    columns = ["exposure_count", "read_usage", "causal_usage", "retention"]
    ranked = data[columns].rank(pct=True).to_numpy(dtype=float)
    x_base = np.column_stack([np.ones(len(data)), ranked[:, 0], ranked[:, 1]])
    x_full = np.column_stack([x_base, ranked[:, 2]])
    y = ranked[:, 3]
    pred_base, pred_full = np.zeros_like(y), np.zeros_like(y)
    for fold in range(4):
        test = np.arange(len(data)) % 4 == fold
        train = ~test
        beta_base = np.linalg.lstsq(x_base[train], y[train], rcond=None)[0]
        beta_full = np.linalg.lstsq(x_full[train], y[train], rcond=None)[0]
        pred_base[test] = x_base[test] @ beta_base
        pred_full[test] = x_full[test] @ beta_full
    sst = float(((y - y.mean()) ** 2).sum())
    base_sse = float(((y - pred_base) ** 2).sum())
    full_sse = float(((y - pred_full) ** 2).sum())
    return {
        "exposure_retention_spearman": float(data.exposure_count.corr(data.retention, method="spearman")),
        "read_retention_spearman": float(data.read_usage.corr(data.retention, method="spearman")),
        "causal_retention_spearman": float(data.causal_usage.corr(data.retention, method="spearman")),
        "heldout_base_r2": 1 - base_sse / sst,
        "heldout_plus_causal_r2": 1 - full_sse / sst,
        "incremental_heldout_r2": (base_sse - full_sse) / sst,
        "item_count": int(len(data)),
    }


def load_model(
    *, mode: str, seed: int, source_run: str, device: torch.device,
) -> tuple[PredictiveETRCM, Path, dict]:
    config = yaml.safe_load((ROOT / "configs/stage1_4.yaml").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "configs/stage1_4_selected.json").read_text(encoding="utf-8"))
    lr = selection["learning_rates"][mode]
    shard = f"{mode}_lr{lr:g}_seed{seed}"
    checkpoint = ROOT / "results/stage1_4" / source_run / shard / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = PredictiveETRCM(Stage14Config.from_mapping(config), mode).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, checkpoint, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("dev-causal", "formal"), required=True)
    parser.add_argument("--model", choices=tuple(PredictiveETRCM.MODES), default="B5_separate")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, checkpoint, config = load_model(
        mode=args.model, seed=args.seed,
        source_run=config_run(args.phase), device=device,
    )
    run_id = (
        "stage1_4-development-causal-v1a1" if args.phase == "dev-causal"
        else config["protocol"]["formal_run_id"]
    )
    if args.phase == "dev-causal":
        if args.model != "B5_separate" or args.seed not in config["training"]["development_seeds"]:
            raise ValueError("development causal selection is B5 on frozen development seeds")
        rows = evaluate_EF(model, seed=args.seed, run_id=run_id, device=device)
    else:
        if args.seed not in config["training"]["formal_seeds"]:
            raise ValueError("formal seed outside frozen split")
        rows = evaluate_A(model, seed=args.seed, run_id=run_id, device=device)
        rows += evaluate_B(model, seed=args.seed, run_id=run_id, device=device)
        if args.model == "B5_separate":
            rows += evaluate_CD(model, seed=args.seed, run_id=run_id, device=device)
            causal_rows = evaluate_EF(model, seed=args.seed, run_id=run_id, device=device)
            rows += causal_rows
            causal_selection = json.loads(
                (ROOT / "configs/stage1_4_causal_selection.json").read_text(encoding="utf-8")
            )
            if causal_selection["experiment_G_authorized"]:
                rows += evaluate_G(
                    model, seed=args.seed, run_id=run_id, device=device,
                    causal_rows=causal_rows,
                )
            rows += evaluate_safety(model, seed=args.seed, run_id=run_id, device=device)
    if not rows:
        raise ValueError("empty evaluation")
    frame = pd.DataFrame(rows)
    finite = frame.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan)
    if frame.select_dtypes(include=["number"]).notna().sum().sum() != finite.notna().sum().sum():
        raise FloatingPointError("non-finite numeric evaluation record")
    out = ROOT / "results/stage1_4" / run_id / f"{args.model}_seed{args.seed}"
    if out.exists():
        raise FileExistsError(f"immutable evaluation shard exists: {out}")
    out.mkdir(parents=True)
    record_path = out / "records.parquet"
    frame.to_parquet(record_path, index=False)
    summary: dict[str, object] = {
        "phase": args.phase, "model": args.model, "seed": args.seed,
        "run_id": run_id, "record_count": len(frame),
        "experiments": frame.experiment.value_counts().to_dict(),
        "checkpoint_sha256": hash_file(checkpoint),
        "records_sha256": hash_file(record_path),
        "config_sha256": hash_file(ROOT / "configs/stage1_4.yaml"),
        "selection_sha256": hash_file(ROOT / "configs/stage1_4_selected.json"),
    }
    if args.phase == "dev-causal" or args.model == "B5_separate":
        summary["causal_regression"] = rank_regression_incremental(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


def config_run(phase: str) -> str:
    return "stage1_4-development-v1" if phase == "dev-causal" else "stage1_4-formal-v1"


if __name__ == "__main__":
    main()
