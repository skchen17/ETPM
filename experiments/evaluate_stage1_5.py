"""Immutable per-experiment formal Stage 1.5 evaluation shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from etrcm.stage1_4.evaluation import evaluate_A, evaluate_B
from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_5.capacity import evaluate_associative_capacity, evaluate_capacity_prediction
from etrcm.stage1_5.evaluation import (
    evaluate_anatomy, evaluate_oracle, evaluate_perturbations,
    evaluate_read_mediation, evaluate_stability,
)
from etrcm.stage1_5.model import AnatomicalETRCM
from etrcm.stage1_5.probes import evaluate_observability, evaluate_timescales


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = (
    "baseline", "stability", "perturbation", "timescale", "capacity",
    "anatomy", "observability", "mediation", "oracle",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(variant: str, seed: int, device: torch.device,
               config: dict, selection: dict) -> tuple[AnatomicalETRCM, Path]:
    if variant not in selection["learning_rates"]:
        raise ValueError(f"unregistered variant {variant}")
    base = variant.split("_h")[0]
    model_config = dict(config["model"])
    if variant != base:
        h_text, m_text = variant.rsplit("_h", 1)[1].split("_m")
        model_config.update(hidden_dim=int(h_text), key_dim=int(m_text), value_dim=int(m_text))
    lr = selection["learning_rates"][variant]
    checkpoint = ROOT / "results/stage1_5" / config["protocol"]["formal_run_id"] / (
        f"{variant}_lr{lr:g}_seed{seed}/checkpoint.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model = AnatomicalETRCM(Stage14Config.from_mapping({"model": model_config}), base).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_path = ROOT / "configs/stage1_5.yaml"
    selection_path = ROOT / "configs/stage1_5_selected.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if args.seed not in config["training"]["formal_seeds"]:
        raise ValueError("formal evaluation seed outside split")
    if selection["protocol_sha256"] != digest(config_path):
        raise ValueError("selected development config hash mismatch")
    if args.variant != "B5_separate" and args.experiment not in ("baseline", "capacity"):
        raise ValueError("anatomy/routing evaluation is primary B5 only")
    if args.experiment == "capacity" and not args.variant.startswith("B5_separate"):
        raise ValueError("capacity grid is B5 only")
    if args.experiment == "baseline" and args.variant.split("_h")[0] not in config["training"]["base_models"]:
        raise ValueError("unknown baseline")
    device = torch.device(args.device)
    model, checkpoint = load_model(args.variant, args.seed, device, config, selection)
    run_id = config["protocol"]["formal_run_id"]
    torch.manual_seed(args.seed * 10000 + 1515)
    if args.experiment == "baseline":
        rows = evaluate_A(model, seed=args.seed, run_id=run_id, device=device, batch=32)
        rows += evaluate_B(model, seed=args.seed, run_id=run_id, device=device,
                           gap_counts=(128, 512, 2048), batch=8)
    elif args.experiment == "stability":
        rows = evaluate_stability(model, seed=args.seed, run_id=run_id, device=device)
    elif args.experiment == "perturbation":
        rows = evaluate_perturbations(model, seed=args.seed, run_id=run_id, device=device)
    elif args.experiment == "timescale":
        rows = evaluate_timescales(
            model, seed=args.seed, run_id=run_id, device=device,
            train_episodes=config["evaluation"]["history_probe_train_episodes"],
            test_episodes=config["evaluation"]["history_probe_test_episodes"],
            ridge=config["evaluation"]["history_probe_ridge"],
        )
    elif args.experiment == "capacity":
        rows = evaluate_associative_capacity(model, seed=args.seed, run_id=run_id, device=device)
        rows += evaluate_capacity_prediction(model, seed=args.seed, run_id=run_id, device=device)
    elif args.experiment == "anatomy":
        rows = evaluate_anatomy(model, seed=args.seed, run_id=run_id, device=device)
    elif args.experiment == "observability":
        rows = evaluate_observability(model, seed=args.seed, run_id=run_id, device=device)
    elif args.experiment == "mediation":
        rows = evaluate_read_mediation(model, seed=args.seed, run_id=run_id, device=device)
    else:
        rows = evaluate_oracle(model, seed=args.seed, run_id=run_id, device=device)
    if not rows:
        raise ValueError("empty formal evaluation shard")
    frame = pd.DataFrame(rows)
    frame["variant"] = args.variant
    numeric = frame.select_dtypes(include=["number"])
    if np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).any():
        raise FloatingPointError("infinite numeric evaluation value")
    out = ROOT / "results/stage1_5" / run_id / "evaluation" / args.experiment / f"{args.variant}_seed{args.seed}"
    if out.exists():
        raise FileExistsError(f"immutable evaluation shard already exists: {out}")
    out.mkdir(parents=True)
    record_path = out / "records.parquet"
    frame.to_parquet(record_path, index=False)
    source_files = (
        "src/etrcm/stage1_5/model.py", "src/etrcm/stage1_5/evaluation.py",
        "src/etrcm/stage1_5/probes.py", "src/etrcm/stage1_5/capacity.py",
        "src/etrcm/stage1_5/routing.py", "experiments/evaluate_stage1_5.py",
    )
    summary = {
        "run_id": run_id, "experiment": args.experiment, "variant": args.variant,
        "seed": args.seed, "row_count": len(frame),
        "checkpoint_sha256": digest(checkpoint), "record_sha256": digest(record_path),
        "config_sha256": digest(config_path), "selection_sha256": digest(selection_path),
        "source_sha256": {name: digest(ROOT / name) for name in source_files},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"experiment": args.experiment, "variant": args.variant,
                      "seed": args.seed, "rows": len(frame)}))


if __name__ == "__main__":
    main()
