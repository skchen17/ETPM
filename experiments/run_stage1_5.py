"""One immutable, equal-budget Stage 1.5 train shard."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from etrcm.stage1_4.model import PredictiveETRCM, Stage14Config
from etrcm.stage1_4.training import train_model


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "formal"), required=True)
    parser.add_argument("--model", choices=tuple(PredictiveETRCM.MODES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--memory-dim", type=int)
    args = parser.parse_args()
    config_path = ROOT / "configs/stage1_5.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    training = config["training"]
    if args.seed not in training[f"{args.mode}_seeds"]:
        raise ValueError("seed outside frozen split")
    if args.lr not in training["candidate_learning_rates"]:
        raise ValueError("learning rate outside frozen grid")
    h = args.hidden_dim or config["model"]["hidden_dim"]
    m = args.memory_dim or config["model"]["key_dim"]
    if (h, m) != (64, 16):
        if args.model != "B5_separate" or {"hidden_dim": h, "memory_dim": m} not in config["evaluation"]["capacity_grid"]:
            raise ValueError("unregistered capacity cell")
    if args.mode == "formal":
        selection = json.loads((ROOT / "configs/stage1_5_selected.json").read_text(encoding="utf-8"))
        chosen = selection["learning_rates"][args.model if (h, m) == (64, 16) else f"B5_separate_h{h}_m{m}"]
        if args.lr != chosen:
            raise ValueError("formal learning rate differs from frozen development selection")
    run_id = config["protocol"][f"{args.mode}_run_id"]
    variant = args.model if (h, m) == (64, 16) else f"B5_separate_h{h}_m{m}"
    shard = f"{variant}_lr{args.lr:g}_seed{args.seed}"
    out = ROOT / "results/stage1_5" / run_id / shard
    if out.exists():
        raise FileExistsError(f"immutable shard already exists: {out}")
    model_config = dict(config["model"])
    model_config.update(hidden_dim=h, key_dim=m, value_dim=m)
    torch.manual_seed(args.seed)
    model = PredictiveETRCM(Stage14Config.from_mapping({"model": model_config}), args.model)
    result = train_model(
        model,
        steps=training[f"{args.mode}_steps"],
        batch_size=training["batch_size"],
        learning_rate=args.lr,
        seed=args.seed,
        device=torch.device(args.device),
        length=16,
        weight_decay=training["weight_decay"],
        gradient_clip=training["gradient_clip"],
    )
    out.mkdir(parents=True)
    checkpoint = out / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    pd.DataFrame(result.logs).to_parquet(out / "training.parquet", index=False)
    summary = {
        "mode": args.mode, "run_id": run_id, "model": args.model, "variant": variant,
        "seed": args.seed, "learning_rate": args.lr, "hidden_dim": h,
        "memory_dim": m, "validation_loss": result.validation_loss,
        "validation_by_family": result.validation_by_family,
        "parameter_count": model.trainable_parameters(),
        "persistent_state_bytes": model.persistent_state_bytes(),
        "training_steps": training[f"{args.mode}_steps"],
        "config_sha256": digest(config_path), "checkpoint_sha256": digest(checkpoint),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"variant": variant, "seed": args.seed, "lr": args.lr,
                      "validation_loss": result.validation_loss}, sort_keys=True))


if __name__ == "__main__":
    main()
