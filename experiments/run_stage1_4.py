"""Train one deterministic Stage 1.4 model/seed/LR shard."""

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "formal"), required=True)
    parser.add_argument("--model", choices=tuple(PredictiveETRCM.MODES), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    config_path = ROOT / "configs/stage1_4.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_seeds = config["training"][f"{args.mode}_seeds"]
    if args.seed not in expected_seeds:
        raise ValueError("seed is outside frozen split")
    if args.lr not in config["training"]["candidate_learning_rates"]:
        raise ValueError("LR outside frozen development grid")
    if args.mode == "formal":
        selection = json.loads((ROOT / "configs/stage1_4_selected.json").read_text(encoding="utf-8"))
        if args.lr != selection["learning_rates"][args.model]:
            raise ValueError("formal LR does not match frozen development selection")
    torch.manual_seed(args.seed)
    model = PredictiveETRCM(Stage14Config.from_mapping(config), args.model)
    device = torch.device(args.device)
    result = train_model(
        model, steps=config["training"][f"{args.mode}_steps"],
        batch_size=config["training"]["batch_size"], learning_rate=args.lr,
        seed=args.seed, device=device, length=config["world"]["training_episode_length"],
        weight_decay=config["training"]["weight_decay"],
        gradient_clip=config["training"]["gradient_clip"],
    )
    shard = f"{args.model}_lr{args.lr:g}_seed{args.seed}"
    out = ROOT / "results/stage1_4" / args.run_id / shard
    if out.exists():
        raise FileExistsError(f"immutable shard exists: {out}")
    out.mkdir(parents=True)
    checkpoint = out / "checkpoint.pt"
    torch.save(model.state_dict(), checkpoint)
    pd.DataFrame(result.logs).to_parquet(out / "training.parquet", index=False)
    summary = {
        "mode": args.mode, "run_id": args.run_id, "model": args.model,
        "seed": args.seed, "learning_rate": args.lr,
        "validation_loss": result.validation_loss,
        "validation_by_family": result.validation_by_family,
        "parameter_count": model.trainable_parameters(),
        "persistent_state_bytes": model.persistent_state_bytes(),
        "training_steps": config["training"][f"{args.mode}_steps"],
        "batch_size": config["training"]["batch_size"],
        "config_sha256": sha256(config_path),
        "checkpoint_sha256": sha256(checkpoint),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
