"""Held-out evaluation of C0–C5 checkpoints with frozen labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from etrcm.stage2d1.engine import health_audit
from etrcm.stage2d4.model import classify
from etrcm.stage2d7.compat import CompatibilityModel


def one(path, out, reps=16):
    torch.set_num_threads(1)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    arm = checkpoint.get("curriculum", checkpoint["arm"])
    rank = checkpoint.get("rank", 4)
    model = CompatibilityModel(arm, rank=rank)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    health = health_audit(model, checkpoint["eval_seed"], reps=reps)
    payload = {"run": path.parent.name, "checkpoint": str(path),
               "arm": arm, "rank": rank,
               "lr_ratio": checkpoint.get("compatibility_lr_ratio"),
               "freeze_step": checkpoint.get("progressive_freeze_step"),
               "init_seed": checkpoint["init_seed"],
               "data_seed": checkpoint["data_seed"],
               "eval_seed": checkpoint["eval_seed"],
               "health": health, "class": classify(health),
               "memory_law_change": checkpoint.get("memory_law_change", "none"),
               "no_latent_or_correct_action_labels": not any(checkpoint.get(k, False)
                   for k in ("latent_z_input", "correct_action_label", "memory_label"))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload))
    print(json.dumps({"run": payload["run"], "arm": payload["arm"],
                      "class": payload["class"], "IHA": health["I_HA"],
                      "BS": health["BS"]}), flush=True)
    return payload


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=16)
    args = p.parse_args()
    one(args.checkpoint, args.out, args.reps)
