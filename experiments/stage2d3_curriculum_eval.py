"""Formal G84/G85 evaluation with zero scaffold at evaluation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from stage2d_evaluate import load_model
from etrcm.stage2d1.engine import classify, health_audit
from etrcm.stage2d2.geometry import phase_states
from etrcm.stage2d3.interaction import anatomy, destroy_native_interaction


def arm_from_name(name):
    return name.split("_", 1)[0]


@torch.no_grad()
def evaluate(run: Path, eval_seed: int, reps: int):
    model, checkpoint = load_model(run / "checkpoint_1500.pt", "cpu")
    health = health_audit(model, eval_seed, reps=reps); label = classify(health)
    state = phase_states(model, eval_seed, reps)["post"]
    anat = anatomy(model, state, eval_seed, reps)
    item = {"run": run.name, "arm": arm_from_name(run.name), "class": label,
            "health": health, "fusion_post": anat["layers"]["fusion_post"],
            "evaluation_scaffold": False, "checkpoint_auxiliary_weight": checkpoint.get("auxiliary_weight", 0.)}
    if item["arm"] == "C1" and label == "healthy":
        item["destruction"] = {mode: destroy_native_interaction(model, state, eval_seed, reps, mode)
                               for mode in ("remove", "reverse", "state_control")}
        item["native"] = anat["output"]
    return item


def main(args):
    runs = sorted(p for p in args.checkpoints.iterdir() if p.is_dir() and (p / "checkpoint_1500.pt").exists())
    rows = [evaluate(run, args.eval_seed, args.reps) for run in runs]
    counts = {arm: dict(Counter(x["class"] for x in rows if x["arm"] == arm)) for arm in ("C0", "C1", "C2", "C3")}
    healthy = {arm: counts[arm].get("healthy", 0) for arm in counts}
    g84 = healthy["C1"] >= 6 and all(healthy["C1"] > healthy[c] for c in ("C0", "C2", "C3"))
    preservation = 0
    for row in rows:
        if row["arm"] != "C1" or row["class"] != "healthy":
            continue
        d = row["destruction"]
        native_i, native_bs = row["native"]["I_HA"], abs(row["native"]["BS"])
        if (row["fusion_post"]["normalized_interaction"] >= .05 and
            d["remove"]["I_HA"] < .5 * native_i and abs(d["remove"]["BS"]) < .5 * native_bs and
            d["remove"]["I_HA"] < d["state_control"]["I_HA"]):
            preservation += 1
    g85 = preservation >= 6 if g84 else None
    payload = {"rows": rows, "healthy_counts": healthy,
               "G84": {"pass": g84, "required": 6, "counts": healthy,
                       "control_requirement": "C1 strictly exceeds C0/C2/C3"},
               "G85": {"pass": g85, "status": ("PASS" if g85 else "FAIL") if g84 else "NOT_RUN_BY_PROTOCOL",
                       "reason": None if g84 else "G84 did not establish a successful curriculum",
                       "diagnostic_count": preservation, "required": 6, "layer": "fusion_post"}}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(payload))
    print(json.dumps({"G84": payload["G84"], "G85": payload["G85"]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--eval-seed", type=int, required=True); p.add_argument("--reps", type=int, default=16)
    p.add_argument("--out", type=Path, required=True); main(p.parse_args())
