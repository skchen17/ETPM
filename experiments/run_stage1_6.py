#!/usr/bin/env python3
"""Independent Stage 1.6 development/formal cells; append-only output tree."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import pandas as pd
import torch
import yaml

from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_6.runner import (ARMS, CONDITIONS, choose_oracle, gradient_diagnostics,
                                    make_model, oracle_probability, run_episode)
from etrcm.stage1_6.world import generate_world


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/stage1_6.yaml"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(phase: str, seed: int, step: int, *, eval_: bool = False) -> int:
    base = 16_000_000 if eval_ else 1_600_000
    return base + seed * 10_000 + (0 if eval_ else step)


@torch.no_grad()
def evaluate(model, *, phase: str, arm: str, seed: int, step: int,
             batch_size: int, device: torch.device, parameter_count: int,
             state_bytes: int, run_id: str) -> list[dict]:
    model.eval()
    world = generate_world(batch=batch_size, seed=_seed(phase, seed, step, eval_=True), device=device)
    conditions = CONDITIONS if arm in {"learned", "oracle", "curriculum"} else (
        "learned", "M_lesion", "F_lesion")
    per_condition = {}
    for condition in conditions:
        _, d = run_episode(model, world, condition=condition,
                           random_seed=40_000 + seed * 31 + step)
        per_condition[condition] = {key: value.detach().cpu().tolist() for key, value in d.items()}
    rows = []
    probability = (oracle_probability(min(step, 2999), 3000) if arm == "curriculum" else
                   (1.0 if arm == "oracle" else 0.0))
    for episode in range(batch_size):
        learned = per_condition["learned"]["ce"][episode]
        zero = per_condition.get("zero", {}).get("ce", [None] * batch_size)[episode]
        oracle = per_condition.get("oracle", {}).get("ce", [None] * batch_size)[episode]
        m_lesion = per_condition["M_lesion"]["ce"][episode]
        f_lesion = per_condition["F_lesion"]["ce"][episode]
        for condition, metrics in per_condition.items():
            row = {
                "run_id": run_id, "phase": phase, "architecture": ARMS[arm], "seed": seed,
                "training_arm": arm, "training_step": step,
                "curriculum_oracle_probability": probability,
                "world_family": "composed_h_scrub_v1", "episode": episode,
                "H_scrub": True, "condition": condition,
                "future_CE": metrics["ce"][episode], "accuracy": metrics["correct"][episode],
                "M_lesion_CE": m_lesion, "F_lesion_CE": f_lesion,
                "MemoryBenefit": None if zero is None else zero - learned,
                "OracleBenefit": None if zero is None else zero - oracle,
                "D_M": m_lesion - learned,
                "read_norm": metrics["read_norm"][episode],
                "read_gate_slow": metrics["read_gate_slow"][episode],
                "candidate_update_norm": metrics["candidate_update_norm"][episode],
                "H_norm": metrics["H_norm"][episode],
                "F_norm": metrics["F_norm"][episode],
                "M_norm": metrics["M_norm"][episode],
                "oracle_norm": metrics["oracle_norm"][episode],
                "access": metrics["access"][episode],
                "gradient_diagnostic": None,
                "parameter_count": parameter_count, "state_bytes": state_bytes,
                "compute_budget_transitions": step * 13 * 32,
                "source_seed": _seed(phase, seed, step, eval_=True),
            }
            rows.append(row)
    return rows


def run_cell(*, phase: str, arm: str, seed: int, learning_rate: float,
             device: torch.device, force: bool = False) -> dict:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    train = config["training"]
    total_steps = int(train["development_steps"] if phase == "development" else train["formal_steps"])
    checkpoints = ({0, total_steps} if phase == "development" else set(train["checkpoints"]))
    model_config = Stage14Config.from_mapping(config)
    model = make_model(model_config, arm, seed=seed, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=float(train["weight_decay"]))
    run_id = config["protocol"][f"{phase}_run_id"]
    cell_id = f"{arm}-seed{seed}-lr{learning_rate:g}"
    cell_dir = ROOT / "results/stage1_6/raw" / run_id / cell_id
    if cell_dir.exists() and any(cell_dir.iterdir()) and not force:
        raise FileExistsError(f"cell already exists: {cell_dir}")
    cell_dir.mkdir(parents=True, exist_ok=True)
    count = sum(p.numel() for p in model.parameters())
    state = model.initial_state(int(train["batch_size"]), device=device)
    state_bytes = (state.H.numel() + state.F.numel() + state.M.numel()) * state.H.element_size() // int(train["batch_size"])
    logs, records, hashes = [], [], {}
    started = time.monotonic()
    for step in range(total_steps + 1):
        if step in checkpoints:
            checkpoint = cell_dir / f"checkpoint_{step:04d}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "arm": arm, "seed": seed,
                        "learning_rate": learning_rate,
                        "config_sha256": sha256(CONFIG_PATH)}, checkpoint)
            hashes[checkpoint.name] = sha256(checkpoint)
            records.extend(evaluate(model, phase=phase, arm=arm, seed=seed, step=step,
                                    batch_size=int(config["evaluation"]["episodes_per_seed"]),
                                    device=device, parameter_count=count,
                                    state_bytes=state_bytes, run_id=run_id))
            world = generate_world(batch=32, seed=_seed(phase, seed, step, eval_=True) + 97,
                                   device=device)
            condition = "oracle" if arm == "oracle" else "learned"
            grad = gradient_diagnostics(model, world, condition=condition,
                                        random_seed=seed + step)
            for row in records:
                if row["training_step"] == step:
                    row["gradient_diagnostic"] = grad["memory_grad_norm"]
                    row["core_gradient_diagnostic"] = grad["core_grad_norm"]
            print(json.dumps({"cell": cell_id, "checkpoint": step,
                              "elapsed_s": round(time.monotonic()-started, 1)}), flush=True)
        if step == total_steps:
            break
        world = generate_world(batch=int(train["batch_size"]),
                               seed=_seed(phase, seed, step), device=device,
                               early_exposures=int(config["world"]["early_exposures"]),
                               distractors=int(config["world"]["distractors"]))
        condition = ("oracle" if arm == "oracle" else
                     ("oracle" if arm == "curriculum" and choose_oracle(step, total_steps, seed=seed) else "learned"))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = run_episode(model, world, condition=condition, random_seed=seed+step)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite training loss at {arm}/{seed}/{step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(train["gradient_clip"]))
        optimizer.step()
        if step % 50 == 0 or step == total_steps - 1:
            logs.append({"run_id": run_id, "arm": arm, "seed": seed, "step": step,
                         "training_condition": condition, "loss": float(loss.detach()),
                         "gradient_norm": float(grad_norm.detach()),
                         "curriculum_oracle_probability": oracle_probability(step, total_steps)
                         if arm == "curriculum" else (1.0 if arm == "oracle" else 0.0),
                         "batch": int(train["batch_size"]), "transitions_per_episode": 13,
                         "learning_rate": learning_rate})
    raw_path = cell_dir / "interventions.parquet"
    pd.DataFrame.from_records(records).to_parquet(raw_path, index=False)
    hashes[raw_path.name] = sha256(raw_path)
    log_path = cell_dir / "train_log.parquet"
    pd.DataFrame.from_records(logs).to_parquet(log_path, index=False)
    hashes[log_path.name] = sha256(log_path)
    final = pd.DataFrame.from_records(records)
    last = final.loc[final.training_step.eq(total_steps)]
    summary = {"run_id": run_id, "phase": phase, "arm": arm, "seed": seed,
               "steps": total_steps, "learning_rate": learning_rate,
               "parameter_count": count, "state_bytes": state_bytes,
               "elapsed_seconds": time.monotonic()-started,
               "final_CE": last.groupby("condition").future_CE.mean().to_dict(),
               "hashes": hashes}
    summary_path = cell_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"cell_complete": cell_id, "elapsed_s": round(summary["elapsed_seconds"], 1)}), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "formal"), required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_cell(phase=args.phase, arm=args.arm, seed=args.seed,
             learning_rate=args.learning_rate, device=torch.device(args.device))


if __name__ == "__main__":
    main()
