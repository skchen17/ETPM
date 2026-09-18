#!/usr/bin/env python3
"""Train, select thresholds, and run frozen ET-RCM Stage-1.3 jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etrcm.stage1_3.baselines import MODEL_NAMES, build_model
from etrcm.stage1_3.evaluation import (
    arbitration_sweep,
    cross_time_association,
    evidence_accumulation,
    interleaved_streaming,
    pattern_discovery,
    revision_after_expression,
    self_output_audit,
    silence_under_noise,
    thought_driven_persistence,
)
from etrcm.stage1_3.model import Stage13Config
from etrcm.stage1_3.training import train_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze() -> None:
    freeze = json.loads((ROOT / "artifacts/stage1_3_protocol.freeze.json").read_text())
    for relative, expected in freeze["files"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"frozen file changed: {relative}: {actual} != {expected}")
    for line in (ROOT / "artifacts/stage1_3_prior_assets.sha256").read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"prior immutable asset changed: {relative}")


def config() -> dict:
    return yaml.safe_load((ROOT / "configs/stage1_3.yaml").read_text())


def train_one(model_name: str, seed: int, lr: float, steps: int, device: torch.device):
    cfg = config()
    torch.manual_seed(seed)
    model = build_model(model_name, Stage13Config.from_mapping(cfg))
    result = train_model(
        model,
        steps=steps,
        batch_size=int(cfg["training"]["batch_size"]),
        learning_rate=lr,
        seed=seed,
        device=device,
        stream_length=int(cfg["evaluation"]["evidence_stream_length"]),
        sufficient_count=int(cfg["evaluation"]["sufficient_evidence_count"]),
        insufficient_count=int(cfg["evaluation"]["insufficient_evidence_count"]),
        emit_weight=float(cfg["training"]["emit_loss_weight"]),
        content_weight=float(cfg["training"]["content_loss_weight"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        gradient_clip=float(cfg["training"]["gradient_clip"]),
    )
    return model, result


def development_job(args, cfg: dict, device: torch.device) -> None:
    run_id = cfg["protocol"]["development_run_id"]
    raw = ROOT / "results/stage1_3/raw" / run_id
    checkpoints = ROOT / "artifacts/stage1_3" / run_id
    raw.mkdir(parents=True, exist_ok=True); checkpoints.mkdir(parents=True, exist_ok=True)
    model, result = train_one(args.model, args.seed, args.lr, int(cfg["training"]["development_steps"]), device)
    tag = f"{args.model}__lr{args.lr:g}__seed{args.seed}"
    torch.save({"model": model.state_dict(), "model_name": args.model, "seed": args.seed, "lr": args.lr}, checkpoints / f"{tag}.pt")
    pd.DataFrame([{**row, "model": args.model, "seed": args.seed, "lr": args.lr} for row in result.logs]).to_parquet(raw / f"{tag}__training.parquet", index=False)
    (raw / f"{tag}__summary.json").write_text(json.dumps({"model": args.model, "seed": args.seed, "lr": args.lr, "validation_loss": result.validation_loss}, indent=2))
    print(json.dumps({"tag": tag, "validation_loss": result.validation_loss}))


def select_lr(cfg: dict) -> None:
    run_id = cfg["protocol"]["development_run_id"]
    raw = ROOT / "results/stage1_3/raw" / run_id
    rows = [json.loads(path.read_text()) for path in raw.glob("*__summary.json")]
    frame = pd.DataFrame(rows)
    expected = len(MODEL_NAMES) * len(cfg["training"]["candidate_learning_rates"]) * len(cfg["training"]["development_seeds"])
    if len(frame) != expected:
        raise RuntimeError(f"development summaries {len(frame)} != {expected}")
    means = frame.groupby(["model", "lr"], as_index=False).validation_loss.mean()
    selected = {}
    for model_name, arm in means.groupby("model"):
        best = arm.sort_values(["validation_loss", "lr"], ascending=[True, True]).iloc[0]
        selected[model_name] = float(best.lr)
    payload = {"protocol": "stage1.3-v1", "development_run_id": run_id, "selected_learning_rates": selected, "selection_rule": "minimum mean development validation loss; exact tie chooses lower LR", "formal_outcomes_visible": False}
    (ROOT / "configs/stage1_3_lr_selected.json").write_text(json.dumps(payload, indent=2))
    means.to_parquet(raw / "learning_rate_selection.parquet", index=False)
    print(json.dumps(payload, indent=2))


def load_checkpoint(model_name: str, seed: int, lr: float, cfg: dict, device: torch.device):
    run_id = cfg["protocol"]["development_run_id"]
    path = ROOT / "artifacts/stage1_3" / run_id / f"{model_name}__lr{lr:g}__seed{seed}.pt"
    model = build_model(model_name, Stage13Config.from_mapping(cfg)).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)["model"])
    model.eval()
    return model


def threshold_job(args, cfg: dict, device: torch.device) -> None:
    selection = json.loads((ROOT / "configs/stage1_3_lr_selected.json").read_text())
    lr = float(selection["selected_learning_rates"][args.model])
    model = load_checkpoint(args.model, args.seed, lr, cfg, device)
    records, _ = evidence_accumulation(
        model, model_name=args.model, seed=args.seed, threshold=1.1, episodes=128,
        length=int(cfg["evaluation"]["evidence_stream_length"]),
        sufficient_count=int(cfg["evaluation"]["sufficient_evidence_count"]),
        insufficient_count=int(cfg["evaluation"]["insufficient_evidence_count"]), device=device,
    )
    noise, _ = silence_under_noise(
        model, model_name=args.model, seed=args.seed, threshold=1.1, length=256,
        streams=16, device=device,
    )
    run_id = cfg["protocol"]["development_run_id"]
    raw = ROOT / "results/stage1_3/raw" / run_id
    pd.DataFrame(records).to_parquet(raw / f"{args.model}__seed{args.seed}__threshold_expression.parquet", index=False)
    pd.DataFrame(noise).to_parquet(raw / f"{args.model}__seed{args.seed}__threshold_noise.parquet", index=False)
    print(json.dumps({"model": args.model, "seed": args.seed, "lr": lr, "expression_rows": len(records), "noise_rows": len(noise)}))


def freeze_selection(cfg: dict) -> None:
    lr_selection = json.loads((ROOT / "configs/stage1_3_lr_selected.json").read_text())
    run_id = cfg["protocol"]["development_run_id"]
    raw = ROOT / "results/stage1_3/raw" / run_id
    thresholds = cfg["threshold_selection"]["candidates"]
    selected_thresholds, sweep_rows = {}, []
    for model_name in MODEL_NAMES:
        expression = pd.concat([pd.read_parquet(raw / f"{model_name}__seed{seed}__threshold_expression.parquet") for seed in cfg["training"]["development_seeds"]], ignore_index=True)
        noise = pd.concat([pd.read_parquet(raw / f"{model_name}__seed{seed}__threshold_noise.parquet") for seed in cfg["training"]["development_seeds"]], ignore_index=True)
        for threshold in thresholds:
            predicted = expression.expression_score.ge(threshold)
            positive = expression.enough_evidence.eq(1)
            content_correct = expression.accuracy.eq(1)
            tp = int((predicted & positive & content_correct).sum())
            fp = int((predicted & ~(positive & content_correct)).sum())
            fn = int((~predicted & positive).sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            noise_false = float(noise.expression_score.ge(threshold).mean())
            sweep_rows.append({"model": model_name, "threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "noise_false_emission_rate": noise_false})
        arm = pd.DataFrame(sweep_rows)
        arm = arm[arm.model.eq(model_name)]
        feasible = arm[(arm.precision >= cfg["threshold_selection"]["minimum_precision"]) & (arm.noise_false_emission_rate <= cfg["threshold_selection"]["maximum_noise_false_emission_rate"])]
        candidates = feasible if len(feasible) else arm
        best = candidates.sort_values(["f1", "precision", "threshold"], ascending=[False, False, True]).iloc[0]
        selected_thresholds[model_name] = float(best.threshold)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_parquet(raw / "threshold_sweep.parquet", index=False)
    payload = {
        "protocol": "stage1.3-v1",
        "development_run_id": run_id,
        "selected_learning_rates": lr_selection["selected_learning_rates"],
        "selected_thresholds": selected_thresholds,
        "threshold_rule": cfg["threshold_selection"]["rule"],
        "development_seeds": cfg["training"]["development_seeds"],
        "formal_outcomes_visible": False,
        "source_hashes": {
            "learning_rate_selection.parquet": sha256(raw / "learning_rate_selection.parquet"),
            "threshold_sweep.parquet": sha256(raw / "threshold_sweep.parquet"),
        },
    }
    (ROOT / "configs/stage1_3_selected.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


def _merge(records, samples, result):
    new_records, new_samples = result
    records.extend(new_records); samples.update(new_samples)


def formal_job(args, cfg: dict, device: torch.device) -> None:
    selected = json.loads((ROOT / "configs/stage1_3_selected.json").read_text())
    lr = float(selected["selected_learning_rates"][args.model])
    threshold = float(selected["selected_thresholds"][args.model])
    run_id = cfg["protocol"]["formal_run_id"]
    raw = ROOT / "results/stage1_3/raw" / run_id
    artifacts = ROOT / "artifacts/stage1_3" / run_id
    raw.mkdir(parents=True, exist_ok=True); artifacts.mkdir(parents=True, exist_ok=True)
    model, result = train_one(args.model, args.seed, lr, int(cfg["training"]["formal_steps"]), device)
    records: list[dict] = []; samples: dict[str, torch.Tensor] = {}
    e = cfg["evaluation"]
    _merge(records, samples, evidence_accumulation(model, model_name=args.model, seed=args.seed, threshold=threshold, episodes=int(e["episodes_expression"]), length=int(e["evidence_stream_length"]), sufficient_count=int(e["sufficient_evidence_count"]), insufficient_count=int(e["insufficient_evidence_count"]), device=device))
    _merge(records, samples, silence_under_noise(model, model_name=args.model, seed=args.seed, threshold=threshold, length=1000, streams=int(e["noise_parallel_streams"]), device=device))
    if args.model in {"B6_arbitration", "B7_nonconserving_self_replay"}:
        _merge(records, samples, silence_under_noise(model, model_name=args.model, seed=args.seed, threshold=threshold, length=10000, streams=int(e["noise_parallel_streams"]), device=device))
        _merge(records, samples, self_output_audit(model, model_name=args.model, seed=args.seed, threshold=threshold, episodes=int(e["episodes_expression"]), tick_counts=list(e["self_output_ticks"]), device=device))
    if args.model in {"B3_joint", "B4_m_only", "B5_f_only", "B6_arbitration"}:
        _merge(records, samples, arbitration_sweep(model, model_name=args.model, seed=args.seed, threshold=threshold, episodes=int(e["episodes_arbitration"]), distractor_counts=list(e["distractor_counts"]), device=device))
    if args.model in {"B3_joint", "B6_arbitration"}:
        _merge(records, samples, thought_driven_persistence(model, model_name=args.model, seed=args.seed, threshold=threshold, episodes=int(e["episodes_persistence"]), fact_count=int(e["thought_fact_count"]), internal_ticks=int(e["thought_internal_ticks"]), interference=int(e["thought_interference"]), device=device))
        _merge(records, samples, interleaved_streaming(model, model_name=args.model, seed=args.seed, threshold=threshold, episodes=int(e["episodes_association"]), intervals=list(e["interleaved_intervals"]), device=device))
        _merge(records, samples, revision_after_expression(model, model_name=args.model, seed=args.seed, threshold=threshold, episodes=int(e["episodes_revision"]), device=device))
    if args.model in {"B0_no_persistent", "B3_joint", "B6_arbitration"}:
        _merge(records, samples, pattern_discovery(model, model_name=args.model, seed=args.seed, threshold=threshold, episodes=int(e["episodes_pattern"]), device=device))
    if args.model in {"B3_joint", "B4_m_only", "B6_arbitration"}:
        _merge(records, samples, cross_time_association(model, model_name=args.model, seed=args.seed, threshold=threshold, episodes=int(e["episodes_association"]), distractors=int(e["cross_time_distractors"]), device=device))
    tag = f"{args.model}__seed{args.seed}"
    pd.DataFrame(records).to_parquet(raw / f"{tag}__records.parquet", index=False)
    pd.DataFrame([{**row, "model": args.model, "seed": args.seed, "lr": lr} for row in result.logs]).to_parquet(raw / f"{tag}__training.parquet", index=False)
    torch.save({"model": model.state_dict(), "model_name": args.model, "seed": args.seed, "lr": lr, "threshold": threshold}, artifacts / f"{tag}__model.pt")
    torch.save(samples, artifacts / f"{tag}__tensor_samples.pt")
    manifest = {"run_id": run_id, "model": args.model, "seed": args.seed, "lr": lr, "threshold": threshold, "record_rows": len(records), "validation_loss": result.validation_loss, "git_revision": __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "protocol_freeze_sha256": sha256(ROOT / "artifacts/stage1_3_protocol.freeze.json"), "selected_config_sha256": sha256(ROOT / "configs/stage1_3_selected.json")}
    (raw / f"{tag}__manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["development", "select-lr", "threshold", "freeze-selection", "formal"], required=True)
    parser.add_argument("--model", choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    verify_freeze(); cfg = config()
    if args.mode == "select-lr":
        select_lr(cfg); return
    if args.mode == "freeze-selection":
        freeze_selection(cfg); return
    if args.model is None or args.seed is None:
        parser.error("job modes require --model and --seed")
    device = torch.device(args.device)
    if args.mode == "development":
        if args.lr is None: parser.error("development requires --lr")
        development_job(args, cfg, device)
    elif args.mode == "threshold":
        threshold_job(args, cfg, device)
    else:
        formal_job(args, cfg, device)


if __name__ == "__main__":
    main()
