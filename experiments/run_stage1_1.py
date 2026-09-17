#!/usr/bin/env python3
"""Run frozen ET-RCM Stage-1.1 development or one resumable formal job."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from etrcm.stage1_1.baselines import TRAINED_BASELINES, build_baseline
from etrcm.stage1_1.evaluation import (
    evaluate_capacity_pressure,
    evaluate_consolidate_before_interference,
    evaluate_frequency_utility,
    evaluate_idle_reasoning,
    evaluate_interleaved_time,
    evaluate_learned_reuse,
    evaluate_reason_before_interruption,
    evaluate_revision,
    evaluate_unknowable,
)
from etrcm.stage1_1.model import LearnedETRCM, LearnedModelConfig
from etrcm.stage1_1.training import (
    TrainSpec,
    evaluate_memory_validation,
    train_graph_model,
    train_memory_model,
    train_unknowable_model,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze() -> dict[str, str]:
    freeze = json.loads((ROOT / "artifacts/stage1_1_protocol.freeze.json").read_text())
    mapping = {
        relative: ROOT / relative for relative in freeze["files"]
    }
    for key, path in mapping.items():
        actual = sha256(path)
        if actual != freeze["files"][key]:
            raise RuntimeError(
                f"frozen file mismatch: {path} {actual} != {freeze['files'][key]}"
            )
    stage1_manifest = ROOT / "artifacts/stage1_frozen_assets.sha256"
    for line in stage1_manifest.read_text().splitlines():
        expected, relative = line.split(maxsplit=1)
        path = ROOT / relative.strip()
        if sha256(path) != expected:
            raise RuntimeError(f"Stage-1 frozen asset changed: {relative}")
    return dict(freeze["files"])


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs/stage1_1.yaml").read_text())


def train_spec(config: dict, steps: int, learning_rate: float) -> TrainSpec:
    train = config["training"]
    return TrainSpec(
        steps=steps,
        batch_size=int(train["batch_size"]),
        learning_rate=learning_rate,
        weight_decay=float(train["weight_decay"]),
        gradient_clip=float(train["gradient_clip"]),
        access_regularization=float(train["access_regularization"]),
        symbol_count=int(config["model"]["symbol_count"]),
    )


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def write_frame(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def development(config: dict, device: torch.device, output: Path) -> None:
    model_config = LearnedModelConfig.from_mapping(config)
    rows: list[dict] = []
    for learning_rate in config["training"]["development_learning_rates"]:
        for seed in config["training"]["development_seeds"]:
            set_seed(seed)
            model = build_baseline("B6_full", model_config).to(device)
            log = train_memory_model(
                model,
                train_spec(config, int(config["training"]["development_steps"]), float(learning_rate)),
                seed=seed,
                device=device,
            )
            score = evaluate_memory_validation(
                model,
                seed=seed + 500_000,
                batch_size=512,
                symbol_count=model_config.symbol_count,
                device=device,
            )
            rows.append(
                {
                    "learning_rate": learning_rate,
                    "seed": seed,
                    "validation_accuracy": score,
                    "last_train_loss": log[-1]["loss"],
                }
            )
    write_frame(rows, output)


def ablation_models(parent: LearnedETRCM, config: LearnedModelConfig, device: torch.device):
    variants = {
        "A1_gamma_zero": LearnedETRCM(replace(config, gamma=0.0)),
        "A2_uniform_transfer": LearnedETRCM(config, consolidation="uniform"),
        "A3_equal_timescales": LearnedETRCM(config, equal_timescales=True),
        "A4_no_null_dynamics": LearnedETRCM(config, idle_updates=False),
        "A5_random_query": LearnedETRCM(config, random_query=True),
        "A6_frozen_H": LearnedETRCM(config, freeze_core=True),
        "A7_nonconserving": LearnedETRCM(config, consolidation="nonconserving"),
    }
    state = parent.state_dict()
    for name, model in variants.items():
        model.load_state_dict(state, strict=True)
        yield name, model.to(device)


def formal_job(
    config: dict,
    model_name: str,
    seed: int,
    learning_rate: float,
    device: torch.device,
    run_dir: Path,
) -> None:
    model_config = LearnedModelConfig.from_mapping(config)
    evaluation = config["evaluation"]
    set_seed(seed)
    model = build_baseline(model_name, model_config).to(device)
    logs = train_memory_model(
        model,
        train_spec(config, int(config["training"]["formal_steps_memory"]), learning_rate),
        seed=seed,
        device=device,
    )
    job = f"{model_name}__seed{seed}"
    checkpoint_dir = ROOT / "artifacts/stage1_1" / run_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": asdict(model_config), "seed": seed},
        checkpoint_dir / f"{job}__memory.pt",
    )
    episodes = int(evaluation["capacity_eval_episodes"])
    symbol_count = model_config.symbol_count
    rows: list[dict] = []
    rows += evaluate_learned_reuse(
        model,
        model_name=model_name,
        seed=seed,
        reuse_counts=list(evaluation["reuse_counts"]),
        episodes=episodes,
        interference=int(evaluation["learned_reuse_interference"]),
        symbol_count=symbol_count,
        device=device,
    )
    rows += evaluate_capacity_pressure(
        model,
        model_name=model_name,
        seed=seed,
        distractor_counts=list(evaluation["distractor_counts"]),
        episodes=episodes,
        symbol_count=symbol_count,
        device=device,
    )
    rows += evaluate_frequency_utility(
        model,
        model_name=model_name,
        seed=seed,
        useless_frequencies=list(evaluation["useless_frequencies"]),
        useful_frequencies=list(evaluation["useful_frequencies"]),
        episodes=episodes,
        symbol_count=symbol_count,
        device=device,
    )
    rows += evaluate_interleaved_time(
        model,
        model_name=model_name,
        seed=seed,
        ticks=8,
        episodes=episodes,
        symbol_count=symbol_count,
        device=device,
    )
    rows += evaluate_consolidate_before_interference(
        model,
        model_name=model_name,
        seed=seed,
        ticks=8,
        interference=128,
        episodes=episodes,
        symbol_count=symbol_count,
        device=device,
    )
    rows += evaluate_revision(
        model,
        model_name=model_name,
        seed=seed,
        old_exposures=list(evaluation["old_exposures"]),
        new_exposures=list(evaluation["new_exposures"]),
        new_reuses=list(evaluation["new_reuses"]),
        episodes=32,
        symbol_count=symbol_count,
        device=device,
    )
    if model_name == "B6_full":
        rows += evaluate_capacity_pressure(
            model,
            model_name=model_name,
            seed=seed,
            distractor_counts=list(evaluation["distractor_counts"]),
            episodes=episodes,
            symbol_count=symbol_count,
            device=device,
            lesion=True,
        )
        for ablation_name, ablation in ablation_models(model, model_config, device):
            rows += evaluate_learned_reuse(
                ablation,
                model_name=ablation_name,
                seed=seed,
                reuse_counts=list(evaluation["reuse_counts"]),
                episodes=episodes,
                interference=int(evaluation["learned_reuse_interference"]),
                symbol_count=symbol_count,
                device=device,
            )

    if model_name in {"B1_gru", "B5_no_idle", "B6_full"}:
        set_seed(seed + 1_000_000)
        graph_model = build_baseline(model_name, model_config).to(device)
        graph_logs = train_graph_model(
            graph_model,
            train_spec(config, int(config["training"]["formal_steps_graph"]), learning_rate),
            seed=seed + 1_000_000,
            device=device,
        )
        for item in graph_logs:
            item["training_arm"] = "graph"
        logs += graph_logs
        torch.save(
            {"model": graph_model.state_dict(), "config": asdict(model_config), "seed": seed},
            checkpoint_dir / f"{job}__graph.pt",
        )
        graph_episodes = int(evaluation["graph_eval_episodes"])
        rows += evaluate_idle_reasoning(
            graph_model,
            model_name=model_name,
            seed=seed,
            tick_counts=list(evaluation["idle_ticks"]),
            episodes=graph_episodes,
            symbol_count=symbol_count,
            device=device,
        )
        rows += evaluate_reason_before_interruption(
            graph_model,
            model_name=model_name,
            seed=seed,
            ticks=8,
            episodes=graph_episodes,
            symbol_count=symbol_count,
            device=device,
        )

    if model_name == "B6_full":
        set_seed(seed + 2_000_000)
        unknown_model = build_baseline(model_name, model_config).to(device)
        unknown_logs = train_unknowable_model(
            unknown_model,
            train_spec(config, int(config["training"]["formal_steps_unknowable"]), learning_rate),
            seed=seed + 2_000_000,
            device=device,
        )
        for item in unknown_logs:
            item["training_arm"] = "unknowable"
        logs += unknown_logs
        torch.save(
            {"model": unknown_model.state_dict(), "config": asdict(model_config), "seed": seed},
            checkpoint_dir / f"{job}__unknowable.pt",
        )
        rows += evaluate_unknowable(
            unknown_model,
            model_name=model_name,
            seed=seed,
            tick_counts=list(evaluation["unknowable_ticks"]),
            episodes=int(evaluation["unknowable_eval_episodes"]),
            symbol_count=symbol_count,
            device=device,
        )

    for row in rows:
        row.update({"run_id": run_dir.name, "git_revision": git_revision()})
    for item in logs:
        item.update({"model": model_name, "seed": seed, "run_id": run_dir.name})
    write_frame(rows, run_dir / f"{job}__records.parquet")
    write_frame(logs, run_dir / f"{job}__training.parquet")
    manifest = {
        "run_id": run_dir.name,
        "job": job,
        "model": model_name,
        "seed": seed,
        "learning_rate": learning_rate,
        "device": str(device),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "record_count": len(rows),
        "checkpoint_dir": str(checkpoint_dir.relative_to(ROOT)),
        "long_stream_run": False,
        "stage2_run": False,
    }
    (run_dir / f"{job}__manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "formal"), required=True)
    parser.add_argument("--model", choices=TRAINED_BASELINES)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id", default="stage1_1-formal-v1")
    args = parser.parse_args()
    freeze = verify_freeze()
    config = load_config()
    device = torch.device(args.device)
    run_dir = ROOT / "results/stage1_1/raw" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "freeze_verified.json").write_text(json.dumps(freeze, indent=2))
    if args.mode == "development":
        development(config, device, run_dir / "development_selection.parquet")
        return
    if args.model is None or args.seed is None:
        parser.error("formal mode requires --model and --seed")
    formal_job(config, args.model, args.seed, args.learning_rate, device, run_dir)


if __name__ == "__main__":
    main()
