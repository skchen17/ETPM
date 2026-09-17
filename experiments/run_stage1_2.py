#!/usr/bin/env python3
"""Frozen, resumable ET-RCM Stage-1.2 development and formal runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import yaml

from etrcm.stage1_1.model import LearnedModelConfig
from etrcm.stage1_1.training import (
    TrainSpec,
    evaluate_memory_validation,
    train_memory_model,
)
from etrcm.stage1_2.baselines import FORMAL_MODELS, build_stage12_baseline
from etrcm.stage1_2.evaluation import (
    autonomous_memory_selection,
    endogenous_time_necessity,
    exposure_reuse_phase,
    functional_query_intervention,
    no_self_evidence,
    selective_scaling,
    sequential_computation,
    storage_vs_use,
)
from etrcm.stage1_2.training import (
    train_autonomous_model,
    train_no_evidence_model,
    train_sequential_model,
)


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_freeze() -> dict[str, str]:
    freeze = json.loads((ROOT / "artifacts/stage1_2_protocol.freeze.json").read_text())
    for relative, expected in freeze["files"].items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Stage-1.2 freeze mismatch: {relative} {actual} != {expected}")
    for line in (ROOT / "artifacts/stage1_2_prior_assets.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        actual = sha256(ROOT / relative.strip())
        if actual != expected:
            raise RuntimeError(f"prior frozen artifact changed: {relative}")
    for freeze_name in (
        "stage1_2_amendment1.freeze.json",
        "stage1_2_formal_selection.freeze.json",
    ):
        extra_path = ROOT / "artifacts" / freeze_name
        if not extra_path.exists():
            continue
        extra = json.loads(extra_path.read_text())
        for section in ("files", "selected_config", "development_artifacts"):
            for relative, expected in extra.get(section, {}).items():
                actual = sha256(ROOT / relative)
                if actual != expected:
                    raise RuntimeError(f"{freeze_name} mismatch: {relative}")
    return dict(freeze["files"])


def load_config() -> dict:
    return yaml.safe_load((ROOT / "configs/stage1_2.yaml").read_text())


def spec(config: dict, steps: int, learning_rate: float) -> TrainSpec:
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


def revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def write_frame(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def development(config: dict, device: torch.device, output_dir: Path) -> None:
    model_config = LearnedModelConfig.from_mapping(config)
    rows = []
    for model_name in FORMAL_MODELS:
        for learning_rate in config["training"]["candidate_learning_rates"]:
            for seed in config["training"]["development_seeds"]:
                set_seed(seed)
                model = build_stage12_baseline(model_name, model_config).to(device)
                logs = train_memory_model(
                    model,
                    spec(config, int(config["training"]["development_steps"]), float(learning_rate)),
                    seed=seed,
                    device=device,
                )
                score = evaluate_memory_validation(
                    model,
                    seed=seed + 600_000,
                    batch_size=384,
                    symbol_count=model_config.symbol_count,
                    device=device,
                )
                rows.append(
                    {
                        "model": model_name,
                        "learning_rate": learning_rate,
                        "seed": seed,
                        "validation_accuracy": score,
                        "last_train_loss": logs[-1]["loss"],
                        "parameter_count": model.trainable_parameters(),
                        "persistent_state_bytes": model.persistent_state_bytes(),
                    }
                )
    write_frame(rows, output_dir / "learning_rate_selection.parquet")
    means = pd.DataFrame(rows).groupby(["model", "learning_rate"], as_index=False).validation_accuracy.mean()
    selected = {}
    for model_name in FORMAL_MODELS:
        arm = means[means.model.eq(model_name)].sort_values(
            ["validation_accuracy", "learning_rate"], ascending=[False, True]
        )
        selected[model_name] = float(arm.iloc[0].learning_rate)

    # Select task difficulty on development data only, using the selected B6 LR.
    difficulty_rows = []
    for seed in config["training"]["development_seeds"]:
        set_seed(seed + 50_000)
        model = build_stage12_baseline("B6_full", model_config).to(device)
        train_memory_model(
            model,
            spec(config, int(config["training"]["development_steps"]), selected["B6_full"]),
            seed=seed + 50_000,
            device=device,
        )
        for interference in config["evaluation"]["endogenous_interference_candidates"]:
            result = endogenous_time_necessity(
                model, seed=seed, ticks=8, interference=int(interference), episodes=64,
                symbol_count=model_config.symbol_count, device=device,
            )
            frame = pd.DataFrame(result)
            difficulty_rows.append(
                {
                    "seed": seed,
                    "interference": interference,
                    "accuracy": frame.accuracy.mean(),
                    "before_accuracy": frame[frame.condition.eq("before_interference")].accuracy.mean(),
                    "after_accuracy": frame[frame.condition.eq("after_interference")].accuracy.mean(),
                }
            )
    write_frame(difficulty_rows, output_dir / "endogenous_difficulty_selection.parquet")
    difficulty = pd.DataFrame(difficulty_rows).groupby("interference").accuracy.mean()
    center = sum(config["evaluation"]["endogenous_target_accuracy_band"]) / 2
    selected_interference = int((difficulty - center).abs().sort_values().index[0])
    selection = {
        "protocol": "stage1.2-v1",
        "development_only": True,
        "selected_learning_rates": selected,
        "selected_endogenous_interference": selected_interference,
        "selection_rule": "maximum mean validation accuracy per architecture; difficulty closest to frozen 0.60 center",
    }
    (output_dir / "selected_hyperparameters.json").write_text(json.dumps(selection, indent=2))


def development_job(
    config: dict,
    device: torch.device,
    output_dir: Path,
    model_name: str,
    learning_rate: float,
    seed: int,
    steps: int,
) -> None:
    model_config = LearnedModelConfig.from_mapping(config)
    set_seed(seed)
    model = build_stage12_baseline(model_name, model_config).to(device)
    logs = train_memory_model(
        model, spec(config, steps, learning_rate), seed=seed, device=device
    )
    score = evaluate_memory_validation(
        model, seed=seed + 600_000, batch_size=384,
        symbol_count=model_config.symbol_count, device=device,
    )
    row = {
        "model": model_name, "learning_rate": learning_rate, "seed": seed,
        "development_steps": steps, "validation_accuracy": score,
        "last_train_loss": logs[-1]["loss"],
        "parameter_count": model.trainable_parameters(),
        "persistent_state_bytes": model.persistent_state_bytes(),
    }
    tag = str(learning_rate).replace(".", "p")
    write_frame([row], output_dir / f"dev__{model_name}__lr{tag}__seed{seed}.parquet")


def collect_development(config: dict, device: torch.device, output_dir: Path, steps: int) -> None:
    files = sorted(output_dir.glob("dev__*.parquet"))
    expected = len(FORMAL_MODELS) * len(config["training"]["candidate_learning_rates"]) * len(config["training"]["development_seeds"])
    if len(files) != expected:
        raise RuntimeError(f"expected {expected} development jobs, found {len(files)}")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    frame.to_parquet(output_dir / "learning_rate_selection.parquet", index=False)
    means = frame.groupby(["model", "learning_rate"], as_index=False).agg(
        validation_accuracy=("validation_accuracy", "mean"),
        last_train_loss=("last_train_loss", "mean"),
    )
    selected = {}
    for model_name in FORMAL_MODELS:
        arm = means[means.model.eq(model_name)].sort_values(
            ["validation_accuracy", "last_train_loss", "learning_rate"],
            ascending=[False, True, True],
        )
        selected[model_name] = float(arm.iloc[0].learning_rate)
    model_config = LearnedModelConfig.from_mapping(config)
    difficulty_rows = []
    for seed in config["training"]["development_seeds"]:
        set_seed(seed + 50_000)
        model = build_stage12_baseline("B6_full", model_config).to(device)
        train_memory_model(
            model, spec(config, steps, selected["B6_full"]),
            seed=seed + 50_000, device=device,
        )
        for interference in config["evaluation"]["endogenous_interference_candidates"]:
            result = endogenous_time_necessity(
                model, seed=seed, ticks=8, interference=int(interference), episodes=64,
                symbol_count=model_config.symbol_count, device=device,
            )
            result_frame = pd.DataFrame(result)
            difficulty_rows.append({
                "seed": seed, "interference": interference,
                "accuracy": result_frame.accuracy.mean(),
                "before_accuracy": result_frame[result_frame.condition.eq("before_interference")].accuracy.mean(),
                "after_accuracy": result_frame[result_frame.condition.eq("after_interference")].accuracy.mean(),
            })
    write_frame(difficulty_rows, output_dir / "endogenous_difficulty_selection.parquet")
    difficulty = pd.DataFrame(difficulty_rows).groupby("interference").accuracy.mean()
    center = sum(config["evaluation"]["endogenous_target_accuracy_band"]) / 2
    selected_interference = int((difficulty - center).abs().sort_values().index[0])
    selection = {
        "protocol": "stage1.2-v1+A1", "development_only": True,
        "development_steps": steps,
        "selected_learning_rates": selected,
        "selected_endogenous_interference": selected_interference,
        "selection_rule": "maximum mean accuracy; exact ties by lower mean loss then lower LR; difficulty closest to frozen 0.60 center",
    }
    (output_dir / "selected_hyperparameters.json").write_text(json.dumps(selection, indent=2))


def formal(config: dict, model_name: str, seed: int, device: torch.device, run_dir: Path) -> None:
    selected_path = ROOT / "configs/stage1_2_selected.json"
    if not selected_path.exists():
        raise RuntimeError("formal config is not frozen: configs/stage1_2_selected.json missing")
    selected = json.loads(selected_path.read_text())
    learning_rate = float(selected["selected_learning_rates"][model_name])
    model_config = LearnedModelConfig.from_mapping(config)
    evaluation = config["evaluation"]
    set_seed(seed)
    model = build_stage12_baseline(model_name, model_config).to(device)
    logs = train_memory_model(
        model,
        spec(config, int(config["training"]["formal_steps_memory"]), learning_rate),
        seed=seed,
        device=device,
    )
    for row in logs:
        row["training_arm"] = "memory"
    job = f"{model_name}__seed{seed}"
    checkpoint_dir = ROOT / "artifacts/stage1_2" / run_dir.name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(model_config), "seed": seed}, checkpoint_dir / f"{job}__memory.pt")
    rows = selective_scaling(
        model, model_name=model_name, seed=seed,
        distractor_counts=list(evaluation["distractor_counts"]),
        useful_count=int(evaluation["useful_fact_count"]),
        episodes=int(evaluation["episodes_scaling"]), symbol_count=model_config.symbol_count,
        device=device,
    )
    if model_name == "B6_full":
        rows += functional_query_intervention(
            model, seed=seed, episodes=int(evaluation["episodes_query"]),
            interference=int(evaluation["phase_interference"]),
            symbol_count=model_config.symbol_count, device=device,
        )
        rows += exposure_reuse_phase(
            model, seed=seed, exposures=list(evaluation["exposure_counts"]),
            reuses=list(evaluation["reuse_counts"]), episodes=int(evaluation["episodes_phase"]),
            interference=int(evaluation["phase_interference"]),
            symbol_count=model_config.symbol_count, device=device,
        )
        rows += endogenous_time_necessity(
            model, seed=seed, ticks=8,
            interference=int(selected["selected_endogenous_interference"]),
            episodes=int(evaluation["episodes_query"]), symbol_count=model_config.symbol_count,
            device=device,
        )
        rows += storage_vs_use(
            model, seed=seed, episodes=int(evaluation["episodes_query"]),
            interference=int(config["selective_scaling"]["high_pressure_distractors"]),
            symbol_count=model_config.symbol_count, device=device,
        )

        set_seed(seed + 1_000_000)
        autonomous = build_stage12_baseline("B6_full", model_config).to(device)
        auto_logs = train_autonomous_model(
            autonomous,
            spec(config, int(config["training"]["formal_steps_autonomous"]), learning_rate),
            seed=seed + 1_000_000, device=device,
        )
        for row in auto_logs:
            row["training_arm"] = "autonomous"
        logs += auto_logs
        rows += autonomous_memory_selection(
            autonomous, seed=seed, tick_counts=list(evaluation["internal_ticks"]),
            episodes=int(evaluation["episodes_autonomous"]),
            symbol_count=model_config.symbol_count, device=device,
        )
        torch.save({"model": autonomous.state_dict(), "config": asdict(model_config), "seed": seed}, checkpoint_dir / f"{job}__autonomous.pt")

        set_seed(seed + 2_000_000)
        sequential = build_stage12_baseline("B6_full", model_config).to(device)
        seq_logs = train_sequential_model(
            sequential,
            spec(config, int(config["training"]["formal_steps_sequential"]), learning_rate),
            seed=seed + 2_000_000, device=device,
            train_max_length=int(evaluation["sequential_train_max_length"]),
        )
        for row in seq_logs:
            row["training_arm"] = "sequential"
        logs += seq_logs
        rows += sequential_computation(
            sequential, seed=seed, path_lengths=list(evaluation["path_lengths"]),
            tick_counts=list(evaluation["internal_ticks"]),
            episodes=int(evaluation["episodes_sequential"]),
            train_max_length=int(evaluation["sequential_train_max_length"]),
            symbol_count=model_config.symbol_count, device=device,
        )
        torch.save({"model": sequential.state_dict(), "config": asdict(model_config), "seed": seed}, checkpoint_dir / f"{job}__sequential.pt")

        set_seed(seed + 3_000_000)
        no_evidence_model = build_stage12_baseline("B6_full", model_config).to(device)
        no_evidence_logs = train_no_evidence_model(
            no_evidence_model,
            spec(config, int(config["training"]["formal_steps_no_evidence"]), learning_rate),
            seed=seed + 3_000_000, device=device,
        )
        for row in no_evidence_logs:
            row["training_arm"] = "no_evidence"
        logs += no_evidence_logs
        rows += no_self_evidence(
            no_evidence_model, seed=seed, tick_counts=list(evaluation["no_evidence_ticks"]),
            episodes=int(evaluation["episodes_no_evidence"]),
            symbol_count=model_config.symbol_count, device=device,
        )
        torch.save({"model": no_evidence_model.state_dict(), "config": asdict(model_config), "seed": seed}, checkpoint_dir / f"{job}__no_evidence.pt")

    git_revision = revision()
    for row in rows:
        row.update({"run_id": run_dir.name, "git_revision": git_revision})
    for row in logs:
        row.update({"run_id": run_dir.name, "model": model_name, "seed": seed})
    write_frame(rows, run_dir / f"{job}__records.parquet")
    write_frame(logs, run_dir / f"{job}__training.parquet")
    manifest = {
        "run_id": run_dir.name,
        "job": job,
        "model": model_name,
        "seed": seed,
        "selected_learning_rate": learning_rate,
        "selected_endogenous_interference": selected["selected_endogenous_interference"],
        "git_revision": git_revision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(rows),
        "long_stream_run": False,
        "stage2_training_run": False,
    }
    (run_dir / f"{job}__manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "development-job", "development-collect", "formal"), required=True)
    parser.add_argument("--model", choices=FORMAL_MODELS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-id")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--development-steps", type=int, default=600)
    args = parser.parse_args()
    freeze = verify_freeze()
    config = load_config()
    run_id = args.run_id or ("stage1_2-development-v1" if args.mode == "development" else config["protocol"]["run_id"])
    output = ROOT / "results/stage1_2/raw" / run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "freeze_verified.json").write_text(json.dumps(freeze, indent=2))
    device = torch.device(args.device)
    if args.mode == "development":
        development(config, device, output)
    elif args.mode == "development-job":
        if args.model is None or args.seed is None or args.learning_rate is None:
            parser.error("development-job requires model, seed and learning-rate")
        development_job(
            config, device, output, args.model, args.learning_rate,
            args.seed, args.development_steps,
        )
    elif args.mode == "development-collect":
        collect_development(config, device, output, args.development_steps)
    else:
        if args.model is None or args.seed is None:
            parser.error("formal mode requires --model and --seed")
        formal(config, args.model, args.seed, device, output)


if __name__ == "__main__":
    main()
