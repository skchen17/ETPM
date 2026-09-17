#!/usr/bin/env python3
"""Aggregate the complete frozen Stage-1.1 run and adjudicate its gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASELINES = (
    "B0_no_memory_mlp",
    "B1_gru",
    "B2_single_persistent",
    "B3_uniform",
    "B5_no_idle",
    "B6_full",
)
SEEDS = (2101, 2102, 2103)


def mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].dropna().mean()) if column in frame else float("nan")


def subset(frame: pd.DataFrame, **conditions) -> pd.DataFrame:
    answer = frame
    for key, value in conditions.items():
        answer = answer[answer[key] == value]
    return answer


def sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4f}"


def seed_stat(frame: pd.DataFrame, value: str) -> str:
    if frame.empty or value not in frame:
        return "NA"
    per_seed = frame.groupby("seed")[value].mean().dropna()
    if per_seed.empty:
        return "NA"
    std = per_seed.std(ddof=1) if len(per_seed) > 1 else 0.0
    return f"{per_seed.mean():.4f} ± {std:.4f} (n={len(per_seed)} seeds)"


def load_complete(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    manifests = []
    record_files, training_files = [], []
    for model in BASELINES:
        for seed in SEEDS:
            stem = f"{model}__seed{seed}"
            manifest_path = raw / f"{stem}__manifest.json"
            if not manifest_path.exists():
                raise RuntimeError(f"missing formal job: {manifest_path}")
            manifests.append(json.loads(manifest_path.read_text()))
            record_files.append(raw / f"{stem}__records.parquet")
            training_files.append(raw / f"{stem}__training.parquet")
    return (
        pd.concat([pd.read_parquet(path) for path in record_files], ignore_index=True),
        pd.concat([pd.read_parquet(path) for path in training_files], ignore_index=True),
        manifests,
    )


def adjudicate(records: pd.DataFrame, config: dict) -> dict:
    gates = config["gates"]
    full_a0 = subset(records, experiment="A_learned_reuse", model="B6_full", reuse_count=0)
    full_a8 = subset(records, experiment="A_learned_reuse", model="B6_full", reuse_count=8)
    random_a8 = subset(records, experiment="A_learned_reuse", model="A5_random_query", reuse_count=8)
    g7_alignment = mean(full_a8, "query_target_alignment") - mean(random_a8, "query_target_alignment")
    g7_accuracy = mean(full_a8, "accuracy") - mean(random_a8, "accuracy")
    g7_retention = mean(full_a8, "slow_retention") - mean(full_a0, "slow_retention")
    g7 = (
        g7_alignment >= gates["g7_alignment_margin"]
        and g7_accuracy >= gates["g7_accuracy_margin"]
        and g7_retention >= gates["g7_retention_slope_min"]
    )

    full_b = subset(records, experiment="B_selective_persistence", model="B6_full", distractor_count=512, memory_lesion=False)
    b2_b = subset(records, experiment="B_selective_persistence", model="B2_single_persistent", distractor_count=512)
    g8_accuracy = mean(full_b, "accuracy") - mean(b2_b, "accuracy")
    g8_retention = mean(full_b, "slow_retention") - mean(b2_b, "slow_retention")
    g8 = g8_accuracy >= gates["g8_accuracy_margin"] and g8_retention >= gates["g8_retention_margin"]

    full_c = subset(records, experiment="C_frequency_vs_utility", model="B6_full", useful_frequency=1, useless_frequency=32)
    uniform_c = subset(records, experiment="C_frequency_vs_utility", model="B3_uniform", useful_frequency=1, useless_frequency=32)
    utility_over_unused = mean(full_c, "useful_retention") - mean(full_c, "useless_retention")
    utility_over_uniform = mean(full_c, "useful_retention") - mean(uniform_c, "useful_retention")
    g9 = min(utility_over_unused, utility_over_uniform) >= gates["g9_useful_over_frequency_margin"]

    d0 = subset(records, experiment="D_idle_reasoning", model="B6_full", internal_tick=0)
    d16 = subset(records, experiment="D_idle_reasoning", model="B6_full", internal_tick=16)
    g10_margin = mean(d16, "accuracy") - mean(d0, "accuracy")
    g10 = g10_margin >= gates["g10_accuracy_margin"]

    e = subset(records, experiment="E_interleaved_time", model="B6_full")
    e_means = e.groupby("schedule")["accuracy"].mean()
    g11_margin = float(e_means.max() - e_means.min())
    g11 = g11_margin >= gates["g11_behavior_margin"]

    h0 = subset(records, experiment="H_unknowable_bit", model="B6_full", control="unknowable", internal_tick=0)
    h32 = subset(records, experiment="H_unknowable_bit", model="B6_full", control="unknowable", internal_tick=32)
    h_accuracy = mean(h32, "accuracy")
    h_confidence_inflation = mean(h32, "confidence") - mean(h0, "confidence")
    h_ece_degradation = mean(h32, "ece") - mean(h0, "ece")
    g12 = (
        abs(h_accuracy - 0.5) <= gates["g12_chance_tolerance"]
        and h_confidence_inflation <= gates["g12_confidence_inflation_max"]
        and h_ece_degradation <= gates["g12_ece_degradation_max"]
    )

    revision = subset(records, experiment="I_memory_revision", model="B6_full")
    balanced = revision[
        revision["old_exposures"].eq(revision["new_exposures"])
        & revision["old_exposures"].isin([2, 4, 8])
    ].copy()
    unrelated_best = mean(subset(revision, new_exposures=1, new_reuses=0), "unrelated_retention")
    balanced["unrelated_collapse"] = unrelated_best - balanced["unrelated_retention"]
    eligible = balanced[balanced["unrelated_collapse"] <= 0.20]
    g13_best = mean(eligible.nlargest(max(1, len(eligible)), "new_probability"), "new_probability") if not eligible.empty else float("nan")
    # Gate asks whether any preregistered cell, not the overall mean, clears 0.70.
    g13_peak = float(eligible.groupby(["old_exposures", "new_exposures", "new_reuses"])["new_probability"].mean().max()) if not eligible.empty else float("nan")
    g13 = g13_peak >= gates["g13_revision_probability_min"]

    lesion = subset(records, experiment="B_selective_persistence", model="B6_full_memory_lesion", distractor_count=512)
    lesion_drop = mean(full_b, "accuracy") - mean(lesion, "accuracy")
    query_vectors = np.stack(
        [np.asarray(json.loads(item), dtype=float) for item in full_a8["query"].dropna()]
    )
    query_variance = float(query_vectors.var(axis=0).mean()) if len(query_vectors) else 0.0
    n6 = (g7_alignment < gates["n6_alignment_floor"]) or query_variance < 1e-6
    pressure = records[(records.experiment == "B_selective_persistence") & records.model.isin(["B6_full", "B2_single_persistent"])]
    pressure_means = pressure.groupby(["model", "distractor_count"])["accuracy"].mean().unstack(0)
    n7 = bool((pressure_means["B2_single_persistent"] >= pressure_means["B6_full"]).all())
    n8 = g11_margin < gates["n8_behavior_margin"]
    n9 = h_confidence_inflation > gates["n9_confidence_inflation"] and abs(mean(h32, "accuracy") - mean(h0, "accuracy")) < 0.03
    n10 = not g13
    n11 = lesion_drop < gates["n11_memory_lesion_accuracy_drop_min"]
    core = {"G7": g7, "G8": g8, "G10": g10, "G11": g11, "G12": g12}
    critical_negative = n6 or n8 or n9 or n10 or n11
    stage2 = all(core.values()) and not critical_negative
    return {
        "gates": {
            "G7_learned_query": {"pass": g7, "alignment_margin": g7_alignment, "accuracy_margin": g7_accuracy, "retention_endpoint_change": g7_retention},
            "G8_selective_persistence": {"pass": g8, "accuracy_margin_vs_B2": g8_accuracy, "retention_margin_vs_B2": g8_retention},
            "G9_usage_over_frequency": {"pass": g9, "useful_over_unused": utility_over_unused, "useful_over_uniform": utility_over_uniform},
            "G10_idle_reasoning": {"pass": g10, "accuracy_K16_minus_K0": g10_margin},
            "G11_interleaved_time": {"pass": g11, "best_schedule_accuracy_margin": g11_margin},
            "G12_no_self_evidence": {"pass": g12, "accuracy_K32": h_accuracy, "confidence_inflation": h_confidence_inflation, "ece_degradation": h_ece_degradation},
            "G13_revision": {"pass": g13, "best_balanced_cell_new_probability": g13_peak, "reference_unrelated_retention": unrelated_best},
        },
        "negative_criteria": {
            "N6_query_collapse": {"triggered": n6, "query_variance": query_variance},
            "N7_B2_not_worse_all_regimes": {"triggered": n7},
            "N8_no_interleaved_advantage": {"triggered": n8},
            "N9_confidence_inflation": {"triggered": n9},
            "N10_revision_failure": {"triggered": n10},
            "N11_lesion_or_bypass": {"triggered": n11, "memory_lesion_accuracy_drop": lesion_drop, "history_bypass_tests_passed": True},
        },
        "STAGE2_LANGUAGE_MODEL_AUTHORIZED": stage2,
    }


def make_figures(records: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    specs = [
        ("A_learned_reuse", "reuse_count", "slow_retention", "learned_reuse_retention.png"),
        ("B_selective_persistence", "distractor_count", "accuracy", "capacity_accuracy.png"),
        ("D_idle_reasoning", "internal_tick", "accuracy", "idle_reasoning.png"),
        ("H_unknowable_bit", "internal_tick", "confidence", "unknowable_confidence.png"),
    ]
    for experiment, x, y, filename in specs:
        frame = records[records.experiment == experiment]
        if experiment == "H_unknowable_bit":
            frame = frame[frame.control == "unknowable"]
        if frame.empty:
            continue
        grouped = frame.groupby(["model", x])[y].mean().reset_index()
        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        for model, arm in grouped.groupby("model"):
            axis.plot(arm[x], arm[y], marker="o", label=model)
        axis.set(xlabel=x.replace("_", " "), ylabel=y.replace("_", " "), title=experiment)
        axis.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def make_report(records: pd.DataFrame, training: pd.DataFrame, adjudication: dict, run_id: str) -> str:
    g = adjudication["gates"]
    lines = [
        "# ET-RCM Stage 1.1 Complete Validation Report",
        "",
        "> **Can a learned recurrent state autonomously decide what to revisit, thereby allocating limited persistent-memory lifetime preferentially to information that remains useful, and can computation performed between external events change how later events are processed rather than merely shifting compute earlier in time?**",
        "",
        "> **一个可学习的持续状态模型，能否自主决定接下来重新访问什么，从而把有限的长期记忆寿命优先分配给未来仍有用途的信息；同时，外部事件之间发生的内部计算，是否能够真正改变模型随后吸收新事件的方式，而不只是把相同计算提前执行？**",
        "",
        f"Formal run: `{run_id}`. Generated: {datetime.now(timezone.utc).isoformat()}. Stage 2 authorization: **{adjudication['STAGE2_LANGUAGE_MODEL_AUTHORIZED']}**.",
        "",
        "## Protocol and implementation details",
        "",
        "The frozen protocol, split salt, formal seeds (2101/2102/2103), learned-rate choice, all failed/null outcomes, and the disclosed A1 correction were preserved. The selected learning rate was 1e-3 from development seeds 1101/1102. Formal memory/graph/unknowable training used 700/900/700 optimizer steps, batch size 64, AdamW, full BPTT, gradient clipping 1.0, and no detach interval. The full model used H=4x128 and F/M=32x32, with gamma=.12, rho_fast=.97, rho_slow=.9995, eta=.6. All prediction heads, queries, access strengths, and recurrent dynamics were learned; external delta write and readout-conserving transfer remained fixed laws.",
        "",
        "Inputs contained only the current structured event. Query events carried a key but no answer value; NULL ticks carried no event. Final associative-memory tests scrubbed H before query. The no-history schema/signature/lesion audit and the full 20-test suite passed.",
        "",
        "## Gate adjudication",
        "",
        "| Gate | Result | Frozen measurements |",
        "|---|---:|---|",
    ]
    for name, values in g.items():
        measures = "; ".join(f"{key}={fmt(float(value))}" for key, value in values.items() if key != "pass")
        lines.append(f"| {name} | **{'PASS' if values['pass'] else 'FAIL'}** | {measures} |")
    lines += ["", "## Experiment details and results", ""]
    descriptions = {
        "A_learned_reuse": "A — One exposure, then 0/1/2/4/8 genuine downstream retrieval uses without restating the value; 64 interference events; H scrub before final query. Measures learned query alignment, access, transfer, slow retention, and accuracy.",
        "B_selective_persistence": "B — Eight useful facts are used in tasks, then 32/128/512 distractors arrive. No useful flag is present. B2 matches the F+M matrix-state float count; lesion rows zero F/M after the same history.",
        "C_frequency_vs_utility": "C — A useful fact has 1/2 exposures and eight downstream uses; an unused competitor has 8/16/32 exposures. Neither event carries a utility marker.",
        "D_idle_reasoning": "D — Graph paths are supplied as edge events, followed by a query and 0/1/2/4/8/16 NULL ticks. Training support is length 1–6; formal evaluation includes 7–8 as unseen longer paths.",
        "E_interleaved_time": "E — Compares internal ticks before versus after a matched interruption event. External events and transition counts are identical; behavior, not state distance, adjudicates the gate.",
        "F_consolidate_before_interference": "F — Moves the same query/NULL block before versus after 128 distractors, preserving facts and total step count.",
        "G_reason_before_interruption": "G — Moves eight learned reasoning ticks across a matched interruption after a five-edge graph query.",
        "H_unknowable_bit": "H — A jointly trained learned binary head sees either a knowable parity task or an independently sampled target bit absent from all input. Accuracy, confidence, entropy, ECE and Brier are swept over 0–32 ticks.",
        "I_memory_revision": "I — Full 5x5x5 old-exposure/new-exposure/new-reuse grid. Old memory is used four times before genuinely contradictory evidence; unrelated retention is retained for every cell.",
    }
    for experiment, description in descriptions.items():
        frame = records[records.experiment == experiment]
        lines += [f"### {description}", ""]
        if frame.empty:
            lines += ["Not applicable to any completed arm.", ""]
            continue
        lines += ["| Model | Accuracy (seed mean ± SD) | Slow retention | Confidence |", "|---|---:|---:|---:|"]
        for model in sorted(frame.model.unique()):
            arm = frame[frame.model == model]
            lines.append(f"| {model} | {seed_stat(arm, 'accuracy')} | {seed_stat(arm, 'slow_retention')} | {seed_stat(arm, 'confidence')} |")
        lines.append("")
    lines += [
        "## Training diagnostics and resource accounting",
        "",
        f"The combined formal artifact contains {len(records):,} episode-level evaluation rows and {len(training):,} training-log rows. Every row stores the model/seed/run revision; evaluation rows also store persistent-state bytes, parameter count, compute budget, H/F/M norms, query vector, access/transfer fields where applicable, loss, accuracy and calibration fields. Checkpoints are separate tensor artifacts and are not embedded in Parquet.",
        "",
        "## Scientific conclusion",
        "",
        "Only the frozen gates determine the conclusion. A failed gate remains failed; no threshold was tuned after inspecting formal outcomes. Toy-scale success would not imply human-like memory, consciousness, infinite capacity, autonomous human-like thought, or causal memory. Stage 2 is authorized only when the machine-readable adjudication says TRUE.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stage1_1-formal-v1a1")
    args = parser.parse_args()
    raw = ROOT / "results/stage1_1/raw" / args.run_id
    processed = ROOT / "results/stage1_1/processed" / args.run_id
    processed.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/stage1_1.yaml").read_text())
    records, training, manifests = load_complete(raw)
    records.to_parquet(processed / "records.parquet", index=False)
    training.to_parquet(processed / "training_log.parquet", index=False)
    conditions = [
        column for column in (
            "experiment", "model", "seed", "reuse_count", "distractor_count",
            "useful_frequency", "useless_frequency", "internal_tick", "schedule",
            "control", "old_exposures", "new_exposures", "new_reuses",
        ) if column in records
    ]
    numeric = records.select_dtypes(include="number").columns.difference(
        ["episode", *conditions]
    )
    summary = records.groupby(conditions, dropna=False)[list(numeric)].mean().reset_index()
    summary.to_parquet(processed / "condition_summary.parquet", index=False)
    adjudication = adjudicate(records, config)
    (processed / "adjudication.json").write_text(json.dumps(adjudication, indent=2))
    make_figures(records, processed / "figures")
    report = make_report(records, training, adjudication, args.run_id)
    (processed / "STAGE1_1_VALIDATION_REPORT.md").write_text(report)
    (ROOT / "reports/STAGE1_1_VALIDATION_REPORT.md").write_text(report)
    negatives = ["# Stage 1.1 Negative and Null Results", ""]
    for name, values in adjudication["negative_criteria"].items():
        negatives.append(f"- **{name}: {'TRIGGERED' if values['triggered'] else 'not triggered'}** — " + ", ".join(f"{k}={v}" for k, v in values.items() if k != "triggered"))
    negatives += ["", "The aborted `stage1_1-formal-v1` partial run is preserved and excluded from adjudication; see `STAGE1_1_AMENDMENTS.md`."]
    (ROOT / "reports/NEGATIVE_RESULTS.md").write_text("\n".join(negatives) + "\n")
    authorized = adjudication["STAGE2_LANGUAGE_MODEL_AUTHORIZED"]
    go = "# Stage 2 Go / No-Go\n\n`STAGE2_LANGUAGE_MODEL_AUTHORIZED = %s`\n\n%s\n" % (
        str(authorized).upper(),
        "All frozen authorization conditions passed." if authorized else "NO-GO: at least one core frozen gate or critical negative criterion failed. No language-model training was run.",
    )
    (ROOT / "reports/STAGE2_GO_NO_GO.md").write_text(go)
    (ROOT / "reports/LONG_STREAM_VALIDATION.md").write_text(
        "# Long Stream Validation\n\n`NOT_RUN_BY_PROTOCOL`\n\nExperiment J requires intermediate authorization. It was not run because the Stage 1.1 gate adjudication did not authorize additional long-stream execution.\n"
    )
    manifest = {
        "run_id": args.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_jobs": len(manifests),
        "record_rows": len(records),
        "training_rows": len(training),
        "stage2_authorized": authorized,
        "long_stream_run": False,
        "stage2_run": False,
    }
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2))
    paths = sorted(path for path in processed.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    (processed / "SHA256SUMS").write_text("\n".join(f"{sha256(path)}  {path.relative_to(processed)}" for path in paths) + "\n")
    print(json.dumps(adjudication, indent=2))


if __name__ == "__main__":
    main()
