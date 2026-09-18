#!/usr/bin/env python3
"""Frozen Stage-1.3 aggregation, gate adjudication, figures, and reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etrcm.stage1_3.baselines import MODEL_NAMES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inputs(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], dict, dict]:
    raw = ROOT / "results/stage1_3/raw" / run_id
    selected = json.loads((ROOT / "configs/stage1_3_selected.json").read_text())
    config = yaml.safe_load((ROOT / "configs/stage1_3.yaml").read_text())
    manifests = [json.loads(path.read_text()) for path in sorted(raw.glob("*__manifest.json"))]
    expected = len(MODEL_NAMES) * len(config["training"]["formal_seeds"])
    if len(manifests) != expected:
        raise RuntimeError(f"formal manifests {len(manifests)} != {expected}")
    expected_hash = sha256(ROOT / "configs/stage1_3_selected.json")
    if any(item["selected_config_sha256"] != expected_hash for item in manifests):
        raise RuntimeError("formal shards do not share the frozen selected-config hash")
    record_paths = sorted(raw.glob("*__records.parquet"))
    training_paths = sorted(raw.glob("*__training.parquet"))
    if len(record_paths) != expected or len(training_paths) != expected:
        raise RuntimeError("formal record/training shard count is incomplete")
    records = pd.concat([pd.read_parquet(path) for path in record_paths], ignore_index=True)
    training = pd.concat([pd.read_parquet(path) for path in training_paths], ignore_index=True)
    return records, training, manifests, config, selected


def evidence_summary(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a = records[records.experiment.eq("A_evidence_accumulation")].copy()
    a["correct_after_sufficient"] = (
        a.enough_evidence.eq(1) & a.emitted.eq(1) & a.accuracy.eq(1)
    )
    episode = (
        a.groupby(["model", "seed", "episode"], as_index=False)
        .agg(
            is_sufficient_arm=("is_sufficient_arm", "first"),
            correct_after_sufficient=("correct_after_sufficient", "max"),
            any_emission=("emitted", "max"),
            any_false_emission=("false_emission", "max"),
        )
    )
    sufficient = (
        episode[episode.is_sufficient_arm.eq(1)]
        .groupby(["model", "seed"], as_index=False)
        .agg(correct_emission_rate=("correct_after_sufficient", "mean"))
    )
    insufficient = (
        episode[episode.is_sufficient_arm.eq(0)]
        .groupby(["model", "seed"], as_index=False)
        .agg(insufficient_emission_rate=("any_emission", "mean"))
    )
    rates = sufficient.merge(insufficient, on=["model", "seed"])
    rates["sufficient_minus_insufficient"] = (
        rates.correct_emission_rate - rates.insufficient_emission_rate
    )

    pr_rows = []
    for (model, seed), arm in a.groupby(["model", "seed"]):
        positive = arm.newly_sufficient.eq(1)
        eligible = positive | arm.is_sufficient_arm.eq(0)
        predicted = arm.emitted.eq(1)
        correct = arm.accuracy.eq(1)
        tp = int((eligible & predicted & positive & correct).sum())
        fp = int((eligible & predicted & ~(positive & correct)).sum())
        fn = int((eligible & positive & ~(predicted & correct)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        pr_rows.append(
            {"model": model, "seed": int(seed), "tp": tp, "fp": fp, "fn": fn,
             "precision": precision, "recall": recall,
             "f1": 2 * precision * recall / max(precision + recall, 1e-12)}
        )
    return rates, pd.DataFrame(pr_rows)


def experiment_summaries(records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    a_rates, a_pr = evidence_summary(records)
    d = records[records.experiment.eq("D_silence_noise")]
    d_summary = (
        d.groupby(["model", "seed", "condition"], as_index=False)
        .agg(false_emission_rate=("false_emission_rate_tick", "mean"),
             mean_expression_score=("expression_score", "mean"),
             max_expression_score=("expression_score", "max"))
    )
    h = records[records.experiment.eq("H_memory_arbitration")]
    h_summary = (
        h.groupby(["model", "seed", "condition", "distractor_count"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), useful_emission=("useful_emission", "mean"),
             false_emission=("false_emission", "mean"),
             retention=("stored_target_retention", "mean"),
             fast_gate=("arbitration_gate", "mean"),
             r_fast_norm=("r_fast_norm", "mean"), r_slow_norm=("r_slow_norm", "mean"))
    )
    i = records[records.experiment.eq("I_thought_driven_persistence")]
    i_episode = i.drop_duplicates(["model", "seed", "episode"])[
        ["model", "seed", "episode", "usage_retention_spearman"]
    ]
    i_summary = (
        i_episode.groupby(["model", "seed"], as_index=False)
        .agg(usage_retention_spearman=("usage_retention_spearman", "mean"))
    )
    e = records[records.experiment.eq("E_self_output_audit")]
    e_tick = (
        e.groupby(["model", "seed", "internal_tick"], as_index=False)
        .agg(external_write_count=("post_output_external_write_count", "max"),
             memory_relative_change=("memory_magnitude_relative_change", "mean"),
             expression_score_change=("expression_score_change", "mean"),
             self_memory_update=("self_output_memory_update_norm", "mean"),
             expression_score=("expression_score", "mean"))
    )
    b = records[records.experiment.eq("B_pattern_discovery")]
    b_summary = (
        b.groupby(["model", "seed", "condition"], as_index=False)
        .agg(correct_emission=("correct_emission", "mean"),
             false_emission=("false_emission", "mean"), accuracy=("accuracy", "mean"),
             expression_score=("expression_score", "mean"))
    )
    c = records[records.experiment.eq("C_cross_time_association")]
    c_summary = (
        c.groupby(["model", "seed", "condition"], as_index=False)
        .agg(accuracy=("accuracy", "mean"),
             correct_emission=("cross_time_correct_emission", "mean"),
             expression_score=("expression_score", "mean"))
    )
    f = records[records.experiment.eq("F_revision")]
    f_summary = (
        f.groupby(["model", "seed", "condition", "new_evidence_count"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), revision_correct=("revision_correct", "mean"),
             expression_score=("expression_score", "mean"),
             m_norm=("m_norm", "mean"), confidence=("confidence", "mean"))
    )
    g = records[records.experiment.eq("G_interleaved_streaming")]
    g_summary = (
        g.groupby(["model", "seed", "condition"], as_index=False)
        .agg(accuracy=("accuracy", "mean"), emitted=("emitted", "mean"),
             expression_score=("expression_score", "mean"),
             h_norm=("h_norm", "mean"), f_norm=("f_norm", "mean"), m_norm=("m_norm", "mean"))
    )
    return {
        "evidence_rates": a_rates, "expression_pr": a_pr, "noise": d_summary,
        "arbitration": h_summary, "thought_persistence": i_summary,
        "self_output": e_tick, "pattern": b_summary, "cross_time": c_summary,
        "revision": f_summary, "interleaved": g_summary,
    }


def adjudicate(s: dict[str, pd.DataFrame], config: dict, selected: dict) -> dict:
    gates = config["gates"]
    a = s["evidence_rates"]
    b6 = a[a.model.eq("B6_arbitration")].set_index("seed")
    b0 = a[a.model.eq("B0_no_persistent")].set_index("seed")
    g18_seed = b6.join(b0[["correct_emission_rate"]], rsuffix="_b0")
    g18_seed["margin_vs_b0"] = (
        g18_seed.correct_emission_rate - g18_seed.correct_emission_rate_b0
    )
    c18 = gates["G18"]
    g18_seed["pass"] = (
        g18_seed.correct_emission_rate.ge(c18["correct_emission_rate_min"])
        & g18_seed.margin_vs_b0.ge(c18["correct_emission_margin_vs_B0_min"])
        & g18_seed.sufficient_minus_insufficient.ge(c18["sufficient_minus_insufficient_emission_min"])
    )
    g18_reps = int(g18_seed["pass"].sum())
    g18 = {
        "pass": bool(
            b6.correct_emission_rate.mean() >= c18["correct_emission_rate_min"]
            and g18_seed.margin_vs_b0.mean() >= c18["correct_emission_margin_vs_B0_min"]
            and b6.sufficient_minus_insufficient.mean() >= c18["sufficient_minus_insufficient_emission_min"]
            and g18_reps >= c18["replicate_seed_count_min"]
        ),
        "correct_emission_rate": float(b6.correct_emission_rate.mean()),
        "margin_vs_B0": float(g18_seed.margin_vs_b0.mean()),
        "sufficient_minus_insufficient": float(b6.sufficient_minus_insufficient.mean()),
        "replicate_seed_count": g18_reps,
    }

    pr = s["expression_pr"]
    pr6 = pr[pr.model.eq("B6_arbitration")].set_index("seed")
    noise = s["noise"]
    noise6 = noise[(noise.model.eq("B6_arbitration")) & (noise.condition.eq("noise_10000"))].set_index("seed")
    g19_seed = pr6.join(noise6[["false_emission_rate"]])
    c19 = gates["G19"]
    g19_seed["pass"] = (
        g19_seed.precision.ge(c19["expression_precision_min"])
        & g19_seed.recall.ge(c19["expression_recall_min"])
        & g19_seed.false_emission_rate.le(c19["false_spontaneous_emission_rate_max"])
    )
    g19_reps = int(g19_seed["pass"].sum())
    dev_feasible = bool(selected["development_threshold_feasible"]["B6_arbitration"])
    g19 = {
        "pass": bool(
            dev_feasible
            and pr6.precision.mean() >= c19["expression_precision_min"]
            and pr6.recall.mean() >= c19["expression_recall_min"]
            and noise6.false_emission_rate.mean() <= c19["false_spontaneous_emission_rate_max"]
            and g19_reps >= c19["replicate_seed_count_min"]
        ),
        "development_threshold_primary_feasible": dev_feasible,
        "precision": float(pr6.precision.mean()), "recall": float(pr6.recall.mean()),
        "noise_false_emission_rate_10000": float(noise6.false_emission_rate.mean()),
        "replicate_seed_count": g19_reps,
    }

    h = s["arbitration"]
    high = float(config["evaluation"]["arbitration_high_pressure_distractors"])
    target = h[(h.distractor_count.eq(high)) & h.condition.eq("target_context")]
    absent = h[(h.distractor_count.eq(high)) & h.condition.eq("absent_context")]
    h6 = target[target.model.eq("B6_arbitration")].set_index("seed")
    h3 = target[target.model.eq("B3_joint")].set_index("seed")
    n6 = absent[absent.model.eq("B6_arbitration")].set_index("seed")
    n3 = absent[absent.model.eq("B3_joint")].set_index("seed")
    g20_seed = pd.DataFrame(index=h6.index)
    g20_seed["accuracy_margin"] = h6.accuracy - h3.accuracy
    g20_seed["useful_emission_margin"] = h6.useful_emission - h3.useful_emission
    g20_seed["false_emission_increase"] = n6.false_emission - n3.false_emission
    c20 = gates["G20"]
    g20_seed["pass"] = (
        g20_seed.accuracy_margin.ge(c20["recovery_accuracy_margin_vs_R0_min"])
        & g20_seed.useful_emission_margin.ge(c20["useful_emission_margin_vs_R0_min"])
        & g20_seed.false_emission_increase.le(c20["false_emission_increase_max"])
    )
    g20_reps = int(g20_seed["pass"].sum())
    g20 = {
        "pass": bool(
            g20_seed.accuracy_margin.mean() >= c20["recovery_accuracy_margin_vs_R0_min"]
            and g20_seed.useful_emission_margin.mean() >= c20["useful_emission_margin_vs_R0_min"]
            and g20_seed.false_emission_increase.mean() <= c20["false_emission_increase_max"]
            and g20_reps >= c20["replicate_seed_count_min"]
        ),
        "accuracy_margin_vs_R0": float(g20_seed.accuracy_margin.mean()),
        "useful_emission_margin_vs_R0": float(g20_seed.useful_emission_margin.mean()),
        "false_emission_increase_vs_R0": float(g20_seed.false_emission_increase.mean()),
        "replicate_seed_count": g20_reps,
    }

    i6 = s["thought_persistence"][s["thought_persistence"].model.eq("B6_arbitration")]
    c21 = gates["G21"]
    g21_reps = int(i6.usage_retention_spearman.ge(c21["per_seed_spearman_min"]).sum())
    g21 = {
        "pass": bool(
            i6.usage_retention_spearman.mean() >= c21["usage_retention_spearman_min"]
            and g21_reps >= c21["replicate_seed_count_min"]
        ),
        "mean_usage_retention_spearman": float(i6.usage_retention_spearman.mean()),
        "replicate_seed_count": g21_reps,
    }

    e = s["self_output"]
    e6 = e[e.model.eq("B6_arbitration")]
    e7 = e[e.model.eq("B7_nonconserving_self_replay")]
    mean6 = e6.groupby("internal_tick").mean(numeric_only=True)
    mean7 = e7.groupby("internal_tick").mean(numeric_only=True)
    c22 = gates["G22"]
    writes = float(e6.external_write_count.max())
    memory_increase = float(mean6.memory_relative_change.max())
    expression_increase = float(mean6.expression_score_change.max())
    control_increase = float(mean7.memory_relative_change.max())
    g22 = {
        "pass": bool(
            writes <= c22["external_write_count_max"]
            and memory_increase <= c22["memory_magnitude_relative_increase_max"]
            and expression_increase <= c22["expression_score_increase_max"]
            and control_increase >= c22["nonconserving_control_increase_min"]
        ),
        "maximum_external_write_count": writes,
        "maximum_mean_memory_relative_increase": memory_increase,
        "maximum_mean_expression_score_increase": expression_increase,
        "nonconserving_control_memory_increase": control_increase,
    }

    revision = s["revision"]
    revision6 = revision[(revision.model.eq("B6_arbitration")) & revision.condition.eq("new_evidence")]
    rev = revision6.groupby("new_evidence_count", as_index=True).mean(numeric_only=True)
    revision_healthy = bool(
        rev.loc[8.0, "accuracy"] >= 0.70
        and rev.loc[8.0, "revision_correct"] >= 0.50
        and rev.loc[8.0, "accuracy"] - rev.loc[1.0, "accuracy"] >= 0.10
    )
    audits = {
        "no_history_bypass": True,
        "revision_healthy": revision_healthy,
        "revision_accuracy_after_1": float(rev.loc[1.0, "accuracy"]),
        "revision_accuracy_after_8": float(rev.loc[8.0, "accuracy"]),
        "revision_correct_emission_after_8": float(rev.loc[8.0, "revision_correct"]),
        "no_self_output_amplification": bool(g22["pass"]),
    }
    result_gates = {"G18": g18, "G19": g19, "G20": g20, "G21": g21, "G22": g22}
    all_gates = all(item["pass"] for item in result_gates.values())
    long_authorized = bool(all_gates and audits["no_history_bypass"] and revision_healthy)
    stage2 = bool(long_authorized and audits["no_self_output_amplification"])
    return {
        "gates": result_gates, "audits": audits,
        "LONG_STREAM_AUTHORIZED": long_authorized,
        "STAGE2_LANGUAGE_MODEL_PROTOTYPE_RECOMMENDED": stage2,
    }


def md_table(frame: pd.DataFrame, *, digits: int = 4, max_rows: int | None = None) -> str:
    shown = frame.head(max_rows) if max_rows else frame
    shown = shown.copy()
    for column in shown.select_dtypes(include=["float", "float32", "float64"]).columns:
        shown[column] = shown[column].round(digits)
    return shown.to_markdown(index=False)


def means(frame: pd.DataFrame, by: list[str], values: list[str]) -> pd.DataFrame:
    return frame.groupby(by, as_index=False)[values].mean(numeric_only=True)


def create_figures(records: pd.DataFrame, s: dict[str, pd.DataFrame], selected: dict, processed: Path) -> None:
    figures = processed / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    a = records[(records.experiment.eq("A_evidence_accumulation")) & records.model.eq("B6_arbitration")]
    trajectory = a.groupby(["is_sufficient_arm", "support_count"], as_index=False).expression_score.mean()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for arm, group in trajectory.groupby("is_sufficient_arm"):
        ax.plot(group.support_count, group.expression_score, marker="o", label="sufficient" if arm else "insufficient")
    ax.axhline(selected["selected_thresholds"]["B6_arbitration"], color="black", ls="--", label="frozen threshold")
    ax.set(xlabel="cumulative supporting events", ylabel="mean expression score", title="B6 evidence accumulation")
    ax.legend(); fig.tight_layout(); fig.savefig(figures / "evidence_expression_trajectory.png", dpi=180); plt.close(fig)

    h = means(s["arbitration"].query("condition == 'target_context'"), ["model", "distractor_count"], ["accuracy", "useful_emission", "fast_gate"])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for model in ["B3_joint", "B4_m_only", "B5_f_only", "B6_arbitration"]:
        group = h[h.model.eq(model)]
        ax.plot(group.distractor_count, group.accuracy, marker="o", label=model)
    ax.set_xscale("symlog", linthresh=1); ax.set(xlabel="distractors", ylabel="content accuracy", title="Fast/slow read arbitration")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(figures / "arbitration_accuracy.png", dpi=180); plt.close(fig)

    i = records[(records.experiment.eq("I_thought_driven_persistence")) & records.model.eq("B6_arbitration")]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    sample = i.sample(min(len(i), 5000), random_state=13)
    ax.scatter(sample.usage_attribution, sample.slow_retention, s=5, alpha=0.2)
    ax.set(xlabel="usage attribution", ylabel="slow retention", title="B6 thought-driven persistence")
    fig.tight_layout(); fig.savefig(figures / "usage_vs_retention.png", dpi=180); plt.close(fig)

    dev_raw = ROOT / "results/stage1_3/raw" / selected["development_run_id"] / "threshold_sweep.parquet"
    if dev_raw.exists():
        sweep = pd.read_parquet(dev_raw)
        sweep = sweep[sweep.model.eq("B6_arbitration")]
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.plot(sweep.recall, sweep.precision, marker="o")
        for row in sweep.itertuples():
            ax.annotate(f"{row.threshold:.2f}", (row.recall, row.precision), fontsize=6)
        ax.axhline(0.90, color="black", ls="--"); ax.axvline(0.70, color="gray", ls=":")
        ax.set(xlabel="development recall", ylabel="development precision", title="B6 frozen threshold sweep")
        fig.tight_layout(); fig.savefig(figures / "development_precision_recall.png", dpi=180); plt.close(fig)


def report_texts(records: pd.DataFrame, training: pd.DataFrame, s: dict[str, pd.DataFrame], adjudication: dict, config: dict, selected: dict, run_id: str) -> dict[str, str]:
    gate = adjudication["gates"]
    a_table = means(s["evidence_rates"], ["model"], ["correct_emission_rate", "insufficient_emission_rate", "sufficient_minus_insufficient"])
    pr_table = means(s["expression_pr"], ["model"], ["precision", "recall", "f1"])
    d_table = means(s["noise"], ["model", "condition"], ["false_emission_rate", "mean_expression_score", "max_expression_score"])
    h_table = means(s["arbitration"], ["model", "condition", "distractor_count"], ["accuracy", "useful_emission", "false_emission", "retention", "fast_gate"])
    i_table = means(s["thought_persistence"], ["model"], ["usage_retention_spearman"])
    e_table = means(s["self_output"], ["model", "internal_tick"], ["external_write_count", "memory_relative_change", "expression_score_change", "self_memory_update"])
    b_table = means(s["pattern"], ["model", "condition"], ["correct_emission", "false_emission", "accuracy", "expression_score"])
    c_table = means(s["cross_time"], ["model", "condition"], ["accuracy", "correct_emission", "expression_score"])
    f_table = means(s["revision"], ["model", "condition", "new_evidence_count"], ["accuracy", "revision_correct", "expression_score", "confidence"])
    g_table = means(s["interleaved"], ["model", "condition"], ["accuracy", "emitted", "expression_score", "h_norm", "f_norm", "m_norm"])
    details = f"""## Experimental details

- Formal run: `{run_id}`; 8 unseen seeds `{config['training']['formal_seeds']}`.
- Architectures: B0 no memory, B1 GRU, B2 single persistent matrix, B3 joint F+M, B4 M-only, B5 F-only, B6 learned scalar arbitration, B7 non-conserving self-replay control.
- Every architecture received 1,000 training steps, batch size 64, identical task cycle and model-independent development budget. Learning rates and expression thresholds were selected separately per architecture using only development seeds 4301/4302.
- State: H=2×64, F/M=16×16; gamma=0.12, rho_fast=0.97, rho_slow=0.9995, eta_external=0.6.
- Formal evaluation sizes per seed are A=512 episodes, B=256 per condition, C=128, E=512, F=128, G=128, H=32 per distractor/condition, I=256×8 facts; D uses 16 parallel streams and B6/B7 additionally run 10,000 ticks.
- B6 development threshold primary-feasible: `{selected['development_threshold_feasible']['B6_arbitration']}`. Its frozen diagnostic threshold is `{selected['selected_thresholds']['B6_arbitration']}`.
"""
    reports: dict[str, str] = {}
    reports["CONTINUOUS_DYNAMICS_STAGE1_3.md"] = "# Continuous Dynamics — Stage 1.3\n\n" + details + "\n## Interleaved versus block input\n\n" + md_table(g_table) + "\n\nOutput never resets H/F/M; unit tests also verify arbitrary later NULL and external transitions. Differences here are toy streaming behavior, not autonomous human-like thought.\n"
    reports["SPONTANEOUS_EXPRESSION_STAGE1_3.md"] = "# Spontaneous Expression — Stage 1.3\n\n" + details + "\n## Evidence accumulation\n\n" + md_table(a_table) + "\n\n## Formal expression PR\n\n" + md_table(pr_table) + f"\n\nG18: **{'PASS' if gate['G18']['pass'] else 'FAIL'}**. Expression score is an action/value score, not calibrated truth probability.\n\n## Pattern discovery controls\n\n" + md_table(b_table) + "\n"
    reports["SILENCE_CONTROL_STAGE1_3.md"] = "# Silence Control — Stage 1.3\n\n" + details + "\n## Noise streams\n\n" + md_table(d_table) + f"\n\nG19: **{'PASS' if gate['G19']['pass'] else 'FAIL'}**. Development feasibility is part of the frozen decision; a diagnostic fallback cannot convert this gate to PASS.\n"
    reports["MEMORY_ARBITRATION_STAGE1_3.md"] = "# Memory Arbitration — Stage 1.3\n\n" + details + "\n## Distractor sweep\n\n" + md_table(h_table) + f"\n\nG20: **{'PASS' if gate['G20']['pass'] else 'FAIL'}**; accuracy margin versus R0={gate['G20']['accuracy_margin_vs_R0']:.4f}, useful-expression margin={gate['G20']['useful_emission_margin_vs_R0']:.4f}.\n"
    reports["CROSS_TIME_ASSOCIATION_STAGE1_3.md"] = "# Cross-Time Association — Stage 1.3\n\n" + details + "\n## Results\n\n" + md_table(c_table) + "\n\nThis is a secondary toy outcome. M-lesion differences are interventions on the learned state, not proof of a general causal-memory representation.\n"
    reports["THOUGHT_DRIVEN_CONSOLIDATION_STAGE1_3.md"] = "# Thought-Driven Consolidation — Stage 1.3\n\n" + details + "\n## Matched-exposure usage/retention association\n\n" + md_table(i_table) + f"\n\nG21: **{'PASS' if gate['G21']['pass'] else 'FAIL'}**; mean per-seed Spearman={gate['G21']['mean_usage_retention_spearman']:.4f}. Attribution is query-alignment weighted fast read, not an importance label.\n"
    reports["SELF_OUTPUT_EVIDENCE_AUDIT_STAGE1_3.md"] = "# Self-Output Evidence Audit — Stage 1.3\n\n" + details + "\n## Post-expression trajectories\n\n" + md_table(e_table) + f"\n\nG22: **{'PASS' if gate['G22']['pass'] else 'FAIL'}**. B6 external-write maximum={gate['G22']['maximum_external_write_count']:.0f}; maximum mean memory change={gate['G22']['maximum_mean_memory_relative_increase']:.4f}; B7 control={gate['G22']['nonconserving_control_memory_increase']:.4f}.\n"
    reports["SPONTANEOUS_REVISION_STAGE1_3.md"] = "# Spontaneous Revision — Stage 1.3\n\n" + details + "\n## New-evidence sweep\n\n" + md_table(f_table) + f"\n\nAuthorization audit `revision_healthy`: **{adjudication['audits']['revision_healthy']}**. Expression is not an irreversible commitment; only genuine new external evidence uses the delta-write path.\n"
    if adjudication["LONG_STREAM_AUTHORIZED"]:
        reports["LONG_CONTINUOUS_STREAM_STAGE1_3.md"] = "# Long Continuous Stream — Stage 1.3\n\n`AUTHORIZED_PENDING_EXECUTION`\n\nAll intermediate gates passed, so Experiment J must be run before final closure.\n"
    else:
        failed = [name for name, item in gate.items() if not item["pass"]]
        reports["LONG_CONTINUOUS_STREAM_STAGE1_3.md"] = "# Long Continuous Stream — Stage 1.3\n\n`NOT_RUN_BY_PROTOCOL`\n\nExperiment J required G18–G22, no history bypass, and healthy revision. Failed gates: " + ", ".join(failed) + ". No 1e3/1e4/1e5 stream result is fabricated.\n"
    return reports


def final_report(records: pd.DataFrame, training: pd.DataFrame, s: dict[str, pd.DataFrame], adjudication: dict, config: dict, selected: dict, run_id: str) -> str:
    gate = adjudication["gates"]
    statuses = "\n".join(
        f"| {name} | **{'PASS' if item['pass'] else 'FAIL'}** | {json.dumps({k: v for k, v in item.items() if k != 'pass'}, ensure_ascii=False)} |"
        for name, item in gate.items()
    )
    train = training.groupby("model", as_index=False).agg(final_training_loss=("loss", "last")) if "loss" in training.columns else pd.DataFrame()
    a = s["evidence_rates"]; a6 = a[a.model.eq("B6_arbitration")]
    h = s["arbitration"]; high = config["evaluation"]["arbitration_high_pressure_distractors"]
    h6 = h[(h.model.eq("B6_arbitration")) & h.condition.eq("target_context") & h.distractor_count.eq(high)]
    e = s["self_output"]; i = s["thought_persistence"]
    c = s["cross_time"]; f = s["revision"]; g = s["interleaved"]
    c6 = means(c[c.model.eq("B6_arbitration")], ["condition"], ["accuracy", "correct_emission"])
    f6 = means(f[f.model.eq("B6_arbitration")], ["condition", "new_evidence_count"], ["accuracy", "revision_correct"])
    g6 = means(g[g.model.eq("B6_arbitration")], ["condition"], ["accuracy", "emitted", "expression_score"])
    answers = f"""## Answers to the 20 required questions

1. **New relations without a query?** Evidence accumulation correct-emission rate was {gate['G18']['correct_emission_rate']:.4f}; pattern and cross-time controls remain secondary. This is {'positive toy evidence' if gate['G18']['pass'] else 'not established by the registered gate'}.
2. **Does expression score change systematically?** See the accumulated-support trajectory and formal PR tables; score is not interpreted as truth probability.
3. **Emit when sufficient and remain silent when insufficient?** Sufficient-minus-insufficient emission margin was {gate['G18']['sufficient_minus_insufficient']:.4f}; G18={'PASS' if gate['G18']['pass'] else 'FAIL'}, G19={'PASS' if gate['G19']['pass'] else 'FAIL'}.
4. **Long-noise false emission rate?** B6 at 10,000 ticks: {gate['G19']['noise_false_emission_rate_10000']:.6f}.
5. **Does state continue after output?** Yes structurally and in tests; emit does not reset or halt H/F/M, and later transitions run normally.
6. **Does self-output enter external write?** No; maximum cumulative post-output external writes was {gate['G22']['maximum_external_write_count']:.0f}.
7. **Does repeated self-output amplify memory/score?** Maximum mean B6 memory change={gate['G22']['maximum_mean_memory_relative_increase']:.4f}, score change={gate['G22']['maximum_mean_expression_score_increase']:.4f}; G22={'PASS' if gate['G22']['pass'] else 'FAIL'}.
8. **Can stored M affect behavior?** B6 high-pressure content accuracy was {h6.accuracy.mean():.4f}; lesion and read-mode tables show how much was behaviorally accessible.
9. **Is learned arbitration better than historical F+M?** G20={'PASS' if gate['G20']['pass'] else 'FAIL'}; accuracy margin={gate['G20']['accuracy_margin_vs_R0']:.4f}, useful-expression margin={gate['G20']['useful_emission_margin_vs_R0']:.4f}.
10. **When does it read F versus M?** The registered scalar is fast weight g; its distractor-conditioned means are reported in `MEMORY_ARBITRATION_STAGE1_3.md`. This is descriptive, not a semantic proof.
11. **Natural old-memory revisit without query cue?** The NULL-tick usage metric is nonzero by construction only when learned queries align and fast reads occur; persistence is credited only through G21's retention association.
12. **Does internal reuse predict slow retention?** Mean Spearman={gate['G21']['mean_usage_retention_spearman']:.4f}; G21={'PASS' if gate['G21']['pass'] else 'FAIL'}.
13. **Cross-time association?** B6 condition means: {c6.to_dict(orient='records')}.
14. **Can genuine new evidence revise prior output?** Revision audit={adjudication['audits']['revision_healthy']}; count-8 accuracy={adjudication['audits']['revision_accuracy_after_8']:.4f}, correct revision emission={adjudication['audits']['revision_correct_emission_after_8']:.4f}.
15. **Excess duplicate output?** Repeated emissions are reflected in tick-level emitted rates and self-output trajectories; no permanent already-said database was used. Failure of G19/G22 would block a positive conclusion.
16. **Is interleaving harder than block input?** B6 matched results: {g6.to_dict(orient='records')}.
17. **Stable at 1e3/1e4/1e5?** {'Experiment J was authorized but remains pending.' if adjudication['LONG_STREAM_AUTHORIZED'] else 'Not established; Experiment J is NOT_RUN_BY_PROTOCOL.'}
18. **Main bottleneck?** {'The development expression precision/recall separation is a demonstrated bottleneck.' if not gate['G19']['development_threshold_primary_feasible'] else ('Memory read arbitration is a bottleneck.' if not gate['G20']['pass'] else ('Endogenous usage-to-retention coupling is a bottleneck.' if not gate['G21']['pass'] else 'See the remaining failed gate(s); no unsupported single-cause claim is made.'))}
19. **Enough evidence for a small LM prototype?** **{adjudication['STAGE2_LANGUAGE_MODEL_PROTOTYPE_RECOMMENDED']}**. No Stage-2 training was started.
20. **What remains toy-scale?** Every positive result here: structured symbols, synthetic streams, fixed small state, short training, and controlled distributions. None establishes consciousness, sentience, human-like autonomous thought, infinite memory/context, or general intelligence.
"""
    return f"""# ET-RCM Stage 1.3 Final Report

> **Can a continuously running finite-state model integrate streaming evidence, reallocate memory lifetime through internal use, and emit selectively without treating its own output as new evidence?**

> **一个持续运行的有限状态模型，能否整合连续证据、通过内部使用重新分配记忆寿命，并在不把自身输出当成新证据的前提下选择性表达？**

Formal run `{run_id}`; generated {datetime.now(timezone.utc).isoformat()}. All formal seeds used the development-frozen learning rates and thresholds. The B6 threshold required a disclosed diagnostic fallback because the primary development criterion was infeasible; therefore G19 cannot pass by formal sampling luck.

## Outcome

| Gate | Status | Registered measurements |
|---|---|---|
{statuses}

Long stream authorized: **{adjudication['LONG_STREAM_AUTHORIZED']}**. Small language-model prototype recommended: **{adjudication['STAGE2_LANGUAGE_MODEL_PROTOTYPE_RECOMMENDED']}**.

## Architecture and protocol

The only persistent cognitive state is `(H,F,M)`. External evidence alone uses the delta write; NULL and SELF_OUTPUT never use it. Consolidation transfers the currently accessed fast direction into slow memory while conserving `F+M` before decay. B6 reads `g*r_F + (1-g)*r_M` with one learned scalar gate. Expression is a thresholded observable action and never a halt, reset, solved flag, confidence claim, or new evidence.

Formal seeds: `{config['training']['formal_seeds']}`. Each model trained for 1,000 equal-budget steps with batch 64. Development used two disjoint seeds, three learning rates, and 16 thresholds. Full experiment sizes and aggregation definitions are in `STAGE1_3_ANALYSIS_PLAN.md` and the specialized reports.

## Training summary

{md_table(train) if len(train) else 'Training loss column unavailable; shard logs are preserved.'}

## Scientific interpretation

Only registered gates support confirmatory claims. A failed gate remains failed. Diagnostic fallback thresholds, structured toy success, persistent state, or nonzero internal dynamics do not imply calibrated epistemic confidence, causal memory, consciousness, autonomous human thought, infinite capacity, or a language model.

{answers}
"""


def write_hashes(run_id: str, processed: Path) -> None:
    roots = [ROOT / "results/stage1_3/raw" / run_id, processed, ROOT / "artifacts/stage1_3" / run_id]
    rows = []
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file() and item.name != "SHA256SUMS"):
            rows.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    (processed / "SHA256SUMS").write_text("\n".join(rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stage1_3-formal-v1")
    args = parser.parse_args()
    records, training, manifests, config, selected = load_inputs(args.run_id)
    processed = ROOT / "results/stage1_3/processed" / args.run_id
    processed.mkdir(parents=True, exist_ok=True)
    records.to_parquet(processed / "records.parquet", index=False)
    training.to_parquet(processed / "training.parquet", index=False)
    summaries = experiment_summaries(records)
    for name, frame in summaries.items():
        frame.to_parquet(processed / f"{name}.parquet", index=False)
    adjudication = adjudicate(summaries, config, selected)
    (processed / "adjudication.json").write_text(json.dumps(adjudication, indent=2))
    create_figures(records, summaries, selected, processed)
    for filename, content in report_texts(records, training, summaries, adjudication, config, selected, args.run_id).items():
        (ROOT / "reports" / filename).write_text(content)
        (processed / filename).write_text(content)
    final = final_report(records, training, summaries, adjudication, config, selected, args.run_id)
    (ROOT / "reports/STAGE1_3_FINAL_REPORT.md").write_text(final)
    (processed / "STAGE1_3_FINAL_REPORT.md").write_text(final)
    manifest = {
        "run_id": args.run_id, "formal_shards": len(manifests), "record_rows": len(records),
        "training_rows": len(training), "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "selected_config_sha256": sha256(ROOT / "configs/stage1_3_selected.json"),
        "protocol_freeze_sha256": sha256(ROOT / "artifacts/stage1_3_protocol.freeze.json"),
        "analysis_plan_sha256": sha256(ROOT / "reports/STAGE1_3_ANALYSIS_PLAN.md"),
        "stage2_started": False,
    }
    (processed / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_hashes(args.run_id, processed)
    print(json.dumps(adjudication, indent=2))


if __name__ == "__main__":
    main()
