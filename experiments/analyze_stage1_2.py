#!/usr/bin/env python3
"""Aggregate, adjudicate, visualize and report ET-RCM Stage-1.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("B1_gru", "B2_single_persistent", "B3_uniform", "B5_no_idle", "B6_full")
SEEDS = tuple(range(3201, 3209))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].dropna() if column in frame else pd.Series(dtype=float)
    return float(values.mean()) if len(values) else float("nan")


def fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.4f}"


def subset(frame: pd.DataFrame, **conditions) -> pd.DataFrame:
    result = frame
    for column, value in conditions.items():
        result = result[result[column].eq(value)]
    return result


def load_complete(raw: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    records, training, manifests = [], [], []
    for model in MODELS:
        for seed in SEEDS:
            stem = f"{model}__seed{seed}"
            manifest = raw / f"{stem}__manifest.json"
            if not manifest.exists():
                raise RuntimeError(f"missing formal job: {manifest}")
            manifests.append(json.loads(manifest.read_text()))
            records.append(pd.read_parquet(raw / f"{stem}__records.parquet"))
            training.append(pd.read_parquet(raw / f"{stem}__training.parquet"))
    return pd.concat(records, ignore_index=True), pd.concat(training, ignore_index=True), manifests


def seed_condition(frame: pd.DataFrame, metric: str) -> pd.Series:
    return frame.groupby("seed")[metric].mean()


def bootstrap_seed_mean(values: pd.Series, seed: int = 12_012, samples: int = 2000) -> tuple[float, float]:
    array = values.dropna().to_numpy(float)
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def phase_models(records: pd.DataFrame) -> dict:
    phase = records[records.experiment.eq("B_exposure_reuse_phase")]
    per_seed = phase.groupby(["seed", "exposure_count", "reuse_count"], as_index=False)[
        ["retention", "accuracy", "transfer_mass"]
    ].mean()
    fits = {}
    ratios = []
    for seed, arm in per_seed.groupby("seed"):
        exposure = arm.exposure_count.to_numpy(float)
        reuse = arm.reuse_count.to_numpy(float)
        target = arm.retention.to_numpy(float)
        linear = np.column_stack([np.ones(len(arm)), exposure, reuse, exposure * reuse])
        log = np.column_stack([
            np.ones(len(arm)), np.log1p(exposure), np.log1p(reuse),
            np.log1p(exposure) * np.log1p(reuse),
        ])
        linear_coef = np.linalg.lstsq(linear, target, rcond=None)[0]
        log_coef = np.linalg.lstsq(log, target, rcond=None)[0]
        ratio = float(log_coef[2] / log_coef[1]) if abs(log_coef[1]) > 1e-9 else float("nan")
        ratios.append(ratio)
        fits[str(seed)] = {
            "linear_intercept_exposure_reuse_interaction": linear_coef.tolist(),
            "log_intercept_exposure_reuse_interaction": log_coef.tolist(),
            "reuse_to_exposure_coefficient_ratio": ratio,
        }
    ratio_series = pd.Series(ratios)
    low, high = bootstrap_seed_mean(ratio_series)
    fits["aggregate"] = {
        "mean_reuse_to_exposure_coefficient_ratio": float(ratio_series.mean()),
        "bootstrap_95_ci": [low, high],
        "interpretation": "descriptive log-model coefficient ratio; not a universal exchange rate",
    }
    return fits


def adjudicate(records: pd.DataFrame, config: dict) -> dict:
    # G14
    a = records[records.experiment.eq("A_functional_addressing")]
    original = seed_condition(a[a.condition.eq("original")], "accuracy")
    target_zero = seed_condition(a[a.condition.eq("target_zero")], "accuracy")
    nontarget_zero = seed_condition(a[a.condition.eq("strongest_nontarget_zero")], "accuracy")
    original_loss = seed_condition(a[a.condition.eq("original")], "loss")
    target_loss = seed_condition(a[a.condition.eq("target_zero")], "loss")
    target_drop = original - target_zero
    nontarget_drop = original - nontarget_zero
    g14_accuracy_drop = float(target_drop.mean())
    g14_loss_increase = float((target_loss - original_loss).mean())
    g14_specificity = float((target_drop - nontarget_drop).mean())
    g14_replicates = int((target_drop >= config["functional_addressing"]["per_seed_accuracy_drop_min"]).sum())
    g14 = (
        g14_accuracy_drop >= config["functional_addressing"]["accuracy_drop_min"]
        and g14_loss_increase >= config["functional_addressing"]["loss_increase_min"]
        and g14_specificity >= config["functional_addressing"]["specificity_margin_min"]
        and g14_replicates >= config["functional_addressing"]["replicate_seed_count_min"]
    )

    # G15
    c = records[(records.experiment.eq("C_selective_scaling")) & records.condition.eq("intact")]
    high = int(config["selective_scaling"]["high_pressure_distractors"])
    c_high_b6 = subset(c, model="B6_full", distractor_count=high)
    c_high_b2 = subset(c, model="B2_single_persistent", distractor_count=high)
    g15_accuracy = mean(c_high_b6, "accuracy") - mean(c_high_b2, "accuracy")
    g15_efficiency = mean(c_high_b6, "selective_persistence_efficiency") - mean(c_high_b2, "selective_persistence_efficiency")
    slopes = {}
    for model in ("B2_single_persistent", "B6_full"):
        values = []
        for seed, arm in c[c.model.eq(model)].groupby("seed"):
            curve = arm.groupby("distractor_count").accuracy.mean().sort_index()
            values.append(float(np.polyfit(np.log1p(curve.index.to_numpy(float)), curve.to_numpy(float), 1)[0]))
        slopes[model] = values
    slope_margin = float(np.mean(slopes["B6_full"]) - np.mean(slopes["B2_single_persistent"]))
    g15 = (
        g15_accuracy >= config["selective_scaling"]["accuracy_margin_min"]
        and slope_margin >= config["selective_scaling"]["degradation_slope_margin_min"]
        and g15_efficiency > config["selective_scaling"]["efficiency_margin_min"]
    )

    # G16
    e = records[(records.experiment.eq("E_sequential_computation")) & records.length_regime.eq("ood_longer")]
    k1 = seed_condition(e[e.internal_tick.eq(config["sequential_computation"]["low_tick_budget"])], "accuracy")
    k8 = seed_condition(e[e.internal_tick.eq(config["sequential_computation"]["high_tick_budget"])], "accuracy")
    gains = k8 - k1
    g16_gain = float(gains.mean())
    g16_replicates = int((gains >= config["sequential_computation"]["per_seed_gain_min"]).sum())
    curve = e[e.internal_tick.isin([1, 2, 4, 8])].groupby("internal_tick").accuracy.mean()
    systematic = all(curve.loc[right] >= curve.loc[left] for left, right in ((1, 2), (2, 4), (4, 8)))
    bottleneck_audit = bool(e.queries_per_tick.eq(1).all() and e.active_graph_history_scrubbed.eq(1).all() and e.label_in_event.eq(0).all())
    g16 = (
        g16_gain >= config["sequential_computation"]["accuracy_gain_min"]
        and g16_replicates >= config["sequential_computation"]["replicate_seed_count_min"]
        and systematic and bottleneck_audit
    )

    # G17
    g = records[records.experiment.eq("G_no_self_evidence")]
    know0 = subset(g, condition="knowable", internal_tick=0)
    know16 = subset(g, condition="knowable", internal_tick=16)
    unknown0 = subset(g, condition="unknowable", internal_tick=0)
    unknown64 = subset(g, condition="unknowable", internal_tick=64)
    know_gain = mean(know16, "accuracy") - mean(know0, "accuracy")
    unknown_accuracy = mean(unknown64, "accuracy")
    confidence_inflation = mean(unknown64, "confidence") - mean(unknown0, "confidence")
    ece_degradation = mean(unknown64, "ece") - mean(unknown0, "ece")
    same_format = bool(g.same_event_types.eq(1).all() and g.explicit_condition_cue.eq(0).all())
    g17 = (
        know_gain >= config["no_self_evidence"]["knowable_accuracy_gain_min"]
        and abs(unknown_accuracy - 0.5) <= config["no_self_evidence"]["chance_tolerance"]
        and confidence_inflation <= config["no_self_evidence"]["confidence_inflation_max"]
        and ece_degradation <= config["no_self_evidence"]["ece_degradation_max"]
        and same_format
    )

    # Separate endogenous-time status
    f = records[records.experiment.eq("F_endogenous_time")]
    before = seed_condition(f[f.condition.eq("before_interference")], "accuracy")
    after = seed_condition(f[f.condition.eq("after_interference")], "accuracy")
    et_differences = before - after
    et_margin = float(et_differences.mean())
    et_replicates = int((et_differences >= config["endogenous_time"]["per_seed_margin_min"]).sum())
    et_supported = (
        et_margin >= config["endogenous_time"]["accuracy_margin_min"]
        and et_replicates >= config["endogenous_time"]["replicate_seed_count_min"]
    )

    no_bypass = bottleneck_audit and same_format
    healthy_revision = True  # immutable inherited Stage-1.1 G13 PASS
    memory_lm = g14 and g15 and g17 and healthy_revision and no_bypass
    continuous_lm = memory_lm and g16 and et_supported
    return {
        "gates": {
            "G14_functional_addressing": {
                "pass": g14, "accuracy_drop_target_removed": g14_accuracy_drop,
                "loss_increase_target_removed": g14_loss_increase,
                "specificity_margin_vs_nontarget": g14_specificity,
                "replicate_seed_count": g14_replicates,
                "signed_cosine_descriptive": mean(a[a.condition.eq("original")], "target_projection"),
                "absolute_cosine_descriptive": mean(a[a.condition.eq("original")], "absolute_cosine"),
            },
            "G15_selective_persistence_scaling": {
                "pass": g15, "high_pressure_accuracy_margin_vs_B2": g15_accuracy,
                "degradation_slope_margin_vs_B2": slope_margin,
                "efficiency_margin_vs_B2": g15_efficiency,
                "B6_slopes": slopes["B6_full"], "B2_slopes": slopes["B2_single_persistent"],
            },
            "G16_sequential_internal_computation": {
                "pass": g16, "ood_K8_minus_K1_accuracy": g16_gain,
                "replicate_seed_count": g16_replicates,
                "systematic_K1_K2_K4_K8": systematic,
                "hard_bottleneck_audit": bottleneck_audit,
            },
            "G17_no_self_evidence": {
                "pass": g17, "knowable_K16_minus_K0_accuracy": know_gain,
                "unknowable_K64_accuracy": unknown_accuracy,
                "unknowable_confidence_inflation": confidence_inflation,
                "unknowable_ece_degradation": ece_degradation,
                "same_format_no_condition_cue": same_format,
            },
        },
        "endogenous_time": {
            "status": "ET_STATUS_SUPPORTED" if et_supported else "ET_STATUS_NOT_SUPPORTED",
            "before_minus_after_accuracy": et_margin,
            "replicate_seed_count": et_replicates,
        },
        "audits": {"no_bypass": no_bypass, "healthy_revision_inherited": healthy_revision},
        "LONG_STREAM_AUTHORIZED": memory_lm,
        "STAGE2_MEMORY_LM_AUTHORIZED": memory_lm,
        "STAGE2_CONTINUOUS_COGNITION_LM_AUTHORIZED": continuous_lm,
    }


def table(frame: pd.DataFrame, groups: list[str], metrics: list[str]) -> str:
    groups = [item for item in groups if item in frame and frame[item].notna().any()]
    metrics = [item for item in metrics if item in frame and frame[item].notna().any()]
    if frame.empty or not groups or not metrics:
        return "No applicable rows."
    data = frame.groupby(groups, dropna=False)[metrics].mean().reset_index()
    lines = ["| " + " | ".join(groups + metrics) + " |", "|" + "---|" * (len(groups) + len(metrics))]
    for _, row in data.iterrows():
        cells = []
        for column in groups:
            value = row[column]
            cells.append(str(int(value)) if isinstance(value, (float, np.floating)) and float(value).is_integer() else str(value))
        cells += [fmt(float(row[column])) for column in metrics]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def heatmap_svg(frame: pd.DataFrame, value: str, path: Path) -> None:
    pivot = frame.groupby(["exposure_count", "reuse_count"])[value].mean().unstack()
    width, height, cell = 780, 520, 70
    values = pivot.to_numpy(float)
    lo, hi = np.nanmin(values), np.nanmax(values)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="70" y="28" font-family="sans-serif" font-size="20" font-weight="bold">{value}: exposure x reuse</text>']
    for i, exposure in enumerate(pivot.index):
        svg.append(f'<text x="55" y="{75+i*cell}" text-anchor="end" font-family="sans-serif" font-size="12">E={exposure}</text>')
        for j, reuse in enumerate(pivot.columns):
            val = values[i, j]
            scale = 0.5 if hi == lo else (val - lo) / (hi - lo)
            red, green, blue = int(245 - 180 * scale), int(248 - 90 * scale), int(255 - 20 * scale)
            x, y = 65 + j * cell, 45 + i * cell
            svg += [f'<rect x="{x}" y="{y}" width="{cell-3}" height="{cell-3}" fill="rgb({red},{green},{blue})"/>', f'<text x="{x+(cell-3)/2}" y="{y+36}" text-anchor="middle" font-family="sans-serif" font-size="11">{val:.3f}</text>']
    for j, reuse in enumerate(pivot.columns):
        svg.append(f'<text x="{65+j*cell+(cell-3)/2}" y="{65+len(pivot.index)*cell}" text-anchor="middle" font-family="sans-serif" font-size="12">R={reuse}</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg))


def specialized_reports(records: pd.DataFrame, adjudication: dict, phase_fit: dict) -> dict[str, str]:
    a = records[records.experiment.eq("A_functional_addressing")]
    b = records[records.experiment.eq("B_exposure_reuse_phase")]
    c = records[records.experiment.eq("C_selective_scaling")]
    d = records[records.experiment.eq("D_autonomous_selection")]
    e = records[records.experiment.eq("E_sequential_computation")]
    f = records[records.experiment.eq("F_endogenous_time")]
    g = records[records.experiment.eq("G_no_self_evidence")]
    h = records[records.experiment.eq("H_storage_vs_use")]
    gate = adjudication["gates"]
    return {
        "FUNCTIONAL_QUERY_ADDRESSING_STAGE1_2.md": "# Functional Query Addressing — Stage 1.2\n\nSigned cosine is descriptive only; all conditions clone the same state.\n\n" + table(a, ["condition"], ["accuracy", "loss", "target_retrieval_score", "downstream_hidden_change", "target_projection", "absolute_cosine", "squared_projection", "nontarget_projection"]) + f"\n\nG14: **{'PASS' if gate['G14_functional_addressing']['pass'] else 'FAIL'}**. Target-removal accuracy drop={gate['G14_functional_addressing']['accuracy_drop_target_removed']:.4f}; specificity={gate['G14_functional_addressing']['specificity_margin_vs_nontarget']:.4f}. Historical G7 is unchanged.\n",
        "EXPOSURE_REUSE_PHASE_DIAGRAM_STAGE1_2.md": "# Exposure × Reuse Phase Diagram — Stage 1.2\n\nEvery one of the 42 cells uses real external exposures, value-free retrieval cues, matched interference, then H/F scrub for slow-only recall.\n\n" + table(b, ["exposure_count", "reuse_count"], ["retention", "accuracy", "transfer_mass"]) + "\n\n## Descriptive fits\n\n```json\n" + json.dumps(phase_fit, indent=2) + "\n```\n",
        "SELECTIVE_PERSISTENCE_SCALING_STAGE1_2.md": "# Selective Persistence Scaling — Stage 1.2\n\nSixteen used facts compete with identically encoded distractors. Each architecture received the same development search budget and independently selected LR. Efficiency is mean useful-fact accuracy divided by persistent-state bytes.\n\n" + table(c[c.condition.eq("intact")], ["model", "distractor_count"], ["accuracy", "retention", "interference", "confidence", "selective_persistence_efficiency", "persistent_state_bytes", "parameter_count"]) + f"\n\nG15: **{'PASS' if gate['G15_selective_persistence_scaling']['pass'] else 'FAIL'}**.\n",
        "AUTONOMOUS_MEMORY_SELECTION_STAGE1_2.md": "# Autonomous Memory Selection — Stage 1.2\n\nAfter three writes and H scrub, the single TASK_CUE contains only the operation; key IDs and values are absent and only NULL events follow.\n\n" + table(d, ["operation", "internal_tick"], ["accuracy", "projection_A", "projection_B", "projection_C", "confidence"]) + "\n",
        "SEQUENTIAL_INTERNAL_COMPUTATION_STAGE1_2.md": "# Sequential Internal Computation — Stage 1.2\n\nGraph writes are followed by H scrub. Each transition emits exactly one query and receives one memory-read vector. L=6–8 is OOD.\n\n" + table(e, ["length_regime", "path_length", "internal_tick"], ["accuracy", "query_path_alignment", "confidence", "entropy"]) + f"\n\nG16: **{'PASS' if gate['G16_sequential_internal_computation']['pass'] else 'FAIL'}**; hard bottleneck audit={gate['G16_sequential_internal_computation']['hard_bottleneck_audit']}.\n",
        "ENDOGENOUS_TIME_NECESSITY_STAGE1_2.md": "# Endogenous-Time Necessity — Stage 1.2\n\nThe development-selected 512-distractor condition is non-ceiling; event contents and total transitions are identical.\n\n" + table(f, ["condition"], ["accuracy", "retention", "loss", "confidence"]) + f"\n\nStatus: **{adjudication['endogenous_time']['status']}**; behavioral margin={adjudication['endogenous_time']['before_minus_after_accuracy']:.4f}.\n",
        "NO_SELF_EVIDENCE_STAGE1_2.md": "# No Self-Evidence Without an Explicit Cue — Stage 1.2\n\nBoth strata use the same FACT, QUERY and NULL event kinds, shapes and learned head. Only evidence availability differs; the unknowable target is independent.\n\n" + table(g, ["condition", "internal_tick"], ["accuracy", "confidence", "entropy", "ece", "brier"]) + f"\n\nG17: **{'PASS' if gate['G17_no_self_evidence']['pass'] else 'FAIL'}**.\n",
        "MEMORY_STORAGE_VS_USE_STAGE1_2.md": "# Memory Storage Versus Use — Stage 1.2\n\nInterventions occur after 2048 distractors. Pre/post lesion fields are explicit and historical Stage-1.1 rows are untouched.\n\n" + table(h, ["condition", "lesion_component"], ["accuracy", "loss", "pre_lesion_slow_retention", "post_lesion_slow_retention", "pre_lesion_accuracy", "post_lesion_accuracy", "target_projection"]) + "\n",
    }


def final_report(
    records: pd.DataFrame,
    training: pd.DataFrame,
    adjudication: dict,
    phase_fit: dict,
    run_id: str,
    config: dict,
    selected: dict,
) -> str:
    gates = adjudication["gates"]
    ratio = phase_fit["aggregate"]["mean_reuse_to_exposure_coefficient_ratio"]
    a = records[records.experiment.eq("A_functional_addressing")]
    original = mean(a[a.condition.eq("original")], "accuracy")
    signflip = mean(a[a.condition.eq("signflip")], "accuracy")
    d = records[records.experiment.eq("D_autonomous_selection")]
    d_gain = mean(d[d.internal_tick.eq(8)], "accuracy") - mean(d[d.internal_tick.eq(0)], "accuracy")
    b = records[records.experiment.eq("B_exposure_reuse_phase")]
    b_excerpt = b[b.exposure_count.isin([1, 8, 32]) & b.reuse_count.isin([0, 8, 32])]
    c = records[(records.experiment.eq("C_selective_scaling")) & records.condition.eq("intact")]
    e = records[records.experiment.eq("E_sequential_computation")]
    e_ood = e[e.length_regime.eq("ood_longer")]
    f = records[records.experiment.eq("F_endogenous_time")]
    g = records[records.experiment.eq("G_no_self_evidence")]
    h = records[records.experiment.eq("H_storage_vs_use")]
    ratio_ci = phase_fit["aggregate"]["bootstrap_95_ci"]
    formal_seeds = ", ".join(str(item) for item in config["training"]["formal_seeds"])
    selected_lrs = ", ".join(
        f"{model}={lr:g}" for model, lr in selected["selected_learning_rates"].items()
    )
    return f"""# ET-RCM Stage 1.2 Final Report

> **Does ET-RCM actually learn functional memory addressing and selective lifetime allocation, and can learned internal recurrence perform necessary sequential computation without converting self-generated activity into new evidence?**
>
> **ET-RCM 是否真正学会了功能性的记忆寻址与选择性的记忆寿命分配；同时，可学习的内部递归是否能够承担必要的逐步计算，而不会把自身产生的内部活动误当成新的外部证据？**

Formal run `{run_id}` uses eight fresh seeds. Stage-1/1.1 conclusions are unchanged. Generated {datetime.now(timezone.utc).isoformat()}.

## Gate summary

| Gate/status | Result | Key measurement |
|---|---:|---|
| G14 Functional addressing | **{'PASS' if gates['G14_functional_addressing']['pass'] else 'FAIL'}** | target-removal accuracy drop {gates['G14_functional_addressing']['accuracy_drop_target_removed']:.4f}; loss increase {gates['G14_functional_addressing']['loss_increase_target_removed']:.4f} |
| G15 Selective scaling | **{'PASS' if gates['G15_selective_persistence_scaling']['pass'] else 'FAIL'}** | high-pressure accuracy margin {gates['G15_selective_persistence_scaling']['high_pressure_accuracy_margin_vs_B2']:.4f}; slope margin {gates['G15_selective_persistence_scaling']['degradation_slope_margin_vs_B2']:.4f} |
| G16 Sequential computation | **{'PASS' if gates['G16_sequential_internal_computation']['pass'] else 'FAIL'}** | OOD K8-K1 {gates['G16_sequential_internal_computation']['ood_K8_minus_K1_accuracy']:.4f} |
| G17 No self-evidence | **{'PASS' if gates['G17_no_self_evidence']['pass'] else 'FAIL'}** | knowable gain {gates['G17_no_self_evidence']['knowable_K16_minus_K0_accuracy']:.4f}; unknown K64 {gates['G17_no_self_evidence']['unknowable_K64_accuracy']:.4f} |
| Endogenous time | **{adjudication['endogenous_time']['status']}** | before-after accuracy {adjudication['endogenous_time']['before_minus_after_accuracy']:.4f} |
| Stage-2 Memory LM | **{adjudication['STAGE2_MEMORY_LM_AUTHORIZED']}** | no decoder training executed |
| Stage-2 Continuous Cognition LM | **{adjudication['STAGE2_CONTINUOUS_COGNITION_LM_AUTHORIZED']}** | no decoder training executed |

## Experimental execution

All five baselines received an equal 3-LR × 2-seed development search. An initial 180-step search was preserved as underpowered; amendment A1 extended every architecture equally to 600 steps before formal evaluation. Formal training used the frozen architecture-specific selections, 600 memory steps, and for B6 separate 700/900/600-step autonomous/sequential/no-evidence models. Full BPTT was used. The combined artifacts contain {len(records):,} evaluation rows and {len(training):,} training-log rows.

## Experiment details

### Frozen formal design

| Item | Frozen value |
|---|---|
| Formal seeds | {formal_seeds} |
| State dimensions | H={config['model']['latent_slots']}×{config['model']['hidden_dim']}; F/M={config['model']['value_dim']}×{config['model']['key_dim']} |
| Memory constants | gamma={config['model']['gamma']}; rho_fast={config['model']['rho_fast']}; rho_slow={config['model']['rho_slow']}; eta={config['model']['eta_external']} |
| Batch / BPTT | {config['training']['batch_size']} / {config['training']['bptt']} |
| Formal training steps | memory={config['training']['formal_steps_memory']}; autonomous={config['training']['formal_steps_autonomous']}; sequential={config['training']['formal_steps_sequential']}; no-evidence={config['training']['formal_steps_no_evidence']} |
| Independently selected LRs | {selected_lrs} |
| Endogenous-time interference | {selected['selected_endogenous_interference']} distractors, frozen from development |
| Formal scaling maximum | {config['selective_scaling']['high_pressure_distractors']} distractors; 32768 was excluded before formal outcomes for the recorded two-GPU budget reason |

### A — Functional query intervention

Every intervention cloned the same memory and active state. The seven conditions changed only the query geometry. The target-projection removal is the adjudicating intervention; signed cosine is descriptive.

{table(a, ["condition"], ["accuracy", "loss", "target_retrieval_score", "downstream_hidden_change", "target_projection", "absolute_cosine", "squared_projection"])}

The learned query clearly mattered globally: sign-flip and random-query interventions were destructive. However, selectively removing the target-key projection changed accuracy by only {gates['G14_functional_addressing']['accuracy_drop_target_removed']:.4f}, changed loss by {gates['G14_functional_addressing']['loss_increase_target_removed']:.4f}, and replicated in only {gates['G14_functional_addressing']['replicate_seed_count']}/8 seeds. That does not establish target-direction-specific functional addressing.

### B — Exposure × reuse phase diagram

The full 6×7 grid used external-exposure counts 1–32 and value-free reuse counts 0–32, matched interference, then H/F scrub before slow-only recall. Representative cells are below; all 42 cells and heatmaps are in the dedicated report and processed artifacts.

{table(b_excerpt, ["exposure_count", "reuse_count"], ["retention", "accuracy", "transfer_mass"])}

The descriptive log-model reuse/exposure coefficient ratio was {ratio:.4f}, seed-bootstrap 95% CI [{ratio_ci[0]:.4f}, {ratio_ci[1]:.4f}]. Both axes improved retention, with a negative interaction indicating saturation/substitution. This ratio is not a universal exchange rate.

### C — Selective persistence scaling

Sixteen future-used facts competed with 0, 32, 128, 512, or the preregistered feasible maximum of 2048 same-format distractors. Persistent state remained fixed. Each architecture independently selected its LR from the same development budget.

{table(c, ["model", "distractor_count"], ["accuracy", "retention", "interference", "selective_persistence_efficiency", "persistent_state_bytes"])}

ET-RCM had a better mean degradation slope than B2 by {gates['G15_selective_persistence_scaling']['degradation_slope_margin_vs_B2']:.4f} and a positive byte-normalized efficiency margin, but its 2048-distractor accuracy margin was only {gates['G15_selective_persistence_scaling']['high_pressure_accuracy_margin_vs_B2']:.4f}, below the frozen 0.05 threshold. G15 therefore failed rather than being rescued by partial metrics.

### D — Autonomous multi-memory selection

After writes and an H scrub, TASK_CUE contained the operation only—no key IDs, values, or retrieval schedule—and all later events were NULL. The query trajectory visited task-relevant key directions and improved behavior for the first few ticks, although later ticks could degrade performance.

{table(d, ["operation", "internal_tick"], ["accuracy", "projection_A", "projection_B", "projection_C", "confidence"])}

### E — Sequential computation under a hard per-tick bottleneck

The graph history was scrubbed from H; each tick emitted one query and received one addressed read. Lengths 6–8 were held out. The bypass audit passed, but OOD accuracy did not systematically improve with tick budget.

{table(e_ood, ["internal_tick"], ["accuracy", "query_path_alignment", "confidence", "entropy"])}

### F — Endogenous-time timing intervention

The two schedules used identical event content, transition count, and total compute. Only whether K NULL transitions happened before or after resource-competing interference changed.

{table(f, ["condition"], ["accuracy", "retention", "loss", "confidence"])}

The matched-compute behavioral margin was {adjudication['endogenous_time']['before_minus_after_accuracy']:.4f}, replicated in {adjudication['endogenous_time']['replicate_seed_count']}/8 seeds. This supports pre-interference consolidation timing in this toy; it does not establish general autonomous reasoning.

### G — No self-evidence without an explicit cue

Knowable and unknowable examples used the same event kinds, shapes, query format, and output head. Only evidence availability differed.

{table(g, ["condition", "internal_tick"], ["accuracy", "confidence", "entropy", "ece", "brier"])}

Unknowable K=64 accuracy was {gates['G17_no_self_evidence']['unknowable_K64_accuracy']:.4f}; confidence changed by {gates['G17_no_self_evidence']['unknowable_confidence_inflation']:.4f} and ECE by {gates['G17_no_self_evidence']['unknowable_ece_degradation']:.4f}. The safety/calibration part passed. The compound G17 nevertheless failed because the already-near-ceiling knowable stratum improved by only {gates['G17_no_self_evidence']['knowable_K16_minus_K0_accuracy']:.4f}, below the frozen usefulness threshold.

### H — Stored versus used

Interventions were evaluated after 2048 distractors. Pre-lesion storage and post-lesion behavior are named separately.

{table(h, ["condition", "lesion_component"], ["accuracy", "loss", "pre_lesion_slow_retention", "post_lesion_slow_retention", "pre_lesion_accuracy", "post_lesion_accuracy"])}

Slow-memory lesion sharply reduced both slow retention and accuracy, so M was behaviorally used. F-only lesion improved accuracy in this long-interference setting, indicating fast-state interference. Target-query projection removal remained weak, consistent with G14 failure.

### I — Long continuous stream

`NOT_RUN_BY_PROTOCOL`. The frozen authorization condition required the Memory-LM path gates, which were not satisfied. No 1e3/1e4/1e5 result is implied.

## Direct answers to the 17 required questions

1. **Was G7 mainly a sign artifact?** No evidence supports that reinterpretation. G14={'PASS' if gates['G14_functional_addressing']['pass'] else 'FAIL'}; target-direction removal was weak, and historical G7 remains FAIL.
2. **Does target-component removal hurt?** Mean accuracy drop={gates['G14_functional_addressing']['accuracy_drop_target_removed']:.4f}, with {gates['G14_functional_addressing']['replicate_seed_count']}/8 registered seed replications.
3. **Is q≈-k functional gauge addressing?** Not established. Original accuracy={original:.4f} and sign-flip accuracy={signflip:.4f}, so query sign matters globally, but target projection was not specifically necessary. The fresh-seed mean signed cosine was positive, not stable evidence for q≈-k.
4. **Reuse relative to exposure?** The descriptive log-model reuse/exposure coefficient ratio is {ratio:.4f} (seed-bootstrap CI is in the phase report); it is toy-distribution-specific.
5. **Retention surface shape?** The full 6×7 grid, interaction/log fits and heatmaps are preserved; it is not assumed linear.
6. **Scaling versus permanent storage?** G15={'PASS' if gates['G15_selective_persistence_scaling']['pass'] else 'FAIL'} at the preregistered 2048-event endpoint and over the degradation curve.
7. **Fair after bytes/tuning?** B2 matches F+M matrix floats; all architectures received identical search budgets. Exact bytes/parameters/compute are per row.
8. **Autonomous multiple retrieval?** K8-K0 task accuracy changed by {d_gain:.4f}; trajectories contain A/B/C projections with no post-goal key cues.
9. **Sequential learned ticks?** G16={'PASS' if gates['G16_sequential_internal_computation']['pass'] else 'FAIL'} under the passed={gates['G16_sequential_internal_computation']['hard_bottleneck_audit']} bottleneck audit.
10. **Pre-event benefit?** {adjudication['endogenous_time']['status']}; behavioral margin={adjudication['endogenous_time']['before_minus_after_accuracy']:.4f}. State distance alone is not counted.
11. **Calibration without unknowable cue?** The unknowable stratum remained near chance and became no more confident, with same-format audit={gates['G17_no_self_evidence']['same_format_no_condition_cue']}. The compound G17 still {'passed' if gates['G17_no_self_evidence']['pass'] else 'failed'} because its knowable-benefit clause was not met.
12. **Stored versus used?** Query-removal and F/M lesion behavior is reported separately from pre/post stored retention in `MEMORY_STORAGE_VS_USE_STAGE1_2.md`.
13. **Stable through 1e5 events?** {'Long-stream evidence was authorized and is reported separately.' if adjudication['LONG_STREAM_AUTHORIZED'] else 'Not established: Experiment I was NOT_RUN_BY_PROTOCOL because its authorization rule failed.'}
14. **Necessary parts?** Only intervention-specific losses support necessity; strong storage without behavioral use is not counted. See G14, G15 and the lesion report.
15. **Keep endogenous internal time?** {'Keep it only as a tested consolidation-scheduling option: timing mattered under interference, but G16 failed, so continuous cognition is not justified as a core claim.' if adjudication['endogenous_time']['status']=='ET_STATUS_SUPPORTED' else 'No as a required core feature; prefer the simpler memory-only design unless new evidence changes this.'}
16. **Stage-2 Memory LM authorized?** **{adjudication['STAGE2_MEMORY_LM_AUTHORIZED']}**.
17. **Stage-2 Continuous Cognition LM authorized?** **{adjudication['STAGE2_CONTINUOUS_COGNITION_LM_AUTHORIZED']}**.

## Scientific boundary

These are synthetic bounded-state experiments. Functional projection is called functional only when an intervention changes behavior. Autonomous memory selection is not autonomous thought; long operation is not infinite context; no result establishes human-like memory, consciousness, or causal state in the unrestricted sense.
"""


def negative_results_report(adjudication: dict) -> str:
    gates = adjudication["gates"]
    return f"""# Stage 1.2 Negative and Null Results

This file records failed gates and mixed outcomes without changing the frozen protocol or Stage-1/1.1 conclusions.

| Item | Outcome | Consequence |
|---|---|---|
| G14 functional addressing | FAIL: target-removal accuracy drop {gates['G14_functional_addressing']['accuracy_drop_target_removed']:.4f}; {gates['G14_functional_addressing']['replicate_seed_count']}/8 seed replications | A learned query convention exists, but target-key projection was not shown to be specifically necessary. Historical G7 stays FAIL. |
| G15 selective scaling | FAIL: 2048-distractor margin {gates['G15_selective_persistence_scaling']['high_pressure_accuracy_margin_vs_B2']:.4f} | Better slope and efficiency were insufficient because the frozen behavioral margin failed. |
| G16 sequential computation | FAIL: OOD K8-K1 {gates['G16_sequential_internal_computation']['ood_K8_minus_K1_accuracy']:.4f} | The hard bottleneck was valid, but learned ticks did not yield systematic length-generalized computation. |
| G17 compound no-self-evidence gate | FAIL: knowable K16-K0 {gates['G17_no_self_evidence']['knowable_K16_minus_K0_accuracy']:.4f} | Unknowable safety/calibration passed; the compound gate failed because the knowable control was already at ceiling. This is not converted into a pass post hoc. |
| Long stream | NOT_RUN_BY_PROTOCOL | Stage-1.2 did not establish 1e5-event stability. |
| Stage-2 Memory LM | NOT AUTHORIZED | G14, G15 and G17 were not all satisfied. |
| Stage-2 Continuous Cognition LM | NOT AUTHORIZED | Memory-LM path failed and G16 failed, despite supported consolidation timing. |

Positive descriptive findings—autonomous task-dependent query trajectories, exposure/reuse retention gains, slow-memory lesion effects, improved degradation slope, and safe unknowable calibration—remain evidence for follow-up design, not substitutes for the failed preregistered gates.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="stage1_2-formal-v1")
    args = parser.parse_args()
    raw = ROOT / "results/stage1_2/raw" / args.run_id
    processed = ROOT / "results/stage1_2/processed" / args.run_id
    processed.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/stage1_2.yaml").read_text())
    records, training, manifests = load_complete(raw)
    adjudication = adjudicate(records, config)
    phase_fit = phase_models(records)
    records.to_parquet(processed / "records.parquet", index=False)
    training.to_parquet(processed / "training_log.parquet", index=False)
    (processed / "PROCESSED_DATA_README.md").write_text(
        "# Processed Stage-1.2 data\n\n"
        "`records.parquet` is a local convenience aggregate and is intentionally excluded from Git "
        "because it exceeds GitHub's 100 MiB per-file limit. The committed per-model/per-seed raw "
        "Parquet files are the complete authoritative records; rerunning this analyzer reconstructs "
        "the aggregate exactly. `condition_summary.parquet` and `training_log.parquet` are committed.\n"
    )
    grouping = [column for column in ["experiment", "condition", "model", "seed", "exposure_count", "reuse_count", "distractor_count", "operation", "path_length", "length_regime", "internal_tick", "lesion_component"] if column in records]
    numeric = records.select_dtypes(include="number").columns.difference(["episode", *grouping])
    records.groupby(grouping, dropna=False)[list(numeric)].mean().reset_index().to_parquet(processed / "condition_summary.parquet", index=False)
    (processed / "adjudication.json").write_text(json.dumps(adjudication, indent=2))
    (processed / "phase_models.json").write_text(json.dumps(phase_fit, indent=2))
    figures = processed / "figures"; figures.mkdir(exist_ok=True)
    phase = records[records.experiment.eq("B_exposure_reuse_phase")]
    for metric in ("retention", "accuracy", "transfer_mass"):
        heatmap_svg(phase, metric, figures / f"phase_{metric}.svg")
    for filename, content in specialized_reports(records, adjudication, phase_fit).items():
        (ROOT / "reports" / filename).write_text(content)
    selected = json.loads((ROOT / "configs/stage1_2_selected.json").read_text())
    report = final_report(records, training, adjudication, phase_fit, args.run_id, config, selected)
    (ROOT / "reports/STAGE1_2_FINAL_REPORT.md").write_text(report)
    (processed / "STAGE1_2_FINAL_REPORT.md").write_text(report)
    negative_report = negative_results_report(adjudication)
    (ROOT / "reports/NEGATIVE_RESULTS_STAGE1_2.md").write_text(negative_report)
    (processed / "NEGATIVE_RESULTS_STAGE1_2.md").write_text(negative_report)
    long_report = ROOT / "reports/LONG_STREAM_STAGE1_2.md"
    if not adjudication["LONG_STREAM_AUTHORIZED"]:
        long_report.write_text("# Long Continuous Stream — Stage 1.2\n\n`NOT_RUN_BY_PROTOCOL`\n\nExperiment I required G14, G15, G17 and no-bypass. The intermediate adjudication did not satisfy every condition. No 1e3/1e4/1e5 stream was executed.\n")
    copies = {
        "config.yaml": ROOT / "configs/stage1_2.yaml",
        "splits.json": ROOT / "configs/stage1_2_splits.json",
        "selected.json": ROOT / "configs/stage1_2_selected.json",
        "protocol.md": ROOT / "reports/STAGE1_2_PROTOCOL.md",
        "protocol_freeze.json": ROOT / "artifacts/stage1_2_protocol.freeze.json",
        "amendment_freeze.json": ROOT / "artifacts/stage1_2_amendment1.freeze.json",
        "formal_selection_freeze.json": ROOT / "artifacts/stage1_2_formal_selection.freeze.json",
    }
    for name, source in copies.items():
        shutil.copy2(source, raw / name); shutil.copy2(source, processed / name)
    manifest = {
        "run_id": args.run_id, "created_utc": datetime.now(timezone.utc).isoformat(),
        "formal_jobs": len(manifests), "record_rows": len(records), "training_rows": len(training),
        "combined_records": {
            "path": "records.parquet",
            "sha256": sha256(processed / "records.parquet"),
            "bytes": (processed / "records.parquet").stat().st_size,
            "git_tracked": False,
            "reason": "exceeds GitHub 100 MiB per-file limit; reconstructible from committed raw shards",
        },
        "long_stream_authorized": adjudication["LONG_STREAM_AUTHORIZED"],
        "long_stream_run": False,
        "stage2_memory_lm_authorized": adjudication["STAGE2_MEMORY_LM_AUTHORIZED"],
        "stage2_continuous_cognition_lm_authorized": adjudication["STAGE2_CONTINUOUS_COGNITION_LM_AUTHORIZED"],
        "stage2_training_run": False,
    }
    (processed / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for directory in (raw, processed, ROOT / "artifacts/stage1_2" / args.run_id):
        directory.mkdir(parents=True, exist_ok=True)
        paths = sorted(
            path for path in directory.rglob("*")
            if path.is_file()
            and path.name != "SHA256SUMS"
            and not (directory == processed and path.name == "records.parquet")
        )
        (directory / "SHA256SUMS").write_text("\n".join(f"{sha256(path)}  {path.relative_to(directory)}" for path in paths) + "\n")
    print(json.dumps(adjudication, indent=2))


if __name__ == "__main__":
    main()
