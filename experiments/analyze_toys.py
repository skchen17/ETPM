#!/usr/bin/env python3
"""Analyze the latest toy run, render compact SVG curves, and write the report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = [
    "retention",
    "accuracy",
    "loss",
    "confidence",
    "calibration",
    "h_norm",
    "f_norm",
    "m_norm",
    "a_norm",
    "delta_norm",
    "transfer_fraction",
    "query_similarity",
    "reuse_count",
    "cumulative_access",
    "h_change",
    "state_convergence",
    "idle_trajectory_length",
    "readout_drift",
    "memory_interference",
    "latency_steps",
    "compute_budget",
    "answer",
]


def _curve(
    final: pd.DataFrame, toy: str, condition: str, metric: str
) -> dict[float, float]:
    rows = final[(final.toy == toy) & (final.condition == condition)]
    rows = rows.dropna(subset=["parameter_value", metric])
    if rows.empty:
        return {}
    grouped = rows.groupby("parameter_value")[metric].mean().sort_index()
    return {float(index): float(value) for index, value in grouped.items()}


def _mean(final: pd.DataFrame, toy: str, condition: str, metric: str) -> float:
    rows = final[(final.toy == toy) & (final.condition == condition)][metric].dropna()
    return float(rows.mean()) if not rows.empty else math.nan


def _endpoint(curve: dict[float, float], end: bool) -> float:
    if not curve:
        return math.nan
    key = max(curve) if end else min(curve)
    return curve[key]


def _monotonic_signal(curve: dict[float, float], tolerance: float = 1e-8) -> bool:
    values = list(curve.values())
    return len(values) >= 2 and values[-1] > values[0] + tolerance and all(
        right + 1e-6 >= left for left, right in zip(values[:-1], values[1:])
    )


def _svg_plot(path: Path, title: str, xlabel: str, ylabel: str, series: dict[str, dict[float, float]]) -> None:
    width, height = 720, 440
    left, right, top, bottom = 78, 28, 54, 66
    points = [(x, y) for values in series.values() for x, y in values.items()]
    if not points:
        return
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(min(ys), 0.0), max(ys)
    if xmax == xmin:
        xmax = xmin + 1.0
    if ymax == ymin:
        ymax = ymin + 1.0
    pad = 0.08 * (ymax - ymin)
    ymin, ymax = ymin - pad, ymax + pad

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * (width - left - right)

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * (height - top - bottom)

    colors = ["#1769aa", "#c0392b", "#2e7d32", "#7b1fa2", "#ef6c00"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{title}</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#333"/>',
    ]
    for tick in range(6):
        y = ymin + tick * (ymax - ymin) / 5
        py = sy(y)
        elements.append(f'<line x1="{left}" y1="{py:.1f}" x2="{width-right}" y2="{py:.1f}" stroke="#ddd"/>')
        elements.append(f'<text x="{left-8}" y="{py+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{y:.3f}</text>')
    for index, (name, values) in enumerate(series.items()):
        ordered = sorted(values.items())
        color = colors[index % len(colors)]
        polyline = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in ordered)
        elements.append(f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in ordered:
            elements.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>')
            elements.append(f'<text x="{sx(x):.1f}" y="{height-bottom+18}" text-anchor="middle" font-family="sans-serif" font-size="10">{x:g}</text>')
        legend_y = top + 17 * index
        elements.append(f'<line x1="{width-210}" y1="{legend_y}" x2="{width-190}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{width-184}" y="{legend_y+4}" font-family="sans-serif" font-size="11">{name}</text>')
    elements.append(f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-family="sans-serif" font-size="13">{xlabel}</text>')
    elements.append(f'<text transform="translate(18 {height/2}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="13">{ylabel}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def _format_curve(curve: dict[float, float]) -> str:
    return ", ".join(f"{x:g}: {y:.4f}" for x, y in curve.items()) or "no data"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "results/raw/latest_run.json")
    parser.add_argument("--tests-passed", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    protocol_details = (ROOT / "docs/EXPERIMENT_PROTOCOL.md").read_text(
        encoding="utf-8"
    )
    records = pd.read_parquet(ROOT / manifest["records"])
    final = records[records.phase == "final"].copy()
    run_id = manifest["run_id"]
    output_dir = ROOT / "results/processed" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    existing_numeric = [column for column in NUMERIC if column in final.columns]
    summary = (
        final.groupby(["toy", "condition", "parameter", "parameter_value"], dropna=False)[existing_numeric]
        .agg(["mean", "std", "count"])
    )
    summary.columns = ["_".join(column).rstrip("_") for column in summary.columns]
    summary = summary.reset_index()
    summary_path = output_dir / "toy_summary.parquet"
    summary.to_parquet(summary_path, index=False, compression="zstd")

    exposure = _curve(final, "toy1_repeated_exposure", "full_etrcm", "retention")
    reuse_full = _curve(final, "toy2_repeated_use", "full_etrcm", "retention")
    reuse_uniform = _curve(final, "toy2_repeated_use", "uniform_transfer", "retention")
    reuse_single = _curve(final, "toy2_repeated_use", "single_persistent", "retention")
    reuse_nonconserving = _curve(final, "toy2_repeated_use", "nonconserving_replay", "retention")
    idle_reasoning = _curve(final, "toy4_idle_reasoning", "idle_compute", "accuracy")
    matched_idle = _curve(final, "toy5_matched_compute", "idle_compute", "accuracy")
    matched_query = _curve(final, "toy5_matched_compute", "query_time_compute", "accuracy")
    idle_consolidation = _curve(final, "toy6_idle_consolidation", "full_etrcm", "retention")
    unknowable_accuracy = _curve(final, "toy7_unknowable_bit", "full_etrcm_null_evidence", "accuracy")
    unknowable_confidence = _curve(final, "toy7_unknowable_bit", "full_etrcm_null_evidence", "confidence")

    revision = final[final.toy == "toy8_memory_revision"]
    best_revision = float(revision.accuracy.max())
    revision_means = {
        float(index): float(value)
        for index, value in revision.groupby("parameter_value").accuracy.mean().sort_index().items()
    }
    toy3_useful = _mean(final, "toy3_same_input_future_utility", "future_useful", "retention")
    toy3_unused = _mean(final, "toy3_same_input_future_utility", "future_unused", "retention")
    toy9_local = _mean(final, "toy9_temporary_persistent", "episode_local", "retention")
    toy9_stable = _mean(final, "toy9_temporary_persistent", "cross_episode_stable", "retention")
    matched_state_difference = float(
        final[final.toy == "toy5_matched_compute"].state_convergence.max()
    )
    nonconserving_drift = float(
        final[
            (final.toy == "toy2_repeated_use")
            & (final.condition == "nonconserving_replay")
        ].readout_drift.max()
    )
    conserving_drift = float(
        final[
            (final.toy == "toy2_repeated_use")
            & (final.condition == "full_etrcm")
        ].readout_drift.max()
    )

    gates = {
        "G1_math": bool(args.tests_passed),
        "G2_repeated_exposure": _monotonic_signal(exposure),
        "G3_repeated_use": _monotonic_signal(reuse_full),
        "G4_idle_cognition": _endpoint(idle_reasoning, True) > _endpoint(idle_reasoning, False) + 0.1,
        "G5_no_self_evidence": (
            max(abs(value - 0.5) for value in unknowable_accuracy.values()) < 0.05
            and max(unknowable_confidence.values()) - min(unknowable_confidence.values()) < 0.01
        ),
        "G6_memory_revision": best_revision >= 0.70,
    }
    negatives = {
        "N1_single_memory_equivalent": _endpoint(reuse_single, True) >= _endpoint(reuse_full, True) - 0.05,
        "N2_uniform_equivalent": _endpoint(reuse_uniform, True) >= _endpoint(reuse_full, True) - 0.05,
        "N3_matched_compute_equivalent": (
            matched_state_difference < 1e-12
            and all(abs(matched_idle[key] - matched_query[key]) < 1e-12 for key in matched_idle)
        ),
        "N4_unknowable_confidence_rises": _endpoint(unknowable_confidence, True) > _endpoint(unknowable_confidence, False) + 0.02,
        "N5_revision_failure": best_revision < 0.70,
    }
    stage2_ready = all(gates.values()) and not any(
        negatives[name] for name in ("N1_single_memory_equivalent", "N2_uniform_equivalent", "N3_matched_compute_equivalent", "N4_unknowable_confidence_rises", "N5_revision_failure")
    )

    result = {
        "run_id": run_id,
        "record_count": len(records),
        "final_record_count": len(final),
        "gates": gates,
        "negative_criteria": negatives,
        "stage2_ready": stage2_ready,
        "curves": {
            "repeated_exposure": exposure,
            "repeated_use_full": reuse_full,
            "repeated_use_uniform": reuse_uniform,
            "repeated_use_single": reuse_single,
            "repeated_use_nonconserving": reuse_nonconserving,
            "idle_reasoning": idle_reasoning,
            "idle_consolidation": idle_consolidation,
            "unknowable_accuracy": unknowable_accuracy,
            "unknowable_confidence": unknowable_confidence,
            "revision_by_new_exposures": revision_means,
        },
        "diagnostics": {
            "toy3_useful_retention": toy3_useful,
            "toy3_unused_retention": toy3_unused,
            "toy9_local_retention": toy9_local,
            "toy9_stable_retention": toy9_stable,
            "matched_final_state_difference": matched_state_difference,
            "conserving_readout_drift_max": conserving_drift,
            "nonconserving_readout_drift_max": nonconserving_drift,
            "best_revision_probability": best_revision,
        },
        "records": manifest["records"],
        "summary": str(summary_path.relative_to(ROOT)),
    }
    summary_json = output_dir / "summary.json"
    summary_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    figure_dir = output_dir / "figures"
    figure_dir.mkdir()
    _svg_plot(figure_dir / "retention_exposure.svg", "Repeated real exposure", "external exposures", "slow retention", {"ET-RCM": exposure})
    _svg_plot(figure_dir / "retention_reuse.svg", "Use-dependent retention and baselines", "internal reuse", "slow retention", {"ET-RCM": reuse_full, "uniform": reuse_uniform, "single": reuse_single, "non-conserving": reuse_nonconserving})
    _svg_plot(figure_dir / "idle_reasoning.svg", "Idle reasoning", "internal ticks", "accuracy", {"idle": idle_reasoning})
    _svg_plot(figure_dir / "matched_compute.svg", "Matched compute", "compute ticks", "accuracy", {"idle": matched_idle, "query-time": matched_query})
    _svg_plot(figure_dir / "idle_consolidation.svg", "Idle consolidation", "idle ticks", "slow retention", {"ET-RCM": idle_consolidation})
    _svg_plot(figure_dir / "unknowable_control.svg", "Unknowable-bit control", "internal ticks", "value", {"accuracy": unknowable_accuracy, "confidence": unknowable_confidence})
    _svg_plot(figure_dir / "memory_revision.svg", "Memory revision", "new external exposures", "P(new value)", {"mean": revision_means})

    report = f"""# ET-RCM Stage-1 toy validation report

> **Can a model with its own internal time transform transient experience into persistent computational state through repeated internal use, while keeping self-repetition from becoming new evidence?**
>
> **一个具有自身内部时间的模型，能否让短暂经历因为后续内部计算中的反复使用而自然转化为持久计算状态，同时避免把自己的重复思考误当成新的证据？**

## Execution scope and verdict

- Run: `{run_id}`; raw records: `{manifest['records']}`.
- Records: {len(records)} total, {len(final)} final episode summaries, {len(manifest['seed_values'])} seeds.
- Stage-2 authorization: **{str(stage2_ready).upper()}**.
- This is a small synthetic validation. It is not evidence of human-like memory,
  consciousness, infinite information capacity, or an intervention-validated
  causal memory in a language model.

{protocol_details}

## Gates

| gate | result |
|---|---|
""" + "\n".join(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in gates.items()) + f"""

## Negative criteria

| criterion | triggered |
|---|---|
""" + "\n".join(f"| {name} | {value} |" for name, value in negatives.items()) + f"""

N1/N3 are decision-relevant null results: the single persistent matrix remains
competitive on this narrow retention endpoint, and deterministic idle compute
is exactly equivalent to moving the same transitions to query time. Idle
execution reduces future latency but does not improve matched-compute accuracy
or final state in Toy 5. These outcomes are retained rather than protocol-tuned.

The Toy-2 uniform baseline uses `gamma/key_dim` on every direction per tick,
matching the isotropic per-direction budget of one rank-one ET-RCM access. B2
uses the slow decay coefficient and is therefore an intentionally strong single
persistent-memory baseline. B0/B1 are implemented and smoke-tested but are not
assigned associative-retention scores without a separately trained sequence
protocol.

## Required questions

1. **Readout conservation?** Yes within the unit-test tolerance when G1 passes;
   maximum accumulated Toy-2 conserving drift was `{conserving_drift:.3e}`.
   The non-conserving replay control drifted by `{nonconserving_drift:.4f}`.
2. **Repeated consolidation analytic formula?** `test_repeated_consolidation_analytic_solution`
   checks `F_K q=(1-gamma)^K F_0 q`; G1 result: `{gates['G1_math']}`.
3. **Repeated real exposure improves retention?** `{gates['G2_repeated_exposure']}`.
   Curve: {_format_curve(exposure)}.
4. **Repeated internal use at matched exposure improves retention?**
   `{gates['G3_repeated_use']}`. Curve: {_format_curve(reuse_full)}.
5. **Same input/different future utility learns different lifetime?** In the
   deliberately small differentiable policy toy, useful/unused retention was
   `{toy3_useful:.4f}` / `{toy3_unused:.4f}`. This is proof of implementation,
   not a general learned-importance result.
6. **Do idle ticks enable reasoning?** Toy-4 accuracy changed from
   `{_endpoint(idle_reasoning, False):.4f}` to `{_endpoint(idle_reasoning, True):.4f}`.
   This shows iterative latent computation, not autonomous human-like thought.
7. **Idle vs matched query-time compute?** Accuracy curves and final states were
   matched; maximum final-state difference was `{matched_state_difference:.3e}`.
   Only readiness/latency placement differed.
8. **Does idle consolidation improve slow retention?**
   `{_monotonic_signal(idle_consolidation)}`; curve: {_format_curve(idle_consolidation)}.
9. **Unknowable-bit self-confidence amplification?** `{negatives['N4_unknowable_confidence_rises']}`.
   Accuracy: {_format_curve(unknowable_accuracy)}; confidence: {_format_curve(unknowable_confidence)}.
10. **Can real new evidence correct old memory?** Best `P(new)` was
    `{best_revision:.4f}`; G6: `{gates['G6_memory_revision']}`. Mean by new
    exposures: {_format_curve(revision_means)}.
11. **Query-dependent better than uniform?** Negative-equivalence criterion N2
    is `{negatives['N2_uniform_equivalent']}`. Full: {_format_curve(reuse_full)};
    uniform: {_format_curve(reuse_uniform)}.
12. **Fast/slow better than single persistent memory?** Not established when N1
    is triggered (`{negatives['N1_single_memory_equivalent']}`). Single:
    {_format_curve(reuse_single)}.
13. **Independent value of endogenous time?** It provides pre-query readiness
    and a slot for consolidation, but Toy 5 does not show a matched-compute state
    or accuracy advantage. Independent value is therefore **not established**.
14. **Enough evidence for Stage 2?** **{str(stage2_ready).upper()}**.
15. **Where is evidence insufficient?** The narrow single-memory baseline is not
    defeated; matched idle/query-time compute is equivalent; Toy 3 uses a tiny
    synthetic context policy; and no language-scale learned dynamics or causal
    state intervention has been tested.

## Temporary vs persistent structure

Toy-9 slow retention was `{toy9_local:.4f}` for the episode-local rule and
`{toy9_stable:.4f}` for the cross-episode repeatedly used rule. Inputs had the
same form; only later task-use statistics differed.

## Artifacts

- Raw: `{manifest['records']}`
- Summary: `{summary_path.relative_to(ROOT)}` and `{summary_json.relative_to(ROOT)}`
- Figures: `{figure_dir.relative_to(ROOT)}`

All negative and null arms remain in the raw Parquet. The protocol was not
modified in response to these results.
"""
    run_report = output_dir / "TOY_VALIDATION_REPORT.md"
    run_report.write_text(report, encoding="utf-8")
    (ROOT / "reports/TOY_VALIDATION_REPORT.md").write_text(report, encoding="utf-8")
    (ROOT / "results/processed/latest_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
