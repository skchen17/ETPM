"""Frozen seed-level adjudication and complete Stage 1.5 integrity audit."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from etrcm.stage1_5.precision import evaluate_precision


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap(values: dict[int, float], *, threshold: float = 0.0,
              seed: int = 1515, resamples: int = 2000) -> dict[str, object]:
    data = np.asarray([values[k] for k in sorted(values)], dtype=float)
    if len(data) != 8 or not np.isfinite(data).all():
        raise ValueError("requires exactly eight finite seed effects")
    generator = np.random.default_rng(seed)
    samples = generator.choice(data, size=(resamples, len(data)), replace=True).mean(1)
    return {
        "mean": float(data.mean()), "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
        "positive_seeds": int((data > threshold).sum()),
        "seed_values": {str(k): float(values[k]) for k in sorted(values)},
    }


def _seed_dict(frame: pd.DataFrame, value: str) -> dict[int, float]:
    return {int(seed): float(group[value].mean()) for seed, group in frame.groupby("seed")}


def _lag_area(frame: pd.DataFrame) -> float:
    ordered = frame.sort_values("history_lag")
    x = np.log2(ordered.history_lag.to_numpy(dtype=float))
    y = ordered.heldout_decoding_gain.to_numpy(dtype=float).clip(min=0)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def main() -> None:
    config_path = ROOT / "configs/stage1_5.yaml"
    selection_path = ROOT / "configs/stage1_5_selected.json"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selection = _json(selection_path)
    run_id = config["protocol"]["formal_run_id"]
    seeds = config["training"]["formal_seeds"]
    base = config["training"]["base_models"]
    capacity_variants = [
        "B5_separate" if (cell["hidden_dim"], cell["memory_dim"]) == (64, 16)
        else f"B5_separate_h{cell['hidden_dim']}_m{cell['memory_dim']}"
        for cell in config["evaluation"]["capacity_grid"]
    ]
    experiments: dict[str, list[str]] = {
        "baseline": base,
        **{name: ["B5_separate"] for name in (
            "stability", "perturbation", "timescale", "anatomy",
            "observability", "mediation", "oracle",
        )},
        "capacity": capacity_variants,
    }
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    checked = 0
    rows_total = 0
    for experiment, variants in experiments.items():
        for variant in variants:
            parts = []
            for seed in seeds:
                shard = ROOT / "results/stage1_5" / run_id / "evaluation" / experiment / f"{variant}_seed{seed}"
                summary = _json(shard / "summary.json")
                path = shard / "records.parquet"
                if (summary["experiment"], summary["variant"], summary["seed"]) != (experiment, variant, seed):
                    raise ValueError(f"evaluation identity mismatch: {shard}")
                if summary["record_sha256"] != digest(path):
                    raise ValueError(f"record hash mismatch: {path}")
                if summary["config_sha256"] != digest(config_path) or summary["selection_sha256"] != digest(selection_path):
                    raise ValueError(f"frozen config/selection mismatch: {shard}")
                for source, expected in summary["source_sha256"].items():
                    if digest(ROOT / source) != expected:
                        raise ValueError(f"evaluation source drift: {source}")
                rate = selection["learning_rates"][variant]
                checkpoint = ROOT / "results/stage1_5" / run_id / f"{variant}_lr{rate:g}_seed{seed}/checkpoint.pt"
                if summary["checkpoint_sha256"] != digest(checkpoint):
                    raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
                frame = pd.read_parquet(path)
                if len(frame) != summary["row_count"] or not frame.seed.eq(seed).all():
                    raise ValueError(f"record identity/count mismatch: {path}")
                numeric = frame.select_dtypes(include=["number"]).to_numpy(dtype=float, na_value=np.nan)
                if np.isinf(numeric).any():
                    raise FloatingPointError(f"nonfinite infinity in {path}")
                parts.append(frame)
                checked += 1
                rows_total += len(frame)
            frames[(experiment, variant)] = pd.concat(parts, ignore_index=True)
    if checked != 160:
        raise ValueError(f"formal evaluation incomplete: {checked}/160")
    changed = subprocess.run(
        ["git", "diff", "--name-status", config["protocol"]["prior_revision"], "HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    if any(not line.startswith("A\t") for line in changed):
        raise ValueError("a tracked pre-Stage-1.5 artifact was modified or removed")

    stability = frames[("stability", "B5_separate")]
    perturbation = frames[("perturbation", "B5_separate")]
    g27_by_seed: dict[int, dict[str, object]] = {}
    for seed in seeds:
        one = stability[stability.seed.eq(seed)]
        two = perturbation[perturbation.seed.eq(seed)]
        max_norm = float(one[["H_norm", "F_norm", "M_norm"]].max().max())
        max_js = float(one.prediction_js_from_t0.max())
        at128 = two[two.internal_tick.eq(128)]
        max_median_growth = float(at128.groupby(
            ["perturbation_component", "epsilon", "world_family"]
        ).full_state_gain.median().max())
        finite = bool(np.isfinite(one.select_dtypes(include=["number"]).to_numpy(dtype=float, na_value=np.nan)[
            ~np.isnan(one.select_dtypes(include=["number"]).to_numpy(dtype=float, na_value=np.nan))
        ]).all())
        passed = finite and max_norm < 1000 and max_js < 0.5 and max_median_growth < 100
        curve = one.groupby("internal_tick")[["H_delta_norm", "F_delta_norm", "M_delta_norm", "H_norm"]].mean()
        tail_delta = float(curve.loc[897:1024, ["H_delta_norm", "F_delta_norm", "M_delta_norm"]].sum(axis=1).median())
        h_norm_drift = float(curve.loc[1024, "H_norm"] - curve.loc[0, "H_norm"])
        if not passed:
            dynamics_class = "unstable_or_divergent_on_sample"
        elif tail_delta < 1e-4:
            dynamics_class = "fixed_point_like_on_sample"
        else:
            dynamics_class = "bounded_drift_or_unresolved_oscillation"
        g27_by_seed[seed] = {
            "max_component_norm": max_norm, "max_prediction_js": max_js,
            "max_median_finite_growth_128": max_median_growth,
            "tail_median_total_delta": tail_delta,
            "H_norm_drift_0_to_1024": h_norm_drift,
            "descriptive_dynamics_class": dynamics_class,
            "all_finite": finite, "pass": passed,
        }
    g27 = all(item["pass"] for item in g27_by_seed.values())

    timescale = frames[("timescale", "B5_separate")]
    areas: dict[int, dict[str, dict[str, float]]] = {}
    g28_fh, g28_mf, g28_ablation = {}, {}, {}
    for seed in seeds:
        by_variant: dict[str, dict[str, float]] = {}
        for variant in timescale.decay_variant.unique():
            by_variant[variant] = {}
            for channel in "HFM":
                subset = timescale[
                    timescale.seed.eq(seed) & timescale.decay_variant.eq(variant)
                    & timescale.component.eq(channel)
                ]
                by_variant[variant][channel] = _lag_area(subset)
        areas[seed] = by_variant
        g28_fh[seed] = by_variant["primary"]["F"] - by_variant["primary"]["H"]
        g28_mf[seed] = by_variant["primary"]["M"] - by_variant["primary"]["F"]
        late = timescale[timescale.seed.eq(seed) & timescale.component.eq("M")
                         & timescale.history_lag.ge(64)]
        primary = float(late[late.decay_variant.eq("primary")].heldout_decoding_gain.mean())
        equal_fast = float(late[late.decay_variant.eq("equal_fast_decay")].heldout_decoding_gain.mean())
        equal_slow = float(late[late.decay_variant.eq("equal_slow_decay")].heldout_decoding_gain.mean())
        g28_ablation[seed] = min(abs(primary - equal_fast), abs(primary - equal_slow))
    g28_fh_stats = bootstrap(g28_fh)
    g28_mf_stats = bootstrap(g28_mf)
    g28_ablation_stats = bootstrap(g28_ablation, threshold=0.01)
    g28 = (g28_fh_stats["mean"] >= 0.02 and g28_fh_stats["positive_seeds"] >= 6
           and g28_mf_stats["mean"] >= 0.02 and g28_mf_stats["positive_seeds"] >= 6
           and g28_ablation_stats["positive_seeds"] >= 6)

    anatomy = frames[("anatomy", "B5_separate")]
    a5 = anatomy[anatomy.experiment.eq("A5_anatomy") & anatomy.internal_tick.eq(4)]
    js_by_channel = {
        channel: _seed_dict(a5[a5.swap_condition.eq(channel)], "prediction_js")
        for channel in ("H", "F", "M", "HF", "HM", "FM", "HFM")
    }
    js_stats = {channel: bootstrap(values) for channel, values in js_by_channel.items()}
    relevant = [channel for channel in ("F", "M")
                if js_stats[channel]["mean"] >= 1e-5
                and sum(value >= 1e-5 for value in js_by_channel[channel].values()) >= 6]
    fm_replicates = sum(value >= 1e-5 for value in js_by_channel["FM"].values())
    g29 = bool(relevant and js_stats["FM"]["mean"] >= 1e-5 and fm_replicates >= 6)
    interactions = anatomy[anatomy.experiment.eq("A5_interaction") & anatomy.internal_tick.eq(4)]
    interaction_stats = {
        name: bootstrap(_seed_dict(interactions[interactions.interaction.eq(name)], "signed_CE_interaction"))
        for name in ("FM", "HF", "HM", "HFM")
    }

    mediation = frames[("mediation", "B5_separate")]
    b2 = mediation[mediation.internal_tick.eq(4)]
    mediation_results: dict[str, object] = {}
    g30 = bool(relevant)
    for channel in ("F", "M"):
        one = b2[b2.swap_condition.eq(channel)]
        pivot = one.groupby(["seed", "intervention_condition"])[["prediction_js", "future_CE"]].mean()
        swap = pivot.xs("swap", level="intervention_condition")
        restore = pivot.xs("swap_read_restored", level="intervention_condition")
        ratios = {}
        for seed in seeds:
            denominator = float(swap.loc[seed, "prediction_js"])
            ratios[seed] = (
                1 - float(restore.loc[seed, "prediction_js"]) / denominator
                if denominator > 1e-5 else np.nan
            )
        identifiable = all(np.isfinite(value) for value in ratios.values())
        ratio_stats = bootstrap(ratios) if identifiable else None
        passed = (channel not in relevant or (
            identifiable and ratio_stats is not None and ratio_stats["mean"] >= 0.5
            and sum(value >= 0.5 for value in ratios.values()) >= 6
        ))
        if channel in relevant:
            g30 = g30 and passed
        mediation_results[channel] = {
            "swap_JS": bootstrap({seed: float(swap.loc[seed, "prediction_js"]) for seed in seeds}),
            "restore_JS": bootstrap({seed: float(restore.loc[seed, "prediction_js"]) for seed in seeds}),
            "swap_CE": bootstrap({seed: float(swap.loc[seed, "future_CE"]) for seed in seeds}),
            "restore_CE": bootstrap({seed: float(restore.loc[seed, "future_CE"]) for seed in seeds}),
            "mediation_fraction": ratio_stats, "identifiable": identifiable,
            "relevant": channel in relevant, "pass_if_relevant": passed,
        }

    oracle = frames[("oracle", "B5_separate")]
    b3 = oracle[oracle.internal_tick.eq(4) & oracle.distractor_count.isin([512, 2048])]
    means = b3.groupby(["seed", "intervention_condition"]).future_CE.mean().unstack()
    oracle_margins = {
        comparator: bootstrap({seed: float(means.loc[seed, comparator] - means.loc[seed, "oracle_static"])
                               for seed in seeds})
        for comparator in ("zero", "random", "shuffled", "learned")
    }
    g31 = all(item["mean"] >= 0.01 and item["positive_seeds"] >= 6
              for item in oracle_margins.values())
    closed_loop = {}
    for tick in (1, 2, 4, 8):
        curve = oracle[oracle.internal_tick.eq(tick)].groupby(
            ["seed", "intervention_condition"]
        ).future_CE.mean().unstack()
        closed_loop[str(tick)] = bootstrap({
            seed: float(curve.loc[seed, "oracle_static"] - curve.loc[seed, "oracle_closed_loop"])
            for seed in seeds
        })

    baseline = {}
    for variant in base:
        frame = frames[("baseline", variant)]
        subset = frame[frame.experiment.eq("B") & frame.intervention_condition.eq("full")
                       & frame.distractor_count.isin([512, 2048])]
        baseline[variant] = bootstrap(_seed_dict(subset, "prediction_loss"))
    observability = frames[("observability", "B5_separate")]
    observable = {
        condition: bootstrap(_seed_dict(
            observability[observability.feature_condition.eq(condition)], "future_event_CE"
        ))
        for condition in ("H", "HF", "HM", "HFM")
    }
    capacity = {}
    for variant in capacity_variants:
        frame = frames[("capacity", variant)]
        cell = {}
        associative = frame[frame.experiment.eq("A4_associative_capacity")]
        predictive = frame[frame.experiment.eq("A4_capacity_prediction")]
        for n in (0, 32, 128, 512, 2048, 8192):
            p = predictive[predictive.distractor_count.eq(n)]
            cell[str(n)] = {
                "prediction_CE": bootstrap(_seed_dict(p, "future_CE")),
                "prediction_accuracy": bootstrap(_seed_dict(p, "behavioral_accuracy")),
                "M_only_retention": bootstrap(_seed_dict(p, "M_only_retention_cosine")),
                "associative_accuracy_by_items": {
                    str(count): float(associative[
                        associative.distractor_count.eq(n) & associative.stored_item_count.eq(count)
                    ].retrieval_top1_accuracy.mean())
                    for count in (1, 2, 4, 8, 16, 32, 64, 128)
                },
            }
        capacity[variant] = cell

    processed = ROOT / "results/stage1_5/processed" / run_id
    if processed.exists():
        raise FileExistsError(f"immutable analysis already exists: {processed}")
    processed.mkdir(parents=True)
    precision = pd.DataFrame(evaluate_precision(run_id=run_id))
    precision.to_parquet(processed / "precision.parquet", index=False)
    access = stability[[
        "run_id", "model", "seed", "world_family", "state_id",
        "external_step", "internal_tick", "r_F_norm", "r_M_norm", "g_F", "g_M",
    ]].copy().sort_values(["state_id", "internal_tick"])
    access["cumulative_raw_F_read_norm"] = access.groupby("state_id").r_F_norm.cumsum()
    access["cumulative_raw_M_read_norm"] = access.groupby("state_id").r_M_norm.cumsum()
    access["read_usage_definition"] = "cumulative_raw_norm_diagnostic_not_causal_use"
    access.to_parquet(processed / "read_usage.parquet", index=False)
    metrics = {
        "run_id": run_id, "formal_training_cells": 96, "formal_evaluation_shards": checked,
        "formal_evaluation_rows": rows_total,
        "G27": {"pass": g27, "by_seed": {str(k): v for k, v in g27_by_seed.items()}},
        "G28": {"pass": g28, "F_minus_H_area": g28_fh_stats, "M_minus_F_area": g28_mf_stats,
                "equal_decay_ablation_min_difference": g28_ablation_stats,
                "lag_areas": {str(k): v for k, v in areas.items()}},
        "G29": {"pass": g29, "prediction_JS_by_channel": js_stats,
                "relevant_single_channels": relevant, "pairwise_CE_interactions": interaction_stats},
        "G30": {"pass": g30, "channels": mediation_results,
                "unmodeled_peripheral_pathway": not g30},
        "G31": {"pass": g31, "oracle_CE_margins": oracle_margins,
                "closed_loop_minus_static_advantage": closed_loop},
        "G32": {"status": "PENDING_AUTHORIZED" if g31 else "NOT_RUN_BY_PROTOCOL"},
        "G33": {"status": "PENDING_G32" if g31 else "NOT_RUN_BY_PROTOCOL"},
        "baseline_long_gap_CE": baseline,
        "observability_event_CE": observable,
        "capacity": capacity,
        "precision_sha256": digest(processed / "precision.parquet"),
        "read_usage_sha256": digest(processed / "read_usage.parquet"),
    }
    (processed / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    adjudication = {
        "run_id": run_id,
        **{name: "PASS" if value else "FAIL" for name, value in
           (("G27", g27), ("G28", g28), ("G29", g29), ("G30", g30), ("G31", g31))},
        "G32": metrics["G32"]["status"], "G33": metrics["G33"]["status"],
        "SMALL_SCALE_SEQUENCE_MODEL_PROTOTYPE_RECOMMENDED": False,
        "LANGUAGE_MODEL_TRAINING_STARTED": False,
    }
    (processed / "adjudication.json").write_text(json.dumps(adjudication, indent=2), encoding="utf-8")
    integrity = {
        "checked_evaluation_shards": checked, "expected_evaluation_shards": 160,
        "formal_evaluation_rows": rows_total,
        "config_sha256": digest(config_path), "selection_sha256": digest(selection_path),
        "metrics_sha256": digest(processed / "metrics.json"),
        "adjudication_sha256": digest(processed / "adjudication.json"),
        "precision_sha256": digest(processed / "precision.parquet"),
        "read_usage_sha256": digest(processed / "read_usage.parquet"),
        "all_shard_hashes_valid": True,
        "prior_revision": config["protocol"]["prior_revision"],
        "prior_tracked_artifacts_unchanged": True,
    }
    (processed / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    print(json.dumps(adjudication, sort_keys=True))


if __name__ == "__main__":
    main()
