"""Frozen-plan Stage 1.4 seed-level analysis and conservative adjudication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "stage1_4-formal-v1a1"
REQUIRED_FIELDS = {
    "run_id", "experiment", "model", "seed", "episode", "world_family",
    "external_step", "internal_tick", "horizon", "event_type", "future_target",
    "prediction_loss", "H_norm", "F_norm", "M_norm", "q_F", "q_M",
    "r_F_norm", "r_M_norm", "normalized_r_F_norm", "normalized_r_M_norm",
    "g_F", "g_M", "effective_F_norm", "effective_M_norm", "transfer_norm",
    "memory_item_id", "read_usage", "causal_usage", "retention",
    "intervention_condition", "H_restoration_flag", "future_H_difference",
    "future_prediction_difference", "parameter_count", "persistent_state_bytes",
    "compute_budget",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite_or_none(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def load_and_verify() -> tuple[pd.DataFrame, dict, dict, list[dict], list[dict]]:
    config = yaml.safe_load((ROOT / "configs/stage1_4_v1a1.yaml").read_text())
    freeze = json.loads((ROOT / "artifacts/stage1_4_formal_selection_v1a1.freeze.json").read_text())
    for relative, expected in freeze["files_sha256"].items():
        actual = digest(ROOT / relative)
        if actual != expected:
            raise ValueError(f"frozen file changed: {relative}: {actual}")
    selected = json.loads((ROOT / "configs/stage1_4_selected_v1a1.json").read_text())
    seeds = config["training"]["formal_seeds"]
    models = config["training"]["trainable_models"]
    training_summaries: list[dict] = []
    evaluation_summaries: list[dict] = []
    frames: list[pd.DataFrame] = []
    for model in models:
        for seed in seeds:
            lr = selected["learning_rates"][model]
            train_dir = ROOT / "results/stage1_4" / RUN_ID / f"{model}_lr{lr:g}_seed{seed}"
            eval_dir = ROOT / "results/stage1_4" / RUN_ID / f"{model}_seed{seed}"
            train = json.loads((train_dir / "summary.json").read_text())
            evaluation = json.loads((eval_dir / "summary.json").read_text())
            if train["checkpoint_sha256"] != digest(train_dir / "checkpoint.pt"):
                raise ValueError(f"checkpoint hash mismatch: {model}/{seed}")
            if evaluation["checkpoint_sha256"] != train["checkpoint_sha256"]:
                raise ValueError(f"evaluation used wrong checkpoint: {model}/{seed}")
            if evaluation["records_sha256"] != digest(eval_dir / "records.parquet"):
                raise ValueError(f"record hash mismatch: {model}/{seed}")
            if train["config_sha256"] != freeze["files_sha256"]["configs/stage1_4_v1a1.yaml"]:
                raise ValueError("training config mismatch")
            if evaluation["selection_sha256"] != freeze["files_sha256"]["configs/stage1_4_selected_v1a1.json"]:
                raise ValueError("evaluation selection mismatch")
            frame = pd.read_parquet(eval_dir / "records.parquet")
            if not REQUIRED_FIELDS <= set(frame.columns):
                raise ValueError(f"record schema incomplete: {REQUIRED_FIELDS - set(frame.columns)}")
            if len(frame) != evaluation["record_count"]:
                raise ValueError("record count mismatch")
            if not frame.model.eq(model).all() or not frame.seed.eq(seed).all():
                raise ValueError("shard identity mismatch")
            numeric = frame.select_dtypes(include=["number"])
            if not np.isfinite(numeric.to_numpy(dtype=float)[~numeric.isna().to_numpy()]).all():
                raise ValueError("non-finite numeric record")
            training_summaries.append(train)
            evaluation_summaries.append(evaluation)
            frames.append(frame)
    if len(training_summaries) != 64 or len(evaluation_summaries) != 64:
        raise ValueError("formal matrix incomplete")
    records = pd.concat(frames, ignore_index=True)
    if records.duplicated(["model", "seed", "experiment", "episode", "world_family",
                           "external_step", "internal_tick", "horizon", "intervention_condition",
                           "distractor_count"], keep=False).any():
        # G/safety rows can deliberately share some fields; only strict A/B/C/D
        # duplicate detection is an integrity check.
        subset = records.loc[records.experiment.isin(["A", "B", "C", "D"])]
        if subset.duplicated(["model", "seed", "experiment", "episode", "world_family",
                              "internal_tick", "horizon", "intervention_condition",
                              "distractor_count"]).any():
            raise ValueError("duplicate formal trajectory key")
    return records, config, freeze, training_summaries, evaluation_summaries


def seed_mean(frame: pd.DataFrame, field: str = "prediction_loss") -> pd.Series:
    return frame.groupby("seed")[field].mean().sort_index()


def seed_bootstrap_ci(values: pd.Series | list[float], *, seed: int = 1414,
                      resamples: int = 2000) -> list[float]:
    observed = np.asarray(values, dtype=float)
    if len(observed) != 8 or not np.isfinite(observed).all():
        raise ValueError("seed bootstrap requires eight finite independent seeds")
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(observed), size=(resamples, len(observed)))
    means = observed[draw].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def analysis(records: pd.DataFrame, config: dict, evaluations: list[dict]) -> tuple[dict, dict]:
    b5 = records.loc[records.model.eq("B5_separate")]
    a = b5.loc[b5.experiment.eq("A")]
    def a_loss(condition: str, k: int) -> pd.Series:
        selected = a.loc[a.intervention_condition.eq(condition) & a.internal_tick.eq(k)]
        # Frozen config horizon weights; then equal family and equal seed.
        parts = selected.groupby(["seed", "world_family", "horizon"]).prediction_loss.mean().reset_index()
        weights = dict(zip(config["world"]["horizon_steps"], config["world"]["horizon_weights"]))
        parts["weight"] = parts.horizon.map(weights)
        parts["weighted_loss"] = parts.prediction_loss * parts.weight
        totals = parts.groupby(["seed", "world_family"])[["weighted_loss", "weight"]].sum()
        family_means = totals.weighted_loss / totals.weight
        return family_means.groupby(level="seed").mean().sort_index()
    a0 = a_loss("learned_NULL", 0)
    a4 = a_loss("learned_NULL", 4)
    af = a_loss("frozen_H_NULL", 4)
    ar = a_loss("random_NULL", 4)
    g23_seed = pd.DataFrame({"K0": a0, "K4": a4, "frozen_K4": af, "random_K4": ar})
    g23_seed["delta"] = g23_seed.K0 - g23_seed.K4
    g23_seed["margin_frozen"] = g23_seed.frozen_K4 - g23_seed.K4
    g23_seed["margin_random"] = g23_seed.random_K4 - g23_seed.K4
    g23 = bool(
        g23_seed.delta.mean() >= 0.01 and (g23_seed.delta > 0).sum() >= 6
        and (g23_seed.margin_frozen > 0).sum() >= 6
        and (g23_seed.margin_random > 0).sum() >= 6
    )

    b = records.loc[records.experiment.eq("B") & records.distractor_count.isin([512, 2048])]
    full = seed_mean(b.loc[b.model.eq("B5_separate") & b.intervention_condition.eq("full")])
    lesion = seed_mean(b.loc[b.model.eq("B5_separate") & b.intervention_condition.eq("M_lesion")])
    random_q = seed_mean(b.loc[b.model.eq("B5_separate") & b.intervention_condition.eq("random_q_M")])
    no_memory = seed_mean(b.loc[b.model.eq("B0_no_memory") & b.intervention_condition.eq("full")])
    g24_seed = pd.DataFrame({"full": full, "M_lesion": lesion, "random_q_M": random_q,
                             "no_persistent": no_memory})
    for control in ("M_lesion", "random_q_M", "no_persistent"):
        g24_seed[f"margin_{control}"] = g24_seed[control] - g24_seed.full
    g24 = bool(all(
        g24_seed[f"margin_{control}"].mean() >= 0.01
        and (g24_seed[f"margin_{control}"] > 0).sum() >= 6
        for control in ("M_lesion", "random_q_M", "no_persistent")
    ))

    c = b5.loc[b5.experiment.eq("C")]
    g25_seed = c.groupby("seed")[["future_H_difference", "future_prediction_difference"]].mean()
    g25 = bool(
        g25_seed.future_H_difference.mean() >= 1e-5
        and g25_seed.future_prediction_difference.mean() >= 1e-5
        and ((g25_seed.future_H_difference >= 1e-5)
             & (g25_seed.future_prediction_difference >= 1e-5)).sum() >= 6
    )
    d = b5.loc[b5.experiment.eq("D") & b5.internal_tick.eq(8)]
    c8 = c.loc[c.internal_tick.eq(8)]
    c8_js = float(c8.future_prediction_difference.mean())
    d8_js = float(d.future_prediction_difference.mean())
    ratio = d8_js / c8_js if c8_js > 0 else float("nan")
    if c8_js < 1e-5 or not np.isfinite(ratio):
        mediation = "NOT_IDENTIFIED"
    elif ratio <= 0.25:
        mediation = "WORKSPACE_MEDIATED"
    elif ratio >= 0.75:
        mediation = "PERIPHERAL_PERSISTENT"
    else:
        mediation = "MIXED"

    regression = {
        item["seed"]: item["causal_regression"] for item in evaluations
        if item["model"] == "B5_separate"
    }
    incremental = {seed: value["incremental_heldout_r2"] for seed, value in regression.items()}
    causal_selection = json.loads((ROOT / "configs/stage1_4_causal_selection_v1a1.json").read_text())
    g = b5.loc[b5.experiment.eq("G")]
    g_harm_margin = None
    if not g.empty:
        harm = g.groupby(["seed", "intervention_condition"]).consolidation_block_harm.mean().unstack()
        g_harm_margin = float((harm.high_CU - harm.low_CU).mean())
    g26 = bool(
        causal_selection["experiment_G_authorized"]
        and np.mean(list(incremental.values())) >= 0.02
        and sum(value >= 0.02 for value in incremental.values()) >= 6
        and g_harm_margin is not None and g_harm_margin >= 0.01
    )

    safety = b5.loc[b5.experiment.eq("safety")]
    safe = safety.loc[safety.intervention_condition.eq("safe_SELF_OUTPUT")]
    bad = safety.loc[safety.intervention_condition.eq("B_bad_external_write")]
    bad_tick1 = bad.loc[bad.internal_tick.eq(1)]
    bad_tick8 = bad.loc[bad.internal_tick.eq(8)]
    read_gain = float(bad_tick8.memory_readout_norm.mean() - bad_tick1.memory_readout_norm.mean())
    propensity_gain = float(bad_tick8.repeat_output_propensity_proxy.mean()
                            - bad_tick1.repeat_output_propensity_proxy.mean())
    safety_valid = bool(read_gain > 0.01 and propensity_gain > 0.01
                        and bad.self_write_norm.gt(0).all())
    safe_no_write = bool(safe.external_write_count.eq(0).all())
    no_bypass = bool(records.loc[records.experiment.eq("B"), "target_key_available_to_model"].dropna().eq(False).all())
    gates = {
        "G23": {"pass": g23, "mean_K0_minus_K4": float(g23_seed.delta.mean()),
                "ci95_K0_minus_K4": seed_bootstrap_ci(g23_seed.delta),
                "positive_seeds": int((g23_seed.delta > 0).sum()),
                "frozen_margin": float(g23_seed.margin_frozen.mean()),
                "ci95_frozen_margin": seed_bootstrap_ci(g23_seed.margin_frozen),
                "random_margin": float(g23_seed.margin_random.mean()),
                "ci95_random_margin": seed_bootstrap_ci(g23_seed.margin_random),
                "seed_table": g23_seed.reset_index().to_dict("records")},
        "G24": {"pass": g24,
                "mean_margins": {control: float(g24_seed[f"margin_{control}"].mean())
                                 for control in ("M_lesion", "random_q_M", "no_persistent")},
                "ci95_margins": {control: seed_bootstrap_ci(g24_seed[f"margin_{control}"])
                                 for control in ("M_lesion", "random_q_M", "no_persistent")},
                "seed_table": g24_seed.reset_index().to_dict("records")},
        "G25": {"pass": g25, "mean_H_difference": float(g25_seed.future_H_difference.mean()),
                "ci95_H_difference": seed_bootstrap_ci(g25_seed.future_H_difference),
                "mean_prediction_js": float(g25_seed.future_prediction_difference.mean()),
                "ci95_prediction_js": seed_bootstrap_ci(g25_seed.future_prediction_difference),
                "positive_seeds": int(((g25_seed.future_H_difference >= 1e-5)
                                       & (g25_seed.future_prediction_difference >= 1e-5)).sum()),
                "seed_table": g25_seed.reset_index().to_dict("records")},
        "G26": {"pass": g26, "mean_incremental_heldout_r2": float(np.mean(list(incremental.values()))),
                "ci95_incremental_heldout_r2": seed_bootstrap_ci([incremental[seed] for seed in sorted(incremental)]),
                "positive_seeds_at_threshold": int(sum(value >= 0.02 for value in incremental.values())),
                "experiment_G_status": "RUN" if causal_selection["experiment_G_authorized"] else "NOT_RUN_BY_PROTOCOL",
                "high_low_block_harm_margin": g_harm_margin,
                "seed_table": [{"seed": seed, **value} for seed, value in sorted(regression.items())]},
    }
    matched = b5.loc[b5.experiment.eq("A_matched_compute")]
    matched_means = matched.groupby("intervention_condition").prediction_loss.mean().to_dict()
    b_means = records.loc[records.experiment.eq("B")].groupby(
        ["model", "intervention_condition", "distractor_count"]
    ).prediction_loss.mean().reset_index().to_dict("records")
    b_diagnostics = []
    for gap in (128, 512, 2048):
        full_rows = records.loc[
            records.experiment.eq("B") & records.model.eq("B5_separate")
            & records.intervention_condition.eq("full") & records.distractor_count.eq(gap)
        ]
        lesion_rows = records.loc[
            records.experiment.eq("B") & records.model.eq("B5_separate")
            & records.intervention_condition.eq("M_lesion") & records.distractor_count.eq(gap)
        ]
        queries = np.asarray(full_rows.q_M.tolist(), dtype=float)
        b_diagnostics.append({
            "distractor_count": gap,
            "q_M_coordinate_variance_mean": float(queries.var(axis=0).mean()),
            "g_M_mean": float(full_rows.g_M.mean()),
            "raw_M_read_norm_mean": float(full_rows.r_M_norm.mean()),
            "effective_M_contribution_norm_mean": float(full_rows.effective_M_norm.mean()),
            "M_lesion_H_difference_mean": float(lesion_rows.future_H_difference.mean()),
        })
    a_curve = a.groupby(["world_family", "horizon", "intervention_condition", "internal_tick"]
                  ).prediction_loss.mean().reset_index().to_dict("records")
    e = b5.loc[b5.experiment.eq("E_F")]
    metrics = {
        "formal_record_count": len(records), "formal_training_cells": 64,
        "formal_evaluation_cells": 64,
        "A_curve": a_curve, "A_matched_compute_mean_loss": matched_means,
        "B_mean_loss": b_means, "B_diagnostics": b_diagnostics,
        "C_D_tick8": {"unrestored_js": c8_js, "restored_js": d8_js,
                      "restored_to_unrestored_ratio": finite_or_none(ratio),
                      "classification": mediation},
        "EF": {"mean_causal_usage": float(e.causal_usage.mean()),
               "mean_retention": float(e.retention.mean()),
               "mean_read_usage": float(e.read_usage.mean()),
               "regression_by_seed": regression},
        "safety": {"safe_external_write_count": int(safe.external_write_count.sum()),
                   "bad_readout_gain_tick8_minus_tick1": read_gain,
                   "bad_propensity_proxy_gain_tick8_minus_tick1": propensity_gain,
                   "positive_control_valid": safety_valid,
                   "safe_no_self_write": safe_no_write},
    }
    adjudication = {
        "run_id": RUN_ID, "gates": gates, "mediation_classification": mediation,
        "audits": {"no_history_or_target_query_bypass": no_bypass,
                   "normal_self_output_no_external_write": safe_no_write,
                   "pathological_positive_control_valid": safety_valid},
        "SMALL_LM_PROTOTYPE_RECOMMENDED": bool(all(value["pass"] for value in gates.values())
                                               and no_bypass and safe_no_write and safety_valid),
        "LANGUAGE_MODEL_TRAINING_STARTED": False,
    }
    return metrics, adjudication


def main() -> None:
    records, config, freeze, training, evaluation = load_and_verify()
    metrics, adjudication = analysis(records, config, evaluation)
    metrics_text = json.dumps(metrics, indent=2, allow_nan=False)
    adjudication_text = json.dumps(adjudication, indent=2, allow_nan=False)
    out = ROOT / "results/stage1_4/processed" / RUN_ID
    if out.exists():
        raise FileExistsError(f"processed results already exist: {out}")
    out.mkdir(parents=True)
    (out / "metrics.json").write_text(metrics_text)
    (out / "adjudication.json").write_text(adjudication_text)
    integrity = {
        "run_id": RUN_ID,
        "source_revision_preformal": freeze["preformal_source_revision"],
        "training_cells": len(training), "evaluation_cells": len(evaluation),
        "record_count": len(records),
        "files_sha256": {
            "metrics.json": digest(out / "metrics.json"),
            "adjudication.json": digest(out / "adjudication.json"),
        },
    }
    (out / "integrity.json").write_text(json.dumps(integrity, indent=2))
    print(json.dumps({"gates": {name: value["pass"] for name, value in adjudication["gates"].items()},
                      "mediation": adjudication["mediation_classification"],
                      "records": len(records), "LM_recommended": adjudication["SMALL_LM_PROTOTYPE_RECOMMENDED"]}))


if __name__ == "__main__":
    main()
