"""Render detailed, append-only reports from immutable formal Stage 1.5 records."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RUN = "stage1_5-formal-v1"
PROCESSED = ROOT / "results/stage1_5/processed" / RUN
PENDING_REPORTS: dict[Path, str] = {}


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, (float, int)):
        return f"{value:.6f}" if isinstance(value, float) else str(value)
    return str(value)


def table(columns: tuple[str, ...], rows: list[tuple[object, ...]]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    out.extend("| " + " | ".join(fmt(item) for item in row) + " |" for row in rows)
    return "\n".join(out)


def records(experiment: str, variant: str = "B5_separate") -> pd.DataFrame:
    parts = [pd.read_parquet(
        ROOT / "results/stage1_5" / RUN / "evaluation" / experiment /
        f"{variant}_seed{seed}/records.parquet"
    ) for seed in range(8501, 8509)]
    return pd.concat(parts, ignore_index=True)


def write(name: str, title: str, methods: str, body: str, limitations: str) -> None:
    path = ROOT / "reports" / name
    if path.exists():
        raise FileExistsError(f"immutable report already exists: {path}")
    PENDING_REPORTS[path] = (
        f"# {title}\n\nFormal run `{RUN}`. Protocol: `reports/STAGE1_5_PROTOCOL.md`; "
        f"analysis plan: `reports/STAGE1_5_ANALYSIS_PLAN.md`. "
        f"Raw Parquet: `results/stage1_5/{RUN}/evaluation/`.\n\n"
        f"## Methods\n\n{methods}\n\n## Results\n\n{body}\n\n"
        f"## Scope and limitations\n\n{limitations}\n",
    )


def main() -> None:
    metrics = json.loads((PROCESSED / "metrics.json").read_text(encoding="utf-8"))
    gates = json.loads((PROCESSED / "adjudication.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((ROOT / "configs/stage1_5.yaml").read_text(encoding="utf-8"))
    if gates["G31"] == "PASS" and gates["G32"] == "PENDING_AUTHORIZED":
        raise RuntimeError("G31 authorized B5/B6; run conditional routing before final reports")
    if metrics["formal_evaluation_shards"] != 160:
        raise ValueError("formal evaluation incomplete")

    stable = records("stability")
    stable_curve = stable[stable.internal_tick.isin([0, 1, 8, 64, 256, 1024])].groupby(
        "internal_tick"
    )[["H_norm", "F_norm", "M_norm", "H_delta_norm", "prediction_js_from_t0", "prediction_entropy"]].mean()
    growth = records("perturbation")
    growth_rows = growth[growth.internal_tick.isin([1, 8, 32, 128])].groupby(
        ["perturbation_component", "epsilon", "internal_tick"]
    )[["H_response_gain", "full_state_gain", "prediction_js"]].mean().reset_index()
    write(
        "ARCHITECTURE_STABILITY_STAGE1_5.md", "Architecture Stability — Stage 1.5",
        "Four genuinely external-history worlds × 8 initial states × 8 fresh checkpoints. "
        "Every NULL tick through 1024 is recorded; no external writes occur. "
        "Finite H/F/M perturbations of 0.001 and 0.01 are tracked to 128; "
        "no JVP substitutes for these rollouts. G27 uses maximum state norm, "
        "prediction JS drift, and median finite-perturbation growth.",
        f"G27: **{gates['G27']}**. Per-seed maximums:\n\n" + table(
            ("Seed", "Max norm", "Max JS", "Max median growth@128", "Tail Δ", "Descriptive class", "Pass"), [
                (seed, one["max_component_norm"], one["max_prediction_js"],
                 one["max_median_finite_growth_128"], one["tail_median_total_delta"],
                 one["descriptive_dynamics_class"], one["pass"])
                for seed, one in metrics["G27"]["by_seed"].items()
            ]) + "\n\nMean trajectories:\n\n" + table(
            ("K", "||H||", "||F||", "||M||", "ΔH", "JS from K0", "Entropy"), [
                (int(k), *[stable_curve.loc[k, column] for column in stable_curve.columns])
                for k in stable_curve.index
            ]) + "\n\nPerturbation growth by component/epsilon/tick:\n\n" + table(
            ("Component", "ε", "K", "H response", "Full response", "Prediction JS"), [
                (row.perturbation_component, row.epsilon, int(row.internal_tick),
                 row.H_response_gain, row.full_state_gain, row.prediction_js)
                for row in growth_rows.itertuples()
            ]),
        "These are finite empirical trajectories on trained toy states. Boundedness "
        "over 1024 ticks is not a global stability theorem; perturbation growth "
        "is not called mathematical chaos.",
    )

    timescale = records("timescale")
    lag = timescale[timescale.decay_variant.eq("primary")].groupby(
        ["component", "history_lag"]
    )[["heldout_decoding_gain", "heldout_accuracy"]].mean().reset_index()
    write(
        "TIMESCALE_CHARACTERIZATION_STAGE1_5.md", "H/F/M Effective Timescales — Stage 1.5",
        "Per seed: 512 train and 256 untouched test histories, 513 real external "
        "events per history, random visible key/value writes. Train-only ridge "
        "probes decode historical visible values at lags 1–512 from H/F/M "
        "separately. Baseline is train-frequency constant CE. Ablations: primary, "
        "equal fast decay, equal slow decay, no transfer. Decoding gain is not MI.",
        f"G28: **{gates['G28']}**. Registered profile areas and ablations:\n\n" + table(
            ("Metric", "Mean", "95% seed CI", "Seeds > threshold"), [
                (name, metrics["G28"][key]["mean"], metrics["G28"][key]["ci95"],
                 metrics["G28"][key]["positive_seeds"])
                for name, key in (
                    ("F−H lag area", "F_minus_H_area"),
                    ("M−F lag area", "M_minus_F_area"),
                    ("minimum equal-decay difference", "equal_decay_ablation_min_difference"),
                )
            ]) + "\n\nPrimary held-out lag curve:\n\n" + table(
            ("State", "Lag", "CE gain (nats)", "Accuracy"), [
                (row.component, int(row.history_lag), row.heldout_decoding_gain, row.heldout_accuracy)
                for row in lag.itertuples()
            ]) + "\n\nFull per-seed/decay curves are in the Parquet shards.",
        "The probe distribution is much longer and more IID than training histories. "
        "Feature dimension differs by state. A decodable trace does not imply "
        "future use or a long-term causal memory.",
    )

    capacity_rows = []
    for variant, cell in metrics["capacity"].items():
        for n in (0, 128, 2048, 8192):
            one = cell[str(n)]
            capacity_rows.append((variant, n, one["prediction_CE"]["mean"],
                                  one["prediction_accuracy"]["mean"],
                                  one["M_only_retention"]["mean"],
                                  one["associative_accuracy_by_items"]["128"]))
    write(
        "CAPACITY_SCALING_STAGE1_5.md", "Capacity and Distractor Scaling — Stage 1.5",
        "Five preregistered sparse model sizes, 8 fresh seeds each; all capacity "
        "models have equal train budgets and development LR search. The trained "
        "prediction arm clones one external stream at distractor checkpoints "
        "0/32/128/512/2048/8192, inserts the same genuine bridge, then 4 NULL "
        "ticks. An independent FP32 continuous-vector associative-cell "
        "microbenchmark varies 1–128 stored items; it does not use learned routing.",
        "Selected points (all intermediate N/item cells remain machine-readable):\n\n" + table(
            ("Variant", "Distractors", "Future CE", "Accuracy", "M-only cosine", "128-item top1"),
            capacity_rows,
        ) + "\n\nModel bytes/parameters and approximate cell operations are recorded "
        "for every shard and capacity condition.",
        "The sparse grid couples H and memory dimensions except at H=256, so "
        "their separate causal scaling effects are not identified. Synthetic "
        "cell accuracy is a mechanistic storage benchmark, not world prediction; "
        "finite dimension and interference preclude infinite-capacity claims.",
    )

    anatomy = records("anatomy")
    a5 = anatomy[anatomy.experiment.eq("A5_anatomy") & anatomy.internal_tick.eq(4)]
    ce_effects = a5.groupby("swap_condition").signed_CE_effect.mean().to_dict()
    write(
        "HFM_CAUSAL_ANATOMY_STAGE1_5.md", "H/F/M Causal Anatomy — Stage 1.5",
        "At identical real external prefix, donor episode `ep−1` supplies exactly "
        "the specified H/F/M components. H, F, M, HF, HM, FM, HFM swaps are "
        "rolled for 1/2/4/8 NULL ticks. Target CE and prediction JS are paired "
        "against intact state; pairwise signed CE interactions are E_AB−E_A−E_B.",
        f"G29: **{gates['G29']}**. Tick-4 finite effects:\n\n" + table(
            ("Swap", "Prediction JS", "JS 95% CI", "Signed CE effect", "JS-positive seeds"), [
                (name, one["mean"], one["ci95"], ce_effects[name],
                 sum(value >= 1e-5 for value in one["seed_values"].values()))
                for name, one in metrics["G29"]["prediction_JS_by_channel"].items()
            ]) + "\n\nSigned interactions:\n\n" + table(
            ("Interaction", "Mean CE", "95% seed CI"), [
                (name, one["mean"], one["ci95"])
                for name, one in metrics["G29"]["pairwise_CE_interactions"].items()
            ]),
        "Swap effects are finite causal interventions for this checkpoint/world, "
        "not a general causal-memory property. Prediction JS, not H distance "
        "alone, determines G29.",
    )

    obs = metrics["observability_event_CE"]
    write(
        "ARCHITECTURAL_OBSERVABILITY_STAGE1_5.md", "Architectural Observability — Stage 1.5",
        "Frozen B5 states from four families; 64 train and 32 held-out examples "
        "per family, per seed. Train-only dual ridge fits predict future H, "
        "future logits and future events from H/HF/HM/HFM. Four future NULL "
        "ticks define the state target. No intervention label trains the probe.",
        "Held-out future-event CE:\n\n" + table(
            ("Features", "Mean CE", "95% seed CI", "ΔCE vs H-only"), [
                (name, one["mean"], one["ci95"], obs["H"]["mean"] - one["mean"])
                for name, one in obs.items()
            ]) + "\n\nFuture-H and future-logit MSE for every family/seed are in the Parquet records.",
        "Probe improvement is observational predictive sufficiency only. It "
        "cannot establish a peripheral causal pathway without A5/B2 finite swaps.",
    )

    med = metrics["G30"]["channels"]
    write(
        "PERIPHERAL_READ_MEDIATION_STAGE1_5.md", "Peripheral Read Mediation — Stage 1.5",
        "Paired F-only and M-only swaps; at every tick the corresponding "
        "*effective* read contribution is replaced by the intact branch's "
        "contribution, while the swapped stored tensor remains untouched. "
        "Absolute JS/CE are measured at 1/2/4/8; ratio only if swap JS>1e-5.",
        f"G30: **{gates['G30']}**. " + (
            "`UNMODELED_PERIPHERAL_PATHWAY` or unidentifiable read mediation.\n\n"
            if not metrics["G30"]["pass"] else "Read mediation meets registered criterion.\n\n"
        ) + table(
            ("Channel", "Relevant", "Swap JS", "Restored JS", "Mediation fraction", "Identifiable"), [
                (name, one["relevant"], one["swap_JS"]["mean"],
                 one["restore_JS"]["mean"],
                 one["mediation_fraction"]["mean"] if one["mediation_fraction"] else None,
                 one["identifiable"])
                for name, one in med.items()
            ]),
        "Mediation fraction is descriptive, not a nonparametric causal "
        "mediation identification theorem. Other state-dependent paths and "
        "post-intervention interactions remain possible.",
    )

    oracle = records("oracle")
    oracle_curve = oracle[oracle.internal_tick.eq(4)].groupby(
        ["distractor_count", "intervention_condition"]
    ).future_CE.mean().reset_index()
    write(
        "ORACLE_RETRIEVAL_CEILING_STAGE1_5.md", "Historical Oracle Retrieval Ceiling — Stage 1.5",
        "For each of 8 seeds: 16 paired long-gap episodes at 128/512/2048 "
        "distractors. Early four real events build a saved memory-read bank. "
        "The observed C→B bridge selects an earlier B source; the oracle "
        "injects only that historical read through the same slow-read "
        "interface for four ticks. It never reads future target labels. "
        "Zero-M, full no-read, norm-matched random, episode-shuffled and "
        "learned-read controls share checkpoint/compute.",
        f"G31: **{gates['G31']}**. Registered 512/2048, tick-4 comparator "
        "minus static-oracle CE margins:\n\n" + table(
            ("Comparator", "Mean CE advantage", "95% seed CI", "Positive seeds"), [
                (name, one["mean"], one["ci95"], one["positive_seeds"])
                for name, one in metrics["G31"]["oracle_CE_margins"].items()
            ]) + "\n\nTick-4 CE by gap and condition:\n\n" + table(
            ("Gap", "Condition", "CE"), [
                (int(row.distractor_count), row.intervention_condition, row.future_CE)
                for row in oracle_curve.itertuples()
            ]),
        "The oracle has privileged access to a saved pre-interference read. "
        "Even a positive ceiling would show integration capacity, not that "
        "the learned persistent M actually preserved or found that vector.",
    )

    write(
        "CLOSED_LOOP_RETRIEVAL_STAGE1_5.md", "Static versus Closed-Loop Oracle — Stage 1.5",
        "On the same long-gap bank, static oracle reuses the last matching "
        "historical contribution. Closed-loop oracle recomputes an H-dependent "
        "mixture of eligible historical reads after every transition; "
        "neither sees a future label. Compare with learned and zero read.",
        "Static CE minus closed-loop CE (positive favors closed-loop):\n\n" + table(
            ("Ticks", "Mean", "95% seed CI", "Positive seeds"), [
                (k, one["mean"], one["ci95"], one["positive_seeds"])
                for k, one in metrics["G31"]["closed_loop_minus_static_advantage"].items()
            ]),
        "This diagnostic H-dependent oracle-bank rule is not learned routing. "
        "Eligible sources may be identical, so closed-loop improvement is "
        "not guaranteed by construction.",
    )

    conditional = (
        "G31 did not pass its frozen oracle ceiling. Accordingly this "
        "experiment is **NOT_RUN_BY_PROTOCOL**. No finite advantage, "
        "student routing, routed NULL time, or secondary utility-retention "
        "effect is imputed from proxy quantities."
    )
    for filename, title, method in (
        ("FINITE_RETRIEVAL_ADVANTAGE_STAGE1_5.md", "Finite Retrieval Advantage", "B5 requires G31 PASS; candidates would be real historical reads, scored by paired finite CE intervention."),
        ("CAUSAL_ROUTING_DISTILLATION_STAGE1_5.md", "Causal Routing Distillation", "B6 requires G31 plus stable held-out finite advantage; no target-key, future-use or remember labels are allowed."),
        ("ROUTED_INTERNAL_TIME_STAGE1_5.md", "Routed Internal Time", "B7 requires learned routing gate G32 PASS; each NULL tick would route anew with matched frozen/random controls."),
    ):
        write(filename, f"{title} — Stage 1.5", method, conditional,
              "Conditional non-execution is not a negative measured effect.")

    precision = pd.read_parquet(PROCESSED / "precision.parquet")
    precision_end = precision[precision.step.eq(1000)]
    write(
        "PRECISION_MEMORY_AUDIT_STAGE1_5.md", "Small-Update Precision Audit — Stage 1.5",
        "Exploratory scalar rank-one-direction reduction: 1000 repeated "
        "consolidations with γ=1e-5 and no decay/external events. FP32 "
        "storage, BF16 storage, and FP32 shadow accumulation with BF16 "
        "consumption are compared to `(1−γ)^K`. This does not enter gates.",
        "At update 1000:\n\n" + table(
            ("Storage/consumption", "F", "M", "F+M", "Analytic F error"), [
                (row.precision_mode, row.F_storage, row.M_storage,
                 row.total_storage, row.analytic_F_error)
                for row in precision_end.itertuples()
            ]),
        "This is a scalar precision regression, not a BF16 full-model "
        "benchmark. FP32 remains the formal scientific condition.",
    )

    answers = [
        f"1. Long NULL rollout: G27 **{gates['G27']}**; max component norm across seeds "
        f"{max(v['max_component_norm'] for v in metrics['G27']['by_seed'].values()):.4f}.",
        f"2. Effective H/F/M timescale separation: G28 **{gates['G28']}**; "
        f"F−H area {metrics['G28']['F_minus_H_area']['mean']:.5f}, "
        f"M−F area {metrics['G28']['M_minus_F_area']['mean']:.5f}.",
        "3. Decay versus dynamics/usage: equal-fast, equal-slow and no-transfer "
        "probe curves are reported; the registered ablation criterion "
        f"{'passes' if metrics['G28']['equal_decay_ablation_min_difference']['positive_seeds'] >= 6 else 'fails'}. "
        "These probes do not identify a unique causal decomposition.",
        "4. Capacity: full sparse-grid × stored-item × distractor curves are "
        "reported; trained prediction and mechanistic cell retrieval are separate.",
        f"5. Independent H/F/M future effects: tick-4 JS means H={metrics['G29']['prediction_JS_by_channel']['H']['mean']:.6g}, "
        f"F={metrics['G29']['prediction_JS_by_channel']['F']['mean']:.6g}, "
        f"M={metrics['G29']['prediction_JS_by_channel']['M']['mean']:.6g}; G29 **{gates['G29']}**.",
        "6. F×M, H×F and H×M signed CE interactions: " + ", ".join(
            f"{k}={metrics['G29']['pairwise_CE_interactions'][k]['mean']:.6g}"
            for k in ("FM", "HF", "HM")
        ) + ".",
        f"7. H-only held-out future-event CE={obs['H']['mean']:.4f}; compare HF/HM/HFM in observability report.",
        f"8. Extra observational F/M information beyond H: HF gain={obs['H']['mean']-obs['HF']['mean']:.5f}, "
        f"HM gain={obs['H']['mean']-obs['HM']['mean']:.5f}, "
        f"HFM gain={obs['H']['mean']-obs['HFM']['mean']:.5f} CE.",
        f"9. Extra causal F/M information is supported only to G29's finite-swap scope: **{gates['G29']}**; "
        "probe gain alone is insufficient.",
        f"10. Explicit read mediation: G30 **{gates['G30']}**; see absolute JS and conditional ratios.",
        f"11. Correct historical oracle read improves all registered controls: G31 **{gates['G31']}**.",
        f"12. Oracle minus learned CE advantage={metrics['G31']['oracle_CE_margins']['learned']['mean']:.5f} "
        "(positive means oracle better).",
        f"13. Static CE minus closed-loop CE at K4={metrics['G31']['closed_loop_minus_static_advantage']['4']['mean']:.5f}.",
        f"14. Finite RetrievalAdvantage signal: {gates['G32']} (B5 conditional prerequisite).",
        f"15. Learned state-dependent finite-benefit router: {gates['G32']}.",
        f"16. Held-out routing generalization: {gates['G32']}.",
        f"17. Routed NULL-tick prediction benefit: {gates['G33']}.",
        "18. Current bottleneck follows the frozen decision tree: " + (
            "architecture stability" if gates["G27"] == "FAIL" else
            "unlocalized causal anatomy" if gates["G29"] == "FAIL" else
            "read mediation / unmodeled peripheral pathway" if gates["G30"] == "FAIL" else
            "integration or historical-read ceiling" if gates["G31"] == "FAIL" else
            "routing / continuous time, pending conditional tests"
        ) + ". Capacity/timescale limits are reported separately.",
        "19. Causal utility should not enter consolidation law in this stage: "
        "routing/finite-use prerequisites have not all been met; transfer law was unchanged.",
        "20. Small sequence/language prototype recommendation: **FALSE**; no LM was trained.",
    ]
    final = ROOT / "reports/STAGE1_5_FINAL_REPORT.md"
    if final.exists():
        raise FileExistsError(final)
    PENDING_REPORTS[final] = (
        "# ET-RCM Stage 1.5 Final Report\n\n"
        "> **What dynamical and causal structure does the ET-RCM architecture itself possess, and can a learned state-dependent routing policy select the peripheral information whose finite use actually improves future computation?**\n\n"
        "> **ET-RCM 架构自身究竟具有怎样的动力学与因果结构；同时，一个可学习的状态依赖路由机制，能否从外围持续状态中选择那些经真实有限干预验证、确实能够改善未来计算的信息？**\n\n"
        f"Formal run `{RUN}`: 96 new train cells (8 inherited architectures "
        "plus 4 additional capacity variants × 8 seeds), "
        f"{metrics['formal_evaluation_shards']} independent evaluation shards, "
        f"{metrics['formal_evaluation_rows']:,} machine-readable formal rows. "
        "Development: 48 train cells, 2 seeds × 2 LRs for 12 variants. "
        "All source/checkpoint/config/record hashes are checked in "
        f"`results/stage1_5/processed/{RUN}/integrity.json`. "
        "Historic Stage 1.4 files remain unchanged.\n\n"
        "## Registered gates\n\n" + table(
            ("Gate", "Outcome", "Criterion"), [
                (name, gates[name], description) for name, description in (
                    ("G27", "numerical/finite dynamics"),
                    ("G28", "nontrivial H/F/M lag hierarchy"),
                    ("G29", "single peripheral predictive causal effect"),
                    ("G30", "explicit read mediation"),
                    ("G31", "historical oracle ceiling"),
                    ("G32", "finite-benefit learned routing"),
                    ("G33", "routed NULL-time gain"),
                )
            ]) + "\n\n"
        "## Experimental design\n\n"
        "FP32 primary. Eight independent seeds 8501–8508, 160 AdamW steps, "
        "batch 32, equal four-family world training; two development seeds "
        "8401–8402 chose LR from 0.001/0.0003 using held-out CE before "
        "formal training. A1 has 4×8 real-history states per seed and every "
        "tick through 1024; A2 uses three independently perturbed components, "
        "two finite epsilons and nine checkpoints. A3 probes 768 histories per "
        "seed under four decay/transfer variants; A4 has five sparse capacities "
        "and six distractor lengths. A5 uses 32 paired states and seven exact "
        "swaps at four horizons; A6 uses 64 train + 32 test examples per "
        "family. B2 read-clamps F/M independently; B3/B4 use 16 paired "
        "episodes per seed/gap and matched controls. The historical oracle "
        "bank sees early real external evidence and the observed bridge only, "
        "never future targets. Full details are in the 12 topic reports.\n\n"
        "## Direct answers to the 20 registered questions\n\n" + "\n".join(
            f"{answer}\n" for answer in answers
        ) + "\n"
        "## Scientific interpretation and boundaries\n\n"
        "Stored information, nonzero read, predictive probe gain, finite causal "
        "benefit and long-term retention are different observables. The "
        "historical oracle's saved pre-interference bank is an upper-bound "
        "intervention, not evidence that current M can retrieve it. Finite "
        "toy swap effects do not justify a general causal-memory label. "
        "No consciousness, human-like autonomy, unlimited capacity, or "
        "spontaneous language content is claimed. All failed, null and "
        "conditionally unexecuted outcomes are preserved; no gate was "
        "changed after formal results.\n",
    )
    for path in PENDING_REPORTS:
        if path.exists():
            raise FileExistsError(f"immutable report already exists: {path}")
    for path, content in PENDING_REPORTS.items():
        path.write_text(content, encoding="utf-8")
    print(f"wrote Stage 1.5 reports; G27–G33 = {[gates[f'G{i}'] for i in range(27, 34)]}")


if __name__ == "__main__":
    main()
