"""Render Stage 1.4 topic and final Markdown from immutable processed JSON."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = "stage1_4-formal-v1a1"
QUESTION = (
    "> **Can a continuously running predictive state autonomously reactivate old persistent "
    "information when that information causally improves future prediction, and does such "
    "causal usefulness explain which transient states should acquire longer memory lifetimes?**\n\n"
    "> **一个持续运行的预测状态系统，能否在旧信息真正能够改善未来预测时自主重新激活这些持久状态；同时，"
    "这种对未来计算的因果效用，能否解释哪些短暂状态应该获得更长的记忆寿命？**\n"
)


def fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def mean_b(metrics: dict, model: str, condition: str, gaps=(512, 2048)) -> float:
    selected = [row["prediction_loss"] for row in metrics["B_mean_loss"]
                if row["model"] == model and row["intervention_condition"] == condition
                and row["distractor_count"] in gaps]
    return sum(selected) / len(selected) if selected else float("nan")


def write_new(name: str, content: str) -> None:
    path = ROOT / "reports" / name
    if path.exists():
        raise FileExistsError(f"report already exists: {path}")
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def main() -> None:
    processed = ROOT / "results/stage1_4/processed" / RUN
    metrics = json.loads((processed / "metrics.json").read_text())
    judgement = json.loads((processed / "adjudication.json").read_text())
    causal_selection = json.loads((ROOT / "configs/stage1_4_causal_selection_v1a1.json").read_text())
    gates = judgement["gates"]
    g23, g24, g25, g26 = (gates[key] for key in ("G23", "G24", "G25", "G26"))
    safety = metrics["safety"]
    mediation = metrics["C_D_tick8"]
    no_lm = not judgement["SMALL_LM_PROTOTYPE_RECOMMENDED"]
    a_rows = [row for row in metrics["A_curve"] if row["intervention_condition"] == "learned_NULL"
              and row["horizon"] == 1]
    k_curve = {}
    for k in (0, 1, 2, 4, 8, 16):
        values = [row["prediction_loss"] for row in a_rows if row["internal_tick"] == k]
        k_curve[k] = sum(values) / len(values) if values else float("nan")
    curve_table = "\n".join(f"| {k} | {fmt(loss)} |" for k, loss in k_curve.items())
    seed_table = "\n".join(
        f"| {row['seed']} | {fmt(row['delta'])} | {fmt(row['margin_frozen'])} | {fmt(row['margin_random'])} |"
        for row in g23["seed_table"]
    )
    predictive = f"""# Predictive Continuous Dynamics — Stage 1.4

{QUESTION}

Eight formal B5 models forecast horizons 1/2/4/8 after the same external
history and K=0/1/2/4/8/16 NULL ticks. Each prefix/target is paired across
K; the only changing input is internal time. Frozen-H and fixed random
recurrent NULL controls use the same pre-event transition count. The matched
compute timing arm places K ticks either before or after the next external
event and forecasts the subsequent event; post-event ticks incur K ticks of
future latency and cannot count as pre-event forecasting.

| K | Mean h=1 CE |
|---:|---:|
{curve_table}

| Seed | L0−L4 | Frozen−learned at K4 | Random−learned at K4 |
|---:|---:|---:|---:|
{seed_table}

G23 **{'PASS' if g23['pass'] else 'FAIL'}**: mean L0−L4={fmt(g23['mean_K0_minus_K4'])},
positive seeds={g23['positive_seeds']}/8, frozen margin={fmt(g23['frozen_margin'])},
random margin={fmt(g23['random_margin'])}. The registered criterion is not
changed for non-monotone curves. Matched timing mean losses:
`{json.dumps(metrics['A_matched_compute_mean_loss'], sort_keys=True)}`.
These are toy-world predictive losses, not evidence of human-like thought.
"""
    write_new("PREDICTIVE_CONTINUOUS_DYNAMICS_STAGE1_4.md", predictive)

    b_table = "\n".join(
        f"| {condition} | {fmt(mean_b(metrics, 'B5_separate', condition))} |"
        for condition in ("full", "M_lesion", "F_lesion", "random_q_M", "shuffled_M")
    )
    baseline_table = "\n".join(
        f"| {model} | {fmt(mean_b(metrics, model, 'full'))} |"
        for model in ("B0_no_memory", "B1_gru", "B2_single_memory", "B3_joint",
                      "B4_shared", "B5_separate", "B6_gamma_zero", "B7_random_query")
    )
    reactivation = f"""# Autonomous Memory Reactivation — Stage 1.4

{QUESTION}

Each long-gap episode has early A→B evidence (1/2/4 true exposures), 128/512/
2048 unrelated writes, a genuine C→B bridge event, and a future A-dependent
token. The model sees the bridge, then four NULL ticks, but **never** receives
a target-key query, target-key auxiliary loss, future-use flag, or future token
as input. Same-checkpoint B5 M/F lesions, random q_M and shuffled M are made
before the bridge. Other baselines were trained independently with equal
development and formal budgets. CE below averages the 512/2048 gaps.

| B5 condition | Future-event CE |
|---|---:|
{b_table}

| Trained architecture | Full-condition CE |
|---|---:|
{baseline_table}

G24 **{'PASS' if g24['pass'] else 'FAIL'}**. Registered control-minus-full
margins: `{json.dumps(g24['mean_margins'], sort_keys=True)}`. Nonzero q_M,
gate weight or read norm is descriptive access, not causal proof. Only paired
lesion and prediction effects count. The M lesion is applied just before the
bridge with H held identical; it tests *reactivation at that point*, not every
possible earlier M→H influence. Learned keys are not orthogonal, and 2048
distractors are an OOD gap relative to the 16-step training worlds.
"""
    write_new("AUTONOMOUS_MEMORY_REACTIVATION_STAGE1_4.md", reactivation)

    peripheral = f"""# Peripheral Causal Intervention — Stage 1.4

{QUESTION}

For 32 paired episodes per each of 8 trained B5 seeds, the manipulated state
copies **exactly** the intact active H and replaces F/M with another episode's
peripheral tensors. Both arms receive the same zero external event over ticks
1/2/4/8. This is an intervention on state, not a correlation of read norm.
The future-event head uses H only, so predictive JS must be mediated by at
least one changed H transition; persistent F/M can still re-enter H later.

G25 **{'PASS' if g25['pass'] else 'FAIL'}**: mean future-H distance
{fmt(g25['mean_H_difference'])}, mean future-prediction JS
{fmt(g25['mean_prediction_js'])}, joint-positive seeds
{g25['positive_seeds']}/8. The predictive endpoint, not H distance alone,
determines whether a peripheral causal effect was identified. This remains
a toy-scale same-H intervention, not a claim of general causal memory.
"""
    write_new("PERIPHERAL_CAUSAL_INTERVENTION_STAGE1_4.md", peripheral)

    workspace = f"""# Workspace Mediation — Stage 1.4

{QUESTION}

After one same-H peripheral-swap NULL transition, the restoration arm replaces
only its new H with the intact arm's H; its altered F/M remain. Both arms then
continue with identical NULL events through tick 8. At tick 8, mean prediction
JS is {fmt(mediation['unrestored_js'])} without restoration and
{fmt(mediation['restored_js'])} with restoration; the ratio is
{fmt(mediation['restored_to_unrestored_ratio'])}. Registered descriptive
classification: **{mediation['classification']}**. This is not a PASS/FAIL
gate. A low effect cannot establish mediation; it is NOT_IDENTIFIED.
"""
    write_new("WORKSPACE_MEDIATION_STAGE1_4.md", workspace)

    ef = metrics["EF"]
    causal = f"""# Causal Memory Usage — Stage 1.4

{QUESTION}

For each of 32 memory-item directions per formal seed, a rank-one key-direction
component is deleted from M before the bridge; the full and lesioned arms see
the same bridge, two NULL ticks and future target. `CU=L_lesion−L_full` uses
future-event CE, and the future-H distance is separately recorded. Exposure,
query alignment/access count and effective M-read magnitude are separately
logged. The mean measured CU is {fmt(ef['mean_causal_usage'])}; mean read
usage is {fmt(ef['mean_read_usage'])}. A direction lesion can overlap other
nonorthogonal learned keys, so it is not a perfect isolated factual item.
Access alone is never labeled causal influence.
"""
    write_new("CAUSAL_MEMORY_USAGE_STAGE1_4.md", causal)

    regression_table = "\n".join(
        f"| {seed} | {fmt(value['exposure_retention_spearman'])} | {fmt(value['read_retention_spearman'])} | "
        f"{fmt(value['causal_retention_spearman'])} | {fmt(value['incremental_heldout_r2'])} |"
        for seed, value in sorted(ef["regression_by_seed"].items(), key=lambda x: int(x[0]))
    )
    retention = f"""# Causal Usage Versus Retention — Stage 1.4

{QUESTION}

After 128 interference writes, F is excluded and each fact's slow-memory
read is compared by cosine to its originally written value vector. Mean M-only
retention={fmt(ef['mean_retention'])}. Fourfold within-seed held-out rank
regression compares exposure+read against exposure+read+CU; seeds, not items,
are replication units. Spearman coefficients are descriptive.

| Seed | Exposure–retention ρ | Read–retention ρ | CU–retention ρ | Incremental R² |
|---:|---:|---:|---:|---:|
{regression_table}

G26 **{'PASS' if g26['pass'] else 'FAIL'}**: mean incremental held-out R²
{fmt(g26['mean_incremental_heldout_r2'])}, seeds at registered +0.02 floor
{g26['positive_seeds_at_threshold']}/8. Correlation cannot substitute for the
required consolidation intervention.
"""
    write_new("CAUSAL_USAGE_RETENTION_STAGE1_4.md", retention)

    development_cu = causal_selection["incremental_heldout_r2_by_seed"]
    consolidation = f"""# Consolidation Intervention — Stage 1.4

{QUESTION}

Development B5 causal-use incremental held-out R² by seed was
`{json.dumps(development_cu, sort_keys=True)}`; mean
`{fmt(causal_selection['mean_incremental_heldout_r2'])}`. The pre-formal
selection froze `experiment_G_authorized={fmt(causal_selection['experiment_G_authorized'])}`.
The matched direction-specific F→M block status is
**{g26['experiment_G_status']}**. Its implementation and conservation unit
test remain in the repository, but no behavioral harm estimate is invented
when G is unauthorized. G26 cannot pass without the confirmatory
intervention. Any later study needs new splits and protocol.
"""
    write_new("CONSOLIDATION_INTERVENTION_STAGE1_4.md", consolidation)

    self_output = f"""# Self-Output Pathological Positive Control — Stage 1.4

{QUESTION}

The normal validated SELF_OUTPUT transition has zero external writes across
all formal records (count={safety['safe_external_write_count']}). In a separate
explicitly unsafe B_bad arm, each repeated self-output deliberately executes
the external delta write. From tick 1 to tick 8, its memory readout strength
changes by {fmt(safety['bad_readout_gain_tick8_minus_tick1'])}; a **fixed
diagnostic proxy**, sigmoid(4×readout−2), changes by
{fmt(safety['bad_propensity_proxy_gain_tick8_minus_tick1'])}. Positive-control
sensitivity is **{'VALID' if safety['positive_control_valid'] else 'INVALID'}**.
The proxy is not a trained expression policy, confidence estimate, or claim
that an agent would literally speak more often. If INVALID, the normal arm's
zero writes cannot by itself validate a full amplification audit.
"""
    write_new("SELF_OUTPUT_CONTROL_STAGE1_4.md", self_output)

    gate_table = "\n".join(
        f"| {name} | {'PASS' if value['pass'] else 'FAIL'} | "
        + ({"G23": f"ΔL4={fmt(g23['mean_K0_minus_K4'])}; {g23['positive_seeds']}/8",
            "G24": f"M lesion margin={fmt(g24['mean_margins']['M_lesion'])}",
            "G25": f"JS={fmt(g25['mean_prediction_js'])}; {g25['positive_seeds']}/8",
            "G26": f"incremental R²={fmt(g26['mean_incremental_heldout_r2'])}; G={g26['experiment_G_status']}"}[name])
        + " |" for name, value in gates.items()
    )
    answers = [
        f"NULL ticks: G23 {'passes' if g23['pass'] else 'fails'}; K0−K4 CE={fmt(g23['mean_K0_minus_K4'])} ({g23['positive_seeds']}/8 seeds positive).",
        f"Matched compute: frozen/random K4 margins are {fmt(g23['frozen_margin'])}/{fmt(g23['random_margin'])}; post-event timing losses are {metrics['A_matched_compute_mean_loss']}, with post-event latency K.",
        f"Autonomous long-gap access is {'supported' if g24['pass'] else 'not established'} under G24; no target query was supplied.",
        f"M-lesion minus intact CE={fmt(g24['mean_margins']['M_lesion'])}; compare direction and replication in G24 table.",
        f"Random-q_M minus intact CE={fmt(g24['mean_margins']['random_q_M'])}; a descriptive q trajectory alone is insufficient.",
        f"Separate B5 versus shared B4 CE at 512/2048 gaps: {fmt(mean_b(metrics,'B5_separate','full'))} versus {fmt(mean_b(metrics,'B4_shared','full'))}; this is a separately trained architecture comparison.",
        f"Same-H peripheral swap changes future H by {fmt(g25['mean_H_difference'])} and prediction JS by {fmt(g25['mean_prediction_js'])}; G25 {'passes' if g25['pass'] else 'fails'}.",
        f"H restoration leaves tick-8 JS ratio {fmt(mediation['restored_to_unrestored_ratio'])}; classification {mediation['classification']}.",
        f"F/M are persistent computational state only to the extent established by G25; otherwise this remains a structural hypothesis. Classification: {mediation['classification']}.",
        "Exposure/read/CU associations with retention are listed seed-by-seed in the causal-usage retention report; no single read norm proves causal use.",
        f"CU incremental held-out R²={fmt(g26['mean_incremental_heldout_r2'])}; G26 {'passes' if g26['pass'] else 'fails'}.",
        f"High-CU consolidation block is {g26['experiment_G_status']}; no harm contrast can be claimed.",
        f"Selective predictive reuse is {'supported' if g24['pass'] and g26['pass'] else 'not established'} by the combined reactivation and causal-use tests.",
        "The world-prediction objective replaces Stage 1.3 expression training, but solving its objective mismatch requires predictive and retrieval gates, not low training loss alone.",
        "Expression should remain secondary until predictive/retrieval controls validate content; it was not a Stage 1.4 gate.",
        f"Small LM prototype recommendation is {'TRUE' if not no_lm else 'FALSE'}; no LM was trained.",
        f"Gate-specific bottlenecks: predictive dynamics G23={'PASS' if g23['pass'] else 'FAIL'}, autonomous retrieval/read interface G24={'PASS' if g24['pass'] else 'FAIL'}, peripheral causal effect G25={'PASS' if g25['pass'] else 'FAIL'}, and causal-utility persistence/consolidation G26={'PASS' if g26['pass'] else 'FAIL'}; model capacity is not isolated by this protocol.",
        "All effects remain toy-scale: 24-symbol worlds, 64-d hidden state, short training histories, controlled interventions, and synthetic long gaps. No consciousness, general intelligence or infinite capacity is inferred.",
    ]
    answer_md = "\n".join(f"{index}. {answer}" for index, answer in enumerate(answers, 1))
    final = f"""# ET-RCM Stage 1.4 Final Report

{QUESTION}

Formal run `{RUN}` contains {metrics['formal_training_cells']} independently
trained model/seed cells (8 trainable architectures × 8 new seeds),
{metrics['formal_evaluation_cells']} corresponding evaluation shards, and
{metrics['formal_record_count']:,} machine-readable evaluation records.
Development used 2 seeds × 2 learning rates × 8 architectures, 100 steps
each; all selected 0.001 by held-out multi-horizon CE. Formal training used
160 steps, batch 32, AdamW and four equal-budget world families. Source,
config, selection, checkpoint and record hashes are verified in
`results/stage1_4/processed/{RUN}/integrity.json` and the frozen manifest.

## Registered outcome

| Gate | Verdict | Primary observation |
|---|---|---|
{gate_table}

Mediation classification: **{mediation['classification']}** (not a gate).
No target/history bypass: {fmt(judgement['audits']['no_history_or_target_query_bypass'])}.
Normal SELF_OUTPUT no external write: {fmt(judgement['audits']['normal_self_output_no_external_write'])}.
Pathological-control audit valid: {fmt(judgement['audits']['pathological_positive_control_valid'])}.
`SMALL_LM_PROTOTYPE_RECOMMENDED = {fmt(judgement['SMALL_LM_PROTOTYPE_RECOMMENDED'])}`.
`LANGUAGE_MODEL_TRAINING_STARTED = FALSE`.

## Experimental methods and exact comparisons

**Worlds.** Latent-transition episodes filter a noisy 8-state trajectory;
latent-regime episodes infer one of four enduring token regimes from an early
event; long-gap relation episodes write B→A in 1/2/4 genuine exposures,
insert 128/512/2048 unrelated memory writes, observe C→B, then forecast an
A-dependent future token; distractor-heavy episodes hold a rare cue amid
frequent future-irrelevant writes. Training length is 16 external events.
Targets are future-shifted by horizons 1/2/4/8 and never supplied as current
events or query labels. The formal 2048 gap is out of training distribution.

**A, predictive internal time.** B5 predicts after 0/1/2/4/8/16 NULL ticks
from the same state/history. Frozen-H and fixed random recurrent controls use
the same tick count. An additional matched timing arm places K compute before
versus after the next event while forecasting the subsequent event; latency
differs and it is not substituted for pre-event prediction. See the full
family×horizon×K curve and all eight seed margins in
`PREDICTIVE_CONTINUOUS_DYNAMICS_STAGE1_4.md`.

**B, autonomous reactivation.** Paired same-checkpoint M/F lesions, random
q_M and shuffled M are applied before the bridge. B0/no-memory, B1/GRU,
B2/single memory, B3/joint, B4/shared, B6/gamma-zero and B7/random query are
separately trained with the same hyperparameter search budget. CE is measured
after four post-bridge NULL ticks. No explicit A key is input. See
`AUTONOMOUS_MEMORY_REACTIVATION_STAGE1_4.md` for every 512/2048 contrast.

**C/D, causal state and mediation.** H is cloned exactly, F/M are swapped
across episodes, then both states receive 1/2/4/8 zero-input ticks. C measures
future-H norm distance and prediction JS; state distance alone cannot pass.
D replaces only H after tick one in the swapped arm, leaving F/M swapped,
then measures the remaining tick-8 JS ratio. See the two intervention reports.

**E/F, causal use versus retention.** Thirty-two item directions per seed
have M rank-one key-direction lesions; future CE differences define CU.
Read usage, true exposures and M-only cosine retention after 128 distractors
are separate quantities. Fourfold held-out rank regression tests whether CU
adds value beyond exposure/read. Learned keys are nonorthogonal, so lesion
overlap is a limitation. All seed correlations and R² values are preserved.

**G and safety.** The pre-formal development CU decision was
`{fmt(causal_selection['experiment_G_authorized'])}`; high-/low-CU
consolidation blocking is `{g26['experiment_G_status']}` when unauthorized.
This decision cannot be reversed by formal results. Normal SELF_OUTPUT
never calls external write; the separate B_bad positive control deliberately
does and is judged by memory readout plus a clearly labeled fixed propensity
proxy, not a learned expression policy.

## Answers to the 18 required questions

{answer_md}

## Scientific boundaries

Predictive usefulness, read/access, causal influence, and persistent retention
are reported separately. No result is called consciousness, human-like
thought, self-awareness, general intelligence, infinite context/capacity, or
causal memory merely because a prediction score improves. Failures and null
effects remain in the raw Parquet shards and gate JSON; no threshold was
changed after formal outcomes.

The first Stage 1.4 development/formal attempt is preserved but invalidated
by Amendment A4: its latent-transition event leaked hidden z in the key field.
The corrected v1a1 run uses new development/formal seeds and enforces a
partially observed latent key plus distinct B/C relation symbols. No metric
from the invalid attempt contributes to this adjudication.
"""
    write_new("STAGE1_4_FINAL_REPORT.md", final)
    print(json.dumps({"reports_created": 9, "run_id": RUN}))


if __name__ == "__main__":
    main()
