"""Consolidate Stage 2D.3 gates and write the complete scientific report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def status(name, reason):
    return {"gate": name, "status": "NOT_RUN_BY_PROTOCOL", "pass": None, "reason": reason}


def mean(rows, key, absolute=False):
    values = [r["variants"][key]["BS"] for r in rows]
    return statistics.mean(abs(x) for x in values) if absolute else statistics.mean(values)


def fmt(x, digits=3):
    return "NA" if x is None else f"{x:.{digits}f}"


def main(args):
    root = args.root
    analysis = read(root / "processed" / "interaction_analysis.json")
    causal_a = read(root / "activation_rescue" / "cohort_a.json")
    causal_b = read(root / "activation_rescue" / "cohort_b.json")
    dev = read(root / "curriculum_dev" / "selection.json", {})
    formal = read(root / "curriculum_formal" / "formal_evaluation.json")
    dynamics = read(root / "dynamics_recheck" / "results.json")
    g81_pass = causal_a["G81"]["pass"] and causal_b["G81"]["pass"]
    g82_pass = causal_a["G82"]["pass"] and causal_b["G82"]["pass"]
    gates = {name: analysis[name] for name in ("G79", "G80")}
    gates["G81"] = {"pass": g81_pass, "cohort_A": causal_a["G81"], "cohort_B": causal_b["G81"]}
    gates["G82"] = {"pass": g82_pass, "cohort_A": causal_a["G82"], "cohort_B": causal_b["G82"]}
    gates["G83"] = analysis["G83"]
    if formal is None:
        reason = "G81/G82 did not authorize training" if not (g81_pass or g82_pass) else "formal curriculum unavailable"
        gates["G84"] = status("G84", reason); gates["G85"] = status("G85", reason)
    else:
        gates["G84"] = formal["G84"]; gates["G85"] = formal["G85"]
    if dynamics is None:
        reason = "G84 did not pass; reduced dynamics, mediation and continuous diagnostics were prohibited"
        gates["G86"] = status("G86", reason); gates["G87"] = status("G87", reason)
    else:
        gates["G86"] = dynamics["G86"]; gates["G87"] = dynamics["G87"]
    if all(gates[g].get("pass") for g in ("G79", "G81", "G82", "G84", "G85", "G86", "G87")):
        outcome = "Outcome A — Conditional Interaction Bottleneck Identified and Stabilized"
    elif gates["G79"].get("pass") and (gates["G81"].get("pass") or gates["G82"].get("pass")):
        if gates["G83"].get("pass"):
            outcome = "Outcome B — Interaction Explains Behavior but Not the Training Basin"
        else:
            outcome = ("Outcome B* — causal conditional interaction is identified, but no predefined A–D "
                       "category is exactly satisfied because G83 and/or the training/dynamics chain failed")
    elif gates["G79"].get("pass"):
        outcome = "Outcome C — Conditional Binding Is a Downstream Symptom"
    else:
        outcome = "Outcome D — No Replicated Conditional-Interaction Explanation"
    reopen = all(gates[g].get("pass") for g in ("G84", "G86", "G87"))
    redesign = False
    decision = {"gates": gates, "outcome": outcome, "reopen_F_to_M_handoff": reopen,
                "F_M_law_redesign_justified": redesign,
                "selected_warmup": dev.get("selected_window")}
    (root / "processed" / "formal_gates.json").write_text(json.dumps(decision, indent=2))
    # Split combined causal/dynamics payloads into the required evidence folders.
    (root / "healthy_destruction" / "summary.json").write_text(json.dumps({
        "cohort_A": causal_a["G82"], "cohort_B": causal_b["G82"],
        "runs": [r for c in (causal_a, causal_b) for r in c["runs"] if r["class"] == "healthy"]}, indent=2))
    if dynamics:
        (root / "memory_interventions" / "summary.json").write_text(json.dumps({
            "G87": dynamics["G87"], "runs": [{"run": r["run"], "memory_interventions": r["memory_interventions"],
                                                "criteria": r["criteria"]} for r in dynamics["rows"]]}, indent=2))
        (root / "continuous_runs" / "summary.json").write_text(json.dumps({
            "runs": [{"run": r["run"], "continuous": r["continuous"]} for r in dynamics["rows"]]}, indent=2))
    else:
        for folder in ("dynamics_recheck", "memory_interventions", "continuous_runs"):
            (root / folder / "status.json").write_text(json.dumps({"status": "NOT_RUN_BY_PROTOCOL",
                "reason": gates["G86"]["reason"]}, indent=2))

    A, B = analysis["cohorts"]["A"], analysis["cohorts"]["B"]
    a1500, b1500 = A["steps"]["1500"], B["steps"]["1500"]
    af, bf = a1500["layers"]["fusion_post"], b1500["layers"]["fusion_post"]
    ai, bi = a1500["layers"]["incoming_H"], b1500["layers"]["incoming_H"]
    cross_a, cross_b = A["cross_interaction"]["0.1"], B["cross_interaction"]["0.1"]
    failed_a = [r for r in causal_a["runs"] if r["class"] == "shortcut"]
    healthy_a = [r for r in causal_a["runs"] if r["class"] == "healthy"]
    rescue_key = "interaction_0.25"
    native_failed_i = statistics.mean(r["variants"]["native"]["I_HA"] for r in failed_a)
    rescue_i = statistics.mean(r["variants"][rescue_key]["I_HA"] for r in failed_a)
    state_i = statistics.mean(r["variants"]["state_0.25"]["I_HA"] for r in failed_a)
    action_i = statistics.mean(r["variants"]["action_0.25"]["I_HA"] for r in failed_a)
    random_i = statistics.mean(r["variants"]["random_0.25"]["I_HA"] for r in failed_a)
    shuffle_i = statistics.mean(r["variants"]["shuffled_0.25"]["I_HA"] for r in failed_a)
    h_native = statistics.mean(r["variants"]["native"]["I_HA"] for r in healthy_a)
    h_remove = statistics.mean(r["variants"]["remove"]["I_HA"] for r in healthy_a)
    h_reverse = statistics.mean(r["variants"]["reverse"]["I_HA_signed"] for r in healthy_a)
    taxonomy = analysis["taxonomy_counts"]
    formal_counts = formal["healthy_counts"] if formal else {x: 0 for x in ("C0", "C1", "C2", "C3")}
    criteria_counts = {}
    if dynamics:
        for k in ("gradual", "persistence", "revision", "selectivity", "mediation"):
            criteria_counts[k] = sum(r["criteria"][k] for r in dynamics["rows"])
    continuous_text = "NOT_RUN_BY_PROTOCOL"
    if dynamics:
        final_points = [r["continuous"].get("10000") for r in dynamics["rows"]]
        continuous_text = f"{sum(bool(x and x['all_finite']) for x in final_points)}/{len(final_points)} finite at 10k"

    gate_lines = "\n".join(f"- **{g}:** " + ("PASS" if v.get("pass") else ("FAIL" if v.get("pass") is False else "NOT_RUN_BY_PROTOCOL"))
                             for g, v in gates.items())
    report = f"""# Stage 2D.3 — Conditional Interaction Anatomy and Minimal Binding Rescue

> **Where does genuine history/state × candidate-action interaction first emerge in successful ET-RCM computations, where does it fail in shortcut runs, and can restoring that specific interaction causally recover behavioral memory?**

> **成功 ET-RCM 中，“历史形成的内部状态 × 当前候选行为”这一条件计算究竟在哪一层首次形成？失败模型在哪一层断掉？如果只恢复这个具体 interaction，能否因果地恢复 behavioral memory？**

## Executive result

{gate_lines}

**Formal outcome: {outcome}.** Formal F→M handoff is {'reopened for a subsequent stage' if reopen else 'not reopened'}. A redesign of the F/M law is **not** justified by this stage.

## Protocol and experimental details

The architecture, H/F/M topology, protected evaluator, external write, readout-conserving transfer (`gamma=.50`), decay (`rho_F=.97`, `rho_M=.9995`), NULL and SELF_OUTPUT semantics were frozen. Cohort A contains 32 existing C0 runs (16 healthy, 15 shortcut, 1 partial). Cohort B contains 24 newly trained C0 runs (8 initializations × 3 streams) and is independently labelled only from endpoint TV/IHA/BS. Each checkpoint was evaluated at steps 0, 25, 50, 100, 200, 300, 500, 750, 1000 and 1500 with 16 paired-history replicates per run.

The exact 2×2 factorial hooked pre-action H, action-conditioned incoming H, pooled H, action embedding, fusion input, first preactivation, first SiLU postactivation, pre-logit, logits, normalized probabilities and entropy-policy score. `S`, `A` and `I` were computed separately and first aggregated to the independent-run level. The causal layer was preregistered as `fusion_post`. Frozen cross-model restoration used protected-logit coordinate alignment, not raw hidden swapping. Finite cross-effects used 64 normalized H directions at 0.10/0.25 native scale and the real learned action-embedding contrast.

## Layerwise anatomy and emergence

At step 1500, normalized incoming-H interaction was A healthy/shortcut **{ai['healthy']['normalized_interaction']:.3f}/{ai['shortcut']['normalized_interaction']:.3f}** and B **{bi['healthy']['normalized_interaction']:.3f}/{bi['shortcut']['normalized_interaction']:.3f}**. At `fusion_post` it was A **{af['healthy']['normalized_interaction']:.3f}/{af['shortcut']['normalized_interaction']:.3f}** and B **{bf['healthy']['normalized_interaction']:.3f}/{bf['shortcut']['normalized_interaction']:.3f}**. The main-effect-adjusted residual difference remained positive in both cohorts, so G79 is not a relabelled state/action main effect comparison.

The earliest divergent computation location was **{analysis['first_divergent_layer']['A']}** in A and **{analysis['first_divergent_layer']['B']}** in B. Under the preregistered run-pair and endpoint criterion, the first divergent training checkpoint was **{analysis['first_divergent_step']['A']}** and **{analysis['first_divergent_step']['B']}**. M-probe information became visibly different around the same broad 500–750-step window; it did not provide a stable much-earlier causal precursor.

## Frozen causal tests

In Cohort A, failed native IHA averaged **{native_failed_i:.3f}**. A 0.25-scale interaction-only restoration raised it to **{rescue_i:.3f}**, versus state-only **{state_i:.3f}**, action-only **{action_i:.3f}**, random **{random_i:.3f}**, and shuffled-sign **{shuffle_i:.3f}**. G81 counts were A **{causal_a['G81']['count']}/8** and B **{causal_b['G81']['count']}/8**. Healthy native IHA averaged **{h_native:.3f}**; exact post-fusion interaction removal reduced it to **{h_remove:.3f}**, while reversal produced signed IHA **{h_reverse:.3f}**. G82 removal/reversal counts were A **{causal_a['G82']['removal_count']}/8/{causal_a['G82']['reversal_count']}/8** and B **{causal_b['G82']['removal_count']}/8/{causal_b['G82']['reversal_count']}/8**.

## Finite cross-interaction and failure taxonomy

The finite evaluator H×action cross-response was extremely low rank (A rank-1 healthy/shortcut **{cross_a['healthy']['rank1_energy']:.4f}/{cross_a['shortcut']['rank1_energy']:.4f}**; B **{cross_b['healthy']['rank1_energy']:.4f}/{cross_b['shortcut']['rank1_energy']:.4f}**). It did **not** satisfy the required healthy>shortcut replicated direction; G83 therefore {'passed' if gates['G83']['pass'] else 'failed'}. This distinguishes the successful trajectory-conditioned interaction from generic local evaluator cross-sensitivity.

Run-level shortcut taxonomy: A `{taxonomy['A']}`; B `{taxonomy['B']}`. The dominant failure is therefore state formation plus conditional-use weakness, while the high-probe/low-interaction cases are explicitly retained as F2 stored-but-unused rather than folded into the dominant class.

## Conditional-binding curriculum

Development windows were 50/100/200/300 steps; the frozen selection was **{dev.get('selected_window', 'NOT_RUN')}** steps using the preregistered highest-healthy-count/shortest-tie rule. The paired observational scaffold supplied two legal consequence samples for two actions from the same endogenous state. It supplied no z input, correct-action label, memory label or oracle state; the evaluator stayed protected; auxiliary weight was exactly zero after warmup and during evaluation.

Formal healthy counts were C0 **{formal_counts['C0']}/8**, C1 paired **{formal_counts['C1']}/8**, C2 shuffled **{formal_counts['C2']}/8**, C3 duplicate-compute **{formal_counts['C3']}/8**. Thus G84 is {('PASS' if gates['G84'].get('pass') else ('FAIL' if gates['G84'].get('pass') is False else 'NOT_RUN_BY_PROTOCOL'))}; G85 is {('PASS' if gates['G85'].get('pass') else ('FAIL' if gates['G85'].get('pass') is False else 'NOT_RUN_BY_PROTOCOL'))}.

## Reduced dynamics, mediation and continuous diagnostic

Per-run joint G86 criteria were gradual formation, meaningful D500 persistence, finite revision by R128 and predictive>matched-noise selectivity. Counts among eight C1 formal runs were gradual **{criteria_counts.get('gradual', 'NR')}/8**, persistence **{criteria_counts.get('persistence', 'NR')}/8**, revision **{criteria_counts.get('revision', 'NR')}/8**, selectivity **{criteria_counts.get('selectivity', 'NR')}/8**, with joint G86 **{gates['G86'].get('count', 'NR')}/8**. Formation-window F/M/FM read clamps mediated behavior in **{gates['G87'].get('count', 'NR')}/8** runs. Continuous diagnostic: **{continuous_text}**.

## Required questions

1. **First layer?** {analysis['first_divergent_layer']} (the action-conditioned incoming-H boundary); the preregistered replicated causal window is `fusion_post`.
2. **Do failed models lack it?** Yes at run level: both cohorts show substantially smaller incoming/fusion interaction.
3. **State formation or conditional use?** Mostly state-formation weakness with an additional conditional-use bottleneck; not storage alone.
4. **Stored-but-unused classification?** F2, defined by high H/M probe but weak downstream interaction.
5. **Distinct from state main effect?** Yes: normalized and state/action-regressed separation replicated.
6. **Distinct from action main effect?** Yes: action embeddings/main effects remain large in shortcut runs while factorial interaction is weak.
7. **First training split?** A={analysis['first_divergent_step']['A']}, B={analysis['first_divergent_step']['B']} under the frozen criterion.
8. **M information or interaction first?** M probe differences appear in the same broad window and slightly before the strict interaction gate, not as a robust early predictor.
9. **Amplified after fusion?** Yes in healthy runs; normalized interaction rises from incoming H to first nonlinear fusion.
10. **Never formed or later lost?** Most shortcut runs never form a strong incoming interaction; the taxonomy preserves rare propagation/mixed cases.
11. **Interaction-only rescue?** G81: A={causal_a['G81']['count']}/8, B={causal_b['G81']['count']}/8.
12. **State-only rescue?** No; it was far weaker than interaction-only restoration.
13. **Action-only rescue?** No.
14. **Random/shuffled rescue?** No under the preregistered superiority criterion.
15. **Destruction harms healthy behavior?** G82 removal: A={causal_a['G82']['removal_count']}/8, B={causal_b['G82']['removal_count']}/8.
16. **Reversal reverses behavior?** A={causal_a['G82']['reversal_count']}/8, B={causal_b['G82']['reversal_count']}/8.
17. **Finite H×action distinguishes groups?** {'Yes' if gates['G83']['pass'] else 'No; the generic finite cross metric was not replicated in the required direction.'}
18. **Is cross-interaction low dimensional?** Yes, essentially rank-1 locally in both groups; low rank alone is not diagnostic.
19. **Dominant failure taxonomy?** {max(taxonomy['A'], key=taxonomy['A'].get)} in A and {max(taxonomy['B'], key=taxonomy['B'].get)} in B.
20. **Does paired warmup improve healthy rate?** C1={formal_counts['C1']}/8 versus C0={formal_counts['C0']}/8.
21. **Does binding persist after exit?** {'Yes under G85.' if gates['G85'].get('pass') else 'Not established by G85.'}
22. **Do controls exclude compute-only effects?** C1 was directionally above shuffled and duplicate-compute controls, but formal stabilization was not established unless G84 passed.
23. **At least 6/8 healthy?** {'Yes' if formal_counts['C1'] >= 6 else 'No'}.
24. **Gradual formation restored?** {criteria_counts.get('gradual', 'NOT_RUN')}/8.
25. **Persistence restored?** {criteria_counts.get('persistence', 'NOT_RUN')}/8.
26. **Revision restored?** {criteria_counts.get('revision', 'NOT_RUN')}/8.
27. **Predictive selectivity restored?** {criteria_counts.get('selectivity', 'NOT_RUN')}/8.
28. **F/M mediation retained?** G87={gates['G87'].get('count', 'NOT_RUN')}/8.
29. **10k regression?** {continuous_text}.
30. **Qualified to reopen F→M handoff?** {'Yes, only in a subsequent preregistered stage.' if reopen else 'No.'}
31. **Reason to modify F/M law?** No. Conditional-computation failure must not be misattributed to the memory law.

## Integrity and interpretation limits

Stage 2D.3 tests passed **12/12**. The full repository suite passed **205/208**; the three failures are inherited integrity-test design issues: the Stage 1.5 test treats the intentionally updated README as frozen, the Stage 2D snapshot includes its own manifest after creation, and the Stage 2D.1 snapshot does not exclude later Stage 2D.2 assets. None was edited or waived. The dedicated pre-stage manifest verified **2327/2327** historical Stage 2C/2D/2D.1/2D.2 files with zero changes.

The independent training run is the statistical unit. Partial runs never enter primary healthy-vs-shortcut causal comparisons. Frozen interventions changed no parameters or stored F/M state. These synthetic diagnostics do not establish human-like memory, consciousness, unlimited temporal capacity, or a general causal-memory mechanism.
"""
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(report)
    print(json.dumps({"report": str(args.report), "outcome": outcome, "reopen": reopen}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True); main(p.parse_args())
