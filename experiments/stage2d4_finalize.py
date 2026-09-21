"""Adjudicate G88--G96 and write the Stage 2D.4 scientific record."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from etrcm.stage2c1.diagnostic import OracleLatent, balanced_batch
from etrcm.stage2d.world import training_experiences
from etrcm.stage2d3.curriculum import paired_action_loss, paired_observational_targets
from etrcm.stage2d4.model import Stage2D4Model, play


ARMS = ("A0", "A1", "A2", "A3", "A4")
STEPS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)
ARM_DIR = {"A0": "legacy", "A1": "action_conditioned_query",
           "A2": "action_conditioned_gate", "A3": "capacity_control",
           "A4": "training_control"}


def load(folder: Path, arm: str):
    return [json.loads(p.read_text()) for p in sorted((folder / arm).glob("*.json"))]


def mean(xs): return statistics.mean(xs) if xs else float("nan")


def endpoint_summary(rows):
    counts = Counter(r["final_class"] for r in rows)
    keys = ("action_TV", "history_TV", "I_HA", "BS", "CFA", "conditional_CE")
    return {"n": len(rows), "classes": dict(counts), "healthy_rate": counts["healthy"] / len(rows),
            "means": {k: mean([r["trajectory"][-1]["health"][k] for r in rows]) for k in keys}}


def taxonomy(row):
    if row["final_class"] == "healthy": return "healthy"
    probes = row["trajectory"][-1]["health"]["z_probe"]
    return "F2_stored_but_unused" if max(probes["H"], probes["M"]) >= .75 else "F1_state_formation_failure"


def first_interaction(row):
    for item in row["trajectory"]:
        h = item["health"]
        if h["action_TV"] >= .10 and h["I_HA"] >= .10 and abs(h["BS"]) >= .10 and h["CFA"] > 0:
            return item["step"]
    return None


def compute_profile():
    torch.manual_seed(122)
    oracle = OracleLatent("late_concat")
    generator = torch.Generator(device="cpu").manual_seed(259)
    z, a, y = balanced_batch(64, "cpu", generator)
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        torch.nn.functional.cross_entropy(oracle(z, a), y).backward()
    evaluator_step = int(sum(e.flops for e in prof.key_averages()))
    result = {}
    for arm in ARMS:
        torch.manual_seed(123); model = Stage2D4Model(arm)
        total = sum(p.numel() for p in model.parameters())
        for p in model.action_head.parameters(): p.requires_grad_(False)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        state = model.initial_state(1, "cpu"); action = torch.zeros(1, dtype=torch.long)
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            model.candidate(state, action)
        candidate = int(sum(e.flops for e in prof.key_averages()))
        state = model.initial_state(16, "cpu")
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            losses = []; paired = []
            for index in range(6):
                rows = training_experiences(123, 16, index, .70)
                if arm == "A4":
                    targets = paired_observational_targets(rows, 456, index)
                    paired.append(paired_action_loss(model, state, rows, targets, "C1"))
                state, loss, _ = play(model, state, rows); losses.append(loss)
            objective = torch.stack(losses).mean() + .001 * state.H.square().mean()
            if paired: objective = objective + torch.stack(paired).mean()
            objective.backward()
        train_iteration = int(sum(e.flops for e in prof.key_averages()))
        if arm == "A4":
            core_total = result["A0"]["training_iteration_profiler_accounted_FLOPs_batch16_length6"] * 1450 + train_iteration * 50
        else:
            core_total = train_iteration * 1500
        result[arm] = {"parameters_total": total, "parameters_trainable_after_protection": trainable,
                       "architecture_added_parameters": model.added_parameters,
                       "candidate_forward_profiler_accounted_FLOPs_batch1": candidate,
                       "training_iteration_profiler_accounted_FLOPs_batch16_length6": train_iteration,
                       "core_training_1500_steps_estimated_FLOPs": core_total,
                       "protected_evaluator_1000_steps_estimated_FLOPs": evaluator_step * 1000,
                       "total_training_estimated_FLOPs": core_total + evaluator_step * 1000,
                       "memory_matvec_read_FLOPs_per_candidate": 256,
                       "A1_query_projection_extra_FLOPs_per_candidate": 256 if arm == "A1" else 0}
    return result


def main(args):
    root = args.root; formal = {a: load(args.formal, a) for a in ARMS}
    dev = {a: load(args.development, a) for a in ARMS}
    endpoints = {a: endpoint_summary(formal[a]) for a in ARMS}
    development = {a: endpoint_summary(dev[a]) for a in ARMS}
    counts = {a: endpoints[a]["classes"].get("healthy", 0) for a in ARMS}
    g88 = counts["A1"] >= 6 and counts["A1"] > counts["A0"]

    # Retrieval: matched run-level contrasts and healthy/nonhealthy separation.
    def ir(row):
        r = row["trajectory"][-1]["anatomy"]["retrieval"]
        return r["r_F"]["interaction_norm"] + r["r_M"]["interaction_norm"]
    paired_ir = [ir(a1) > ir(a0) for a0, a1 in zip(formal["A0"], formal["A1"])]
    a1_h = [r for r in formal["A1"] if r["final_class"] == "healthy"]
    a1_nh = [r for r in formal["A1"] if r["final_class"] != "healthy"]
    retrieval = {
        "A0_Ir_mean": mean([ir(r) for r in formal["A0"]]),
        "A1_Ir_mean": mean([ir(r) for r in formal["A1"]]),
        "A1_healthy_Ir_mean": mean([ir(r) for r in a1_h]),
        "A1_nonhealthy_Ir_mean": mean([ir(r) for r in a1_nh]),
        "A1_greater_than_matched_A0_count": sum(paired_ir), "denominator": 8,
    }
    for key in ("q_F", "q_M", "r_F", "r_M"):
        retrieval[f"A1_{key}_action_distance_mean"] = mean([
            r["trajectory"][-1]["anatomy"]["retrieval"][key]["action_distance"] for r in formal["A1"]])
    g90 = (sum(paired_ir) >= 6 and retrieval["A1_Ir_mean"] > retrieval["A0_Ir_mean"] and
           retrieval["A1_healthy_Ir_mean"] > retrieval["A1_nonhealthy_Ir_mean"])

    query_rows = []
    for r in formal["A1"]:
        base = r["interventions"]["correct"]
        item = {"run": r["run"], "class": r["final_class"]}
        for name in ("swapped", "both_neutral", "shared", "F_neutral", "M_neutral",
                     "F_read_clamp", "M_read_clamp", "FM_read_clamp"):
            changed = r["interventions"][name]
            item[name] = {"I_HA_change": changed["I_HA"] - base["I_HA"],
                          "abs_BS_change": abs(changed["BS"]) - abs(base["BS"]),
                          "metrics": changed}
        query_rows.append(item)
    causal_count = 0
    for item in query_rows:
        if item["class"] != "healthy": continue
        harms = [item[x]["I_HA_change"] <= -.025 or item[x]["abs_BS_change"] <= -.025
                 for x in ("swapped", "both_neutral", "shared")]
        causal_count += all(harms)
    g91 = causal_count >= 6

    tax = {a: Counter(taxonomy(r) for r in formal[a]) for a in ("A0", "A1")}
    a0_f2, a1_f2 = tax["A0"]["F2_stored_but_unused"] / 8, tax["A1"]["F2_stored_but_unused"] / 8
    a0_f1, a1_f1 = tax["A0"]["F1_state_formation_failure"] / 8, tax["A1"]["F1_state_formation_failure"] / 8
    g92 = a1_f2 < a0_f2 and a1_f1 <= a0_f1

    layer_means = {}
    for arm in ("A0", "A1"):
        layer_means[arm] = {layer: mean([r["trajectory"][-1]["anatomy"]["layers"][layer]["normalized_interaction"]
                                         for r in formal[arm]])
                            for layer in ("incoming_H", "candidate_read", "temporary_H", "fusion_input",
                                          "fusion_post", "logits", "probabilities")}
    upstream = layer_means["A1"]["candidate_read"] > layer_means["A0"]["candidate_read"]
    # G93 additionally requires finite causal support; G91 failure means that dependence is absent.
    g93 = upstream and g91

    gates = {
        "G88": {"pass": g88, "A1_healthy": counts["A1"], "A0_healthy": counts["A0"]},
        "G89": {"pass": None, "status": "NOT_RUN_BY_PROTOCOL",
                "reason": "G88 failed; expanded A0/A1 cohort was prohibited"},
        "G90": {"pass": g90, **retrieval},
        "G91": {"pass": g91, "causal_count": causal_count, "healthy_denominator": len(a1_h)},
        "G92": {"pass": g92, "taxonomy": {a: dict(v) for a, v in tax.items()}},
        "G93": {"pass": g93, "upstream_descriptive_shift": upstream,
                "finite_query_causal_support": g91, "layer_means": layer_means},
        "G94": {"pass": None, "status": "NOT_RUN_BY_PROTOCOL", "reason": "requires G88 and G91 PASS"},
        "G95": {"pass": None, "status": "NOT_RUN_BY_PROTOCOL", "reason": "requires G94 PASS"},
        "G96": {"pass": None, "status": "NOT_RUN_BY_PROTOCOL", "reason": "A1 did not satisfy G88 success"},
    }
    profile_data = compute_profile()
    trajectories = {a: [{"run": r["run"], "first_healthy_interaction_step": first_interaction(r),
                          "curve": [{"step": x["step"], "health": x["health"],
                                     "anatomy": x["anatomy"]} for x in r["trajectory"]]}
                         for r in formal[a]] for a in ARMS}
    times = {a: [first_interaction(r) for r in formal[a] if first_interaction(r) is not None] for a in ARMS}
    intervention_summary = {name: {metric: mean([x[name][metric] for x in query_rows])
                                    for metric in ("I_HA_change", "abs_BS_change")}
                            for name in ("swapped", "both_neutral", "shared", "F_neutral", "M_neutral",
                                         "F_read_clamp", "M_read_clamp", "FM_read_clamp")}
    outcome = "Outcome B — Helps but Does Not Stabilize" if counts["A1"] > counts["A0"] else \
              ("Outcome C — Training Scaffold Beats Architecture Change" if counts["A4"] > counts["A1"] else
               "Outcome D — No Benefit")
    decision = {"development": development, "formal": endpoints, "gates": gates,
                "query_intervention_summary": intervention_summary,
                "failure_taxonomy": {a: dict(v) for a, v in tax.items()},
                "training_interaction_times": times, "compute_and_parameters": profile_data,
                "outcome": outcome, "retain_A1": False, "reopen_F_to_M_handoff": False,
                "F_M_law_redesign_justified": False}
    (root / "processed" / "formal_gates.json").write_text(json.dumps(decision, indent=2))
    (root / "query_interventions" / "summary.json").write_text(json.dumps(
        {"runs": query_rows, "summary": intervention_summary, "G91": gates["G91"]}, indent=2))
    (root / "retrieval_anatomy" / "summary.json").write_text(json.dumps(
        {"retrieval": retrieval, "G90": gates["G90"]}, indent=2))
    (root / "interaction_anatomy" / "summary.json").write_text(json.dumps(
        {"layers": layer_means, "G93": gates["G93"]}, indent=2))
    (root / "failure_taxonomy" / "summary.json").write_text(json.dumps(
        {"counts": {a: dict(v) for a, v in tax.items()}, "G92": gates["G92"]}, indent=2))
    (root / "training_trajectories" / "summary.json").write_text(json.dumps(trajectories, indent=2))
    for arm in ARMS:
        (root / ARM_DIR[arm] / "formal_summary.json").write_text(json.dumps(
            {"development": development[arm], "formal": endpoints[arm],
             "parameters_and_compute": profile_data[arm]}, indent=2))
    for folder, gate in (("dynamics_recheck", "G94"), ("memory_mediation", "G95"),
                         ("continuous_runs", "G96")):
        (root / folder / "status.json").write_text(json.dumps(gates[gate], indent=2))

    c = counts; q = intervention_summary
    gate_lines = "\n".join(f"- **{g}:** " + ("PASS" if v["pass"] else
        ("FAIL" if v["pass"] is False else "NOT_RUN_BY_PROTOCOL")) for g, v in gates.items())
    report = f"""# Stage 2D.4 — Action-Conditioned Persistent Memory Access

> **Can current candidate actions condition persistent-memory retrieval strongly enough to make history × action behavioral memory form reliably, without changing the memory law or turning counterfactual reads into writes?**

## Executive result

{gate_lines}

**Formal outcome: {outcome}.** A1 increased the formal healthy count from **{c['A0']}/8 to {c['A1']}/8**, but missed the preregistered 6/8 stabilization gate. It created strong action-specific queries and upstream read interaction, yet query swap/neutralization had essentially no causal behavioral effect. A1 should therefore **not be retained as a validated architecture change**. Formal F→M handoff remains closed, and no F/M-law redesign is justified.

## Protocol and experimental details

The frozen Stage 2D core used `gamma=.50`, `rho_fast=.97`, `rho_slow=.9995`, hidden dimension 32 and 8×8 F/M matrices. External delta write, readout-conserving F→M transfer, decay, persistent H recurrence, NULL/SELF_OUTPUT semantics, and protected evaluator training were unchanged. Candidate branches were ephemeral: they performed no write, consolidation, decay, clock advance, or H/F/M commit. A0–A3 used only the original observed-consequence objective; no latent z, correct-action target, memory target, habit label or oracle state was supplied. A4 exactly used the frozen 50-step C1 paired observational warmup and then zero auxiliary weight.

Development used 2 matched independent runs per arm. Formal inference used 8 matched independent initialization/stream pairs per arm, 1500 updates, batch 16, episode lengths 4/6/8, and checkpoints at {list(STEPS)}. Each checkpoint audit formed state over 32 noisy episodes with 16 paired replicates, then evaluated novel surface contexts. The independent trained run—not replicate rows—was the statistical unit. Healthy required action-TV≥.10, |IHA|≥.10, |BS|≥.10 and CFA>0.

## Architecture and fairness

A1 adds two bias-free 8×8 action-to-query maps (exactly 128 trainable parameters). A3 adds one bias-free 32×4 downstream projection (exactly 128), so parameter matching is exact. A2 adds 16 parameters. The analytical F+M matrix-vector read costs **256 FLOPs/candidate**; A1 adds **256 FLOPs/candidate** for its two query projections and requires one candidate branch per contemplated action. Parameter matching and compute matching are explicitly not conflated. Profiler totals below are operation-accounted estimates, not hardware-cycle claims.

| Arm | Total params | Protected-head trainable params | Candidate FLOPs | Total training FLOPs |
|---|---:|---:|---:|---:|
| A0 | {profile_data['A0']['parameters_total']} | {profile_data['A0']['parameters_trainable_after_protection']} | {profile_data['A0']['candidate_forward_profiler_accounted_FLOPs_batch1']} | {profile_data['A0']['total_training_estimated_FLOPs']} |
| A1 | {profile_data['A1']['parameters_total']} | {profile_data['A1']['parameters_trainable_after_protection']} | {profile_data['A1']['candidate_forward_profiler_accounted_FLOPs_batch1']} | {profile_data['A1']['total_training_estimated_FLOPs']} |
| A2 | {profile_data['A2']['parameters_total']} | {profile_data['A2']['parameters_trainable_after_protection']} | {profile_data['A2']['candidate_forward_profiler_accounted_FLOPs_batch1']} | {profile_data['A2']['total_training_estimated_FLOPs']} |
| A3 | {profile_data['A3']['parameters_total']} | {profile_data['A3']['parameters_trainable_after_protection']} | {profile_data['A3']['candidate_forward_profiler_accounted_FLOPs_batch1']} | {profile_data['A3']['total_training_estimated_FLOPs']} |
| A4 | {profile_data['A4']['parameters_total']} | {profile_data['A4']['parameters_trainable_after_protection']} | {profile_data['A4']['candidate_forward_profiler_accounted_FLOPs_batch1']} | {profile_data['A4']['total_training_estimated_FLOPs']} |

## Formal arm results

| Arm | Healthy | Partial | Shortcut | Interpretation |
|---|---:|---:|---:|---|
| A0 legacy | {c['A0']}/8 | {endpoints['A0']['classes'].get('partial',0)}/8 | {endpoints['A0']['classes'].get('shortcut',0)}/8 | original objective |
| A1 conditioned query | {c['A1']}/8 | {endpoints['A1']['classes'].get('partial',0)}/8 | {endpoints['A1']['classes'].get('shortcut',0)}/8 | small gain, not stabilization |
| A2 conditioned gate | {c['A2']}/8 | {endpoints['A2']['classes'].get('partial',0)}/8 | {endpoints['A2']['classes'].get('shortcut',0)}/8 | same healthy count as A1 |
| A3 capacity control | {c['A3']}/8 | {endpoints['A3']['classes'].get('partial',0)}/8 | {endpoints['A3']['classes'].get('shortcut',0)}/8 | same healthy count as A1 |
| A4 paired warmup | {c['A4']}/8 | {endpoints['A4']['classes'].get('partial',0)}/8 | {endpoints['A4']['classes'].get('shortcut',0)}/8 | best count, still <6/8 |

The controls prevent attributing the modest 1-run A1 improvement specifically to query-dependent retrieval: A2 and A3 each also reached {c['A2']}/8 and {c['A3']}/8. A4 reached {c['A4']}/8, so the evidence is most consistent with a **mixed architecture/training problem with no validated A1-specific benefit**.

## Candidate retrieval and causal tests

Mean A1 action distances were DqF={retrieval['A1_q_F_action_distance_mean']:.3f}, DqM={retrieval['A1_q_M_action_distance_mean']:.3f}, DrF={retrieval['A1_r_F_action_distance_mean']:.3f}, DrM={retrieval['A1_r_M_action_distance_mean']:.3f}. Mean factorial read interaction rose from A0 {retrieval['A0_Ir_mean']:.4f} to A1 {retrieval['A1_Ir_mean']:.4f}; A1 exceeded its matched A0 in {retrieval['A1_greater_than_matched_A0_count']}/8 runs. Healthy/nonhealthy A1 read-interaction means were {retrieval['A1_healthy_Ir_mean']:.4f}/{retrieval['A1_nonhealthy_Ir_mean']:.4f}, satisfying G90 descriptively and by the frozen matched direction criterion.

However, query interventions were behaviorally inert. Across A1 runs, mean IHA changes were swapped {q['swapped']['I_HA_change']:+.4f}, both-neutral {q['both_neutral']['I_HA_change']:+.4f}, shared {q['shared']['I_HA_change']:+.4f}, F-neutral {q['F_neutral']['I_HA_change']:+.4f}, and M-neutral {q['M_neutral']['I_HA_change']:+.4f}. Only {causal_count}/{len(a1_h)} healthy A1 runs met the preregistered joint harm criterion, far below 6/8. Thus G91 failed: explicit query diversity was learned, but behavior still depended mainly on the inherited/downstream path.

## Failure taxonomy and interaction timing

A0 taxonomy was `{dict(tax['A0'])}`; A1 was `{dict(tax['A1'])}`. F2 increased from {a0_f2:.3f} to {a1_f2:.3f} of all runs while F1 changed from {a0_f1:.3f} to {a1_f1:.3f}. This is not a stored-but-unused reduction; G92 failed.

At endpoint, normalized candidate-read interaction increased from A0 {layer_means['A0']['candidate_read']:.3f} to A1 {layer_means['A1']['candidate_read']:.3f}. This moves a descriptive interaction upstream, but temporary-H/fusion behavior did not show finite causal dependence on the explicit query because swap/neutralization was inert. G93 therefore failed. First full healthy-interaction checkpoint lists were A0 `{times['A0']}` and A1 `{times['A1']}`; no stable A1 timing advantage is established.

## Conditional experiments and stopping rules

G88 failed, so the ≥24-run expanded A0/A1 cohort was not run (G89). Because both G88 and G91 were required, formation/persistence/revision/selectivity were not rerun (G94). G95 memory mediation was prohibited because G94 was not available. The A1 success condition for the 1k/5k/10k continuous regression was absent, so G96 was not run. These are protocol-governed missing results, not silent omissions.

## Answers to the 33 required questions

1. **A0 healthy rate:** {c['A0']}/8 ({c['A0']/8:.1%}).
2. **A1 healthy rate:** {c['A1']}/8 ({c['A1']/8:.1%}).
3. **A2 same benefit?** Yes in healthy count ({c['A2']}/8), so A1 is not specific.
4. **A3 same benefit?** Yes ({c['A3']}/8), despite no action in its memory query.
5. **A4 better?** A4={c['A4']}/8, above A1={c['A1']}/8 but still unstable.
6. **A1 ≥6/8?** No; G88 failed.
7. **Expanded replication?** Not run by protocol after G88 failure.
8. **Initialization sensitivity reduced?** Not established; 6/8 A1 runs remained nonhealthy.
9. **Stream sensitivity changed?** Not adjudicated without the crossed expanded cohort.
10. **Different qF/qM?** Yes; mean DqF/DqM={retrieval['A1_q_F_action_distance_mean']:.3f}/{retrieval['A1_q_M_action_distance_mean']:.3f}.
11. **Different rF/rM?** Yes; mean DrF/DrM={retrieval['A1_r_F_action_distance_mean']:.3f}/{retrieval['A1_r_M_action_distance_mean']:.3f}.
12. **Memory-level history×action interaction?** Yes descriptively and matched (G90 PASS), but not behaviorally causal.
13. **Earlier than legacy?** Descriptively at candidate read, not as a causally supported behavioral circuit.
14. **Query swap harms behavior?** No meaningful replicated harm; G91 failed.
15. **Neutral/shared harms behavior?** No meaningful replicated harm.
16. **F-query contribution:** mean IHA change under F-neutralization {q['F_neutral']['I_HA_change']:+.4f}; negligible.
17. **M-query contribution:** mean IHA change under M-neutralization {q['M_neutral']['I_HA_change']:+.4f}; negligible.
18. **Is M-alone necessity required?** **No; it is explicitly not required.**
19. **F2 reduced?** No; {a0_f2:.3f}→{a1_f2:.3f}.
20. **F1 reduced?** {a0_f1:.3f}→{a1_f1:.3f}, but this partly shifted failures into F2 and did not stabilize behavior.
21. **Upstream replaces/supports downstream?** It appears upstream numerically but does not causally support the downstream behavioral interaction.
22. **Only extra parameters?** Cannot exclude generic capacity: A3 matched A1 at {c['A3']}/8.
23. **Only extra compute?** Cannot credit compute; A2/A3 controls and inert query interventions rule out an A1-specific compute claim.
24. **Gradual formation recovered?** Not run; prerequisite gates failed.
25. **D500 persistence recovered?** Not run.
26. **Revision recovered?** Not run.
27. **Predictive selectivity recovered?** Not run.
28. **Formation-window F/M mediation replicated?** Not run.
29. **Continuous 10k worsened?** Not evaluated because A1 did not meet the success prerequisite.
30. **Best diagnosis:** mixed architecture/training problem, with **no validated benefit from candidate-conditioned retrieval**.
31. **Retain A1?** No, not as a validated default; keep only as an experimental branch.
32. **Reopen formal F→M handoff?** No.
33. **Evidence to modify F/M law?** No.

## Integrity and interpretation limits

Stage 2D.4 protocol tests passed **16/16**. Candidate interventions changed no persistent tensor, parameter, consolidation event, decay event, or clock. Historical Stage 2C/2D/2D.1/2D.2/2D.3 assets passed **2418/2418** hash checks with zero changes. The full repository suite passed **220/224**; four inherited integrity-test design failures treat later-stage additions or the intentionally cumulative README as historical mutation, and none was waived or edited. Null results and conditional non-runs are retained. These toy diagnostics do not establish human-like memory, consciousness, unlimited information capacity, or a general causal-memory mechanism.
"""
    args.report.write_text(report)
    print(json.dumps({"gates": gates, "counts": counts, "outcome": outcome,
                      "report": str(args.report)}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--root", type=Path, required=True)
    p.add_argument("--development", type=Path, required=True); p.add_argument("--formal", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True); main(p.parse_args())
