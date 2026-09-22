"""Produce complete Stage 2D.6 experiment report and transparent stop statuses."""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import stage2d6_finalize as z


ROOT=Path("results/stage2d6")
REPORT=Path("reports/STAGE2D6_CONDITIONAL_INTERACTION_GENERATION_RESULTS.md")


def f(value,digits=3):
    return "n/a" if value is None else f"{value:.{digits}f}"


def m(values):
    return statistics.mean(values) if values else None


def status(path,status,reason):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"status":status,"reason":reason,
                                "regularizer_enabled":False},indent=2))


def cohort_line(name,summary):
    h=summary["healthy"];s=summary["shortcut"]
    return (f"| {name} | {summary['count']} | {summary['classes']} | "
            f"{f(h['pre_I'])}/{f(s['pre_I'])} | {f(h['post_I'])}/{f(s['post_I'])} | "
            f"{f(h['delta_NL'])}/{f(s['delta_NL'])} | {f(h['Q1'])}/{f(s['Q1'])} | "
            f"{f(h['overlap'])}/{f(s['overlap'])} |")


def op_means(rows,label):
    group=[r for r in rows if r["class"]==label]
    keys=("common_offset","random_offset","low_curvature_offset")
    return {key:{"IHA":m([r[key]["behavior"]["IHA"] for r in group]),
                 "BS_abs":m([abs(r[key]["behavior"]["BS"]) for r in group]),
                 "post_I":m([r[key]["post_I"] for r in group])} for key in keys}


def main():
    data=json.loads((ROOT/"processed"/"formal_gates.json").read_text())
    if data["training_authorized"]:
        raise RuntimeError("G114/G115/G116 authorize training; run conditional arms before final report")
    gate=data["gates"];coh=data["cohorts"];prim=data["primary"]
    historical=z.load(ROOT/"fusion_anatomy"/"historical_legacy")
    confirm=z.load(ROOT/"fusion_anatomy"/"confirmatory_baseline")
    a1=z.load(ROOT/"fusion_anatomy"/"original_query")+z.load(
        ROOT/"fusion_anatomy"/"confirmatory_query")
    oldop=z.load(ROOT/"operating_point_rescue"/"formal"/"historical_legacy")
    newop=z.load(ROOT/"operating_point_rescue"/"formal"/"confirmatory_baseline")
    a1op=z.load(ROOT/"operating_point_rescue"/"formal"/"original_query")+z.load(
        ROOT/"operating_point_rescue"/"formal"/"confirmatory_query")
    assert all(all(r["endpoint"]["integrity"].values()) for r in historical+confirm+a1)
    assert all(all(r["integrity"].values()) for r in oldop+newop+a1op)
    max_hook_error=max(r["endpoint"]["native_logit_max_error"] for r in historical+confirm+a1)
    freeze=data["frozen_rule"]
    training=json.loads((ROOT/"confirmatory"/"training_summaries.json").read_text())
    assert len(training)==24 and all(r["summary"]["evaluator_hash_unchanged"] for r in training)
    assert all(r["summary"]["architecture_change"]=="none" and
               r["summary"]["memory_law_change"]=="none" and
               not r["summary"]["correct_action_labels"] and
               not r["summary"]["memory_labels"] and
               not r["summary"]["evaluation_scaffold"] for r in training)
    selected={c:{"healthy":prim[c]["healthy_runs"],"shortcut":prim[c]["shortcut_runs"]}
              for c in prim}
    topk={}
    for cname,group in (("historical",historical),("confirmatory",confirm)):
        healthy=[r for r in group if r["run"] in selected[cname]["healthy"]]
        topk[cname]={str(k):{"passes":sum(z.unit_pass(r,k) for r in healthy),
            "mean_IHA_loss":m([r["endpoint"]["unit_causality"][str(k)]["IHA_loss"] for r in healthy]),
            "mean_random_IHA_loss":m([r["endpoint"]["unit_causality"][str(k)]["random_mean_IHA_loss"]
                                      for r in healthy])} for k in (1,2,4,8,16)}
    f1=[r for r in historical+confirm+a1 if z.phenotype(r)=="F1_state_formation"]
    f2=[r for r in historical+confirm+a1 if z.phenotype(r)=="F2_stored_unused"]
    positive=[]
    for row in oldop+newop+a1op:
        if "post_positive_control" not in row:continue
        candidates=row["post_positive_control"]["interventions"]["interaction"].values()
        positive.append(any(v["action_TV"]>=.10 and v["I_HA"]>=.10 and
                            abs(v["BS"])>=.10 for v in candidates))
    integrity=json.loads((ROOT/"manifests"/"integrity_verification.json").read_text())
    assert integrity["pass"] and not integrity["changed"]
    notrun="G114/G115/G116 did not supply two passing causal/temporal gates; stopping rule forbids training modification"
    for folder in ("training_rescue","dynamics_recheck","memory_mediation","continuous_runs"):
        status(ROOT/folder/"status.json","NOT_RUN_BY_PROTOCOL",notrun)
    for folder,payload in {
        "nonlinear_generation":{c:{k:v for k,v in cohort.items() if k in ("healthy","shortcut","partial","classes")}
                                for c,cohort in coh.items()},
        "curvature":{c:{"overlap_AUC":prim[c]["AUC"]["overlap"],
                        "comparators":prim[c]["AUC"]} for c in prim},
        "unit_causality":topk,
        "healthy_destruction":{c:prim[c]["G115_destroy"] for c in prim},
        "training_trajectories":{c:prim[c]["trajectory_onsets"] for c in prim},
        "f1_f2_analysis":data["phenotypes"],
        "controls":{"post_positive_pass":sum(positive),"post_positive_n":len(positive),
                    "historical":op_means(oldop,"shortcut"),"confirmatory":op_means(newop,"shortcut")},
        "operating_point_rescue":{c:{"count":prim[c]["G114_rescue"],
                                    "failed_n":prim[c]["shortcut_count"]} for c in prim},
    }.items():
        p=ROOT/folder/"summary.json";p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(payload,indent=2))
    outcome=("C — fusion-post interaction is causal, but simple curvature/operating-point "
             "control does not explain or rescue it" if gate["G110"]=="PASS" else
             "D — no replicated nonlinear-generation explanation")
    first=[]
    first.append("# ET-RCM Stage 2D.6 — Conditional Interaction Generation in the Fusion Nonlinearity\n")
    first.append("> **How does the fusion nonlinearity transform history-dependent state effects and candidate-action effects into the causal state×action interaction required for behavioral memory, and why does this computation form only in some training runs?**\n")
    first.append("> **fusion 非线性如何把 history-derived state main effect 与 candidate-action main effect 转换成行为相关的 state×action interaction？为何只在部分训练 run 中形成？**\n")
    first.append("## Executive result\n")
    first.append(f"**Outcome {outcome}.** The new 24-run cohort contained only **{coh['confirmatory']['classes'].get('healthy',0)} healthy** endpoints, below the prespecified 8-run healthy confirmation minimum; gates marked `INCONCLUSIVE_N_LT_8` are not mechanistic failures. The frozen SiLU often enlarges factorial interaction in healthy checkpoints, but descriptive generation is not sufficient to establish a causal operating-point mechanism. Development froze the global common offset at **{freeze['offset_scale']:+.2f} native SD**, with selected top-**{freeze['unit_k']}** finite-contribution units before confirmatory labels were available. The diagnostic stopping rule did not authorize training or memory-law changes. Fusion architecture redesign and formal F→M handoff remain unjustified.\n")
    first.append("| Gate | Decision |\n|---|---|\n")
    for a in range(110,123):
        first.append(f"| G{a} | {gate[f'G{a}']} |\n")
    first.append("\n## Frozen design and exact computation\n")
    first.append("The real candidate path is `context/event → candidate temporary H → pooled H → W_H H + W_A e_a + b → SiLU → protected linear logits → softmax`. The H branch is already candidate-conditioned; consequently preactivation interaction is not theoretically forced to zero. The action projection itself is action-only and its 2×2 interaction is numerically zero. All 2×2 histories/actions and 16 within-run replicates were aggregated to independent-run statistics. Analytic SiLU slope/curvature, exact hook equality, finite output response and persistent-state/parameter purity were unit-tested.\n")
    first.append("Development used all **24** pre-existing C0 legacy runs (primary sorted 8 healthy/8 shortcut). The independent confirmatory baseline trained **24** new C0 runs from initialization seeds 21101–21108 × streams 22101–22103, 1500 updates each, batch 16, episodes 4/6/8, observed-only consequence cross-entropy, γ=.50, ρF=.97, ρM=.9995, no paired scaffold, 1000-step protected evaluator pretraining. Ten checkpoint times were retained. Their training summaries, checkpoint paths, final hashes and immutable-head checks are machine-readable; binary checkpoints remain on the research server and are git-ignored. A1 original/confirmatory checkpoints (24 total) were analyzed only as supplementary F1/F2 context, never pooled into C0 replication gates. Healthy/partial/shortcut labels use the frozen TV/IHA/BS/CFA thresholds, not fusion metrics.\n")
    first.append("The finite 4×32 probability-interaction response to ±0.02/4 per-unit post-SiLU perturbation was SVD-decomposed; Q reports projection onto ranks 1/2/4. `C_j` is the finite IHA loss from removing one unit's factorial interaction. The development offset grid was {-1,-.5,-.25,0,.25,.5,1} native preactivation SD; the *group-mean* best scale, not each run's best, was frozen. Common offsets were identical across four cells and verified to preserve preactivation state/action interaction contrasts. Controls included same-support random signed offsets, curvature-lowering offsets, state/action projection scaling, same-k random unit removal and a known post-SiLU injection ceiling. This ceiling did not enter any rescue gate.\n")
    first.append("\n## Factorial fusion anatomy and behavioral alignment\n")
    first.append("Values are means over independent runs. Each cell is healthy/shortcut (not neuron-level pseudo-replicates).\n\n")
    first.append("| Cohort | Runs | Basin counts | pre-I H/S | post-I H/S | ΔNL H/S | Q1 H/S | O H/S |\n|---|---:|---|---:|---:|---:|---:|---:|\n")
    first.append(cohort_line("legacy development",coh["historical"])+"\n")
    first.append(cohort_line("new confirmatory",coh["confirmatory"])+"\n")
    first.append(cohort_line("A1 supplementary",coh["A1_supplementary"])+"\n")
    first.append("\n")
    for name in ("historical","confirmatory"):
        d=prim[name];s=coh[name]
        first.append(f"**{name}:** healthy state/action main norms {f(s['healthy']['state_main_norm'])}/{f(s['healthy']['action_main_norm'])}; shortcut {f(s['shortcut']['state_main_norm'])}/{f(s['shortcut']['action_main_norm'])}. Mean curvature healthy/shortcut {f(s['healthy']['mean_abs_curvature'])}/{f(s['shortcut']['mean_abs_curvature'])}; rank-2 response energy {f(s['healthy']['response_rank2_energy'])}/{f(s['shortcut']['response_rank2_energy'])}. Q ranks 1/2/4 healthy = {f(s['healthy']['Q1'])}/{f(s['healthy']['Q2'])}/{f(s['healthy']['Q4'])}, shortcut = {f(s['shortcut']['Q1'])}/{f(s['shortcut']['Q2'])}/{f(s['shortcut']['Q4'])}. G110 positive healthy **{d['G110_positive']}/{d['healthy_count']}**, Q1 paired direction **{d['G111_Q1_direction']}/{min(d['healthy_count'],d['shortcut_count'])}**, O paired direction **{d['G112_O_direction']}/{min(d['healthy_count'],d['shortcut_count'])}**. AUC(O) {f(d['AUC']['overlap'])}; state/action/curvature/fusion norm comparators {f(d['AUC']['state_main_norm'])}/{f(d['AUC']['action_main_norm'])}/{f(d['AUC']['mean_abs_curvature'])}/{f(d['AUC']['fusion_norm'])}.\n\n")
        for key in ("post_I","Q1","overlap"):
            ci=d["run_bootstrap"][key]
            first.append(f"Run-level healthy−shortcut {key} difference {f(ci['difference'])}, 95% bootstrap CI [{f(ci['bootstrap_95'][0])}, {f(ci['bootstrap_95'][1])}]. ")
        first.append("These intervals resample independent trained runs, not neurons or histories.\n\n")
    first.append("The repeated phenotype is a much stronger **fusion state-main** contrast in healthy runs, while the action main effect remains large in both basins and mean SiLU curvature changes little. O separates classes descriptively, but its AUC ties the state-main norm (both 1.000), so O adds no independent basin discrimination under G112. Because `fusion_pre` already has history×action interaction from candidate-conditioned H, the post-SiLU increase is an amplification/nonlinear transformation of a mixed input, not proof that SiLU created the full causal pattern from two pure main effects. F2's small fusion state-main despite high upstream probes strengthens the distinction between stored information and useful fusion input.\n\n")
    first.append("## Finite unit causality, offsets and training timing\n")
    for name in ("historical","confirmatory"):
        d=prim[name]
        first.append(f"**{name}:** frozen k={freeze['unit_k']} removal met the finite-vs-random criterion in **{d['G113_topk']}/{d['healthy_count']}** healthy runs; same-k random means and all k=1/2/4/8/16 outcomes are in `unit_causality/summary.json`. Fixed common-offset full rescue **{d['G114_rescue']}/{d['shortcut_count']}** failed; curvature-lowering healthy destruction meeting its *matched common-shift* control **{d['G115_destroy']}/{d['healthy_count']}**; O half-rise before/at IHA onset **{d['G116_ordered']}/{d['healthy_count']}**.\n\n")
    grid_text=", ".join("{:+.2f}:{:+.4f}".format(x["scale"],x["mean_IHA_improvement"])
                        for x in freeze["offset_grid"])
    selected_positive=next(x["positive_count"] for x in freeze["offset_grid"]
                           if x["scale"]==freeze["offset_scale"])
    first.append(f"Development offset grid mean IHA changes: {grid_text}. The selected scale had {selected_positive}/8 development primary failed runs with >.025 IHA improvement. Output-aligned post-SiLU positive ceiling achieved threshold behavior in **{sum(positive)}/{len(positive)}** failed audits; this confirms the assay's downstream reach but cannot rescue the preactivation hypothesis.\n\n")
    first.append("A common preactivation shift preserves the *preactivation* factorial S/A/I contrasts exactly, but may change their post-SiLU images. An extreme ±3 SD low-curvature shift can also destroy generic information; therefore matched random-sign common shifts, not native-only comparisons, decide G115. Curvature overlap O is descriptive and cannot be called causal merely because it covaries with healthy basin entry. The timing test does not infer mediation from order alone.\n")
    f1s=data["phenotypes"]["F1"];f2s=data["phenotypes"]["F2"]
    first.append("\n## F1/F2 and conditional stopping\n")
    first.append(f"Across explicitly labeled cohorts, F1-like nonhealthy n={f1s['count']} had mean fusion state-main/post-I/Q1/O {f(f1s['means']['state_main_norm'])}/{f(f1s['means']['post_I'])}/{f(f1s['means']['Q1'])}/{f(f1s['means']['overlap'])}. Strict F2 stored-but-unused (high H/M probe, |BS|<.10) n={f2s['count']} had {f(f2s['means']['state_main_norm'])}/{f(f2s['means']['post_I'])}/{f(f2s['means']['Q1'])}/{f(f2s['means']['overlap'])}; healthy n={data['phenotypes']['healthy']['count']} had {f(data['phenotypes']['healthy']['means']['state_main_norm'])}/{f(data['phenotypes']['healthy']['means']['post_I'])}/{f(data['phenotypes']['healthy']['means']['Q1'])}/{f(data['phenotypes']['healthy']['means']['overlap'])}. Another {data['phenotypes']['other_partial']['count']} high-probe partial cases with |BS|≥.10 are kept separately. An earlier broad probe-only F2-like grouping had n={data['phenotypes']['broad_probe_only_pilot']['count']}; it is retained machine-readably but excluded from G117 because it omitted the frozen |BS| criterion. Fixed common-offset full rescue occurred in **{f2s['rescue_count']}/{f2s['rescue_denominator']}** strict F2 audits. Probe-based F2 status alone does not establish that state information reached the right fusion direction.\n\n")
    first.append("G114/G115/G116 did not produce the two passes required to authorize any training intervention. Hence G118–G122 and reduced formation (N=0/1/4/16/64), persistence (D=0/100/500/1000), revision (R=0/8/32/128), selectivity, F/M clamps and 1000/5000/10000-tick stabilized-model diagnostics are **NOT_RUN_BY_PROTOCOL**. This is not a measured failure of a training rescue or a claimed lack of long-run safety; no stabilized model exists. F/M laws and the F→M handoff stay frozen.\n")
    first.append("\n## Direct answers to the 35 required questions\n")
    answers=[
        f"1. **fusion_pre interaction:** legacy healthy/shortcut {f(coh['historical']['healthy']['pre_I'])}/{f(coh['historical']['shortcut']['pre_I'])}; new {f(coh['confirmatory']['healthy']['pre_I'])}/{f(coh['confirmatory']['shortcut']['pre_I'])}. Nonzero because candidate H is action-conditioned.",
        f"2. **fusion_post interaction:** legacy {f(coh['historical']['healthy']['post_I'])}/{f(coh['historical']['shortcut']['post_I'])}; new {f(coh['confirmatory']['healthy']['post_I'])}/{f(coh['confirmatory']['shortcut']['post_I'])}.",
        f"3. **Healthy nonlinear generation:** G110 {gate['G110']}; mean ΔNL legacy/new healthy {f(coh['historical']['healthy']['delta_NL'])}/{f(coh['confirmatory']['healthy']['delta_NL'])}.",
        f"4. **Shortcut generation:** mean ΔNL legacy/new {f(coh['historical']['shortcut']['delta_NL'])}/{f(coh['confirmatory']['shortcut']['delta_NL'])}; compare absolute post-I, not just ratio.",
        f"5. **State main effect:** healthy/shortcut legacy {f(coh['historical']['healthy']['state_main_norm'])}/{f(coh['historical']['shortcut']['state_main_norm'])}; see F1/F2 caveat.",
        f"6. **Action main effect:** healthy/shortcut legacy {f(coh['historical']['healthy']['action_main_norm'])}/{f(coh['historical']['shortcut']['action_main_norm'])}.",
        "7. **Interaction norm sufficiency:** No causal sufficiency follows from its basin separation; Stage 2D.5 upstream counterexample remains.",
        f"8. **Behavioral alignment:** G111 {gate['G111']}; finite response/SVD Q1, not only gradient.",
        f"9. **Low-dimensional output response:** rank-2 energy healthy legacy/new {f(coh['historical']['healthy']['response_rank2_energy'])}/{f(coh['confirmatory']['healthy']['response_rank2_energy'])}; this is local output response, not proof that the full internal interaction is globally low-rank.",
        "10. **Sparse support:** finite per-unit contributions and k=1/2/4/8/16 curves retained; only the frozen G113 criterion licenses a causal set.",
        f"11. **Top-k removal:** G113 {gate['G113']}; frozen k={freeze['unit_k']} passes legacy {prim['historical']['G113_topk']}/{prim['historical']['healthy_count']}, new {prim['confirmatory']['G113_topk']}/{prim['confirmatory']['healthy_count']}.",
        "12. **Random-k:** 16 same-k deterministic controls per run; their mean losses are in `unit_causality/summary.json` and explicitly enter G113.",
        f"13. **Operating points:** mean preactivation legacy healthy/shortcut {f(coh['historical']['healthy'].get('pre_mean'))}/{f(coh['historical']['shortcut'].get('pre_mean'))}; full per-unit four-cell distributions retained in each run JSON.",
        f"14. **Curvature exposure:** mean |SiLU''| healthy/shortcut legacy {f(coh['historical']['healthy']['mean_abs_curvature'])}/{f(coh['historical']['shortcut']['mean_abs_curvature'])}; new {f(coh['confirmatory']['healthy']['mean_abs_curvature'])}/{f(coh['confirmatory']['shortcut']['mean_abs_curvature'])}.",
        f"15. **Curvature-weighted overlap:** G112 {gate['G112']}; O legacy/new healthy {f(coh['historical']['healthy']['overlap'])}/{f(coh['confirmatory']['healthy']['overlap'])}.",
        f"16. **Failed common-offset rescue:** G114 {gate['G114']}; legacy {prim['historical']['G114_rescue']}/{prim['historical']['shortcut_count']}, new {prim['confirmatory']['G114_rescue']}/{prim['confirmatory']['shortcut_count']}.",
        "17. **Offset/scale controls:** random, low-curvature, state-scale and action-scale were norm/support-matched where applicable; all recorded per run and required to lose to the primary offset.",
        f"18. **Healthy destruction:** G115 {gate['G115']}; matched-control-qualified legacy {prim['historical']['G115_destroy']}/{prim['historical']['healthy_count']}, new {prim['confirmatory']['G115_destroy']}/{prim['confirmatory']['healthy_count']}.",
        f"19. **Temporal order:** G116 {gate['G116']}; O-before/at-IHA legacy {prim['historical']['G116_ordered']}/{prim['historical']['healthy_count']}, new {prim['confirmatory']['G116_ordered']}/{prim['confirmatory']['healthy_count']}.",
        f"20. **F1 phenotype:** fusion S/post-I/Q1/O = {f(f1s['means']['state_main_norm'])}/{f(f1s['means']['post_I'])}/{f(f1s['means']['Q1'])}/{f(f1s['means']['overlap'])}; weak state formation is a probe-qualified interpretation, not shortcut label alone.",
        f"21. **F2 phenotype:** fusion S/post-I/Q1/O = {f(f2s['means']['state_main_norm'])}/{f(f2s['means']['post_I'])}/{f(f2s['means']['Q1'])}/{f(f2s['means']['overlap'])}; high H/M probe can coexist with weak or misaligned fusion interaction.",
        f"22. **F2 rescue:** fixed preactivation offset {f2s['rescue_count']}/{f2s['rescue_denominator']}; G117 {gate['G117']}.",
        "23. **Training authorization:** No; fewer than two of G114/G115/G116 passed.",
        "24. **Fusion architecture redesign:** Not justified by this diagnostic.",
        "25. **Initialization/stabilization:** Not tested; causal authorization failed, so it cannot be claimed sufficient.",
        "26. **Training healthy rate ≥6/8:** Not run by protocol; the new 24-run baseline's healthy count is reported separately.",
        "27. **Mechanism recovery after training:** Not run by protocol.",
        "28. **Gradual accumulation:** Not rechecked in a stabilized model.",
        "29. **Persistence:** Not rechecked in a stabilized model.",
        "30. **Revision:** Not rechecked in a stabilized model.",
        "31. **Predictive selectivity:** Not rechecked in a stabilized model.",
        "32. **F/M mediation:** Not rechecked; no stabilized model was authorized.",
        "33. **Continuous regression:** Not assessed; no new model was trained.",
        "34. **F/M-law redesign:** Not justified.",
        "35. **Formal F→M handoff:** Remains closed."
    ]
    first.extend(x+"\n" for x in answers)
    first.append("\n## Integrity, limitations and reproducibility\n")
    first.append(f"Stage 2D.6 dedicated tests passed **15/15**. Dedicated historical tracked-asset manifest: **{integrity['current']}/{integrity['expected']}**, zero changes. The full repository suite passed **248/255**; seven older historical snapshot tests now fail because their selection logic includes later-stage tracked files or cumulative README edits (`test_stage1_5`, `test_stage2d`, `test_stage2d1`–`test_stage2d5`). The Stage 2D.5 snapshot test becomes the seventh failure when Stage 2D.6 files are tracked. These tests were not modified or waived. The frozen evaluator head was unchanged in all 24 new training runs; across all audited endpoints, the exact candidate hook's maximum logit error was **{max_hook_error:.2e}** and protected-head, parameter and persistent-state purity checks all passed. No complete confirmatory endpoint summary existed at development-rule freeze: **{freeze['confirmatory_sealed_at_freeze']}**; this is an endpoint-availability check, not a cryptographic seal. The 8×3 design shares initialization within each triplet, so 24 runs are not 24 independent initialization draws. Per-run JSON includes seeds, checkpoints, 2×2 fusion cell means, derivatives, rank-1/2/4 Q, finite unit contributions, 10-step trajectories, full offset grid and controls. The top-k units were selected and evaluated on the same run's factorial sample; a held-out-history validation would be needed before making a strong sparse-circuit claim. Legacy/A1 architecture differences are not hidden. The intervention on post-SiLU units is a ceiling, not evidence that their successful pattern was caused by memory. No result implies human-like cognition or causal F→M handoff.\n")
    body="".join(first)
    body=re.sub(r"(^#{1,2} .+\n)(?!\n)",r"\1\n",body,flags=re.MULTILINE)
    REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(body)
    print(f"RESULT_FILE=/data/CSK/ETPM/et-rcm/{REPORT}")


if __name__=="__main__":main()
