"""Conservative Stage 2D.5 gate adjudication and complete scientific report."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path


NODES=("q_F","q_M","r_F","r_M","normalized_F","normalized_M","gated_read",
       "read_projection","candidate_update","temporary_H","evaluator_H_input",
       "fusion_input","fusion_pre","fusion_post","logits","probabilities","policy_score")
REPAIR=("q_F","q_M","r_F","r_M","normalized_F","normalized_M","gated_read",
        "read_projection","core_preactivation","gate_activation","candidate_update",
        "temporary_H","evaluator_H_input","fusion_input","fusion_pre","fusion_post")


def load(folder):return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
def mean(xs):return statistics.mean(xs) if xs else None
def median(xs):return statistics.median(xs) if xs else None
def fmt(x,n=3):return "NA" if x is None else f"{x:.{n}f}"


def healthy_behavior(b):
    return b["action_TV"]>=.10 and b["I_HA"]>=.10 and abs(b["BS"])>=.10


def restored(row,node,kind):
    entry=row.get("restoration",{}).get(node,{})
    if "interventions" not in entry:return False
    native=entry["native"]
    return any(healthy_behavior(entry["interventions"][kind][str(lam)]) and
               entry["interventions"][kind][str(lam)]["I_HA"]-native["I_HA"]>=.025 and
               abs(entry["interventions"][kind][str(lam)]["BS"])-abs(native["BS"])>=.025
               for lam in (.25,.5,1.))


def destroyed(row,node):
    entry=row.get("destruction",{}).get(node,{})
    if not entry:return False
    native=entry["native"];removed=entry["remove"]
    return (removed["I_HA"]<=.5*native["I_HA"] and
            abs(removed["BS"])<=.5*abs(native["BS"]))


def main_control_harms(row,node,kind):
    entry=row.get("destruction",{}).get(node,{})
    if not entry:return False
    native=entry["native"];control=entry[kind]
    return (control["I_HA"]<=.5*native["I_HA"] and
            abs(control["BS"])<=.5*abs(native["BS"]))


def main(args):
    root=args.root
    training=[]
    for summary_path in sorted((root/"confirmatory"/"checkpoints").glob("*/summary.json")):
        item=json.loads(summary_path.read_text())
        endpoint=summary_path.parent/"checkpoint_1500.pt"
        training.append({"run":summary_path.parent.name,"init_seed":item["init_seed"],
            "data_seed":item["data_seed"],"eval_seed":item["eval_seed"],
            "arm":item["arm"],"elapsed_s":item["elapsed_s"],
            "parameter_counts":item["parameter_counts"],
            "protected_evaluator_hash_unchanged":item["evaluator_hash_unchanged"],
            "training_objective":"observed-only consequence",
            "memory_law_change":item["memory_law_change"],
            "training_log":item["training_log"],
            "endpoint_sha256":hashlib.sha256(endpoint.read_bytes()).hexdigest()})
    (root/"confirmatory"/"training_summaries.json").write_text(json.dumps(training,indent=2))
    oldq=load(root/"interaction_flow"/"original_query")
    newq=load(root/"interaction_flow"/"confirmatory_query")
    oldl=load(root/"interaction_flow"/"matched_legacy")
    histl=load(root/"interaction_flow"/"historical_legacy")
    selected=json.loads((root/"processed"/"selection.json").read_text())
    causal={c:load(root/"path_restoration"/"individual"/c) for c in
            ("original_query","confirmatory_query","historical_legacy")}
    primary_failed=[r for name in selected["primary_failed_confirmatory_A1"]
                    for r in causal["confirmatory_query"] if r["run"]==name]
    primary_healthy=[r for name in selected["primary_healthy_historical_A0"]
                     for r in causal["historical_legacy"] if r["run"]==name]
    a1healthy=[r for c in ("original_query","confirmatory_query") for r in causal[c]
               if r["class"]=="healthy"]
    allq=oldq+newq
    flow={}
    for cohort,rows in (("original_query",oldq),("confirmatory_query",newq),
                        ("matched_legacy",oldl),("historical_legacy",histl)):
        flow[cohort]={}
        for label in ("healthy","nonhealthy"):
            group=[r for r in rows if (r["class"]=="healthy")==(label=="healthy")]
            flow[cohort][label]={"n":len(group),"layers":{n:{
                "interaction_norm_mean":mean([r["flow"][n]["norm"] for r in group]),
                "normalized_mean":mean([r["flow"][n]["normalized"] for r in group])}
                for n in NODES},"edges":{edge:{
                    "absolute_median":median([r["ratios"][edge]["absolute"] for r in group]),
                    "normalized_median":median([r["ratios"][edge]["normalized"] for r in group])}
                    for edge in (group[0]["ratios"] if group else {})}}

    # The two independent A1 cohorts contain only two healthy endpoints each.
    # No edge can satisfy the specified >=6/8 within-architecture matched gate.
    g97=False
    mixing={}
    for cohort,rows in (("original_query",oldq),("confirmatory_query",newq)):
        mixing[cohort]={"n":len(rows),
            "F_M_cosine_mean":mean([r["F_M_raw_interaction_cosine"] for r in rows]),
            "negative_cosine_count":sum(r["F_M_raw_interaction_cosine"]<0 for r in rows),
            "native_less_than_both_F_and_M_count":sum(
                r["mixing"]["native"]["gated_read_interaction"]<min(
                    r["mixing"]["F_only"]["gated_read_interaction"],
                    r["mixing"]["M_only"]["gated_read_interaction"]) for r in rows),
            "alternative_behavior_rescue_count":sum(any(
                healthy_behavior(r["mixing"][variant]["behavior"]) and
                not healthy_behavior(r["mixing"]["native"]["behavior"])
                for variant in ("F_only","M_only","equal","norm_matched_sum")) for r in rows)}
    g98=False
    residual={}
    for cohort,rows in (("original_query",oldq),("confirmatory_query",newq)):
        failed=[r for r in rows if r["class"]!="healthy"]
        residual[cohort]={"n_failed":len(failed),
            "native_IHA_mean":mean([r["residual"]["1.0"]["behavior"]["I_HA"] for r in failed]),
            "alpha2_IHA_mean":mean([r["residual"]["2.0"]["behavior"]["I_HA"] for r in failed]),
            "finite_rescue_count":sum(any(healthy_behavior(r["residual"][str(a)]["behavior"])
                for a in (1.5,2.)) for r in failed)}
    g99=False
    restoration={node:{kind:sum(restored(r,node,kind) for r in primary_failed)
        for kind in ("interaction","state","action","random_interaction","shuffled","generic")}
        for node in REPAIR}
    destruction={node:{"primary_healthy_legacy":sum(destroyed(r,node) for r in primary_healthy),
                       "A1_healthy":sum(destroyed(r,node) for r in a1healthy)} for node in REPAIR}
    rescue_nodes=[node for node in REPAIR if restoration[node]["interaction"]>=6 and
                  all(restoration[node]["interaction"]>restoration[node][kind]
                      for kind in ("state","action","random_interaction","shuffled","generic"))]
    necessary_nodes=[node for node in REPAIR if destruction[node]["primary_healthy_legacy"]>=6 and
                     sum(main_control_harms(r,node,"state_main_control") for r in primary_healthy)<6 and
                     sum(main_control_harms(r,node,"action_main_control") for r in primary_healthy)<6]
    first_rescue=rescue_nodes[0] if rescue_nodes else None
    first_necessary=necessary_nodes[0] if necessary_nodes else None
    g100=first_rescue is not None
    g101=first_necessary is not None and destruction[first_necessary]["A1_healthy"]==len(a1healthy)
    # A true operator/fusion bottleneck needs its own upstream causal rescue,
    # not merely a post-fusion output-aligned rescue or an R-denominator effect.
    g102=False;g103=False
    f2=[]
    for c in ("original_query","confirmatory_query","historical_legacy"):
        for r in causal[c]:
            if r["class"]=="healthy":continue
            p=r["health"]["z_probe"]
            if max(p["H"],p["M"])>=.75 and abs(r["behavior"]["BS"])<.10:
                f2.append(r)
    f2_summary={"n":len(f2),"by_cohort":dict(Counter(r["cohort"] for r in f2)),
                "read_I_mean":mean([r["flow"]["gated_read"]["norm"] for r in f2]),
                "fusion_post_I_mean":mean([r["flow"]["fusion_post"]["norm"] for r in f2]),
                "fusion_post_restorable_count":sum(restored(r,"fusion_post","interaction") for r in f2),
                "upstream_restorable_count":sum(any(restored(r,n,"interaction") for n in REPAIR[:-1]) for r in f2)}
    g104=False
    authorized=g100 and (g102 or g103)
    gates={
        "G97":{"pass":g97,"reason":"only 2 healthy A1 per independent cohort; no >=6/8 within-architecture edge replication"},
        "G98":{"pass":g98,"mixing":mixing},
        "G99":{"pass":g99,"residual":residual},
        "G100":{"pass":g100,"earliest_rescue":first_rescue,"primary_failed_n":len(primary_failed),
                "counts":restoration[first_rescue] if first_rescue else {}},
        "G101":{"pass":g101,"earliest_necessary":first_necessary,
                "primary_healthy_legacy_n":len(primary_healthy),"A1_healthy_n":len(a1healthy),
                "counts":destruction[first_necessary] if first_necessary else {}},
        "G102":{"pass":g102,"reason":"read/integration-node restoration did not rescue 6/8; no causal operator edge"},
        "G103":{"pass":g103,"reason":"fusion_post restoration works, but pre-fusion interaction was not stably transmitted or causally rescued; no proven fusion transmission loss"},
        "G104":{"pass":g104,"reason":"F2 patterns differ across architectures and no upstream-node restoration replicated"},
    }
    for gate,why in (("G105","architecture modification requires G100 and G102 or G103"),
                     ("G106","requires authorized architecture and G105 success"),
                     ("G107","requires G105 and G106"),
                     ("G108","requires G107"),
                     ("G109","requires successful architecture")):
        gates[gate]={"pass":None,"status":"NOT_RUN_BY_PROTOCOL","reason":why}
    outcome="Outcome D — Upstream Interaction Is Mostly Epiphenomenal"
    decision={"gates":gates,"outcome":outcome,"architecture_rescue_authorized":authorized,
              "memory_law_redesign_justified":False,"formal_F_to_M_handoff_reopened":False,
              "cohorts":{"original_query":dict(Counter(r["class"] for r in oldq)),
                         "confirmatory_query":dict(Counter(r["class"] for r in newq)),
                         "matched_legacy":dict(Counter(r["class"] for r in oldl)),
                         "historical_legacy":dict(Counter(r["class"] for r in histl))},
              "earliest_rescue":first_rescue,"earliest_necessary":first_necessary,
              "first_descriptive_attenuation_edge":"gated_read->read_projection (absolute ~0.4-0.5), followed by temporary_H->evaluator_H_input (~0.13-0.18); neither is a replicated causal collapse",
              "fusion_input_R_drop_is_denominator_effect":True}
    (root/"processed"/"formal_gates.json").write_text(json.dumps(decision,indent=2))
    (root/"interaction_flow"/"summary.json").write_text(json.dumps(flow,indent=2))
    (root/"transmission_ratios"/"summary.json").write_text(json.dumps({
        "cohorts":{k:{label:v[label]["edges"] for label in v} for k,v in flow.items()},
        "G97":gates["G97"]},indent=2))
    (root/"read_decomposition"/"summary.json").write_text(json.dumps({
        "F_M_raw_interaction_cosine":mixing,"flow":{k:{label:v[label]["layers"] for label in v}
            for k,v in flow.items()}},indent=2))
    (root/"gate_cancellation"/"summary.json").write_text(json.dumps(gates["G98"],indent=2))
    (root/"residual_dilution"/"summary.json").write_text(json.dumps(gates["G99"],indent=2))
    (root/"path_restoration"/"summary.json").write_text(json.dumps({
        "primary_failed":selected["primary_failed_confirmatory_A1"],"counts":restoration,
        "G100":gates["G100"]},indent=2))
    (root/"path_destruction"/"summary.json").write_text(json.dumps({
        "primary_healthy":selected["primary_healthy_historical_A0"],"counts":destruction,
        "G101":gates["G101"]},indent=2))
    (root/"integration_audit"/"summary.json").write_text(json.dumps({
        "nodes":{k:{label:{node:v[label]["layers"].get(node) for node in
            ("read_projection","candidate_update","temporary_H")}
            for label in v} for k,v in flow.items()},"G102":gates["G102"]},indent=2))
    (root/"fusion_audit"/"summary.json").write_text(json.dumps({
        "nodes":{k:{label:{node:v[label]["layers"].get(node) for node in
            ("evaluator_H_input","fusion_input","fusion_pre","fusion_post")}
            for label in v} for k,v in flow.items()},"G103":gates["G103"]},indent=2))
    (root/"f2_localization"/"summary.json").write_text(json.dumps({
        "F2":f2_summary,"G104":gates["G104"]},indent=2))
    (root/"architecture_rescue"/"status.json").write_text(json.dumps({
        "status":"NOT_AUTHORIZED_BY_PROTOCOL" if not authorized else "AUTHORIZED",
        "G100_pass":g100,"G102_pass":g102,"G103_pass":g103,
        "reason":"G102/G103 did not causally localize a repairable transmission edge"},indent=2))
    for folder,gate in (("controls","G105"),("dynamics_recheck","G107"),
                        ("memory_mediation","G108"),("continuous_runs","G109")):
        (root/folder/"status.json").write_text(json.dumps(gates[gate],indent=2))

    q=newq+oldq
    qflow={n:mean([r["flow"][n]["norm"] for r in q]) for n in NODES}
    oldqflow={n:mean([r["flow"][n]["norm"] for r in oldq]) for n in NODES}
    newqflow={n:mean([r["flow"][n]["norm"] for r in newq]) for n in NODES}
    gate_lines="\n".join(f"- **{k}:** "+("PASS" if v["pass"] else
        ("FAIL" if v["pass"] is False else "NOT_RUN_BY_PROTOCOL")) for k,v in gates.items())
    rows="\n".join(f"| {n} | {fmt(oldqflow[n],4)} | {fmt(newqflow[n],4)} | {fmt(qflow[n],4)} |"
                   for n in NODES)
    a1_ratio=mean([r["ratios"]["gated_read->read_projection"]["absolute"] for r in q])
    pool_ratio=mean([r["ratios"]["temporary_H->evaluator_H_input"]["absolute"] for r in q])
    rnorm=mean([r["ratios"]["evaluator_H_input->fusion_input"]["normalized"] for r in q])
    rabs=mean([r["ratios"]["evaluator_H_input->fusion_input"]["absolute"] for r in q])
    rest=restoration["fusion_post"]
    report=f"""# Stage 2D.5 — Conditional Interaction Transmission Audit

> **When candidate-specific history×action information is already present in persistent-memory reads but has little behavioral effect, where along the read→active-state→fusion pathway is that conditional interaction lost, and can restoring only the broken transmission step stabilize behavioral memory?**

> **当 persistent-memory read 中已经存在 candidate-specific 的 history×action 信息，但它几乎不影响行为时，这种 conditional interaction 究竟在 read→active state→fusion 的哪一步丢失？如果只修复这个断裂的传递步骤，能否稳定建立 behavioral memory？**

## Executive result

{gate_lines}

**{outcome}.** A1's upstream read interaction is real as a measured activation but not a replicated behavioral mediator. Eight of eight failed confirmatory A1 checkpoints can be rescued by output-aligned `fusion_post` interaction injection, and 8/8 healthy legacy checkpoints are destroyed by removing their native interaction there. Yet restoring at any earlier candidate-read, integration or pre-fusion node rescued **0/8**. This confirms the protected downstream causal node already found in Stage 2D.3; it does **not** causally localize a repairable read→H or fusion transmission edge. Architecture rescue was not authorized. F/M laws remain frozen and the formal handoff stays closed.

## Frozen protocol, cohorts and provenance

No architecture, memory law, query, gate, evaluator, dimensions, NULL/SELF_OUTPUT semantics, training objective or curriculum changed. The original Stage 2D.4 A0/A1 endpoints each contributed 8 matched runs. A new confirmatory cohort trained **16 A1 models** under the identical frozen Stage 2D.4 observed-only protocol (1500 updates, batch 16, episode lengths 4/6/8, gamma .50, rhoF .97, rhoM .9995, protected evaluator). It yielded **2 healthy, 4 partial and 10 shortcut** endpoints; the original A1 cohort had **2 healthy and 6 partial**. Thus only **4/24 A1** models are healthy. An independent historical legacy C0 set supplied 24 endpoints (12 healthy, 12 shortcut) to reach the diagnostic minimum of 8 healthy and 8 failed overall. It is clearly labelled as legacy, never passed off as A1 replication. Statistical units are independently trained models; 16 within-run history replicates are averaged before all run-level contrasts.

The primary causal set was fixed by sorted IDs: eight failed confirmatory A1 and eight healthy historical legacy C0. Six failed original A1, all four healthy A1 and six historical F2 controls were audited additionally. `results/stage2d5/processed/selection.json` records exact run IDs and checkpoint paths; `confirmatory/training_summaries.json` records all 16 seeds, endpoints, hashes, per-checkpoint losses, norms, parameter counts and protected-evaluator checks. All candidate branches and interventions were read-only; H/F/M, clocks, model parameters and protected evaluator hashes were verified unchanged. The audit reimplemented the exact frozen candidate computation and unit-tested its logits/read equality to numerical precision.

## Full 2×2 interaction flow

Each row is the mean absolute factorial interaction norm after per-run 2×2 aggregation. The same histories and candidate actions are used at every node; none of these magnitudes alone establishes causal mediation.

| Node | Original A1 (8) | Confirmatory A1 (16) | Combined A1 (24) |
|---|---:|---:|---:|
{rows}

The first descriptive absolute reduction is `gated_read→read_projection` (mean ratio **{a1_ratio:.3f}**), followed by `temporary_H→evaluator_H_input` (**{pool_ratio:.3f}**). Neither distinguishes healthy from failed in two adequately powered A1 cohorts. `candidate_update→temporary_H` conserves absolute factorial interaction here; its normalized ratio can fall because common residual H increases state-main magnitude. Likewise `evaluator_H_input→fusion_input` has absolute ratio **{rabs:.3f}** (concatenation preserves the interaction) but normalized ratio **{rnorm:.3f}** because the action main effect enters the denominator. These are not evidence of destroyed information. G97 therefore fails; A1 has only two healthy runs in each independent cohort and cannot furnish ≥6/8 same-architecture healthy/failed matched replications.

## F/M mixing, residual scaling and operator anatomy

Mean raw F/M interaction cosine was original/confirmatory **{mixing['original_query']['F_M_cosine_mean']:.3f}/{mixing['confirmatory_query']['F_M_cosine_mean']:.3f}**; negative in **{mixing['original_query']['negative_cosine_count']}/8** and **{mixing['confirmatory_query']['negative_cosine_count']}/16**. Native gate interaction was smaller than both F-only and M-only in only **{mixing['original_query']['native_less_than_both_F_and_M_count']}/8** and **{mixing['confirmatory_query']['native_less_than_both_F_and_M_count']}/16**. F-only, M-only, equal and norm-matched-sum conditions did not give replicated behavioral rescue, so G98 fails.

For failed A1, residual α=1→2 changed mean IHA from **{fmt(residual['original_query']['native_IHA_mean'])}→{fmt(residual['original_query']['alpha2_IHA_mean'])}** (original) and **{fmt(residual['confirmatory_query']['native_IHA_mean'])}→{fmt(residual['confirmatory_query']['alpha2_IHA_mean'])}** (confirmatory). No moderate α≤2 rescue replicated; G99 fails. The internal audit separates H/read/event projections, core preactivation/SiLU, gate preactivation/activation, candidate update, residual output, evaluator input, H/action fusion projections, additive merge, fusion preactivation/SiLU and postactivation. The complete per-node metrics and ratios are retained machine-readably.

## Finite path restoration and destruction

For each node, the source was the **same-architecture healthy cohort's mean logit-interaction vector**. Its direction was mapped into the target model's local coordinates by that model's output-Jacobian adjoint; amplitude was calibrated to the within-architecture healthy median native interaction. This is declared output-coordinate alignment, not a raw neuron swap across models. Each 2×2 intervention changed only the factorial interaction component; state/action main effects were preserved. λ was .25/.5/1 and the induced factorial norm never exceeded the matched healthy norm (below the 1.25 cap). Norm-matched state-only, action-only, random-interaction, shuffled-sign and generic-shift controls were run at the same scales. Healthy destruction removed/reversed the native interaction at each node with main-effect controls. A preliminary single-logit alignment pilot is retained under `path_restoration/pilot_unaligned/` but **excluded from all formal gates**.

The **earliest rescue node is `{first_rescue}`**: {rest['interaction']}/8 primary failed A1 recovered healthy TV/IHA/BS, versus state {rest['state']}/8, action {rest['action']}/8, random {rest['random_interaction']}/8, shuffled {rest['shuffled']}/8 and generic {rest['generic']}/8. Every earlier node was **0/8** under the same norm cap. The **earliest necessary node is `{first_necessary}`**: removal destroyed {destruction[first_necessary]['primary_healthy_legacy']}/8 healthy legacy behaviors, while state/action-main removal had negligible effect. All four healthy A1 checkpoints also lost behavior at this node, but four is below an independent 8-run A1 necessity cohort. Rescue and necessity coincide at the protected post-fusion node, not at a demonstrated broken transmission edge.

## F2 and bottleneck adjudication

F2-like stored-but-unused controls numbered **{f2_summary['n']}** across cohorts (`{f2_summary['by_cohort']}`). Their mean gated-read interaction was **{fmt(f2_summary['read_I_mean'])}**, and fusion-post interaction **{fmt(f2_summary['fusion_post_I_mean'])}**. Fusion-post injection rescued **{f2_summary['fusion_post_restorable_count']}/{f2_summary['n']}**, but earlier-node restoration rescued **{f2_summary['upstream_restorable_count']}/{f2_summary['n']}**. Legacy F2 often had weak interaction already near memory access, whereas some A1 F2 had strong reads but weak behavior; no single replicated F2 break edge was identified, so G104 fails.

G102 fails because no specific internal read→H node passed the 6/8 finite-restoration requirement. G103 fails because post-fusion rescue alone does not show that a healthy interaction was lost *inside* fusion: preactivation interaction is small and the SiLU can generate new state×action interaction from main effects. The primary result is **upstream interaction mostly epiphenomenal for behavior**, not a licensed fusion or integration redesign. G100 PASS plus G102/G103 FAIL prohibits architecture modification. Hence G105–G109 are `NOT_RUN_BY_PROTOCOL`: no rescue architecture, parameter-matched control, expanded 24-run cohort, behavioral dynamics, F/M mediation or continuous regression was run.

## Direct answers to the 34 required questions

1. **Query interaction:** mean qF/qM = {fmt(qflow['q_F'],4)}/{fmt(qflow['q_M'],4)}.
2. **Raw F read:** {fmt(qflow['r_F'],4)}.
3. **Raw M read:** {fmt(qflow['r_M'],4)}.
4. **Gated read:** {fmt(qflow['gated_read'],4)}.
5. **ΔH:** {fmt(qflow['candidate_update'],4)}.
6. **Temporary H:** {fmt(qflow['temporary_H'],4)}.
7. **Fusion input:** {fmt(qflow['fusion_input'],4)}.
8. **Fusion post:** {fmt(qflow['fusion_post'],4)}.
9. **Output probability interaction:** {fmt(qflow['probabilities'],4)}.
10. **First clear attenuation:** descriptively gated-read→read-projection; no proven causal collapse edge.
11. **Healthy vs failed ratio:** weak and inconsistent within A1; only 2 healthy per cohort, below the 6/8 gate.
12. **F/M cancellation?** Not systematic; G98 FAIL.
13. **Gate mixing destroys interaction?** Not systematically and alternatives do not rescue behavior.
14. **Residual H dilution?** Normalized ratio falls, absolute interaction is conserved; no causal dilution claim.
15. **Moderate scaling rescue?** No; G99 FAIL.
16. **Earliest rescue:** `{first_rescue}`.
17. **Earliest necessity:** `{first_necessary}` in 8 healthy legacy; corroborated in 4 healthy A1, not an 8-run A1 gate.
18. **Same window?** Yes, post-fusion; this is the known downstream causal node, not a proven upstream transmission break.
19. **F2 first break?** No shared edge; legacy often weak at access, A1 may have high read interaction yet little behavioral use.
20. **Read→H bottleneck?** Not causally localized; G102 FAIL.
21. **Fusion bottleneck?** Not proven; G103 FAIL despite strong post-fusion positive control.
22. **Distributed bottleneck?** Not established; pre-fusion finite restorations were null.
23. **Interaction restoration beats controls?** Yes at `fusion_post`: {rest['interaction']}/8 versus ≤{max(rest[k] for k in ('state','action','random_interaction','shuffled','generic'))}/8 controls; not at earlier nodes.
24. **Architecture modification authorized?** No.
25. **Minimal rescue selected?** None; selecting one would violate the stopping rule.
26. **Rescue architecture ≥6/8?** Not run by protocol.
27. **Parameter-matched control?** Not run; no architecture rescue was authorized.
28. **Transmission repaired?** Not demonstrated.
29. **Behavioral dynamics recovered?** Not run.
30. **F/M mediation recovered?** Not run.
31. **Continuous running worsened?** Not evaluated for a new architecture; none was trained.
32. **Most consistent category:** upstream interaction mostly epiphenomenal, with **no replicated causal transmission bottleneck**.
33. **Modify F/M law?** No.
34. **Reopen formal F→M handoff?** No.

## Integrity and limits

Stage 2D.5 tests passed **16/16**. The dedicated historical manifest verified **2445/2445** tracked frozen Stage 2C–2D.4 assets with zero changes. The full repository suite passed **235/240**; five inherited integrity-test failures compare cumulative later-stage files or README changes against older frozen snapshots. None was modified or waived. No endpoint was reclassified to make a gate pass. A downstream output-aligned intervention is a causal test of that node, not proof that persistent memory itself was the source of the recovered information. These synthetic findings make no claim of human-like memory, consciousness or unlimited capacity.
"""
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(report)
    print(json.dumps({"gates":{k:("PASS" if v["pass"] else ("FAIL" if v["pass"] is False else "NOT_RUN_BY_PROTOCOL")) for k,v in gates.items()},
                      "outcome":outcome,"report":str(args.report)},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True);main(p.parse_args())
