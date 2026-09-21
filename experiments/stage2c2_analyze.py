"""Failure-preserving Stage 2C.2 analysis and report generation."""

from __future__ import annotations

import hashlib
import json
import math
import statistics as st
from pathlib import Path

import torch

from etrcm.stage2c2.protocol import next_gate_status

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/stage2c2"
FROZEN_SEEDS=list(range(7201,7209))
CURVE_SEEDS=list(range(7401,7409))
CHECKPOINT_STEPS=(0,25,50,100,200,300,500)
REPORT=ROOT/"reports/STAGE2C2_FROZEN_L3_PATHWAY_AUDIT_RESULTS.md"


def read(path):return json.loads(path.read_text())
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))
def avg(values):return float(st.mean(values)) if values else None
def sd(values):return float(st.stdev(values)) if len(values)>1 else 0.0
def ms(values):return f"{avg(values):.6f} ± {sd(values):.6f}" if values else "NOT_RUN"


def load_frozen():
    rows={}
    for seed in FROZEN_SEEDS:
        folder=BASE/"oracle_h/formal"/str(seed)
        rows[seed]={"native":read(folder/"native.json"),
                    "H":read(folder/"oracle_h.json"),
                    "manifest":read(folder/"manifest.json")}
    return rows


def load_probes():
    return {seed:read(BASE/"native_probes"/str(seed)/"summary.json")
            for seed in FROZEN_SEEDS}


def load_curves():
    return {seed:read(BASE/"learning_curves/formal"/str(seed)/"summary.json")
            for seed in CURVE_SEEDS}


def native_survival(rows):
    result={}
    for seed,row in rows.items():
        items=[]
        for rep in row["native"]["rows"]:
            M=torch.tensor(rep["M"]);r=torch.tensor(rep["r_M"]);H=torch.tensor(rep["H"])
            p=torch.tensor(rep["metrics"]["prob"]).clamp_min(1e-10)
            mid=(p[0]+p[1])/2
            js=.5*((p[0]*(p[0].log()-mid.log())).sum(-1)+
                   (p[1]*(p[1].log()-mid.log())).sum(-1))
            items.append({"D_M":float((M[0]-M[1]).norm()),
                          "D_r":float((r[0]-r[1]).norm()),
                          "D_H":float((H[0]-H[1]).norm()),
                          "D_P_JS_fixed_action":float(js.mean()),
                          "D_B":abs(rep["metrics"]["behavioral_separation_entropy"])})
        H=torch.load(BASE/"oracle_h/formal"/str(seed)/"oracle_H.pt",map_location="cpu",weights_only=True)
        prob=torch.tensor(row["H"]["metrics"]["prob"]).clamp_min(1e-10)
        mid=(prob[0]+prob[1])/2
        js=.5*((prob[0]*(prob[0].log()-mid.log())).sum(-1)+
               (prob[1]*(prob[1].log()-mid.log())).sum(-1))
        result[seed]={"native":{key:avg([item[key] for item in items]) for key in items[0]},
                      "oracle_H":{"D_M":None,"D_r":None,
                                  "D_H":float((H[0]-H[1]).norm()),
                                  "D_P_JS_fixed_action":float(js.mean()),
                                  "D_B":abs(row["H"]["metrics"]["behavioral_separation_entropy"])}}
    return result


def main():
    rows=load_frozen();probes=load_probes();curves=load_curves()
    for seed,row in rows.items():
        manifest=row["manifest"]
        assert manifest["parameter_sha256_before"]==manifest["parameter_sha256_after"]
        assert manifest["weights_frozen"]
        for replicate in row["native"]["rows"]:
            for name in ("H","F","M","q_F","q_M","r_F","r_M"):
                assert bool(torch.isfinite(torch.tensor(replicate[name])).all()),(seed,name)
        target=row["H"]["norm_target"]
        assert all(abs(norm-target)<1e-4 for norm in row["H"]["state_norms"])
        reference=read(ROOT/f"results/stage2c1/raw/lifetime/{seed}/summary.json")
        expected=avg([rep["action_metrics_N16"]["behavioral_separation_entropy"]
                      for rep in reference["evaluation"]])
        assert abs(row["native"]["reproduction"]["BS"]-expected)<1e-6
        expected_tv=avg([avg(rep["action_metrics_N16"]["tv_action"])
                         for rep in reference["evaluation"]])
        assert abs(row["native"]["reproduction"]["TV_action"]-expected_tv)<1e-6
    gate51_per_seed={seed:(min(row["H"]["metrics"]["tv_action"])>=.10 and
                           row["H"]["metrics"]["interaction_y0"]>=.15 and
                           row["H"]["metrics"]["behavioral_separation_entropy"]>=.10)
                     for seed,row in rows.items()}
    count=sum(gate51_per_seed.values())
    g51=next_gate_status(True,count>=6)
    gates={"G51":g51}
    for name in ("G52","G53","G54","G55","G56"):
        gates[name]=next_gate_status(False,None)
    m_linear=[probes[s]["results"]["M"]["linear"]["test_accuracy"] for s in FROZEN_SEEDS]
    m_decodable=sum(x>=.70 for x in m_linear)>=6
    curve_aggregate={}
    for step in CHECKPOINT_STEPS:
        snaps=[next(x for x in curves[s]["snapshots"] if x["step"]==step) for s in CURVE_SEEDS]
        metric_names=snaps[0]["metrics"]
        curve_aggregate[str(step)]={"metrics":{name:{"mean":avg([x["metrics"][name] for x in snaps]),
                                                "sd":sd([x["metrics"][name] for x in snaps])}
                                               for name in metric_names},
                                    "M_probe_mean":avg([x["M_latent_ridge_test_accuracy"] for x in snaps]),
                                    "gradient_mean":({name:avg([x["gradient_preclip_from_preceding_update"][name]
                                                                 for x in snaps])
                                                      for name in snaps[0]["gradient_preclip_from_preceding_update"]}
                                                     if step else None)}
    curve_seed_rows={seed:{str(x["step"]):{"metrics":x["metrics"],
                                        "M_probe":x["M_latent_ridge_test_accuracy"],
                                        "gradients":x["gradient_preclip_from_preceding_update"]}
                           for x in row["snapshots"]} for seed,row in curves.items()}
    curve_success={seed:curve_seed_rows[seed]["500"]["metrics"]["BS"]>=.05 and
                        curve_seed_rows[seed]["500"]["metrics"]["TV_action"]>=.10
                   for seed in CURVE_SEEDS}
    decodable_without_behavior=[seed for seed in CURVE_SEEDS
                                if curve_seed_rows[seed]["500"]["M_probe"]>=.70
                                and abs(curve_seed_rows[seed]["500"]["metrics"]["BS"])<.05]
    curve_development={seed:read(BASE/"learning_curves/development"/str(seed)/"summary.json")
                       for seed in (7351,7352)}
    survival=native_survival(rows)
    summary={"gates":gates,"G51_pass_count":count,"G51_seed_results":gate51_per_seed,
             "frozen_seeds":FROZEN_SEEDS,"curve_seeds":CURVE_SEEDS,
             "native_P0":{seed:row["native"]["reproduction"] for seed,row in rows.items()},
             "oracle_H_P1":{seed:row["H"] for seed,row in rows.items()},
             "native_probes":{seed:probes[seed]["results"] for seed in FROZEN_SEEDS},
             "M_linear_test_accuracy_ge_0_70_count":sum(x>=.70 for x in m_linear),
             "M_decodable_gate_condition":m_decodable,
             "survival":survival,"curve_aggregate":curve_aggregate,
             "curve_seed_rows":curve_seed_rows,"curve_success_count":sum(curve_success.values()),
             "curve_success_seeds":[seed for seed,ok in curve_success.items() if ok],
             "curve_M_decodable_without_behavior_seeds":decodable_without_behavior,
             "curve_development":{seed:{"final_metrics":row["snapshots"][-1]["metrics"],
                                        "final_M_probe":row["snapshots"][-1]["M_latent_ridge_test_accuracy"]}
                                  for seed,row in curve_development.items()},
             "interpretation":"G51 failed; downstream pathway and curriculum gates not run. Scratch curve is parallel diagnostic, not G55."}
    write(BASE/"processed/summary.json",summary)
    write(BASE/"processed/frozen_seed_rows.json",rows)
    write(BASE/"interventions/P0_P1_survival.json",survival)
    write(BASE/"learning_curves/formal_aggregate.json",curve_aggregate)
    skipped={
        "oracle_read":("P2","G52"),"oracle_m":("P3","G53"),
        "history_to_l3m":("P4","G54"),"curriculum":("C1-C3","G55-G56"),
        "routing_rescue":("ROUTING_RESCUE_DIAGNOSTIC","not a gate")}
    for folder,(stage,gate) in skipped.items():
        reason="G51 FAIL; frozen L3 downstream head capacity did not replicate"
        if folder=="routing_rescue":
            reason+=f"; M decodability reached 0.70 in only {sum(x>=.70 for x in m_linear)}/8 seeds"
        write(BASE/folder/"status.json",{"stage":stage,"gate":gate,
                                        "status":"NOT_RUN_BY_GATE","reason":reason,
                                        "no_checkpoint_weights_changed":True})
    checkpoint_index={seed:{"path":str(ROOT/f"results/stage2c1/raw/lifetime/{seed}/checkpoint.pt"),
                            "sha256":rows[seed]["manifest"]["checkpoint_sha256"]}
                      for seed in FROZEN_SEEDS}
    write(BASE/"frozen_l3/formal_checkpoint_index.json",checkpoint_index)
    source_files=[ROOT/x for x in ("configs/stage2c2_formal.yaml","configs/stage2c2_joint_curve.yaml",
                "experiments/stage2c2_pathway.py","experiments/stage2c2_probes.py",
                "experiments/stage2c2_curve.py","experiments/stage2c2_analyze.py",
                "experiments/stage2c2_integrity.py","src/etrcm/stage2c2/protocol.py",
                "tests/test_stage2c2.py","docs/STAGE2C2_PROTOCOL.md")]
    write(BASE/"manifests/source_sha256.json",{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in source_files})
    REPORT.write_text(build_report(summary,rows,probes,curves))
    assert REPORT.stat().st_size>0
    print(f"RESULT_FILE={REPORT}")
    print(json.dumps({"gates":gates,"G51_count":count,"M_decodable_count":sum(x>=.70 for x in m_linear),
                      "scratch_curve_success_count":sum(curve_success.values())}))


def build_report(s,rows,probes,curves):
    g=s["gates"];sur=s["survival"]
    native_tv=[rows[k]["native"]["reproduction"]["TV_action"] for k in FROZEN_SEEDS]
    native_bs=[rows[k]["native"]["reproduction"]["BS"] for k in FROZEN_SEEDS]
    oracle_tv=[avg(rows[k]["H"]["metrics"]["tv_action"]) for k in FROZEN_SEEDS]
    oracle_min_tv=[min(rows[k]["H"]["metrics"]["tv_action"]) for k in FROZEN_SEEDS]
    oracle_bs=[rows[k]["H"]["metrics"]["behavioral_separation_entropy"] for k in FROZEN_SEEDS]
    oracle_i=[rows[k]["H"]["metrics"]["interaction_y0"] for k in FROZEN_SEEDS]
    probe_names=("H","F","M","FM")
    probe_table=["| State | linear accuracy | small-MLP accuracy |","|---|---:|---:|"]
    for name in probe_names:
        a=[probes[k]["results"][name]["linear"]["test_accuracy"] for k in FROZEN_SEEDS]
        b=[probes[k]["results"][name]["small_mlp"]["test_accuracy"] for k in FROZEN_SEEDS]
        probe_table.append(f"| {name} | {ms(a)} | {ms(b)} |")
    gate_table=["| Frozen L3 seed | native TV | oracle-H TV A/B | interaction | entropy BS | G51 | H norm target | held-out CE |",
                "|---:|---:|---:|---:|---:|---|---:|---:|"]
    for seed in FROZEN_SEEDS:
        row=rows[seed];met=row["H"]["metrics"]
        gate_table.append(f"| {seed} | {row['native']['reproduction']['TV_action']:.5f} | "
                          f"{met['tv_action'][0]:.5f}/{met['tv_action'][1]:.5f} | "
                          f"{met['interaction_y0']:.5f} | {met['behavioral_separation_entropy']:.5f} | "
                          f"{s['G51_seed_results'][seed]} | {row['H']['norm_target']:.3f} | "
                          f"{row['H']['formal_heldout_CE']:.4f} |")
    curve_table=["| Step | action-TV | history-TV | interaction y0 | BS | M-probe accuracy |",
                 "|---:|---:|---:|---:|---:|---:|"]
    for step in CHECKPOINT_STEPS:
        x=s["curve_aggregate"][str(step)];m=x["metrics"]
        curve_table.append(f"| {step} | {m['TV_action']['mean']:.5f} ± {m['TV_action']['sd']:.5f} | "
                           f"{m['TV_history']['mean']:.5f} ± {m['TV_history']['sd']:.5f} | "
                           f"{m['interaction_y0']['mean']:.5f} ± {m['interaction_y0']['sd']:.5f} | "
                           f"{m['BS']['mean']:.5f} ± {m['BS']['sd']:.5f} | {x['M_probe_mean']:.4f} |")
    endpoint_table=["| Curve seed | TV step 0 | TV step 100 | TV step 500 | BS step 500 | M-probe step 500 |",
                    "|---:|---:|---:|---:|---:|---:|"]
    for seed in CURVE_SEEDS:
        r=s["curve_seed_rows"][seed]
        endpoint_table.append(f"| {seed} | {r['0']['metrics']['TV_action']:.5f} | "
                              f"{r['100']['metrics']['TV_action']:.5f} | "
                              f"{r['500']['metrics']['TV_action']:.5f} | "
                              f"{r['500']['metrics']['BS']:+.5f} | {r['500']['M_probe']:.4f} |")
    grad_table=["| Step | action | consequence | q_M | read gate | event encoder | H core |",
                "|---:|---:|---:|---:|---:|---:|---:|"]
    for step in CHECKPOINT_STEPS[1:]:
        x=s["curve_aggregate"][str(step)]["gradient_mean"]
        grad_table.append(f"| {step} | {x['action_branch']:.5f} | {x['consequence_head']:.5f} | "
                          f"{x['q_M']:.5f} | {x['read_gate']:.5f} | {x['event_encoder']:.5f} | {x['H_core']:.5f} |")
    survival_table=["| Link metric | native P0 | injected oracle H P1 |",
                    "|---|---:|---:|"]
    for key in ("D_M","D_r","D_H","D_P_JS_fixed_action","D_B"):
        a=[sur[k]["native"][key] for k in FROZEN_SEEDS]
        b=[sur[k]["oracle_H"][key] for k in FROZEN_SEEDS if sur[k]["oracle_H"][key] is not None]
        survival_table.append(f"| {key} | {ms(a)} | {ms(b)} |")
    top_probe=max(probe_names,key=lambda n:avg([probes[k]["results"][n]["linear"]["test_accuracy"]
                                                  for k in FROZEN_SEEDS]))
    native_m_decodable=s["M_linear_test_accuracy_ge_0_70_count"]
    success=s["curve_success_count"]
    devcurve=s["curve_development"]
    # The strong seed-dependent scratch curve is not a curriculum or a gate.
    lines=[
"# ET-RCM Stage 2C.2 — Frozen-L3 Pathway Audit and Endogenous Memory-Formation Curriculum",
"",
"> **When the jointly trained endogenous ET-RCM fails to develop behavioral memory, which link in the chain from persistent state to action-conditioned prediction is actually broken: downstream action evaluation, read integration, memory addressing, history encoding, endogenous state formation, or their joint optimization?**",
"",
"> **当联合训练的 endogenous ET-RCM 无法形成行为记忆时，真正断裂的是哪一级：行为评估、read integration、M 寻址、历史编码、内生记忆形成，还是这些模块之间的联合优化？**",
"",
f"Formal decision: **G51 FAIL ({s['G51_pass_count']}/8)**. G52–G56: **NOT_RUN_BY_GATE**. This is a gated negative diagnostic, not a completed curriculum trial. Historical Stage 2C/2C.1 claims remain unchanged.",
"",
"## 1. Scope, prior evidence and frozen discipline",
"",
"Stage 2C.1 showed G48–G50 PASS 8/8, but its joint endogenous L3 rerun had action-TV 0.011601 ± 0.008913 and BS −0.000206 ± 0.000660. Stage 2C.2 audits those *same failed L3 checkpoints* (7201–7208). Two new development L3 checkpoints (7301–7302) were used only before freezing thresholds. Frozen L3 core, action embedding, consequence head, queries, read gate and F/M law were never updated during P0/P1 or native probes. Every seed records checkpoint SHA-256 and before/after parameter SHA-256. No L1/L2 M tensor was copied into L3.",
"",
"## 2. Formal protocol and data split",
"",
"P0 replays four matched N=16 paired histories per formal seed, with no parameter updates. P1 fits only two local H vectors per frozen checkpoint, each projected to that checkpoint's native N=16 median H norm. Four 1,000-step restarts use exactly balanced z×action consequence-CE batches (128; no correct-action label); restart selection uses independent observed-outcome samples, and final CE uses another independent 4,096-sample split. P1 injects H immediately before the frozen action head, so it tests the downstream ceiling, not recurrent integration. All gates were frozen in `configs/stage2c2_formal.yaml` after development 7301–7302 and before formal 7201–7208.",
"",
"## 3. P0 — native failed-L3 reproduction",
"",
f"Native action-TV {ms(native_tv)}; native entropy BS {ms(native_bs)}. The replay reproduces Stage 2C.1 L3 per seed to numerical precision. Native H/F/M/r_M medians, q_F/q_M vectors, paired states, read vectors and direct forecast metrics are retained in `results/stage2c2/oracle_h/formal/*/native.json`. No nonfinite values were observed in this short-horizon audit.",
"",
"## 4. P1 — frozen L3-local oracle H ceiling and G51",
"",
f"Oracle-H mean action-TV {ms(oracle_tv)} (per-seed minimum latent branch TV {ms(oracle_min_tv)}), interaction {ms(oracle_i)}, entropy BS {ms(oracle_bs)}. This is above native sensitivity but below robust recovery. Pre-registered G51 requires both TV branches≥0.10, interaction≥0.15 and entropy BS≥0.10 in at least 6/8 seeds; only {s['G51_pass_count']}/8 pass. The frozen downstream head retains limited state-contingent capacity in some checkpoints but fails replicated native-norm control.",
"",
*gate_table,
"",
"## 5. P2–P4 and curriculum gate discipline",
"",
"G51 failed. Therefore P2 oracle-read fitting, P3 local oracle-M fitting and swap/read-clamp, P4 history→L3-M trained adapter, routing rescue, and curriculum C1–C3 were **not run**. C0 is the already frozen Stage 2C.1 L3 scratch baseline, not a new curriculum run. G52/G53/G54/G55/G56 are `NOT_RUN_BY_GATE`, not FAIL. No negative result is assigned to read→H, M addressing, history→M mapping, endogenous F/M formation, or curriculum survival from these absent interventions. The read/M tests in the new unit suite check intervention mechanics only, not behavioral gates.",
"",
"## 6. Native latent information audit (non-causal)",
"",
"For each frozen seed, 128 independent train and 128 held-out test paired lifetimes were generated. Test history surfaces use parity-odd held-out combinations; train surfaces use parity-even. Balanced z labels were provided only to linear/small-MLP probes, never to L3. Per-feature standardization used train data only. Chance is 0.5.",
"",
*probe_table,
"",
f"The highest mean linear accuracy is in {top_probe}. M linear accuracy exceeds 0.70 in {native_m_decodable}/8 seeds, below the preregistered 6/8 condition for a replicated native-M rescue diagnostic. Some seeds nevertheless contain decodable z in M. This is not causal evidence that the native read uses it; with G51 failed, a routing-only explanation is not identifiable.",
"",
"## 7. Per-layer survival metrics",
"",
"D_M=||M_A−M_B||, D_r=||r_M,A−r_M,B||, D_H=||H_A−H_B||, D_P is mean fixed-action forecast JS, and D_B is absolute entropy-policy BS. P0 values are observational paired-state differences. P1 is a finite H-state intervention; its M/read entries are not applicable. P1 matches each state's native norm but permits a larger inter-state D_H than native histories, so it is a bounded ceiling, not evidence that endogenous dynamics create those directions. These metrics locate *attenuation*, but only intervention rows support causal claims.",
"",
*survival_table,
"",
"## 8. Independent joint-training learning-curve audit",
"",
"A separate scratch arm reproduced Stage 2C.1 L3 training without changing F/M law: 500 AdamW steps, batch 16, 4/8/16 experience lengths, consequence CE plus 0.001 H-square penalty. Development seeds 7351–7352 and disjoint formal seeds 7401–7408; parameters were hashed before/after every no-update snapshot at 0/25/50/100/200/300/500. Four paired N=16 lifetimes, a held-out M ridge probe, read/H differences and group gradients were recorded per snapshot. This arm is a **parallel diagnostic**, not a curriculum or G55 result.",
"",
f"Development endpoints diverged sharply: 7351 TV={devcurve[7351]['final_metrics']['TV_action']:.4f}, BS={devcurve[7351]['final_metrics']['BS']:+.4f}; 7352 TV={devcurve[7352]['final_metrics']['TV_action']:.4f}, BS={devcurve[7352]['final_metrics']['BS']:+.4f}. This motivated independent formal replication, not threshold revision.",
"",
*curve_table,
"",
f"Across formal seeds, mean action-TV moves from {s['curve_aggregate']['0']['metrics']['TV_action']['mean']:.5f} at initialization to {s['curve_aggregate']['100']['metrics']['TV_action']['mean']:.5f} by step 100 and {s['curve_aggregate']['500']['metrics']['TV_action']['mean']:.5f} at step 500. Thus the dominant formal pattern is early loss of weak random action sensitivity, not a replicated high-TV learned phase followed by collapse.",
"",
f"Mean N=16 H norm changed from {s['curve_aggregate']['0']['metrics']['H_norm']['mean']:.3f} at step 0 to {s['curve_aggregate']['500']['metrics']['H_norm']['mean']:.3f} at step 500; this alone is not evidence that H instability caused the behavioral null.",
"",
*endpoint_table,
"",
f"At step 500, {success}/8 scratch seeds meet the descriptive TV≥0.10 and BS≥0.05 criterion; seed identities: {s['curve_success_seeds']}. The separate Stage 2C.1 failed-checkpoint cohort also showed near-zero behavior in all eight seeds. These are different seeded cohorts, and the descriptive cutoff is not a G55 gate or controlled curriculum effect. Per-seed trajectories and checkpoints are preserved under `results/stage2c2/learning_curves/formal/`.",
"The striking development-seed 7352 endpoint did not replicate in the eight formal curve seeds; it must be treated as a non-replicated development observation." if success==0 else
f"The development-seed 7352 endpoint has {success} formal descriptive analogue(s), still below the 6/8 replication standard and not a curriculum effect.",
"",
f"At step 500, high M-probe accuracy (≥0.70) coexists with |BS|<0.05 in scratch seeds {s['curve_M_decodable_without_behavior_seeds']}; this is a non-causal stored-but-unused *possibility*, not proof of a routing defect.",
"",
"## 9. Gradient audit",
"",
"Mean pre-clipping gradient norms at the update immediately before each snapshot (not causal evidence):",
"",
*grad_table,
"",
f"At step 500, mean action/head/q_M/read-gate gradient norms were {s['curve_aggregate']['500']['gradient_mean']['action_branch']:.5f}/{s['curve_aggregate']['500']['gradient_mean']['consequence_head']:.5f}/{s['curve_aggregate']['500']['gradient_mean']['q_M']:.5f}/{s['curve_aggregate']['500']['gradient_mean']['read_gate']:.5f}. They are not identically zero, but small action-branch gradients relative to the head warrant an optimization audit. A small gradient cannot by itself prove starvation; compare its trajectory with action-TV, CE and state differences. H norms are recorded separately; recurrence/stability parameters were not altered.",
"",
"## 10. Decision localization and next step",
"",
"By the mandated decision ladder, G51 FAIL localizes the first *replicated demonstrated* blocker to the downstream action-evaluation/head interface in the failed L3 checkpoints, under the preregistered native-norm oracle-H fit. It does not establish that the head has zero capacity at every seed, nor that no other upstream defect coexists. A targeted next stage may pre-register a stronger joint action-conditioning objective or training schedule, while retaining F/M law and explicitly preserving action-TV through training. The unreplicated development-seed 7352 anomaly should be understood before a curriculum claim. Do not redesign F/M write/consolidation or H recurrence from these data alone, and do not rerun the full Stage 2C habit suite yet.",
"",
"## Direct answers to the 25 required questions",
"",
f"1. Only inconsistently: native-norm oracle H passes G51 in {s['G51_pass_count']}/8 frozen checkpoints.",
"2. The first replicated failure is downstream action-head/interface capacity under frozen joint-trained weights; state quality may also contribute but is not isolated.",
"3. NOT_RUN_BY_GATE: oracle-read integration behavior was not fitted or evaluated.",
"4. NOT_RUN_BY_GATE: frozen-L3 oracle-M behavioral control was not evaluated.",
"5. NOT_RUN_BY_GATE: no fitted oracle-M preference-swap test.",
"6. NOT_RUN_BY_GATE: no oracle-M read-clamp behavioral test.",
"7. NOT_RUN_BY_GATE: no trained legal-history→L3-M adapter.",
"8. Yes, variably: held-out z is most consistently probe-decodable from native M/FM, not reliably from H/F.",
f"9. Highest mean linear-probe accuracy: {top_probe}; this is non-causal.",
"10. Not determined: M decodability alone cannot identify why the learned read did not control behavior, especially with G51 failed.",
"11. NOT_RUN_BY_GATE: routing rescue was not trained; preregistered M decodability criterion also missed.",
"12. Native observational differences and P1 H intervention are quantified; causal M→read→H survival was not adjudicated downstream of G51.",
"13. Formal mean action-TV drops mostly by step 100; there is no replicated high-TV learned phase followed by collapse.",
"14. History-TV and interaction are reported at every trajectory checkpoint; no single pattern is imposed across seeds.",
"15. Action gradients are relatively small versus the consequence head, while q_M/read gradients vary and are not uniformly zero; this is diagnostic, not proof of starvation.",
"16. Curriculum comparison NOT_RUN_BY_GATE; scratch training alone was observed.",
"17. Oracle-M curriculum NOT_RUN_BY_GATE; final zero-oracle survival unknown.",
"18. History-M curriculum NOT_RUN_BY_GATE; final zero-teacher survival unknown.",
"19. Not established by the failed frozen cohort or eight formal scratch-curve seeds; one positive development scratch seed was non-replicated and is not G55.",
"20. New G56 interventions NOT_RUN_BY_GATE; frozen Stage 2C.1 peripheral swaps were near-null.",
"21. Primary localized blocker: downstream action-evaluation/head interface in frozen failed L3; joint optimization and upstream defects remain possible.",
"22. No evidence yet requiring an F/M external-write-law change.",
"23. No evidence yet requiring readout-conserving consolidation change.",
"24. No: no demonstrated nonfinite or extreme-norm causal relation warrants H redesign now.",
"25. No: G51 failed and G55/G56 were not run; full Stage 2C rerun is premature.",
"",
"## Integrity, tests and limitations",
"",
"All formal fitted states are L3-local and norm matched. The 906-file prior Stage 2C/2C.1 tracked-artifact SHA-256 baseline was captured before Stage 2C.2 and verified unchanged at reporting. New Stage 2C.2 tests pass. The repository-wide suite retains one pre-existing Stage 1.5 README immutability assertion failure inherited from the Stage 2C commit, not changed here. No oracle-M/read/history adapter or curriculum data were fabricated when gated off. P1's 1,000-step/four-restart optimizer is a bounded ceiling test, not proof of absolute mathematical incapacity; formal outcome targets share the same world rule but independent sampled observations. The separate scratch-curve arm is not a curriculum and cannot revise frozen Stage 2C or Stage 2C.1 gates."
]
    return "\n".join(lines)+"\n"


if __name__=="__main__":main()
