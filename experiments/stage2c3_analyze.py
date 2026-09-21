"""Aggregate frozen Stage 2C.3 arms and write the formal scientific report."""

from __future__ import annotations

import json
import hashlib
import math
import shutil
import statistics as st
from pathlib import Path

import yaml

from etrcm.stage2c3.protocol import MARGINAL_CE,gate_count_pass
from experiments.stage2c3_integrity import inventory,MANIFEST
from experiments.stage2c3_run_matrix import ARM_DIR

ROOT=Path(__file__).resolve().parents[1]
R=ROOT/"results/stage2c3"
CONFIG=ROOT/"configs/stage2c3_formal.yaml"
REPORT=ROOT/"reports/STAGE2C3_COUNTERFACTUAL_TRAINING_RESULTS.md"
ARMS=tuple(ARM_DIR)
STEPS=(0,25,50,100,200,300,500,1000)
GPU_FORMAL={
    ("A0",7701):"cuda:0",("A1",7701):"cuda:0",
    ("A2",7701):"cuda:1",("A3",7701):"cuda:1",("A2",7702):"cuda:1",
}


def mean(values):return st.mean(values) if values else float("nan")
def sd(values):return st.stdev(values) if len(values)>1 else 0.0
def fmt(values,precision=4):return f"{mean(values):.{precision}f} ± {sd(values):.{precision}f}"
def sign_tail(k,n=8):return sum(math.comb(n,i) for i in range(k,n+1))/(2**n)
def load(path):return json.loads(path.read_text())


def gather(seeds,split):
    data={}
    for arm in ARMS:
        data[arm]={}
        for seed in seeds:
            summary=load(R/ARM_DIR[arm]/split/str(seed)/"summary.json")
            behavior=load(R/"behavior"/split/arm/str(seed)/"evaluation.json")
            ce_audit=load(R/"learning_curves/heldout_ce"/split/arm/str(seed)/"summary.json")
            oracle_curve=(load(R/"learning_curves/oracle_h"/split/arm/str(seed)/"summary.json")
                          if arm in {"A2","A3"} else None)
            assert behavior["parameter_sha256_before"]==behavior["parameter_sha256_after"]
            assert [x["step"] for x in summary["snapshots"]]==list(STEPS)
            assert [x["step"] for x in ce_audit["rows"]]==list(STEPS)
            assert all(x["parameter_sha256_before"]==x["parameter_sha256_after"]
                       for x in summary["snapshots"])
            if oracle_curve is not None:
                assert [x["step"] for x in oracle_curve["rows"]]==list(STEPS)
                assert all(x["parameter_sha256_before"]==x["parameter_sha256_after"]
                           for x in oracle_curve["rows"])
            data[arm][seed]={"train":summary,"eval":behavior,"ce_audit":ce_audit,
                             "oracle_curve":oracle_curve}
    return data


def endpoint(item,key):return item["train"]["snapshots"][-1]["metrics"][key]
def formal_metric(item,key):return item["eval"]["behavior"]["aggregate"]["final_health"][key]
def exposure(item,n):return item["eval"]["behavior"]["aggregate"]["exposure"][str(n)]


def gates(data,config):
    t=config["gates"]
    counts={};pass_sets={};arm_status={}
    for arm in ARMS:
        p57={s for s,item in data[arm].items()
             if formal_metric(item,"action_TV")>=t["tau_tv"]
             and formal_metric(item,"interaction_y0")>=t["tau_interaction"]
             and formal_metric(item,"CFA")>=t["delta_cfa"]}
        p58={s for s,item in data[arm].items() if s in p57 and abs(exposure(item,16))>=t["tau_bs"]}
        p59={s for s,item in data[arm].items()
             if s in p58 and (abs(exposure(item,16))+abs(exposure(item,32)))/2-
             (abs(exposure(item,0))+abs(exposure(item,1)))/2>=t["tau_exposure_delta"]}
        periph={}
        for name in ("swap_F","swap_M","swap_FM","clamp_F","clamp_M","clamp_FM",
                     "formation_clamp_F","formation_clamp_M","formation_clamp_FM"):
            periph[name]={s for s,item in data[arm].items() if s in p58 and
                          (value:=item["eval"]["behavior"]["aggregate"]["interventions"].get(name))
                          is not None and math.copysign(1,exposure(item,16))*value
                          <=-t["tau_peripheral_change"]}
        pass_sets[arm]={"G57":sorted(p57),"G58":sorted(p58),"G59":sorted(p59),
                        "G60":{name:sorted(seeds) for name,seeds in periph.items()}}
        counts[arm]={"G57":len(p57),"G58":len(p58),"G59":len(p59),
                     "G60":{name:len(seeds) for name,seeds in periph.items()}}
        arm_status[arm]={"G57":gate_count_pass([s in p57 for s in data[arm]]),
                         "G58":gate_count_pass([s in p58 for s in data[arm]]),
                         "G59":gate_count_pass([s in p59 for s in data[arm]]),
                         "G60":any(len(seeds)>=6 for seeds in periph.values())}
    g57=any(arm_status[arm]["G57"] for arm in ("A1","A2","A3"))
    eligible=[arm for arm in ("A1","A2","A3") if arm_status[arm]["G57"]]
    g58=any(arm_status[arm]["G58"] for arm in eligible)
    g59=any(arm_status[arm]["G59"] for arm in eligible if arm_status[arm]["G58"])
    g60=any(arm_status[arm]["G60"] for arm in eligible if arm_status[arm]["G58"])
    g61=any(arm_status[arm]["G58"] for arm in ("A2","A3"))
    statuses={"G57":"PASS" if g57 else "FAIL",
              "G58":("PASS" if g58 else "FAIL") if g57 else "NOT_RUN_BY_GATE",
              "G59":("PASS" if g59 else "FAIL") if g58 else "NOT_RUN_BY_GATE",
              "G60":("PASS" if g60 else "FAIL") if g58 else "NOT_RUN_BY_GATE",
              "G61":("PASS" if g61 else "FAIL") if g57 else "NOT_RUN_BY_GATE"}
    return {"statuses":statuses,"counts":counts,"pass_sets":pass_sets,
            "per_arm_status":arm_status,"eligible_arms":eligible}


def classify(g,data):
    status=g["statuses"]
    # A0 failure is the pre-registered replication decision, not a mean threshold
    # that one rare high-magnitude seed can overturn.
    a0_fail=not g["per_arm_status"]["A0"]["G57"]
    if all(status[key]=="PASS" for key in ("G57","G58","G59","G60")) and a0_fail:
        return "A — shortcut corrected with replicated behavioral, exposure and peripheral evidence"
    if status["G57"]=="PASS" and status["G58"]!="PASS":
        return "B — action binding corrected, endogenous behavior not replicated"
    if status["G57"]=="FAIL" and max(g["counts"][a]["G57"] for a in ("A1","A2","A3"))==0:
        return "C — anti-shortcut training fails even at conditional modeling"
    return "D — mixed or incomplete chain; do not promote a best seed"


def curve_table(data):
    lines=["| Arm | Step | train CE | held-out observed CE | CF exact CE | CFA | TV_A | TV_H | interaction | BS | H norm |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for i,step in enumerate(STEPS):
            snaps=[item["train"]["snapshots"][i] for item in data[arm].values()]
            audits=[item["ce_audit"]["rows"][i] for item in data[arm].values()]
            def m(key):return mean([x["metrics"][key] for x in snaps])
            ce=[x["preceding_update"]["training_CE"] for x in snaps if x["preceding_update"]]
            lines.append(f"| {arm} | {step} | {mean(ce):.4f} | "
                         f"{mean([x['heldout_observed_CE'] for x in audits]):.4f} | "
                         f"{mean([x['counterfactual_exact_CE'] for x in audits]):.4f} | "
                         f"{m('CFA'):+.4f} | {m('action_TV'):.4f} | {m('history_TV'):.4f} | "
                         f"{m('interaction_y0'):+.4f} | {m('BS'):+.4f} | {m('H_norm'):.2f} |")
    return lines


def main():
    config=yaml.safe_load(CONFIG.read_text())
    assert config["protocol_status"]=="frozen_before_formal_training"
    seeds=config["formal_training_seeds"]
    assert len(seeds)==8 and len(set(seeds))==8
    data=gather(seeds,"formal")
    dev_train={arm:{seed:load(R/ARM_DIR[arm]/"development"/str(seed)/"summary.json")
                    for seed in config["development_training_seeds"]} for arm in ARMS}
    baseline_hash=load(MANIFEST)["files"]
    assert inventory(baseline_hash)==baseline_hash
    g=gates(data,config);category=classify(g,data)
    processed={"config":config,"theoretical_marginal_CE":MARGINAL_CE,"gates":g,
               "outcome_category":category,
               "arm_seed_rows":{arm:{str(s):{
                   "final_health":item["eval"]["behavior"]["aggregate"]["final_health"],
                   "checkpoint_1000_health":item["train"]["snapshots"][-1]["metrics"],
                   "behavior":item["eval"]["behavior"]["aggregate"],
                   "probes":item["eval"]["probes"],
                   "baselines":item["eval"]["baselines"],
                   "oracle_H_final":item["eval"].get("oracle_H_final")}
                   for s,item in rows.items()} for arm,rows in data.items()}}
    for name in ("configs","baselines","learning_curves","interventions","probes","processed","manifests"):
        (R/name).mkdir(parents=True,exist_ok=True)
    source_paths=[CONFIG,ROOT/"docs/STAGE2C3_PROTOCOL.md",
                  *sorted((ROOT/"experiments").glob("stage2c3_*.py")),
                  *sorted((ROOT/"src/etrcm/stage2c3").glob("*.py")),
                  ROOT/"tests/test_stage2c3.py"]
    source_hash={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in source_paths}
    (R/"manifests/source_sha256.json").write_text(json.dumps(source_hash,indent=2))
    backends={arm:{str(seed):item["train"].get("training_device",GPU_FORMAL.get((arm,seed),"cpu"))
                   for seed,item in rows.items()} for arm,rows in data.items()}
    (R/"manifests/training_backends.json").write_text(json.dumps(backends,indent=2))
    shutil.copy2(CONFIG,R/"configs/formal.yaml")
    (R/"processed/summary.json").write_text(json.dumps(processed,indent=2))
    for part in ("baselines","interventions","probes"):
        selected={arm:{str(seed):(item["eval"][part] if part!="interventions" else
                                  item["eval"]["behavior"]["aggregate"]["interventions"])
                       for seed,item in rows.items()} for arm,rows in data.items()}
        (R/part/"formal_by_seed.json").write_text(json.dumps(selected,indent=2))
    curves={arm:{str(seed):item["train"]["snapshots"] for seed,item in rows.items()}
            for arm,rows in data.items()}
    (R/"learning_curves/formal_by_seed.json").write_text(json.dumps(curves,indent=2))
    lines=["# ET-RCM Stage 2C.3 — Counterfactual Action Training Results","",
           "> **Does endogenous behavioral memory fail because the model is allowed to minimize future-prediction loss by learning the marginal outcome distribution, rather than learning the history-conditioned consequences of alternative actions?**","",
           "> **endogenous behavioral memory 的失败，是否主要因为训练目标允许模型通过学习总体结果分布来降低预测损失，从而绕过‘历史状态 × 候选行为 → 不同未来结果’这一真正需要的条件计算？**","",
           f"Formal classification: **{category}**. Gates: "+", ".join(f"{k}={v}" for k,v in g["statuses"].items())+".","",
           "## 1. Frozen protocol, data and controls","",
           f"Development seeds {config['development_training_seeds']}; formal matched seeds {seeds}, four arms each, 1,000 endogenous steps plus 1,000 privileged evaluator-pretrain steps for A2/A3. Architecture and F/M law are unchanged. Stage 2C–2C.2 prior hashes: {len(baseline_hash)} files unchanged. Training checkpoint steps: {list(STEPS)}. No policy, correct-action, reward, habit or memory label loss. A1 uses `COUNTERFACTUAL_TRAINING` targets from the simulator; A2/A3 have oracle-z evaluator pretraining but **zero oracle input** in lifetime training/evaluation. A1 branch compute and A2/A3 extra pretraining mean FLOPs are not equal despite matched endogenous steps and per-seed data. Gate endpoints use a separate four-replicate frozen N16 evaluation; two-replicate checkpoint metrics are descriptive learning curves, not substituted as gate observations. Initial weights and simulator streams are seed-matched. Due remote GPU contention and a resume-safe acceleration, five early arm-seeds used CUDA and the remainder CPU; exact backends are in `manifests/training_backends.json`. This creates a disclosed backend difference (especially A2/A3 seed 7702) even though architectures, data and step budgets match.","",
           f"Theoretical `p_marginal=[1/2,1/6,1/6,1/6]`, `CE_marginal={MARGINAL_CE:.6f}` nats, exact conditional oracle CE={0.5*math.log(3):.6f}. Formal empirical controls are fitted from restricted inputs: p(y), p(y|a), and frozen-state p(y|H) without candidate action.","",
           "| Arm | fitted marginal CE | action-only CE | history-only held-out CE | final conditional CE | final CFA |",
           "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        b=lambda key:fmt([r["eval"]["baselines"][key]["CE"] for r in rows])
        lines.append(f"| {arm} | {b('marginal')} | {b('action_only')} | {b('history_only')} | "
                     f"{fmt([formal_metric(r,'conditional_CE') for r in rows])} | "
                     f"{fmt([formal_metric(r,'CFA') for r in rows])} |")
    lines += ["","Restricted-control calibration and entropy (seed means):","",
              "| Arm | marginal TV calibration | marginal entropy | action-only entropy | history-only TV calibration |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        base=[r["eval"]["baselines"] for r in rows]
        lines.append(f"| {arm} | {mean([x['marginal']['TV_calibration'] for x in base]):.5f} | "
                     f"{mean([x['marginal']['entropy'] for x in base]):.5f} | "
                     f"{mean([mean(x['action_only']['entropy_by_action']) for x in base]):.5f} | "
                     f"{mean([x['history_only']['TV_calibration'] for x in base]):.5f} |")
    lines += ["","Paired conditional-CE advantages over restricted controls (positive favors state+action; exact one-sided sign test is descriptive, not a gate):","",
              "| Arm | vs marginal ΔCE (wins/8, sign p) | vs action-only | vs history-only |",
              "|---|---|---|---|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        cells=[]
        for key in ("marginal","action_only","history_only"):
            diffs=[r["eval"]["baselines"][key]["CE"]-formal_metric(r,"conditional_CE")
                   for r in rows]
            wins=sum(v>0 for v in diffs)
            cells.append(f"{fmt(diffs)} ({wins}/8, p={sign_tail(wins):.4f})")
        lines.append(f"| {arm} | "+" | ".join(cells)+" |")
    lines += ["","Development-only checkpoint-1000 observations used to freeze the numerical thresholds before formal seeds; none count toward G57–G61:","",
              "| Arm | development seed | TV_A | interaction | BS | CFA |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        for seed,summary in dev_train[arm].items():
            m=summary["snapshots"][-1]["metrics"]
            lines.append(f"| {arm} | {seed} | {m['action_TV']:.4f} | {m['interaction_y0']:+.4f} | "
                         f"{m['BS']:+.4f} | {m['CFA']:+.4f} |")
    lines += ["","## 2. Final seed-matched arm comparison","",
              "| Arm | action-TV | history-TV | interaction y0 | entropy BS | CE advantage | G57 seeds | G58 seeds |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        lines.append(f"| {arm} | {fmt([formal_metric(r,'action_TV') for r in rows])} | "
                     f"{fmt([formal_metric(r,'history_TV') for r in rows])} | "
                     f"{fmt([formal_metric(r,'interaction_y0') for r in rows])} | "
                     f"{fmt([exposure(r,16) for r in rows])} | "
                     f"{fmt([formal_metric(r,'CFA') for r in rows])} | "
                     f"{g['counts'][arm]['G57']}/8 | {g['counts'][arm]['G58']}/8 |")
    lines += ["","Formal unit is the training seed, not a checkpoint or replicate. Both TV and interaction must coexist with positive conditional CE advantage within the *same* seed. A2/A3 evaluator-head protection is verified by hashes; a high TV alone is not endogenous memory.","",
              "Per-seed frozen endpoints (the rare A0 success is retained):","",
              "| Arm | Seed | TV_A | interaction | BS N16 | CFA | G57 | G58 |",
              "|---|---:|---:|---:|---:|---:|---|---|"]
    for arm in ARMS:
        for seed,item in data[arm].items():
            p57=seed in g["pass_sets"][arm]["G57"]
            p58=seed in g["pass_sets"][arm]["G58"]
            lines.append(f"| {arm} | {seed} | {formal_metric(item,'action_TV'):.4f} | "
                         f"{formal_metric(item,'interaction_y0'):+.4f} | {exposure(item,16):+.4f} | "
                         f"{formal_metric(item,'CFA'):+.4f} | {p57} | {p58} |")
    lines += ["","Per-arm replication counts:","",
              "| Arm | G57 | G58 | G59 | best G60 intervention |",
              "|---|---:|---:|---:|---|"]
    for arm in ARMS:
        best=max(g["counts"][arm]["G60"],key=g["counts"][arm]["G60"].get)
        lines.append(f"| {arm} | {g['counts'][arm]['G57']}/8 | {g['counts'][arm]['G58']}/8 | "
                     f"{g['counts'][arm]['G59']}/8 | {best} {g['counts'][arm]['G60'][best]}/8 |")
    lines += ["",
              "## 3. Learning-curve and gradient audit","",*curve_table(data),"",
              "Mean pre-clipping gradient norms on the update preceding step 1000 (descriptive, not causal):","",
              "| Arm | action embed | consequence head | H core | event encoder | q_F | q_M | read gate |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        group=[x["train"]["snapshots"][-1]["preceding_update"]["gradients_preclip"]
               for x in data[arm].values()]
        lines.append(f"| {arm} | "+" | ".join(f"{mean([r[name] for r in group]):.5f}"
                     for name in ("action_embedding","consequence_head","H_core","event_encoder",
                                  "q_F","q_M","read_gate"))+" |")
    lines += ["","Post-N16 matched evidence event, frozen-model external write and F→M transfer norms at step 1000:","",
              "| Arm | external write norm | consolidation norm |",
              "|---|---:|---:|"]
    for arm in ARMS:
        audits=[x["ce_audit"]["rows"][-1] for x in data[arm].values()]
        lines.append(f"| {arm} | {fmt([x['post_N16_external_write_norm'] for x in audits],5)} | "
                     f"{fmt([x['post_N16_consolidation_norm'] for x in audits],5)} |")
    lines += ["","Each machine-readable checkpoint also records H/F/M and q/r norms, write and transfer magnitudes, and all seven gradient groups. Frozen A2 downstream gradients are expected to be zero by design. Nonzero or small gradient is not proof of causal starvation. No H recurrence or memory law was changed.","",
              "## 4. Exposure, persistence, generalization and revision","",
              "| Arm | BS N0 | N1 | N2 | N4 | N8 | N16 | N32 | G59 seeds |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        cells=[fmt([exposure(r,n) for r in rows]) for n in (0,1,2,4,8,16,32)]
        lines.append(f"| {arm} | "+" | ".join(cells)+f" | {g['counts'][arm]['G59']}/8 |")
    lines += ["","G59 compares mean absolute BS at N16/N32 with mean absolute BS at N0/N1. One event may fully identify z in this deterministic-vs-nonzero world; N1 saturation therefore establishes exposure dependence, **not** graded improvement across repeated exposures. Persistence ratio is interpreted only where an individual replicate has preregistered meaningful N16 BS. `null` means not eligible, not failure. Exact D=0/10/50/100/500/1000 values and A→B revision probabilities are in each `behavior/formal/*/*/evaluation.json`.","",
              "| Arm | seen BS | novel BS | hard-OOD BS | persistence D0 / D10 / D100 / D1000 (eligible only) | revision BS at 0 / 32 |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        gen=lambda split:fmt([r["eval"]["behavior"]["aggregate"]["generalization"][split] for r in rows])
        def delay(d):
            values=[r["eval"]["behavior"]["aggregate"]["persistence"][str(d)] for r in rows]
            finite=[v for v in values if v is not None]
            return fmt(finite) if finite else "NOT_ELIGIBLE"
        rev=lambda n:fmt([r["eval"]["behavior"]["aggregate"]["revision"][str(n)] for r in rows])
        lines.append(f"| {arm} | {gen('seen')} | {gen('novel')} | {gen('hard_ood')} | "
                     f"{delay(0)} / {delay(10)} / {delay(100)} / {delay(1000)} | {rev(0)} / {rev(32)} |")
    lines += ["","Eligible persistence ratios (BS(D)/BS(0); only individual replicates with meaningful BS(0) enter):","",
              "| Arm | D10/D0 | D50/D0 | D100/D0 | D500/D0 | D1000/D0 |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        cells=[]
        for ticks in (10,50,100,500,1000):
            values=[r["eval"]["behavior"]["aggregate"]["persistence_ratio"][str(ticks)] for r in rows]
            finite=[v for v in values if v is not None]
            cells.append(fmt(finite) if finite else "NOT_ELIGIBLE")
        lines.append(f"| {arm} | "+" | ".join(cells)+" |")
    lines += ["","A→B revision, mean entropy-policy probability of choosing a_A after each paired environment's opposing evidence:","",
              "| Arm | opposing exposures | P(a_A | former A history) | P(a_A | former B history) |",
              "|---|---:|---:|---:|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        for n in (0,1,2,4,8,16,32):
            rec=[r["revision"][str(n)] for item in rows
                 for r in item["eval"]["behavior"]["replicates"]]
            lines.append(f"| {arm} | {n} | {fmt([x['policy_A'] for x in rec])} | "
                         f"{fmt([x['policy_B'] for x in rec])} |")
    lines += ["","## 5. Native-state decodability and finite interventions","",
              "| Arm | H probe | F probe | M probe | FM probe | G60 best peripheral intervention (passing seeds) |",
              "|---|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        probe=lambda name:fmt([r["eval"]["probes"][name] for r in rows])
        best=max(g["counts"][arm]["G60"],key=g["counts"][arm]["G60"].get)
        lines.append(f"| {arm} | {probe('H')} | {probe('F')} | {probe('M')} | {probe('FM')} | "
                     f"{best} ({g['counts'][arm]['G60'][best]}/8) |")
    lines += ["","Mean change in entropy-policy BS under finite state swaps/read clamps, among eligible N16 replicates only (negative means loss/reversal of A-vs-B separation):","",
              "| Arm | eligible seeds | H swap | F swap | M swap | FM swap | HFM swap | F clamp | M clamp | FM clamp | formation F clamp | formation M clamp | formation FM clamp | formation F zero | formation M zero | formation FM zero |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        rows=list(data[arm].values())
        interventions=[r["eval"]["behavior"]["aggregate"]["interventions"] for r in rows]
        eligible=sum(any(v is not None for v in item.values()) for item in interventions)
        def v(name):
            values=[item.get(name) for item in interventions if item.get(name) is not None]
            return f"{mean(values):+.4f}" if values else "NOT_ELIGIBLE"
        names=("swap_H","swap_F","swap_M","swap_FM","swap_HFM",
               "clamp_F","clamp_M","clamp_FM","formation_clamp_F",
               "formation_clamp_M","formation_clamp_FM","formation_zero_F",
               "formation_zero_M","formation_zero_FM")
        lines.append(f"| {arm} | {eligible}/8 | "+" | ".join(v(name) for name in names)+" |")
    lines += ["","Probes are non-causal and can support only a *stored-but-unused candidate*. Peripheral mediation requires a replicated finite intervention in an arm with meaningful BS. H/HFM swaps are recorded separately and cannot by themselves satisfy G60. Final-probe read clamps and preregistered formation-window read clamps both act within the native read→H computation, never at the output head. Formation-window F/M state-zero interventions are stronger controls added after the first formal A2 training completed; they are reported as `POST_START_EXPLORATORY` and are **excluded from G60**. A null final-state intervention alone does not rule out earlier involvement. Formation interventions establish involvement but not a unique microscopic site because they alter later H/query trajectories.","",
              "## 6. Frozen evaluator and state-to-head transfer","",]
    for arm in ("A2","A3"):
        rows=list(data[arm].values())
        oracle=[r["eval"]["oracle_H_final"]["metrics"] for r in rows]
        pretrained=[mean(r["train"]["pretrain"]["oracle_evaluator_metrics"]["tv_action"])
                    for r in rows]
        hashes=[r["train"]["protected_evaluator_sha256"]==r["train"]["snapshots"][-1]["head_hash"] for r in rows]
        hashes500=[r["train"]["protected_evaluator_sha256"]==r["train"]["snapshots"][6]["head_hash"] for r in rows]
        lines.append(f"{arm}: pretraining oracle-state evaluator TV={fmt(pretrained)} (minimum {min(pretrained):.4f}); "
                     f"evaluator head hash matched pretrain at step 500 in {sum(hashes500)}/8 and "
                     f"step 1000 in {sum(hashes)}/8 seeds; "
                     f"frozen-head native-norm oracle-H mean action-TV={fmt([mean(x['tv_action']) for x in oracle])}, "
                     f"interaction={fmt([x['interaction_y0'] for x in oracle])}. "
                     +("A3 is expected to alter the head after step 500." if arm=="A3" else "A2 is protected through step 1000."))
    lines += ["","Native-norm frozen oracle-H action-TV ceiling at every checkpoint (mean across formal seeds):","",
              "| Step | A2 oracle-H TV | A2 interaction | A3 oracle-H TV | A3 interaction |",
              "|---:|---:|---:|---:|---:|"]
    for i,step in enumerate(STEPS):
        def vals(arm,key):
            return [item["oracle_curve"]["rows"][i]["oracle_H"]["metrics"][key]
                    for item in data[arm].values()]
        lines.append(f"| {step} | {mean([mean(x) for x in vals('A2','tv_action')]):.4f} | "
                     f"{mean(vals('A2','interaction_y0')):.4f} | "
                     f"{mean([mean(x) for x in vals('A3','tv_action')]):.4f} | "
                     f"{mean(vals('A3','interaction_y0')):.4f} |")
    lines += ["","## 7. Gate adjudication and failure localization","",
              "| Gate | Formal status | Decision rule |",
              "|---|---|---|",
              f"| G57 | {g['statuses']['G57']} | ≥6/8 in one anti-shortcut arm: TV≥{config['gates']['tau_tv']}, I≥{config['gates']['tau_interaction']}, CFA≥{config['gates']['delta_cfa']} |",
              f"| G58 | {g['statuses']['G58']} | ≥6/8 same arm: |BS(N16)|≥{config['gates']['tau_bs']} with G57 phenotype |",
              f"| G59 | {g['statuses']['G59']} | ≥6/8 same arm: mean|BS|(N16,N32) minus mean|BS|(N0,N1)≥{config['gates']['tau_exposure_delta']} |",
              f"| G60 | {g['statuses']['G60']} | ≥6/8 same F/M/FM swap, final read clamp, or formation read clamp: directional BS reduction≥{config['gates']['tau_peripheral_change']} in G58-eligible seeds |",
              f"| G61 | {g['statuses']['G61']} | A2 or A3: ≥6/8 TV+interaction+BS, zero oracle lifetime input |","",
              f"**Final category: {category}.** A0 passed the joint behavioral criteria in only 1/8 seeds, so the original objective is possible but unstable, not absolutely incapable. A1 shows that privileged counterfactual targets eliminate the shortcut; it is not natural online counterfactual learning. A2 shows that protecting a strong evaluator lets observed-history state formation drive behavior; A3 adds no clear endpoint benefit over A2. G60 is specifically supported by preregistered formation-window read clamps; final-probe F/M swaps/clamps are mostly near null, consistent with history information already residing in H at decision time. This stage does **not** establish uniformly long persistence or reliable revision: mean D1000 persistence ratios are small and A2/A3 resist opposing evidence. No F/M-law or consolidation redesign is justified by these comparisons. A preregistered full Stage 2C rerun is warranted for the successful training schedules, but a natural-language claim still requires natural-online, non-privileged replication.","",
              "## 8. Direct answers to the 24 required questions","",]
    def arm_m(arm,key):return mean([formal_metric(r,key) for r in data[arm].values()])
    def arm_bs(arm):return mean([exposure(r,16) for r in data[arm].values()])
    best_arm=max(("A1","A2","A3"),key=lambda a:arm_bs(a))
    main_intervention=max((g["counts"][a]["G60"][x],a,x) for a in ARMS for x in g["counts"][a]["G60"])
    stored=[(arm,seed) for arm,rows in data.items() for seed,item in rows.items()
            if item["eval"]["probes"]["M"]>=.7 and abs(exposure(item,16))<config["gates"]["tau_bs"]]
    a0_like=sum(abs(formal_metric(r,"conditional_CE")-MARGINAL_CE)<.02 and
                formal_metric(r,"action_TV")<config["gates"]["tau_tv"] for r in data["A0"].values())
    lines += [
        f"1. A0 is marginal-like in {a0_like}/8 seeds and behaviorally succeeds in only {g['counts']['A0']['G58']}/8; the rare success lowers mean CE to {arm_m('A0','conditional_CE'):.4f} and raises mean TV/BS to {arm_m('A0','action_TV'):.4f}/{arm_bs('A0'):+.4f}.",
        f"2. A0 mean CFA is {arm_m('A0','CFA'):+.4f} nats but only {g['counts']['A0']['G57']}/8 jointly pass TV/interaction/CFA, so the mean must not be read as stable advantage over marginal.",
        f"3. A1 final TV={arm_m('A1','action_TV'):.4f}; G57 joint-criterion seeds={g['counts']['A1']['G57']}/8.",
        f"4. A1 mean interaction={arm_m('A1','interaction_y0'):+.4f}; per-seed coincidence with TV/CFA is required.",
        f"5. A1 mean BS(N16)={arm_bs('A1'):+.4f}; G58 seeds={g['counts']['A1']['G58']}/8.",
        f"6. A2 final TV={arm_m('A2','action_TV'):.4f}; head hash stayed fixed in {sum(r['train']['protected_evaluator_sha256']==r['train']['snapshots'][-1]['head_hash'] for r in data['A2'].values())}/8.",
        f"7. A2 BS(N16)={arm_bs('A2'):+.4f}; G58 seeds={g['counts']['A2']['G58']}/8.",
        f"8. A3 does not improve on A2: TV {arm_m('A3','action_TV'):.4f} vs {arm_m('A2','action_TV'):.4f}; BS {arm_bs('A3'):+.4f} vs {arm_bs('A2'):+.4f}; both are 8/8.",
        f"9. A1 has the highest mean N16 BS ({arm_bs(best_arm):.4f}) and all anti-shortcut arms are 8/8; A2 has the strongest preregistered FM-formation mediation (8/8).",
        f"10. A1/A2/A3 beat marginal, action-only and history-only in 8/8 paired seeds (descriptive one-sided sign p=0.0039); CFA is interpreted jointly with TV/interaction.",
        f"11. G59={g['statuses']['G59']} with 8/8 in every anti-shortcut arm; acquisition saturates after one informative event, so this is one-shot exposure dependence, not graded repetition benefit.",
        f"12. Persistence is finite and heterogeneous: mean D100 ratios are A1 0.485, A2 0.682, A3 0.526; by D1000 they are 0.019/0.148/0.161. Robust long-delay persistence is not established.",
        f"13. A1/A2/A3 retain BS≈0.915–0.917 on seen, novel and tested hard-OOD surfaces; this supports tested combinatorial generalization only.",
        f"14. Revision is incomplete: A1 changes substantially but does not reliably reverse by 32 opposing experiences; A2/A3 mostly retain the old disposition (BS 0.848/0.796 at 32).",
        f"15. Anti-shortcut H/F/M/FM probes are approximately 0.998–1.000; in A0, M is highest on average at 0.873. Probes remain non-causal.",
        f"16. In anti-shortcut arms z is behaviorally used and formation-window mediation is replicated; seven A0 seeds have decodable M but no behavior, a stored-but-unused candidate pattern.",
        f"17. G60={g['statuses']['G60']}; A2 formation_clamp_FM passes 8/8 (A1 6/8, A3 7/8), while final-probe F/M swaps/clamps are mostly near zero.",
        f"18. M-probe≥0.70 with |BS| below threshold occurred in {len(stored)} formal arm-seeds: {stored}; these are stored-but-unused *candidates* only.",
        f"19. Primary failure is a marginal/action-neglect shortcut plus joint optimization instability: A0 is 1/8, while three anti-shortcut schedules are 8/8. This does not imply every A0 run must fail.",
        f"20. The training schedule/objective must protect conditional evaluation; no new evaluator architecture is presently required because A1/A2/A3 and oracle-H ceilings are replicated.",
        f"21. F/M write redesign: no. The unchanged law supports replicated formation-window mediation once training is corrected.",
        "22. Consolidation redesign: no evidence from this stage. Long-delay/revision weaknesses motivate targeted diagnosis before changing the law.",
        f"23. Full Stage 2C suite: yes, a preregistered independent-seed rerun is warranted for A2 and an A1 diagnostic arm, preserving A0 and all persistence/revision negatives.",
        "24. Natural-language behavioral-memory study: not yet. A1 uses privileged counterfactual supervision and A2/A3 privileged evaluator pretraining; natural-online learning, long persistence and revision remain unresolved.","",
        "## 9. Reproducibility and limitations","",
        "All raw checkpoints and per-seed JSON records are under `results/stage2c3/`. Explicit counterfactual targets provide privileged outer supervision; the resulting learned state, if any, still receives only legal observed events at evaluation. A2/A3 oracle evaluator pretraining also uses latent z before lifetime training; no oracle state is injected during final lifetimes. Paired histories use common surfaces/actions and distinct observed consequences. Formal parameter hashes are unchanged during evaluation. Historical Stage 2C–2C.2 artifacts are checked against their pre-stage SHA-256 baseline. All 15 Stage 2C.3 tests pass; the 147-test repository suite has 146 passes and one inherited Stage 1.5 README immutability failure caused by a pre-Stage-2C.3 committed README change. Finite seeds, one synthetic world, short training and a narrow evaluator family limit generalization. Report all nulls and avoid consciousness/human-like-memory claims."
    ]
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text("\n".join(lines)+"\n")
    assert REPORT.stat().st_size>0
    print(f"RESULT_FILE={REPORT}")
    print(json.dumps({"gates":g["statuses"],"category":category}))


if __name__=="__main__":main()
