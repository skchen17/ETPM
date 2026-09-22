"""Run-level Stage 2D.6 gate adjudication; preserves all null results."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import Counter
from pathlib import Path


ROOT=Path("results/stage2d6")
STEPS=(0,25,50,100,200,300,500,750,1000,1500)


def load(folder):
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def mean(values):
    return statistics.mean(values) if values else None


def metric(row,key):
    f=row["endpoint"]["fusion"]
    return f["Q"][key[1:]] if key in ("Q1","Q2","Q4") else f[key]


def auc(healthy,failed,key):
    return mean([float(metric(h,key)>metric(f,key))+.5*float(metric(h,key)==metric(f,key))
                 for h in healthy for f in failed])


def bootstrap_difference(healthy,failed,key,seed):
    """Run-level, not neuron-level, independent resampling CI."""
    if not healthy or not failed:return None
    generator=random.Random(seed)
    h=[metric(r,key) for r in healthy];f=[metric(r,key) for r in failed]
    values=[]
    for _ in range(4000):
        values.append(mean([generator.choice(h) for _ in h])-
                      mean([generator.choice(f) for _ in f]))
    values.sort()
    return {"difference":mean(h)-mean(f),"bootstrap_95":[values[100],values[3899]]}


def summary(rows):
    result={"count":len(rows),"classes":dict(Counter(r["endpoint"]["class"] for r in rows))}
    for label in ("healthy","shortcut","partial"):
        group=[r for r in rows if r["endpoint"]["class"]==label]
        result[label]={key:mean([metric(r,key) for r in group]) for key in (
            "pre_I","post_I","delta_NL","state_main_norm","action_main_norm",
            "mean_abs_curvature","overlap","pre_mean","Q1","Q2","Q4","fusion_norm",
            "response_rank2_energy")}
        result[label]["IHA"]=mean([r["endpoint"]["fusion"]["behavior"]["IHA"] for r in group])
        result[label]["BS_abs"]=mean([abs(r["endpoint"]["fusion"]["behavior"]["BS"]) for r in group])
    return result


def unit_pass(row,k):
    u=row["endpoint"]["unit_causality"][str(k)]
    native=row["endpoint"]["fusion"]["behavior"]["IHA"]
    return u["IHA_loss"]>=.05 and u["IHA_loss"]>=.5*native and \
           u["IHA_loss"]-u["random_mean_IHA_loss"]>=.025


def rescue_pass(row):
    native=row["native"];changed=row["common_offset"];new=changed["behavior"]
    return changed["post_I"]-row["native_post_I"]>=.025 and new["TV"]>=.10 and \
        new["IHA"]>=.10 and abs(new["BS"])>=.10 and \
        new["IHA"]-native["IHA"]>=.025 and abs(new["BS"])-abs(native["BS"])>=.025 and \
        new["IHA"]>row["random_offset"]["behavior"]["IHA"]+.025 and \
        new["IHA"]>row["low_curvature_offset"]["behavior"]["IHA"]+.025 and \
        new["IHA"]>row["state_scale"]["IHA"]+.025 and \
        new["IHA"]>row["action_scale"]["IHA"]+.025


def destruction_pass(row):
    native=row["native"];destroy=row["healthy_destruction"]
    control=row["healthy_matched_shift"]
    return destroy["post_I"]<=.75*row["native_post_I"] and \
        destroy["behavior"]["IHA"]<=.5*native["IHA"] and \
        abs(destroy["behavior"]["BS"])<=.5*abs(native["BS"]) and \
        control["behavior"]["IHA"]-destroy["behavior"]["IHA"]>=.025


def onset(row):
    trace=row.get("trajectory",[])
    if not trace:return None
    initial=trace[0]["fusion"]["overlap"]
    final=trace[-1]["fusion"]["overlap"]
    midpoint=initial+.5*(final-initial)
    o=next((x["step"] for x in trace if x["fusion"]["overlap"]>=midpoint),None)
    b=next((x["step"] for x in trace if x["fusion"]["behavior"]["IHA"]>=.10),None)
    return {"O_half_onset":o,"IHA_threshold_onset":b,"O_rise":final-initial,
            "precedes":bool(o is not None and b is not None and o<=b and final>initial)}


def phenotype(row):
    if row["endpoint"]["class"]=="healthy":return "healthy"
    probes=row["endpoint"]["health"]["z_probe"]
    high=max(probes.get("H",0),probes.get("M",0))>=.75
    low_behavior=abs(row["endpoint"]["health"]["BS"])<.10
    if high and low_behavior:return "F2_stored_unused"
    if not high:return "F1_state_formation"
    return "other_partial"


def phenotype_means(rows):
    return {key:mean([metric(r,key) for r in rows]) for key in
            ("pre_I","post_I","delta_NL","state_main_norm","action_main_norm",
             "overlap","Q1","mean_abs_curvature")}


def group_detail(rows,operating):
    op={r["run"]:r for r in operating}
    return {"runs":[r["run"] for r in rows],
            "G110_positive":sum(metric(r,"delta_NL")>=.10 for r in rows),
            "G113_unit_pass":sum(unit_pass(r,8) for r in rows if r["endpoint"]["class"]=="healthy"),
            "G114_rescue":sum(rescue_pass(op[r["run"]]) for r in rows if r["endpoint"]["class"]=="shortcut"),
            "G115_destroy":sum(destruction_pass(op[r["run"]]) for r in rows if r["endpoint"]["class"]=="healthy"),
            "trajectory_precedes":sum(bool(onset(r) and onset(r)["precedes"]) for r in rows),
            "trajectories":{r["run"]:onset(r) for r in rows}}


def main():
    freeze=json.loads((ROOT/"processed"/"frozen_rule.json").read_text())
    historical=load(ROOT/"fusion_anatomy"/"historical_legacy")
    confirm=load(ROOT/"fusion_anatomy"/"confirmatory_baseline")
    a1=load(ROOT/"fusion_anatomy"/"original_query")+load(ROOT/"fusion_anatomy"/"confirmatory_query")
    assert len(historical)==len(confirm)==len(a1)==24
    old_h=[r for r in historical if r["run"] in freeze["primary_healthy"]]
    old_f=[r for r in historical if r["run"] in freeze["primary_shortcut"]]
    new_h=[r for r in confirm if r["endpoint"]["class"]=="healthy"][:8]
    new_f=[r for r in confirm if r["endpoint"]["class"]=="shortcut"][:8]
    old_op=load(ROOT/"operating_point_rescue"/"formal"/"historical_legacy")
    new_op=load(ROOT/"operating_point_rescue"/"formal"/"confirmatory_baseline")
    a1_op=load(ROOT/"operating_point_rescue"/"formal"/"original_query")+load(
        ROOT/"operating_point_rescue"/"formal"/"confirmatory_query")
    pairs={"historical":(old_h,old_f,old_op),"confirmatory":(new_h,new_f,new_op)}
    details={}
    for cohort,(healthy,failed,operating) in pairs.items():
        op={r["run"]:r for r in operating}
        direction={key:sum(metric(h,key)>=metric(f,key)+(.10 if key=="Q1" else 0.)
                           for h,f in zip(healthy,failed)) for key in ("Q1","overlap")}
        details[cohort]={"healthy_count":len(healthy),"shortcut_count":len(failed),
            "healthy_runs":[r["run"] for r in healthy],"shortcut_runs":[r["run"] for r in failed],
            "G110_positive":sum(metric(r,"delta_NL")>=.10 for r in healthy),
            "G111_Q1_direction":direction["Q1"],"G112_O_direction":direction["overlap"],
            "AUC":{key:auc(healthy,failed,key) for key in ("overlap","state_main_norm",
                            "action_main_norm","mean_abs_curvature","fusion_norm","Q1","post_I")},
            "run_bootstrap":{key:bootstrap_difference(healthy,failed,key,
                20260922+len(cohort)*100+i) for i,key in enumerate(
                    ("pre_I","post_I","delta_NL","Q1","overlap","state_main_norm"))},
            "G113_topk":sum(unit_pass(r,freeze["unit_k"]) for r in healthy),
            "G114_rescue":sum(rescue_pass(op[r["run"]]) for r in failed),
            "G115_destroy":sum(destruction_pass(op[r["run"]]) for r in healthy),
            "G116_ordered":sum(bool(onset(r) and onset(r)["precedes"]) for r in healthy),
            "trajectory_onsets":{r["run"]:onset(r) for r in healthy+failed}}
    def decision(key, *, need_healthy=True, need_shortcut=True, extra=None):
        for cohort in ("historical","confirmatory"):
            d=details[cohort]
            if (need_healthy and d["healthy_count"]<8) or (need_shortcut and d["shortcut_count"]<8):
                return "INCONCLUSIVE_N_LT_8"
            if d[key]<6 or (extra is not None and not extra(cohort)):
                return "FAIL"
        return "PASS"
    statuses={
        "G110":decision("G110_positive",need_shortcut=False),
        "G111":decision("G111_Q1_direction"),
        "G112":decision("G112_O_direction",extra=lambda c:details[c]["AUC"]["overlap"]>
            max(details[c]["AUC"][k] for k in ("state_main_norm","action_main_norm",
                "mean_abs_curvature","fusion_norm"))),
        "G113":decision("G113_topk",need_shortcut=False),
        "G114":decision("G114_rescue",need_healthy=False),
        "G115":decision("G115_destroy",need_shortcut=False),
        "G116":decision("G116_ordered",extra=lambda c:details[c]["G116_ordered"]>
            sum(bool(onset(r) and onset(r)["precedes"]) for r in
                (old_f if c=="historical" else new_f))),
    }
    allrows=historical+confirm+a1
    f1=[r for r in allrows if phenotype(r)=="F1_state_formation"]
    f2=[r for r in allrows if phenotype(r)=="F2_stored_unused"]
    other=[r for r in allrows if phenotype(r)=="other_partial"]
    broad_probe_only=[r for r in allrows if r["endpoint"]["class"]!="healthy" and
        max(r["endpoint"]["health"]["z_probe"].get("H",0),
            r["endpoint"]["health"]["z_probe"].get("M",0))>=.75]
    healthy=[r for r in allrows if phenotype(r)=="healthy"]
    opall={r["run"]:r for r in old_op+new_op+a1_op}
    f2_rescue=[rescue_pass(opall[r["run"]]) for r in f2 if r["run"] in opall]
    healthy_state_q1=mean([r["endpoint"]["fusion"]["Q"]["1"] for r in healthy])
    healthy_state_main=mean([metric(r,"state_main_norm") for r in healthy])
    f2_mean_state=mean([metric(r,"state_main_norm") for r in f2])
    f2_mean_q1=mean([metric(r,"Q1") for r in f2])
    g117=bool(len(f2)>=8 and len(f2_rescue)>=8 and sum(f2_rescue)>=6 and
             f2_mean_state>=.5*healthy_state_main and f2_mean_q1<healthy_state_q1)
    gates={**statuses,"G117":"PASS" if g117 else "FAIL"}
    training_authorized=sum(gates[f"G{i}"]=="PASS" for i in (114,115,116))>=2
    for i in range(118,123):gates[f"G{i}"]="PENDING_AUTHORIZED" if training_authorized else "NOT_RUN_BY_PROTOCOL"
    result={"gates":gates,"training_authorized":training_authorized,
            "frozen_rule":freeze,"cohorts":{"historical":summary(historical),
                "confirmatory":summary(confirm),"A1_supplementary":summary(a1)},
            "primary":details,
            "phenotypes":{"F1":{"count":len(f1),"summary":summary(f1),
                                 "means":phenotype_means(f1)},
                          "F2":{"count":len(f2),"summary":summary(f2),
                                "means":phenotype_means(f2),
                                "rescue_count":sum(f2_rescue),"rescue_denominator":len(f2_rescue)},
                          "other_partial":{"count":len(other),"summary":summary(other),
                                           "means":phenotype_means(other)},
                          "broad_probe_only_pilot":{"count":len(broad_probe_only),
                              "means":phenotype_means(broad_probe_only),
                              "status":"EXCLUDED_FROM_FORMAL_G117; missing strict |BS|<.10 criterion"},
                          "healthy":{"count":len(healthy),"summary":summary(healthy),
                                     "means":phenotype_means(healthy)}},
            "development_grid":freeze["offset_grid"],
            "limitations":["A1 cohorts contain only four healthy endpoints in total",
                           "causal-unit within-run selection is diagnostic, not a deployable label-free training rule",
                           "cross-model neuron indices are not presumed aligned"]}
    out=ROOT/"processed"/"formal_gates.json";out.write_text(json.dumps(result,indent=2))
    (ROOT/"processed"/"selection.json").write_text(json.dumps({
        "development_primary_healthy":[{"run":r["run"],"checkpoint":r["checkpoint"]} for r in old_h],
        "development_primary_shortcut":[{"run":r["run"],"checkpoint":r["checkpoint"]} for r in old_f],
        "confirmatory_primary_healthy":[{"run":r["run"],"checkpoint":r["checkpoint"]} for r in new_h],
        "confirmatory_primary_shortcut":[{"run":r["run"],"checkpoint":r["checkpoint"]} for r in new_f],
        "rule":"development IDs frozen sorted; confirmatory first eight per frozen behavioral class sorted by run ID; no fusion-based selection"
    },indent=2))
    print(json.dumps({"gates":gates,"counts":{k:v["classes"] for k,v in result["cohorts"].items()},
                      "F1":len(f1),"F2":len(f2)}))


if __name__=="__main__":main()
