"""Adjudicate predeclared Stage 2B gates from independent training seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter,defaultdict
from pathlib import Path

from etrcm.stage2b.data import COLORS,FAMILIES,NAMES,make_corpus


ARMS=("GRU","RNN","E0","E1","E2")
SEEDS=(2401,2402,2403,2404,2405)
STRICT_OOD_FAMILIES={"attribute","location","revision","interference"}


def avg(xs):return sum(xs)/len(xs) if xs else float("nan")
def sd(xs):return statistics.stdev(xs) if len(xs)>1 else 0.0
def nice(x):
    if isinstance(x,bool):return "PASS" if x else "FAIL"
    if isinstance(x,int):return str(x)
    return f"{x:.3f}" if isinstance(x,float) else str(x)


def table(rows):
    if not rows:return "No records."
    cols=list(rows[0])
    return "| "+" | ".join(cols)+" |\n|"+"|".join("---" for _ in cols)+"|\n"+"\n".join(
        "| "+" | ".join(nice(row.get(c,"")) for c in cols)+" |" for row in rows)


def paired_ci(diffs, seed=142, draws=10000):
    rng=random.Random(seed)
    means=sorted(avg([rng.choice(diffs) for _ in diffs]) for _ in range(draws))
    return means[int(0.025*draws)],means[int(0.975*draws)]


def load(root,size,seed,arm):
    category=("active_parameter_control/small" if size=="matched" else
              "ordering_control/small" if arm=="E0_late" else
              "formal/small" if size=="small" else "medium")
    base=root/category/f"seed{seed}"/arm
    train=json.loads((base/"summary.json").read_text())
    processed_category=("processed/active_parameter_control/small" if size=="matched" else
                        "processed/ordering_control/small" if arm=="E0_late" else
                        "processed/small" if size=="small" else "processed/medium")
    evaluation=json.loads((root/processed_category/
                           f"seed{seed}"/arm/"evaluation.json").read_text())
    strict=[row for row in evaluation["main_records"]
            if row["split"]=="ood" and row["gap"]<=128 and row["family"] in STRICT_OOD_FAMILIES]
    evaluation["main"]["lexical_ood_strict"]={
        "n":len(strict),
        "candidate_accuracy":avg([row["candidate_correct"] for row in strict]),
        "raw_accuracy":avg([row["raw_exact"] for row in strict]),
        "answer_ce":avg([row["answer_ce"] for row in strict]),
    }
    return train,evaluation


def get(eval_record,path):
    value=eval_record
    for item in path:value=value[item]
    return value


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args()
    data={}
    missing=[]
    for seed in SEEDS:
        for arm in ARMS:
            try:data[("small",seed,arm)]=load(args.root,"small",seed,arm)
            except FileNotFoundError:missing.append(f"small/{seed}/{arm}")
    for arm in ARMS:
        try:data[("medium",2501,arm)]=load(args.root,"medium",2501,arm)
        except FileNotFoundError:missing.append(f"medium/2501/{arm}")
    ordering_control=[]
    for seed in SEEDS:
        try:ordering_control.append((seed,load(args.root,"small",seed,"E0_late")))
        except FileNotFoundError:pass
    active_control=[]
    for seed in SEEDS:
        try:active_control.append((seed,load(args.root,"matched",seed,"GRU")))
        except FileNotFoundError:pass
    if missing:
        raise RuntimeError("formal training/evaluation incomplete: "+", ".join(missing))

    def metric(seed,arm,path):return get(data[("small",seed,arm)][1],path)
    gain={}
    for arm in ("E1","E2"):
        accuracy=[metric(s,arm,["main","in_distribution","candidate_accuracy"])-
                  metric(s,"E0",["main","in_distribution","candidate_accuracy"]) for s in SEEDS]
        ce=[metric(s,"E0",["main","in_distribution","answer_ce"])-
            metric(s,arm,["main","in_distribution","answer_ce"]) for s in SEEDS]
        ood=[metric(s,arm,["main","lexical_ood_strict","candidate_accuracy"])-
             metric(s,"E0",["main","lexical_ood_strict","candidate_accuracy"]) for s in SEEDS]
        joint=[metric(s,arm,["counterfactual","full","joint_accuracy"])-
               metric(s,"E0",["counterfactual","full","joint_accuracy"]) for s in SEEDS]
        hreset=[metric(s,arm,["counterfactual","H_reset","candidate_accuracy"])-
                metric(s,arm,["counterfactual","H_reset_FM_zero","candidate_accuracy"]) for s in SEEDS]
        benefits={control:[metric(s,arm,["interventions",control,"answer_ce"])-
                           metric(s,arm,["interventions","full","answer_ce"]) for s in SEEDS]
                  for control in ("zero","random","shuffle")}
        gain[arm]={"accuracy":accuracy,"ce":ce,"ood":ood,"joint":joint,"hreset":hreset,"benefits":benefits,
                   "accuracy_ci":paired_ci(accuracy),"ce_ci":paired_ci(ce)}
    g38={arm:((sum(x>0 for x in gain[arm]["accuracy"])>=4 and gain[arm]["accuracy_ci"][0]>0)
               or (sum(x>0 for x in gain[arm]["ce"])>=4 and gain[arm]["ce_ci"][0]>0))
         for arm in ("E1","E2")}
    g39={arm:(sum(x>=0.05 for x in gain[arm]["joint"])>=4 and
               avg([metric(s,arm,["counterfactual","full","joint_accuracy"]) for s in SEEDS])>0.25)
         for arm in ("E1","E2")}
    g40={arm:sum(x>=0.03 for x in gain[arm]["hreset"])>=4 for arm in ("E1","E2")}
    g41={arm:all(sum(x>0 for x in gain[arm]["benefits"][control])>=4 and
                 avg(gain[arm]["benefits"][control])>0 for control in ("zero","random","shuffle"))
         for arm in ("E1","E2")}
    gates={"G38":any(g38.values()),"G39":any(g39.values()),"G40":any(g40.values()),"G41":any(g41.values())}
    ordering_specific=False
    if len(ordering_control)==5:
        late_by_seed={seed:record[1]["main"]["in_distribution"]["candidate_accuracy"]
                      for seed,record in ordering_control}
        ordering_specific=any(
            avg([metric(seed,arm,["main","in_distribution","candidate_accuracy"])-late_by_seed[seed]
                 for seed in SEEDS])>=0.03
            and sum(metric(seed,arm,["main","in_distribution","candidate_accuracy"])>late_by_seed[seed]
                    for seed in SEEDS)>=4
            for arm in ("E1","E2"))
    if all(gates.values()) and ordering_specific:
        outcome="A — contextual KV significantly fixes language memory (within this controlled protocol)"
    elif gates["G38"] or any(avg(gain[arm]["accuracy"])>0.03 and avg(gain[arm]["ood"])>0.03
                                    and sum(x>0 for x in gain[arm]["accuracy"])>=3 for arm in ("E1","E2")):
        outcome="B — contextual KV helps, but retrieval/integration/stability remains limiting"
    else:outcome="C — contextual KV does not materially improve memory behavior"

    train_rows=[]; recall_rows=[]; gap_rows=[]; cf_rows=[]; lesion_rows=[]; generation_rows=[]; stability_rows=[]; null_rows=[]
    for size,seeds in (("small",SEEDS),("medium",(2501,))):
        for arm in ARMS:
            records=[data[(size,s,arm)] for s in seeds]
            train_rows.append({"size":size,"arm":arm,"n_seeds":len(seeds),
              "nominal":records[0][0]["nominal_parameters"],"active":records[0][0]["active_parameters"],
              "state_B":records[0][0]["allocated_state_bytes"],
              "effective_FM_B":records[0][0]["effective_memory_bytes"],
              "ms_per_token":round(1000*avg([r[0]["compute_seconds_per_token"] for r in records]),3),
              "basic_CE":avg([r[0]["validation"]["basic"]["CE"] for r in records]),
              "basic_PPL":avg([r[0]["validation"]["basic"]["PPL"] for r in records]),
              "memory_ID_CE":avg([r[0]["validation"]["memory_id"]["CE"] for r in records]),
              "memory_ID_PPL":avg([r[0]["validation"]["memory_id"]["PPL"] for r in records]),
              "memory_OOD_CE":avg([r[0]["validation"]["memory_ood"]["CE"] for r in records]),
              "memory_OOD_PPL":avg([r[0]["validation"]["memory_ood"]["PPL"] for r in records]),
              "reasoning_CE":avg([r[0]["validation"]["reasoning"]["CE"] for r in records]),
              "reasoning_PPL":avg([r[0]["validation"]["reasoning"]["PPL"] for r in records])})
            for split,key in (("ID","in_distribution"),("strict lexical OOD","lexical_ood_strict"),
                              ("all OOD cells","lexical_ood"),("256+","long_extrapolation")):
                recall_rows.append({"size":size,"arm":arm,"split":split,
                    "candidate_acc":avg([r[1]["main"][key]["candidate_accuracy"] for r in records]),
                    "seed_SD":sd([r[1]["main"][key]["candidate_accuracy"] for r in records]),
                    "raw_exact":avg([r[1]["main"][key]["raw_accuracy"] for r in records]),
                    "answer_CE":avg([r[1]["main"][key]["answer_ce"] for r in records])})
            for split in ("train","ood"):
                for gap in (32,64,128,256,512,1024):
                    family_scope=("strict 4" if split=="ood" and gap<=128 else
                                  "long 2" if gap>128 else "all 6")
                    by_seed=[]
                    for _,evaluation in records:
                        rows=[row for row in evaluation["main_records"]
                              if row["split"]==split and row["gap"]==gap
                              and (family_scope!="strict 4" or row["family"] in STRICT_OOD_FAMILIES)]
                        if rows:
                            by_seed.append({"n":len(rows),"candidate":avg([r["candidate_correct"] for r in rows]),
                                            "raw":avg([r["raw_exact"] for r in rows]),
                                            "CE":avg([r["answer_ce"] for r in rows])})
                    if by_seed:
                        gap_rows.append({"size":size,"arm":arm,"split":split,"gap":gap,
                                         "families":family_scope,"n_per_seed":by_seed[0]["n"],
                                         "candidate_acc":avg([r["candidate"] for r in by_seed]),
                                         "raw_exact":avg([r["raw"] for r in by_seed]),
                                         "answer_CE":avg([r["CE"] for r in by_seed])})
            if arm in {"E0","E1","E2"}:
                for condition in ("full","F_swap","M_swap","FM_swap","zero","random","H_reset","H_reset_FM_zero"):
                    cf_rows.append({"size":size,"arm":arm,"condition":condition,
                      "joint_acc":avg([r[1]["counterfactual"][condition]["joint_accuracy"] for r in records]),
                      "candidate_acc":avg([r[1]["counterfactual"][condition]["candidate_accuracy"] for r in records]),
                      "answer_CE":avg([r[1]["counterfactual"][condition]["mean_answer_ce"] for r in records])})
                for condition in ("full","H","F","M","FM","zero","random","shuffle"):
                    lesion_rows.append({"size":size,"arm":arm,"condition":condition,
                      "candidate_acc":avg([r[1]["interventions"][condition]["candidate_accuracy"] for r in records]),
                      "answer_CE":avg([r[1]["interventions"][condition]["answer_ce"] for r in records])})
                for K in (0,1,2,4,8,16):
                    null_rows.append({"size":size,"arm":arm,"K":K,
                      "candidate_acc":avg([r[1]["null"][f"K{K}"]["candidate_accuracy"] for r in records]),
                      "answer_CE":avg([r[1]["null"][f"K{K}"]["answer_ce"] for r in records]),
                      "H":avg([r[1]["null"][f"K{K}"]["H"] for r in records]),
                      "F":avg([r[1]["null"][f"K{K}"]["F"] for r in records]),
                      "M":avg([r[1]["null"][f"K{K}"]["M"] for r in records]),
                      "r_F":avg([r[1]["null"][f"K{K}"]["r_F"] for r in records]),
                      "r_M":avg([r[1]["null"][f"K{K}"]["r_M"] for r in records])})
            for decoder in ("greedy","temp_0.7","top_k_8"):
                gen=[g for _,r in records for g in r["generations"] if g["decoder"]==decoder]
                generation_rows.append({"size":size,"arm":arm,"decoder":decoder,"n":len(gen),
                  "semantic_acc":avg([g["semantic_correct"] for g in gen]),
                  "whole_exact":avg([g["whole_answer_exact"] for g in gen]),
                  "EOS":avg([g["EOS"] for g in gen]),
                  "punct_only":avg([g["punctuation_only"] for g in gen]),
                  "repetition":avg([g["repetition"] for g in gen]),
                  "self_writes":sum(g["self_output_external_writes"] for g in gen)})
            if arm in {"E1","E2"}:
                for seed in seeds:
                    st=data[(size,seed,arm)][1]["stability"]
                    stability_rows.append({"size":size,"seed":seed,"arm":arm,"completed":st["completed"],
                      "first_H_100":st["first_H_gt_100"],"first_H_1000":st["first_H_gt_1000"],
                      "first_bad":st["first_nonfinite"],
                      "H_100":st["snapshots"]["100"]["H"] if st["snapshots"]["100"] else None,
                      "H_1000":st["snapshots"]["1000"]["H"] if st["snapshots"]["1000"] else None,
                      "H_5000":st["snapshots"]["5000"]["H"] if st["snapshots"]["5000"] else None,
                      "probe_CE_1000":st["snapshots"]["1000"].get("probe_answer_ce") if st["snapshots"]["1000"] else None,
                      "probe_correct_1000":st["snapshots"]["1000"].get("probe_correct") if st["snapshots"]["1000"] else None})

    seed_rows=[]
    for seed in SEEDS:
        for arm in ARMS:
            record=data[("small",seed,arm)][1]
            row={"seed":seed,"arm":arm,"ID_acc":record["main"]["in_distribution"]["candidate_accuracy"],
                 "OOD_acc":record["main"]["lexical_ood_strict"]["candidate_accuracy"],
                 "long_acc":record["main"]["long_extrapolation"]["candidate_accuracy"],
                 "ID_CE":record["main"]["in_distribution"]["answer_ce"]}
            if arm in {"E0","E1","E2"}:
                row.update({"pair_joint":record["counterfactual"]["full"]["joint_accuracy"],
                            "Hreset_acc":record["counterfactual"]["H_reset"]["candidate_accuracy"],
                            "Hreset_FM0_acc":record["counterfactual"]["H_reset_FM_zero"]["candidate_accuracy"],
                            "mem_benefit_zero":record["interventions"]["zero"]["answer_ce"]-
                                               record["interventions"]["full"]["answer_ce"]})
            seed_rows.append(row)

    audit_corpus=make_corpus(2402,4800,"train")
    answer_balance={}
    for family in FAMILIES:
        counts=Counter(ex.answer for ex in audit_corpus if ex.family==family)
        answer_balance[family]={"distinct_labels":len(counts),"minimum":min(counts.values()),
                                "maximum":max(counts.values()),"max_min_ratio":max(counts.values())/min(counts.values())}
    reps=[]
    for arm in ("E0","E1","E2","GRU"):
        ex=data[("small",2401,arm)][1]["generations"]
        reps.extend([{"arm":arm,**g} for g in ex[:12]])
    representation_rows=[]
    for arm in ("E1","E2"):
        for seed in SEEDS:
            rep=data[("small",seed,arm)][1]["representation"]
            representation_rows.append({"arm":arm,"seed":seed,"key_norm":rep["key_norm_mean"],
                                        "value_norm":rep["value_norm_mean"],"q_dot_k":rep["q_dot_k_mean"],
                                        "same_entity_diff_value":rep.get("key_similarity",{}).get("same_entity_diff_value",{}).get("mean"),
                                        "same_value_diff_entity":rep.get("key_similarity",{}).get("same_value_diff_entity",{}).get("mean"),
                                        "cross_relation":rep.get("key_similarity",{}).get("cross_relation",{}).get("mean")})
    stabilization_rows=[]
    for arm in ("E1","E2"):
        path=args.root/"processed/stability/seed2401"/arm/"stabilization.json"
        if path.exists():
            diagnostics=json.loads(path.read_text())
            for item in diagnostics["results"]:
                stabilization_rows.append({"arm":arm,"alpha":item["alpha"],
                    "recall_candidate":item["recall"]["candidate_accuracy"],
                    "answer_CE":item["recall"]["answer_ce"],
                    "H_1000":item["H_at_1000"],"H_5000":item["H_at_5000"],
                    "first_H_1000":item["first_H_gt_1000"],"first_nonfinite":item["first_nonfinite"]})
    ordering_rows=[]
    for seed,(train_record,eval_record) in ordering_control:
        ordering_rows.append({"seed":seed,"arm":"E0_late","ID_acc":eval_record["main"]["in_distribution"]["candidate_accuracy"],
                              "OOD_acc":eval_record["main"]["lexical_ood_strict"]["candidate_accuracy"],
                              "ID_answer_CE":eval_record["main"]["in_distribution"]["answer_ce"],
                              "active_parameters":train_record["active_parameters"]})
    active_rows=[]
    for seed,(train_record,eval_record) in active_control:
        active_rows.append({"seed":seed,"hidden":train_record["hidden_dim"],
            "active_parameters":train_record["active_parameters"],
            "ID_acc":eval_record["main"]["in_distribution"]["candidate_accuracy"],
            "OOD_acc":eval_record["main"]["lexical_ood_strict"]["candidate_accuracy"],
            "ID_answer_CE":eval_record["main"]["in_distribution"]["answer_ce"]})

    report=["# ET-RCM Stage 2B — Contextual Associative Memory Results", "",
            "> **Can contextual language associations be compressed into fixed-size recurrent F/M memory and later retrieved behaviorally?**", "",
            "> **由语言上下文形成的关联信息，能否不依赖保存所有历史 KV，而被压缩进固定规模的 F/M persistent state，并在未来通过 learned query 被重新取回并影响行为？**", "",
            f"**Formal conclusion: Outcome {outcome}.** This adjudication uses five independent small-model training seeds. "
            "A single medium-size seed is a scale check, not a replication. No M-necessity gate was imposed.", "",
            "## 1. Protocol, lineage and architecture", "",
            "Stage 1.x and 2A were treated as frozen. E0 reuses the unchanged Stage 2A raw-token KV implementation, "
            "including its write-before-read order. E1 derives normalized K and unconstrained V from contextual H; "
            "E2 concatenates the current token embedding. E1/E2 read old F/M, update H, derive KV, then external delta-write, "
            "consolidate and decay. This necessary ordering difference is a confound in E0 comparisons and prevents "
            "attributing any gain solely to representation without an E0-late ordering control. That optional "
            "ordering-matched raw-token control is reported separately when available; it never substitutes for E0 gates. The integration operator, "
            "query, F/M reads, consolidation law, decay, NULL and SELF_OUTPUT rules were not modified. "
            "Early auxiliary-data OOD leakage was found before formal completion, interrupted, and corrected; see "
            "`reports/STAGE2B_PROTOCOL_AMENDMENT.md`. The two 100-step development seeds are not adjudicative.", "",
            "## 2. Data, lexical OOD and shortcut checks", "",
            "Synthetic English has entity-attribute, location, revision, transfer relation, temporal revision and "
            "8/16/32-entity interference families. Training gaps are 16/32/64/128; evaluation 32/64/128 is in-distribution "
            "and 256/512/1024 extrapolates. Every lexical item appears in training; the parity of name+value indexes "
            "makes answer-bearing train and OOD combinations disjoint for attribute/location/revision/interference. "
            "Those four families define the strict lexical-OOD headline; all-family OOD cells, including temporal "
            "and transfer relation, remain visible separately. Auxiliary basic text never co-occurs a name with a color/place, "
            "and auxiliary reasoning excludes name-color binding. Counterfactual pairs have exactly the same token bag, "
            "different swapped relations, shared distractor/question, and randomized statement position. "
            "Name, answer, order and gap are randomized but not perfectly balanced, especially across 32 owner names. "
            "Relation-transfer and last-update temporal tasks still admit "
            "recency/template shortcuts; causal memory interventions, not surface accuracy, determine memory claims. "
            f"Training answer-label audit (seed 2402; min/max counts): {answer_balance}. "
            "Unit tests verify the lexical split, paired token inventory and a bounded attribute-label imbalance; "
            "the relation-name imbalance remains a limitation.", "",
            "## 3. Training scale, parameters and language modeling", "",
            "All formal runs use 1000 optimizer steps, batch 16, AdamW 5e-4, parameter-gradient clip 1, no state clip; "
            "objective = mean next-token CE + 4×answer-token CE. The answer loss is ordinary task supervision, not a memory key, "
            "importance or entity label. Train-only lowercase word/punctuation vocabulary, OOV `<unk>`. "
            "Nominal parameters include inactive inherited branches; active parameters are counted from non-null gradients "
            "on a training batch. Compute/token is measured wall time; not hardware-normalized FLOPs. "
            "Allocated state includes H/F/M even in no-memory baselines; effective F/M bytes are separately zero there. "
            "Training logs include loss, preclip gradient and H/F/M norms. Validation sets share templates but use separate RNG streams.", "",
            table(train_rows), "",
            "## 4. Associative recall — E0/E1/E2, GRU and RNN", "",
            "Candidate accuracy is constrained to each family's answer set; unrestricted next-token accuracy and answer CE "
            "are separate. Pooled candidate chance is not uniform across families (transfer-owner has 32 names); "
            "paired arm differences are the primary comparison. Long extrapolation uses attribute/interference only.", "",
            table(recall_rows), "", "### Gap-by-gap recall, including 256/512/1024 extrapolation", "",
            "For strict OOD gaps 32–128, only the four train/OOD-disjoint answer-pair families are included; "
            "long gaps use attribute/interference, exactly as generated. Each cell gives examples per seed.", "",
            table(gap_rows), "", "### Independent formal seed records", "",table(seed_rows), "",
            "### Optional E0-late ordering control (not part of G38–G41)", "",
            table(ordering_rows), "",
            "The optional controls were trained on CPU while mandatory formal arms used GPU. "
            "This is a numerical/hardware caveat for close differences; data, seeds and optimizer schedule are otherwise matched.", "",
            "### Optional GRU-76 active-parameter control (not part of G38–G41)", "",
            table(active_rows), "",
            "## 5. Counterfactual same-token/different-relation binding", "",
            "Each pair swaps the two colors while preserving the complete token inventory; the query relation changes "
            "its correct answer. Candidate choices are the two colors, so pairwise chance joint accuracy is 0.25. "
            "F/M swaps preserve the receiver episode's H and the other memory component. "
            "Answer CE and downstream candidate decisions are intervention outcomes, not geometric proxies.", "",
            table(cf_rows), "",
            "## 6. Functional memory interventions and H reset", "",
            "Lesions, zero, norm-matched random and batch-deranged shuffled memory are applied after the premise/distractor "
            "and before the question. Correct-memory benefit is `CE_control−CE_full`; a positive value is required. "
            "H reset restores only active H and leaves F/M bit-exact, then the question is replayed. "
            "Shuffled controls are performed only in groups with at least two examples and use a cyclic derangement. "
            "F/M swap and H-reset/FM-zero results above are the stronger relation-specific interventions.", "",
            table(lesion_rows), "", "### Seed-level contextual gains and gate inputs", ""]
    for arm in ("E1","E2"):
        item=gain[arm]
        report += [f"- {arm}: ID candidate gain vs E0 = {item['accuracy']} (mean {avg(item['accuracy']):.3f}, "
                   f"bootstrap 95% CI {item['accuracy_ci']}); ID answer-CE benefit = {item['ce']} "
                   f"(CI {item['ce_ci']}); OOD candidate gain = {item['ood']}; pair-joint gain = {item['joint']}; "
                   f"H-reset peripheral gain = {item['hreset']}; correct-memory CE benefits vs "
                   f"zero/random/shuffle = {item['benefits']}."]
    report += ["", "## 7. Representation diagnostics and semantic alignment", "",
               "K/V norms and q·k are descriptive only; they cannot establish semantic storage by themselves. "
               "The functional memory interventions above are the causal behavioral check. "
               "Per-token vectors and episode labels are preserved in raw evaluation JSON.", "",table(representation_rows), "",
               "## 8. Generation and answer extraction", "",
               "Unrestricted 6-token generation uses greedy, temperature 0.7 or top-k 8. The first alphabetic token after "
               "`answer:` is scored as the semantic answer; whole-answer exact requires that the entire alphabetic output "
               "be the single target. EOS, punctuation-only, repetition and SELF_OUTPUT write violations are recorded. "
               "These measures are not silently replaced by candidate accuracy. The next 48 literal samples include failures.", "",
               table(generation_rows), ""]
    for i,g in enumerate(reps,1):
        report.append(f"{i}. {g['arm']} {g['decoder']}: target `{g['target']}`, prompt `{g.get('prompt',g['family'])}` "
                      f"(gap {g['gap']}) → `{g['output']}`; semantic `{g['first_semantic']}`; "
                      f"correct={g['semantic_correct']}; EOS={g['EOS']}.")
    report += ["", "## 9. NULL ticks and stability", "",
               "NULL ticks have no external token or delta write. The table shows CE, candidate accuracy and H/F/M/read norms "
               "on identical questions at K=0/1/2/4/8/16. It does not assume monotone improvement.", "",
               table(null_rows), "", "Mixed-stream stability uses 60% external, 20% NULL and 20% SELF_OUTPUT ticks "
               "without reset or state clipping. Every trajectory is saved with first H>100, H>1000, NaN/Inf, "
               "read and H-step norms; finite 5000 ticks do not prove asymptotic stability. "
               "An in-distribution `alice has the red key` fact is injected; at 100/500/1000/5000 ticks a cloned, "
               "off-path state answers its question, recording CE/candidate without altering the continuing stream.", "",table(stability_rows), "",
               "### Separate post-training residual-scale diagnostic", "",
               "A forward hook scales the frozen `core_out` proposal by alpha in one E1/E2 seed. This is an "
               "exploratory evaluation-only perturbation, not retraining, and is excluded from G38–G41. "
               "Any norm reduction must be read alongside answer accuracy/CE; a changed operator cannot be credited "
               "to contextual KV.", "",table(stabilization_rows), "",
               "## 10. Formal gates and outcome", "",
               table([{"gate":name,"PASS":passed,"E1":{ "G38":g38,"G39":g39,"G40":g40,"G41":g41}[name]["E1"],
                       "E2":{ "G38":g38,"G39":g39,"G40":g40,"G41":g41}[name]["E2"]}
                      for name,passed in gates.items()]), "",
               f"Selected outcome: **{outcome}**. G38 requires >=4/5 same-direction seeds and a positive paired bootstrap CI "
               "for either ID candidate accuracy or answer CE. G39 requires pair-joint accuracy above 0.25 and >=0.05 "
               "gain vs E0 in >=4/5 seeds. G40 requires >=0.03 H-reset peripheral gain in >=4/5. "
               "G41 requires positive correct-memory CE benefit against zero/random/shuffled in >=4/5. "
               f"No gate was altered after inspecting formal outcomes. Outcome A additionally requires a contextual "
               f"advantage over the E0-late ordering control; observed check={ordering_specific}. "
               "M necessity was explicitly excluded.", "",
               "## 11. Explicit answers and negative results", ""]
    best=max(("E1","E2"),key=lambda arm:avg(gain[arm]["accuracy"]))
    def pick(rows,**filters):
        return next(row for row in rows if all(row.get(k)==v for k,v in filters.items()))
    def train_ce(arm,field="basic_CE"):return pick(train_rows,size="small",arm=arm)[field]
    def recall(arm,split,field):return pick(recall_rows,size="small",arm=arm,split=split)[field]
    def cf(arm,condition,field):return pick(cf_rows,size="small",arm=arm,condition=condition)[field]
    def intervention(arm,condition,field):return pick(lesion_rows,size="small",arm=arm,condition=condition)[field]
    def gen(arm,decoder,field):return pick(generation_rows,size="small",arm=arm,decoder=decoder)[field]
    def null(arm,K,field):return pick(null_rows,size="small",arm=arm,K=K)[field]
    q_and_a=[
      ("1. Contextual KV improves basic LM?",f"E0/E1/E2 mean CE {train_ce('E0'):.3f}/{train_ce('E1'):.3f}/{train_ce('E2'):.3f}; lower is better."),
      ("2. Improves associative recall?",f"ID candidate E0/E1/E2 {recall('E0','ID','candidate_acc'):.3f}/{recall('E1','ID','candidate_acc'):.3f}/{recall('E2','ID','candidate_acc'):.3f}; G38={gates['G38']}."),
      ("3. E1 or E2 more stable?",f"Compare all 5-seed first-H>1000/H5000 rows above; E1 ID accuracy {recall('E1','ID','candidate_acc'):.3f}, E2 {recall('E2','ID','candidate_acc'):.3f}. No single-sample stability claim."),
      ("4. Raw token KV the main bottleneck?",f"G38={gates['G38']}; ordering-matched E0_late has {len(ordering_rows)}/5 seeds. Any gain is not solely encoding without this control."),
      ("5. Counterfactual binding established?",f"Pair joint E0/E1/E2 {cf('E0','full','joint_acc'):.3f}/{cf('E1','full','joint_acc'):.3f}/{cf('E2','full','joint_acc'):.3f} vs chance 0.25; G39={gates['G39']}."),
      ("6. Same tokens, swapped relations distinguished?",f"Correct paired joint accuracy best contextual {cf(best,'full','joint_acc'):.3f}; both answers must switch, not just one."),
      ("7. Correct memory reduces answer CE?",f"Best arm {best}: full CE {intervention(best,'full','answer_CE'):.3f}, zero CE {intervention(best,'zero','answer_CE'):.3f}."),
      ("8. Zero/random/shuffled worse?",f"Best arm {best} CE controls {intervention(best,'zero','answer_CE'):.3f}/{intervention(best,'random','answer_CE'):.3f}/{intervention(best,'shuffle','answer_CE'):.3f} vs correct {intervention(best,'full','answer_CE'):.3f}; G41={gates['G41']}."),
      ("9. H-reset peripheral benefit?",f"Best arm H-reset candidate {cf(best,'H_reset','candidate_acc'):.3f} vs H-reset+FM0 {cf(best,'H_reset_FM_zero','candidate_acc'):.3f}; G40={gates['G40']}."),
      ("10. Separate F/M lesions?",f"Best arm full/F0/M0/FM0 answer CE {intervention(best,'full','answer_CE'):.3f}/{intervention(best,'F','answer_CE'):.3f}/{intervention(best,'M','answer_CE'):.3f}/{intervention(best,'FM','answer_CE'):.3f}; M necessity was not a gate."),
      ("11. Lexical OOD benefit?",f"Strict OOD candidate E0/E1/E2 {recall('E0','strict lexical OOD','candidate_acc'):.3f}/{recall('E1','strict lexical OOD','candidate_acc'):.3f}/{recall('E2','strict lexical OOD','candidate_acc'):.3f}."),
      ("12. GRU still stronger?",f"GRU/E0/{best} basic CE {train_ce('GRU'):.3f}/{train_ce('E0'):.3f}/{train_ce(best):.3f}; ID candidate {recall('GRU','ID','candidate_acc'):.3f}/{recall('E0','ID','candidate_acc'):.3f}/{recall(best,'ID','candidate_acc'):.3f}."),
      ("13. Long-gap advantage?",f"256+ candidate GRU/E0/{best} {recall('GRU','256+','candidate_acc'):.3f}/{recall('E0','256+','candidate_acc'):.3f}/{recall(best,'256+','candidate_acc'):.3f}; this is extrapolation."),
      ("14. Semantic organization in KV?",f"See same-entity/different-value, same-value/different-entity and cross-relation cosine diagnostics; geometry alone is descriptive."),
      ("15. Causal intervention support?",f"G41={gates['G41']}; correct-vs-swapped/zero CE, not cosine, carries the behavioral evidence."),
      ("16. Better free generation?",f"Greedy semantic-answer accuracy E0/E1/E2 {gen('E0','greedy','semantic_acc'):.3f}/{gen('E1','greedy','semantic_acc'):.3f}/{gen('E2','greedy','semantic_acc'):.3f}; different data/objective prevent a controlled Stage 2A cross-stage estimate."),
      ("17. EOS/punctuation collapse?",f"Best arm {best} greedy EOS {gen(best,'greedy','EOS'):.3f}, punctuation-only {gen(best,'greedy','punct_only'):.3f}, whole-answer exact {gen(best,'greedy','whole_exact'):.3f}."),
      ("18. NULL ticks harmful?",f"{best} answer CE K0→K16 {null(best,0,'answer_CE'):.3f}→{null(best,16,'answer_CE'):.3f}; H {null(best,0,'H'):.2f}→{null(best,16,'H'):.2f}."),
      ("19. H norm growth remains?",f"First-H>1000 and H5000 are per seed in Section 9; a finite trajectory is not a boundedness proof."),
      ("20. Modify H dynamics?",f"The separate alpha diagnostic, if available, tests norm-control vs recall tradeoff. Do not merge it with contextual-KV gates."),
      ("21. Priority next?",f"Given Outcome {outcome[0]}, prioritize functional routing/encoding discrimination, generation feedback and H stability before capacity scaling."),
      ("22. Larger language prototype justified?",f"Only if robust binding, intervention benefit and safe long-running behavior replicate; current gate vector is {gates}."),
    ]
    report += ["### Direct 22-question answer matrix", "",table([{"question":q,"measured answer":a} for q,a in q_and_a]), ""]
    report += [f"1. Contextual KV basic LM: compare CE/PPL in Section 3; GRU remains the training-strength reference.",
               f"2. Associative recall: best ID candidate gain vs E0 is {best} {avg(gain[best]['accuracy']):+.3f}; "
               "see OOD and CE separately, not only a pooled headline.",
               f"3. E1/E2 stability: compare all 5-seed CE/gain and trajectory tables; size-128 has only one seed.",
               "4. Raw-token encoding as Stage 2A's *main* bottleneck is not established without both robust functional "
               "gain and an ordering-matched raw control; E0 differs in step order.",
               "5–6. Same-token counterfactual binding and swapped answer identity are judged by pair-joint accuracy, "
               "not token frequency or q·k cosine.",
               "7–10. Correct/zero/random/shuffled/F-swap/M-swap effects and H-reset peripheral recall are in Sections 5–6; "
               "zero effect or negative benefit is retained rather than dismissed.",
               "11. Lexical OOD effects are in Section 4 and the seed-level gain list; all vocabulary tokens are known.",
               "12–13. GRU and long-gap comparisons are in Section 4. Cross-stage Stage 2A percentages are not directly "
               "comparable because data, objective and lengths changed.",
               "14–15. K/V organization is descriptive; finite memory interventions determine whether it is behaviorally useful.",
               "16–17. Generation quality, punctuation/EOS collapse and whole-answer exact are in Section 8; "
               "teacher-forced CE is not substituted for successful free answering.",
               "18–20. NULL effects and H drift are in Section 9. Any H stabilization should be a separately matched experiment, "
               "not folded into the contextual-KV comparison.",
               "21–22. Prioritize whichever failure is measured: lexical writing if G38 fails, routing/integration if memory "
               "interventions are null, long-run H stability if trajectories drift, and SELF_OUTPUT-aware decoding if free "
               "generation collapses. Do not scale to a larger language prototype on template-only, single-size evidence.", "",
               "## 12. Reproduction, artifacts and scientific boundaries", "",
               "All 1000-step checkpoints, tokenizer metadata, configs, training logs, evaluation records, seed summaries, "
               "generation examples, intervention vectors and 5000-tick trajectories are under `results/stage2b/`. "
               "The SHA-256 manifest is `results/stage2b/manifest.json`; the Stage 1.5 frozen manifest and Stage 2A "
               "commit-byte tests must pass. This controlled synthetic experiment cannot establish LLM capability, "
               "human-like thought, general memory, autonomous intelligence, consciousness or infinite context. "
               "A negative gate remains negative even if an individual sample looks compelling.", ""]
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text("\n".join(report))
    processed=args.root/"processed"
    processed.mkdir(parents=True,exist_ok=True)
    summary={"outcome":outcome,"gates":gates,"g38_by_arm":g38,"g39_by_arm":g39,"g40_by_arm":g40,
             "g41_by_arm":g41,"ordering_specific":ordering_specific,
             "seed_rows":seed_rows,"train_rows":train_rows,"recall_rows":recall_rows,"gap_rows":gap_rows,
             "counterfactual_rows":cf_rows,"lesion_rows":lesion_rows,"generation_rows":generation_rows,
             "stability_rows":stability_rows,"null_rows":null_rows,"contextual_gains":gain,
             "ordering_control_rows":ordering_rows,"active_control_rows":active_rows,
             "stabilization_rows":stabilization_rows,"answer_label_audit":answer_balance}
    (processed/"formal_summary.json").write_text(json.dumps(summary,indent=2))
    project=args.report.parent.parent
    manifest={}
    artifact_paths=[args.report]+[path for path in args.root.rglob("*")
                                 if path.is_file() and path.name!="manifest.json"]
    for path in artifact_paths:
        manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ("configs/stage2b.yaml","STAGE2B_START_HERE.md",
                     "reports/STAGE2B_PROTOCOL_AMENDMENT.md","src/etrcm/stage2b/model.py",
                     "src/etrcm/stage2b/data.py","src/etrcm/stage2b/interventions.py",
                     "experiments/stage2b_train.py","experiments/stage2b_eval.py",
                     "experiments/stage2b_report.py","experiments/stage2b_stability_diagnostic.py",
                     "tests/test_stage2b.py"):
        path=project/relative
        manifest[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    (args.root/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"report":str(args.report),"report_bytes":args.report.stat().st_size,
                      "outcome":outcome,"gates":gates,"best_arm":best,
                      "ID_candidate_gain":avg(gain[best]["accuracy"])}),flush=True)


if __name__=="__main__":main()
