"""Deterministic shortcut and schedule audit; no learned checkpoint required."""

from __future__ import annotations

import json
import random
from pathlib import Path

from etrcm.stage2c.world import ABSTRACT,ACTION,NUISANCE,OUTCOME,paired_histories,surface


def audit():
    checks={}
    paired=[]
    for seed in range(6201,6209):
        a,b=paired_histories(seed,32)
        same_present=all((x.color,x.shape,x.nuisance,x.action)==
                         (y.color,y.shape,y.nuisance,y.action) for x,y in zip(a,b))
        balanced={str(n):sum(x.action==0 for x in a[:n])==n//2
                  for n in (2,4,8,16,32)}
        paired.append({"seed":seed,"same_surface_action_clock":same_present,
                       "prefix_action_balanced":balanced,
                       "outcome_single_symbol_each":all(x.outcome in OUTCOME and y.outcome in OUTCOME
                                                       for x,y in zip(a,b))})
    checks["paired_histories"]=paired
    train=set();novel=set();hard=set()
    rng=random.Random(404)
    for _ in range(500):
        train.add(surface(rng,"train")[:2]);novel.add(surface(rng,"novel")[:2])
        hard.add(surface(rng,"hard_ood")[:2])
    checks["surface_split"]={"train_n":len(train),"novel_n":len(novel),"hard_ood_n":len(hard),
                             "train_novel_disjoint":train.isdisjoint(novel),
                             "train_hard_disjoint":train.isdisjoint(hard),
                             "novel_single_features_seen":
                                 {x[0] for x in novel}<={x[0] for x in train} and
                                 {x[1] for x in novel}<={x[1] for x in train}}
    checks["event_design"]={"probe_latent_field":False,
                             "shared_abstract_token":ABSTRACT,
                             "actions":list(ACTION),"unrelated_noise_tokens":list(NUISANCE),
                             "unrelated_noise_contains_action":bool(set(NUISANCE)&set(ACTION)),
                             "unrelated_noise_contains_outcome":bool(set(NUISANCE)&set(OUTCOME)),
                             "external_outcome_write_once_per_experience":True,
                             "context_action_events_write":False}
    checks["passed"]=(all(x["same_surface_action_clock"] and
                          all(x["prefix_action_balanced"].values()) and
                          x["outcome_single_symbol_each"] for x in paired)
                      and checks["surface_split"]["train_novel_disjoint"]
                      and checks["surface_split"]["novel_single_features_seen"]
                      and not checks["event_design"]["unrelated_noise_contains_action"]
                      and not checks["event_design"]["unrelated_noise_contains_outcome"])
    return checks


if __name__=="__main__":
    result=audit()
    out=Path("results/stage2c/manifests/shortcut_audit.json")
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2))
    print(json.dumps({"shortcut_audit_passed":result["passed"],"path":str(out)}))
    if not result["passed"]:raise SystemExit(1)
