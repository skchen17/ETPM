#!/usr/bin/env python3
"""Exploratory original-objective scaling, separately from G34–G37."""

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/stage1_6/legacy_scaling"
STEPS=(0,50,100,160,300,500,1000,2000,3000)
SEEDS=(8701,8702,8703,8704,8705,8706,8707,8708)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def main()->None:
    frames=[]; manifest=[]
    for seed in SEEDS:
        cell=BASE/f"B5_seed{seed}"
        summary=json.loads((cell/"summary.json").read_text())
        assert all(sha256(cell/name)==digest for name,digest in summary["hashes"].items())
        frame=pd.read_parquet(cell/"oracle_interventions.parquet")
        assert set(frame.training_step)==set(STEPS)
        frames.append(frame)
        manifest.append({"seed":seed,"summary_sha256":sha256(cell/"summary.json")})
    data=pd.concat(frames,ignore_index=True)
    main=data.loc[data.internal_tick.eq(4)]
    seed=main.groupby(["training_seed","training_step","intervention_condition"]).future_CE.mean().unstack()
    seed["oracle_vs_no_read"]=seed["no_read"]-seed["oracle_static"]
    seed["oracle_vs_random"]=seed["random"]-seed["oracle_static"]
    seed["oracle_vs_shuffled"]=seed["shuffled"]-seed["oracle_static"]
    seed["learned_vs_no_read"]=seed["no_read"]-seed["learned"]
    seed=seed.reset_index()
    seed.to_parquet(BASE/"seed_checkpoint_effects.parquet",index=False)
    means=seed.groupby("training_step").mean(numeric_only=True).drop(columns=["training_seed"])
    replicate=seed.groupby("training_step")["oracle_vs_no_read"].apply(lambda x:int((x>=.01).sum()))
    delta=means.loc[3000]-means.loc[160]
    result={"scope":"exploratory original Stage 1.5 objective, 128-distractor long-gap only",
            "mean_effect_3000_minus_160":{k:float(v) for k,v in delta.items()},
            "oracle_benefit_seeds_at_least_0.01_by_checkpoint":{str(k):int(v) for k,v in replicate.items()},
            "seed_manifest":manifest}
    (BASE/"summary.json").write_text(json.dumps(result,indent=2))
    report=("# Original-objective training-length scaling (exploratory)\n\n"
            "This is an extension, **not G34–G37**. Eight new B5 seeds were trained on the frozen Stage 1.5 four-family future-event objective for 3000 steps with its selected LR .001, batch 32 and original data schedule. The old historical oracle evaluator was reused on the 128-distractor long-gap slice, tick 4. It does not cover the old 512/2048 long-gap grid. In that old world, past A is also the future class, so its oracle read can approximate the answer itself; this does not test compositional integration the way the new H-scrub world does. Below are seed-equal mean CE and finite read interventions; positive `oracle_vs_no_read` means the historical read helps.\n\n"
            +means.round(5).reset_index().to_markdown(index=False)
            +"\n\nAt 3000 minus 160 steps: "+json.dumps(result["mean_effect_3000_minus_160"])
            +". Replication counts (oracle vs no-read ≥.01): "+json.dumps(result["oracle_benefit_seeds_at_least_0.01_by_checkpoint"])
            +". Training loss alone is not evidence of memory use. Full paired records and hashes are in `results/stage1_6/legacy_scaling/`.\n")
    (ROOT/"reports/TRAINING_LENGTH_SCALING_ORIGINAL_OBJECTIVE_STAGE1_6.md").write_text(report)
    print(json.dumps(result))


if __name__=="__main__":
    main()
