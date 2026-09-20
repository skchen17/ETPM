#!/usr/bin/env python3
"""Post-trigger descriptive auxiliary curriculum analysis; no gate changes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_stage1_6 import sha256


ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/stage1_6/auxiliary_F"
FORMAL=ROOT/"results/stage1_6/processed/stage1_6-formal-v1/seed_checkpoint_effects.parquet"
SEEDS=range(8701,8709)


def bootstrap(values:np.ndarray)->list[float]:
    generator=np.random.default_rng(1617)
    means=generator.choice(values,(2000,len(values)),replace=True).mean(1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def main()->None:
    frames=[]
    for seed in SEEDS:
        cell=BASE/f"seed{seed}"
        summary=json.loads((cell/"summary.json").read_text())
        assert summary["seed"]==seed and summary["steps"]==3000
        assert all(sha256(cell/name)==digest for name,digest in summary["hashes"].items())
        records=pd.read_parquet(cell/"interventions.parquet")
        assert set(records.training_step)=={0,1000,2000,3000}
        assert set(records.condition)=={"learned","oracle","zero","random","shuffled","M_lesion","F_lesion"}
        frames.append(records)
    records=pd.concat(frames,ignore_index=True)
    assert np.isfinite(records.future_CE).all()
    paired=records.groupby(["seed","training_step","condition"]).agg(
        CE=("future_CE","mean"),accuracy=("accuracy","mean")).reset_index()
    wide=paired.pivot_table(index=["seed","training_step"],columns="condition",values="CE").reset_index()
    for name,series in {"D_R":wide.zero-wide.learned,
                        "D_O":wide.zero-wide.oracle,
                        "D_M":wide.M_lesion-wide.learned,
                        "D_F":wide.F_lesion-wide.learned}.items():
        wide[name]=series
    final=wide.loc[wide.training_step.eq(3000)].set_index("seed")
    old=pd.read_parquet(FORMAL)
    old=old.loc[old.training_arm.eq("curriculum") & old.training_step.eq(3000)].set_index("seed")
    paired_delta=final.D_R-old.D_R
    within_denominator=final.D_O.where(final.D_O>=.05)
    ratio=final.D_R/within_denominator
    effects={
        "aux_D_R":final.D_R,
        "aux_D_O":final.D_O,
        "aux_D_M":final.D_M,
        "aux_D_F":final.D_F,
        "aux_minus_formal_B3_D_R":paired_delta,
        "aux_oracle_fraction_valid_only":ratio,
    }
    summaries={}
    for name,values in effects.items():
        finite=values[np.isfinite(values)]
        summaries[name]={"mean":float(finite.mean()) if len(finite) else None,
                         "ci95":bootstrap(finite.to_numpy()) if len(finite) else [None,None],
                         "valid_seeds":int(len(finite)),
                         "seed_values":{str(k):(float(v) if np.isfinite(v) else None)
                                        for k,v in values.items()}}
    summaries["aux_D_R"]["seeds_above_0.025"]=int((final.D_R>=.025).sum())
    summaries["aux_D_M"]["seeds_above_0.02"]=int((final.D_M>=.02).sum())
    summaries["aux_D_F"]["seeds_above_0.02"]=int((final.D_F>=.02).sum())
    wide.to_parquet(BASE/"seed_checkpoint_effects.parquet",index=False)
    output={"scope":"post-trigger auxiliary F, not G34-G37",
            "schedule":"1000 oracle, 1000 50/50, 1000 learned",
            "matched_oracle_budget":1500,"effects":summaries,
            "final_accuracy":paired.loc[paired.training_step.eq(3000)].groupby("condition").accuracy.mean().to_dict()}
    (BASE/"analysis.json").write_text(json.dumps(output,indent=2))
    table=(paired.loc[paired.training_step.eq(3000)]
           .groupby("condition")[["CE","accuracy"]].mean().round(5).reset_index().to_markdown(index=False))
    summary_table=pd.DataFrame([{"effect":name,"mean":value["mean"],"95% seed CI":value["ci95"],
                                 "valid seeds":value["valid_seeds"]}
                                for name,value in summaries.items()]).to_markdown(index=False)
    report=("# Conditional auxiliary F: phase-wise oracle-to-learned curriculum\n\n"
            "This is a **post-trigger exploratory experiment**, not a fifth gate. It was triggered because formal G35 passed while ordinary learned-read training had finite D_R≥.01 in only 2/8 seeds. The three-phase schedule, total oracle budget, optimizer, seeds, data and controls were fixed in `reports/STAGE1_6_AUX_F_PROTOCOL.md` before training. It uses no query/path labels. The formal B3 five-block curriculum has the same expected 1500 oracle deliveries and is seed-paired here.\n\n"
            "## Final condition outcomes\n\n"+table+"\n\n## Seed-paired effects\n\n"+summary_table+"\n\n"
            f"Auxiliary learned read clears ≥.025 CE benefit in {summaries['aux_D_R']['seeds_above_0.025']}/8 seeds. "
            f"Scrub-boundary M and F lesions clear ≥.02 in {summaries['aux_D_M']['seeds_above_0.02']}/8 and {summaries['aux_D_F']['seeds_above_0.02']}/8 seeds. "
            "The auxiliary-versus-formal B3 D_R difference is descriptive and paired by seed; the frozen G37 outcome is unchanged. An oracle-reintroduction denominator below .05 is reported as missing, never as a pass.\n\n"
            "Raw episodes, phase checkpoints, logs, hashes and seed-checkpoint effects are under `results/stage1_6/auxiliary_F/`. This toy result does not authorize a language model or establish persistent M necessity.\n")
    (ROOT/"reports/AUXILIARY_MEMORY_USE_CURRICULUM_STAGE1_6.md").write_text(report)
    print(json.dumps({"aux_D_R_seeds":summaries["aux_D_R"]["seeds_above_0.025"],
                      "aux_D_M_seeds":summaries["aux_D_M"]["seeds_above_0.02"],
                      "aux_D_F_seeds":summaries["aux_D_F"]["seeds_above_0.02"]}))


if __name__=="__main__":
    main()
