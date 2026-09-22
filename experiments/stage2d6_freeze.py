"""Freeze run-level primary metrics and one global operating-point rule on legacy development."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


ROOT=Path("results/stage2d6")


def rows(folder):
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def mean(x):
    return statistics.mean(x) if x else 0.


def main():
    old=rows(ROOT/"fusion_anatomy"/"historical_legacy")
    healthy=[r for r in old if r["endpoint"]["class"]=="healthy"][:8]
    failed=[r for r in old if r["endpoint"]["class"]=="shortcut"][:8]
    assert len(healthy)==len(failed)==8
    options=[]
    for k in (1,2,4,8,16):
        pass_count=0;deltas=[]
        for r in healthy:
            u=r["endpoint"]["unit_causality"][str(k)]
            loss=u["IHA_loss"];control=u["random_mean_IHA_loss"]
            ok=loss>=.05 and loss>=.5*r["endpoint"]["fusion"]["behavior"]["IHA"] and loss-control>=.025
            pass_count+=bool(ok);deltas.append(loss-control)
        options.append({"k":k,"pass_count":pass_count,"mean_excess_loss":mean(deltas)})
    eligible=[o["k"] for o in options if o["pass_count"]>=6]
    k=min(eligible) if eligible else 8
    grids=rows(ROOT/"operating_point_rescue"/"development_grid"/"historical_legacy")
    by_run={r["run"]:r for r in grids}
    scales=(-1.,-.5,-.25,0.,.25,.5,1.)
    scale_rows=[]
    for scale in scales:
        improvements=[]
        for r in failed:
            row=by_run[r["run"]]
            improvements.append(row["offset_grid"][str(scale)]["behavior"]["IHA"]-
                                row["native"]["IHA"])
        scale_rows.append({"scale":scale,"mean_IHA_improvement":mean(improvements),
                           "positive_count":sum(x>.025 for x in improvements)})
    best=max(scale_rows,key=lambda r:(r["mean_IHA_improvement"],-abs(r["scale"]),-r["scale"]))
    frozen={"development_cohort":"historical_legacy", "primary_healthy":[r["run"] for r in healthy],
            "primary_shortcut":[r["run"] for r in failed],"primary_unit_rule":"within-run descending finite IHA loss, deterministic tie-break",
            "unit_k":k,"unit_k_options":options,
            "offset_rule":"same signed global scale times native preactivation SD on selected units; no per-run search",
            "offset_scale":best["scale"],"offset_grid":scale_rows,
            "formal_success":"TV>=.10, IHA>=.10, abs(BS)>=.10 and IHA/abs(BS) each improve >=.025; post_I improves >=.025",
            "G110_threshold":"post_I-pre_I >=.10 in >=6/8 healthy per cohort",
            "G111_threshold":"rank-1 Q healthy exceeds shortcut by >=.10 in >=6/8 sorted independent pairs per cohort",
            "G112_threshold":"O healthy exceeds shortcut in >=6/8 pairs and AUROC(O) exceeds each single-component AUROC in both cohorts",
            "G113_threshold":"top-k IHA loss >=.05 and >=50% native and >=.025 more than matched random mean in >=6/8 healthy per cohort",
            "G115_threshold":"low-curvature shift causes post_I >=25% fall and IHA and abs(BS) >=50% fall, exceeding matched-shift loss by >=.025 in >=6/8 healthy per cohort",
            "G116_threshold":"O crosses 50% endpoint increase no later than IHA crosses .10 in >=6/8 healthy per cohort, failed lacks same pattern",
            "statistics_unit":"independent trained run; neuron/history measurements aggregated within run",
            "confirmatory_sealed_at_freeze":not (ROOT/"confirmatory"/"training_summaries.json").exists()}
    target=ROOT/"processed"/"frozen_rule.json";target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(frozen,indent=2))
    print(json.dumps({"k":k,"offset_scale":best["scale"],"positive_count":best["positive_count"],
                      "confirmatory_sealed":frozen["confirmatory_sealed_at_freeze"]}))


if __name__=="__main__":main()
