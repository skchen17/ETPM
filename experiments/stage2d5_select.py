"""Freeze Stage 2D.5 diagnostic cohorts and within-architecture norm references."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from stage2d5_audit import REPAIR_NODES


def load(folder):
    return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]


def main(args):
    root=args.root
    old_a0=load(root/"interaction_flow"/"matched_legacy")
    old_a1=load(root/"interaction_flow"/"original_query")
    new_a1=load(root/"interaction_flow"/"confirmatory_query")
    hist_a0=load(root/"interaction_flow"/"historical_legacy")
    all_a0=old_a0+hist_a0; all_a1=old_a1+new_a1
    healthy_norms={arm:{node:statistics.median(r["flow"][node]["norm"] for r in rows if r["class"]=="healthy")
                        for node in REPAIR_NODES}
                   for arm,rows in (("A0",all_a0),("A1",all_a1))}
    for arm,rows in (("A0",all_a0),("A1",all_a1)):
        healthy=[r for r in rows if r["class"]=="healthy"]
        healthy_norms[arm]["output_template"]=[statistics.mean(
            r["flow"]["logits"]["interaction_vector"][j] for r in healthy)
            for j in range(4)]
    # Keep all primary matched A0/A1 and confirmatory A1 in the descriptive map.
    # Causal primary: eight independent failed confirmatory A1, eight healthy
    # historical legacy C0. The architecture difference is never hidden.
    primary_failed=[r for r in new_a1 if r["class"]!="healthy"][:8]
    primary_healthy=[r for r in hist_a0 if r["class"]=="healthy"][:8]
    old_failed=[r for r in old_a1 if r["class"]!="healthy"]
    q_healthy=[r for r in all_a1 if r["class"]=="healthy"]
    selected=primary_failed+primary_healthy+old_failed+q_healthy
    unique={(r["cohort"],r["run"]):r for r in selected}
    manifest={"primary_failed_confirmatory_A1":[r["run"] for r in primary_failed],
              "primary_healthy_historical_A0":[r["run"] for r in primary_healthy],
              "secondary_failed_original_A1":[r["run"] for r in old_failed],
              "A1_healthy_all":[r["run"] for r in q_healthy],
              "A1_healthy_count":len(q_healthy),
              "A1_healthy_insufficient_for_eight_model_same_architecture_necessity":len(q_healthy)<8,
              "selected":[{"cohort":r["cohort"],"run":r["run"],"arm":r["arm"],
                           "class":r["class"],"checkpoint":r["checkpoint"]} for r in unique.values()],
              "selection_rule":"sorted run names; no outcome-based selection beyond preregistered phenotype",
              "healthy_norm_reference":"within architecture median of all healthy endpoints; factorial interaction norm"}
    (root/"processed"/"selection.json").write_text(json.dumps(manifest,indent=2))
    (root/"processed"/"healthy_norms.json").write_text(json.dumps(healthy_norms,indent=2))
    print(json.dumps({k:manifest[k] for k in ("primary_failed_confirmatory_A1","primary_healthy_historical_A0",
                        "secondary_failed_original_A1","A1_healthy_all")},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);main(p.parse_args())
