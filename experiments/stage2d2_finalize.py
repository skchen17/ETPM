"""Create compact output routing and conditional-stage status records."""
from __future__ import annotations
import argparse,json
from pathlib import Path

def write(path,obj): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2))
def main(args):
    analysis=json.loads(args.analysis.read_text()); causal=json.loads(args.causal.read_text())
    write(args.root/"history_alignment/summary.json",{"cohort_A":analysis["alignment_trajectory"],"cohort_B":analysis["confirmatory_alignment_trajectory"]})
    write(args.root/"memory_alignment/summary.json",{"cohort_A":analysis["alignment_trajectory"],"cohort_B":analysis["confirmatory_alignment_trajectory"],"rank_energy":analysis["rank_energy"]})
    write(args.root/"controllability/summary.json",{"development":analysis["development_candidates"],"frozen":analysis["frozen_predictor"],"confirmatory":analysis["confirmatory"],"G75":analysis["G75"]})
    write(args.root/"frozen_rescue/results.json",{"G73":causal["G73"],"runs":[r for r in causal["runs"] if r["class"]!="healthy"]})
    write(args.root/"healthy_destruction/results.json",{"G74":causal["G74"],"runs":[r for r in causal["runs"] if r["class"]=="healthy"]})
    authorized=causal["G73"]["pass"] or causal["G74"]["pass"] or analysis["G75"]["pass"]
    status={"authorized":authorized,"reason":"G73, G74, and G75 all failed; Experiment H forbidden by protocol" if not authorized else "diagnostic gate passed"}
    write(args.root/"alignment_warmup/status.json",status)
    write(args.root/"controls/status.json",status)
    downstream={"status":"NOT_RUN_BY_PROTOCOL","reason":"G76 cannot be adjudicated because alignment warmup was not authorized"}
    write(args.root/"dynamics_recheck/status.json",downstream);write(args.root/"continuous_runs/status.json",downstream)
    gates={"G73":causal["G73"],"G74":causal["G74"],"G75":analysis["G75"],
           "G76":{"status":"NOT_RUN_BY_PROTOCOL"},"G77":{"status":"NOT_RUN_BY_PROTOCOL"},"G78":{"status":"NOT_RUN_BY_PROTOCOL"}}
    write(args.root/"processed/formal_gates.json",gates);print(json.dumps(gates))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--analysis",type=Path,required=True);p.add_argument("--causal",type=Path,required=True);p.add_argument("--root",type=Path,required=True);main(p.parse_args())
