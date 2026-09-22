"""Audit development and confirmatory runs with frozen endpoint labels."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from stage2d6_audit import one


def historical_inputs():
    folder=Path("results/stage2d5/interaction_flow/historical_legacy")
    rows=[json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    return [(Path(r["checkpoint"]),"A0","historical_legacy",r["class"])
            for r in rows]


def confirmatory_inputs(root):
    return [(p,"A0","confirmatory_baseline",None) for p in sorted(
        (root/"confirmatory"/"checkpoints").glob("*/checkpoint_1500.pt"))]


def a1_inputs():
    folders=("original_query","confirmatory_query")
    rows=[]
    for cohort in folders:
        path=Path("results/stage2d5/interaction_flow")/cohort
        rows.extend([(Path(r["checkpoint"]),"A1",cohort,r["class"])
                     for r in (json.loads(p.read_text()) for p in sorted(path.glob("*.json")))])
    return rows


def main(args):
    if args.cohort=="historical": inputs=historical_inputs()
    elif args.cohort=="confirmatory": inputs=confirmatory_inputs(args.root)
    else: inputs=a1_inputs()
    if not args.allow_incomplete:
        assert len(inputs)==24, len(inputs)
    # Every checkpoint is a run-level unit. Full trajectories for all baseline
    # runs; A1 is F1/F2 supplementary and endpoint-only.
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[]
        for path,arm,cohort,_ in inputs:
            out=args.root/"fusion_anatomy"/cohort/f"{path.parent.name}.json"
            if args.skip_existing and out.exists():
                continue
            futures.append(pool.submit(one,path,arm,cohort,out,args.reps,
                                       args.cohort!="a1"))
        for future in as_completed(futures):
            result=future.result()
            print("COMPLETE",result["cohort"],result["run"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--cohort",choices=("historical","confirmatory","a1"),required=True)
    p.add_argument("--jobs",type=int,default=12)
    p.add_argument("--reps",type=int,default=16)
    p.add_argument("--allow-incomplete",action="store_true")
    p.add_argument("--skip-existing",action="store_true")
    main(p.parse_args())
