"""Audit remaining A1 failures and historical stored-but-unused controls."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from stage2d5_audit import one


def main(args):
    root=args.root
    norms=json.loads((root/"processed"/"healthy_norms.json").read_text())
    items=[]
    for cohort in ("confirmatory_query","historical_legacy"):
        for p in sorted((root/"interaction_flow"/cohort).glob("*.json")):
            row=json.loads(p.read_text())
            if cohort=="confirmatory_query" and row["class"]=="healthy":continue
            if cohort=="historical_legacy":
                probes=row["health"]["z_probe"]
                if row["class"]=="healthy" or max(probes["H"],probes["M"])<.75 or abs(row["behavior"]["BS"])>=.1:continue
            out=root/"path_restoration"/"individual"/cohort/f"{row['run']}.json"
            if out.exists():continue
            items.append((Path(row["checkpoint"]),row["arm"],cohort,out,norms,True,args.reps))
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(one,*x) for x in items]
        for f in as_completed(futures):
            r=f.result();print("COMPLETE",r["cohort"],r["run"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True)
    p.add_argument("--jobs",type=int,default=12);p.add_argument("--reps",type=int,default=16)
    main(p.parse_args())
