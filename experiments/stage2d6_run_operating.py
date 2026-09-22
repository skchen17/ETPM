"""Parallel frozen operating-point interventions."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

from stage2d6_operating_point import one


def main(args):
    folder=args.root/"fusion_anatomy"/args.cohort
    paths=sorted(folder.glob("*.json"))
    if args.mode=="grid":
        paths=[p for p in paths if json.loads(p.read_text())["endpoint"]["class"]!="healthy"]
    norms=json.loads(Path("results/stage2d5/processed/healthy_norms.json").read_text())
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures=[]
        for path in paths:
            name="development_grid" if args.mode=="grid" else "formal"
            out=args.root/"operating_point_rescue"/name/args.cohort/path.name
            if args.skip_existing and out.exists():
                continue
            futures.append(pool.submit(one,path,out,mode=args.mode,
                offset_scale=args.offset_scale,k=args.k,reps=args.reps,healthy_norms=norms))
        for future in as_completed(futures):
            r=future.result();print("COMPLETE",r["cohort"],r["run"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--cohort",required=True)
    p.add_argument("--mode",choices=("grid","formal"),required=True)
    p.add_argument("--offset-scale",type=float,default=.5)
    p.add_argument("--k",type=int,default=8)
    p.add_argument("--reps",type=int,default=16)
    p.add_argument("--jobs",type=int,default=12)
    p.add_argument("--skip-existing",action="store_true")
    main(p.parse_args())
