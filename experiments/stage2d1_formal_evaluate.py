"""Parallel full frozen Stage 2D evaluation for the selected curriculum."""

from __future__ import annotations
import argparse,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path


def main(args):
    runs=sorted(p for p in args.checkpoints.iterdir() if p.is_dir())
    def one(run):
        seed=int(run.name.split("_d")[1]); out=args.out/f"{run.name}.json"; log=args.out/f"{run.name}.log"; args.out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"experiments/stage2d_evaluate.py","--checkpoint",str(run/"checkpoint_1500.pt"),
             "--seed",str(seed+50000),"--replicates",str(args.replicates),"--scope","full","--out",str(out)]
        with log.open("w") as h: subprocess.run(cmd,stdout=h,stderr=subprocess.STDOUT,check=True)
        return run.name
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        fs=[pool.submit(one,r) for r in runs]
        for f in as_completed(fs): print("COMPLETED",f.result(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoints",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--replicates",type=int,default=32); p.add_argument("--jobs",type=int,default=8); main(p.parse_args())
