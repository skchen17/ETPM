"""Matched two-seed development screen for C1/C2/C3."""

from __future__ import annotations

import argparse, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main(args):
    tasks=[]
    for seed,data in ((9201,12201),(9202,12202)):
        tasks += [("C1",.65,100,seed,data),("C3",.65,100,seed,data)]
        for floor in (.5,.65,.75):
            for window in (50,100,200): tasks.append(("C2",floor,window,seed,data))
    def one(t):
        curriculum,floor,window,seed,data=t; name=f"{curriculum}_f{floor:.2f}_w{window}_i{seed}_d{data}"
        out=args.out/name; log=args.out/f"{name}.log"; args.out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"experiments/stage2d1_train.py","--init-seed",str(seed),"--data-seed",str(data),
             "--eval-seed",str(args.eval_seed),"--curriculum",curriculum,"--gate-floor",str(floor),
             "--gate-window",str(window),"--checkpoints","0","1500","--out",str(out)]
        with log.open("w") as h: subprocess.run(cmd,stdout=h,stderr=subprocess.STDOUT,check=True)
        return name
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        fs=[pool.submit(one,t) for t in tasks]
        for f in as_completed(fs): print("COMPLETED",f.result(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); p.add_argument("--eval-seed",type=int,default=15301)
    p.add_argument("--jobs",type=int,default=22); main(p.parse_args())
