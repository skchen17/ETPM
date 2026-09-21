"""Eight independent paired-seed runs for the frozen selected curriculum."""

from __future__ import annotations
import argparse,subprocess,sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path


def main(args):
    pairs=list(zip(range(9301,9309),range(12301,12309)))
    def one(pair):
        i,d=pair; out=args.out/f"i{i}_d{d}"; log=args.out/f"i{i}_d{d}.log"; args.out.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"experiments/stage2d1_train.py","--init-seed",str(i),"--data-seed",str(d),"--eval-seed","15501",
             "--curriculum",args.curriculum,"--gate-floor",str(args.gate_floor),"--gate-window",str(args.gate_window),
             "--checkpoints","0","1500","--out",str(out)]
        with log.open("w") as h: subprocess.run(cmd,stdout=h,stderr=subprocess.STDOUT,check=True)
        return pair
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(one,p) for p in pairs]
        for f in as_completed(fs): print("COMPLETED",f.result(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--curriculum",choices=("C1","C2","C3"),required=True)
    p.add_argument("--gate-floor",type=float,default=.65); p.add_argument("--gate-window",type=int,default=100); p.add_argument("--out",type=Path,required=True); main(p.parse_args())
