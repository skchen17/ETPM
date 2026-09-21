"""Run final frozen behavior/control/probe audits on two GPUs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

from experiments.stage2c3_run_matrix import ARM_DIR


def one(split,arm,seed,gpu,threshold):
    root=Path("results/stage2c3")
    checkpoint=root/ARM_DIR[arm]/split/str(seed)/"checkpoint_1000.pt"
    out=root/"behavior"/split/arm/str(seed)/"evaluation.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"experiments/stage2c3_evaluate.py","--checkpoint",str(checkpoint),
             "--seed",str(seed),"--device",f"cuda:{gpu}","--bs-threshold",str(threshold),
             "--out",str(out)]
    with (out.parent/"stdout.log").open("w") as stream:
        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    return f"EVALUATED {split} {arm} {seed} GPU{gpu}"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=("development","formal"),required=True)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    p.add_argument("--arms",nargs="+",choices=tuple(ARM_DIR),default=tuple(ARM_DIR))
    p.add_argument("--bs-threshold",type=float,default=.1)
    args=p.parse_args()
    jobs=[(arm,seed) for seed in args.seeds for arm in args.arms]
    queues=[jobs[::2],jobs[1::2]]
    def worker(gpu,queue):
        for arm,seed in queue:print(one(args.split,arm,seed,gpu,args.bs_threshold),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu,queue) for gpu,queue in enumerate(queues)]
        for future in as_completed(futures):future.result()


if __name__=="__main__":main()
