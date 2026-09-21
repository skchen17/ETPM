"""Parallel held-out CE audits for all eight checkpoints and all arms."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

from experiments.stage2c3_run_matrix import ARM_DIR


def one(split,arm,seed,gpu):
    out=Path("results/stage2c3/learning_curves/heldout_ce")/split/arm/str(seed)
    out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"experiments/stage2c3_checkpoint_audit.py","--split",split,
             "--arm",arm,"--seed",str(seed),"--device",f"cuda:{gpu}"]
    with (out/"stdout.log").open("w") as stream:
        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    return f"CE_AUDITED {split} {arm} {seed} GPU{gpu}"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=("development","formal"),required=True)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    p.add_argument("--arms",nargs="+",choices=tuple(ARM_DIR),default=tuple(ARM_DIR))
    args=p.parse_args()
    jobs=[(arm,seed) for seed in args.seeds for arm in args.arms]
    queues=[jobs[::2],jobs[1::2]]
    def worker(gpu,queue):
        for arm,seed in queue:print(one(args.split,arm,seed,gpu),flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(worker,gpu,queue) for gpu,queue in enumerate(queues)]):
            future.result()


if __name__=="__main__":main()
