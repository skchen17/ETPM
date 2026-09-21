"""Run a seed/arm matrix on two GPUs with per-run machine-readable records."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ARM_DIR={"A0":"a0_joint","A1":"a1_counterfactual",
         "A2":"a2_frozen_evaluator","A3":"a3_gradual_unfreeze"}


def one(root: Path, split: str, arm: str, seed: int, gpu: int, steps: int,
        pretrain_steps: int) -> str:
    out=root/ARM_DIR[arm]/split/str(seed)
    out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"experiments/stage2c3_train.py","--arm",arm,
             "--seed",str(seed),"--device",f"cuda:{gpu}","--steps",str(steps),
             "--evaluator-pretrain-steps",str(pretrain_steps),"--out",str(out)]
    with (out/"stdout.log").open("w") as stream:
        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    return f"COMPLETED {split} {arm} {seed} GPU{gpu}"


def run(args):
    root=Path("results/stage2c3")
    jobs=[(arm,seed) for seed in args.seeds for arm in args.arms]
    # At most one active process per GPU; each worker owns a disjoint queue.
    queues=[jobs[::2],jobs[1::2]]
    def worker(gpu,queue):
        done=[]
        for arm,seed in queue:
            result=one(root,args.split,arm,seed,gpu,args.steps,args.pretrain_steps)
            print(result,flush=True)
            done.append(result)
        return done
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,gpu,queue) for gpu,queue in enumerate(queues)]
        for future in as_completed(futures):future.result()


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=("development","formal"),required=True)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    p.add_argument("--arms",nargs="+",choices=tuple(ARM_DIR),default=tuple(ARM_DIR))
    p.add_argument("--steps",type=int,default=1000)
    p.add_argument("--pretrain-steps",type=int,default=1000)
    run(p.parse_args())
