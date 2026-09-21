"""Resume-safe CPU scheduler for a frozen formal training matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path

import yaml

from experiments.stage2c3_run_matrix import ARM_DIR


def one(arm,seed,steps,pretrain_steps):
    out=Path("results/stage2c3")/ARM_DIR[arm]/"formal"/str(seed)
    summary=out/"summary.json"
    if summary.exists():return f"SKIPPED_COMPLETED formal {arm} {seed}"
    out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,"experiments/stage2c3_train.py","--arm",arm,
             "--seed",str(seed),"--device","cpu","--steps",str(steps),
             "--evaluator-pretrain-steps",str(pretrain_steps),"--out",str(out)]
    with (out/"stdout.log").open("w") as stream:
        subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    return f"COMPLETED formal {arm} {seed} CPU"


def main():
    p=argparse.ArgumentParser();p.add_argument("--workers",type=int,default=8)
    args=p.parse_args()
    cfg=yaml.safe_load(Path("configs/stage2c3_formal.yaml").read_text())
    assert cfg["protocol_status"]=="frozen_before_formal_training"
    jobs=[(arm,seed) for seed in cfg["formal_training_seeds"] for arm in cfg["formal_arm_priority"]]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,arm,seed,cfg["training"]["endogenous_steps"],
                             cfg["training"]["evaluator_pretrain_steps_A2_A3"])
                 for arm,seed in jobs]
        for future in as_completed(futures):print(future.result(),flush=True)


if __name__=="__main__":main()
