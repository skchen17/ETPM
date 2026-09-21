"""CPU-side frozen audits that follow completed formal training seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from experiments.stage2c3_run_matrix import ARM_DIR


def command(log: Path,argv: list[str]):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w") as stream:
        subprocess.run([sys.executable,*argv],check=True,stdout=stream,stderr=subprocess.STDOUT)


def audit(arm,seed,threshold):
    root=Path("results/stage2c3")
    checkpoint=root/ARM_DIR[arm]/"formal"/str(seed)/"checkpoint_1000.pt"
    behavior=root/"behavior/formal"/arm/str(seed)
    if not (behavior/"evaluation.json").exists():
        command(behavior/"stdout.log",["experiments/stage2c3_evaluate.py",
            "--checkpoint",str(checkpoint),"--seed",str(seed),"--device","cpu",
            "--bs-threshold",str(threshold),"--out",str(behavior/"evaluation.json")])
    ce=root/"learning_curves/heldout_ce/formal"/arm/str(seed)
    if not (ce/"summary.json").exists():
        command(ce/"stdout.log",["experiments/stage2c3_checkpoint_audit.py",
            "--split","formal","--arm",arm,"--seed",str(seed),"--device","cpu"])
    if arm in {"A2","A3"}:
        oracle=root/"learning_curves/oracle_h/formal"/arm/str(seed)
        if not (oracle/"summary.json").exists():
            command(oracle/"stdout.log",["experiments/stage2c3_oracle_curve.py",
                "--split","formal","--arm",arm,"--seed",str(seed),"--device","cpu"])
    return f"POSTPROCESSED formal {arm} {seed}"


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=int,default=4)
    p.add_argument("--max-wait-s",type=int,default=21600)
    args=p.parse_args()
    cfg=yaml.safe_load(Path("configs/stage2c3_formal.yaml").read_text())
    assert cfg["protocol_status"]=="frozen_before_formal_training"
    jobs={(arm,seed) for seed in cfg["formal_training_seeds"] for arm in cfg["formal_arm_priority"]}
    pending=set(jobs);active={};done=set();start=time.monotonic()
    root=Path("results/stage2c3")
    threshold=cfg["gates"]["tau_bs"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while pending or active:
            if time.monotonic()-start>args.max_wait_s:
                raise TimeoutError(f"formal audit timed out; pending={sorted(pending)}")
            for job,future in list(active.items()):
                if future.done():
                    print(future.result(),flush=True)
                    done.add(job);del active[job]
            for arm,seed in sorted(pending,key=lambda x:(x[1],cfg["formal_arm_priority"].index(x[0]))):
                if len(active)>=args.workers:break
                training=root/ARM_DIR[arm]/"formal"/str(seed)/"summary.json"
                if training.exists():
                    active[(arm,seed)]=pool.submit(audit,arm,seed,threshold)
                    pending.remove((arm,seed))
            if pending or active:time.sleep(10)
    assert done==jobs
    print(f"POSTPROCESS_COMPLETE={len(done)}",flush=True)


if __name__=="__main__":main()
