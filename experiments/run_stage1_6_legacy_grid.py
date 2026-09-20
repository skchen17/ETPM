#!/usr/bin/env python3
"""Two-GPU exploratory old-objective scaling launcher."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
SEEDS=(8701,8702,8703,8704,8705,8706,8707,8708)


def worker(gpu:int)->list[dict]:
    output=ROOT/"results/stage1_6/legacy_scaling"
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in SEEDS[gpu::2]:
        summary=output/f"B5_seed{seed}"/"summary.json"
        if summary.exists():
            rows.append({"seed":seed,"exit_code":0,"existing_complete":True})
            continue
        log=output/f"B5_seed{seed}.log"
        with log.open("w") as handle:
            result=subprocess.run([sys.executable,str(ROOT/"experiments/run_stage1_6_legacy_scaling.py"),
                                   "--seed",str(seed),"--device",f"cuda:{gpu}"],cwd=ROOT,
                                  env=dict(os.environ,PYTHONPATH=str(ROOT/"src")),
                                  stdout=handle,stderr=subprocess.STDOUT)
        rows.append({"seed":seed,"exit_code":result.returncode,"log":str(log.relative_to(ROOT))})
    return rows


def main()->None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(worker,0); b=pool.submit(worker,1)
        rows=a.result()+b.result()
    output=ROOT/"results/stage1_6/legacy_scaling/grid_status.json"
    output.write_text(json.dumps(rows,indent=2))
    print(json.dumps({"cells":len(rows),"failures":[r for r in rows if r["exit_code"]]}))
    if any(r["exit_code"] for r in rows):
        raise SystemExit(1)


if __name__=="__main__":
    main()
