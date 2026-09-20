#!/usr/bin/env python3
"""Eight independent CPU workers for the post-trigger auxiliary F curriculum."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT=Path(__file__).resolve().parents[1]
SEEDS=range(8701,8709)


def cell(seed:int)->dict:
    out=ROOT/"results/stage1_6/auxiliary_F"
    out.mkdir(parents=True,exist_ok=True)
    summary=out/f"seed{seed}"/"summary.json"
    if summary.exists():
        return {"seed":seed,"exit_code":0,"existing_complete":True}
    log=out/f"seed{seed}.log"
    with log.open("w") as handle:
        result=subprocess.run([sys.executable,str(ROOT/"experiments/run_stage1_6_aux_F.py"),
                               "--seed",str(seed),"--device","cpu"],cwd=ROOT,
                              env=dict(os.environ,PYTHONPATH=str(ROOT/"src")),
                              stdout=handle,stderr=subprocess.STDOUT)
    return {"seed":seed,"exit_code":result.returncode,"log":str(log.relative_to(ROOT))}


def main()->None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(cell,SEEDS))
    (ROOT/"results/stage1_6/auxiliary_F/grid_status.json").write_text(json.dumps(rows,indent=2))
    print(json.dumps({"cells":len(rows),"failures":[r for r in rows if r["exit_code"]]}))
    if any(r["exit_code"] for r in rows):
        raise SystemExit(1)


if __name__=="__main__":
    main()
