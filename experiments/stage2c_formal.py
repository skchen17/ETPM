"""Run one GPU worker on disjoint preregistered formal training seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


VARIANTS=("full","no_memory","gru","gamma_zero","f_only","m_disabled")
SEEDS=tuple(range(6201,6209))


def run(cmd, log: Path):
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open("w") as handle:
        subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,choices=(0,1),required=True)
    p.add_argument("--root",type=Path,default=Path("results/stage2c"))
    args=p.parse_args()
    root=args.root;device=f"cuda:{args.worker}"
    seeds=SEEDS[:4] if args.worker==0 else SEEDS[4:]
    progress=[]
    for seed in seeds:
        for variant in VARIANTS:
            start=time.monotonic()
            checkpoint_dir=root/"checkpoints"/variant/str(seed)
            raw_dir=root/"raw"/variant/str(seed)
            if checkpoint_dir.exists() or raw_dir.exists():
                raise FileExistsError(f"refusing to overwrite frozen run {variant} {seed}; "
                                      "choose a fresh --root for reproduction")
            train=[sys.executable,"experiments/stage2c_train.py","--variant",variant,
                   "--seed",str(seed),"--steps","500","--batch","16",
                   "--state-penalty","0.001","--device",device,
                   "--out",str(checkpoint_dir),"--log-every","50"]
            evaluate=[sys.executable,"experiments/stage2c_eval.py","--checkpoint",
                      str(checkpoint_dir/"checkpoint.pt"),"--out",str(raw_dir),
                      "--device",device,"--pairs","4"]
            record={"seed":seed,"variant":variant,"worker":args.worker}
            try:
                run(train,root/"manifests"/f"train_{variant}_{seed}.log")
                run(evaluate,root/"manifests"/f"eval_{variant}_{seed}.log")
                if variant=="full":
                    posthoc=evaluate.copy()
                    posthoc[posthoc.index(str(raw_dir))]=str(root/"raw"/"gamma_zero_posthoc"/str(seed))
                    run(posthoc+["--posthoc-gamma-zero"],
                        root/"manifests"/f"eval_gamma_zero_posthoc_{seed}.log")
                record["status"]="completed"
            except subprocess.CalledProcessError as exc:
                record["status"]="failed";record["returncode"]=exc.returncode
            record["elapsed_s"]=time.monotonic()-start
            progress.append(record)
            target=root/"manifests"/f"worker{args.worker}_progress.json"
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(json.dumps(progress,indent=2))
            print(json.dumps(record),flush=True)


if __name__=="__main__":main()
