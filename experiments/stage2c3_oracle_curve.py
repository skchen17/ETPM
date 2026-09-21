"""Frozen native-norm oracle-H capacity sweep across A2/A3 checkpoints."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from etrcm.stage2c3.protocol import CHECKPOINTS
from experiments.stage2c1_lifetime import paired_state
from experiments.stage2c2_pathway import fit_oracle_h,parameter_hash
from experiments.stage2c3_evaluate import load
from experiments.stage2c3_run_matrix import ARM_DIR
from experiments.stage2c3_train import head_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--arm",choices=("A2","A3"),required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--split",choices=("development","formal"),required=True)
    p.add_argument("--device",default="cuda:0")
    args=p.parse_args();torch.set_num_threads(1)
    root=Path("results/stage2c3")
    source=root/ARM_DIR[args.arm]/args.split/str(args.seed)
    rows=[]
    for step in CHECKPOINTS:
        model,data=load(source/f"checkpoint_{step}.pt",args.device)
        before=parameter_hash(model)
        norms=[]
        with torch.no_grad():
            for rep in range(4):
                state=paired_state(model,args.seed*101+rep+step*100_000+321,16,args.device)
                norms.extend(float(v) for v in state.H.norm(dim=(-2,-1)))
        target=statistics.median(norms)
        fitted=fit_oracle_h(model,args.seed*13+step+8_000_000,target,args.device,
                            steps=500,restarts=2)
        fitted.pop("state_tensors")
        assert before==parameter_hash(model)
        rows.append({"step":step,"native_H_norm_median":target,"head_hash":head_hash(model),
                     "parameter_sha256_before":before,"parameter_sha256_after":parameter_hash(model),
                     "oracle_H":fitted})
        print(json.dumps({"arm":args.arm,"seed":args.seed,"step":step,
                          "oracle_tv":fitted["metrics"]["tv_action"]}),flush=True)
    protected=rows[0]["head_hash"]
    assert all(row["head_hash"]==protected for row in rows if args.arm=="A2" or row["step"]<=500)
    out=root/"learning_curves"/"oracle_h"/args.split/args.arm/str(args.seed)/"summary.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps({"arm":args.arm,"seed":args.seed,"rows":rows},indent=2))


if __name__=="__main__":main()
