#!/usr/bin/env python3
"""Conditional phase-wise oracle/use curriculum, outside frozen G34–G37."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import pandas as pd
import torch
import yaml

from etrcm.stage1_4.model import Stage14Config
from etrcm.stage1_6.runner import make_model, run_episode
from etrcm.stage1_6.world import generate_world
from run_stage1_6 import evaluate, sha256


ROOT=Path(__file__).resolve().parents[1]
CONFIG=yaml.safe_load((ROOT/"configs/stage1_6.yaml").read_text())
CHECKPOINTS=(0,1000,2000,3000)


def p_oracle(step:int)->float:
    return 1.0 if step<1000 else (0.5 if step<2000 else 0.0)


def choose(step:int,seed:int)->bool:
    digest=hashlib.sha256(f"stage16-auxF-{seed}-{step}".encode()).digest()
    return int.from_bytes(digest[:8],"little")/2**64<p_oracle(step)


def run(seed:int,device:torch.device)->None:
    torch.set_num_threads(1)
    model=make_model(Stage14Config.from_mapping(CONFIG),"curriculum",seed=seed,device=device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    output=ROOT/"results/stage1_6/auxiliary_F"/f"seed{seed}"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    count=sum(p.numel() for p in model.parameters())
    initial=model.initial_state(32,device=device)
    state_bytes=(initial.H.numel()+initial.F.numel()+initial.M.numel())*initial.H.element_size()//32
    rows=[]; logs=[]; hashes={}; started=time.monotonic()
    for step in range(3001):
        if step in CHECKPOINTS:
            checkpoint=output/f"checkpoint_{step:04d}.pt"
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                        "seed":seed,"step":step,"schedule":"oracle_1000_mix_1000_learned_1000",
                        "config_sha256":sha256(ROOT/"configs/stage1_6.yaml")},checkpoint)
            hashes[checkpoint.name]=sha256(checkpoint)
            records=evaluate(model,phase="formal",arm="curriculum",seed=seed,step=step,
                             batch_size=256,device=device,parameter_count=count,
                             state_bytes=state_bytes,run_id="stage1_6-auxiliary-F-v1")
            for row in records:
                row["training_arm"]="auxiliary_F"
                row["curriculum_oracle_probability"]=p_oracle(min(step,2999))
                row["post_trigger_exploratory"]=True
            rows.extend(records)
            print(json.dumps({"seed":seed,"checkpoint":step,
                              "elapsed_s":round(time.monotonic()-started,1)}),flush=True)
        if step==3000:
            break
        world=generate_world(batch=32,seed=1_600_000+seed*10_000+step,device=device)
        condition="oracle" if choose(step,seed) else "learned"
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss,_=run_episode(model,world,condition=condition,random_seed=seed+step)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"aux F nonfinite seed={seed} step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step%50==0:
            logs.append({"seed":seed,"step":step,"loss":float(loss.detach()),
                         "condition":condition,"p_oracle":p_oracle(step),"batch":32})
    for filename,data in (("interventions.parquet",rows),("train_log.parquet",logs)):
        path=output/filename
        pd.DataFrame(data).to_parquet(path,index=False)
        hashes[filename]=sha256(path)
    (output/"summary.json").write_text(json.dumps({"seed":seed,"steps":3000,
        "arm":"auxiliary_F","elapsed_seconds":time.monotonic()-started,
        "hashes":hashes},indent=2))


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--device",default="cpu")
    args=parser.parse_args()
    run(args.seed,torch.device(args.device))


if __name__=="__main__":
    main()
