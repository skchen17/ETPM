#!/usr/bin/env python3
"""Exploratory exact-old-objective B5 scaling; never changes frozen gates."""

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
from etrcm.stage1_4.training import (FAMILIES, prefix_index, run_prefix, seed_everything,
                                      world_loss)
from etrcm.stage1_4.world import generate_world
from etrcm.stage1_5.evaluation import evaluate_oracle
from etrcm.stage1_5.model import AnatomicalETRCM


ROOT=Path(__file__).resolve().parents[1]
CFG=yaml.safe_load((ROOT/"configs/stage1_5.yaml").read_text())
CHECKPOINTS=(0,50,100,160,300,500,1000,2000,3000)


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def run_seed(seed:int,device:torch.device)->None:
    torch.set_num_threads(1)
    seed_everything(seed)
    model=AnatomicalETRCM(Stage14Config.from_mapping(CFG),"B5_separate").to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,
                                weight_decay=CFG["training"]["weight_decay"])
    out=ROOT/"results/stage1_6/legacy_scaling"/f"B5_seed{seed}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True)
    logs=[]; rows=[]; hashes={}
    started=time.monotonic()
    for step in range(3001):
        if step in CHECKPOINTS:
            checkpoint=out/f"checkpoint_{step:04d}.pt"
            torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),
                        "seed":seed,"step":step,"source_config_sha256":sha256(ROOT/"configs/stage1_5.yaml")},checkpoint)
            hashes[checkpoint.name]=sha256(checkpoint)
            model.eval()
            diagnostics=evaluate_oracle(model,seed=seed,run_id="stage1_6-legacy-scaling-exploratory",
                                        device=device,gaps=(128,),batch=32)
            for row in diagnostics:
                row.update({"training_step":step,"training_seed":seed,
                            "exploratory_not_gate":True})
            rows.extend(diagnostics)
            print(json.dumps({"seed":seed,"checkpoint":step,
                              "elapsed_s":round(time.monotonic()-started,1)}),flush=True)
        if step==3000:
            break
        family=FAMILIES[step%len(FAMILIES)]
        world=generate_world(family,batch=32,length=16,
                             seed=seed*1_000_000+step*17+41,device=device)
        prefix=prefix_index(world,step)
        null_ticks=(0,1,2,4)[(step//len(FAMILIES))%4]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        state,_=run_prefix(model,world,prefix=prefix,null_ticks=null_ticks)
        loss,_=world_loss(model,world,state,prefix=prefix)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"legacy scaling nonfinite loss seed={seed} step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CFG["training"]["gradient_clip"])
        optimizer.step()
        if step%50==0:
            logs.append({"seed":seed,"step":step,"family":family,"loss":float(loss.detach())})
    for filename,data in (("oracle_interventions.parquet",rows),("train_log.parquet",logs)):
        path=out/filename
        pd.DataFrame.from_records(data).to_parquet(path,index=False)
        hashes[filename]=sha256(path)
    (out/"summary.json").write_text(json.dumps({"seed":seed,"steps":3000,
                                                "source":"frozen Stage 1.5 four-family objective",
                                                "eval_gap":128,"hashes":hashes},indent=2))


def main()->None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    run_seed(args.seed,torch.device(args.device))


if __name__=="__main__":
    main()
