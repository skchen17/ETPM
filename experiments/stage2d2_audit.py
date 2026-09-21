"""Finite-subspace trajectory audit for Stage 2D.2 cohorts."""

from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import torch
from stage2d_evaluate import load_model
from etrcm.stage2d1.engine import classify,health_audit
from etrcm.stage2d2.geometry import alignment_audit,finite_behavioral_subspace,finite_controllability,phase_states
from etrcm.stage2d2.protocol import CHECKPOINTS


def one(run,out,eval_seed,reps,directions):
    torch.set_num_threads(1)
    rows=[]
    for step in CHECKPOINTS:
        model,meta=load_model(run/f"checkpoint_{step:04d}.pt","cpu")
        states=phase_states(model,eval_seed,reps); sub=finite_behavioral_subspace(model,states["post"],eval_seed,reps,
            directions=directions,direction_seed=eval_seed+step+meta.get("init_seed",meta["seed"])*17)
        item={"step":step,"health":health_audit(model,eval_seed,reps=reps),"subspace":sub,
              "alignment":alignment_audit(model,sub,eval_seed,reps)}
        item["class"]=classify(item["health"])
        if step in (0,25): item["controllability"]=finite_controllability(model,sub,eval_seed,reps=max(2,reps//2))
        rows.append(item)
    final_class=rows[-1]["class"]
    payload={"run":run.name,"init_seed":meta.get("init_seed",meta["seed"]),"data_seed":meta.get("data_seed"),
             "eval_seed":eval_seed,"final_class":final_class,"trajectory":rows}
    out.mkdir(parents=True,exist_ok=True); target=out/f"{run.name}.json"; target.write_text(json.dumps(payload)); return target


def main(args):
    torch.set_num_threads(1); runs=sorted(p for p in args.checkpoints.iterdir() if p.is_dir() and (p/"checkpoint_1500.pt").exists())
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        fs=[pool.submit(one,r,args.out,args.eval_seed,args.reps,args.directions) for r in runs]
        for f in as_completed(fs): print("COMPLETED",f.result(),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoints",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--eval-seed",type=int,required=True); p.add_argument("--reps",type=int,default=8); p.add_argument("--directions",type=int,default=128); p.add_argument("--jobs",type=int,default=16); main(p.parse_args())
