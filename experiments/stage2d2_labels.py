"""Freeze confirmatory basin labels at the Stage 2D.1 evaluation resolution."""
from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import torch
from stage2d_evaluate import load_model
from etrcm.stage2d1.engine import classify,health_audit

def one(run,seed):
    torch.set_num_threads(1); model,_=load_model(run/"checkpoint_1500.pt","cpu"); h=health_audit(model,seed,reps=16)
    return {"run":run.name,"class":classify(h),"health":h}

def main(args):
    runs=sorted(p for p in args.checkpoints.iterdir() if p.is_dir())
    with ProcessPoolExecutor(max_workers=16) as pool: rows=[f.result() for f in as_completed([pool.submit(one,r,args.eval_seed) for r in runs])]
    args.out.write_text(json.dumps({"runs":rows})); print({c:sum(x["class"]==c for x in rows) for c in ("healthy","partial","shortcut")})

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--checkpoints",type=Path,required=True);p.add_argument("--eval-seed",type=int,required=True);p.add_argument("--out",type=Path,required=True);main(p.parse_args())
