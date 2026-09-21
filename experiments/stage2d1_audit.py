"""Audit all high-resolution Stage 2D.1 checkpoints."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from etrcm.stage2d1.engine import auxiliary_routing_audit, classify, health_audit
from stage2d_evaluate import load_model


def one(run_dir: Path, out_dir: Path, eval_seed: int, reps: int):
    summary = json.loads((run_dir / "summary.json").read_text()); rows = []
    for checkpoint in sorted(run_dir.glob("checkpoint_*.pt")):
        model, meta = load_model(checkpoint, "cpu")
        health = health_audit(model, eval_seed, reps=reps)
        rows.append({"step": meta["steps"], "health": health, "class": classify(health),
                     "gradients_preceding_update": meta["gradients_preceding_update"]})
    final_model, _ = load_model(run_dir / "checkpoint_1500.pt", "cpu")
    payload = {"run": run_dir.name, "init_seed": summary["init_seed"], "data_seed": summary["data_seed"],
               "eval_seed": eval_seed, "trajectory": rows,
               "auxiliary_final": auxiliary_routing_audit(final_model, eval_seed + 99, reps=max(4,reps//2))}
    out_dir.mkdir(parents=True, exist_ok=True); target = out_dir / f"{run_dir.name}.json"
    target.write_text(json.dumps(payload)); return target


def main(args):
    runs = sorted(p for p in args.checkpoints.iterdir() if p.is_dir() and (p/"summary.json").exists())
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures=[pool.submit(one,r,args.out,args.eval_seed,args.reps) for r in runs]
        for f in as_completed(futures): print("COMPLETED", f.result(), flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--checkpoints",type=Path,required=True); p.add_argument("--out",type=Path,required=True)
    p.add_argument("--eval-seed",type=int,default=15101); p.add_argument("--reps",type=int,default=16); p.add_argument("--jobs",type=int,default=16)
    main(p.parse_args())
