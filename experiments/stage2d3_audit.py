"""Layerwise factorial-interaction trajectories for Stage 2D.3 cohorts."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

from stage2d_evaluate import load_model
from etrcm.stage2d1.engine import classify, health_audit
from etrcm.stage2d2.geometry import phase_states
from etrcm.stage2d3.interaction import anatomy, finite_cross_interaction


CHECKPOINTS = (0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500)


def one(run: Path, out: Path, eval_seed: int, reps: int) -> Path:
    torch.set_num_threads(1)
    trajectory = []
    metadata = None
    for step in CHECKPOINTS:
        model, metadata = load_model(run / f"checkpoint_{step:04d}.pt", "cpu")
        state = phase_states(model, eval_seed, reps)["post"]
        item = {"step": step, "health": health_audit(model, eval_seed, reps=reps),
                "anatomy": anatomy(model, state, eval_seed, reps)}
        item["class"] = classify(item["health"])
        trajectory.append(item)
    assert metadata is not None
    model, _ = load_model(run / "checkpoint_1500.pt", "cpu")
    final_state = phase_states(model, eval_seed, reps)["post"]
    cross = finite_cross_interaction(model, final_state, eval_seed, reps,
                                     directions=64,
                                     direction_seed=eval_seed + int(metadata["init_seed"]) * 19)
    payload = {"run": run.name, "init_seed": metadata["init_seed"],
               "data_seed": metadata["data_seed"], "eval_seed": eval_seed,
               "final_class": trajectory[-1]["class"], "trajectory": trajectory,
               "cross_interaction": cross}
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{run.name}.json"
    target.write_text(json.dumps(payload))
    return target


def main(args):
    runs = sorted(p for p in args.checkpoints.iterdir()
                  if p.is_dir() and (p / "checkpoint_1500.pt").exists())
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(one, run, args.out, args.eval_seed, args.reps) for run in runs]
        for future in as_completed(futures):
            print("COMPLETED", future.result(), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--eval-seed", type=int, required=True)
    p.add_argument("--reps", type=int, default=16)
    p.add_argument("--jobs", type=int, default=16)
    main(p.parse_args())
