"""Resume-safe parallel Stage 2D training/evaluation matrix."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from pathlib import Path

from etrcm.stage2d.protocol import COARSE_CONFIGS


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/stage2d"
PRIMARY = tuple(range(7801, 7809))
CONTROLS = tuple(range(7801, 7806))
DEV_SWEEP = (7621, 7622)


def run(command: list[str], log: Path, expected: Path):
    if expected.exists() and expected.stat().st_size:
        return {"status": "SKIP", "expected": str(expected)}
    log.parent.mkdir(parents=True, exist_ok=True)
    expected.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        completed = subprocess.run(command, cwd=ROOT, stdout=stream,
                                   stderr=subprocess.STDOUT, text=True)
    if completed.returncode or not expected.exists():
        raise RuntimeError(f"job failed rc={completed.returncode}; see {log}")
    return {"status": "DONE", "expected": str(expected)}


def train_command(arm, variant, seed, out, gamma=.12, rf=.97, rm=.9995):
    return [sys.executable, "experiments/stage2d_train.py", "--arm", arm,
            "--variant", variant, "--seed", str(seed), "--steps", "1500",
            "--evaluator-steps", "1000", "--batch", "16", "--train-p", ".70",
            "--lengths", "16,32,64", "--gamma", str(gamma), "--rho-fast", str(rf),
            "--rho-slow", str(rm), "--device", "cpu", "--out", str(out)]


def formal_train_jobs():
    jobs = []
    for seed in PRIMARY:
        out = RESULTS / "checkpoints/formal/primary" / str(seed)
        jobs.append((train_command("A2", "full", seed, out),
                     RESULTS / f"logs/formal_train_primary_{seed}.log", out / "checkpoint.pt"))
    controls = (("a0", "A0", "full"), ("gamma_zero", "A2", "gamma_zero"),
                ("f_only", "A2", "f_only"), ("no_memory", "A2", "no_memory"),
                ("gru", "A2", "gru"))
    for label, arm, variant in controls:
        for seed in CONTROLS:
            out = RESULTS / "checkpoints/formal/controls" / label / str(seed)
            jobs.append((train_command(arm, variant, seed, out),
                         RESULTS / f"logs/formal_train_{label}_{seed}.log", out / "checkpoint.pt"))
    return jobs


def sweep_train_jobs():
    jobs = []
    for name, gamma, rf, rm in COARSE_CONFIGS:
        for seed in DEV_SWEEP:
            out = RESULTS / "checkpoints/sweep" / name / str(seed)
            jobs.append((train_command("A2", "full", seed, out, gamma, rf, rm),
                         RESULTS / f"logs/sweep_train_{name}_{seed}.log", out / "checkpoint.pt"))
    return jobs


def eval_command(checkpoint, seed, scope, out, reps):
    return [sys.executable, "experiments/stage2d_evaluate.py", "--checkpoint", str(checkpoint),
            "--seed", str(seed), "--replicates", str(reps), "--scope", scope,
            "--device", "cpu", "--out", str(out)]


def formal_eval_jobs():
    jobs = []
    for seed in PRIMARY:
        checkpoint = RESULTS / "checkpoints/formal/primary" / str(seed) / "checkpoint.pt"
        out = RESULTS / "processed/formal/primary" / f"{seed}.json"
        jobs.append((eval_command(checkpoint, seed, "full", out, 32),
                     RESULTS / f"logs/formal_eval_primary_{seed}.log", out))
    for label in ("a0", "gamma_zero", "f_only", "no_memory", "gru"):
        for seed in CONTROLS:
            checkpoint = RESULTS / "checkpoints/formal/controls" / label / str(seed) / "checkpoint.pt"
            out = RESULTS / "processed/formal/controls" / label / f"{seed}.json"
            scope = "full" if label == "gamma_zero" else "control"
            jobs.append((eval_command(checkpoint, seed, scope, out, 24),
                         RESULTS / f"logs/formal_eval_{label}_{seed}.log", out))
    return jobs


def sweep_eval_jobs():
    jobs = []
    for name, _, _, _ in COARSE_CONFIGS:
        for seed in DEV_SWEEP:
            checkpoint = RESULTS / "checkpoints/sweep" / name / str(seed) / "checkpoint.pt"
            out = RESULTS / "parameter_sweeps" / name / f"{seed}.json"
            jobs.append((eval_command(checkpoint, seed, "phase", out, 8),
                         RESULTS / f"logs/sweep_eval_{name}_{seed}.log", out))
    return jobs


def shortlist_train_jobs():
    jobs=[]; name="g050_f0970_m09995"
    for seed in PRIMARY:
        out=RESULTS/"checkpoints/formal/shortlist"/name/str(seed)
        jobs.append((train_command("A2","full",seed,out,.50,.97,.9995),
                     RESULTS/f"logs/shortlist_train_{name}_{seed}.log",out/"checkpoint.pt"))
    return jobs


def shortlist_eval_jobs():
    jobs=[]; name="g050_f0970_m09995"
    for seed in PRIMARY:
        checkpoint=RESULTS/"checkpoints/formal/shortlist"/name/str(seed)/"checkpoint.pt"
        out=RESULTS/"processed/formal/shortlist"/name/f"{seed}.json"
        jobs.append((eval_command(checkpoint,seed,"full",out,32),
                     RESULTS/f"logs/shortlist_eval_{name}_{seed}.log",out))
    return jobs


def shortlist_control_train_jobs():
    jobs=[]; name="f_only_g050"
    for seed in CONTROLS:
        out=RESULTS/"checkpoints/formal/controls"/name/str(seed)
        jobs.append((train_command("A2","f_only",seed,out,.50,.97,.9995),
                     RESULTS/f"logs/shortlist_train_{name}_{seed}.log",out/"checkpoint.pt"))
    return jobs


def shortlist_control_eval_jobs():
    jobs=[]; name="f_only_g050"
    for seed in CONTROLS:
        checkpoint=RESULTS/"checkpoints/formal/controls"/name/str(seed)/"checkpoint.pt"
        out=RESULTS/"processed/formal/controls"/name/f"{seed}.json"
        jobs.append((eval_command(checkpoint,seed,"control",out,24),
                     RESULTS/f"logs/shortlist_eval_{name}_{seed}.log",out))
    return jobs


def main(args):
    mapping = {"formal-train": formal_train_jobs, "formal-eval": formal_eval_jobs,
               "sweep-train": sweep_train_jobs, "sweep-eval": sweep_eval_jobs,
               "shortlist-train": shortlist_train_jobs, "shortlist-eval": shortlist_eval_jobs,
               "shortlist-control-train": shortlist_control_train_jobs,
               "shortlist-control-eval": shortlist_control_eval_jobs}
    jobs = mapping[args.phase]()
    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, *job): job for job in jobs}
        for future in cf.as_completed(futures):
            value = future.result(); rows.append(value); print(json.dumps(value), flush=True)
    print(json.dumps({"phase": args.phase, "jobs": len(rows),
                      "done": sum(x["status"] == "DONE" for x in rows),
                      "skipped": sum(x["status"] == "SKIP" for x in rows)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("formal-train", "formal-eval", "sweep-train", "sweep-eval",
                                          "shortlist-train", "shortlist-eval",
                                          "shortlist-control-train", "shortlist-control-eval"))
    parser.add_argument("--workers", type=int, default=32)
    main(parser.parse_args())
