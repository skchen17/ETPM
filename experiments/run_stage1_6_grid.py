#!/usr/bin/env python3
"""Two-GPU bounded grid runner; per-cell logs and failure manifest."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/stage1_6.yaml").read_text())
ARMS = CONFIG["training"]["arms"]


def cell(phase: str, arm: str, seed: int, lr: float, gpu: int) -> dict:
    log_dir = ROOT / "results/stage1_6/processed" / CONFIG["protocol"][f"{phase}_run_id"]
    log_dir.mkdir(parents=True, exist_ok=True)
    cell_id = f"{arm}-seed{seed}-lr{lr:g}"
    log = log_dir / f"{cell_id}.log"
    summary = ROOT / "results/stage1_6/raw" / CONFIG["protocol"][f"{phase}_run_id"] / cell_id / "summary.json"
    if summary.exists():
        return {"cell": cell_id, "gpu": gpu, "exit_code": 0,
                "log": str(log.relative_to(ROOT)), "existing_complete": True}
    command = [sys.executable, str(ROOT / "experiments/run_stage1_6.py"),
               "--phase", phase, "--arm", arm, "--seed", str(seed),
               "--learning-rate", str(lr), "--device", f"cuda:{gpu}"]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=env,
                                stdout=handle, stderr=subprocess.STDOUT)
    return {"cell": cell_id, "gpu": gpu, "exit_code": result.returncode,
            "log": str(log.relative_to(ROOT))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "formal"), required=True)
    parser.add_argument("--learning-rate", type=float, help="required for formal")
    args = parser.parse_args()
    if args.phase == "formal" and args.learning_rate is None:
        parser.error("formal phase requires selected development learning rate")
    seeds = CONFIG["training"][f"{args.phase}_seeds"]
    rates = (CONFIG["training"]["learning_rate_candidates"] if args.phase == "development"
             else [args.learning_rate])
    grid = [(arm, seed, lr) for seed in seeds for arm in ARMS for lr in rates]
    # Each worker owns one GPU and executes its queue serially.
    buckets = [grid[::2], grid[1::2]]

    def worker(gpu: int) -> list[dict]:
        return [cell(args.phase, arm, seed, lr, gpu) for arm, seed, lr in buckets[gpu]]

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu) for gpu in (0, 1)]
        outcomes = []
        for future in as_completed(futures):
            outcomes.extend(future.result())
            print(json.dumps({"worker_complete": len(outcomes), "total": len(grid)}), flush=True)
    output = ROOT / "results/stage1_6/processed" / CONFIG["protocol"][f"{args.phase}_run_id"] / "grid_status.json"
    output.write_text(json.dumps(outcomes, indent=2))
    failures = [row for row in outcomes if row["exit_code"] != 0]
    print(json.dumps({"phase": args.phase, "cells": len(outcomes), "failures": failures}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
