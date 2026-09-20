"""Two-worker executor for all registered Stage 1.5 formal evaluations."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def worker(jobs: list[tuple[str, str, int]], device: str) -> None:
    logs = ROOT / "results/stage1_5/logs"
    logs.mkdir(parents=True, exist_ok=True)
    for experiment, variant, seed in jobs:
        out = ROOT / "results/stage1_5/stage1_5-formal-v1/evaluation" / experiment / (
            f"{variant}_seed{seed}/summary.json"
        )
        if out.exists():
            summary = json.loads(out.read_text(encoding="utf-8"))
            if summary["experiment"] != experiment or summary["variant"] != variant or summary["seed"] != seed:
                raise ValueError(f"evaluation shard identity mismatch: {out}")
            continue
        command = [
            str(ROOT / ".venv/bin/python"), "experiments/evaluate_stage1_5.py",
            "--experiment", experiment, "--variant", variant,
            "--seed", str(seed), "--device", device,
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (logs / f"eval_{experiment}_{variant}_seed{seed}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError(f"evaluation failed: {experiment}/{variant}/{seed}: {result.stderr[-2000:]}")
        print(f"{experiment} {variant} seed {seed} done", flush=True)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/stage1_5.yaml").read_text(encoding="utf-8"))
    seeds = config["training"]["formal_seeds"]
    base = config["training"]["base_models"]
    jobs: list[tuple[str, str, int]] = []
    jobs += [("baseline", variant, seed) for variant in base for seed in seeds]
    primary = ("stability", "perturbation", "timescale", "anatomy",
               "observability", "mediation", "oracle")
    jobs += [(experiment, "B5_separate", seed) for experiment in primary for seed in seeds]
    capacity_variants = [
        "B5_separate" if (cell["hidden_dim"], cell["memory_dim"]) == (64, 16)
        else f"B5_separate_h{cell['hidden_dim']}_m{cell['memory_dim']}"
        for cell in config["evaluation"]["capacity_grid"]
    ]
    jobs += [("capacity", variant, seed) for variant in capacity_variants for seed in seeds]
    if len(jobs) != 160:
        raise AssertionError(f"unexpected registered job count: {len(jobs)}")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, jobs[index::2], f"cuda:{index}") for index in range(2)]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
