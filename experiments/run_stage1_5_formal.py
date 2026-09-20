"""Run all preregistered fresh formal train cells, two isolated GPU workers."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def worker(jobs: list[tuple[str, int, int, int, float]], device: str) -> None:
    logs = ROOT / "results/stage1_5/logs"
    logs.mkdir(parents=True, exist_ok=True)
    for variant, h, m, seed, lr in jobs:
        output = ROOT / "results/stage1_5/stage1_5-formal-v1" / f"{variant}_lr{lr:g}_seed{seed}/summary.json"
        if output.exists():
            summary = json.loads(output.read_text(encoding="utf-8"))
            if summary["seed"] != seed or summary["variant"] != variant:
                raise ValueError(f"existing formal shard identity mismatch: {output}")
            continue
        command = [
            str(ROOT / ".venv/bin/python"), "experiments/run_stage1_5.py",
            "--mode", "formal", "--model", variant.split("_h")[0],
            "--seed", str(seed), "--lr", str(lr), "--device", device,
            "--hidden-dim", str(h), "--memory-dim", str(m),
        ]
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
        (logs / f"formal_{variant}_seed{seed}.log").write_text(
            result.stdout + result.stderr, encoding="utf-8"
        )
        if result.returncode:
            raise RuntimeError(f"formal cell failed: {variant}/{seed}: {result.stderr[-1000:]}")
        print(f"{variant} seed {seed} done", flush=True)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/stage1_5.yaml").read_text(encoding="utf-8"))
    selection = json.loads((ROOT / "configs/stage1_5_selected.json").read_text(encoding="utf-8"))
    variants = [(model, 64, 16) for model in config["training"]["base_models"]]
    variants += [
        (f"B5_separate_h{cell['hidden_dim']}_m{cell['memory_dim']}", cell["hidden_dim"], cell["memory_dim"])
        for cell in config["evaluation"]["capacity_grid"]
        if (cell["hidden_dim"], cell["memory_dim"]) != (64, 16)
    ]
    jobs = [
        (variant, h, m, seed, selection["learning_rates"][variant])
        for variant, h, m in variants
        for seed in config["training"]["formal_seeds"]
    ]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(worker, jobs[worker_id::2], f"cuda:{worker_id}")
            for worker_id in range(2)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
