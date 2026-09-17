#!/usr/bin/env python3
"""Run one or all frozen Stage-1 toy protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from etrcm.toys import run_all_toys, run_named_toy  # noqa: E402


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy", default="all", help="1..9 or all")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/toy_default.yaml")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config = yaml.safe_load(config_bytes)
    count = args.seeds or int(config.get("experiments", {}).get("seeds", 8))
    base_seed = int(config.get("training", {}).get("seed", 42))
    seeds = [base_seed + index for index in range(count)]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("toy-%Y%m%dT%H%M%SZ")
    output_dir = ROOT / "results/raw" / run_id
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    records = (
        run_all_toys(config, seeds)
        if args.toy.lower() == "all"
        else run_named_toy(args.toy, config, seeds)
    )
    frame = pd.DataFrame.from_records(records)
    record_path = output_dir / "toy_records.parquet"
    frame.to_parquet(record_path, index=False, compression="zstd")

    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "toy": args.toy,
        "seed_values": seeds,
        "record_count": len(frame),
        "final_record_count": int((frame["phase"] == "final").sum()),
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "records": str(record_path.relative_to(ROOT)),
        "git_revision": _git_revision(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "results/raw/latest_run.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

