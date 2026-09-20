#!/usr/bin/env python3
"""Independent post-run integrity, pairing and historical-immutability audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd
import yaml


ROOT=Path(__file__).resolve().parents[1]
CONFIG=yaml.safe_load((ROOT/"configs/stage1_6.yaml").read_text())
RUN=CONFIG["protocol"]["formal_run_id"]


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def main()->None:
    base=ROOT/"results/stage1_6/raw"/RUN
    expected_seeds=CONFIG["training"]["formal_seeds"]
    expected_steps=CONFIG["training"]["checkpoints"]
    cells=0
    for arm in CONFIG["training"]["arms"]:
        for seed in expected_seeds:
            cell=base/f"{arm}-seed{seed}-lr0.001"
            summary=json.loads((cell/"summary.json").read_text())
            assert summary["arm"]==arm and summary["seed"]==seed and summary["steps"]==3000
            assert all(sha256(cell/name)==digest for name,digest in summary["hashes"].items())
            records=pd.read_parquet(cell/"interventions.parquet")
            assert set(records.training_step)==set(expected_steps)
            assert set(records.episode)==set(range(CONFIG["evaluation"]["episodes_per_seed"]))
            expected_conditions=(set(CONFIG["evaluation"]["intervention_conditions"])
                                 if arm in {"learned","oracle","curriculum"}
                                 else {"learned","M_lesion","F_lesion"})
            assert set(records.condition)==expected_conditions
            assert bool(records.H_scrub.all())
            assert records.run_id.eq(RUN).all()
            assert records.seed.eq(seed).all()
            assert records.training_arm.eq(arm).all()
            cells+=1
    changed=subprocess.check_output(["git","diff","--name-only",CONFIG["protocol"]["frozen_parent"],"--"],
                                    cwd=ROOT,text=True).splitlines()
    assert all("stage1_6" in path or "STAGE1_6" in path or path=="README.md" for path in changed),changed
    frozen=subprocess.run(["sha256sum","-c","artifacts/stage1_5_all_assets.sha256"],
                          cwd=ROOT,capture_output=True,text=True)
    assert frozen.returncode==0,frozen.stderr
    processed=ROOT/"results/stage1_6/processed"/RUN
    manifest=json.loads((processed/"integrity.json").read_text())
    assert all(sha256(ROOT/item["path"])==item["sha256"] for item in manifest)
    print(json.dumps({"cells":cells,"checkpoints_per_cell":len(expected_steps),
                      "historical_stage1_5_hashes":"PASS","stage1_6_manifest_files":len(manifest),
                      "tracked_parent_diff_scoped":"PASS"}))


if __name__=="__main__":
    main()
