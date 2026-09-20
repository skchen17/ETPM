"""Archive compact pre-schema rows, then replay every fixed formal checkpoint."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT=Path("results/stage2c")
SEEDS=tuple(range(6201,6209))
VARIANTS=("full","no_memory","gru","gamma_zero","f_only","m_disabled")


def key(row):
    return (row["section"],row["exposure_count"],row["delay"],row["probe_type"],
            row["revision_count"],row["useful_noise_condition"],row["intervention"])


def task(variant,seed):
    raw=ROOT/"raw"/variant/str(seed)
    checkpoint=ROOT/"checkpoints"/variant/str(seed)/"checkpoint.pt"
    if variant=="gamma_zero_posthoc":
        checkpoint=ROOT/"checkpoints"/"full"/str(seed)/"checkpoint.pt"
    old=ROOT/"development"/"pre_final_schema_rows"/variant/str(seed)/"rows.jsonl"
    cmd=[sys.executable,"experiments/stage2c_eval.py","--checkpoint",str(checkpoint),
         "--out",str(raw),"--device","cpu","--pairs","4"]
    if variant=="gamma_zero_posthoc":cmd.append("--posthoc-gamma-zero")
    log=ROOT/"manifests"/f"final_schema_reeval_{variant}_{seed}.log"
    with log.open("w") as handle:
        subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,check=True)
    before={key(x):x["BS"] for x in (json.loads(line) for line in old.read_text().splitlines())}
    after={key(x):x["BS"] for x in (json.loads(line) for line in (raw/"rows.jsonl").read_text().splitlines())}
    if before.keys()!=after.keys():raise AssertionError(f"condition set changed {variant} {seed}")
    maximum=max(abs(before[k]-after[k]) for k in before)
    if maximum>1e-6:raise AssertionError(f"behavior changed {variant} {seed}: {maximum}")
    return {"variant":variant,"seed":seed,"max_abs_BS_change":maximum}


def main():
    if Path("reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md").exists():
        raise FileExistsError("formal report already exists; do not replay frozen results")
    target=ROOT/"development"/"pre_final_schema_rows"
    if target.exists():raise FileExistsError(target)
    expected=[(variant,seed) for seed in SEEDS for variant in VARIANTS]
    expected += [("gamma_zero_posthoc",seed) for seed in SEEDS]
    for variant,seed in expected:
        raw=ROOT/"raw"/variant/str(seed)
        checkpoint=ROOT/"checkpoints"/("full" if variant=="gamma_zero_posthoc" else variant)/str(seed)/"checkpoint.pt"
        if not (raw/"rows.jsonl").exists() or not checkpoint.exists():
            raise FileNotFoundError((raw,checkpoint))
    for variant,seed in expected:
        raw=ROOT/"raw"/variant/str(seed)
        dest=target/variant/str(seed)
        dest.mkdir(parents=True,exist_ok=False)
        for name in ("rows.jsonl","manifest.json"):
            shutil.copy2(raw/name,dest/name)
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(task,variant,seed):(variant,seed) for variant,seed in expected}
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            print(json.dumps(result),flush=True)
    manifest={"reason":"uniform final telemetry/metadata schema; weights/gates unchanged",
              "runs":len(results),"max_abs_BS_change":max(x["max_abs_BS_change"] for x in results),
              "archived_rows":str(target)}
    (ROOT/"manifests"/"final_schema_replay.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps({"COMPLETE":manifest}))


if __name__=="__main__":main()
