"""Repair only the documented prefix-balance evaluation bug from fixed checkpoints."""

import subprocess
import sys
from pathlib import Path

root=Path("results/stage2c")
archive=root/"development"/"prefix_imbalance_eval"
for old_rows in sorted(archive.glob("*/*/rows.jsonl")):
    variant,seed=old_rows.parent.parent.name,old_rows.parent.name
    checkpoint_variant="full" if variant=="gamma_zero_posthoc" else variant
    checkpoint=root/"checkpoints"/checkpoint_variant/seed/"checkpoint.pt"
    out=root/"raw"/variant/seed
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    cmd=[sys.executable,"experiments/stage2c_eval.py","--checkpoint",str(checkpoint),
         "--out",str(out),"--device","cpu","--pairs","4"]
    if variant=="gamma_zero_posthoc":cmd.append("--posthoc-gamma-zero")
    log=root/"manifests"/f"prefix_balanced_reeval_{variant}_{seed}.log"
    with log.open("w") as handle:
        subprocess.run(cmd,check=True,stdout=handle,stderr=subprocess.STDOUT)
    print(f"REPAIRED {variant} {seed}",flush=True)
