"""Hash inherited Stage 2C artifacts without touching them."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"results/stage2c1/manifests/historical_stage2c_sha256.json"
SOURCES=("configs/stage2c_formal.yaml","reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md",
         "src/etrcm/stage2c/model.py","src/etrcm/stage2c/world.py",
         "src/etrcm/stage2c/rollout.py","experiments/stage2c_train.py",
         "experiments/stage2c_eval.py")


def hashes():
    files=[ROOT/x for x in SOURCES]+sorted((ROOT/"results/stage2c").rglob("*"))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files if p.is_file()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("mode",choices=["snapshot","verify"])
    args=parser.parse_args()
    if args.mode=="snapshot":
        if MANIFEST.exists():raise RuntimeError("Refusing to overwrite baseline manifest")
        data={"stage2c_git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "captured_note":"Captured after L0-L2 but before L3 completion; git tracked files independently checked against HEAD",
              "files":hashes()}
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        MANIFEST.write_text(json.dumps(data,indent=2))
        print(json.dumps({"baseline_files":len(data["files"])}))
    else:
        expected=json.loads(MANIFEST.read_text())["files"]
        actual=hashes()
        assert actual==expected, {"missing":sorted(set(expected)-set(actual)),
                                  "new":sorted(set(actual)-set(expected)),
                                  "changed":sorted(k for k in expected.keys()&actual.keys() if expected[k]!=actual[k])}
        subprocess.run(["git","diff","--exit-code","HEAD","--","results/stage2c",
                        "src/etrcm/stage2c","configs/stage2c_formal.yaml",
                        "reports/STAGE2C_BEHAVIORAL_MEMORY_RESULTS.md"],cwd=ROOT,check=True)
        print(json.dumps({"historical_stage2c_unchanged":True,"file_count":len(actual)}))


if __name__=="__main__":main()
