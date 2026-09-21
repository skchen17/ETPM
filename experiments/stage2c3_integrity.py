"""SHA-256 guard for immutable Stage 2C, 2C.1, and 2C.2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"results/stage2c3/manifests/frozen_prior_sha256.json"
PREFIXES=("results/stage2c/","results/stage2c1/","results/stage2c2/",
          "src/etrcm/stage2c/","src/etrcm/stage2c1/","src/etrcm/stage2c2/",
          "experiments/stage2c","configs/stage2c","reports/STAGE2C",
          "docs/STAGE2C","tests/test_stage2c")


def inventory(paths=None):
    if paths is None and MANIFEST.exists():
        paths=json.loads(MANIFEST.read_text())["files"]
    files=(subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines()
           if paths is None else paths)
    return {name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in files if name.startswith(PREFIXES) and (ROOT/name).is_file()}


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("snapshot","verify"))
    args=p.parse_args()
    if args.mode=="snapshot":
        if MANIFEST.exists():raise RuntimeError("snapshot already exists")
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        data={"head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "files":inventory()}
        MANIFEST.write_text(json.dumps(data,indent=2))
        print(json.dumps({"snapshot_count":len(data["files"]),"head":data["head"]}))
    else:
        expected=json.loads(MANIFEST.read_text())["files"]
        actual=inventory(expected)
        assert expected==actual,"historical artifact changed"
        subprocess.run(["git","diff","--exit-code","HEAD","--","README.md",*expected],cwd=ROOT,check=True)
        print(json.dumps({"unchanged":True,"count":len(expected)}))


if __name__=="__main__":main()
