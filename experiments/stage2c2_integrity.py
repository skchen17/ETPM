"""Read-only inventory of frozen Stage 2C and 2C.1 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/"results/stage2c2/manifests/frozen_prior_sha256.json"
PREFIXES=("results/stage2c/","results/stage2c1/","src/etrcm/stage2c/",
          "src/etrcm/stage2c1/","experiments/stage2c","configs/stage2c",
          "reports/STAGE2C","docs/STAGE2C","tests/test_stage2c")


def tracked():
    paths=subprocess.check_output(["git","ls-files"],cwd=ROOT,text=True).splitlines()
    return [ROOT/p for p in paths if p.startswith(PREFIXES)]


def hashes():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in tracked() if p.is_file()}


def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=["snapshot","verify"]);args=p.parse_args()
    if args.mode=="snapshot":
        if MANIFEST.exists():raise RuntimeError("Historical baseline already exists")
        MANIFEST.parent.mkdir(parents=True,exist_ok=True)
        data={"head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
              "files":hashes()}
        MANIFEST.write_text(json.dumps(data,indent=2))
        print(json.dumps({"snapshot":len(data["files"]),"head":data["head"]}))
    else:
        expected=json.loads(MANIFEST.read_text())["files"]
        actual=hashes()
        assert expected==actual,"Frozen Stage 2C/2C.1 file hash mismatch"
        paths=list(expected)
        subprocess.run(["git","diff","--exit-code","HEAD","--",*paths],cwd=ROOT,check=True)
        subprocess.run(["git","diff","--exit-code","HEAD","--","README.md"],cwd=ROOT,check=True)
        print(json.dumps({"unchanged":True,"count":len(expected)}))


if __name__=="__main__":main()
