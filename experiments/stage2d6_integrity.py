"""Snapshot/verify tracked frozen Stage 2C--2D.5 scientific assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def snapshot():
    files = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    selected = [p for p in files if any(k in p.lower() for k in ("stage2c", "stage2d"))
                and "stage2d6" not in p.lower()]
    return {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in selected}


def main(args):
    current=snapshot(); args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.mode == "snapshot":
        args.out.write_text(json.dumps(current, indent=2)); print(json.dumps({"files":len(current)})); return
    expected=json.loads(args.manifest.read_text())
    changed={p:(expected.get(p),current.get(p)) for p in set(expected)|set(current)
             if expected.get(p)!=current.get(p)}
    result={"expected":len(expected),"current":len(current),"changed":changed,"pass":not changed}
    args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
    raise SystemExit(bool(changed))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("snapshot","verify"),required=True)
    p.add_argument("--manifest",type=Path)
    p.add_argument("--out",type=Path,required=True)
    main(p.parse_args())
