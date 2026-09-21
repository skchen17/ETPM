"""Hash and verify all tracked Stage 2C.x/Stage 2D frozen assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def frozen_files():
    files=subprocess.check_output(["git","ls-files"],text=True).splitlines()
    return [p for p in files if ("stage2c" in p.lower() or "stage2d" in p.lower()) and "stage2d1" not in p.lower()]


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(): return {p:digest(p) for p in frozen_files()}


def main(args):
    current=snapshot()
    if args.write:
        args.manifest.parent.mkdir(parents=True,exist_ok=True); args.manifest.write_text(json.dumps(current,indent=2)); print(len(current)); return
    expected=json.loads(args.manifest.read_text()); changed={p:(expected.get(p),current.get(p)) for p in set(expected)|set(current) if expected.get(p)!=current.get(p)}
    result={"expected_files":len(expected),"current_files":len(current),"changed":changed,"pass":not changed}
    if args.out: args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps(result));
    if changed: raise SystemExit(1)


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--manifest",type=Path,required=True); p.add_argument("--write",action="store_true"); p.add_argument("--out",type=Path); main(p.parse_args())
