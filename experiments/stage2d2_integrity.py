"""Verify all tracked Stage 2C/2D/2D.1 assets against the pre-stage manifest."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
def snapshot():
    files=subprocess.check_output(["git","ls-files"],text=True).splitlines()
    files=[p for p in files if any(k in p.lower() for k in ("stage2c","stage2d","stage2d1")) and "stage2d2" not in p.lower()]
    return {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
def main(args):
    expected=json.loads(args.manifest.read_text());current=snapshot();changed={p:(expected.get(p),current.get(p)) for p in set(expected)|set(current) if expected.get(p)!=current.get(p)}
    result={"expected":len(expected),"current":len(current),"changed":changed,"pass":not changed};args.out.write_text(json.dumps(result,indent=2));print(json.dumps(result));raise SystemExit(bool(changed))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--out",type=Path,required=True);main(p.parse_args())
