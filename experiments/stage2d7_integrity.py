"""Hash tracked Stage 2C–2D.6 assets before/after this additive stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def selected(path):
    return any(part.startswith("stage2c") or
               (part.startswith("stage2d") and not part.startswith("stage2d7"))
               for part in Path(path).parts)


def snapshot():
    files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in files if name and selected(name)}


def main(args):
    current = snapshot()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.baseline:
        old = json.loads(args.baseline.read_text())["files"]
        missing = sorted(set(old)-set(current))
        changed = sorted(name for name, value in old.items()
                         if name in current and current[name] != value)
        extra = sorted(set(current)-set(old))
        payload = {"baseline": str(args.baseline), "count": len(old),
                   "missing": missing, "changed": changed, "extra": extra,
                   "unchanged": not missing and not changed and not extra}
        args.out.write_text(json.dumps(payload, indent=2))
        print(json.dumps({k: payload[k] for k in ("count", "unchanged", "missing", "changed", "extra")}))
        if not payload["unchanged"]:
            raise SystemExit(1)
    else:
        args.out.write_text(json.dumps({"files": current}, indent=2))
        print("HISTORICAL_BASELINE", len(current))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    main(p.parse_args())
