"""Create/verify immutable historical Stage 2C.x SHA-256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def historical_files(root: Path) -> list[Path]:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    keys = ("stage2c", "stage2c1", "stage2c2", "stage2c3", "STAGE2C")
    return [root / item for item in tracked if any(key in item for key in keys)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): digest(path) for path in historical_files(root)}


def main(args):
    current = snapshot(args.root)
    if args.mode == "create":
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps({"count": len(current), "files": current}, indent=2))
        print(json.dumps({"created": str(args.manifest), "count": len(current)}))
        return
    expected = json.loads(args.manifest.read_text())["files"]
    changed = sorted(key for key in set(expected) | set(current) if expected.get(key) != current.get(key))
    print(json.dumps({"historical_unchanged": not changed, "count": len(current), "changed": changed}, indent=2))
    if changed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mode", choices=("create", "verify"), required=True)
    main(parser.parse_args())
