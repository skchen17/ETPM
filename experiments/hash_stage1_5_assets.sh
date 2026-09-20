#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
manifest="artifacts/stage1_5_all_assets.sha256"
if [[ -e "$manifest" ]]; then
  echo "refusing to overwrite immutable manifest: $manifest" >&2
  exit 1
fi
mkdir -p artifacts
{
  find configs -maxdepth 1 -type f -name 'stage1_5*' -print0
  find reports -maxdepth 1 -type f -name '*STAGE1_5*' -print0
  find src/etrcm/stage1_5 -type f ! -path '*/__pycache__/*' -print0
  find experiments -maxdepth 1 -type f -name '*stage1_5*' -print0
  find tests -maxdepth 1 -type f -name 'test_stage1_5*' -print0
  find results/stage1_5 -type f -print0
} | sort -zu | xargs -0 sha256sum > "$manifest"
sha256sum -c --status "$manifest"
printf '%s hashed Stage 1.5 assets\n' "$(wc -l < "$manifest")"
