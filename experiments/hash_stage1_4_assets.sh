#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
manifest=artifacts/stage1_4_all_assets.sha256
if [[ -e "$manifest" ]]; then
  echo "Manifest already exists: $manifest" >&2
  exit 1
fi
{
  find results/stage1_4 -type f -print0
  find reports -maxdepth 1 -type f -name '*STAGE1_4*.md' -print0
  find docs -maxdepth 1 -type f -name 'STAGE1_4*.md' -print0
  find configs -maxdepth 1 -type f -name 'stage1_4*' -print0
  find src/etrcm/stage1_4 -maxdepth 1 -type f -name '*.py' -print0
  find experiments -maxdepth 1 -type f -name '*stage1_4*' -print0
  find artifacts -maxdepth 1 -type f -name 'stage1_4*' ! -name 'stage1_4_all_assets.sha256' -print0
} | sort -z | xargs -0 sha256sum > "$manifest"
sha256sum -c --status "$manifest"
echo "SHA256 manifest complete: $(wc -l < "$manifest") files"
