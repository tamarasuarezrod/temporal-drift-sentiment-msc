#!/usr/bin/env bash
#
# Download and unpack the trained checkpoints (~14.7 GB) from Zenodo into
# checkpoints/<id>-seed<n>/ (for example checkpoints/tea-seed1/), which is where
# the pipeline expects them. Resumable and idempotent, and verifies the MD5.
#
#   ./download_checkpoints.sh
#
# Needs curl, unzip and python3. Re-run to resume an interrupted download.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECORD=21850584
API="https://zenodo.org/api/records/${RECORD}"
ZIP="$HERE/checkpoints.zip"
DEST="$HERE/checkpoints"

if [[ -d "$DEST" ]] && [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
  echo "checkpoints/ already populated — nothing to do."
  exit 0
fi

# Resolve the download URL, MD5 and size from the Zenodo record.
read -r URL MD5 SIZE < <(curl -sL --fail "$API" | python3 -c '
import json, sys
d = json.load(sys.stdin)
f = next(x for x in d["files"] if x["key"] == "checkpoints.zip")
print(f["links"]["self"], f.get("checksum", "md5:").split(":")[-1], f["size"])
')
printf "checkpoints.zip: %.2f GB from %s\n" "$(python3 -c "print($SIZE/1e9)")" "$URL"

echo "Downloading (resumable) ..."
curl -L -C - --fail --retry 3 -o "$ZIP" "$URL"

# Verify the MD5 with whichever tool is available.
if command -v md5sum >/dev/null 2>&1; then GOT="$(md5sum "$ZIP" | cut -d' ' -f1)"
elif command -v md5 >/dev/null 2>&1; then GOT="$(md5 -q "$ZIP")"
else GOT=""; fi
if [[ -n "$MD5" && -n "$GOT" && "$MD5" != "$GOT" ]]; then
  echo "MD5 mismatch (expected $MD5, got $GOT). Delete $ZIP and re-run."
  exit 1
fi
[[ -n "$GOT" ]] && echo "MD5 OK ($GOT)"

echo "Unzipping into $DEST/ ..."
unzip -q -o "$ZIP" -d "$HERE"
echo "Done. $(ls "$DEST" 2>/dev/null | wc -l | tr -d ' ') checkpoint folders under checkpoints/"
