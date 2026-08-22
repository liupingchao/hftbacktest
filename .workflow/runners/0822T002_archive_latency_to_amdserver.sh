#!/usr/bin/env bash
set -euo pipefail

TASK_ID="0822T002"
PACKAGE_ROOT="local_live_analysis/skhynix_c6in_hyperliquid_execution_latency_0822T002"
REMOTE_ALIAS="amdserver"
REMOTE_PARENT="/home/molly/project/durable_archives/skhynix_c6in_latency"

if [[ ! -f "${PACKAGE_ROOT}/measurement_manifest.json" ]]; then
  printf '%s has no formal package; archive is forbidden\n' "${TASK_ID}" >&2
  exit 3
fi

COMPOSITE="$(
  python3 - "${PACKAGE_ROOT}/measurement_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
print(manifest["composite_identity"])
PY
)"
REMOTE_FINAL="${REMOTE_PARENT}/${COMPOSITE}"
REMOTE_TEMP="${REMOTE_PARENT}/.${COMPOSITE}.tmp-${TASK_ID}"

ssh "${REMOTE_ALIAS}" "test ! -e '${REMOTE_FINAL}' && rm -rf '${REMOTE_TEMP}' && mkdir -p '${REMOTE_TEMP}'"
rsync -a --delete "${PACKAGE_ROOT}/" "${REMOTE_ALIAS}:${REMOTE_TEMP}/"
ssh "${REMOTE_ALIAS}" "mv '${REMOTE_TEMP}' '${REMOTE_FINAL}'"
printf '%s\n' "${REMOTE_FINAL}"
