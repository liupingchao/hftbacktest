#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${HOST:-c6in-winner}"
REMOTE_TMP="/tmp/0822T002-trading-runtime-discovery.py"

scp \
  "${ROOT}/examples/hyperliquid/trading_runtime_discovery.py" \
  "${HOST}:${REMOTE_TMP}"

ssh "${HOST}" \
  "/home/admin/0729T003-venv/bin/python ${REMOTE_TMP} install \
    --home /home/admin \
    --runtime-root /home/admin/trading \
    --repo-target /home/admin/hftbacktest-cross-exchange \
    --env-target /home/admin/XEMM_rust_latest/.env \
    --venv-target /home/admin/0729T003-venv \
    --python-target /home/admin/0729T003-venv/bin/python \
    --json"

ssh "${HOST}" \
  "/usr/bin/python3 -c 'from pathlib import Path; Path(\"${REMOTE_TMP}\").unlink(missing_ok=True)'"
