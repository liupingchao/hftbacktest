#!/usr/bin/env bash
set -euo pipefail

# Sync this example folder to amdserver:~/project/hftbacktest/examples/binance_tick_mm
# Usage:
#   ./sync_to_amdserver.sh
#   ./sync_to_amdserver.sh amdserver ~/project/hftbacktest

REMOTE_HOST="${1:-amdserver}"
REMOTE_PROJECT_ROOT="${2:-~/project/hftbacktest}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rsync -az --delete \
  --exclude '__pycache__' \
  --exclude '.DS_Store' \
  "${SCRIPT_DIR}/" \
  "${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/examples/binance_tick_mm/"

echo "Synced ${SCRIPT_DIR} -> ${REMOTE_HOST}:${REMOTE_PROJECT_ROOT}/examples/binance_tick_mm/"
