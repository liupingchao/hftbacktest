#!/usr/bin/env bash
set -euo pipefail

REMOTE_PATH=/home/admin/hftbacktest-cross-exchange
PY=/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python
ENV_FILE=/home/admin/XEMM_rust_latest/.env
TASK_ID=0716T006
ROOT=/home/admin/hftbacktest-cross-exchange-artifacts/cross_exchange_controlled_role_evidence_0716T006_20260716T073133Z

cd "$REMOTE_PATH"
echo "continue_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

for W in 02 03; do
  WDIR="$ROOT/window_${W}"
  mkdir -p "$WDIR"
  echo "window_${W}_start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  set +e
  "$PY" examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py \
    --event-driven-edge-gate-live \
    --hyperliquid-l2book-fast \
    --watcher-seconds 1800 \
    --max-order-size 0.005 \
    --max-real-order-submissions 2 \
    --quote-hold-seconds 3 \
    --wait-seconds 10 \
    --env-file "$ENV_FILE" \
    --artifact-task-id "$TASK_ID" \
    --output-dir "$WDIR" \
    > "$WDIR/runner_stdout.log" 2> "$WDIR/runner_stderr.log"
  RC=$?
  set -e

  echo "window_${W}_runner_rc=$RC"
  "$PY" - "$ENV_FILE" "$TASK_ID" "$W" "$WDIR/independent_remote_open_orders_check.json" <<'PY'
import json
import pathlib
import sys
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor

env_file, task_id, window, output = sys.argv[1:5]
executor.load_env_file(pathlib.Path(env_file))
client = executor.build_live_client_from_env()
orders = client.open_orders()
payload = {
    "task_id": task_id,
    "window": window,
    "final_open_orders": executor.redact(orders),
    "final_open_orders_count": len(orders),
    "final_open_orders_empty": len(orders) == 0,
    "private_read_only": True,
    "order_endpoint_called": False,
    "cancel_endpoint_called": False,
    "credentials_written": False,
    "secret_values_written": False,
    "raw_signatures_written": False,
}
pathlib.Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps({"window": window, "final_open_orders_count": len(orders), "final_open_orders_empty": len(orders) == 0}, sort_keys=True))
if orders:
    raise SystemExit(42)
PY

  echo "window_${W}_end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ "$RC" -ne 0 ]; then
    echo "window_${W}_runner_nonzero_stop"
    exit "$RC"
  fi
done

find "$ROOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$ROOT/remote_sha256_manifest.txt"
printf '{"task_id":"%s","remote_root":"%s","completed_at_utc":"%s","windows_completed":"01,02,03"}\n' \
  "$TASK_ID" "$ROOT" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$ROOT/run_complete.json"
echo "continue_complete_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
