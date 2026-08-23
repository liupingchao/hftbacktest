#!/usr/bin/env bash
set -euo pipefail

TASK_ID="0822T002"
REMOTE_ALIAS="c6in-winner"
EXPECTED_COMMIT="${1:-$(git rev-parse HEAD)}"
SHORT_COMMIT="${EXPECTED_COMMIT:0:12}"
REMOTE_ROOT="/tmp/${TASK_ID}-${SHORT_COMMIT}"
REMOTE_VENV="${REMOTE_ROOT}/venv"
REMOTE_PYTHON="${REMOTE_VENV}/bin/python"
REMOTE_BUNDLE="${REMOTE_ROOT}/${TASK_ID}.bundle"
REMOTE_REPO="${REMOTE_ROOT}/repo"
REMOTE_EVIDENCE="${REMOTE_ROOT}/evidence"
LOCAL_EVIDENCE=".workflow/reports/${TASK_ID}-c6in-gate2-${SHORT_COMMIT}"
TRADING_INSPECT="/home/admin/trading/inspect"
TRADING_CREDENTIALS="/home/admin/trading/credentials.env"

if [[ "$(git rev-parse HEAD)" != "${EXPECTED_COMMIT}" ]]; then
  printf 'expected commit is not local HEAD\n' >&2
  exit 2
fi
if [[ -e "${LOCAL_EVIDENCE}" ]]; then
  printf 'local evidence path already exists: %s\n' "${LOCAL_EVIDENCE}" >&2
  exit 2
fi

WORK_ROOT="$(mktemp -d "/tmp/${TASK_ID}-c6in.XXXXXX")"
cleanup() {
  rm -rf "${WORK_ROOT}"
}
trap cleanup EXIT

BUNDLE="${WORK_ROOT}/${TASK_ID}.bundle"
git bundle create "${BUNDLE}" HEAD
ssh "${REMOTE_ALIAS}" \
  "test ! -e '${REMOTE_ROOT}' && mkdir -p '${REMOTE_ROOT}'"
scp -q "${BUNDLE}" "${REMOTE_ALIAS}:${REMOTE_BUNDLE}"

ssh "${REMOTE_ALIAS}" \
  "EXPECTED_COMMIT='${EXPECTED_COMMIT}' \
REMOTE_BUNDLE='${REMOTE_BUNDLE}' \
REMOTE_REPO='${REMOTE_REPO}' \
REMOTE_EVIDENCE='${REMOTE_EVIDENCE}' \
REMOTE_VENV='${REMOTE_VENV}' \
REMOTE_PYTHON='${REMOTE_PYTHON}' \
TRADING_INSPECT='${TRADING_INSPECT}' \
TRADING_CREDENTIALS='${TRADING_CREDENTIALS}' \
bash -s" <<'REMOTE'
set -euo pipefail

if [[ ! -x "${REMOTE_PYTHON}" ]]; then
  python3 -m venv --copies "${REMOTE_VENV}"
  "${REMOTE_PYTHON}" -m pip install \
    --disable-pip-version-check \
    --no-input \
    "hyperliquid-python-sdk==0.24.0" \
    "pytest==8.4.2"
fi
"${REMOTE_PYTHON}" - <<'PY'
import importlib.metadata

assert importlib.metadata.version("hyperliquid-python-sdk") == "0.24.0"
assert importlib.metadata.version("pytest") == "8.4.2"
PY

test ! -e "${REMOTE_REPO}"
test ! -e "${REMOTE_EVIDENCE}"
git clone -q "${REMOTE_BUNDLE}" "${REMOTE_REPO}"
git -C "${REMOTE_REPO}" checkout -q --detach "${EXPECTED_COMMIT}"
test -z "$(git -C "${REMOTE_REPO}" status --porcelain)"
mkdir -p "${REMOTE_EVIDENCE}"

cd "${REMOTE_REPO}"
"${REMOTE_PYTHON}" \
  .workflow/workflow-kit/validate_research_package_task.py \
  --task .workflow/tasks/0822T002.md \
  --matrix .workflow/contracts/0822T002-surface-matrix.json \
  >"${REMOTE_EVIDENCE}/gate0-validator.json"

"${REMOTE_PYTHON}" -m pytest -q \
  examples/hyperliquid/test_skhynix_c6in_latency_v2.py \
  examples/hyperliquid/test_skhynix_c6in_latency_package_v2.py \
  examples/hyperliquid/test_hyperliquid_maker_order_manager.py \
  examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py \
  >"${REMOTE_EVIDENCE}/gate1-pytest.txt"

"${REMOTE_PYTHON}" examples/hyperliquid/skhynix_c6in_latency_v2.py \
  hostile-preflight \
  --task .workflow/tasks/0822T002.md \
  --matrix .workflow/contracts/0822T002-surface-matrix.json \
  --output "${REMOTE_EVIDENCE}/hostile-preflight.json" \
  >"${REMOTE_EVIDENCE}/hostile-preflight.stdout.json"

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  "${REMOTE_PYTHON}" examples/hyperliquid/skhynix_c6in_latency_v2.py \
  gate2-preflight \
  --expected-commit "${EXPECTED_COMMIT}" \
  --output-root "${REMOTE_EVIDENCE}/gate2-notional" \
  >"${REMOTE_EVIDENCE}/gate2-notional.stdout.json" \
  2>"${REMOTE_EVIDENCE}/gate2-notional.stderr.txt"

"${REMOTE_PYTHON}" - "${REMOTE_EVIDENCE}/gate2-notional/gate2_preflight_receipt.json" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
market = json.loads(
    Path(sys.argv[1]).with_name("market_identity.json").read_text(
        encoding="ascii"
    )
)
authorization = json.loads(
    Path(sys.argv[1]).with_name("authorization_envelope.json").read_text(
        encoding="ascii"
    )
)
assert receipt["status"] == "notional_subgate_pass"
assert receipt["gate2_complete"] is False
assert receipt["blocking_error_code"] == ""
assert receipt["credential_file_read"] is False
assert receipt["private_endpoint_called"] is False
assert receipt["order_endpoint_called"] is False
assert receipt["cancel_endpoint_called"] is False
assert receipt["public_quote_safety_collection_started"] is False
assert receipt["h0b_outcome_accessed"] is False
assert receipt["h0a_tuple_mutated"] is False
assert market["minimum_order_notional_status"] == "pass"
assert float(market["minimum_valid_order_notional"]) <= 15.0
assert authorization["per_order_notional_cap_usdc"] == 15.0
assert authorization["aggregate_position_cap_usdc"] == 30.0
assert authorization["max_loss_usdc"] == 3.0
PY

"${REMOTE_PYTHON}" examples/hyperliquid/skhynix_c6in_latency_v2.py \
  freeze-schedule \
  --output "${REMOTE_EVIDENCE}/collection-window-schedule-frozen.csv" \
  >"${REMOTE_EVIDENCE}/schedule-freeze.stdout.json" \
  2>"${REMOTE_EVIDENCE}/schedule-freeze.stderr.txt"

"${TRADING_INSPECT}" \
  --repo "${REMOTE_REPO}" \
  --env-file "${TRADING_CREDENTIALS}" \
  --python "${REMOTE_PYTHON}" \
  --json \
  >"${REMOTE_EVIDENCE}/trading-runtime-inspect.json"

"${REMOTE_PYTHON}" - \
  "${REMOTE_EVIDENCE}/trading-runtime-inspect.json" \
  "${EXPECTED_COMMIT}" \
  "${TRADING_CREDENTIALS}" \
  "${REMOTE_PYTHON}" <<'PY'
import json
import sys
from pathlib import Path

inspection = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_commit = sys.argv[2]
expected_credentials = sys.argv[3]
expected_python = sys.argv[4]
flags = inspection["boundary_flags"]
selected = inspection["selected"]
credentials = selected["credentials"]
python_runtime = selected["python"]
repo = selected["repo"]

assert inspection["discovery_status"] == "complete"
assert inspection["lookup_ready"] is True
assert inspection["execution_runtime_ready"] is True
assert inspection["execution_runtime_blockers"] == []
assert flags == {
    "account_endpoint_called": False,
    "cancel_endpoint_called": False,
    "credential_file_read": True,
    "credential_values_copied": False,
    "credential_values_emitted": False,
    "order_endpoint_called": False,
    "private_endpoint_called": False,
    "wallet_client_constructed": False,
}
assert credentials["path"] == expected_credentials
assert credentials["is_symlink"] is True
assert credentials["permission_secure"] is True
assert credentials["target_mode"] == "600"
assert credentials["exchange_status"]["hyperliquid"]["ready"] is True
assert credentials["credential_values_emitted"] is False
assert python_runtime["path"] == expected_python
assert python_runtime["resolved_path"] == expected_python
assert python_runtime["is_symlink"] is False
assert python_runtime["hyperliquid_importable"] is True
assert python_runtime["hyperliquid_sdk_version"] == "0.24.0"
assert python_runtime["hyperliquid_order_cancel_surface_ready"] is True
assert repo["commit"] == expected_commit
assert repo["is_git_checkout"] is True
assert repo["working_tree_clean"] is True
assert repo["dirty_count"] == 0
assert repo["order_runtime_sources_present"] is True
PY

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  "${REMOTE_PYTHON}" examples/hyperliquid/skhynix_c6in_latency_v2.py \
  gate2-full \
  --expected-commit "${EXPECTED_COMMIT}" \
  --credential-file "${TRADING_CREDENTIALS}" \
  --output-root "${REMOTE_EVIDENCE}/gate2-full" \
  >"${REMOTE_EVIDENCE}/gate2-full.stdout.json" \
  2>"${REMOTE_EVIDENCE}/gate2-full.stderr.txt"

"${REMOTE_PYTHON}" - "${REMOTE_EVIDENCE}/gate2-full" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipt = json.loads(
    (root / "gate2_full_receipt.json").read_text(encoding="ascii")
)
market = json.loads(
    (root / "market_identity.json").read_text(encoding="ascii")
)
authorization = json.loads(
    (root / "authorization_envelope.json").read_text(encoding="ascii")
)
quote_safety = json.loads(
    (root / "public_quote_safety.json").read_text(encoding="ascii")
)
account = json.loads(
    (root / "account_baseline.json").read_text(encoding="ascii")
)
assert receipt["status"] == "pass"
assert receipt["gate2_complete"] is True
assert receipt["credential_file_read"] is True
assert receipt["private_endpoint_called"] is True
assert receipt["order_endpoint_called"] is False
assert receipt["cancel_endpoint_called"] is False
assert receipt["public_quote_safety_status"] == "pass"
assert quote_safety["observed_duration_seconds"] >= 900
assert quote_safety["valid_pair_count"] >= 1000
assert market["quote_distance_safety_status"] == "pass"
assert account["open_order_count"] == 0
assert account["target_position_zero"] is True
assert account["available_margin_at_least_aggregate_cap"] is True
assert account["configured_identity_matches_signer"] is True
assert account["configured_identity_role"] == "agent"
assert account["signer_role"] == "agent"
assert account["account_role"] == "user"
assert account["account_abstraction"] == "unifiedAccount"
assert account["account_source"] == "derived_from_configured_agent_role"
assert account["agent_approved"] is True
assert account["agent_expired"] is False
assert account["available_collateral_source"] == (
    "unified_spot_usdc_available"
)
assert authorization["per_order_notional_cap_usdc"] == 15.0
assert authorization["aggregate_position_cap_usdc"] == 30.0
assert authorization["max_loss_usdc"] == 3.0
PY

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  "${REMOTE_PYTHON}" examples/hyperliquid/skhynix_c6in_latency_v2.py \
  collect-active \
  --expected-commit "${EXPECTED_COMMIT}" \
  --credential-file "${TRADING_CREDENTIALS}" \
  --gate2-root "${REMOTE_EVIDENCE}/gate2-full" \
  --schedule "${REMOTE_EVIDENCE}/collection-window-schedule-frozen.csv" \
  --output-root "${REMOTE_EVIDENCE}/active" \
  >"${REMOTE_EVIDENCE}/active.stdout.json" \
  2>"${REMOTE_EVIDENCE}/active.stderr.txt"

"${REMOTE_PYTHON}" - "${REMOTE_EVIDENCE}/active" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
receipt = json.loads(
    (root / "collection_receipt.json").read_text(encoding="ascii")
)
assert receipt["status"] == "complete"
assert receipt["attempt_count"] <= 120
assert receipt["final_open_orders_count"] == 0
assert receipt["final_position_zero"] is True
assert receipt["h0b_outcome_accessed"] is False
assert receipt["h0a_tuple_mutated"] is False
assert receipt["latency_values_accessed_by_l0"] is False
PY

env -i \
  PATH="/usr/bin:/bin" \
  PYTHONPATH="${REMOTE_REPO}/examples/hyperliquid" \
  "${REMOTE_PYTHON}" examples/hyperliquid/skhynix_c6in_latency_v2.py \
  summarize \
  --sealed-root "${REMOTE_EVIDENCE}/active/sealed" \
  --output "${REMOTE_EVIDENCE}/l1-a" \
  >"${REMOTE_EVIDENCE}/l1-a.stdout.json" \
  2>"${REMOTE_EVIDENCE}/l1-a.stderr.txt"

env -i \
  PATH="/usr/bin:/bin" \
  PYTHONPATH="${REMOTE_REPO}/examples/hyperliquid" \
  "${REMOTE_PYTHON}" examples/hyperliquid/skhynix_c6in_latency_v2.py \
  summarize \
  --sealed-root "${REMOTE_EVIDENCE}/active/sealed" \
  --output "${REMOTE_EVIDENCE}/l1-b" \
  >"${REMOTE_EVIDENCE}/l1-b.stdout.json" \
  2>"${REMOTE_EVIDENCE}/l1-b.stderr.txt"

diff -qr "${REMOTE_EVIDENCE}/l1-a" "${REMOTE_EVIDENCE}/l1-b"

"${REMOTE_PYTHON}" - "${REMOTE_EVIDENCE}" "${EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name == "sha256_inventory.json":
        continue
    rows.append(
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
payload = {
    "schema_version": "skhynix_c6in_latency_gate2_inventory_v2",
    "task_id": "0822T002",
    "source_commit": sys.argv[2],
    "file_count": len(rows),
    "files": rows,
}
(root / "sha256_inventory.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY
REMOTE

mkdir -p "${LOCAL_EVIDENCE}"
rsync -a "${REMOTE_ALIAS}:${REMOTE_EVIDENCE}/" "${LOCAL_EVIDENCE}/"

python3 - "${LOCAL_EVIDENCE}" "${EXPECTED_COMMIT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
inventory = json.loads(
    (root / "sha256_inventory.json").read_text(encoding="ascii")
)
assert inventory["source_commit"] == sys.argv[2]
for row in inventory["files"]:
    path = root / row["path"]
    assert path.stat().st_size == row["bytes"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
print(
    json.dumps(
        {
            "status": "gate2_pass",
            "source_commit": sys.argv[2],
            "evidence_root": str(root),
            "file_count": inventory["file_count"],
        },
        sort_keys=True,
    )
)
PY

PACKAGE_ROOT="local_live_analysis/skhynix_c6in_hyperliquid_execution_latency_0822T002"
test ! -e "${PACKAGE_ROOT}"
python3 examples/hyperliquid/skhynix_c6in_latency_v2.py \
  build-package \
  --evidence-root "${LOCAL_EVIDENCE}" \
  --package-root "${PACKAGE_ROOT}" \
  --source-commit "${EXPECTED_COMMIT}" \
  >".workflow/reports/${TASK_ID}-build-receipt.json"
python3 examples/hyperliquid/skhynix_c6in_latency_v2.py \
  verify-package \
  --package-root "${PACKAGE_ROOT}" \
  >".workflow/reports/${TASK_ID}-verify-receipt.json"
