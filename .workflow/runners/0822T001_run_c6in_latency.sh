#!/usr/bin/env bash
set -euo pipefail

TASK_ID="0822T001"
REMOTE_ALIAS="c6in-winner"
REMOTE_PYTHON="/home/admin/0729T003-venv/bin/python"
EXPECTED_COMMIT="${1:-$(git rev-parse HEAD)}"
SHORT_COMMIT="${EXPECTED_COMMIT:0:12}"
REMOTE_BUNDLE="/home/admin/${TASK_ID}-${SHORT_COMMIT}.bundle"
REMOTE_REPO="/home/admin/hftbacktest-${TASK_ID}-${SHORT_COMMIT}"
REMOTE_EVIDENCE="/home/admin/hftbacktest-artifacts/${TASK_ID}-gate2-${SHORT_COMMIT}"
LOCAL_EVIDENCE=".workflow/reports/${TASK_ID}-c6in-gate2-${SHORT_COMMIT}"

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
scp -q "${BUNDLE}" "${REMOTE_ALIAS}:${REMOTE_BUNDLE}"

ssh "${REMOTE_ALIAS}" \
  "EXPECTED_COMMIT='${EXPECTED_COMMIT}' \
REMOTE_BUNDLE='${REMOTE_BUNDLE}' \
REMOTE_REPO='${REMOTE_REPO}' \
REMOTE_EVIDENCE='${REMOTE_EVIDENCE}' \
REMOTE_PYTHON='${REMOTE_PYTHON}' \
bash -s" <<'REMOTE'
set -euo pipefail

test ! -e "${REMOTE_REPO}"
test ! -e "${REMOTE_EVIDENCE}"
git clone -q "${REMOTE_BUNDLE}" "${REMOTE_REPO}"
git -C "${REMOTE_REPO}" checkout -q --detach "${EXPECTED_COMMIT}"
test -z "$(git -C "${REMOTE_REPO}" status --porcelain)"
mkdir -p "${REMOTE_EVIDENCE}"

cd "${REMOTE_REPO}"
"${REMOTE_PYTHON}" \
  .workflow/workflow-kit/validate_research_package_task.py \
  --task .workflow/tasks/0822T001.md \
  --matrix .workflow/contracts/0822T001-surface-matrix.json \
  >"${REMOTE_EVIDENCE}/gate0-validator.json"

"${REMOTE_PYTHON}" -m pytest -q \
  examples/hyperliquid/test_skhynix_c6in_latency.py \
  examples/hyperliquid/test_skhynix_c6in_latency_package.py \
  examples/hyperliquid/test_hyperliquid_maker_order_manager.py \
  examples/hyperliquid/test_hyperliquid_tiny_live_real_order_executor.py \
  >"${REMOTE_EVIDENCE}/gate1-pytest.txt"

"${REMOTE_PYTHON}" examples/hyperliquid/skhynix_c6in_latency.py \
  hostile-preflight \
  --task .workflow/tasks/0822T001.md \
  --matrix .workflow/contracts/0822T001-surface-matrix.json \
  --output "${REMOTE_EVIDENCE}/hostile-preflight.json" \
  >"${REMOTE_EVIDENCE}/hostile-preflight.stdout.json"

set +e
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  "${REMOTE_PYTHON}" examples/hyperliquid/skhynix_c6in_latency.py \
  gate2-preflight \
  --expected-commit "${EXPECTED_COMMIT}" \
  --output-root "${REMOTE_EVIDENCE}/gate2" \
  >"${REMOTE_EVIDENCE}/gate2.stdout.json" \
  2>"${REMOTE_EVIDENCE}/gate2.stderr.txt"
gate2_rc=$?
set -e
test "${gate2_rc}" -eq 3

"${REMOTE_PYTHON}" - "${REMOTE_EVIDENCE}/gate2/gate2_preflight_receipt.json" <<'PY'
import json
import sys
from pathlib import Path

receipt = json.loads(Path(sys.argv[1]).read_text(encoding="ascii"))
assert receipt["status"] == "blocked"
assert receipt["blocking_error_code"] == "LATENCY_AUTHORIZATION_MISMATCH"
assert receipt["credential_file_read"] is False
assert receipt["private_endpoint_called"] is False
assert receipt["order_endpoint_called"] is False
assert receipt["cancel_endpoint_called"] is False
assert receipt["public_quote_safety_collection_started"] is False
assert receipt["h0b_outcome_accessed"] is False
assert receipt["h0a_tuple_mutated"] is False
PY

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
    "schema_version": "skhynix_c6in_latency_blocked_preflight_inventory_v1",
    "task_id": "0822T001",
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
            "status": "blocked",
            "source_commit": sys.argv[2],
            "evidence_root": str(root),
            "file_count": inventory["file_count"],
        },
        sort_keys=True,
    )
)
PY
