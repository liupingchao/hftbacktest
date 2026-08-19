#!/bin/zsh
set -euo pipefail

export HOME="/Users/liu"
export USER="liu"
export LOGNAME="liu"
export AWS_PROFILE="codex-cli"
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

TASK_ID="0806T001"
SOURCE_HOST="c6in-winner"
AMD_HOST="amdserver"
SERVICE="hftbacktest-0806t001-skhynix-2h.service"
SOURCE_ROOT="/home/admin/0806T001_skhynix_2h_continuous"
AMD_PARENT="/home/molly/project/hftbacktest/local_live_analysis"
AMD_FINAL="${AMD_PARENT}/0806T001_skhynix_2h_continuous"
AMD_TEMP="${AMD_PARENT}/.0806T001_skhynix_2h_continuous.tmp"
AMD_EVIDENCE="${AMD_PARENT}/0806T001_skhynix_2h_continuous.transfer.json"
STATE_DIR="/Users/liu/Documents/hftbacktest/local_live_analysis/0806T001_transfer"
SOURCE_INVENTORY="${STATE_DIR}/source_inventory.tsv"
AMD_INVENTORY="${STATE_DIR}/amd_inventory.tsv"
STATUS_FILE="${STATE_DIR}/status.txt"
POLL_SECONDS=30
DEADLINE_SECONDS=10800
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)

mkdir -p "${STATE_DIR}"

log() {
  local message
  message="$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"
  print -r -- "${message}"
  print -r -- "${message}" > "${STATUS_FILE}"
}

inventory() {
  local host="$1"
  local root="$2"
  ssh "${SSH_OPTIONS[@]}" "${host}" python3 - "${root}" <<'PY'
import hashlib
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
if not root.is_dir():
    raise SystemExit(f"inventory root missing: {root}")
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    relative = path.relative_to(root).as_posix()
    print(f"{digest.hexdigest()}\t{path.stat().st_size}\t{relative}")
PY
}

validate_manifest() {
  ssh "${SSH_OPTIONS[@]}" "${SOURCE_HOST}" python3 - \
    "${SOURCE_ROOT}/campaign_manifest.json" "${TASK_ID}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
task_id = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
run_status_path = path.parent / "run_status.json"
abort_path = path.parent / "abort_manifest.json"
run_status = json.loads(run_status_path.read_text(encoding="utf-8"))
checks = {
    "task_id": payload.get("task_id") == task_id,
    "execution_mode": payload.get("execution_mode") == "collection_only",
    "collection_mode": payload.get("collection_mode") == "continuous_single_segment",
    "network_collection_complete": payload.get("network_collection_complete") is True,
    "postprocess_pending": payload.get("postprocess_pending") is True,
    "passes": payload.get("passes") is True,
    "one_segment": len(payload.get("segments", [])) == 1,
    "run_status_complete": run_status.get("state") == "complete",
    "run_status_phase": run_status.get("phase") == "collection_only_complete",
    "abort_absent": not abort_path.exists(),
}
failed = [name for name, passed in checks.items() if not passed]
if failed:
    raise SystemExit("campaign manifest failed: " + ",".join(failed))
print(json.dumps(checks, sort_keys=True))
PY
}

started_epoch="$(date +%s)"
log "watcher_started service=${SERVICE}"

while true; do
  now_epoch="$(date +%s)"
  if (( now_epoch - started_epoch > DEADLINE_SECONDS )); then
    log "failed timeout_waiting_for_service"
    exit 1
  fi

  active_state="$(
    ssh "${SSH_OPTIONS[@]}" "${SOURCE_HOST}" \
      "systemctl is-active '${SERVICE}' 2>/dev/null || true"
  )"
  case "${active_state}" in
    active|activating|reloading)
      sleep "${POLL_SECONDS}"
      ;;
    *)
      break
      ;;
  esac
done

service_properties="$(
  ssh "${SSH_OPTIONS[@]}" "${SOURCE_HOST}" \
    "systemctl show '${SERVICE}' -p LoadState -p ActiveState -p SubState -p Result -p ExecMainStatus --no-pager"
)"
print -r -- "${service_properties}" > "${STATE_DIR}/service_terminal_state.txt"
print -r -- "${service_properties}" | grep -qx "ActiveState=inactive"
load_state="$(print -r -- "${service_properties}" | awk -F= '$1 == "LoadState" {print $2}')"
if [[ "${load_state}" == "loaded" ]]; then
  print -r -- "${service_properties}" | grep -qx "Result=success"
  print -r -- "${service_properties}" | grep -qx "ExecMainStatus=0"
elif [[ "${load_state}" != "not-found" ]]; then
  log "failed unexpected_service_load_state=${load_state}"
  exit 1
fi
validate_manifest > "${STATE_DIR}/manifest_validation.json"
log "collection_complete manifest_validated"

inventory "${SOURCE_HOST}" "${SOURCE_ROOT}" > "${SOURCE_INVENTORY}"
test -s "${SOURCE_INVENTORY}"

ssh "${SSH_OPTIONS[@]}" "${AMD_HOST}" \
  "test ! -e '${AMD_FINAL}' && test ! -e '${AMD_TEMP}' && mkdir -p '${AMD_TEMP}'"

log "transfer_started"
ssh "${SSH_OPTIONS[@]}" "${SOURCE_HOST}" \
  "tar -C '${SOURCE_ROOT}' -cf - ." \
  | ssh "${SSH_OPTIONS[@]}" "${AMD_HOST}" "tar -C '${AMD_TEMP}' -xf -"

inventory "${AMD_HOST}" "${AMD_TEMP}" > "${AMD_INVENTORY}"
cmp -s "${SOURCE_INVENTORY}" "${AMD_INVENTORY}"
file_count="$(wc -l < "${SOURCE_INVENTORY}" | tr -d ' ')"
byte_count="$(awk -F '\t' '{sum += $2} END {printf "%.0f", sum}' "${SOURCE_INVENTORY}")"
inventory_sha="$(shasum -a 256 "${SOURCE_INVENTORY}" | awk '{print $1}')"

ssh "${SSH_OPTIONS[@]}" "${AMD_HOST}" \
  "test ! -e '${AMD_FINAL}' && mv '${AMD_TEMP}' '${AMD_FINAL}'"

ssh "${SSH_OPTIONS[@]}" "${AMD_HOST}" python3 - \
  "${AMD_EVIDENCE}" "${TASK_ID}" "${SOURCE_HOST}" "${SOURCE_ROOT}" \
  "${AMD_FINAL}" "${file_count}" "${byte_count}" "${inventory_sha}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    evidence_path,
    task_id,
    source_host,
    source_root,
    destination_root,
    file_count,
    byte_count,
    inventory_sha,
) = sys.argv[1:]
payload = {
    "schema_version": "cross_exchange_raw_transfer_v1",
    "task_id": task_id,
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "source_host": source_host,
    "source_root": source_root,
    "destination_root": destination_root,
    "file_count": int(file_count),
    "byte_count": int(byte_count),
    "inventory_sha256": inventory_sha,
    "relative_path_size_sha256_match": True,
    "atomic_publish": True,
    "passes": True,
}
path = Path(evidence_path)
temporary = path.with_suffix(path.suffix + ".tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY

log "transfer_complete destination=${AMD_FINAL} files=${file_count} bytes=${byte_count} inventory_sha256=${inventory_sha}"
