#!/usr/bin/env bash
set -euo pipefail

TASK_ID="0821T001"
REMOTE_ALIAS="amdserver"
PACKAGE_ROOT="local_live_analysis/skhynix_continuous_conditional_risk_v2_stage_h0a_support_only"
BUILD_RECEIPT=".workflow/reports/0821T001-build-receipt.json"
ARCHIVE_RECEIPT=".workflow/reports/0821T001-archive-receipt.json"
DEFAULT_KERNEL_REPORT=".workflow/reports/0821T001-amdserver-kernel-only.json"
REMOTE_PARENT="/home/molly/project/durable_archives/skhynix_continuous_conditional_risk_v2/stage_h0a_support_only"

MODE="${1:-}"
if [[ "${MODE}" != "--archive" && "${MODE}" != "--verify-kernel-only" ]]; then
  printf 'usage: %s --archive|--verify-kernel-only <output.json>\n' "$0" >&2
  exit 2
fi
if [[ "${MODE}" == "--verify-kernel-only" && "$#" -ne 2 ]]; then
  printf 'usage: %s --verify-kernel-only <output.json>\n' "$0" >&2
  exit 2
fi

python3 examples/hyperliquid/skhynix_stage_h0a.py verify \
  --package "${PACKAGE_ROOT}" >/dev/null

COMPOSITE_IDENTITY="$(
  python3 - "${PACKAGE_ROOT}/h0a_manifest.json" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_bytes())
print(value["composite_identity"])
PY
)"
REMOTE_FINAL="${REMOTE_PARENT}/${COMPOSITE_IDENTITY}"
REMOTE_TEMP="${REMOTE_PARENT}/.${COMPOSITE_IDENTITY}.tmp-${TASK_ID}"

WORK_ROOT="$(mktemp -d "/tmp/${TASK_ID}-archive.XXXXXX")"
cleanup_local() {
  python3 - "${WORK_ROOT}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(path)
PY
}
trap cleanup_local EXIT

inventory_tree() {
  local root="$1"
  local output="$2"
  python3 - "${root}" "${output}" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
observed_root = root.lstat()
if not stat.S_ISDIR(observed_root.st_mode) or stat.S_ISLNK(observed_root.st_mode):
    raise SystemExit("inventory root must be a real directory")
entries = []
pending = [root]
while pending:
    directory = pending.pop()
    children = []
    with os.scandir(directory) as iterator:
        names = sorted(entry.name for entry in iterator)
    for name in names:
        path = directory / name
        observed = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            entries.append(
                {
                    "path": relative,
                    "entry_type": "directory",
                    "mode": stat.S_IMODE(observed.st_mode),
                }
            )
            children.append(path)
        elif stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append(
                {
                    "path": relative,
                    "entry_type": "regular_file",
                    "mode": stat.S_IMODE(observed.st_mode),
                    "bytes": observed.st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        else:
            raise SystemExit(f"forbidden archive entry: {relative}")
    pending.extend(reversed(children))
entries.sort(key=lambda row: row["path"])
files = [row for row in entries if row["entry_type"] == "regular_file"]
directories = [row for row in entries if row["entry_type"] == "directory"]
file_inventory = [
    {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
    for row in files
]
payload = {
    "schema_version": "h0a_exact_tree_inventory_v1",
    "entries": entries,
    "file_inventory": file_inventory,
    "file_count": len(files),
    "directory_count": len(directories),
    "total_bytes": sum(row["bytes"] for row in files),
    "tree_inventory_sha256": hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest(),
    "file_inventory_sha256": hashlib.sha256(
        json.dumps(
            file_inventory, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest(),
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY
}

SOURCE_INVENTORY="${WORK_ROOT}/source_inventory.json"
REMOTE_INVENTORY="${WORK_ROOT}/remote_inventory.json"
KERNEL_RAW="${WORK_ROOT}/kernel_raw.json"
VERIFIER_ROOT="${WORK_ROOT}/verifier"
inventory_tree "${PACKAGE_ROOT}" "${SOURCE_INVENTORY}"

build_verifier() {
  python3 - "${VERIFIER_ROOT}" <<'PY'
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "examples/hyperliquid"))
import research_package_trust_cli as kernel  # noqa: E402

destination = Path(sys.argv[1]) / "repo"
for row in kernel.source_tree_inventory():
    source = Path.cwd() / row["path"]
    target = destination / row["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
PY
}

remote_inventory() {
  local remote_root="$1"
  ssh "${REMOTE_ALIAS}" python3 - "${remote_root}" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
entries = []
pending = [root]
while pending:
    directory = pending.pop()
    children = []
    with os.scandir(directory) as iterator:
        names = sorted(entry.name for entry in iterator)
    for name in names:
        path = directory / name
        observed = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            entries.append(
                {
                    "path": relative,
                    "entry_type": "directory",
                    "mode": stat.S_IMODE(observed.st_mode),
                }
            )
            children.append(path)
        elif stat.S_ISREG(observed.st_mode) and not stat.S_ISLNK(observed.st_mode):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            entries.append(
                {
                    "path": relative,
                    "entry_type": "regular_file",
                    "mode": stat.S_IMODE(observed.st_mode),
                    "bytes": observed.st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        else:
            raise SystemExit(f"forbidden archive entry: {relative}")
    pending.extend(reversed(children))
entries.sort(key=lambda row: row["path"])
files = [row for row in entries if row["entry_type"] == "regular_file"]
directories = [row for row in entries if row["entry_type"] == "directory"]
file_inventory = [
    {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
    for row in files
]
payload = {
    "schema_version": "h0a_exact_tree_inventory_v1",
    "entries": entries,
    "file_inventory": file_inventory,
    "file_count": len(files),
    "directory_count": len(directories),
    "total_bytes": sum(row["bytes"] for row in files),
    "tree_inventory_sha256": hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest(),
    "file_inventory_sha256": hashlib.sha256(
        json.dumps(
            file_inventory, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest(),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
}

verify_kernel_only() {
  local output="$1"
  local observed_root="${2:-${REMOTE_FINAL}}"
  local reported_root="${3:-${observed_root}}"
  local remote_verifier="/tmp/${TASK_ID}-kernel-verifier-$$"
  build_verifier
  ssh "${REMOTE_ALIAS}" python3 - "${remote_verifier}" <<'PY'
import os
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if os.path.lexists(path):
    shutil.rmtree(path)
path.mkdir(mode=0o700)
PY
  rsync -a "${VERIFIER_ROOT}/" "${REMOTE_ALIAS}:${remote_verifier}/"
  ssh "${REMOTE_ALIAS}" \
    "cd '${remote_verifier}/repo' && PYTHONPATH=examples/hyperliquid python3 - <<'PY'
import research_package_trust_cli as kernel
print(kernel.source_tree_sha256())
PY" > "${WORK_ROOT}/remote_kernel_source_sha256.txt"
  ssh "${REMOTE_ALIAS}" \
    "PYTHONPATH='${remote_verifier}/repo/examples/hyperliquid:${observed_root}/runtime_source' \
      python3 '${observed_root}/runtime_source/skhynix_stage_h0a.py' \
      verify --package '${observed_root}'" > "${KERNEL_RAW}"
  remote_inventory "${observed_root}" > "${REMOTE_INVENTORY}"
  python3 - \
    "${SOURCE_INVENTORY}" \
    "${REMOTE_INVENTORY}" \
    "${KERNEL_RAW}" \
    "${WORK_ROOT}/remote_kernel_source_sha256.txt" \
    "${PACKAGE_ROOT}/h0a_manifest.json" \
    "${reported_root}" \
    "${output}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    source_text,
    remote_text,
    kernel_text,
    source_sha_text,
    manifest_text,
    remote_root,
    output_text,
) = sys.argv[1:]
source = json.loads(Path(source_text).read_bytes())
remote = json.loads(Path(remote_text).read_bytes())
kernel = json.loads(Path(kernel_text).read_bytes())
manifest = json.loads(Path(manifest_text).read_bytes())
source_sha = Path(source_sha_text).read_text(encoding="ascii").strip()
expected_source_sha = (
    "cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203"
)
if source != remote:
    raise SystemExit("ARCHIVE_TREE_MISMATCH: local/remote inventory differs")
if source_sha != expected_source_sha:
    raise SystemExit("KERNEL_SOURCE_IDENTITY_MISMATCH")
for field in (
    "research_data_identity",
    "code_contract_identity",
    "evidence_identity",
    "composite_identity",
):
    if kernel[field] != manifest[field]:
        raise SystemExit(f"COMPOSITE_IDENTITY_BINDING_MISMATCH: {field}")
result = {
    "schema_version": "skhynix_stage_h0a_amdserver_kernel_admission_v1",
    "task_id": "0821T001",
    "remote_archive_path": remote_root,
    "remote_tree_inventory_sha256": remote["tree_inventory_sha256"],
    "remote_file_inventory_sha256": remote["file_inventory_sha256"],
    "file_count": remote["file_count"],
    "directory_count": remote["directory_count"],
    "total_bytes": remote["total_bytes"],
    "kernel_source_tree_sha256": source_sha,
    "research_data_identity": kernel["research_data_identity"],
    "code_contract_identity": kernel["code_contract_identity"],
    "evidence_identity": kernel["evidence_identity"],
    "composite_identity": kernel["composite_identity"],
    "kernel_package_admission_portable": True,
    "full_source_semantic_replay_portable": False,
    "source_semantic_replay_executed": False,
    "zero_write": kernel["zero_write"],
    "verified_at_utc": datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    ).replace("+00:00", "Z"),
    "verified": True,
}
Path(output_text).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY
  ssh "${REMOTE_ALIAS}" python3 - "${remote_verifier}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(path)
PY
}

if [[ "${MODE}" == "--verify-kernel-only" ]]; then
  verify_kernel_only "$2" "${REMOTE_FINAL}" "${REMOTE_FINAL}"
  cat "$2"
  exit 0
fi

if [[ -e "${ARCHIVE_RECEIPT}" ]]; then
  printf 'archive receipt already exists: %s\n' "${ARCHIVE_RECEIPT}" >&2
  exit 2
fi
if [[ ! -f "${BUILD_RECEIPT}" ]]; then
  printf 'build receipt missing: %s\n' "${BUILD_RECEIPT}" >&2
  exit 2
fi

ARCHIVE_STARTED_AT_UTC="$(
  python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"))
PY
)"

ssh "${REMOTE_ALIAS}" python3 - \
  "${REMOTE_PARENT}" "${REMOTE_FINAL}" "${REMOTE_TEMP}" <<'PY'
import os
import stat
import sys
from pathlib import Path

parent = Path(sys.argv[1])
final = Path(sys.argv[2])
temp = Path(sys.argv[3])
project = Path("/home/molly/project")
if project not in parent.parents:
    raise SystemExit("remote archive parent escaped /home/molly/project")
current = project
for part in parent.relative_to(project).parts:
    current = current / part
    if os.path.lexists(current):
        observed = current.lstat()
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise SystemExit(f"remote parent type drift: {current}")
    else:
        current.mkdir(mode=0o755)
if os.path.lexists(final) or os.path.lexists(temp):
    raise SystemExit("remote final or temp already exists")
temp.mkdir(mode=0o755)
PY

rsync -a --numeric-ids "${PACKAGE_ROOT}/" "${REMOTE_ALIAS}:${REMOTE_TEMP}/"
ssh "${REMOTE_ALIAS}" python3 - "${REMOTE_TEMP}" <<'PY'
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in sorted(
    [item for item in root.rglob("*") if item.is_file()],
    key=lambda item: item.as_posix(),
):
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
directories = sorted(
    [item for item in root.rglob("*") if item.is_dir()],
    key=lambda item: len(item.parts),
    reverse=True,
)
for directory in [*directories, root]:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
PY

PREPUBLICATION_KERNEL_REPORT="${WORK_ROOT}/prepublication-kernel-only.json"
verify_kernel_only \
  "${PREPUBLICATION_KERNEL_REPORT}" \
  "${REMOTE_TEMP}" \
  "${REMOTE_FINAL}"

ssh "${REMOTE_ALIAS}" python3 - "${REMOTE_TEMP}" "${REMOTE_FINAL}" <<'PY'
import os
import sys
from pathlib import Path

temp = Path(sys.argv[1])
final = Path(sys.argv[2])
if os.path.lexists(final):
    raise SystemExit("remote final appeared before publication")
descriptor = os.open(temp.parent, os.O_RDONLY)
try:
    os.rename(temp, final)
    os.fsync(descriptor)
finally:
    os.close(descriptor)
if os.path.lexists(temp) or not final.is_dir():
    raise SystemExit("remote atomic publication failed")
PY

verify_kernel_only \
  "${DEFAULT_KERNEL_REPORT}" \
  "${REMOTE_FINAL}" \
  "${REMOTE_FINAL}"

ARCHIVE_COMPLETED_AT_UTC="$(
  python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"))
PY
)"

python3 - \
  "${BUILD_RECEIPT}" \
  "${DEFAULT_KERNEL_REPORT}" \
  "${PACKAGE_ROOT}/h0a_manifest.json" \
  "${PACKAGE_ROOT}" \
  "${REMOTE_FINAL}" \
  "${ARCHIVE_STARTED_AT_UTC}" \
  "${ARCHIVE_COMPLETED_AT_UTC}" \
  "${ARCHIVE_RECEIPT}" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path

(
    build_text,
    kernel_text,
    manifest_text,
    local_root,
    remote_root,
    started_text,
    completed_text,
    output_text,
) = sys.argv[1:]
build = json.loads(Path(build_text).read_bytes())
kernel = json.loads(Path(kernel_text).read_bytes())
manifest = json.loads(Path(manifest_text).read_bytes())

def parse(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

publication = build["publication_completed_at_utc"]
strict = parse(publication) < parse(started_text) < parse(completed_text)
if not strict:
    raise SystemExit("ARCHIVE_TREE_MISMATCH: chronology is not strict")
for field in (
    "research_data_identity",
    "code_contract_identity",
    "evidence_identity",
    "composite_identity",
):
    if kernel[field] != manifest[field]:
        raise SystemExit(f"COMPOSITE_IDENTITY_BINDING_MISMATCH: {field}")
result = {
    "schema_version": "skhynix_stage_h0a_archive_receipt_v1",
    "task_id": "0821T001",
    "local_package_path": local_root,
    "remote_archive_path": remote_root,
    "research_data_identity": manifest["research_data_identity"],
    "code_contract_identity": manifest["code_contract_identity"],
    "evidence_identity": manifest["evidence_identity"],
    "composite_identity": manifest["composite_identity"],
    "package_publication_completed_at_utc": publication,
    "archive_started_at_utc": started_text,
    "archive_completed_at_utc": completed_text,
    "local_tree_inventory_sha256": kernel["remote_tree_inventory_sha256"],
    "remote_tree_inventory_sha256": kernel["remote_tree_inventory_sha256"],
    "kernel_package_admission_portable": True,
    "full_source_semantic_replay_portable": False,
    "foreground_process_closed": True,
    "strict_chronology": strict,
    "verified": True,
}
Path(output_text).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY

cat "${ARCHIVE_RECEIPT}"
