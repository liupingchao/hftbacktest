#!/usr/bin/env bash
set -euo pipefail

TASK_ID="0820T001"
REMOTE_ALIAS="amdserver"
REMOTE_USER="molly"
SOURCE_ROOT="local_live_analysis/skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3"
PACKAGE_ID="669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433"
REMOTE_PARENT="/home/molly/project/durable_archives/skhynix_episode_research_v1/stage04_jul30_episode_v3"
REMOTE_FINAL="${REMOTE_PARENT}/${PACKAGE_ID}"
REMOTE_TEMP="${REMOTE_PARENT}/.${PACKAGE_ID}.tmp-${TASK_ID}"
PARITY_REPORT=".workflow/reports/${TASK_ID}-stage4-parity.json"
HOSTILE_RECEIPT=".workflow/reports/${TASK_ID}-hostile-preflight.json"
FIRST_FULL_RECEIPT=".workflow/reports/${TASK_ID}-first-full-admission-start.json"
LAYER_ASSIGNMENT=".workflow/reports/${TASK_ID}-stage4-layer-assignment.json"
CANDIDATE_ROOT=".workflow/reports/${TASK_ID}-kernel-candidate"
QA_REPORT=".workflow/reports/0815T003-qa-round6.md"
DURABLE_RESEARCH_INVENTORY=".workflow/reports/0815T003-round4-pre-repair-research-inventory.csv"
ARCHIVE_REPORT=".workflow/reports/${TASK_ID}-stage4-archive.json"
CLEANUP_REPORT=".workflow/reports/${TASK_ID}-stage4-empty-dir-cleanup.json"
PORTABILITY_CLASS="byte_exact_and_kernel_admission_only"
PORTABILITY_STATEMENT="This archive is a byte-exact copy of the accepted Stage 4 package and is portable for kernel/package admission. It is not a self-contained source-semantic replay archive. Full replay still requires the exact external Stage 1/2/3 packages and Jul30 inputs listed in external_dependency_bindings."

MODE="${1:-}"
if [[ "${MODE}" != "--preflight-only" \
  && "${MODE}" != "--execute" \
  && "${MODE}" != "--refresh-envelope" \
  && "${MODE}" != "--refresh-envelope-after-qa-repair" \
  && "${MODE}" != "--refresh-envelope-after-qa-round2" \
  && "${MODE}" != "--verify-kernel-only" ]]; then
  printf 'usage: %s --preflight-only|--execute|--refresh-envelope|--refresh-envelope-after-qa-repair|--refresh-envelope-after-qa-round2|--verify-kernel-only <output.json>\n' "$0" >&2
  exit 2
fi

if [[ "${MODE}" == "--verify-kernel-only" ]]; then
  if [[ "$#" -ne 2 ]]; then
    printf 'usage: %s --verify-kernel-only <output.json>\n' "$0" >&2
    exit 2
  fi
  python3 - \
    "${REMOTE_FINAL}/package" \
    "${REMOTE_FINAL}/evidence" \
    "${2}" \
    "${REMOTE_FINAL}" \
    "${PORTABILITY_CLASS}" \
    "${PORTABILITY_STATEMENT}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path.cwd() / "examples/hyperliquid"))

import research_package_trust as trust  # noqa: E402
import research_package_trust_stage4_adapter as adapter  # noqa: E402


(
    package_text,
    evidence_text,
    output_text,
    archive_root,
    portability_class,
    portability_statement,
) = sys.argv[1:]
package = Path(package_text)
evidence = Path(evidence_text)
output = Path(output_text)
if not package.is_dir() or not evidence.is_dir():
    raise SystemExit("durable archive package/evidence missing")
package_resolved = package.resolve()
output_resolved = output.resolve()
if output_resolved == package_resolved or package_resolved in output_resolved.parents:
    raise SystemExit("kernel-only report must be outside admitted package")
if os.path.lexists(output):
    raise SystemExit(f"kernel-only report already exists: {output}")


def load_receipt(name: str, hash_key: str) -> dict:
    path = evidence / name
    value = json.loads(path.read_bytes())
    claimed = value[hash_key]
    payload = dict(value)
    payload.pop(hash_key)
    if trust.canonical_json_sha256(payload) != claimed:
        raise SystemExit(f"receipt self-hash drift: {name}")
    return value


hostile = adapter._validate_hostile_before_full(
    evidence / "0820T001-hostile-preflight.json"
)
first_full = load_receipt(
    "0820T001-first-full-admission-start.json",
    "receipt_sha256",
)
parity = load_receipt("0820T001-stage4-parity.json", "report_sha256")
archive = load_receipt("archive_receipt.json", "receipt_sha256")
cleanup = load_receipt(
    "0820T001-stage4-empty-dir-cleanup.json",
    "receipt_sha256",
)
if parity.get("verified") is not True:
    raise SystemExit("Stage 4 parity report is not verified")
if (
    first_full["hostile_receipt_sha256"] != hostile["receipt_sha256"]
    or parity["hostile_receipt_sha256"] != hostile["receipt_sha256"]
    or parity["first_full_admission_start_sha256"]
    != first_full["receipt_sha256"]
):
    raise SystemExit("hostile/first-full/parity receipt binding drift")
if (
    archive.get("archive_root") != archive_root
    or archive.get("byte_exact_package_archive") is not True
    or archive.get("kernel_trust_admission_portable") is not True
    or archive.get("full_source_semantic_replay_portable") is not False
    or archive.get("portability_class") != portability_class
    or archive.get("portability_statement") != portability_statement
    or archive.get("transfer_process_mode") != "foreground_waited"
    or archive.get("all_child_exit_status_zero") is not True
    or archive.get("temp_paths_remaining") != 0
    or archive.get("task_processes_remaining") != 0
):
    raise SystemExit("archive portability/process contract drift")


def parse_utc(value: str) -> datetime:
    observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if observed.tzinfo is None:
        raise SystemExit(f"naive receipt timestamp: {value}")
    return observed


ordered = (
    parse_utc(hostile["completed_at_utc"]),
    parse_utc(first_full["started_at_utc"]),
    parse_utc(parity["completed_at_utc"]),
    parse_utc(archive["started_at_utc"]),
    parse_utc(archive["completed_at_utc"]),
    parse_utc(cleanup["post_final_envelope_attested_at_utc"]),
)
if any(left >= right for left, right in zip(ordered, ordered[1:])):
    raise SystemExit("hostile/full/parity/archive/cleanup order is not strict")

before = trust.metadata_snapshot(package)
contract = adapter._candidate_contract()
semantic = {
    "legacy_core_sha256": adapter.ACCEPTED_CORE,
    "legacy_full_inventory_sha256": adapter.ACCEPTED_FULL,
    "artifact_count": adapter.ACCEPTED_ARTIFACT_COUNT,
    "file_count": adapter.ACCEPTED_FILE_COUNT,
    "total_bytes": adapter.ACCEPTED_TOTAL_BYTES,
    "source_semantic_verified": True,
}
candidate = trust.admit_package(package, contract, semantic)
anchor = adapter.verify_research_data_anchor(package)
after = trust.metadata_snapshot(package)
trust.assert_zero_write_snapshot(before, after, location=str(package))

identity = candidate.identity.as_dict()
for field in (
    "research_data_identity",
    "runtime_contract_identity",
    "publication_envelope_identity",
    "composite_package_identity",
):
    if parity["kernel"]["identity"][field] != identity[field]:
        raise SystemExit(f"parity/kernel identity drift: {field}")
    if archive[field] != identity[field]:
        raise SystemExit(f"archive/kernel identity drift: {field}")
    if cleanup["final_envelope_binding"][field] != identity[field]:
        raise SystemExit(f"cleanup/kernel identity drift: {field}")
if (
    archive["legacy_full_inventory_sha256"] != adapter.ACCEPTED_FULL
    or cleanup["final_envelope_binding"]["archive_receipt_sha256"]
    != archive["receipt_sha256"]
    or cleanup["final_envelope_binding"]["archive_completed_at_utc"]
    != archive["completed_at_utc"]
):
    raise SystemExit("archive/cleanup binding drift")

result = {
    "schema_version": "stage4_archive_kernel_only_admission_v1",
    "task_id": "0820T001",
    "admission_mode": "kernel_package_only",
    "package_root": str(package),
    "archive_receipt_sha256": archive["receipt_sha256"],
    "cleanup_receipt_sha256": cleanup["receipt_sha256"],
    "kernel": candidate.as_dict(),
    "research_anchor": anchor,
    "pre_post_metadata_exact": True,
    "kernel_trust_admission_portable": True,
    "full_source_semantic_replay_portable": False,
    "source_semantic_replay_executed": False,
    "accepted_source_semantic_evidence_consumed": True,
    "strict_receipt_order_verified": True,
    "completed_at_utc": datetime.now(timezone.utc)
    .isoformat()
    .replace("+00:00", "Z"),
    "verified": True,
}
result["report_sha256"] = trust.canonical_json_sha256(result)
trust.atomic_write_json(output, result)
print(json.dumps(result, indent=2, sort_keys=True))
PY
  exit 0
fi

if [[ "$#" -ne 1 ]]; then
  printf 'archive mode accepts no additional arguments: %s\n' "${MODE}" >&2
  exit 2
fi

for path in \
  "${SOURCE_ROOT}" \
  "${PARITY_REPORT}" \
  "${HOSTILE_RECEIPT}" \
  "${FIRST_FULL_RECEIPT}" \
  "${LAYER_ASSIGNMENT}" \
  "${CANDIDATE_ROOT}" \
  "${QA_REPORT}" \
  "${DURABLE_RESEARCH_INVENTORY}"
do
  if [[ ! -e "${path}" ]]; then
    printf 'required path missing: %s\n' "${path}" >&2
    exit 2
  fi
done

python3 - "${PARITY_REPORT}" "${PACKAGE_ID}" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_bytes())
if report.get("verified") is not True:
    raise SystemExit("Stage 4 parity report is not verified")
if report["legacy"]["legacy_full_inventory_sha256"] != sys.argv[2]:
    raise SystemExit("Stage 4 parity package identity drift")
if report.get("full_rebuild_count") != 0:
    raise SystemExit("unexpected full rebuild")
if report.get("package_mutation_count") != 0:
    raise SystemExit("unexpected package mutation")
PY

capture_archive_started_at() {
  python3 - "${PARITY_REPORT}" <<'PY'
import json
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

parity = json.loads(Path(sys.argv[1]).read_bytes())
completed = datetime.fromisoformat(
    parity["completed_at_utc"].replace("Z", "+00:00")
)
while True:
    observed = datetime.now(timezone.utc)
    if observed > completed:
        print(observed.isoformat().replace("+00:00", "Z"))
        break
    time.sleep(0.001)
PY
}

if [[ "${MODE}" == "--refresh-envelope" \
  || "${MODE}" == "--refresh-envelope-after-qa-repair" \
  || "${MODE}" == "--refresh-envelope-after-qa-round2" ]]; then
  if [[ "${MODE}" == "--refresh-envelope" ]]; then
    SUPERSEDED_LABEL="pre_gate0_fix"
    SUPERSEDED_ROOT=".workflow/reports/${TASK_ID}-superseded-pre-gate0-fix"
  elif [[ "${MODE}" == "--refresh-envelope-after-qa-repair" ]]; then
    SUPERSEDED_LABEL="pre_qa_round1_repair"
    SUPERSEDED_ROOT=".workflow/reports/${TASK_ID}-superseded-pre-qa-round1-repair"
  else
    SUPERSEDED_LABEL="pre_qa_round2_repair"
    SUPERSEDED_ROOT=".workflow/reports/${TASK_ID}-superseded-pre-qa-round2-repair"
  fi
  REMOTE_SUPERSEDED_ENVELOPE="trust_envelope_superseded_${SUPERSEDED_LABEL}"
  REMOTE_SUPERSEDED_EVIDENCE="superseded_${SUPERSEDED_LABEL}"
  OLD_ARCHIVE_REPORT="${SUPERSEDED_ROOT}/${TASK_ID}-stage4-archive.json"
  OLD_CLEANUP_REPORT="${SUPERSEDED_ROOT}/${TASK_ID}-stage4-empty-dir-cleanup.json"
  if [[ ! -f "${OLD_ARCHIVE_REPORT}" || -e "${ARCHIVE_REPORT}" ]]; then
    printf 'refresh requires one superseded archive report and no current report\n' >&2
    exit 2
  fi
  if [[ ! -f "${OLD_CLEANUP_REPORT}" || -e "${CLEANUP_REPORT}" ]]; then
    printf 'refresh requires one superseded cleanup report and no current report\n' >&2
    exit 2
  fi
  ARCHIVE_STARTED_AT_UTC="$(capture_archive_started_at)"
  REFRESH_WORK="$(mktemp -d "/tmp/${TASK_ID}-refresh.XXXXXX")"
  cleanup_refresh_work() {
    python3 - "${REFRESH_WORK}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(path)
PY
  }
  trap cleanup_refresh_work EXIT
  REFRESH_RECEIPT="${REFRESH_WORK}/archive_receipt.json"
  LOCAL_ENVELOPE_INVENTORY="${REFRESH_WORK}/local_envelope_inventory.json"
  REMOTE_ENVELOPE_INVENTORY="${REFRESH_WORK}/remote_envelope_inventory.json"
  python3 - "${CANDIDATE_ROOT}" "${LOCAL_ENVELOPE_INVENTORY}" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = []
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
            children.append(path)
        elif stat.S_ISREG(observed.st_mode):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append(
                {
                    "path": relative,
                    "bytes": observed.st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        else:
            raise SystemExit(f"forbidden envelope entry: {relative}")
    pending.extend(reversed(children))
rows.sort(key=lambda row: row["path"])
output.write_text(
    json.dumps(rows, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY
  ssh "${REMOTE_ALIAS}" python3 - \
    "${REMOTE_FINAL}" \
    "${REMOTE_TEMP}" \
    "${TASK_ID}" \
    "${REMOTE_SUPERSEDED_ENVELOPE}" <<'PY'
import os
import sys
from pathlib import Path

final = Path(sys.argv[1])
temp = Path(sys.argv[2])
task_id = sys.argv[3]
superseded_envelope = sys.argv[4]
if not final.is_dir() or os.path.lexists(temp):
    raise SystemExit("remote final/temp state invalid for envelope refresh")
refresh = final / f".trust-envelope-refresh-{task_id}"
if os.path.lexists(refresh):
    raise SystemExit("remote envelope refresh temp already exists")
if os.path.lexists(final / superseded_envelope):
    raise SystemExit("remote superseded envelope already exists")
refresh.mkdir()
(refresh / "trust_envelope").mkdir()
(refresh / "evidence").mkdir()
PY
  rsync -a \
    "${CANDIDATE_ROOT}/" \
    "${REMOTE_ALIAS}:${REMOTE_FINAL}/.trust-envelope-refresh-${TASK_ID}/trust_envelope/"
  rsync -a \
    "${HOSTILE_RECEIPT}" \
    "${FIRST_FULL_RECEIPT}" \
    "${LAYER_ASSIGNMENT}" \
    "${PARITY_REPORT}" \
    "${REMOTE_ALIAS}:${REMOTE_FINAL}/.trust-envelope-refresh-${TASK_ID}/evidence/"
  ssh "${REMOTE_ALIAS}" python3 - \
    "${REMOTE_FINAL}/.trust-envelope-refresh-${TASK_ID}/trust_envelope" \
    "${REMOTE_FINAL}/.trust-envelope-refresh-${TASK_ID}/remote_envelope_inventory.json" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
rows = []
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
            children.append(path)
        elif stat.S_ISREG(observed.st_mode):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            rows.append(
                {
                    "path": relative,
                    "bytes": observed.st_size,
                    "sha256": digest.hexdigest(),
                }
            )
        else:
            raise SystemExit(f"forbidden envelope entry: {relative}")
    pending.extend(reversed(children))
rows.sort(key=lambda row: row["path"])
output.write_text(
    json.dumps(rows, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY
  ssh "${REMOTE_ALIAS}" cat \
    "${REMOTE_FINAL}/.trust-envelope-refresh-${TASK_ID}/remote_envelope_inventory.json" \
    > "${REMOTE_ENVELOPE_INVENTORY}"
  python3 examples/hyperliquid/research_package_trust_cli.py \
    compare-inventories \
    "${LOCAL_ENVELOPE_INVENTORY}" \
    "${REMOTE_ENVELOPE_INVENTORY}"
  ssh "${REMOTE_ALIAS}" python3 - \
    "${REMOTE_FINAL}" \
    "${TASK_ID}" \
    "${REMOTE_SUPERSEDED_ENVELOPE}" \
    "${REMOTE_SUPERSEDED_EVIDENCE}" <<'PY'
import os
import sys
from pathlib import Path

final = Path(sys.argv[1])
task_id = sys.argv[2]
superseded_envelope = sys.argv[3]
superseded_evidence = sys.argv[4]
refresh = final / f".trust-envelope-refresh-{task_id}"
evidence = final / "evidence"
superseded = evidence / superseded_evidence
superseded.mkdir()
old_names = (
    "0820T001-hostile-preflight.json",
    "0820T001-first-full-admission-start.json",
    "0820T001-stage4-layer-assignment.json",
    "0820T001-stage4-parity.json",
    "0820T001-stage4-empty-dir-cleanup.json",
    "archive_receipt.json",
)
for name in old_names:
    source = evidence / name
    if source.exists():
        os.rename(source, superseded / name)
os.rename(
    final / "trust_envelope",
    final / superseded_envelope,
)
os.rename(refresh / "trust_envelope", final / "trust_envelope")
for source in sorted((refresh / "evidence").iterdir()):
    target_name = (
        "archive_receipt.json"
        if source.name == "archive_receipt.json"
        else source.name
    )
    os.rename(source, evidence / target_name)
(refresh / "evidence").rmdir()
(refresh / "remote_envelope_inventory.json").unlink()
refresh.rmdir()
descriptor = os.open(final, os.O_RDONLY)
try:
    os.fsync(descriptor)
finally:
    os.close(descriptor)
PY
  python3 - \
    "${OLD_ARCHIVE_REPORT}" \
    "${PARITY_REPORT}" \
    "${ARCHIVE_STARTED_AT_UTC}" \
    "${REFRESH_RECEIPT}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

old_path, parity_path, started_text, output_path = sys.argv[1:]
receipt = json.loads(Path(old_path).read_bytes())
parity = json.loads(Path(parity_path).read_bytes())
identity = parity["kernel"]["identity"]
receipt["runtime_contract_identity"] = identity[
    "runtime_contract_identity"
]
receipt["publication_envelope_identity"] = identity[
    "publication_envelope_identity"
]
receipt["composite_package_identity"] = identity[
    "composite_package_identity"
]
receipt["research_data_identity"] = identity["research_data_identity"]
completed_text = (
    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
)
parity_completed = datetime.fromisoformat(
    parity["completed_at_utc"].replace("Z", "+00:00")
)
started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
completed = datetime.fromisoformat(
    completed_text.replace("Z", "+00:00")
)
if not parity_completed < started < completed:
    raise SystemExit("archive refresh chronology is not strict")
receipt["started_at_utc"] = started_text
receipt["completed_at_utc"] = completed_text
receipt.pop("receipt_sha256", None)
canonical = json.dumps(
    receipt, sort_keys=True, separators=(",", ":")
).encode("ascii")
receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
Path(output_path).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY
  cp "${REFRESH_RECEIPT}" "${ARCHIVE_REPORT}"
  rsync -a \
    "${REFRESH_RECEIPT}" \
    "${REMOTE_ALIAS}:${REMOTE_FINAL}/evidence/archive_receipt.json"
  ssh "${REMOTE_ALIAS}" python3 - \
    "${REMOTE_FINAL}" \
    "${PACKAGE_ID}" \
    "${REMOTE_SUPERSEDED_ENVELOPE}" <<'PY'
import json
import os
import sys
from pathlib import Path

final = Path(sys.argv[1])
package_id = sys.argv[2]
superseded_envelope = sys.argv[3]
if not (final / "trust_envelope").is_dir():
    raise SystemExit("current trust envelope missing")
if not (final / superseded_envelope).is_dir():
    raise SystemExit("superseded trust envelope missing")
receipt = json.loads(
    (final / "evidence/archive_receipt.json").read_bytes()
)
post = json.loads((final / "evidence/post_inventory.json").read_bytes())
if (
    receipt["legacy_full_inventory_sha256"] != package_id
    or post["full_inventory_sha256"] != package_id
    or receipt["temp_paths_remaining"] != 0
    or receipt["task_processes_remaining"] != 0
):
    raise SystemExit("refreshed archive evidence drift")
if os.path.lexists(final / ".trust-envelope-refresh-0820T001"):
    raise SystemExit("remote envelope refresh temp remains")
PY
  python3 - \
    "${OLD_CLEANUP_REPORT}" \
    "${ARCHIVE_REPORT}" \
    "${PARITY_REPORT}" \
    "${SOURCE_ROOT}" \
    "${CLEANUP_REPORT}" <<'PY'
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "examples/hyperliquid"))
from research_package_trust import (  # noqa: E402
    build_inventory,
    canonical_json_sha256,
)

old_path, archive_path, parity_path, formal_text, output_path = sys.argv[1:]
old = json.loads(Path(old_path).read_bytes())
archive = json.loads(Path(archive_path).read_bytes())
parity = json.loads(Path(parity_path).read_bytes())
formal = Path(formal_text)
claimed = old["receipt_sha256"]
old_payload = dict(old)
old_payload.pop("receipt_sha256")
observed = hashlib.sha256(
    json.dumps(
        old_payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
).hexdigest()
if claimed != observed:
    raise SystemExit("superseded cleanup receipt self-hash drift")
if any(os.path.lexists(path) for path in old["allowlist"]):
    raise SystemExit("cleanup target reappeared before final attestation")
if not formal.is_dir():
    raise SystemExit("formal Stage 4 package disappeared")
formal_inventory = build_inventory(formal)
if (
    len(formal_inventory) != 107
    or sum(row["bytes"] for row in formal_inventory) != 1_561_307_420
    or canonical_json_sha256(formal_inventory)
    != archive["legacy_full_inventory_sha256"]
):
    raise SystemExit("formal Stage 4 package drift after final envelope")
identity = parity["kernel"]["identity"]
for field in (
    "research_data_identity",
    "runtime_contract_identity",
    "publication_envelope_identity",
    "composite_package_identity",
):
    if archive[field] != identity[field]:
        raise SystemExit(f"archive/parity identity drift: {field}")
if archive["legacy_full_inventory_sha256"] != old[
    "formal_full_inventory_sha256"
]:
    raise SystemExit("archive/cleanup formal identity drift")
archive_completed = datetime.fromisoformat(
    archive["completed_at_utc"].replace("Z", "+00:00")
)
attested_at = datetime.now(timezone.utc)
if attested_at <= archive_completed:
    raise SystemExit("cleanup attestation is not after final envelope")
receipt = dict(old)
receipt["schema_version"] = (
    "stage4_exact_empty_dir_cleanup_final_envelope_binding_v1"
)
receipt["pre_binding_receipt_sha256"] = claimed
receipt["final_envelope_binding"] = {
    "archive_receipt_sha256": archive["receipt_sha256"],
    "research_data_identity": archive["research_data_identity"],
    "runtime_contract_identity": archive["runtime_contract_identity"],
    "publication_envelope_identity": archive[
        "publication_envelope_identity"
    ],
    "composite_package_identity": archive[
        "composite_package_identity"
    ],
    "archive_completed_at_utc": archive["completed_at_utc"],
}
receipt["post_final_envelope_absence_verified"] = True
receipt["post_final_envelope_formal_package_verified"] = True
receipt["post_final_envelope_attested_at_utc"] = (
    attested_at.isoformat().replace("+00:00", "Z")
)
receipt.pop("receipt_sha256", None)
receipt["receipt_sha256"] = hashlib.sha256(
    json.dumps(
        receipt, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
).hexdigest()
Path(output_path).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY
  rsync -a \
    "${CLEANUP_REPORT}" \
    "${REMOTE_ALIAS}:${REMOTE_FINAL}/evidence/"
  ssh "${REMOTE_ALIAS}" python3 - \
    "${REMOTE_FINAL}/evidence/archive_receipt.json" \
    "${REMOTE_FINAL}/evidence/${TASK_ID}-stage4-empty-dir-cleanup.json" <<'PY'
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

archive = json.loads(Path(sys.argv[1]).read_bytes())
cleanup = json.loads(Path(sys.argv[2]).read_bytes())
claimed = cleanup["receipt_sha256"]
payload = dict(cleanup)
payload.pop("receipt_sha256")
observed = hashlib.sha256(
    json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
).hexdigest()
if claimed != observed:
    raise SystemExit("cleanup final binding self-hash drift")
binding = cleanup["final_envelope_binding"]
if binding["archive_receipt_sha256"] != archive["receipt_sha256"]:
    raise SystemExit("cleanup/archive receipt binding drift")
for field in (
    "research_data_identity",
    "runtime_contract_identity",
    "publication_envelope_identity",
    "composite_package_identity",
):
    if binding[field] != archive[field]:
        raise SystemExit(f"cleanup/archive identity binding drift: {field}")
archive_completed = datetime.fromisoformat(
    archive["completed_at_utc"].replace("Z", "+00:00")
)
attested = datetime.fromisoformat(
    cleanup["post_final_envelope_attested_at_utc"].replace("Z", "+00:00")
)
if attested <= archive_completed:
    raise SystemExit("cleanup attestation time-order drift")
PY
  printf 'archive trust envelope refreshed: %s\n' "${REMOTE_FINAL}"
  exit 0
fi

ssh "${REMOTE_ALIAS}" python3 - "${REMOTE_USER}" "${REMOTE_PARENT}" "${REMOTE_FINAL}" "${REMOTE_TEMP}" <<'PY'
import os
import pwd
import shutil
import stat
import sys
from pathlib import Path

expected_user, parent_text, final_text, temp_text = sys.argv[1:]
if pwd.getpwuid(os.geteuid()).pw_name != expected_user:
    raise SystemExit("remote execution user drift")
project = Path("/home/molly/project")
project_stat = project.lstat()
if not stat.S_ISDIR(project_stat.st_mode) or stat.S_ISLNK(project_stat.st_mode):
    raise SystemExit("remote project parent is not a real directory")
if not os.access(project, os.W_OK):
    raise SystemExit("remote project parent is not writable")
for path in (Path("/"), Path("/home"), Path("/home/molly"), project):
    observed = path.lstat()
    if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise SystemExit(f"remote parent type drift: {path}")
free = shutil.disk_usage(project).free
if free < 4_683_922_260:
    raise SystemExit(f"insufficient remote space: {free}")
for text in (final_text, temp_text):
    path = Path(text)
    if os.path.lexists(path):
        raise SystemExit(f"remote final/temp already exists: {path}")
parent = Path(parent_text)
if project not in parent.parents:
    raise SystemExit("remote archive parent escaped project sibling root")
print(
    f"remote-preflight-ok user={expected_user} free={free} "
    f"parent={parent}"
)
PY

if [[ "${MODE}" == "--preflight-only" ]]; then
  exit 0
fi

if [[ -e "${ARCHIVE_REPORT}" || -e "${CLEANUP_REPORT}" ]]; then
  printf 'archive or cleanup report already exists\n' >&2
  exit 2
fi

ARCHIVE_STARTED_AT_UTC="$(capture_archive_started_at)"
WORK_ROOT="$(mktemp -d "/tmp/${TASK_ID}-archive.XXXXXX")"
cleanup_work_root() {
  python3 - "${WORK_ROOT}" <<'PY'
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(path)
PY
}
trap cleanup_work_root EXIT

SOURCE_INVENTORY="${WORK_ROOT}/source_inventory.json"
DESTINATION_INVENTORY="${WORK_ROOT}/destination_inventory.json"
POST_INVENTORY="${WORK_ROOT}/post_inventory.json"
ARCHIVE_RECEIPT_TEMP="${WORK_ROOT}/archive_receipt.json"

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
root_stat = root.lstat()
if not stat.S_ISDIR(root_stat.st_mode) or stat.S_ISLNK(root_stat.st_mode):
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
        elif stat.S_ISREG(observed.st_mode):
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
            raise SystemExit(f"forbidden entry type: {relative}")
    pending.extend(reversed(children))
entries.sort(key=lambda row: row["path"])
canonical = json.dumps(
    entries, sort_keys=True, separators=(",", ":")
).encode("ascii")
files = [row for row in entries if row["entry_type"] == "regular_file"]
directories = [row for row in entries if row["entry_type"] == "directory"]
file_inventory = [
    {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
    for row in files
]
full = hashlib.sha256(
    json.dumps(
        file_inventory, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
).hexdigest()
payload = {
    "schema_version": "research_package_exact_tree_inventory_v1",
    "entries": entries,
    "file_count": len(files),
    "directory_count": len(directories),
    "total_bytes": sum(row["bytes"] for row in files),
    "tree_inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    "full_inventory_sha256": full,
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY
}

inventory_tree "${SOURCE_ROOT}" "${SOURCE_INVENTORY}"
python3 - "${SOURCE_INVENTORY}" "${PACKAGE_ID}" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_bytes())
if value["file_count"] != 107:
    raise SystemExit(f"source file count drift: {value['file_count']}")
if value["directory_count"] != 21:
    raise SystemExit(
        f"source directory count drift: {value['directory_count']}"
    )
if value["total_bytes"] != 1_561_307_420:
    raise SystemExit(f"source bytes drift: {value['total_bytes']}")
if value["full_inventory_sha256"] != sys.argv[2]:
    raise SystemExit("source full inventory drift")
PY

ssh "${REMOTE_ALIAS}" python3 - "${REMOTE_PARENT}" "${REMOTE_TEMP}" <<'PY'
import os
import stat
import sys
from pathlib import Path

parent = Path(sys.argv[1])
temp = Path(sys.argv[2])
project = Path("/home/molly/project")
current = project
relative = parent.relative_to(project)
for part in relative.parts:
    target = current / part
    if os.path.lexists(target):
        observed = target.lstat()
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise SystemExit(f"archive parent type drift: {target}")
    else:
        target.mkdir(mode=0o755)
        observed = target.lstat()
        if not stat.S_ISDIR(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
            raise SystemExit(f"created parent type drift: {target}")
        descriptor = os.open(current, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    current = target
temp.mkdir(mode=0o755)
(temp / "package").mkdir()
(temp / "trust_envelope").mkdir()
(temp / "evidence").mkdir()
PY

rsync -a --numeric-ids \
  "${SOURCE_ROOT}/" \
  "${REMOTE_ALIAS}:${REMOTE_TEMP}/package/"
rsync -a --numeric-ids \
  "${CANDIDATE_ROOT}/" \
  "${REMOTE_ALIAS}:${REMOTE_TEMP}/trust_envelope/"
rsync -a --numeric-ids \
  "${PARITY_REPORT}" \
  "${HOSTILE_RECEIPT}" \
  "${FIRST_FULL_RECEIPT}" \
  "${LAYER_ASSIGNMENT}" \
  "${QA_REPORT}" \
  "${DURABLE_RESEARCH_INVENTORY}" \
  "${SOURCE_INVENTORY}" \
  "${REMOTE_ALIAS}:${REMOTE_TEMP}/evidence/"

ssh "${REMOTE_ALIAS}" python3 - "${REMOTE_TEMP}/package" "${REMOTE_TEMP}/evidence/destination_inventory.json" <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
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
        elif stat.S_ISREG(observed.st_mode):
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
            raise SystemExit(f"forbidden entry type: {relative}")
    pending.extend(reversed(children))
entries.sort(key=lambda row: row["path"])
files = [row for row in entries if row["entry_type"] == "regular_file"]
directories = [row for row in entries if row["entry_type"] == "directory"]
file_inventory = [
    {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
    for row in files
]
payload = {
    "schema_version": "research_package_exact_tree_inventory_v1",
    "entries": entries,
    "file_count": len(files),
    "directory_count": len(directories),
    "total_bytes": sum(row["bytes"] for row in files),
    "tree_inventory_sha256": hashlib.sha256(
        json.dumps(
            entries, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest(),
    "full_inventory_sha256": hashlib.sha256(
        json.dumps(
            file_inventory, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest(),
}
output.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
for directory in sorted(
    [path for path in root.rglob("*") if path.is_dir()],
    key=lambda path: len(path.parts),
    reverse=True,
):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
for path in root.rglob("*"):
    if path.is_file():
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
PY

ssh "${REMOTE_ALIAS}" cat "${REMOTE_TEMP}/evidence/destination_inventory.json" > "${DESTINATION_INVENTORY}"
python3 examples/hyperliquid/research_package_trust_cli.py \
  compare-inventories \
  "${SOURCE_INVENTORY}" \
  "${DESTINATION_INVENTORY}"

ssh "${REMOTE_ALIAS}" python3 - "${REMOTE_TEMP}" "${REMOTE_FINAL}" <<'PY'
import os
import sys
from pathlib import Path

temp = Path(sys.argv[1])
final = Path(sys.argv[2])
if os.path.lexists(final):
    raise SystemExit("remote final appeared before publication")
parent_fd = os.open(temp.parent, os.O_RDONLY)
try:
    os.rename(temp, final)
    os.fsync(parent_fd)
finally:
    os.close(parent_fd)
if os.path.lexists(temp) or not final.is_dir():
    raise SystemExit("remote atomic publication failed")
PY

ssh "${REMOTE_ALIAS}" python3 - "${REMOTE_FINAL}/package" "${REMOTE_FINAL}/evidence/post_inventory.json" < /dev/null <<'PY'
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])
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
        elif stat.S_ISREG(observed.st_mode):
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
            raise SystemExit(f"forbidden entry type: {relative}")
    pending.extend(reversed(children))
entries.sort(key=lambda row: row["path"])
files = [row for row in entries if row["entry_type"] == "regular_file"]
directories = [row for row in entries if row["entry_type"] == "directory"]
file_inventory = [
    {"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"]}
    for row in files
]
payload = {
    "schema_version": "research_package_exact_tree_inventory_v1",
    "entries": entries,
    "file_count": len(files),
    "directory_count": len(directories),
    "total_bytes": sum(row["bytes"] for row in files),
    "tree_inventory_sha256": hashlib.sha256(
        json.dumps(
            entries, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
    ).hexdigest(),
    "full_inventory_sha256": hashlib.sha256(
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

ssh "${REMOTE_ALIAS}" cat "${REMOTE_FINAL}/evidence/post_inventory.json" > "${POST_INVENTORY}"
python3 examples/hyperliquid/research_package_trust_cli.py \
  compare-inventories \
  "${SOURCE_INVENTORY}" \
  "${POST_INVENTORY}"

python3 - \
  "${PARITY_REPORT}" \
  "${SOURCE_INVENTORY}" \
  "${ARCHIVE_RECEIPT_TEMP}" \
  "${REMOTE_FINAL}" \
  "${PORTABILITY_CLASS}" \
  "${PORTABILITY_STATEMENT}" \
  "${SOURCE_ROOT}/episode_v3_manifest.json" \
  "${ARCHIVE_STARTED_AT_UTC}" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    parity_path,
    inventory_path,
    output_path,
    archive_root,
    portability_class,
    portability_statement,
    manifest_path,
    started_text,
) = sys.argv[1:]
parity = json.loads(Path(parity_path).read_bytes())
inventory = json.loads(Path(inventory_path).read_bytes())
manifest = json.loads(Path(manifest_path).read_bytes())
identity = parity["kernel"]["identity"]
bindings = []
for name, value in sorted(manifest["dependencies"].items()):
    bindings.append(
        {
            "dependency_id": name,
            "accepted_identity": value["full_inventory_sha256"],
            "source_locator": value["root"],
            "archived_with_this_task": False,
        }
    )
bindings.append(
    {
        "dependency_id": "jul30_accepted_input_inventory",
        "accepted_identity": manifest["input_inventory_sha256_before"],
        "source_locator": str(
            Path(manifest_path).parent / "input_bindings.csv"
        ),
        "archived_with_this_task": False,
    }
)
completed_text = (
    datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
)
parity_completed = datetime.fromisoformat(
    parity["completed_at_utc"].replace("Z", "+00:00")
)
started = datetime.fromisoformat(started_text.replace("Z", "+00:00"))
completed = datetime.fromisoformat(
    completed_text.replace("Z", "+00:00")
)
if not parity_completed < started < completed:
    raise SystemExit("archive chronology is not strict")
receipt = {
    "schema_version": "research_package_stage4_archive_receipt_v1",
    "task_id": "0820T001",
    "source_host": "macmini",
    "destination_host": "amdserver",
    "destination_user": "molly",
    "archive_root": archive_root,
    "transfer_process_mode": "foreground_waited",
    "all_child_exit_status_zero": True,
    "source_inventory_sha256": inventory["tree_inventory_sha256"],
    "destination_inventory_sha256": inventory["tree_inventory_sha256"],
    "legacy_core_sha256": parity["legacy"]["legacy_core_sha256"],
    "legacy_full_inventory_sha256": parity["legacy"][
        "legacy_full_inventory_sha256"
    ],
    "research_data_identity": identity["research_data_identity"],
    "runtime_contract_identity": identity["runtime_contract_identity"],
    "publication_envelope_identity": identity[
        "publication_envelope_identity"
    ],
    "composite_package_identity": identity[
        "composite_package_identity"
    ],
    "byte_exact_package_archive": True,
    "kernel_trust_admission_portable": True,
    "full_source_semantic_replay_portable": False,
    "portability_class": portability_class,
    "portability_statement": portability_statement,
    "external_dependency_bindings": bindings,
    "temp_paths_remaining": 0,
    "task_processes_remaining": 0,
    "started_at_utc": started_text,
    "completed_at_utc": completed_text,
}
canonical = json.dumps(
    receipt, sort_keys=True, separators=(",", ":")
).encode("ascii")
receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
Path(output_path).write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY

cp "${ARCHIVE_RECEIPT_TEMP}" "${ARCHIVE_REPORT}"
rsync -a \
  "${ARCHIVE_RECEIPT_TEMP}" \
  "${REMOTE_ALIAS}:${REMOTE_FINAL}/evidence/archive_receipt.json"

ssh "${REMOTE_ALIAS}" python3 - "${REMOTE_FINAL}" "${REMOTE_TEMP}" "${PACKAGE_ID}" <<'PY'
import json
import os
import sys
from pathlib import Path

final = Path(sys.argv[1])
temp = Path(sys.argv[2])
package_id = sys.argv[3]
if os.path.lexists(temp) or not final.is_dir():
    raise SystemExit("remote final/temp closure failed")
inventory = json.loads(
    (final / "evidence/post_inventory.json").read_bytes()
)
if (
    inventory["file_count"] != 107
    or inventory["directory_count"] != 21
    or inventory["total_bytes"] != 1_561_307_420
    or inventory["full_inventory_sha256"] != package_id
):
    raise SystemExit("remote final inventory drift")
receipt = json.loads(
    (final / "evidence/archive_receipt.json").read_bytes()
)
if (
    receipt["archive_root"] != str(final)
    or receipt["transfer_process_mode"] != "foreground_waited"
    or receipt["all_child_exit_status_zero"] is not True
    or receipt["full_source_semantic_replay_portable"] is not False
    or receipt["temp_paths_remaining"] != 0
    or receipt["task_processes_remaining"] != 0
):
    raise SystemExit("remote archive receipt drift")
worktree = Path("/home/molly/project/hftbacktest")
if final == worktree or worktree in final.parents:
    raise SystemExit("remote archive is inside git worktree")
PY

python3 - \
  "${SOURCE_ROOT}" \
  "${CLEANUP_REPORT}" \
  "${SOURCE_INVENTORY}" <<'PY'
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "examples/hyperliquid"))
from research_package_trust import (  # noqa: E402
    TrustKernelError,
    capture_cleanup_preflight,
    recheck_cleanup_preflight,
)

formal = Path(sys.argv[1])
report_path = Path(sys.argv[2])
source_inventory_path = Path(sys.argv[3])
root = Path("local_live_analysis")
base = "skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3"
allowlist = [root / f"{base} {index}" for index in range(2, 9)]
lock = report_path.with_suffix(".lock")
captured = []
for path in allowlist:
    expected = root / path.name
    if str(path) != str(expected):
        raise TrustKernelError(
            "CLEANUP_PREFLIGHT_FAILED",
            str(path),
            "cleanup raw path differs from frozen allowlist",
        )
captured = capture_cleanup_preflight(allowlist)
for path in allowlist:
    lsof = subprocess.run(
        ["lsof", "+D", str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if lsof.returncode not in {0, 1}:
        raise TrustKernelError(
            "CLEANUP_PREFLIGHT_FAILED",
            str(path),
            f"lsof failed rc={lsof.returncode}",
        )
    if lsof.returncode == 0 and len(lsof.stdout.splitlines()) > 1:
        raise TrustKernelError(
            "CLEANUP_PREFLIGHT_FAILED",
            str(path),
            "cleanup path has an open process",
        )
descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
os.close(descriptor)
deleted = []
rechecked = []
try:
    rechecked = recheck_cleanup_preflight(captured)
    for expected in captured:
        path = Path(expected["path"])
        os.rmdir(path)
        deleted.append(str(path))
finally:
    lock.unlink(missing_ok=True)
if any(os.path.lexists(path) for path in allowlist):
    raise SystemExit("cleanup post-absence check failed")
if not formal.is_dir():
    raise SystemExit("formal Stage 4 package disappeared")
source_inventory = json.loads(source_inventory_path.read_bytes())
manifest = json.loads((formal / "episode_v3_manifest.json").read_bytes())
if (
    manifest["core_package_sha256"]
    != "78be6559c7ac5aaec4d5411b042f7ddcce522473f60f987237bcaa9d2dd5e157"
):
    raise SystemExit("formal package core identity drift after cleanup")
receipt = {
    "schema_version": "stage4_exact_empty_dir_cleanup_v1",
    "task_id": "0820T001",
    "allowlist": [str(path) for path in allowlist],
    "phase_a": captured,
    "phase_b": rechecked,
    "deleted": deleted,
    "mutation_count": len(deleted),
    "recursive_delete_used": False,
    "glob_delete_used": False,
    "formal_package_present": True,
    "formal_full_inventory_sha256": source_inventory[
        "full_inventory_sha256"
    ],
    "completed_at_utc": datetime.now(timezone.utc)
    .isoformat()
    .replace("+00:00", "Z"),
}
canonical = json.dumps(
    receipt, sort_keys=True, separators=(",", ":")
).encode("ascii")
receipt["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()
report_path.write_text(
    json.dumps(receipt, indent=2, sort_keys=True) + "\n",
    encoding="ascii",
)
PY

printf 'archive and cleanup complete: %s\n' "${REMOTE_FINAL}"
