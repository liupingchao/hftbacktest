#!/usr/bin/env python3
"""Stage 4 compatibility adapter for the Trust Kernel v1 candidate."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

try:
    import research_package_trust as trust
    import research_package_trust_cli as kernel_cli
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid import research_package_trust as trust
    from examples.hyperliquid import research_package_trust_cli as kernel_cli


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    REPO_ROOT
    / "local_live_analysis/"
    "skhynix_trigger_aligned_episode_research_v1_stage04_jul30_episode_v3"
)
DURABLE_RESEARCH_INVENTORY = (
    REPO_ROOT
    / ".workflow/reports/"
    "0815T003-round4-pre-repair-research-inventory.csv"
)
LAYER_ASSIGNMENT_PATH = (
    REPO_ROOT / ".workflow/reports/0820T001-stage4-layer-assignment.json"
)
FIRST_FULL_START_PATH = (
    REPO_ROOT / ".workflow/reports/0820T001-first-full-admission-start.json"
)
PARITY_REPORT_PATH = (
    REPO_ROOT / ".workflow/reports/0820T001-stage4-parity.json"
)

ACCEPTED_FILE_COUNT = 107
ACCEPTED_ARTIFACT_COUNT = 106
ACCEPTED_TOTAL_BYTES = 1_561_307_420
ACCEPTED_CORE = "78be6559c7ac5aaec4d5411b042f7ddcce522473f60f987237bcaa9d2dd5e157"
ACCEPTED_FULL = "669fb7d12f25cfa7828aec0fb1546398b1def754952de2290cd19784a477a433"
ACCEPTED_CONTRACT = (
    "b80b9bae2d6cf18cb6e7be4f133467138527fec20eef7054ab38b7f8c70cefde"
)
ACCEPTED_MANIFEST = (
    "2c802336c1446eaf50eaf5ef546b115046abe3a1b69fdd1fcaea31d22a8e64a6"
)
ACCEPTED_RESEARCH = (
    "bb5aed2099b1a97da5b476d4b09dfe4a7bac06f0bf331f8ca864a88daf5c9232"
)
C_PATHS = (
    "frozen_episode_v3_contract.json",
    "runtime_source/cross_exchange_jul30_episode_v3_admission.py",
    "runtime_source/cross_exchange_trigger_aligned_episodes.py",
    "runtime_tests/test_cross_exchange_jul30_episode_v3_admission.py",
    "runtime_tests/test_cross_exchange_trigger_aligned_episodes.py",
)
E_PATHS = (
    "episode_v3_manifest.json",
    "input_bindings.csv",
    "reports/jul30_episode_v3.md",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _durable_rows() -> list[dict[str, Any]]:
    with DURABLE_RESEARCH_INVENTORY.open(
        newline="",
        encoding="ascii",
    ) as handle:
        rows = [
            {
                "path": row["path"],
                "bytes": int(row["bytes"]),
                "sha256": row["sha256"],
            }
            for row in csv.DictReader(handle)
        ]
    trust.validate_inventory_against_surface(
        rows,
        {"inventory_sha256": ACCEPTED_RESEARCH},
    )
    return rows


def stage4_layer_assignment() -> dict[str, list[str]]:
    return {
        "R": [row["path"] for row in _durable_rows()],
        "C": list(C_PATHS),
        "E": list(E_PATHS),
    }


def _candidate_contract() -> dict[str, Any]:
    return {
        "tree_contract": {
            "allowed_entry_types": ["regular_file", "directory"],
        },
        "surface_assignments": stage4_layer_assignment(),
        "runtime_contract": {
            "schema_version": "stage4_kernel_runtime_contract_bridge_v1",
            "kernel_source_tree_sha256": kernel_cli.source_tree_sha256(),
            "stage4_contract_sha256": ACCEPTED_CONTRACT,
            "c_paths": list(C_PATHS),
        },
        "publication_envelope": {
            "schema_version": "stage4_publication_envelope_bridge_v1",
            "legacy_core_sha256": ACCEPTED_CORE,
            "legacy_full_inventory_sha256": ACCEPTED_FULL,
            "manifest_raw_sha256": ACCEPTED_MANIFEST,
            "atomic_publication_required": True,
            "verify_only_zero_write_required": True,
            "e_paths": list(E_PATHS),
        },
        "semantic_evidence_contract": {
            "required_keys": [
                "legacy_core_sha256",
                "legacy_full_inventory_sha256",
                "artifact_count",
                "file_count",
                "total_bytes",
                "source_semantic_verified",
            ],
            "field_types": {
                "legacy_core_sha256": "string",
                "legacy_full_inventory_sha256": "string",
                "artifact_count": "integer",
                "file_count": "integer",
                "total_bytes": "integer",
                "source_semantic_verified": "boolean",
            },
            "exact_values": {
                "legacy_core_sha256": ACCEPTED_CORE,
                "legacy_full_inventory_sha256": ACCEPTED_FULL,
                "artifact_count": ACCEPTED_ARTIFACT_COUNT,
                "file_count": ACCEPTED_FILE_COUNT,
                "total_bytes": ACCEPTED_TOTAL_BYTES,
                "source_semantic_verified": True,
            },
        },
    }


def verify_research_data_anchor(
    package_root: Path = PACKAGE_ROOT,
) -> dict[str, Any]:
    durable = _durable_rows()
    observed = trust.build_inventory(
        package_root,
        {"paths": [row["path"] for row in durable]},
    )
    if observed != durable:
        raise trust.TrustKernelError(
            "RESEARCH_DATA_INVENTORY_MISMATCH",
            str(package_root),
            "observed research files differ from durable accepted rows",
        )
    identity = trust.compute_research_data_identity(observed)
    if identity != ACCEPTED_RESEARCH:
        raise trust.TrustKernelError(
            "RESEARCH_DATA_IDENTITY_MISMATCH",
            str(package_root),
            f"expected {ACCEPTED_RESEARCH}, observed {identity}",
        )
    return {
        "verified": True,
        "research_file_count": len(observed),
        "research_total_bytes": sum(row["bytes"] for row in observed),
        "research_inventory_sha256": identity,
    }


def _validate_hostile_before_full(receipt_path: Path) -> dict[str, Any]:
    receipt = kernel_cli.validate_hostile_receipt(receipt_path)
    completed = datetime.fromisoformat(
        receipt["completed_at_utc"].replace("Z", "+00:00")
    )
    if completed >= datetime.now(timezone.utc):
        raise trust.TrustKernelError(
            "HOSTILE_PREFLIGHT_TIME_INVALID",
            "$.completed_at_utc",
            receipt["completed_at_utc"],
        )
    return receipt


def _write_first_full_start(receipt: Mapping[str, Any]) -> dict[str, Any]:
    if os.path.lexists(FIRST_FULL_START_PATH):
        raise trust.TrustKernelError(
            "FIRST_FULL_ADMISSION_ALREADY_STARTED",
            str(FIRST_FULL_START_PATH),
            "first full-admission receipt already exists",
        )
    start = {
        "schema_version": (
            "research_package_first_full_admission_start_v1"
        ),
        "task_id": kernel_cli.TASK_ID,
        "hostile_receipt_sha256": receipt["receipt_sha256"],
        "kernel_source_tree_sha256": receipt[
            "kernel_source_tree_sha256"
        ],
        "surface_matrix_sha256": receipt["surface_matrix_sha256"],
        "started_at_utc": _utc_now(),
    }
    start["receipt_sha256"] = trust.canonical_json_sha256(start)
    hostile_completed = datetime.fromisoformat(
        receipt["completed_at_utc"].replace("Z", "+00:00")
    )
    started = datetime.fromisoformat(
        start["started_at_utc"].replace("Z", "+00:00")
    )
    if hostile_completed >= started:
        raise trust.TrustKernelError(
            "HOSTILE_FIRST_ORDER_VIOLATION",
            "$.started_at_utc",
            "full admission did not start after hostile completion",
        )
    trust.atomic_write_json(FIRST_FULL_START_PATH, start)
    return start


def _legacy_admission(package_root: Path) -> dict[str, Any]:
    try:
        import cross_exchange_trigger_aligned_episodes as episode_v3
    except ModuleNotFoundError:  # pragma: no cover - package import path
        from examples.hyperliquid import (
            cross_exchange_trigger_aligned_episodes as episode_v3,
        )

    manifest = episode_v3.verify_package(package_root)
    episode_v3._validate_source_semantic_verification_evidence(manifest)
    inventory = episode_v3._directory_inventory(package_root)
    return {
        "legacy_core_sha256": manifest["core_package_sha256"],
        "legacy_full_inventory_sha256": episode_v3.canonical_json_sha256(
            inventory
        ),
        "contract_sha256": episode_v3.sha256_file(
            package_root / "frozen_episode_v3_contract.json"
        ),
        "manifest_raw_sha256": episode_v3.sha256_file(
            package_root / "episode_v3_manifest.json"
        ),
        "artifact_count": len(manifest["artifacts"]),
        "file_count": len(inventory),
        "total_bytes": sum(int(row["bytes"]) for row in inventory),
        "source_semantic_verified": True,
    }


def run_full_parity(
    receipt_path: Path,
    package_root: Path = PACKAGE_ROOT,
) -> dict[str, Any]:
    hostile = _validate_hostile_before_full(receipt_path)
    start = _write_first_full_start(hostile)
    before = trust.metadata_snapshot(package_root)
    legacy = _legacy_admission(package_root)
    expected_legacy = {
        "legacy_core_sha256": ACCEPTED_CORE,
        "legacy_full_inventory_sha256": ACCEPTED_FULL,
        "contract_sha256": ACCEPTED_CONTRACT,
        "manifest_raw_sha256": ACCEPTED_MANIFEST,
        "artifact_count": ACCEPTED_ARTIFACT_COUNT,
        "file_count": ACCEPTED_FILE_COUNT,
        "total_bytes": ACCEPTED_TOTAL_BYTES,
        "source_semantic_verified": True,
    }
    if legacy != expected_legacy:
        raise trust.TrustKernelError(
            "STAGE4_LEGACY_PARITY_MISMATCH",
            str(package_root),
            f"expected={expected_legacy} observed={legacy}",
        )
    contract = _candidate_contract()
    semantic = {
        key: legacy[key]
        for key in (
            "legacy_core_sha256",
            "legacy_full_inventory_sha256",
            "artifact_count",
            "file_count",
            "total_bytes",
            "source_semantic_verified",
        )
    }
    candidate = trust.admit_package(package_root, contract, semantic)
    if candidate.identity.research_data_identity != ACCEPTED_RESEARCH:
        raise trust.TrustKernelError(
            "RESEARCH_DATA_IDENTITY_MISMATCH",
            str(package_root),
            candidate.identity.research_data_identity,
        )
    after = trust.metadata_snapshot(package_root)
    trust.assert_zero_write_snapshot(
        before,
        after,
        location=str(package_root),
    )
    assignment = {
        "schema_version": "stage4_layer_assignment_v1",
        "task_id": kernel_cli.TASK_ID,
        "package_root": str(package_root),
        "surface_assignments": contract["surface_assignments"],
        "file_count": sum(
            len(paths)
            for paths in contract["surface_assignments"].values()
        ),
    }
    trust.atomic_write_json(LAYER_ASSIGNMENT_PATH, assignment)
    result = {
        "schema_version": "stage4_trust_kernel_parity_v1",
        "task_id": kernel_cli.TASK_ID,
        "hostile_receipt_sha256": hostile["receipt_sha256"],
        "first_full_admission_start_sha256": start["receipt_sha256"],
        "legacy": legacy,
        "kernel": candidate.as_dict(),
        "research_anchor": verify_research_data_anchor(package_root),
        "pre_post_metadata_exact": True,
        "full_rebuild_count": 0,
        "package_mutation_count": 0,
        "completed_at_utc": _utc_now(),
        "verified": True,
    }
    result["report_sha256"] = trust.canonical_json_sha256(result)
    trust.atomic_write_json(PARITY_REPORT_PATH, result)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        default=str(PACKAGE_ROOT),
    )
    parser.add_argument("--verify-research-data-anchor", action="store_true")
    parser.add_argument("--full-admission", action="store_true")
    parser.add_argument("--require-hostile-receipt")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    package_root = Path(args.package_dir)
    try:
        if args.full_admission:
            if not args.require_hostile_receipt:
                raise trust.TrustKernelError(
                    "HOSTILE_PREFLIGHT_REQUIRED",
                    "$.argv",
                    "--require-hostile-receipt is required",
                )
            result = run_full_parity(
                Path(args.require_hostile_receipt),
                package_root,
            )
        elif args.verify_research_data_anchor:
            result = verify_research_data_anchor(package_root)
        else:
            raise trust.TrustKernelError(
                "VERIFY_MODE_REQUIRED",
                "$.argv",
                "select --verify-research-data-anchor or --full-admission",
            )
    except (OSError, trust.TrustKernelError, KeyError, ValueError) as exc:
        error = (
            exc.as_dict()
            if isinstance(exc, trust.TrustKernelError)
            else {
                "code": "STAGE4_ADAPTER_ERROR",
                "location": "$",
                "detail": str(exc),
            }
        )
        print(json.dumps({"verified": False, "error": error}, indent=2))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
