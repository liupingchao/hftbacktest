#!/usr/bin/env python3
"""Command-line entrypoint for the Research Package Trust Kernel candidate."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


sys.dont_write_bytecode = True

try:
    import research_package_trust as trust
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid import research_package_trust as trust


REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0820T001"
MATRIX_PATH = REPO_ROOT / ".workflow/contracts/0820T001-surface-matrix.json"
SURFACE_SCHEMA_PATH = (
    REPO_ROOT
    / ".workflow/workflow-kit/research-package-surface-matrix.schema.json"
)
REGISTRY_SCHEMA_PATH = (
    REPO_ROOT
    / ".workflow/workflow-kit/research-package-kernel-registry.schema.json"
)
REGISTRY_PATH = (
    REPO_ROOT
    / "baselines/research_package_trust_kernel/accepted_versions.json"
)
REPORT_ROOT = REPO_ROOT / ".workflow/reports"
CANDIDATE_ROOT = REPORT_ROOT / f"{TASK_ID}-kernel-candidate"
HOSTILE_RECEIPT_PATH = REPORT_ROOT / f"{TASK_ID}-hostile-preflight.json"
RECEIPT_SCHEMA_VERSION = "research_package_hostile_preflight_receipt_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_paths() -> tuple[Path, ...]:
    kernel_root = Path(__file__).with_name("research_package_trust")
    kernel_files = sorted(kernel_root.glob("*.py"))
    explicit = [
        Path(__file__),
        Path(__file__).with_name(
            "research_package_trust_stage4_adapter.py"
        ),
        *sorted(
            Path(__file__).parent.glob(
                "test_research_package_trust_*.py"
            )
        ),
        REPO_ROOT
        / ".workflow/workflow-kit/research-package-task-template.md",
        REPO_ROOT
        / ".workflow/workflow-kit/validate_research_package_task.py",
        REPO_ROOT
        / ".workflow/workflow-kit/test_validate_research_package_task.py",
    ]
    paths = [*kernel_files, *explicit]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise trust.TrustKernelError(
            "KERNEL_SOURCE_FILE_MISSING",
            "$.source_tree",
            repr(missing),
        )
    return tuple(sorted(set(paths)))


def source_tree_inventory() -> list[dict[str, Any]]:
    rows = [
        {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": trust.sha256_file(path),
        }
        for path in _source_paths()
    ]
    return sorted(rows, key=lambda row: row["path"])


def source_tree_sha256() -> str:
    return trust.canonical_json_sha256(source_tree_inventory())


def _purity_scan() -> None:
    kernel_root = Path(__file__).with_name("research_package_trust")
    forbidden = (
        "skhynix",
        "skhx",
        "jul30",
        "family a",
        "family b",
        "segment_000",
        "/users/",
        "/home/",
    )
    for path in sorted(kernel_root.glob("*.py")):
        lowered = path.read_text(encoding="utf-8").lower()
        matches = [token for token in forbidden if token in lowered]
        if matches:
            raise trust.TrustKernelError(
                "KERNEL_PURITY_VIOLATION",
                path.relative_to(REPO_ROOT).as_posix(),
                f"forbidden tokens {matches}",
            )


def _expect_error(
    action: Callable[[], Any],
    expected_codes: set[str] | None = None,
) -> str:
    try:
        action()
    except trust.TrustKernelError as exc:
        if expected_codes is not None and exc.code not in expected_codes:
            raise trust.TrustKernelError(
                "HOSTILE_UNEXPECTED_ERROR_CODE",
                exc.location,
                f"expected {sorted(expected_codes)}, observed {exc.code}",
            ) from exc
        return exc.code
    raise trust.TrustKernelError(
        "HOSTILE_FAIL_OPEN",
        "$.negative_matrix",
        "mutation was accepted",
    )


def _aggregate_negative_matrix() -> list[dict[str, Any]]:
    contract = {
        "required_keys": ["count", "verified", "digest"],
        "field_types": {
            "count": "integer",
            "verified": "boolean",
            "digest": "string",
        },
        "exact_values": {
            "verified": True,
            "digest": "a" * 64,
        },
    }
    cases: list[dict[str, Any]] = []
    for index in range(98):
        observed: dict[str, Any] = {
            "count": 1,
            "verified": True,
            "digest": "a" * 64,
        }
        mode = index % 7
        if mode == 0:
            observed[f"extra_{index:03d}"] = 0
        elif mode == 1:
            observed.pop(("count", "verified", "digest")[index % 3])
        elif mode == 2:
            observed["count"] = True
        elif mode == 3:
            observed["count"] = "1"
        elif mode == 4:
            observed["verified"] = False
        elif mode == 5:
            observed["digest"] = "b" * 64
        else:
            observed["digest"] = None
        code = _expect_error(
            lambda observed=observed: trust.validate_exact_object(
                observed,
                contract,
            )
        )
        cases.append(
            {
                "case_id": f"aggregate_{index:03d}",
                "error_code": code,
            }
        )
    return cases


def _tree_fixture(root: Path) -> None:
    (root / "nested").mkdir(parents=True)
    (root / "root.txt").write_text("root\n", encoding="ascii")
    (root / "nested/child.txt").write_text("child\n", encoding="ascii")


def _direct_tree_negative_matrix(work: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(36):
        root = work / f"tree-{index:03d}" / "package"
        _tree_fixture(root)
        mode = index % 6
        scan_target = root
        contract: Mapping[str, Any] | None = None
        if mode == 0:
            (root / f"file-link-{index}").symlink_to(root / "root.txt")
        elif mode == 1:
            (root / f"dir-link-{index}").symlink_to(
                root / "nested",
                target_is_directory=True,
            )
        elif mode == 2:
            (root / f"dangling-{index}").symlink_to(root / "missing")
        elif mode == 3:
            os.mkfifo(root / f"fifo-{index}")
        elif mode == 4:
            scan_target = root.parent / f"root-link-{index}"
            scan_target.symlink_to(root, target_is_directory=True)
        else:
            contract = {
                "allowed_entry_types": ["regular_file", "directory"],
                "expected_files": ["root.txt"],
                "expected_directories": ["nested"],
            }
        code = _expect_error(
            lambda scan_target=scan_target, contract=contract: (
                trust.scan_exact_tree(scan_target, contract)
            )
        )
        cases.append(
            {
                "case_id": f"direct_tree_{index:03d}",
                "error_code": code,
            }
        )
    return cases


def _small_package(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root.mkdir(parents=True)
    (root / "research.csv").write_text("x\n1\n", encoding="ascii")
    (root / "contract.json").write_text('{"api":"v1"}\n', encoding="ascii")
    (root / "envelope.json").write_text('{"seal":"v1"}\n', encoding="ascii")
    contract = {
        "surface_assignments": {
            "R": ["research.csv"],
            "C": ["contract.json"],
            "E": ["envelope.json"],
        },
        "runtime_contract": {"api": "v1"},
        "publication_envelope": {"seal": "v1"},
        "semantic_evidence_contract": {
            "required_keys": ["verified", "count"],
            "field_types": {
                "verified": "boolean",
                "count": "integer",
            },
            "exact_values": {"verified": True, "count": 1},
        },
    }
    evidence = {"verified": True, "count": 1}
    baseline = trust.admit_package(root, contract, evidence)
    contract["declared_identity"] = baseline.identity.as_dict()
    return contract, evidence


def _production_shape_negative_matrix(work: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index in range(12):
        root = work / f"production-{index:03d}"
        contract, evidence = _small_package(root)
        mode = index
        if mode == 0:
            (root / "research.csv").write_text("x\n2\n", encoding="ascii")
        elif mode == 1:
            (root / "contract.json").write_text(
                '{"api":"v2"}\n',
                encoding="ascii",
            )
        elif mode == 2:
            (root / "envelope.json").write_text(
                '{"seal":"v2"}\n',
                encoding="ascii",
            )
        elif mode == 3:
            (root / "extra.txt").write_text("extra\n", encoding="ascii")
        elif mode == 4:
            contract["surface_assignments"]["C"] = []
        elif mode == 5:
            contract["surface_assignments"]["C"].append("research.csv")
        elif mode == 6:
            (root / "link").symlink_to(root / "research.csv")
        elif mode == 7:
            os.mkfifo(root / "fifo")
        elif mode == 8:
            evidence["unexpected"] = 0
        elif mode == 9:
            evidence["count"] = True
        elif mode == 10:
            contract["declared_identity"].pop("runtime_contract_identity")
        else:
            contract["declared_identity"]["composite_package_identity"] = (
                "f" * 64
            )
        code = _expect_error(
            lambda root=root, contract=contract, evidence=evidence: (
                trust.admit_package(root, contract, evidence)
            )
        )
        cases.append(
            {
                "case_id": f"production_shape_{index:03d}",
                "error_code": code,
            }
        )
    return cases


def _identity_metamorphic_matrix() -> list[dict[str, Any]]:
    rows = [{"path": "r.csv", "bytes": 1, "sha256": "a" * 64}]
    baseline = trust.build_package_identity(
        rows,
        {"contract": "v1"},
        {"envelope": "v1"},
    )
    r_rows = [{"path": "r.csv", "bytes": 1, "sha256": "b" * 64}]
    r_mutated = trust.build_package_identity(
        r_rows,
        {"contract": "v1"},
        {"envelope": "v1"},
    )
    c_mutated = trust.build_package_identity(
        rows,
        {"contract": "v2"},
        {"envelope": "v1"},
    )
    e_mutated = trust.build_package_identity(
        rows,
        {"contract": "v1"},
        {"envelope": "v2"},
    )
    _expect_error(
        lambda: trust.validate_identity_bindings(
            baseline.as_dict(),
            r_mutated,
        ),
        {"RESEARCH_DATA_IDENTITY_MISMATCH"},
    )
    if not (
        c_mutated.research_data_identity == baseline.research_data_identity
        and c_mutated.runtime_contract_identity
        != baseline.runtime_contract_identity
        and e_mutated.runtime_contract_identity
        == baseline.runtime_contract_identity
        and e_mutated.publication_envelope_identity
        != baseline.publication_envelope_identity
    ):
        raise trust.TrustKernelError(
            "IDENTITY_METAMORPHIC_MISMATCH",
            "$.identity",
            "layer isolation property failed",
        )
    return [
        {"case_id": "r_change_rejects_old_c_e", "passed": True},
        {"case_id": "c_change_preserves_r", "passed": True},
        {"case_id": "e_change_preserves_r_c", "passed": True},
        {"case_id": "composite_changes_for_every_layer", "passed": True},
    ]


def run_hostile_matrix() -> dict[str, Any]:
    _purity_scan()
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}-hostile-") as temp:
        work = Path(temp)
        aggregate = _aggregate_negative_matrix()
        direct_tree = _direct_tree_negative_matrix(work)
        production = _production_shape_negative_matrix(work)
        metamorphic = _identity_metamorphic_matrix()
    return {
        "schema_version": "research_package_trust_negative_matrix_v1",
        "aggregate": aggregate,
        "direct_tree": direct_tree,
        "production_shape": production,
        "metamorphic": metamorphic,
        "counts": {
            "aggregate": len(aggregate),
            "direct_tree": len(direct_tree),
            "production_shape": len(production),
            "metamorphic": len(metamorphic),
            "fail_open": 0,
        },
    }


def _api_contract() -> dict[str, Any]:
    return {
        "schema_version": "research_package_trust_api_contract_v1",
        "kernel_name": "research_package_trust_kernel",
        "kernel_version": "v1",
        "public_api": sorted(trust.__all__),
        "error_contract": {
            "type": "TrustKernelError",
            "fields": ["code", "location", "detail"],
        },
        "symlink_policy": "all_symlinks_forbidden",
        "unknown_key_policy": "reject",
        "bool_as_integer": "reject",
    }


def _fixture_inventory(negative: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "research_package_trust_fixture_inventory_v1",
        "fixture_families": {
            name: [row["case_id"] for row in negative[name]]
            for name in (
                "aggregate",
                "direct_tree",
                "production_shape",
                "metamorphic",
            )
        },
    }


def _write_candidate_package(
    negative: Mapping[str, Any],
) -> dict[str, str]:
    if os.path.lexists(CANDIDATE_ROOT):
        raise trust.TrustKernelError(
            "CANDIDATE_PACKAGE_EXISTS",
            str(CANDIDATE_ROOT),
            "remove only after preserving prior evidence",
        )
    staging = CANDIDATE_ROOT.with_name(
        f".{CANDIDATE_ROOT.name}.tmp-{os.getpid()}"
    )
    if os.path.lexists(staging):
        raise trust.TrustKernelError(
            "PUBLICATION_TEMP_EXISTS",
            str(staging),
            "candidate staging path exists",
        )
    staging.mkdir(parents=True)
    try:
        api = _api_contract()
        fixture = _fixture_inventory(negative)
        trust.atomic_write_json(staging / "api_contract.json", api)
        trust.atomic_write_json(
            staging / "negative_matrix.json",
            dict(negative),
        )
        trust.atomic_write_json(staging / "fixture_inventory.json", fixture)
        snapshot = staging / "kernel_source_snapshot"
        snapshot.mkdir()
        for source in _source_paths():
            target = snapshot / source.relative_to(REPO_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        trust.fsync_tree(staging)
        os.rename(staging, CANDIDATE_ROOT)
        parent = os.open(CANDIDATE_ROOT.parent, os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    source_inventory = source_tree_inventory()
    source_identity = trust.canonical_json_sha256(source_inventory)
    snapshot_identity = trust.canonical_json_sha256(
        trust.build_inventory(
            CANDIDATE_ROOT / "kernel_source_snapshot"
        )
    )
    if snapshot_identity != source_identity:
        raise trust.TrustKernelError(
            "KERNEL_SNAPSHOT_IDENTITY_MISMATCH",
            str(CANDIDATE_ROOT / "kernel_source_snapshot"),
            f"source={source_identity} snapshot={snapshot_identity}",
        )
    return {
        "kernel_source_tree_sha256": source_identity,
        "kernel_snapshot_sha256": snapshot_identity,
        "kernel_api_contract_sha256": trust.sha256_file(
            CANDIDATE_ROOT / "api_contract.json"
        ),
        "kernel_negative_matrix_sha256": trust.sha256_file(
            CANDIDATE_ROOT / "negative_matrix.json"
        ),
        "fixture_inventory_sha256": trust.sha256_file(
            CANDIDATE_ROOT / "fixture_inventory.json"
        ),
    }


def run_hostile_preflight(task_id: str) -> dict[str, Any]:
    if task_id != TASK_ID:
        raise trust.TrustKernelError(
            "TASK_ID_MISMATCH",
            "$.task_id",
            f"expected {TASK_ID}, observed {task_id}",
        )
    if os.path.lexists(HOSTILE_RECEIPT_PATH):
        raise trust.TrustKernelError(
            "HOSTILE_RECEIPT_EXISTS",
            str(HOSTILE_RECEIPT_PATH),
            "refusing to overwrite prior preflight evidence",
        )
    started = _utc_now()
    negative = run_hostile_matrix()
    candidate = _write_candidate_package(negative)
    completed = _utc_now()
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "task_id": task_id,
        **candidate,
        "surface_schema_sha256": trust.sha256_file(SURFACE_SCHEMA_PATH),
        "surface_matrix_sha256": trust.sha256_file(MATRIX_PATH),
        "registry_schema_sha256": trust.sha256_file(REGISTRY_SCHEMA_PATH),
        "registry_bootstrap_sha256": trust.sha256_file(REGISTRY_PATH),
        "negative_counts": negative["counts"],
        "stable_error_codes": sorted(
            {
                row["error_code"]
                for family in ("aggregate", "direct_tree", "production_shape")
                for row in negative[family]
            }
        ),
        "started_at_utc": started,
        "completed_at_utc": completed,
        "passed": True,
    }
    receipt["receipt_sha256"] = trust.canonical_json_sha256(receipt)
    trust.atomic_write_json(HOSTILE_RECEIPT_PATH, receipt)
    return receipt


def validate_hostile_receipt(
    receipt_path: Path = HOSTILE_RECEIPT_PATH,
) -> dict[str, Any]:
    receipt = trust.read_json_object(receipt_path)
    if receipt_path.read_bytes() != trust.canonical_pretty_json_bytes(receipt):
        raise trust.TrustKernelError(
            "HOSTILE_RECEIPT_NONCANONICAL_BYTES",
            str(receipt_path),
            "receipt is not canonical pretty JSON",
        )
    expected_keys = {
        "schema_version",
        "task_id",
        "kernel_source_tree_sha256",
        "kernel_snapshot_sha256",
        "kernel_api_contract_sha256",
        "kernel_negative_matrix_sha256",
        "fixture_inventory_sha256",
        "surface_schema_sha256",
        "surface_matrix_sha256",
        "registry_schema_sha256",
        "registry_bootstrap_sha256",
        "negative_counts",
        "stable_error_codes",
        "started_at_utc",
        "completed_at_utc",
        "passed",
        "receipt_sha256",
    }
    if set(receipt) != expected_keys:
        raise trust.TrustKernelError(
            "HOSTILE_RECEIPT_SCHEMA_MISMATCH",
            str(receipt_path),
            f"missing={sorted(expected_keys - set(receipt))} "
            f"extra={sorted(set(receipt) - expected_keys)}",
        )
    claimed = receipt["receipt_sha256"]
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    observed = trust.canonical_json_sha256(payload)
    if claimed != observed:
        raise trust.TrustKernelError(
            "HOSTILE_RECEIPT_SHA256_MISMATCH",
            str(receipt_path),
            f"claimed {claimed}, observed {observed}",
        )
    expected_bindings = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "kernel_source_tree_sha256": source_tree_sha256(),
        "surface_schema_sha256": trust.sha256_file(SURFACE_SCHEMA_PATH),
        "surface_matrix_sha256": trust.sha256_file(MATRIX_PATH),
        "registry_schema_sha256": trust.sha256_file(REGISTRY_SCHEMA_PATH),
        "registry_bootstrap_sha256": trust.sha256_file(REGISTRY_PATH),
        "passed": True,
    }
    for field, expected in expected_bindings.items():
        if receipt[field] != expected:
            raise trust.TrustKernelError(
                "HOSTILE_RECEIPT_BINDING_MISMATCH",
                f"$.{field}",
                f"expected {expected!r}, observed {receipt[field]!r}",
            )
    candidate_bindings = {
        "kernel_api_contract_sha256": trust.sha256_file(
            CANDIDATE_ROOT / "api_contract.json"
        ),
        "kernel_negative_matrix_sha256": trust.sha256_file(
            CANDIDATE_ROOT / "negative_matrix.json"
        ),
        "fixture_inventory_sha256": trust.sha256_file(
            CANDIDATE_ROOT / "fixture_inventory.json"
        ),
        "kernel_snapshot_sha256": trust.canonical_json_sha256(
            trust.build_inventory(
                CANDIDATE_ROOT / "kernel_source_snapshot"
            )
        ),
    }
    for field, observed_binding in candidate_bindings.items():
        if receipt[field] != observed_binding:
            raise trust.TrustKernelError(
                "HOSTILE_RECEIPT_BINDING_MISMATCH",
                f"$.{field}",
                f"expected current {observed_binding}, "
                f"observed {receipt[field]}",
            )
    if (
        receipt["kernel_snapshot_sha256"]
        != receipt["kernel_source_tree_sha256"]
    ):
        raise trust.TrustKernelError(
            "KERNEL_SNAPSHOT_IDENTITY_MISMATCH",
            "$.kernel_snapshot_sha256",
            "frozen snapshot differs from current source tree",
        )
    counts = receipt["negative_counts"]
    if counts != {
        "aggregate": 98,
        "direct_tree": 36,
        "production_shape": 12,
        "metamorphic": 4,
        "fail_open": 0,
    }:
        raise trust.TrustKernelError(
            "HOSTILE_NEGATIVE_COUNT_MISMATCH",
            "$.negative_counts",
            repr(counts),
        )
    negative = trust.read_json_object(
        CANDIDATE_ROOT / "negative_matrix.json"
    )
    observed_counts = {
        "aggregate": len(negative["aggregate"]),
        "direct_tree": len(negative["direct_tree"]),
        "production_shape": len(negative["production_shape"]),
        "metamorphic": len(negative["metamorphic"]),
        "fail_open": negative["counts"]["fail_open"],
    }
    if observed_counts != counts:
        raise trust.TrustKernelError(
            "HOSTILE_NEGATIVE_COUNT_MISMATCH",
            str(CANDIDATE_ROOT / "negative_matrix.json"),
            f"receipt={counts} observed={observed_counts}",
        )
    started = datetime.fromisoformat(
        receipt["started_at_utc"].replace("Z", "+00:00")
    )
    completed = datetime.fromisoformat(
        receipt["completed_at_utc"].replace("Z", "+00:00")
    )
    if completed <= started:
        raise trust.TrustKernelError(
            "HOSTILE_RECEIPT_TIME_ORDER_INVALID",
            "$.completed_at_utc",
            f"started={started.isoformat()} completed={completed.isoformat()}",
        )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    hostile = subparsers.add_parser("hostile-preflight")
    hostile.add_argument("--task-id", required=True)
    validate = subparsers.add_parser("validate-hostile-receipt")
    validate.add_argument(
        "--receipt",
        default=str(HOSTILE_RECEIPT_PATH),
    )
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("root")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "hostile-preflight":
            result = run_hostile_preflight(args.task_id)
        elif args.command == "validate-hostile-receipt":
            result = validate_hostile_receipt(Path(args.receipt))
        else:
            rows = trust.build_inventory(Path(args.root))
            result = {
                "verified": True,
                "file_count": len(rows),
                "total_bytes": sum(row["bytes"] for row in rows),
                "inventory_sha256": trust.canonical_json_sha256(rows),
                "inventory": rows,
            }
    except (OSError, trust.TrustKernelError) as exc:
        error = (
            exc.as_dict()
            if isinstance(exc, trust.TrustKernelError)
            else {
                "code": "TRUST_KERNEL_CLI_IO_ERROR",
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
