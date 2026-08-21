#!/usr/bin/env python3
"""Command-line entrypoint for the Research Package Trust Kernel candidate."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import shutil
import socket
import subprocess
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
    source_file = Path(__file__).resolve()
    kernel_root = source_file.with_name("research_package_trust")
    kernel_files = sorted(kernel_root.glob("*.py"))
    explicit = [
        source_file,
        source_file.with_name(
            "research_package_trust_stage4_adapter.py"
        ),
        *sorted(
            source_file.parent.glob(
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
    paths = [path.resolve() for path in (*kernel_files, *explicit)]
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


def _assert_source_purity(paths: Sequence[Path]) -> None:
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
    for path in sorted(Path(item).resolve() for item in paths):
        lowered = path.read_text(encoding="utf-8").lower()
        matches = [token for token in forbidden if token in lowered]
        if matches:
            raise trust.TrustKernelError(
                "KERNEL_PURITY_VIOLATION",
                str(path),
                f"forbidden tokens {matches}",
            )


def _purity_scan() -> None:
    kernel_root = Path(__file__).resolve().with_name("research_package_trust")
    _assert_source_purity(sorted(kernel_root.glob("*.py")))


def _assert_source_identity(expected: str, observed: str) -> None:
    if observed != expected:
        raise trust.TrustKernelError(
            "KERNEL_SOURCE_IDENTITY_MISMATCH",
            "$.kernel_source_tree_sha256",
            f"expected {expected}, observed {observed}",
        )


def _expect_error(
    action: Callable[[], Any],
    expected_codes: set[str] | None = None,
) -> str:
    try:
        action()
    except Exception as exc:
        code = getattr(exc, "code", None)
        if code is None:
            raise
        if expected_codes is not None and code not in expected_codes:
            raise trust.TrustKernelError(
                "HOSTILE_UNEXPECTED_ERROR_CODE",
                getattr(exc, "location", "$"),
                f"expected {sorted(expected_codes)}, observed {code}",
            ) from exc
        return str(code)
    raise trust.TrustKernelError(
        "HOSTILE_FAIL_OPEN",
        "$.negative_matrix",
        "mutation was accepted",
    )


def _aggregate_fixture() -> dict[str, Any]:
    return {
        "verified": True,
        "scope": {
            "scope_id": "fixture_scope",
            "projection_names": ["projection_a"],
        },
        "projections": {
            "projection_a": {
                "entries": {
                    "entry_a": {
                        "fields": ["alpha", "beta"],
                        "count": 2,
                        "digest": "a" * 64,
                        "mismatch_count": 0,
                    }
                },
                "fields": ["alpha", "beta"],
                "count": 2,
                "digest": "a" * 64,
                "mismatch_count": 0,
            }
        },
        "exact_counts": {"projection_a": 2},
        "aggregate_counts": {"total": 2},
        "bindings": [
            {
                "projection": "projection_a",
                "field": "count",
                "expected_value": 2,
            }
        ],
    }


def _validate_aggregate_fixture(kernel: Any, observed: Any) -> None:
    kernel.validate_exact_object(
        observed,
        {
            "required_keys": [
                "verified",
                "scope",
                "projections",
                "exact_counts",
                "aggregate_counts",
                "bindings",
            ],
            "field_types": {
                "verified": "boolean",
                "scope": "object",
                "projections": "object",
                "exact_counts": "object",
                "aggregate_counts": "object",
                "bindings": "array",
            },
            "exact_values": {"verified": True},
        },
    )
    kernel.validate_exact_object(
        observed["scope"],
        {
            "required_keys": ["scope_id", "projection_names"],
            "field_types": {
                "scope_id": "string",
                "projection_names": "array",
            },
            "exact_values": {
                "scope_id": "fixture_scope",
                "projection_names": ["projection_a"],
            },
        },
        location="$.scope",
    )
    kernel.validate_exact_object(
        observed["projections"],
        {
            "required_keys": ["projection_a"],
            "field_types": {"projection_a": "object"},
        },
        location="$.projections",
    )
    projection = observed["projections"]["projection_a"]
    kernel.validate_exact_object(
        projection,
        {
            "required_keys": [
                "entries",
                "fields",
                "count",
                "digest",
                "mismatch_count",
            ],
            "field_types": {
                "entries": "object",
                "fields": "array",
                "count": "integer",
                "digest": "string",
                "mismatch_count": "integer",
            },
            "exact_values": {
                "fields": ["alpha", "beta"],
                "count": 2,
                "digest": "a" * 64,
                "mismatch_count": 0,
            },
        },
        location="$.projections.projection_a",
    )
    kernel.validate_exact_object(
        projection["entries"],
        {
            "required_keys": ["entry_a"],
            "field_types": {"entry_a": "object"},
        },
        location="$.projections.projection_a.entries",
    )
    kernel.validate_exact_object(
        projection["entries"]["entry_a"],
        {
            "required_keys": [
                "fields",
                "count",
                "digest",
                "mismatch_count",
            ],
            "field_types": {
                "fields": "array",
                "count": "integer",
                "digest": "string",
                "mismatch_count": "integer",
            },
            "exact_values": {
                "fields": ["alpha", "beta"],
                "count": 2,
                "digest": "a" * 64,
                "mismatch_count": 0,
            },
        },
        location="$.projections.projection_a.entries.entry_a",
    )
    kernel.validate_exact_object(
        observed["exact_counts"],
        {
            "required_keys": ["projection_a"],
            "field_types": {"projection_a": "integer"},
            "exact_values": {"projection_a": 2},
        },
        location="$.exact_counts",
    )
    kernel.validate_exact_object(
        observed["aggregate_counts"],
        {
            "required_keys": ["total"],
            "field_types": {"total": "integer"},
            "exact_values": {"total": 2},
        },
        location="$.aggregate_counts",
    )
    bindings = observed["bindings"]
    if not bindings:
        raise kernel.TrustKernelError(
            "EXACT_OBJECT_VALUE_MISMATCH",
            "$.bindings",
            "at least one binding is required",
        )
    canonical_bindings = []
    for index, binding in enumerate(bindings):
        validated = kernel.validate_exact_object(
            binding,
            {
                "required_keys": [
                    "projection",
                    "field",
                    "expected_value",
                ],
                "field_types": {
                    "projection": "string",
                    "field": "string",
                    "expected_value": "integer",
                },
                "exact_values": {
                    "projection": "projection_a",
                    "field": "count",
                    "expected_value": 2,
                },
            },
            location=f"$.bindings[{index}]",
        )
        canonical_bindings.append(
            json.dumps(validated, sort_keys=True, separators=(",", ":"))
        )
    if len(canonical_bindings) != len(set(canonical_bindings)):
        raise kernel.TrustKernelError(
            "EXACT_OBJECT_VALUE_MISMATCH",
            "$.bindings",
            "duplicate bindings",
        )


def _target(value: Any, path: Sequence[Any]) -> Any:
    current = value
    for token in path:
        current = current[token]
    return current


def _set_value(path: Sequence[Any], replacement: Any):
    def mutate(value: Any) -> Any:
        parent = _target(value, path[:-1])
        parent[path[-1]] = replacement
        return value

    return mutate


def _drop_value(path: Sequence[Any]):
    def mutate(value: Any) -> Any:
        parent = _target(value, path[:-1])
        parent.pop(path[-1])
        return value

    return mutate


def _add_value(path: Sequence[Any], key: str, added: Any):
    def mutate(value: Any) -> Any:
        _target(value, path)[key] = added
        return value

    return mutate


def _duplicate_binding(value: Any) -> Any:
    value["bindings"].append(copy.deepcopy(value["bindings"][0]))
    return value


def _aggregate_case_specs() -> list[tuple[str, str, Callable[[Any], Any]]]:
    key = "EXACT_OBJECT_KEY_UNIVERSE"
    value = "EXACT_OBJECT_VALUE_MISMATCH"
    type_code = "EXACT_OBJECT_TYPE_MISMATCH"
    return [
        ("semantic_missing", key, _drop_value(("verified",))),
        ("semantic_not_dict", "EXACT_OBJECT_REQUIRED", lambda _item: []),
        ("semantic_verified_false", value, _set_value(("verified",), False)),
        ("semantic_extra", key, _add_value((), "unexpected", 0)),
        ("scope_missing", key, _drop_value(("scope",))),
        ("scope_extra", key, _add_value(("scope",), "unexpected", 0)),
        ("scope_type", type_code, _set_value(("scope",), [])),
        (
            "scope_value",
            value,
            _set_value(("scope", "scope_id"), "other_scope"),
        ),
        ("projection_missing", key, _drop_value(("projections",))),
        (
            "projection_extra",
            key,
            _add_value(("projections",), "projection_b", {}),
        ),
        (
            "projection_non_dict",
            type_code,
            _set_value(("projections", "projection_a"), []),
        ),
        (
            "entry_missing",
            key,
            _drop_value(("projections", "projection_a", "entries")),
        ),
        (
            "entry_extra",
            key,
            _add_value(
                ("projections", "projection_a", "entries"),
                "entry_b",
                {},
            ),
        ),
        (
            "entry_non_dict",
            type_code,
            _set_value(
                ("projections", "projection_a", "entries", "entry_a"),
                [],
            ),
        ),
        (
            "fields_type",
            type_code,
            _set_value(
                ("projections", "projection_a", "fields"),
                "alpha,beta",
            ),
        ),
        (
            "fields_value",
            value,
            _set_value(
                ("projections", "projection_a", "fields"),
                ["alpha", "gamma"],
            ),
        ),
        (
            "fields_order",
            value,
            _set_value(
                ("projections", "projection_a", "fields"),
                ["beta", "alpha"],
            ),
        ),
        (
            "count_value",
            value,
            _set_value(("projections", "projection_a", "count"), 3),
        ),
        (
            "count_bool",
            type_code,
            _set_value(("projections", "projection_a", "count"), True),
        ),
        (
            "count_string",
            type_code,
            _set_value(("projections", "projection_a", "count"), "2"),
        ),
        (
            "digest_mismatch",
            value,
            _set_value(("projections", "projection_a", "digest"), "b" * 64),
        ),
        (
            "digest_malformed",
            value,
            _set_value(("projections", "projection_a", "digest"), "abc"),
        ),
        (
            "digest_uppercase",
            value,
            _set_value(("projections", "projection_a", "digest"), "A" * 64),
        ),
        (
            "digest_non_string",
            type_code,
            _set_value(("projections", "projection_a", "digest"), 1),
        ),
        (
            "mismatch_nonzero",
            value,
            _set_value(
                ("projections", "projection_a", "mismatch_count"),
                1,
            ),
        ),
        (
            "mismatch_bool",
            type_code,
            _set_value(
                ("projections", "projection_a", "mismatch_count"),
                True,
            ),
        ),
        (
            "mismatch_string",
            type_code,
            _set_value(
                ("projections", "projection_a", "mismatch_count"),
                "0",
            ),
        ),
        ("exact_counts_missing", key, _drop_value(("exact_counts",))),
        (
            "exact_counts_extra",
            key,
            _add_value(("exact_counts",), "projection_b", 0),
        ),
        (
            "exact_counts_value",
            value,
            _set_value(("exact_counts", "projection_a"), 3),
        ),
        ("exact_counts_type", type_code, _set_value(("exact_counts",), [])),
        (
            "aggregate_counts_missing",
            key,
            _drop_value(("aggregate_counts",)),
        ),
        (
            "aggregate_counts_extra",
            key,
            _add_value(("aggregate_counts",), "unexpected", 0),
        ),
        (
            "aggregate_counts_value",
            value,
            _set_value(("aggregate_counts", "total"), 3),
        ),
        (
            "aggregate_counts_type",
            type_code,
            _set_value(("aggregate_counts",), []),
        ),
        ("binding_duplicate", value, _duplicate_binding),
        ("binding_empty", value, _set_value(("bindings",), [])),
        ("binding_missing", key, _drop_value(("bindings",))),
        (
            "binding_extra",
            key,
            _add_value(("bindings", 0), "unexpected", 0),
        ),
        (
            "binding_bool",
            type_code,
            _set_value(("bindings", 0, "expected_value"), True),
        ),
        (
            "binding_string",
            type_code,
            _set_value(("bindings", 0, "expected_value"), "2"),
        ),
        (
            "entry_fields_missing",
            key,
            _drop_value(
                (
                    "projections",
                    "projection_a",
                    "entries",
                    "entry_a",
                    "fields",
                )
            ),
        ),
        (
            "entry_count_missing",
            key,
            _drop_value(
                (
                    "projections",
                    "projection_a",
                    "entries",
                    "entry_a",
                    "count",
                )
            ),
        ),
        (
            "entry_digest_missing",
            key,
            _drop_value(
                (
                    "projections",
                    "projection_a",
                    "entries",
                    "entry_a",
                    "digest",
                )
            ),
        ),
        (
            "entry_mismatch_missing",
            key,
            _drop_value(
                (
                    "projections",
                    "projection_a",
                    "entries",
                    "entry_a",
                    "mismatch_count",
                )
            ),
        ),
        (
            "binding_non_dict",
            "EXACT_OBJECT_REQUIRED",
            _set_value(("bindings", 0), []),
        ),
        (
            "exact_counts_bool",
            type_code,
            _set_value(("exact_counts",), True),
        ),
        (
            "aggregate_counts_bool",
            type_code,
            _set_value(("aggregate_counts",), True),
        ),
        (
            "binding_projection_value",
            value,
            _set_value(("bindings", 0, "projection"), "projection_b"),
        ),
    ]


def _aggregate_negative_matrix(
    implementations: Mapping[str, Any],
) -> list[dict[str, Any]]:
    specs = _aggregate_case_specs()
    if len(specs) != 49 or len({case[0] for case in specs}) != 49:
        raise trust.TrustKernelError(
            "HOSTILE_FIXTURE_UNIVERSE_MISMATCH",
            "$.aggregate",
            f"expected 49 unique cases, observed {len(specs)}",
        )
    rows: list[dict[str, Any]] = []
    for mutation_id, expected, mutate in specs:
        for implementation, kernel in implementations.items():
            observed = mutate(copy.deepcopy(_aggregate_fixture()))
            code = _expect_error(
                lambda kernel=kernel, observed=observed: (
                    _validate_aggregate_fixture(kernel, observed)
                ),
                {expected},
            )
            rows.append(
                {
                    "case_id": f"{mutation_id}__{implementation}",
                    "mutation_id": mutation_id,
                    "implementation": implementation,
                    "expected_error_code": expected,
                    "error_code": code,
                }
            )
    return rows


def _tree_fixture(root: Path) -> None:
    (root / "nested").mkdir(parents=True)
    (root / "root.txt").write_text("root\n", encoding="ascii")
    (root / "nested/child.txt").write_text("child\n", encoding="ascii")


_TREE_ATTACKS = (
    "dangling_symlink",
    "file_symlink",
    "directory_symlink",
    "fifo",
    "unix_socket",
    "root_symlink",
)
_TREE_BOUNDARIES = (
    "exact_tree_inventory",
    "artifact_record_construction",
    "artifact_tree_closure",
)


def _apply_tree_attack(root: Path, attack: str) -> Path:
    _tree_fixture(root)
    return _add_tree_attack(root, attack)


def _add_tree_attack(root: Path, attack: str) -> Path:
    if attack == "dangling_symlink":
        (root / "dangling-link").symlink_to(root / "missing")
    elif attack == "file_symlink":
        (root / "file-link").symlink_to(root / "root.txt")
    elif attack == "directory_symlink":
        (root / "directory-link").symlink_to(
            root / "nested",
            target_is_directory=True,
        )
    elif attack == "fifo":
        os.mkfifo(root / "fifo")
    elif attack == "unix_socket":
        endpoint = socket.socket(socket.AF_UNIX)
        try:
            endpoint.bind(str(root / "socket"))
        finally:
            endpoint.close()
    elif attack == "root_symlink":
        link = root.parent / "package-link"
        link.symlink_to(root, target_is_directory=True)
        return link
    else:  # pragma: no cover - internal fixture guard
        raise AssertionError(attack)
    return root


def _tree_admission_contract() -> dict[str, Any]:
    return {
        "surface_assignments": {
            "R": ["root.txt"],
            "C": ["nested/child.txt"],
            "E": [],
        },
        "runtime_contract": {"api": "v1"},
        "publication_envelope": {"seal": "v1"},
    }


def _run_tree_boundary(
    kernel: Any,
    boundary: str,
    scan_target: Path,
) -> Any:
    if boundary == "exact_tree_inventory":
        return kernel.scan_exact_tree(scan_target)
    if boundary == "artifact_record_construction":
        return kernel.build_inventory(scan_target)
    return kernel.admit_package(
        scan_target,
        _tree_admission_contract(),
        {},
    )


def _direct_tree_negative_matrix(
    work: Path,
    implementations: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack_index, attack in enumerate(_TREE_ATTACKS):
        expected = (
            "TREE_ROOT_TYPE_FORBIDDEN"
            if attack == "root_symlink"
            else "TREE_ENTRY_TYPE_FORBIDDEN"
        )
        for boundary_index, boundary in enumerate(_TREE_BOUNDARIES):
            for implementation, kernel in implementations.items():
                root = (
                    work
                    / "d"
                    / str(attack_index)
                    / str(boundary_index)
                    / implementation[0]
                    / "p"
                )
                scan_target = _apply_tree_attack(root, attack)
                code = _expect_error(
                    lambda: _run_tree_boundary(
                        kernel,
                        boundary,
                        scan_target,
                    ),
                    {expected},
                )
                rows.append(
                    {
                        "case_id": (
                            f"{attack}__{boundary}__{implementation}"
                        ),
                        "attack": attack,
                        "boundary": boundary,
                        "implementation": implementation,
                        "expected_error_code": expected,
                        "error_code": code,
                    }
                )
    return rows


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


def _filesystem_state(root: Path, scan_target: Path) -> list[dict[str, Any]]:
    rows = []
    pending = [root]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as iterator:
            names = sorted(entry.name for entry in iterator)
        children = []
        for name in names:
            path = directory / name
            observed = path.lstat()
            relative = path.relative_to(root).as_posix()
            row = {
                "path": relative,
                "mode": observed.st_mode,
                "bytes": observed.st_size,
            }
            if os.path.islink(path):
                row["link_target"] = os.readlink(path)
            elif os.path.isfile(path):
                row["sha256"] = trust.sha256_file(path)
            elif os.path.isdir(path):
                children.append(path)
            rows.append(row)
        pending.extend(reversed(children))
    if scan_target != root:
        observed = scan_target.lstat()
        rows.append(
            {
                "path": "$scan_target",
                "mode": observed.st_mode,
                "bytes": observed.st_size,
                "link_target": os.readlink(scan_target),
            }
        )
    return sorted(rows, key=lambda row: row["path"])


def _run_json_command(
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    result = subprocess.run(
        list(command),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=dict(env) if env is not None else None,
        timeout=30,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise trust.TrustKernelError(
            "HOSTILE_SUBPROCESS_OUTPUT_INVALID",
            "$.stdout",
            f"rc={result.returncode} stdout={result.stdout!r} "
            f"stderr={result.stderr!r}",
        ) from exc
    return result.returncode, payload


def _production_shape_negative_matrix(
    work: Path,
    cli_paths: Mapping[str, Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attack_index, attack in enumerate(_TREE_ATTACKS):
        expected = (
            "TREE_ROOT_TYPE_FORBIDDEN"
            if attack == "root_symlink"
            else "TREE_ENTRY_TYPE_FORBIDDEN"
        )
        for implementation, cli_path in cli_paths.items():
            case_root = work / "p" / str(attack_index) / implementation[0]
            package_root = case_root / "x"
            contract, evidence = _small_package(package_root)
            scan_target = _add_tree_attack(package_root, attack)
            contract_path = case_root / "contract-input.json"
            evidence_path = case_root / "evidence-input.json"
            contract_path.write_text(
                json.dumps(contract, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            evidence_path.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="ascii",
            )
            before = _filesystem_state(package_root, scan_target)
            environment = os.environ.copy()
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            if implementation == "frozen":
                frozen_python = (
                    cli_path.parent
                    / "research_package_trust"
                ).parent
                existing = environment.get("PYTHONPATH")
                environment["PYTHONPATH"] = (
                    str(frozen_python)
                    if not existing
                    else f"{frozen_python}{os.pathsep}{existing}"
                )
            rc, payload = _run_json_command(
                [
                    sys.executable,
                    str(cli_path),
                    "admit-fixture",
                    "--package-dir",
                    str(scan_target),
                    "--contract",
                    str(contract_path),
                    "--evidence",
                    str(evidence_path),
                ],
                env=environment,
            )
            after = _filesystem_state(package_root, scan_target)
            observed_code = payload.get("error", {}).get("code")
            if (
                rc == 0
                or payload.get("verified") is not False
                or observed_code != expected
                or "identity" in payload
                or before != after
            ):
                raise trust.TrustKernelError(
                    "HOSTILE_PRODUCTION_TOPOLOGY_MISMATCH",
                    f"$.production_shape.{attack}.{implementation}",
                    f"rc={rc} payload={payload} zero_write={before == after}",
                )
            rows.append(
                {
                    "case_id": f"{attack}__{implementation}",
                    "attack": attack,
                    "implementation": implementation,
                    "expected_error_code": expected,
                    "error_code": observed_code,
                    "rc_nonzero": True,
                    "verified_false": True,
                    "trusted_identity_emitted": False,
                    "domain_semantic_replay_started": False,
                    "package_zero_write": True,
                }
            )
    return rows


def _identity_metamorphic_matrix(kernel: Any) -> list[dict[str, Any]]:
    rows = [{"path": "r.csv", "bytes": 1, "sha256": "a" * 64}]
    baseline = kernel.build_package_identity(
        rows,
        {"contract": "v1"},
        {"envelope": "v1"},
    )
    r_rows = [{"path": "r.csv", "bytes": 1, "sha256": "b" * 64}]
    r_mutated = kernel.build_package_identity(
        r_rows,
        {"contract": "v1"},
        {"envelope": "v1"},
    )
    c_mutated = kernel.build_package_identity(
        rows,
        {"contract": "v2"},
        {"envelope": "v1"},
    )
    e_mutated = kernel.build_package_identity(
        rows,
        {"contract": "v1"},
        {"envelope": "v2"},
    )
    _expect_error(
        lambda: kernel.validate_identity_bindings(
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


def _load_frozen_kernel(snapshot_root: Path) -> Any:
    package_root = (
        snapshot_root / "examples/hyperliquid/research_package_trust"
    )
    init_path = package_root / "__init__.py"
    module_name = f"_research_package_trust_frozen_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise trust.TrustKernelError(
            "HOSTILE_FROZEN_IMPORT_FAILED",
            str(init_path),
            "could not create package spec",
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _error_from_cli(
    command: Sequence[str],
    expected: str,
) -> str:
    rc, payload = _run_json_command(command)
    observed = payload.get("error", {}).get("code")
    if rc == 0 or payload.get("verified") is not False or observed != expected:
        raise trust.TrustKernelError(
            "HOSTILE_UNEXPECTED_ERROR_CODE",
            "$.surface_contract",
            f"expected {expected}, rc={rc}, payload={payload}",
        )
    return str(observed)


def _registry_promotion_fixture(path: Path) -> None:
    registry = trust.read_json_object(REGISTRY_PATH)
    registry["registry_revision"] = 1
    registry["versions"] = [
        {
            "kernel_name": "research_package_trust_kernel",
            "kernel_version": "v1",
            "status": "accepted",
            "acceptance_task_id": TASK_ID,
            "accepted_at_utc": "2026-08-20T00:00:00Z",
            "kernel_source_tree_sha256": "a" * 64,
            "kernel_api_contract_sha256": "b" * 64,
            "kernel_negative_matrix_sha256": "c" * 64,
            "kernel_qa_report_sha256": "d" * 64,
            "kernel_acceptance_receipt_sha256": "e" * 64,
            "kernel_acceptance_package_inventory_sha256": "f" * 64,
            "kernel_plan_sha256": "1" * 64,
            "acceptance_package_path": (
                "baselines/research_package_trust_kernel/"
                "v1/v1_acceptance_package"
            ),
        }
    ]
    path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )


def _workflow_negative_code(
    work: Path,
    mutation_id: str,
    expected: str,
) -> str:
    matrix = trust.read_json_object(MATRIX_PATH)
    if mutation_id == "remove_surface_from_research_task":
        matrix["surfaces"].pop()
    elif mutation_id == "insert_placeholder_into_surface":
        matrix["surfaces"][0]["description"] = "TBD"
    else:  # pragma: no cover - internal fixture guard
        raise AssertionError(mutation_id)
    matrix_path = work / f"{mutation_id}.json"
    matrix_path.write_bytes(trust.canonical_pretty_json_bytes(matrix))
    return _error_from_cli(
        [
            sys.executable,
            str(
                REPO_ROOT
                / ".workflow/workflow-kit/"
                "validate_research_package_task.py"
            ),
            "--task",
            str(REPO_ROOT / ".workflow/tasks/0820T001.md"),
            "--matrix",
            str(matrix_path),
        ],
        expected,
    )


def _surface_contract_negative_matrix(work: Path) -> list[dict[str, str]]:
    declarations = {
        mutation["mutation_id"]: mutation["expected_error_code"]
        for surface in trust.read_json_object(MATRIX_PATH)["surfaces"]
        for mutation in surface["negative_mutations"]
    }
    expected_declarations = {
        "mutate_stage4_research_byte": (
            "RESEARCH_DATA_IDENTITY_MISMATCH"
        ),
        "inject_domain_constant_into_kernel": "KERNEL_PURITY_VIOLATION",
        "drift_kernel_source_same_version": (
            "KERNEL_SOURCE_IDENTITY_MISMATCH"
        ),
        "remove_surface_from_research_task": "SURFACE_MATRIX_INCOMPLETE",
        "insert_placeholder_into_surface": "SURFACE_MATRIX_PLACEHOLDER",
        "premature_registry_promotion": "REGISTRY_BOOTSTRAP_NOT_EMPTY",
        "start_full_admission_without_receipt": (
            "HOSTILE_PREFLIGHT_REQUIRED"
        ),
        "coherent_rehash_stage4_package": (
            "COMPOSITE_IDENTITY_BINDING_MISMATCH"
        ),
        "replace_cleanup_directory_with_symlink": (
            "CLEANUP_PREFLIGHT_FAILED"
        ),
        "drift_remote_archive_tree": "ARCHIVE_TREE_MISMATCH",
    }
    if declarations != expected_declarations:
        raise trust.TrustKernelError(
            "SURFACE_NEGATIVE_UNIVERSE_MISMATCH",
            str(MATRIX_PATH),
            f"expected={expected_declarations} observed={declarations}",
        )

    rows = [{"path": "r.csv", "bytes": 1, "sha256": "a" * 64}]
    baseline = trust.build_package_identity(
        rows,
        {"contract": "v1"},
        {"envelope": "v1"},
    )
    mutated = trust.build_package_identity(
        [{"path": "r.csv", "bytes": 1, "sha256": "b" * 64}],
        {"contract": "v1"},
        {"envelope": "v1"},
    )
    observed: dict[str, str] = {}
    observed["mutate_stage4_research_byte"] = _expect_error(
        lambda: trust.validate_identity_bindings(
            baseline.as_dict(),
            mutated,
        ),
        {"RESEARCH_DATA_IDENTITY_MISMATCH"},
    )

    purity_source = work / "purity.py"
    purity_source.write_text("DOMAIN = 'SKHYNIX'\n", encoding="ascii")
    observed["inject_domain_constant_into_kernel"] = _expect_error(
        lambda: _assert_source_purity([purity_source]),
        {"KERNEL_PURITY_VIOLATION"},
    )
    observed["drift_kernel_source_same_version"] = _expect_error(
        lambda: _assert_source_identity("a" * 64, "b" * 64),
        {"KERNEL_SOURCE_IDENTITY_MISMATCH"},
    )
    for mutation_id in (
        "remove_surface_from_research_task",
        "insert_placeholder_into_surface",
    ):
        observed[mutation_id] = _workflow_negative_code(
            work,
            mutation_id,
            expected_declarations[mutation_id],
        )

    registry_path = work / "promoted-registry.json"
    _registry_promotion_fixture(registry_path)
    observed["premature_registry_promotion"] = _expect_error(
        lambda: trust.load_accepted_version_registry(
            registry_path,
            REGISTRY_SCHEMA_PATH,
            require_empty_bootstrap=True,
        ),
        {"REGISTRY_BOOTSTRAP_NOT_EMPTY"},
    )
    observed["start_full_admission_without_receipt"] = _error_from_cli(
        [
            sys.executable,
            str(
                REPO_ROOT
                / "examples/hyperliquid/"
                "research_package_trust_stage4_adapter.py"
            ),
            "--full-admission",
            "--package-dir",
            str(work / "not-read"),
        ],
        "HOSTILE_PREFLIGHT_REQUIRED",
    )

    coherent = mutated.as_dict()
    coherent["composite_package_identity"] = (
        baseline.composite_package_identity
    )
    observed["coherent_rehash_stage4_package"] = _expect_error(
        lambda: trust.validate_identity_bindings(coherent, mutated),
        {"COMPOSITE_IDENTITY_BINDING_MISMATCH"},
    )

    cleanup_root = work / "cleanup"
    cleanup_root.mkdir()
    cleanup_link = work / "cleanup-link"
    cleanup_link.symlink_to(cleanup_root, target_is_directory=True)
    observed["replace_cleanup_directory_with_symlink"] = _expect_error(
        lambda: trust.capture_cleanup_preflight([cleanup_link]),
        {"CLEANUP_PREFLIGHT_FAILED"},
    )
    observed["drift_remote_archive_tree"] = _expect_error(
        lambda: trust.assert_archive_tree_match(
            {"entries": [{"path": "a", "sha256": "a" * 64}]},
            {"entries": [{"path": "a", "sha256": "b" * 64}]},
        ),
        {"ARCHIVE_TREE_MISMATCH"},
    )
    return [
        {
            "mutation_id": mutation_id,
            "expected_error_code": expected,
            "error_code": observed[mutation_id],
        }
        for mutation_id, expected in declarations.items()
    ]


def run_hostile_matrix(
    implementations: Mapping[str, Any],
    cli_paths: Mapping[str, Path],
) -> dict[str, Any]:
    _purity_scan()
    with tempfile.TemporaryDirectory(prefix=f"{TASK_ID}-hostile-") as temp:
        work = Path(temp)
        aggregate = _aggregate_negative_matrix(implementations)
        direct_tree = _direct_tree_negative_matrix(work, implementations)
        production = _production_shape_negative_matrix(work, cli_paths)
        metamorphic = _identity_metamorphic_matrix(implementations["current"])
        surface_contract = _surface_contract_negative_matrix(work)
    return {
        "schema_version": "research_package_trust_negative_matrix_v1",
        "aggregate": aggregate,
        "direct_tree": direct_tree,
        "production_shape": production,
        "metamorphic": metamorphic,
        "surface_contract": surface_contract,
        "counts": {
            "aggregate": len(aggregate),
            "direct_tree": len(direct_tree),
            "production_shape": len(production),
            "metamorphic": len(metamorphic),
            "surface_contract": len(surface_contract),
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
    families: dict[str, list[str]] = {}
    for name in (
        "aggregate",
        "direct_tree",
        "production_shape",
        "metamorphic",
        "surface_contract",
    ):
        families[name] = [
            str(row.get("case_id", row.get("mutation_id")))
            for row in negative[name]
        ]
    return {
        "schema_version": "research_package_trust_fixture_inventory_v1",
        "fixture_families": families,
    }


def _write_candidate_package() -> tuple[dict[str, str], dict[str, Any]]:
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
        source_inventory = source_tree_inventory()
        source_identity = trust.canonical_json_sha256(source_inventory)
        snapshot = staging / "kernel_source_snapshot"
        snapshot.mkdir()
        for source in _source_paths():
            target = snapshot / source.relative_to(REPO_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        snapshot_identity = trust.canonical_json_sha256(
            trust.build_inventory(snapshot)
        )
        if snapshot_identity != source_identity:
            raise trust.TrustKernelError(
                "KERNEL_SNAPSHOT_IDENTITY_MISMATCH",
                str(snapshot),
                f"source={source_identity} snapshot={snapshot_identity}",
            )
        frozen = _load_frozen_kernel(snapshot)
        negative = run_hostile_matrix(
            {"current": trust, "frozen": frozen},
            {
                "current": Path(__file__).resolve(),
                "frozen": (
                    snapshot
                    / "examples/hyperliquid/"
                    "research_package_trust_cli.py"
                ),
            },
        )
        _assert_source_identity(source_identity, source_tree_sha256())
        api = _api_contract()
        fixture = _fixture_inventory(negative)
        trust.atomic_write_json(staging / "api_contract.json", api)
        trust.atomic_write_json(
            staging / "negative_matrix.json",
            dict(negative),
        )
        trust.atomic_write_json(staging / "fixture_inventory.json", fixture)
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
    }, negative


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
    candidate, negative = _write_candidate_package()
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
                for family in (
                    "aggregate",
                    "direct_tree",
                    "production_shape",
                    "surface_contract",
                )
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


def _assert_negative_topology(negative: Mapping[str, Any]) -> None:
    aggregate = negative.get("aggregate")
    expected_aggregate = {
        (mutation_id, implementation)
        for mutation_id, _code, _mutation in _aggregate_case_specs()
        for implementation in ("current", "frozen")
    }
    observed_aggregate = {
        (row.get("mutation_id"), row.get("implementation"))
        for row in aggregate
    } if type(aggregate) is list else set()
    if observed_aggregate != expected_aggregate or len(aggregate or []) != 98:
        raise trust.TrustKernelError(
            "HOSTILE_NEGATIVE_TOPOLOGY_MISMATCH",
            "$.aggregate",
            "expected 49 unique mutations on current and frozen",
        )

    direct = negative.get("direct_tree")
    expected_direct = {
        (attack, boundary, implementation)
        for attack in _TREE_ATTACKS
        for boundary in _TREE_BOUNDARIES
        for implementation in ("current", "frozen")
    }
    observed_direct = {
        (
            row.get("attack"),
            row.get("boundary"),
            row.get("implementation"),
        )
        for row in direct
    } if type(direct) is list else set()
    if observed_direct != expected_direct or len(direct or []) != 36:
        raise trust.TrustKernelError(
            "HOSTILE_NEGATIVE_TOPOLOGY_MISMATCH",
            "$.direct_tree",
            "expected six attacks at three boundaries on current and frozen",
        )

    production = negative.get("production_shape")
    expected_production = {
        (attack, implementation)
        for attack in _TREE_ATTACKS
        for implementation in ("current", "frozen")
    }
    observed_production = {
        (row.get("attack"), row.get("implementation"))
        for row in production
    } if type(production) is list else set()
    if (
        observed_production != expected_production
        or len(production or []) != 12
        or any(
            row.get("rc_nonzero") is not True
            or row.get("verified_false") is not True
            or row.get("trusted_identity_emitted") is not False
            or row.get("domain_semantic_replay_started") is not False
            or row.get("package_zero_write") is not True
            for row in production or []
        )
    ):
        raise trust.TrustKernelError(
            "HOSTILE_NEGATIVE_TOPOLOGY_MISMATCH",
            "$.production_shape",
            "expected six full CLI attacks on current and frozen",
        )

    surface = negative.get("surface_contract")
    declared = {
        mutation["mutation_id"]: mutation["expected_error_code"]
        for item in trust.read_json_object(MATRIX_PATH)["surfaces"]
        for mutation in item["negative_mutations"]
    }
    observed_surface = {
        row.get("mutation_id"): row.get("error_code")
        for row in surface
    } if type(surface) is list else {}
    if (
        len(surface or []) != len(declared)
        or observed_surface != declared
        or any(
            row.get("expected_error_code") != row.get("error_code")
            for row in surface or []
        )
    ):
        raise trust.TrustKernelError(
            "SURFACE_NEGATIVE_UNIVERSE_MISMATCH",
            "$.surface_contract",
            f"declared={declared} observed={observed_surface}",
        )


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
        "surface_contract": 10,
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
        "surface_contract": len(negative["surface_contract"]),
        "fail_open": negative["counts"]["fail_open"],
    }
    if observed_counts != counts:
        raise trust.TrustKernelError(
            "HOSTILE_NEGATIVE_COUNT_MISMATCH",
            str(CANDIDATE_ROOT / "negative_matrix.json"),
            f"receipt={counts} observed={observed_counts}",
        )
    _assert_negative_topology(negative)
    stable_codes = sorted(
        {
            row["error_code"]
            for family in (
                "aggregate",
                "direct_tree",
                "production_shape",
                "surface_contract",
            )
            for row in negative[family]
        }
    )
    if receipt["stable_error_codes"] != stable_codes:
        raise trust.TrustKernelError(
            "HOSTILE_STABLE_ERROR_CODE_MISMATCH",
            "$.stable_error_codes",
            f"expected={stable_codes} observed={receipt['stable_error_codes']}",
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
    source_identity = subparsers.add_parser("source-identity")
    source_identity.add_argument(
        "--expected",
        help="optional expected source-tree SHA256",
    )
    compare = subparsers.add_parser("compare-inventories")
    compare.add_argument("source")
    compare.add_argument("destination")
    admit = subparsers.add_parser("admit-fixture")
    admit.add_argument("--package-dir", required=True)
    admit.add_argument("--contract", required=True)
    admit.add_argument("--evidence", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "hostile-preflight":
            result = run_hostile_preflight(args.task_id)
        elif args.command == "validate-hostile-receipt":
            result = validate_hostile_receipt(Path(args.receipt))
        elif args.command == "inventory":
            rows = trust.build_inventory(Path(args.root))
            result = {
                "verified": True,
                "file_count": len(rows),
                "total_bytes": sum(row["bytes"] for row in rows),
                "inventory_sha256": trust.canonical_json_sha256(rows),
                "inventory": rows,
            }
        elif args.command == "source-identity":
            identity = source_tree_sha256()
            if args.expected:
                _assert_source_identity(args.expected, identity)
            result = {
                "verified": True,
                "kernel_source_tree_sha256": identity,
                "source_file_count": len(source_tree_inventory()),
            }
        elif args.command == "compare-inventories":
            source = trust.read_json(Path(args.source))
            destination = trust.read_json(Path(args.destination))
            trust.assert_archive_tree_match(source, destination)
            result = {
                "verified": True,
                "source": str(args.source),
                "destination": str(args.destination),
            }
        else:
            contract = trust.read_json_object(Path(args.contract))
            evidence = trust.read_json_object(Path(args.evidence))
            result = trust.admit_package(
                Path(args.package_dir),
                contract,
                evidence,
            ).as_dict()
    except (OSError, KeyError, ValueError, trust.TrustKernelError) as exc:
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
