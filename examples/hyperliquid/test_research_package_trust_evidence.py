from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_package_trust import (
    TrustKernelError,
    canonical_json_sha256,
    canonical_pretty_json_bytes,
    get_accepted_version,
    load_accepted_version_registry,
    read_json_object,
    sha256_file,
    validate_accepted_version_package,
    validate_exact_object,
    validate_json_schema,
    validate_pinned_version,
    validate_registry_append_only,
)


ROOT = Path(__file__).resolve().parents[2]
SURFACE_SCHEMA = (
    ROOT
    / ".workflow/workflow-kit/research-package-surface-matrix.schema.json"
)
REGISTRY_SCHEMA = (
    ROOT
    / ".workflow/workflow-kit/research-package-kernel-registry.schema.json"
)
MATRIX = ROOT / ".workflow/contracts/0820T001-surface-matrix.json"
REGISTRY = (
    ROOT / "baselines/research_package_trust_kernel/accepted_versions.json"
)


def test_frozen_surface_matrix_validates_without_external_dependency():
    schema = read_json_object(SURFACE_SCHEMA)
    matrix = read_json_object(MATRIX)
    validate_json_schema(matrix, schema)


@pytest.mark.parametrize(
    ("path", "value", "code"),
    [
        (
            ("produces_research_package",),
            False,
            "JSON_SCHEMA_CONST_MISMATCH",
        ),
        (
            ("surfaces", 0, "identity_layer"),
            "X",
            "JSON_SCHEMA_ENUM_MISMATCH",
        ),
        (
            ("exit_criteria",),
            [],
            "JSON_SCHEMA_MIN_ITEMS",
        ),
    ],
)
def test_frozen_surface_schema_rejects_local_shape_drift(path, value, code):
    schema = read_json_object(SURFACE_SCHEMA)
    matrix = read_json_object(MATRIX)
    target = matrix
    for token in path[:-1]:
        target = target[token]
    target[path[-1]] = value
    with pytest.raises(TrustKernelError) as caught:
        validate_json_schema(matrix, schema)
    assert caught.value.code == code


def test_empty_registry_bootstrap_is_canonical_and_unaccepted():
    registry = load_accepted_version_registry(
        REGISTRY,
        REGISTRY_SCHEMA,
        require_empty_bootstrap=True,
    )
    assert registry == {
        "schema_version": (
            "research_package_trust_kernel_accepted_versions_v1"
        ),
        "registry_revision": 0,
        "versions": [],
    }


def test_registry_bootstrap_rejects_premature_promotion(tmp_path):
    registry = read_json_object(REGISTRY)
    registry["registry_revision"] = 1
    registry["versions"] = [
        {
            "kernel_name": "research_package_trust_kernel",
            "kernel_version": "v1",
            "status": "accepted",
            "acceptance_task_id": "0820T001",
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
    target = tmp_path / "registry.json"
    import json

    target.write_text(json.dumps(registry, indent=2) + "\n", encoding="ascii")
    with pytest.raises(TrustKernelError) as caught:
        load_accepted_version_registry(
            target,
            REGISTRY_SCHEMA,
            require_empty_bootstrap=True,
        )
    assert caught.value.code == "REGISTRY_BOOTSTRAP_NOT_EMPTY"


def _accepted_registry_fixture(tmp_path: Path):
    source_path = tmp_path / "examples/hyperliquid/kernel.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("KERNEL_VERSION = 'v1'\n", encoding="ascii")
    source_inventory = [
        {
            "path": "examples/hyperliquid/kernel.py",
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
        }
    ]
    package = (
        tmp_path
        / "baselines/research_package_trust_kernel/"
        "v1/v1_acceptance_package"
    )
    package.mkdir(parents=True)
    payloads = {
        "api_contract.json": b'{"api":"v1"}\n',
        "execution_plan.md": b"# execution plan\n",
        "fixture_inventory.json": b'{"fixtures":[]}\n',
        "negative_matrix.json": b'{"fail_open":0}\n',
        "qa_report.md": b"# QA\n\npassed\n",
        "stage4_parity.json": b'{"verified":true}\n',
    }
    for name, payload in payloads.items():
        (package / name).write_bytes(payload)
    accepted_at = "2026-08-21T00:00:00Z"
    receipt = {
        "schema_version": "research_package_trust_kernel_acceptance_v1",
        "kernel_name": "research_package_trust_kernel",
        "kernel_version": "v1",
        "status": "accepted",
        "acceptance_task_id": "0820T001",
        "accepted_at_utc": accepted_at,
        "kernel_source_tree_sha256": canonical_json_sha256(
            source_inventory
        ),
        "kernel_source_inventory": source_inventory,
        "kernel_api_contract_sha256": sha256_file(
            package / "api_contract.json"
        ),
        "kernel_negative_matrix_sha256": sha256_file(
            package / "negative_matrix.json"
        ),
        "fixture_inventory_sha256": sha256_file(
            package / "fixture_inventory.json"
        ),
        "stage4_parity_sha256": sha256_file(
            package / "stage4_parity.json"
        ),
        "kernel_qa_report_sha256": sha256_file(package / "qa_report.md"),
        "kernel_plan_sha256": sha256_file(package / "execution_plan.md"),
    }
    receipt_path = package / "kernel_acceptance.json"
    receipt_path.write_bytes(canonical_pretty_json_bytes(receipt))
    inventory_files = (
        "api_contract.json",
        "execution_plan.md",
        "fixture_inventory.json",
        "kernel_acceptance.json",
        "negative_matrix.json",
        "qa_report.md",
        "stage4_parity.json",
    )
    package_inventory = {
        "schema_version": (
            "research_package_trust_kernel_acceptance_inventory_v1"
        ),
        "kernel_name": "research_package_trust_kernel",
        "kernel_version": "v1",
        "acceptance_task_id": "0820T001",
        "files": [
            {
                "path": name,
                "bytes": (package / name).stat().st_size,
                "sha256": sha256_file(package / name),
            }
            for name in inventory_files
        ],
    }
    package_inventory["inventory_sha256"] = canonical_json_sha256(
        package_inventory
    )
    inventory_path = package / "acceptance_package_inventory.json"
    inventory_path.write_bytes(canonical_pretty_json_bytes(package_inventory))
    entry = {
        "kernel_name": "research_package_trust_kernel",
        "kernel_version": "v1",
        "status": "accepted",
        "acceptance_task_id": "0820T001",
        "accepted_at_utc": accepted_at,
        "kernel_source_tree_sha256": receipt[
            "kernel_source_tree_sha256"
        ],
        "kernel_api_contract_sha256": receipt[
            "kernel_api_contract_sha256"
        ],
        "kernel_negative_matrix_sha256": receipt[
            "kernel_negative_matrix_sha256"
        ],
        "kernel_qa_report_sha256": receipt["kernel_qa_report_sha256"],
        "kernel_acceptance_receipt_sha256": sha256_file(receipt_path),
        "kernel_acceptance_package_inventory_sha256": sha256_file(
            inventory_path
        ),
        "kernel_plan_sha256": receipt["kernel_plan_sha256"],
        "acceptance_package_path": (
            "baselines/research_package_trust_kernel/"
            "v1/v1_acceptance_package"
        ),
    }
    registry = {
        "schema_version": (
            "research_package_trust_kernel_accepted_versions_v1"
        ),
        "registry_revision": 1,
        "versions": [entry],
    }
    registry_path = (
        tmp_path
        / "baselines/research_package_trust_kernel/accepted_versions.json"
    )
    registry_path.write_text(
        json.dumps(registry, indent=2, ensure_ascii=True) + "\n",
        encoding="ascii",
    )
    previous = {
        "schema_version": registry["schema_version"],
        "registry_revision": 0,
        "versions": [],
    }
    return registry_path, registry, previous, entry, package, source_path


def test_future_accepted_registry_binds_exact_package_and_pin(tmp_path):
    registry_path, _registry, previous, entry, _package, _source = (
        _accepted_registry_fixture(tmp_path)
    )
    loaded = load_accepted_version_registry(
        registry_path,
        REGISTRY_SCHEMA,
        repository_root=tmp_path,
        previous_registry=previous,
    )
    accepted = get_accepted_version(
        loaded,
        "research_package_trust_kernel",
        "v1",
    )
    pin = {
        "kernel_name": entry["kernel_name"],
        "kernel_version": entry["kernel_version"],
        "kernel_source_tree_sha256": entry[
            "kernel_source_tree_sha256"
        ],
        "kernel_api_contract_sha256": entry[
            "kernel_api_contract_sha256"
        ],
        "kernel_negative_matrix_sha256": entry[
            "kernel_negative_matrix_sha256"
        ],
        "kernel_qa_report_sha256": entry["kernel_qa_report_sha256"],
        "kernel_acceptance_task_id": entry["acceptance_task_id"],
        "registry_entry_sha256": canonical_json_sha256(entry),
    }
    validate_pinned_version(pin, accepted)


def test_accepted_registry_rejects_package_or_source_drift(tmp_path):
    registry_path, _registry, previous, entry, package, source = (
        _accepted_registry_fixture(tmp_path)
    )
    (package / "qa_report.md").write_text("forged\n", encoding="ascii")
    with pytest.raises(TrustKernelError) as caught:
        load_accepted_version_registry(
            registry_path,
            REGISTRY_SCHEMA,
            repository_root=tmp_path,
            previous_registry=previous,
        )
    assert caught.value.code == "ACCEPTANCE_PACKAGE_INVENTORY_MISMATCH"

    _accepted_registry_fixture(tmp_path / "fresh")
    fresh_root = tmp_path / "fresh"
    fresh_registry = (
        fresh_root
        / "baselines/research_package_trust_kernel/accepted_versions.json"
    )
    fresh_entry = read_json_object(fresh_registry)["versions"][0]
    source = fresh_root / "examples/hyperliquid/kernel.py"
    source.write_text("KERNEL_VERSION = 'v2'\n", encoding="ascii")
    with pytest.raises(TrustKernelError) as caught:
        validate_accepted_version_package(fresh_entry, fresh_root)
    assert caught.value.code == "KERNEL_SOURCE_IDENTITY_MISMATCH"


def test_registry_append_only_rejects_mutation_and_deletion(tmp_path):
    _path, registry, previous, _entry, _package, _source = (
        _accepted_registry_fixture(tmp_path)
    )
    validate_registry_append_only(registry, previous)
    with pytest.raises(TrustKernelError) as caught:
        validate_registry_append_only(previous, registry)
    assert caught.value.code == "REGISTRY_APPEND_ONLY_VIOLATION"
    mutated = copy.deepcopy(registry)
    mutated["versions"][0]["kernel_qa_report_sha256"] = "0" * 64
    with pytest.raises(TrustKernelError) as caught:
        validate_registry_append_only(mutated, registry)
    assert caught.value.code == "REGISTRY_APPEND_ONLY_VIOLATION"


def test_exact_object_rejects_bool_as_integer_and_extra_key():
    contract = {
        "required_keys": ["count", "verified"],
        "field_types": {
            "count": "integer",
            "verified": "boolean",
        },
        "exact_values": {"verified": True},
    }
    assert validate_exact_object(
        {"count": 3, "verified": True},
        contract,
    )["count"] == 3
    with pytest.raises(TrustKernelError) as caught:
        validate_exact_object(
            {"count": True, "verified": True},
            contract,
        )
    assert caught.value.code == "EXACT_OBJECT_TYPE_MISMATCH"
    with pytest.raises(TrustKernelError) as caught:
        validate_exact_object(
            {"count": 3, "verified": True, "unexpected": 0},
            contract,
        )
    assert caught.value.code == "EXACT_OBJECT_KEY_UNIVERSE"


def test_schema_rejects_unknown_key_in_nested_surface():
    schema = read_json_object(SURFACE_SCHEMA)
    matrix = read_json_object(MATRIX)
    mutated = copy.deepcopy(matrix)
    mutated["surfaces"][0]["unexpected"] = 0
    with pytest.raises(TrustKernelError) as caught:
        validate_json_schema(mutated, schema)
    assert caught.value.code == "JSON_SCHEMA_ADDITIONAL_PROPERTY"
