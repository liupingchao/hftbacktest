from __future__ import annotations

import copy
from pathlib import Path

import pytest

from research_package_trust import (
    TrustKernelError,
    load_accepted_version_registry,
    read_json_object,
    validate_exact_object,
    validate_json_schema,
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
