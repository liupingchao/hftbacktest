from __future__ import annotations

import copy
import csv
from pathlib import Path

import pytest

import skhynix_stage_h0b as h0b
import skhynix_stage_h0b_contracts as contracts


def valid_hostile_receipt() -> tuple[dict[str, object], dict[str, object]]:
    dispatch = h0b.validate_dispatch(h0b.TASK_PATH, h0b.MATRIX_PATH)
    matrix = h0b.read_json(h0b.MATRIX_PATH)
    rows = [
        {
            "mutation_id": surface["negative_mutations"][0]["mutation_id"],
            "expected_error_code": surface["negative_mutations"][0][
                "expected_error_code"
            ],
            "error_code": surface["negative_mutations"][0][
                "expected_error_code"
            ],
        }
        for surface in matrix["surfaces"]
    ]
    return (
        {
            "schema_version": "skhynix_stage_h0b_hostile_preflight_v2",
            "task_id": contracts.TASK_ID,
            "dispatch": dispatch,
            "runtime_source_tree_sha256": h0b.runtime_source_tree_sha256(),
            "frozen_runtime_source_tree_sha256": (
                h0b.runtime_source_tree_sha256()
            ),
            "surface_contract": rows,
            "frozen_surface_contract": copy.deepcopy(rows),
            "current_negative_mutation_count": len(rows),
            "frozen_negative_mutation_count": len(rows),
            "fail_open_count": 0,
            "outcome_predicate_evaluated": False,
            "stage4_bytes_opened": False,
            "network_private_order_cancel_live_access": False,
        },
        dispatch,
    )


def test_hostile_receipt_is_bound_to_current_and_frozen_runtime(
    tmp_path: Path,
) -> None:
    receipt, dispatch = valid_hostile_receipt()
    path = tmp_path / "hostile.json"
    h0b.write_json(path, receipt)
    h0b.validate_hostile_preflight_receipt(
        path,
        matrix_path=h0b.MATRIX_PATH,
        expected_dispatch=dispatch,
    )
    mutations = []
    stale_runtime = copy.deepcopy(receipt)
    stale_runtime["runtime_source_tree_sha256"] = "0" * 64
    mutations.append(stale_runtime)
    missing_frozen = copy.deepcopy(receipt)
    missing_frozen["frozen_surface_contract"] = []
    mutations.append(missing_frozen)
    wrong_count = copy.deepcopy(receipt)
    wrong_count["frozen_negative_mutation_count"] = 0
    mutations.append(wrong_count)
    extra_key = copy.deepcopy(receipt)
    extra_key["unexpected"] = True
    mutations.append(extra_key)
    forbidden_access = copy.deepcopy(receipt)
    forbidden_access["stage4_bytes_opened"] = True
    mutations.append(forbidden_access)
    for index, mutated in enumerate(mutations):
        mutated_path = tmp_path / f"hostile-mutated-{index}.json"
        h0b.write_json(mutated_path, mutated)
        with pytest.raises(contracts.H0BError):
            h0b.validate_hostile_preflight_receipt(
                mutated_path,
                matrix_path=h0b.MATRIX_PATH,
                expected_dispatch=dispatch,
            )


def test_dispatch_and_surface_assignment_oracles() -> None:
    result = h0b.validate_dispatch(h0b.TASK_PATH, h0b.MATRIX_PATH)
    assert result["verified"] is True
    projection = h0b.surface_assignment_projection()
    assert len(projection) == 76
    assert len({(row["path"], row["surface_id"]) for row in projection}) == 76


def test_semantic_inventory_is_outcome_blind_and_portable() -> None:
    with h0b.SEMANTIC_INVENTORY_PATH.open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 106
    assert all(not row["relative_path"].startswith("/") for row in rows)
    assert all("stage04" not in row["relative_path"] for row in rows)
    assert all("/outcomes/" not in row["relative_path"] for row in rows)


def test_preoutcome_contract_exact_key_universe() -> None:
    payload = h0b.preoutcome_contract_payload()
    assert set(payload) == {
        "schema_version",
        "task_id",
        "primary_plan_sha256",
        "diagnostic_plan_sha256",
        "diagnostic_review_sha256",
        "surface_matrix_sha256",
        "source_inventory_contract_sha256",
        "likelihood_contract_sha256",
        "design_matrix_contract_sha256",
        "walk_forward_contract_sha256",
        "resampling_contract_sha256",
        "classification_contract_sha256",
        "output_contract_sha256",
        "runtime_source_tree_sha256",
    }


def test_exact_tree_rejects_extra_path(tmp_path: Path) -> None:
    for directory in contracts.PACKAGE_DIRECTORIES:
        (tmp_path / directory).mkdir()
    for relative in contracts.EXACT_PACKAGE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    contracts.validate_exact_package_tree(tmp_path)
    (tmp_path / "extra.txt").write_text("forbidden", encoding="ascii")
    with pytest.raises(contracts.H0BError) as captured:
        contracts.validate_exact_package_tree(tmp_path)
    assert captured.value.code == "H0B_PACKAGE_TREE_MISMATCH"


def test_latency_roles_keep_850_non_rescue() -> None:
    rows = h0b.latency_role_rows()
    diagnostic = next(row for row in rows if row["latency_ms"] == 850)
    primary = next(row for row in rows if row["latency_ms"] == 6600)
    assert diagnostic["diagnostic"] is True
    assert diagnostic["primary"] is False
    assert diagnostic["can_rescue_primary"] is False
    assert primary["primary"] is True
    assert primary["can_rescue_primary"] is False


def test_package_report_template_orders_plan_identities_first() -> None:
    source = h0b.package_report_text.__code__.co_consts
    template_lines = [value for value in source if isinstance(value, str)]
    classification = template_lines.index("- classification: `")
    primary_plan = template_lines.index("- primary_plan_sha256: `")
    diagnostic_plan = template_lines.index("- diagnostic_plan_sha256: `")
    diagnostic_review = template_lines.index(
        "- diagnostic_review_sha256: `"
    )
    formal_sessions = template_lines.index(
        "- formal_sessions: `jul30,aug04`"
    )
    assert classification < primary_plan < diagnostic_plan
    assert diagnostic_plan < diagnostic_review < formal_sessions
