from __future__ import annotations

import csv
import json
from pathlib import Path

from account_inventory_source import (
    ALL_FIELDS,
    PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION,
    PRIVATE_ORDER_CONTEXT_TASK_ID,
    REPLAY_LIFECYCLE_CONTEXT_FINAL_RECOMMENDATION,
    REPLAY_LIFECYCLE_CONTEXT_TASK_ID,
    SOURCE_FINAL_RECOMMENDATION,
    SOURCE_TASK_ID,
    SYNTHESIS_FINAL_RECOMMENDATION,
    SYNTHESIS_TASK_ID,
    generate_artifacts,
    synthetic_fixture_cases,
    validate_artifact,
    validate_rows,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(ALL_FIELDS + ["fixture_case_id"] + [field for row in rows for field in row if field not in ALL_FIELDS]))
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _reason_codes(rows: list[dict[str, str]]) -> set[str]:
    return {issue.reason_code for issue in validate_rows(rows).issues}


def test_valid_complete_snapshot_passes() -> None:
    result = validate_rows(synthetic_fixture_cases()["valid_complete_snapshot"])

    assert result.passed is True
    assert result.status == "pass"


def test_valid_account_observed_transition_passes() -> None:
    result = validate_rows(synthetic_fixture_cases()["valid_account_observed_transition"])

    assert result.passed is True


def test_missing_required_field_fails_closed() -> None:
    assert "missing_required_field" in _reason_codes(synthetic_fixture_cases()["missing_required_field"])


def test_unknown_enum_value_fails_closed() -> None:
    assert "unknown_enum_value" in _reason_codes(synthetic_fixture_cases()["unknown_enum_value"])


def test_forbidden_endpoint_field_fails_closed() -> None:
    assert "forbidden_endpoint_or_action_field" in _reason_codes(synthetic_fixture_cases()["forbidden_endpoint_field"])


def test_snapshot_fail_closed_label_accepted_fails_closed() -> None:
    assert "snapshot_fail_closed_label_accepted" in _reason_codes(
        synthetic_fixture_cases()["snapshot_fail_closed_label_accepted"]
    )


def test_transition_fail_closed_label_accepted_fails_closed() -> None:
    assert "transition_fail_closed_label_accepted" in _reason_codes(
        synthetic_fixture_cases()["transition_fail_closed_label_accepted"]
    )


def test_available_locked_total_nonconserving_fails_closed() -> None:
    assert "available_locked_total_nonconserving" in _reason_codes(
        synthetic_fixture_cases()["available_locked_total_nonconserving"]
    )


def test_before_after_delta_nonconserving_fails_closed() -> None:
    assert "before_after_delta_nonconserving" in _reason_codes(
        synthetic_fixture_cases()["before_after_delta_nonconserving"]
    )


def test_precision_policy_invalid_fails_closed() -> None:
    assert "precision_policy_invalid" in _reason_codes(synthetic_fixture_cases()["precision_policy_invalid"])


def test_duplicate_transition_identity_fails_closed() -> None:
    assert "duplicate_transition_identity" in _reason_codes(synthetic_fixture_cases()["duplicate_transition_identity"])


def test_out_of_order_transition_fails_closed() -> None:
    assert "out_of_order_transition" in _reason_codes(synthetic_fixture_cases()["out_of_order_transition"])


def test_order_fills_alone_inventory_proof_overclaim_fails_closed() -> None:
    assert "order_fills_alone_inventory_proof_overclaim" in _reason_codes(
        synthetic_fixture_cases()["order_fills_alone_inventory_proof_overclaim"]
    )


def test_inventory_lifecycle_proof_overclaim_fails_closed() -> None:
    assert "inventory_lifecycle_proof_overclaim" in _reason_codes(
        synthetic_fixture_cases()["inventory_lifecycle_proof_overclaim"]
    )


def test_pnl_or_economics_overclaim_fails_closed() -> None:
    assert "pnl_or_economics_overclaim" in _reason_codes(synthetic_fixture_cases()["pnl_or_economics_overclaim"])


def test_non_account_authority_inventory_overclaim_fails_closed() -> None:
    assert "non_account_authority_inventory_overclaim" in _reason_codes(
        synthetic_fixture_cases()["non_account_authority_inventory_overclaim"]
    )


def test_validate_artifact_reads_csv(tmp_path: Path) -> None:
    rows = synthetic_fixture_cases()["valid_account_observed_transition"]
    path = tmp_path / "valid_account_observed_transition.csv"
    _write_csv(path, rows)

    result = validate_artifact(path)

    assert result.passed is True
    assert len(result.rows) == 1


def test_generate_artifacts_writes_manifest_and_expected_results(tmp_path: Path) -> None:
    manifest = generate_artifacts(tmp_path)

    assert manifest["source_task_id"] == SOURCE_TASK_ID
    assert manifest["source_final_recommendation"] == SOURCE_FINAL_RECOMMENDATION
    assert manifest["synthesis_task_id"] == SYNTHESIS_TASK_ID
    assert manifest["synthesis_final_recommendation"] == SYNTHESIS_FINAL_RECOMMENDATION
    assert manifest["private_order_context_task_id"] == PRIVATE_ORDER_CONTEXT_TASK_ID
    assert manifest["private_order_context_final_recommendation"] == PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION
    assert manifest["replay_lifecycle_context_task_id"] == REPLAY_LIFECYCLE_CONTEXT_TASK_ID
    assert manifest["replay_lifecycle_context_final_recommendation"] == REPLAY_LIFECYCLE_CONTEXT_FINAL_RECOMMENDATION
    assert manifest["final_recommendation"] == "account_inventory_artifact_skeleton_ready_for_qa"
    assert manifest["all_expected_statuses_matched"] is True

    manifest_path = tmp_path / "account_inventory_validation_manifest.json"
    summary_path = tmp_path / "validator_result_summary.csv"
    boundary_path = tmp_path / "boundary_validation.csv"
    conservation_path = tmp_path / "conservation_check_policy.csv"
    reconciliation_path = tmp_path / "reconciliation_boundary_policy.csv"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert boundary_path.exists()
    assert conservation_path.exists()
    assert reconciliation_path.exists()
    assert json.loads(manifest_path.read_text()) == manifest

    with summary_path.open("r", newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert len(summary_rows) == 16
    assert all(row["expected_status"] == row["actual_status"] for row in summary_rows)

    with boundary_path.open("r", newline="", encoding="utf-8") as fh:
        boundary_rows = list(csv.DictReader(fh))
    assert boundary_rows
    assert {row["status"] for row in boundary_rows} == {"pass"}
