from __future__ import annotations

import csv
import json
from pathlib import Path

from replay_lifecycle_validation_gate import (
    ALL_FIELDS,
    PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION,
    PRIVATE_ORDER_CONTEXT_TASK_ID,
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
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALL_FIELDS + ["fixture_case_id"])
        writer.writeheader()
        writer.writerows(rows)


def _reason_codes(rows: list[dict[str, str]]) -> set[str]:
    return {issue.reason_code for issue in validate_rows(rows).issues}


def test_valid_same_order_cancel_lifecycle_passes() -> None:
    result = validate_rows(synthetic_fixture_cases()["valid_same_order_cancel_lifecycle"])

    assert result.passed is True
    assert result.status == "pass"


def test_valid_fill_before_cancel_context_passes() -> None:
    result = validate_rows(synthetic_fixture_cases()["valid_fill_before_cancel_context"])

    assert result.passed is True


def test_missing_required_field_fails_closed() -> None:
    assert "missing_required_field" in _reason_codes(synthetic_fixture_cases()["missing_required_field"])


def test_unknown_enum_value_fails_closed() -> None:
    assert "unknown_enum_value" in _reason_codes(synthetic_fixture_cases()["unknown_enum_value"])


def test_merged_timestamp_domain_fails_closed() -> None:
    assert "timestamp_domain_merged_or_invalid" in _reason_codes(synthetic_fixture_cases()["merged_timestamp_domain"])


def test_non_monotonic_same_order_sequence_fails_closed() -> None:
    assert "non_monotonic_same_order_sequence" in _reason_codes(synthetic_fixture_cases()["non_monotonic_same_order_sequence"])


def test_flagged_event_fails_closed() -> None:
    reasons = _reason_codes(synthetic_fixture_cases()["ambiguous_conflicting_out_of_order_event"])

    assert "fail_closed_flag_accepted" in reasons


def test_duplicate_lifecycle_event_identity_fails_closed() -> None:
    assert "duplicate_lifecycle_event_identity" in _reason_codes(synthetic_fixture_cases()["duplicate_lifecycle_event_identity"])


def test_duplicate_terminal_state_fails_closed() -> None:
    assert "duplicate_terminal_state" in _reason_codes(synthetic_fixture_cases()["duplicate_terminal_state"])


def test_cross_order_causal_overclaim_fails_closed() -> None:
    assert "cross_order_causal_overclaim" in _reason_codes(synthetic_fixture_cases()["cross_order_causal_overclaim"])


def test_replay_as_execution_proof_overclaim_fails_closed() -> None:
    assert "replay_as_execution_proof_overclaim" in _reason_codes(
        synthetic_fixture_cases()["replay_as_execution_proof_overclaim"]
    )


def test_queue_priority_proof_overclaim_fails_closed() -> None:
    assert "queue_priority_proof_overclaim" in _reason_codes(synthetic_fixture_cases()["queue_priority_proof_overclaim"])


def test_cancel_fill_race_metric_overclaim_fails_closed() -> None:
    assert "cancel_fill_race_metric_overclaim" in _reason_codes(
        synthetic_fixture_cases()["cancel_fill_race_metric_overclaim"]
    )


def test_validate_artifact_reads_csv(tmp_path: Path) -> None:
    rows = synthetic_fixture_cases()["valid_fill_before_cancel_context"]
    path = tmp_path / "valid_fill_before_cancel_context.csv"
    _write_csv(path, rows)

    result = validate_artifact(path)

    assert result.passed is True
    assert len(result.rows) == 3


def test_generate_artifacts_writes_manifest_and_expected_results(tmp_path: Path) -> None:
    manifest = generate_artifacts(tmp_path)

    assert manifest["source_task_id"] == SOURCE_TASK_ID
    assert manifest["source_final_recommendation"] == SOURCE_FINAL_RECOMMENDATION
    assert manifest["synthesis_task_id"] == SYNTHESIS_TASK_ID
    assert manifest["synthesis_final_recommendation"] == SYNTHESIS_FINAL_RECOMMENDATION
    assert manifest["private_order_context_task_id"] == PRIVATE_ORDER_CONTEXT_TASK_ID
    assert manifest["private_order_context_final_recommendation"] == PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION
    assert manifest["final_recommendation"] == "replay_lifecycle_validation_gate_ready_for_qa"
    assert manifest["all_expected_statuses_matched"] is True

    manifest_path = tmp_path / "replay_lifecycle_validation_manifest.json"
    summary_path = tmp_path / "validator_result_summary.csv"
    boundary_path = tmp_path / "boundary_validation.csv"
    policy_path = tmp_path / "ordering_reconciliation_policy.csv"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert boundary_path.exists()
    assert policy_path.exists()
    assert json.loads(manifest_path.read_text()) == manifest

    with summary_path.open("r", newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert len(summary_rows) == 13
    assert all(row["expected_status"] == row["actual_status"] for row in summary_rows)

    with boundary_path.open("r", newline="", encoding="utf-8") as fh:
        boundary_rows = list(csv.DictReader(fh))
    assert boundary_rows
    assert {row["status"] for row in boundary_rows} == {"pass"}
