from __future__ import annotations

import csv
import json
from pathlib import Path

from private_order_response_source import (
    ALL_FIELDS,
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


def test_valid_accepted_lifecycle_passes() -> None:
    rows = synthetic_fixture_cases()["valid_accepted_lifecycle"]

    result = validate_rows(rows)

    assert result.passed is True
    assert result.status == "pass"


def test_valid_post_only_reject_passes() -> None:
    rows = synthetic_fixture_cases()["valid_post_only_reject"]

    result = validate_rows(rows)

    assert result.passed is True


def test_missing_required_field_fails_closed() -> None:
    rows = synthetic_fixture_cases()["missing_required_field"]

    assert "missing_required_field" in _reason_codes(rows)


def test_unknown_enum_value_fails_closed() -> None:
    rows = synthetic_fixture_cases()["unknown_enum_value"]

    assert "unknown_enum_value" in _reason_codes(rows)


def test_conflicting_terminal_state_fails_closed() -> None:
    rows = synthetic_fixture_cases()["conflicting_terminal_state"]

    assert "terminal_state_conflict" in _reason_codes(rows)


def test_bad_timestamp_fails_closed() -> None:
    rows = synthetic_fixture_cases()["bad_timestamp"]

    assert "missing_timestamp" in _reason_codes(rows)


def test_duplicate_event_identity_fails_closed() -> None:
    rows = synthetic_fixture_cases()["duplicate_event_identity"]

    assert "duplicate_event_identity" in _reason_codes(rows)


def test_unsupported_source_fails_closed() -> None:
    rows = synthetic_fixture_cases()["unsupported_evidence_source"]

    assert "unsupported_evidence_source" in _reason_codes(rows)


def test_incomplete_lifecycle_fails_closed() -> None:
    rows = synthetic_fixture_cases()["incomplete_lifecycle_evidence"]

    assert "terminal_consistency_invalid" in _reason_codes(rows)


def test_validate_artifact_reads_csv(tmp_path: Path) -> None:
    rows = synthetic_fixture_cases()["valid_post_only_reject"]
    path = tmp_path / "valid_post_only_reject.csv"
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
    assert manifest["final_recommendation"] == "private_order_response_artifact_skeleton_ready_for_qa"
    assert manifest["all_expected_statuses_matched"] is True

    manifest_path = tmp_path / "private_order_response_skeleton_manifest.json"
    summary_path = tmp_path / "validator_result_summary.csv"
    boundary_path = tmp_path / "boundary_validation.csv"
    assert manifest_path.exists()
    assert summary_path.exists()
    assert boundary_path.exists()
    assert json.loads(manifest_path.read_text()) == manifest

    with summary_path.open("r", newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert len(summary_rows) == 9
    assert all(row["expected_status"] == row["actual_status"] for row in summary_rows)

    with boundary_path.open("r", newline="", encoding="utf-8") as fh:
        boundary_rows = list(csv.DictReader(fh))
    assert boundary_rows
    assert {row["status"] for row in boundary_rows} == {"pass"}
