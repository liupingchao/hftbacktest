from __future__ import annotations

import csv
import json
from pathlib import Path

from private_order_response_read_only_collector import (
    FINAL_RECOMMENDATION,
    forbidden_input_rows,
    generate_artifacts,
    synthetic_input_rows,
    transform_rows,
)


def test_transform_rows_redacts_and_validates() -> None:
    inputs = synthetic_input_rows()

    result = transform_rows(inputs)

    assert result.passed is True
    assert len(result.rows) == 2
    artifact_text = json.dumps(result.rows)
    assert "synthetic_client_alpha" not in artifact_text
    assert "synthetic_exchange_alpha" not in artifact_text
    assert result.rows[0]["client_order_ref_hash"].startswith("client_ref_sha256_")
    assert result.rows[0]["exchange_order_ref_hash"].startswith("exchange_ref_sha256_")


def test_forbidden_action_field_fails_closed() -> None:
    result = transform_rows(forbidden_input_rows())

    assert result.passed is False
    assert result.validation_status == "fail_closed"
    assert {issue.reason_code for issue in result.issues} == {"forbidden_input_field"}


def test_missing_order_reference_fails_closed() -> None:
    row = dict(synthetic_input_rows()[0])
    row["client_order_ref_input"] = ""
    row["exchange_order_ref_input"] = ""

    result = transform_rows([row])

    assert result.passed is False
    assert "missing_order_reference" in {issue.reason_code for issue in result.issues}


def test_missing_timestamp_domain_fails_closed() -> None:
    row = dict(synthetic_input_rows()[0])
    row["exchange_event_time"] = ""

    result = transform_rows([row])

    assert result.passed is False
    assert "missing_timestamp_domain" in {issue.reason_code for issue in result.issues}


def test_generate_artifacts_writes_expected_outputs(tmp_path: Path) -> None:
    manifest = generate_artifacts(tmp_path)

    assert manifest["final_recommendation"] == FINAL_RECOMMENDATION
    assert manifest["valid_transform_passed"] is True
    assert manifest["forbidden_action_failed_closed"] is True
    assert all(manifest["boundary_flags"].values())

    output_artifact = tmp_path / "collector_output_artifact.csv"
    summary = tmp_path / "collector_validation_summary.csv"
    redaction = tmp_path / "redaction_audit.csv"
    safety = tmp_path / "no_trading_safety_audit.csv"
    boundary = tmp_path / "boundary_validation.csv"
    assert output_artifact.exists()
    assert summary.exists()
    assert redaction.exists()
    assert safety.exists()
    assert boundary.exists()

    with output_artifact.open("r", newline="", encoding="utf-8") as fh:
        output_rows = list(csv.DictReader(fh))
    assert len(output_rows) == 2

    with summary.open("r", newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert {row["actual_status"] for row in summary_rows} == {"pass", "fail_closed"}

    with redaction.open("r", newline="", encoding="utf-8") as fh:
        redaction_rows = list(csv.DictReader(fh))
    assert {row["raw_client_ref_persisted_in_artifact"] for row in redaction_rows} == {"False"}
    assert {row["raw_exchange_ref_persisted_in_artifact"] for row in redaction_rows} == {"False"}

    with safety.open("r", newline="", encoding="utf-8") as fh:
        safety_rows = list(csv.DictReader(fh))
    assert safety_rows
    assert {row["status"] for row in safety_rows} == {"pass"}

    with boundary.open("r", newline="", encoding="utf-8") as fh:
        boundary_rows = list(csv.DictReader(fh))
    assert boundary_rows
    assert {row["status"] for row in boundary_rows} == {"pass"}
