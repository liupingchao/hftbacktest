from __future__ import annotations

import csv
import json
from pathlib import Path

import hyperliquid_private_order_artifact_validator as validator


def test_accepted_fixture_rows_pass() -> None:
    rows = validator.fixture_rows()["accepted"]

    result = validator.validate_rows(rows)

    assert result.passed is True
    assert result.status == "pass"


def test_fail_closed_fixture_rows_fail_closed() -> None:
    rows = validator.fixture_rows()["fail_closed"]

    result = validator.validate_rows(rows)

    assert result.passed is False
    assert result.status == "fail_closed"
    reasons = {issue.reason_code for issue in result.issues}
    assert "missing_or_invalid_timestamp" in reasons
    assert "terminal_consistency_invalid" in reasons
    assert "forbidden_field_present" in reasons
    assert "duplicate_event_identity" in reasons


def test_unknown_enum_fails_closed() -> None:
    row = dict(validator.fixture_rows()["accepted"][0])
    row["venue"] = "binance"

    result = validator.validate_rows([row])

    assert "unknown_enum_value" in {issue.reason_code for issue in result.issues}


def test_missing_required_field_fails_closed() -> None:
    row = dict(validator.fixture_rows()["accepted"][0])
    row["opaque_client_order_ref"] = ""

    result = validator.validate_rows([row])

    assert "missing_required_field" in {issue.reason_code for issue in result.issues}


def test_generate_artifacts_writes_expected_outputs(tmp_path: Path) -> None:
    manifest = validator.generate_artifacts(tmp_path)

    assert manifest["final_recommendation"] == validator.FINAL_RECOMMENDATION
    assert manifest["accepted_fixture_status"] == "pass"
    assert manifest["fail_closed_fixture_status"] == "fail_closed"
    assert all(manifest["boundary_flags"].values())

    manifest_path = tmp_path / "hyperliquid_private_order_validator_manifest.json"
    accepted_path = tmp_path / "accepted_artifact_rows.csv"
    fail_closed_path = tmp_path / "fail_closed_artifact_rows.csv"
    summary_path = tmp_path / "validator_result_summary.csv"
    boundary_path = tmp_path / "boundary_validation.csv"
    assert json.loads(manifest_path.read_text()) == manifest
    assert accepted_path.exists()
    assert fail_closed_path.exists()
    assert summary_path.exists()
    assert boundary_path.exists()

    with accepted_path.open(newline="", encoding="utf-8") as fh:
        accepted_rows = list(csv.DictReader(fh))
    assert len(accepted_rows) == 5

    with summary_path.open(newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert {row["actual_status"] for row in summary_rows} == {"pass", "fail_closed"}

    with boundary_path.open(newline="", encoding="utf-8") as fh:
        boundary_rows = list(csv.DictReader(fh))
    assert {row["status"] for row in boundary_rows} == {"pass"}
