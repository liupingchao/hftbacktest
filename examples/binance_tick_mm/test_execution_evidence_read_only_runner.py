from __future__ import annotations

import csv
from pathlib import Path

from execution_evidence_read_only_runner import (
    FINAL_RECOMMENDATION,
    SourceSpec,
    generate_artifacts,
    private_order_response_source,
    run_runner,
)


def test_runner_emits_only_proof_limited_rows() -> None:
    result = run_runner()

    assert result.status == "pass"
    assert result.rows
    assert {row["runner_output_status"] for row in result.rows} == {"proof_limited"}
    assert all("ready" not in row["proof_limit_class"] for row in result.rows)
    assert all("promotion" in row["forbidden_interpretation"] or "proof" in row["forbidden_interpretation"] for row in result.rows)


def test_missing_source_fails_closed() -> None:
    spec = SourceSpec(
        "private_order_response_source_line",
        Path("local_live_analysis/does_not_exist_for_0615T007.csv"),
        private_order_response_source.validate_artifact,
        "proof_limited_local_artifact_only",
        "local_receive_time",
        ("client_order_ref_hash",),
        "no_execution_proof",
    )

    result = run_runner(specs=[spec])

    assert result.status == "fail_closed"
    assert result.rows[0]["proof_limit_class"] == "unavailable_missing_source"
    assert result.rows[0]["fail_closed_reason"] == "missing_source_artifact"


def test_overclaim_request_fails_closed() -> None:
    result = run_runner(requested_output="pnl_promotion")

    assert result.status == "fail_closed"
    assert result.rows[0]["proof_limit_class"] == "blocked_overclaim_rejected"
    assert result.rows[0]["fail_closed_reason"] == "metric_overclaim_requested"


def test_generate_artifacts_writes_expected_outputs(tmp_path: Path) -> None:
    manifest = generate_artifacts(tmp_path)

    assert manifest["final_recommendation"] == FINAL_RECOMMENDATION
    assert manifest["missing_source_failed_closed"] is True
    assert manifest["overclaim_failed_closed"] is True
    assert manifest["forbids_live_until_task"] == "0615T009"

    evidence = tmp_path / "execution_evidence_rows.csv"
    summary = tmp_path / "runner_validation_summary.csv"
    safety = tmp_path / "no_live_safety_audit.csv"
    boundary = tmp_path / "boundary_validation.csv"
    assert evidence.exists()
    assert summary.exists()
    assert safety.exists()
    assert boundary.exists()

    with evidence.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert {row["runner_output_status"] for row in rows} == {"proof_limited"}

    with summary.open("r", newline="", encoding="utf-8") as fh:
        summary_rows = list(csv.DictReader(fh))
    assert {row["case_id"] for row in summary_rows} == {
        "valid_local_sources",
        "missing_source_fail_closed",
        "overclaim_request_fail_closed",
    }

    with safety.open("r", newline="", encoding="utf-8") as fh:
        assert {row["status"] for row in csv.DictReader(fh)} == {"pass"}

    with boundary.open("r", newline="", encoding="utf-8") as fh:
        assert {row["status"] for row in csv.DictReader(fh)} == {"pass"}
