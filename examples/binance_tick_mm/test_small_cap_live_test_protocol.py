from __future__ import annotations

import csv
from pathlib import Path

from small_cap_live_test_protocol import FINAL_RECOMMENDATION, default_protocol_row, generate_artifacts, validate_protocol


def test_protocol_validates_default_row() -> None:
    result = validate_protocol(default_protocol_row())
    assert result.passed is True
    assert result.status == "pass"


def test_protocol_rejects_default_on() -> None:
    row = dict(default_protocol_row())
    row["default_on_allowed"] = "true"
    result = validate_protocol(row)
    assert result.passed is False


def test_generate_artifacts_writes_expected_outputs(tmp_path: Path) -> None:
    manifest = generate_artifacts(tmp_path)
    assert manifest["final_recommendation"] == FINAL_RECOMMENDATION
    assert manifest["first_live_capable_task"] == "0615T009"
    assert manifest["requires_explicit_total_control_approval"] is True

    protocol = tmp_path / "small_cap_live_test_protocol.csv"
    gates = tmp_path / "risk_gate_matrix.csv"
    kill = tmp_path / "kill_switch_rules.csv"
    required = tmp_path / "required_live_artifacts.csv"
    dry_run = tmp_path / "dry_run_acceptance.csv"
    boundary = tmp_path / "boundary_validation.csv"
    assert protocol.exists()
    assert gates.exists()
    assert kill.exists()
    assert required.exists()
    assert dry_run.exists()
    assert boundary.exists()

    with protocol.open("r", newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 1

    with gates.open("r", newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 6
