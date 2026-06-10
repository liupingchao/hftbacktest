from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import basis_positive_execution_evidence_fail_closed_runner as runner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_T003_DIR = (
    PROJECT_ROOT / "local_live_analysis" / "basis_positive_execution_evidence_source_gate_0610T003"
)
OFFICIAL_T002_DIR = (
    PROJECT_ROOT / "local_live_analysis" / "basis_positive_execution_evidence_runner_contract_0610T002"
)


def _copy_tree(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def test_official_runner_outputs_seven_unavailable_rows(tmp_path: Path) -> None:
    result = runner.build_fail_closed_artifacts(
        t003_dir=OFFICIAL_T003_DIR,
        t002_dir=OFFICIAL_T002_DIR,
        output_dir=tmp_path / "out",
    )

    manifest = result["manifest"]
    rows = result["status_rows"]
    assert manifest["final_recommendation"] == "fail_closed_runner_skeleton_ready_for_qa"
    assert manifest["gap_count"] == 7
    assert {row["execution_gap_id"] for row in rows} == set(runner.REQUIRED_GAPS)
    assert all(row["metric_value"] == "unavailable" for row in rows)
    assert all(row["proof_status"] == "unavailable_proof_limited" for row in rows)
    assert all(row["validation_status"] == "pass" for row in rows)
    assert all(row["overclaim_reject_status"].startswith("reject:") for row in rows)
    assert set(rows[0]) == set(runner.STATUS_FIELDNAMES)
    assert (tmp_path / "out" / "fail_closed_runner_manifest.json").exists()
    assert (tmp_path / "out" / "execution_gap_status_rows.csv").exists()
    assert (tmp_path / "out" / "boundary_validation.csv").exists()


def test_rejects_t003_recommendation_not_ready(tmp_path: Path) -> None:
    t003_dir = _copy_tree(OFFICIAL_T003_DIR, tmp_path / "t003")
    manifest_path = t003_dir / "source_gate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_recommendation"] = "needs_private_order_source_design_first"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(runner.FailClosedRunnerError, match="not ready"):
        runner.build_fail_closed_artifacts(t003_dir=t003_dir, t002_dir=OFFICIAL_T002_DIR, output_dir=tmp_path / "out")


def test_rejects_source_promoted_to_execution_proof(tmp_path: Path) -> None:
    t003_dir = _copy_tree(OFFICIAL_T003_DIR, tmp_path / "t003")
    matrix_path = t003_dir / "source_availability_matrix.csv"
    rows = _read_csv(matrix_path)
    rows[0]["current_artifacts_allowed_use"] = "execution_proof"
    _write_csv(matrix_path, rows, list(rows[0].keys()))

    with pytest.raises(runner.FailClosedRunnerError, match="promoted beyond design context"):
        runner.build_fail_closed_artifacts(t003_dir=t003_dir, t002_dir=OFFICIAL_T002_DIR, output_dir=tmp_path / "out")


def test_rejects_contract_forbidden_output_schema_weakened(tmp_path: Path) -> None:
    t002_dir = _copy_tree(OFFICIAL_T002_DIR, tmp_path / "t002")
    schema_path = t002_dir / "runner_output_schema_contract.csv"
    rows = _read_csv(schema_path)
    for row in rows:
        if row["column_name"] == "order_side":
            row["category"] = "read_only_metric"
            row["allowed_for_future_runner"] = "true"
            break
    _write_csv(schema_path, rows, list(rows[0].keys()))

    with pytest.raises(runner.FailClosedRunnerError, match="forbidden output columns weakened"):
        runner.build_fail_closed_artifacts(t003_dir=OFFICIAL_T003_DIR, t002_dir=t002_dir, output_dir=tmp_path / "out")


def test_rejects_missing_required_gap(tmp_path: Path) -> None:
    t003_dir = _copy_tree(OFFICIAL_T003_DIR, tmp_path / "t003")
    matrix_path = t003_dir / "source_availability_matrix.csv"
    rows = [row for row in _read_csv(matrix_path) if row["execution_gap_id"] != "real_order_lifecycle"]
    _write_csv(matrix_path, rows, list(rows[0].keys()))

    with pytest.raises(runner.FailClosedRunnerError, match="exactly seven required gaps"):
        runner.build_fail_closed_artifacts(t003_dir=t003_dir, t002_dir=OFFICIAL_T002_DIR, output_dir=tmp_path / "out")


def test_output_schema_validation_rejects_action_field() -> None:
    output_schema = _read_csv(OFFICIAL_T002_DIR / "runner_output_schema_contract.csv")
    with pytest.raises(runner.FailClosedRunnerError, match="forbidden output columns present"):
        runner._validate_output_schema(output_schema, runner.STATUS_FIELDNAMES + ["quote_price"])
