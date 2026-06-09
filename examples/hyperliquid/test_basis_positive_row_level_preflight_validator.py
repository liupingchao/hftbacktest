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

import basis_positive_row_level_preflight_validator as validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_T006_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_generator_design_0609T006"


def _copy_design(tmp_path: Path) -> Path:
    target = tmp_path / "t006"
    shutil.copytree(OFFICIAL_T006_DIR, target)
    return target


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def test_official_t006_artifacts_pass(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = validator.build_preflight_artifacts(input_dir=OFFICIAL_T006_DIR, output_dir=out)

    assert result["final_recommendation"] == "preflight_validator_ready_for_qa"
    assert result["manifest"]["failed_check_count"] == 0
    assert (out / "preflight_validator_manifest.json").exists()
    assert (out / "preflight_validation_summary.csv").exists()
    assert (out / "preflight_reject_check_results.csv").exists()
    assert (out / "preflight_validation_report.md").exists()


def test_rejects_future_label_as_input(tmp_path: Path) -> None:
    design_dir = _copy_design(tmp_path)
    schema_path = design_dir / "proposed_row_level_output_schema.csv"
    rows = _read_csv(schema_path)
    for row in rows:
        if row["column_name"] == "horizon_ms":
            row["column_category"] = "decision_time_visible_context"
            break
    _write_csv(schema_path, rows, list(rows[0].keys()))

    with pytest.raises(validator.PreflightValidationError, match="schema_future_labels_output_only"):
        validator.build_preflight_artifacts(input_dir=design_dir, output_dir=tmp_path / "out")


def test_rejects_action_capable_schema_column(tmp_path: Path) -> None:
    design_dir = _copy_design(tmp_path)
    schema_path = design_dir / "proposed_row_level_output_schema.csv"
    rows = _read_csv(schema_path)
    rows.append(
        {
            "column_name": "order_side",
            "column_category": "decision_time_visible_context",
            "required_status": "allowed",
            "source_contract": "bad_fixture",
            "definition": "bad action field",
            "allowed_use": "bad action use",
            "forbidden_use": "",
            "validator_reference": "reject_action_field_definitions",
        }
    )
    _write_csv(schema_path, rows, list(rows[0].keys()))

    with pytest.raises(validator.PreflightValidationError, match="schema_no_action_columns"):
        validator.build_preflight_artifacts(input_dir=design_dir, output_dir=tmp_path / "out")


def test_rejects_missing_boundary_flag(tmp_path: Path) -> None:
    design_dir = _copy_design(tmp_path)
    manifest_path = design_dir / "generator_design_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["boundary_flags"]["no_row_level_generation"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(validator.PreflightValidationError, match="manifest_required_boundary_flags"):
        validator.build_preflight_artifacts(input_dir=design_dir, output_dir=tmp_path / "out")


def test_rejects_missing_reject_condition_coverage(tmp_path: Path) -> None:
    design_dir = _copy_design(tmp_path)
    reject_path = design_dir / "generator_reject_conditions.csv"
    rows = [row for row in _read_csv(reject_path) if row["condition_id"] != "reject_future_label_as_input"]
    _write_csv(reject_path, rows, list(rows[0].keys()))

    with pytest.raises(validator.PreflightValidationError, match="generator_reject_conditions_required_coverage"):
        validator.build_preflight_artifacts(input_dir=design_dir, output_dir=tmp_path / "out")
