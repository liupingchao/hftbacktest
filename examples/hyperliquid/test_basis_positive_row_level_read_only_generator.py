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

import basis_positive_row_level_read_only_generator as generator


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


def test_official_generator_outputs_read_only_rows(tmp_path: Path) -> None:
    result = generator.build_row_level_artifacts(
        t006_dir=OFFICIAL_T006_DIR,
        preflight_output_dir=tmp_path / "preflight",
        output_dir=tmp_path / "out",
    )

    manifest = result["manifest"]
    rows = result["rows"]
    assert manifest["final_recommendation"] == "row_level_read_only_artifacts_ready_for_qa"
    assert manifest["generated_row_count"] == 3545
    assert manifest["generated_sample_count"] == 7
    assert manifest["preflight_final_recommendation"] == "preflight_validator_ready_for_qa"
    assert rows
    assert {row["case_label"] for row in rows} == {"basis_positive_clean_context"}
    assert {row["horizon_ms"] for row in rows} == {"1000"}
    assert all(row["validator_no_action_fields_pass"] is True for row in rows)
    assert all(row["validator_future_label_output_only_pass"] is True for row in rows)
    assert (tmp_path / "out" / "row_level_read_only_cases.csv").exists()


def test_rejects_preflight_failure_before_row_generation(tmp_path: Path) -> None:
    design_dir = _copy_design(tmp_path)
    manifest_path = design_dir / "generator_design_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["boundary_flags"]["no_row_level_generation"] = False
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(Exception, match="manifest_required_boundary_flags"):
        generator.build_row_level_artifacts(
            t006_dir=design_dir,
            preflight_output_dir=tmp_path / "preflight",
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out" / "row_level_read_only_cases.csv").exists()


def test_rejects_action_capable_output_schema(tmp_path: Path) -> None:
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

    with pytest.raises(Exception, match="schema_no_action_columns"):
        generator.build_row_level_artifacts(
            t006_dir=design_dir,
            preflight_output_dir=tmp_path / "preflight",
            output_dir=tmp_path / "out",
        )


def test_rejects_future_label_schema_weakening(tmp_path: Path) -> None:
    design_dir = _copy_design(tmp_path)
    schema_path = design_dir / "proposed_row_level_output_schema.csv"
    rows = _read_csv(schema_path)
    for row in rows:
        if row["column_name"] == "hyperliquid_future_mid_move_ticks":
            row["column_category"] = "decision_time_visible_context"
            break
    _write_csv(schema_path, rows, list(rows[0].keys()))

    with pytest.raises(Exception, match="schema_future_labels_output_only"):
        generator.build_row_level_artifacts(
            t006_dir=design_dir,
            preflight_output_dir=tmp_path / "preflight",
            output_dir=tmp_path / "out",
        )


def test_rejects_source_path_outside_allowlist(tmp_path: Path) -> None:
    allowed = {Path("/tmp/allowed.csv").resolve()}
    with pytest.raises(generator.RowLevelGeneratorError, match="outside T006/T003 allowlist"):
        generator._validate_source_path_allowed(tmp_path / "private_order_rows.csv", allowed)


def test_execution_gap_markers_must_remain_true() -> None:
    checks = generator._validate_execution_gap_rows(
        [
            {
                "source_row_reference": "sample:0",
                "fill_probability_unproven": True,
                "queue_position_unproven": True,
                "post_only_reject_unproven": True,
                "cancel_fill_race_unproven": True,
                "fees_rebates_spread_capture_unproven": True,
                "inventory_lifecycle_unproven": True,
                "real_order_lifecycle_unproven": False,
            }
        ]
    )
    assert checks[0]["status"] == "fail"
    assert "real_order_lifecycle_unproven" in checks[0]["detail"]
