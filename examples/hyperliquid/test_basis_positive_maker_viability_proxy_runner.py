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

import basis_positive_maker_viability_proxy_runner as runner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_T009_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_execution_evidence_contract_0609T009"
OFFICIAL_T008_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_generator_0609T008"


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


def test_official_runner_outputs_read_only_proxy_rows(tmp_path: Path) -> None:
    result = runner.build_proxy_artifacts(
        t009_dir=OFFICIAL_T009_DIR,
        t008_dir=OFFICIAL_T008_DIR,
        output_dir=tmp_path / "out",
    )

    manifest = result["manifest"]
    rows = result["rows"]
    assert manifest["final_recommendation"] == "read_only_proxy_evidence_ready_for_qa"
    assert manifest["source_row_count"] == 3545
    assert manifest["generated_proxy_row_count"] == 3545 * 6
    assert manifest["proxy_metric_count"] == 6
    assert set(manifest["per_metric_proxy_rows"]) == runner.ALLOWED_PROXY_METRICS
    assert rows
    assert "execution_viability_decision" not in {row["proxy_metric_id"] for row in rows}
    assert all(row["future_label_output_only"] is True for row in rows)
    assert all(row["no_action_fields_pass"] is True for row in rows)
    assert all(row["execution_gap_preserved"] is True for row in rows)
    assert (tmp_path / "out" / "proxy_metric_outputs.csv").exists()


def test_rejects_t009_recommendation_not_ready(tmp_path: Path) -> None:
    t009_dir = _copy_tree(OFFICIAL_T009_DIR, tmp_path / "t009")
    manifest_path = t009_dir / "execution_evidence_contract_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_recommendation"] = "needs_more_proxy_contract_detail"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(runner.ProxyRunnerError, match="not ready"):
        runner.build_proxy_artifacts(t009_dir=t009_dir, t008_dir=OFFICIAL_T008_DIR, output_dir=tmp_path / "out")


def test_rejects_forbidden_output_schema_column(tmp_path: Path) -> None:
    t009_dir = _copy_tree(OFFICIAL_T009_DIR, tmp_path / "t009")
    schema_path = t009_dir / "proxy_runner_output_schema_contract.csv"
    rows = _read_csv(schema_path)
    for row in rows:
        if row["column_name"] == "order_side":
            row["category"] = "proxy_output"
            row["required"] = "True"
            break
    _write_csv(schema_path, rows, list(rows[0].keys()))

    with pytest.raises(runner.ProxyRunnerError, match="forbidden output columns"):
        runner.build_proxy_artifacts(t009_dir=t009_dir, t008_dir=OFFICIAL_T008_DIR, output_dir=tmp_path / "out")


def test_rejects_future_label_as_wrong_metric_scope() -> None:
    rows = [
        {
            "source_row_reference": "r1",
            "proxy_metric_id": "spread_capture_fee_rebate_proxy",
            "future_label_value": "12",
            "future_label_output_only": True,
        }
    ]
    checks = runner._validate_future_label_output_only(rows)
    assert any(row["status"] == "fail" and row["check_id"] == "future_label_metric_scope" for row in checks)


def test_rejects_missing_execution_gap_marker(tmp_path: Path) -> None:
    t008_dir = _copy_tree(OFFICIAL_T008_DIR, tmp_path / "t008")
    rows = _read_csv(t008_dir / "row_level_read_only_cases.csv")
    rows[0]["real_order_lifecycle_unproven"] = "False"
    _write_csv(t008_dir / "row_level_read_only_cases.csv", rows, list(rows[0].keys()))

    with pytest.raises(runner.ProxyRunnerError, match="execution gap marker weakened"):
        runner.build_proxy_artifacts(t009_dir=OFFICIAL_T009_DIR, t008_dir=t008_dir, output_dir=tmp_path / "out")


def test_rejects_invalid_proof_class() -> None:
    rows = [{"proof_class": "invalid", "proxy_metric_id": "public_book_post_only_feasibility_proxy"}]
    contracts = {"execution_gap_taxonomy": [], "proxy_metric_contract": []}
    checks = runner._validate_proof_classes(rows, contracts)
    assert checks[0]["status"] == "fail"


def test_rejected_sources_are_all_reported_absent() -> None:
    contracts = {"rejected_input_artifact_contract": [{"source_class": "private_account_endpoint_artifacts"}]}
    checks = runner._validate_rejected_sources(contracts)
    assert checks == [
        {
            "check_id": "rejected_source_absent",
            "source_class": "private_account_endpoint_artifacts",
            "status": "pass",
            "detail": "not used by T010 runner",
        }
    ]
