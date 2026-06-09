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

import basis_positive_proxy_evidence_synthesis as runner


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_T010_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_maker_viability_proxy_0609T010"
OFFICIAL_T010_QA = PROJECT_ROOT / ".workflow" / "reports" / "0609T010-qa.md"
OFFICIAL_T010_BUSINESS = PROJECT_ROOT / ".workflow" / "reports" / "0609T010-business.md"


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


def test_official_t010_synthesis_outputs_required_matrices(tmp_path: Path) -> None:
    result = runner.build_synthesis_artifacts(
        t010_dir=OFFICIAL_T010_DIR,
        t010_qa_report=OFFICIAL_T010_QA,
        t010_business_report=OFFICIAL_T010_BUSINESS,
        output_dir=tmp_path / "out",
    )

    manifest = result["manifest"]
    assert manifest["final_recommendation"] == "continue_to_execution_evidence_design"
    assert manifest["source_final_recommendation"] == "read_only_proxy_evidence_ready_for_qa"
    assert manifest["metric_decision_row_count"] == 6
    assert manifest["sample_decision_row_count"] == 7
    assert manifest["proof_class_decision_row_count"] == 2
    assert manifest["execution_evidence_gap_count"] == 7
    assert manifest["boundary_validation_passed"] is True
    assert set(row["proxy_metric_id"] for row in result["metric_rows"]) == runner.ALLOWED_PROXY_METRICS
    assert all(row["status"] == "pass" for row in result["boundary_rows"])
    assert all("source_row_reference" not in row for row in result["sample_rows"])

    out = tmp_path / "out"
    for name in runner.OUTPUT_ARTIFACTS.values():
        assert (out / name).exists()


def test_rejects_t010_qa_not_passed(tmp_path: Path) -> None:
    qa_path = tmp_path / "0609T010-qa.md"
    qa_path.write_text("# QA\n\n任务ID：\n- 0609T010\n\n状态：\n- 未通过\n", encoding="utf-8")

    with pytest.raises(runner.ProxyEvidenceSynthesisError, match="QA report"):
        runner.build_synthesis_artifacts(
            t010_dir=OFFICIAL_T010_DIR,
            t010_qa_report=qa_path,
            t010_business_report=OFFICIAL_T010_BUSINESS,
            output_dir=tmp_path / "out",
        )


def test_rejects_t010_recommendation_not_ready(tmp_path: Path) -> None:
    t010_dir = _copy_tree(OFFICIAL_T010_DIR, tmp_path / "t010")
    manifest_path = t010_dir / "proxy_runner_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["final_recommendation"] = "needs_more_proxy_runner_coverage"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(runner.ProxyEvidenceSynthesisError, match="final recommendation"):
        runner.build_synthesis_artifacts(
            t010_dir=t010_dir,
            t010_qa_report=OFFICIAL_T010_QA,
            t010_business_report=OFFICIAL_T010_BUSINESS,
            output_dir=tmp_path / "out",
        )


def test_rejects_failed_t010_validation_artifact(tmp_path: Path) -> None:
    t010_dir = _copy_tree(OFFICIAL_T010_DIR, tmp_path / "t010")
    path = t010_dir / "overclaim_reject_validation.csv"
    rows = _read_csv(path)
    rows[0]["status"] = "fail"
    rows[0]["detail"] = "overclaim"
    _write_csv(path, rows, list(rows[0].keys()))

    with pytest.raises(runner.ProxyEvidenceSynthesisError, match="validation artifacts"):
        runner.build_synthesis_artifacts(
            t010_dir=t010_dir,
            t010_qa_report=OFFICIAL_T010_QA,
            t010_business_report=OFFICIAL_T010_BUSINESS,
            output_dir=tmp_path / "out",
        )


def test_boundary_validation_detects_action_capable_columns() -> None:
    artifact_rows = {
        "bad": (
            [{"quote_price": "123"}],
            ["quote_price"],
        )
    }
    rows = runner._build_boundary_validation(
        t010_inputs={"manifest": {"final_recommendation": "read_only_proxy_evidence_ready_for_qa"}},
        final_recommendation="continue_to_execution_evidence_design",
        artifact_rows=artifact_rows,
        report_text="read-only report",
    )
    assert any(
        row["check_id"] == "t011_no_action_capable_output_columns" and row["status"] == "fail"
        for row in rows
    )


def test_recommendation_taxonomy_rejects_unknown() -> None:
    rows = runner._build_boundary_validation(
        t010_inputs={"manifest": {"final_recommendation": "read_only_proxy_evidence_ready_for_qa"}},
        final_recommendation="promotion_ready",
        artifact_rows={},
        report_text="read-only report",
    )
    assert any(
        row["check_id"] == "t011_final_recommendation_taxonomy" and row["status"] == "fail"
        for row in rows
    )


def test_gap_requirements_preserve_all_execution_gaps() -> None:
    rows = runner._build_gap_requirements()
    assert {row["execution_gap_id"] for row in rows} == {
        "fill_probability",
        "queue_priority",
        "post_only_reject_behavior",
        "cancel_fill_race",
        "fees_rebates_spread_capture",
        "inventory_lifecycle",
        "real_order_lifecycle",
    }
    assert all(row["current_t010_status"] == "not_provable_from_t010_proxy" for row in rows)
