from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_public_flow_interval_artifact_repair_design_0714T001 as design


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _make_source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    _write_json(
        source / "quote_fill_probability_manifest.json",
        {
            "task_id": "0713T003",
            "final_route": "route_to_public_flow_artifact_repair",
            "attempt_count": 2,
            "resting_attempt_count": 1,
            "public_trade_summary_row_count": 1,
            "supported_no_fill_reasons": [
                "no_attempt_keyed_interval_public_trades",
                "insufficient_exact_interval_public_flow_reconstruction",
            ],
            "unsupported_claims": ["queue_priority", "final_mvp_pass"],
        },
    )
    _write_csv(
        source / "attempt_level_quote_fill_evidence_matrix.csv",
        [
            {
                "window_id": "window_01",
                "evaluation_id": "1",
                "order_attempt_id": "",
                "order_status_type": "skipped",
                "censoring_status": "not_applicable_no_order_submitted",
                "trade_through_status": "not_applicable_no_order_submitted",
            },
            {
                "window_id": "window_01",
                "evaluation_id": "2",
                "order_attempt_id": "1",
                "order_status_type": "resting",
                "censoring_status": "short_hold_censored",
                "trade_through_status": "rolling_proxy_present_resting_interval_missing",
                "exact_timestamp_caveat": "proxy_interval_from_local_order_response_and_cancel_ack",
                "depth_proxy_status": "depth_proxy_present",
            },
        ],
    )
    _write_csv(
        source / "resting_interval_public_trades_depletion_summary.csv",
        [
            {
                "window_id": "window_01",
                "evaluation_id": "2",
                "order_attempt_id": "1",
                "capture_status": "no_matching_attempt_keyed_public_trades",
                "lifecycle_status": "proxy_interval_from_local_order_response_and_cancel_ack",
                "depth_status": "l2_snapshot_proxy_not_after_order_resting",
                "trade_through_status": "rolling_proxy_present_resting_interval_missing",
            }
        ],
    )
    for name in [
        "final_route.json",
        "censoring_horizon_matrix.csv",
        "same_side_depth_proxy_matrix.csv",
        "boundary_manifest.json",
        "input_source_manifest.json",
    ]:
        path = source / name
        if name.endswith(".json"):
            _write_json(path, {})
        else:
            _write_csv(path, [])
    return source


def test_zero_captured_rows_are_not_no_exchange_trades() -> None:
    assert (
        design.public_trade_zero_interpretation(
            "no_matching_attempt_keyed_public_trades",
            "rolling_proxy_present_resting_interval_missing",
        )
        == "artifact_gap_not_no_exchange_trades"
    )
    assert (
        design.public_trade_zero_interpretation(
            "complete_interval_coverage_zero_trades",
            "irrelevant",
        )
        == "zero_public_trades_observed_with_complete_interval_coverage"
    )


def test_run_analysis_emits_capture_contract_repair_route(tmp_path: Path) -> None:
    source = _make_source(tmp_path)

    manifest = design.run_analysis(source_dir=source, qa_report=tmp_path / "0713T003-qa.md", output_dir=tmp_path / "out")

    assert manifest["final_route"] == design.FINAL_ROUTE
    assert manifest["resting_attempt_count"] == 1
    assert manifest["artifact_gap_count"] == 5
    assert manifest["zero_public_trade_interpretation"] == ["artifact_gap_not_no_exchange_trades"]
    gaps = _read_csv(tmp_path / "out" / "artifact_gap_matrix.csv")
    trade_gap = next(row for row in gaps if row["gap_id"] == "G02")
    assert trade_gap["unsupported_interpretation"] == "no_exchange_public_trades_occurred"
    gates = _read_csv(tmp_path / "out" / "acceptance_gate_matrix.csv")
    assert next(row for row in gates if row["gate_id"] == "A05")["result"] == "pass"


def test_boundary_manifest_is_offline_only(tmp_path: Path) -> None:
    source = _make_source(tmp_path)

    design.run_analysis(source_dir=source, qa_report=tmp_path / "0713T003-qa.md", output_dir=tmp_path / "out")
    boundary = json.loads((tmp_path / "out" / "boundary_manifest.json").read_text(encoding="utf-8"))

    assert boundary["offline_only"] is True
    assert boundary["live_submit"] is False
    assert boundary["live_retry"] is False
    assert boundary["threshold_change"] is False
    assert boundary["quote_policy_design"] is False
