from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_public_flow_interval_artifact_repair as repair


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _make_inputs(tmp_path: Path) -> tuple[Path, Path]:
    qfp = tmp_path / "qfp"
    t011 = tmp_path / "t011"
    _write_json(qfp / "quote_fill_probability_manifest.json", {"task_id": "0710T001", "final_recommendation": "route_to_public_flow_artifact_repair"})
    _write_csv(
        qfp / "attempt_level_fill_probability_matrix.csv",
        [
            {
                "window_id": "0708T001",
                "attempt_id": "1",
                "source_kind": "prior_accepted_replay_reference",
                "source_path": ".workflow/reports/0708T002-qa.md",
                "live_classification": "submitted_resting_no_fill",
                "order_status_type": "resting",
                "side": "",
                "limit_px": "",
                "size_btc": "0.002",
                "hold_elapsed_seconds": "",
                "censoring_status": "horizon_missing",
                "no_fill_state": "resting_no_fill_observed",
            },
            {
                "window_id": "0709T001_window_02",
                "attempt_id": "1",
                "source_kind": "t011_live_window_artifact",
                "source_path": str(t011 / "window_02"),
                "live_classification": "submitted_resting_no_fill",
                "order_status_type": "resting",
                "side": "buy",
                "limit_px": "100.0",
                "size_btc": "0.005",
                "hold_elapsed_seconds": "3.0",
                "censoring_status": "short_hold_censored",
                "no_fill_state": "resting_no_fill_observed",
            },
        ],
    )
    _write_csv(
        qfp / "same_side_depth_proxy_matrix.csv",
        [
            {
                "window_id": "0709T001_window_02",
                "attempt_id": "1",
                "same_side_top_qty_btc": "0.010",
                "same_side_top_order_count": "2",
                "top_depth_multiple_of_order": "2",
            }
        ],
    )
    _write_csv(
        qfp / "trade_through_depletion_matrix.csv",
        [
            {
                "window_id": "0709T001_window_02",
                "attempt_id": "1",
                "public_depletion_status": "depleted_top_plus_order_proxy",
            }
        ],
    )
    _write_csv(
        qfp / "censoring_and_horizon_matrix.csv",
        [
            {
                "window_id": "0709T001_window_02",
                "attempt_id": "1",
                "hold_elapsed_seconds": "3.0",
            }
        ],
    )
    window = t011 / "window_02"
    _write_csv(
        window / "quote_attempt_matrix.csv",
        [
            {
                "attempt": "1",
                "event_sequence": "42",
                "order_endpoint_called": "True",
            }
        ],
    )
    _write_csv(
        window / "event_driven_latency_matrix.csv",
        [
            {
                "attempt": "1",
                "phase": "exchange_order_response",
                "event_sequence": "42",
                "end_unix_seconds": "1000.500",
            }
        ],
    )
    _write_csv(
        window / "rolling_flow_state.csv",
        [
            {
                "event_sequence": "42",
                "source_channel": "trades",
                "source_event_exchange_time_ms": "999900",
            }
        ],
    )
    return qfp, t011


def test_current_artifact_gap_routes_to_controlled_interval_capture(tmp_path: Path) -> None:
    qfp, t011 = _make_inputs(tmp_path)

    manifest = repair.run_analysis(qfp_dir=qfp, t011_root=t011, output_dir=tmp_path / "out")

    assert manifest["accepted_resting_no_fill_attempt_count"] == 2
    assert manifest["final_route"] == repair.CONTROLLED_CAPTURE_ROUTE
    assert manifest["public_trades_reconstruction_status_counts"][repair.NOT_RECONSTRUCTABLE] == 2
    rows = _read_csv(tmp_path / "out" / "resting_interval_contract_matrix.csv")
    live = next(row for row in rows if row["source_kind"] == "t011_live_window_artifact")
    assert live["resting_start_ts"] == "1000.5"
    assert live["cancel_or_shutdown_ts"] == "1003.5"
    assert live["depth_reconstruction_status"] == "pre_submit_reprice_l2_proxy_not_exact_resting_start_l2"
    assert live["public_trades_reconstruction_status"] == repair.NOT_RECONSTRUCTABLE


def test_future_interval_public_trade_contract_can_be_reconstructed(tmp_path: Path) -> None:
    qfp, t011 = _make_inputs(tmp_path)
    _write_csv(
        t011 / "window_02" / "resting_interval_public_trades.csv",
        [
            {"attempt": "1", "exchange_time_ms": "1001000", "px": "100.0", "size_btc": "0.006"},
            {"attempt": "1", "exchange_time_ms": "1002000", "px": "99.5", "size_btc": "0.004"},
        ],
    )

    repair.run_analysis(qfp_dir=qfp, t011_root=t011, output_dir=tmp_path / "out")

    rows = _read_csv(tmp_path / "out" / "resting_interval_contract_matrix.csv")
    live = next(row for row in rows if row["source_kind"] == "t011_live_window_artifact")
    assert live["public_trades_reconstruction_status"] == "exact_interval_public_trades_present"
    assert live["trade_through_count_during_interval"] == "2"
    assert live["touch_trade_qty_btc"] == "0.006"
    assert live["strict_trade_through_qty_btc"] == "0.004"
    assert live["at_or_through_trade_qty_btc"] == "0.01"
    assert live["depletion_estimate_status"] == "visible_depletion_proxy_from_interval_public_trades"
