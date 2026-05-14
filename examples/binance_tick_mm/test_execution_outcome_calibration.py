from __future__ import annotations

import csv
import json
from pathlib import Path

from execution_outcome_calibration import run_execution_outcome_calibration


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sample_tree(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "sample"
    output_dir = run_dir / "stage6"
    sidecar = run_dir / "t009_fixed_sidecar"
    replay_dir = run_dir / "out" / "backtest_audit_replay"
    ns = lambda ms: 1_000_000_000 + ms * 1_000_000
    fieldnames = [
        "run_id",
        "symbol",
        "strategy_seq",
        "event_type",
        "event_source",
        "event_seq",
        "ts_local",
        "ts_exch",
        "order_id",
        "action",
        "planned_order_id",
        "planned_action",
        "throttle_reason",
        "reject_reason",
        "best_bid",
        "best_ask",
        "mid",
        "fair",
        "reservation",
        "position",
        "inventory_score",
        "feed_latency_ns",
        "latency_signal_ms",
        "bid_size",
        "ask_size",
        "bid_top5_ticks",
        "bid_top5_qtys",
        "ask_top5_ticks",
        "ask_top5_qtys",
        "market_view_source",
        "top5_source",
        "book_view_stale_ms",
        "target_bid_tick",
        "target_ask_tick",
        "working_bid_tick",
        "working_ask_tick",
        "order_side",
        "order_price",
        "order_price_tick",
        "order_qty",
        "order_remaining_qty",
        "order_executed_qty",
        "order_status",
        "lifecycle_state",
        "linked_strategy_seq",
        "linked_action",
        "linked_order_id",
        "cancel_request_ts",
        "cancel_ack_ts",
        "fill_ts",
        "fill_qty",
        "fill_price",
        "fill_after_cancel_request",
    ]

    live_rows = [
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 1, "event_type": "decision", "event_source": "strategy", "event_seq": 1, "ts_local": ns(0), "ts_exch": ns(0), "best_bid": "99.9", "best_ask": "100.1", "mid": "100.0", "fair": "100.2", "reservation": "100.15", "position": "0.0", "inventory_score": "0.1", "feed_latency_ns": "1000000", "latency_signal_ms": "1.0", "bid_size": "2.0", "ask_size": "1.0", "bid_top5_ticks": "999|998|997|996|995", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1001|1002|1003|1004|1005", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "1.0", "target_bid_tick": "999", "target_ask_tick": "1001", "working_bid_tick": "", "working_ask_tick": ""},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 1, "event_type": "order_submit_sent", "event_source": "strategy", "event_seq": 2, "ts_local": ns(0), "ts_exch": ns(0), "order_id": "L1", "action": "submit_buy", "planned_action": "submit_buy", "best_bid": "99.9", "best_ask": "100.1", "mid": "100.0", "fair": "100.2", "reservation": "100.15", "position": "0.0", "inventory_score": "0.1", "feed_latency_ns": "1000000", "latency_signal_ms": "1.0", "bid_size": "2.0", "ask_size": "1.0", "bid_top5_ticks": "999|998|997|996|995", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1001|1002|1003|1004|1005", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "1.0", "target_bid_tick": "999", "target_ask_tick": "1001", "order_side": "buy", "order_price": "99.9", "order_price_tick": "999", "order_qty": "1.0", "lifecycle_state": "order_submit_sent", "linked_strategy_seq": "1", "linked_action": "submit_buy", "linked_order_id": "L1", "fill_after_cancel_request": "0"},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 2, "event_type": "decision", "event_source": "strategy", "event_seq": 3, "ts_local": ns(100), "ts_exch": ns(100), "best_bid": "100.0", "best_ask": "100.2", "mid": "100.1", "fair": "100.25", "reservation": "100.2", "position": "1.0", "inventory_score": "0.2", "feed_latency_ns": "1000000", "latency_signal_ms": "1.0", "bid_size": "2.0", "ask_size": "1.0", "bid_top5_ticks": "1000|999|998|997|996", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1002|1003|1004|1005|1006", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "1.0", "target_bid_tick": "1000", "target_ask_tick": "1002", "working_bid_tick": "999", "working_ask_tick": ""},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 2, "event_type": "fill", "event_source": "ws", "event_seq": 4, "ts_local": ns(100), "ts_exch": ns(100), "order_id": "L1", "action": "fill", "best_bid": "100.0", "best_ask": "100.2", "mid": "100.1", "fair": "100.25", "reservation": "100.2", "position": "1.0", "inventory_score": "0.2", "feed_latency_ns": "1000000", "latency_signal_ms": "1.0", "bid_size": "2.0", "ask_size": "1.0", "bid_top5_ticks": "1000|999|998|997|996", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1002|1003|1004|1005|1006", "ask_top5_qtys": "1|1|1|1|1", "order_side": "buy", "order_price": "99.9", "order_price_tick": "999", "order_qty": "1.0", "order_remaining_qty": "0.0", "order_executed_qty": "1.0", "order_status": "filled", "lifecycle_state": "fill", "linked_strategy_seq": "2", "linked_action": "fill", "linked_order_id": "L1", "fill_ts": str(ns(100)), "fill_qty": "1.0", "fill_price": "99.9", "fill_after_cancel_request": "0"},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 3, "event_type": "decision", "event_source": "strategy", "event_seq": 5, "ts_local": ns(200), "ts_exch": ns(200), "best_bid": "100.1", "best_ask": "100.3", "mid": "100.2", "fair": "100.0", "reservation": "100.05", "position": "1.0", "inventory_score": "0.6", "feed_latency_ns": "1000000", "latency_signal_ms": "5.0", "bid_size": "3.0", "ask_size": "1.0", "bid_top5_ticks": "1001|1000|999|998|997", "bid_top5_qtys": "3|1|1|1|1", "ask_top5_ticks": "1003|1004|1005|1006|1007", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "3.0", "target_bid_tick": "1001", "target_ask_tick": "1003", "working_bid_tick": "", "working_ask_tick": ""},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 3, "event_type": "order_submit_sent", "event_source": "strategy", "event_seq": 6, "ts_local": ns(200), "ts_exch": ns(200), "order_id": "L2", "action": "submit_sell", "planned_action": "submit_sell", "best_bid": "100.1", "best_ask": "100.3", "mid": "100.2", "fair": "100.0", "reservation": "100.05", "position": "1.0", "inventory_score": "0.6", "feed_latency_ns": "1000000", "latency_signal_ms": "5.0", "bid_size": "3.0", "ask_size": "1.0", "bid_top5_ticks": "1001|1000|999|998|997", "bid_top5_qtys": "3|1|1|1|1", "ask_top5_ticks": "1003|1004|1005|1006|1007", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "3.0", "target_bid_tick": "1001", "target_ask_tick": "1003", "order_side": "sell", "order_price": "100.3", "order_price_tick": "1003", "order_qty": "1.0", "lifecycle_state": "order_submit_sent", "linked_strategy_seq": "3", "linked_action": "submit_sell", "linked_order_id": "L2", "fill_after_cancel_request": "0"},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 4, "event_type": "cancel_sent", "event_source": "strategy", "event_seq": 7, "ts_local": ns(260), "ts_exch": ns(260), "order_id": "L2", "action": "cancel_sell", "best_bid": "100.1", "best_ask": "100.3", "mid": "100.2", "position": "1.0", "inventory_score": "0.6", "linked_strategy_seq": "4", "linked_action": "cancel_sell", "linked_order_id": "L2", "order_side": "sell", "cancel_request_ts": str(ns(260))},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 5, "event_type": "cancel_ack", "event_source": "ws", "event_seq": 8, "ts_local": ns(300), "ts_exch": ns(300), "order_id": "L2", "action": "cancel_ack", "best_bid": "100.1", "best_ask": "100.3", "mid": "100.2", "position": "1.0", "inventory_score": "0.6", "linked_strategy_seq": "5", "linked_action": "cancel_ack", "linked_order_id": "L2", "order_side": "sell", "cancel_request_ts": str(ns(260)), "cancel_ack_ts": str(ns(300))},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 6, "event_type": "decision", "event_source": "strategy", "event_seq": 9, "ts_local": ns(500), "ts_exch": ns(500), "best_bid": "100.2", "best_ask": "100.4", "mid": "100.3", "fair": "100.35", "reservation": "100.3", "position": "1.0", "inventory_score": "0.3", "feed_latency_ns": "2000000", "latency_signal_ms": "2.0", "bid_size": "2.0", "ask_size": "2.0", "bid_top5_ticks": "1002|1001|1000|999|998", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1004|1005|1006|1007|1008", "ask_top5_qtys": "2|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "2.0", "target_bid_tick": "1002", "target_ask_tick": "1004", "working_bid_tick": "", "working_ask_tick": ""},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 7, "event_type": "decision", "event_source": "strategy", "event_seq": 10, "ts_local": ns(1000), "ts_exch": ns(1000), "best_bid": "100.0", "best_ask": "100.2", "mid": "100.1", "fair": "100.05", "reservation": "100.0", "position": "1.0", "inventory_score": "0.4", "feed_latency_ns": "2000000", "latency_signal_ms": "2.5", "bid_size": "2.0", "ask_size": "2.0", "bid_top5_ticks": "1000|999|998|997|996", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1002|1003|1004|1005|1006", "ask_top5_qtys": "2|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "2.0", "target_bid_tick": "1000", "target_ask_tick": "1002", "working_bid_tick": "", "working_ask_tick": ""},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 8, "event_type": "decision", "event_source": "strategy", "event_seq": 11, "ts_local": ns(5000), "ts_exch": ns(5000), "best_bid": "100.4", "best_ask": "100.6", "mid": "100.5", "fair": "100.4", "reservation": "100.45", "position": "1.0", "inventory_score": "0.4", "feed_latency_ns": "3000000", "latency_signal_ms": "3.0", "bid_size": "4.0", "ask_size": "1.0", "bid_top5_ticks": "1004|1003|1002|1001|1000", "bid_top5_qtys": "4|1|1|1|1", "ask_top5_ticks": "1006|1007|1008|1009|1010", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "4.0", "target_bid_tick": "1004", "target_ask_tick": "1006", "working_bid_tick": "", "working_ask_tick": ""},
    ]

    replay_rows = [dict(row) for row in live_rows]
    for row in replay_rows:
        if row.get("order_id") == "L1" and row.get("event_type") == "fill":
            row["ts_local"] = str(ns(500))
            row["ts_exch"] = str(ns(500))
            row["strategy_seq"] = "6"
            row["linked_strategy_seq"] = "6"
            row["position"] = "1.0"
            row["mid"] = "100.3"
            row["fill_ts"] = str(ns(500))
            row["fill_price"] = "99.9"
        if row.get("order_id") == "L2" and row.get("event_type") == "cancel_ack":
            row["event_type"] = "fill"
            row["event_source"] = "ws"
            row["action"] = "fill"
            row["ts_local"] = str(ns(300))
            row["ts_exch"] = str(ns(300))
            row["fill_ts"] = str(ns(300))
            row["fill_qty"] = "1.0"
            row["fill_price"] = "100.3"
            row["order_status"] = "filled"
            row["order_executed_qty"] = "1.0"
            row["order_remaining_qty"] = "0.0"
            row["fill_after_cancel_request"] = "1"
        if row.get("order_id") == "L2" and row.get("event_type") == "cancel_sent":
            row["fill_after_cancel_request"] = "0"

    _write_csv(run_dir / "audit_live_sample.csv", live_rows, fieldnames)
    _write_csv(replay_dir / "audit_bt_audit_replay.csv", replay_rows, fieldnames)

    _write_csv(
        sidecar / "joined_decisions.csv",
        [
            {
                "strategy_seq": str(seq),
                "top5_join_age_ms": "5.0",
                "depth_join_age_ms": "5.0",
                "bookticker_join_age_ms": "5.0",
                "max_join_age_ms": "5.0",
                "join_stale": "0",
                "join_gap_crossed": "0",
                "join_missing": "0",
                "join_used_future": "0",
            }
            for seq in range(1, 9)
        ],
        ["strategy_seq", "top5_join_age_ms", "depth_join_age_ms", "bookticker_join_age_ms", "max_join_age_ms", "join_stale", "join_gap_crossed", "join_missing", "join_used_future"],
    )
    _write_json(run_dir / "maker_acceptance_stage3.json", {"market_view": {"classification": "passes_pricing_research_market_view"}})
    _write_json(sidecar / "metrics.json", {"snapshot_alignment_status": "present"})
    _write_json(sidecar / "joined_decisions.metrics.json", {"decision_join_coverage": 1.0})
    return run_dir, output_dir


def test_run_execution_outcome_calibration_outputs_required_artifacts(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)
    manifest = run_execution_outcome_calibration(run_dir=run_dir, output_dir=output_dir, tick_size=0.1)
    assert manifest["task_id"] == "0514T007"
    expected = {
        "execution_calibration_summary.md",
        "submit_key_coverage.csv",
        "fill_horizon_gap.csv",
        "time_to_fill_gap.csv",
        "final_state_gap.csv",
        "cancel_race_gap.csv",
        "markout_gap.csv",
        "placement_strata_gap.csv",
        "inventory_strata_gap.csv",
        "latency_strata_gap.csv",
        "coverage_gap.csv",
        "run_manifest.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})


def test_submit_key_coverage_and_fill_gap(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)
    run_execution_outcome_calibration(run_dir=run_dir, output_dir=output_dir, tick_size=0.1)
    coverage_rows = _read_csv(output_dir / "submit_key_coverage.csv")
    assert len(coverage_rows) == 2
    assert all(row["matched"] == "1" for row in coverage_rows)
    fill_gap_rows = _read_csv(output_dir / "fill_horizon_gap.csv")
    row_100 = next(row for row in fill_gap_rows if row["horizon_ms"] == "100")
    assert row_100["live_fill_rate_all"] == "0.5"
    assert row_100["replay_fill_rate_all"] == "0.5"
    assert row_100["aligned"] == "1"


def test_cancel_race_and_strata_outputs(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)
    manifest = run_execution_outcome_calibration(run_dir=run_dir, output_dir=output_dir, tick_size=0.1)
    assert manifest["decision_state"] == "diagnostic_only_gap_too_large"
    cancel_rows = _read_csv(output_dir / "cancel_race_gap.csv")
    fill_after_cancel = next(row for row in cancel_rows if row["metric_name"] == "fill_after_cancel_request_rate")
    assert fill_after_cancel["live_value"] == "0.0"
    assert fill_after_cancel["replay_value"] == "0.5"
    placement_rows = _read_csv(output_dir / "placement_strata_gap.csv")
    assert any(row["strata_family"] == "placement_bucket" for row in placement_rows)
    latency_rows = _read_csv(output_dir / "latency_strata_gap.csv")
    assert any(row["strata_family"] == "join_stale" for row in latency_rows)
