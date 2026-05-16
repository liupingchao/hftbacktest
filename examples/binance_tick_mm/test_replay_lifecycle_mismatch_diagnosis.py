from __future__ import annotations

import csv
import json
from pathlib import Path

from replay_lifecycle_mismatch_diagnosis import (
    run_queue_priority_evidence_diagnosis,
    run_replay_lifecycle_mismatch_diagnosis,
    run_residual_replay_fill_diagnosis,
    run_single_replay_fill_trigger_diagnosis,
)


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
    output_dir = run_dir / "diag"
    sidecar = run_dir / "t009_fixed_sidecar"
    replay_dir = run_dir / "out" / "backtest_audit_replay"
    ns = lambda ms: 1_000_000_000 + ms * 1_000_000
    fieldnames = [
        "run_id", "symbol", "strategy_seq", "event_type", "event_source", "event_seq", "ts_local", "ts_exch",
        "order_id", "action", "planned_order_id", "planned_action", "throttle_reason", "reject_reason",
        "best_bid", "best_ask", "mid", "fair", "reservation", "position", "inventory_score",
        "feed_latency_ns", "latency_signal_ms", "bid_size", "ask_size", "bid_top5_ticks", "bid_top5_qtys",
        "ask_top5_ticks", "ask_top5_qtys", "market_view_source", "top5_source", "book_view_stale_ms",
        "target_bid_tick", "target_ask_tick", "working_bid_tick", "working_ask_tick", "order_side",
        "order_price", "order_price_tick", "order_qty", "order_remaining_qty", "order_executed_qty",
        "order_status", "lifecycle_state", "linked_strategy_seq", "linked_action", "linked_order_id",
        "cancel_request_ts", "cancel_ack_ts", "fill_ts", "fill_qty", "fill_price", "fill_after_cancel_request",
    ]
    live_rows = [
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 1, "event_type": "decision", "event_source": "strategy", "event_seq": 1, "ts_local": ns(0), "ts_exch": ns(0), "best_bid": "99.9", "best_ask": "100.1", "mid": "100.0", "fair": "100.2", "reservation": "100.15", "position": "0.0", "inventory_score": "0.1", "feed_latency_ns": "1000000", "latency_signal_ms": "1.0", "bid_size": "2.0", "ask_size": "1.0", "bid_top5_ticks": "999|998|997|996|995", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1001|1002|1003|1004|1005", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "1.0", "target_bid_tick": "999", "target_ask_tick": "1001", "working_bid_tick": "", "working_ask_tick": ""},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 1, "event_type": "order_submit_sent", "event_source": "strategy", "event_seq": 2, "ts_local": ns(0), "ts_exch": ns(0), "order_id": "L1", "action": "submit_buy", "planned_action": "submit_buy", "best_bid": "99.9", "best_ask": "100.1", "mid": "100.0", "fair": "100.2", "reservation": "100.15", "position": "0.0", "inventory_score": "0.1", "feed_latency_ns": "1000000", "latency_signal_ms": "1.0", "bid_size": "2.0", "ask_size": "1.0", "bid_top5_ticks": "999|998|997|996|995", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1001|1002|1003|1004|1005", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "1.0", "target_bid_tick": "999", "target_ask_tick": "1001", "order_side": "buy", "order_price": "99.9", "order_price_tick": "999", "order_qty": "1.0", "lifecycle_state": "order_submit_sent", "linked_strategy_seq": "1", "linked_action": "submit_buy", "linked_order_id": "L1", "fill_after_cancel_request": "0"},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 1, "event_type": "cancel_sent", "event_source": "strategy", "event_seq": 2, "ts_local": ns(90), "ts_exch": ns(90), "order_id": "L1", "action": "cancel_buy", "planned_action": "cancel_buy", "best_bid": "99.9", "best_ask": "100.1", "mid": "100.0", "fair": "100.2", "reservation": "100.15", "position": "0.0", "inventory_score": "0.1", "feed_latency_ns": "1000000", "latency_signal_ms": "1.0", "bid_size": "2.0", "ask_size": "1.0", "bid_top5_ticks": "999|998|997|996|995", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1001|1002|1003|1004|1005", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "1.0", "target_bid_tick": "999", "target_ask_tick": "1001", "order_side": "buy", "order_price": "99.9", "order_price_tick": "999", "order_qty": "1.0", "cancel_request_ts": str(ns(90)), "lifecycle_state": "cancel_sent", "linked_strategy_seq": "1", "linked_action": "cancel_buy", "linked_order_id": "L1", "fill_after_cancel_request": "0"},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 2, "event_type": "decision", "event_source": "strategy", "event_seq": 3, "ts_local": ns(100), "ts_exch": ns(100), "best_bid": "100.0", "best_ask": "100.2", "mid": "100.1", "fair": "100.25", "reservation": "100.2", "position": "1.0", "inventory_score": "0.2", "feed_latency_ns": "1000000", "latency_signal_ms": "1.0", "bid_size": "2.0", "ask_size": "1.0", "bid_top5_ticks": "1000|999|998|997|996", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1002|1003|1004|1005|1006", "ask_top5_qtys": "1|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "1.0", "target_bid_tick": "1000", "target_ask_tick": "1002", "working_bid_tick": "999", "working_ask_tick": ""},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 2, "event_type": "fill", "event_source": "ws", "event_seq": 4, "ts_local": ns(100), "ts_exch": ns(100), "order_id": "L1", "action": "fill", "best_bid": "100.0", "best_ask": "100.2", "mid": "100.1", "fair": "100.25", "reservation": "100.2", "position": "1.0", "inventory_score": "0.2", "feed_latency_ns": "1000000", "latency_signal_ms": "1.0", "bid_size": "2.0", "ask_size": "1.0", "bid_top5_ticks": "1000|999|998|997|996", "bid_top5_qtys": "2|1|1|1|1", "ask_top5_ticks": "1002|1003|1004|1005|1006", "ask_top5_qtys": "1|1|1|1|1", "order_side": "buy", "order_price": "99.9", "order_price_tick": "999", "order_qty": "1.0", "order_remaining_qty": "0.0", "order_executed_qty": "1.0", "order_status": "filled", "lifecycle_state": "fill", "linked_strategy_seq": "2", "linked_action": "fill", "linked_order_id": "L1", "cancel_request_ts": str(ns(90)), "fill_ts": str(ns(100)), "fill_qty": "1.0", "fill_price": "99.9", "fill_after_cancel_request": "1"},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 3, "event_type": "decision", "event_source": "strategy", "event_seq": 5, "ts_local": ns(200), "ts_exch": ns(200), "best_bid": "100.1", "best_ask": "100.3", "mid": "100.2", "fair": "100.0", "reservation": "100.05", "position": "1.0", "inventory_score": "0.6", "feed_latency_ns": "1000000", "latency_signal_ms": "5.0", "bid_size": "3.0", "ask_size": "4.0", "bid_top5_ticks": "1001|1000|999|998|997", "bid_top5_qtys": "3|1|1|1|1", "ask_top5_ticks": "1003|1004|1005|1006|1007", "ask_top5_qtys": "4|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "3.0", "target_bid_tick": "1001", "target_ask_tick": "1003", "working_bid_tick": "", "working_ask_tick": ""},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 3, "event_type": "order_submit_sent", "event_source": "strategy", "event_seq": 6, "ts_local": ns(200), "ts_exch": ns(200), "order_id": "L2", "action": "submit_sell", "planned_action": "submit_sell", "best_bid": "100.1", "best_ask": "100.3", "mid": "100.2", "fair": "100.0", "reservation": "100.05", "position": "1.0", "inventory_score": "0.6", "feed_latency_ns": "1000000", "latency_signal_ms": "5.0", "bid_size": "3.0", "ask_size": "4.0", "bid_top5_ticks": "1001|1000|999|998|997", "bid_top5_qtys": "3|1|1|1|1", "ask_top5_ticks": "1003|1004|1005|1006|1007", "ask_top5_qtys": "4|1|1|1|1", "market_view_source": "live_depth", "top5_source": "live_depth", "book_view_stale_ms": "3.0", "target_bid_tick": "1001", "target_ask_tick": "1003", "order_side": "sell", "order_price": "100.3", "order_price_tick": "1003", "order_qty": "1.0", "lifecycle_state": "order_submit_sent", "linked_strategy_seq": "3", "linked_action": "submit_sell", "linked_order_id": "L2", "fill_after_cancel_request": "0"},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 4, "event_type": "cancel_sent", "event_source": "strategy", "event_seq": 7, "ts_local": ns(450), "ts_exch": ns(450), "order_id": "L2", "action": "cancel_sell", "best_bid": "100.1", "best_ask": "100.3", "mid": "100.2", "position": "1.0", "inventory_score": "0.6", "linked_strategy_seq": "4", "linked_action": "cancel_sell", "linked_order_id": "L2", "order_side": "sell", "cancel_request_ts": str(ns(450))},
        {"run_id": "sample", "symbol": "BTCUSDT", "strategy_seq": 5, "event_type": "cancel_ack", "event_source": "ws", "event_seq": 8, "ts_local": ns(500), "ts_exch": ns(500), "order_id": "L2", "action": "cancel_ack", "best_bid": "100.1", "best_ask": "100.3", "mid": "100.2", "position": "1.0", "inventory_score": "0.6", "linked_strategy_seq": "5", "linked_action": "cancel_ack", "linked_order_id": "L2", "order_side": "sell", "cancel_request_ts": str(ns(450)), "cancel_ack_ts": str(ns(500))},
    ]
    replay_rows = [dict(row) for row in live_rows]
    for row in replay_rows:
        if row.get("order_id") == "L2" and row.get("event_type") == "cancel_ack":
            row["event_type"] = "fill"
            row["event_source"] = "ws"
            row["action"] = "fill"
            row["fill_ts"] = str(ns(300))
            row["fill_qty"] = "1.0"
            row["fill_price"] = "100.3"
            row["order_status"] = "filled"
            row["order_executed_qty"] = "1.0"
            row["order_remaining_qty"] = "0.0"
            row["fill_after_cancel_request"] = "1"
            row["ts_local"] = str(ns(300))
            row["ts_exch"] = str(ns(300))
        if row.get("order_id") == "L2" and row.get("event_type") == "fill":
            row["linked_strategy_seq"] = "5"
            row["linked_action"] = "fill"
        if row.get("order_id") == "L1" and row.get("event_type") == "fill":
            row["event_type"] = "cancel_ack"
            row["action"] = "cancel_ack"
            row["order_status"] = "canceled"
            row["cancel_request_ts"] = str(ns(90))
            row["cancel_ack_ts"] = str(ns(500))
            row["fill_ts"] = ""
            row["fill_qty"] = "0.0"
            row["fill_price"] = "0.0"
            row["fill_after_cancel_request"] = "0"
            row["order_executed_qty"] = "0.0"
            row["order_remaining_qty"] = "1.0"
            row["ts_local"] = str(ns(500))
            row["ts_exch"] = str(ns(500))
    _write_csv(run_dir / "audit_live_sample.csv", live_rows, fieldnames)
    _write_csv(replay_dir / "audit_bt_audit_replay.csv", replay_rows, fieldnames)
    _write_csv(
        sidecar / "joined_decisions.csv",
        [
            {"strategy_seq": str(seq), "top5_join_age_ms": "5.0", "depth_join_age_ms": "5.0", "bookticker_join_age_ms": "5.0", "max_join_age_ms": "5.0", "join_stale": "0", "join_gap_crossed": "0", "join_missing": "0", "join_used_future": "0"}
            for seq in range(1, 6)
        ],
        ["strategy_seq", "top5_join_age_ms", "depth_join_age_ms", "bookticker_join_age_ms", "max_join_age_ms", "join_stale", "join_gap_crossed", "join_missing", "join_used_future"],
    )
    _write_csv(
        sidecar / "top5_sidecar.csv",
        [
            {
                "raw_seq": "1",
                "event_type": "depthUpdate",
                "local_ts": str(ns(0)),
                "exch_ts": str(ns(0)),
                "sync_aligned": "True",
                "bid_top5_px": "99.9|99.8|99.7|99.6|99.5",
                "bid_top5_qtys": "5|1|1|1|1",
                "ask_top5_px": "100.3|100.4|100.5|100.6|100.7",
                "ask_top5_qtys": "4|1|1|1|1",
                "bookticker_depth_age_ms": "1.0",
            },
            {
                "raw_seq": "2",
                "event_type": "depthUpdate",
                "local_ts": str(ns(260)),
                "exch_ts": str(ns(260)),
                "sync_aligned": "True",
                "bid_top5_px": "100.1|100.0|99.9|99.8|99.7",
                "bid_top5_qtys": "3|1|1|1|1",
                "ask_top5_px": "100.3|100.4|100.5|100.6|100.7",
                "ask_top5_qtys": "4|1|1|1|1",
                "bookticker_depth_age_ms": "1.0",
            },
        ],
        [
            "raw_seq",
            "event_type",
            "local_ts",
            "exch_ts",
            "sync_aligned",
            "bid_top5_px",
            "bid_top5_qtys",
            "ask_top5_px",
            "ask_top5_qtys",
            "bookticker_depth_age_ms",
        ],
    )
    raw_dir = run_dir / "raw_market_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    import gzip
    raw_lines = [
        f"{ns(89)} " + json.dumps({"stream": "btcusdt@bookTicker", "data": {"e": "bookTicker", "u": 1, "b": "99.9", "B": "5.0", "a": "100.1", "A": "5.0"}}),
        f"{ns(99)} " + json.dumps({"stream": "btcusdt@trade", "data": {"e": "trade", "p": "99.9", "q": "1.0", "m": True}}),
        f"{ns(298)} " + json.dumps({"stream": "btcusdt@trade", "data": {"e": "trade", "p": "100.3", "q": "1.0", "m": False}}),
    ]
    with gzip.open(raw_dir / "btcusdt_20260513.gz", "wt", encoding="utf-8") as fh:
        for line in raw_lines:
            fh.write(line + "\n")
    _write_json(run_dir / "maker_acceptance_stage3.json", {"market_view": {"classification": "passes_pricing_research_market_view"}})
    _write_json(sidecar / "metrics.json", {"snapshot_alignment_status": "present"})
    _write_json(sidecar / "joined_decisions.metrics.json", {"decision_join_coverage": 1.0})
    return run_dir, output_dir


def test_run_replay_lifecycle_mismatch_diagnosis_outputs_required_artifacts(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)
    manifest = run_replay_lifecycle_mismatch_diagnosis(run_dir=run_dir, output_dir=output_dir, tick_size=0.1)
    assert manifest["task_id"] == "0515T001"
    expected = {
        "REPLAY_LIFECYCLE_MISMATCH_DIAGNOSIS_SUMMARY.md",
        "matched_submit_state_diff.csv",
        "replay_only_fill_cases.csv",
        "live_cancel_replay_fill_cases.csv",
        "cancel_fill_timeline_diff.csv",
        "terminal_state_transition_diff.csv",
        "cancel_ack_delay_diff.csv",
        "state_diff_by_placement.csv",
        "state_diff_by_inventory.csv",
        "state_diff_by_latency.csv",
        "replay_only_fill_by_horizon.csv",
        "cancel_race_gap_by_bucket.csv",
        "run_manifest.json",
    }
    assert expected.issubset({path.name for path in output_dir.iterdir()})


def test_replay_only_fill_and_live_cancel_replay_fill_cases(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)
    run_replay_lifecycle_mismatch_diagnosis(run_dir=run_dir, output_dir=output_dir, tick_size=0.1)
    replay_only_rows = _read_csv(output_dir / "replay_only_fill_cases.csv")
    assert len(replay_only_rows) == 1
    assert replay_only_rows[0]["case_label"] == "live_canceled_replay_filled"
    cancel_fill_rows = _read_csv(output_dir / "live_cancel_replay_fill_cases.csv")
    assert len(cancel_fill_rows) == 1
    timeline_rows = _read_csv(output_dir / "cancel_fill_timeline_diff.csv")
    assert len(timeline_rows) == 2
    assert any(row["replay_fill_after_cancel_request"] == "1" for row in timeline_rows)


def test_grouped_state_diff_outputs(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)
    run_replay_lifecycle_mismatch_diagnosis(run_dir=run_dir, output_dir=output_dir, tick_size=0.1)
    placement_rows = _read_csv(output_dir / "state_diff_by_placement.csv")
    assert any(row["group_name"] == "placement_bucket" for row in placement_rows)
    latency_rows = _read_csv(output_dir / "state_diff_by_latency.csv")
    assert any(row["group_name"] == "latency_signal_ms_bucket" for row in latency_rows)
    horizon_rows = _read_csv(output_dir / "replay_only_fill_by_horizon.csv")
    assert len(horizon_rows) == 4


def test_run_residual_replay_fill_diagnosis_classifies_two_residual_modes(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)
    residual_dir = output_dir / "residual"
    manifest = run_residual_replay_fill_diagnosis(run_dir=run_dir, output_dir=residual_dir, tick_size=0.1)

    assert manifest["task_id"] == "0515T004"
    rows = _read_csv(residual_dir / "residual_case_diagnosis.csv")
    assert len(rows) == 2
    by_case = {row["case_label"]: row for row in rows}
    assert by_case["live_filled_replay_canceled"]["residual_trigger_class"] == "cancel_race_window_too_short"
    assert by_case["live_canceled_replay_filled"]["residual_trigger_class"] == "touch_fill_assumption_too_optimistic"


def test_run_single_replay_fill_trigger_diagnosis_classifies_queue_proxy_bias(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)
    single_dir = output_dir / "single"
    manifest = run_single_replay_fill_trigger_diagnosis(
        run_dir=run_dir,
        output_dir=single_dir,
        target_order_id="L2",
        tick_size=0.1,
    )

    assert manifest["task_id"] == "0515T006"
    rows = _read_csv(single_dir / "single_case_diagnosis.csv")
    assert len(rows) == 1
    assert rows[0]["target_order_id"] == "L2"
    assert rows[0]["single_case_trigger_class"] == "queue_exposure_proxy_bias_possible"
    assert int(rows[0]["replay_supportive_trade_count_10ms"]) > 0


def test_run_queue_priority_evidence_diagnosis_compares_trade_qty_to_visible_queue(tmp_path: Path) -> None:
    run_dir, output_dir = _sample_tree(tmp_path)
    queue_dir = output_dir / "queue"
    manifest = run_queue_priority_evidence_diagnosis(
        run_dir=run_dir,
        output_dir=queue_dir,
        target_order_id="L2",
        tick_size=0.1,
    )

    assert manifest["task_id"] == "0516T001"
    rows = _read_csv(queue_dir / "queue_priority_evidence.csv")
    assert len(rows) == 1
    assert rows[0]["target_order_id"] == "L2"
    assert rows[0]["queue_priority_diagnosis_class"] == "queue_ahead_depth_can_absorb_observed_trades"
    assert float(rows[0]["same_price_trade_qty_submit_to_replay_fill"]) < float(rows[0]["submit_order_price_visible_qty"])
    assert rows[0]["evidence_sufficient_for_repair"] == "0"

    window_rows = _read_csv(queue_dir / "queue_priority_window_trade_qty.csv")
    assert any(row["window_ms"] == "10" for row in window_rows)
    depth_rows = _read_csv(queue_dir / "queue_priority_depth_timeline.csv")
    assert depth_rows
    trade_rows = _read_csv(queue_dir / "queue_priority_supportive_trades.csv")
    assert len(trade_rows) == 1
