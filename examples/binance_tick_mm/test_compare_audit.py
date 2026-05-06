from __future__ import annotations

from pathlib import Path

from audit_schema import AUDIT_FIELDS
from compare_audit import _cadence_stats, _nearest_lag_stats, compare


def test_series_int_preserves_nanosecond_precision() -> None:
    rows = [{"ts_local": "1777342117432305601"}]

    stats = _cadence_stats(rows)

    assert stats["ts_local_first"] == 1777342117432305601


def test_cadence_stats_reports_negative_delta_count() -> None:
    rows = [{"ts_local": "300"}, {"ts_local": "100"}, {"ts_local": "200"}]

    stats = _cadence_stats(rows)

    assert stats["negative_delta_count"] == 1


def test_cadence_stats_reports_ts_local_deltas() -> None:
    rows = [{"ts_local": "100"}, {"ts_local": "150"}, {"ts_local": "300"}]

    stats = _cadence_stats(rows)

    assert stats["rows"] == 3
    assert stats["delta_ns"]["count"] == 2
    assert stats["delta_ns"]["p50"] == 100.0
    assert stats["delta_ns"]["max"] == 150.0


def test_nearest_lag_stats_reports_abs_and_signed_lag() -> None:
    bt_rows = [{"ts_local": "90"}, {"ts_local": "210"}, {"ts_local": "300"}]
    live_rows = [{"ts_local": "100"}, {"ts_local": "200"}]

    stats = _nearest_lag_stats(bt_rows, live_rows)

    assert stats["count"] == 2
    assert stats["abs_ns"]["p50"] == 10.0
    assert stats["signed_ns"]["mean"] == 0.0


def _write_audit(path: Path, rows: list[dict[str, str]]) -> None:
    fields = AUDIT_FIELDS
    path.write_text(",".join(fields) + "\n")
    with path.open("a") as f:
        for row in rows:
            merged = {field: "0" for field in fields}
            merged.update(row)
            f.write(",".join(str(merged[field]) for field in fields) + "\n")


def _write_legacy_audit(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        field
        for field in AUDIT_FIELDS
        if field not in {
            "local_open_orders",
            "rest_open_orders",
            "open_order_diff",
            "safety_detail",
            "working_bid_qty",
            "working_ask_qty",
            "working_bid_status",
            "working_ask_status",
            "working_bid_req",
            "working_ask_req",
            "working_bid_pending_cancel",
            "working_ask_pending_cancel",
        }
    ]
    path.write_text(",".join(fields) + "\n")
    with path.open("a") as f:
        for row in rows:
            merged = {field: "0" for field in fields}
            merged.update(row)
            f.write(",".join(str(merged[field]) for field in fields) + "\n")


def test_compare_accepts_legacy_audit_without_open_order_detail_columns(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    row = {"event_type": "decision", "strategy_seq": "1", "ts_local": "100", "action": "keep"}
    _write_legacy_audit(bt, [row])
    _write_legacy_audit(live, [row])

    report = compare(bt, live, align_mode="both", max_lag_ms=1.0)

    assert report["alignment_seq"]["common_rows"] == 1
    assert report["alignment_seq"]["action_match_rate"] == 1.0


def test_compare_reports_seq_and_nearest_ts_alignment(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {"strategy_seq": "1", "ts_local": "100", "action": "keep", "planned_action": "keep"},
            {"strategy_seq": "2", "ts_local": "210", "action": "submit_buy", "planned_action": "submit_buy"},
        ],
    )
    _write_audit(
        live,
        [
            {"strategy_seq": "1", "ts_local": "101", "action": "keep", "planned_action": "keep"},
            {"strategy_seq": "3", "ts_local": "205", "action": "submit_buy", "planned_action": "submit_buy"},
        ],
    )

    report = compare(bt, live, align_mode="both", max_lag_ms=1.0)

    assert report["alignment_seq"]["common_rows"] == 1
    assert report["alignment_nearest_ts"]["nearest_ts_matched_rows"] == 2
    assert report["alignment_nearest_ts"]["nearest_ts_unmatched_rows"] == 0


def test_compare_nearest_ts_excludes_rows_over_max_lag(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(bt, [{"strategy_seq": "1", "ts_local": "100", "action": "keep"}])
    _write_audit(live, [{"strategy_seq": "1", "ts_local": "2_000_000", "action": "keep"}])

    report = compare(bt, live, align_mode="nearest_ts", max_lag_ms=1.0)

    assert report["alignment_nearest_ts"]["nearest_ts_matched_rows"] == 0
    assert report["alignment_nearest_ts"]["nearest_ts_unmatched_rows"] == 1


def test_compare_ignores_lifecycle_rows_for_alignment_and_summary(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {"event_type": "decision", "strategy_seq": "1", "ts_local": "100", "action": "keep"},
            {"event_type": "fill", "strategy_seq": "1", "ts_local": "101", "action": "fill", "position": "1.0"},
        ],
    )
    _write_audit(
        live,
        [
            {"event_type": "decision", "strategy_seq": "1", "ts_local": "100", "action": "keep"},
            {"event_type": "cancel_sent", "strategy_seq": "1", "ts_local": "102", "action": "cancel_sell"},
        ],
    )

    report = compare(bt, live, align_mode="both", max_lag_ms=1.0)

    assert report["bt_event_rows"] == 2
    assert report["live_event_rows"] == 2
    assert report["bt_summary"]["rows"] == 1
    assert report["live_summary"]["rows"] == 1
    assert report["alignment_seq"]["common_rows"] == 1


def test_compare_reports_api_throttle_breakdown_by_planned_action(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "ts_exch": "100",
                "action": "keep",
                "planned_action": "submit_buy",
                "throttle_reason": "min_quote_update_interval",
                "reject_reason": "quote_throttle",
                "dropped_by_api_limit": "1",
                "target_bid_tick": "10",
                "target_ask_tick": "12",
                "working_bid_tick": "-1",
                "working_ask_tick": "-1",
            },
            {
                "event_type": "decision",
                "strategy_seq": "2",
                "ts_local": "200",
                "ts_exch": "200",
                "action": "keep",
                "planned_action": "submit_sell",
                "throttle_reason": "min_quote_update_interval",
                "reject_reason": "quote_throttle",
                "dropped_by_api_limit": "1",
                "target_bid_tick": "20",
                "target_ask_tick": "22",
                "working_bid_tick": "20",
                "working_ask_tick": "-1",
            },
            {
                "event_type": "decision",
                "strategy_seq": "3",
                "ts_local": "300",
                "ts_exch": "300",
                "action": "keep",
                "planned_action": "keep",
                "throttle_reason": "",
                "reject_reason": "",
                "dropped_by_api_limit": "0",
                "target_bid_tick": "30",
                "target_ask_tick": "32",
                "working_bid_tick": "30",
                "working_ask_tick": "32",
            },
        ],
    )
    _write_audit(
        live,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "ts_exch": "100",
                "action": "keep",
                "planned_action": "keep",
                "throttle_reason": "",
                "reject_reason": "",
                "dropped_by_api_limit": "0",
                "target_bid_tick": "10",
                "target_ask_tick": "12",
                "working_bid_tick": "-1",
                "working_ask_tick": "-1",
            },
            {
                "event_type": "decision",
                "strategy_seq": "2",
                "ts_local": "200",
                "ts_exch": "200",
                "action": "keep",
                "planned_action": "submit_sell",
                "throttle_reason": "api_interval",
                "reject_reason": "api_interval_guard",
                "dropped_by_api_limit": "1",
                "target_bid_tick": "20",
                "target_ask_tick": "22",
                "working_bid_tick": "20",
                "working_ask_tick": "-1",
            },
            {
                "event_type": "decision",
                "strategy_seq": "3",
                "ts_local": "300",
                "ts_exch": "300",
                "action": "keep",
                "planned_action": "keep",
                "throttle_reason": "",
                "reject_reason": "",
                "dropped_by_api_limit": "0",
                "target_bid_tick": "30",
                "target_ask_tick": "32",
                "working_bid_tick": "30",
                "working_ask_tick": "32",
            },
        ],
    )

    report = compare(bt, live, align_mode="seq", max_lag_ms=1.0)
    breakdown = report["alignment_seq"]["api_throttle"]

    assert breakdown["all"]["rows"] == 3
    assert breakdown["all"]["api_drop_abs_diff"] == 1 / 3
    assert breakdown["planned_action_match"]["rows"] == 2
    assert breakdown["planned_action_match"]["api_drop_abs_diff"] == 0.0
    assert breakdown["planned_action_mismatch"]["rows"] == 1
    assert breakdown["mismatch_attribution"]["mismatch_rows"] == 2


def test_compare_reports_replay_lag_gate_breakdown(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "keep",
                "replay_lag_abs_ns": "50",
            },
            {
                "event_type": "decision",
                "strategy_seq": "2",
                "ts_local": "200",
                "action": "keep",
                "planned_action": "submit_buy",
                "dropped_by_api_limit": "1",
                "throttle_reason": "api_interval",
                "replay_lag_abs_ns": "300000000",
            },
        ],
    )
    _write_audit(
        live,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "keep",
                "replay_lag_abs_ns": "0",
            },
            {
                "event_type": "decision",
                "strategy_seq": "2",
                "ts_local": "200",
                "action": "keep",
                "planned_action": "submit_buy",
                "dropped_by_api_limit": "0",
                "throttle_reason": "",
                "replay_lag_abs_ns": "0",
            },
        ],
    )

    report = compare(bt, live, align_mode="seq", max_replay_lag_ms=250.0)
    replay_lag = report["alignment_seq"]["replay_lag"]

    assert replay_lag["rows_with_lag"] == 2
    assert replay_lag["in_gate_rows"] == 1
    assert replay_lag["outside_gate_rows"] == 1
    assert replay_lag["buckets"]["le_250ms"]["rows"] == 1
    assert report["alignment_seq"]["api_throttle"]["replay_lag_abs_diff_gt_gate"]["rows"] == 1


def test_compare_reports_dual_replay_lag_gate_breakdown(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "ts_exch": "1000",
                "bt_feed_ts_exch": "1000",
                "action": "keep",
                "planned_action": "keep",
                "replay_lag_abs_ns": "50",
            },
            {
                "event_type": "decision",
                "strategy_seq": "2",
                "ts_local": "200",
                "ts_exch": "2000",
                "bt_feed_ts_exch": "2000",
                "action": "keep",
                "planned_action": "submit_buy",
                "dropped_by_api_limit": "1",
                "throttle_reason": "api_interval",
                "replay_lag_abs_ns": "50",
            },
        ],
    )
    _write_audit(
        live,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "ts_exch": "1020",
                "action": "keep",
                "planned_action": "keep",
            },
            {
                "event_type": "decision",
                "strategy_seq": "2",
                "ts_local": "200",
                "ts_exch": "3000",
                "action": "keep",
                "planned_action": "submit_buy",
                "dropped_by_api_limit": "0",
                "throttle_reason": "",
            },
        ],
    )

    report = compare(bt, live, align_mode="seq", max_replay_lag_ms=0.00025)
    replay_lag = report["alignment_seq"]["replay_lag"]
    api_throttle = report["alignment_seq"]["api_throttle"]

    assert replay_lag["in_gate_rows"] == 2
    assert replay_lag["in_exchange_gate_rows"] == 1
    assert replay_lag["in_dual_gate_rows"] == 1
    assert replay_lag["outside_dual_gate_rows"] == 1
    assert replay_lag["exchange_abs_ns"]["max"] == 1000.0
    assert replay_lag["stateful_gate"]["strict_full_window_passed"] is False
    assert replay_lag["stateful_gate"]["clean_prefix_rows"] == 1
    assert replay_lag["stateful_gate"]["state_contaminated_rows"] == 1
    assert replay_lag["stateful_gate"]["first_outside_dual_gate"]["strategy_seq"] == 2
    assert replay_lag["lifecycle"]["dual_gate_subset"]["rows"] == 1
    assert replay_lag["lifecycle"]["outside_dual_gate_subset"]["rows"] == 1
    assert api_throttle["replay_dual_lag_abs_diff_gt_gate"]["rows"] == 1
    assert api_throttle["mismatch_attribution"]["replay_exchange_lag_gt_gate_rows"] == 1


def test_compare_reports_working_order_lifecycle_breakdown(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "submit_buy",
                "working_bid_tick": "1000",
                "working_ask_tick": "-1",
                "working_buy_order_id": "7",
                "working_sell_order_id": "",
                "local_open_order_count": "1",
                "rest_open_order_count": "0",
                "local_open_orders": "7:buy:1000:0.001:new",
                "rest_open_orders": "",
                "open_order_diff": "local_only=7",
                "dropped_by_api_limit": "1",
                "throttle_reason": "api_interval",
            }
        ],
    )
    _write_audit(
        live,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "keep",
                "working_bid_tick": "-1",
                "working_ask_tick": "-1",
                "working_buy_order_id": "",
                "working_sell_order_id": "",
                "local_open_order_count": "0",
                "rest_open_order_count": "0",
                "local_open_orders": "",
                "rest_open_orders": "",
                "open_order_diff": "",
                "dropped_by_api_limit": "0",
                "throttle_reason": "",
            }
        ],
    )

    report = compare(bt, live, align_mode="seq", max_lag_ms=1.0)
    lifecycle = report["alignment_seq"]["working_order_lifecycle"]
    categories = {row["key"]: row["count"] for row in lifecycle["category_rows"]}

    assert lifecycle["mismatch_rows"] == 1
    assert lifecycle["planned_action_mismatch_with_lifecycle_mismatch_rows"] == 1
    assert lifecycle["api_throttle_mismatch_with_lifecycle_mismatch_rows"] == 1
    assert categories["working_tick"] == 1
    assert categories["working_order_id"] == 1
    assert categories["local_open_order_count"] == 1
    assert categories["local_open_order_detail"] == 1
    assert "rest_open_order_detail" not in categories
    assert report["alignment_seq"]["api_throttle"]["mismatch_attribution"][
        "planned_action_mismatch_rows"
    ] == 1


def test_compare_does_not_treat_missing_rest_snapshot_as_rest_local_divergence(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "keep",
                "working_bid_tick": "1000",
                "working_ask_tick": "1004",
                "working_buy_order_id": "7",
                "working_sell_order_id": "",
                "local_open_order_count": "1",
                "rest_open_order_count": "0",
                "local_open_orders": "7:buy:1000:0.001:new:req=new:cxl=0",
                "rest_open_orders": "",
                "open_order_diff": "",
                "safety_status": "",
            }
        ],
    )
    _write_audit(
        live,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "keep",
                "working_bid_tick": "1000",
                "working_ask_tick": "1004",
                "working_buy_order_id": "7",
                "working_sell_order_id": "",
                "local_open_order_count": "1",
                "rest_open_order_count": "0",
                "local_open_orders": "7:buy:1000:0.001:new:req=new:cxl=0",
                "rest_open_orders": "",
                "open_order_diff": "",
                "safety_status": "ok",
            }
        ],
    )

    report = compare(bt, live, align_mode="seq", max_lag_ms=1.0)
    lifecycle = report["alignment_seq"]["working_order_lifecycle"]

    assert lifecycle["rest_local_divergence_rows"] == 0


def test_compare_uses_explicit_working_order_semantics_when_present(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "keep",
                "working_bid_tick": "1000",
                "working_ask_tick": "-1",
                "working_buy_order_id": "1",
                "working_bid_qty": "0.001",
                "working_bid_status": "new",
                "working_bid_req": "none",
                "working_bid_pending_cancel": "0",
                "local_open_orders": "1:buy:1000:0.001:new:req=none:cxl=1",
            }
        ],
    )
    _write_audit(
        live,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "keep",
                "working_bid_tick": "1000",
                "working_ask_tick": "-1",
                "working_buy_order_id": "99",
                "working_bid_qty": "0.001",
                "working_bid_status": "new",
                "working_bid_req": "none",
                "working_bid_pending_cancel": "0",
                "local_open_orders": "99:buy:1000:0.001:new:req=none:cxl=1",
            }
        ],
    )

    report = compare(bt, live, align_mode="seq", max_lag_ms=1.0)
    lifecycle = report["alignment_seq"]["working_order_lifecycle"]

    assert lifecycle["semantic_mismatch_rows"] == 0
    assert lifecycle["identity_only_mismatch_rows"] == 1
    assert lifecycle["first_identity_only_divergence"]["strategy_seq"] == 1


def test_compare_splits_semantic_and_identity_only_lifecycle_mismatches(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "submit_buy",
                "working_bid_tick": "1000",
                "working_ask_tick": "1004",
                "working_buy_order_id": "7",
                "working_sell_order_id": "",
                "local_open_order_count": "1",
                "local_open_orders": "7:buy:1000:0.001:new:req=new:cxl=0",
                "rest_open_order_count": "0",
                "rest_open_orders": "",
                "open_order_diff": "",
            }
        ],
    )
    _write_audit(
        live,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "submit_buy",
                "working_bid_tick": "1000",
                "working_ask_tick": "1004",
                "working_buy_order_id": "99",
                "working_sell_order_id": "",
                "local_open_order_count": "1",
                "local_open_orders": "99:buy:1000:0.001:new:req=new:cxl=0",
                "rest_open_order_count": "0",
                "rest_open_orders": "",
                "open_order_diff": "",
            }
        ],
    )

    report = compare(bt, live, align_mode="seq", max_lag_ms=1.0)
    lifecycle = report["alignment_seq"]["working_order_lifecycle"]
    categories = {row["key"]: row["count"] for row in lifecycle["category_rows"]}
    semantic_categories = {row["key"]: row["count"] for row in lifecycle["semantic_category_rows"]}
    identity_categories = {row["key"]: row["count"] for row in lifecycle["identity_category_rows"]}

    assert lifecycle["mismatch_rows"] == 1
    assert lifecycle["semantic_mismatch_rows"] == 0
    assert lifecycle["identity_only_mismatch_rows"] == 1
    assert categories["working_order_id"] == 1
    assert identity_categories["working_order_id"] == 1
    assert semantic_categories == {}
    assert lifecycle["first_identity_only_divergence"]["strategy_seq"] == 1


def test_compare_reports_first_semantic_divergence_context(tmp_path: Path) -> None:
    bt = tmp_path / "bt.csv"
    live = tmp_path / "live.csv"
    _write_audit(
        bt,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "event_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "submit_buy",
                "working_bid_tick": "1000",
                "working_ask_tick": "1004",
                "working_buy_order_id": "7",
                "working_sell_order_id": "",
                "local_open_order_count": "1",
                "local_open_orders": "7:buy:1000:0.001:new:req=new:cxl=0",
            },
            {
                "event_type": "fill",
                "strategy_seq": "1",
                "event_seq": "2",
                "ts_local": "101",
                "action": "fill",
                "planned_action": "submit_buy",
                "order_id": "7",
                "order_side": "buy",
                "order_price_tick": "1000",
                "order_qty": "0.001",
                "order_status": "filled",
                "lifecycle_state": "fill",
            },
        ],
    )
    _write_audit(
        live,
        [
            {
                "event_type": "decision",
                "strategy_seq": "1",
                "event_seq": "1",
                "ts_local": "100",
                "action": "keep",
                "planned_action": "submit_buy",
                "working_bid_tick": "1000",
                "working_ask_tick": "1008",
                "working_buy_order_id": "7",
                "working_sell_order_id": "",
                "local_open_order_count": "1",
                "local_open_orders": "7:buy:1000:0.001:new:req=new:cxl=0",
            },
            {
                "event_type": "cancel_sent",
                "strategy_seq": "1",
                "event_seq": "2",
                "ts_local": "101",
                "action": "cancel_buy",
                "planned_action": "submit_buy",
                "order_id": "7",
                "order_side": "buy",
                "order_price_tick": "1000",
                "order_qty": "0.001",
                "lifecycle_state": "cancel_sent",
            },
        ],
    )

    report = compare(bt, live, align_mode="seq", max_lag_ms=1.0)
    context = report["alignment_seq"]["working_order_lifecycle"]["first_semantic_divergence_context"]

    assert context["divergence"]["strategy_seq"] == 1
    assert len(context["bt_rows"]) == 2
    assert len(context["live_rows"]) == 2
    assert context["bt_rows"][0]["event_type"] == "decision"
    assert context["live_rows"][1]["event_type"] == "cancel_sent"
