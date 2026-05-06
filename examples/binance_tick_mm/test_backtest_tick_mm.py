from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hftbacktest import (
    BUY_EVENT,
    DEPTH_CLEAR_EVENT,
    DEPTH_EVENT,
    DEPTH_SNAPSHOT_EVENT,
    EXCH_EVENT,
    LOCAL_EVENT,
    BUY,
    SELL_EVENT,
    SELL,
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    event_dtype,
)

from audit_schema import AUDIT_FIELDS
from backtest_tick_mm import (
    AUDIT_REPLAY_DECISION_MARKER_EVENT,
    AuditReplayScheduleEntry,
    FeedLatencyOracle,
    _empty_replay_lag_gate_stats,
    _alignment_init_config,
    _apply_alignment_initial_position,
    _apply_initial_snapshot,
    _audit_replay_decision_due,
    _backtest_cadence_config,
    _backtest_cadence_interval_ns,
    _continuous_run_metadata,
    _latency_guard_signal_ns,
    _load_audit_cadence_schedule,
    _load_audit_cadence_schedule_with_stats,
    _load_audit_replay_schedule_with_stats,
    _load_live_strategy_position_by_decision_ts,
    _load_live_market_state_by_decision_ts,
    _insert_audit_replay_decision_markers,
    _live_local_feed_compat_data,
    _live_visible_working_orders,
    _market_data_replay_config,
    _round_position_qty,
    _strip_local_snapshot_events,
    _audit_replay_schedule_decision_due,
    _select_data_for_asset,
    _should_skip_strategy_decision,
    _slice_data_by_absolute_local_ts,
    _validate_manifest_paths,
)
from strategy_core import (
    Action,
    ExtraOrder,
    GreekValues,
    LiveSafetyConfig,
    OrderLifecycleTracker,
    OrderSnapshot,
    PendingLocalOrder,
    QuoteThrottleState,
    WorkingOrders,
    build_audit_row,
    build_lifecycle_event_row,
    decide_actions,
    evaluate_live_safety,
    format_rest_open_orders,
    is_pure_cancel_extra,
    merge_pending_orders,
    open_order_diff,
    update_quote_throttle_state,
)



class FakeAsset:
    def __init__(self) -> None:
        self.snapshots: list[str] = []

    def initial_snapshot(self, path: str) -> None:
        self.snapshots.append(path)


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return str(path)


def test_live_open_order_diagnostics_are_in_audit_schema() -> None:
    for field in [
        "local_open_orders",
        "rest_open_orders",
        "open_order_diff",
        "safety_detail",
        "predicted_entry_ns",
        "event_source",
        "event_seq",
        "cancel_request_ts",
        "fill_after_cancel_request",
        "lifecycle_detail",
    ]:
        assert field in AUDIT_FIELDS


def test_build_lifecycle_event_row_records_fill_after_cancel_request() -> None:
    order = OrderSnapshot(
        order_id=7,
        side="sell",
        price=101.0,
        price_tick=1010,
        qty=0.001,
        leaves_qty=0.0,
        exec_qty=0.001,
        exec_price_tick=1010,
        status="filled",
        req="none",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=False,
    )

    row = build_lifecycle_event_row(
        run_id="run",
        symbol="BTCUSDT",
        strategy_seq=3,
        event_seq=4,
        event_type="fill",
        event_source="backtest_exchange",
        ts_local=130,
        order=order,
        cancel_requested=True,
        cancel_request_ts=121,
        fill_ts=123,
        fill_qty=0.001,
        fill_price=101.0,
        fill_after_cancel_request=True,
        local_order_seen=True,
        lifecycle_detail="fill_after_cancel_request",
    )

    assert set(row) == set(AUDIT_FIELDS)
    assert row["event_type"] == "fill"
    assert row["event_source"] == "backtest_exchange"
    assert row["event_seq"] == 4
    assert row["order_id"] == "7"
    assert row["order_status"] == "filled"
    assert row["cancel_requested"] == 1
    assert row["cancel_request_ts"] == 121
    assert row["fill_after_cancel_request"] == 1
    assert row["lifecycle_detail"] == "fill_after_cancel_request"


def test_order_lifecycle_tracker_marks_fill_after_cancel_request() -> None:
    tracker = OrderLifecycleTracker.create()
    tracker.mark_cancel_requested(7, 100)

    assert tracker.cancel_request_ts(7) == 100


def test_format_rest_open_orders_and_diff_extract_local_ids() -> None:
    rows = [
        {
            "clientOrderId": "hft-101",
            "orderId": 987654,
            "side": "BUY",
            "price": "77000.10",
            "origQty": "0.001",
            "executedQty": "0",
            "status": "NEW",
            "timeInForce": "GTX",
            "updateTime": 1777440000000,
        }
    ]

    rest_open_orders = format_rest_open_orders(rows, tick_size=0.1)
    diff = open_order_diff(
        "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1;202:sell:770101:0.001:new:req=none:cxl=1:exch=1:local=1",
        rest_open_orders,
    )

    assert "101:buy:770001:0.001:price=77000.10" in rest_open_orders
    assert diff == "local_only=sell:770101:0.001"


def test_open_order_diff_ignores_client_ids_when_quote_keys_match() -> None:
    diff = open_order_diff(
        "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1",
        "999:buy:770001:0.001:price=77000.10:exec=0:status=new:tif=gtx",
    )

    assert diff == ""


def test_open_order_diff_uses_quote_key_when_rest_client_id_is_random() -> None:
    rest_open_orders = format_rest_open_orders(
        [
            {
                "clientOrderId": "mmrandom",
                "orderId": 987654,
                "side": "BUY",
                "price": "77000.10",
                "origQty": "0.00100000",
                "executedQty": "0",
                "status": "NEW",
                "timeInForce": "GTX",
            }
        ],
        tick_size=0.1,
    )

    diff = open_order_diff(
        "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1",
        rest_open_orders,
    )

    assert diff == ""


def test_pure_cancel_extra_can_bypass_api_interval_guard() -> None:
    assert is_pure_cancel_extra([Action("cancel", "extra", 1, 0.0, 0.0)]) is True
    assert is_pure_cancel_extra([Action("cancel", "buy", 1, 0.0, 0.0)]) is False
    assert is_pure_cancel_extra(
        [
            Action("cancel", "extra", 1, 0.0, 0.0),
            Action("submit", "buy", 2, 100.0, 1.0),
        ]
    ) is False


def test_update_quote_throttle_state_marks_submit_and_normal_cancel() -> None:
    state = QuoteThrottleState()

    update_quote_throttle_state(
        state,
        ts_local=100,
        target_bid_tick=1000,
        target_ask_tick=1002,
        actions=[Action("submit", "buy", 1, 100.0, 0.001)],
    )
    assert state.last_sent_api_ts == 100
    assert state.last_sent_target_bid_tick == 1000
    assert state.last_sent_target_ask_tick == 1002

    update_quote_throttle_state(
        state,
        ts_local=200,
        target_bid_tick=1001,
        target_ask_tick=1003,
        actions=[Action("cancel", "sell", 2, 0.0, 0.0)],
    )
    assert state.last_sent_api_ts == 200
    assert state.last_sent_target_bid_tick == 1001
    assert state.last_sent_target_ask_tick == 1003


def test_update_quote_throttle_state_ignores_pure_extra_cancel() -> None:
    state = QuoteThrottleState(last_sent_api_ts=100, last_sent_target_bid_tick=1000, last_sent_target_ask_tick=1002)

    update_quote_throttle_state(
        state,
        ts_local=200,
        target_bid_tick=1001,
        target_ask_tick=1003,
        actions=[Action("cancel", "extra", 3, 0.0, 0.0)],
    )

    assert state.last_sent_api_ts == 100
    assert state.last_sent_target_bid_tick == 1000
    assert state.last_sent_target_ask_tick == 1002


def test_live_uses_shared_quote_throttle_state_update() -> None:
    live_source = Path(__file__).with_name("live_tick_mm.py").read_text()
    direct_mark_sent = "throttle_state." + "mark" + "_sent("

    assert direct_mark_sent not in live_source
    assert "update_quote_throttle_state(" in live_source


def test_decide_actions_waits_for_pending_extra_cancel() -> None:
    actions, next_order_id = decide_actions(
        working=WorkingOrders(
            buy=None,
            sell=None,
            extras=[ExtraOrder(7, "buy", 1000, req="cancel", cancellable=True)],
        ),
        target_bid_tick=999,
        target_ask_tick=1001,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
    )

    assert actions == []
    assert next_order_id == 10


def test_pending_submit_occupies_working_side() -> None:
    working = merge_pending_orders(
        WorkingOrders(buy=None, sell=None, extras=[]),
        [
            PendingLocalOrder(
                order_id=7,
                side=BUY,
                price=100.0,
                price_tick=1000,
                qty=0.001,
                leaves_qty=0.001,
                local_timestamp=123,
            )
        ],
    )

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
    )

    assert working.buy is not None
    assert int(working.buy.order_id) == 7
    assert [action.side for action in actions] == ["sell"]
    assert next_order_id == 11


def test_pending_submit_does_not_duplicate_engine_order() -> None:
    engine_order = OrderSnapshot(
        order_id=7,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="none",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=True,
    )

    working = merge_pending_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [
            PendingLocalOrder(
                order_id=7,
                side=BUY,
                price=100.0,
                price_tick=1000,
                qty=0.001,
                leaves_qty=0.001,
                local_timestamp=121,
            )
        ],
    )

    assert working.buy is engine_order
    assert working.extras == []


def test_merge_pending_orders_can_replace_engine_view_for_audit_overlay() -> None:
    engine_order = OrderSnapshot(
        order_id=7,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="none",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=True,
    )
    overlay = PendingLocalOrder(
        order_id=7,
        side=BUY,
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=121,
        release_ts=200,
    )

    working = merge_pending_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [overlay],
        replace_existing=True,
    )

    assert working.buy is overlay
    assert working.extras == []


def test_merge_pending_orders_deduplicates_existing_extra_order_id() -> None:
    working = merge_pending_orders(
        WorkingOrders(
            buy=PendingLocalOrder(
                order_id=7,
                side=BUY,
                price=100.0,
                price_tick=1000,
                qty=0.001,
                leaves_qty=0.001,
                local_timestamp=100,
            ),
            sell=None,
            extras=[],
        ),
        [
            PendingLocalOrder(
                order_id=7,
                side=SELL,
                price=100.1,
                price_tick=1001,
                qty=0.001,
                leaves_qty=0.001,
                local_timestamp=100,
            ),
            PendingLocalOrder(
                order_id=8,
                side=SELL,
                price=100.2,
                price_tick=1002,
                qty=0.001,
                leaves_qty=0.001,
                local_timestamp=100,
            ),
        ],
    )

    assert int(working.buy.order_id) == 7
    assert int(working.sell.order_id) == 8
    assert working.extras == []


def test_live_visible_working_orders_masks_unreleased_submit_state() -> None:
    engine_order = OrderSnapshot(
        order_id=7,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="none",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=True,
    )
    overlay = PendingLocalOrder(
        order_id=7,
        side=BUY,
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=121,
        release_ts=200,
    )

    before_release = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [overlay],
        [],
        decision_ts=150,
    )
    after_release = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [overlay],
        [],
        decision_ts=200,
    )

    assert before_release.buy is overlay
    assert before_release.buy.cancellable is False
    assert after_release.buy is not engine_order
    assert after_release.buy.order_id == 7
    assert after_release.buy.req == 0
    assert after_release.buy.cancellable is True


def test_live_visible_working_orders_hides_live_absent_engine_order() -> None:
    engine_order = OrderSnapshot(
        order_id=7,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="cancel",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=False,
    )

    before_absent = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [],
        decision_ts=150,
        absent_after_seen_ts={7: 200},
    )
    after_absent = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [],
        decision_ts=200,
        absent_after_seen_ts={7: 200},
    )

    assert before_absent.buy is engine_order
    assert after_absent.buy is None


def test_live_visible_working_orders_promotes_visible_extra_after_hidden_primary() -> None:
    old_engine_order = OrderSnapshot(
        order_id=7,
        side="buy",
        price=99.0,
        price_tick=990,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="cancel",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=False,
    )
    new_overlay = PendingLocalOrder(
        order_id=8,
        side=BUY,
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=130,
        req=0,
        cancellable=True,
        release_ts=150,
    )

    working = _live_visible_working_orders(
        WorkingOrders(
            buy=old_engine_order,
            sell=None,
            extras=[ExtraOrder(8, "buy", 1000, req="none", cancellable=True)],
        ),
        [new_overlay],
        [],
        decision_ts=200,
        absent_after_seen_ts={7: 190},
    )

    assert working.buy is not None
    assert int(working.buy.order_id) == 8
    assert working.extras == []


def test_live_visible_working_orders_promotes_engine_extra_after_hidden_primary() -> None:
    hidden_primary = OrderSnapshot(
        order_id=7,
        side="buy",
        price=99.0,
        price_tick=990,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="cancel",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=False,
    )
    visible_extra = OrderSnapshot(
        order_id=8,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="none",
        time_in_force="gtx",
        exch_timestamp=124,
        local_timestamp=121,
        cancellable=True,
    )

    working = _live_visible_working_orders(
        WorkingOrders(
            buy=hidden_primary,
            sell=None,
            extras=[ExtraOrder(8, "buy", 1000, req="none", cancellable=True, source_order=visible_extra)],
        ),
        [],
        [],
        decision_ts=200,
        absent_after_seen_ts={7: 190},
    )

    assert working.buy is visible_extra
    assert working.extras == []


def test_live_visible_working_orders_cancel_overlay_switches_at_visible_ts() -> None:
    engine_order = OrderSnapshot(
        order_id=7,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="cancel",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=False,
    )
    overlay = PendingLocalOrder(
        order_id=7,
        side=BUY,
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=121,
        req=0,
        cancellable=True,
        visible_ts=200,
        release_ts=300,
    )

    before_visible = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [overlay],
        decision_ts=150,
    )
    after_visible = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [overlay],
        decision_ts=200,
    )

    assert before_visible.buy.req == 0
    assert before_visible.buy.cancellable is True
    assert after_visible.buy.req == 4
    assert after_visible.buy.cancellable is False


def test_evaluate_live_safety_keeps_open_order_details() -> None:
    state = evaluate_live_safety(
        cfg=LiveSafetyConfig(open_order_mismatch_confirmations=2),
        rest_position=0.001,
        local_position=0.001,
        rest_open_order_count=0,
        local_open_order_count=1,
        rest_error="",
        rest_open_orders="",
        local_open_orders="7:buy:770000:0.001:new:req=none:cxl=1:exch=1:local=1",
        open_order_diff="local_only=7",
        ts_local=10_000_000_000,
        last_api_ts=0,
        open_order_mismatch_count=1,
    )

    assert state.safety_status == "open_order_mismatch"
    assert state.local_open_orders.startswith("7:buy")
    assert state.open_order_diff == "local_only=7"
    assert state.safety_detail == "local_only=7"


def test_build_audit_row_writes_open_order_details() -> None:
    row = build_audit_row(
        run_id="run",
        symbol="BTCUSDT",
        strategy_seq=1,
        ts_local=100,
        ts_exch=90,
        action_order_id="",
        action_name="keep",
        planned_order_id="",
        planned_action="keep",
        throttle_reason="",
        reject_reason="",
        req_ts=0,
        exch_ts=0,
        resp_ts=0,
        entry_latency_ns=0,
        resp_latency_ns=0,
        predicted_entry_ns=0,
        best_bid=100.0,
        best_ask=101.0,
        mid=100.5,
        fair=100.5,
        reservation=100.5,
        half_spread=1.0,
        position=0.0,
        auditlatency_ms=0.0,
        dropped_by_latency=False,
        dropped_by_api_limit=False,
        pos_limit=False,
        impact_cost_val=0.0,
        spread_bps=1.0,
        vol_bps=0.0,
        inventory_score=1.0,
        feed_latency_ns=0,
        latency_signal_ns=0,
        bid_size=1.0,
        ask_size=1.0,
        greek_values=GreekValues(0.0, 0.0, 0.0, 0.0),
        greek_adjustment=0.0,
        target_bid_tick=1000,
        target_ask_tick=1010,
        working_bid_tick=-1,
        working_ask_tick=-1,
        working_buy_order_id="",
        working_sell_order_id="",
        extra_order_ids="",
        extra_order_sides="",
        extra_order_price_ticks="",
        rest_position=0.0,
        position_mismatch=0.0,
        rest_open_order_count=0,
        local_open_order_count=1,
        safety_status="open_order_mismatch",
        local_open_orders="7:buy:1000:0.001:new:req=none:cxl=1:exch=1:local=1",
        rest_open_orders="",
        open_order_diff="local_only=7",
        safety_detail="local_only=7",
    )

    assert row["local_open_orders"].startswith("7:buy")
    assert row["rest_open_orders"] == ""
    assert row["open_order_diff"] == "local_only=7"
    assert row["safety_detail"] == "local_only=7"
    assert row["predicted_entry_ns"] == 0
    assert row["working_bid_qty"] == "0.001"
    assert row["working_ask_qty"] == ""
    assert row["working_bid_status"] == "new"
    assert row["working_bid_req"] == "none"
    assert row["working_bid_pending_cancel"] == "0"


def test_build_audit_row_writes_replay_feed_timestamps() -> None:
    row = build_audit_row(
        run_id="run",
        symbol="BTCUSDT",
        strategy_seq=1,
        ts_local=1_000,
        ts_exch=900,
        action_order_id="",
        action_name="keep",
        planned_order_id="",
        planned_action="keep",
        throttle_reason="",
        reject_reason="",
        req_ts=0,
        exch_ts=0,
        resp_ts=0,
        entry_latency_ns=0,
        resp_latency_ns=0,
        predicted_entry_ns=0,
        best_bid=100.0,
        best_ask=101.0,
        mid=100.5,
        fair=100.5,
        reservation=100.5,
        half_spread=1.0,
        position=0.0,
        auditlatency_ms=0.0,
        dropped_by_latency=False,
        dropped_by_api_limit=False,
        pos_limit=False,
        impact_cost_val=0.0,
        spread_bps=1.0,
        vol_bps=0.0,
        inventory_score=1.0,
        feed_latency_ns=0,
        latency_signal_ns=0,
        bid_size=1.0,
        ask_size=1.0,
        greek_values=GreekValues(0.0, 0.0, 0.0, 0.0),
        greek_adjustment=0.0,
        target_bid_tick=1000,
        target_ask_tick=1010,
        working_bid_tick=-1,
        working_ask_tick=-1,
        working_buy_order_id="",
        working_sell_order_id="",
        extra_order_ids="",
        extra_order_sides="",
        extra_order_price_ticks="",
        replay_scheduled_ts_local=1_000,
        bt_feed_ts_local=1_250,
        bt_feed_ts_exch=900,
        replay_lag_ns=250,
    )

    assert row["replay_scheduled_ts_local"] == 1_000
    assert row["bt_feed_ts_local"] == 1_250
    assert row["bt_feed_ts_exch"] == 900
    assert row["replay_lag_ns"] == 250
    assert row["replay_lag_abs_ns"] == 250


def test_validate_manifest_accepts_ordered_continuous_manifest(tmp_path: Path) -> None:
    snapshot = _touch(tmp_path / "snapshot_before_day1.npz")
    day1 = _touch(tmp_path / "btcusdt_20260101.npz")
    day2 = _touch(tmp_path / "btcusdt_20260102.npz")
    day3 = _touch(tmp_path / "btcusdt_20260103.npz")
    manifest = {
        "start_day": "2026-01-01",
        "end_day": "2026-01-03",
        "initial_snapshot": snapshot,
        "data_files": [day1, day2, day3],
    }

    data_files, initial_snapshot = _validate_manifest_paths(manifest)

    assert data_files == [day1, day2, day3]
    assert initial_snapshot == snapshot


def test_validate_manifest_rejects_empty_data_files() -> None:
    with pytest.raises(ValueError, match="manifest.data_files must contain at least one file"):
        _validate_manifest_paths({"data_files": []})


def test_validate_manifest_rejects_missing_data_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing_day.npz"
    manifest = {"data_files": [str(missing)]}

    with pytest.raises(FileNotFoundError, match="manifest.data_files does not exist"):
        _validate_manifest_paths(manifest)


def test_validate_manifest_rejects_missing_initial_snapshot(tmp_path: Path) -> None:
    data_file = _touch(tmp_path / "btcusdt_20260101.npz")
    missing_snapshot = tmp_path / "missing_snapshot.npz"
    manifest = {
        "initial_snapshot": str(missing_snapshot),
        "data_files": [data_file],
    }

    with pytest.raises(FileNotFoundError, match="manifest.initial_snapshot does not exist"):
        _validate_manifest_paths(manifest)



def test_validate_manifest_rejects_duplicate_data_files(tmp_path: Path) -> None:
    data_file = _touch(tmp_path / "btcusdt_20260101.npz")
    manifest = {"data_files": [data_file, data_file]}

    with pytest.raises(ValueError, match="manifest.data_files contains duplicate path"):
        _validate_manifest_paths(manifest)


def _write_data_npz(path: Path, local_ts_values: list[int]) -> str:
    dtype = np.dtype([
        ("ev", "i8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
    ])
    data = np.zeros(len(local_ts_values), dtype=dtype)
    data["local_ts"] = local_ts_values
    data["exch_ts"] = local_ts_values
    np.savez_compressed(path, data=data)
    return str(path)


def test_select_data_for_asset_full_day_uses_all_files_in_manifest_order(tmp_path: Path) -> None:
    day1 = _write_data_npz(tmp_path / "btcusdt_20260101.npz", [1, 2])
    day2 = _write_data_npz(tmp_path / "btcusdt_20260102.npz", [3, 4])
    day3 = _write_data_npz(tmp_path / "btcusdt_20260103.npz", [5, 6])

    data_for_asset = _select_data_for_asset([day1, day2, day3], "full_day")

    assert data_for_asset == [day1, day2, day3]


def test_select_data_for_asset_windowed_mode_slices_only_first_file(tmp_path: Path) -> None:
    day1 = _write_data_npz(
        tmp_path / "btcusdt_20260101.npz",
        [0, 1_000_000_000, 6 * 60 * 60 * 1_000_000_000 + 1],
    )
    day2 = _write_data_npz(tmp_path / "btcusdt_20260102.npz", [0])

    data_for_asset = _select_data_for_asset([day1, day2], "first_6h")

    assert len(data_for_asset) == 1
    sliced = data_for_asset[0]
    assert isinstance(sliced, np.ndarray)
    assert sliced["local_ts"].tolist() == [0, 1_000_000_000]


def test_strip_local_snapshot_events_removes_local_side_from_snapshots() -> None:
    rows = np.asarray(
        [
            _event(1_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 10.0),
            _event(1_000_000_001, DEPTH_SNAPSHOT_EVENT | BUY_EVENT, 101.0, 1.0),
            _event(1_000_000_002, DEPTH_CLEAR_EVENT | SELL_EVENT, 102.0, 0.0),
            (
                DEPTH_SNAPSHOT_EVENT | BUY_EVENT | LOCAL_EVENT,
                1_000_000_003,
                1_000_000_003,
                99.0,
                2.0,
                0,
                0,
                0.0,
            ),
        ],
        dtype=event_dtype,
    )

    filtered, stats = _strip_local_snapshot_events(rows)

    assert len(filtered) == 3
    assert stats["input_rows"] == 4
    assert stats["output_rows"] == 3
    assert stats["local_flag_removed_snapshot_rows"] == 2
    assert stats["dropped_local_only_snapshot_rows"] == 1
    assert int(filtered[0]["ev"]) & LOCAL_EVENT == LOCAL_EVENT
    assert int(filtered[0]["ev"]) & EXCH_EVENT == EXCH_EVENT
    assert int(filtered[1]["ev"]) & EXCH_EVENT == EXCH_EVENT
    assert int(filtered[1]["ev"]) & LOCAL_EVENT == 0
    assert int(filtered[2]["ev"]) & EXCH_EVENT == EXCH_EVENT
    assert int(filtered[2]["ev"]) & LOCAL_EVENT == 0


def test_live_local_feed_compat_fuses_snapshot_like_connector() -> None:
    rows = np.asarray(
        [
            _event(1_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 1.0),
            _event(1_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 1.0),
            (
                DEPTH_SNAPSHOT_EVENT | BUY_EVENT | LOCAL_EVENT,
                900_000_000,
                1_100_000_000,
                102.0,
                2.0,
                0,
                0,
                0.0,
            ),
            (
                DEPTH_SNAPSHOT_EVENT | SELL_EVENT | LOCAL_EVENT,
                900_000_000,
                1_100_000_000,
                101.1,
                3.0,
                0,
                0,
                0.0,
            ),
            _event(1_200_000_000, DEPTH_EVENT | BUY_EVENT, 100.5, 1.5),
        ],
        dtype=event_dtype,
    )

    fused, stats = _live_local_feed_compat_data(rows, tick_size=0.1, lot_size=0.001)
    local_rows = [row for row in fused if int(row["ev"]) & LOCAL_EVENT == LOCAL_EVENT]

    assert stats["local_input_rows"] == 5
    assert stats["local_output_rows"] == 4
    assert stats["dropped_outdated_depth_rows"] == 1
    assert all(int(row["ev"]) & EXCH_EVENT == 0 for row in local_rows)
    assert any(float(row["px"]) == pytest.approx(101.1) for row in local_rows)
    assert not any(float(row["px"]) == pytest.approx(102.0) for row in local_rows)
    assert all((int(row["ev"]) & 0xff) == DEPTH_EVENT for row in local_rows)


def test_market_data_replay_config_defaults_off() -> None:
    cfg = {"market": {"tick_size": 0.1, "lot_size": 0.001}}
    assert _market_data_replay_config(cfg) == {
        "live_local_feed_compat": False,
        "insert_audit_replay_decision_markers": True,
        "tick_size": 0.1,
        "lot_size": 0.001,
    }
    assert _market_data_replay_config(
        {"market": {"tick_size": 0.1, "lot_size": 0.001}, "market_data_replay": {"live_local_feed_compat": True}}
    ) == {
        "live_local_feed_compat": True,
        "insert_audit_replay_decision_markers": True,
        "tick_size": 0.1,
        "lot_size": 0.001,
    }


def test_insert_audit_replay_decision_markers_sorts_at_live_feed_timestamp() -> None:
    rows = np.asarray(
        [
            _event(1_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 1.0),
            _event(1_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 1.0),
            _event(2_000_000_000, DEPTH_EVENT | SELL_EVENT, 102.0, 2.0),
        ],
        dtype=event_dtype,
    )
    schedule = [AuditReplayScheduleEntry(ts_local=1_500_000_000, ts_exch=1_400_000_000, decision_ts_local=1_600_000_000)]

    marked_items, stats = _insert_audit_replay_decision_markers([rows], schedule)
    marked = marked_items[0]

    assert stats["enabled"] is True
    assert stats["scheduled_count"] == 1
    assert stats["inserted_count"] == 1
    assert stats["duplicate_timestamp_count"] == 0
    assert marked["local_ts"].tolist() == [
        1_000_000_000,
        1_000_000_000,
        1_500_000_000,
        2_000_000_000,
    ]
    marker = marked[2]
    assert int(marker["ev"]) == AUDIT_REPLAY_DECISION_MARKER_EVENT
    assert int(marker["exch_ts"]) == 1_400_000_000
    assert float(marker["px"]) == pytest.approx(0.0)
    assert float(marker["qty"]) == pytest.approx(0.0)


def test_insert_audit_replay_decision_markers_skips_existing_local_timestamp() -> None:
    rows = np.asarray(
        [
            _event(1_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 1.0),
            _event(2_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 1.0),
        ],
        dtype=event_dtype,
    )
    schedule = [AuditReplayScheduleEntry(ts_local=1_000_000_000, ts_exch=1_000_000_000, decision_ts_local=1_000_000_010)]

    marked_items, stats = _insert_audit_replay_decision_markers([rows], schedule)

    assert len(marked_items[0]) == 2
    assert stats["inserted_count"] == 0
    assert stats["duplicate_timestamp_count"] == 1


def test_slice_data_by_absolute_local_ts_is_inclusive() -> None:
    dtype = np.dtype([
        ("ev", "i8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
    ])
    data = np.zeros(4, dtype=dtype)
    data["local_ts"] = [100, 200, 300, 400]

    sliced = _slice_data_by_absolute_local_ts(data, 200, 300)

    assert sliced["local_ts"].tolist() == [200, 300]


def test_slice_data_by_absolute_local_ts_rejects_start_after_end() -> None:
    dtype = np.dtype([
        ("ev", "i8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
    ])
    data = np.zeros(1, dtype=dtype)
    data["local_ts"] = [100]

    with pytest.raises(ValueError, match="slice_ts_local_start must be <= slice_ts_local_end"):
        _slice_data_by_absolute_local_ts(data, 300, 200)


def test_select_data_for_asset_absolute_slice_rejects_multi_file_manifest(tmp_path: Path) -> None:
    day1 = _write_data_npz(tmp_path / "btcusdt_20260101.npz", [100, 200])
    day2 = _write_data_npz(tmp_path / "btcusdt_20260102.npz", [300, 400])

    with pytest.raises(ValueError, match="absolute ts_local slicing currently supports exactly one data file"):
        _select_data_for_asset([day1, day2], "full_day", slice_ts_local_start=100, slice_ts_local_end=400)


def test_select_data_for_asset_absolute_slice_rejects_named_relative_window(tmp_path: Path) -> None:
    day1 = _write_data_npz(tmp_path / "btcusdt_20260101.npz", [100, 200, 300])

    with pytest.raises(ValueError, match="absolute ts_local slicing requires window='full_day'"):
        _select_data_for_asset([day1], "first_5m", slice_ts_local_start=100, slice_ts_local_end=200)


def test_select_data_for_asset_absolute_slice_rejects_empty_result(tmp_path: Path) -> None:
    day1 = _write_data_npz(tmp_path / "btcusdt_20260101.npz", [100, 200, 300])

    with pytest.raises(ValueError, match="absolute ts_local slice selected zero rows"):
        _select_data_for_asset([day1], "full_day", slice_ts_local_start=500, slice_ts_local_end=600)


def test_audit_replay_decision_due_waits_until_next_schedule() -> None:
    due, idx, lag, skipped, breach, consumed_ts = _audit_replay_decision_due(99, [100, 200], 0, 0)
    assert (due, idx, lag, skipped, breach, consumed_ts) == (False, 0, 0, 0, False, 0)

    due, idx, lag, skipped, breach, consumed_ts = _audit_replay_decision_due(100, [100, 200], 0, 0)
    assert (due, idx, lag, skipped, breach, consumed_ts) == (True, 1, 0, 0, False, 100)


def test_audit_replay_decision_due_uses_tolerance() -> None:
    due, idx, lag, skipped, breach, consumed_ts = _audit_replay_decision_due(98, [100], 0, 2)

    assert (due, idx, lag, skipped, breach, consumed_ts) == (True, 1, -2, 0, False, 100)


def test_audit_replay_decision_due_consumes_one_schedule_per_feed_event() -> None:
    due, idx, lag, skipped, breach, consumed_ts = _audit_replay_decision_due(250, [100, 200, 300], 0, 0)

    assert (due, idx, lag, skipped, breach, consumed_ts) == (True, 1, 150, 0, False, 100)


def test_audit_replay_decision_due_drain_due_consumes_backlog() -> None:
    due, idx, lag, skipped, breach, consumed_ts = _audit_replay_decision_due(
        250,
        [100, 200, 300],
        0,
        0,
        replay_mode="drain_due",
        max_lag_ns=40,
    )

    assert (due, idx, lag, skipped, breach, consumed_ts) == (True, 2, 50, 1, True, 200)


def test_empty_replay_lag_gate_stats_marks_strict_gate_only_when_enabled() -> None:
    disabled = _empty_replay_lag_gate_stats(max_lag_ns=0, strict=True)
    enabled = _empty_replay_lag_gate_stats(max_lag_ns=250_000_000, strict=True)
    exchange_enabled = _empty_replay_lag_gate_stats(max_lag_ns=0, max_exch_lag_ns=250_000_000, strict=True)

    assert disabled["enabled"] is False
    assert disabled["strict"] is False
    assert disabled["action"] == "report"
    assert enabled["enabled"] is True
    assert enabled["strict"] is True
    assert enabled["action"] == "report"
    assert enabled["max_lag_ns"] == 250_000_000
    assert exchange_enabled["enabled"] is True
    assert exchange_enabled["strict"] is True
    assert exchange_enabled["max_exch_lag_ns"] == 250_000_000


def test_load_audit_cadence_schedule_skips_fractional_nanoseconds(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,ts_local,action\nlive,100.5,keep\nlive,200.0,keep\n")

    schedule = _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")

    assert schedule == [200]


def test_load_audit_cadence_schedule_filters_run_id_and_dedupes(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "run_id,ts_local,action\n"
        "other,100,keep\n"
        "live,300,keep\n"
        "live,100,keep\n"
        "live,300,keep\n"
        "live,200,submit_buy\n"
    )

    schedule = _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")

    assert schedule == [100, 200, 300]


def test_load_audit_cadence_schedule_filters_non_decision_event_rows(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "run_id,event_type,ts_local,action\n"
        "live,decision,100,keep\n"
        "live,order_submit_sent,100,submit_buy\n"
        "live,cancel_sent,101,cancel_buy\n"
        "live,decision,100,keep\n"
        "live,,200,keep\n"
        "live,0,300,keep\n"
        "other,decision,400,keep\n"
    )

    schedule, stats = _load_audit_cadence_schedule_with_stats(audit, run_id="live", ts_column="ts_local")

    assert schedule == [100, 200, 300]
    assert stats["raw_row_count"] == 7
    assert stats["run_id_match_row_count"] == 6
    assert stats["decision_row_count"] == 4
    assert stats["ignored_non_decision_row_count"] == 2
    assert stats["valid_decision_timestamp_count"] == 4
    assert stats["deduped_decision_timestamp_count"] == 1
    assert stats["unique_schedule_count"] == 3
    assert stats["has_event_type_column"] is True


def test_load_audit_replay_schedule_preserves_live_exchange_timestamps(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "run_id,event_type,ts_local,ts_exch,action\n"
        "live,decision,100,90,keep\n"
        "live,order_submit_sent,101,91,submit_buy\n"
        "live,decision,200,190,keep\n"
    )

    schedule, stats = _load_audit_replay_schedule_with_stats(audit, run_id="live", ts_column="ts_local")

    assert schedule == [
        AuditReplayScheduleEntry(ts_local=100, ts_exch=90, decision_ts_local=100),
        AuditReplayScheduleEntry(ts_local=200, ts_exch=190, decision_ts_local=200),
    ]
    assert stats["has_ts_exch_column"] is True
    assert stats["valid_ts_exch_timestamp_count"] == 2
    assert stats["ignored_non_decision_row_count"] == 1


def test_load_audit_replay_schedule_uses_live_feed_local_timestamps(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "run_id,event_type,ts_local,ts_exch,feed_latency_ns,action\n"
        "live,decision,1000,900,25,keep\n"
        "live,decision,2000,1900,40,keep\n"
    )

    schedule, stats = _load_audit_replay_schedule_with_stats(audit, run_id="live", ts_column="ts_local")

    assert schedule == [
        AuditReplayScheduleEntry(ts_local=925, ts_exch=900, decision_ts_local=1000),
        AuditReplayScheduleEntry(ts_local=1940, ts_exch=1900, decision_ts_local=2000),
    ]
    assert stats["feed_ts_local_source"] == "ts_exch_plus_feed_latency_ns"
    assert stats["derived_feed_ts_local_count"] == 2
    assert stats["first_decision_ts_local"] == 1000
    assert stats["first_feed_ts_local"] == 925


def test_load_live_strategy_position_by_decision_ts_reads_decision_rows(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "run_id,event_type,ts_local,position\n"
        "live,decision,100,0.001\n"
        "live,fill,101,0.002\n"
        "other,decision,102,0.003\n"
        "live,decision,103,-0.001\n"
    )

    positions = _load_live_strategy_position_by_decision_ts(audit, run_id="live")

    assert positions == {100: 0.001, 103: -0.001}


def test_load_live_market_state_by_decision_ts_reads_decision_rows(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "run_id,event_type,ts_local,best_bid,best_ask,mid,bid_size,ask_size,"
        "fair,reservation,half_spread,target_bid_tick,target_ask_tick\n"
        "other,decision,99,1,2,1.5,3,4,5,6,7,8,9\n"
        "live,order_submit_sent,100,1,2,1.5,3,4,5,6,7,8,9\n"
        "live,decision,100,100.0,101.0,100.5,1.2,3.4,99.9,99.8,0.7,998,1012\n"
        "live,decision,101,100.0,99.0,99.5,1.2,3.4,99.9,99.8,0.7,998,1012\n"
    )

    states = _load_live_market_state_by_decision_ts(audit, run_id="live")

    assert list(states) == [100]
    state = states[100]
    assert state.best_bid == pytest.approx(100.0)
    assert state.best_ask == pytest.approx(101.0)
    assert state.bid_size == pytest.approx(1.2)
    assert state.ask_size == pytest.approx(3.4)
    assert state.target_bid_tick == 998
    assert state.target_ask_tick == 1012


def test_load_audit_cadence_schedule_keeps_legacy_csv_without_event_type(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "run_id,ts_local,action\n"
        "live,100,keep\n"
        "live,200,order_submit_sent\n"
    )

    schedule, stats = _load_audit_cadence_schedule_with_stats(audit, run_id="live", ts_column="ts_local")

    assert schedule == [100, 200]
    assert stats["decision_row_count"] == 2
    assert stats["ignored_non_decision_row_count"] == 0
    assert stats["has_event_type_column"] is False


def test_audit_replay_schedule_decision_due_checks_local_and_exchange_lag() -> None:
    schedule = [AuditReplayScheduleEntry(ts_local=100, ts_exch=1_000, decision_ts_local=500)]

    result = _audit_replay_schedule_decision_due(
        ts_local=120,
        bt_feed_ts_exch=1_400,
        schedule=schedule,
        schedule_idx=0,
        tolerance_ns=0,
        max_lag_ns=50,
        max_exch_lag_ns=250,
    )

    (
        due,
        idx,
        local_lag,
        exch_lag,
        skipped,
        local_breach,
        exch_breach,
        consumed_decision,
        consumed_feed,
        consumed_exch,
    ) = result
    assert due is True
    assert idx == 1
    assert local_lag == 20
    assert exch_lag == 400
    assert skipped == 0
    assert local_breach is False
    assert exch_breach is True
    assert consumed_decision == 500
    assert consumed_feed == 100
    assert consumed_exch == 1_000


def test_load_audit_cadence_schedule_raises_for_empty_filter(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,ts_local,action\nother,100,keep\n")

    with pytest.raises(ValueError, match="no cadence timestamps loaded"):
        _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")


def test_load_audit_cadence_schedule_raises_for_missing_column(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,wrong_ts\nlive,100\n")

    with pytest.raises(KeyError, match="missing cadence timestamp column"):
        _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")


def test_feed_latency_oracle_uses_latest_live_audit_latency(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "run_id,ts_local,feed_latency_ns\n"
        "other,100,900\n"
        "live,100,3000000\n"
        "live,200,7000000\n"
    )

    oracle = FeedLatencyOracle.from_audit_csv(audit, run_id="live")

    assert oracle.feed_latency_ns(50, fallback=123) == 3_000_000
    assert oracle.feed_latency_ns(150, fallback=123) == 3_000_000
    assert oracle.feed_latency_ns(250, fallback=123) == 7_000_000


def test_feed_latency_oracle_disabled_returns_raw_fallback() -> None:
    oracle = FeedLatencyOracle.disabled()

    assert oracle.feed_latency_ns(100, fallback=6_000_000) == 6_000_000


def test_feed_latency_oracle_requires_latency_rows(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,ts_local,feed_latency_ns\nother,100,900\n")

    with pytest.raises(ValueError, match="no feed latency rows loaded"):
        FeedLatencyOracle.from_audit_csv(audit, run_id="live")


def test_backtest_cadence_config_defaults_to_fixed_interval() -> None:
    cfg = _backtest_cadence_config({})

    assert cfg == {
        "mode": "fixed_interval",
        "min_interval_ns": 0,
        "audit_csv": "",
        "run_id": "",
        "ts_column": "ts_local",
        "tolerance_ns": 0,
        "replay_mode": "single",
        "max_lag_ns": 0,
        "max_exch_lag_ns": 0,
        "strict_lag_gate": False,
        "lag_gate_action": "report",
        "trigger_ts_source": "ts_local",
        "feed_latency_column": "feed_latency_ns",
        "market_state_overlay": "off",
    }


def test_backtest_cadence_config_reads_audit_replay() -> None:
    config = {
        "backtest_cadence": {
            "mode": "audit_replay",
            "audit_csv": "/tmp/live.csv",
            "run_id": "live_run",
            "ts_column": "ts_local",
            "tolerance_ms": 2.5,
            "replay_mode": "drain_due",
            "max_lag_ms": 250.0,
        }
    }

    cfg = _backtest_cadence_config(config)

    assert cfg["mode"] == "audit_replay"
    assert cfg["audit_csv"] == "/tmp/live.csv"
    assert cfg["run_id"] == "live_run"
    assert cfg["ts_column"] == "ts_local"
    assert cfg["tolerance_ns"] == 2_500_000
    assert cfg["replay_mode"] == "drain_due"
    assert cfg["max_lag_ns"] == 250_000_000
    assert cfg["max_exch_lag_ns"] == 250_000_000
    assert cfg["strict_lag_gate"] is True
    assert cfg["lag_gate_action"] == "fail"
    assert cfg["trigger_ts_source"] == "feed_local"
    assert cfg["feed_latency_column"] == "feed_latency_ns"
    assert cfg["market_state_overlay"] == "off"


def test_backtest_cadence_config_reads_distinct_exchange_lag_gate() -> None:
    cfg = _backtest_cadence_config(
        {
            "backtest_cadence": {
                "mode": "audit_replay",
                "max_lag_ms": 250.0,
                "max_exch_lag_ms": 1000.0,
            }
        }
    )

    assert cfg["max_lag_ns"] == 250_000_000
    assert cfg["max_exch_lag_ns"] == 1_000_000_000
    assert cfg["strict_lag_gate"] is True


def test_backtest_cadence_config_accepts_report_lag_gate() -> None:
    cfg = _backtest_cadence_config(
        {
            "backtest_cadence": {
                "mode": "audit_replay",
                "max_lag_ms": 250.0,
                "strict_lag_gate": False,
                "lag_gate_action": "report",
            }
        }
    )

    assert cfg["max_lag_ns"] == 250_000_000
    assert cfg["strict_lag_gate"] is False
    assert cfg["lag_gate_action"] == "report"


def test_backtest_cadence_config_accepts_emit_due_replay_mode() -> None:
    cfg = _backtest_cadence_config(
        {
            "backtest_cadence": {
                "mode": "audit_replay",
                "replay_mode": "emit_due",
            }
        }
    )

    assert cfg["replay_mode"] == "emit_due"


def test_backtest_cadence_config_accepts_audit_market_state_overlay() -> None:
    cfg = _backtest_cadence_config(
        {
            "backtest_cadence": {
                "mode": "audit_replay",
                "market_state_overlay": "audit",
            }
        }
    )

    assert cfg["market_state_overlay"] == "audit"


def test_backtest_cadence_config_rejects_unknown_market_state_overlay() -> None:
    with pytest.raises(ValueError, match="Unsupported backtest_cadence.market_state_overlay"):
        _backtest_cadence_config(
            {
                "backtest_cadence": {
                    "mode": "audit_replay",
                    "market_state_overlay": "unknown",
                }
            }
        )


def test_backtest_cadence_config_rejects_unknown_replay_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported audit replay_mode"):
        _backtest_cadence_config(
            {
                "backtest_cadence": {
                    "mode": "audit_replay",
                    "replay_mode": "unknown",
                }
            }
        )


def test_backtest_cadence_config_rejects_unknown_lag_gate_action() -> None:
    with pytest.raises(ValueError, match="Unsupported backtest_cadence.lag_gate_action"):
        _backtest_cadence_config(
            {
                "backtest_cadence": {
                    "mode": "audit_replay",
                    "lag_gate_action": "unknown",
                }
            }
        )


def test_backtest_cadence_config_keeps_legacy_enabled_fixed_interval() -> None:
    config = {"backtest_cadence": {"enabled": True, "min_decision_interval_ms": 5.0}}

    cfg = _backtest_cadence_config(config)

    assert cfg["mode"] == "fixed_interval"
    assert cfg["min_interval_ns"] == 5_000_000




def test_backtest_cadence_interval_from_config_reads_ms() -> None:
    config = {"backtest_cadence": {"enabled": True, "min_decision_interval_ms": 12.5}}

    assert _backtest_cadence_interval_ns(config) == 12_500_000


def test_backtest_cadence_interval_from_config_disabled_returns_zero() -> None:
    config = {"backtest_cadence": {"enabled": False, "min_decision_interval_ms": 12.5}}

    assert _backtest_cadence_interval_ns(config) == 0


def test_should_skip_strategy_decision_is_disabled_for_zero_interval() -> None:
    assert _should_skip_strategy_decision(200, 100, 0) is False


def test_should_skip_strategy_decision_allows_first_decision() -> None:
    assert _should_skip_strategy_decision(100, None, 50) is False


def test_should_skip_strategy_decision_skips_until_interval_elapsed() -> None:
    assert _should_skip_strategy_decision(149, 100, 50) is True
    assert _should_skip_strategy_decision(150, 100, 50) is False


def test_latency_guard_signal_uses_feed_latency_only() -> None:
    assert _latency_guard_signal_ns(3_000_000) == 3_000_000


def test_latency_guard_signal_does_not_use_predicted_entry_latency() -> None:
    predicted_entry_ns = 20_000_000
    assert _latency_guard_signal_ns(3_000_000) != predicted_entry_ns


def test_live_does_not_reject_after_send_with_observed_entry_latency() -> None:
    live_source = Path(__file__).with_name("live_tick_mm.py").read_text()

    assert "sent_api and entry_latency_ns > latency_guard_ns" not in live_source


def test_apply_initial_snapshot_calls_asset_once_when_snapshot_is_present(tmp_path: Path) -> None:
    snapshot = str(tmp_path / "snapshot_before_day1.npz")
    asset = FakeAsset()

    _apply_initial_snapshot(asset, snapshot)

    assert asset.snapshots == [snapshot]


def test_apply_initial_snapshot_does_not_call_asset_when_snapshot_is_absent() -> None:
    asset = FakeAsset()

    _apply_initial_snapshot(asset, None)

    assert asset.snapshots == []


def test_alignment_init_config_defaults_to_disabled() -> None:
    cfg = _alignment_init_config({})

    assert cfg["enabled"] is False
    assert cfg["position_mode"] == "off"
    assert cfg["position"] == 0.0


def test_alignment_init_config_reads_synthetic_fill() -> None:
    cfg = _alignment_init_config(
        {
            "alignment_init": {
                "enabled": True,
                "position_mode": "synthetic_fill",
                "position": "-0.001",
                "source": "position",
                "ts_local": "100",
                "order_id_start": "900",
            }
        }
    )

    assert cfg == {
        "enabled": True,
        "position_mode": "synthetic_fill",
        "position": -0.001,
        "source": "position",
        "ts_local": 100,
        "order_id_start": 900,
    }


def test_alignment_init_config_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported alignment_init.position_mode"):
        _alignment_init_config({"alignment_init": {"enabled": True, "position_mode": "direct"}})


def test_round_position_qty_rounds_to_lot_size() -> None:
    assert _round_position_qty(0.0014, 0.001) == pytest.approx(0.001)
    assert _round_position_qty(-0.0016, 0.001) == pytest.approx(0.002)
    assert _round_position_qty(0.0004, 0.001) == pytest.approx(0.0)


def _event(ts: int, ev: int, px: float, qty: float) -> tuple[int, int, int, float, float, int, int, float]:
    return (ev | EXCH_EVENT | LOCAL_EVENT, ts, ts, px, qty, 0, 0, 0.0)


def _simple_market_data() -> np.ndarray:
    rows = [
        _event(1_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 10.0),
        _event(1_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 10.0),
        _event(2_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 10.0),
        _event(2_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 10.0),
        _event(3_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 10.0),
        _event(3_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 10.0),
    ]
    return np.asarray(rows, dtype=event_dtype)


def _make_hbt_from_data(data: np.ndarray) -> ROIVectorMarketDepthBacktest:
    asset = (
        BacktestAsset()
        .linear_asset(1.0)
        .data(data)
        .no_partial_fill_exchange()
        .constant_order_latency(0, 0)
        .power_prob_queue_model3(3.0)
        .tick_size(0.1)
        .lot_size(0.001)
        .roi_lb(90.0)
        .roi_ub(110.0)
    )
    return ROIVectorMarketDepthBacktest([asset])


def _make_simple_hbt() -> ROIVectorMarketDepthBacktest:
    return _make_hbt_from_data(_simple_market_data())


def _load_first_valid_feed(hbt: ROIVectorMarketDepthBacktest) -> tuple[float, float]:
    while True:
        rc = hbt.wait_next_feed(True, 1_000_000_000)
        assert rc != 1
        depth = hbt.depth(0)
        best_bid = float(depth.best_bid)
        best_ask = float(depth.best_ask)
        if best_bid > 0 and best_ask > best_bid:
            return best_bid, best_ask


def test_apply_alignment_initial_position_crosses_buy_to_target_position() -> None:
    hbt = _make_simple_hbt()
    try:
        best_bid, best_ask = _load_first_valid_feed(hbt)

        meta = _apply_alignment_initial_position(
            hbt,
            0,
            target_position=0.001,
            order_id=900,
            best_bid=best_bid,
            best_ask=best_ask,
            lot_size=0.001,
            source="position",
            source_ts_local=100,
        )

        assert meta["success"] is True
        assert meta["side"] == "buy"
        assert meta["price"] == pytest.approx(best_ask)
        assert hbt.position(0) == pytest.approx(0.001)
    finally:
        hbt.close()


def test_audit_replay_decision_marker_advances_feed_clock_without_depth_change() -> None:
    rows = np.asarray(
        [
            _event(1_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 10.0),
            _event(1_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 10.0),
            (
                AUDIT_REPLAY_DECISION_MARKER_EVENT,
                1_400_000_000,
                1_500_000_000,
                0.0,
                0.0,
                0,
                0,
                0.0,
            ),
            _event(2_000_000_000, DEPTH_EVENT | SELL_EVENT, 100.5, 10.0),
            _event(3_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 10.0),
        ],
        dtype=event_dtype,
    )
    hbt = _make_hbt_from_data(rows)
    try:
        best_bid, best_ask = _load_first_valid_feed(hbt)
        assert best_bid == pytest.approx(100.0)
        assert best_ask == pytest.approx(101.0)

        rc = hbt.wait_next_feed(True, 1_000_000_000)
        assert rc != 1
        depth = hbt.depth(0)
        feed_latency = hbt.feed_latency(0)
        assert int(hbt.current_timestamp) == 1_500_000_000
        assert feed_latency == (1_400_000_000, 1_500_000_000)
        assert float(depth.best_bid) == pytest.approx(100.0)
        assert float(depth.best_ask) == pytest.approx(101.0)

        rc = hbt.wait_next_feed(True, 1_000_000_000)
        assert rc != 1
        assert float(hbt.depth(0).best_ask) == pytest.approx(100.5)
    finally:
        hbt.close()


def test_apply_alignment_initial_position_crosses_sell_to_target_position() -> None:
    hbt = _make_simple_hbt()
    try:
        best_bid, best_ask = _load_first_valid_feed(hbt)

        meta = _apply_alignment_initial_position(
            hbt,
            0,
            target_position=-0.001,
            order_id=901,
            best_bid=best_bid,
            best_ask=best_ask,
            lot_size=0.001,
            source="position",
            source_ts_local=100,
        )

        assert meta["success"] is True
        assert meta["side"] == "sell"
        assert meta["price"] == pytest.approx(best_bid)
        assert hbt.position(0) == pytest.approx(-0.001)
    finally:
        hbt.close()


def test_continuous_run_metadata_marks_multi_file_full_day_as_continuous() -> None:
    metadata = _continuous_run_metadata(
        window="full_day",
        data_files=["day1.npz", "day2.npz"],
        initial_snapshot="snapshot_before_day1.npz",
    )

    assert metadata == {
        "continuous_run": True,
        "initial_snapshot": "snapshot_before_day1.npz",
        "data_file_count": 2,
        "data_files": ["day1.npz", "day2.npz"],
    }


def test_continuous_run_metadata_marks_single_file_full_day_as_not_continuous() -> None:
    metadata = _continuous_run_metadata(
        window="full_day",
        data_files=["day1.npz"],
        initial_snapshot=None,
    )

    assert metadata["continuous_run"] is False
    assert metadata["data_file_count"] == 1
    assert metadata["initial_snapshot"] is None


def test_continuous_run_metadata_marks_windowed_multi_file_as_not_continuous() -> None:
    metadata = _continuous_run_metadata(
        window="first_6h",
        data_files=["day1.npz", "day2.npz"],
        initial_snapshot="snapshot_before_day1.npz",
    )

    assert metadata["continuous_run"] is False
    assert metadata["data_file_count"] == 2
