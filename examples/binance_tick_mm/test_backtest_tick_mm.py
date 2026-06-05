from __future__ import annotations

import csv
import io
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
    CANCELED,
    FILLED,
    NEW,
    SELL_EVENT,
    SELL,
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    event_dtype,
)
from hftbacktest.order import PARTIALLY_FILLED, REJECTED

from audit_schema import AUDIT_FIELDS
from backtest_tick_mm import (
    AUDIT_REPLAY_DECISION_MARKER_EVENT,
    AuditReplayScheduleEntry,
    CompactLifecycleAuditWriter,
    FeedLatencyOracle,
    LiveShortCancelRaceFillConstraint,
    LiveTerminalLifecycleConstraint,
    _LiveLocalFeedFuser,
    _apply_live_short_cancel_race_fill_constraint,
    _apply_live_terminal_lifecycle_constraint,
    _collect_forced_live_terminal_events,
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
    _load_live_decision_rows_by_ts,
    _load_live_lifecycle_events_by_decision_ts,
    _load_live_short_cancel_race_fill_constraints_by_order_id,
    _load_live_terminal_constraints_by_order_id,
    _load_live_order_absent_after_seen_ts,
    _pending_order_from_live_token,
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
    _sync_live_state_visibility_overlays,
    _update_terminal_live_visibility_overlays,
    _validate_manifest_paths,
    _working_orders_from_live_state,
    _apply_live_inflight_replay_after_decision,
)
from live_tick_mm import (
    SHUTDOWN_CANCEL_ACK_TIMEOUT_NS,
    SHUTDOWN_TERMINAL_CONFIRMATION_SOURCES,
    SHUTDOWN_WAIT_OUTCOMES,
    cancel_working_orders_for_shutdown,
)
from strategy_core import (
    Action,
    ExtraOrder,
    GreekValues,
    InFlightExposureTracker,
    LiveSafetyConfig,
    OrderLifecycleTracker,
    OrderSnapshot,
    PendingLocalOrder,
    TokenBucket,
    QuoteThrottleState,
    WorkingOrders,
    QuoteThrottleConfig,
    add_side_toxic_timing_guard_side_blocks,
    adverse_timing_guard_side_blocks,
    add_side_soft_limit_qty_from_risk,
    build_market_view_from_depth,
    build_audit_row,
    build_quote_update_audit_fields,
    build_lifecycle_event_row,
    cancel_race_guard_side_blocks,
    decide_actions,
    evaluate_live_safety,
    format_rest_open_orders,
    format_top5_levels,
    is_pure_cancel_extra,
    merge_pending_orders,
    open_order_diff,
    quote_throttle_reason,
    update_quote_throttle_state,
)



class _ShutdownFakeOrder:
    def __init__(
        self,
        order_id: int,
        *,
        cancellable: bool = True,
        side: int = BUY,
        status: int = NEW,
    ) -> None:
        self.order_id = order_id
        self.cancellable = cancellable
        self.side = side
        self.status = status
        self.price_tick = 1000
        self.req = 1


class _ShutdownFakeOrderValues:
    def __init__(self, orders: list[_ShutdownFakeOrder]) -> None:
        self.orders = orders
        self.index = 0

    def has_next(self) -> bool:
        return self.index < len(self.orders)

    def get(self) -> _ShutdownFakeOrder:
        order = self.orders[self.index]
        self.index += 1
        return order


class _ShutdownFakeOrderDict:
    def __init__(self, orders: list[_ShutdownFakeOrder]) -> None:
        self.orders = orders

    def values(self) -> _ShutdownFakeOrderValues:
        return _ShutdownFakeOrderValues(self.orders)


class _ShutdownFakeHbt:
    def __init__(
        self,
        *,
        cancel_fail_order_ids: set[int] | None = None,
        wait_fail_order_ids: set[int] | None = None,
        wait_results_by_order_id: dict[int, int] | None = None,
        final_active_order_ids: set[int] | None = None,
        final_order_statuses_by_order_id: dict[int, int] | None = None,
        orders_fail: bool = False,
    ) -> None:
        self.cancel_fail_order_ids = cancel_fail_order_ids or set()
        self.wait_fail_order_ids = wait_fail_order_ids or set()
        self.wait_results_by_order_id = wait_results_by_order_id or {}
        self.final_active_order_ids = final_active_order_ids or set()
        self.final_order_statuses_by_order_id = final_order_statuses_by_order_id or {}
        self.orders_fail = orders_fail
        self.events: list[tuple[object, ...]] = []
        self.status_checks: list[tuple[object, ...]] = []

    def cancel(self, asset_no: int, order_id: int, wait: bool) -> None:
        self.events.append(("cancel", asset_no, order_id, wait))
        if order_id in self.cancel_fail_order_ids:
            raise RuntimeError(f"cancel failed {order_id}")

    def wait_order_response(self, asset_no: int, order_id: int, timeout_ns: int) -> int:
        self.events.append(("wait", asset_no, order_id, timeout_ns))
        if order_id in self.wait_fail_order_ids:
            raise RuntimeError(f"wait failed {order_id}")
        return self.wait_results_by_order_id.get(order_id, 0)

    def orders(self, asset_no: int) -> _ShutdownFakeOrderDict:
        self.status_checks.append(("orders", asset_no))
        if self.orders_fail:
            raise RuntimeError("orders unavailable")
        statuses = {
            order_id: NEW for order_id in self.final_active_order_ids
        } | self.final_order_statuses_by_order_id
        return _ShutdownFakeOrderDict(
            [
                _ShutdownFakeOrder(order_id, status=status)
                for order_id, status in sorted(statuses.items())
            ]
        )

    def close(self) -> None:
        self.events.append(("close",))


def _shutdown_working_orders(
    *,
    buy: _ShutdownFakeOrder | None = None,
    sell: _ShutdownFakeOrder | None = None,
    extras: list[ExtraOrder] | None = None,
) -> WorkingOrders:
    return WorkingOrders(buy=buy, sell=sell, extras=extras or [])


def _expected_shutdown_cancel_wait_events(order_ids: list[int]) -> list[tuple[object, ...]]:
    events: list[tuple[object, ...]] = []
    for order_id in order_ids:
        events.append(("cancel", 0, order_id, False))
        events.append(("wait", 0, order_id, SHUTDOWN_CANCEL_ACK_TIMEOUT_NS))
    return events


@pytest.mark.parametrize(
    ("case_name", "working", "expected_order_ids"),
    [
        (
            "buy_only_cancellable",
            _shutdown_working_orders(buy=_ShutdownFakeOrder(101)),
            [101],
        ),
        (
            "sell_only_cancellable",
            _shutdown_working_orders(sell=_ShutdownFakeOrder(201)),
            [201],
        ),
        (
            "buy_sell_cancellable",
            _shutdown_working_orders(
                buy=_ShutdownFakeOrder(101),
                sell=_ShutdownFakeOrder(201),
            ),
            [101, 201],
        ),
        (
            "extra_buy_exists",
            _shutdown_working_orders(extras=[ExtraOrder(301, "buy", 1000)]),
            [301],
        ),
        (
            "mixed_buy_sell_extras",
            _shutdown_working_orders(
                buy=_ShutdownFakeOrder(101),
                sell=_ShutdownFakeOrder(201),
                extras=[
                    ExtraOrder(301, "buy", 1000),
                    ExtraOrder(401, "sell", 1002),
                ],
            ),
            [101, 201, 301, 401],
        ),
    ],
)
def test_live_shutdown_cancel_waits_for_order_response_for_cancellable_orders(
    case_name: str,
    working: WorkingOrders,
    expected_order_ids: list[int],
) -> None:
    hbt = _ShutdownFakeHbt()

    results = cancel_working_orders_for_shutdown(hbt, working)

    assert [result.order_id for result in results] == expected_order_ids, case_name
    assert hbt.events == _expected_shutdown_cancel_wait_events(expected_order_ids)
    assert all(result.cancel_sent for result in results)
    assert all(result.wait_requested for result in results)
    assert all(result.wait_result == 0 for result in results)
    assert all(result.error == "" for result in results)


def test_live_shutdown_does_not_cancel_non_cancellable_primary_orders() -> None:
    hbt = _ShutdownFakeHbt()
    working = _shutdown_working_orders(
        buy=_ShutdownFakeOrder(101, cancellable=False),
        sell=_ShutdownFakeOrder(201, cancellable=False),
    )

    results = cancel_working_orders_for_shutdown(hbt, working)

    assert results == []
    assert hbt.events == []


def test_live_shutdown_does_not_cancel_non_cancellable_or_cancel_pending_extra_orders() -> None:
    hbt = _ShutdownFakeHbt()
    working = _shutdown_working_orders(
        extras=[
            ExtraOrder(301, "buy", 1000, cancellable=False),
            ExtraOrder(302, "buy", 1001, req="cancel", cancellable=True),
            ExtraOrder(303, "sell", 1002, req="none", cancellable=True),
        ]
    )

    results = cancel_working_orders_for_shutdown(hbt, working)

    assert [result.order_id for result in results] == [303]
    assert hbt.events == _expected_shutdown_cancel_wait_events([303])


def test_live_shutdown_cancel_exception_does_not_stop_remaining_orders() -> None:
    hbt = _ShutdownFakeHbt(cancel_fail_order_ids={101})
    working = _shutdown_working_orders(
        buy=_ShutdownFakeOrder(101),
        sell=_ShutdownFakeOrder(201),
        extras=[ExtraOrder(301, "buy", 1000)],
    )

    results = cancel_working_orders_for_shutdown(hbt, working)

    assert [result.order_id for result in results] == [101, 201, 301]
    assert hbt.events == [
        ("cancel", 0, 101, False),
        ("cancel", 0, 201, False),
        ("wait", 0, 201, SHUTDOWN_CANCEL_ACK_TIMEOUT_NS),
        ("cancel", 0, 301, False),
        ("wait", 0, 301, SHUTDOWN_CANCEL_ACK_TIMEOUT_NS),
    ]
    assert results[0].cancel_sent is False
    assert results[0].wait_requested is False
    assert results[0].error.startswith("cancel_error:RuntimeError:")
    assert results[1].cancel_sent is True
    assert results[1].wait_requested is True
    assert results[2].cancel_sent is True
    assert results[2].wait_requested is True


def test_live_shutdown_wait_exception_does_not_stop_remaining_orders() -> None:
    hbt = _ShutdownFakeHbt(wait_fail_order_ids={101})
    working = _shutdown_working_orders(
        buy=_ShutdownFakeOrder(101),
        sell=_ShutdownFakeOrder(201),
    )

    results = cancel_working_orders_for_shutdown(hbt, working)

    assert hbt.events == _expected_shutdown_cancel_wait_events([101, 201])
    assert results[0].cancel_sent is True
    assert results[0].wait_requested is True
    assert results[0].error.startswith("wait_error:RuntimeError:")
    assert results[1].cancel_sent is True
    assert results[1].wait_requested is True
    assert results[1].error == ""


def test_live_shutdown_waits_for_cancel_ack_before_close() -> None:
    hbt = _ShutdownFakeHbt()
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    cancel_working_orders_for_shutdown(hbt, working)
    hbt.close()

    assert hbt.events == [
        ("cancel", 0, 101, False),
        ("wait", 0, 101, SHUTDOWN_CANCEL_ACK_TIMEOUT_NS),
        ("close",),
    ]


class _ShutdownWaitResultFakeHbt:
    def __init__(
        self,
        wait_result: int,
        *,
        final_active_order_ids: set[int] | None = None,
        final_order_statuses_by_order_id: dict[int, int] | None = None,
        exchange_open_order_ids: set[int] | None = None,
        orders_fail: bool = False,
    ) -> None:
        self.wait_result = wait_result
        self.final_active_order_ids = final_active_order_ids or set()
        self.final_order_statuses_by_order_id = final_order_statuses_by_order_id or {}
        self.exchange_open_order_ids = exchange_open_order_ids or set()
        self.orders_fail = orders_fail
        self.events: list[tuple[object, ...]] = []
        self.status_checks: list[tuple[object, ...]] = []

    def cancel(self, asset_no: int, order_id: int, wait: bool) -> None:
        self.events.append(("cancel", asset_no, order_id, wait))

    def wait_order_response(self, asset_no: int, order_id: int, timeout_ns: int) -> int:
        self.events.append(("wait", asset_no, order_id, timeout_ns))
        return self.wait_result

    def orders(self, asset_no: int) -> list[object]:
        self.status_checks.append(("orders", asset_no))
        if self.orders_fail:
            raise RuntimeError("orders unavailable")
        statuses = {
            order_id: NEW for order_id in self.final_active_order_ids
        } | self.final_order_statuses_by_order_id
        return _ShutdownFakeOrderDict(
            [
                _ShutdownFakeOrder(order_id, status=status)
                for order_id, status in sorted(statuses.items())
            ]
        )

    def open_orders(self, asset_no: int) -> list[object]:
        self.status_checks.append(("open_orders", asset_no))
        return [
            _ShutdownFakeOrder(order_id, status=NEW)
            for order_id in sorted(self.exchange_open_order_ids)
        ]

    def order_status(self, asset_no: int, order_id: int) -> str:
        self.status_checks.append(("order_status", asset_no, order_id))
        return "new"


def _read_repo_text(relative_path: str) -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / relative_path).read_text()


def test_wait_order_response_live_binding_result_codes_are_not_cancel_ack_states() -> None:
    live_binding = _read_repo_text("py-hftbacktest/src/live.rs")

    assert "Ok(ElapseResult::Ok) => 0" in live_binding
    assert "Ok(ElapseResult::OrderResponse) => 3" in live_binding
    assert "Err(BotError::Timeout) => 17" in live_binding


def test_live_wait_order_response_timeout_and_batch_response_share_zero_code() -> None:
    live_bot = _read_repo_text("hftbacktest/src/live/bot.rs")

    assert "if received_order_resp {\n                    return Ok(ElapseResult::OrderResponse);" in live_bot
    assert "if wait_resp_received {\n                        return Ok(ElapseResult::Ok);" in live_bot
    assert "Err(BotError::Timeout) => {\n                    return Ok(ElapseResult::Ok);" in live_bot


def test_shutdown_wait_zero_with_local_terminal_proof_keeps_dimensions_independent() -> None:
    hbt = _ShutdownWaitResultFakeHbt(wait_result=0)
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    results = cancel_working_orders_for_shutdown(hbt, working)

    assert hbt.events == _expected_shutdown_cancel_wait_events([101])
    assert hbt.status_checks == [("orders", 0)]
    assert len(results) == 1
    assert results[0].wait_requested is True
    assert results[0].wait_result == 0
    assert results[0].wait_result_raw == 0
    assert results[0].order_response_received is False
    assert results[0].wait_outcome == "ok_unknown_or_timeout"
    assert results[0].terminal_confirmed is True
    assert results[0].terminal_confirmation_source == "local_orders"
    assert results[0].final_order_status == "local_absent_from_local_orders"
    assert results[0].error == ""


def test_shutdown_wait_three_with_local_order_still_active_is_not_terminal_confirmed() -> None:
    hbt = _ShutdownWaitResultFakeHbt(wait_result=3, final_active_order_ids={101})
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    results = cancel_working_orders_for_shutdown(hbt, working)

    assert hbt.events == _expected_shutdown_cancel_wait_events([101])
    assert hbt.status_checks == [("orders", 0)]
    assert len(results) == 1
    assert results[0].wait_requested is True
    assert results[0].wait_result == 3
    assert results[0].wait_result_raw == 3
    assert results[0].order_response_received is True
    assert results[0].wait_outcome == "order_response_received"
    assert results[0].terminal_confirmed is False
    assert results[0].terminal_confirmation_source == "local_orders"
    assert results[0].final_order_status == "local_active_order:new"
    assert results[0].error == ""


def test_shutdown_wait_three_with_partially_filled_local_order_is_not_terminal_confirmed() -> None:
    hbt = _ShutdownWaitResultFakeHbt(
        wait_result=3,
        final_order_statuses_by_order_id={101: PARTIALLY_FILLED},
    )
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    result = cancel_working_orders_for_shutdown(hbt, working)[0]

    assert result.order_response_received is True
    assert result.wait_outcome == "order_response_received"
    assert result.terminal_confirmed is False
    assert result.terminal_confirmation_source == "local_orders"
    assert result.final_order_status == "local_active_order:partially_filled"


@pytest.mark.parametrize(
    ("terminal_status", "expected_detail"),
    [
        (FILLED, "local_terminal_order:filled"),
        (CANCELED, "local_terminal_order:canceled"),
        (REJECTED, "local_terminal_order:rejected"),
    ],
)
def test_shutdown_wait_zero_with_local_terminal_status_confirms_terminal(
    terminal_status: int,
    expected_detail: str,
) -> None:
    hbt = _ShutdownWaitResultFakeHbt(
        wait_result=0,
        final_order_statuses_by_order_id={101: terminal_status},
    )
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    result = cancel_working_orders_for_shutdown(hbt, working)[0]

    assert result.order_response_received is False
    assert result.wait_outcome == "ok_unknown_or_timeout"
    assert result.terminal_confirmed is True
    assert result.terminal_confirmation_source == "local_orders"
    assert result.final_order_status == expected_detail


def test_shutdown_wait_three_with_local_terminal_proof_can_confirm_both_dimensions() -> None:
    hbt = _ShutdownWaitResultFakeHbt(wait_result=3)
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    result = cancel_working_orders_for_shutdown(hbt, working)[0]

    assert result.order_response_received is True
    assert result.wait_outcome == "order_response_received"
    assert result.terminal_confirmed is True
    assert result.terminal_confirmation_source == "local_orders"
    assert result.final_order_status == "local_absent_from_local_orders"


def test_shutdown_local_absent_terminal_confirmed_is_local_only_without_exchange_check() -> None:
    hbt = _ShutdownWaitResultFakeHbt(wait_result=3, exchange_open_order_ids={101})
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    result = cancel_working_orders_for_shutdown(hbt, working)[0]

    assert result.order_response_received is True
    assert result.terminal_confirmed is True
    assert result.terminal_confirmation_source == "local_orders"
    assert result.final_order_status == "local_absent_from_local_orders"
    assert hbt.status_checks == [("orders", 0)]


def test_shutdown_result_has_no_exchange_final_proof_fields() -> None:
    hbt = _ShutdownWaitResultFakeHbt(wait_result=0, exchange_open_order_ids={101})
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    result = cancel_working_orders_for_shutdown(hbt, working)[0]

    assert result.terminal_confirmed is True
    assert result.final_order_status == "local_absent_from_local_orders"
    assert not hasattr(result, "exchange_reconciliation_checked")
    assert not hasattr(result, "exchange_open_order_absent")
    assert not hasattr(result, "exchange_confirmation_source")
    assert not hasattr(result, "final_proof_level")


def test_shutdown_summary_has_no_exchange_reconciliation_counter() -> None:
    live_tick_mm = _read_repo_text("examples/binance_tick_mm/live_tick_mm.py")
    summary_start = live_tick_mm.index('"Shutdown cancel attempts=%d')
    summary_end = live_tick_mm.index("for result in cancel_results:", summary_start)
    summary_block = live_tick_mm[summary_start:summary_end]

    assert "wait_requests=%d" in summary_block
    assert "order_responses=%d" in summary_block
    assert "terminal_confirmed=%d" in summary_block
    assert "unknown_or_timeout=%d" in summary_block
    assert "exchange" not in summary_block
    assert "open_orders" not in summary_block
    assert "reconciliation" not in summary_block


def test_shutdown_audit_tail_final_exchange_state_not_written() -> None:
    live_tick_mm = _read_repo_text("examples/binance_tick_mm/live_tick_mm.py")
    shutdown_start = live_tick_mm.index("# ---- Graceful shutdown: cancel all open orders")
    shutdown_end = live_tick_mm.index("# Read position before closing", shutdown_start)
    shutdown_block = live_tick_mm[shutdown_start:shutdown_end]

    assert "cancel_working_orders_for_shutdown" in shutdown_block
    assert "audit" not in shutdown_block.lower()
    assert "reconciliation" not in shutdown_block.lower()
    assert "final no-open-order" not in shutdown_block.lower()


@pytest.mark.parametrize("wait_result", [0, 3])
def test_shutdown_wait_result_without_final_status_proof_never_confirms_terminal(
    wait_result: int,
) -> None:
    hbt = _ShutdownWaitResultFakeHbt(wait_result=wait_result, orders_fail=True)
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    result = cancel_working_orders_for_shutdown(hbt, working)[0]

    assert result.wait_result_raw == wait_result
    assert result.order_response_received is (wait_result == 3)
    assert result.terminal_confirmed is False
    assert result.terminal_confirmation_source == "none"
    assert result.final_order_status.startswith("local_orders_unavailable:RuntimeError:")


def test_shutdown_summary_counters_split_wait_response_and_terminal_confirmation() -> None:
    timeout_like = _ShutdownWaitResultFakeHbt(wait_result=0)
    response_like = _ShutdownWaitResultFakeHbt(wait_result=3)
    working = _shutdown_working_orders(buy=_ShutdownFakeOrder(101))

    timeout_result = cancel_working_orders_for_shutdown(timeout_like, working)[0]
    response_result = cancel_working_orders_for_shutdown(response_like, working)[0]

    wait_requests = sum(1 for result in [timeout_result, response_result] if result.wait_requested)
    order_responses = sum(
        1 for result in [timeout_result, response_result] if result.order_response_received
    )
    terminal_confirmed = sum(
        1 for result in [timeout_result, response_result] if result.terminal_confirmed
    )
    unknown_or_timeout = sum(
        1
        for result in [timeout_result, response_result]
        if result.wait_outcome == "ok_unknown_or_timeout"
    )

    assert wait_requests == 2
    assert order_responses == 1
    assert terminal_confirmed == 2
    assert unknown_or_timeout == 1
    assert timeout_result.error == ""
    assert response_result.error == ""


def test_shutdown_result_enums_are_fixed_and_rest_source_is_unused() -> None:
    hbt = _ShutdownFakeHbt(
        wait_fail_order_ids={101},
        wait_results_by_order_id={201: 3, 301: 0},
        orders_fail=True,
    )
    working = _shutdown_working_orders(
        buy=_ShutdownFakeOrder(101),
        sell=_ShutdownFakeOrder(201),
        extras=[ExtraOrder(301, "buy", 1000)],
    )

    results = cancel_working_orders_for_shutdown(hbt, working)

    assert {result.wait_outcome for result in results} <= SHUTDOWN_WAIT_OUTCOMES
    assert {
        result.terminal_confirmation_source for result in results
    } <= SHUTDOWN_TERMINAL_CONFIRMATION_SOURCES
    assert all(result.terminal_confirmation_source != "rest_open_orders" for result in results)


class FakeAsset:
    def __init__(self) -> None:
        self.snapshots: list[str] = []

    def initial_snapshot(self, path: str) -> None:
        self.snapshots.append(path)


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return str(path)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


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
        "add_side_submit_eligible_buy",
        "add_side_submit_eligible_sell",
        "add_side_submit_blocked_buy",
        "add_side_submit_blocked_sell",
        "add_side_submit_reduce_side_allowed_buy",
        "add_side_submit_reduce_side_allowed_sell",
        "target_move_since_last_quote_or_cancel_buy",
        "target_move_since_last_quote_or_cancel_sell",
        "last_cancel_request_age_ms_buy",
        "last_cancel_fill_age_ms_buy",
        "toxic_timing_guard_until_ts_buy",
    ]:
        assert field in AUDIT_FIELDS


def _compact_lifecycle_row(event_type: str, order_id: str = "") -> dict[str, str]:
    row = {field: "" for field in AUDIT_FIELDS}
    row["event_type"] = event_type
    row["order_id"] = order_id
    row["linked_order_id"] = order_id
    row["strategy_seq"] = "1"
    return row


def test_compact_lifecycle_audit_writer_deduplicates_terminal_rows() -> None:
    buffer = io.StringIO()
    csv_writer = csv.DictWriter(buffer, fieldnames=AUDIT_FIELDS)
    csv_writer.writeheader()
    writer = CompactLifecycleAuditWriter(csv_writer)

    assert writer.writerow(_compact_lifecycle_row("decision"))
    assert writer.writerow(_compact_lifecycle_row("cancel_ack", "7"))
    assert not writer.writerow(_compact_lifecycle_row("cancel_ack", "7"))
    assert writer.writerow(_compact_lifecycle_row("fill", "7"))
    assert writer.writerow(_compact_lifecycle_row("cancel_ack", "8"))

    rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
    assert [row["event_type"] for row in rows] == ["decision", "cancel_ack", "fill", "cancel_ack"]
    assert writer.written_rows == 4
    assert writer.skipped_duplicate_terminal_rows == 1


def test_compact_lifecycle_audit_writer_keeps_duplicate_non_terminal_rows() -> None:
    buffer = io.StringIO()
    csv_writer = csv.DictWriter(buffer, fieldnames=AUDIT_FIELDS)
    csv_writer.writeheader()
    writer = CompactLifecycleAuditWriter(csv_writer)

    assert writer.writerow(_compact_lifecycle_row("order_submit_sent", "9"))
    assert writer.writerow(_compact_lifecycle_row("order_submit_sent", "9"))
    assert writer.writerow(_compact_lifecycle_row("cancel_sent", "9"))
    assert writer.writerow(_compact_lifecycle_row("cancel_sent", "9"))

    rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
    assert [row["event_type"] for row in rows] == [
        "order_submit_sent",
        "order_submit_sent",
        "cancel_sent",
        "cancel_sent",
    ]
    assert writer.written_rows == 4
    assert writer.skipped_duplicate_terminal_rows == 0


def test_market_view_provenance_fields_are_in_audit_schema() -> None:
    for field in [
        "market_view_source",
        "top5_source",
        "market_overlay_source",
        "top5_overlay_source",
        "book_view_ts_local",
        "book_view_ts_exch",
        "book_view_feed_latency_ns",
        "book_view_stale_ms",
        "top5_depth_best_bid_tick",
        "top5_depth_best_ask_tick",
    ]:
        assert field in AUDIT_FIELDS


def test_step_8c_quote_update_fields_are_in_audit_schema() -> None:
    for field in [
        "quote_update_intent",
        "quote_update_action",
        "quote_update_reason",
        "min_move_passed",
        "quote_age_ms",
        "join_age_ms",
        "anchor_age_ms",
        "latency_bucket",
        "throttle_state",
        "token_bucket_state",
        "cancel_readd_bucket",
        "reject_throttle_drop_cause",
        "post_only_pre_check",
        "post_only_post_check",
        "inventory_request_id",
    ]:
        assert field in AUDIT_FIELDS


def test_step_8c_quote_update_helper_is_wired_into_live_and_backtest() -> None:
    for filename in ["backtest_tick_mm.py", "live_tick_mm.py"]:
        source = Path(__file__).with_name(filename).read_text()
        assert "build_quote_update_audit_fields(" in source
        assert "quote_update_fields=" in source
        assert "inventory_request_id=\"\"" in source


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


SAME_COUNT_OPEN_ORDER_DRIFT_CASES = [
    pytest.param(
        "single_side_drift_bid_vs_ask",
        "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1",
        "999:sell:770001:0.001:price=77000.10:exec=0:status=new:tif=gtx",
        "local_only=buy:770001:0.001;rest_only=sell:770001:0.001",
        id="single_side_drift_bid_vs_ask",
    ),
    pytest.param(
        "single_price_tick_drift",
        "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1",
        "999:buy:770002:0.001:price=77000.20:exec=0:status=new:tif=gtx",
        "local_only=buy:770001:0.001;rest_only=buy:770002:0.001",
        id="single_price_tick_drift",
    ),
    pytest.param(
        "single_qty_drift",
        "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1",
        "999:buy:770001:0.002:price=77000.10:exec=0:status=new:tif=gtx",
        "local_only=buy:770001:0.001;rest_only=buy:770001:0.002",
        id="single_qty_drift",
    ),
    pytest.param(
        "single_price_and_qty_drift",
        "101:sell:770101:0.001:new:req=none:cxl=1:exch=1:local=1",
        "999:sell:770102:0.002:price=77010.20:exec=0:status=new:tif=gtx",
        "local_only=sell:770101:0.001;rest_only=sell:770102:0.002",
        id="single_price_and_qty_drift",
    ),
    pytest.param(
        "two_orders_one_leg_price_drift",
        "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1;202:sell:770101:0.001:new:req=none:cxl=1:exch=1:local=1",
        "888:buy:770001:0.001:price=77000.10:exec=0:status=new:tif=gtx;999:sell:770102:0.001:price=77010.20:exec=0:status=new:tif=gtx",
        "local_only=sell:770101:0.001;rest_only=sell:770102:0.001",
        id="two_orders_one_leg_price_drift",
    ),
    pytest.param(
        "two_orders_same_count_disjoint_set",
        "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1;202:sell:770101:0.001:new:req=none:cxl=1:exch=1:local=1",
        "888:buy:770003:0.001:price=77000.30:exec=0:status=new:tif=gtx;999:sell:770103:0.002:price=77010.30:exec=0:status=new:tif=gtx",
        "local_only=buy:770001:0.001|sell:770101:0.001;rest_only=buy:770003:0.001|sell:770103:0.002",
        id="two_orders_same_count_disjoint_set",
    ),
    pytest.param(
        "single_random_client_quote_key_different",
        "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1",
        "?:buy:770002:0.001:price=77000.20:exec=0:status=new:tif=gtx:client=mmrandom",
        "local_only=buy:770001:0.001;rest_only=buy:770002:0.001",
        id="single_random_client_quote_key_different",
    ),
]


@pytest.mark.parametrize(
    ("case_name", "local_open_orders", "rest_open_orders", "expected_diff"),
    SAME_COUNT_OPEN_ORDER_DRIFT_CASES,
)
def test_evaluate_live_safety_flags_same_count_open_order_drift_pending(
    case_name: str,
    local_open_orders: str,
    rest_open_orders: str,
    expected_diff: str,
) -> None:
    diff = open_order_diff(local_open_orders, rest_open_orders)

    state = evaluate_live_safety(
        cfg=LiveSafetyConfig(open_order_mismatch_confirmations=2),
        rest_position=0.001,
        local_position=0.001,
        rest_open_order_count=local_open_orders.count(";") + 1,
        local_open_order_count=local_open_orders.count(";") + 1,
        rest_error="",
        rest_open_orders=rest_open_orders,
        local_open_orders=local_open_orders,
        open_order_diff=diff,
        ts_local=10_000_000_000,
        last_api_ts=0,
        open_order_mismatch_count=0,
    )

    assert case_name
    assert diff == expected_diff
    assert state.safety_status == "open_order_mismatch_pending"
    assert state.open_order_diff == expected_diff
    assert state.safety_detail == expected_diff


@pytest.mark.parametrize(
    ("case_name", "local_open_orders", "rest_open_orders", "expected_diff"),
    SAME_COUNT_OPEN_ORDER_DRIFT_CASES,
)
def test_evaluate_live_safety_flags_same_count_open_order_drift_confirmed(
    case_name: str,
    local_open_orders: str,
    rest_open_orders: str,
    expected_diff: str,
) -> None:
    diff = open_order_diff(local_open_orders, rest_open_orders)

    state = evaluate_live_safety(
        cfg=LiveSafetyConfig(open_order_mismatch_confirmations=2),
        rest_position=0.001,
        local_position=0.001,
        rest_open_order_count=local_open_orders.count(";") + 1,
        local_open_order_count=local_open_orders.count(";") + 1,
        rest_error="",
        rest_open_orders=rest_open_orders,
        local_open_orders=local_open_orders,
        open_order_diff=diff,
        ts_local=10_000_000_000,
        last_api_ts=0,
        open_order_mismatch_count=1,
    )

    assert case_name
    assert diff == expected_diff
    assert state.safety_status == "open_order_mismatch"
    assert state.open_order_diff == expected_diff
    assert state.safety_detail == expected_diff


def test_evaluate_live_safety_allows_same_quote_key_with_different_ids() -> None:
    local_open_orders = "101:buy:770001:0.001:new:req=none:cxl=1:exch=1:local=1"
    rest_open_orders = "999:buy:770001:0.001:price=77000.10:exec=0:status=new:tif=gtx"
    diff = open_order_diff(local_open_orders, rest_open_orders)

    state = evaluate_live_safety(
        cfg=LiveSafetyConfig(open_order_mismatch_confirmations=2),
        rest_position=0.001,
        local_position=0.001,
        rest_open_order_count=1,
        local_open_order_count=1,
        rest_error="",
        rest_open_orders=rest_open_orders,
        local_open_orders=local_open_orders,
        open_order_diff=diff,
        ts_local=10_000_000_000,
        last_api_ts=0,
        open_order_mismatch_count=1,
    )

    assert diff == ""
    assert state.safety_status == "ok"
    assert state.safety_detail == ""


def test_evaluate_live_safety_keeps_metadata_diff_boundary_on_quote_key_only() -> None:
    local_open_orders = "101:buy:770001:0.001:new:req=cancel:cxl=0:exch=1:local=1"
    rest_open_orders = "999:buy:770001:0.001:price=77000.10:exec=0.0005:status=partially_filled:tif=gtx"
    diff = open_order_diff(local_open_orders, rest_open_orders)

    state = evaluate_live_safety(
        cfg=LiveSafetyConfig(open_order_mismatch_confirmations=2),
        rest_position=0.001,
        local_position=0.001,
        rest_open_order_count=1,
        local_open_order_count=1,
        rest_error="",
        rest_open_orders=rest_open_orders,
        local_open_orders=local_open_orders,
        open_order_diff=diff,
        ts_local=10_000_000_000,
        last_api_ts=0,
        open_order_mismatch_count=1,
    )

    assert diff == ""
    assert state.safety_status == "ok"
    assert state.safety_detail == ""


def _live_fatal_mismatch_breaks(cfg: LiveSafetyConfig, safety_status: str) -> bool:
    return cfg.fail_on_mismatch and safety_status not in {
        "ok",
        "safety_disabled",
        "open_order_grace",
        "open_order_mismatch_pending",
        "position_mismatch_pending",
    }


def _live_current_position_mismatch_pause_trading(cfg: LiveSafetyConfig, safety_status: str) -> bool:
    return cfg.position_mismatch_pause_trading and safety_status == "position_mismatch_pending"


def _live_position_mismatch_state(
    cfg: LiveSafetyConfig,
    *,
    position_mismatch_count: int,
) -> str:
    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.004,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=0,
        rest_error="",
        position_mismatch_count=position_mismatch_count,
    )
    return state.safety_status


def test_live_safety_position_mismatch_fatal_mode_breaks_only_after_confirmation() -> None:
    cfg = LiveSafetyConfig(fail_on_mismatch=True, position_mismatch_confirmations=2)

    pending_status = _live_position_mismatch_state(cfg, position_mismatch_count=0)
    confirmed_status = _live_position_mismatch_state(cfg, position_mismatch_count=1)

    assert pending_status == "position_mismatch_pending"
    assert confirmed_status == "position_mismatch"
    assert _live_fatal_mismatch_breaks(cfg, pending_status) is False
    assert _live_fatal_mismatch_breaks(cfg, confirmed_status) is True


def test_live_safety_nonfatal_pause_mode_stops_pending_but_allows_confirmed_position_mismatch() -> None:
    cfg = LiveSafetyConfig(
        fail_on_mismatch=False,
        position_mismatch_pause_trading=True,
        position_mismatch_confirmations=2,
    )

    pending_status = _live_position_mismatch_state(cfg, position_mismatch_count=0)
    confirmed_status = _live_position_mismatch_state(cfg, position_mismatch_count=1)

    assert pending_status == "position_mismatch_pending"
    assert confirmed_status == "position_mismatch"
    assert _live_fatal_mismatch_breaks(cfg, pending_status) is False
    assert _live_fatal_mismatch_breaks(cfg, confirmed_status) is False
    assert _live_current_position_mismatch_pause_trading(cfg, pending_status) is True
    assert _live_current_position_mismatch_pause_trading(cfg, confirmed_status) is False


def test_live_safety_nonfatal_pause_mode_confirmations_one_skips_pending_and_allows_trading() -> None:
    cfg = LiveSafetyConfig(
        fail_on_mismatch=False,
        position_mismatch_pause_trading=True,
        position_mismatch_confirmations=1,
    )

    status = _live_position_mismatch_state(cfg, position_mismatch_count=0)

    assert status == "position_mismatch"
    assert _live_fatal_mismatch_breaks(cfg, status) is False
    assert _live_current_position_mismatch_pause_trading(cfg, status) is False


def test_live_safety_nonfatal_pause_disabled_allows_pending_and_confirmed_position_mismatch() -> None:
    cfg = LiveSafetyConfig(
        fail_on_mismatch=False,
        position_mismatch_pause_trading=False,
        position_mismatch_confirmations=2,
    )

    pending_status = _live_position_mismatch_state(cfg, position_mismatch_count=0)
    confirmed_status = _live_position_mismatch_state(cfg, position_mismatch_count=1)

    assert pending_status == "position_mismatch_pending"
    assert confirmed_status == "position_mismatch"
    assert _live_current_position_mismatch_pause_trading(cfg, pending_status) is False
    assert _live_current_position_mismatch_pause_trading(cfg, confirmed_status) is False


def test_evaluate_live_safety_position_mismatch_ok_resets_on_next_loop_contract() -> None:
    cfg = LiveSafetyConfig(position_mismatch_confirmations=2)
    mismatch_state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.004,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=0,
        rest_error="",
        position_mismatch_count=1,
    )
    ok_state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=0,
        rest_error="",
        position_mismatch_count=2,
    )

    assert mismatch_state.safety_status == "position_mismatch"
    assert ok_state.safety_status == "ok"
    assert ok_state.position_mismatch == 0.0


def test_live_safety_rest_error_is_fatal_but_not_pause_only() -> None:
    fail_cfg = LiveSafetyConfig(fail_on_mismatch=True)
    nonfatal_cfg = LiveSafetyConfig(fail_on_mismatch=False, position_mismatch_pause_trading=True)
    state = evaluate_live_safety(
        cfg=nonfatal_cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=0,
        rest_error="timeout",
    )

    assert state.safety_status == "rest_error"
    assert _live_fatal_mismatch_breaks(fail_cfg, state.safety_status) is True
    assert _live_fatal_mismatch_breaks(nonfatal_cfg, state.safety_status) is False
    assert _live_current_position_mismatch_pause_trading(nonfatal_cfg, state.safety_status) is False


def test_live_safety_open_order_confirmed_has_no_position_pause_path() -> None:
    cfg = LiveSafetyConfig(
        fail_on_mismatch=False,
        position_mismatch_pause_trading=True,
        open_order_mismatch_confirmations=2,
    )
    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=1,
        local_open_order_count=0,
        rest_error="",
        ts_local=10_000_000_000,
        last_api_ts=0,
        open_order_mismatch_count=1,
    )

    assert state.safety_status == "open_order_mismatch"
    assert _live_fatal_mismatch_breaks(cfg, state.safety_status) is False
    assert _live_current_position_mismatch_pause_trading(cfg, state.safety_status) is False


def test_pure_cancel_extra_can_bypass_api_interval_guard() -> None:
    assert is_pure_cancel_extra([Action("cancel", "extra", 1, 0.0, 0.0)]) is True
    assert is_pure_cancel_extra([Action("cancel", "buy", 1, 0.0, 0.0)]) is False
    assert is_pure_cancel_extra(
        [
            Action("cancel", "extra", 1, 0.0, 0.0),
            Action("submit", "buy", 2, 100.0, 1.0),
        ]
    ) is False


def test_build_market_view_from_depth_records_top5_and_provenance() -> None:
    class _Depth:
        best_bid = 100.0
        best_ask = 101.0
        best_bid_tick = 1000
        best_ask_tick = 1010
        roi_lb_tick = 990
        roi_ub_tick = 1020

        @staticmethod
        def bid_qty_at_tick(tick: int) -> float:
            return {1000: 1.0, 999: 0.5}.get(tick, 0.0)

        @staticmethod
        def ask_qty_at_tick(tick: int) -> float:
            return {1010: 2.0, 1011: 0.25}.get(tick, 0.0)

    view = build_market_view_from_depth(
        _Depth(),
        source="replay_depth",
        ts_local=123,
        ts_exch=100,
        feed_latency_ns=23,
    )

    assert view.source == "replay_depth"
    assert view.top5_source == "replay_depth"
    assert view.best_bid == 100.0
    assert view.best_ask == 101.0
    assert view.mid == 100.5
    assert view.spread == 1.0
    assert view.bid_size == 1.5
    assert view.ask_size == 2.25
    assert view.bid_top5_ticks == "1000|999|998|997|996"
    assert view.ask_top5_qtys == "2.0|0.25|0.0|0.0|0.0"
    assert view.best_bid_tick == 1000
    assert view.best_ask_tick == 1010
    assert view.ts_local == 123
    assert view.ts_exch == 100
    assert view.feed_latency_ns == 23


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


def test_quote_throttle_reason_respects_position_limit_bypass() -> None:
    state = QuoteThrottleState(
        last_sent_api_ts=1_000_000_000,
        last_sent_target_bid_tick=1000,
        last_sent_target_ask_tick=1002,
    )
    actions = [Action("submit", "sell", 7, 100.2, 0.001)]

    assert (
        quote_throttle_reason(
            actions=actions,
            state=state,
            ts_local=1_050_000_000,
            target_bid_tick=1000,
            target_ask_tick=1003,
            min_interval_ns=100_000_000,
            min_move_ticks=2,
            pos_limit=False,
        )
        == "min_quote_update_interval"
    )
    assert (
        quote_throttle_reason(
            actions=actions,
            state=state,
            ts_local=1_050_000_000,
            target_bid_tick=1000,
            target_ask_tick=1003,
            min_interval_ns=100_000_000,
            min_move_ticks=2,
            pos_limit=True,
        )
        == ""
    )


def test_live_uses_shared_quote_throttle_state_update() -> None:
    live_source = Path(__file__).with_name("live_tick_mm.py").read_text()
    direct_mark_sent = "throttle_state." + "mark" + "_sent("

    assert direct_mark_sent not in live_source
    assert "update_quote_throttle_state(" in live_source


def test_quote_update_audit_fields_include_step_8c_columns() -> None:
    quote_anchor_safety = type(
        "Q",
        (),
        {
            "enabled": True,
            "anchor_age_ms": 12.5,
            "anchor_source": "bookticker",
            "anchor_bid_tick": 1000,
            "anchor_ask_tick": 1002,
            "original_bid_tick": 1001,
            "original_ask_tick": 1001,
            "post_only_risk_after_recheck": False,
            "stale_anchor": False,
            "missing_anchor": False,
            "suppress_buy": False,
            "suppress_sell": False,
            "bid_clamped": True,
            "ask_clamped": False,
            "bid_rounding_changed": False,
            "ask_rounding_changed": False,
        },
    )()
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=1000, last_sent_target_ask_tick=1002)
    bucket = TokenBucket(capacity=10.0, refill_per_sec=5.0, tokens=7.5, last_ts=1_000_000_000)
    fields = build_quote_update_audit_fields(
        planned_actions=[Action("submit", "buy", 1, 100.0, 0.001)],
        executed_actions=[],
        quote_throttle_cfg=QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2),
        quote_throttle_state=state,
        token_bucket=bucket,
        ts_local=1_050_000_000,
        target_bid_tick=1000,
        target_ask_tick=1002,
        quote_anchor_safety=quote_anchor_safety,
        book_view_stale_ms=8.0,
        auditlatency_ms=4.0,
        feed_latency_ns=4_000_000,
        latency_signal_ns=4_000_000,
        reject_reason="",
        throttle_reason="",
        dropped_by_latency=False,
        dropped_by_api_limit=False,
        pos_limit=False,
        working_bid_req="none",
        working_ask_req="none",
        last_cancel_request_age_ms_buy=12.0,
        last_cancel_request_age_ms_sell=12.0,
        last_cancel_fill_age_ms_buy=5.0,
        last_cancel_fill_age_ms_sell=5.0,
        inventory_request_id="",
        api_enabled=True,
    )

    assert set(
        [
            "quote_update_intent",
            "quote_update_action",
            "quote_update_reason",
            "min_move_passed",
            "quote_age_ms",
            "join_age_ms",
            "anchor_age_ms",
            "latency_bucket",
            "throttle_state",
            "token_bucket_state",
            "cancel_readd_bucket",
            "reject_throttle_drop_cause",
            "post_only_pre_check",
            "post_only_post_check",
            "inventory_request_id",
        ]
    ).issubset(fields)
    assert fields["quote_update_intent"] == "submit"
    assert fields["quote_update_action"] == "drop"
    assert fields["quote_update_reason"] == "post_only_risk"
    assert fields["min_move_passed"] == 0
    assert fields["quote_age_ms"] == 50.0
    assert fields["anchor_age_ms"] == 12.5
    assert fields["token_bucket_state"].startswith("enabled=1|capacity=10")
    assert fields["reject_throttle_drop_cause"] == ""


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


def test_decide_actions_blocks_add_side_that_would_cross_qty_cap() -> None:
    actions, next_order_id = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=160.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
        position=0.002,
        max_position_qty=0.002,
    )

    assert [(action.kind, action.side) for action in actions] == [("submit", "sell")]
    assert next_order_id == 11


def test_decide_actions_counts_working_add_side_leaves_before_qty_cap() -> None:
    working = WorkingOrders(
        buy=OrderSnapshot(
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
        ),
        sell=None,
        extras=[],
    )

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=998,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=80.0,
        next_order_id=10,
        two_phase_replace_enabled=False,
        position=0.001,
        max_position_qty=0.002,
    )

    assert [(action.kind, action.side) for action in actions] == [
        ("cancel", "buy"),
        ("submit", "sell"),
    ]
    assert next_order_id == 11


def test_add_side_soft_limit_qty_from_risk_prefers_explicit_qty() -> None:
    assert add_side_soft_limit_qty_from_risk(
        {
            "max_position_qty": 0.003,
            "inventory_add_side_soft_limit_ratio": 0.8,
            "inventory_add_side_soft_limit_qty": 0.0015,
        }
    ) == pytest.approx(0.0015)


def test_add_side_soft_limit_qty_from_risk_uses_ratio() -> None:
    assert add_side_soft_limit_qty_from_risk(
        {
            "max_position_qty": 0.003,
            "inventory_add_side_soft_limit_ratio": 0.75,
        }
    ) == pytest.approx(0.00225)


def test_decide_actions_soft_limit_allows_existing_add_order_below_limit() -> None:
    working = WorkingOrders(
        buy=OrderSnapshot(
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
        ),
        sell=None,
        extras=[],
    )

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=80.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
        position=0.001,
        max_position_qty=0.0031,
        add_side_soft_limit_qty=0.0025,
    )

    assert [(action.kind, action.side) for action in actions] == [("submit", "sell")]
    assert next_order_id == 11


def test_decide_actions_soft_limit_blocks_new_add_order_above_limit() -> None:
    actions, next_order_id = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=160.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
        position=0.002,
        max_position_qty=0.0031,
        add_side_soft_limit_qty=0.0025,
    )

    assert [(action.kind, action.side) for action in actions] == [("submit", "sell")]
    assert next_order_id == 11


def test_decide_actions_cooldown_blocks_add_side_but_allows_reduce_side() -> None:
    long_actions, long_next_order_id = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=80.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
        position=0.001,
        max_position_qty=0.003,
        add_side_cooldown_block_buy=True,
    )
    short_actions, short_next_order_id = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=-80.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
        position=-0.001,
        max_position_qty=0.003,
        add_side_cooldown_block_buy=True,
    )

    assert [(action.kind, action.side) for action in long_actions] == [("submit", "sell")]
    assert long_next_order_id == 11
    assert [(action.kind, action.side) for action in short_actions] == [
        ("submit", "buy"),
        ("submit", "sell"),
    ]
    assert short_next_order_id == 22


def test_add_side_toxic_timing_guard_default_off_never_blocks() -> None:
    tracker = InFlightExposureTracker.create()
    tracker.mark_submitted(Action("submit", "buy", 7, 100.0, 0.001))
    tracker.mark_cancel_requested(7)

    result = add_side_toxic_timing_guard_side_blocks(
        enabled=False,
        pending_cancel_enabled=True,
        post_cancel_fill_enabled=True,
        target_move_enabled=True,
        cooldown_ns=100_000_000,
        min_target_move_ticks=2,
        latency_threshold_ns=0,
        ts_local=1_000_000_000,
        position=0.001,
        target_bid_tick=998,
        target_ask_tick=1005,
        buy_submit_eligible=True,
        sell_submit_eligible=True,
        buy_reduce_side_allowed=False,
        sell_reduce_side_allowed=True,
        inflight_exposure=tracker,
        last_buy_cancel_ts=950_000_000,
        last_quote_or_cancel_target_bid_tick=1000,
    )

    assert result.buy_eligible is True
    assert result.buy_block is False
    assert result.sell_block is False
    assert result.buy_target_move_ticks == 2
    assert result.buy_last_cancel_request_age_ms == pytest.approx(50.0)


def test_add_side_toxic_timing_guard_blocks_only_add_side_submit_path() -> None:
    tracker = InFlightExposureTracker.create()
    tracker.mark_submitted(Action("submit", "buy", 7, 100.0, 0.001))
    tracker.mark_cancel_requested(7)

    result = add_side_toxic_timing_guard_side_blocks(
        enabled=True,
        pending_cancel_enabled=True,
        post_cancel_fill_enabled=True,
        target_move_enabled=True,
        cooldown_ns=100_000_000,
        min_target_move_ticks=2,
        latency_threshold_ns=0,
        ts_local=1_000_000_000,
        position=0.001,
        target_bid_tick=998,
        target_ask_tick=1005,
        buy_submit_eligible=True,
        sell_submit_eligible=True,
        buy_reduce_side_allowed=False,
        sell_reduce_side_allowed=True,
        inflight_exposure=tracker,
        last_buy_cancel_ts=950_000_000,
        last_quote_or_cancel_target_bid_tick=1000,
    )

    assert result.buy_block is True
    assert result.sell_block is False
    assert "pending_cancel" in result.buy_reason
    assert "target_move" in result.buy_reason
    assert result.buy_until_ts == 1_100_000_000
    assert result.sell_reduce_side_allowed is True


def test_decide_actions_toxic_timing_blocks_add_side_but_allows_reduce_side() -> None:
    long_actions, long_next_order_id = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=80.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
        position=0.001,
        add_side_toxic_timing_guard_block_buy=True,
        add_side_toxic_timing_guard_block_sell=True,
    )
    short_actions, short_next_order_id = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=-80.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
        position=-0.001,
        add_side_toxic_timing_guard_block_buy=True,
        add_side_toxic_timing_guard_block_sell=True,
    )

    assert [(action.kind, action.side) for action in long_actions] == [("submit", "sell")]
    assert long_next_order_id == 11
    assert [(action.kind, action.side) for action in short_actions] == [("submit", "buy")]
    assert short_next_order_id == 21


def test_inflight_exposure_tracker_keeps_cancel_requested_order_until_terminal() -> None:
    tracker = InFlightExposureTracker.create()
    tracker.mark_submitted(Action("submit", "buy", 7, 100.0, 0.001))
    tracker.mark_cancel_requested(7)

    assert tracker.side_qty("buy") == pytest.approx(0.001)
    assert tracker.cancel_requested_side_qty("buy") == pytest.approx(0.001)

    tracker.observe_lifecycle(
        "cancel_sent",
        OrderSnapshot(
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
        ),
    )
    assert tracker.side_qty("buy") == pytest.approx(0.001)
    assert tracker.cancel_requested_side_qty("buy") == pytest.approx(0.001)

    tracker.observe_lifecycle(
        "cancel_ack",
        OrderSnapshot(
            order_id=7,
            side="buy",
            price=100.0,
            price_tick=1000,
            qty=0.001,
            leaves_qty=0.001,
            exec_qty=0.0,
            exec_price_tick=0,
            status="canceled",
            req="none",
            time_in_force="gtx",
            exch_timestamp=124,
            local_timestamp=120,
            cancellable=False,
        ),
    )
    assert tracker.side_qty("buy") == pytest.approx(0.0)
    assert tracker.cancel_requested_side_qty("buy") == pytest.approx(0.0)


def test_cancel_race_guard_blocks_pending_cancel_and_post_fill_cooldown() -> None:
    tracker = InFlightExposureTracker.create()
    tracker.mark_submitted(Action("submit", "buy", 7, 100.0, 0.001))
    tracker.mark_cancel_requested(7)

    buy_block, sell_block = cancel_race_guard_side_blocks(
        enabled=True,
        pending_cancel_block=True,
        post_fill_cooldown_ns=0,
        ts_local=1_000,
        inflight_exposure=tracker,
    )

    assert buy_block is True
    assert sell_block is False

    buy_block, sell_block = cancel_race_guard_side_blocks(
        enabled=True,
        pending_cancel_block=False,
        post_fill_cooldown_ns=1_000,
        ts_local=1_500,
        inflight_exposure=InFlightExposureTracker.create(),
        last_sell_cancel_fill_ts=1_000,
    )

    assert buy_block is False
    assert sell_block is True


def test_decide_actions_inflight_exposure_blocks_add_side_stacking() -> None:
    actions, next_order_id = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=80.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
        position=0.001,
        max_position_qty=0.003,
        add_side_inflight_buy_qty=0.002,
    )

    assert [(action.kind, action.side) for action in actions] == [("submit", "sell")]
    assert next_order_id == 11


def test_decide_actions_cancel_race_guard_blocks_add_side_but_allows_reduce_side() -> None:
    long_actions, long_next_order_id = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=80.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
        position=0.001,
        max_position_qty=0.003,
        cancel_race_guard_block_buy=True,
    )
    short_actions, short_next_order_id = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=-80.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
        position=-0.001,
        max_position_qty=0.003,
        cancel_race_guard_block_buy=True,
    )

    assert [(action.kind, action.side) for action in long_actions] == [("submit", "sell")]
    assert long_next_order_id == 11
    assert [(action.kind, action.side) for action in short_actions] == [
        ("submit", "buy"),
        ("submit", "sell"),
    ]
    assert short_next_order_id == 22


def test_adverse_timing_guard_disabled_has_no_blocks() -> None:
    result = adverse_timing_guard_side_blocks(
        enabled=False,
        target_deterioration_enabled=True,
        pending_cancel_enabled=True,
        post_cancel_fill_enabled=True,
        cooldown_ns=100_000_000,
        min_target_move_ticks=2,
        ts_local=1_000_000_000,
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=998,
        target_ask_tick=1002,
        inflight_exposure=InFlightExposureTracker.create(),
        last_buy_cancel_fill_ts=999_999_999,
        last_sell_cancel_fill_ts=999_999_999,
    )

    assert result.buy_block is False
    assert result.sell_block is False
    assert result.buy_reason == ""
    assert result.sell_reason == ""


def test_adverse_timing_guard_target_deterioration_and_pending_cancel() -> None:
    tracker = InFlightExposureTracker.create()
    tracker.mark_submitted(Action("submit", "sell", 8, 100.3, 0.001))
    tracker.mark_cancel_requested(8)

    result = adverse_timing_guard_side_blocks(
        enabled=True,
        target_deterioration_enabled=True,
        pending_cancel_enabled=True,
        post_cancel_fill_enabled=False,
        cooldown_ns=100_000_000,
        min_target_move_ticks=2,
        ts_local=1_000_000_000,
        working=WorkingOrders(
            buy=None,
            sell=OrderSnapshot(
                order_id=8,
                side="sell",
                price=100.3,
                price_tick=1003,
                qty=0.001,
                leaves_qty=0.001,
                exec_qty=0.0,
                exec_price_tick=0,
                status="new",
                req="cancel",
                time_in_force="gtx",
                exch_timestamp=0,
                local_timestamp=0,
                cancellable=False,
            ),
            extras=[],
        ),
        target_bid_tick=998,
        target_ask_tick=1006,
        inflight_exposure=tracker,
    )

    assert result.buy_block is False
    assert result.sell_block is True
    assert result.sell_reason == "target_deterioration"
    assert result.sell_target_move_ticks == 3
    assert result.sell_until_ts == 1_100_000_000


def test_adverse_timing_guard_cancel_fill_cooldown_expires() -> None:
    active = adverse_timing_guard_side_blocks(
        enabled=True,
        target_deterioration_enabled=False,
        pending_cancel_enabled=False,
        post_cancel_fill_enabled=True,
        cooldown_ns=100,
        min_target_move_ticks=2,
        ts_local=1_050,
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=998,
        target_ask_tick=1002,
        inflight_exposure=InFlightExposureTracker.create(),
        last_buy_cancel_fill_ts=1_000,
    )
    expired = adverse_timing_guard_side_blocks(
        enabled=True,
        target_deterioration_enabled=False,
        pending_cancel_enabled=False,
        post_cancel_fill_enabled=True,
        cooldown_ns=100,
        min_target_move_ticks=2,
        ts_local=1_101,
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=998,
        target_ask_tick=1002,
        inflight_exposure=InFlightExposureTracker.create(),
        last_buy_cancel_fill_ts=1_000,
    )

    assert active.buy_block is True
    assert active.buy_reason == "cancel_requested_fill_timing"
    assert active.buy_until_ts == 1_100
    assert expired.buy_block is False


def test_decide_actions_adverse_timing_blocks_add_side_but_allows_reduce_side() -> None:
    long_actions, _ = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=80.0,
        next_order_id=10,
        two_phase_replace_enabled=True,
        position=0.001,
        adverse_timing_guard_block_buy=True,
    )
    short_actions, _ = decide_actions(
        working=WorkingOrders(buy=None, sell=None, extras=[]),
        target_bid_tick=1000,
        target_ask_tick=1005,
        qty=0.001,
        tick_size=0.1,
        pos_limit=False,
        position_notional=-80.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
        position=-0.001,
        adverse_timing_guard_block_buy=True,
    )

    assert [(action.kind, action.side) for action in long_actions] == [("submit", "sell")]
    assert [(action.kind, action.side) for action in short_actions] == [
        ("submit", "buy"),
        ("submit", "sell"),
    ]


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


def test_live_visible_working_orders_keeps_submit_overlay_without_release_ts() -> None:
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
        release_ts=0,
    )

    visible = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [overlay],
        [],
        decision_ts=250,
    )

    assert visible.buy is overlay
    assert visible.buy.req == 1
    assert visible.buy.cancellable is False


def test_pending_order_from_live_token_preserves_req_new_state() -> None:
    order = _pending_order_from_live_token(
        {
            "order_id": "17",
            "side": "sell",
            "price_tick": "1001",
            "qty": "0.001",
            "status": "new",
            "req": "new",
            "cxl": "0",
            "exch": "0",
            "local": "123",
        },
        tick_size=0.1,
        decision_ts=150,
    )

    assert order is not None
    assert order.order_id == 17
    assert order.side == SELL
    assert order.price == pytest.approx(100.1)
    assert order.status == 1
    assert order.req == 1
    assert order.cancellable is False
    assert order.local_timestamp == 123


def test_working_orders_from_live_state_reconstructs_primary_sides() -> None:
    buy = PendingLocalOrder(
        order_id=17,
        side=BUY,
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=123,
        status=1,
        req=0,
        cancellable=True,
    )
    sell = PendingLocalOrder(
        order_id=18,
        side=SELL,
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=124,
        status=1,
        req=4,
        cancellable=False,
    )

    working = _working_orders_from_live_state({17: buy, 18: sell})

    assert working.buy is buy
    assert working.sell is sell
    assert working.extras == []


def test_live_inflight_replay_uses_live_action_then_lifecycle_ordering(tmp_path: Path) -> None:
    audit = _write_csv(
        tmp_path / "audit.csv",
        [
            "run_id",
            "event_type",
            "ts_local",
            "action",
            "order_id",
            "target_bid_tick",
            "target_ask_tick",
            "order_side",
            "order_price_tick",
            "order_qty",
            "order_status",
            "cancel_requested",
            "fill_qty",
            "fill_price",
        ],
        [
            {
                "run_id": "run",
                "event_type": "decision",
                "ts_local": "100",
                "action": "submit_buy",
                "order_id": "17",
                "target_bid_tick": "1000",
                "target_ask_tick": "1002",
            },
            {
                "run_id": "run",
                "event_type": "fill",
                "ts_local": "200",
                "action": "fill",
                "order_id": "17",
                "order_side": "buy",
                "order_price_tick": "1000",
                "order_qty": "0.001",
                "order_status": "filled",
                "cancel_requested": "0",
                "fill_qty": "0.001",
                "fill_price": "100.0",
            },
        ],
    )
    decisions = _load_live_decision_rows_by_ts(audit, run_id="run")
    lifecycle = _load_live_lifecycle_events_by_decision_ts(
        audit,
        run_id="run",
        tick_size=0.1,
    )
    inflight = InFlightExposureTracker.create()

    _apply_live_inflight_replay_after_decision(
        inflight,
        live_decision_row=decisions[100],
        live_lifecycle_events=[],
        tick_size=0.1,
        qty=0.001,
    )
    assert inflight.side_qty("buy") == pytest.approx(0.001)

    _apply_live_inflight_replay_after_decision(
        inflight,
        live_decision_row=None,
        live_lifecycle_events=lifecycle[200],
        tick_size=0.1,
        qty=0.001,
    )
    assert inflight.side_qty("buy") == pytest.approx(0.0)


def test_load_live_terminal_constraints_by_order_id_reads_latest_terminal_row(tmp_path: Path) -> None:
    audit = _write_csv(
        tmp_path / "audit.csv",
        [
            "run_id",
            "event_type",
            "ts_local",
            "order_id",
            "order_status",
            "cancel_request_ts",
            "cancel_ack_ts",
        ],
        [
            {
                "run_id": "run",
                "event_type": "cancel_ack",
                "ts_local": "200",
                "order_id": "17",
                "order_status": "canceled",
                "cancel_request_ts": "150",
                "cancel_ack_ts": "200",
            },
            {
                "run_id": "run",
                "event_type": "expired",
                "ts_local": "220",
                "order_id": "18",
                "order_status": "expired",
                "cancel_request_ts": "",
                "cancel_ack_ts": "",
            },
        ],
    )

    constraints = _load_live_terminal_constraints_by_order_id(audit, run_id="run")

    assert constraints[17] == LiveTerminalLifecycleConstraint(
        order_id=17,
        final_state="canceled",
        cancel_request_ts=150,
        cancel_ack_ts=200,
        terminal_ts=200,
    )
    assert constraints[18] == LiveTerminalLifecycleConstraint(
        order_id=18,
        final_state="expired",
        cancel_request_ts=0,
        cancel_ack_ts=0,
        terminal_ts=220,
    )


def test_load_live_short_cancel_race_fill_constraints_by_order_id_reads_short_race_only(tmp_path: Path) -> None:
    audit = _write_csv(
        tmp_path / "audit.csv",
        [
            "run_id",
            "event_type",
            "ts_local",
            "ts_exch",
            "order_id",
            "cancel_request_ts",
            "fill_ts",
            "fill_qty",
            "fill_price",
            "fill_after_cancel_request",
        ],
        [
            {
                "run_id": "run",
                "event_type": "fill",
                "ts_local": "160",
                "ts_exch": "160",
                "order_id": "17",
                "cancel_request_ts": "150",
                "fill_ts": "155",
                "fill_qty": "0.001",
                "fill_price": "100.1",
                "fill_after_cancel_request": "1",
            },
                {
                    "run_id": "run",
                    "event_type": "fill",
                    "ts_local": "40000000",
                    "ts_exch": "40000000",
                    "order_id": "18",
                    "cancel_request_ts": "150",
                    "fill_ts": "40000000",
                    "fill_qty": "0.001",
                    "fill_price": "100.2",
                    "fill_after_cancel_request": "1",
                },
        ],
    )

    constraints = _load_live_short_cancel_race_fill_constraints_by_order_id(
        audit,
        tick_size=0.1,
        run_id="run",
    )

    assert constraints[17] == LiveShortCancelRaceFillConstraint(
        order_id=17,
        fill_ts_local=160,
        fill_ts_exch=155,
        cancel_request_ts=150,
        fill_qty=0.001,
        fill_price=100.1,
        fill_price_tick=1001,
    )
    assert 18 not in constraints


def test_apply_live_terminal_lifecycle_constraint_converts_replay_fill_after_live_cancel() -> None:
    order = OrderSnapshot(
        order_id=17,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.0,
        exec_qty=0.001,
        exec_price_tick=1000,
        status="filled",
        req="none",
        time_in_force="gtx",
        exch_timestamp=260,
        local_timestamp=250,
        cancellable=False,
    )

    constrained = _apply_live_terminal_lifecycle_constraint(
        order,
        decision_ts=250,
        live_constraint=LiveTerminalLifecycleConstraint(
            order_id=17,
            final_state="canceled",
            cancel_request_ts=150,
            cancel_ack_ts=200,
            terminal_ts=200,
        ),
    )

    assert constrained is not None
    lifecycle_type, snapshot = constrained
    assert lifecycle_type == "cancel_ack"
    assert snapshot.status == "canceled"
    assert snapshot.exec_qty == pytest.approx(0.0)
    assert snapshot.leaves_qty == pytest.approx(0.0)
    assert snapshot.exch_timestamp == 200


def test_apply_live_short_cancel_race_fill_constraint_converts_cancel_ack_to_fill() -> None:
    order = OrderSnapshot(
        order_id=17,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="canceled",
        req="none",
        time_in_force="gtx",
        exch_timestamp=260,
        local_timestamp=250,
        cancellable=False,
    )

    constrained = _apply_live_short_cancel_race_fill_constraint(
        order,
        decision_ts=250,
        lifecycle_type="cancel_ack",
        live_constraint=LiveShortCancelRaceFillConstraint(
            order_id=17,
            fill_ts_local=245,
            fill_ts_exch=244,
            cancel_request_ts=235,
            fill_qty=0.001,
            fill_price=100.1,
            fill_price_tick=1001,
        ),
    )

    assert constrained is not None
    lifecycle_type, snapshot = constrained
    assert lifecycle_type == "fill"
    assert snapshot.status == "filled"
    assert snapshot.exec_qty == pytest.approx(0.001)
    assert snapshot.exec_price_tick == 1001
    assert snapshot.exch_timestamp == 244


def test_apply_live_short_cancel_race_fill_constraint_skips_non_cancel_ack() -> None:
    order = OrderSnapshot(
        order_id=17,
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
        exch_timestamp=260,
        local_timestamp=250,
        cancellable=False,
    )

    constrained = _apply_live_short_cancel_race_fill_constraint(
        order,
        decision_ts=250,
        lifecycle_type="order_update",
        live_constraint=LiveShortCancelRaceFillConstraint(
            order_id=17,
            fill_ts_local=245,
            fill_ts_exch=244,
            cancel_request_ts=235,
            fill_qty=0.001,
            fill_price=100.1,
            fill_price_tick=1001,
        ),
    )

    assert constrained is None


def test_apply_live_terminal_lifecycle_constraint_keeps_fill_before_live_terminal() -> None:
    order = OrderSnapshot(
        order_id=17,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.0,
        exec_qty=0.001,
        exec_price_tick=1000,
        status="filled",
        req="none",
        time_in_force="gtx",
        exch_timestamp=180,
        local_timestamp=180,
        cancellable=False,
    )

    constrained = _apply_live_terminal_lifecycle_constraint(
        order,
        decision_ts=180,
        live_constraint=LiveTerminalLifecycleConstraint(
            order_id=17,
            final_state="canceled",
            cancel_request_ts=150,
            cancel_ack_ts=200,
            terminal_ts=200,
        ),
    )

    assert constrained is None


def test_collect_forced_live_terminal_events_injects_missing_cancel_ack() -> None:
    tracker = OrderLifecycleTracker.create()
    tracker.last_by_order_id[17] = OrderSnapshot(
        order_id=17,
        side="sell",
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="cancel",
        time_in_force="gtx",
        exch_timestamp=180,
        local_timestamp=180,
        cancellable=False,
    )

    forced = _collect_forced_live_terminal_events(
        tracker,
        decision_ts=250,
        live_constraints_by_order_id={
            17: LiveTerminalLifecycleConstraint(
                order_id=17,
                final_state="canceled",
                cancel_request_ts=150,
                cancel_ack_ts=200,
                terminal_ts=200,
            )
        },
        live_order_state_by_id={},
        emitted_order_ids=set(),
    )

    assert len(forced) == 1
    lifecycle_type, snapshot = forced[0]
    assert lifecycle_type == "cancel_ack"
    assert snapshot.status == "canceled"
    assert snapshot.exch_timestamp == 200
    assert 17 not in tracker.last_by_order_id


def test_collect_forced_live_terminal_events_skips_live_visible_orders() -> None:
    tracker = OrderLifecycleTracker.create()
    tracker.last_by_order_id[17] = OrderSnapshot(
        order_id=17,
        side="sell",
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="new",
        req="cancel",
        time_in_force="gtx",
        exch_timestamp=180,
        local_timestamp=180,
        cancellable=False,
    )
    live_state = PendingLocalOrder(
        order_id=17,
        side=SELL,
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=220,
        status=1,
        req=4,
        cancellable=False,
    )

    forced = _collect_forced_live_terminal_events(
        tracker,
        decision_ts=250,
        live_constraints_by_order_id={
            17: LiveTerminalLifecycleConstraint(
                order_id=17,
                final_state="canceled",
                cancel_request_ts=150,
                cancel_ack_ts=200,
                terminal_ts=200,
            )
        },
        live_order_state_by_id={17: live_state},
        emitted_order_ids=set(),
    )

    assert forced == []
    assert 17 in tracker.last_by_order_id


def test_live_order_absent_after_seen_uses_lifecycle_seen_then_decision_absent(tmp_path: Path) -> None:
    path = _write_csv(
        tmp_path / "audit.csv",
        ["run_id", "event_type", "ts_local", "local_open_orders"],
        [
            {"run_id": "run", "event_type": "decision", "ts_local": "100", "local_open_orders": ""},
            {
                "run_id": "run",
                "event_type": "order_new",
                "ts_local": "110",
                "local_open_orders": "17:sell:1001:0.001:new:req=new:cxl=0:exch=0:local=110",
            },
            {"run_id": "run", "event_type": "decision", "ts_local": "120", "local_open_orders": ""},
        ],
    )

    assert _load_live_order_absent_after_seen_ts(path, run_id="run") == {17: 120}


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
        req=4,
        cancellable=False,
        visible_ts=200,
        release_ts=300,
    )

    before_visible = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [overlay],
        decision_ts=150,
        live_pending_cancel_order_ids=set(),
    )
    after_visible = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [overlay],
        decision_ts=200,
        live_pending_cancel_order_ids={7},
    )

    assert before_visible.buy is engine_order
    assert before_visible.buy.req == "cancel"
    assert before_visible.buy.cancellable is False
    assert after_visible.buy.req == 4
    assert after_visible.buy.cancellable is False


def test_live_visible_working_orders_cancel_retention_overlay_hides_early_cancel_state() -> None:
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
    retention_overlay = PendingLocalOrder(
        order_id=7,
        side=BUY,
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=121,
        req=0,
        cancellable=True,
        visible_ts=150,
        release_ts=200,
    )

    before_release = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [],
        [retention_overlay],
        decision_ts=175,
        live_pending_cancel_order_ids=set(),
    )
    after_release = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [],
        [retention_overlay],
        decision_ts=200,
        live_pending_cancel_order_ids=set(),
    )

    assert before_release.buy is retention_overlay
    assert before_release.buy.req == 0
    assert before_release.buy.cancellable is True
    assert after_release.buy is engine_order
    assert after_release.buy.req == "cancel"
    assert after_release.buy.cancellable is False


def test_terminal_live_visibility_overlay_keeps_filled_order_until_live_absent() -> None:
    filled_order = OrderSnapshot(
        order_id=17,
        side="sell",
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.0,
        exec_qty=0.001,
        exec_price_tick=1001,
        status="filled",
        req="none",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=False,
    )
    cancel_overlays: dict[int, PendingLocalOrder] = {}
    retention_overlays: dict[int, PendingLocalOrder] = {}

    _update_terminal_live_visibility_overlays(
        pending_cancel_overlays=cancel_overlays,
        pending_cancel_retention_overlays=retention_overlays,
        order=filled_order,
        absent_after_seen_ts={17: 200},
        cancel_pending_ts={},
        live_pending_cancel_order_ids=set(),
        decision_ts=150,
        tick_size=0.1,
    )

    working = _live_visible_working_orders(
        WorkingOrders(buy=None, sell=None, extras=[]),
        [],
        [],
        list(retention_overlays.values()),
        decision_ts=150,
        live_pending_cancel_order_ids=set(),
    )

    assert cancel_overlays == {}
    assert working.sell is retention_overlays[17]
    assert working.sell.status == 1
    assert working.sell.req == 0
    assert working.sell.cancellable is True

    _update_terminal_live_visibility_overlays(
        pending_cancel_overlays=cancel_overlays,
        pending_cancel_retention_overlays=retention_overlays,
        order=filled_order,
        absent_after_seen_ts={17: 200},
        cancel_pending_ts={},
        live_pending_cancel_order_ids=set(),
        decision_ts=200,
        tick_size=0.1,
    )

    assert retention_overlays == {}


def test_terminal_live_visibility_overlay_preserves_live_req_new_state() -> None:
    filled_order = OrderSnapshot(
        order_id=17,
        side="sell",
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.0,
        exec_qty=0.001,
        exec_price_tick=1001,
        status="filled",
        req="none",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=False,
    )
    live_state = PendingLocalOrder(
        order_id=17,
        side=SELL,
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=140,
        status=1,
        req=1,
        cancellable=False,
    )
    cancel_overlays: dict[int, PendingLocalOrder] = {}
    retention_overlays: dict[int, PendingLocalOrder] = {}

    _update_terminal_live_visibility_overlays(
        pending_cancel_overlays=cancel_overlays,
        pending_cancel_retention_overlays=retention_overlays,
        order=filled_order,
        absent_after_seen_ts={17: 200},
        cancel_pending_ts={},
        live_pending_cancel_order_ids=set(),
        live_order_state_by_id={17: live_state},
        decision_ts=150,
        tick_size=0.1,
    )

    working = _live_visible_working_orders(
        WorkingOrders(buy=None, sell=None, extras=[]),
        [],
        [],
        list(retention_overlays.values()),
        decision_ts=150,
        live_pending_cancel_order_ids=set(),
    )

    assert working.sell is retention_overlays[17]
    assert working.sell.req == 1
    assert working.sell.cancellable is False
    assert working.sell.local_timestamp == 140


def test_terminal_live_visibility_overlay_uses_cancel_state_only_when_live_pending() -> None:
    canceled_order = OrderSnapshot(
        order_id=17,
        side="buy",
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        exec_qty=0.0,
        exec_price_tick=0,
        status="canceled",
        req="cancel",
        time_in_force="gtx",
        exch_timestamp=123,
        local_timestamp=120,
        cancellable=False,
    )
    cancel_overlays: dict[int, PendingLocalOrder] = {}
    retention_overlays: dict[int, PendingLocalOrder] = {}

    _update_terminal_live_visibility_overlays(
        pending_cancel_overlays=cancel_overlays,
        pending_cancel_retention_overlays=retention_overlays,
        order=canceled_order,
        absent_after_seen_ts={17: 250},
        cancel_pending_ts={17: 140},
        live_pending_cancel_order_ids={17},
        decision_ts=150,
        tick_size=0.1,
    )

    pending_visible = _live_visible_working_orders(
        WorkingOrders(buy=None, sell=None, extras=[]),
        [],
        list(cancel_overlays.values()),
        list(retention_overlays.values()),
        decision_ts=150,
        live_pending_cancel_order_ids={17},
    )
    pending_hidden = _live_visible_working_orders(
        WorkingOrders(buy=None, sell=None, extras=[]),
        [],
        list(cancel_overlays.values()),
        list(retention_overlays.values()),
        decision_ts=150,
        live_pending_cancel_order_ids=set(),
    )

    assert pending_visible.buy.req == 4
    assert pending_visible.buy.cancellable is False
    assert pending_hidden.buy.req == 0
    assert pending_hidden.buy.cancellable is True


def test_live_visible_working_orders_prefers_cancel_or_retention_overlay_by_live_pending_state() -> None:
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
    cancel_overlay = PendingLocalOrder(
        order_id=7,
        side=BUY,
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=121,
        req=4,
        cancellable=False,
        visible_ts=150,
        release_ts=300,
    )
    retention_overlay = PendingLocalOrder(
        order_id=7,
        side=BUY,
        price=100.0,
        price_tick=1000,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=121,
        req=0,
        cancellable=True,
        visible_ts=150,
        release_ts=300,
    )

    pending_visible = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [cancel_overlay],
        [retention_overlay],
        decision_ts=200,
        live_pending_cancel_order_ids={7},
    )
    pending_hidden = _live_visible_working_orders(
        WorkingOrders(buy=engine_order, sell=None, extras=[]),
        [],
        [cancel_overlay],
        [retention_overlay],
        decision_ts=200,
        live_pending_cancel_order_ids=set(),
    )

    assert pending_visible.buy.req == 4
    assert pending_visible.buy.cancellable is False
    assert pending_hidden.buy.req == 0
    assert pending_hidden.buy.cancellable is True


def test_live_state_visibility_overlay_refreshes_stale_retention_state() -> None:
    stale_retention = PendingLocalOrder(
        order_id=17,
        side=SELL,
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=120,
        status=1,
        req=4,
        cancellable=False,
        visible_ts=120,
        release_ts=250,
    )
    stale_cancel = PendingLocalOrder(
        order_id=17,
        side=SELL,
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=120,
        status=1,
        req=4,
        cancellable=False,
        visible_ts=120,
        release_ts=250,
    )
    live_state = PendingLocalOrder(
        order_id=17,
        side=SELL,
        price=100.1,
        price_tick=1001,
        qty=0.001,
        leaves_qty=0.001,
        local_timestamp=150,
        status=1,
        req=0,
        cancellable=True,
    )
    cancel_overlays = {17: stale_cancel}
    retention_overlays = {17: stale_retention}

    _sync_live_state_visibility_overlays(
        pending_cancel_overlays=cancel_overlays,
        pending_cancel_retention_overlays=retention_overlays,
        live_order_state_by_id={17: live_state},
        absent_after_seen_ts={17: 250},
        live_pending_cancel_order_ids=set(),
        decision_ts=150,
    )

    assert cancel_overlays == {}
    assert retention_overlays[17].req == 0
    assert retention_overlays[17].cancellable is True
    assert retention_overlays[17].local_timestamp == 150
    assert retention_overlays[17].release_ts == 250


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
        bid_top5_ticks="1000|999|998|997|996",
        bid_top5_qtys="1.0|0.5|0.0|0.0|0.0",
        ask_top5_ticks="1010|1011|1012|1013|1014",
        ask_top5_qtys="1.0|0.5|0.0|0.0|0.0",
        market_view_source="audit_overlay",
        top5_source="replay_depth",
        market_overlay_source="audit",
        top5_overlay_source="",
        book_view_ts_local=100,
        book_view_ts_exch=90,
        book_view_feed_latency_ns=10,
        book_view_stale_ms=0.01,
        top5_depth_best_bid_tick=1000,
        top5_depth_best_ask_tick=1010,
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
    assert row["bid_top5_ticks"] == "1000|999|998|997|996"
    assert row["ask_top5_qtys"] == "1.0|0.5|0.0|0.0|0.0"
    assert row["market_view_source"] == "audit_overlay"
    assert row["top5_source"] == "replay_depth"
    assert row["market_overlay_source"] == "audit"
    assert row["top5_overlay_source"] == ""
    assert row["book_view_ts_local"] == 100
    assert row["book_view_ts_exch"] == 90
    assert row["book_view_feed_latency_ns"] == 10
    assert row["book_view_stale_ms"] == 0.01
    assert row["top5_depth_best_bid_tick"] == 1000
    assert row["top5_depth_best_ask_tick"] == 1010
    assert row["quote_update_intent"] == ""
    assert row["quote_update_action"] == ""
    assert row["quote_update_reason"] == ""
    assert row["token_bucket_state"] == ""
    assert row["inventory_request_id"] == ""


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
        bid_top5_ticks="1000|999|998|997|996",
        bid_top5_qtys="1.0|0.5|0.0|0.0|0.0",
        ask_top5_ticks="1010|1011|1012|1013|1014",
        ask_top5_qtys="1.0|0.5|0.0|0.0|0.0",
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


def test_format_top5_levels_serializes_ticks_and_qtys() -> None:
    class _Depth:
        best_bid_tick = 1000
        best_ask_tick = 1010
        roi_lb_tick = 900
        roi_ub_tick = 1100

        @staticmethod
        def bid_qty_at_tick(tick: int) -> float:
            return {1000: 1.0, 999: 0.5}.get(tick, 0.0)

        @staticmethod
        def ask_qty_at_tick(tick: int) -> float:
            return {1010: 2.0, 1011: 0.25}.get(tick, 0.0)

    bid_ticks, bid_qtys, ask_ticks, ask_qtys = format_top5_levels(_Depth())

    assert bid_ticks == "1000|999|998|997|996"
    assert bid_qtys == "1.0|0.5|0.0|0.0|0.0"
    assert ask_ticks == "1010|1011|1012|1013|1014"
    assert ask_qtys == "2.0|0.25|0.0|0.0|0.0"


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


def test_live_local_feed_compat_clear_then_snapshot_removes_prior_levels() -> None:
    def _local_only(ts: int, ev: int, px: float, qty: float) -> tuple[int, int, int, float, float, int, int, float]:
        return (ev | LOCAL_EVENT, ts, ts, px, qty, 0, 0, 0.0)

    rows = np.asarray(
        [
            _event(1_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 1.0),
            _event(1_000_000_000, DEPTH_EVENT | SELL_EVENT, 100.2, 1.0),
            _local_only(1_050_000_000, DEPTH_EVENT | BUY_EVENT, 99.9, 1.0),
            _local_only(1_100_000_000, DEPTH_CLEAR_EVENT | BUY_EVENT, 0.0, 0.0),
            _local_only(1_100_000_000, DEPTH_SNAPSHOT_EVENT | BUY_EVENT, 100.0, 2.0),
            _local_only(1_100_000_000, DEPTH_SNAPSHOT_EVENT | SELL_EVENT, 100.2, 3.0),
            _event(1_200_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 1.5),
        ],
        dtype=event_dtype,
    )

    fused, _stats = _live_local_feed_compat_data(rows, tick_size=0.1, lot_size=0.001)
    local_depth_rows = [
        (round(float(row["px"]), 1), float(row["qty"]))
        for row in fused
        if (
            int(row["ev"]) & LOCAL_EVENT == LOCAL_EVENT
            and int(row["ev"]) & EXCH_EVENT == 0
            and (int(row["ev"]) & 0xff) == DEPTH_EVENT
        )
    ]

    assert (99.9, 0.0) in local_depth_rows


def test_live_local_feed_fuser_drops_crossed_opposite_levels() -> None:
    fuser = _LiveLocalFeedFuser(tick_size=0.1, lot_size=0.001)
    rows = np.asarray(
        [
            _event(1_000_000_000, DEPTH_EVENT | SELL_EVENT, 100.2, 1.0),
            _event(1_000_000_001, DEPTH_EVENT | SELL_EVENT, 100.3, 1.0),
            _event(1_000_000_002, DEPTH_EVENT | BUY_EVENT, 100.0, 1.0),
            _event(1_000_000_003, DEPTH_EVENT | BUY_EVENT, 100.2, 1.0),
        ],
        dtype=event_dtype,
    )

    assert fuser.update_ask(rows[0]) is True
    assert fuser.update_ask(rows[1]) is True
    assert fuser.update_bid(rows[2]) is True
    assert fuser.best_ask_tick == 1002

    assert fuser.update_bid(rows[3]) is True
    assert fuser.best_bid_tick == 1002
    assert fuser.best_ask_tick is None or fuser.best_ask_tick > 1002
    assert 1002 not in fuser.ask_depth




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
        "lag_gate_startup_exclusion_ns": 0,
        "strict_lag_gate": False,
        "lag_gate_action": "report",
        "trigger_ts_source": "ts_local",
        "feed_latency_column": "feed_latency_ns",
        "market_state_overlay": "off",
        "strategy_position_overlay": "audit",
        "working_order_overlay": "off",
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
            "lag_gate_startup_exclusion_ms": 3000.0,
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
    assert cfg["lag_gate_startup_exclusion_ns"] == 3_000_000_000
    assert cfg["strict_lag_gate"] is True
    assert cfg["lag_gate_action"] == "fail"
    assert cfg["trigger_ts_source"] == "feed_local"
    assert cfg["feed_latency_column"] == "feed_latency_ns"
    assert cfg["market_state_overlay"] == "off"
    assert cfg["strategy_position_overlay"] == "audit"
    assert cfg["working_order_overlay"] == "off"


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


def test_backtest_cadence_config_accepts_off_strategy_position_overlay() -> None:
    cfg = _backtest_cadence_config(
        {
            "backtest_cadence": {
                "mode": "audit_replay",
                "strategy_position_overlay": "off",
            }
        }
    )

    assert cfg["strategy_position_overlay"] == "off"


def test_backtest_cadence_config_accepts_audit_working_order_overlay() -> None:
    cfg = _backtest_cadence_config(
        {
            "backtest_cadence": {
                "mode": "audit_replay",
                "working_order_overlay": "audit",
            }
        }
    )

    assert cfg["working_order_overlay"] == "audit"


def test_backtest_cadence_config_rejects_unknown_working_order_overlay() -> None:
    with pytest.raises(ValueError, match="Unsupported backtest_cadence.working_order_overlay"):
        _backtest_cadence_config(
            {
                "backtest_cadence": {
                    "mode": "audit_replay",
                    "working_order_overlay": "unknown",
                }
            }
        )


def test_backtest_cadence_config_rejects_unknown_strategy_position_overlay() -> None:
    with pytest.raises(ValueError, match="Unsupported backtest_cadence.strategy_position_overlay"):
        _backtest_cadence_config(
            {
                "backtest_cadence": {
                    "mode": "audit_replay",
                    "strategy_position_overlay": "unknown",
                }
            }
        )


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
