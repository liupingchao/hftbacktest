from __future__ import annotations

from audit_schema import AUDIT_FIELDS, REQUIRED_ALIGNMENT_FIELDS
from strategy_core import (
    Action,
    ExtraOrder,
    GreekValues,
    LiveSafetyConfig,
    LiveSafetyState,
    QuoteThrottleConfig,
    QuoteThrottleState,
    WorkingOrders,
    build_audit_row,
    decide_actions,
    evaluate_live_safety,
    format_actions,
    format_working_order_diagnostics,
    inventory_score_from_risk,
    is_position_limit_reached,
    is_pure_cancel_extra,
    should_throttle_quote_update,
)


class _Order:
    def __init__(self, order_id: int, side: int, price_tick: int, cancellable: bool = True) -> None:
        self.order_id = order_id
        self.side = side
        self.price_tick = price_tick
        self.cancellable = cancellable
        self.status = 0


def test_position_limit_uses_max_position_qty_when_configured() -> None:
    risk = {"max_position_qty": 0.003, "max_notional_pos": 1_000_000.0}

    assert is_position_limit_reached(position=0.003, position_notional=233.0, risk=risk) is True
    assert is_position_limit_reached(position=0.002, position_notional=155.0, risk=risk) is False


def test_position_limit_falls_back_to_max_notional_pos() -> None:
    risk = {"max_notional_pos": 250.0}

    assert is_position_limit_reached(position=0.002, position_notional=300.0, risk=risk) is True
    assert is_position_limit_reached(position=0.004, position_notional=200.0, risk=risk) is False


def test_inventory_score_uses_max_position_qty_when_configured() -> None:
    risk = {"max_position_qty": 0.004, "max_notional_pos": 1_000_000.0}

    assert inventory_score_from_risk(position=0.001, position_notional=80.0, risk=risk) == 0.75
    assert inventory_score_from_risk(position=0.004, position_notional=320.0, risk=risk) == 0.0


def test_inventory_score_falls_back_to_max_notional_pos() -> None:
    risk = {"max_notional_pos": 250.0}

    assert inventory_score_from_risk(position=0.001, position_notional=125.0, risk=risk) == 0.5



    order_id, action = format_actions([])

    assert order_id == ""
    assert action == "keep"


def test_format_actions_joins_order_ids_and_kind_side_names() -> None:
    actions = [
        Action("cancel", "buy", 1, 0.0, 0.0),
        Action("submit", "buy", 2, 100.0, 0.01),
    ]

    order_id, action = format_actions(actions)

    assert order_id == "1|2"
    assert action == "cancel_buy|submit_buy"


def test_format_working_order_diagnostics_with_primary_and_extras() -> None:
    class Order:
        def __init__(self, order_id: int, price_tick: int) -> None:
            self.order_id = order_id
            self.price_tick = price_tick

    working = WorkingOrders(
        buy=Order(11, 100),
        sell=Order(22, 110),
        extras=[
            ExtraOrder(order_id=33, side="buy", price_tick=101),
            ExtraOrder(order_id=44, side="sell", price_tick=111),
        ],
    )

    diagnostics = format_working_order_diagnostics(working)

    assert diagnostics == {
        "working_buy_order_id": "11",
        "working_sell_order_id": "22",
        "extra_order_ids": "33|44",
        "extra_order_sides": "buy|sell",
        "extra_order_price_ticks": "101|111",
    }


def test_evaluate_live_safety_ok_within_position_tolerance() -> None:
    cfg = LiveSafetyConfig(enabled=True, position_tolerance=0.003, open_order_check=True)

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.045,
        local_position=0.043,
        rest_open_order_count=1,
        local_open_order_count=1,
        rest_error="",
    )

    assert state.safety_status == "ok"
    assert state.position_mismatch == 0.002
    assert state.rest_position == 0.045
    assert state.rest_open_order_count == 1
    assert state.local_open_order_count == 1


def test_evaluate_live_safety_flags_position_mismatch_over_tolerance() -> None:
    cfg = LiveSafetyConfig(
        enabled=True,
        position_tolerance=0.003,
        open_order_check=True,
        position_mismatch_confirmations=1,
    )

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.045,
        local_position=-0.029,
        rest_open_order_count=0,
        local_open_order_count=1,
        rest_error="",
    )

    assert state.safety_status == "position_mismatch"
    assert state.position_mismatch == 0.074


def test_evaluate_live_safety_flags_open_order_mismatch() -> None:
    cfg = LiveSafetyConfig(
        enabled=True,
        position_tolerance=0.003,
        open_order_check=True,
        open_order_mismatch_confirmations=1,
    )

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=1,
        rest_error="",
    )

    assert state.safety_status == "open_order_mismatch"
    assert state.rest_open_order_count == 0
    assert state.local_open_order_count == 1


def test_evaluate_live_safety_reports_rest_error() -> None:
    cfg = LiveSafetyConfig(enabled=True, position_tolerance=0.003, open_order_check=True)

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=0,
        rest_error="timeout",
    )

    assert state.safety_status == "rest_error"


def test_evaluate_live_safety_position_mismatch_pending_before_confirmation() -> None:
    cfg = LiveSafetyConfig(
        enabled=True,
        position_tolerance=0.003,
        open_order_check=True,
        position_mismatch_confirmations=2,
    )

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.004,
        local_position=0.0,
        rest_open_order_count=1,
        local_open_order_count=1,
        rest_error="",
        position_mismatch_count=0,
    )

    assert state.safety_status == "position_mismatch_pending"
    assert state.position_mismatch == 0.004


def test_evaluate_live_safety_position_mismatch_after_confirmation() -> None:
    cfg = LiveSafetyConfig(
        enabled=True,
        position_tolerance=0.003,
        open_order_check=True,
        position_mismatch_confirmations=2,
    )

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.004,
        local_position=0.0,
        rest_open_order_count=1,
        local_open_order_count=1,
        rest_error="",
        position_mismatch_count=1,
    )

    assert state.safety_status == "position_mismatch"


def test_evaluate_live_safety_config_parses_position_dampers() -> None:
    cfg = LiveSafetyConfig.from_config({
        "position_mismatch_confirmations": 3,
        "position_mismatch_pause_trading": False,
        "use_rest_position_for_strategy": False,
    })

    assert cfg.position_mismatch_confirmations == 3
    assert cfg.position_mismatch_pause_trading is False
    assert cfg.use_rest_position_for_strategy is False



    cfg = LiveSafetyConfig(
        enabled=True,
        position_tolerance=0.003,
        open_order_check=True,
        open_order_mismatch_confirmations=2,
    )

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=1,
        rest_error="",
        ts_local=2_000_000_000,
        last_api_ts=0,
        open_order_mismatch_count=0,
    )

    assert state.safety_status == "open_order_mismatch_pending"


def test_evaluate_live_safety_open_order_mismatch_after_confirmation() -> None:
    cfg = LiveSafetyConfig(
        enabled=True,
        position_tolerance=0.003,
        open_order_check=True,
        open_order_mismatch_confirmations=2,
    )

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=0,
        local_open_order_count=1,
        rest_error="",
        ts_local=2_000_000_000,
        last_api_ts=0,
        open_order_mismatch_count=1,
    )

    assert state.safety_status == "open_order_mismatch"


def test_evaluate_live_safety_open_order_mismatch_uses_api_grace_window() -> None:
    cfg = LiveSafetyConfig(
        enabled=True,
        position_tolerance=0.003,
        open_order_check=True,
        open_order_grace_ns=1_000_000_000,
        open_order_mismatch_confirmations=2,
    )

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.0,
        local_position=0.0,
        rest_open_order_count=1,
        local_open_order_count=2,
        rest_error="",
        ts_local=2_500_000_000,
        last_api_ts=2_000_000_000,
        open_order_mismatch_count=1,
    )

    assert state.safety_status == "open_order_grace"


def test_evaluate_live_safety_config_parses_open_order_dampers() -> None:
    cfg = LiveSafetyConfig.from_config({
        "open_order_grace_ms": 250,
        "open_order_mismatch_confirmations": 3,
    })

    assert cfg.open_order_grace_ns == 250_000_000
    assert cfg.open_order_mismatch_confirmations == 3


def test_build_audit_row_includes_live_safety_fields() -> None:
    kwargs = _base_audit_kwargs()
    kwargs.update({
        "rest_position": 0.045,
        "position_mismatch": 0.074,
        "rest_open_order_count": 0,
        "local_open_order_count": 1,
        "safety_status": "position_mismatch",
    })

    row = build_audit_row(**kwargs)

    assert row["rest_position"] == 0.045
    assert row["position_mismatch"] == 0.074
    assert row["rest_open_order_count"] == 0
    assert row["local_open_order_count"] == 1
    assert row["safety_status"] == "position_mismatch"


def test_is_pure_cancel_extra_false_for_empty_actions() -> None:
    assert is_pure_cancel_extra([]) is False


def test_is_pure_cancel_extra_true_for_only_cancel_extra_actions() -> None:
    actions = [
        Action(kind="cancel", side="extra", order_id=1, price=0.0, qty=0.0),
        Action(kind="cancel", side="extra", order_id=2, price=0.0, qty=0.0),
    ]
    assert is_pure_cancel_extra(actions) is True


def test_is_pure_cancel_extra_false_for_mixed_actions() -> None:
    actions = [
        Action(kind="cancel", side="extra", order_id=1, price=0.0, qty=0.0),
        Action(kind="submit", side="buy", order_id=2, price=100.0, qty=0.01),
    ]
    assert is_pure_cancel_extra(actions) is False


def test_pure_cancel_extra_bypasses_interval_condition_but_quote_update_does_not() -> None:
    cancel_extra = [Action(kind="cancel", side="extra", order_id=1, price=0.0, qty=0.0)]
    quote_update = [Action(kind="cancel", side="buy", order_id=2, price=100.0, qty=0.01)]

    assert is_pure_cancel_extra(cancel_extra) is True
    assert is_pure_cancel_extra(quote_update) is False

    cfg = QuoteThrottleConfig(enabled=False, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "buy", 1, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_010_000_000,
        target_bid_tick=101,
        target_ask_tick=111,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == ""


def test_throttle_blocks_small_update_inside_interval() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "buy", 1, 0.0, 0.0), Action("submit", "buy", 2, 10.1, 1.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_050_000_000,
        target_bid_tick=101,
        target_ask_tick=111,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == "min_quote_update_interval"


def test_throttle_allows_update_outside_interval() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "buy", 1, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_150_000_000,
        target_bid_tick=101,
        target_ask_tick=111,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == ""


def test_throttle_allows_large_tick_move_inside_interval() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "buy", 1, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_050_000_000,
        target_bid_tick=103,
        target_ask_tick=111,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == ""


def test_throttle_does_not_block_cancel_extra() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "extra", 99, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_010_000_000,
        target_bid_tick=100,
        target_ask_tick=110,
        planned_actions=actions,
        pos_limit=False,
    )

    assert reason == ""


def test_throttle_does_not_block_pos_limit_actions() -> None:
    cfg = QuoteThrottleConfig(enabled=True, min_interval_ns=100_000_000, min_move_ticks=2)
    state = QuoteThrottleState(last_sent_api_ts=1_000_000_000, last_sent_target_bid_tick=100, last_sent_target_ask_tick=110)
    actions = [Action("cancel", "sell", 1, 0.0, 0.0)]

    reason = should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=1_010_000_000,
        target_bid_tick=100,
        target_ask_tick=110,
        planned_actions=actions,
        pos_limit=True,
    )

    assert reason == ""


def test_quote_throttle_config_defaults_disabled() -> None:
    cfg = QuoteThrottleConfig.from_config({})

    assert cfg.enabled is False
    assert cfg.min_interval_ns == 100_000_000
    assert cfg.min_move_ticks == 2


def test_quote_throttle_config_clamps_negative_values() -> None:
    cfg = QuoteThrottleConfig.from_config({
        "quote_throttle_enabled": True,
        "min_quote_update_interval_ms": -1,
        "min_quote_move_ticks": -5,
    })

    assert cfg.enabled is True
    assert cfg.min_interval_ns == 0
    assert cfg.min_move_ticks == 0


def test_decide_actions_same_cycle_replace_when_two_phase_disabled() -> None:
    working = WorkingOrders(
        buy=_Order(10, 1, 100),
        sell=None,
        extras=[],
    )

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=105,
        target_ask_tick=110,
        qty=0.01,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=20,
        two_phase_replace_enabled=False,
    )

    assert format_actions(actions) == ("10|20|21", "cancel_buy|submit_buy|submit_sell")
    assert next_order_id == 22


def test_decide_actions_cancel_only_when_two_phase_enabled() -> None:
    working = WorkingOrders(
        buy=_Order(10, 1, 100),
        sell=None,
        extras=[],
    )

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=105,
        target_ask_tick=110,
        qty=0.01,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
    )

    assert format_actions(actions) == ("10|20", "cancel_buy|submit_sell")
    assert next_order_id == 21


def test_decide_actions_submits_when_side_absent_with_two_phase_enabled() -> None:
    working = WorkingOrders(buy=None, sell=None, extras=[])

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=105,
        target_ask_tick=110,
        qty=0.01,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
    )

    assert format_actions(actions) == ("20|21", "submit_buy|submit_sell")
    assert next_order_id == 22


def test_decide_actions_extras_keep_priority_with_two_phase_enabled() -> None:
    working = WorkingOrders(
        buy=_Order(10, 1, 100),
        sell=None,
        extras=[ExtraOrder(order_id=99, side="buy", price_tick=101)],
    )

    actions, next_order_id = decide_actions(
        working=working,
        target_bid_tick=105,
        target_ask_tick=110,
        qty=0.01,
        tick_size=0.1,
        pos_limit=False,
        position_notional=0.0,
        next_order_id=20,
        two_phase_replace_enabled=True,
    )

    assert format_actions(actions) == ("99", "cancel_extra")
    assert next_order_id == 20


def _base_audit_kwargs() -> dict[str, object]:
    return {
        "run_id": "test",
        "symbol": "BTCUSDT",
        "strategy_seq": 1,
        "ts_local": 1_000,
        "ts_exch": 900,
        "action_order_id": "",
        "action_name": "keep",
        "planned_order_id": "1|2",
        "planned_action": "cancel_buy|submit_buy",
        "throttle_reason": "api_interval",
        "reject_reason": "api_interval_guard",
        "req_ts": 0,
        "exch_ts": 0,
        "resp_ts": 0,
        "entry_latency_ns": 0,
        "resp_latency_ns": 0,
        "predicted_entry_ns": 0,
        "best_bid": 100.0,
        "best_ask": 100.1,
        "mid": 100.05,
        "fair": 100.0,
        "reservation": 100.0,
        "half_spread": 1.0,
        "position": 0.0,
        "auditlatency_ms": 0.0,
        "dropped_by_latency": False,
        "dropped_by_api_limit": True,
        "pos_limit": False,
        "impact_cost_val": 0.0,
        "spread_bps": 1.0,
        "vol_bps": 0.0,
        "inventory_score": 1.0,
        "feed_latency_ns": 0,
        "latency_signal_ns": 0,
        "bid_size": 1.0,
        "ask_size": 1.0,
        "greek_values": GreekValues(0.0, 0.0, 0.0, 0.0),
        "greek_adjustment": 0.0,
        "target_bid_tick": 1000,
        "target_ask_tick": 1001,
        "working_bid_tick": 999,
        "working_ask_tick": 1002,
        "working_buy_order_id": "",
        "working_sell_order_id": "",
        "extra_order_ids": "",
        "extra_order_sides": "",
        "extra_order_price_ticks": "",
        "rest_position": 0.0,
        "position_mismatch": 0.0,
        "rest_open_order_count": 0,
        "local_open_order_count": 0,
        "safety_status": "ok",
    }


def test_audit_schema_contains_planned_action_fields() -> None:
    assert "planned_order_id" in AUDIT_FIELDS
    assert "planned_action" in AUDIT_FIELDS
    assert "throttle_reason" in AUDIT_FIELDS
    assert "planned_order_id" in REQUIRED_ALIGNMENT_FIELDS
    assert "planned_action" in REQUIRED_ALIGNMENT_FIELDS
    assert "throttle_reason" in REQUIRED_ALIGNMENT_FIELDS


def test_build_audit_row_for_quote_throttle_reject() -> None:
    kwargs = _base_audit_kwargs()
    kwargs.update({
        "planned_order_id": "1|2",
        "planned_action": "cancel_buy|submit_buy",
        "action_order_id": "",
        "action_name": "keep",
        "reject_reason": "quote_throttle",
        "throttle_reason": "min_quote_update_interval",
        "dropped_by_api_limit": True,
    })

    row = build_audit_row(**kwargs)

    assert row["planned_action"] == "cancel_buy|submit_buy"
    assert row["action"] == "keep"
    assert row["reject_reason"] == "quote_throttle"
    assert row["throttle_reason"] == "min_quote_update_interval"




def test_build_audit_row_preserves_separate_planned_and_executed_fields() -> None:
    kwargs = _base_audit_kwargs()
    kwargs.update({
        "planned_order_id": "1|2|3",
        "planned_action": "cancel_buy|submit_buy|submit_sell",
        "action_order_id": "1|2",
        "action_name": "cancel_buy|submit_buy",
        "reject_reason": "",
        "throttle_reason": "",
        "dropped_by_api_limit": False,
    })

    row = build_audit_row(**kwargs)

    assert row["planned_order_id"] == "1|2|3"
    assert row["planned_action"] == "cancel_buy|submit_buy|submit_sell"
    assert row["order_id"] == "1|2"
    assert row["action"] == "cancel_buy|submit_buy"
    assert row["reject_reason"] == ""
    assert row["throttle_reason"] == ""
    assert row["extra_order_ids"] == ""



def test_evaluate_live_safety_disabled() -> None:
    cfg = LiveSafetyConfig(enabled=False, position_tolerance=0.003, open_order_check=True)

    state = evaluate_live_safety(
        cfg=cfg,
        rest_position=0.045,
        local_position=-0.029,
        rest_open_order_count=0,
        local_open_order_count=1,
        rest_error="",
    )

    assert state.safety_status == "safety_disabled"
