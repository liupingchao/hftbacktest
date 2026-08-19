from __future__ import annotations

from collections import Counter

import pytest

import cross_exchange_liquidity_response_trigger as trigger


def _timeline_state(
    ts_ns: int,
    *,
    bid_px: float = 99.0,
    bid_qty: float = 10.0,
    ask_px: float = 100.0,
    ask_qty: float = 10.0,
    fast_source_ts_ns: int | None = None,
) -> trigger.TimelineState:
    return trigger.TimelineState(
        ts_ns=ts_ns,
        binance_bid_px=(bid_px, bid_px - 0.01),
        binance_bid_qty=(bid_qty, 2.0),
        binance_ask_px=(ask_px, ask_px + 0.01),
        binance_ask_qty=(ask_qty, 2.0),
        fast_source_ts_ns=(
            ts_ns - 1 if fast_source_ts_ns is None else fast_source_ts_ns
        ),
        fast_age_ms=0.0,
        fast_bid_px=(98.0, 97.99),
        fast_bid_qty=(5.0, 5.0),
        fast_ask_px=(98.01, 98.02),
        fast_ask_qty=(5.0, 5.0),
    )


def _candidate(
    burst: list[dict[str, object]],
    timeline: list[trigger.TimelineState],
    *,
    bbo: list[trigger.BboState] | None = None,
    boundary_end_ns: int = 5_000_000_000,
) -> dict[str, object] | None:
    bbo = bbo or []
    audit, _, _ = trigger.candidate_from_burst(
        burst,
        timeline=timeline,
        timeline_ts=[state.ts_ns for state in timeline],
        bbo=bbo,
        bbo_ts=[state.ts_ns for state in bbo],
        boundary_end_ns=boundary_end_ns,
        candidate_seq=1,
        campaign_id="campaign",
        segment_id="segment_0001",
        profile_id="skhynix",
    )
    return audit


def test_frozen_contract_identity_and_audit_grammar() -> None:
    assert trigger.CONTRACT_VERSION == "cross_exchange_queue_shock_trigger_v1"
    assert len(trigger.AUDIT_FIELDS) == 36
    assert trigger.BURST_WINDOW_MS == 10
    assert trigger.IMPACT_THRESHOLD == 0.30
    assert trigger.CONFIRMATION_WINDOW_MS == 100
    assert trigger.TRADE_DRIVEN_THRESHOLD == 0.70
    assert trigger.MIXED_THRESHOLD == 0.30
    assert trigger.DEDUP_WINDOW_MS == 50


def test_burst_uses_fixed_first_trade_origin_and_zero_trade_terminates() -> None:
    counts: Counter[str] = Counter()
    trades = [
        {"ts_ns": 1_000_000_000, "side": "buy", "px": 100.0, "qty": 1.0},
        {"ts_ns": 1_009_000_000, "side": "buy", "px": 100.0, "qty": 1.0},
        {"ts_ns": 1_011_000_000, "side": "buy", "px": 100.0, "qty": 1.0},
        {"ts_ns": 1_012_000_000, "side": "sell", "px": 99.0, "qty": 1.0},
        {"ts_ns": 1_013_000_000, "side": "sell", "px": 0.0, "qty": 0.0},
        {"ts_ns": 1_014_000_000, "side": "sell", "px": 99.0, "qty": 1.0},
    ]

    bursts = list(trigger.iter_trade_bursts(trades, counts))

    assert [len(burst) for burst in bursts] == [2, 1, 1, 1]
    assert counts == {"economic_trade": 5, "zero_economic_trade": 1}


def test_candidate_threshold_is_exactly_ge_030() -> None:
    pre = _timeline_state(1_000_000_000)
    decision = _timeline_state(1_010_000_000, ask_qty=7.0)
    below = [
        {"ts_ns": 1_001_000_000, "side": "buy", "px": 100.0, "qty": 2.999}
    ]
    exact = [
        {"ts_ns": 1_001_000_000, "side": "buy", "px": 100.0, "qty": 3.0}
    ]

    assert _candidate(below, [pre, decision]) is None
    audit = _candidate(exact, [pre, decision])
    assert audit is not None
    assert audit["shock_impact_ratio"] == pytest.approx(0.3)


def test_pre_state_is_strictly_before_burst_start() -> None:
    burst_start = 1_001_000_000
    strict_pre = _timeline_state(burst_start - 1, ask_qty=10.0)
    same_time_future = _timeline_state(burst_start, ask_qty=1.0)
    burst = [
        {"ts_ns": burst_start, "side": "buy", "px": 100.0, "qty": 1.0}
    ]

    assert _candidate(burst, [strict_pre, same_time_future]) is None


def test_confirmation_window_is_inclusive_at_100ms() -> None:
    pre = _timeline_state(1_000_000_000)
    shock_ts = 1_001_000_000
    at_boundary = _timeline_state(shock_ts + 100_000_000, ask_qty=7.0)
    burst = [{"ts_ns": shock_ts, "side": "buy", "px": 100.0, "qty": 3.0}]

    audit = _candidate(burst, [pre, at_boundary])

    assert audit is not None
    assert audit["decision_ts_ns"] == at_boundary.ts_ns
    assert audit["rejection_reason"] == "missing_prior_hyperliquid_bbo"


def test_attribution_excludes_post_decision_trade_and_keeps_thresholds() -> None:
    pre = _timeline_state(1_000_000_000)
    decision = _timeline_state(1_005_000_000, ask_qty=0.0, ask_px=100.01)
    pre_bbo = trigger.BboState(
        ts_ns=999_999_999,
        bid_px=98.0,
        bid_qty=8.0,
        ask_px=98.01,
        ask_qty=10.0,
    )
    burst = [
        {"ts_ns": 1_001_000_000, "side": "buy", "px": 100.0, "qty": 7.0},
        {"ts_ns": 1_008_000_000, "side": "buy", "px": 100.0, "qty": 3.0},
    ]

    audit = _candidate(burst, [pre, decision], bbo=[pre_bbo])

    assert audit is not None
    assert audit["touch_trade_qty"] == pytest.approx(10.0)
    assert audit["touch_trade_qty_through_decision"] == pytest.approx(7.0)
    assert audit["post_decision_burst_trade_count"] == 1
    assert audit["trade_explained_ratio"] == pytest.approx(0.7)
    assert audit["attribution"] == "trade_driven"


def test_rejection_order_keeps_attribution_before_missing_bbo() -> None:
    pre = _timeline_state(1_000_000_000)
    decision = _timeline_state(1_005_000_000, ask_qty=6.0)
    burst = [{"ts_ns": 1_001_000_000, "side": "buy", "px": 100.0, "qty": 3.0}]

    audit = _candidate(burst, [pre, decision], bbo=[])

    assert audit is not None
    assert audit["attribution"] == "trade_driven"
    assert audit["rejection_reason"] == "missing_prior_hyperliquid_bbo"

    mixed_decision = _timeline_state(1_005_000_000, ask_qty=0.0, ask_px=100.01)
    mixed = _candidate(burst, [pre, mixed_decision], bbo=[])
    assert mixed is not None
    assert mixed["attribution"] == "mixed"
    assert mixed["rejection_reason"] == "attribution_mixed"


def test_rejection_order_keeps_fast_l2_before_response_room() -> None:
    pre = _timeline_state(1_000_000_000, fast_source_ts_ns=1_001_000_000)
    decision = _timeline_state(
        1_005_000_000,
        ask_qty=6.0,
        fast_source_ts_ns=1_001_000_000,
    )
    pre_bbo = trigger.BboState(
        ts_ns=999_999_999,
        bid_px=98.0,
        bid_qty=8.0,
        ask_px=98.01,
        ask_qty=10.0,
    )
    burst = [{"ts_ns": 1_001_000_000, "side": "buy", "px": 100.0, "qty": 4.0}]

    audit = _candidate(
        burst,
        [pre, decision],
        bbo=[pre_bbo],
        boundary_end_ns=1_100_000_000,
    )

    assert audit is not None
    assert audit["rejection_reason"] == "missing_prior_hyperliquid_fast_l2"


def test_primary_selection_reuse_precedes_same_side_dedup() -> None:
    state = trigger.PrimarySelectionState()
    first = {
        "rejection_reason": "",
        "aggressor_side": "buy",
        "decision_ts_ns": 1_100,
        "pre_best_px": 100.0,
        "shock_ts_ns": 1_000,
        "primary_episode": "false",
    }
    second = {
        **first,
        "shock_ts_ns": 1_010,
        "primary_episode": "false",
    }

    assert state.apply(first) is True
    assert state.apply(second) is False
    assert second["rejection_reason"] == "confirmation_reuse_excluded"


def test_detect_candidates_keeps_rejected_rows_and_contiguous_sequence() -> None:
    pre = _timeline_state(1_000_000_000)
    decision = _timeline_state(1_005_000_000, ask_qty=6.0)
    bursts = [
        [{"ts_ns": 1_001_000_000, "side": "buy", "px": 100.0, "qty": 4.0}],
        [{"ts_ns": 1_020_000_000, "side": "buy", "px": 100.0, "qty": 4.0}],
    ]

    detections = list(
        trigger.detect_candidates(
            bursts,
            timeline=[pre, decision],
            timeline_ts=[pre.ts_ns, decision.ts_ns],
            bbo=[],
            bbo_ts=[],
            boundary_end_ns=5_000_000_000,
            campaign_id="campaign",
            segment_id="segment_0001",
            profile_id="skhynix",
        )
    )

    assert [row.audit["candidate_seq"] for row in detections] == [1, 2]
    assert [row.audit["rejection_reason"] for row in detections] == [
        "missing_prior_hyperliquid_bbo",
        "no_depth_confirmation_within_100ms",
    ]
