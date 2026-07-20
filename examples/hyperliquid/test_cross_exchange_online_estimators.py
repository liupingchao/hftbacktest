from __future__ import annotations

import json
import time
from pathlib import Path

from examples.hyperliquid import cross_exchange_online_estimators as estimators
from examples.hyperliquid import cross_exchange_shared_signal_kernel as shared_kernel
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


BASE_TS = 1_700_000_000_000


def _event_book(
    event_time_ms: int,
    *,
    local_receive_time_ms: int | None = None,
    bid_px: float = 100.0,
    ask_px: float = 101.0,
    bid_depth_btc: float = 0.02,
    ask_depth_btc: float = 0.03,
) -> dict:
    return {
        "event_kind": "book",
        "event_time_ms": event_time_ms,
        "local_receive_time_ms": (
            event_time_ms
            if local_receive_time_ms is None
            else local_receive_time_ms
        ),
        "bid_px": bid_px,
        "ask_px": ask_px,
        "bid_depth_btc": bid_depth_btc,
        "ask_depth_btc": ask_depth_btc,
        "trade_px": "",
        "trade_size_btc": "",
        "aggressor_side": "",
        "trade_id": "",
    }


def _event_trade(
    event_time_ms: int,
    *,
    trade_px: float,
    trade_size_btc: float,
    aggressor_side: str,
    trade_id: str,
    local_receive_time_ms: int | None = None,
) -> dict:
    return {
        "event_kind": "trade",
        "event_time_ms": event_time_ms,
        "local_receive_time_ms": (
            event_time_ms
            if local_receive_time_ms is None
            else local_receive_time_ms
        ),
        "bid_px": "",
        "ask_px": "",
        "bid_depth_btc": "",
        "ask_depth_btc": "",
        "trade_px": trade_px,
        "trade_size_btc": trade_size_btc,
        "aggressor_side": aggressor_side,
        "trade_id": trade_id,
    }


def _confirmed_interval(
    *,
    attempt_key: str,
    attempt: int,
    side: str,
    quote_px: float,
    start_local_receive_time_ms: int = BASE_TS,
    end_local_receive_time_ms: int = BASE_TS + 3_000,
    resting_confirmed: bool = True,
    reconnect_count_start: int = 0,
    reconnect_count_end: int = 0,
    disconnect_count_start: int = 0,
    disconnect_count_end: int = 0,
    **extra: object,
) -> dict:
    return {
        "attempt_key": attempt_key,
        "attempt": attempt,
        "side": side,
        "quote_px": quote_px,
        "start_local_receive_time_ms": start_local_receive_time_ms,
        "end_local_receive_time_ms": end_local_receive_time_ms,
        "resting_confirmed": resting_confirmed,
        "reconnect_count_start": reconnect_count_start,
        "reconnect_count_end": reconnect_count_end,
        "disconnect_count_start": disconnect_count_start,
        "disconnect_count_end": disconnect_count_end,
        **extra,
    }


def _book(estimator: estimators.EventTimeOnlineEstimator, ts: int, bid: float, ask: float) -> None:
    result = estimator.ingest_book(
        event_time_ms=ts,
        local_receive_time_ms=ts,
        bid_px=bid,
        ask_px=ask,
        bid_depth_btc=0.02,
        ask_depth_btc=0.01,
    )
    assert result.accepted is True


def _exposure_rows(estimator: estimators.EventTimeOnlineEstimator, side: str) -> None:
    for index, (distance, arrivals) in enumerate(((1, 8), (2, 4), (3, 2), (4, 1))):
        mid = 100.0
        quote = mid - distance if side == "buy" else mid + distance
        result = estimator.observe_quote_exposure(
            exposure_id=f"{side}-{index}",
            side=side,
            quote_px=quote,
            reference_mid_px=mid,
            start_exchange_time_ms=BASE_TS + index * 1_000,
            end_exchange_time_ms=BASE_TS + index * 1_000 + 1_000,
            arrival_count=arrivals,
            arrival_volume_btc=arrivals * 0.001,
        )
        assert result["status"] == "accepted"


def test_event_time_bucket_dedupes_state_and_quarantines_invalid_ordering() -> None:
    estimator = estimators.EventTimeOnlineEstimator()

    _book(estimator, BASE_TS, 100.0, 101.0)
    duplicate = estimator.ingest_book(
        event_time_ms=BASE_TS + 100,
        local_receive_time_ms=BASE_TS + 100,
        bid_px=100.0,
        ask_px=101.0,
        bid_depth_btc=0.02,
        ask_depth_btc=0.01,
    )
    assert duplicate.accepted is False
    assert duplicate.reason == "same_bucket_state_deduped"

    _book(estimator, BASE_TS + 1_000, 100.0, 101.0)
    _trade = estimator.ingest_trade(
        event_time_ms=BASE_TS + 1_000,
        local_receive_time_ms=BASE_TS + 1_000,
        trade_px=101.0,
        trade_size_btc=0.001,
        aggressor_side="buy",
        trade_id="ordered-trade",
    )
    assert _trade.accepted is True
    out_of_order = estimator.ingest_trade(
        event_time_ms=BASE_TS + 500,
        local_receive_time_ms=BASE_TS + 500,
        trade_px=101.0,
        trade_size_btc=0.001,
        aggressor_side="buy",
    )
    assert out_of_order.accepted is False
    assert out_of_order.reason == "out_of_order_event"

    future = estimator.ingest_book(
        event_time_ms=BASE_TS + 10_000,
        local_receive_time_ms=BASE_TS,
        bid_px=100.0,
        ask_px=101.0,
        bid_depth_btc=0.02,
        ask_depth_btc=0.01,
    )
    assert future.accepted is False
    assert future.reason == "future_event_beyond_allowed_skew"
    assert {row["reason"] for row in estimator.quarantine_rows()} == {
        "out_of_order_event",
        "future_event_beyond_allowed_skew",
    }


def test_bucket_metrics_include_volatility_liquidity_toxicity_and_sweep() -> None:
    estimator = estimators.EventTimeOnlineEstimator()
    _book(estimator, BASE_TS, 100.0, 101.0)
    _book(estimator, BASE_TS + 1_000, 101.0, 102.0)
    buy = estimator.ingest_trade(
        event_time_ms=BASE_TS + 1_100,
        local_receive_time_ms=BASE_TS + 1_100,
        trade_px=102.0,
        trade_size_btc=0.02,
        aggressor_side="buy",
        trade_id="buy-1",
    )
    sell = estimator.ingest_trade(
        event_time_ms=BASE_TS + 1_200,
        local_receive_time_ms=BASE_TS + 1_200,
        trade_px=101.0,
        trade_size_btc=0.01,
        aggressor_side="sell",
        trade_id="sell-1",
    )
    assert buy.accepted is True
    assert sell.accepted is True

    row = estimator.bucket_rows()[1]
    assert row["spread_ticks"] == 1
    assert row["liquidity_depth_btc"] == 0.01
    assert row["trade_volume_btc"] == 0.03
    assert row["toxicity"] == round((0.02 - 0.01) / 0.03, 8)
    assert row["buy_adverse_volume_btc"] == 0.01
    assert row["sell_adverse_volume_btc"] == 0.02
    assert row["buy_sweep_depth_penetration"] == 2
    assert row["realized_volatility"] > 0


def test_side_intensity_fit_dynamic_candidate_and_rate_limit() -> None:
    estimator = estimators.EventTimeOnlineEstimator()
    _book(estimator, BASE_TS, 100.0, 101.0)
    _book(estimator, BASE_TS + 1_000, 101.0, 102.0)
    estimator.ingest_trade(
        event_time_ms=BASE_TS + 1_100,
        local_receive_time_ms=BASE_TS + 1_100,
        trade_px=102.0,
        trade_size_btc=0.001,
        aggressor_side="buy",
        trade_id="candidate-trade",
    )
    _exposure_rows(estimator, "buy")
    _exposure_rows(estimator, "sell")

    buy_fit = estimator.fit_intensity("buy")
    sell_fit = estimator.fit_intensity("sell")
    assert buy_fit["status"] == "pass"
    assert sell_fit["status"] == "pass"
    assert buy_fit["observation_count"] == 4
    assert buy_fit["A"] > 0
    assert buy_fit["k"] > 0
    assert buy_fit["confidence"] > 0
    assert buy_fit["A_confidence_low"] <= buy_fit["A"] <= buy_fit["A_confidence_high"]
    assert buy_fit["k_confidence_low"] <= buy_fit["k"] <= buy_fit["k_confidence_high"]

    candidate = estimator.dynamic_half_spread_candidate()
    assert candidate.status == "pass"
    assert candidate.bounded is True
    assert candidate.half_spread_ticks >= 0.5
    assert candidate.to_dict()["activation_enabled"] is False

    rate_limited = estimators.compute_dynamic_half_spread(
        base_half_spread_ticks=0.5,
        volatility=0.01,
        intensity_a=1.0,
        intensity_k=1.0,
        risk_aversion=1.0,
        inventory_ratio=0.0,
        liquidity_depth_btc=0.01,
        toxicity=1.0,
        previous_half_spread_ticks=0.5,
        elapsed_seconds=0.0,
    )
    assert rate_limited.rate_limited is True
    assert rate_limited.half_spread_ticks == 0.5
    lower_arrival_intensity = estimators.compute_dynamic_half_spread(
        base_half_spread_ticks=0.5,
        volatility=0.01,
        intensity_a=0.1,
        intensity_k=1.0,
        risk_aversion=1.0,
        inventory_ratio=0.0,
        liquidity_depth_btc=0.01,
        toxicity=0.5,
    )
    higher_arrival_intensity = estimators.compute_dynamic_half_spread(
        base_half_spread_ticks=0.5,
        volatility=0.01,
        intensity_a=10.0,
        intensity_k=1.0,
        risk_aversion=1.0,
        inventory_ratio=0.0,
        liquidity_depth_btc=0.01,
        toxicity=0.5,
    )
    assert lower_arrival_intensity.half_spread_ticks > higher_arrival_intensity.half_spread_ticks


def test_quote_exposure_uses_directional_trade_and_pre_trade_depth() -> None:
    estimator = estimators.EventTimeOnlineEstimator()
    _book(estimator, BASE_TS, 100.0, 101.0)
    estimator.ingest_trade(
        event_time_ms=BASE_TS + 100,
        local_receive_time_ms=BASE_TS + 100,
        trade_px=100.0,
        trade_size_btc=0.01,
        aggressor_side="sell",
        trade_id="hits-bid",
    )
    estimator.ingest_trade(
        event_time_ms=BASE_TS + 200,
        local_receive_time_ms=BASE_TS + 200,
        trade_px=101.0,
        trade_size_btc=0.02,
        aggressor_side="buy",
        trade_id="hits-ask",
    )

    result = estimator.observe_quote_exposure_from_public_flow(
        exposure_id="directional-buy",
        side="buy",
        quote_px=100.0,
        reference_mid_px=100.5,
        start_exchange_time_ms=BASE_TS,
        end_exchange_time_ms=BASE_TS + 1_000,
    )

    row = result["row"]
    assert row["arrival_count"] == 1
    assert row["arrival_volume_btc"] == 0.01
    assert row["pre_trade_side_depth_btc"] == 0.02
    assert row["max_sweep_depth_penetration"] == 0.5
    assert row["arrival_evidence_source"] == (
        "pre_trade_l2_directional_at_or_through_trade_and_exposure_interval"
    )


def test_confirmed_resting_exposure_builds_two_sides_across_partial_buckets() -> None:
    event_rows = [
        _event_book(BASE_TS + 100),
        _event_trade(
            BASE_TS + 400,
            trade_px=100.0,
            trade_size_btc=0.005,
            aggressor_side="sell",
            trade_id="buy-arrival-1",
        ),
        _event_book(
            BASE_TS + 900,
            bid_depth_btc=0.025,
            ask_depth_btc=0.035,
        ),
        _event_book(
            BASE_TS + 1_100,
            bid_px=101.0,
            ask_px=102.0,
            bid_depth_btc=0.03,
            ask_depth_btc=0.04,
        ),
        _event_trade(
            BASE_TS + 1_400,
            trade_px=102.0,
            trade_size_btc=0.006,
            aggressor_side="buy",
            trade_id="sell-arrival-1",
        ),
        _event_book(
            BASE_TS + 2_100,
            bid_px=102.0,
            ask_px=103.0,
            bid_depth_btc=0.04,
            ask_depth_btc=0.05,
        ),
        _event_trade(
            BASE_TS + 2_400,
            trade_px=100.0,
            trade_size_btc=0.007,
            aggressor_side="sell",
            trade_id="buy-arrival-2",
        ),
        _event_book(
            BASE_TS + 2_900,
            bid_px=102.0,
            ask_px=103.0,
            bid_depth_btc=0.045,
            ask_depth_btc=0.055,
        ),
    ]
    interval_rows = [
        _confirmed_interval(
            attempt_key="task:window_01:attempt_1",
            attempt=1,
            side="buy",
            quote_px=100.0,
        ),
        _confirmed_interval(
            attempt_key="task:window_01:attempt_2",
            attempt=2,
            side="sell",
            quote_px=102.0,
        ),
    ]

    rows, quarantine = estimators.build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=interval_rows,
    )

    assert quarantine == []
    assert len(rows) == 6
    assert {row["side"] for row in rows} == {"buy", "sell"}
    assert {
        (row["side"], row["start_exchange_time_ms"], row["duration_seconds"])
        for row in rows
    } == {
        ("buy", BASE_TS + 100, 0.9),
        ("buy", BASE_TS + 1_000, 1),
        ("buy", BASE_TS + 2_000, 0.9),
        ("sell", BASE_TS + 100, 0.9),
        ("sell", BASE_TS + 1_000, 1),
        ("sell", BASE_TS + 2_000, 0.9),
    }
    assert sum(row["arrival_count"] for row in rows if row["side"] == "buy") == 2
    assert sum(row["arrival_count"] for row in rows if row["side"] == "sell") == 1
    assert all(row["resting_confirmed"] is True for row in rows)
    assert all(
        row["source"] == "manager_confirmed_resting_event_time_bucket"
        for row in rows
    )
    assert all(set(row) == set(estimators.quote_exposure_fieldnames()) for row in rows)
    assert len(
        {
            (
                row["exposure_id"].rsplit(":bucket_", 1)[0],
                row["side"],
                int(row["start_exchange_time_ms"]) // 1_000,
            )
            for row in rows
        }
    ) == len(rows)


def test_confirmed_resting_exposure_rejects_unconfirmed_and_conflicting_attempts() -> None:
    event_rows = [
        _event_book(BASE_TS + 100),
        _event_book(BASE_TS + 900, bid_depth_btc=0.03),
    ]
    valid = _confirmed_interval(
        attempt_key="valid-buy",
        attempt=1,
        side="buy",
        quote_px=100.0,
    )
    unconfirmed = _confirmed_interval(
        attempt_key="rejected-sell",
        attempt=2,
        side="sell",
        quote_px=101.0,
        resting_confirmed=False,
        response_status_types="rejected",
    )
    conflict_a = _confirmed_interval(
        attempt_key="conflicting-buy",
        attempt=3,
        side="buy",
        quote_px=99.0,
    )
    conflict_b = {
        **conflict_a,
        "quote_px": 98.0,
    }

    rows, quarantine = estimators.build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=[valid, unconfirmed, conflict_a, conflict_b],
    )

    assert len(rows) == 1
    assert rows[0]["side"] == "buy"
    assert rows[0]["exposure_id"].startswith("valid-buy:")
    reasons = [row["reason"] for row in quarantine]
    assert "interval_not_confirmed_resting" in reasons
    assert reasons.count("duplicate_attempt_side_bucket") == 2
    assert all(
        set(row) == set(estimators.resting_exposure_quarantine_fieldnames())
        for row in quarantine
    )


def test_confirmed_resting_exposure_quarantines_bounds_continuity_and_no_events() -> None:
    event_rows = [
        _event_book(BASE_TS + 100),
        _event_book(BASE_TS + 900, bid_depth_btc=0.03),
    ]
    missing_start = _confirmed_interval(
        attempt_key="missing-start",
        attempt=1,
        side="buy",
        quote_px=100.0,
    )
    del missing_start["start_local_receive_time_ms"]
    inverted = _confirmed_interval(
        attempt_key="inverted",
        attempt=2,
        side="sell",
        quote_px=101.0,
        start_local_receive_time_ms=BASE_TS + 2_000,
        end_local_receive_time_ms=BASE_TS + 1_000,
    )
    reconnected = _confirmed_interval(
        attempt_key="reconnected",
        attempt=3,
        side="buy",
        quote_px=100.0,
        reconnect_count_end=1,
    )
    disconnected = _confirmed_interval(
        attempt_key="disconnected",
        attempt=4,
        side="sell",
        quote_px=101.0,
        disconnect_count_end=1,
    )
    no_events = _confirmed_interval(
        attempt_key="no-events",
        attempt=5,
        side="buy",
        quote_px=100.0,
        start_local_receive_time_ms=BASE_TS + 4_000,
        end_local_receive_time_ms=BASE_TS + 5_000,
    )

    rows, quarantine = estimators.build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=[
            missing_start,
            inverted,
            reconnected,
            disconnected,
            no_events,
        ],
    )

    assert rows == []
    assert {row["reason"] for row in quarantine} == {
        "invalid_local_receive_interval",
        "public_stream_continuity_changed",
        "no_public_event_inside_local_bounds",
    }


def test_confirmed_resting_exposure_dedupes_dense_book_and_trade_rows() -> None:
    sparse_rows = [
        _event_book(BASE_TS + 100),
        _event_trade(
            BASE_TS + 400,
            trade_px=100.0,
            trade_size_btc=0.01,
            aggressor_side="sell",
            trade_id="same-trade",
        ),
        _event_book(BASE_TS + 900, bid_depth_btc=0.03),
    ]
    dense_rows = [
        sparse_rows[0],
        _event_book(BASE_TS + 200),
        sparse_rows[1],
        _event_trade(
            BASE_TS + 500,
            trade_px=100.0,
            trade_size_btc=0.01,
            aggressor_side="sell",
            trade_id="same-trade",
        ),
        sparse_rows[2],
    ]
    interval_rows = [
        _confirmed_interval(
            attempt_key="dense-sparse",
            attempt=1,
            side="buy",
            quote_px=100.0,
        )
    ]

    sparse_exposure, sparse_quarantine = (
        estimators.build_confirmed_resting_exposure_rows(
            event_rows=sparse_rows,
            interval_rows=interval_rows,
        )
    )
    dense_exposure, dense_quarantine = (
        estimators.build_confirmed_resting_exposure_rows(
            event_rows=dense_rows,
            interval_rows=interval_rows,
        )
    )

    assert sparse_quarantine == []
    assert dense_quarantine == []
    assert dense_exposure == sparse_exposure
    assert dense_exposure[0]["arrival_count"] == 1
    assert dense_exposure[0]["arrival_volume_btc"] == 0.01


def test_confirmed_resting_exposure_dedupes_trade_ids_across_buckets() -> None:
    event_rows = [
        _event_book(BASE_TS + 100),
        _event_trade(
            BASE_TS + 400,
            trade_px=100.0,
            trade_size_btc=0.01,
            aggressor_side="sell",
            trade_id="replayed-across-buckets",
        ),
        _event_book(
            BASE_TS + 1_100,
            bid_depth_btc=0.03,
        ),
        _event_trade(
            BASE_TS + 1_400,
            trade_px=100.0,
            trade_size_btc=0.01,
            aggressor_side="sell",
            trade_id="replayed-across-buckets",
        ),
        _event_book(
            BASE_TS + 1_900,
            bid_depth_btc=0.04,
        ),
    ]

    rows, quarantine = estimators.build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=[
            _confirmed_interval(
                attempt_key="global-trade-id",
                attempt=1,
                side="buy",
                quote_px=100.0,
            )
        ],
    )

    assert quarantine == []
    assert sum(row["arrival_count"] for row in rows) == 1
    assert sum(row["arrival_volume_btc"] for row in rows) == 0.01


def test_confirmed_resting_exposure_clips_exchange_time_to_private_bounds() -> None:
    interval_start_ms = BASE_TS + 1_000
    interval_end_ms = BASE_TS + 3_000
    event_rows = [
        _event_book(
            BASE_TS + 100,
            local_receive_time_ms=BASE_TS + 1_100,
            bid_depth_btc=0.5,
        ),
        _event_book(
            BASE_TS + 1_200,
            local_receive_time_ms=BASE_TS + 1_200,
            bid_depth_btc=0.02,
        ),
        _event_trade(
            BASE_TS + 1_500,
            local_receive_time_ms=BASE_TS + 1_500,
            trade_px=100.0,
            trade_size_btc=0.01,
            aggressor_side="sell",
            trade_id="inside-private-bounds",
        ),
        _event_book(
            BASE_TS + 2_500,
            local_receive_time_ms=BASE_TS + 2_500,
            bid_depth_btc=0.03,
        ),
        _event_book(
            BASE_TS + 3_500,
            local_receive_time_ms=BASE_TS + 2_900,
            bid_depth_btc=0.9,
        ),
    ]

    rows, quarantine = estimators.build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=[
            _confirmed_interval(
                attempt_key="private-bounds",
                attempt=1,
                side="buy",
                quote_px=100.0,
                start_local_receive_time_ms=interval_start_ms,
                end_local_receive_time_ms=interval_end_ms,
            )
        ],
    )

    assert quarantine == []
    assert rows
    assert min(row["start_exchange_time_ms"] for row in rows) > interval_start_ms
    assert max(row["end_exchange_time_ms"] for row in rows) <= interval_end_ms
    assert rows[0]["start_exchange_time_ms"] == BASE_TS + 1_200
    assert rows[0]["pre_trade_side_depth_btc"] == 0.02


def test_confirmed_resting_exposure_does_not_look_ahead_for_pre_trade_depth() -> None:
    event_rows = [
        _event_book(
            BASE_TS + 100,
            bid_depth_btc=0.02,
        ),
        _event_trade(
            BASE_TS + 500,
            trade_px=100.0,
            trade_size_btc=0.01,
            aggressor_side="sell",
            trade_id="arrival-before-late-book",
        ),
        _event_book(
            BASE_TS + 500,
            bid_depth_btc=0.5,
        ),
        _event_book(
            BASE_TS + 900,
            bid_depth_btc=0.03,
        ),
    ]

    rows, quarantine = estimators.build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=[
            _confirmed_interval(
                attempt_key="no-lookahead",
                attempt=1,
                side="buy",
                quote_px=100.0,
            )
        ],
    )

    assert quarantine == []
    assert len(rows) == 1
    assert rows[0]["arrival_count"] == 1
    assert rows[0]["pre_trade_side_depth_btc"] == 0.02
    assert rows[0]["max_sweep_depth_penetration"] == 0.5


def test_confirmed_resting_exposure_is_directional_and_uses_latest_pre_trade_depth() -> None:
    event_rows = [
        _event_book(BASE_TS + 100),
        _event_book(
            BASE_TS + 300,
            bid_depth_btc=0.04,
            ask_depth_btc=0.05,
        ),
        _event_trade(
            BASE_TS + 400,
            trade_px=100.0,
            trade_size_btc=0.01,
            aggressor_side="sell",
            trade_id="buy-at",
        ),
        _event_trade(
            BASE_TS + 450,
            trade_px=100.1,
            trade_size_btc=0.02,
            aggressor_side="sell",
            trade_id="buy-not-through",
        ),
        _event_trade(
            BASE_TS + 500,
            trade_px=102.0,
            trade_size_btc=0.015,
            aggressor_side="buy",
            trade_id="sell-at",
        ),
        _event_trade(
            BASE_TS + 550,
            trade_px=101.9,
            trade_size_btc=0.025,
            aggressor_side="buy",
            trade_id="sell-not-through",
        ),
        _event_book(
            BASE_TS + 900,
            bid_depth_btc=0.06,
            ask_depth_btc=0.07,
        ),
    ]
    interval_rows = [
        _confirmed_interval(
            attempt_key="direction-buy",
            attempt=1,
            side="buy",
            quote_px=100.0,
        ),
        _confirmed_interval(
            attempt_key="direction-sell",
            attempt=2,
            side="sell",
            quote_px=102.0,
        ),
    ]

    rows, quarantine = estimators.build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=interval_rows,
    )

    assert quarantine == []
    by_side = {row["side"]: row for row in rows}
    assert by_side["buy"]["arrival_count"] == 1
    assert by_side["buy"]["arrival_volume_btc"] == 0.01
    assert by_side["buy"]["pre_trade_side_depth_btc"] == 0.04
    assert by_side["buy"]["max_sweep_depth_penetration"] == 0.25
    assert by_side["sell"]["arrival_count"] == 1
    assert by_side["sell"]["arrival_volume_btc"] == 0.015
    assert by_side["sell"]["pre_trade_side_depth_btc"] == 0.05
    assert by_side["sell"]["max_sweep_depth_penetration"] == 0.3


def test_confirmed_resting_exposure_shrinks_to_first_interval_book() -> None:
    event_rows = [
        _event_trade(
            BASE_TS + 100,
            trade_px=100.0,
            trade_size_btc=0.01,
            aggressor_side="sell",
            trade_id="before-first-book",
        ),
        _event_book(BASE_TS + 250),
        _event_book(BASE_TS + 900, bid_depth_btc=0.03),
    ]

    rows, quarantine = estimators.build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=[
            _confirmed_interval(
                attempt_key="shrink",
                attempt=1,
                side="buy",
                quote_px=100.0,
            )
        ],
    )

    assert quarantine == []
    assert len(rows) == 1
    assert rows[0]["start_exchange_time_ms"] == BASE_TS + 250
    assert rows[0]["end_exchange_time_ms"] == BASE_TS + 900
    assert rows[0]["duration_seconds"] == 0.65
    assert rows[0]["arrival_count"] == 0
    assert rows[0]["pre_trade_side_depth_btc"] == 0.02


def test_confirmed_resting_exposure_revalidates_event_order_and_future_skew() -> None:
    event_rows = [
        _event_book(BASE_TS + 100),
        _event_book(
            BASE_TS + 50,
            local_receive_time_ms=BASE_TS + 200,
            bid_depth_btc=0.025,
        ),
        _event_trade(
            BASE_TS + 10_000,
            local_receive_time_ms=BASE_TS + 100,
            trade_px=100.0,
            trade_size_btc=0.01,
            aggressor_side="sell",
            trade_id="future",
        ),
        {
            **_event_trade(
                BASE_TS + 300,
                trade_px=100.0,
                trade_size_btc=0.01,
                aggressor_side="sell",
                trade_id="bad-kind",
            ),
            "event_kind": "funding",
        },
        _event_book(BASE_TS + 900, bid_depth_btc=0.03),
    ]

    rows, quarantine = estimators.build_confirmed_resting_exposure_rows(
        event_rows=event_rows,
        interval_rows=[
            _confirmed_interval(
                attempt_key="event-validation",
                attempt=1,
                side="buy",
                quote_px=100.0,
            )
        ],
    )

    assert len(rows) == 1
    assert rows[0]["arrival_count"] == 0
    assert {row["reason"] for row in quarantine} == {
        "out_of_order_event",
        "future_event_beyond_allowed_skew",
        "event_kind_invalid",
    }


def test_confirmed_resting_replay_is_deterministic_and_candidate_stays_observe_only() -> None:
    event_rows = [_event_book(BASE_TS + 100)]
    event_time_ms = BASE_TS + 200
    trade_id = 0
    for trade_px, count in ((100.0, 8), (99.0, 4), (98.0, 2), (97.0, 1)):
        for _ in range(count):
            event_rows.append(
                _event_trade(
                    event_time_ms,
                    trade_px=trade_px,
                    trade_size_btc=0.001,
                    aggressor_side="sell",
                    trade_id=f"sell-{trade_id}",
                )
            )
            event_time_ms += 1
            trade_id += 1
    for trade_px, count in ((101.0, 8), (102.0, 4), (103.0, 2), (104.0, 1)):
        for _ in range(count):
            event_rows.append(
                _event_trade(
                    event_time_ms,
                    trade_px=trade_px,
                    trade_size_btc=0.001,
                    aggressor_side="buy",
                    trade_id=f"buy-{trade_id}",
                )
            )
            event_time_ms += 1
            trade_id += 1
    event_rows.extend(
        [
            _event_book(BASE_TS + 900, bid_depth_btc=0.025, ask_depth_btc=0.035),
            _event_book(
                BASE_TS + 1_100,
                bid_px=101.0,
                ask_px=102.0,
                bid_depth_btc=0.03,
                ask_depth_btc=0.04,
            ),
            _event_trade(
                BASE_TS + 1_200,
                trade_px=97.0,
                trade_size_btc=0.001,
                aggressor_side="sell",
                trade_id="latest-sell",
            ),
            _event_trade(
                BASE_TS + 1_201,
                trade_px=104.0,
                trade_size_btc=0.001,
                aggressor_side="buy",
                trade_id="latest-buy",
            ),
            _event_book(
                BASE_TS + 1_900,
                bid_px=101.0,
                ask_px=102.0,
                bid_depth_btc=0.035,
                ask_depth_btc=0.045,
            ),
        ]
    )
    interval_rows = []
    attempt = 1
    for side, quote_prices in (
        ("buy", (100.0, 99.0, 98.0, 97.0)),
        ("sell", (101.0, 102.0, 103.0, 104.0)),
    ):
        for quote_px in quote_prices:
            interval_rows.append(
                _confirmed_interval(
                    attempt_key=f"fit-{side}-{attempt}",
                    attempt=attempt,
                    side=side,
                    quote_px=quote_px,
                )
            )
            attempt += 1

    first = estimators.replay_estimator_rows(
        event_rows=event_rows,
        confirmed_resting_interval_rows=interval_rows,
    )
    second = estimators.replay_estimator_rows(
        event_rows=event_rows,
        confirmed_resting_interval_rows=interval_rows,
    )

    assert first.quote_exposure_rows() == second.quote_exposure_rows()
    assert first.intensity_rows() == second.intensity_rows()
    assert first.snapshot() == second.snapshot()
    assert first.fit_intensity("buy")["status"] == "pass"
    assert first.fit_intensity("sell")["status"] == "pass"
    assert first.snapshot()["dynamic_half_spread_candidate"]["status"] == "pass"
    assert first.snapshot()["dynamic_half_spread_candidate"]["activation_enabled"] is False
    assert first.snapshot()["activation_enabled"] is False
    assert first.snapshot()["actual_quote_behavior_changed"] is False

    legacy = estimators.replay_estimator_rows(event_rows=event_rows)
    assert legacy.quote_exposure_rows() == []
    assert legacy.snapshot()["dynamic_half_spread_candidate"]["status"] == "fallback_fixed"
    assert legacy.snapshot()["activation_enabled"] is False


def test_cold_start_falls_back_to_fixed_and_replay_is_deterministic() -> None:
    def run() -> tuple[list[dict], dict]:
        estimator = estimators.EventTimeOnlineEstimator()
        _book(estimator, BASE_TS, 100.0, 101.0)
        estimator.ingest_trade(
            event_time_ms=BASE_TS + 100,
            local_receive_time_ms=BASE_TS + 100,
            trade_px=101.0,
            trade_size_btc=0.001,
            aggressor_side="buy",
            trade_id="same-trade",
        )
        return estimator.bucket_rows(), estimator.snapshot()

    first_rows, first_snapshot = run()
    second_rows, second_snapshot = run()
    assert first_rows == second_rows
    assert first_snapshot == second_snapshot
    assert first_snapshot["dynamic_half_spread_candidate"]["status"] == "fallback_fixed"
    assert first_snapshot["dynamic_half_spread_candidate"]["half_spread_ticks"] == 0.5


def test_recorded_event_rows_replay_to_identical_snapshot(tmp_path: Path) -> None:
    estimator = estimators.EventTimeOnlineEstimator()
    _book(estimator, BASE_TS, 100.0, 101.0)
    _book(estimator, BASE_TS + 1_000, 101.0, 102.0)
    estimator.ingest_trade(
        event_time_ms=BASE_TS + 1_100,
        local_receive_time_ms=BASE_TS + 1_100,
        trade_px=102.0,
        trade_size_btc=0.001,
        aggressor_side="buy",
        trade_id="replay-trade",
    )
    _exposure_rows(estimator, "buy")
    _exposure_rows(estimator, "sell")
    input_dir = tmp_path / "input"
    estimators._write_csv(
        input_dir / "online_estimator_event_rows.csv",
        estimator.event_rows(),
        estimators.estimator_event_fieldnames(),
    )
    estimators._write_csv(
        input_dir / "quote_exposure_intervals.csv",
        estimator.quote_exposure_rows(),
        estimators.quote_exposure_fieldnames(),
    )
    estimators._write_json(
        input_dir / "online_estimator_core_snapshot.json",
        estimator.snapshot(),
    )

    manifest = estimators.build_replay_artifacts(
        input_dir=input_dir,
        output_dir=tmp_path / "replay",
    )

    assert manifest["snapshot_match"] is True
    assert manifest["event_row_count"] == 3
    assert manifest["quote_exposure_row_count"] == 8


def test_replay_artifacts_rebuild_confirmed_exposure_from_interval_contract(
    tmp_path: Path,
) -> None:
    event_rows = [
        _event_book(BASE_TS + 100),
        _event_trade(
            BASE_TS + 500,
            trade_px=100.0,
            trade_size_btc=0.004,
            aggressor_side="sell",
            trade_id="touch",
        ),
        _event_book(
            BASE_TS + 1_100,
            bid_px=99.0,
            ask_px=100.0,
            bid_depth_btc=0.03,
        ),
        _event_book(
            BASE_TS + 2_100,
            bid_px=100.0,
            ask_px=101.0,
            bid_depth_btc=0.025,
        ),
    ]
    interval_rows = [
        _confirmed_interval(
            attempt_key="0720T028:window_01:attempt_1",
            attempt=1,
            side="buy",
            quote_px=100.0,
            response_status_types="resting",
            interval_status="pass",
            interval_reason="",
        )
    ]
    source = estimators.replay_estimator_rows(
        event_rows=event_rows,
        confirmed_resting_interval_rows=interval_rows,
    )
    input_dir = tmp_path / "confirmed-input"
    estimators._write_csv(
        input_dir / "online_estimator_event_rows.csv",
        event_rows,
        estimators.estimator_event_fieldnames(),
    )
    estimators._write_csv(
        input_dir / "quote_exposure_intervals.csv",
        source.quote_exposure_rows(),
        estimators.quote_exposure_fieldnames(),
    )
    estimators._write_csv(
        input_dir / "confirmed_resting_interval_contract.csv",
        interval_rows,
        watcher.manager_resting_interval_fieldnames(),
    )
    estimators._write_json(
        input_dir / "online_estimator_core_snapshot.json",
        source.snapshot(),
    )

    manifest = estimators.build_replay_artifacts(
        input_dir=input_dir,
        output_dir=tmp_path / "confirmed-replay",
    )

    assert manifest["snapshot_match"] is True
    assert manifest["confirmed_resting_exposure_match"] is True
    assert manifest["confirmed_resting_interval_row_count"] == 1
    assert manifest["rebuilt_confirmed_exposure_row_count"] == 3

    tampered = [dict(row) for row in source.quote_exposure_rows()]
    tampered[0]["arrival_count"] = 999
    estimators._write_csv(
        input_dir / "quote_exposure_intervals.csv",
        tampered,
        estimators.quote_exposure_fieldnames(),
    )
    tampered_manifest = estimators.build_replay_artifacts(
        input_dir=input_dir,
        output_dir=tmp_path / "tampered-replay",
    )
    assert tampered_manifest["confirmed_resting_exposure_match"] is False
    assert tampered_manifest["snapshot_match"] is False


def test_shared_kernel_overlay_keeps_fixed_quote_authoritative() -> None:
    overlay = shared_kernel.build_observe_only_pricing_overlay(
        fixed_half_spread_ticks=0.5,
        dynamic_candidate_half_spread_ticks=1.25,
    )
    assert overlay["authoritative_half_spread_ticks"] == 0.5
    assert overlay["dynamic_candidate_half_spread_ticks"] == 1.25
    assert overlay["activation_enabled"] is False
    assert overlay["quote_behavior_changed"] is False


def test_public_shadow_writes_observe_only_estimator_artifacts(tmp_path: Path, monkeypatch) -> None:
    control_dir = tmp_path / "control"
    executor.initialize_control_state(control_dir)
    monkeypatch.setattr(executor, "DEFAULT_CONTROL_STATE_DIR", control_dir)
    monkeypatch.setattr(watcher, "DEFAULT_CONTROL_STATE_DIR", control_dir)
    now_ms = int(time.time() * 1000)
    messages = [
        {
            "channel": "l2Book",
            "data": {
                "time": now_ms,
                "levels": [
                    [{"px": "100", "sz": "0.02", "n": 4}],
                    [{"px": "101", "sz": "0.01", "n": 2}],
                ],
            },
        },
        {
            "channel": "l2Book",
            "data": {
                "time": now_ms + 1_000,
                "levels": [
                    [{"px": "101", "sz": "0.02", "n": 4}],
                    [{"px": "102", "sz": "0.01", "n": 2}],
                ],
            },
        },
        {
            "channel": "trades",
            "data": [
                {
                    "time": now_ms + 1_001,
                    "px": "102",
                    "sz": "0.001",
                    "side": "B",
                    "tid": "integration-trade",
                }
            ],
        },
    ]

    manifest = watcher.run_event_driven_public_shadow_source(
        output_dir=tmp_path / "shadow",
        watcher_seconds=1,
        event_source_fn=lambda: ((time.time_ns(), message) for message in messages),
        binance_public_state_provider=lambda: None,
        public_source_mode="t019_unit_observe_only",
    )

    assert manifest["dynamic_spread_activation_enabled"] is False
    assert manifest["actual_quote_behavior_changed"] is False
    snapshot = json.loads((tmp_path / "shadow" / "online_estimator_snapshot.json").read_text())
    assert snapshot["activation_enabled"] is False
    assert snapshot["actual_quote_behavior_changed"] is False
    assert (tmp_path / "shadow" / "online_estimator_bucket_matrix.csv").exists()
    assert (tmp_path / "shadow" / "online_estimator_event_rows.csv").exists()
    assert (tmp_path / "shadow" / "online_estimator_quarantine.csv").exists()
    assert (tmp_path / "shadow" / "quote_exposure_intervals.csv").exists()
    replay = estimators.build_replay_artifacts(
        input_dir=tmp_path / "shadow",
        output_dir=tmp_path / "shadow-replay",
    )
    assert replay["snapshot_match"] is True
    assert manifest["order_endpoint_called"] is False
    assert manifest["private_endpoint_called"] is False
    assert manifest["credentials_read"] is False
