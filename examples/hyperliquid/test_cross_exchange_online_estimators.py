from __future__ import annotations

import json
import time
from pathlib import Path

from examples.hyperliquid import cross_exchange_online_estimators as estimators
from examples.hyperliquid import cross_exchange_shared_signal_kernel as shared_kernel
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


BASE_TS = 1_700_000_000_000


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
