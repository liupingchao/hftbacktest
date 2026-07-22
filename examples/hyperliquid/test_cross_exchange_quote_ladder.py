from __future__ import annotations

import pytest

from examples.hyperliquid import cross_exchange_shared_signal_kernel as kernel
from examples.hyperliquid import hyperliquid_maker_order_manager as manager
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher


PRECISION = {"tick_size": 1.0, "sz_decimals": 5, "lot_size": 0.00001}


def _build(config: kernel.QuoteLadderConfigV1) -> dict[str, object]:
    return kernel.build_default_off_quote_ladder(
        reservation_px=100.0,
        half_spread_ticks=0.5,
        best_bid=99.0,
        best_ask=101.0,
        precision=PRECISION,
        bid_size_btc=0.001,
        ask_size_btc=0.001,
        config=config,
    )


def test_level_zero_matches_authoritative_single_level_quote_path() -> None:
    base = kernel.compute_two_sided_quotes(
        reservation_px=100.0,
        half_spread_ticks=0.5,
        best_bid=99.0,
        best_ask=101.0,
        precision=PRECISION,
    )
    result = _build(kernel.QuoteLadderConfigV1())

    rows = {row["side"]: row for row in result["ladder_rows"]}
    assert result["status"] == "pass_observe_only"
    assert result["reason"] == "single_level_authoritative"
    assert rows["buy"]["level"] == 0
    assert rows["sell"]["level"] == 0
    assert rows["buy"]["quote_px"] == base.bid_px
    assert rows["sell"]["quote_px"] == base.ask_px
    assert result["quote_intents"] == []
    assert result["activation_enabled"] is False
    assert result["actual_quote_behavior_changed"] is False


def test_deeper_levels_are_deterministic_bounded_and_default_off() -> None:
    config = kernel.QuoteLadderConfigV1(
        levels=3,
        gap_ticks=1.0,
        size_decay=0.4,
        max_total_size_btc=0.01,
    )
    first = _build(config)
    second = _build(config)

    assert first == second
    assert first["status"] == "blocked"
    assert first["reason"] == "single_level_lifecycle_prerequisite_not_satisfied"
    assert len(first["hypothetical_quote_intents"]) == 6
    assert first["quote_intents"] == []
    assert first["activation_enabled"] is False
    assert first["working_exposure_btc"] <= 0.01
    for row in first["ladder_rows"]:
        assert row["post_only"] is True
        assert row["time_in_force"] == "Alo"
        assert row["size_btc"] >= 0.00001


def test_active_multi_level_ladder_emits_executable_intents() -> None:
    config = kernel.QuoteLadderConfigV1(
        levels=2,
        gap_ticks=1.0,
        size_decay=0.5,
        max_total_size_btc=0.01,
        activation_enabled=True,
        single_level_lifecycle_prerequisite=True,
    )
    first = _build(config)
    second = _build(config)

    assert first == second
    assert first["status"] == "pass"
    assert first["reason"] == "multi_level_activation_enabled"
    assert first["activation_enabled"] is True
    assert first["actual_quote_behavior_changed"] is True
    assert first["quote_intents"] == first["ladder_rows"]
    assert len(first["quote_intents"]) == 4
    buys = [
        row
        for row in first["quote_intents"]
        if row["side"] == "buy"
    ]
    sells = [
        row
        for row in first["quote_intents"]
        if row["side"] == "sell"
    ]
    assert [row["quote_px"] for row in buys] == [99.5, 98.5]
    assert [row["quote_px"] for row in sells] == [100.5, 101.5]
    assert [row["size_btc"] for row in buys] == [0.001, 0.0005]
    assert [row["size_btc"] for row in sells] == [0.001, 0.0005]
    assert all(row["post_only"] for row in first["quote_intents"])
    assert all(
        row["time_in_force"] == "Alo"
        for row in first["quote_intents"]
    )
    assert all(
        row["price_key"]
        == manager.executor.canonical_price_key(
            row["quote_px"],
            sz_decimals=5,
        )
        for row in first["quote_intents"]
    )


def test_rounded_duplicate_prices_coalesce_or_fail_closed() -> None:
    coalesced = _build(
        kernel.QuoteLadderConfigV1(
            levels=3,
            gap_ticks=0.01,
            size_decay=0.5,
            max_total_size_btc=0.01,
            coalesce_duplicate_prices=True,
        )
    )
    assert coalesced["status"] == "blocked"
    assert any(len(row["coalesced_levels"]) > 1 for row in coalesced["ladder_rows"])
    assert len(
        {(row["side"], row["price_key"]) for row in coalesced["ladder_rows"]}
    ) == len(coalesced["ladder_rows"])

    rejected = _build(
        kernel.QuoteLadderConfigV1(
            levels=3,
            gap_ticks=0.01,
            size_decay=0.5,
            max_total_size_btc=0.01,
            coalesce_duplicate_prices=False,
        )
    )
    assert rejected["status"] == "fail_closed"
    assert "duplicate_rounded_price" in rejected["reason"]
    assert rejected["quote_intents"] == []


def test_invalid_size_and_aggregate_exposure_fail_closed() -> None:
    too_small = _build(
        kernel.QuoteLadderConfigV1(
            levels=2,
            size_decay=0.001,
            min_size_btc=0.0001,
            max_total_size_btc=0.01,
        )
    )
    assert too_small["status"] == "fail_closed"
    assert "invalid_level_size" in too_small["reason"]

    too_large = _build(
        kernel.QuoteLadderConfigV1(
            levels=3,
            size_decay=1.0,
            max_total_size_btc=0.005,
        )
    )
    assert too_large["status"] == "fail_closed"
    assert too_large["reason"] == "aggregate_ladder_exposure_cap_exceeded"

    with pytest.raises(ValueError, match="size_decay"):
        kernel.QuoteLadderConfigV1(size_decay=0.0)
    with pytest.raises(ValueError, match="levels"):
        kernel.QuoteLadderConfigV1(
            levels=kernel.MAX_QUOTE_LADDER_LEVELS + 1,
        )


def test_manager_and_status_keep_multi_level_gate_fail_closed() -> None:
    gate = manager.MakerOrderManager.multi_level_prerequisite_gate(
        requested_levels=3,
        activation_enabled=True,
        single_level_lifecycle_prerequisite=False,
    )
    assert gate["status"] == "blocked"
    assert gate["reason"] == "single_level_lifecycle_prerequisite_not_satisfied"
    assert gate["activation_enabled"] is False

    status = watcher.task7_status_payload(
        run_id="run-t021",
        window_id=1,
        config_hash="config",
    )
    assert status["multi_level"]["status"] == "single_level_authoritative"
    assert status["multi_level"]["activation_enabled"] is False
    assert status["multi_level"]["actual_quote_behavior_changed"] is False

    active = manager.MakerOrderManager.multi_level_prerequisite_gate(
        requested_levels=2,
        activation_enabled=True,
        single_level_lifecycle_prerequisite=True,
    )
    assert active["status"] == "pass"
    assert active["reason"] == ""
    assert active["activation_enabled"] is True
    assert active["actual_quote_behavior_changed"] is True
