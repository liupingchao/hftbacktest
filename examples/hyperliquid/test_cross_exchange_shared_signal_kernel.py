from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).with_name("cross_exchange_shared_signal_kernel.py")
SPEC = importlib.util.spec_from_file_location("cross_exchange_shared_signal_kernel", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _contract() -> dict[str, object]:
    return {
        "schema_version": "cross_exchange_signal_contract_v1",
        "task_id": "0625T003",
        "candidate_id": "binance_lead_composite",
        "feature_schema": [
            "input_binance_top5_imbalance",
            "input_binance_microprice_minus_mid_ticks",
            "input_binance_mid_move_ticks_from_prev",
        ],
        "horizon_ms": 1000,
        "effective_horizon_row_condition": "valid_for_1000ms_signal_acceptance=true and 1000ms <= effective_future_age_ms <= 1250ms",
        "normalization": "train_fold_z_score_mean_std",
        "threshold_abs_z": 1.0,
        "side_mapping": "positive_signal_buy_negative_signal_sell",
        "acceptance_limits": {"fee_adverse_buffer_ticks": 1.5},
        "edge_formula": {
            "fair_mid_px": "current_hyperliquid_mid + signed_expected_move_ticks * tick_size",
            "buy_edge_ticks": "(fair_mid_px - quote_px) / tick_size",
            "sell_edge_ticks": "(quote_px - fair_mid_px) / tick_size",
        },
    }


def _stats() -> dict[str, dict[str, float]]:
    return MODULE.fixture_normalization_stats(_contract())


def _pricing_config(
    stats: dict[str, dict[str, float]] | None = None,
    **overrides: object,
) -> object:
    values = {
        "expected_move_ticks_per_signal_z": 4.0,
        "base_half_spread_ticks": 0.5,
        "inventory_skew_ticks_at_max": 0.0,
        "max_position_btc": 0.01,
        "enable_microprice": False,
        "enable_inventory_skew": False,
        "enable_dynamic_spread": False,
        "enable_fill_feedback": False,
        "levels": 1,
    }
    values.update(overrides)
    return MODULE.PricingConfigV1.from_normalization_stats(stats or _stats(), **values)


def test_kernel_would_submit_buy_and_sell_from_accepted_side_mapping() -> None:
    contract = _contract()
    common = {
        "hyperliquid_bid_px": 100.0,
        "hyperliquid_ask_px": 101.0,
        "hyperliquid_mid_px": 100.5,
        "tick_size": 1.0,
        "sz_decimals": 5,
    }
    buy = MODULE.evaluate_shared_kernel(
        {
            **common,
            "decision_id": "buy",
            "input_binance_top5_imbalance": 1.2,
            "input_binance_microprice_minus_mid_ticks": 1.1,
            "input_binance_mid_move_ticks_from_prev": 1.0,
        },
        contract=contract,
        normalization_stats=_stats(),
        pricing_config=_pricing_config(),
    )
    sell = MODULE.evaluate_shared_kernel(
        {
            **common,
            "decision_id": "sell",
            "input_binance_top5_imbalance": -1.2,
            "input_binance_microprice_minus_mid_ticks": -1.1,
            "input_binance_mid_move_ticks_from_prev": -1.0,
        },
        contract=contract,
        normalization_stats=_stats(),
        pricing_config=_pricing_config(),
    )

    assert buy["action"] == "would_submit"
    assert buy["side"] == "buy"
    assert buy["quote_bid_px"] < buy["quote_ask_px"]
    assert {row["side"] for row in buy["quote_intents"]} == {"buy", "sell"}
    assert buy["quote_intent"]["time_in_force"] == "Alo"
    assert buy["order_endpoint_called"] is False
    assert sell["action"] == "would_submit"
    assert sell["side"] == "sell"
    assert sell["quote_intent"]["post_only"] is True
    assert buy["pricing_config_hash"] == _pricing_config().config_hash


def test_kernel_blocks_missing_feature_and_below_threshold() -> None:
    contract = _contract()
    common = {
        "hyperliquid_bid_px": 100.0,
        "hyperliquid_ask_px": 101.0,
        "hyperliquid_mid_px": 100.5,
        "tick_size": 1.0,
        "sz_decimals": 5,
        "input_binance_top5_imbalance": 0.2,
        "input_binance_microprice_minus_mid_ticks": 0.1,
        "input_binance_mid_move_ticks_from_prev": 0.0,
    }
    below = MODULE.evaluate_shared_kernel(
        common,
        contract=contract,
        normalization_stats=_stats(),
        pricing_config=_pricing_config(),
    )
    missing = MODULE.evaluate_shared_kernel(
        {key: value for key, value in common.items() if key != "input_binance_mid_move_ticks_from_prev"},
        contract=contract,
        normalization_stats=_stats(),
        pricing_config=_pricing_config(),
    )

    assert below["action"] == "would_submit"
    assert below["block_reason"] == ""
    assert below["confidence_bucket"] == "below_threshold"
    assert below["quote_eligibility"] == "eligible"
    assert missing["action"] == "block"
    assert missing["block_reason"] == "signal_missing_feature:input_binance_mid_move_ticks_from_prev"


def test_build_fixture_artifacts_is_deterministic_and_boundary_closed(tmp_path: Path) -> None:
    contract_path = tmp_path / "accepted_signal_contract.json"
    signal_manifest_path = tmp_path / "signal_acceptance_manifest.json"
    contract_path.write_text(json.dumps(_contract(), indent=2) + "\n", encoding="utf-8")
    signal_manifest_path.write_text(
        json.dumps(
            {
                "blocking_or_warning_reasons": [
                    {
                        "severity": "warning",
                        "reason": "some_source_age_or_basis_buckets_have_negative_adjusted_proxy",
                        "bucket_count": 1,
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    first = MODULE.build_fixture_artifacts(
        contract_path=contract_path,
        signal_manifest_path=signal_manifest_path,
        output_dir=tmp_path / "out1",
    )
    second = MODULE.build_fixture_artifacts(
        contract_path=contract_path,
        signal_manifest_path=signal_manifest_path,
        output_dir=tmp_path / "out2",
    )

    assert first["fixture_inputs"]["market_views"] == second["fixture_inputs"]["market_views"]
    assert first["fixture_outputs"] == second["fixture_outputs"]
    assert first["manifest"]["would_submit_fixture_count"] == 4
    assert first["manifest"]["block_fixture_count"] == 1
    assert first["manifest"]["warning_bucket_visible_in_fixture"] is True
    assert first["manifest"]["pricing_config_hash"] == first["fixture_inputs"]["pricing_config_hash"]
    assert first["boundary_manifest"]["no_live_orders"] is True
    assert first["boundary_manifest"]["no_private_account_order_cancel_endpoints"] is True


def test_pricing_config_is_validated_serialized_and_hash_stable() -> None:
    config = _pricing_config()
    restored = MODULE.PricingConfigV1.from_dict(config.to_dict())

    assert restored == config
    assert restored.config_hash == config.config_hash
    assert len(config.config_hash) == 64
    with pytest.raises(ValueError, match="fields_mismatch"):
        MODULE.PricingConfigV1.from_dict({**config.to_dict(), "unexpected": True})


def test_alpha_moves_forecast_center_without_selecting_only_one_quote() -> None:
    contract = _contract()
    stats = _stats()
    common = {
        "hyperliquid_bid_px": 90.0,
        "hyperliquid_ask_px": 110.0,
        "hyperliquid_mid_px": 100.0,
        "tick_size": 1.0,
        "sz_decimals": 5,
        "input_binance_top5_imbalance": 0.2,
        "input_binance_microprice_minus_mid_ticks": 0.2,
        "input_binance_mid_move_ticks_from_prev": 0.2,
    }
    alpha = MODULE.evaluate_shared_kernel(
        {**common, "decision_id": "alpha"},
        contract=contract,
        normalization_stats=stats,
        pricing_config=_pricing_config(base_half_spread_ticks=2.0),
    )
    zero = MODULE.evaluate_shared_kernel(
        {
            **common,
            "decision_id": "zero",
            "input_binance_top5_imbalance": 0.0,
            "input_binance_microprice_minus_mid_ticks": 0.0,
            "input_binance_mid_move_ticks_from_prev": 0.0,
        },
        contract=contract,
        normalization_stats=stats,
        pricing_config=_pricing_config(base_half_spread_ticks=2.0),
    )

    assert alpha["action"] == "would_submit"
    assert alpha["side"] == "buy"
    assert alpha["forecast_mid_px"] == 100.8
    assert alpha["quote_bid_px"] == 98.8
    assert alpha["quote_ask_px"] == 102.8
    assert zero["side"] == "both"
    assert zero["forecast_mid_px"] == 100.0
    assert zero["quote_bid_px"] == 98.0
    assert zero["quote_ask_px"] == 102.0


def test_microprice_requires_same_fresh_snapshot_and_falls_back_to_mid() -> None:
    contract = _contract()
    stats = _stats()
    config = _pricing_config(enable_microprice=True)
    common = {
        "hyperliquid_bid_px": 100.0,
        "hyperliquid_ask_px": 101.0,
        "hyperliquid_mid_px": 100.5,
        "tick_size": 1.0,
        "sz_decimals": 5,
        "input_binance_top5_imbalance": 0.0,
        "input_binance_microprice_minus_mid_ticks": 0.0,
        "input_binance_mid_move_ticks_from_prev": 0.0,
        "hyperliquid_bid_qty": 3.0,
        "hyperliquid_ask_qty": 1.0,
        "bbo_snapshot_id": "snapshot-1",
        "bbo_snapshot_coherent": True,
        "bbo_snapshot_age_ms": 10.0,
    }
    fresh = MODULE.evaluate_shared_kernel(
        common,
        contract=contract,
        normalization_stats=stats,
        pricing_config=config,
    )
    stale = MODULE.evaluate_shared_kernel(
        {**common, "bbo_snapshot_age_ms": 300.0},
        contract=contract,
        normalization_stats=stats,
        pricing_config=config,
    )
    incoherent = MODULE.evaluate_shared_kernel(
        {**common, "bbo_snapshot_coherent": False},
        contract=contract,
        normalization_stats=stats,
        pricing_config=config,
    )

    assert fresh["hl_micro_px"] == 100.75
    assert fresh["fair_base"] == "hl_micro_px"
    assert fresh["microprice_reason"] == "same_snapshot_bbo_qty"
    assert stale["hl_micro_px"] is None
    assert stale["fair_base"] == "mid_fallback"
    assert stale["microprice_reason"] == "stale_microprice_snapshot"
    assert incoherent["action"] == "block"
    assert incoherent["block_reason"] == "incoherent_bbo_snapshot"


def test_stale_signal_and_normalization_hash_mismatch_fail_closed() -> None:
    contract = _contract()
    stats = _stats()
    config = _pricing_config()
    market = {
        "hyperliquid_bid_px": 100.0,
        "hyperliquid_ask_px": 101.0,
        "hyperliquid_mid_px": 100.5,
        "tick_size": 1.0,
        "input_binance_top5_imbalance": 1.2,
        "input_binance_microprice_minus_mid_ticks": 1.1,
        "input_binance_mid_move_ticks_from_prev": 1.0,
    }
    stale = MODULE.evaluate_shared_kernel(
        {**market, "signal_stale": True},
        contract=contract,
        normalization_stats=stats,
        pricing_config=config,
    )
    mismatch = MODULE.evaluate_shared_kernel(
        market,
        contract=contract,
        normalization_stats={**stats, "extra": {"mean": 0.0, "std": 1.0}},
        pricing_config=config,
    )

    assert stale["action"] == "block"
    assert stale["block_reason"] == "stale_signal"
    assert mismatch["action"] == "block"
    assert mismatch["block_reason"] == "normalization_stats_hash_mismatch"


def test_reservation_price_skew_sign_ratio_notional_and_bound() -> None:
    long = MODULE.compute_reservation_price(
        forecast_mid_px=100.0,
        position_btc=0.005,
        mid_px=100.0,
        max_position_btc=0.01,
        inventory_skew_ticks_at_max=2.0,
        price_increment=0.5,
    )
    short = MODULE.compute_reservation_price(
        forecast_mid_px=100.0,
        position_btc=-0.005,
        mid_px=100.0,
        max_position_btc=0.01,
        inventory_skew_ticks_at_max=2.0,
        price_increment=0.5,
    )
    over_cap = MODULE.compute_reservation_price(
        forecast_mid_px=100.0,
        position_btc=0.02,
        mid_px=100.0,
        max_position_btc=0.01,
        inventory_skew_ticks_at_max=2.0,
        price_increment=0.5,
    )

    assert long.raw_position_ratio == 0.5
    assert long.position_notional == 0.5
    assert long.inventory_penalty_ticks == 1.0
    assert long.reservation_px == 99.5
    assert short.reservation_px == 100.5
    assert over_cap.bounded_position_ratio == 1.0
    assert over_cap.inventory_penalty_ticks == 2.0
    assert over_cap.hard_cap_breached is True


def test_two_sided_quotes_record_desired_final_clamp_and_invariant() -> None:
    quotes = MODULE.compute_two_sided_quotes(
        reservation_px=105.0,
        half_spread_ticks=0.5,
        best_bid=100.0,
        best_ask=101.0,
        precision={"tick_size": 1.0, "sz_decimals": 5},
    )

    assert quotes.desired_bid_px == 104.5
    assert quotes.bid_px == 100.0
    assert quotes.bid_clamp_reason == "post_only_crossing_clamp_to_best_bid"
    assert quotes.ask_px > 100.0
    assert quotes.post_only_invariant is True
    assert quotes.bid_edge_change_ticks < 0


def test_near_cap_suppresses_add_side_and_preserves_reduce_side() -> None:
    contract = _contract()
    stats = _stats()
    config = _pricing_config(
        enable_inventory_skew=True,
        inventory_skew_ticks_at_max=1.0,
        base_half_spread_ticks=2.0,
    )
    common = {
        "hyperliquid_bid_px": 90.0,
        "hyperliquid_ask_px": 110.0,
        "hyperliquid_mid_px": 100.0,
        "tick_size": 1.0,
        "sz_decimals": 5,
        "input_binance_top5_imbalance": 0.2,
        "input_binance_microprice_minus_mid_ticks": 0.2,
        "input_binance_mid_move_ticks_from_prev": 0.2,
    }
    long = MODULE.evaluate_shared_kernel(
        {**common, "position_btc": 0.009},
        contract=contract,
        normalization_stats=stats,
        pricing_config=config,
    )
    short = MODULE.evaluate_shared_kernel(
        {**common, "position_btc": -0.009},
        contract=contract,
        normalization_stats=stats,
        pricing_config=config,
    )

    assert long["reservation_px"] < long["forecast_mid_px"]
    assert long["quote_eligibility"] == "reduce_only"
    assert [row["side"] for row in long["quote_intents"]] == ["sell"]
    assert long["signal_side"] == "buy"
    assert long["side"] == "sell"
    assert long["quote_px"] == long["quote_ask_px"]
    assert long["inventory_worsening_side"] == "buy"
    assert short["reservation_px"] > short["forecast_mid_px"]
    assert [row["side"] for row in short["quote_intents"]] == ["buy"]
    assert short["inventory_worsening_side"] == "sell"
