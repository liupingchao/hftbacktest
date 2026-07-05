from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("cross_exchange_shared_signal_kernel.py")
SPEC = importlib.util.spec_from_file_location("cross_exchange_shared_signal_kernel", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
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
    return MODULE.default_normalization_stats(_contract())


def test_kernel_would_submit_buy_and_sell_from_accepted_side_mapping() -> None:
    contract = _contract()
    common = {
        "hyperliquid_bid_px": 100.0,
        "hyperliquid_ask_px": 101.0,
        "hyperliquid_mid_px": 100.5,
        "tick_size": 1.0,
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
    )

    assert buy["action"] == "would_submit"
    assert buy["side"] == "buy"
    assert buy["quote_intent"]["time_in_force"] == "Alo"
    assert buy["order_endpoint_called"] is False
    assert sell["action"] == "would_submit"
    assert sell["side"] == "sell"
    assert sell["quote_intent"]["post_only"] is True


def test_kernel_blocks_missing_feature_and_below_threshold() -> None:
    contract = _contract()
    common = {
        "hyperliquid_bid_px": 100.0,
        "hyperliquid_ask_px": 101.0,
        "hyperliquid_mid_px": 100.5,
        "tick_size": 1.0,
        "input_binance_top5_imbalance": 0.2,
        "input_binance_microprice_minus_mid_ticks": 0.1,
        "input_binance_mid_move_ticks_from_prev": 0.0,
    }
    below = MODULE.evaluate_shared_kernel(common, contract=contract, normalization_stats=_stats())
    missing = MODULE.evaluate_shared_kernel(
        {key: value for key, value in common.items() if key != "input_binance_mid_move_ticks_from_prev"},
        contract=contract,
        normalization_stats=_stats(),
    )

    assert below["action"] == "block"
    assert below["block_reason"] == "signal_below_threshold"
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
    assert first["manifest"]["would_submit_fixture_count"] == 3
    assert first["manifest"]["block_fixture_count"] == 2
    assert first["manifest"]["warning_bucket_visible_in_fixture"] is True
    assert first["boundary_manifest"]["no_live_orders"] is True
    assert first["boundary_manifest"]["no_private_account_order_cancel_endpoints"] is True
