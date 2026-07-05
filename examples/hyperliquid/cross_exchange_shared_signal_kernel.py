#!/usr/bin/env python3
"""Shared pure signal/fair-mid/quote-intent kernel for the cross-exchange MVP."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0625T004"
SCHEMA_VERSION = "cross_exchange_shared_signal_kernel_v1"
DEFAULT_CONTRACT_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_mvp_signal_acceptance_0625T003"
    / "accepted_signal_contract.json"
)
DEFAULT_SIGNAL_MANIFEST_PATH = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_mvp_signal_acceptance_0625T003"
    / "signal_acceptance_manifest.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_shared_kernel_0625T004"

POST_ONLY_TIF = "Alo"
DEFAULT_EXPECTED_MOVE_TICKS_PER_SIGNAL_Z = 4.0
DEFAULT_REQUIRED_EDGE_TICKS = 1.5


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _finite_positive(value: Any) -> float | None:
    parsed = _float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def load_signal_contract(path: Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    contract = _read_json(path)
    if contract.get("schema_version") != "cross_exchange_signal_contract_v1":
        raise ValueError("unsupported_signal_contract_schema")
    if contract.get("horizon_ms") != 1000:
        raise ValueError("unsupported_signal_contract_horizon")
    if contract.get("side_mapping") != "positive_signal_buy_negative_signal_sell":
        raise ValueError("unsupported_signal_contract_side_mapping")
    feature_schema = contract.get("feature_schema")
    if not isinstance(feature_schema, list) or not feature_schema:
        raise ValueError("signal_contract_missing_feature_schema")
    return contract


def default_normalization_stats(contract: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        str(field): {"mean": 0.0, "std": 1.0}
        for field in contract.get("feature_schema", [])
    }


def normalize_signal(
    market_view: dict[str, Any],
    *,
    contract: dict[str, Any],
    normalization_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    z_values: list[float] = []
    component_rows: list[dict[str, Any]] = []
    for field in contract["feature_schema"]:
        raw = _float(market_view.get(field))
        stats = normalization_stats.get(field) or {}
        mean = _float(stats.get("mean"))
        std = _float(stats.get("std"))
        if raw is None:
            return {
                "status": "block",
                "reason": f"signal_missing_feature:{field}",
                "components": component_rows,
            }
        if mean is None or std is None or std <= 0:
            return {
                "status": "block",
                "reason": f"signal_invalid_normalization:{field}",
                "components": component_rows,
            }
        z = (raw - mean) / std
        z_values.append(z)
        component_rows.append({"field": field, "raw": raw, "mean": mean, "std": std, "z": round(z, 8)})
    if not z_values:
        return {"status": "block", "reason": "signal_no_components", "components": component_rows}
    score = math.fsum(z_values) / len(z_values)
    threshold = float(contract["threshold_abs_z"])
    abs_score = abs(score)
    if abs_score < threshold:
        return {
            "status": "block",
            "reason": "signal_below_threshold",
            "candidate_id": contract["candidate_id"],
            "signal_score": round(score, 8),
            "signal_abs_z": round(abs_score, 8),
            "threshold_abs_z": threshold,
            "components": component_rows,
        }
    return {
        "status": "pass",
        "reason": "",
        "candidate_id": contract["candidate_id"],
        "signal_score": round(score, 8),
        "signal_abs_z": round(abs_score, 8),
        "threshold_abs_z": threshold,
        "components": component_rows,
    }


def side_from_signal(signal_score: float, side_mapping: str) -> str:
    if side_mapping != "positive_signal_buy_negative_signal_sell":
        raise ValueError("unsupported_side_mapping")
    return "buy" if signal_score > 0 else "sell"


def evaluate_shared_kernel(
    market_view: dict[str, Any],
    *,
    contract: dict[str, Any],
    normalization_stats: dict[str, dict[str, Any]],
    expected_move_ticks_per_signal_z: float = DEFAULT_EXPECTED_MOVE_TICKS_PER_SIGNAL_Z,
    required_edge_ticks: float | None = None,
) -> dict[str, Any]:
    decision_id = str(market_view.get("decision_id") or "")
    signal = normalize_signal(market_view, contract=contract, normalization_stats=normalization_stats)
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "decision_id": decision_id,
        "candidate_id": contract.get("candidate_id", ""),
        "horizon_ms": contract.get("horizon_ms"),
        "normalization": contract.get("normalization", ""),
        "side_mapping": contract.get("side_mapping", ""),
        "post_only_tif": POST_ONLY_TIF,
        "order_endpoint_called": False,
        "private_endpoint_called": False,
        "credential_read": False,
        "live_client_initialized": False,
    }
    if signal["status"] != "pass":
        return {
            **base,
            "signal_status": "block",
            "signal_reason": signal["reason"],
            "action": "block",
            "block_reason": signal["reason"],
            "signal_components": signal.get("components", []),
        }

    tick_size = _finite_positive(market_view.get("tick_size"))
    bid_px = _finite_positive(market_view.get("hyperliquid_bid_px"))
    ask_px = _finite_positive(market_view.get("hyperliquid_ask_px"))
    mid_px = _finite_positive(market_view.get("hyperliquid_mid_px"))
    if mid_px is None and bid_px is not None and ask_px is not None and ask_px > bid_px:
        mid_px = (bid_px + ask_px) / 2.0
    if tick_size is None:
        return {**base, **signal, "signal_status": "pass", "action": "block", "block_reason": "invalid_tick_size"}
    if bid_px is None or ask_px is None or ask_px <= bid_px:
        return {**base, **signal, "signal_status": "pass", "action": "block", "block_reason": "invalid_hyperliquid_bbo"}
    if mid_px is None:
        return {**base, **signal, "signal_status": "pass", "action": "block", "block_reason": "missing_hyperliquid_mid"}

    signal_score = float(signal["signal_score"])
    side = side_from_signal(signal_score, str(contract["side_mapping"]))
    signed_expected_move_ticks = signal_score * float(expected_move_ticks_per_signal_z)
    fair_mid_px = mid_px + signed_expected_move_ticks * tick_size
    quote_px = bid_px if side == "buy" else ask_px
    if side == "buy":
        edge_ticks = (fair_mid_px - quote_px) / tick_size
    else:
        edge_ticks = (quote_px - fair_mid_px) / tick_size
    required_edge = (
        float(required_edge_ticks)
        if required_edge_ticks is not None
        else float(
            (contract.get("acceptance_limits") or {}).get(
                "fee_adverse_buffer_ticks",
                DEFAULT_REQUIRED_EDGE_TICKS,
            )
        )
    )
    quote_intent = {
        "side": side,
        "quote_px": round(quote_px, 8),
        "quote_type": "touch",
        "time_in_force": POST_ONLY_TIF,
        "post_only": True,
        "single_layer": True,
    }
    decision = {
        **base,
        "signal_status": "pass",
        "signal_reason": "",
        "signal_score": round(signal_score, 8),
        "signal_abs_z": signal["signal_abs_z"],
        "threshold_abs_z": signal["threshold_abs_z"],
        "signal_components": signal["components"],
        "side": side,
        "signed_expected_move_ticks": round(signed_expected_move_ticks, 8),
        "hyperliquid_mid_px": round(mid_px, 8),
        "fair_mid_px": round(fair_mid_px, 8),
        "quote_intent": quote_intent,
        "quote_px": round(quote_px, 8),
        "edge_ticks": round(edge_ticks, 8),
        "required_edge_ticks": required_edge,
        "edge_formula": contract.get("edge_formula", {}),
        "source_age_bucket": market_view.get("source_age_bucket", ""),
        "basis_bucket": market_view.get("basis_bucket", ""),
        "warning_bucket": bool(market_view.get("warning_bucket", False)),
    }
    if edge_ticks <= required_edge:
        return {**decision, "action": "block", "block_reason": "edge_below_required_buffer"}
    return {
        **decision,
        "action": "would_submit",
        "block_reason": "",
        "shadow_action": "would_submit_if_real_order_task_authorized",
    }


def fixture_market_views() -> list[dict[str, Any]]:
    return [
        {
            "decision_id": "fixture_buy_pass",
            "hyperliquid_bid_px": 65000.0,
            "hyperliquid_ask_px": 65001.0,
            "hyperliquid_mid_px": 65000.5,
            "tick_size": 1.0,
            "input_binance_top5_imbalance": 1.3,
            "input_binance_microprice_minus_mid_ticks": 1.1,
            "input_binance_mid_move_ticks_from_prev": 1.0,
            "source_age_bucket": "binance_source_age_low",
            "basis_bucket": "basis_mid",
        },
        {
            "decision_id": "fixture_sell_pass",
            "hyperliquid_bid_px": 65000.0,
            "hyperliquid_ask_px": 65001.0,
            "hyperliquid_mid_px": 65000.5,
            "tick_size": 1.0,
            "input_binance_top5_imbalance": -1.4,
            "input_binance_microprice_minus_mid_ticks": -1.2,
            "input_binance_mid_move_ticks_from_prev": -1.1,
            "source_age_bucket": "binance_source_age_high",
            "basis_bucket": "basis_low",
        },
        {
            "decision_id": "fixture_signal_below_threshold",
            "hyperliquid_bid_px": 65000.0,
            "hyperliquid_ask_px": 65001.0,
            "hyperliquid_mid_px": 65000.5,
            "tick_size": 1.0,
            "input_binance_top5_imbalance": 0.2,
            "input_binance_microprice_minus_mid_ticks": 0.1,
            "input_binance_mid_move_ticks_from_prev": 0.0,
            "source_age_bucket": "binance_source_age_low",
            "basis_bucket": "basis_mid",
        },
        {
            "decision_id": "fixture_missing_feature_block",
            "hyperliquid_bid_px": 65000.0,
            "hyperliquid_ask_px": 65001.0,
            "hyperliquid_mid_px": 65000.5,
            "tick_size": 1.0,
            "input_binance_top5_imbalance": 1.3,
            "input_binance_microprice_minus_mid_ticks": 1.1,
            "source_age_bucket": "binance_source_age_low",
            "basis_bucket": "basis_mid",
        },
        {
            "decision_id": "fixture_warning_bucket_visible",
            "hyperliquid_bid_px": 65000.0,
            "hyperliquid_ask_px": 65001.0,
            "hyperliquid_mid_px": 65000.5,
            "tick_size": 1.0,
            "input_binance_top5_imbalance": 1.1,
            "input_binance_microprice_minus_mid_ticks": 1.0,
            "input_binance_mid_move_ticks_from_prev": 1.0,
            "source_age_bucket": "binance_source_age_mid",
            "basis_bucket": "basis_low",
            "warning_bucket": True,
            "warning_source": "xemm_0627_t001_hlfast_utc17_b/binance_source_age_mid",
        },
    ]


def build_fixture_artifacts(
    *,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    signal_manifest_path: Path = DEFAULT_SIGNAL_MANIFEST_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    contract = load_signal_contract(contract_path)
    normalization_stats = default_normalization_stats(contract)
    kernel_parameters = {
        "expected_move_ticks_per_signal_z": DEFAULT_EXPECTED_MOVE_TICKS_PER_SIGNAL_Z,
        "required_edge_ticks": float((contract.get("acceptance_limits") or {}).get("fee_adverse_buffer_ticks", DEFAULT_REQUIRED_EDGE_TICKS)),
        "quote_policy": "single_layer_touch_post_only",
    }
    market_views = fixture_market_views()
    decisions = [
        evaluate_shared_kernel(
            row,
            contract=contract,
            normalization_stats=normalization_stats,
            expected_move_ticks_per_signal_z=kernel_parameters["expected_move_ticks_per_signal_z"],
            required_edge_ticks=kernel_parameters["required_edge_ticks"],
        )
        for row in market_views
    ]
    signal_manifest = _read_json(signal_manifest_path) if signal_manifest_path.exists() else {}
    warnings = signal_manifest.get("blocking_or_warning_reasons", [])
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_inputs = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "contract_path": str(contract_path),
        "normalization_stats": normalization_stats,
        "kernel_parameters": kernel_parameters,
        "market_views": market_views,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task_id": "0625T003",
        "source_contract_path": str(contract_path),
        "accepted_candidate": contract.get("candidate_id", ""),
        "feature_schema": contract.get("feature_schema", []),
        "horizon_ms": contract.get("horizon_ms"),
        "effective_horizon_row_condition": contract.get("effective_horizon_row_condition", ""),
        "normalization_policy": contract.get("normalization", ""),
        "threshold_abs_z": contract.get("threshold_abs_z"),
        "side_mapping": contract.get("side_mapping", ""),
        "kernel_parameters": kernel_parameters,
        "fixture_input_count": len(market_views),
        "fixture_output_count": len(decisions),
        "would_submit_fixture_count": sum(1 for row in decisions if row.get("action") == "would_submit"),
        "block_fixture_count": sum(1 for row in decisions if row.get("action") == "block"),
        "t003_warning_reasons": warnings,
        "warning_bucket_visible_in_fixture": any(row.get("warning_bucket") for row in decisions),
        "final_recommendation": "shared_signal_quote_intent_kernel_ready_for_qa",
        "output_files": {
            "shared_kernel_manifest": str(output_dir / "shared_kernel_manifest.json"),
            "fixture_inputs": str(output_dir / "fixture_inputs.json"),
            "fixture_outputs": str(output_dir / "fixture_outputs.json"),
            "boundary_manifest": str(output_dir / "boundary_manifest.json"),
        },
    }
    boundary_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "public_market_data_only": True,
        "offline_local_processing_only": True,
        "no_network_collection": True,
        "no_aws_execution": True,
        "no_remote_alignment": True,
        "no_credentials": True,
        "no_private_account_order_cancel_endpoints": True,
        "no_user_stream": True,
        "no_live_client_initialization": True,
        "no_live_orders": True,
        "no_shadow_execution": True,
        "no_strategy_or_watcher_change": True,
        "no_production_config_change": True,
        "no_signal_feature_search": True,
        "no_threshold_tuning": True,
        "no_side_mapping_change": True,
        "no_horizon_change": True,
        "no_canary_or_promotion_authorization": True,
    }
    _write_json(output_dir / "fixture_inputs.json", fixture_inputs)
    _write_json(output_dir / "fixture_outputs.json", {"schema_version": SCHEMA_VERSION, "task_id": TASK_ID, "decisions": decisions})
    _write_json(output_dir / "shared_kernel_manifest.json", manifest)
    _write_json(output_dir / "boundary_manifest.json", boundary_manifest)
    return {
        "manifest": manifest,
        "fixture_inputs": fixture_inputs,
        "fixture_outputs": decisions,
        "boundary_manifest": boundary_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate shared signal/quote-intent kernel fixtures")
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--signal-manifest-path", type=Path, default=DEFAULT_SIGNAL_MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generate-fixtures", action="store_true")
    args = parser.parse_args()
    if not args.generate_fixtures:
        parser.print_help()
        return
    result = build_fixture_artifacts(
        contract_path=args.contract_path,
        signal_manifest_path=args.signal_manifest_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
