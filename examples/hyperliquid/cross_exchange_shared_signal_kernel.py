#!/usr/bin/env python3
"""Shared pure signal/fair-mid/quote-intent kernel for the cross-exchange MVP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import cross_exchange_price_math


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
PRICING_CONFIG_SCHEMA_VERSION = "pricing_config_v1"
MAX_MICROPRICE_AGE_MS = 250.0


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def normalization_stats_hash(normalization_stats: Mapping[str, Mapping[str, Any]]) -> str:
    """Return the stable identity of the exact normalization artifact."""

    try:
        canonical = _canonical_json(normalization_stats)
    except (TypeError, ValueError) as exc:
        raise ValueError("normalization_stats_not_serializable") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PricingConfigV1:
    schema_version: str
    normalization_stats_hash: str
    expected_move_ticks_per_signal_z: float
    base_half_spread_ticks: float
    inventory_skew_ticks_at_max: float
    max_position_btc: float
    enable_microprice: bool
    enable_inventory_skew: bool
    enable_dynamic_spread: bool
    enable_fill_feedback: bool
    levels: int

    def __post_init__(self) -> None:
        if self.schema_version != PRICING_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported_pricing_config_schema")
        if (
            not isinstance(self.normalization_stats_hash, str)
            or len(self.normalization_stats_hash) != 64
            or any(character not in "0123456789abcdef" for character in self.normalization_stats_hash)
        ):
            raise ValueError("invalid_normalization_stats_hash")
        for field_name in (
            "expected_move_ticks_per_signal_z",
            "base_half_spread_ticks",
            "inventory_skew_ticks_at_max",
            "max_position_btc",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
                raise ValueError(f"invalid_pricing_config:{field_name}")
        if self.expected_move_ticks_per_signal_z < 0:
            raise ValueError("invalid_pricing_config:expected_move_ticks_per_signal_z")
        if self.base_half_spread_ticks <= 0:
            raise ValueError("invalid_pricing_config:base_half_spread_ticks")
        if self.inventory_skew_ticks_at_max < 0:
            raise ValueError("invalid_pricing_config:inventory_skew_ticks_at_max")
        if self.max_position_btc <= 0:
            raise ValueError("invalid_pricing_config:max_position_btc")
        for field_name in (
            "enable_microprice",
            "enable_inventory_skew",
            "enable_dynamic_spread",
            "enable_fill_feedback",
        ):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"invalid_pricing_config:{field_name}")
        if isinstance(self.levels, bool) or not isinstance(self.levels, int) or self.levels < 1:
            raise ValueError("invalid_pricing_config:levels")

    @classmethod
    def from_normalization_stats(
        cls,
        normalization_stats: Mapping[str, Mapping[str, Any]],
        *,
        expected_move_ticks_per_signal_z: float = DEFAULT_EXPECTED_MOVE_TICKS_PER_SIGNAL_Z,
        base_half_spread_ticks: float = 0.5,
        inventory_skew_ticks_at_max: float = 0.0,
        max_position_btc: float = 0.01,
        enable_microprice: bool = False,
        enable_inventory_skew: bool = False,
        enable_dynamic_spread: bool = False,
        enable_fill_feedback: bool = False,
        levels: int = 1,
    ) -> "PricingConfigV1":
        return cls(
            schema_version=PRICING_CONFIG_SCHEMA_VERSION,
            normalization_stats_hash=normalization_stats_hash(normalization_stats),
            expected_move_ticks_per_signal_z=float(expected_move_ticks_per_signal_z),
            base_half_spread_ticks=float(base_half_spread_ticks),
            inventory_skew_ticks_at_max=float(inventory_skew_ticks_at_max),
            max_position_btc=float(max_position_btc),
            enable_microprice=enable_microprice,
            enable_inventory_skew=enable_inventory_skew,
            enable_dynamic_spread=enable_dynamic_spread,
            enable_fill_feedback=enable_fill_feedback,
            levels=levels,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PricingConfigV1":
        required = {
            "schema_version",
            "normalization_stats_hash",
            "expected_move_ticks_per_signal_z",
            "base_half_spread_ticks",
            "inventory_skew_ticks_at_max",
            "max_position_btc",
            "enable_microprice",
            "enable_inventory_skew",
            "enable_dynamic_spread",
            "enable_fill_feedback",
            "levels",
        }
        if set(payload) != required:
            raise ValueError("pricing_config_fields_mismatch")
        return cls(**{field: payload[field] for field in required})

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "normalization_stats_hash": self.normalization_stats_hash,
            "expected_move_ticks_per_signal_z": self.expected_move_ticks_per_signal_z,
            "base_half_spread_ticks": self.base_half_spread_ticks,
            "inventory_skew_ticks_at_max": self.inventory_skew_ticks_at_max,
            "max_position_btc": self.max_position_btc,
            "enable_microprice": self.enable_microprice,
            "enable_inventory_skew": self.enable_inventory_skew,
            "enable_dynamic_spread": self.enable_dynamic_spread,
            "enable_fill_feedback": self.enable_fill_feedback,
            "levels": self.levels,
        }

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict()).encode("utf-8")).hexdigest()


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
    """Return identity stats for offline fixtures only.

    Production, shadow, and replay callers must supply the accepted
    normalization artifact instead of using this helper.
    """

    return {
        str(field): {"mean": 0.0, "std": 1.0}
        for field in contract.get("feature_schema", [])
    }


def fixture_normalization_stats(contract: dict[str, Any]) -> dict[str, dict[str, float]]:
    return default_normalization_stats(contract)


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
            "status": "pass",
            "reason": "",
            "candidate_id": contract["candidate_id"],
            "signal_score": round(score, 8),
            "signal_abs_z": round(abs_score, 8),
            "threshold_abs_z": threshold,
            "confidence_bucket": "below_threshold",
            "confidence_reason": "signal_below_threshold",
            "components": component_rows,
        }
    return {
        "status": "pass",
        "reason": "",
        "candidate_id": contract["candidate_id"],
        "signal_score": round(score, 8),
        "signal_abs_z": round(abs_score, 8),
        "threshold_abs_z": threshold,
        "confidence_bucket": "above_threshold",
        "confidence_reason": "",
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
    pricing_config: PricingConfigV1,
    required_edge_ticks: float | None = None,
    expected_move_ticks_per_signal_z: float | None = None,
) -> dict[str, Any]:
    decision_id = str(market_view.get("decision_id") or "")
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "decision_id": decision_id,
        "candidate_id": contract.get("candidate_id", ""),
        "horizon_ms": contract.get("horizon_ms"),
        "normalization": contract.get("normalization", ""),
        "side_mapping": contract.get("side_mapping", ""),
        "post_only_tif": POST_ONLY_TIF,
        "pricing_config_schema_version": pricing_config.schema_version,
        "pricing_config_hash": pricing_config.config_hash,
        "normalization_stats_hash": pricing_config.normalization_stats_hash,
        "pricing_config": pricing_config.to_dict(),
        "order_endpoint_called": False,
        "private_endpoint_called": False,
        "credential_read": False,
        "live_client_initialized": False,
    }
    actual_stats_hash = normalization_stats_hash(normalization_stats)
    if actual_stats_hash != pricing_config.normalization_stats_hash:
        return {
            **base,
            "signal_status": "block",
            "signal_reason": "normalization_stats_hash_mismatch",
            "action": "block",
            "block_reason": "normalization_stats_hash_mismatch",
        }
    if expected_move_ticks_per_signal_z is not None and not math.isclose(
        float(expected_move_ticks_per_signal_z),
        pricing_config.expected_move_ticks_per_signal_z,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return {
            **base,
            "signal_status": "block",
            "signal_reason": "pricing_config_parameter_mismatch",
            "action": "block",
            "block_reason": "pricing_config_parameter_mismatch",
        }

    signal = normalize_signal(market_view, contract=contract, normalization_stats=normalization_stats)
    if signal["status"] != "pass":
        return {
            **base,
            "signal_status": "block",
            "signal_reason": signal["reason"],
            "action": "block",
            "block_reason": signal["reason"],
            "signal_components": signal.get("components", []),
        }

    if market_view.get("signal_stale") is True:
        return {
            **base,
            **signal,
            "signal_status": "block",
            "action": "block",
            "block_reason": "stale_signal",
        }
    signal_age_ms = _float(market_view.get("signal_age_ms"))
    max_signal_age_ms = _float(market_view.get("max_signal_age_ms"))
    if signal_age_ms is not None and max_signal_age_ms is not None and signal_age_ms > max_signal_age_ms:
        return {
            **base,
            **signal,
            "signal_status": "block",
            "action": "block",
            "block_reason": "stale_signal",
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

    if market_view.get("bbo_snapshot_coherent") is False:
        return {
            **base,
            **signal,
            "signal_status": "pass",
            "action": "block",
            "block_reason": "incoherent_bbo_snapshot",
        }

    hl_micro_px: float | None = None
    micro_reason = "microprice_disabled"
    if pricing_config.enable_microprice:
        bid_qty = _float(market_view.get("hyperliquid_bid_qty"))
        ask_qty = _float(market_view.get("hyperliquid_ask_qty"))
        snapshot_id = str(market_view.get("bbo_snapshot_id") or "")
        snapshot_age_ms = _float(market_view.get("bbo_snapshot_age_ms"))
        if (
            bid_qty is not None
            and ask_qty is not None
            and bid_qty > 0
            and ask_qty > 0
            and snapshot_id
            and market_view.get("bbo_snapshot_coherent") is True
            and snapshot_age_ms is not None
            and snapshot_age_ms <= MAX_MICROPRICE_AGE_MS
        ):
            hl_micro_px = (ask_px * bid_qty + bid_px * ask_qty) / (bid_qty + ask_qty)
            micro_reason = "same_snapshot_bbo_qty"
        elif snapshot_age_ms is not None and snapshot_age_ms > MAX_MICROPRICE_AGE_MS:
            micro_reason = "stale_microprice_snapshot"
        elif bid_qty is None or ask_qty is None or bid_qty <= 0 or ask_qty <= 0:
            micro_reason = "invalid_microprice_qty"
        else:
            micro_reason = "microprice_snapshot_unproven"

    fair_base = "hl_micro_px" if hl_micro_px is not None else "mid_fallback"
    pricing_base_px = hl_micro_px if hl_micro_px is not None else mid_px
    signal_score = float(signal["signal_score"])
    side = side_from_signal(signal_score, str(contract["side_mapping"])) if signal_score else "both"
    try:
        signed_expected_move_ticks = signal_score * pricing_config.expected_move_ticks_per_signal_z
        forecast_mid_px = pricing_base_px + signed_expected_move_ticks * tick_size
        position_btc = _float(market_view.get("position_btc")) or 0.0
        inventory_ratio = max(-1.0, min(1.0, position_btc / pricing_config.max_position_btc))
        inventory_penalty_ticks = (
            inventory_ratio * pricing_config.inventory_skew_ticks_at_max
            if pricing_config.enable_inventory_skew
            else 0.0
        )
        reservation_px = forecast_mid_px - inventory_penalty_ticks * tick_size
        half_spread_ticks = pricing_config.base_half_spread_ticks
        bid_desired_px = reservation_px - half_spread_ticks * tick_size
        ask_desired_px = reservation_px + half_spread_ticks * tick_size
        sz_decimals = market_view.get("sz_decimals", 5)
        quote_bid_px = cross_exchange_price_math.post_only_price(
            bid_desired_px,
            side="buy",
            best_bid=bid_px,
            best_ask=ask_px,
            sz_decimals=sz_decimals,
        )
        quote_ask_px = cross_exchange_price_math.post_only_price(
            ask_desired_px,
            side="sell",
            best_bid=bid_px,
            best_ask=ask_px,
            sz_decimals=sz_decimals,
        )
    except (TypeError, ValueError) as exc:
        return {
            **base,
            **signal,
            "signal_status": "pass",
            "action": "block",
            "block_reason": f"invalid_hyperliquid_price:{exc}",
        }
    edge_ticks = (
        (forecast_mid_px - quote_bid_px) / tick_size
        if side == "buy"
        else (quote_ask_px - forecast_mid_px) / tick_size
        if side == "sell"
        else 0.0
    )
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
    quote_intents = [
        {
            "side": "buy",
            "quote_px": round(quote_bid_px, 8),
            "quote_type": "forecast_bid",
            "time_in_force": POST_ONLY_TIF,
            "post_only": True,
            "level": 1,
        },
        {
            "side": "sell",
            "quote_px": round(quote_ask_px, 8),
            "quote_type": "forecast_ask",
            "time_in_force": POST_ONLY_TIF,
            "post_only": True,
            "level": 1,
        },
    ]
    legacy_quote_px = quote_bid_px if side == "buy" else quote_ask_px if side == "sell" else None
    legacy_quote_intent = next((row for row in quote_intents if row["side"] == side), None)
    decision = {
        **base,
        "signal_status": "pass",
        "signal_reason": "",
        "signal_score": round(signal_score, 8),
        "signal_abs_z": signal["signal_abs_z"],
        "threshold_abs_z": signal["threshold_abs_z"],
        "confidence_bucket": signal["confidence_bucket"],
        "confidence_reason": signal["confidence_reason"],
        "signal_components": signal["components"],
        "side": side,
        "alpha_adjustment_ticks": round(signed_expected_move_ticks, 8),
        "signed_expected_move_ticks": round(signed_expected_move_ticks, 8),
        "hl_mid_px": round(mid_px, 8),
        "hl_micro_px": round(hl_micro_px, 8) if hl_micro_px is not None else None,
        "fair_base": fair_base,
        "microprice_reason": micro_reason,
        "forecast_mid_px": round(forecast_mid_px, 8),
        "fair_mid_px": round(forecast_mid_px, 8),
        "inventory_penalty_ticks": round(inventory_penalty_ticks, 8),
        "reservation_px": round(reservation_px, 8),
        "half_spread_ticks": half_spread_ticks,
        "quote_bid_px": round(quote_bid_px, 8),
        "quote_ask_px": round(quote_ask_px, 8),
        "quote_intents": quote_intents,
        "quote_intent": legacy_quote_intent,
        "quote_px": round(legacy_quote_px, 8) if legacy_quote_px is not None else None,
        "edge_ticks": round(edge_ticks, 8),
        "bid_edge_ticks": round((forecast_mid_px - quote_bid_px) / tick_size, 8),
        "ask_edge_ticks": round((quote_ask_px - forecast_mid_px) / tick_size, 8),
        "required_edge_ticks": required_edge,
        "edge_gate_status": "audit_only",
        "edge_formula": contract.get("edge_formula", {}),
        "source_age_bucket": market_view.get("source_age_bucket", ""),
        "basis_bucket": market_view.get("basis_bucket", ""),
        "warning_bucket": bool(market_view.get("warning_bucket", False)),
        "quote_eligibility": "eligible",
    }
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
    normalization_stats = fixture_normalization_stats(contract)
    pricing_config = PricingConfigV1.from_normalization_stats(
        normalization_stats,
        expected_move_ticks_per_signal_z=DEFAULT_EXPECTED_MOVE_TICKS_PER_SIGNAL_Z,
        base_half_spread_ticks=0.5,
        levels=1,
    )
    kernel_parameters = {
        "expected_move_ticks_per_signal_z": DEFAULT_EXPECTED_MOVE_TICKS_PER_SIGNAL_Z,
        "required_edge_ticks": float((contract.get("acceptance_limits") or {}).get("fee_adverse_buffer_ticks", DEFAULT_REQUIRED_EDGE_TICKS)),
        "quote_policy": "two_sided_forecast_post_only",
        "pricing_config": pricing_config.to_dict(),
        "pricing_config_hash": pricing_config.config_hash,
    }
    market_views = fixture_market_views()
    decisions = [
        evaluate_shared_kernel(
            row,
            contract=contract,
            normalization_stats=normalization_stats,
            pricing_config=pricing_config,
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
        "pricing_config": pricing_config.to_dict(),
        "pricing_config_hash": pricing_config.config_hash,
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
        "normalization_stats_hash": pricing_config.normalization_stats_hash,
        "pricing_config": pricing_config.to_dict(),
        "pricing_config_hash": pricing_config.config_hash,
        "threshold_abs_z": contract.get("threshold_abs_z"),
        "side_mapping": contract.get("side_mapping", ""),
        "kernel_parameters": kernel_parameters,
        "fixture_input_count": len(market_views),
        "fixture_output_count": len(decisions),
        "would_submit_fixture_count": sum(1 for row in decisions if row.get("action") == "would_submit"),
        "block_fixture_count": sum(1 for row in decisions if row.get("action") == "block"),
        "t003_warning_reasons": warnings,
        "warning_bucket_visible_in_fixture": any(row.get("warning_bucket") for row in decisions),
        "quote_eligibility_decisions": sum(1 for row in decisions if row.get("quote_eligibility") == "eligible"),
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
