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
NEAR_POSITION_CAP_RATIO = 0.8
MAX_QUOTE_LADDER_LEVELS = 8


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
        if (
            isinstance(self.levels, bool)
            or not isinstance(self.levels, int)
            or not 1 <= self.levels <= MAX_QUOTE_LADDER_LEVELS
        ):
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


@dataclass(frozen=True)
class ReservationResult:
    forecast_mid_px: float
    position_btc: float
    mid_px: float
    max_position_btc: float
    raw_position_ratio: float
    bounded_position_ratio: float
    position_notional: float
    inventory_skew_ticks_at_max: float
    inventory_penalty_ticks: float
    inventory_penalty_px: float
    reservation_px: float
    hard_cap_breached: bool


@dataclass(frozen=True)
class TwoSidedQuotes:
    desired_bid_px: float
    desired_ask_px: float
    bid_px: float
    ask_px: float
    bid_clamp_reason: str
    ask_clamp_reason: str
    bid_edge_change_ticks: float
    ask_edge_change_ticks: float
    post_only_invariant: bool

    def quote_intents(self) -> list[dict[str, Any]]:
        return [
            {
                "side": "buy",
                "quote_px": round(self.bid_px, 8),
                "desired_quote_px": round(self.desired_bid_px, 8),
                "quote_type": "reservation_bid",
                "time_in_force": POST_ONLY_TIF,
                "post_only": True,
                "level": 1,
                "clamp_reason": self.bid_clamp_reason,
                "edge_change_ticks": round(self.bid_edge_change_ticks, 8),
            },
            {
                "side": "sell",
                "quote_px": round(self.ask_px, 8),
                "desired_quote_px": round(self.desired_ask_px, 8),
                "quote_type": "reservation_ask",
                "time_in_force": POST_ONLY_TIF,
                "post_only": True,
                "level": 1,
                "clamp_reason": self.ask_clamp_reason,
                "edge_change_ticks": round(self.ask_edge_change_ticks, 8),
            },
        ]


LADDER_CONFIG_SCHEMA_VERSION = "cross_exchange_quote_ladder_config_v1"


@dataclass(frozen=True)
class QuoteLadderConfigV1:
    schema_version: str = LADDER_CONFIG_SCHEMA_VERSION
    levels: int = 1
    gap_ticks: float = 1.0
    size_decay: float = 1.0
    min_size_btc: float = 0.00001
    max_size_btc: float = 0.005
    max_total_size_btc: float = 0.01
    activation_enabled: bool = False
    single_level_lifecycle_prerequisite: bool = False
    coalesce_duplicate_prices: bool = True

    def __post_init__(self) -> None:
        if self.schema_version != LADDER_CONFIG_SCHEMA_VERSION:
            raise ValueError("unsupported_quote_ladder_config_schema")
        if (
            isinstance(self.levels, bool)
            or not isinstance(self.levels, int)
            or not 1 <= self.levels <= MAX_QUOTE_LADDER_LEVELS
        ):
            raise ValueError("quote_ladder_levels_must_be_positive_integer")
        for field_name in (
            "gap_ticks",
            "min_size_btc",
            "max_size_btc",
            "max_total_size_btc",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"quote_ladder_invalid:{field_name}")
            if float(value) <= 0:
                raise ValueError(f"quote_ladder_{field_name}_must_be_positive")
        if (
            isinstance(self.size_decay, bool)
            or not isinstance(self.size_decay, (int, float))
            or not math.isfinite(float(self.size_decay))
            or not 0 < float(self.size_decay) <= 1
        ):
            raise ValueError("quote_ladder_size_decay_must_be_between_zero_and_one")
        if self.min_size_btc > self.max_size_btc:
            raise ValueError("quote_ladder_min_size_exceeds_max_size")
        if self.max_total_size_btc < self.max_size_btc:
            raise ValueError("quote_ladder_total_cap_below_single_quote_cap")
        if type(self.activation_enabled) is not bool:
            raise ValueError("quote_ladder_activation_enabled_must_be_bool")
        if type(self.single_level_lifecycle_prerequisite) is not bool:
            raise ValueError("quote_ladder_lifecycle_prerequisite_must_be_bool")
        if type(self.coalesce_duplicate_prices) is not bool:
            raise ValueError("quote_ladder_coalesce_duplicate_prices_must_be_bool")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "levels": self.levels,
            "gap_ticks": self.gap_ticks,
            "size_decay": self.size_decay,
            "min_size_btc": self.min_size_btc,
            "max_size_btc": self.max_size_btc,
            "max_total_size_btc": self.max_total_size_btc,
            "activation_enabled": self.activation_enabled,
            "single_level_lifecycle_prerequisite": self.single_level_lifecycle_prerequisite,
            "coalesce_duplicate_prices": self.coalesce_duplicate_prices,
        }


def multi_level_prerequisite_gate(
    *,
    requested_levels: int,
    activation_enabled: bool = False,
    single_level_lifecycle_prerequisite: bool = False,
) -> dict[str, Any]:
    if isinstance(requested_levels, bool) or not isinstance(requested_levels, int) or requested_levels < 1:
        raise ValueError("requested_levels_must_be_positive_integer")
    if requested_levels == 1:
        return {
            "status": "single_level_authoritative",
            "reason": "multi_level_not_requested",
            "requested_levels": requested_levels,
            "activation_enabled": False,
            "single_level_lifecycle_prerequisite": single_level_lifecycle_prerequisite,
            "actual_quote_behavior_changed": False,
        }
    if not single_level_lifecycle_prerequisite:
        reason = "single_level_lifecycle_prerequisite_not_satisfied"
    elif not activation_enabled:
        reason = "multi_level_activation_disabled"
    else:
        reason = ""
    return {
        "status": "pass" if not reason else "blocked",
        "reason": reason,
        "requested_levels": requested_levels,
        "activation_enabled": bool(not reason),
        "single_level_lifecycle_prerequisite": single_level_lifecycle_prerequisite,
        "actual_quote_behavior_changed": bool(not reason),
    }


def _floor_to_lot(size_btc: float, lot_size_btc: float) -> float:
    units = math.floor((size_btc / lot_size_btc) + 1e-9)
    return round(units * lot_size_btc, 12)


def _ladder_price_key(price: float, *, sz_decimals: int) -> str:
    normalized = cross_exchange_price_math.normalize_hl_perp_price(
        price,
        sz_decimals=sz_decimals,
        side="nearest",
    )
    return f"{normalized:.12f}".rstrip("0").rstrip(".") or "0"


def build_default_off_quote_ladder(
    *,
    reservation_px: float,
    half_spread_ticks: float,
    best_bid: float,
    best_ask: float,
    precision: Mapping[str, Any] | float,
    bid_size_btc: float,
    ask_size_btc: float,
    config: QuoteLadderConfigV1,
    eligible_sides: tuple[str, ...] = ("buy", "sell"),
) -> dict[str, Any]:
    """Build a guarded ladder while preserving default-off behavior."""

    reservation = _finite_positive(reservation_px)
    half_spread = _float(half_spread_ticks)
    bid = _finite_positive(best_bid)
    ask = _finite_positive(best_ask)
    bid_size = _finite_positive(bid_size_btc)
    ask_size = _finite_positive(ask_size_btc)
    if isinstance(precision, Mapping):
        tick_size = _finite_positive(precision.get("tick_size"))
        sz_decimals = precision.get("sz_decimals", 5)
        lot_size = _finite_positive(precision.get("lot_size"))
    else:
        tick_size = _finite_positive(precision)
        sz_decimals = 5
        lot_size = 10 ** (-int(sz_decimals)) if tick_size is not None else None
    if reservation is None or half_spread is None or half_spread <= 0:
        return {
            "status": "fail_closed",
            "reason": "invalid_reservation_or_half_spread",
            "ladder_rows": [],
            "quote_intents": [],
            "activation_enabled": False,
            "actual_quote_behavior_changed": False,
        }
    if bid is None or ask is None or ask <= bid or tick_size is None or lot_size is None:
        return {
            "status": "fail_closed",
            "reason": "invalid_bbo_or_precision",
            "ladder_rows": [],
            "quote_intents": [],
            "activation_enabled": False,
            "actual_quote_behavior_changed": False,
        }
    if bid_size is None or ask_size is None or sz_decimals is None:
        return {
            "status": "fail_closed",
            "reason": "invalid_ladder_size_input",
            "ladder_rows": [],
            "quote_intents": [],
            "activation_enabled": False,
            "actual_quote_behavior_changed": False,
        }
    if any(side not in {"buy", "sell"} for side in eligible_sides):
        raise ValueError("ladder_eligible_side_invalid")

    gate = multi_level_prerequisite_gate(
        requested_levels=config.levels,
        activation_enabled=config.activation_enabled,
        single_level_lifecycle_prerequisite=config.single_level_lifecycle_prerequisite,
    )
    base_quotes = compute_two_sided_quotes(
        reservation_px=reservation,
        half_spread_ticks=half_spread,
        best_bid=bid,
        best_ask=ask,
        precision={"tick_size": tick_size, "sz_decimals": sz_decimals},
    )
    base_by_side = {"buy": base_quotes.bid_px, "sell": base_quotes.ask_px}
    base_size_by_side = {"buy": bid_size, "sell": ask_size}
    rows: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for side in eligible_sides:
        for level in range(config.levels):
            if side == "buy":
                desired_px = reservation - (half_spread + level * config.gap_ticks) * tick_size
            else:
                desired_px = reservation + (half_spread + level * config.gap_ticks) * tick_size
            if level == 0:
                quote_px = base_by_side[side]
            else:
                normalized = cross_exchange_price_math.normalize_hl_perp_price(
                    desired_px,
                    sz_decimals=sz_decimals,
                    side=side,
                )
                quote_px = cross_exchange_price_math.post_only_price(
                    normalized,
                    side=side,
                    best_bid=bid,
                    best_ask=ask,
                    sz_decimals=sz_decimals,
                )
            size_before_lot = base_size_by_side[side] * (config.size_decay ** level)
            size_btc = _floor_to_lot(size_before_lot, lot_size)
            if size_btc < config.min_size_btc or size_btc > config.max_size_btc:
                return {
                    "status": "fail_closed",
                    "reason": f"invalid_level_size:{side}:level_{level}",
                    "ladder_rows": rows,
                    "quote_intents": [],
                    "activation_enabled": False,
                    "actual_quote_behavior_changed": False,
                }
            if quote_px <= 0 or (side == "buy" and quote_px >= ask) or (side == "sell" and quote_px <= bid):
                return {
                    "status": "fail_closed",
                    "reason": f"post_only_invariant_failed:{side}:level_{level}",
                    "ladder_rows": rows,
                    "quote_intents": [],
                    "activation_enabled": False,
                    "actual_quote_behavior_changed": False,
                }
            price_key = _ladder_price_key(
                quote_px,
                sz_decimals=int(sz_decimals),
            )
            key = (side, price_key)
            existing = by_key.get(key)
            if existing is not None:
                if not config.coalesce_duplicate_prices:
                    return {
                        "status": "fail_closed",
                        "reason": f"duplicate_rounded_price:{side}:{price_key}",
                        "ladder_rows": rows,
                        "quote_intents": [],
                        "activation_enabled": False,
                        "actual_quote_behavior_changed": False,
                    }
                existing["size_btc"] = round(existing["size_btc"] + size_btc, 12)
                existing["coalesced_levels"].append(level)
                if existing["size_btc"] > config.max_size_btc + 1e-12:
                    return {
                        "status": "fail_closed",
                        "reason": f"coalesced_size_exceeds_max:{side}:{price_key}",
                        "ladder_rows": rows,
                        "quote_intents": [],
                        "activation_enabled": False,
                        "actual_quote_behavior_changed": False,
                    }
                continue
            row = {
                "side": side,
                "level": level,
                "desired_px": round(desired_px, 12),
                "quote_px": round(quote_px, 12),
                "price_key": price_key,
                "size_btc": size_btc,
                "coalesced_levels": [level],
                "post_only": True,
                "time_in_force": POST_ONLY_TIF,
            }
            by_key[key] = row
            rows.append(row)

    total_size = sum(float(row["size_btc"]) for row in rows)
    if total_size > config.max_total_size_btc + 1e-12:
        return {
            "status": "fail_closed",
            "reason": "aggregate_ladder_exposure_cap_exceeded",
            "ladder_rows": rows,
            "quote_intents": [],
            "activation_enabled": False,
            "actual_quote_behavior_changed": False,
        }
    rows = sorted(rows, key=lambda row: (str(row["side"]), int(row["level"]), str(row["price_key"])))
    executable: list[dict[str, Any]] = []
    reason = gate["reason"]
    if config.levels == 1:
        reason = "single_level_authoritative"
    elif not reason:
        reason = "multi_level_activation_enabled"
        executable = [dict(row) for row in rows]
    activation_enabled = bool(executable)
    return {
        "status": (
            "pass"
            if activation_enabled
            else (
                "pass_observe_only"
                if config.levels == 1
                else "blocked"
            )
        ),
        "reason": reason,
        "gate": gate,
        "config": config.to_dict(),
        "ladder_rows": rows,
        "hypothetical_quote_intents": [dict(row) for row in rows],
        "quote_intents": executable,
        "working_exposure_btc": round(total_size, 12),
        "activation_enabled": activation_enabled,
        "actual_quote_behavior_changed": activation_enabled,
        "inference_scope": (
            "guarded_multi_level_ladder_execution"
            if activation_enabled
            else "default_off_multi_level_ladder_contract"
        ),
    }


def inventory_quote_sides(position_ratio: float) -> tuple[tuple[str, ...], str]:
    ratio = _float(position_ratio)
    if ratio is None:
        raise ValueError("position_ratio_must_be_finite")
    if ratio >= NEAR_POSITION_CAP_RATIO:
        return ("sell",), "reduce_only_long"
    if ratio <= -NEAR_POSITION_CAP_RATIO:
        return ("buy",), "reduce_only_short"
    return ("buy", "sell"), "two_sided"


def compute_reservation_price(
    *,
    forecast_mid_px: float,
    position_btc: float,
    mid_px: float,
    max_position_btc: float,
    inventory_skew_ticks_at_max: float,
    price_increment: float = 1.0,
) -> ReservationResult:
    forecast = _finite_positive(forecast_mid_px)
    mid = _finite_positive(mid_px)
    position = _float(position_btc)
    max_position = _finite_positive(max_position_btc)
    skew = _float(inventory_skew_ticks_at_max)
    increment = _finite_positive(price_increment)
    if forecast is None:
        raise ValueError("forecast_mid_px_must_be_finite_positive")
    if mid is None:
        raise ValueError("mid_px_must_be_finite_positive")
    if position is None:
        raise ValueError("position_btc_must_be_finite")
    if max_position is None:
        raise ValueError("max_position_btc_must_be_finite_positive")
    if skew is None or skew < 0:
        raise ValueError("inventory_skew_ticks_at_max_must_be_finite_nonnegative")
    if increment is None:
        raise ValueError("price_increment_must_be_finite_positive")
    raw_ratio = position / max_position
    bounded_ratio = max(-1.0, min(1.0, raw_ratio))
    penalty_ticks = bounded_ratio * skew
    penalty_px = penalty_ticks * increment
    reservation_px = forecast - penalty_px
    if reservation_px <= 0:
        raise ValueError("reservation_px_must_be_positive_after_inventory_penalty")
    return ReservationResult(
        forecast_mid_px=forecast,
        position_btc=position,
        mid_px=mid,
        max_position_btc=max_position,
        raw_position_ratio=raw_ratio,
        bounded_position_ratio=bounded_ratio,
        position_notional=abs(position) * mid,
        inventory_skew_ticks_at_max=skew,
        inventory_penalty_ticks=penalty_ticks,
        inventory_penalty_px=penalty_px,
        reservation_px=reservation_px,
        hard_cap_breached=abs(position) > max_position,
    )


def _quote_clamp_reason(
    *,
    desired_px: float,
    normalized_px: float,
    final_px: float,
    side: str,
) -> str:
    if final_px == desired_px:
        return ""
    if final_px == normalized_px:
        return "precision_normalization"
    if side == "buy":
        return "post_only_crossing_clamp_to_best_bid"
    return "post_only_crossing_clamp_to_best_ask"


def compute_two_sided_quotes(
    *,
    reservation_px: float,
    half_spread_ticks: float,
    best_bid: float,
    best_ask: float,
    precision: Mapping[str, Any] | float,
) -> TwoSidedQuotes:
    reservation = _finite_positive(reservation_px)
    bid = _finite_positive(best_bid)
    ask = _finite_positive(best_ask)
    half_spread = _float(half_spread_ticks)
    if reservation is None:
        raise ValueError("reservation_px_must_be_finite_positive")
    if bid is None or ask is None or ask <= bid:
        raise ValueError("best_ask_must_exceed_best_bid")
    if half_spread is None or half_spread <= 0:
        raise ValueError("half_spread_ticks_must_be_finite_positive")
    if isinstance(precision, Mapping):
        tick_size = _finite_positive(precision.get("tick_size"))
        sz_decimals = precision.get("sz_decimals", 5)
    else:
        tick_size = _finite_positive(precision)
        sz_decimals = 5
    if tick_size is None:
        raise ValueError("tick_size_must_be_finite_positive")
    desired_bid = reservation - half_spread * tick_size
    desired_ask = reservation + half_spread * tick_size
    normalized_bid = cross_exchange_price_math.normalize_hl_perp_price(
        desired_bid,
        sz_decimals=sz_decimals,
        side="buy",
    )
    normalized_ask = cross_exchange_price_math.normalize_hl_perp_price(
        desired_ask,
        sz_decimals=sz_decimals,
        side="sell",
    )
    final_bid = cross_exchange_price_math.post_only_price(
        desired_bid,
        side="buy",
        best_bid=bid,
        best_ask=ask,
        sz_decimals=sz_decimals,
    )
    final_ask = cross_exchange_price_math.post_only_price(
        desired_ask,
        side="sell",
        best_bid=bid,
        best_ask=ask,
        sz_decimals=sz_decimals,
    )
    post_only_invariant = final_bid < ask and final_ask > bid and final_bid < final_ask
    if not post_only_invariant:
        raise ValueError("two_sided_post_only_invariant_failed")
    return TwoSidedQuotes(
        desired_bid_px=desired_bid,
        desired_ask_px=desired_ask,
        bid_px=final_bid,
        ask_px=final_ask,
        bid_clamp_reason=_quote_clamp_reason(
            desired_px=desired_bid,
            normalized_px=normalized_bid,
            final_px=final_bid,
            side="buy",
        ),
        ask_clamp_reason=_quote_clamp_reason(
            desired_px=desired_ask,
            normalized_px=normalized_ask,
            final_px=final_ask,
            side="sell",
        ),
        bid_edge_change_ticks=(final_bid - desired_bid) / tick_size,
        ask_edge_change_ticks=(final_ask - desired_ask) / tick_size,
        post_only_invariant=post_only_invariant,
    )


def build_observe_only_pricing_overlay(
    *,
    fixed_half_spread_ticks: float,
    dynamic_candidate_half_spread_ticks: float | None = None,
    activation_enabled: bool = False,
) -> dict[str, Any]:
    """Describe an estimator overlay without changing the authoritative quote."""

    fixed = _finite_positive(fixed_half_spread_ticks)
    candidate = _float(dynamic_candidate_half_spread_ticks)
    if fixed is None:
        raise ValueError("fixed_half_spread_ticks_must_be_finite_positive")
    if candidate is not None and candidate <= 0:
        raise ValueError("dynamic_candidate_half_spread_ticks_must_be_positive")
    if activation_enabled:
        raise ValueError("observe_only_pricing_overlay_cannot_activate_dynamic_spread")
    return {
        "fixed_half_spread_ticks": fixed,
        "dynamic_candidate_half_spread_ticks": "" if candidate is None else candidate,
        "activation_enabled": False,
        "authoritative_half_spread_ticks": fixed,
        "quote_behavior_changed": False,
        "inference_scope": "dynamic_spread_candidate_audit_only",
    }


def build_bounded_dynamic_pricing_overlay(
    *,
    fixed_half_spread_ticks: float,
    dynamic_candidate: Mapping[str, Any] | None,
    activation_enabled: bool,
    min_half_spread_ticks: float = 0.5,
    max_half_spread_ticks: float = 10.0,
) -> dict[str, Any]:
    """Resolve the sole bounded dynamic-spread input for live quoting.

    The caller still owns the activation decision. This pure resolver makes
    invalid, stale, or out-of-range candidates fall back to the fixed base
    without relaxing any quote or risk guard.
    """

    fixed = _finite_positive(fixed_half_spread_ticks)
    minimum = _finite_positive(min_half_spread_ticks)
    maximum = _finite_positive(max_half_spread_ticks)
    if fixed is None or minimum is None or maximum is None or minimum > maximum:
        raise ValueError("invalid_dynamic_spread_bounds")

    candidate = dict(dynamic_candidate or {})
    candidate_value = _float(
        candidate.get("half_spread_ticks", candidate.get("bounded_half_spread_ticks"))
    )
    status = str(candidate.get("status") or "")
    bounded = candidate.get("bounded") is True
    valid_candidate = (
        activation_enabled
        and status == "pass"
        and bounded
        and candidate_value is not None
        and minimum <= candidate_value <= maximum
    )
    authoritative = candidate_value if valid_candidate else fixed
    fallback_reason = ""
    if activation_enabled and not valid_candidate:
        fallback_reason = (
            str(candidate.get("reason") or "dynamic_candidate_unavailable_or_invalid")
        )
    return {
        "fixed_half_spread_ticks": fixed,
        "dynamic_candidate_half_spread_ticks": (
            "" if candidate_value is None else candidate_value
        ),
        "candidate_status": status,
        "candidate_bounded": bounded,
        "activation_enabled": bool(activation_enabled),
        "authoritative_half_spread_ticks": authoritative,
        "quote_behavior_changed": not math.isclose(
            authoritative, fixed, rel_tol=0.0, abs_tol=1e-12
        ),
        "fallback_to_fixed": bool(activation_enabled and not valid_candidate),
        "fallback_reason": fallback_reason,
        "hard_bounds": {
            "min_half_spread_ticks": minimum,
            "max_half_spread_ticks": maximum,
        },
        "inference_scope": (
            "bounded_dynamic_spread_quote_input"
            if activation_enabled
            else "dynamic_spread_candidate_audit_only"
        ),
    }


def build_bounded_fill_feedback_pricing_overlay(
    *,
    fixed_half_spread_ticks: float,
    fill_feedback_candidate: Mapping[str, Any] | None,
    activation_enabled: bool,
    min_half_spread_ticks: float = 0.5,
    max_half_spread_ticks: float = 10.0,
    max_abs_offset_ticks: float = 2.0,
) -> dict[str, Any]:
    """Resolve the sole bounded fill-feedback quote input.

    A missing target, censored lifecycle, invalid candidate, or out-of-range
    offset keeps the fixed base authoritative. This resolver never owns
    kill-switch, risk, or post-only decisions.
    """

    fixed = _finite_positive(fixed_half_spread_ticks)
    minimum = _finite_positive(min_half_spread_ticks)
    maximum = _finite_positive(max_half_spread_ticks)
    max_offset = _finite_positive(max_abs_offset_ticks)
    if (
        fixed is None
        or minimum is None
        or maximum is None
        or max_offset is None
        or minimum > maximum
    ):
        raise ValueError("invalid_fill_feedback_bounds")
    candidate = dict(fill_feedback_candidate or {})
    offset = _float(
        candidate.get("bounded_offset_ticks", candidate.get("raw_offset_ticks"))
    )
    status = str(candidate.get("status") or "")
    candidate_activation = candidate.get("activation_enabled") is True
    candidate_observe_only = candidate.get("observe_only") is True
    valid_candidate = (
        activation_enabled
        and candidate_activation
        and not candidate_observe_only
        and status == "pass"
        and offset is not None
        and abs(offset) <= max_offset
    )
    authoritative = fixed
    if valid_candidate:
        authoritative = max(minimum, min(maximum, fixed + offset))
    fallback_reason = ""
    if activation_enabled and not valid_candidate:
        fallback_reason = str(
            candidate.get("reason")
            or "fill_feedback_candidate_unavailable_or_invalid"
        )
    return {
        "fixed_half_spread_ticks": fixed,
        "candidate_offset_ticks": "" if offset is None else offset,
        "candidate_status": status,
        "candidate_activation_enabled": candidate_activation,
        "candidate_observe_only": candidate_observe_only,
        "activation_enabled": bool(activation_enabled),
        "authoritative_half_spread_ticks": authoritative,
        "quote_behavior_changed": not math.isclose(
            authoritative, fixed, rel_tol=0.0, abs_tol=1e-12
        ),
        "fallback_to_fixed": bool(activation_enabled and not valid_candidate),
        "fallback_reason": fallback_reason,
        "hard_bounds": {
            "min_half_spread_ticks": minimum,
            "max_half_spread_ticks": maximum,
            "max_abs_offset_ticks": max_offset,
        },
        "inference_scope": (
            "bounded_fill_feedback_quote_input"
            if activation_enabled
            else "fill_feedback_candidate_audit_only"
        ),
    }


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
    signal_side = side_from_signal(signal_score, str(contract["side_mapping"])) if signal_score else "both"
    try:
        signed_expected_move_ticks = signal_score * pricing_config.expected_move_ticks_per_signal_z
        forecast_mid_px = pricing_base_px + signed_expected_move_ticks * tick_size
        position_raw = market_view.get("position_btc")
        position_btc = 0.0 if position_raw is None else _float(position_raw)
        if position_btc is None:
            raise ValueError("position_btc_must_be_finite")
        reservation = compute_reservation_price(
            forecast_mid_px=forecast_mid_px,
            position_btc=position_btc,
            mid_px=mid_px,
            max_position_btc=pricing_config.max_position_btc,
            inventory_skew_ticks_at_max=(
                pricing_config.inventory_skew_ticks_at_max
                if pricing_config.enable_inventory_skew
                else 0.0
            ),
            price_increment=tick_size,
        )
        two_sided_quotes = compute_two_sided_quotes(
            reservation_px=reservation.reservation_px,
            half_spread_ticks=pricing_config.base_half_spread_ticks,
            best_bid=bid_px,
            best_ask=ask_px,
            precision={
                "tick_size": tick_size,
                "sz_decimals": market_view.get("sz_decimals", 5),
            },
        )
    except (TypeError, ValueError) as exc:
        return {
            **base,
            **signal,
            "signal_status": "pass",
            "action": "block",
            "block_reason": f"invalid_hyperliquid_price:{exc}",
        }
    quote_bid_px = two_sided_quotes.bid_px
    quote_ask_px = two_sided_quotes.ask_px
    eligible_side_tuple, inventory_mode = inventory_quote_sides(reservation.raw_position_ratio)
    eligible_sides = set(eligible_side_tuple)
    near_or_over_cap = inventory_mode != "two_sided"
    side = (
        signal_side
        if signal_side in eligible_sides
        else eligible_side_tuple[0]
        if len(eligible_side_tuple) == 1
        else "both"
    )
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
    all_quote_intents = two_sided_quotes.quote_intents()
    quote_intents = [row for row in all_quote_intents if row["side"] in eligible_sides]
    legacy_quote_px = (
        quote_bid_px
        if side == "buy" and side in eligible_sides
        else quote_ask_px
        if side == "sell" and side in eligible_sides
        else None
    )
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
        "signal_side": signal_side,
        "alpha_adjustment_ticks": round(signed_expected_move_ticks, 8),
        "signed_expected_move_ticks": round(signed_expected_move_ticks, 8),
        "hl_mid_px": round(mid_px, 8),
        "hl_micro_px": round(hl_micro_px, 8) if hl_micro_px is not None else None,
        "fair_base": fair_base,
        "microprice_reason": micro_reason,
        "forecast_mid_px": round(forecast_mid_px, 8),
        "fair_mid_px": round(forecast_mid_px, 8),
        "position_btc": reservation.position_btc,
        "position_ratio": round(reservation.raw_position_ratio, 8),
        "bounded_position_ratio": round(reservation.bounded_position_ratio, 8),
        "position_notional": round(reservation.position_notional, 8),
        "inventory_skew_enabled": pricing_config.enable_inventory_skew,
        "inventory_penalty_ticks": round(reservation.inventory_penalty_ticks, 8),
        "inventory_penalty_px": round(reservation.inventory_penalty_px, 8),
        "reservation_px": round(reservation.reservation_px, 8),
        "hard_cap_breached": reservation.hard_cap_breached,
        "inventory_mode": inventory_mode,
        "inventory_worsening_side": (
            "buy"
            if reservation.position_btc > 0
            else "sell"
            if reservation.position_btc < 0
            else ""
        ),
        "half_spread_ticks": pricing_config.base_half_spread_ticks,
        "quote_bid_px": round(quote_bid_px, 8),
        "quote_ask_px": round(quote_ask_px, 8),
        "desired_bid_px": round(two_sided_quotes.desired_bid_px, 8),
        "desired_ask_px": round(two_sided_quotes.desired_ask_px, 8),
        "bid_clamp_reason": two_sided_quotes.bid_clamp_reason,
        "ask_clamp_reason": two_sided_quotes.ask_clamp_reason,
        "bid_edge_change_ticks": round(two_sided_quotes.bid_edge_change_ticks, 8),
        "ask_edge_change_ticks": round(two_sided_quotes.ask_edge_change_ticks, 8),
        "post_only_invariant": two_sided_quotes.post_only_invariant,
        "all_quote_intents": all_quote_intents,
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
        "quote_eligibility": "reduce_only" if near_or_over_cap else "eligible",
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
