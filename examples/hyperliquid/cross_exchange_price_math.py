"""Authoritative Hyperliquid perp price normalization and post-only math."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP
from typing import Literal


PriceSide = Literal["buy", "sell", "nearest"]
MAX_SIGNIFICANT_DIGITS = 5


def _as_positive_decimal(value: object, *, field: str) -> Decimal:
    try:
        decimal = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"{field}_must_be_finite_positive") from exc
    if not decimal.is_finite() or decimal <= 0:
        raise ValueError(f"{field}_must_be_finite_positive")
    return decimal


def _validate_sz_decimals(sz_decimals: int) -> int:
    if isinstance(sz_decimals, bool) or not isinstance(sz_decimals, int) or sz_decimals < 0:
        raise ValueError("sz_decimals_must_be_nonnegative_integer")
    return sz_decimals


def _price_quantum(price: Decimal, *, sz_decimals: int) -> Decimal:
    max_price_decimals = max(0, 6 - _validate_sz_decimals(sz_decimals))
    significant_exponent = price.adjusted() - (MAX_SIGNIFICANT_DIGITS - 1)
    exponent = max(significant_exponent, -max_price_decimals)
    return Decimal(1).scaleb(exponent)


def normalize_hl_perp_price(
    px: float | int | str | Decimal,
    *,
    sz_decimals: int,
    side: PriceSide = "nearest",
) -> float:
    """Normalize a positive price to Hyperliquid's precision constraints."""

    if side not in {"buy", "sell", "nearest"}:
        raise ValueError(f"unsupported_price_side:{side}")
    price = _as_positive_decimal(px, field="price")
    quantum = _price_quantum(price, sz_decimals=sz_decimals)
    rounding = {
        "buy": ROUND_FLOOR,
        "sell": ROUND_CEILING,
        "nearest": ROUND_HALF_UP,
    }[side]
    normalized = price.quantize(quantum, rounding=rounding)
    if normalized <= 0:
        raise ValueError("normalized_price_must_be_positive")
    assert_hl_perp_price_valid(normalized, sz_decimals=sz_decimals)
    return float(normalized)


def assert_hl_perp_price_valid(
    px: float | int | str | Decimal,
    *,
    sz_decimals: int,
) -> None:
    """Raise when a price violates significant-digit or decimal-place limits."""

    price = _as_positive_decimal(px, field="price")
    max_price_decimals = max(0, 6 - _validate_sz_decimals(sz_decimals))
    normalized = price.normalize()
    significant_digits = len(normalized.as_tuple().digits)
    decimal_places = max(0, -normalized.as_tuple().exponent)
    if significant_digits > MAX_SIGNIFICANT_DIGITS:
        raise ValueError("price_exceeds_five_significant_digits")
    if decimal_places > max_price_decimals:
        raise ValueError("price_exceeds_sz_decimals_precision")


def post_only_price(
    desired_px: float | int | str | Decimal,
    *,
    side: Literal["buy", "sell"],
    best_bid: float | int | str | Decimal,
    best_ask: float | int | str | Decimal,
    sz_decimals: int,
) -> float:
    """Return a normalized quote that is strictly non-crossing."""

    if side not in {"buy", "sell"}:
        raise ValueError(f"unsupported_post_only_side:{side}")
    bid = _as_positive_decimal(best_bid, field="best_bid")
    ask = _as_positive_decimal(best_ask, field="best_ask")
    if ask <= bid:
        raise ValueError("best_ask_must_exceed_best_bid")

    normalized = normalize_hl_perp_price(desired_px, sz_decimals=sz_decimals, side=side)
    normalized_decimal = Decimal(str(normalized))
    if side == "buy" and normalized_decimal >= ask:
        normalized = normalize_hl_perp_price(bid, sz_decimals=sz_decimals, side="buy")
        normalized_decimal = Decimal(str(normalized))
    elif side == "sell" and normalized_decimal <= bid:
        normalized = normalize_hl_perp_price(ask, sz_decimals=sz_decimals, side="sell")
        normalized_decimal = Decimal(str(normalized))

    if side == "buy" and normalized_decimal >= ask:
        raise ValueError("post_only_buy_would_cross_ask")
    if side == "sell" and normalized_decimal <= bid:
        raise ValueError("post_only_sell_would_cross_bid")
    assert_hl_perp_price_valid(normalized, sz_decimals=sz_decimals)
    return normalized
