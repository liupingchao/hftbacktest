from __future__ import annotations

from decimal import Decimal

import pytest

from examples.hyperliquid import cross_exchange_price_math as price_math


@pytest.mark.parametrize(
    ("value", "sz_decimals", "expected_buy", "expected_sell"),
    [
        (65000.99, 5, 65000.0, 65001.0),
        (12.34567, 0, 12.345, 12.346),
        (0.123456, 0, 0.12345, 0.12346),
    ],
)
def test_directional_normalization_respects_precision(
    value: float,
    sz_decimals: int,
    expected_buy: float,
    expected_sell: float,
) -> None:
    buy = price_math.normalize_hl_perp_price(value, sz_decimals=sz_decimals, side="buy")
    sell = price_math.normalize_hl_perp_price(value, sz_decimals=sz_decimals, side="sell")

    assert buy == expected_buy
    assert sell == expected_sell
    price_math.assert_hl_perp_price_valid(buy, sz_decimals=sz_decimals)
    price_math.assert_hl_perp_price_valid(sell, sz_decimals=sz_decimals)


@pytest.mark.parametrize("side", ["buy", "sell", "nearest"])
def test_normalization_is_idempotent(side: str) -> None:
    first = price_math.normalize_hl_perp_price(12345.678, sz_decimals=5, side=side)  # type: ignore[arg-type]
    second = price_math.normalize_hl_perp_price(first, sz_decimals=5, side=side)  # type: ignore[arg-type]

    assert second == first


def test_post_only_clamps_crossing_quotes_and_preserves_side() -> None:
    buy = price_math.post_only_price(101.0, side="buy", best_bid=100.0, best_ask=100.1, sz_decimals=5)
    sell = price_math.post_only_price(99.0, side="sell", best_bid=100.0, best_ask=100.1, sz_decimals=5)

    assert buy == 100.0
    assert sell == 100.1
    assert buy < 100.1
    assert sell > 100.0


def test_low_price_decimal_precision_and_significant_digits() -> None:
    value = price_math.normalize_hl_perp_price(Decimal("0.001234567"), sz_decimals=0, side="nearest")

    assert value == 0.001235
    price_math.assert_hl_perp_price_valid(value, sz_decimals=0)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), "not-a-price"])
def test_invalid_prices_fail_closed(value: object) -> None:
    with pytest.raises(ValueError):
        price_math.normalize_hl_perp_price(value, sz_decimals=5)


def test_invalid_bbo_and_nonpositive_size_decimals_fail_closed() -> None:
    with pytest.raises(ValueError, match="best_ask"):
        price_math.post_only_price(100.0, side="buy", best_bid=100.0, best_ask=99.0, sz_decimals=5)
    with pytest.raises(ValueError, match="sz_decimals"):
        price_math.normalize_hl_perp_price(100.0, sz_decimals=-1)
