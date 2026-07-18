# Cross-Exchange Price Taxonomy Contract

## Scope

All Hyperliquid perp prices that can reach a quote intent must use
`examples/hyperliquid/cross_exchange_price_math.py`. Callers must not maintain
independent significant-digit or decimal-place rounding rules.

## Precision

For `szDecimals`, the valid price has:

- no more than five significant digits;
- no more than `max(0, 6 - szDecimals)` decimal places;
- a finite, strictly positive value.

`normalize_hl_perp_price()` is directional:

- `buy` floors;
- `sell` ceils;
- `nearest` uses half-up rounding.

Normalization is idempotent. Invalid values fail closed.

## Post-Only

`post_only_price()` first normalizes in the quote direction, then clamps a
crossing buy to the best bid or a crossing sell to the best ask. It rejects a
crossed or invalid BBO and verifies that the final price remains strictly:

- buy `< best_ask`;
- sell `> best_bid`.

The helper returns the final price, while callers remain responsible for
recording desired and final values when they have an audit artifact.
