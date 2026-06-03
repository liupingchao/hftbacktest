# Pricing-Signal Recommendation

Task: `0601T005`

## Recommendation

- `keep_for_read_only_research`
- Reason: all hard gates pass; excluded rows are reported and primary rows remain sufficient.

## Scope Boundary

- This is read-only pricing-signal research over accepted local public artifacts.
- This is not strategy-ready, signal-ready, tiny-live-ready, default-on-ready, or promotion-ready evidence.
- Binance and Hyperliquid trade pressure remain disabled because public side semantics are unverified.

## Data Quality

- Primary rows: `3596`
- Excluded rows: `3`
- Horizon label summary rows: `30`

## Primary Allowlist Features

- `binance_top5_imbalance`: stable outcomes `5`, missing `0`
- `binance_microprice_minus_mid_ticks`: stable outcomes `3`, missing `0`
- `binance_mid_move_ticks_from_prev`: stable outcomes `3`, missing `1`
- `binance_top5_bid_qty`: stable outcomes `5`, missing `0`

## Next Boundary

- More synchronized public samples may improve external validity.
- A later task must still separately approve any private/order, strategy, parameter-search, live, default-on, tiny-live, or promotion work.
