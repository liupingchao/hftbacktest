# Lead-Lag Recommendation

Task: `0601T003`

## Scope

- This is read-only lead-lag research evidence over local-observation-time joined features.
- It is not a strategy signal, parameter search, tiny-live readiness, default-on readiness, or promotion artifact.

## Verdict Counts

- stable_enough_for_pricing_research: `18`
- watch_only: `6`
- unstable: `30`
- insufficient_samples: `0`

## Interpretation

- Stable-enough feature/outcome pairs exist for later read-only pricing-signal runner design.
- `binance_microprice_minus_mid_ticks` -> `basis_microprice_response_ticks`: 2 passing horizons share negative sign
- `binance_microprice_minus_mid_ticks` -> `hyperliquid_mid_move_ticks`: 2 passing horizons share positive sign
- `binance_microprice_minus_mid_ticks` -> `hyperliquid_top5_imbalance_change`: 4 passing horizons share positive sign
- `binance_mid_move_ticks_from_prev` -> `basis_microprice_response_ticks`: 3 passing horizons share negative sign
- `binance_mid_move_ticks_from_prev` -> `basis_mid_response_ticks`: 2 passing horizons share negative sign
- `binance_mid_move_ticks_from_prev` -> `hyperliquid_mid_move_ticks`: 3 passing horizons share positive sign
- `binance_top5_bid_qty` -> `basis_microprice_response_ticks`: 3 passing horizons share positive sign
- `binance_top5_bid_qty` -> `basis_mid_response_ticks`: 3 passing horizons share positive sign
- `binance_top5_bid_qty` -> `hyperliquid_microprice_minus_mid_change_ticks`: 4 passing horizons share positive sign
- `binance_top5_bid_qty` -> `hyperliquid_mid_move_ticks`: 6 passing horizons share positive sign

## Data Quality

- Input rows: `3599`
- Primary rows: `3596`
- Excluded rows: `3`
- Horizons ms: `100,250,500,1000,5000,10000`
