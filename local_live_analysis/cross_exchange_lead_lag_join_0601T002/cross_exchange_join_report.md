# Cross-Exchange Lead/Lag Join Report

Task: `0601T002`

Source sample: `/home/molly/project/hftbacktest/local_live_analysis/cross_exchange_public_sample_0602T001`

## Scope

- Reads only accepted synchronized public-data artifacts from `0602T001`.
- Joins Binance USD-M Futures `BTCUSDT` lead rows to Hyperliquid `BTC` lag decision timestamps.
- Uses local controller capture timestamps as the cross-venue clock: `binance_local_ts <= hyperliquid_decision_ts`.
- Does not calculate lead-lag stability, predictive edge, strategy readiness, tiny-live readiness, or promotion.

## Row Counts

- Binance lead feature rows: `67211`
- Hyperliquid lag context rows: `3599`
- Joined feature rows: `3599`
- Primary usable joined rows: `3596`
- Watch/diagnostic joined rows: `3`

## Join Quality

- Future cross-exchange joins: `0`
- Missing Binance joins: `0`
- Stale Binance source rows: `0`
- Binance source age p99 ms: `32.91798816`

## Disabled Features

- Binance trade pressure: `disabled_unverified_side_semantics`
- Hyperliquid trade pressure: `disabled_unverified_side_semantics`

## Caveats

- Basis/dislocation fields are diagnostic only because they compare Binance USD-M Futures `BTCUSDT` with Hyperliquid `BTC` contract context.
- Output is synchronized public-data joined-feature input for `0601T003` only.
