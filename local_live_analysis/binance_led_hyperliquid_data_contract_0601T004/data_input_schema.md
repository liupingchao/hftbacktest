# Data Input Schema

Task: `0601T004`

This schema is for a later read-only pricing-signal runner. It is not a live strategy schema.

## Required Row Identity

- `sample_id`: accepted local sample identifier.
- `hyperliquid_decision_ts`: Hyperliquid decision/context timestamp from local capture clock.
- `binance_local_ts`: Binance source timestamp from local capture clock.
- `joined_row_quality`: primary row classifier.
- `horizon_ms`: nominal future label horizon.
- `effective_future_age_ms`: actual future-row age used for the label.

## Binance Lead Input Columns

Primary allowlist:

- `binance_top5_imbalance`
- `binance_microprice_minus_mid_ticks`
- `binance_mid_move_ticks_from_prev`
- `binance_top5_bid_qty`

Diagnostic/context only:

- `binance_top5_microprice_px`
- `binance_rolling_abs_mid_move_ticks_5`
- `binance_rolling_rv_ticks_20`
- `binance_top5_ask_qty`
- `binance_top5_total_qty`

Disabled:

- `binance_trade_pressure`
- Any Binance feature derived from future rows, private state, order lifecycle, fills, or after-the-fact labels.

## Hyperliquid Venue-State Input Columns

Allowed context fields:

- `hyperliquid_mid_px`
- `hyperliquid_spread_ticks`
- `hyperliquid_top1_imbalance`
- `hyperliquid_top3_imbalance`
- `hyperliquid_top5_imbalance`
- `hyperliquid_microprice_minus_mid_ticks`
- `hyperliquid_top5_bid_qty`
- `hyperliquid_top5_ask_qty`
- `hyperliquid_join_age_ms`
- `hyperliquid_l2book_cadence_ms`
- `hyperliquid_recovery_state`
- `hyperliquid_market_view_quality`

Disabled:

- `hyperliquid_trade_pressure`
- Private/account/order endpoint fields.
- Submit/cancel/fill/order lifecycle fields.

## Joint Context

- `basis_mid_dislocation_ticks`: read-only decision-time context only after `0608T005`; keep the execution-PnL caveat.
- `basis_microprice_dislocation_ticks`: diagnostic context only.
- `basis_caveat`

Basis fields must carry the Binance USD-M `BTCUSDT` vs Hyperliquid `BTC` contract caveat. `basis_mid_dislocation_ticks` may be used as context in later read-only research only; it must not be treated as executable strategy PnL, private execution proof, case-library trigger, shadow decision, live/default-on/tiny-live, or promotion evidence.

## Future Labels

Future labels are labels only, never decision-time inputs:

- `hyperliquid_future_mid_move_ticks`
- `hyperliquid_future_microprice_minus_mid_change_ticks`
- `hyperliquid_future_top5_imbalance_change`
- `basis_future_mid_response_ticks`
- `basis_future_microprice_response_ticks`

Every label row must include:

- `horizon_ms`
- `future_hyperliquid_decision_ts`
- `effective_future_age_ms`
- `label_row_quality`

## Primary Exclusion Rules

Exclude from primary analysis:

- `joined_row_quality != primary_usable`
- missing Binance as-of source
- future Binance source
- stale Binance source beyond accepted gate
- missing Hyperliquid future row for the requested horizon
- disabled trade-pressure features
- private/order/fill/lifecycle-derived fields

Rows may remain in watch/diagnostic summaries if the output records the exclusion reason.
