# Canonical Signal Quality Ranking Report

Task: `0604T006`

## Scope

- Input directory: `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Output directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_signal_quality_ranking_0604T006`
- Inputs are existing `0604T003` canonical event-mode aggregate artifacts only.
- `100/250ms` evidence is treated as weakly independent; ranking prioritizes `500ms+` and especially `1000ms+` rows.

## Ranking

- Rank `1` `binance_mid_move_ticks_from_prev`: `keep_for_read_only_research`, score `0.8047634`; stable_1000_plus=15/15; consistency_1000_plus=1; independent_future_row_delta_1000_plus=4
- Rank `2` `binance_top5_imbalance`: `watch_regime_dependent`, score `0.7184303`; stable_1000_plus=13/15; consistency_1000_plus=0.95555556; independent_future_row_delta_1000_plus=4; sample_concentration_watch=0.7473146; short_horizon_reliance_watch=0.35714286
- Rank `3` `binance_top5_bid_qty`: `watch_regime_dependent`, score `0.64361754`; stable_1000_plus=11/15; consistency_1000_plus=0.91111111; independent_future_row_delta_1000_plus=4; sample_concentration_watch=0.77663751; short_horizon_reliance_watch=0.38461538
- Rank `4` `binance_microprice_minus_mid_ticks`: `watch_regime_dependent`, score `0.5473174`; stable_1000_plus=7/15; consistency_1000_plus=0.82222222; independent_future_row_delta_1000_plus=4; sample_concentration_watch=0.77993258

## Controller Interpretation Check

- `binance_mid_move_ticks_from_prev` remains the strongest global signal candidate when using `1000ms+` canonical evidence.
- `binance_top5_imbalance` remains the strongest book-pressure candidate but is watch-labeled by the strict concentration and short-horizon reliance caveats.
- `binance_top5_bid_qty` remains useful liquidity/context evidence but is ranked behind the two stronger global candidates.
- `binance_microprice_minus_mid_ticks` is classified as watch/regime-dependent because short horizons are unstable and broader evidence is less consistent.

## Boundary

- This is read-only signal quality ranking only.
- No new data collection, regime selection, case-library construction, shadow decision generation, strategy implementation, private/order endpoints, order lifecycle, parameter search, live/default-on/tiny-live, or promotion is authorized.
