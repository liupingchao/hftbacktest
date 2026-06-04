# Canonical Signal Quality Ranking Report

Task: `0604T006`

## Scope

- Input directory: `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Output directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_signal_quality_ranking_0604T006`
- Inputs are existing local `0604T003` canonical event-mode artifacts loaded through `0604T004` foundation.
- Ranking is limited to the four `0601T004` primary Binance lead allowlist features.

## Ranking Method

- Score dimensions: direction consistency, effect size, mean absolute correlation, usable row/sample count, independent future-row-delta count, sample concentration, and 100/250ms versus 500ms+/1000ms+ reliance.
- `100/250ms` evidence is tracked as weakly independent context; keep decisions require stable 500ms+ and 1000ms+ canonical evidence.
- Buckets are limited to `keep_for_read_only_research`, `watch_regime_dependent`, and `reject_for_canonical_signal_ranking`.

## Feature Ranking

| Rank | Feature | Bucket | Score | Reason | Controller Alignment |
|---:|---|---|---:|---|---|
| 1 | `binance_mid_move_ticks_from_prev` | `keep_for_read_only_research` | 0.8366 | stable multi-sample canonical evidence at 500ms+ and 1000ms+ | `matches_current_controller_interpretation` |
| 2 | `binance_top5_imbalance` | `keep_for_read_only_research` | 0.8056 | stable multi-sample canonical evidence at 500ms+ and 1000ms+ | `matches_current_controller_interpretation` |
| 3 | `binance_top5_bid_qty` | `watch_regime_dependent` | 0.7008 | aggregate score remains below keep threshold; contract/controller treats bid quantity as liquidity/context | `matches_current_controller_interpretation` |
| 4 | `binance_microprice_minus_mid_ticks` | `watch_regime_dependent` | 0.5859 | 500ms+ stability is below keep threshold; 1000ms+ stability is below keep threshold; aggregate score remains below keep threshold; controller interpretation is regime-dependent microprice dislocation | `matches_current_controller_interpretation` |

## Keep / Watch / Reject

### keep_for_read_only_research

- `binance_mid_move_ticks_from_prev`: stable multi-sample canonical evidence at 500ms+ and 1000ms+
- `binance_top5_imbalance`: stable multi-sample canonical evidence at 500ms+ and 1000ms+

### watch_regime_dependent

- `binance_top5_bid_qty`: aggregate score remains below keep threshold; contract/controller treats bid quantity as liquidity/context
- `binance_microprice_minus_mid_ticks`: 500ms+ stability is below keep threshold; 1000ms+ stability is below keep threshold; aggregate score remains below keep threshold; controller interpretation is regime-dependent microprice dislocation

### reject_for_canonical_signal_ranking

- None

## Boundary

- This is read-only signal quality ranking evidence only.
- It does not authorize regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, parameter search, live/default-on/tiny-live, or promotion.
