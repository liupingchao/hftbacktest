# Canonical Regime Synthesis Report

Task: `0604T009`

## Scope

- Input directory: `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Signal ranking directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_signal_quality_ranking_0604T006`
- Horizon/regime diagnostics directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_horizon_regime_diagnostics_0604T007`
- Output directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_regime_synthesis_0604T009`
- Inputs are existing canonical event-mode artifacts only and are guarded through the accepted loader/source-lock path.
- Primary candidate anchor is limited to `binance_mid_move_ticks_from_prev`.
- `binance_top5_imbalance`, `binance_top5_bid_qty`, and `binance_microprice_minus_mid_ticks` are secondary context only.

## Result

- Classification counts: `{"candidate_for_milestone3_executability": 1, "reject_concentrated_or_aliased": 2, "reject_unstable_direction": 9, "watch_needs_more_samples": 6}`
- Candidate rows for Milestone 3 executability assessment: `1`
- Top candidate/watch rows:
  - `regime_011_1000_spread_10_20_ticks` horizon `1000`: binance_mid_move_ticks_from_prev=high_or_positive_impulse_bucket; hyperliquid_context_quality=primary_usable; hyperliquid_join_age_bucket=fresh_0_50ms; hyperliquid_spread_bucket=spread_10_20_ticks; reason `primary_anchor_and_context_supported_at_1000ms_plus`

## Artifacts

- `candidate_regime_definitions.csv`
- `candidate_regime_evidence_summary.csv`
- `candidate_regime_watch_reject_list.csv`
- `canonical_regime_synthesis_manifest.json`
- `canonical_regime_synthesis_report.md`

## Boundary

- This is read-only regime synthesis only.
- No maker side, quote placement, order behavior, strategy action, case-library construction, shadow decision generation, private/order endpoints, order lifecycle, parameter search, live/default-on/tiny-live, or promotion is authorized.
