# Canonical Horizon / Regime Diagnostics Report

Task: `0604T007`

## Scope

- Input directory: `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Output directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_horizon_regime_diagnostics_0604T007`
- Inputs are existing local canonical event-mode aggregate artifacts only.
- The runner uses the `0604T004` canonical loader/guard path and rejects diagnostic-only synthetic inputs.

## Horizon Findings

- Canonical sample count: `3`
- Horizon support counts: `{"diagnostic_supported": 4, "watch_needs_more_samples": 2}`
- Diagnostic-supported horizons: `500, 1000, 5000, 10000`
- Watch/reject horizons: `100, 250`
- `100/250ms` horizons remain watch-only because public `l2Book` cadence can weakly alias short nominal horizons.
- `500ms+` horizons are reported separately from preferred `1000ms+` support.

## Regime Conditioning Findings

- Regime bucket support counts: `{"diagnostic_supported": 10, "reject_aliased_or_concentrated": 2, "watch_needs_more_samples": 6}`
- Regime rows include sample count, row count, direction consistency, effect concentration, and horizon-level correlation context.
- All reported buckets remain watch-only diagnostics, not final high-confidence regime definitions.

## Artifacts

- `horizon_independence_diagnostics.csv`
- `regime_conditioning_diagnostics.csv`
- `regime_watch_list.csv`
- `horizon_regime_diagnostics_manifest.json`
- `horizon_regime_diagnostics_report.md`

## Boundary

- No new collection is authorized.
- No final regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, parameter search, live/default-on/tiny-live, or promotion is authorized.
