# Canonical Maker Executability Report

Task: `0608T002`

## Scope

- Formal input directory: `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Candidate directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_regime_synthesis_0604T009`
- Output directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_maker_executability_0608T002`
- Assessed candidate: `regime_011_1000_spread_10_20_ticks` only.
- Inputs are existing local canonical event-mode public-data artifacts guarded through the accepted source-lock path.
- All fill, spread, adverse-selection, post-only, latency, and inventory findings are read-only public-data proxies.

## Result

- Final recommendation: `reject_not_maker_executable`
- Public proxy row count: `242`
- Fill opportunity proxy: `fillable_proxy_supported`
- Spread capture proxy: `spread_capture_negative`
- Adverse selection proxy: `adverse_selection_reject`
- Quote churn proxy: `manageable_churn`
- Post-only risk proxy: `reject_post_only_risk`
- Latency sensitivity proxy: `latency_reject`
- Inventory what-if: `flat_only_watch`

## Boundary

- This report does not prove real fill probability, real post-only reject rate, private order lifecycle, inventory behavior, or live execution.
- No maker side, quote price, size, cancel rule, strategy action, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, or promotion is authorized.
