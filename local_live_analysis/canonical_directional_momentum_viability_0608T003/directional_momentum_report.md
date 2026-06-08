# Canonical Directional Momentum Viability Report

Task: `0608T003`

## Scope

- Formal input directory: `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- Candidate directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_regime_synthesis_0604T009`
- T002 maker executability directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_maker_executability_0608T002`
- Output directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_directional_momentum_viability_0608T003`
- Assessed regime: `regime_011_1000_spread_10_20_ticks` only.
- Row-level files are resolved from `multi_sample_manifest.json` `samples[].pricing_signal_rows`.
- Inputs are existing local canonical event-mode public-data artifacts guarded through the accepted source-lock path.

## Result

- Final recommendation: `reject_directional_edge_unstable`
- Directionality: `directional_signal_reject`
- Cost-adjusted viability: `net_edge_negative_proxy`
- Stability: `sample_unstable_reject`
- Tail risk: `tail_risk_reject`
- Base row count: `242`
- Mean signed future move ticks: `-0.92975207`
- Net directional edge proxy ticks: `-8.07024793`

## Boundary

- This report is a read-only public-data proxy assessment, not executable strategy PnL or private execution proof.
- No executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, shadow decision generation, deployment recommendation, or promotion is authorized.
