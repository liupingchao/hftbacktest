# Feature-Conditioned Signal Validity Report

Task: `0608T004`

## Scope

- Formal input directory: `/home/molly/project/hftbacktest/local_live_analysis/event_mode_canonical_pricing_signal_0604T003`
- T003 prerequisite directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_directional_momentum_viability_0608T003`
- Output directory: `/home/molly/project/hftbacktest/local_live_analysis/canonical_feature_conditioned_signal_validity_0608T004`
- Assessed regime: `regime_011_1000_spread_10_20_ticks` only.
- Row-level files are resolved from `multi_sample_manifest.json` `samples[].pricing_signal_rows`.
- Feature scope is fixed to T004's listed fields; this is not a broad feature search.

## Result

- Final recommendation: `watch_needs_contract_visibility_clarification`
- Valid supported pattern count: `0`
- Watch pattern count: `0`
- Invalid pattern count: `21`

## Strongest Recomputed Patterns

- `input_binance_top5_bid_qty` / `sign_positive`: validity `invalid_tail_or_cost_reject`, rows `242`, hit rate `0.51239669`, net edge `-8.07024793`, tail `tail_risk_reject`.
- `context_basis_mid_ticks` / `sign_negative`: validity `invalid_not_decision_visible`, rows `185`, hit rate `0.62162162`, net edge `22.27027027`, tail `tail_risk_reject`.
- `context_hyperliquid_microprice_minus_mid_ticks` / `sign_positive`: validity `invalid_redundant_or_leakage_risk`, rows `133`, hit rate `0.84210526`, net edge `42.91729323`, tail `tail_risk_reject`.
- `context_hyperliquid_top5_imbalance` / `sign_positive`: validity `invalid_redundant_or_leakage_risk`, rows `133`, hit rate `0.84210526`, net edge `42.91729323`, tail `tail_risk_reject`.
- `input_binance_microprice_minus_mid_ticks` / `sign_negative`: validity `invalid_redundant_or_leakage_risk`, rows `123`, hit rate `0.62601626`, net edge `23.03252033`, tail `tail_risk_reject`.

## Boundary

- This report is a read-only public-data proxy diagnosis, not executable strategy PnL or private execution proof.
- No executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, strategy action, shadow decision, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, deployment recommendation, or promotion is authorized.
