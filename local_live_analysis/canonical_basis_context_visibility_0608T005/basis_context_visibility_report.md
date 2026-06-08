# Basis Context Visibility Report

Task: `0608T005`

## Result

- Contract decision: `upgrade_to_context_only_supported`
- Lineage: `lineage_confirmed_decision_time_formula`
- Timestamp: `timestamp_clean_asof`
- Max sample row share: `0.38596491`
- Decision reason: basis is as-of clean and formula-derived; retain execution-PnL caveat but remove decision-visibility blocker

## Scope

- Assessed regime: `regime_011_1000_spread_10_20_ticks` only.
- Assessed pattern: `context_basis_mid_ticks > 0` only.
- Basis formula audited as `(binance_mid_px - hyperliquid_mid_px) / 0.1` from decision-time as-of joined rows.

## Boundary

- This report is a read-only public-data proxy visibility / lineage diagnosis, not executable strategy PnL or private execution proof.
- No executable trading instruction, actual order side, quote price, size, leverage, stop rule, take-profit rule, strategy action, shadow decision, private/order endpoint, order lifecycle, live/default-on/tiny-live, parameter search, case-library implementation, deployment recommendation, or promotion is authorized.
