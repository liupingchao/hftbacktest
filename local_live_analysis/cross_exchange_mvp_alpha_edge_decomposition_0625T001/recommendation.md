# Alpha / Edge Decomposition Recommendation

Task: `0625T001`

## Recommendation

- `needs_more_public_samples`

## Finding

- Historical event-mode evidence supports directional Binance top5 alpha across three samples.
- The current production public-shadow signal shape is not ready to freeze: only four rows reached edge, one source was stale, and the fresh rows did not provide edge above the current buffer.
- Production anti-drift rows have no same-window future markout, so the gate cannot yet be classified as helpful or over-filtering.
- This result requests more synchronized public evidence; it does not lower the seven-tick edge policy.

## Root Causes

1. `production_edge_sample_coverage_insufficient`: only four production candidates reached fair-mid/edge; one was stale
2. `candidate_side_not_aligned_with_observed_lead_move`: opposed=2, zero=1; valid edge values=[-24.5, -24.5, 0.5]
3. `effective_horizon_timing_mismatch`: materially delayed nominal horizons=[100, 250]; status_by_horizon={100: 'materially_delayed', 250: 'materially_delayed', 500: 'aligned', 1000: 'aligned'}
4. `basis_conditioning`: max bucket mean range=47.0199146 ticks for basis_mid_ticks at 1000ms
5. `hyperliquid_venue_state_conditioning`: max bucket mean range=34.33546961 ticks for hyperliquid_top5_imbalance at 1000ms
6. `anti_drift_throughput_dominates`: anti-drift block=64, pass=4; same-window future markout is absent
7. `fair_mid_source_freshness`: fair-mid pass=3, block=1
8. `historical_alpha_exists_but_live_projection_is_unfrozen`: 4/4 allowlist features stable across three samples at 1000ms

## Required T002 Evidence

- Separated windows: at least `3`
- Duration per window: at least `30` minutes
- Edge-evaluable rows: at least `100` aggregate and `20` per window
- Regimes: at least `2` distinct volatility/liquidity regimes
- Required on every decision: dual top5, local/exchange timestamps, source seq/age, signal components, lead_move_ticks, candidate side/quote, anti-drift result, fair-mid, edge, and future HL mid/microprice labels.

## Boundary

- Offline/public-only/no-submit.
- No live behavior change, credentials, private/order endpoints, quote relaxation, canary, M3, stable-PnL, default-on, or promotion authorization.
