# Hyperliquid Tiny-Live Optimistic PnL Proxy

Task: `0617T006`

## Scope

This is a read-only optimistic PnL proxy over local public pricing-signal rows.
It assumes every eligible theoretical Hyperliquid maker intent fills at the
theoretical quote, then settles against future public Hyperliquid mid labels.

This is not real PnL. It does not model fill probability, queue priority,
post-only reject behavior, live order lifecycle, account inventory,
fees/rebates, spread-capture settlement, execution quality, maker viability,
deployment readiness, or live authorization.

## Sample-Set Reconciliation

The user originally requested `6` datasets, but accepted local manifests expose:

- `7` canonical event-mode samples in `0609T008`
- `8` total inputs consumed by accepted `0617T005` when `0601T005` is included

The user subsequently selected `canonical_7` as the formal sample-set口径. The
runner therefore records:

- `requested_six`: superseded by user-selected `canonical_7`
- `canonical_7`: official computed sample set
- `0617T005_8_input`: computed diagnostic comparison set

Reconciliation output:

- `local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006/sample_set_reconciliation.csv`

## Rule Source

The runner uses the `0617T004` / `0617T005` rule sources:

- positive eligible signal -> Hyperliquid maker buy intent
- negative eligible signal -> Hyperliquid maker sell intent
- primary threshold candidate: `75` ticks, persistence `2`
- stricter fallback: `75` ticks, persistence `3`
- sensitivity grid: thresholds `50,75,100`, persistence `1,2,3`

Eligibility uses the same primary/fresh/persistence checks as the `0617T005`
replay, but the main PnL output is explicitly `unconstrained_all_intents`; it
does not apply the `0617T005` simulated position cap.

## Formula

Tick size: `0.1`

Order size: `0.01 BTC`

USDC per tick: `0.001`

- buy: `optimistic_mid_pnl_ticks = hyperliquid_future_mid_move_ticks + context_hyperliquid_spread_ticks / 2`
- sell: `optimistic_mid_pnl_ticks = -hyperliquid_future_mid_move_ticks + context_hyperliquid_spread_ticks / 2`
- USDC: `optimistic_mid_pnl_usdc = optimistic_mid_pnl_ticks * 0.1 * 0.01`

## Primary Diagnostic Result

At `75` ticks / persistence `2` / `1000ms` fixed horizon:

- `canonical_7`: `7231` fixed-horizon rows, `295.985 USDC` optimistic proxy,
  mean `40.932789` ticks per intent, all `7/7` samples positive.
- `0617T005_8_input`: `7597` fixed-horizon rows, `299.38 USDC` optimistic
  proxy, mean `39.407661` ticks per intent, all `8/8` samples positive.

The official `canonical_7` sample set classifies as `materially_positive` under
the optimistic public-data upper-bound interpretation. The `0617T005_8_input`
diagnostic comparison set is also positive.

## Oracle Upper Bound

The runner also emits `non_tradeable_oracle_upper_bound`, which chooses the
best future horizon per eligible intent. This is intentionally not tradable and
must not be interpreted as a strategy result.

At `75` ticks / persistence `2`:

- `canonical_7`: `7238` oracle intents, `1098.535 USDC` optimistic proxy.
- `0617T005_8_input`: `7604` oracle intents, `1124.89 USDC` optimistic proxy.

## Outputs

- `optimistic_pnl_proxy_manifest.json`
- `sample_set_reconciliation.csv`
- `sample_set_membership.csv`
- `eligibility_summary.csv`
- `fixed_horizon_pnl_summary.csv`
- `aggregate_fixed_horizon_pnl_summary.csv`
- `oracle_best_horizon_summary.csv`
- `diagnostic_interpretation.csv`
- `row_level_audit_sample.csv`

Output directory:

- `local_live_analysis/hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006/`

## Final Recommendation

- `hyperliquid_tiny_live_optimistic_pnl_proxy_ready_for_qa`

The optimistic proxy is materially positive on the official `canonical_7`
sample set. This remains read-only evidence only and does not authorize live
execution.
