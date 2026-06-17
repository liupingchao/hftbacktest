# Next Runner Contract

Task: `0601T004`

## Allowed Task Type

The next implementation task may be a read-only pricing-signal runner. It must not be a strategy implementation or live-trading task.

## Inputs

- Accepted `0601T002` style joined-feature artifacts or later QA-accepted equivalent.
- Accepted `0601T003` style lead-lag verdict artifacts or later QA-accepted equivalent.
- Accepted `0531T001` style Hyperliquid public market-view artifacts or later QA-accepted equivalent.
- `0601T004` feature decision table and data input schema.
- `0608T005` basis-context visibility / lineage decision may be used only to treat `context_basis_mid_ticks` as read-only decision-time context with execution-PnL caveat retained.

## Required Outputs

- `run_manifest.json`
- `pricing_signal_rows.csv`
- `pricing_signal_feature_quality.csv`
- `horizon_label_summary.csv`
- `feature_stability_by_regime.csv`
- `venue_state_conditioning_summary.csv`
- `pricing_signal_recommendation.md`

## Required Quality Reporting

- Primary row count and excluded row count.
- Future join count and missing Binance join count.
- Binance source age p50/p90/p99/max.
- Hyperliquid venue-state join age/cadence/recovery quality.
- Nominal horizon and effective future-age distribution.
- Trade-pressure disabled status.
- Boundary flags for no private/order/live/parameter/default-on/tiny-live/promotion.

## Allowed Conclusions

- `keep_for_read_only_research`
- `needs_more_public_samples`
- `reject_for_runner_design`

The runner must not output `strategy_ready`, `signal_ready`, `tiny_live_ready`, `default_on_ready`, or `promotion_ready`.

`context_basis_mid_ticks` may appear only as read-only context. It must not be converted into an executable signal, case-library trigger, shadow decision, or live/promotion claim.

## Hard Boundaries

- No private keys.
- No private account endpoints.
- No order endpoints.
- No submit/cancel/fill/order lifecycle logic.
- No strategy behavior changes.
- No live trading process.
- No parameter search.
- No default-on behavior.
- No tiny-live.
- No promotion.
- No standard npz schema, canonical audit schema, connector, or core API change.

## Acceptance Gate For The Later Runner

The later runner should pass QA only if it proves that all decision-time inputs are visible at `hyperliquid_decision_ts`, all future labels are separated, and all primary evidence is derived from accepted local public artifacts.
