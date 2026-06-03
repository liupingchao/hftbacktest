# Binance-Led Hyperliquid Maker Data Input Contract

Task: `0601T004`

## Decision

The next Hyperliquid maker research step may use Binance public market data as the lead-side pricing input and Hyperliquid public market data as the lag-venue state input. This is a data-input contract for a later read-only pricing-signal runner only.

This contract does not authorize strategy implementation, private/order endpoints, order lifecycle modeling, parameter search, live trading, default-on behavior, tiny-live, or promotion.

## Accepted Source Evidence

- `0601T002` passed QA as a local-observation-time as-of join over synchronized public Binance and Hyperliquid artifacts. The join used `binance_local_ts <= hyperliquid_decision_ts`, had `future_join_count=0`, `missing_binance_join_count=0`, `primary_usable_row_count=3596`, and kept Binance/Hyperliquid trade pressure disabled.
- `0601T003` passed QA as the read-only lead-lag stability analyzer over `0601T002`. It reported `3596` primary rows, `3` excluded rows, horizons `100/250/500/1000/5000/10000ms`, and verdict counts `18 stable_enough_for_pricing_research`, `6 watch_only`, `30 unstable`, `0 insufficient_samples`.
- `0601T003` QA accepted effective future-age audit fields. Because the Hyperliquid decision grid is roughly 500ms, nominal `100/250/500ms` horizons may share the same future row. A later runner must keep both nominal horizon and effective future age.
- `0531T001` passed QA as a Hyperliquid public market-data research consumer. It keeps Hyperliquid trade pressure disabled because public trade side semantics remain unverified.

## Venue Ownership

Binance owns lead-side pricing inputs:

- Top5 imbalance and microprice-minus-mid are allowed as lead-side pricing inputs because they produced stable read-only evidence against Hyperliquid mid, book-pressure, and basis response families.
- Binance previous mid move is allowed as a short-horizon momentum input because it produced stable evidence against Hyperliquid mid and basis responses.
- Binance top5 bid quantity is allowed as an asymmetric liquidity-pressure lead input, but a later runner must report that ask quantity and total quantity did not pass the same stability bar.
- Binance rolling realized-volatility fields are allowed only as regime/context fields, not directional pricing inputs.

Hyperliquid owns lag-venue state and execution context:

- Hyperliquid mid, spread, top-N imbalance, microprice-minus-mid, book pressure, join age, cadence, recovery state, and market-view quality are venue-state/context fields.
- Basis/dislocation fields are diagnostic context because current inputs compare Binance USD-M Futures `BTCUSDT` to Hyperliquid `BTC` contract state.
- Post-only, reject/throttle, queue, private order, and fill lifecycle fields are not available in this public-only contract. They may appear only in a later explicitly scoped private/order task after separate approval.

## Feature Decisions

Use `local_live_analysis/binance_led_hyperliquid_data_contract_0601T004/feature_decision_table.csv` as the machine-readable feature decision table.

Primary allowlist for the next read-only pricing-signal runner:

- `binance_top5_imbalance`
- `binance_microprice_minus_mid_ticks`
- `binance_mid_move_ticks_from_prev`
- `binance_top5_bid_qty`

Diagnostic-only inputs:

- `binance_top5_microprice_px`: absolute price level showed some stable basis response, but it should not be treated as a normalized lead signal without demeaning or differencing.
- `binance_rolling_abs_mid_move_ticks_5` and `binance_rolling_rv_ticks_20`: regime/context only.
- `binance_top5_ask_qty` and `binance_top5_total_qty`: liquidity context only unless later evidence proves stable directional value.
- Hyperliquid venue-state fields: allowed as conditioning/context, not as future labels.
- Basis/dislocation: diagnostic context only because venue contracts differ.

Rejected or disabled inputs:

- Binance trade pressure and Hyperliquid trade pressure remain disabled until public trade side semantics are separately proven.
- Any feature/outcome pair classified `unstable` in `0601T003` may not drive candidate signal decisions.
- Any after-the-fact fill, order, private account, or future outcome label may not be used as a decision-time input.

## Timestamp And No-Future Policy

The next runner must preserve the `0601T002` timestamp policy:

- Join clock: `local_controller_capture_ts_ns`.
- Binance-to-Hyperliquid as-of join: `binance_local_ts <= hyperliquid_decision_ts`.
- Future outcome construction: first Hyperliquid row where `future_hyperliquid_decision_ts >= hyperliquid_decision_ts + horizon_ms`.
- The output must record `effective_future_age_ms_min`, `effective_future_age_ms_mean`, and `effective_future_age_ms_max` by horizon and label family.
- Rows with missing Binance join, future Binance join, stale Binance source, or non-primary joined-row quality must be excluded from primary analysis and reported separately.

## Quality Gates

Minimum gates for a later read-only pricing-signal runner:

- Input source must be an accepted local artifact set, not a fresh network collection inside the runner.
- `future_join_count` must be `0`.
- `missing_binance_join_count` must be `0` for primary rows.
- Primary rows must use `joined_row_quality=primary_usable`.
- Binance source age should keep p99 within the accepted `0601T002` envelope unless the run is explicitly classified as watch/diagnostic.
- Hyperliquid public market-view rows must pass the accepted public consumer quality checks or be classified as watch/diagnostic.
- Trade pressure must remain disabled unless a separate QA-accepted side-semantics task enables it.

## Next Runner Boundary

The later implementation task, if created, should be named as a read-only pricing-signal runner, not a strategy task.

Inputs:

- `0601T002` style joined features or a later QA-accepted equivalent.
- `0601T003` feature verdict table or a later QA-accepted stability artifact.
- Hyperliquid public market-view features from `0531T001` style consumer outputs or a later QA-accepted equivalent.

Outputs:

- Candidate pricing-signal rows with decision-time features only.
- Horizon label summaries with nominal horizon and effective future age.
- Feature stability by volatility/liquidity regime.
- Venue-state conditioning summaries.
- A recommendation that can only be `keep_for_read_only_research`, `needs_more_public_samples`, or `reject_for_runner_design`.

Labels:

- Hyperliquid future mid move.
- Hyperliquid future microprice-minus-mid change.
- Hyperliquid future top5 imbalance/book-pressure change.
- Basis mid and basis microprice response as diagnostic labels only.

Forbidden:

- No strategy implementation.
- No live trading process.
- No private keys or private/account/order endpoints.
- No order lifecycle, submit/cancel/fill logic, queue model, or execution simulator.
- No parameter search, default-on behavior, tiny-live, or promotion.
- No standard npz schema, canonical audit schema, connector, or core API change.

## What This Evidence Supports

The evidence supports designing a later read-only pricing-signal runner that tests whether Binance top5-derived lead features can improve Hyperliquid pricing research labels under explicit venue-state quality gates.

The evidence does not support a trading strategy, signal-ready claim, order placement logic, live test, parameter optimization, default-on behavior, tiny-live, or promotion.
