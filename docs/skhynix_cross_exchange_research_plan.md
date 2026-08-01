# SKHYNIX Cross-Exchange Research Plan

Date: 2026-07-30

Task: `0730T012`

## 1. Objective

Use the pulled-back four-hour SKHYNIX public-data campaign to answer five
separate questions in order:

1. Can Binance and Hyperliquid state be aligned without future leakage at the
   horizons required by the strategy?
2. Does the cross-venue basis contain stable predictive information, rather
   than only a persistent contract or venue-level price difference?
3. Does Binance lead Hyperliquid, at which horizons, and through which market
   variables?
4. Is any observed dislocation executable after fees, depth, latency and fill
   uncertainty?
5. Does a Binance-led signal improve a Hyperliquid maker policy relative to
   Hyperliquid-only and basis-only baselines?

The output of this research may qualify a signal for public shadow or further
sample expansion. This dataset alone cannot authorize live orders, promotion,
stable-PnL claims or exact fill claims.

## 2. Input Dataset And Current Facts

Primary package:

- extracted campaign:
  `local_live_analysis/cross_exchange_collection_campaign_0730T011_skhynix_4h_8x30m/`
- archive:
  `local_live_analysis/0729T010_skhynix_4h_8x30m_postprocessed_0730T011.tar.gz`
- archive SHA-256:
  `df35a37aa421e92f2d4267ccb78cb5c31c365eaef54a0af5c78f9bd1910b71c4`
- venues:
  - Binance USD-M `SKHYNIXUSDT`
  - Hyperliquid `xyz:SKHX`
- duration:
  - eight 30-minute segments
  - approximately four hours of market data
- common L2 timeline:
  - `556,861` rows
  - `527,660` total Binance triggers
  - `527,652` Binance depth-update triggers
  - `8` Binance snapshot triggers
  - `26,519` Hyperliquid fast-L2 triggers
  - `2,682` Hyperliquid standard-L2 triggers

Raw event inventory across all segments:

| source | event count | approximate rate |
| --- | ---: | ---: |
| Binance depth | 527,652 | 36.64/s |
| Binance bookTicker | 5,626,969 | 390.76/s |
| Binance trades | 5,117,764 | 355.40/s |
| Hyperliquid fast L2 | 26,528 | 1.84/s |
| Hyperliquid BBO | 142,898 | 9.92/s |
| Hyperliquid trades | 66,915 | 4.65/s |
| Hyperliquid standard L2 | 2,690 | 0.19/s |
| Hyperliquid asset context | 14,101 | 0.98/s |
| Hyperliquid main allMids | 2,856 | 0.20/s |
| Hyperliquid target-dex allMids | 2,857 | 0.20/s |

Observed alignment facts on the current common timeline:

- strict as-of future joins: `0`
- source timestamp regressions: `0`
- Hyperliquid fast-L2 age at Binance-triggered rows:
  - p50: approximately `271.66ms`
  - p90: approximately `499.23ms`
  - p99: approximately `628.33ms`
  - `90.11%` are no older than `500ms`
  - `99.96%` are no older than `1000ms`
- Hyperliquid standard-L2 age at Binance-triggered rows:
  - p50: approximately `2681.34ms`
  - p99: approximately `5358.82ms`
- segment restart gaps:
  - minimum: approximately `1080.12ms`
  - maximum: approximately `1433.69ms`
- raw midpoint basis, defined only as
  `10000 * log(Binance mid / Hyperliquid fast mid)`:
  - median: approximately `13.21bps`
  - p90: approximately `36.84bps`
  - p99: approximately `48.40bps`
  - range: approximately `-33.83bps` to `76.67bps`

The raw basis distribution is a research observation, not an arbitrage
conclusion. The registry correctly labels this pair's basis as diagnostic
only.

## 3. Non-Negotiable Research Boundaries

### 3.1 Clock And Leakage

- Cross-venue decision joins use same-host local receipt timestamps.
- Exchange timestamps remain diagnostic fields.
- Every decision input must satisfy
  `source_local_ts_ns <= decision_ts_ns`.
- Future labels may look forward but must never enter feature construction,
  normalization or decision state.
- Normalization, coefficients, thresholds and bucket boundaries are fitted on
  training segments only.

### 3.2 Segment Boundaries

- Never compute returns, OFI windows, labels, basis changes or queue proxies
  across segment boundaries.
- Treat each segment as a fresh book epoch.
- Exclude each segment's initialization period until all required hot-state
  inputs are available.
- Preserve the observed `1.08-1.43s` restart gaps in every report.

### 3.3 Auxiliary Degraded Intervals

Mask auxiliary features in segment 2 for:

- `asset_context`: `927.889576ms`
- `main_all_mids`: `10029.488302ms`

Core Binance and Hyperliquid L2/BBO/trade research remains usable in these
intervals. Any model using funding, oracle, mark, premium, main allMids or
target-dex allMids must either exclude the corresponding interval or emit an
explicit missing/degraded flag. Unlimited forward-fill is prohibited.

### 3.4 Execution Claims

Public L2 does not provide exact queue position. Therefore:

- no exact fill probability claim;
- no exact order-level replay claim;
- no exact maker PnL claim;
- fill results must be reported as scenario bounds;
- contract economic equivalence must pass before any result is called
  executable arbitrage.

## 4. Research Data Model

The common L2 timeline is necessary but not sufficient for this study because
it excludes the highest-rate Binance bookTicker/trade streams and the
Hyperliquid BBO/trade streams.

Build one research event store per segment from the original raw files:

1. Replayed Binance depth state.
2. Binance bookTicker and signed trade flow.
3. Hyperliquid BBO.
4. Hyperliquid trades.
5. Hyperliquid fast top-5 L2.
6. Hyperliquid standard top-20 L2.
7. Masked asset context and allMids auxiliary state.

Required views:

### 4.1 Decision-Event View

Use Binance information arrivals as candidate decision events:

- best bid/ask change;
- midpoint or microprice change;
- top-level OFI change;
- signed trade-flow burst;
- depth-imbalance change.

As-of join the latest Hyperliquid BBO, fast L2, standard L2 and auxiliary
state. Deduplicate events that produce an identical feature state.

### 4.2 Fixed-Grid View

Construct independent `10ms`, `25ms`, `50ms`, `100ms`, `250ms` and `500ms`
grids for return cross-correlation and robustness checks.

The grid must carry source age and update flags. Repeated forward-filled state
must not be counted as a new information arrival or given independent-event
weight.

### 4.3 Response-Event View

For each accepted Binance shock, locate:

- first Hyperliquid BBO update;
- first Hyperliquid fast-L2 update;
- first Hyperliquid trade;
- first Hyperliquid midpoint move in the predicted direction.

This view measures response latency and response magnitude without pretending
that the approximately `539ms` fast-L2 cadence is a strategy tick-to-order
latency.

## 5. Goal 1: Alignment Acceptance

This goal must pass before signal or arbitrage conclusions are produced.

### 5.1 Build And Reconciliation

- Reconstruct Binance L2 from the accepted snapshot bridge and continuous
  depth stream.
- Reconcile the reconstructed top of book against Binance bookTicker.
- Reconcile Hyperliquid fast-L2 top of book against Hyperliquid BBO.
- Compare fast top-5 with the first five standard-L2 levels when both update
  near the same local receipt time.
- Emit mismatch counts, durations and price-distance distributions.

### 5.2 Source-Age Tiers

Collection quality limits are not signal freshness limits. Use:

| track | primary | watch | reject for decision use |
| --- | --- | --- | --- |
| Hyperliquid BBO | `<=250ms` | `250-500ms` | `>500ms` |
| Hyperliquid fast L2 | `<=500ms` | `500-1000ms` | `>1000ms` |
| Hyperliquid standard L2 | `<=3000ms` | `3000-6000ms` | `>6000ms` |
| auxiliary snapshots | channel-specific | explicit degraded flag | degraded interval or failed freshness |

Standard L2 is a depth and liquidity-regime feature. It is not a primary
millisecond decision trigger.

### 5.3 Label Alignment

Evaluate horizons:

`10, 25, 50, 100, 250, 500, 1000, 2000ms`.

For each label, persist:

- nominal horizon;
- effective horizon;
- label source event timestamp;
- source age at decision;
- whether a price update occurred inside the horizon;
- whether the label is unchanged because no new Hyperliquid information
  arrived.

Use two explicitly named label modes:

- primary response label:
  the first Hyperliquid BBO update at or after `decision_ts + horizon`;
- diagnostic wall-clock label:
  the latest Hyperliquid BBO state at or before `decision_ts + horizon`, with
  the BBO age and no-new-information flag retained.

Freeze primary response-label tolerances before model fitting:

| nominal horizon | accepted effective horizon |
| --- | --- |
| `10, 25, 50ms` | `[h, h + 50ms]` |
| `100, 250, 500ms` | `[h, h + 100ms]` |
| `1000, 2000ms` | `[h, h + 250ms]` |

Primary signal acceptance should focus on horizons whose effective-label
coverage passes. The expected initial primary set is `100, 250, 500, 1000ms`;
the `10-50ms` set is diagnostic unless BBO coverage proves otherwise.

### 5.4 Alignment Exit Gate

Pass only when:

- future decision joins are `0`;
- source timestamp regressions are `0`;
- no feature or label crosses a segment boundary;
- raw event counts and hashes reconcile;
- top-of-book reconciliation exceptions are quantified and explainable;
- every accepted horizon has at least `95%` effective-label coverage inside
  its frozen tolerance;
- freshness coverage is reported by segment and regime;
- auxiliary degraded intervals are masked exactly.

Deliverables:

- `alignment_manifest.json`
- `alignment_quality_by_segment.csv`
- `source_age_distribution.csv`
- `top_of_book_reconciliation.csv`
- `effective_horizon_coverage.csv`
- `alignment_acceptance.md`

## 6. Goal 2: Basis Signal Effectiveness

The purpose is to distinguish a stable predictive residual from a persistent
cross-contract price level difference.

### 6.1 Basis Definitions

Calculate all of the following:

- midpoint basis:
  `log(binance_mid / hl_mid)`;
- microprice basis;
- executable sell-Binance/buy-Hyperliquid edge:
  `binance_bid - hl_ask`;
- executable sell-Hyperliquid/buy-Binance edge:
  `hl_bid - binance_ask`;
- rolling basis residual and z-score;
- basis innovation over
  `10, 25, 50, 100, 250, 500, 1000ms`;
- basis relative to Hyperliquid mark, oracle and premium state;
- basis conditioned on both venues' spread, depth, volatility and source age.

Keep the raw basis level, basis change and rolling residual as separate
features.

### 6.2 Predictive Decomposition

For each horizon, decompose basis closure into:

- future Hyperliquid move;
- future Binance reversal;
- simultaneous movement;
- no convergence.

Fit simple, pre-registered models:

```text
future_hl_return ~ basis_residual
future_hl_return ~ binance_return + basis_residual
future_basis_change ~ basis_residual
```

Do not reuse the accepted BTC basis coefficient. SKHYNIX receives new
training-only normalization and coefficients.

### 6.3 Basis Metrics

- Pearson and Spearman IC;
- signed hit rate;
- top-minus-bottom quantile future return;
- monotonicity across basis deciles;
- coefficient and sign stability by segment;
- incremental out-of-sample R-squared over Binance-lead-only;
- half-life and stationarity diagnostics;
- performance by source-age, spread, volatility and liquidity regime;
- block-bootstrap confidence intervals with multiple-testing control.

### 6.4 Basis Exit Gate

Classify the result as one of:

- `basis_predictive_candidate`;
- `basis_context_only`;
- `basis_not_supported`;
- `needs_more_samples`.

`basis_predictive_candidate` requires:

- the same directional effect in at least four of five anchored
  walk-forward test segments;
- a block-bootstrap `95%` interval excluding zero at two adjacent accepted
  horizons;
- monotonic or near-monotonic decile response;
- positive incremental out-of-sample value over the lead-only model;
- no single test segment contributing more than `50%` of total effect;
- no dependence on stale or degraded-only buckets.

This gate validates a forecasting feature, not executable arbitrage.

## 7. Goal 3: Binance Lead / Hyperliquid Lag Validation

### 7.1 Hypotheses

- H1: Binance returns and order-flow innovations predict future Hyperliquid
  returns.
- H2: the reverse Hyperliquid-to-Binance relationship is weaker at the same
  horizons.
- H3: the effect survives spread, volatility, source-age and liquidity
  conditioning.
- H4: the effect is concentrated in actual Binance information shocks, not
  duplicate rows carrying unchanged Hyperliquid state.

### 7.2 Candidate Binance Features

- top-1/top-5/top-20 imbalance;
- top-1/top-5 microprice-minus-mid;
- OFI over `10-1000ms`;
- signed trade count, quantity and notional;
- aggressive buy/sell imbalance;
- midpoint return and realized volatility;
- spread and depth changes;
- bookTicker acceleration and update intensity.

Hyperliquid local-state controls:

- BBO spread and midpoint;
- fast-L2 imbalance and microprice;
- standard-L2 depth slope and concentration;
- recent Hyperliquid trade flow;
- basis residual;
- source age and update flags;
- mark/oracle/premium and liquidity regime when not degraded.

### 7.3 Tests

1. Fixed-grid cross-correlation of returns, not price levels.
2. Binance-shock event studies with first-Hyperliquid-response timing.
3. Distributed-lag regressions:

```text
hl_return(t, t+h) ~ binance_innovation(t-w, t) + hl_state(t)
binance_return(t, t+h) ~ hl_innovation(t-w, t) + binance_state(t)
```

4. Incremental model comparison against Hyperliquid-only state.
5. Granger/VAR diagnostics on stationary fixed-grid series as secondary
   evidence only.

### 7.4 Lead-Lag Exit Gate

Classify as:

- `binance_lead_supported`;
- `bidirectional_or_regime_dependent`;
- `no_stable_lead`;
- `needs_more_samples`.

`binance_lead_supported` requires:

- Binance-to-Hyperliquid effect exceeds the reverse effect at two adjacent
  accepted horizons;
- sign consistency in at least four of five walk-forward test segments;
- positive incremental out-of-sample IC or R-squared;
- response-event timing agrees with the regression direction;
- effect remains after excluding stale fast-L2/BBO rows;
- result is not driven only by segment 1 or one high-volatility interval.

## 8. Goal 4: Arbitrage And Executable Edge

This goal has two different studies and must report them separately.

### 8.1 Economic-Equivalence Gate

Before using the word arbitrage, verify and freeze:

- underlying/index definition;
- contract multiplier and price scale;
- settlement and collateral currency;
- mark/oracle construction;
- funding calculation and payment timing;
- trading-hours or market-state differences;
- tick/lot/minimum-notional rules;
- maker/taker fees and rebates;
- transfer, capital and inventory constraints.

If this gate does not pass, all outputs remain
`cross_venue_dislocation` or `statistical_basis_trade`.

### 8.2 Taker-Taker Dislocation Study

At the same local decision timestamp, walk both reconstructed L2 books for
candidate sizes and calculate:

- executable bid/ask edge;
- taker fees;
- depth slippage;
- hedge latency;
- funding carry;
- stale-state penalty;
- residual inventory if one leg fails.

Stress latency at:

`1, 2, 5, 10, 25, 50, 100, 250, 500ms`.

Use p50 and p99 results separately. Report opportunity duration and independent
opportunity count after a cooldown, not raw row count.

### 8.3 Hyperliquid-Maker / Binance-Taker Study

Model:

```text
Hyperliquid post-only quote
  -> scenario-bounded Hyperliquid fill
  -> Binance taker hedge after fill
  -> fee, slippage, latency and inventory PnL
```

Because queue position is unknown, report:

- optimistic bound: touch and trade interaction;
- base proxy: displayed queue ahead plus observed trade depletion;
- conservative bound: full visible queue depletion or trade-through before
  fill.

No bound may be described as exact fill simulation.

### 8.4 Arbitrage Exit Gate

An executable candidate requires:

- economic-equivalence gate passed;
- positive net edge after frozen fees, depth and funding;
- positive lower-bound result under p99 latency stress;
- enough independent opportunities for block-bootstrap confidence;
- no profitability caused by future joins or stale snapshots;
- bounded residual inventory and one-leg failure loss.

Otherwise classify as:

- `statistical_basis_only`;
- `gross_dislocation_not_executable`;
- `economic_equivalence_unresolved`;
- `needs_more_samples`.

## 9. Goal 5: Hyperliquid Maker Signal Validation

### 9.1 Prediction Target

Predict the future Hyperliquid executable fair value:

```text
future_hl_mid_move(h)
future_hl_microprice_move(h)
future_hl_markout_after_quote(h)
```

Primary horizons:

`100, 250, 500, 1000ms`.

Diagnostic horizons:

`10, 25, 50, 2000ms`.

The local strategy tick-to-order target of `1-2ms` is evaluated as an
execution delay. It does not imply that the public market-data prediction
horizon must also be `1-2ms`.

### 9.2 Frozen Baselines

Compare four policies:

1. `hl_only`
   - Hyperliquid spread, microprice, imbalance, trade flow.
2. `basis_only`
   - rolling basis residual with Hyperliquid freshness controls.
3. `binance_lead_only`
   - Binance return, OFI, microprice and trade-flow innovations.
4. `combined`
   - Binance lead, basis residual and Hyperliquid local state.

Do not compare only against a zero-signal policy.

### 9.3 Fair-Value And Quote Policy

Candidate structure:

```text
forecast_hl_move =
    beta_lead * binance_lead_features
  + beta_basis * basis_residual
  + beta_hl * hyperliquid_local_state

fair_hl_px = hl_mid_px + forecast_hl_move
reservation_px = fair_hl_px - inventory_penalty
```

Quote simulations:

- join best bid/ask;
- one tick behind touch;
- signal-gated one-sided quote;
- two-sided quote with signal-dependent skew;
- post-only price validation;
- cancel/requote on fair-value drift or stale input.

Inventory-aware reservation price is evaluated only after the directional
signal has out-of-sample value.

### 9.4 Maker Evaluation

Prediction metrics:

- out-of-sample Pearson/Spearman IC;
- direction hit rate;
- decile spread and monotonicity;
- calibration error;
- improvement over each frozen baseline.

Execution-proxy metrics:

- would-submit count and rate;
- quote survival;
- optimistic/base/conservative fill range;
- gross spread capture;
- fee-adjusted edge;
- adverse selection at `50, 100, 250, 500, 1000ms`;
- Binance hedge slippage;
- fill-conditioned and all-decision PnL;
- inventory distribution and maximum exposure;
- cancel/requote rate;
- drawdown and segment contribution.

Latency stress:

- local tick-to-order: `1, 2, 5, 10ms`;
- full signal-to-hedge path: `10, 25, 50, 100, 250, 500ms`;
- freshness buckets for Hyperliquid BBO and fast L2.

### 9.5 Maker Exit Gate

The maximum conclusion from this dataset is
`candidate_for_multi_window_public_shadow`.

Require:

- combined model beats `hl_only`, `basis_only` and `binance_lead_only` out of
  sample;
- directional and markout improvement has the same sign in at least four of
  five walk-forward test segments;
- conservative fill proxy produces positive net edge after frozen costs;
- adverse selection is lower than the Hyperliquid-only baseline;
- result survives `1-2ms` local latency and at least one slower full-path
  latency stress;
- no segment contributes more than `50%` of total PnL;
- inventory and drawdown stay inside pre-registered bounds;
- stale and degraded rows are not required for profitability.

Failure classifications:

- `signal_predictive_but_not_maker_executable`;
- `maker_proxy_positive_needs_more_samples`;
- `no_incremental_binance_value`;
- `reject_current_signal_shape`.

## 10. Train, Validation And Statistical Discipline

Use anchored walk-forward testing:

| fold | train | test |
| --- | --- | --- |
| 1 | segments 1-3 | segment 4 |
| 2 | segments 1-4 | segment 5 |
| 3 | segments 1-5 | segment 6 |
| 4 | segments 1-6 | segment 7 |
| 5 | segments 1-7 | segment 8 |

Additional diagnostics may use leave-one-segment-out, but the primary result
must be the time-ordered walk-forward result.

Controls:

- freeze candidate features and horizons before reading test results;
- fit all scalers and thresholds on the training side only;
- use time-block bootstrap or HAC errors;
- control false discovery across features and horizons;
- report zero-return/no-new-information labels separately;
- report both event-weighted and time-weighted metrics;
- preserve per-segment results, not only pooled aggregates.

Four hours from one trading session is enough for method validation and
candidate rejection. It is not enough for production acceptance. A surviving
candidate must later be tested across multiple dates, market sessions and
symbols such as BTC, ETH and MU.

## 11. Ordered Execution

1. `R0 Research dataset builder`
   - enrich the common L2 timeline with raw BBO/trade/bookTicker events;
   - persist hashes, masks, source ages and segment boundaries.
2. `R1 Alignment acceptance`
   - finish reconciliation, freshness and effective-horizon gates.
3. `R2 Basis effectiveness`
   - determine predictive, context-only or unsupported status.
4. `R3 Lead-lag validation`
   - establish direction, horizon, magnitude and reverse-causality controls.
5. `R4 Executable edge`
   - run economic-equivalence, taker-taker and maker-taker studies.
6. `R5 Hyperliquid maker signal`
   - compare frozen baselines and run latency/fill scenario stress.
7. `R6 Robustness decision`
   - decide reject, recommend more data, or move to public shadow.

Do not begin R4 or R5 acceptance before R1 passes. R2 and R3 may share the
same accepted research table but must retain separate conclusions.

R6 may only recommend a new collection. It must not start one. Any new AWS or
public-data collection requires:

- a separate formal task;
- the user's explicit authorization before launch;
- a user-confirmed active trading window;
- frozen symbols, venue mappings, host, duration, expected cost and output
  path;
- a preflight proving that no private/account/order endpoint is involved.

Without that authorization, execution stops at the collection recommendation
and continues only with already available local data.

## 12. Final Deliverables

Recommended output root:

`local_live_analysis/skhynix_cross_exchange_research_0730T012/`

Required artifacts:

- `research_input_manifest.json`
- `segment_and_mask_index.csv`
- `alignment/`
- `basis/`
- `lead_lag/`
- `arbitrage/`
- `maker_signal/`
- `walk_forward_split_manifest.json`
- `cost_and_latency_assumptions.json`
- `research_boundary_manifest.json`
- `research_recommendation.md`

The final recommendation must state one of:

- `reject_current_signal_shape`;
- `needs_more_samples`;
- `candidate_for_multi_window_public_shadow`.

It must separately report:

- alignment status;
- basis status;
- lead-lag status;
- economic-equivalence status;
- executable-edge status;
- Hyperliquid maker-signal status.
