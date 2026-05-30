# Hyperliquid Public Market-Data Research Consumer Design

Task: `0530T001`

Status: design-only / read-only. This document does not authorize a private connector, order lifecycle, strategy live logic, parameter search, default-on behavior, tiny-live, promotion, connector/core API changes, standard npz schema changes, or canonical audit schema changes.

Conclusion: official docs rechecked successfully.

## Purpose

`0529T004` proved that one fresh Hyperliquid public-only BTC sample can be collected, provenanced, aligned, and classified as `passes_pricing_research_market_view`. The next useful boundary is a read-only consumer contract that turns those accepted public artifacts into deterministic pricing and market-view research tables.

This consumer is a research data product. It should answer whether public Hyperliquid book/trade views are good enough to study pricing signals. It must not answer whether a trading strategy can place, cancel, fill, or promote orders.

## References Rechecked

Official Hyperliquid docs were reachable during this task and were rechecked before relying on current public schema facts:

- `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api`
- `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket`
- `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions`
- `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint`

Local accepted facts:

- `.workflow/reports/0529T004-qa.md`
- `local_live_analysis/hyperliquid_public_sample_0529T004/collection_manifest.json`
- `local_live_analysis/hyperliquid_public_sample_0529T004/alignment/metrics.json`
- `local_live_analysis/hyperliquid_public_sample_0529T004/alignment/topn_sidecar.csv`
- `local_live_analysis/hyperliquid_public_sample_0529T004/alignment/synthetic_joined_views.csv`

Key official public-schema facts used:

- WebSocket mainnet endpoint is `wss://api.hyperliquid.xyz/ws`; testnet endpoint is `wss://api.hyperliquid-testnet.xyz/ws`.
- Public subscription messages use `{"method": "subscribe", "subscription": {...}}`.
- `l2Book` subscription uses `{"type": "l2Book", "coin": "<coin_symbol>"}` and emits `WsBook`.
- `trades` subscription uses `{"type": "trades", "coin": "<coin_symbol>"}` and emits `WsTrade[]`.
- Successful subscriptions emit `subscriptionResponse`; stream messages then use the subscribed channel.
- `WsBook` has `coin`, `levels`, and `time`; each level has `px`, `sz`, and `n`.
- Public Info `l2Book` is available through `POST https://api.hyperliquid.xyz/info` with body type `l2Book` and returns at most 20 levels per side.
- Automated WebSocket users should handle server-side disconnect and reconnect; missed data can be recovered through snapshot ack or corresponding info requests.

## Accepted T004 Entry Point

The first implementation should default to:

- input directory: `local_live_analysis/hyperliquid_public_sample_0529T004/`
- coin: `BTC`
- network: `mainnet`
- source classification: `passes_pricing_research_market_view`

Required inputs:

- `raw.gz`
- `raw.sha256`
- `collection_manifest.json`
- `recovery_snapshots.jsonl`
- `alignment/run_manifest.json`
- `alignment/converter_manifest.json`
- `alignment/data.npz`
- `alignment/raw_provenance.csv`
- `alignment/raw_to_npz_mapping.csv`
- `alignment/topn_sidecar.csv`
- `alignment/synthetic_joined_views.csv`
- `alignment/metrics.json`
- `alignment/acceptance_report.md`

T004 quality facts that the implementation should preserve in its run manifest:

- raw sha256: `137018ef937b3692a5de0c12ee009c4a93a0e6d62ff15321061c377fc514389c`
- session id: `hl-66959ecb09e941cab61b020ac1e06419`
- channels: `l2Book`, `trades`
- message counts: `l2Book=222`, `trades=111`, `subscriptionResponse=2`
- subscription ack by channel: `l2Book=1`, `trades=1`
- reconnect count: `0`
- recovery snapshot count: `1`
- raw parse errors: `0`
- `data.npz` rows: `4279`
- trade events: `418`
- event-order validation: `passed`
- top-N coverage: `1.0`
- synthetic join coverage: `1.0`
- future joins: `0`
- missing joins: `0`
- join-age p99: `1114.135914ms`
- l2Book cadence p99: `1317.185171ms`

## Consumer Contract

Proposed later CLI:

```text
python examples/hyperliquid/hyperliquid_market_data_research.py \
  --input-dir local_live_analysis/hyperliquid_public_sample_0529T004 \
  --output-dir local_live_analysis/hyperliquid_market_data_research_0530T002 \
  --coin BTC \
  --top-n 5
```

The exact future task ID may change, but the implementation boundary should remain read-only and public-only.

Required outputs:

- `run_manifest.json`
- `market_view_timeseries.csv`
- `pricing_features.csv`
- `feature_quality_summary.json`
- `sample_session_quality_summary.json`
- `research_recommendation.md`

Optional diagnostic output:

- `diagnostic_labels.csv`, only for after-the-fact research labels and never as decision inputs.

### `run_manifest.json`

Required fields:

- schema version, task id, generated timestamp, git commit
- input directory and all input artifact paths
- raw sha256 from file and manifest, plus match boolean
- official doc recheck status, references, and timestamp
- coin, network, channels, top-N, synthetic interval
- source sample classification from T004
- consumer output paths
- quality gate verdicts
- boundary flags:
  - `no_private_keys=true`
  - `no_private_account_endpoints=true`
  - `no_order_endpoints=true`
  - `no_order_lifecycle=true`
  - `no_strategy_live_process=true`
  - `no_parameter_search=true`
  - `no_default_on=true`
  - `no_tiny_live=true`
  - `no_promotion=true`

### `market_view_timeseries.csv`

One row per synthetic joined market-view timestamp, derived from `synthetic_joined_views.csv` and the as-of `topn_sidecar.csv`.

Required fields:

- `view_seq`
- `view_ts`
- `coin`
- `session_id`
- `connection_attempt`
- `joined_raw_seq`
- `joined_l2book_local_ts`
- `joined_l2book_event_ts`
- `join_age_ms`
- `future_join`
- `missing_join`
- `reconnect_recovery_crossed`
- `best_bid_px`
- `best_ask_px`
- `mid_px`
- `spread_px`
- `spread_ticks`
- `bid_topn_px`
- `ask_topn_px`
- `bid_topn_qty`
- `ask_topn_qty`
- `bid_topn_order_count`
- `ask_topn_order_count`
- `book_freshness_bucket`
- `market_view_quality`

### `pricing_features.csv`

One row per market-view row, with only public, decision-time-visible features:

- `mid_px`
- `spread_ticks`
- `top1_imbalance`
- `top3_imbalance`
- `top5_imbalance`
- `top1_microprice_px`
- `top3_microprice_px`
- `top5_microprice_px`
- `microprice_minus_mid_ticks`
- `bid_depth_top5_qty`
- `ask_depth_top5_qty`
- `bid_depth_top5_notional`
- `ask_depth_top5_notional`
- `book_pressure_bucket`
- `spread_bucket`
- `join_age_bucket`
- `l2book_cadence_bucket`
- `recovery_context`
- `trade_pressure_qty`
- `trade_pressure_count`
- `trade_pressure_bucket`
- `feature_row_quality`

Trade pressure must be derived only from public `trades` messages whose local/exchange timestamps are already available before the feature timestamp. If the side semantics are not freshly confirmed or are ambiguous, the implementation must emit `trade_pressure_status=unverified_side_semantics` and exclude trade pressure from candidate-ready features.

### `feature_quality_summary.json`

Required groups:

- input artifact presence
- raw sha256 match
- manifest boundary flags
- official docs recheck status
- top-N coverage
- synthetic join coverage
- future/missing joins
- join-age distribution
- l2Book and trade cadence distributions
- recovery snapshot count
- feature null counts
- feature outlier counts
- final classification and reason

### `sample_session_quality_summary.json`

Required fields:

- session id
- connection attempts
- reconnect count
- close reason
- subscription ack counts
- first/last local timestamps by channel
- message counts by channel
- recovery snapshot count and statuses
- official public references used
- explicit private/order endpoint absence flags

### `research_recommendation.md`

Required sections:

- input sample and classification
- public feature coverage
- quality-gate result
- whether the sample is acceptable for pricing / market-view research
- whether more public samples are needed before any pricing signal conclusion
- next recommended task
- explicit non-authorization statement for private/order/live/promotion work

## Feature Rules

Allowed decision-time-visible features:

- BBO, mid, spread
- top-N depth quantity and notional
- top-N order-count context from level `n`
- top-N imbalance:
  - `(bid_qty - ask_qty) / (bid_qty + ask_qty)` when denominator is positive
- top-N microprice proxy:
  - `(best_ask_px * bid_qty + best_bid_px * ask_qty) / (bid_qty + ask_qty)` when denominator is positive
- microprice-minus-mid in ticks
- book freshness and join-age buckets
- public trade pressure based only on already observed trades
- session/reconnect/recovery context as quality context

Diagnostic-only labels:

- future mid return
- future markout
- realized later spread movement
- sample-local price move after the feature timestamp
- any label derived from future rows

Diagnostic labels may be useful for offline research scoring, but they must live in a separate output and must never be described as live decision inputs.

## Quality Gates

`passes_pricing_research_market_view` requires:

- all required input artifacts exist
- raw sha256 matches `raw.sha256`, `collection_manifest.json`, and `run_manifest.json`
- `no_private_keys`, `no_private_account_endpoints`, `no_order_endpoints`, and `no_strategy_process` are true in the source manifest
- `l2Book` and `trades` message counts are positive
- subscription ack exists for `l2Book` and `trades`
- recovery snapshot count is at least `1`
- raw parse errors are `0`
- event-order validation is `passed`
- top-N coverage is at least `0.99`
- synthetic join coverage is at least `0.99`
- future joins are `0`
- missing joins are `0`
- join-age p99 is not worse than `max(2000ms, 2 * l2Book_cadence_p99_ms)`
- feature null rates are explained and do not affect core BBO/top-N features

`limited_pricing_research` applies when:

- raw parsing and top-N reconstruction are usable
- core BBO/top-N features are present
- one supporting evidence layer is incomplete, such as recovery snapshot evidence, subscription ack evidence, or elevated but explainable join age
- no private/order endpoint contamination exists

`market_view_diagnostic_only` applies when:

- public data can be parsed
- but join coverage, freshness, top-N coverage, or feature null rates prevent pricing research claims

`unusable` applies when:

- required artifacts are missing
- raw parsing fails materially
- `l2Book` is absent
- public/private boundary flags are missing or contradicted
- future joins or missing joins indicate the market view cannot be trusted

## Later Implementation Boundary

The next task should implement only the read-only consumer described above. It may:

- read accepted `0529T004` public artifacts
- generate deterministic CSV/JSON/Markdown research artifacts
- run focused tests around feature computation and quality gates
- classify the sample for public pricing / market-view research

It must not:

- collect a new sample
- connect to private endpoints
- use API keys
- subscribe to user streams
- submit, cancel, amend, or query orders
- model fills or order lifecycle
- run strategy live logic
- change strategy behavior
- run parameter search
- default-enable anything
- start tiny-live
- make promotion claims

## Evidence Required Before Non-Read-Only Work

`0529T004` plus `0530T001` is not enough to plan private connector, order lifecycle, strategy live, parameter search, default-on, tiny-live, or promotion.

Before any future non-read-only Hyperliquid task can even be planned, the controller would need separate accepted evidence for:

- multiple fresh public samples across regimes
- stable public market-view quality and feature coverage across those samples
- a read-only pricing study showing positive signal quality out of sample
- a private connector design that separately covers auth, signing, nonce handling, account safety, private WebSocket/Info schemas, and rate limits
- an order lifecycle evidence plan for submit/ack/reject/cancel/fill/terminal states
- replay/live execution acceptance gates comparable in rigor to the Binance workflow
- risk controls, deployment gates, and QA acceptance for any live micro test

The immediate next task is therefore: implement the Hyperliquid public market-data research consumer as a read-only local artifact generator over the accepted `0529T004` sample.
