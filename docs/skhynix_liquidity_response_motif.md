# Hyperliquid Liquidity-Response Motif Family

## Scope

This task builds a reproducible episode table for one narrowly defined motif
family:

```text
Binance aggressive-trade shock
-> Hyperliquid BBO / fast-L2 withdrawal, replenishment, or follow
-> Hyperliquid price response and adverse markout
```

The output is a public aggregate-market-data study. It does not identify a
specific maker, reconstruct queue ownership, simulate exact fills, or establish
maker PnL.

## Accepted Inputs

- Accepted R0 event store:
  `local_live_analysis/skhynix_cross_exchange_research_0730T013/`
- Accepted R1 alignment:
  `local_live_analysis/skhynix_cross_exchange_research_0730T013/alignment/`
- Join clock: same-host local receipt timestamp.
- Segment epochs are independent. No pre-state or response may cross a segment
  boundary.
- R1 primary horizons: `1000ms`, `2000ms`.
- Diagnostic-only horizons in this task: `100ms`, `250ms`, `500ms`.

The builder fails closed unless both source manifests pass and their
provenance relationship is intact.

The v2 builder additionally requires exact R1 horizon tolerances and accepted
gate state, exact campaign/segment/profile/symbol identity, and full
construction-time stability for all eight segment manifests.

## Trigger Contract

Trades are grouped into a burst when they:

- have the same aggressor side;
- are consecutive in the normalized Binance trade stream; and
- occur no later than `10ms` after the first trade in the burst.

Normalized Binance rows with both `trade_px=0` and `trade_qty=0` are counted
as zero-economic trade records, excluded from quantity, and treated as burst
boundaries. Any other non-positive price or quantity fails closed.

For each burst, the pre-state is the latest Binance L2 state strictly before
the first trade. The impacted queue is:

- aggressive buy: pre-shock Binance best ask;
- aggressive sell: pre-shock Binance best bid.

Only trade quantity at or through that pre-shock best price counts as touch
quantity. `shock_ts` is the timestamp of the first trade that makes cumulative
touch quantity reach at least `30%` of the pre-shock impacted queue.

Within `100ms` after `shock_ts`, the Binance L2 state must show either:

- the original impacted best price is depleted; or
- the same best price remains and its quantity is at most `70%` of the
  pre-shock quantity.

The first such L2 state is `decision_ts`.

## Attribution And Eligibility

Confirmed removed quantity is the full pre-queue when its price level is
depleted, otherwise the observed reduction at the unchanged best price.
Attribution uses only touch trades whose receipt timestamps are no later than
`decision_ts`; later trades from the same `10ms` burst are counted separately
and cannot leak into attribution.

```text
trade_explained_ratio =
    min(cumulative_touch_trade_qty, confirmed_removed_qty)
    / confirmed_removed_qty
```

Attribution classes:

- `trade_driven`: ratio `>= 0.70`;
- `mixed`: ratio `>= 0.30` and `< 0.70`;
- `cancel_driven`: ratio `< 0.30`;
- `uncertain`: missing or invalid queue evidence.

All candidates remain in `trigger_audit.csv.gz`. A primary episode additionally
requires:

- `trade_driven` attribution;
- prior Hyperliquid BBO and fast-L2 state inside the same segment;
- pre-state and decision outside excluded boundaries;
- unique `(aggressor side, decision_ts, pre-shock best price)` confirmation;
- no same-direction accepted shock within the preceding `50ms`;
- enough same-segment room for a `2000ms` response target.

Opposite-direction primary shocks inside the following `2000ms` are retained
but marked as response contamination. Every horizon also records the count of
subsequent same-direction and opposite-direction confirmed shock candidates,
plus an `isolated` flag. Isolation is therefore available for downstream
motif analysis rather than assumed.

## Response Features

The event is direction-normalized so positive markout means movement in the
Binance shock direction and therefore adverse movement for a hypothetical
Hyperliquid passive quote on the impacted side.

For Hyperliquid BBO and fast L2 the episode records:

- pre-shock spread, midpoint, impacted/opposite queue, top-5 depth and
  imbalance;
- first BBO response latency;
- first impacted-side withdrawal;
- first impacted-side replenishment after withdrawal;
- first one-tick directional midpoint follow;
- endpoint spread, impacted/opposite queue ratios, fast-L2 age and top-5
  imbalance;
- direction-normalized midpoint markout in inferred Hyperliquid ticks.

`100/250/500ms` fields are diagnostic wall-clock/as-of observations.
`1000/2000ms` fields use R1 primary-response semantics: the first Hyperliquid
BBO at or after the target, within the frozen R1 tolerance.

## M1 Output

```text
local_live_analysis/skhynix_liquidity_response_0730T017/
  motif_episode_manifest.json
  trigger_audit.csv.gz
  segment_summary.csv
  episodes/
    segment_0001.csv.gz
    ...
    segment_0008.csv.gz
```

M1 stops after deterministic episode construction and QA. Clustering,
walk-forward motif stability, economic filtering and shadow-signal evaluation
require later formal tasks.

Existing output replacement uses an operating-system atomic directory
exchange (`renamex_np(RENAME_SWAP)` on macOS or
`renameat2(RENAME_EXCHANGE)` on Linux). The published output path therefore
remains continuously visible during replacement.
