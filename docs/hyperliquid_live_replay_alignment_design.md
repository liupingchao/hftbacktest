# Hyperliquid Live/Replay Alignment Design

Task: `0528T003`

Status: design-only. This document does not authorize trading logic changes, Hyperliquid connector implementation, live collection, remote deploy, parameter search, default-on behavior, tiny-live, or promotion.

## Summary

The first Hyperliquid track should mirror the proven Binance alignment discipline, but it should not copy Binance exchange assumptions. The right first boundary is a read-only market-data alignment path:

1. collect raw Hyperliquid `l2Book` and `trades` WebSocket messages with local receive timestamps and session metadata;
2. convert the raw file to hftbacktest `npz` using the existing Hyperliquid converter contract;
3. build a Hyperliquid top-N provenance / decision as-of join sidecar;
4. run market-view acceptance on Hyperliquid-specific metrics;
5. only after that, consider private lifecycle and trading connector design in separate tasks.

Official Hyperliquid GitBook API docs and the official `hyperliquid-python-sdk` are the schema and endpoint facts. The local XEMM project is useful as engineering reference only.

## Reference Priority

Primary references:

- Hyperliquid API overview: `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api`
- WebSocket endpoint and reconnect guidance: `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket`
- WebSocket subscriptions and payload types: `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions`
- Info endpoint, including `l2Book`, `openOrders`, `userFillsByTime`, and `orderStatus`: `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint`
- Tick and lot size rules: `https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size`
- Official Python SDK: `https://github.com/hyperliquid-dex/hyperliquid-python-sdk`

Local references:

- Current Binance workflow under `examples/binance_tick_mm/`
- Hyperliquid converter at `py-hftbacktest/hftbacktest/data/utils/hyperliquid.py`
- Collector Hyperliquid module under `collector/src/hyperliquid/`
- XEMM reference repo at `/home/molly/project/XEMM_CROSS_EXCHANGE_MARKET_MAKING_PACIFICA_HYPERLIQUID/`

Use XEMM for practical hints around reconnect loops, SDK wrapping, and order surface inventory. Do not use it as schema truth, replay evidence, or acceptance definition.

## Current Binance Pipeline Decomposition

The current Binance live/replay alignment path is centered on `examples/binance_tick_mm/align_live_run.py`.

Exchange-neutral pieces:

- Run artifact layout:
  - `local_live_analysis/<run_id>/`
  - `raw_market_data/`
  - `out/live_raw/<symbol>/manifest_*.json`
  - `out/backtest_normal/`
  - `out/backtest_audit_replay/`
  - `live_alignment_summary.md`
  - archive and sha256 files
- Live window extraction from `audit_live*.csv` via `ts_local`.
- Order latency model generation from live audit rows.
- Raw-to-npz manifest contract:
  - symbol
  - start/end day
  - source
  - data files
  - strict timestamp mode
  - optional snapshot metadata
- Normal replay and audit replay orchestration.
- Backtest config generation for normal versus audit replay cadence.
- Audit comparison shape:
  - action match
  - planned action match
  - reject/throttle match
  - lag gate diagnostics
- Archive/checksum/report generation.
- Later diagnostic runner pattern:
  - Stage 5 execution outcome labels
  - Step 5C quote-anchor safety diagnostics
  - Stage 6 replay/live lifecycle calibration
  - Stage 9 candidate replay / bucket analysis

Binance-specific pieces:

- `run_live.sh` launches `collector ... binancefuturesum`, `connector ... binancefutures`, and `live_tick_mm.py`.
- `pipeline_live_raw.py` calls `hftbacktest.data.utils.binancefutures.convert`.
- Binance collector raw gzip is combined-stream JSON with line-local timestamp followed by payload.
- Binance continuity is built around `depthUpdate` fields `U/u/pu`.
- Binance bootstrap uses snapshot `lastUpdateId` and buffered pre-snapshot depth replay.
- `bookTicker` is a Binance-specific fast BBO anchor.
- `binance_top5_provenance.py` is tied to Binance `depthUpdate`, snapshot, and bookTicker fields.
- Binance maker acceptance currently includes Binance top5/bookTicker market-view gates.
- Binance private lifecycle assumptions are REST/listenKey/order-id/status specific.

Do not generalize `pu`, `lastUpdateId`, or `bookTicker` into an exchange-neutral interface. The neutral contract is "raw sequence plus reconstructed market view plus decision as-of join," not the Binance mechanism used to build it.

## Official Hyperliquid API Mapping

First-stage market-data inputs:

- `l2Book`
  - WebSocket subscription: `{ "type": "l2Book", "coin": "<coin_symbol>" }`
  - Message channel: `l2Book`
  - Data shape: `coin`, `levels`, `time`
  - `levels` is `[bids, asks]`; each level has `px`, `sz`, and `n`
  - Official docs describe it as a snapshot feed pushed on blocks, not an incremental Binance-style depth diff
- `trades`
  - WebSocket subscription: `{ "type": "trades", "coin": "<coin_symbol>" }`
  - Message channel: `trades`
  - Data shape: array of trades
  - Each trade includes `coin`, `side`, `px`, `sz`, `hash`, `time`, `tid`, and `users`
  - Official docs note `tid` should be paired with block time and coin for global uniqueness

Optional first-stage support:

- REST/info `l2Book`
  - `POST /info` with body `{ "type": "l2Book", "coin": "<coin>" }`
  - returns at most 20 levels per side
  - supports `nSigFigs` and `mantissa`
  - use this for bootstrap, reconnect recovery, and sidecar validation, not as a replacement for raw WebSocket replay evidence
- `bbo`
  - May be useful as an auxiliary BBO sanity feed
  - It must not replace full `l2Book` top-N provenance in the first alignment implementation

Connection behavior:

- WebSocket mainnet URL: `wss://api.hyperliquid.xyz/ws`
- WebSocket testnet URL: `wss://api.hyperliquid-testnet.xyz/ws`
- Automated users must handle server-side disconnects and reconnect.
- Missed data during reconnect can be recovered via reconnect snapshots or matching info requests.
- Therefore the collector contract needs explicit session ids, reconnect counters, subscribe ack timestamps, snapshot provenance, and gap/recovery markers.

Precision and rounding:

- Prices have up to 5 significant figures.
- Perp prices also have no more than `6 - szDecimals` decimal places.
- Spot prices use `8 - szDecimals`.
- Integer prices are allowed regardless of significant figures.
- Sizes are rounded to the asset `szDecimals`.
- `szDecimals` comes from the meta response.
- These rules are not part of first-stage market-data replay, but they must be captured in the design because any later order lifecycle adapter must not inherit Binance tick/lot assumptions.

Future private lifecycle surfaces, not first-stage implementation:

- WebSocket `orderUpdates`
- WebSocket `userEvents`, whose channel name is `user`
- WebSocket `userFills`, including `isSnapshot` behavior
- Info `openOrders`
- Info `userFills` and `userFillsByTime`
- Info `orderStatus` by oid or cloid
- SDK `Info.subscribe`, `Info.l2_snapshot`, `Info.user_fills_by_time`, `Info.query_order_by_oid`, `Info.query_order_by_cloid`
- SDK `Exchange.order`, `Exchange.market_open`, `Exchange.cancel`, `Exchange.cancel_by_cloid`
- `Alo`, `Ioc`, `Gtc`, `cloid`, signing, nonce, API wallet, and rate-limit behavior

These belong in a later private lifecycle / execution adapter task.

## Existing Hyperliquid Code In This Repo

`py-hftbacktest/hftbacktest/data/utils/hyperliquid.py` already converts raw Hyperliquid stream files with lines shaped as:

```text
<local_timestamp_ns> <raw_json>
```

It handles:

- `trades` messages by emitting hftbacktest trade events with exchange time from trade `time`.
- `l2Book` messages by treating each message as a book snapshot, diffing against prior snapshot via `DiffOrderBookSnapshot`, and emitting depth changes/deletes.
- Event ordering and local timestamp correction through the standard data validation helpers.

Current limitations for this task's future implementation:

- It converts to `npz`, but it does not emit raw provenance, raw-to-npz mapping, top-N sidecar, or decision join artifacts.
- It assumes a fixed local timestamp width and raw-line format.
- It does not capture session/reconnect metadata.
- It needs explicit tick size / lot size inputs rather than deriving them from Hyperliquid meta.
- It does not define market-view acceptance metrics.

`collector/src/hyperliquid/` already has a read-only WebSocket collector:

- It subscribes to configured Hyperliquid subscription types per symbol.
- Current `collector/src/main.rs` includes `trades`, `l2Book`, and `bbo`.
- It writes raw JSON lines through the common rotating gzip writer using local receive time.
- It reconnects with backoff and uses JSON ping.

Current limitations:

- It does not write a manifest with URL, network, subscription list, session id, reconnect count, subscribe ack sequence, or recovery snapshot facts.
- It filters `subscriptionResponse` out before writing raw data.
- It does not persist disconnect/reconnect markers into the raw artifact.
- It does not call REST/info snapshots for bootstrap or validation.
- It is a collector primitive, not the full alignment workflow.

## Exchange-Neutral Boundary Proposal

Create a neutral alignment layer around these contracts:

### 1. Raw Collection Contract

Required fields:

- exchange
- network
- symbol/coin
- channel
- raw sequence
- line number
- local receive timestamp in ns
- exchange event timestamp in ms/ns when available
- raw payload
- session id
- connection attempt id
- subscription ack id/time when available
- reconnect reason
- recovery snapshot pointer when used

The raw gzip format can stay line-oriented, but the manifest must make the above metadata explicit. Hyperliquid should preserve `subscriptionResponse` and reconnect markers in either the raw stream or a sidecar log so the replay evidence can explain gaps.

### 2. Raw-To-NPZ Contract

The neutral pipeline should require:

- one or more `npz` files
- a manifest listing source raw files and conversion parameters
- tick size and lot size source
- converter version
- channel set
- data row counts
- min/max exchange and local timestamps
- validation result from `validate_event_order`

For Hyperliquid, first use the existing converter, but wrap it with a task-specific adapter that records the missing manifest/provenance facts.

### 3. Provenance / Sidecar Contract

Binance sidecar uses `raw_seq -> final npz rows -> reconstructed top5 book -> decision rows`. Hyperliquid should keep the same conceptual chain but replace exchange-specific columns.

Required Hyperliquid sidecar outputs:

- `raw_provenance.csv`
  - `raw_seq`
  - `line_no`
  - `channel`
  - `coin`
  - `local_ts`
  - `event_ts`
  - `message_kind`
  - `session_id`
  - `connection_attempt`
  - `generated_event_count`
  - `final_row_indices`
- `raw_to_npz_mapping.csv`
  - mapping after converter ordering
- `topn_sidecar.csv`
  - raw/event sequence
  - `l2Book` time
  - bid/ask top N prices, ticks, sizes, and order counts `n`
  - snapshot/reconnect provenance
  - stale/gap/recovery flags
- `joined_decisions.csv`
  - decision sequence
  - as-of joined top-N row
  - join age
  - future join flag
  - missing join flag
  - reconnect/recovery-crossed flag
- `metrics.json`
  - top-N coverage
  - join coverage
  - future join count
  - missing join count
  - p50/p90/p99 join age
  - l2Book cadence distribution
  - trade cadence distribution
  - reconnect count
  - recovery snapshot count
  - recovered interval count

Do not include Binance fields such as `U`, `u`, `pu`, `lastUpdateId`, or `bookticker_u`.

### 4. Market-View Acceptance Contract

Hyperliquid acceptance should begin as read-only market-data quality classification:

- pass/fail hard gates:
  - raw parse success
  - non-empty `l2Book` and `trades` streams
  - npz conversion row count > 0
  - event order validation passes
  - top-N sidecar coverage is complete after startup/reconnect exclusions
  - decision join coverage is complete for the accepted window
  - future joins are zero
  - missing joins are zero outside excluded startup/reconnect windows
- quality gates:
  - join age p99 threshold
  - l2Book cadence p99 threshold
  - reconnect count and recovery coverage
  - trade stream availability
  - REST snapshot versus WebSocket top-N sanity at bootstrap/recovery points

The first implementation should classify samples as:

- `unusable`
- `compressed_action_path_only`
- `limited_pricing_research`
- `passes_pricing_research_market_view`

Exact queue position, full L2 equivalence, private fill lifecycle proof, and live PnL remain out of scope.

## Hyperliquid Adapter Inventory

Recommended adapter sequence:

1. `hyperliquid_raw_market_data_adapter`
   - wraps or extends current collector output contract
   - records `l2Book` and `trades`
   - preserves local timestamps, subscription acks, reconnect metadata, and optional REST snapshot evidence
2. `hyperliquid_raw_to_npz_adapter`
   - invokes `py-hftbacktest/hftbacktest/data/utils/hyperliquid.py`
   - records tick/lot/meta source and converter parameters
   - writes a manifest compatible with the existing replay runner
3. `hyperliquid_topn_provenance_adapter`
   - reconstructs top-N state from `l2Book`
   - writes raw provenance, raw-to-npz mapping, top-N sidecar, and decision join artifacts
4. `hyperliquid_market_view_acceptance_adapter`
   - evaluates Hyperliquid-specific sidecar/join metrics
   - excludes Binance update-id/bookTicker gates
5. `exchange_neutral_alignment_orchestrator`
   - later refactor of `align_live_run.py`
   - common phases: artifact fetch/local input, latency model, raw conversion, replay, audit compare, sidecar, acceptance, archive
   - exchange adapters plug into raw conversion and market-view metrics
6. `hyperliquid_private_lifecycle_adapter`
   - later task only
   - private order lifecycle, fills, order status, ALO/post-only, cloid, signing, nonce, rate limits

## XEMM Reference Assessment

Useful ideas from XEMM:

- Rust WebSocket client structure for `l2Book`.
- Reconnect/backoff and ping loops.
- Typed `L2BookData`, `BookLevel`, and `TimeInForce` models.
- SDK-style private surface inventory:
  - meta / asset id
  - L2 snapshot
  - price and size rounding
  - `Alo`, `Ioc`, `Gtc`
  - `cloid`
  - user fills and state
- A generic exchange trait can inform future adapter design.

Unsafe to copy directly:

- BBO-only or cache-only market-data handling.
- Trading and hedging services.
- REST polling as the main replay evidence.
- XEMM order lifecycle assumptions as hftbacktest acceptance facts.
- Any code path that does not preserve raw stream, local timestamp, session/reconnect metadata, and raw-to-decision provenance.

The local hftbacktest Binance workflow remains the alignment model; XEMM is only a practical Hyperliquid interface reference.

## First Implementation Task Recommendation

Recommended next task:

```text
Hyperliquid read-only raw market-data sample / converter / sidecar validation
```

Scope:

- No trading, no private keys, no order submit/cancel.
- Use Hyperliquid read-only market-data only.
- Collect or use one short raw sample containing `l2Book` and `trades`.
- Preserve subscription ack and reconnect/session metadata.
- Generate `npz` through the existing Hyperliquid converter.
- Build a minimal Hyperliquid top-N sidecar and decision-free validation first.
- If no Hyperliquid strategy audit exists yet, join against synthetic decision timestamps or a simple sampling cadence only for market-view age/coverage validation.

Expected outputs:

- raw gzip
- collection manifest
- converter manifest
- `data.npz`
- raw provenance
- raw-to-npz mapping
- top-N sidecar
- market-data-only metrics
- short acceptance report

Explicitly defer:

- exchange-neutral refactor of the whole Binance alignment runner
- Hyperliquid live strategy
- private lifecycle adapter
- order submission and cancel
- maker strategy port
- parameter search
- tiny-live
- promotion

## Risks And Non-Goals

Risks:

- Hyperliquid `l2Book` is a snapshot feed, so Binance incremental-depth continuity gates do not apply.
- Reconnect evidence must be explicit; otherwise missed blocks can look like normal sparse snapshots.
- REST `l2Book` returns limited depth and should not become the only replay source.
- `bbo` cadence and semantics differ from Binance `bookTicker`; using it as a hard post-only anchor requires a later design.
- Private lifecycle will need a separate evidence model because Hyperliquid order ids, cloids, snapshots, and streaming user events differ from Binance.

Non-goals:

- No Binance strategy changes.
- No Hyperliquid connector implementation.
- No private order lifecycle implementation.
- No canonical audit schema change.
- No standard npz schema change.
- No live collection started by this design task.
- No remote deploy.
- No default-on behavior.
- No guard relaxation.
- No parameter search.
- No tiny-live or promotion claim.
