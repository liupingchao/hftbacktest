# Cross-Exchange Collection Supervisor And Common L2 Timeline

Task: `0729T009`

## Objective

Collect multiple Binance/Hyperliquid symbol pairs under one controlled
campaign and reconstruct each pair's L2 state in one same-host local receipt
time framework.

Supported profiles:

| profile | Binance | Hyperliquid |
| --- | --- | --- |
| `btc` | `BTCUSDT` | `BTC` |
| `eth` | `ETHUSDT` | `ETH` |
| `skhynix` | `SKHYNIXUSDT` | `xyz:SKHX` |
| `mu` | `MUUSDT` | `xyz:MU` |

The mappings are public-data mappings only. They do not define order sizing,
tick/lot rules, economic equivalence or execution-PnL equivalence.

## Campaign Layout

```text
campaign/
  campaign_manifest.json
  run_status.json
  heartbeat.json
  supervisor_events.jsonl
  timeline_index.csv
  segments/
    segment_0001/
      segment_manifest.json
      btc/
        sample/
        common_l2_timeline.csv.gz
        common_l2_timeline_manifest.json
      eth/
      skhynix/
      mu/
```

Profiles in one segment run concurrently. Segments run sequentially and fail
closed. All configured segments are collected back-to-back before strict
quality validation and timeline expansion begin. This keeps inter-segment
downtime limited to child exit, manifest seal and connection restart instead
of blocking collection on CSV generation.

## Usage

An eight-hour campaign split into 30-minute independently recoverable
segments:

```bash
python examples/hyperliquid/cross_exchange_collection_supervisor.py \
  --output-dir /data/cross_exchange_campaign_8h \
  --campaign-id cross-exchange-8h-001 \
  --profiles btc,eth,skhynix,mu \
  --total-duration-seconds 28800 \
  --segment-duration-seconds 1800 \
  --python-executable /path/to/venv/bin/python
```

The output directory must be empty. Reusing a campaign path requires the
explicit destructive acknowledgement `--clean-output`; otherwise the
supervisor fails before starting children so stale successful artifacts cannot
be confused with a new failed run.

An already collected campaign can be revalidated and have its timelines
rebuilt without reconnecting to either exchange:

```bash
python examples/hyperliquid/cross_exchange_collection_supervisor.py \
  --output-dir /data/existing_campaign \
  --campaign-id existing-campaign-id \
  --profiles skhynix \
  --total-duration-seconds 14400 \
  --segment-duration-seconds 1800 \
  --task-id postprocess-task-id \
  --postprocess-only
```

Postprocess-only mode forbids `--clean-output`, requires successful child
results and sample manifests for every configured segment, never spawns a
collector, and preserves previous terminal control files under
`postprocess_history/`. Original collection source remains in
`runtime_source.json`; the new validator/timeline source is sealed separately
in `postprocess_runtime_source.json`.

## Supervisor Contract

The supervisor owns:

- process-group lifecycle for every synchronized collector child;
- timeout, terminate, kill and reap;
- atomic status and heartbeat writes;
- child pid, command, return code and log paths;
- post-exit process-group inspection and orphan descendant cleanup;
- strict post-collection quality validation;
- timeline generation only after collection quality passes;
- campaign abort on any profile/segment failure.
- sealed SHA-256 provenance for the supervisor, synchronized collector,
  timeline builder and symbol registry.

The supervisor does not own:

- credentials or account state;
- order placement or cancellation;
- strategy processes;
- exchange instance lifecycle.

## Time Contract

Three clock domains are preserved:

1. `common_ts_ns`
   - same-host Python `time.time_ns()` receipt timestamp;
   - the only cross-venue as-of join clock.
2. `source_local_ts_ns`
   - local receipt timestamp of the current venue/track book state.
3. `source_exchange_ts_ns`
   - venue-provided event timestamp when available;
   - diagnostic only.

For every timeline row and source:

```text
source_local_ts_ns <= common_ts_ns
```

Any future join, source timestamp regression or missing initialized book state
fails the timeline quality gate.

## L2 Reconstruction

Binance:

- initialize from the accepted depth snapshot embedded in raw;
- apply every accepted `depthUpdate`;
- quantity `0` deletes a price level;
- verify `pu` continuity while replaying;
- emit a new book state after each snapshot/depth event.

Hyperliquid fast:

- consume the independent `fast=true` raw track;
- each `l2Book` payload replaces the full fast shallow state.

Hyperliquid standard:

- consume the independent standard raw track;
- each `l2Book` payload replaces the full deeper state.

The primary timeline is the ordered union of those three L2 event streams.
After the three books are initialized, every source event emits one row with
all three books as-of that local receipt timestamp.

Main-DEX profiles (`BTC`, `ETH`) require four research tracks: fast market,
standard L2, asset context and main allMids. Named-DEX profiles
(`xyz:SKHX`, `xyz:MU`) additionally require target-DEX allMids.

## Timeline Fields

Each row includes:

- campaign, segment and profile identity;
- `common_seq`, `common_ts_ns`;
- trigger venue/track/raw sequence;
- Binance local/exchange timestamp and age;
- Hyperliquid fast local/exchange timestamp and age;
- Hyperliquid standard local/exchange timestamp and age;
- top-N bid/ask prices and quantities for all three states;
- Hyperliquid `n` order-count values;
- segment-boundary and quality fields.

## Segment Boundary

Each segment starts from fresh exchange snapshots. Segment timelines may be
indexed and analyzed together, but the implementation does not claim exact
continuity across the gap between two segments. Boundary gaps and source
windows are explicit in `timeline_index.csv`.

The final two-segment c6in canary observed common-timeline restart gaps of
about `1.22-1.32s`. These explicit restart boundaries no longer include
timeline post-processing, but they are still not zero and remain excluded
from exact-continuity claims.

## Strict Quality

A profile segment passes only when:

- collector child exits `0`;
- Binance snapshot bridge and replay readiness pass;
- Binance continuity gaps are `0`;
- required Binance event streams are nonempty;
- Hyperliquid research-max and all track gates pass;
- fast L2 is shallow and standard L2 is deeper;
- required raw files and SHA values match;
- Binance, Hyperliquid fast L2 and Hyperliquid standard L2 reconnect counts
  are `0`;
- auxiliary snapshot tracks may reconnect only when connection/disconnect
  counts reconcile, every subscription is acknowledged again, every required
  channel resumes, raw rows reconcile, parse errors are zero and all
  coverage/freshness/gap gates pass;
- every accepted auxiliary reconnect produces an explicit degraded interval
  from the last pre-disconnect required-channel message through the first
  post-ACK resumed message;
- collection overlap meets the configured ratio;
- required-channel coverage, tail freshness and maximum arrival gap pass;
- timeline has no future joins, source regressions or replay gaps.
- timeline source ages remain within their configured per-track limits.

Timeline gzip and manifest writes are fail-closed. A rebuild removes stale
terminal files first, writes gzip to a temporary path and atomically replaces
the final path only after replay completes.

Default source-age limits are `2000ms` for Binance L2, `2000ms` for
Hyperliquid fast L2 and `15000ms` for Hyperliquid standard L2.

Auxiliary degraded intervals are not silently forward-filled. Research code
that consumes `asset_context`, `main_all_mids` or `target_dex_all_mids` must
exclude or explicitly mask those intervals. They do not enter the primary
three-track common L2 timeline.

## Research Use

The common timeline is suitable for:

- cross-venue lead/lag feature construction on one local receipt clock;
- separate study of Hyperliquid shallow/faster and deep/slower state;
- depth imbalance, spread, microprice and state-age features;
- later L2-level execution models that explicitly state their assumptions.

For maker research, Binance events can be selected as decision triggers while
the row still carries the latest Hyperliquid fast and standard states. Source
age columns make stale execution-side state observable rather than silently
forward-filling it.

## Boundaries

This task supports public L2 reconstruction and same-host as-of alignment.
It does not prove:

- L3/L4 queue position;
- exact fill probability or fill simulation;
- exact continuity across segment restarts;
- venue contract economic equivalence;
- strategy profitability or live readiness.
