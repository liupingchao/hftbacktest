# SKHYNIX Cross-Exchange Research Dataset

Tasks: `0730T013`, fail-closed repair `0730T014`

## Purpose

This R0 dataset builder converts the accepted local collection layout into an
auditable research event store without reconnecting to either exchange.

The store is a composition of:

- existing `common_l2_timeline.csv.gz` files as replayed L2 state;
- normalized Binance bookTicker and trade sidecars;
- normalized Hyperliquid BBO and individual trade-item sidecars;
- auxiliary event indexes retaining full compact payload JSON;
- segment-boundary and degraded-interval masks.

Original raw files remain the information-complete source of truth.

## Usage

```bash
/Users/liu/.local/conda/envs/hftbacktest/bin/python \
  examples/hyperliquid/cross_exchange_research_dataset.py \
  --campaign-dir \
  local_live_analysis/cross_exchange_collection_campaign_0730T011_skhynix_4h_8x30m \
  --output-dir \
  local_live_analysis/skhynix_cross_exchange_research_0730T013 \
  --symbol-profile skhynix
```

The output path must be absent or empty. Replacing an existing output requires
the explicit `--clean-output` acknowledgement. A clean rebuild is produced in
a sibling temporary directory while the previous output remains intact; final
publication uses backup/rename/rollback only after the complete build passes.

## Output Layout

```text
research-output/
  research_input_manifest.json
  segment_and_mask_index.csv
  segments/
    segment_0001/
      binance_hot_events.csv.gz
      hyperliquid_hot_events.csv.gz
      hyperliquid_auxiliary_events.csv.gz
      segment_event_store_manifest.json
```

## Reconciliation

The builder fails closed unless:

- campaign and strict segment quality pass;
- timeline index campaign/profile/segment order and continuity policy match;
- every timeline SHA, row count, first/last timestamp and segment gap matches
  the manifest and actual gzip CSV;
- every source raw SHA matches its collection, research-bundle and timeline
  provenance;
- Binance snapshot/depth/bookTicker/trade counts and symbols match exactly;
- Hyperliquid fast/standard/auxiliary channel and control-message sets, raw
  row counts, coins and target dex match exactly;
- every Hyperliquid track reports reconciled raw rows and zero parse errors;
- Hyperliquid trade batches can be exploded without invalid coin rows;
- all source-local timestamps are nondecreasing inside each source;
- Binance symbol and Hyperliquid coin/dex identities match the profile;
- source hashes are unchanged after the build.

The final campaign manifest records both source-message counts and normalized
row counts. Hyperliquid trade-message count and exploded trade-item count are
kept separate.

## Masks

`segment_and_mask_index.csv` contains:

- one `segment_epoch` row per segment;
- explicit `auxiliary_degraded_interval` rows.

Auxiliary output rows carry both `degraded` and `degraded_interval_id`. This
allows later as-of feature code to reject an interval even when the underlying
socket recovered and the overall segment remained valid.

## Boundaries

This task is local and offline:

- no AWS or SSH;
- no new public-data collection;
- no private/account/order endpoint;
- no strategy process;
- no signal fitting or parameter search;
- no arbitrage, exact queue, exact fill or PnL conclusion.

Any later data collection requires a separate formal task, explicit user
authorization and a user-confirmed active trading window.
