# Binance Top5 Provenance Sidecars

`examples/binance_tick_mm/binance_top5_provenance.py` builds a standalone Binance raw-data provenance layer for the maker workflow.

The tool deliberately keeps the standard hftbacktest `data` npz array unchanged. The npz remains the replay-compatible event stream with the canonical fields:

- `ev`
- `exch_ts`
- `local_ts`
- `px`
- `qty`
- `order_id`
- `ival`
- `fval`

Binance-specific provenance is written to sidecars instead:

- `raw_provenance.csv`: raw message metadata, including `raw_seq`, depth `U/u/pu`, snapshot `lastUpdateId`, and bookTicker fields.
- `raw_to_npz_mapping.csv`: explicit mapping from each `raw_seq` to final standard npz rows after local timestamp and event-order correction.
- `top5_sidecar.csv`: reconstructed top5 book state, sync status, bookTicker/depth BBO match, and source timestamps.
- `joined_decisions.csv`: optional read-only as-of join from live decision rows to top5 sidecar rows.
- `sidecar_manifest.json`: schema version, raw file identity, converter options, top5 level count, tick size, and output paths.

Decision joins are as-of joins only: a decision row may join only to a top5 row whose `local_ts` is less than or equal to the decision `ts_local`. Future joins are not allowed and must report `future_join_count = 0`.

This is a top5-only data-quality layer. It can support later top5 pricing, top5 OFI proxy, and top5 microprice proxy studies if join-age and mismatch diagnostics pass. It does not prove full L2 equivalence, exact queue position, or strategy PnL, and it does not make the live strategy consume these fields in real time.
