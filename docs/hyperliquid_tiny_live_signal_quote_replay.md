# Hyperliquid Tiny-Live Signal / Quote Replay

Task: `0617T005`

## Scope

This is a read-only replay / calibration task for the `0617T004` protocol.
It uses local public/read-only artifacts only. It does not place orders, cancel
orders, amend orders, query accounts, read credentials, call private endpoints,
start a live bot, claim fills, claim PnL, or claim maker viability.

## Inputs

Direct full replay input available on this host:

- `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/pricing_signal_rows.csv`

Historical row-level calibration input available on this host:

- `local_live_analysis/basis_positive_row_level_generator_0609T008/row_level_read_only_cases.csv`
- `local_live_analysis/basis_positive_row_level_generator_0609T008/source_artifact_manifest.csv`

The `0609T008` manifest lists seven historical event-mode samples. Their
original `pricing_signal_rows.csv` paths are old absolute paths and are not
present on this host, so they are used for basis-threshold distribution only,
not full quote replay.

## Replay Rule

The runner consumes the `0617T004` protocol rule:

- `basis_mid = binance_mid - hyperliquid_mid`
- `basis_mid_ticks = basis_mid / hyperliquid_tick_size`
- positive eligible signal -> Hyperliquid maker buy intent
- negative eligible signal -> Hyperliquid maker sell intent
- maker-only / post-only
- no crossing
- no taker fallback
- max order size `0.01 BTC`
- max position `0.04 BTC`

Threshold grid:

- `10, 20, 30, 40, 50, 75, 100` ticks

Persistence grid:

- `1, 2, 3` observations

## Results

The direct full quote replay evaluated `3595` de-duplicated decision rows from
the locally available `0601T005` pricing-signal artifact.

Observed full replay behavior:

- trigger intent count remains very small across tested thresholds
- sell intents dominate the one fully replayed local sample
- stale/data-gap rejection dominates because the strict fresh-source filter is
  conservative
- post-only crossing rejections are zero in the tested replay
- simulated cap / reduce-side-only triggers are high relative to accepted
  intents, so position policy needs further calibration before live

The historical row-level calibration input covers seven source samples with
`3545` basis-positive rows. It gives useful basis magnitude distribution:

- per-sample median absolute basis ranges from `43.5` to `62.5` ticks
- per-sample p90 absolute basis ranges from `91.5` to `235.5` ticks
- all row-level cases are basis-positive by construction, so this artifact does
  not calibrate negative-side behavior

## Boundary

This task supports further calibration, but it does not make `0616T008`
live-ready. The original seven historical `pricing_signal_rows.csv` files
should be restored or regenerated locally before claiming full multi-sample
quote replay coverage.

## Final Recommendation

- `hyperliquid_tiny_live_signal_quote_replay_needs_threshold_calibration`

The next step should either retrieve/regenerate the original historical
pricing-signal rows for full multi-sample quote replay, or explicitly approve a
narrower tiny-live threshold policy using only the available replay evidence and
its limitations.
