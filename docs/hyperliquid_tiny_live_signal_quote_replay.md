# Hyperliquid Tiny-Live Signal / Quote Replay

Task: `0617T005`

## Scope

This is a read-only replay / calibration task for the `0617T004` protocol.
It uses local public/read-only artifacts only. It does not place orders, cancel
orders, amend orders, query accounts, read credentials, call private endpoints,
start a live bot, claim fills, claim PnL, or claim maker viability.

## Inputs

Full quote replay inputs available on this host:

- `local_live_analysis/binance_led_hyperliquid_pricing_signal_0601T005/pricing_signal_rows.csv`
- `local_live_analysis/event_horizon_comparison_0604T002/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_a_event/pricing_signal_rows.csv`
- `local_live_analysis/event_horizon_comparison_0604T002/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_b_event/pricing_signal_rows.csv`
- `local_live_analysis/event_horizon_comparison_0604T002/binance_led_hyperliquid_pricing_signal_xemm_0603_quiet_c_event/pricing_signal_rows.csv`
- `local_live_analysis/basis_positive_targeted_public_collection_0609T002/binance_led_hyperliquid_pricing_signal_xemm_0609_normal_a_event/pricing_signal_rows.csv`
- `local_live_analysis/basis_positive_targeted_public_collection_0609T002/binance_led_hyperliquid_pricing_signal_xemm_0609_normal_b_event/pricing_signal_rows.csv`
- `local_live_analysis/basis_positive_targeted_public_collection_0609T002/binance_led_hyperliquid_pricing_signal_xemm_0609_active_a_event/pricing_signal_rows.csv`
- `local_live_analysis/basis_positive_targeted_public_collection_0609T002/binance_led_hyperliquid_pricing_signal_xemm_0609_active_b_event/pricing_signal_rows.csv`

Historical row-level calibration input available on this host:

- `local_live_analysis/basis_positive_row_level_generator_0609T008/row_level_read_only_cases.csv`
- `local_live_analysis/basis_positive_row_level_generator_0609T008/source_artifact_manifest.csv`

The `0609T008` manifest lists seven historical event-mode samples. Their
manifest paths resolve on this host under `local_live_analysis/`.
`source_availability.csv` records all seven as
`local_direct_file_available=true` and `replay_source_used=pricing_signal_rows`.

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

The full quote replay consumed `8` pricing-signal inputs, replayed `161455`
raw pricing rows, and evaluated `26948` de-duplicated decision rows across the
threshold / persistence grid.

Observed full replay behavior:

- buy and sell intents are balanced across the complete local replay set
- stale/data-gap rejection dominates because the strict fresh-source filter is
  conservative
- post-only crossing rejections are zero in the tested replay
- simulated cap / reduce-side-only triggers are high relative to accepted
  intents, so the candidate remains read-only until QA/controller ratification

Calibration summary:

- primary read-only candidate: `75` ticks with `2` observations of persistence
- primary candidate accepted `654` theoretical intents: `313` buy and `341`
  sell, or `2.4269%` of `26948` decision rows
- primary candidate covers `8/8` samples with any intent, `7/8` with buy
  intent, and `8/8` with sell intent
- stricter low-activity fallback: `75` ticks with `3` observations of
  persistence
- stricter fallback accepted `327` theoretical intents: `148` buy and `179`
  sell, or `1.2134%` of `26948` decision rows
- `100` ticks with `3` observations is too sparse for balanced tiny-live
  bootstrap evidence: `193` intents, `0.7162%` of rows, and only `5/8`
  samples with buy intent

The historical row-level calibration input covers seven source samples with
`3545` basis-positive rows and is used as a basis-distribution cross-check:

- per-sample median absolute basis ranges from `43.5` to `62.5` ticks
- per-sample p90 absolute basis ranges from `91.5` to `235.5` ticks
- all row-level cases are basis-positive by construction, so this artifact does
  not calibrate negative-side behavior

## Boundary

This task establishes full local multi-sample quote replay coverage and a
read-only threshold candidate, but it does not itself authorize `0616T008` live
execution. Any live task still needs QA acceptance and controller ratification
of the chosen threshold / persistence policy.

## Final Recommendation

- `hyperliquid_tiny_live_signal_quote_replay_ready_for_qa`

The next step is QA acceptance of `0617T005`. If accepted, the controller can
ratify `75` ticks with `2` observations as the primary tiny-live read-only
calibration, with `75` ticks and `3` observations as a stricter low-activity
fallback.
