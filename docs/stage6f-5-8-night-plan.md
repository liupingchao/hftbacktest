# Stage 6F 5-8-night Plan

## Goal

Collect a fresh `1h` current-format live sample and archive it as `5-8-night`.

The sample is for Stage 6F measurement, not immediate strategy-rule tuning. The specific question is whether cancel-requested fill risk is a stable cross-window problem before adding a narrow same-side re-add rule.

## Live Run Setup

- Remote host: `admin@awsserver1`
- Remote root: `/home/admin/hft_live`
- Remote run directory: `/home/admin/hft_live/runs/5-8-night`
- Symbol: `BTCUSDT`
- Duration: `1h`
- Local archive directory: `local_live_analysis/5-8-night`
- Archive tarball: `local_live_analysis/archive/5-8-night.tar.gz`

Run config requirements:

- keep current live/backtest alignment fields
- use the top-ranked latest 5-7 Stage 6E candidate:
  - `strategy.min_quote_update_interval_ms = 150`
  - `risk.base_spread = 0.4`
  - `risk.k_vol = 1.5`
  - `risk.k_inv = 0.0003`
  - `risk.order_notional = 50.0`
- `risk.max_notional_pos = 250.0`
- `risk.max_position_qty = 0.003`
- `risk.inventory_inflight_exposure_enabled = true`
- `risk.inventory_add_side_cancel_cooldown_ms = 0.0`
- do not use broad cooldown as the production candidate behavior

## Collection Procedure

1. Sync the current Stage 6E code needed by live/replay to awsserver1.
2. Create `/home/admin/hft_live/runs/5-8-night/{data,logs,output}`.
3. Write a run-local `config_live.toml` with `run_id_prefix = "5-8-night"`.
4. Copy the connector config into the run directory.
5. Launch `examples/binance_tick_mm/deploy/run_live.sh` with `DATA_DIR` pointed at the run-local data directory.
6. Let it run for approximately `3600s`.
7. Stop bot, connector, and collector panes.
8. Capture tmux logs into the run-local `logs/` directory.

## Local Replay And Archive

Run:

```bash
python examples/binance_tick_mm/align_live_run.py --run-id 5-8-night
```

Expected outputs:

- `local_live_analysis/5-8-night/audit_live_5-8-night.csv`
- `local_live_analysis/5-8-night/out/live_raw/btcusdt/manifest_*.json`
- `local_live_analysis/5-8-night/out/backtest_audit_replay/summary_audit_replay.json`
- `local_live_analysis/5-8-night/alignment_report_audit_replay.json`
- `local_live_analysis/5-8-night/live_alignment_summary.md`
- `local_live_analysis/archive/5-8-night.tar.gz`
- `local_live_analysis/archive/5-8-night.tar.gz.sha256`

## Stage 6F Measurement

Extract or add metrics from live/audit replay:

- `fill_after_cancel_request_count`
- `fill_after_cancel_request_qty`
- `fill_after_cancel_request_notional`
- `same_side_readd_while_cancel_requested_count`
- `same_side_readd_while_cancel_requested_qty`
- `worsening_fill_after_cancel_request_count`
- inventory before/after cancel-requested fills
- max-position contribution from cancel-requested fills

## Acceptance

The sample is accepted for Stage 6F if:

- live duration is close to `1h`
- live audit and collector gzip are present
- replay conversion succeeds
- audit replay runs on the same live window
- startup-excluded replay lag gate passes
- blocking working-order/action/throttle gates do not regress
- archive checksum is written

The sample does not by itself approve a narrow cancel-requested rule. That decision requires repeated evidence across multiple OOS windows.
