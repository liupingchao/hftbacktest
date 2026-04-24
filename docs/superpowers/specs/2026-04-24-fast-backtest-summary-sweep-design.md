# Fast Backtest Output and Parameter-Parallel Sweep Design

## Goal

Make large-scale backtesting practical by reducing audit I/O, producing summary metrics during the run, writing per-day summaries, and speeding hyperparameter tuning by running independent parameter configurations in parallel while preserving exact multi-day continuity within each run.

## Context

A 10-day full-day BTCUSDT backtest produced:

- 47,186,748 decision rows
- 22GB audit CSV
- 1532.3 seconds runtime (~153.2 sec/day)
- projected 356-day runtime: ~15.2 hours
- projected 356-day full audit size: hundreds of GB

Compute time is acceptable. Full audit output size and post-run CSV analysis are the main bottlenecks.

## Non-goals

- Do not parallelize a single multi-day strategy run by day, because that resets order/inventory state at day boundaries and breaks exactness.
- Do not change strategy logic or pricing formulas.
- Do not replace the hftbacktest engine.
- Do not implement automatic hyperparameter optimization algorithms yet; implement grid execution and metrics first.

## Architecture

### 1. Audit output modes

Add config keys under `[audit]`:

```toml
[audit]
mode = "full"          # full | actions_only | sampled | off
sample_every = 1000
output_csv = "audit_bt.csv"
flush_every = 1000
```

Mode behavior:

| Mode | Behavior |
|------|----------|
| `full` | Write every decision row. This is current behavior and remains default. |
| `actions_only` | Write rows where `action != "keep"` or `reject_reason != ""`. |
| `sampled` | Write all action/reject rows plus every `sample_every`th decision row. |
| `off` | Do not create audit CSV; write only summary outputs. |

The returned `run_backtest` result includes `audit_csv` only when an audit file is written.

### 2. Streaming summary metrics

Add a metrics accumulator updated inside the backtest loop. It computes metrics without reading the audit CSV:

- `rows`
- `pnl_mtm`
- `max_drawdown_mtm`
- `avg_inventory_score`
- `avg_abs_position_notional`
- `max_abs_position_notional`
- `avg_spread_bps`
- `avg_vol_bps`
- `drop_latency_rate`
- `drop_api_rate`
- action counts
- reject reason counts
- feed latency summary
- entry latency summary
- latency signal summary

Latency percentiles use a fixed histogram to avoid retaining all values in memory. The histogram should cover 0-100ms with enough resolution for p50/p90/p99 diagnostics. Values above the cap go into the last bucket.

Add config keys under `[summary]`:

```toml
[summary]
enabled = true
output_json = "summary.json"
daily_csv = "daily_summary.csv"
```

If `[summary]` is absent, summary still exists in the Python return dict but files are not written unless enabled.

### 3. Per-day summaries

Detect UTC day boundaries using `ts_local`. Maintain both total metrics and current-day metrics. When day changes, flush the current day to `daily_summary.csv` and reset the daily accumulator.

Daily CSV fields include:

- `day`
- `rows`
- `pnl_mtm`
- `max_drawdown_mtm`
- `avg_inventory_score`
- `avg_abs_position_notional`
- `max_abs_position_notional`
- `avg_spread_bps`
- `avg_vol_bps`
- `drop_latency_rate`
- `drop_api_rate`
- action counts as columns, e.g. `action_keep`, `action_submit_buy`
- reject counts as columns, e.g. `reject_latency_guard`, `reject_api_interval_guard`

The total summary JSON includes the same aggregate fields plus paths to audit and daily summary files.

### 4. Parameter-parallel sweep runner

Add `sweep_backtest.py`.

CLI:

```bash
python sweep_backtest.py \
  --base-config config.toml \
  --manifest manifest.json \
  --grid sweep.toml \
  --workers 8 \
  --window full_day \
  --out ./out/sweeps
```

`grid.toml` example:

```toml
[risk]
k_inv = [0.0005, 0.001, 0.002]
base_spread = [0.5, 1.0, 1.5]
k_pos = [0.00005, 0.0001, 0.0002]

[fair]
w_imb = [0.0, 0.1, 0.25, 0.5]
```

Each parameter combination runs as a separate process. Each process runs an exact sequential multi-day backtest using the full manifest and writes:

```text
sweeps/run_0001/config.toml
sweeps/run_0001/summary.json
sweeps/run_0001/daily_summary.csv
```

After all workers finish, write:

```text
sweeps/sweep_summary.csv
sweeps/sweep_summary.json
```

The sweep runner should default child runs to:

```toml
[audit]
mode = "off"

[summary]
enabled = true
```

unless explicitly overridden.

### 5. Error handling

- Invalid `audit.mode` fails fast with a clear error.
- `sampled` mode requires `sample_every > 0`.
- Sweep runner validates grid values are arrays and each target config section/key exists or is explicitly created.
- Failed sweep workers record the error in sweep summary and do not stop other workers unless `--fail-fast` is set.

## Test Plan

### 1. Summary equivalence test

Run a short backtest with `audit.mode = "full"`. Compute metrics once via the new streaming summary and once by reading the audit CSV. Verify:

- row counts match
- PnL matches within floating point tolerance
- drop rates match
- average inventory/spread/vol metrics match

### 2. Audit mode test

Run `first_5m` backtests with each audit mode:

- `full`: audit row count equals summary rows
- `actions_only`: every audit row has `action != "keep"` or non-empty `reject_reason`
- `sampled`: includes sampled keep rows plus all action/reject rows
- `off`: no audit file exists, summary JSON and daily CSV still exist

### 3. Per-day summary test

Run a two-day backtest. Verify:

- daily CSV has exactly two data rows
- sum of daily rows equals total summary rows
- sum of daily PnL approximately equals total PnL

### 4. Sweep runner test

Run a tiny grid on `first_5m`, e.g. two configs:

```toml
[risk]
base_spread = [0.5, 1.0]
```

Verify:

- two run directories are created
- each has config + summary + daily CSV
- sweep summary has two rows
- parameter columns are present

### 5. Performance test

Run the existing 10-day dataset with:

```toml
[audit]
mode = "off"

[summary]
enabled = true
```

Compare against the previous baseline:

- baseline: 1532.3 sec, 22GB audit CSV
- expected: no audit CSV, lower runtime, summary files only

## Success Criteria

- A 356-day run can be executed without creating hundreds of GB of audit CSV.
- Summary metrics are available immediately at run completion.
- Daily summaries allow diagnosing bad days without reading full audit files.
- Hyperparameter sweeps use multiple CPU cores while preserving exact sequential state within each run.
- Existing behavior remains available with `audit.mode = "full"`.
