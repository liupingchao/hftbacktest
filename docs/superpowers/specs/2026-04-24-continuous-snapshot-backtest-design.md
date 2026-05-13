# Continuous Multi-Day Snapshot Backtest Design

## Goal

Make long-range validation explicit and testable: one initial book snapshot at the beginning, one continuous multi-day backtest process, and strategy/account state preserved across all daily data files.

## Scope

This design covers the current Binance tick market-making backtest path under `examples/binance_tick_mm`.

It does not add checkpoint/resume, daily account resets, or new strategy logic.

## Current State

The existing code already supports the core path:

- `pipeline.py` can build a date-range manifest with ordered `data_files` and an `initial_snapshot`.
- `backtest_tick_mm.py` passes `manifest["initial_snapshot"]` to `BacktestAsset.initial_snapshot()`.
- For `window = "full_day"`, `backtest_tick_mm.py` passes all manifest `data_files` to the asset.
- Metrics are accumulated continuously while per-day rows are emitted for reporting.

The missing part is explicit validation and tests proving this behavior is reliable for long ranges.

## Recommended Approach

Use manifest-level continuous runs.

A long-range manifest should contain:

```json
{
  "start_day": "2026-01-01",
  "end_day": "2026-12-31",
  "initial_snapshot": "/path/to/snapshot_before_20260101.npz",
  "data_files": [
    "/path/to/btcusdt_20260101.npz",
    "/path/to/btcusdt_20260102.npz"
  ]
}
```

The backtest should run one simulator instance over all files:

```text
initial snapshot once
+ day 1 events
+ day 2 events
+ ...
+ final day events
```

Inventory, cash, working simulator state, and aggregate metrics remain continuous. Daily summary rows are only reporting boundaries.

## Data Flow

```text
Tardis trades + incremental_book_L2
        ↓
pipeline.py converts daily NPZ files
        ↓
optional initial snapshot before first day
        ↓
manifest with ordered data_files
        ↓
backtest_tick_mm.py single continuous run
        ↓
summary.json + daily_summary.csv
```

## Backtest Behavior

### Full-Day Mode

For `window = "full_day"` and a manifest with multiple data files:

- Use every manifest file in order.
- Pass file paths directly to `BacktestAsset.data()`.
- Apply `initial_snapshot` once if present.
- Do not reset strategy metrics, inventory, or simulator state between files.

### Windowed Mode

For `first_5m`, `first_2h`, or `first_6h`:

- Keep the current behavior.
- Load only the first data file.
- Slice it to the requested window.
- This mode is for smoke tests and short debugging, not final long-range validation.

## Manifest Validation

Add a small validation function in the backtest path.

It should fail fast when:

- `data_files` is missing or empty.
- Any data file path does not exist.
- `initial_snapshot` is set but the path does not exist.
- A full-day multi-file manifest contains duplicate data file paths.

It should preserve manifest order. It should not sort files silently, because silently reordering the event stream could hide data-pipeline mistakes.

## Summary Metadata

When `summary.enabled = true`, include continuous-run metadata in `summary.json`:

```json
{
  "continuous_run": true,
  "initial_snapshot": "/path/to/snapshot_before_20260101.npz",
  "data_file_count": 365,
  "data_files": ["..."]
}
```

`continuous_run` is true when `window == "full_day"` and `data_file_count > 1`.

This metadata is for auditability. It should not change strategy behavior.

## Error Handling

Use clear `ValueError` or `FileNotFoundError` messages that identify the bad manifest field and path.

Examples:

```text
manifest.data_files must contain at least one file
manifest.initial_snapshot does not exist: /path/to/snapshot.npz
manifest.data_files contains duplicate path: /path/to/day.npz
```

## Test Plan

### Unit Tests

1. **Manifest validation accepts continuous manifest**
   - Create three temporary NPZ placeholder files and one snapshot placeholder.
   - Build a manifest with ordered `data_files` and `initial_snapshot`.
   - Assert validation passes and keeps order unchanged.

2. **Manifest validation rejects missing snapshot**
   - Set `initial_snapshot` to a missing file.
   - Assert a clear `FileNotFoundError`.

3. **Manifest validation rejects missing data file**
   - Include one missing path in `data_files`.
   - Assert a clear `FileNotFoundError`.

4. **Manifest validation rejects duplicate data files**
   - Include the same data file twice.
   - Assert a clear `ValueError`.

5. **Full-day mode uses all data files**
   - Monkeypatch or fake the asset builder boundary.
   - For `window = "full_day"` and three files, assert the asset receives all three paths.

6. **Windowed mode uses only the first sliced file**
   - Use a tiny structured NPZ fixture.
   - For `window = "first_6h"`, assert only the first file is loaded and sliced.

7. **Initial snapshot applied once**
   - Use a fake asset object.
   - Assert `initial_snapshot()` is called exactly once with the manifest snapshot path.

### Integration / Slow Test

8. **Two-day continuous smoke test**
   - Prepare two converted days with a start snapshot.
   - Run with `audit.mode = "off"` and `summary.enabled = true`.
   - Assert `summary.json` exists.
   - Assert `summary.json.continuous_run == true`.
   - Assert `summary.json.data_file_count == 2`.
   - Assert `daily_summary.csv` has two day rows.

## Success Criteria

- Long-range full-day manifests are validated before running.
- Missing files or bad snapshots fail before the expensive simulation starts.
- Summary output clearly records whether the run was continuous.
- Tests prove multi-day full-day backtests use all files and apply the initial snapshot once.
- Existing short-window behavior remains unchanged.
