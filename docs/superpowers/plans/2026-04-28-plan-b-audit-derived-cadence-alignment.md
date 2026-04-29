# Plan B Audit-Derived Cadence Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fixed-interval cadence guessing with live-audit-derived decision replay so the backtest decision schedule matches the live bot's actual `ts_local` cadence.

**Architecture:** Add or validate an `audit_replay` mode under `[backtest_cadence]` that loads live audit timestamps, gates strategy decisions until the backtest clock reaches each live decision timestamp, and records consumed/unconsumed schedule counts plus lag distribution. Use the existing same-window P3 live raw replay inputs, then compare backtest vs live audit with cadence, action, reject, latency, and MAE metrics.

**Tech Stack:** Python, TOML config, CSV audit files, NumPy-backed hftbacktest replay, `pytest`, existing `examples/binance_tick_mm/backtest_tick_mm.py` and `examples/binance_tick_mm/compare_audit.py`.

---

## Context

Plan A tested fixed minimum decision intervals (`0, 2, 5, 8, 10, 15, 20 ms`). It did not solve row-count mismatch: `2ms` reduced backtest rows to `165,625`, below live `210,110`, and action match dropped from `0.8490` to `0.7918`. Fixed intervals improved some fair/spread MAE metrics but damaged action path and API/latency behavior.

Plan B should model live burst/gap structure directly by replaying the live audit decision timestamps.

Use this live run as the initial validation target:

- Run id: `live_btcusdt_1777342116`
- Live audit CSV: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/audit_live_two_phase_align_live_btcusdt_1777342116_20260428_134818.csv`
- P2 manifest: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json`
- Backtest config: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml`
- Same-window start `ts_local`: `1777342117432305664`
- Same-window end `ts_local`: `1777347796789512192`

## Target acceptance criteria

Plan B is successful if the audit-derived replay run produces:

1. Backtest decision rows close to live audit rows (`210,110`) because the schedule is loaded from the live audit.
2. `audit_replay_consumed_count` close to `audit_replay_scheduled_count`, with low `audit_replay_unconsumed_count`.
3. `compare_audit.py` cadence report shows live-to-backtest nearest lag distribution near zero relative to Plan A fixed-interval runs.
4. Action match does not regress below the P3 baseline `0.8490` unless the report clearly shows a different bottleneck.
5. Reject reason match improves from the P3 baseline `0.4513`, or the report isolates that the remaining mismatch is caused by latency/order lifecycle rather than cadence.
6. Results are documented in the local artifact index and summary.

## Files

- Modify: `examples/binance_tick_mm/backtest_tick_mm.py`
  - Add/validate `audit_replay` cadence config parsing.
  - Add/validate live audit timestamp loader.
  - Gate strategy decisions against loaded live timestamps.
  - Include replay schedule counters and lag distribution in backtest result JSON.
- Modify: `examples/binance_tick_mm/test_backtest_tick_mm.py`
  - Unit-test timestamp parsing, run-id filtering, deduplication, tolerance behavior, config parsing, and one-schedule-per-feed-event semantics.
- Modify: `examples/binance_tick_mm/compare_audit.py`
  - Add/validate cadence stats for both audit files and nearest live-to-backtest timestamp lag distribution.
- Modify: `examples/binance_tick_mm/config.example.toml`
  - Document `fixed_interval` vs `audit_replay` cadence config.
- Modify: `local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml`
  - Set `[backtest_cadence]` to `mode = "audit_replay"` for the Plan B run.
- Modify: `local_live_analysis/live_btcusdt_1777342116/ARTIFACT_INDEX.md`
  - Add Plan B artifact paths and metrics after execution.
- Modify: `local_live_analysis/live_btcusdt_1777342116/live_summary.md`
  - Add a concise Plan B result summary after execution.

---

## Task 1: Add audit replay schedule unit tests

**Files:**
- Modify: `/home/molly/project/hftbacktest/examples/binance_tick_mm/test_backtest_tick_mm.py`
- Test target: `/home/molly/project/hftbacktest/examples/binance_tick_mm/backtest_tick_mm.py`

- [ ] **Step 1: Add imports for audit replay helpers if missing**

At the top import block in `examples/binance_tick_mm/test_backtest_tick_mm.py`, ensure these names are imported from `backtest_tick_mm`:

```python
from backtest_tick_mm import (
    _audit_replay_decision_due,
    _backtest_cadence_config,
    _load_audit_cadence_schedule,
)
```

If the file already imports from `backtest_tick_mm`, add only the missing names to the existing import list.

- [ ] **Step 2: Add timestamp precision test**

Append this test near the existing cadence tests:

```python
def test_load_audit_cadence_schedule_preserves_nanosecond_precision(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,ts_local,action\nlive,1777342117432305601,keep\n")

    schedule = _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")

    assert schedule == [1777342117432305601]
```

- [ ] **Step 3: Add run-id filter and dedupe test**

Append this test near the existing cadence tests:

```python
def test_load_audit_cadence_schedule_filters_run_id_and_dedupes(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text(
        "run_id,ts_local,action\n"
        "other,100,keep\n"
        "live,300,keep\n"
        "live,100,keep\n"
        "live,300,keep\n"
        "live,200,submit_buy\n"
    )

    schedule = _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")

    assert schedule == [100, 200, 300]
```

- [ ] **Step 4: Add empty-filter error test**

Append this test near the existing cadence tests:

```python
def test_load_audit_cadence_schedule_raises_for_empty_filter(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,ts_local,action\nother,100,keep\n")

    with pytest.raises(ValueError, match="no cadence timestamps loaded"):
        _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")
```

- [ ] **Step 5: Add missing-column error test**

Append this test near the existing cadence tests:

```python
def test_load_audit_cadence_schedule_raises_for_missing_column(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,wrong_ts\nlive,100\n")

    with pytest.raises(KeyError, match="missing cadence timestamp column"):
        _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")
```

- [ ] **Step 6: Add decision due tests**

Append these tests near the existing cadence tests:

```python
def test_audit_replay_decision_due_waits_until_next_schedule() -> None:
    due, idx, lag = _audit_replay_decision_due(99, [100, 200], 0, 0)
    assert (due, idx, lag) == (False, 0, 0)

    due, idx, lag = _audit_replay_decision_due(100, [100, 200], 0, 0)
    assert (due, idx, lag) == (True, 1, 0)


def test_audit_replay_decision_due_uses_tolerance() -> None:
    due, idx, lag = _audit_replay_decision_due(98, [100], 0, 2)

    assert (due, idx, lag) == (True, 1, -2)


def test_audit_replay_decision_due_consumes_one_schedule_per_feed_event() -> None:
    due, idx, lag = _audit_replay_decision_due(250, [100, 200, 300], 0, 0)

    assert (due, idx, lag) == (True, 1, 150)
```

- [ ] **Step 7: Add config parsing test**

Append this test near the existing config tests:

```python
def test_backtest_cadence_config_reads_audit_replay() -> None:
    config = {
        "backtest_cadence": {
            "mode": "audit_replay",
            "audit_csv": "/tmp/live.csv",
            "run_id": "live_run",
            "ts_column": "ts_local",
            "tolerance_ms": 2.5,
        }
    }

    cfg = _backtest_cadence_config(config)

    assert cfg["mode"] == "audit_replay"
    assert cfg["audit_csv"] == "/tmp/live.csv"
    assert cfg["run_id"] == "live_run"
    assert cfg["ts_column"] == "ts_local"
    assert cfg["tolerance_ns"] == 2_500_000
```

- [ ] **Step 8: Run tests and verify they fail before implementation if helpers are missing**

Run:

```bash
cd /home/molly/project/hftbacktest && pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected before implementation if helpers are missing: FAIL with import/name errors for `_load_audit_cadence_schedule`, `_audit_replay_decision_due`, or missing config fields.

Expected if implementation already exists: PASS.

---

## Task 2: Implement audit replay schedule loading and config parsing

**Files:**
- Modify: `/home/molly/project/hftbacktest/examples/binance_tick_mm/backtest_tick_mm.py`
- Test: `/home/molly/project/hftbacktest/examples/binance_tick_mm/test_backtest_tick_mm.py`

- [ ] **Step 1: Add or validate timestamp parser**

In `examples/binance_tick_mm/backtest_tick_mm.py`, ensure this helper exists near other parsing helpers:

```python
def _parse_int_timestamp(raw: str) -> int:
    text = str(raw).strip()
    if not text:
        raise ValueError("empty timestamp")
    try:
        return int(text)
    except ValueError:
        return int(float(text))
```

- [ ] **Step 2: Add or validate audit schedule loader**

In `examples/binance_tick_mm/backtest_tick_mm.py`, ensure this helper exists before `run_backtest`:

```python
def _load_audit_cadence_schedule(audit_csv: Path, run_id: str, ts_column: str) -> list[int]:
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if ts_column not in set(reader.fieldnames or []):
            raise KeyError(f"missing cadence timestamp column: {ts_column}")
        out: set[int] = set()
        for row in reader:
            if run_id and row.get("run_id", "") != run_id:
                continue
            raw = row.get(ts_column, "")
            if not raw:
                continue
            try:
                ts = _parse_int_timestamp(raw)
            except ValueError:
                continue
            if ts > 0:
                out.add(ts)
    schedule = sorted(out)
    if not schedule:
        raise ValueError(f"no cadence timestamps loaded from {audit_csv}")
    return schedule
```

- [ ] **Step 3: Add or validate decision due helper**

In `examples/binance_tick_mm/backtest_tick_mm.py`, ensure this helper exists before `run_backtest`:

```python
def _audit_replay_decision_due(
    ts_local: int,
    schedule: list[int],
    schedule_idx: int,
    tolerance_ns: int,
) -> tuple[bool, int, int]:
    if schedule_idx >= len(schedule):
        return False, schedule_idx, 0
    scheduled_ts = schedule[schedule_idx]
    if ts_local + tolerance_ns < scheduled_ts:
        return False, schedule_idx, 0

    return True, schedule_idx + 1, ts_local - scheduled_ts
```

This intentionally consumes only one scheduled decision per feed event. Do not loop through multiple missed schedule points in one feed event; doing so would create multiple strategy decisions at the same market state and distort action path.

- [ ] **Step 4: Add or validate cadence config parser**

In `examples/binance_tick_mm/backtest_tick_mm.py`, ensure `_backtest_cadence_config` returns these fields:

```python
def _backtest_cadence_config(config: dict[str, Any]) -> dict[str, Any]:
    cadence_cfg = config.get("backtest_cadence", {})
    mode = str(cadence_cfg.get("mode", "")).strip()
    if not mode:
        mode = "fixed_interval"
    if mode not in {"fixed_interval", "audit_replay"}:
        raise ValueError(f"Unsupported backtest_cadence.mode: {mode}")

    enabled = bool(cadence_cfg.get("enabled", mode == "audit_replay"))
    min_interval_ns = 0
    if enabled and mode == "fixed_interval":
        min_interval_ns = int(float(cadence_cfg.get("min_decision_interval_ms", 0.0)) * 1_000_000)

    return {
        "mode": mode,
        "min_interval_ns": min_interval_ns,
        "audit_csv": str(cadence_cfg.get("audit_csv", "")),
        "run_id": str(cadence_cfg.get("run_id", "")),
        "ts_column": str(cadence_cfg.get("ts_column", "ts_local")),
        "tolerance_ns": int(float(cadence_cfg.get("tolerance_ms", 0.0)) * 1_000_000),
    }
```

- [ ] **Step 5: Run cadence unit tests**

Run:

```bash
cd /home/molly/project/hftbacktest && pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: PASS.

---

## Task 3: Wire audit replay into the backtest loop

**Files:**
- Modify: `/home/molly/project/hftbacktest/examples/binance_tick_mm/backtest_tick_mm.py`
- Test: `/home/molly/project/hftbacktest/examples/binance_tick_mm/test_backtest_tick_mm.py`

- [ ] **Step 1: Initialize audit replay state after latency/API config setup**

Inside `run_backtest`, after `latency_guard_ns` is computed, ensure this block exists:

```python
cadence_cfg = _backtest_cadence_config(config)
cadence_interval_ns = int(cadence_cfg["min_interval_ns"])
cadence_mode = str(cadence_cfg["mode"])
audit_replay_schedule: list[int] = []
audit_replay_idx = 0
cadence_skipped_feed_events = 0
cadence_lags_ns: list[int] = []
if cadence_mode == "audit_replay":
    audit_csv = str(cadence_cfg["audit_csv"]).strip()
    if not audit_csv:
        raise ValueError("backtest_cadence.audit_csv is required for audit_replay mode")
    audit_replay_schedule = _load_audit_cadence_schedule(
        _expand(audit_csv),
        run_id=str(cadence_cfg["run_id"]),
        ts_column=str(cadence_cfg["ts_column"]),
    )
```

- [ ] **Step 2: Gate strategy decisions before incrementing `strategy_seq`**

Inside the `while True` feed loop, after valid `best_bid`/`best_ask` checks and before `strategy_seq += 1`, ensure this cadence gate exists:

```python
if cadence_mode == "audit_replay":
    due, audit_replay_idx, lag_ns = _audit_replay_decision_due(
        ts_local,
        audit_replay_schedule,
        audit_replay_idx,
        int(cadence_cfg["tolerance_ns"]),
    )
    if not due:
        cadence_skipped_feed_events += 1
        continue
    cadence_lags_ns.append(lag_ns)
elif _should_skip_strategy_decision(ts_local, last_decision_ts, cadence_interval_ns):
    cadence_skipped_feed_events += 1
    continue
```

- [ ] **Step 3: Include audit replay metadata in result JSON**

In the `result = { ... }` dictionary returned by `run_backtest`, ensure these fields exist:

```python
"backtest_cadence_mode": cadence_mode,
"backtest_cadence_interval_ns": cadence_interval_ns,
"audit_replay_scheduled_count": len(audit_replay_schedule),
"audit_replay_consumed_count": audit_replay_idx,
"audit_replay_unconsumed_count": max(0, len(audit_replay_schedule) - audit_replay_idx),
"cadence_skipped_feed_events": cadence_skipped_feed_events,
"audit_replay_lag_ns": _distribution(cadence_lags_ns),
```

- [ ] **Step 4: Run unit tests**

Run:

```bash
cd /home/molly/project/hftbacktest && pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: PASS.

---

## Task 4: Add cadence comparison metrics to audit comparison

**Files:**
- Modify: `/home/molly/project/hftbacktest/examples/binance_tick_mm/compare_audit.py`
- Test command: use a small Python smoke test or run the script against the live/backtest files after Task 6.

- [ ] **Step 1: Add integer timestamp reader**

In `compare_audit.py`, ensure these helpers exist before `_summary`:

```python
def _parse_int_timestamp(raw: str) -> int:
    text = raw.strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        return int(float(text))


def _series_int(rows: list[dict[str, str]], key: str) -> list[int]:
    out = []
    for r in rows:
        try:
            v = _parse_int_timestamp(r.get(key, "") or "0")
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    return out
```

- [ ] **Step 2: Add cadence distribution helper**

In `compare_audit.py`, ensure this helper exists before `compare`:

```python
def _dist(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    s = sorted(vals)
    return {
        "count": float(len(vals)),
        "mean": float(mean(vals)),
        "p50": float(_quantile(s, 0.50)),
        "p90": float(_quantile(s, 0.90)),
        "p99": float(_quantile(s, 0.99)),
        "max": float(s[-1]),
    }
```

- [ ] **Step 3: Add cadence stats helper**

In `compare_audit.py`, ensure this helper exists before `compare`:

```python
def _cadence_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    ts = _series_int(rows, "ts_local")
    raw_deltas = [b - a for a, b in zip(ts, ts[1:])]
    deltas = [float(delta) for delta in raw_deltas if delta >= 0]
    return {
        "rows": len(rows),
        "ts_local_count": len(ts),
        "ts_local_first": ts[0] if ts else 0,
        "ts_local_last": ts[-1] if ts else 0,
        "negative_delta_count": sum(delta < 0 for delta in raw_deltas),
        "delta_ns": _dist(deltas),
    }
```

- [ ] **Step 4: Add nearest lag stats helper**

In `compare_audit.py`, ensure this helper exists before `compare`:

```python
def _nearest_lag_stats(bt_rows: list[dict[str, str]], live_rows: list[dict[str, str]]) -> dict[str, Any]:
    import bisect

    bt_ts = sorted(_series_int(bt_rows, "ts_local"))
    live_ts = _series_int(live_rows, "ts_local")
    signed = []
    for ts in live_ts:
        i = bisect.bisect_left(bt_ts, ts)
        candidates = []
        if i < len(bt_ts):
            candidates.append(bt_ts[i])
        if i > 0:
            candidates.append(bt_ts[i - 1])
        if candidates:
            nearest = min(candidates, key=lambda x: abs(x - ts))
            signed.append(float(nearest - ts))
    return {
        "count": len(signed),
        "abs_ns": _dist([abs(v) for v in signed]),
        "signed_ns": _dist(signed),
    }
```

- [ ] **Step 5: Include cadence section in comparison report**

In `compare`, ensure the returned dictionary includes:

```python
"cadence": {
    "bt": _cadence_stats(bt_rows),
    "live": _cadence_stats(live_rows),
    "nearest_lag_live_to_bt": _nearest_lag_stats(bt_rows, live_rows),
},
```

- [ ] **Step 6: Run a syntax check**

Run:

```bash
cd /home/molly/project/hftbacktest && python -m py_compile examples/binance_tick_mm/compare_audit.py
```

Expected: exits `0` with no output.

---

## Task 5: Configure Plan B audit replay run

**Files:**
- Modify: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml`
- Modify: `/home/molly/project/hftbacktest/examples/binance_tick_mm/config.example.toml`

- [ ] **Step 1: Set local Plan B config**

In `local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml`, ensure this exact section exists:

```toml
[backtest_cadence]
mode = "audit_replay"
enabled = true
audit_csv = "/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/audit_live_two_phase_align_live_btcusdt_1777342116_20260428_134818.csv"
run_id = "live_btcusdt_1777342116"
ts_column = "ts_local"
tolerance_ms = 0.0
```

- [ ] **Step 2: Document cadence modes in example config**

In `examples/binance_tick_mm/config.example.toml`, ensure this section exists:

```toml
[backtest_cadence]
# fixed_interval | audit_replay
# fixed_interval uses min_decision_interval_ms.
# audit_replay loads live audit ts_local values and only runs strategy decisions
# when the backtest clock reaches the next live decision timestamp.
mode = "fixed_interval"
enabled = false
min_decision_interval_ms = 0.0
audit_csv = ""
run_id = ""
ts_column = "ts_local"
tolerance_ms = 0.0
```

- [ ] **Step 3: Verify config parses**

Run:

```bash
cd /home/molly/project/hftbacktest && python - <<'PY'
import tomllib
from pathlib import Path
p = Path('/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml')
cfg = tomllib.loads(p.read_text())
print(cfg['backtest_cadence'])
PY
```

Expected output includes:

```text
'mode': 'audit_replay'
'audit_csv': '/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/audit_live_two_phase_align_live_btcusdt_1777342116_20260428_134818.csv'
'run_id': 'live_btcusdt_1777342116'
```

---

## Task 6: Run Plan B same-window audit replay backtest

**Files:**
- Use: `/home/molly/project/hftbacktest/examples/binance_tick_mm/backtest_tick_mm.py`
- Use: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml`
- Use: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json`

- [ ] **Step 1: Run same-window audit replay**

Run:

```bash
cd /home/molly/project/hftbacktest && HFTBACKTEST_USE_LOCAL_PY=1 python examples/binance_tick_mm/backtest_tick_mm.py \
  --config /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml \
  --manifest /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json \
  --window full_day \
  --slice-ts-local-start 1777342117432305664 \
  --slice-ts-local-end 1777347796789512192
```

Expected: command exits `0` and printed JSON includes:

```json
{
  "backtest_cadence_mode": "audit_replay",
  "audit_replay_scheduled_count": 210110,
  "audit_replay_consumed_count": 210110,
  "audit_replay_unconsumed_count": 0
}
```

If `audit_replay_consumed_count` is lower than `audit_replay_scheduled_count`, continue to Task 7 and document the gap; do not patch strategy logic in this plan.

- [ ] **Step 2: Capture output paths and counters**

From the printed JSON, record these values for documentation:

```text
audit_csv
summary_json
daily_summary_csv
rows
audit_rows
backtest_cadence_mode
audit_replay_scheduled_count
audit_replay_consumed_count
audit_replay_unconsumed_count
cadence_skipped_feed_events
audit_replay_lag_ns
```

---

## Task 7: Compare Plan B backtest audit against live audit

**Files:**
- Use: `/home/molly/project/hftbacktest/examples/binance_tick_mm/compare_audit.py`
- Use: generated Plan B backtest audit CSV from Task 6
- Use: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/audit_live_two_phase_align_live_btcusdt_1777342116_20260428_134818.csv`
- Write: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json`

- [ ] **Step 1: Run comparison report**

Replace `<PLAN_B_BT_AUDIT>` with the `audit_csv` path printed in Task 6:

```bash
cd /home/molly/project/hftbacktest && python examples/binance_tick_mm/compare_audit.py \
  --bt <PLAN_B_BT_AUDIT> \
  --live /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/audit_live_two_phase_align_live_btcusdt_1777342116_20260428_134818.csv \
  --out /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json
```

Expected: command exits `0` and writes:

```text
/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json
```

- [ ] **Step 2: Extract Plan B metrics**

Run:

```bash
cd /home/molly/project/hftbacktest && python - <<'PY'
import json
from pathlib import Path
p = Path('/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json')
r = json.loads(p.read_text())
print('bt_rows', r['bt_summary']['rows'])
print('live_rows', r['live_summary']['rows'])
print('common_rows', r['alignment']['common_rows'])
print('action_match_rate', r['alignment']['action_match_rate'])
print('reject_reason_match_rate', r['alignment']['reject_reason_match_rate'])
print('bt_drop_latency_rate', r['bt_summary']['drop_latency_rate'])
print('live_drop_latency_rate', r['live_summary']['drop_latency_rate'])
print('bt_drop_api_rate', r['bt_summary']['drop_api_rate'])
print('live_drop_api_rate', r['live_summary']['drop_api_rate'])
print('mae', r['alignment']['mae'])
print('cadence_bt', r['cadence']['bt'])
print('cadence_live', r['cadence']['live'])
print('nearest_lag_live_to_bt', r['cadence']['nearest_lag_live_to_bt'])
PY
```

Expected: printed metrics include cadence stats and nearest lag. Use these to judge whether Plan B improved cadence alignment vs Plan A.

---

## Task 8: Interpret Plan B result

**Files:**
- Read: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json`
- Read: Task 6 backtest result JSON or `summary_json`

- [ ] **Step 1: Classify cadence alignment**

Use these rules:

```text
If audit_replay_unconsumed_count == 0 and nearest_lag_live_to_bt.abs_ns.p99 is small relative to live delta_ns.p50:
  cadence alignment = good
Else if unconsumed_count is low but nearest lag p99 is large:
  cadence alignment = partial; investigate feed event availability/tolerance
Else:
  cadence alignment = poor; audit schedule could not be replayed from available market data timestamps
```

- [ ] **Step 2: Classify action/reject alignment**

Use these rules against P3 baseline:

```text
P3 baseline action_match_rate = 0.8490457379467897
P3 baseline reject_reason_match_rate = 0.4512636238160963

If action_match_rate >= 0.8490457379467897 and reject_reason_match_rate > 0.4512636238160963:
  Plan B improved path and reject alignment
Else if action_match_rate >= 0.8490457379467897 and reject_reason_match_rate <= 0.4512636238160963:
  cadence is not the main reject mismatch; inspect latency guard/API path next
Else:
  audit-derived cadence changed action path negatively; inspect lag/tolerance and timestamp semantics before using Plan B as default
```

- [ ] **Step 3: Identify next bottleneck**

Use these rules:

```text
If cadence alignment is good but reject_reason_match_rate remains low:
  next bottleneck = latency model / guard timing
If cadence alignment is good and reject alignment improves but position MAE remains high:
  next bottleneck = fill path / order lifecycle / ORDER_TRADE_UPDATE gap
If cadence alignment is poor:
  next bottleneck = schedule/timestamp semantics, not strategy logic
```

Do not modify strategy behavior during this interpretation task.

---

## Task 9: Document Plan B artifacts and conclusion

**Files:**
- Modify: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/ARTIFACT_INDEX.md`
- Modify: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/live_summary.md`

- [ ] **Step 1: Add Plan B section to artifact index**

Append this section to `ARTIFACT_INDEX.md`, replacing the example values only with actual values from Tasks 6-8:

```markdown
## Plan B audit-derived cadence replay result

Status: **complete** / **blocked**.

- Cadence mode: `audit_replay`
- Live audit CSV: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/audit_live_two_phase_align_live_btcusdt_1777342116_20260428_134818.csv`
- Backtest audit CSV: `<PLAN_B_BT_AUDIT>`
- Alignment report JSON: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json`
- Scheduled live decisions: `<AUDIT_REPLAY_SCHEDULED_COUNT>`
- Consumed decisions: `<AUDIT_REPLAY_CONSUMED_COUNT>`
- Unconsumed decisions: `<AUDIT_REPLAY_UNCONSUMED_COUNT>`
- Backtest rows: `<BT_ROWS>`
- Live rows: `<LIVE_ROWS>`
- Common strategy sequence rows: `<COMMON_ROWS>`
- Action match rate: `<ACTION_MATCH_RATE>`
- Reject reason match rate: `<REJECT_REASON_MATCH_RATE>`
- Backtest latency guard/drop rate: `<BT_DROP_LATENCY_RATE>`
- Live latency guard/drop rate: `<LIVE_DROP_LATENCY_RATE>`
- Backtest API drop rate: `<BT_DROP_API_RATE>`
- Live API drop rate: `<LIVE_DROP_API_RATE>`
- Nearest live-to-backtest cadence lag p99 ns: `<NEAREST_LAG_ABS_NS_P99>`

Conclusion: `<one or two sentences: whether audit-derived cadence fixed row/cadence mismatch, whether action/reject alignment improved, and what bottleneck remains>`.
```

- [ ] **Step 2: Add concise Plan B summary to live summary**

Append this section to `live_summary.md`, replacing the example values only with actual values from Tasks 6-8:

```markdown
## Plan B audit-derived cadence replay

- Status: **complete** / **blocked**
- Cadence mode: `audit_replay`
- Scheduled/consumed/unconsumed decisions: `<SCHEDULED>` / `<CONSUMED>` / `<UNCONSUMED>`
- Backtest/live rows: `<BT_ROWS>` / `<LIVE_ROWS>`
- Action match rate: `<ACTION_MATCH_RATE>`
- Reject reason match rate: `<REJECT_REASON_MATCH_RATE>`
- Alignment report: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json`

Interpretation: `<one concise sentence identifying whether cadence is now aligned and which mismatch remains>`.
```

---

## Task 10: Final verification and report

**Files:**
- No code changes unless a previous verification failure identifies a specific bug.

- [ ] **Step 1: Run unit tests**

Run:

```bash
cd /home/molly/project/hftbacktest && pytest examples/binance_tick_mm/test_backtest_tick_mm.py examples/binance_tick_mm/test_compare_audit.py -q
```

Expected: all tests pass.

- [ ] **Step 2: Verify generated report exists and has cadence section**

Run:

```bash
cd /home/molly/project/hftbacktest && python - <<'PY'
import json
from pathlib import Path
p = Path('/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json')
r = json.loads(p.read_text())
assert 'cadence' in r
assert 'nearest_lag_live_to_bt' in r['cadence']
assert 'alignment' in r
print('ok', p)
PY
```

Expected output starts with:

```text
ok /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json
```

- [ ] **Step 3: Report final Plan B status**

Report exactly these bullets to the user:

```text
Plan B status: complete / blocked
Cadence mode: audit_replay
Backtest audit: <path>
Alignment report: /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_plan_b_audit_replay.json
Schedule counters:
- scheduled: <n>
- consumed: <n>
- unconsumed: <n>
Key metrics:
- bt rows: <n>
- live rows: <n>
- common rows: <n>
- action match rate: <value>
- reject reason match rate: <value>
- nearest lag p99 ns: <value>
Conclusion: <one sentence>
Next bottleneck: <cadence / latency / fill-order lifecycle>
```

## Caveats

1. Plan B must not change strategy quoting logic. It only changes when backtest decisions are evaluated.
2. Plan B is still using the old live run where `ORDER_TRADE_UPDATE` was not subscribed, so order lifecycle and fill-path conclusions remain diagnostic.
3. If cadence aligns but reject reasons remain poor, continue with latency guard/API timing alignment before touching strategy parameters.
4. If position/fill differences dominate after cadence improves, run Plan C first to collect a cleaner live audit with order lifecycle events.

## Verification

- `pytest examples/binance_tick_mm/test_backtest_tick_mm.py examples/binance_tick_mm/test_compare_audit.py -q` passes.
- Plan B backtest command exits successfully.
- Plan B result JSON contains `backtest_cadence_mode = "audit_replay"`.
- Plan B comparison JSON contains `cadence.nearest_lag_live_to_bt`.
- `ARTIFACT_INDEX.md` and `live_summary.md` contain Plan B artifact paths and interpretation.

## Self-review

- Spec coverage: The plan covers audit schedule loading, decision gating, result counters, comparison metrics, execution, interpretation, and documentation.
- Placeholder scan: The implementation steps contain exact code and commands. Result documentation templates contain explicit replacement markers because values are only known after execution.
- Type consistency: Cadence config fields are consistently named `mode`, `audit_csv`, `run_id`, `ts_column`, `tolerance_ms`, and internal nanosecond fields are `min_interval_ns` and `tolerance_ns`.
