# P3 Partial Backtest Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a partial backtest using the P2 live-collected market-data manifest and compare its audit metrics against the 2026-04-28 live audit.

**Architecture:** Reuse existing `examples/binance_tick_mm/backtest_tick_mm.py` for replay and existing `examples/binance_tick_mm/compare_audit.py` for first-pass audit comparison. Add one small local orchestration/report script only if manual commands prove repetitive; do not change strategy logic or backtest/live schemas unless a failing test exposes a real contract mismatch.

**Tech Stack:** Python, NumPy `.npz`, existing hftbacktest replay engine, TOML config, CSV audit output, JSON comparison report.

---

## Context

P2 is complete. The raw Binance collector gzip was converted to an hftbacktest-compatible NPZ and manifest:

- P2 manifest: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json`
- P2 NPZ: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/btcusdt_20260428.npz`
- NPZ rows: `29,491,708`
- Local timestamp coverage: `1777335329858658560` to `1777360821081938432`

Backtest config already exists and points to the live-derived order latency artifact:

- Config: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml`
- Live audit: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/audit_live_two_phase_align_live_btcusdt_1777342116_20260428_134818.csv`

Important update: `backtest_tick_mm.py` now supports absolute same-window slicing by `ts_local` bounds. For exact live-window P3 alignment, use `--window full_day` with `--slice-ts-local-start` and `--slice-ts-local-end`; do not use `first_5m`/`first_2h` for exact live-window alignment.

Same-window slice bounds for `live_btcusdt_1777342116`:

- start `ts_local`: `1777342117432305664`
- end `ts_local`: `1777347796789512192`
- expected P2 NPZ slice rows: `6,952,762`

Same-window command:

```bash
cd /home/molly/project/hftbacktest && HFTBACKTEST_USE_LOCAL_PY=1 python examples/binance_tick_mm/backtest_tick_mm.py \
  --config /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml \
  --manifest /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json \
  --window full_day \
  --slice-ts-local-start 1777342117432305664 \
  --slice-ts-local-end 1777347796789512192
```

Audit-derived cadence replay command:

```bash
conda run -n hftbacktest python examples/binance_tick_mm/backtest_tick_mm.py \
  --config /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml \
  --manifest /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json \
  --window full_day \
  --slice-ts-local-start 1777342117432305664 \
  --slice-ts-local-end 1777347796789512192
```

Audit-derived cadence config:

```toml
[backtest_cadence]
mode = "audit_replay"
enabled = true
audit_csv = "/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/audit_live_two_phase_align_live_btcusdt_1777342116_20260428_134818.csv"
run_id = "live_btcusdt_1777342116"
ts_column = "ts_local"
tolerance_ms = 0.0
```


## Files

- Use: `examples/binance_tick_mm/backtest_tick_mm.py`
  - Existing backtest runner.
  - Consumes manifest `data_files` and `initial_snapshot`.
  - Writes audit CSV to config `[paths].output_root` / `[audit].output_csv`.
- Use: `examples/binance_tick_mm/compare_audit.py`
  - Existing comparison report script.
- Use: `local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml`
  - Existing aligned config.
- Use: `local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json`
  - P2 manifest.
- Modify: `local_live_analysis/live_btcusdt_1777342116/ARTIFACT_INDEX.md`
  - Add P3 generated artifact paths and status.
- Modify: `local_live_analysis/live_btcusdt_1777342116/live_summary.md`
  - Add concise P3 result summary.
- Optional create: `local_live_analysis/live_btcusdt_1777342116/p3_alignment_notes.md`
  - Only if results need more interpretation than fits in the index/summary.

## Task 1: Validate P2 manifest and config paths before running P3

**Files:**
- Read: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json`
- Read: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml`

- [ ] **Step 1: Verify manifest data file exists**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
manifest = Path('/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json')
m = json.loads(manifest.read_text())
print('manifest', manifest)
for p in m['data_files']:
    path = Path(p)
    print('data_file', path, 'exists=', path.exists(), 'size=', path.stat().st_size if path.exists() else None)
print('initial_snapshot', m.get('initial_snapshot'))
PY
```

Expected: data file exists and has nonzero size; `initial_snapshot` is `None`.

- [ ] **Step 2: Verify config output and latency paths**

Run:

```bash
python - <<'PY'
import tomllib
from pathlib import Path
cfg = tomllib.loads(Path('/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml').read_text())
print('output_root', cfg['paths']['output_root'])
print('audit_csv', cfg['audit']['output_csv'])
print('latency_npz', cfg['latency']['order_latency_npz'], 'exists=', Path(cfg['latency']['order_latency_npz']).exists())
print('window', cfg['backtest']['window'])
PY
```

Expected: latency NPZ exists; output root is writable or creatable.

## Task 2: Run same-window P3 backtest

**Files:**
- Use: `examples/binance_tick_mm/backtest_tick_mm.py`
- Use: P2 manifest and aligned config.

- [ ] **Step 1: Verify same-window slice row count**

Run:

```bash
python - <<'PY'
import numpy as np
p = '/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/btcusdt_20260428.npz'
start = 1777342117432305664
end = 1777347796789512192
d = np.load(p)['data']
s = d[(d['local_ts'] >= start) & (d['local_ts'] <= end)]
print(len(s))
print(int(s['local_ts'][0]), int(s['local_ts'][-1]))
PY
```

Expected:

```text
6952762
1777342117447261440 1777347796767164928
```

- [ ] **Step 2: Run exact live-window replay**

Run:

```bash
cd /home/molly/project/hftbacktest && HFTBACKTEST_USE_LOCAL_PY=1 python examples/binance_tick_mm/backtest_tick_mm.py \
  --config /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/config_backtest_two_phase_align.toml \
  --manifest /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/out/live_raw/btcusdt/manifest_2026-04-28_to_2026-04-28.json \
  --window full_day \
  --slice-ts-local-start 1777342117432305664 \
  --slice-ts-local-end 1777347796789512192
```

Expected: command exits `0`, prints JSON with `audit_csv`, `audit_rows`, `rows`, `slice_ts_local_start`, and `slice_ts_local_end`; `rows > 0`.

- [ ] **Step 3: Record output paths**

From the printed JSON, capture:

```text
audit_csv
summary_json
daily_summary_csv
audit_rows
rows
slice_ts_local_start
slice_ts_local_end
```

Expected: `audit_csv` points under the config output root and exists.

## Task 3: Compare same-window backtest audit against live audit

**Files:**
- Use: `examples/binance_tick_mm/compare_audit.py`
- Use: generated backtest audit CSV.
- Use: live audit CSV.

- [ ] **Step 1: Run comparison report**

Replace `<BT_AUDIT>` with the `audit_csv` path printed by the chosen backtest run, preferably `first_2h` if it completed:

```bash
cd /home/molly/project/hftbacktest && python examples/binance_tick_mm/compare_audit.py \
  --bt <BT_AUDIT> \
  --live /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/audit_live_two_phase_align_live_btcusdt_1777342116_20260428_134818.csv \
  --out /home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_p3_live_raw.json
```

Expected: command exits `0` and writes `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_p3_live_raw.json`.

- [ ] **Step 2: Extract report metrics**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_p3_live_raw.json')
r = json.loads(p.read_text())
print('bt_rows', r['bt_summary']['rows'])
print('live_rows', r['live_summary']['rows'])
print('common_rows', r['alignment']['common_rows'])
print('action_match_rate', r['alignment']['action_match_rate'])
print('reject_reason_match_rate', r['alignment']['reject_reason_match_rate'])
print('mae', r['alignment']['mae'])
print('bt_drop_latency_rate', r['bt_summary']['drop_latency_rate'])
print('live_drop_latency_rate', r['live_summary']['drop_latency_rate'])
print('bt_spread_bps', r['bt_summary']['spread_bps'])
print('live_spread_bps', r['live_summary']['spread_bps'])
print('bt_vol_bps', r['bt_summary']['vol_bps'])
print('live_vol_bps', r['live_summary']['vol_bps'])
PY
```

Expected: report prints distribution and sequence-alignment metrics. If `common_rows == 0`, document that sequence-level comparison is invalid and use distribution metrics only.

## Task 4: Document P3 result locally

**Files:**
- Modify: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/ARTIFACT_INDEX.md`
- Modify: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/live_summary.md`

- [ ] **Step 1: Add P3 artifact paths to `ARTIFACT_INDEX.md`**

Append this section, filling values from actual command output:

```markdown
## P3 partial backtest alignment result

Status: **complete** / **blocked**.

- Backtest window used: `full_day` with absolute `ts_local` slicing
- Slice start `ts_local`: `1777342117432305664`
- Slice end `ts_local`: `1777347796789512192`
- Backtest audit CSV: `<BT_AUDIT>`
- Backtest summary JSON: `<SUMMARY_JSON>`
- Backtest daily summary CSV: `<DAILY_SUMMARY_CSV>`
- Alignment report JSON: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_p3_live_raw.json`
- Backtest rows: `<BT_ROWS>`
- Live rows: `<LIVE_ROWS>`
- Common strategy sequence rows: `<COMMON_ROWS>`
- Action match rate: `<ACTION_MATCH_RATE>`
- Reject reason match rate: `<REJECT_REASON_MATCH_RATE>`

Interpretation: `<one or two concise sentences explaining whether this is same-window-valid or distribution-only>`.
```

- [ ] **Step 2: Add concise P3 summary to `live_summary.md`**

Append this section, filling values from actual command output:

```markdown
## P3 partial backtest alignment

- Status: **complete** / **blocked**
- Window: `full_day` with absolute `ts_local` slicing
- Slice start `ts_local`: `1777342117432305664`
- Slice end `ts_local`: `1777347796789512192`
- Backtest audit: `<BT_AUDIT>`
- Alignment report: `/home/molly/project/hftbacktest/local_live_analysis/live_btcusdt_1777342116/alignment_report_p3_live_raw.json`
- Common rows: `<COMMON_ROWS>`
- Action match rate: `<ACTION_MATCH_RATE>`
- Reject reason match rate: `<REJECT_REASON_MATCH_RATE>`

Caveat: This run uses absolute `ts_local` bounds from the live audit, so market-data replay is aligned to the live bot window. Fill-path equivalence is still limited by the backtest exchange model and the live run's missing `ORDER_TRADE_UPDATE` stream.
```

## Task 5: Report final P3 status to user

**Files:**
- No file changes.

- [ ] **Step 1: Summarize P3 status**

Report exactly these bullets:

```text
P2 documented: yes
P3 status: complete / blocked
Backtest window: <window>
Backtest audit: <path>
Alignment report: <path>
Key metrics:
- bt rows: <n>
- live rows: <n>
- common rows: <n>
- action match rate: <value>
- reject reason match rate: <value>
Main caveat: <same-window caveat or blocker>
```

## Caveats

1. This plan does not change strategy behavior. If alignment is poor, that is a result to analyze, not a reason to patch strategy code during P3 execution.
2. `first_2h` means first two hours of P2 raw data, not necessarily first two hours of the live bot run.
3. The live run stopped on open-order alignment caveat; order lifecycle metrics remain diagnostic because `ORDER_TRADE_UPDATE` was not subscribed.
4. The backtest uses `.no_partial_fill_exchange()`, so exact fill-path equivalence is not expected.

## Verification

- P2 converter tests remain green:

```bash
cd /home/molly/project/hftbacktest && pytest examples/binance_tick_mm/test_pipeline_live_raw.py -q
```

- Backtest command exits successfully and writes audit CSV.
- `compare_audit.py` exits successfully and writes `alignment_report_p3_live_raw.json`.
- `ARTIFACT_INDEX.md` and `live_summary.md` contain the generated P3 paths.

## Self-review

- Spec coverage: P3 run, comparison, and documentation are covered.
- Placeholder scan: Implementation commands are concrete; only result values are explicitly filled after execution.
- Type consistency: Uses existing manifest, config, audit CSV, and comparison JSON schemas without changing them.
