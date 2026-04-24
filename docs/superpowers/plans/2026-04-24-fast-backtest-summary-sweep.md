# Fast Backtest Summary and Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add memory-safe summary metrics, configurable audit output modes, per-day summaries, and exact parameter-parallel sweeps for large-scale backtesting.

**Architecture:** Add a focused `backtest_metrics.py` module for streaming metrics and audit write decisions, wire it into `backtest_tick_mm.py`, then add a standalone `sweep_backtest.py` for multiprocessing parameter grids. Existing `audit.mode = full` behavior remains the default for backwards compatibility.

**Tech Stack:** Python 3.13, stdlib `csv/json/tomllib/multiprocessing`, hftbacktest 2.4.4, TOML configs

---

## File Structure

### Create

- `examples/binance_tick_mm/backtest_metrics.py` — streaming metric accumulators, latency histogram, audit mode policy, daily summary writer helpers.
- `examples/binance_tick_mm/sweep_backtest.py` — parameter-grid runner using process workers; each worker runs one exact sequential multi-day backtest.

### Modify

- `examples/binance_tick_mm/backtest_tick_mm.py` — use audit modes, update metrics during loop, write summary JSON and daily CSV.
- `examples/binance_tick_mm/config.example.toml` — add audit mode/sample config and `[summary]` section.
- `examples/binance_tick_mm/walk_forward.py` — stop reading huge audit CSV when `run_backtest` returns summary metrics.
- `examples/binance_tick_mm/README.md` — document fast backtest modes and sweeps.

---

## Task 1: Add streaming metrics module

**Files:**
- Create: `examples/binance_tick_mm/backtest_metrics.py`

- [ ] **Step 1: Create `backtest_metrics.py`**

Create the file with this content:

```python
#!/usr/bin/env python3
"""Streaming metrics and audit-output helpers for binance_tick_mm backtests."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_AUDIT_MODES = {"full", "actions_only", "sampled", "off"}


@dataclass
class AuditPolicy:
    mode: str = "full"
    sample_every: int = 1000

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AuditPolicy":
        mode = str(cfg.get("mode", "full"))
        sample_every = int(cfg.get("sample_every", 1000))
        if mode not in VALID_AUDIT_MODES:
            raise ValueError(f"Unsupported audit.mode={mode!r}; expected one of {sorted(VALID_AUDIT_MODES)}")
        if mode == "sampled" and sample_every <= 0:
            raise ValueError("audit.sample_every must be > 0 when audit.mode='sampled'")
        return cls(mode=mode, sample_every=sample_every)

    def should_write(self, row: dict[str, Any], seq: int) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "full":
            return True

        action = str(row.get("action", "keep"))
        reject_reason = str(row.get("reject_reason", ""))
        is_action_or_reject = action != "keep" or bool(reject_reason)

        if self.mode == "actions_only":
            return is_action_or_reject
        if self.mode == "sampled":
            return is_action_or_reject or (seq % self.sample_every == 0)
        raise AssertionError(f"unreachable audit mode: {self.mode}")


@dataclass
class LatencyHistogram:
    """Fixed-memory latency histogram in milliseconds."""

    max_ms: float = 100.0
    bucket_ms: float = 0.1
    buckets: list[int] = field(default_factory=list)
    count: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        if not self.buckets:
            n = int(self.max_ms / self.bucket_ms) + 1
            self.buckets = [0] * n

    def add(self, value_ms: float) -> None:
        if not math.isfinite(value_ms) or value_ms < 0:
            return
        idx = int(value_ms / self.bucket_ms)
        if idx >= len(self.buckets):
            idx = len(self.buckets) - 1
        self.buckets[idx] += 1
        self.count += 1
        self.total += value_ms

    def percentile(self, q: float) -> float:
        if self.count == 0:
            return 0.0
        target = max(1, int(math.ceil(self.count * q)))
        seen = 0
        for idx, n in enumerate(self.buckets):
            seen += n
            if seen >= target:
                return idx * self.bucket_ms
        return self.max_ms

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
        return {
            "count": float(self.count),
            "mean": self.total / self.count,
            "p50": self.percentile(0.50),
            "p90": self.percentile(0.90),
            "p99": self.percentile(0.99),
        }


@dataclass
class MetricAccumulator:
    rows: int = 0
    pnl_mtm: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown_mtm: float = 0.0
    prev_mid: float | None = None
    prev_position: float | None = None
    inventory_score_sum: float = 0.0
    spread_bps_sum: float = 0.0
    vol_bps_sum: float = 0.0
    abs_position_notional_sum: float = 0.0
    max_abs_position_notional: float = 0.0
    drop_latency_count: int = 0
    drop_api_count: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    reject_counts: Counter[str] = field(default_factory=Counter)
    feed_latency_ms: LatencyHistogram = field(default_factory=LatencyHistogram)
    entry_latency_ms: LatencyHistogram = field(default_factory=LatencyHistogram)
    latency_signal_ms: LatencyHistogram = field(default_factory=LatencyHistogram)

    def update(self, row: dict[str, Any]) -> None:
        mid = float(row["mid"])
        position = float(row["position"])

        if self.prev_mid is not None and self.prev_position is not None:
            self.pnl_mtm += self.prev_position * (mid - self.prev_mid)
            if self.pnl_mtm > self.peak_pnl:
                self.peak_pnl = self.pnl_mtm
            drawdown = self.peak_pnl - self.pnl_mtm
            if drawdown > self.max_drawdown_mtm:
                self.max_drawdown_mtm = drawdown

        self.prev_mid = mid
        self.prev_position = position
        self.rows += 1

        self.inventory_score_sum += float(row["inventory_score"])
        self.spread_bps_sum += float(row["spread_bps"])
        self.vol_bps_sum += float(row["vol_bps"])

        abs_notional = abs(position * mid)
        self.abs_position_notional_sum += abs_notional
        if abs_notional > self.max_abs_position_notional:
            self.max_abs_position_notional = abs_notional

        if int(row["dropped_by_latency"]) > 0:
            self.drop_latency_count += 1
        if int(row["dropped_by_api_limit"]) > 0:
            self.drop_api_count += 1

        self.action_counts[str(row["action"])] += 1
        reject_reason = str(row.get("reject_reason", ""))
        if reject_reason:
            self.reject_counts[reject_reason] += 1

        self.feed_latency_ms.add(float(row["feed_latency_ns"]) / 1_000_000.0)
        self.entry_latency_ms.add(float(row["entry_latency_ns"]) / 1_000_000.0)
        self.latency_signal_ms.add(float(row["latency_signal_ms"]))

    def summary(self) -> dict[str, Any]:
        n = float(self.rows)
        if self.rows == 0:
            return {
                "rows": 0,
                "pnl_mtm": 0.0,
                "max_drawdown_mtm": 0.0,
                "avg_inventory_score": 0.0,
                "avg_abs_position_notional": 0.0,
                "max_abs_position_notional": 0.0,
                "avg_spread_bps": 0.0,
                "avg_vol_bps": 0.0,
                "drop_latency_rate": 0.0,
                "drop_api_rate": 0.0,
                "actions": {},
                "reject_reasons": {},
                "feed_latency_ms": self.feed_latency_ms.summary(),
                "entry_latency_ms": self.entry_latency_ms.summary(),
                "latency_signal_ms": self.latency_signal_ms.summary(),
            }
        return {
            "rows": self.rows,
            "pnl_mtm": self.pnl_mtm,
            "max_drawdown_mtm": self.max_drawdown_mtm,
            "avg_inventory_score": self.inventory_score_sum / n,
            "avg_abs_position_notional": self.abs_position_notional_sum / n,
            "max_abs_position_notional": self.max_abs_position_notional,
            "avg_spread_bps": self.spread_bps_sum / n,
            "avg_vol_bps": self.vol_bps_sum / n,
            "drop_latency_rate": self.drop_latency_count / n,
            "drop_api_rate": self.drop_api_count / n,
            "actions": dict(self.action_counts),
            "reject_reasons": dict(self.reject_counts),
            "feed_latency_ms": self.feed_latency_ms.summary(),
            "entry_latency_ms": self.entry_latency_ms.summary(),
            "latency_signal_ms": self.latency_signal_ms.summary(),
        }


def utc_day_from_ns(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def flatten_summary(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in summary.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, dict):
                    for leaf_key, leaf_val in sub_val.items():
                        out[f"{name}_{sub_key}_{leaf_key}"] = leaf_val
                else:
                    out[f"{name}_{sub_key}"] = sub_val
        else:
            out[name] = value
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def write_daily_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
```

- [ ] **Step 2: Run syntax check**

Run:

```bash
python -m py_compile examples/binance_tick_mm/backtest_metrics.py
```

Expected: exits 0.

- [ ] **Step 3: Commit**

```bash
git add examples/binance_tick_mm/backtest_metrics.py
git commit -m "feat: add streaming backtest metrics helpers"
```

---

## Task 2: Wire audit modes and summaries into backtest_tick_mm.py

**Files:**
- Modify: `examples/binance_tick_mm/backtest_tick_mm.py`

- [ ] **Step 1: Add imports**

Add this import block after existing `strategy_core` imports:

```python
from backtest_metrics import (
    AuditPolicy,
    MetricAccumulator,
    flatten_summary,
    utc_day_from_ns,
    write_daily_csv,
    write_json,
)
```

- [ ] **Step 2: Add config parsing before opening audit output**

Inside `run_backtest`, after `audit_path = output_root / audit_name`, add:

```python
    audit_cfg = config.get("audit", {})
    audit_policy = AuditPolicy.from_config(audit_cfg)
    summary_cfg = config.get("summary", {})
    summary_enabled = bool(summary_cfg.get("enabled", False))
    summary_json_path = output_root / str(summary_cfg.get("output_json", "summary.json"))
    daily_csv_path = output_root / str(summary_cfg.get("daily_csv", "daily_summary.csv"))
```

- [ ] **Step 3: Replace the audit file context manager**

Replace:

```python
    rows_written = 0
    with audit_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()

        while True:
            ...
```

with:

```python
    rows_written = 0
    audit_file = None
    writer = None
    if audit_policy.mode != "off":
        audit_file = audit_path.open("w", newline="")
        writer = csv.DictWriter(audit_file, fieldnames=AUDIT_FIELDS)
        writer.writeheader()

    metrics = MetricAccumulator()
    day_metrics = MetricAccumulator()
    current_day: str | None = None
    daily_rows: list[dict[str, Any]] = []

    try:
        while True:
            ...
    finally:
        if current_day is not None and day_metrics.rows > 0:
            daily_rows.append({"day": current_day, **flatten_summary("", day_metrics.summary())})
        if audit_file is not None:
            audit_file.flush()
            audit_file.close()
```

Keep the loop body indentation correct.

- [ ] **Step 4: Update row writing logic**

Replace:

```python
            writer.writerow(row)
            rows_written += 1

            if rows_written % int(config["audit"].get("flush_every", 1000)) == 0:
                f.flush()
```

with:

```python
            metrics.update(row)

            row_day = utc_day_from_ns(ts_local)
            if current_day is None:
                current_day = row_day
            elif row_day != current_day:
                daily_rows.append({"day": current_day, **flatten_summary("", day_metrics.summary())})
                day_metrics = MetricAccumulator()
                current_day = row_day
            day_metrics.update(row)

            if writer is not None and audit_policy.should_write(row, strategy_seq):
                writer.writerow(row)
                rows_written += 1
                if rows_written % int(audit_cfg.get("flush_every", 1000)) == 0 and audit_file is not None:
                    audit_file.flush()
```

- [ ] **Step 5: Update return payload and summary file writing**

Replace the final return block:

```python
    return {
        "run_id": run_id,
        "audit_csv": str(audit_path),
        "rows": rows_written,
    }
```

with:

```python
    summary = metrics.summary()
    result = {
        "run_id": run_id,
        "audit_csv": str(audit_path) if audit_policy.mode != "off" else "",
        "audit_rows": rows_written,
        "rows": summary["rows"],
        "summary": summary,
        "daily_summary_csv": str(daily_csv_path) if summary_enabled else "",
        "summary_json": str(summary_json_path) if summary_enabled else "",
    }

    if summary_enabled:
        write_daily_csv(daily_csv_path, daily_rows)
        write_json(summary_json_path, result)

    return result
```

- [ ] **Step 6: Run syntax check**

Run:

```bash
python -m py_compile examples/binance_tick_mm/backtest_tick_mm.py
```

Expected: exits 0.

- [ ] **Step 7: Commit**

```bash
git add examples/binance_tick_mm/backtest_tick_mm.py
git commit -m "feat: add audit modes and streaming summaries to backtest"
```

---

## Task 3: Add config defaults

**Files:**
- Modify: `examples/binance_tick_mm/config.example.toml`

- [ ] **Step 1: Extend `[audit]` and add `[summary]`**

Change the `[audit]` section from:

```toml
[audit]
output_csv = "audit_bt.csv"
flush_every = 1000
```

to:

```toml
[audit]
# full | actions_only | sampled | off
mode = "full"
# Used only when mode = "sampled". Action/reject rows are always written.
sample_every = 1000
output_csv = "audit_bt.csv"
flush_every = 1000

[summary]
enabled = true
output_json = "summary.json"
daily_csv = "daily_summary.csv"
```

- [ ] **Step 2: Validate TOML**

Run:

```bash
python -c "import tomllib; tomllib.load(open('examples/binance_tick_mm/config.example.toml','rb')); print('TOML OK')"
```

Expected: `TOML OK`

- [ ] **Step 3: Commit**

```bash
git add examples/binance_tick_mm/config.example.toml
git commit -m "feat: add fast backtest audit and summary config"
```

---

## Task 4: Add sweep_backtest.py

**Files:**
- Create: `examples/binance_tick_mm/sweep_backtest.py`

- [ ] **Step 1: Create sweep_backtest.py**

Create the file with this content:

```python
#!/usr/bin/env python3
"""Parameter-parallel exact backtest sweeps for binance_tick_mm."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import os
import tomllib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from backtest_metrics import flatten_summary
from backtest_tick_mm import _expand, _load_manifest, _load_toml, run_backtest


def _set_nested(cfg: dict[str, Any], section: str, key: str, value: Any) -> None:
    if section not in cfg or not isinstance(cfg[section], dict):
        cfg[section] = {}
    cfg[section][key] = value


def _grid_items(grid: dict[str, Any]) -> list[tuple[str, str, list[Any]]]:
    items: list[tuple[str, str, list[Any]]] = []
    for section, values in grid.items():
        if not isinstance(values, dict):
            raise ValueError(f"Grid section {section!r} must be a table")
        for key, raw_values in values.items():
            if not isinstance(raw_values, list):
                raise ValueError(f"Grid value {section}.{key} must be an array")
            if not raw_values:
                raise ValueError(f"Grid value {section}.{key} must not be empty")
            items.append((section, key, raw_values))
    return items


def _param_combinations(grid: dict[str, Any]) -> list[dict[str, Any]]:
    items = _grid_items(grid)
    combos: list[dict[str, Any]] = []
    for values in itertools.product(*(item[2] for item in items)):
        combo: dict[str, Any] = {}
        for (section, key, _), value in zip(items, values, strict=True):
            combo[f"{section}.{key}"] = value
        combos.append(combo)
    return combos


def _write_toml_scalar(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _write_config(path: Path, cfg: dict[str, Any]) -> None:
    lines: list[str] = []
    for section, values in cfg.items():
        if not isinstance(values, dict):
            continue
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, dict):
                continue
            lines.append(f"{key} = {_write_toml_scalar(value)}")
        lines.append("")
    path.write_text("\n".join(lines))


def _run_one(args: tuple[int, dict[str, Any], dict[str, Any], dict[str, Any], str | None, str]) -> dict[str, Any]:
    run_id, base_cfg, manifest, params, window, out_dir_raw = args
    out_dir = Path(out_dir_raw) / f"run_{run_id:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = copy.deepcopy(base_cfg)
    for dotted, value in params.items():
        section, key = dotted.split(".", 1)
        _set_nested(cfg, section, key, value)

    cfg.setdefault("audit", {})["mode"] = str(cfg.get("audit", {}).get("mode", "off"))
    cfg["audit"]["mode"] = "off"
    cfg.setdefault("summary", {})["enabled"] = True
    cfg["summary"]["output_json"] = "summary.json"
    cfg["summary"]["daily_csv"] = "daily_summary.csv"
    cfg.setdefault("paths", {})["output_root"] = str(out_dir)

    config_path = out_dir / "config.toml"
    _write_config(config_path, cfg)

    try:
        result = run_backtest(cfg, manifest, window_override=window)
        summary = result["summary"]
        flat = flatten_summary("", summary)
        return {
            "run_id": run_id,
            "status": "ok",
            "run_dir": str(out_dir),
            **params,
            **flat,
        }
    except Exception as exc:
        return {
            "run_id": run_id,
            "status": "error",
            "run_dir": str(out_dir),
            "error": repr(exc),
            **params,
        }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter-parallel exact backtest sweeps")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--window", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fail-fast", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = _load_toml(_expand(args.base_config))
    manifest = _load_manifest(_expand(args.manifest))
    grid = _load_toml(_expand(args.grid))
    combos = _param_combinations(grid)

    out_dir = _expand(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [(i + 1, base_cfg, manifest, params, args.window, str(out_dir)) for i, params in enumerate(combos)]
    rows: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=int(args.workers)) as executor:
        futures = [executor.submit(_run_one, job) for job in jobs]
        for fut in as_completed(futures):
            row = fut.result()
            rows.append(row)
            print(json.dumps({"run_id": row["run_id"], "status": row["status"]}, ensure_ascii=True))
            if args.fail_fast and row["status"] != "ok":
                raise RuntimeError(f"Sweep run failed: {row}")

    rows.sort(key=lambda r: int(r["run_id"]))
    _write_rows(out_dir / "sweep_summary.csv", rows)
    (out_dir / "sweep_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=True))
    print(json.dumps({"runs": len(rows), "out": str(out_dir)}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run syntax check**

Run:

```bash
python -m py_compile examples/binance_tick_mm/sweep_backtest.py
```

Expected: exits 0.

- [ ] **Step 3: Commit**

```bash
git add examples/binance_tick_mm/sweep_backtest.py
git commit -m "feat: add parameter-parallel sweep runner"
```

---

## Task 5: Update walk_forward.py to use returned summaries

**Files:**
- Modify: `examples/binance_tick_mm/walk_forward.py`

- [ ] **Step 1: Add helper to prefer returned summary**

After `_audit_metrics`, add:

```python
def _result_metrics(result: dict[str, Any]) -> dict[str, float]:
    summary = result.get("summary")
    if isinstance(summary, dict):
        return {
            "rows": float(summary.get("rows", 0.0)),
            "pnl_mtm": float(summary.get("pnl_mtm", 0.0)),
            "max_drawdown_mtm": float(summary.get("max_drawdown_mtm", 0.0)),
            "drop_latency_rate": float(summary.get("drop_latency_rate", 0.0)),
            "drop_api_rate": float(summary.get("drop_api_rate", 0.0)),
            "avg_inventory_score": float(summary.get("avg_inventory_score", 0.0)),
            "avg_spread_bps": float(summary.get("avg_spread_bps", 0.0)),
            "avg_vol_bps": float(summary.get("avg_vol_bps", 0.0)),
            "avg_abs_position_notional": float(summary.get("avg_abs_position_notional", 0.0)),
        }
    audit_csv = str(result.get("audit_csv", ""))
    if not audit_csv:
        return {
            "rows": 0.0,
            "pnl_mtm": 0.0,
            "max_drawdown_mtm": 0.0,
            "drop_latency_rate": 0.0,
            "drop_api_rate": 0.0,
            "avg_inventory_score": 0.0,
            "avg_spread_bps": 0.0,
            "avg_vol_bps": 0.0,
            "avg_abs_position_notional": 0.0,
        }
    return _audit_metrics(Path(audit_csv))
```

- [ ] **Step 2: Replace metrics calls**

Replace:

```python
        train_metrics = _audit_metrics(Path(train_result["audit_csv"]))
```

with:

```python
        train_metrics = _result_metrics(train_result)
```

Replace:

```python
        test_metrics = _audit_metrics(Path(test_result["audit_csv"]))
```

with:

```python
        test_metrics = _result_metrics(test_result)
```

- [ ] **Step 3: Run syntax check**

Run:

```bash
python -m py_compile examples/binance_tick_mm/walk_forward.py
```

Expected: exits 0.

- [ ] **Step 4: Commit**

```bash
git add examples/binance_tick_mm/walk_forward.py
git commit -m "feat: use streaming backtest summaries in walk_forward"
```

---

## Task 6: Verification tests with existing 10-day data

**Files:**
- No code changes unless failures require fixes.

- [ ] **Step 1: Create a small manifest for one NPZ day**

Run:

```bash
python - <<'PY'
import json
manifest = {
  "start_day": "2026-04-01",
  "end_day": "2026-04-01",
  "initial_snapshot": None,
  "data_files": ["/tmp/bt10days/BTCUSDT_20260401.npz"]
}
open('/tmp/bt10days/manifest_1day.json','w').write(json.dumps(manifest, indent=2))
PY
```

Expected: `/tmp/bt10days/manifest_1day.json` exists.

- [ ] **Step 2: Create fast config with audit off**

Run:

```bash
python - <<'PY'
import tomllib
cfg = tomllib.load(open('/tmp/config_calibration.toml','rb'))
cfg['paths']['output_root'] = '/tmp/bt10days/verify_off'
cfg['backtest']['window'] = 'first_5m'
cfg['latency']['order_latency_npz'] = '/tmp/bt10days/live_order_latency_clean.npz'
cfg['audit']['mode'] = 'off'
cfg['summary'] = {'enabled': True, 'output_json': 'summary.json', 'daily_csv': 'daily_summary.csv'}

def scalar(v):
    if isinstance(v, str): return '"' + v.replace('"','\\"') + '"'
    if isinstance(v, bool): return 'true' if v else 'false'
    return str(v)
lines=[]
for section, values in cfg.items():
    if isinstance(values, dict):
        lines.append(f'[{section}]')
        for k,v in values.items():
            if not isinstance(v, dict): lines.append(f'{k} = {scalar(v)}')
        lines.append('')
open('/tmp/bt10days/config_verify_off.toml','w').write('\n'.join(lines))
PY
```

Expected: config file exists.

- [ ] **Step 3: Run audit-off verification**

Run:

```bash
cd examples/binance_tick_mm
python backtest_tick_mm.py \
  --config /tmp/bt10days/config_verify_off.toml \
  --manifest /tmp/bt10days/manifest_1day.json \
  --window first_5m
```

Expected:

- JSON output contains `"audit_csv": ""`
- `/tmp/bt10days/verify_off/summary.json` exists
- `/tmp/bt10days/verify_off/daily_summary.csv` exists
- no audit CSV exists in `/tmp/bt10days/verify_off`

- [ ] **Step 4: Create sampled config**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('/tmp/bt10days/config_verify_off.toml').read_text()
text = text.replace('output_root = "/tmp/bt10days/verify_off"', 'output_root = "/tmp/bt10days/verify_sampled"')
text = text.replace('mode = "off"', 'mode = "sampled"\nsample_every = 100')
Path('/tmp/bt10days/config_verify_sampled.toml').write_text(text)
PY
```

- [ ] **Step 5: Run sampled verification**

Run:

```bash
cd examples/binance_tick_mm
python backtest_tick_mm.py \
  --config /tmp/bt10days/config_verify_sampled.toml \
  --manifest /tmp/bt10days/manifest_1day.json \
  --window first_5m
wc -l /tmp/bt10days/verify_sampled/audit_bt.csv
```

Expected: audit CSV exists and row count is far smaller than summary rows.

- [ ] **Step 6: Create tiny sweep grid**

Run:

```bash
cat > /tmp/bt10days/grid_tiny.toml <<'TOML'
[risk]
base_spread = [0.5, 1.0]
TOML
```

- [ ] **Step 7: Run tiny sweep**

Run:

```bash
cd examples/binance_tick_mm
python sweep_backtest.py \
  --base-config /tmp/bt10days/config_verify_off.toml \
  --manifest /tmp/bt10days/manifest_1day.json \
  --grid /tmp/bt10days/grid_tiny.toml \
  --workers 2 \
  --window first_5m \
  --out /tmp/bt10days/sweep_tiny
```

Expected:

- `/tmp/bt10days/sweep_tiny/run_0001/summary.json` exists
- `/tmp/bt10days/sweep_tiny/run_0002/summary.json` exists
- `/tmp/bt10days/sweep_tiny/sweep_summary.csv` has 2 data rows

---

## Task 7: 10-day performance benchmark

**Files:**
- No code changes unless failures require fixes.

- [ ] **Step 1: Create 10-day manifest**

Run:

```bash
python - <<'PY'
import json
files = [f'/tmp/bt10days/BTCUSDT_202604{d:02d}.npz' for d in range(1, 11)]
manifest = {"start_day":"2026-04-01", "end_day":"2026-04-10", "initial_snapshot":None, "data_files":files}
open('/tmp/bt10days/manifest_10day.json','w').write(json.dumps(manifest, indent=2))
PY
```

- [ ] **Step 2: Create 10-day audit-off config**

Run:

```bash
python - <<'PY'
from pathlib import Path
text = Path('/tmp/bt10days/config_verify_off.toml').read_text()
text = text.replace('output_root = "/tmp/bt10days/verify_off"', 'output_root = "/tmp/bt10days/bench_10day_off"')
text = text.replace('window = "first_5m"', 'window = "full_day"')
Path('/tmp/bt10days/config_10day_off.toml').write_text(text)
PY
```

- [ ] **Step 3: Run 10-day benchmark**

Run:

```bash
cd examples/binance_tick_mm
/usr/bin/time -p python backtest_tick_mm.py \
  --config /tmp/bt10days/config_10day_off.toml \
  --manifest /tmp/bt10days/manifest_10day.json \
  --window full_day
```

Expected:

- no large audit CSV is created
- `/tmp/bt10days/bench_10day_off/summary.json` exists
- `/tmp/bt10days/bench_10day_off/daily_summary.csv` exists
- runtime is lower than or comparable to previous 1532.3 seconds, with near-zero audit disk output

---

## Task 8: Documentation update

**Files:**
- Modify: `examples/binance_tick_mm/README.md`

- [ ] **Step 1: Add fast backtest section**

Add a section documenting:

```markdown
## 大规模回测：summary-only 与参数并行

### 避免写出完整 audit

[audit]
mode = "off"          # full | actions_only | sampled | off
sample_every = 1000

[summary]
enabled = true
output_json = "summary.json"
daily_csv = "daily_summary.csv"

### 参数并行 sweep

python sweep_backtest.py \
  --base-config config.toml \
  --manifest manifest.json \
  --grid sweep.toml \
  --workers 8 \
  --window full_day \
  --out ./out/sweeps

### sweep.toml 示例

[risk]
k_inv = [0.0005, 0.001, 0.002]
base_spread = [0.5, 1.0, 1.5]
k_pos = [0.00005, 0.0001, 0.0002]

[fair]
w_imb = [0.0, 0.1, 0.25, 0.5]
```

Also mention the measured baseline:

- 10-day full audit: 1532.3 sec, 22GB CSV
- expected large-scale mode: summary JSON + daily CSV only

- [ ] **Step 2: Commit**

```bash
git add examples/binance_tick_mm/README.md
git commit -m "docs: document fast backtest summaries and sweeps"
```

---

## Self-Review Checklist

- Spec coverage: Tasks 1-3 implement audit modes, summary JSON, daily CSV. Task 4 implements parameter-parallel sweep. Tasks 6-7 verify behavior and performance. Task 8 documents it.
- Placeholder scan: no TBD/TODO placeholders; all commands and code are specified.
- Type consistency: `AuditPolicy`, `MetricAccumulator`, `LatencyHistogram`, `flatten_summary`, `write_daily_csv`, and `write_json` are defined before use.
