# Continuous Snapshot Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make long-range Binance tick market-making backtests explicitly continuous, with one initial snapshot applied once, all full-day files consumed in manifest order, and summary metadata proving the run shape.

**Architecture:** Add small manifest validation and data-selection helpers inside `examples/binance_tick_mm/backtest_tick_mm.py`, then route `run_backtest()` through them. Add focused pytest tests in `examples/binance_tick_mm/test_backtest_tick_mm.py` for validation, file selection, snapshot application, and summary metadata without running expensive hftbacktest simulations.

**Tech Stack:** Python 3.11+, pytest, NumPy, existing `hftbacktest` Python bindings, existing Binance tick MM scripts.

---

## File Structure

- Modify: `examples/binance_tick_mm/backtest_tick_mm.py`
  - Add `_validate_manifest_paths()` to fail fast on bad manifests.
  - Add `_select_data_for_asset()` to make full-day continuous versus windowed behavior testable.
  - Add `_apply_initial_snapshot()` to make snapshot-once behavior testable.
  - Add `_continuous_run_metadata()` and include it in returned/written summary JSON.
- Create: `examples/binance_tick_mm/test_backtest_tick_mm.py`
  - Unit tests for validation, data-file selection, snapshot-once application, and summary metadata.
- No changes: `strategy_core.py`, `backtest_metrics.py`, `pipeline.py`
  - The pipeline already writes `initial_snapshot` and ordered `data_files`.
  - Strategy logic should remain unchanged.

---

### Task 1: Add manifest validation tests

**Files:**
- Create: `examples/binance_tick_mm/test_backtest_tick_mm.py`
- Modify later: `examples/binance_tick_mm/backtest_tick_mm.py`

- [ ] **Step 1: Write the failing tests**

Create `examples/binance_tick_mm/test_backtest_tick_mm.py` with this initial content:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from backtest_tick_mm import _validate_manifest_paths


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"placeholder")
    return str(path)


def test_validate_manifest_accepts_ordered_continuous_manifest(tmp_path: Path) -> None:
    snapshot = _touch(tmp_path / "snapshot_before_day1.npz")
    day1 = _touch(tmp_path / "btcusdt_20260101.npz")
    day2 = _touch(tmp_path / "btcusdt_20260102.npz")
    day3 = _touch(tmp_path / "btcusdt_20260103.npz")
    manifest = {
        "start_day": "2026-01-01",
        "end_day": "2026-01-03",
        "initial_snapshot": snapshot,
        "data_files": [day1, day2, day3],
    }

    data_files, initial_snapshot = _validate_manifest_paths(manifest)

    assert data_files == [day1, day2, day3]
    assert initial_snapshot == snapshot


def test_validate_manifest_rejects_empty_data_files() -> None:
    with pytest.raises(ValueError, match="manifest.data_files must contain at least one file"):
        _validate_manifest_paths({"data_files": []})


def test_validate_manifest_rejects_missing_data_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing_day.npz"
    manifest = {"data_files": [str(missing)]}

    with pytest.raises(FileNotFoundError, match="manifest.data_files does not exist"):
        _validate_manifest_paths(manifest)


def test_validate_manifest_rejects_missing_initial_snapshot(tmp_path: Path) -> None:
    data_file = _touch(tmp_path / "btcusdt_20260101.npz")
    missing_snapshot = tmp_path / "missing_snapshot.npz"
    manifest = {
        "initial_snapshot": str(missing_snapshot),
        "data_files": [data_file],
    }

    with pytest.raises(FileNotFoundError, match="manifest.initial_snapshot does not exist"):
        _validate_manifest_paths(manifest)


def test_validate_manifest_rejects_duplicate_data_files(tmp_path: Path) -> None:
    data_file = _touch(tmp_path / "btcusdt_20260101.npz")
    manifest = {"data_files": [data_file, data_file]}

    with pytest.raises(ValueError, match="manifest.data_files contains duplicate path"):
        _validate_manifest_paths(manifest)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: FAIL during import with:

```text
ImportError: cannot import name '_validate_manifest_paths'
```

- [ ] **Step 3: Commit is intentionally skipped**

Do not commit after a failing test. Commit only after the implementation passes.

---

### Task 2: Implement manifest validation

**Files:**
- Modify: `examples/binance_tick_mm/backtest_tick_mm.py:97-135`
- Test: `examples/binance_tick_mm/test_backtest_tick_mm.py`

- [ ] **Step 1: Add validation helper**

In `examples/binance_tick_mm/backtest_tick_mm.py`, add this function after `_load_manifest()`:

```python
def _validate_manifest_paths(manifest: dict[str, Any]) -> tuple[list[str], str | None]:
    raw_data_files = manifest.get("data_files")
    if not raw_data_files:
        raise ValueError("manifest.data_files must contain at least one file")

    data_files = [str(_expand(str(path))) for path in raw_data_files]
    seen: set[str] = set()
    for path in data_files:
        if path in seen:
            raise ValueError(f"manifest.data_files contains duplicate path: {path}")
        seen.add(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"manifest.data_files does not exist: {path}")

    raw_initial_snapshot = manifest.get("initial_snapshot")
    initial_snapshot = None
    if raw_initial_snapshot:
        initial_snapshot = str(_expand(str(raw_initial_snapshot)))
        if not Path(initial_snapshot).exists():
            raise FileNotFoundError(f"manifest.initial_snapshot does not exist: {initial_snapshot}")

    return data_files, initial_snapshot
```

- [ ] **Step 2: Run validation tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: PASS for the manifest validation tests.

- [ ] **Step 3: Commit**

Only if the user explicitly requested commits in this session, run:

```bash
git add examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/test_backtest_tick_mm.py
git commit -m "test: cover continuous backtest manifest validation"
```

Otherwise, do not commit.

---

### Task 3: Add data-selection tests

**Files:**
- Modify: `examples/binance_tick_mm/test_backtest_tick_mm.py`
- Modify later: `examples/binance_tick_mm/backtest_tick_mm.py`

- [ ] **Step 1: Extend the test imports**

Change the imports at the top of `examples/binance_tick_mm/test_backtest_tick_mm.py` to:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backtest_tick_mm import _select_data_for_asset, _validate_manifest_paths
```

- [ ] **Step 2: Add tiny structured data helper and selection tests**

Append these tests to `examples/binance_tick_mm/test_backtest_tick_mm.py`:

```python
def _write_data_npz(path: Path, local_ts_values: list[int]) -> str:
    dtype = np.dtype([
        ("ev", "i8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
    ])
    data = np.zeros(len(local_ts_values), dtype=dtype)
    data["local_ts"] = local_ts_values
    data["exch_ts"] = local_ts_values
    np.savez_compressed(path, data=data)
    return str(path)


def test_select_data_for_asset_full_day_uses_all_files_in_manifest_order(tmp_path: Path) -> None:
    day1 = _write_data_npz(tmp_path / "btcusdt_20260101.npz", [1, 2])
    day2 = _write_data_npz(tmp_path / "btcusdt_20260102.npz", [3, 4])
    day3 = _write_data_npz(tmp_path / "btcusdt_20260103.npz", [5, 6])

    data_for_asset = _select_data_for_asset([day1, day2, day3], "full_day")

    assert data_for_asset == [day1, day2, day3]


def test_select_data_for_asset_windowed_mode_slices_only_first_file(tmp_path: Path) -> None:
    day1 = _write_data_npz(
        tmp_path / "btcusdt_20260101.npz",
        [0, 1_000_000_000, 6 * 60 * 60 * 1_000_000_000 + 1],
    )
    day2 = _write_data_npz(tmp_path / "btcusdt_20260102.npz", [0])

    data_for_asset = _select_data_for_asset([day1, day2], "first_6h")

    assert len(data_for_asset) == 1
    sliced = data_for_asset[0]
    assert isinstance(sliced, np.ndarray)
    assert sliced["local_ts"].tolist() == [0, 1_000_000_000]
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: FAIL during import with:

```text
ImportError: cannot import name '_select_data_for_asset'
```

---

### Task 4: Implement data-selection helper and use manifest validation in run_backtest

**Files:**
- Modify: `examples/binance_tick_mm/backtest_tick_mm.py:135-173`
- Test: `examples/binance_tick_mm/test_backtest_tick_mm.py`

- [ ] **Step 1: Add data-selection helper**

In `examples/binance_tick_mm/backtest_tick_mm.py`, add this function after `_slice_data_by_window()`:

```python
def _select_data_for_asset(data_files: list[str], window: str) -> list[Any]:
    if window == "full_day":
        return data_files

    first_data = np.load(data_files[0])["data"]
    sliced = _slice_data_by_window(first_data, window)
    return [sliced]
```

- [ ] **Step 2: Update run_backtest to use validation and helper**

In `run_backtest()`, replace:

```python
    data_files = [str(_expand(p)) for p in manifest["data_files"]]
    initial_snapshot = manifest.get("initial_snapshot")
```

with:

```python
    data_files, initial_snapshot = _validate_manifest_paths(manifest)
```

Then replace:

```python
    data_for_asset: list[Any]
    if window == "full_day" and len(data_files) >= 1:
        data_for_asset = data_files
    else:
        first_data = np.load(data_files[0])["data"]
        sliced = _slice_data_by_window(first_data, window)
        data_for_asset = [sliced]
```

with:

```python
    data_for_asset = _select_data_for_asset(data_files, window)
```

- [ ] **Step 3: Run tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: PASS for manifest validation and data-selection tests.

- [ ] **Step 4: Commit**

Only if commits were explicitly requested, run:

```bash
git add examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/test_backtest_tick_mm.py
git commit -m "feat: validate continuous backtest manifest inputs"
```

Otherwise, do not commit.

---

### Task 5: Add snapshot-once tests

**Files:**
- Modify: `examples/binance_tick_mm/test_backtest_tick_mm.py`
- Modify later: `examples/binance_tick_mm/backtest_tick_mm.py`

- [ ] **Step 1: Extend imports**

Change the import from `backtest_tick_mm` to:

```python
from backtest_tick_mm import _apply_initial_snapshot, _select_data_for_asset, _validate_manifest_paths
```

- [ ] **Step 2: Add fake asset and tests**

Append this code to `examples/binance_tick_mm/test_backtest_tick_mm.py`:

```python
class FakeAsset:
    def __init__(self) -> None:
        self.snapshots: list[str] = []

    def initial_snapshot(self, path: str) -> None:
        self.snapshots.append(path)


def test_apply_initial_snapshot_calls_asset_once_when_snapshot_is_present(tmp_path: Path) -> None:
    snapshot = str(tmp_path / "snapshot_before_day1.npz")
    asset = FakeAsset()

    _apply_initial_snapshot(asset, snapshot)

    assert asset.snapshots == [snapshot]


def test_apply_initial_snapshot_does_not_call_asset_when_snapshot_is_absent() -> None:
    asset = FakeAsset()

    _apply_initial_snapshot(asset, None)

    assert asset.snapshots == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: FAIL during import with:

```text
ImportError: cannot import name '_apply_initial_snapshot'
```

---

### Task 6: Implement snapshot-once helper and use it

**Files:**
- Modify: `examples/binance_tick_mm/backtest_tick_mm.py:177-193`
- Test: `examples/binance_tick_mm/test_backtest_tick_mm.py`

- [ ] **Step 1: Add snapshot helper**

In `examples/binance_tick_mm/backtest_tick_mm.py`, add this function after `_select_data_for_asset()`:

```python
def _apply_initial_snapshot(asset: Any, initial_snapshot: str | None) -> None:
    if initial_snapshot:
        asset.initial_snapshot(initial_snapshot)
```

- [ ] **Step 2: Route existing asset setup through helper**

In `run_backtest()`, replace:

```python
    if initial_snapshot:
        asset.initial_snapshot(str(_expand(initial_snapshot)))
```

with:

```python
    _apply_initial_snapshot(asset, initial_snapshot)
```

- [ ] **Step 3: Run tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

Only if commits were explicitly requested, run:

```bash
git add examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/test_backtest_tick_mm.py
git commit -m "feat: make initial snapshot application explicit"
```

Otherwise, do not commit.

---

### Task 7: Add continuous-run metadata tests

**Files:**
- Modify: `examples/binance_tick_mm/test_backtest_tick_mm.py`
- Modify later: `examples/binance_tick_mm/backtest_tick_mm.py`

- [ ] **Step 1: Extend imports**

Change the import from `backtest_tick_mm` to:

```python
from backtest_tick_mm import (
    _apply_initial_snapshot,
    _continuous_run_metadata,
    _select_data_for_asset,
    _validate_manifest_paths,
)
```

- [ ] **Step 2: Add metadata tests**

Append these tests to `examples/binance_tick_mm/test_backtest_tick_mm.py`:

```python
def test_continuous_run_metadata_marks_multi_file_full_day_as_continuous() -> None:
    metadata = _continuous_run_metadata(
        window="full_day",
        data_files=["day1.npz", "day2.npz"],
        initial_snapshot="snapshot_before_day1.npz",
    )

    assert metadata == {
        "continuous_run": True,
        "initial_snapshot": "snapshot_before_day1.npz",
        "data_file_count": 2,
        "data_files": ["day1.npz", "day2.npz"],
    }


def test_continuous_run_metadata_marks_single_file_full_day_as_not_continuous() -> None:
    metadata = _continuous_run_metadata(
        window="full_day",
        data_files=["day1.npz"],
        initial_snapshot=None,
    )

    assert metadata["continuous_run"] is False
    assert metadata["data_file_count"] == 1
    assert metadata["initial_snapshot"] is None


def test_continuous_run_metadata_marks_windowed_multi_file_as_not_continuous() -> None:
    metadata = _continuous_run_metadata(
        window="first_6h",
        data_files=["day1.npz", "day2.npz"],
        initial_snapshot="snapshot_before_day1.npz",
    )

    assert metadata["continuous_run"] is False
    assert metadata["data_file_count"] == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: FAIL during import with:

```text
ImportError: cannot import name '_continuous_run_metadata'
```

---

### Task 8: Implement continuous-run metadata and include it in result JSON

**Files:**
- Modify: `examples/binance_tick_mm/backtest_tick_mm.py:135-459`
- Test: `examples/binance_tick_mm/test_backtest_tick_mm.py`

- [ ] **Step 1: Add metadata helper**

In `examples/binance_tick_mm/backtest_tick_mm.py`, add this function after `_apply_initial_snapshot()`:

```python
def _continuous_run_metadata(window: str, data_files: list[str], initial_snapshot: str | None) -> dict[str, Any]:
    return {
        "continuous_run": window == "full_day" and len(data_files) > 1,
        "initial_snapshot": initial_snapshot,
        "data_file_count": len(data_files),
        "data_files": data_files,
    }
```

- [ ] **Step 2: Include metadata in run_backtest result**

In `run_backtest()`, before building `result`, add:

```python
    continuous_metadata = _continuous_run_metadata(window, data_files, initial_snapshot)
```

Then change the `result` dict from:

```python
    result = {
        "run_id": run_id,
        "audit_csv": str(audit_path) if audit_policy.mode != "off" else "",
        "audit_rows": rows_written,
        "rows": summary["rows"],
        "summary": summary,
        "daily_summary_csv": str(daily_csv_path) if summary_enabled else "",
        "summary_json": str(summary_json_path) if summary_enabled else "",
    }
```

to:

```python
    result = {
        "run_id": run_id,
        "audit_csv": str(audit_path) if audit_policy.mode != "off" else "",
        "audit_rows": rows_written,
        "rows": summary["rows"],
        "summary": summary,
        "daily_summary_csv": str(daily_csv_path) if summary_enabled else "",
        "summary_json": str(summary_json_path) if summary_enabled else "",
        **continuous_metadata,
    }
```

- [ ] **Step 3: Run tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

Only if commits were explicitly requested, run:

```bash
git add examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/test_backtest_tick_mm.py
git commit -m "feat: record continuous backtest metadata"
```

Otherwise, do not commit.

---

### Task 9: Run focused validation and inspect diff

**Files:**
- Verify: `examples/binance_tick_mm/backtest_tick_mm.py`
- Verify: `examples/binance_tick_mm/test_backtest_tick_mm.py`

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
python -m pytest examples/binance_tick_mm/test_backtest_tick_mm.py -q
```

Expected:

```text
12 passed
```

The exact count may differ if additional tests are added, but all tests in this file must pass.

- [ ] **Step 2: Run syntax check**

Run:

```bash
python -m py_compile examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/test_backtest_tick_mm.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git diff -- examples/binance_tick_mm/backtest_tick_mm.py examples/binance_tick_mm/test_backtest_tick_mm.py
```

Expected:

- `backtest_tick_mm.py` only adds helpers, validation routing, snapshot helper routing, and result metadata.
- `test_backtest_tick_mm.py` contains focused unit tests.
- No strategy decision logic changed.

---

### Task 10: Optional slow two-day smoke test

**Files:**
- Runtime check only, no required code changes.

This task is optional because it depends on local converted data/snapshot availability and can be expensive.

- [ ] **Step 1: Identify an existing two-day manifest with an initial snapshot**

Use an existing manifest from the pipeline output if available. It must have:

```json
{
  "initial_snapshot": "/existing/snapshot.npz",
  "data_files": ["/existing/day1.npz", "/existing/day2.npz"]
}
```

- [ ] **Step 2: Run audit-off summary-on backtest**

Run with the existing config and manifest:

```bash
python examples/binance_tick_mm/backtest_tick_mm.py \
  --config /tmp/bt10days/tune_stage1/base_config_first5m.toml \
  --manifest /path/to/two_day_manifest.json \
  --window full_day
```

Expected:

- Command exits 0.
- JSON output contains:

```json
{
  "continuous_run": true,
  "data_file_count": 2
}
```

- [ ] **Step 3: Verify output files**

Check the configured `summary_json` and `daily_summary_csv` output paths.

Expected:

- `summary.json` contains `continuous_run: true`.
- `summary.json` contains `data_file_count: 2`.
- `daily_summary.csv` has one header row and two day rows.

---

## Self-Review

### Spec Coverage

- Continuous full-day multi-file behavior: Task 4.
- Initial snapshot applied once: Tasks 5-6.
- Manifest validation: Tasks 1-2.
- Summary metadata: Tasks 7-8.
- Existing windowed behavior unchanged: Tasks 3-4.
- Two-day slow smoke test: Task 10.

### Placeholder Scan

No `TBD`, `TODO`, or unspecified implementation steps remain. Optional Task 10 names the required manifest shape and exact command template because the path depends on local prepared data.

### Type Consistency

Helper names are consistent across tests and implementation:

- `_validate_manifest_paths(manifest) -> tuple[list[str], str | None]`
- `_select_data_for_asset(data_files, window) -> list[Any]`
- `_apply_initial_snapshot(asset, initial_snapshot) -> None`
- `_continuous_run_metadata(window, data_files, initial_snapshot) -> dict[str, Any]`
