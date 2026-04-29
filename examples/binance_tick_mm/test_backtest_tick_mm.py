from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backtest_tick_mm import (
    _apply_initial_snapshot,
    _audit_replay_decision_due,
    _backtest_cadence_config,
    _backtest_cadence_interval_ns,
    _continuous_run_metadata,
    _load_audit_cadence_schedule,
    _select_data_for_asset,
    _should_skip_strategy_decision,
    _slice_data_by_absolute_local_ts,
    _validate_manifest_paths,
)



class FakeAsset:
    def __init__(self) -> None:
        self.snapshots: list[str] = []

    def initial_snapshot(self, path: str) -> None:
        self.snapshots.append(path)


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



def test_slice_data_by_absolute_local_ts_is_inclusive() -> None:
    dtype = np.dtype([
        ("ev", "i8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
    ])
    data = np.zeros(4, dtype=dtype)
    data["local_ts"] = [100, 200, 300, 400]

    sliced = _slice_data_by_absolute_local_ts(data, 200, 300)

    assert sliced["local_ts"].tolist() == [200, 300]


def test_slice_data_by_absolute_local_ts_rejects_start_after_end() -> None:
    dtype = np.dtype([
        ("ev", "i8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
    ])
    data = np.zeros(1, dtype=dtype)
    data["local_ts"] = [100]

    with pytest.raises(ValueError, match="slice_ts_local_start must be <= slice_ts_local_end"):
        _slice_data_by_absolute_local_ts(data, 300, 200)


def test_select_data_for_asset_absolute_slice_rejects_multi_file_manifest(tmp_path: Path) -> None:
    day1 = _write_data_npz(tmp_path / "btcusdt_20260101.npz", [100, 200])
    day2 = _write_data_npz(tmp_path / "btcusdt_20260102.npz", [300, 400])

    with pytest.raises(ValueError, match="absolute ts_local slicing currently supports exactly one data file"):
        _select_data_for_asset([day1, day2], "full_day", slice_ts_local_start=100, slice_ts_local_end=400)


def test_select_data_for_asset_absolute_slice_rejects_named_relative_window(tmp_path: Path) -> None:
    day1 = _write_data_npz(tmp_path / "btcusdt_20260101.npz", [100, 200, 300])

    with pytest.raises(ValueError, match="absolute ts_local slicing requires window='full_day'"):
        _select_data_for_asset([day1], "first_5m", slice_ts_local_start=100, slice_ts_local_end=200)


def test_select_data_for_asset_absolute_slice_rejects_empty_result(tmp_path: Path) -> None:
    day1 = _write_data_npz(tmp_path / "btcusdt_20260101.npz", [100, 200, 300])

    with pytest.raises(ValueError, match="absolute ts_local slice selected zero rows"):
        _select_data_for_asset([day1], "full_day", slice_ts_local_start=500, slice_ts_local_end=600)


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


def test_load_audit_cadence_schedule_skips_fractional_nanoseconds(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,ts_local,action\nlive,100.5,keep\nlive,200.0,keep\n")

    schedule = _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")

    assert schedule == [200]


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


def test_load_audit_cadence_schedule_raises_for_empty_filter(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,ts_local,action\nother,100,keep\n")

    with pytest.raises(ValueError, match="no cadence timestamps loaded"):
        _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")


def test_load_audit_cadence_schedule_raises_for_missing_column(tmp_path: Path) -> None:
    audit = tmp_path / "audit.csv"
    audit.write_text("run_id,wrong_ts\nlive,100\n")

    with pytest.raises(KeyError, match="missing cadence timestamp column"):
        _load_audit_cadence_schedule(audit, run_id="live", ts_column="ts_local")


def test_backtest_cadence_config_defaults_to_fixed_interval() -> None:
    cfg = _backtest_cadence_config({})

    assert cfg == {
        "mode": "fixed_interval",
        "min_interval_ns": 0,
        "audit_csv": "",
        "run_id": "",
        "ts_column": "ts_local",
        "tolerance_ns": 0,
    }


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


def test_backtest_cadence_config_keeps_legacy_enabled_fixed_interval() -> None:
    config = {"backtest_cadence": {"enabled": True, "min_decision_interval_ms": 5.0}}

    cfg = _backtest_cadence_config(config)

    assert cfg["mode"] == "fixed_interval"
    assert cfg["min_interval_ns"] == 5_000_000




def test_backtest_cadence_interval_from_config_reads_ms() -> None:
    config = {"backtest_cadence": {"enabled": True, "min_decision_interval_ms": 12.5}}

    assert _backtest_cadence_interval_ns(config) == 12_500_000


def test_backtest_cadence_interval_from_config_disabled_returns_zero() -> None:
    config = {"backtest_cadence": {"enabled": False, "min_decision_interval_ms": 12.5}}

    assert _backtest_cadence_interval_ns(config) == 0


def test_should_skip_strategy_decision_is_disabled_for_zero_interval() -> None:
    assert _should_skip_strategy_decision(200, 100, 0) is False


def test_should_skip_strategy_decision_allows_first_decision() -> None:
    assert _should_skip_strategy_decision(100, None, 50) is False


def test_should_skip_strategy_decision_skips_until_interval_elapsed() -> None:
    assert _should_skip_strategy_decision(149, 100, 50) is True
    assert _should_skip_strategy_decision(150, 100, 50) is False


def test_apply_initial_snapshot_calls_asset_once_when_snapshot_is_present(tmp_path: Path) -> None:
    snapshot = str(tmp_path / "snapshot_before_day1.npz")
    asset = FakeAsset()

    _apply_initial_snapshot(asset, snapshot)

    assert asset.snapshots == [snapshot]


def test_apply_initial_snapshot_does_not_call_asset_when_snapshot_is_absent() -> None:
    asset = FakeAsset()

    _apply_initial_snapshot(asset, None)

    assert asset.snapshots == []


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
