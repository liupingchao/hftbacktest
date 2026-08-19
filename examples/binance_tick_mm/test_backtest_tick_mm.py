from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from backtest_tick_mm import (
    _apply_initial_snapshot,
    _continuous_run_metadata,
    _select_data_for_asset,
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
