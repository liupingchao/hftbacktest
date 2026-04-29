from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np

from pipeline_live_raw import build_live_raw_manifest, convert_live_raw_file


def _write_gz(path: Path, rows: list[tuple[int, dict]]) -> None:
    with gzip.open(path, "wt") as f:
        for ts, msg in rows:
            f.write(f"{ts} {json.dumps(msg, separators=(',', ':'))}\n")


def _sample_rows() -> list[tuple[int, dict]]:
    return [
        (
            1777335329819270838,
            {
                "stream": "btcusdt@depth@0ms",
                "data": {
                    "e": "depthUpdate",
                    "E": 1777335329819,
                    "T": 1777335329818,
                    "s": "BTCUSDT",
                    "U": 1,
                    "u": 2,
                    "pu": 0,
                    "b": [["77299.90", "0.124"]],
                    "a": [["77300.00", "8.900"]],
                },
            },
        ),
        (
            1777335329829270838,
            {
                "stream": "btcusdt@trade",
                "data": {
                    "e": "trade",
                    "E": 1777335329820,
                    "T": 1777335329819,
                    "s": "BTCUSDT",
                    "t": 123,
                    "p": "77300.00",
                    "q": "0.001",
                    "X": "MARKET",
                    "m": False,
                },
            },
        ),
    ]


def test_convert_live_raw_file_writes_npz_with_data_key(tmp_path: Path) -> None:
    raw = tmp_path / "btcusdt_20260428.gz"
    out = tmp_path / "btcusdt_20260428.npz"
    _write_gz(raw, _sample_rows())

    data = convert_live_raw_file(raw, out, buffer_size=16)

    assert out.exists()
    loaded = np.load(out)
    assert "data" in loaded.files
    assert len(loaded["data"]) == len(data)
    assert len(data) == 3
    assert data.dtype.names == ("ev", "exch_ts", "local_ts", "px", "qty", "order_id", "ival", "fval")
    assert int(data["local_ts"][0]) == 1777335329819270838


def test_build_live_raw_manifest_matches_backtest_contract(tmp_path: Path) -> None:
    data_file = tmp_path / "btcusdt_20260428.npz"
    np.savez_compressed(
        data_file,
        data=np.empty(
            0,
            dtype=[
                ("ev", "u1"),
                ("exch_ts", "i8"),
                ("local_ts", "i8"),
                ("px", "f8"),
                ("qty", "f8"),
                ("order_id", "u8"),
                ("ival", "i8"),
                ("fval", "f8"),
            ],
        ),
    )

    manifest = build_live_raw_manifest(
        symbol="BTCUSDT",
        start_day="2026-04-28",
        end_day="2026-04-28",
        data_files=[data_file],
        initial_snapshot=None,
        strict_timestamps=True,
    )

    assert manifest["symbol"] == "BTCUSDT"
    assert manifest["start_day"] == "2026-04-28"
    assert manifest["end_day"] == "2026-04-28"
    assert manifest["data_files"] == [str(data_file.resolve())]
    assert manifest["initial_snapshot"] is None

def test_convert_live_raw_file_tolerates_missing_gzip_trailer(tmp_path: Path) -> None:
    raw = tmp_path / "btcusdt_20260428_truncated.gz"
    out = tmp_path / "btcusdt_20260428_truncated.npz"
    _write_gz(raw, _sample_rows())
    raw.write_bytes(raw.read_bytes()[:-8])

    data = convert_live_raw_file(raw, out, buffer_size=16)

    assert out.exists()
    assert len(data) == 3
    assert int(data["local_ts"][-1]) == 1777335329829270838


def test_convert_live_raw_file_ignores_incomplete_final_json_line(tmp_path: Path) -> None:
    raw = tmp_path / "btcusdt_20260428_partial_line.gz"
    out = tmp_path / "btcusdt_20260428_partial_line.npz"
    rows = _sample_rows()
    with gzip.open(raw, "wt") as f:
        for ts, msg in rows:
            f.write(f"{ts} {json.dumps(msg, separators=(',', ':'))}\n")
        f.write(f"1777335329839270838 {{\"stream\":\"btcusdt@depth@0ms\",\"data\":{{\"e\":\"depthUpdate\"")
    raw.write_bytes(raw.read_bytes()[:-8])

    data = convert_live_raw_file(raw, out, buffer_size=16)

    assert out.exists()
    assert len(data) == 3
    assert int(data["local_ts"][-1]) == 1777335329829270838
