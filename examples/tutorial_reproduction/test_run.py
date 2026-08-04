from __future__ import annotations

import gzip
from pathlib import Path

from examples.tutorial_reproduction.run import (
    _resolve_tardis_file,
    _slice_tardis_file,
)


def _write_rows(path: Path, timestamps: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write("exchange,symbol,timestamp,local_timestamp,price,amount,side\n")
        for timestamp in timestamps:
            stream.write(
                f"binance-futures,BTCUSDT,{timestamp},{timestamp + 1},100.0,0.1,buy\n"
            )


def test_resolve_tardis_file_supports_flat_and_daily_layouts(tmp_path: Path) -> None:
    flat = tmp_path / "trades" / "BTCUSDT.csv.gz"
    daily = tmp_path / "trades" / "2025" / "01" / "01" / "BTCUSDT.csv.gz"
    _write_rows(flat, [1])

    assert _resolve_tardis_file(tmp_path, "trades", "2025-01-01", "BTCUSDT") == flat

    _write_rows(daily, [2])
    assert _resolve_tardis_file(tmp_path, "trades", "2025-01-01", "BTCUSDT") == daily


def test_slice_tardis_file_uses_half_open_timestamp_window(tmp_path: Path) -> None:
    source = tmp_path / "trades.csv.gz"
    destination = tmp_path / "staged.csv.gz"
    _write_rows(source, [99, 100, 150, 199, 200, 201])

    rows = _slice_tardis_file(source, destination, 100, 200)

    assert rows == 3
    with gzip.open(destination, "rt", encoding="utf-8") as stream:
        timestamps = [int(line.split(",")[2]) for line in list(stream)[1:]]
    assert timestamps == [100, 150, 199]
