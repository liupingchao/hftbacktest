from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from binance_top5_provenance import build_sidecars, join_decisions


def _write_gz(path: Path, rows: list[tuple[int, dict]]) -> None:
    with gzip.open(path, "wt") as fh:
        for ts, msg in rows:
            fh.write(f"{ts} {json.dumps(msg, separators=(',', ':'))}\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fixture_rows() -> list[tuple[int, dict]]:
    return [
        (
            1_000_000_000_000,
            {
                "lastUpdateId": 10,
                "T": 1_000_000,
                "bids": [["100.00", "1.0"], ["99.90", "2.0"], ["99.80", "3.0"]],
                "asks": [["100.10", "1.5"], ["100.20", "2.5"], ["100.30", "3.5"]],
            },
        ),
        (
            1_000_001_000_000,
            {
                "stream": "btcusdt@depth@100ms",
                "data": {
                    "e": "depthUpdate",
                    "E": 1_000_001,
                    "T": 1_000_001,
                    "s": "BTCUSDT",
                    "U": 11,
                    "u": 12,
                    "pu": 10,
                    "b": [["100.00", "1.2"], ["99.70", "4.0"]],
                    "a": [["100.10", "1.4"], ["100.40", "4.5"]],
                },
            },
        ),
        (
            1_000_001_500_000,
            {
                "stream": "btcusdt@bookTicker",
                "data": {
                    "e": "bookTicker",
                    "E": 1_000_001,
                    "T": 1_000_001,
                    "s": "BTCUSDT",
                    "u": 12,
                    "b": "100.00",
                    "B": "1.2",
                    "a": "100.10",
                    "A": "1.4",
                },
            },
        ),
        (
            1_000_002_000_000,
            {
                "stream": "btcusdt@depth@100ms",
                "data": {
                    "e": "depthUpdate",
                    "E": 1_000_002,
                    "T": 1_000_002,
                    "s": "BTCUSDT",
                    "U": 13,
                    "u": 13,
                    "pu": 12,
                    "b": [["100.00", "0.8"]],
                    "a": [["100.10", "1.8"]],
                },
            },
        ),
    ]


def test_build_sidecars_preserves_standard_npz_and_final_row_mapping(tmp_path: Path) -> None:
    raw = tmp_path / "btcusdt.gz"
    out = tmp_path / "out"
    _write_gz(raw, _fixture_rows())

    result = build_sidecars(
        raw,
        out,
        sample_id="fixture",
        symbol="BTCUSDT",
        tick_size=0.1,
        buffer_size=64,
    )

    loaded = np.load(out / "data.npz")
    assert loaded["data"].dtype.names == ("ev", "exch_ts", "local_ts", "px", "qty", "order_id", "ival", "fval")
    assert len(loaded["data"]) == len(result.data)

    mapping = _read_csv(out / "raw_to_npz_mapping.csv")
    assert mapping[0]["event_type"] == "snapshot"
    assert int(mapping[0]["row_count"]) > 0
    assert int(mapping[1]["row_count"]) > 0
    assert mapping[2]["event_type"] == "bookTicker"
    assert mapping[2]["row_count"] == "0"
    assert mapping[2]["row_reason"] == "bookticker_not_in_npz_without_opt_t"
    assert result.metrics["final_data_row_mapping_coverage"] == 1.0

    manifest = json.loads((out / "sidecar_manifest.json").read_text())
    assert manifest["schema_version"] == "binance_top5_provenance_v1"
    assert manifest["top5_levels"] == 5
    assert manifest["tick_size"] == 0.1


def test_top5_sidecar_records_snapshot_alignment_and_bookticker_match(tmp_path: Path) -> None:
    raw = tmp_path / "btcusdt.gz"
    out = tmp_path / "out"
    _write_gz(raw, _fixture_rows())

    result = build_sidecars(raw, out, sample_id="fixture", symbol="BTCUSDT", tick_size=0.1, buffer_size=64)

    top5 = _read_csv(out / "top5_sidecar.csv")
    first_depth = next(row for row in top5 if row["event_type"] == "depthUpdate" and row["first_valid_update_aligned"])
    assert first_depth["first_valid_update_aligned"] == "true"
    assert first_depth["sync_aligned"] == "True"
    assert first_depth["snapshot_lastUpdateId"] == "10"
    assert first_depth["bid_top5_ticks"].startswith("1000|999")
    assert result.metrics["first_valid_update_aligned"] == "true"

    second_depth = [row for row in top5 if row["event_type"] == "depthUpdate"][-1]
    assert second_depth["bookticker_u"] == "12"
    assert second_depth["bookticker_bbo_match"] == "true"
    assert second_depth["bookticker_depth_age_ms"] != ""


def test_bookticker_maps_to_final_rows_when_opt_t_enabled(tmp_path: Path) -> None:
    raw = tmp_path / "btcusdt.gz"
    out = tmp_path / "out"
    _write_gz(raw, _fixture_rows())

    build_sidecars(raw, out, sample_id="fixture", symbol="BTCUSDT", tick_size=0.1, opt="t", buffer_size=64)

    mapping = _read_csv(out / "raw_to_npz_mapping.csv")
    bookticker = mapping[2]
    assert bookticker["event_type"] == "bookTicker"
    assert int(bookticker["row_count"]) >= 2
    assert bookticker["final_row_indices"]


def test_join_decisions_uses_asof_history_and_reports_no_future_join(tmp_path: Path) -> None:
    raw = tmp_path / "btcusdt.gz"
    out = tmp_path / "out"
    _write_gz(raw, _fixture_rows())
    build_sidecars(raw, out, sample_id="fixture", symbol="BTCUSDT", tick_size=0.1, buffer_size=64)

    audit = tmp_path / "audit.csv"
    with audit.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["strategy_seq", "event_type", "ts_local"])
        writer.writeheader()
        writer.writerow({"strategy_seq": "1", "event_type": "decision", "ts_local": "1000002500000"})
        writer.writerow({"strategy_seq": "2", "event_type": "decision", "ts_local": "999999999999"})

    joined = out / "joined_decisions.csv"
    metrics = join_decisions(audit, out / "top5_sidecar.csv", joined, max_age_ms=10_000)
    rows = _read_csv(joined)

    assert metrics["decision_count"] == 2
    assert metrics["joined_decision_count"] == 1
    assert metrics["future_join_count"] == 0
    assert rows[0]["strategy_seq"] == "1"
    assert rows[0]["join_used_future"] == "false"
    assert rows[0]["joined_raw_seq"] == "3"
    assert float(rows[0]["top5_join_age_ms"]) == 0.5
    assert rows[1]["strategy_seq"] == "2"
    assert rows[1]["join_missing"] == "true"
