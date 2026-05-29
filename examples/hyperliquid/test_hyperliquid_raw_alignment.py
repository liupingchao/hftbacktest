from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import hyperliquid_raw_alignment as alignment


def _write_gz(path: Path, rows: list[tuple[int, dict]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for ts, msg in rows:
            fh.write(f"{ts} {json.dumps(msg, separators=(',', ':'))}\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _l2book(ts_ms: int, bid_qty: str = "1.0", ask_qty: str = "2.0") -> dict:
    return {
        "channel": "l2Book",
        "data": {
            "coin": "BTC",
            "time": ts_ms,
            "levels": [
                [
                    {"px": "100.0", "sz": bid_qty, "n": 2},
                    {"px": "99.9", "sz": "3.0", "n": 1},
                ],
                [
                    {"px": "100.1", "sz": ask_qty, "n": 3},
                    {"px": "100.2", "sz": "4.0", "n": 1},
                ],
            ],
        },
    }


def _trades(ts_ms: int) -> dict:
    return {
        "channel": "trades",
        "data": [
            {
                "coin": "BTC",
                "side": "B",
                "px": "100.1",
                "sz": "0.1",
                "hash": "0x1",
                "time": ts_ms,
                "tid": 1,
                "users": ["0x0", "0x1"],
            }
        ],
    }


def test_build_alignment_outputs_hyperliquid_sidecars_without_binance_fields(tmp_path: Path) -> None:
    raw = tmp_path / "sample.gz"
    out = tmp_path / "out"
    _write_gz(
        raw,
        [
            (1_000_000_000_000_000_000, _l2book(1_000_000)),
            (1_000_000_000_500_000_000, _trades(1_000_100)),
            (1_000_000_001_000_000_000, _l2book(1_001_000, bid_qty="1.5", ask_qty="2.5")),
        ],
    )

    result = alignment.build_alignment(
        input_gzip=raw,
        output_dir=out,
        tick_size=0.1,
        lot_size=0.001,
        num_levels=2,
        top_n=2,
        synthetic_interval_ms=500,
        buffer_size=128,
        source_label="test_fixture",
    )

    metrics = result["metrics"]
    assert metrics["l2book_message_count"] == 2
    assert metrics["trade_event_count"] == 1
    assert metrics["npz_row_count"] > 0
    assert metrics["topn_coverage"] == 1.0
    assert metrics["decision_join_coverage"] == 1.0
    assert metrics["future_join_count"] == 0
    assert metrics["missing_join_count"] == 0
    assert metrics["sample_classification"] == "limited_pricing_research"

    for name in [
        "run_manifest.json",
        "collection_manifest.json",
        "converter_manifest.json",
        "data.npz",
        "raw_provenance.csv",
        "raw_to_npz_mapping.csv",
        "topn_sidecar.csv",
        "synthetic_joined_views.csv",
        "metrics.json",
        "acceptance_report.md",
    ]:
        assert (out / name).exists()

    topn_rows = _read_csv(out / "topn_sidecar.csv")
    assert topn_rows[0]["bid_topn_px"] == "100.0|99.9"
    assert topn_rows[0]["ask_topn_n"] == "3|1"
    assert not {"depth_U", "depth_u", "depth_pu", "lastUpdateId", "bookticker_u"}.intersection(topn_rows[0])


def test_missing_trade_stream_is_unusable(tmp_path: Path) -> None:
    raw = tmp_path / "sample.gz"
    out = tmp_path / "out"
    _write_gz(raw, [(1_000_000_000_000_000_000, _l2book(1_000_000))])

    result = alignment.build_alignment(
        input_gzip=raw,
        output_dir=out,
        tick_size=0.1,
        lot_size=0.001,
        num_levels=2,
        top_n=2,
        synthetic_interval_ms=500,
        buffer_size=128,
        source_label="test_fixture",
    )

    assert result["metrics"]["sample_classification"] == "unusable"
    assert result["metrics"]["classification_reason"] == "missing_required_l2book_or_trades"


def test_synthetic_join_uses_asof_topn_row() -> None:
    rows = [
        alignment.TopNRow(
            raw_seq=1,
            line_no=1,
            channel="l2Book",
            coin="BTC",
            local_ts=1_000_000_000_000,
            event_ts=999_999_000_000,
            bid_px="100.0",
            bid_ticks="1000",
            bid_qty="1.0",
            bid_n="1",
            ask_px="100.1",
            ask_ticks="1001",
            ask_qty="1.0",
            ask_n="1",
        ),
        alignment.TopNRow(
            raw_seq=2,
            line_no=2,
            channel="l2Book",
            coin="BTC",
            local_ts=1_001_000_000_000,
            event_ts=1_000_999_000_000,
            bid_px="101.0",
            bid_ticks="1010",
            bid_qty="1.0",
            bid_n="1",
            ask_px="101.1",
            ask_ticks="1011",
            ask_qty="1.0",
            ask_n="1",
        ),
    ]

    joined = alignment.build_synthetic_join_rows(rows, interval_ms=500)

    assert joined[0].joined_raw_seq == "1"
    assert joined[1].joined_raw_seq == "1"
    assert joined[1].join_age_ms == "500.000000"
    assert joined[2].joined_raw_seq == "2"
