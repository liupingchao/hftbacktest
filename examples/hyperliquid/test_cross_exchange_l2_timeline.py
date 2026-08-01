from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_l2_timeline as timeline
from cross_exchange_symbol_registry import available_profile_ids, get_symbol_profile, normalize_signal_symbol


def _write_raw(path: Path, rows: list[tuple[int, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for local_ts, payload in rows:
            fh.write(f"{local_ts} {json.dumps(payload, separators=(',', ':'))}\n")


def _sample(tmp_path: Path, *, gap: bool = False, regression: bool = False) -> Path:
    sample_dir = tmp_path / "sample"
    _write_raw(
        sample_dir / "binance_public_raw" / "raw.gz",
        [
            (
                100,
                {
                    "lastUpdateId": 10,
                    "T": 1,
                    "bids": [["100", "2"], ["99", "3"]],
                    "asks": [["101", "4"], ["102", "5"]],
                },
            ),
            (
                400,
                {
                    "stream": "btcusdt@depth@0ms",
                    "data": {
                        "e": "depthUpdate",
                        "s": "BTCUSDT",
                        "T": 2,
                        "U": 10,
                        "u": 12,
                        "pu": 10,
                        "b": [["100", "0"], ["100.5", "7"]],
                        "a": [],
                    },
                },
            ),
            (
                350 if regression else 700,
                {
                    "stream": "btcusdt@depth@0ms",
                    "data": {
                        "e": "depthUpdate",
                        "s": "BTCUSDT",
                        "T": 3,
                        "U": 13,
                        "u": 14,
                        "pu": 999 if gap else 12,
                        "b": [],
                        "a": [["101", "6"]],
                    },
                },
            ),
        ],
    )
    _write_raw(
        sample_dir / "hyperliquid_public_sample" / "raw.gz",
        [
            (
                200,
                {
                    "channel": "l2Book",
                    "data": {
                        "coin": "BTC",
                        "time": 1,
                        "levels": [
                            [{"px": "100.1", "sz": "1", "n": 2}],
                            [{"px": "100.9", "sz": "2", "n": 3}],
                        ],
                    },
                },
            ),
            (
                500,
                {
                    "channel": "l2Book",
                    "data": {
                        "coin": "BTC",
                        "time": 2,
                        "levels": [
                            [{"px": "100.2", "sz": "3", "n": 4}],
                            [{"px": "100.8", "sz": "4", "n": 5}],
                        ],
                    },
                },
            ),
        ],
    )
    _write_raw(
        sample_dir / "hyperliquid_public_sample" / "research_tracks" / "standard_l2" / "raw.gz",
        [
            (
                300,
                {
                    "channel": "l2Book",
                    "data": {
                        "coin": "BTC",
                        "time": 1,
                        "levels": [
                            [
                                {"px": "100.1", "sz": "10", "n": 6},
                                {"px": "100.0", "sz": "11", "n": 7},
                            ],
                            [
                                {"px": "100.9", "sz": "12", "n": 8},
                                {"px": "101.0", "sz": "13", "n": 9},
                            ],
                        ],
                    },
                },
            ),
            (
                600,
                {
                    "channel": "l2Book",
                    "data": {
                        "coin": "BTC",
                        "time": 2,
                        "levels": [
                            [{"px": "100.3", "sz": "14", "n": 10}],
                            [{"px": "100.7", "sz": "15", "n": 11}],
                        ],
                    },
                },
            ),
        ],
    )
    return sample_dir


def _read_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_registry_exposes_all_research_profiles_and_aliases() -> None:
    assert available_profile_ids() == ("btc", "eth", "mu", "skhynix")
    assert get_symbol_profile("eth").binance_symbol == "ETHUSDT"
    assert get_symbol_profile("mu").hyperliquid_coin == "xyz:MU"
    assert normalize_signal_symbol("MUUSDT") == "XYZ:MU"
    assert normalize_signal_symbol("eth/usdc") == "ETH"


def test_build_common_timeline_replays_books_without_future_joins(tmp_path: Path) -> None:
    sample_dir = _sample(tmp_path)
    manifest = timeline.build_common_l2_timeline(
        sample_dir=sample_dir,
        output_dir=tmp_path / "out",
        profile_id="btc",
        binance_symbol="BTCUSDT",
        hyperliquid_coin="BTC",
        top_n=2,
        campaign_id="campaign",
        segment_id="segment_0001",
    )
    rows = _read_rows(tmp_path / "out" / "common_l2_timeline.csv.gz")

    assert manifest["input_event_count_by_track"] == {
        "binance": 3,
        "hyperliquid_fast": 2,
        "hyperliquid_standard": 2,
    }
    assert manifest["warmup_event_count"] == 2
    assert manifest["timeline_row_count"] == 5
    assert [int(row["common_ts_ns"]) for row in rows] == [300, 400, 500, 600, 700]
    for row in rows:
        common_ts = int(row["common_ts_ns"])
        for track in timeline.TRACKS:
            assert int(row[f"{track}_local_ts_ns"]) <= common_ts
            assert float(row[f"{track}_age_ms"]) >= 0

    update_row = rows[1]
    assert update_row["trigger_track"] == "binance"
    assert update_row["binance_bid_1_px"] == "100.5"
    assert update_row["binance_bid_2_px"] == "99"
    assert update_row["hyperliquid_fast_bid_1_n"] == "2"
    assert update_row["hyperliquid_standard_bid_2_n"] == "7"
    assert manifest["capability_boundary"]["exact_fill_simulation"] is False
    assert manifest["segment_boundary"]["cross_segment_continuity_claimed"] is False


def test_binance_replay_gap_fails_closed(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "common_l2_timeline.csv.gz").write_bytes(b"stale")
    (output_dir / "common_l2_timeline_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(timeline.TimelineError, match="Binance replay gap"):
        timeline.build_common_l2_timeline(
            sample_dir=_sample(tmp_path, gap=True),
            output_dir=output_dir,
            profile_id="btc",
            binance_symbol="BTCUSDT",
            hyperliquid_coin="BTC",
        )
    assert not (output_dir / "common_l2_timeline.csv.gz").exists()
    assert not (output_dir / "common_l2_timeline.csv.gz.tmp").exists()
    assert not (output_dir / "common_l2_timeline_manifest.json").exists()


def test_binance_snapshot_bridge_failure_fails_closed(tmp_path: Path) -> None:
    sample_dir = _sample(tmp_path)
    path = sample_dir / "binance_public_raw" / "raw.gz"
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            local_ts, payload = line.split(" ", 1)
            rows.append((int(local_ts), json.loads(payload)))
    rows[1][1]["data"]["U"] = 200
    rows[1][1]["data"]["u"] = 201
    rows[1][1]["data"]["pu"] = 199
    _write_raw(path, rows)
    with pytest.raises(timeline.TimelineError, match="snapshot bridge failed"):
        timeline.build_common_l2_timeline(
            sample_dir=sample_dir,
            output_dir=tmp_path / "out",
            profile_id="btc",
            binance_symbol="BTCUSDT",
            hyperliquid_coin="BTC",
        )


def test_local_timestamp_regression_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(timeline.TimelineError, match="timestamp regressed"):
        timeline.build_common_l2_timeline(
            sample_dir=_sample(tmp_path, regression=True),
            output_dir=tmp_path / "out",
            profile_id="btc",
            binance_symbol="BTCUSDT",
            hyperliquid_coin="BTC",
        )


def test_wrong_hyperliquid_coin_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(timeline.TimelineError, match="expected Hyperliquid coin ETH"):
        timeline.build_common_l2_timeline(
            sample_dir=_sample(tmp_path),
            output_dir=tmp_path / "out",
            profile_id="eth",
            binance_symbol="BTCUSDT",
            hyperliquid_coin="ETH",
        )


def test_timeline_source_age_gate_rejects_unbounded_forward_fill(tmp_path: Path) -> None:
    sample_dir = _sample(tmp_path)
    binance_path = sample_dir / "binance_public_raw" / "raw.gz"
    rows = []
    with gzip.open(binance_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            local_ts, payload = line.split(" ", 1)
            rows.append((int(local_ts), json.loads(payload)))
    rows[-1] = (20_000_000_000, rows[-1][1])
    _write_raw(binance_path, rows)
    manifest = timeline.build_common_l2_timeline(
        sample_dir=sample_dir,
        output_dir=tmp_path / "out",
        profile_id="btc",
        binance_symbol="BTCUSDT",
        hyperliquid_coin="BTC",
        max_hyperliquid_fast_age_ms=1_000,
        max_hyperliquid_standard_age_ms=30_000,
    )
    assert manifest["passes"] is False
    assert "hyperliquid_fast_source_age_exceeds_limit" in manifest["failures"]


def test_timeline_rejects_nonfinite_age_limits(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source age limits"):
        timeline.build_common_l2_timeline(
            sample_dir=_sample(tmp_path),
            output_dir=tmp_path / "out",
            profile_id="btc",
            binance_symbol="BTCUSDT",
            hyperliquid_coin="BTC",
            max_hyperliquid_fast_age_ms=float("inf"),
        )


def test_timeline_cli_returns_nonzero_when_quality_manifest_fails(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(timeline, "build_common_l2_timeline", lambda **_kwargs: {"passes": False})
    assert timeline.main(
        [
            "--sample-dir",
            str(tmp_path / "sample"),
            "--output-dir",
            str(tmp_path / "out"),
            "--symbol-profile",
            "btc",
        ]
    ) == 4
