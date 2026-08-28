from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from examples.hyperliquid.skhynix_obi_reversal_track_a0 import (
    CaptureBinding,
    OBISequence,
    control_common_support,
    control_candidates,
    detect_reversals,
    load_obi_sequence,
    match_controls,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binding(path: Path) -> CaptureBinding:
    return CaptureBinding(
        capture_id="fixture",
        research_date="2026-08-01",
        role="historical_method_development",
        start_utc="2026-08-01T00:00:00+00:00",
        end_utc="2026-08-01T00:01:00+00:00",
        duration_seconds=60.0,
        raw_path=path,
        raw_size_bytes=path.stat().st_size,
        raw_sha256=_sha256(path),
        raw_size_verified=True,
        raw_hash_verified=True,
        depth_gap_count=0,
        cache_path=path,
        cache_sha256=_sha256(path),
        cache_size_bytes=path.stat().st_size,
    )


def test_standard_obi_recovers_quantities_from_log_features(tmp_path: Path) -> None:
    names = [
        *(f"bid_qty_log_l{i}" for i in range(1, 6)),
        *(f"ask_qty_log_l{i}" for i in range(1, 6)),
        "midpoint_delta_ticks",
    ]
    values = np.zeros((2, len(names)), dtype=np.float32)
    values[:, :5] = np.log1p(2.0)
    values[:, 5:10] = np.log1p(1.0)
    values[:, 10] = [1000.0, -1000.0]
    path = tmp_path / "fixture.npz"
    np.savez(
        path,
        base_features=values,
        feature_names=np.asarray(names),
        ts_ns=np.asarray([100_000_000, 200_000_000]),
        valid=np.asarray([True, True]),
        capture_id=np.asarray("fixture"),
        research_date=np.asarray("2026-08-01"),
        role=np.asarray("historical_method_development"),
    )
    sequence = load_obi_sequence(_binding(path))
    assert np.allclose(sequence.obi, 1 / 3)


def test_forbidden_price_column_does_not_change_obi(tmp_path: Path) -> None:
    names = [
        *(f"bid_qty_log_l{i}" for i in range(1, 6)),
        *(f"ask_qty_log_l{i}" for i in range(1, 6)),
        "midpoint_delta_ticks",
    ]
    values = np.zeros((3, len(names)), dtype=np.float32)
    values[:, :5] = np.log1p([[2, 2, 2, 2, 2]] * 3)
    values[:, 5:10] = np.log1p([[1, 1, 1, 1, 1]] * 3)
    outputs = []
    for index, price_values in enumerate(([0, 0, 0], [999, -999, 123])):
        candidate = values.copy()
        candidate[:, 10] = price_values
        path = tmp_path / f"fixture_{index}.npz"
        np.savez(
            path,
            base_features=candidate,
            feature_names=np.asarray(names),
            ts_ns=np.arange(3) * 100_000_000,
            valid=np.ones(3, dtype=bool),
            capture_id=np.asarray("fixture"),
            research_date=np.asarray("2026-08-01"),
            role=np.asarray("historical_method_development"),
        )
        outputs.append(load_obi_sequence(_binding(path)).obi)
    assert np.array_equal(outputs[0], outputs[1])


def test_reversal_detection_uses_confirmation_timestamp() -> None:
    obi = np.asarray([0.7] * 12 + [0.2, 0.0, -0.2] + [-0.7] * 12)
    ts = np.arange(len(obi), dtype=np.int64) * 100_000_000
    binding = CaptureBinding(
        capture_id="fixture",
        research_date="2026-08-01",
        role="historical_method_development",
        start_utc="",
        end_utc="",
        duration_seconds=len(obi) / 10,
        raw_path=Path("raw"),
        raw_size_bytes=0,
        raw_sha256="",
        raw_size_verified=True,
        raw_hash_verified=True,
        depth_gap_count=0,
        cache_path=Path("cache"),
        cache_sha256="",
        cache_size_bytes=0,
    )
    sequence = OBISequence(
        binding=binding,
        ts_ns=ts,
        obi=obi,
        valid=np.ones(len(obi), dtype=bool),
        informative=True,
        obi_std=float(np.std(obi)),
        unique_rounded=len(np.unique(obi)),
    )
    rows = detect_reversals(sequence)
    assert len(rows) == 1
    assert rows[0]["cross_index"] == 13
    assert rows[0]["detect_index"] == 24
    assert rows[0]["detect_ts_ns"] > rows[0]["cross_ts_ns"]


def test_controls_exclude_recent_opposite_prestate() -> None:
    obi = np.asarray([-0.7] * 10 + [0.0] * 10 + [0.7] * 200)
    ts = np.arange(len(obi), dtype=np.int64) * 100_000_000
    binding = CaptureBinding(
        capture_id="fixture",
        research_date="2026-08-01",
        role="historical_method_development",
        start_utc="",
        end_utc="",
        duration_seconds=len(obi) / 10,
        raw_path=Path("raw"),
        raw_size_bytes=0,
        raw_sha256="",
        raw_size_verified=True,
        raw_hash_verified=True,
        depth_gap_count=0,
        cache_path=Path("cache"),
        cache_sha256="",
        cache_size_bytes=0,
    )
    sequence = OBISequence(
        binding=binding,
        ts_ns=ts,
        obi=obi,
        valid=np.ones(len(obi), dtype=bool),
        informative=True,
        obi_std=float(np.std(obi)),
        unique_rounded=len(np.unique(obi)),
    )
    rows = control_candidates(sequence, [])
    assert rows
    assert min(row["detect_index"] for row in rows) >= 109


def test_common_support_is_distinct_from_no_reuse_matching() -> None:
    reversals = [
        {
            "capture_id": f"reversal_{index}",
            "research_date": "2026-08-01",
            "role": "historical_method_development",
            "side": 1,
            "detect_ts_ns": index,
            "obi_at_detection": 0.51,
            "remaining_ms": 10_000,
        }
        for index in range(3)
    ]
    controls = [
        {
            "capture_id": "control",
            "research_date": "2026-08-01",
            "role": "historical_method_development",
            "side": 1,
            "detect_ts_ns": 100,
            "obi_at_detection": 0.51,
            "obi_bin": 0,
            "remaining_ms": 10_000,
        }
    ]

    matched, unmatched = match_controls(reversals, controls)
    overall, minimum_date, covered_count = control_common_support(
        reversals,
        controls,
    )

    assert len(matched) == 1
    assert len(unmatched) == 2
    assert overall == 1.0
    assert minimum_date == 1.0
    assert covered_count == 3
