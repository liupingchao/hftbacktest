from __future__ import annotations

import numpy as np

from latency_from_audit import build_latency_series, build_observed_latency_series


def test_observed_latency_series_preserves_long_tail() -> None:
    rows = np.array(
        [
            (100, 105, 109),
            (200, 200_000_200, 200_000_260),
        ],
        dtype=np.int64,
    )

    out, stats = build_observed_latency_series(rows)

    assert out["req_ts"].tolist() == [100, 200]
    assert (out["exch_ts"] - out["req_ts"]).tolist() == [5, 200_000_000]
    assert stats["mode"] == "observed"
    assert stats["entry_max_ms"] == 200.0


def test_observed_latency_series_filters_invalid_rows() -> None:
    rows = np.array(
        [
            (100, 99, 110),
            (200, 205, 204),
            (300, 305, 309),
        ],
        dtype=np.int64,
    )

    out, stats = build_observed_latency_series(rows)

    assert out["req_ts"].tolist() == [300]
    assert stats["rows"] == 1.0


def test_synthetic_latency_series_still_clips() -> None:
    rows = np.array([(100, 100_000_100, 100_000_200)], dtype=np.int64)

    out, stats = build_latency_series(
        base_rows=rows,
        entry_min_ms=1.0,
        entry_max_ms=2.0,
        resp_min_ms=1.0,
        resp_max_ms=2.0,
        spike_prob=0.0,
        spike_min_ms=8.0,
        spike_max_ms=10.0,
        seed=1,
    )

    assert (out["exch_ts"] - out["req_ts"]).tolist() == [2_000_000]
    assert stats["mode"] == "synthetic"
    assert stats["entry_max_ms"] == 2.0
