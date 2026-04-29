from __future__ import annotations

from compare_audit import _cadence_stats, _nearest_lag_stats


def test_series_int_preserves_nanosecond_precision() -> None:
    rows = [{"ts_local": "1777342117432305601"}]

    stats = _cadence_stats(rows)

    assert stats["ts_local_first"] == 1777342117432305601


def test_cadence_stats_reports_negative_delta_count() -> None:
    rows = [{"ts_local": "300"}, {"ts_local": "100"}, {"ts_local": "200"}]

    stats = _cadence_stats(rows)

    assert stats["negative_delta_count"] == 1


def test_cadence_stats_reports_ts_local_deltas() -> None:
    rows = [{"ts_local": "100"}, {"ts_local": "150"}, {"ts_local": "300"}]

    stats = _cadence_stats(rows)

    assert stats["rows"] == 3
    assert stats["delta_ns"]["count"] == 2
    assert stats["delta_ns"]["p50"] == 100.0
    assert stats["delta_ns"]["max"] == 150.0


def test_nearest_lag_stats_reports_abs_and_signed_lag() -> None:
    bt_rows = [{"ts_local": "90"}, {"ts_local": "210"}, {"ts_local": "300"}]
    live_rows = [{"ts_local": "100"}, {"ts_local": "200"}]

    stats = _nearest_lag_stats(bt_rows, live_rows)

    assert stats["count"] == 2
    assert stats["abs_ns"]["p50"] == 10.0
    assert stats["signed_ns"]["mean"] == 0.0
