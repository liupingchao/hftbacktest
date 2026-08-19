from __future__ import annotations

from pathlib import Path

import pytest

from backtest_tick_mm import (
    _audit_replay_decision_due,
    _backtest_cadence_config,
    _load_audit_cadence_schedule,
)


def _write_csv(path: Path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_load_audit_cadence_schedule_preserves_nanosecond_precision(tmp_path: Path) -> None:
    csv_path = tmp_path / "audit.csv"
    _write_csv(
        csv_path,
        [
            "run_id,ts_local,action",
            "live,1777342117432305601,keep",
        ],
    )

    schedule = _load_audit_cadence_schedule(csv_path, run_id="live", ts_column="ts_local")

    assert schedule == [1777342117432305601]


def test_load_audit_cadence_schedule_filters_run_id_and_dedupes(tmp_path: Path) -> None:
    csv_path = tmp_path / "audit.csv"
    _write_csv(
        csv_path,
        [
            "run_id,ts_local,action",
            "other,100,keep",
            "live,300,keep",
            "live,100,keep",
            "live,300,keep",
            "live,200,keep",
        ],
    )

    schedule = _load_audit_cadence_schedule(csv_path, run_id="live", ts_column="ts_local")

    assert schedule == [100, 200, 300]


def test_load_audit_cadence_schedule_raises_for_empty_filter(tmp_path: Path) -> None:
    csv_path = tmp_path / "audit.csv"
    _write_csv(
        csv_path,
        [
            "run_id,ts_local,action",
            "other,100,keep",
        ],
    )

    with pytest.raises(ValueError, match="no cadence timestamps loaded"):
        _load_audit_cadence_schedule(csv_path, run_id="live", ts_column="ts_local")


def test_load_audit_cadence_schedule_raises_for_missing_column(tmp_path: Path) -> None:
    csv_path = tmp_path / "audit.csv"
    _write_csv(
        csv_path,
        [
            "run_id,wrong_ts,action",
            "live,100,keep",
        ],
    )

    with pytest.raises(KeyError, match="missing cadence timestamp column"):
        _load_audit_cadence_schedule(csv_path, run_id="live", ts_column="ts_local")


def test_audit_replay_decision_due_waits_until_timestamp() -> None:
    due, next_idx, lag_ns = _audit_replay_decision_due(
        ts_local=99,
        schedule=[100],
        schedule_idx=0,
        tolerance_ns=0,
    )

    assert due is False
    assert next_idx == 0
    assert lag_ns == 0


def test_audit_replay_decision_due_honors_tolerance() -> None:
    due, next_idx, lag_ns = _audit_replay_decision_due(
        ts_local=98,
        schedule=[100],
        schedule_idx=0,
        tolerance_ns=2,
    )

    assert due is True
    assert next_idx == 1
    assert lag_ns == -2


def test_audit_replay_decision_due_consumes_one_schedule_per_feed_event() -> None:
    schedule = [100, 100]

    first_due, first_idx, first_lag = _audit_replay_decision_due(
        ts_local=150,
        schedule=schedule,
        schedule_idx=0,
        tolerance_ns=0,
    )
    second_due, second_idx, second_lag = _audit_replay_decision_due(
        ts_local=150,
        schedule=schedule,
        schedule_idx=first_idx,
        tolerance_ns=0,
    )
    third_due, third_idx, third_lag = _audit_replay_decision_due(
        ts_local=150,
        schedule=schedule,
        schedule_idx=second_idx,
        tolerance_ns=0,
    )

    assert (first_due, first_idx, first_lag) == (True, 1, 50)
    assert (second_due, second_idx, second_lag) == (True, 2, 50)
    assert (third_due, third_idx, third_lag) == (False, 2, 0)


def test_backtest_cadence_config_parses_audit_replay_fields() -> None:
    mode, min_interval_ns, audit_csv, run_id, ts_column, tolerance_ns = _backtest_cadence_config(
        {
            "cadence": {
                "mode": "audit_replay",
                "audit_csv": "/tmp/live.csv",
                "run_id": "live_run",
                "ts_column": "ts_local",
                "tolerance_ms": 2.5,
            }
        }
    )

    assert mode == "audit_replay"
    assert min_interval_ns == 0
    assert audit_csv == Path("/tmp/live.csv")
    assert run_id == "live_run"
    assert ts_column == "ts_local"
    assert tolerance_ns == 2_500_000
