from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analyze_cancel_fill_risk import analyze_audit_csv


FIELDS = [
    "run_id",
    "strategy_seq",
    "event_type",
    "ts_local",
    "order_id",
    "action",
    "mid",
    "position",
    "order_side",
    "order_price",
    "order_qty",
    "order_executed_qty",
    "order_status",
    "cancel_requested",
    "cancel_request_ts",
    "fill_qty",
    "fill_price",
    "fill_after_cancel_request",
]


def _write_audit(path: Path, rows: list[dict[str, object]]) -> Path:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
    return path


def _decision(ts: int, mid: float, position: float = 0.0, action: str = "keep") -> dict[str, object]:
    return {
        "run_id": "test",
        "strategy_seq": ts,
        "event_type": "decision",
        "ts_local": ts,
        "mid": mid,
        "position": position,
        "action": action,
    }


def test_detects_fill_after_cancel_request_and_side_adjusted_markout(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path / "audit_live_test.csv",
        [
            _decision(1_000, 100.0, 0.0),
            {
                "run_id": "test",
                "event_type": "cancel_sent",
                "ts_local": 1_100,
                "order_id": "1",
                "action": "cancel_buy",
                "order_side": "buy",
                "cancel_requested": 1,
                "cancel_request_ts": 1_100,
            },
            {
                "run_id": "test",
                "event_type": "fill",
                "ts_local": 1_200,
                "order_id": "1",
                "action": "fill",
                "order_side": "buy",
                "position": 0.001,
                "fill_qty": 0.001,
                "fill_price": 100.0,
                "order_status": "filled",
                "cancel_requested": 1,
                "cancel_request_ts": 1_100,
                "fill_after_cancel_request": 1,
            },
            _decision(1_000_001_200, 99.5, 0.001),
        ],
    )

    result = analyze_audit_csv(audit, run_id="test", sample_class="current_format")

    summary = result["summary"]
    assert summary["fill_after_cancel_request_count"] == 1
    assert summary["worsening_fill_after_cancel_request_count"] == 1
    assert summary["cancel_to_fill_latency_ms_p50"] == pytest.approx(0.0001)
    assert summary["markout_1s_median"] == pytest.approx(-0.5)
    assert summary["source_path_inventory_worsening_no_readd_count"] == 1
    assert summary["cancel_latency_bucket_le_10ms_count"] == 1
    assert result["events"][0]["source_path"] == "inventory_worsening_no_readd"
    assert result["events"][0]["position_before"] == pytest.approx(0.0)
    assert result["events"][0]["position_after"] == pytest.approx(0.001)


def test_pending_cancel_terminal_event_prevents_same_side_readd_count(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path / "audit_live_test.csv",
        [
            _decision(1_000, 100.0, 0.0),
            {
                "run_id": "test",
                "event_type": "cancel_sent",
                "ts_local": 1_100,
                "order_id": "1",
                "action": "cancel_sell",
                "order_side": "sell",
                "cancel_requested": 1,
                "cancel_request_ts": 1_100,
            },
            {
                "run_id": "test",
                "event_type": "cancel_ack",
                "ts_local": 1_200,
                "order_id": "1",
                "order_side": "sell",
                "order_status": "canceled",
            },
            {
                "run_id": "test",
                "event_type": "order_submit_sent",
                "ts_local": 1_300,
                "order_id": "2",
                "action": "submit_sell",
                "order_side": "sell",
                "order_qty": 0.001,
            },
        ],
    )

    result = analyze_audit_csv(audit, run_id="test", sample_class="current_format")

    assert result["summary"]["same_side_readd_while_cancel_requested_count"] == 0


def test_same_side_readd_while_cancel_requested_and_later_fill_overlap(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path / "audit_live_test.csv",
        [
            _decision(1_000, 100.0, position=-0.001),
            {
                "run_id": "test",
                "event_type": "cancel_sent",
                "ts_local": 1_100,
                "order_id": "1",
                "action": "cancel_sell",
                "order_side": "sell",
                "cancel_requested": 1,
                "cancel_request_ts": 1_100,
            },
            {
                "run_id": "test",
                "event_type": "order_submit_sent",
                "ts_local": 1_150,
                "order_id": "2",
                "action": "submit_sell",
                "order_side": "sell",
                "order_qty": 0.001,
            },
            {
                "run_id": "test",
                "event_type": "fill",
                "ts_local": 1_200,
                "order_id": "1",
                "action": "fill",
                "order_side": "sell",
                "position": -0.002,
                "fill_qty": 0.001,
                "fill_price": 100.0,
                "order_status": "filled",
                "cancel_requested": 1,
                "cancel_request_ts": 1_100,
                "fill_after_cancel_request": 1,
            },
        ],
    )

    result = analyze_audit_csv(audit, run_id="test", sample_class="current_format")

    summary = result["summary"]
    assert summary["same_side_readd_while_cancel_requested_count"] == 1
    assert summary["inventory_worsening_readd_while_cancel_requested_count"] == 1
    assert summary["same_side_readd_then_cancel_fill_count"] == 1
    assert summary["worsening_fill_after_cancel_request_count"] == 1
    assert summary["source_path_same_side_readd_inventory_worsening_count"] == 1
    assert result["events"][0]["same_side_readd_before_fill"] == 1
    assert result["events"][0]["source_path"] == "same_side_readd_inventory_worsening"


def test_sell_fill_markout_is_adverse_when_future_mid_rises(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path / "audit_live_test.csv",
        [
            _decision(1_000, 100.0, 0.0),
            {
                "run_id": "test",
                "event_type": "fill",
                "ts_local": 1_200,
                "order_id": "1",
                "action": "fill",
                "order_side": "sell",
                "position": -0.001,
                "fill_qty": 0.001,
                "fill_price": 100.0,
                "order_status": "filled",
                "cancel_requested": 1,
                "cancel_request_ts": 1_100,
                "fill_after_cancel_request": 1,
            },
            _decision(1_000_001_200, 100.25, -0.001),
        ],
    )

    result = analyze_audit_csv(audit, run_id="test", sample_class="current_format")

    assert result["summary"]["markout_1s_median"] == pytest.approx(-0.25)


def test_inventory_reducing_cancel_race_can_still_have_adverse_markout(tmp_path: Path) -> None:
    audit = _write_audit(
        tmp_path / "audit_live_test.csv",
        [
            _decision(1_000, 100.0, position=-0.002),
            {
                "run_id": "test",
                "event_type": "fill",
                "ts_local": 1_200,
                "order_id": "1",
                "action": "fill",
                "order_side": "buy",
                "position": -0.001,
                "fill_qty": 0.001,
                "fill_price": 100.0,
                "order_status": "filled",
                "cancel_requested": 1,
                "cancel_request_ts": 1_100,
                "fill_after_cancel_request": 1,
            },
            _decision(1_000_001_200, 99.75, -0.001),
        ],
    )

    result = analyze_audit_csv(audit, run_id="test", sample_class="current_format")

    summary = result["summary"]
    assert summary["worsening_fill_after_cancel_request_count"] == 0
    assert summary["source_path_inventory_reducing_cancel_race_count"] == 1
    assert summary["source_path_inventory_reducing_cancel_race_adverse_markout_1s_count"] == 1
    assert result["events"][0]["source_path"] == "inventory_reducing_cancel_race"
    assert result["events"][0]["markout_1s"] == pytest.approx(-0.25)
