from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import hyperliquid_tiny_live_m2_no_fill_diagnosis as diagnosis


def test_quote_position_classifies_touch_and_cross() -> None:
    metrics = diagnosis.BookMetrics(
        bid=diagnosis.Decimal("100"),
        ask=diagnosis.Decimal("101"),
        mid=diagnosis.Decimal("100.5"),
        spread_ticks=diagnosis.Decimal("1"),
        bid_top_qty=diagnosis.Decimal("2"),
        ask_top_qty=diagnosis.Decimal("3"),
        bid_top_n=4,
        ask_top_n=5,
        bid_top5_qty=diagnosis.Decimal("5"),
        ask_top5_qty=diagnosis.Decimal("6"),
        bid_top5_notional=diagnosis.Decimal("500"),
        ask_top5_notional=diagnosis.Decimal("600"),
    )

    assert diagnosis.quote_position("buy", diagnosis.Decimal("100"), metrics) == "same_side_touch_join_back"
    assert diagnosis.quote_position("sell", diagnosis.Decimal("101"), metrics) == "same_side_touch_join_back"
    assert diagnosis.quote_position("buy", diagnosis.Decimal("101"), metrics) == "cross_or_taker_reject_expected"
    assert diagnosis.quote_position("sell", diagnosis.Decimal("100"), metrics) == "cross_or_taker_reject_expected"


def test_l2_book_metrics_computes_top_depth() -> None:
    metrics = diagnosis.l2_book_metrics(
        {
            "levels": [
                [{"px": "100", "sz": "2", "n": 3}, {"px": "99", "sz": "4", "n": 1}],
                [{"px": "101", "sz": "5", "n": 2}, {"px": "102", "sz": "6", "n": 1}],
            ]
        }
    )

    assert metrics.bid == diagnosis.Decimal("100")
    assert metrics.ask == diagnosis.Decimal("101")
    assert metrics.spread_ticks == diagnosis.Decimal("1")
    assert metrics.bid_top5_qty == diagnosis.Decimal("6")
    assert metrics.ask_top5_qty == diagnosis.Decimal("11")


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_run_diagnosis_outputs_decision_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "hyperliquid_tiny_live_m2_fill_loop_0618T010"
    window_dir = root / "window_1" / "pulled_back_awsserver1"
    window_dir.mkdir(parents=True)
    (window_dir / "market_markout_snapshot.json").write_text(
        json.dumps(
            {
                "pre_l2": {
                    "time": 1,
                    "levels": [
                        [{"px": "100", "sz": "2", "n": 3}, {"px": "99", "sz": "1", "n": 1}],
                        [{"px": "101", "sz": "5", "n": 4}, {"px": "102", "sz": "1", "n": 1}],
                    ],
                },
                "post_l2": {
                    "time": 2,
                    "levels": [
                        [{"px": "99", "sz": "1", "n": 1}],
                        [{"px": "100", "sz": "1", "n": 1}],
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    _write_csv(
        window_dir / "quote_attempt_matrix.csv",
        [
            {
                "attempt": "1",
                "side": "buy",
                "limit_px": "100",
                "size_btc": "0.01",
                "bid": "100",
                "ask": "101",
                "post_only_tif": "Alo",
                "order_status_types": "resting",
                "fill_count_after_attempt": "0",
                "crossing_guard_status": "pass",
            }
        ],
        [
            "attempt",
            "side",
            "limit_px",
            "size_btc",
            "bid",
            "ask",
            "post_only_tif",
            "order_status_types",
            "fill_count_after_attempt",
            "crossing_guard_status",
        ],
    )

    manifest = diagnosis.run_diagnosis([root], tmp_path / "out")

    assert manifest["design_decision"] == "do_not_blind_retry; run_read_only_public_flow_diagnosis_next"
    assert manifest["summary"]["attempt_count"] == 1
    rows = list(csv.DictReader((tmp_path / "out" / "quote_attempt_diagnostics.csv").open()))
    assert rows[0]["quote_position"] == "same_side_touch_join_back"
    assert rows[0]["top_depth_multiple_of_order"] == "200"
    assert (tmp_path / "out" / "evidence_gap_matrix.csv").exists()
    assert (tmp_path / "out" / "design_decision_matrix.csv").exists()
