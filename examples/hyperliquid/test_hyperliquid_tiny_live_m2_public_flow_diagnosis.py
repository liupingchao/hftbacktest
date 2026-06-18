from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from examples.hyperliquid import hyperliquid_tiny_live_m2_public_flow_diagnosis as flow


def _write_raw(path: Path, rows: list[tuple[int, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for local_ts, payload in rows:
            fh.write(f"{local_ts} {json.dumps(payload, separators=(',', ':'), sort_keys=True)}\n")


def _book(ts_ms: int, bid: str, ask: str, bid_size: str = "1.0", ask_size: str = "1.0") -> dict:
    return {
        "channel": "l2Book",
        "data": {
            "coin": "BTC",
            "time": ts_ms,
            "levels": [
                [{"px": bid, "sz": bid_size, "n": 3}],
                [{"px": ask, "sz": ask_size, "n": 4}],
            ],
        },
    }


def _trades(ts_ms: int, trades: list[dict]) -> dict:
    return {"channel": "trades", "data": [{"coin": "BTC", "time": ts_ms, **trade} for trade in trades]}


def test_trade_through_filters_are_side_aware() -> None:
    buy_quote = flow.Decimal("100")
    sell_quote = flow.Decimal("101")
    sell_aggressor_below_bid = flow.TradeEvent(1, 1, flow.Decimal("99"), flow.Decimal("0.1"), "A", "1")
    buy_aggressor_above_ask = flow.TradeEvent(1, 1, flow.Decimal("102"), flow.Decimal("0.1"), "B", "2")

    assert flow.trade_through_filters("buy", buy_quote, sell_aggressor_below_bid) == (False, True, True)
    assert flow.trade_through_filters("sell", sell_quote, buy_aggressor_above_ask) == (False, True, True)
    assert flow.trade_through_filters("buy", buy_quote, buy_aggressor_above_ask) == (False, False, False)


def test_quote_aging_detects_adverse_lost_touch() -> None:
    books = [
        flow.BookEvent(1, 1000, flow.Decimal("100"), flow.Decimal("101"), flow.Decimal("1"), flow.Decimal("1"), 1, 1),
        flow.BookEvent(2, 1500, flow.Decimal("99"), flow.Decimal("100"), flow.Decimal("1"), flow.Decimal("1"), 1, 1),
    ]

    status, first_not_touch_ms, first_adverse_ms = flow.classify_quote_aging("buy", flow.Decimal("100"), 1000, books)

    assert status == "adverse_lost_touch"
    assert first_not_touch_ms == 500
    assert first_adverse_ms == 500


def test_run_diagnosis_from_fixture_raw_outputs_hypothesis_artifacts(tmp_path: Path) -> None:
    raw = tmp_path / "raw.gz"
    _write_raw(
        raw,
        [
            (1, _book(1000, "100", "101", "1.0", "1.0")),
            (2, _trades(1100, [{"px": "100", "sz": "0.2", "side": "A", "tid": 1}])),
            (3, _trades(1200, [{"px": "99", "sz": "0.2", "side": "A", "tid": 2}])),
            (4, _book(1300, "99", "100", "1.0", "1.0")),
            (5, _trades(1400, [{"px": "102", "sz": "2.0", "side": "B", "tid": 3}])),
            (6, _book(6000, "100", "101", "2.0", "2.0")),
        ],
    )

    out = tmp_path / "out"
    manifest = flow.run_diagnosis(
        output_dir=out,
        raw_input=raw,
        order_size=flow.Decimal("0.1"),
        quote_hold_seconds=1.0,
        candidate_stride_seconds=100.0,
    )

    assert manifest["private_or_order_endpoint_actions"] == "none"
    assert manifest["real_fill_claim"] is False
    assert manifest["queue_position_claim"] == "public_depth_and_trade_depletion_proxy_only_not_exact_priority"
    assert manifest["summary"]["candidate_count"] == 2
    assert (out / "candidate_flow_diagnostics.csv").exists()
    assert (out / "hypothesis_decision_matrix.csv").exists()

    candidate_rows = list(csv.DictReader((out / "candidate_flow_diagnostics.csv").open()))
    buy_row = next(row for row in candidate_rows if row["side"] == "buy")
    sell_row = next(row for row in candidate_rows if row["side"] == "sell")
    assert buy_row["strict_trade_through_qty_btc"] == "0.2"
    assert buy_row["quote_aging_status"] == "adverse_lost_touch"
    assert sell_row["strict_trade_through_qty_btc"] == "2"
    assert sell_row["public_depletion_status"] == "depleted_top_plus_order_proxy"

    decision_rows = list(csv.DictReader((out / "hypothesis_decision_matrix.csv").open()))
    statuses = {row["hypothesis"]: row["status"] for row in decision_rows}
    assert statuses["no_trade_through"] == "rejected_for_sample"
    assert statuses["wrong_time_of_day"] == "inconclusive"
