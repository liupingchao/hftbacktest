from __future__ import annotations

import json
from pathlib import Path

from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_loop as loop
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as window
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


def test_build_top_of_book_maker_intent_stays_below_ask_and_under_caps() -> None:
    precision = executor.PrecisionFacts(symbol="BTC", sz_decimals=5, tick_size=1.0, lot_size=0.00001, mid_px=65000.0, source="unit")

    intent = window.build_top_of_book_maker_intent(
        precision=precision,
        bid=65000.0,
        ask=65002.0,
        quote_offset_ticks=1,
        window_id=1,
    )

    assert intent.time_in_force == "Alo"
    assert intent.limit_px == 65001.0
    assert intent.limit_px < 65002.0
    assert intent.size_btc <= executor.MAX_ORDER_SIZE_BTC
    assert intent.notional_usdc <= executor.MAX_ORDER_NOTIONAL_USDC


def test_build_top_of_book_sell_intent_stays_above_bid() -> None:
    precision = executor.PrecisionFacts(symbol="BTC", sz_decimals=5, tick_size=1.0, lot_size=0.00001, mid_px=65000.0, source="unit")

    intent = window.build_top_of_book_maker_intent(
        precision=precision,
        bid=65000.0,
        ask=65002.0,
        quote_offset_ticks=1,
        window_id=1,
        is_buy=False,
    )

    assert intent.time_in_force == "Alo"
    assert intent.limit_px == 65001.0
    assert intent.limit_px > 65000.0
    assert intent.notional_usdc <= executor.MAX_ORDER_NOTIONAL_USDC


def test_build_top_of_book_maker_intent_falls_back_when_spread_one_tick() -> None:
    precision = executor.PrecisionFacts(symbol="BTC", sz_decimals=5, tick_size=1.0, lot_size=0.00001, mid_px=65000.0, source="unit")

    intent = window.build_top_of_book_maker_intent(
        precision=precision,
        bid=65000.0,
        ask=65001.0,
        quote_offset_ticks=1,
        window_id=1,
    )

    assert intent.limit_px == 65000.0
    assert intent.limit_px < 65001.0


def test_live_fill_rows_filters_to_tracked_oid_and_maker() -> None:
    intent = executor.OrderIntent(symbol="BTC", is_buy=True, size_btc=0.01, limit_px=65000.0)
    rows = window.live_fill_rows(
        fills=[
            {"oid": 1, "side": "B", "sz": "0.01", "px": "65000", "crossed": False, "fee": "0.13", "hash": "abc"},
            {"oid": 2, "side": "B", "sz": "0.01", "px": "65000", "crossed": False, "fee": "0.13", "hash": "def"},
        ],
        tracked_oids={"1"},
        intent=intent,
        mark_px=65010.0,
        window_id=1,
        user_add_rate=0.0002,
    )

    assert len(rows) == 1
    assert rows[0]["liquidity"] == "maker"
    assert rows[0]["side"] == "buy"
    assert rows[0]["fee_usdc"] == 0.13


def test_aggregate_live_fills_collects_window_rows(tmp_path: Path) -> None:
    window_dir = tmp_path / "window_1" / "pulled_back_awsserver1"
    window_dir.mkdir(parents=True)
    (window_dir / "live_fill_ledger.csv").write_text(
        "source_window,fill_id,side,qty_btc,price_usdc,intent_price_usdc,mark_price_usdc,fee_usdc,rebate_usdc,liquidity\n"
        "window_1,f1,buy,0.01,65000,65000,65010,0.13,0,maker\n",
        encoding="utf-8",
    )

    aggregate = loop.aggregate_live_fills(tmp_path)

    assert "window_1" in aggregate.read_text(encoding="utf-8")


def test_side_for_attempt_alternates() -> None:
    assert window.side_for_attempt("alternate", 1) is True
    assert window.side_for_attempt("alternate", 2) is False


def test_run_window_manifest_shape_with_mocked_client(tmp_path: Path, monkeypatch) -> None:
    class MockInfo:
        def l2_snapshot(self, name: str):
            return {"levels": [[{"px": "65000", "sz": "1"}], [{"px": "65002", "sz": "1"}]]}

        def user_fees(self, address: str):
            return {"userAddRate": "0.0002"}

        def user_fills_by_time(self, address: str, start_time: int, end_time: int, aggregate_by_time: bool = False):
            return [{"oid": 618001000, "side": "B", "sz": "0.01", "px": "65001", "crossed": False, "fee": "0.13"}]

    class MockClient(executor.MockHyperliquidClient):
        account_address = "0x" + "1" * 40
        info = MockInfo()

        def all_mids(self):
            return {"BTC": "65001"}

        def meta(self):
            return {"universe": [{"name": "BTC", "szDecimals": 5}]}

        def user_state(self, address=None):
            return {"assetPositions": [{"position": {"coin": "BTC", "szi": "0.01"}}]}

        def open_orders(self, address=None):
            return []

    monkeypatch.setattr(executor, "load_env_file", lambda path: {"loaded_keys": []})
    monkeypatch.setattr(executor, "build_live_client_from_env", lambda: MockClient())

    manifest = window.run_window(
        output_dir=tmp_path,
        env_file=tmp_path / ".env",
        window_id=1,
        wait_seconds=1,
        quote_offset_ticks=1,
        requote_attempts=1,
        quote_hold_seconds=1,
        side_policy="buy",
    )

    assert manifest["fill_count"] == 1
    assert manifest["maker_fill_count"] == 1
    assert manifest["final_recommendation"] == window.READY_RECOMMENDATION
    assert json.loads((tmp_path / "executor_manifest.json").read_text(encoding="utf-8"))["order_submission_attempted"] is True
    assert (tmp_path / "quote_attempt_matrix.csv").exists()
