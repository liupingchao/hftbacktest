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


def test_flow_aware_side_scoring_prefers_buy_and_penalizes_sell() -> None:
    precision = executor.PrecisionFacts(symbol="BTC", sz_decimals=5, tick_size=1.0, lot_size=0.00001, mid_px=65000.0, source="unit")
    l2 = {"levels": [[{"px": "65000", "sz": "0.02", "n": 2}], [{"px": "65002", "sz": "0.02", "n": 2}]]}
    public_summary = {
        "by_side": {
            "buy": {"candidate_count": 10, "strict_trade_through_candidate_count": 7, "public_depletion_candidate_count": 4, "adverse_lost_touch_candidate_count": 2},
            "sell": {"candidate_count": 10, "strict_trade_through_candidate_count": 1, "public_depletion_candidate_count": 0, "adverse_lost_touch_candidate_count": 8},
        }
    }

    decision = window.select_flow_aware_side(
        l2_snapshot=l2,
        precision=precision,
        quote_offset_ticks=0,
        window_id=1,
        attempt_id=1,
        max_order_size_btc=0.00999,
        max_top_depth_multiple=500,
        public_flow_summary=public_summary,
    )

    assert decision["allowed"] is True
    assert decision["selected_side"] == "buy"
    assert decision["scores"]["buy"]["score"] > decision["scores"]["sell"]["score"]


def test_flow_aware_rejects_sell_unless_materially_better() -> None:
    precision = executor.PrecisionFacts(symbol="BTC", sz_decimals=5, tick_size=1.0, lot_size=0.00001, mid_px=65000.0, source="unit")
    l2 = {"levels": [[{"px": "65000", "sz": "3.0", "n": 2}], [{"px": "65002", "sz": "0.02", "n": 2}]]}

    decision = window.select_flow_aware_side(
        l2_snapshot=l2,
        precision=precision,
        quote_offset_ticks=0,
        window_id=1,
        attempt_id=1,
        max_order_size_btc=0.00999,
        max_top_depth_multiple=500,
        public_flow_summary={},
    )

    assert decision["allowed"] is False
    assert decision["skip_reason"] == "sell_without_public_flow_support"


def test_dynamic_fresh_touch_size_caps_and_floors() -> None:
    decision = window.dynamic_fresh_touch_size(
        bucket="quality_a",
        recent_same_side_at_or_through_qty_btc=0.04,
        lot_size=0.00001,
    )

    assert decision["status"] == "pass"
    assert decision["floored_size_btc"] == 0.005
    assert decision["bucket_cap_btc"] == 0.005


def test_fresh_touch_quality_b_uses_smaller_hold() -> None:
    decision = window.classify_fresh_touch_quality(
        side="buy",
        top_depth_multiple=75.0,
        same_side_top_order_count=8,
        strict_through_supported=True,
        touch_freshness_present=True,
        recent_same_side_at_or_through_qty_btc=0.01,
    )

    assert decision["allowed"] is True
    assert decision["quality_bucket"] == "quality_b"
    assert decision["hold_seconds"] == window.FRESH_TOUCH_QUALITY_B_HOLD_SECONDS


def test_select_fresh_touch_candidate_buy_only_dynamic_size(tmp_path: Path) -> None:
    precision = executor.PrecisionFacts(symbol="BTC", sz_decimals=5, tick_size=1.0, lot_size=0.00001, mid_px=65000.0, source="unit")
    candidates = tmp_path / "candidate_flow_diagnostics.csv"
    candidates.write_text(
        "start_exchange_time_ms,side,quote_px,bid,ask,spread_ticks,order_size_btc,same_side_top_qty_btc,same_side_top_order_count,top_depth_multiple_of_order,hold_seconds,book_updates_in_window,trades_in_window,touch_trade_qty_btc,strict_trade_through_qty_btc,at_or_through_trade_qty_btc,opposite_trade_qty_btc,required_depletion_qty_btc,queue_depletion_multiple,public_depletion_status,first_touch_trade_ms,first_strict_trade_through_ms,quote_aging_status,first_not_touch_ms,first_adverse_lost_touch_ms,window_mid_move_ticks,inference_scope\n"
        "1000,buy,65000,65000,65001,1,0.005,0.02,4,4,3,2,3,0.01,0.01,0.04,0,0.025,1.6,depleted_top_plus_order_proxy,100,200,stayed_touch,,,0,unit\n"
        "1000,sell,65001,65000,65001,1,0.005,0.02,4,4,3,2,3,0.01,0.01,0.04,0,0.025,1.6,depleted_top_plus_order_proxy,100,200,stayed_touch,,,0,unit\n",
        encoding="utf-8",
    )
    public_precheck = {
        "summary": {"last_book_exchange_time_ms": "1500"},
        "diagnosis_manifest": {"output_files": {"candidate_flow_diagnostics": str(candidates)}},
    }
    l2 = {"levels": [[{"px": "65000", "sz": "0.02", "n": 4}], [{"px": "65001", "sz": "1.0", "n": 9}]]}

    decision = window.select_fresh_touch_candidate(
        l2_snapshot=l2,
        precision=precision,
        window_id=1,
        attempt_id=1,
        public_flow_precheck=public_precheck,
        max_order_size_btc=0.005,
    )

    assert decision["allowed"] is True
    assert decision["selected_side"] == "buy"
    assert decision["quality_bucket"] == "quality_a"
    assert decision["intent_size_btc"] == 0.005
    assert all(row["side"] == "buy" for row in decision["candidate_rows"])


def test_select_fresh_touch_recomputes_depth_for_quality_b_size(tmp_path: Path) -> None:
    precision = executor.PrecisionFacts(symbol="BTC", sz_decimals=5, tick_size=1.0, lot_size=0.00001, mid_px=65000.0, source="unit")
    candidates = tmp_path / "candidate_flow_diagnostics.csv"
    candidates.write_text(
        "start_exchange_time_ms,side,quote_px,bid,ask,spread_ticks,order_size_btc,same_side_top_qty_btc,same_side_top_order_count,top_depth_multiple_of_order,hold_seconds,book_updates_in_window,trades_in_window,touch_trade_qty_btc,strict_trade_through_qty_btc,at_or_through_trade_qty_btc,opposite_trade_qty_btc,required_depletion_qty_btc,queue_depletion_multiple,public_depletion_status,first_touch_trade_ms,first_strict_trade_through_ms,quote_aging_status,first_not_touch_ms,first_adverse_lost_touch_ms,window_mid_move_ticks,inference_scope\n"
        "1000,buy,65000,65000,65001,1,0.005,0.15,10,30,3,2,3,0.01,0.01,0.04,0,0.025,1.6,depleted_top_plus_order_proxy,100,200,stayed_touch,,,0,unit\n",
        encoding="utf-8",
    )
    public_precheck = {
        "summary": {"last_book_exchange_time_ms": "1500"},
        "diagnosis_manifest": {"output_files": {"candidate_flow_diagnostics": str(candidates)}},
    }
    l2 = {"levels": [[{"px": "65000", "sz": "0.15", "n": 10}], [{"px": "65001", "sz": "1.0", "n": 9}]]}

    decision = window.select_fresh_touch_candidate(
        l2_snapshot=l2,
        precision=precision,
        window_id=1,
        attempt_id=1,
        public_flow_precheck=public_precheck,
        max_order_size_btc=0.005,
    )

    assert decision["allowed"] is True
    assert decision["quality_bucket"] == "quality_b"
    assert decision["intent_size_btc"] == 0.002
    assert decision["selected_candidate"]["top_depth_multiple_of_order"] == 75.0


def test_quote_aging_guard_detects_adverse_lost_touch() -> None:
    intent = executor.OrderIntent(symbol="BTC", is_buy=True, size_btc=0.00999, limit_px=65000.0)

    guard = window.quote_aging_guard(
        intent=intent,
        pre_bid=65000.0,
        pre_ask=65001.0,
        post_bid=64999.0,
        post_ask=65000.0,
        tick_size=1.0,
    )

    assert guard["status"] == "cancel_requote"
    assert "adverse_drift" in guard["reason"]


def test_flow_aware_precheck_fail_writes_no_order_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(window, "run_public_flow_precheck", lambda **kwargs: {"status": "fail_closed", "reason": "ssl_eof", "summary": {}})

    manifest = window.run_window(
        output_dir=tmp_path,
        env_file=tmp_path / ".env",
        window_id=1,
        wait_seconds=1,
        quote_offset_ticks=0,
        requote_attempts=1,
        quote_hold_seconds=1,
        side_policy="flow_aware",
        max_order_size=0.00999,
    )

    assert manifest["final_recommendation"] == window.BLOCKED_RECOMMENDATION
    assert manifest["real_order_endpoint_called"] is False
    assert manifest["private_endpoint_called"] is False
    assert manifest["flow_guard_status"] == "public_flow_precheck_blocked"
    assert (tmp_path / "quote_attempt_matrix.csv").exists()


def test_fresh_touch_precheck_fail_writes_required_no_order_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(window, "run_public_flow_precheck", lambda **kwargs: {"status": "fail_closed", "reason": "ssl_eof", "summary": {}})

    manifest = window.run_window(
        output_dir=tmp_path,
        env_file=tmp_path / ".env",
        window_id=1,
        wait_seconds=1,
        quote_offset_ticks=0,
        requote_attempts=1,
        quote_hold_seconds=1,
        side_policy="fresh_touch",
        max_order_size=0.005,
    )

    assert manifest["final_recommendation"] == window.BLOCKED_RECOMMENDATION
    assert manifest["real_order_endpoint_called"] is False
    assert manifest["private_endpoint_called"] is False
    assert manifest["fresh_touch_guard_status"] == "public_flow_precheck_blocked"
    assert (tmp_path / "touch_freshness_matrix.csv").exists()
    assert (tmp_path / "dynamic_size_decision_matrix.csv").exists()
    assert (tmp_path / "session_side_eligibility.csv").exists()
    assert (tmp_path / "time_gate_decision_matrix.csv").exists()


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


def test_run_window_flow_aware_with_mocked_client_uses_smaller_size(tmp_path: Path, monkeypatch) -> None:
    class MockInfo:
        def l2_snapshot(self, name: str):
            return {"levels": [[{"px": "65000", "sz": "0.02", "n": 1}], [{"px": "65001", "sz": "1.0", "n": 8}]]}

        def user_fees(self, address: str):
            return {"userAddRate": "0.0002"}

        def user_fills_by_time(self, address: str, start_time: int, end_time: int, aggregate_by_time: bool = False):
            return [{"oid": 618001000, "side": "B", "sz": "0.00999", "px": "65000", "crossed": False, "fee": "0.13"}]

    class MockClient(executor.MockHyperliquidClient):
        account_address = "0x" + "1" * 40
        info = MockInfo()

        def all_mids(self):
            return {"BTC": "65000.5"}

        def meta(self):
            return {"universe": [{"name": "BTC", "szDecimals": 5}]}

        def user_state(self, address=None):
            return {"assetPositions": [{"position": {"coin": "BTC", "szi": "0.00999"}}]}

        def open_orders(self, address=None):
            return []

    public_summary = {
        "by_side": {
            "buy": {"candidate_count": 10, "strict_trade_through_candidate_count": 7, "public_depletion_candidate_count": 4, "adverse_lost_touch_candidate_count": 1},
            "sell": {"candidate_count": 10, "strict_trade_through_candidate_count": 1, "public_depletion_candidate_count": 0, "adverse_lost_touch_candidate_count": 8},
        }
    }
    monkeypatch.setattr(window, "run_public_flow_precheck", lambda **kwargs: {"status": "pass", "reason": "", "summary": public_summary})
    monkeypatch.setattr(executor, "load_env_file", lambda path: {"loaded_keys": []})
    monkeypatch.setattr(executor, "build_live_client_from_env", lambda: MockClient())

    manifest = window.run_window(
        output_dir=tmp_path,
        env_file=tmp_path / ".env",
        window_id=1,
        wait_seconds=1,
        quote_offset_ticks=0,
        requote_attempts=1,
        quote_hold_seconds=1,
        side_policy="flow_aware",
        max_order_size=0.00999,
    )

    rows = (tmp_path / "order_intent_audit.csv").read_text(encoding="utf-8")
    assert manifest["final_recommendation"] == window.READY_RECOMMENDATION
    assert manifest["side_policy"] == "flow_aware"
    assert "0.00999" in rows
    assert (tmp_path / "flow_side_score_matrix.csv").exists()
    assert (tmp_path / "quote_aging_guard_matrix.csv").exists()


def test_run_window_fresh_touch_with_mocked_client_caps_size_and_writes_matrices(tmp_path: Path, monkeypatch) -> None:
    class MockInfo:
        def l2_snapshot(self, name: str):
            return {"levels": [[{"px": "65000", "sz": "0.02", "n": 4}], [{"px": "65001", "sz": "1.0", "n": 8}]]}

        def user_fees(self, address: str):
            return {"userAddRate": "0.0002"}

        def user_fills_by_time(self, address: str, start_time: int, end_time: int, aggregate_by_time: bool = False):
            return [{"oid": 618001000, "side": "B", "sz": "0.005", "px": "65000", "crossed": False, "fee": "0.065"}]

    class MockClient(executor.MockHyperliquidClient):
        account_address = "0x" + "1" * 40
        info = MockInfo()

        def all_mids(self):
            return {"BTC": "65000.5"}

        def meta(self):
            return {"universe": [{"name": "BTC", "szDecimals": 5}]}

        def user_state(self, address=None):
            return {"assetPositions": [{"position": {"coin": "BTC", "szi": "0.005"}}]}

        def open_orders(self, address=None):
            return []

    candidates = tmp_path / "candidate_flow_diagnostics.csv"
    candidates.write_text(
        "start_exchange_time_ms,side,quote_px,bid,ask,spread_ticks,order_size_btc,same_side_top_qty_btc,same_side_top_order_count,top_depth_multiple_of_order,hold_seconds,book_updates_in_window,trades_in_window,touch_trade_qty_btc,strict_trade_through_qty_btc,at_or_through_trade_qty_btc,opposite_trade_qty_btc,required_depletion_qty_btc,queue_depletion_multiple,public_depletion_status,first_touch_trade_ms,first_strict_trade_through_ms,quote_aging_status,first_not_touch_ms,first_adverse_lost_touch_ms,window_mid_move_ticks,inference_scope\n"
        "1000,buy,65000,65000,65001,1,0.005,0.02,4,4,3,2,3,0.01,0.01,0.04,0,0.025,1.6,depleted_top_plus_order_proxy,100,200,stayed_touch,,,0,unit\n",
        encoding="utf-8",
    )
    public_summary = {
        "last_book_exchange_time_ms": "1500",
        "utc_hours": ["10"],
        "by_side": {
            "buy": {"candidate_count": 1, "strict_trade_through_candidate_count": 1, "public_depletion_candidate_count": 1},
            "sell": {"candidate_count": 0, "strict_trade_through_candidate_count": 0, "public_depletion_candidate_count": 0},
        },
    }
    monkeypatch.setattr(
        window,
        "run_public_flow_precheck",
        lambda **kwargs: {"status": "pass", "reason": "", "summary": public_summary, "diagnosis_manifest": {"output_files": {"candidate_flow_diagnostics": str(candidates)}}},
    )
    monkeypatch.setattr(executor, "load_env_file", lambda path: {"loaded_keys": []})
    monkeypatch.setattr(executor, "build_live_client_from_env", lambda: MockClient())

    manifest = window.run_window(
        output_dir=tmp_path,
        env_file=tmp_path / ".env",
        window_id=1,
        wait_seconds=3,
        quote_offset_ticks=0,
        requote_attempts=1,
        quote_hold_seconds=1,
        side_policy="fresh_touch",
        max_order_size=0.005,
    )

    order_rows = (tmp_path / "order_intent_audit.csv").read_text(encoding="utf-8")
    assert manifest["final_recommendation"] == window.READY_RECOMMENDATION
    assert manifest["policy_version"] == window.FRESH_TOUCH_POLICY_VERSION
    assert manifest["fresh_touch_submitted_count"] == 1
    assert "0.005" in order_rows
    assert (tmp_path / "touch_freshness_matrix.csv").exists()
    assert (tmp_path / "dynamic_size_decision_matrix.csv").exists()
    assert (tmp_path / "session_side_eligibility.csv").exists()
    assert (tmp_path / "time_gate_decision_matrix.csv").exists()
