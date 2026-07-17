from __future__ import annotations

from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


def _intent() -> executor.OrderIntent:
    return executor.OrderIntent(symbol="BTC", is_buy=True, size_btc=0.005, limit_px=65335.0)


def test_live_fill_rows_prefers_tracked_oid() -> None:
    rows = fill_window.live_fill_rows(
        fills=[
            {"coin": "BTC", "oid": 123, "side": "B", "sz": "0.005", "px": "65336", "fee": "0.01", "crossed": False},
            {"coin": "BTC", "oid": 999, "side": "B", "sz": "0.005", "px": "65335", "fee": "0.02", "crossed": False},
        ],
        tracked_oids={"123"},
        intent=_intent(),
        mark_px=65335.5,
        window_id=1,
        user_add_rate=0.0,
    )

    assert len(rows) == 1
    assert rows[0]["price_usdc"] == 65336.0
    assert rows[0]["attribution_status"] == "matched_tracked_oid"
    assert rows[0]["liquidity"] == "maker"


def test_live_fill_rows_falls_back_to_price_size_without_oid() -> None:
    rows = fill_window.live_fill_rows(
        fills=[
            {"coin": "BTC", "dir": "Open Long", "sz": "0.00136", "px": "65335", "fee": "0.013328", "time": 1784122080000},
            {"coin": "BTC", "dir": "Open Long", "sz": "0.00364", "px": "65335", "fee": "0.035672", "time": 1784122080000},
            {"coin": "BTC", "dir": "Open Long", "sz": "0.005", "px": "65336", "fee": "0.01", "time": 1784122080000},
        ],
        tracked_oids={"123"},
        intent=_intent(),
        mark_px=65335.5,
        window_id=1,
        user_add_rate=0.0,
    )

    assert len(rows) == 2
    assert sum(float(row["qty_btc"]) for row in rows) == 0.005
    assert {row["attribution_status"] for row in rows} == {"matched_price_size_without_oid"}
    assert {row["liquidity"] for row in rows} == {"unknown"}
    assert all(row["source_has_liquidity_role"] is False for row in rows)


def test_live_fill_rows_no_fill_when_fallback_exceeds_intent_size() -> None:
    rows = fill_window.live_fill_rows(
        fills=[
            {"coin": "BTC", "dir": "Open Long", "sz": "0.004", "px": "65335", "fee": "0.01"},
            {"coin": "BTC", "dir": "Open Long", "sz": "0.004", "px": "65335", "fee": "0.01"},
        ],
        tracked_oids=set(),
        intent=_intent(),
        mark_px=65335.5,
        window_id=1,
        user_add_rate=0.0,
    )

    assert len(rows) == 1
    assert rows[0]["qty_btc"] == 0.004


def test_cancel_result_mentions_filled_detects_hyperliquid_ambiguous_cancel() -> None:
    assert fill_window.cancel_result_mentions_filled(
        [
            {
                "result": {
                    "response": {
                        "data": {
                            "statuses": [
                                {"error": "Order was never placed, already canceled, or filled. asset=0"}
                            ]
                        }
                    }
                }
            }
        ]
    )


def test_live_fill_ledger_fieldnames_include_attribution_contract() -> None:
    fieldnames = fill_window.live_fill_ledger_fieldnames()

    assert "attribution_status" in fieldnames
    assert "attribution_source" in fieldnames
    assert "source_oid_present" in fieldnames
    assert "source_has_liquidity_role" in fieldnames


def test_fill_rows_preserve_window_and_attempt_key() -> None:
    rows = fill_window.live_fill_rows(
        fills=[
            {
                "coin": "BTC",
                "oid": 123,
                "side": "B",
                "sz": "0.005",
                "px": "65336",
                "fee": "0.01",
                "crossed": False,
            }
        ],
        tracked_oids={"123"},
        intent=_intent(),
        mark_px=65335.5,
        window_id=2,
        attempt_id=3,
        task_id="0717T007",
        user_add_rate=0.0,
    )

    assert rows[0]["source_window"] == "window_02"
    assert rows[0]["window_id"] == "window_02"
    assert rows[0]["attempt_id"] == 3
    assert rows[0]["attempt_key"] == "0717T007:window_02:attempt_3"


def test_two_windows_do_not_share_attempt_keys() -> None:
    keys = {
        fill_window.artifact_attempt_key(task_id="0717T007", window_id=window_id, attempt_id=1)
        for window_id in (1, 2)
    }

    assert keys == {
        "0717T007:window_01:attempt_1",
        "0717T007:window_02:attempt_1",
    }


def test_fill_liquidity_role_evidence_rows_gate_fee_pnl_role() -> None:
    rows = fill_window.fill_liquidity_role_evidence_rows(
        [
            {
                "source_window": "window_1",
                "fill_id": "a",
                "liquidity": "maker",
                "source_has_liquidity_role": True,
                "source_oid_present": True,
                "attribution_status": "matched_tracked_oid",
            },
            {
                "source_window": "window_1",
                "fill_id": "b",
                "liquidity": "unknown",
                "source_has_liquidity_role": False,
                "source_oid_present": False,
                "attribution_status": "matched_price_size_without_oid",
            },
        ]
    )

    assert rows[0]["liquidity_role_status"] == "confirmed_maker"
    assert rows[0]["fee_pnl_role_gate"] == "pass_role_known"
    assert rows[1]["liquidity_role_status"] == "unknown_liquidity_role"
    assert rows[1]["fee_pnl_role_gate"] == "block_unknown_liquidity_role"
