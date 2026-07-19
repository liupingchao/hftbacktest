from __future__ import annotations

import time

import pytest

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
    assert {row["attribution_status"] for row in rows} == {"matched_unique_time_bounded_fallback"}
    assert {row["liquidity"] for row in rows} == {"unknown"}
    assert all(row["source_has_liquidity_role"] is False for row in rows)


def test_live_fill_rows_fail_closed_when_identical_synthetic_fills_exceed_intent_size() -> None:
    fill_time_ms = int(time.time() * 1000)
    rows = fill_window.live_fill_rows(
        fills=[
            {"coin": "BTC", "dir": "Open Long", "sz": "0.004", "px": "65335", "fee": "0.01", "time": fill_time_ms},
            {"coin": "BTC", "dir": "Open Long", "sz": "0.004", "px": "65335", "fee": "0.01", "time": fill_time_ms},
        ],
        tracked_oids=set(),
        intent=_intent(),
        mark_px=65335.5,
        window_id=1,
        user_add_rate=0.0,
    )

    assert rows == []


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


def test_no_fill_reconciliation_accepts_success_then_redundant_ambiguous_cancel() -> None:
    reconciliation = fill_window.no_fill_reconciliation(
        real_order_endpoint_called=True,
        cancel_results=[
            {
                "method": "cancel",
                "attempt": 1,
                "oid": 123,
                "cloid": "0xabc",
                "result": {
                    "status": "ok",
                    "response": {"data": {"statuses": ["success"]}},
                },
            },
            {
                "method": "cancel_by_cloid",
                "attempt": 1,
                "oid": None,
                "cloid": "0xabc",
                "result": {
                    "status": "ok",
                    "response": {
                        "data": {
                            "statuses": [
                                {
                                    "error": (
                                        "Order was never placed, already canceled, or filled. "
                                        "asset=0"
                                    )
                                }
                            ]
                        }
                    },
                },
            },
        ],
        tracked_refs=[{"attempt": 1, "oid": 123, "cloid": "0xabc"}],
        final_open_orders=[],
        fill_rows=[],
        fill_attribution_summary={
            "attributed_fill_count": 0,
            "unattributed_fill_count": 0,
            "fail_closed_reasons": [],
        },
        user_fills_pullbacks=[{"fill_count": 0, "fills": []}],
        post_state={"assetPositions": []},
        shutdown_status="pass",
    )

    assert reconciliation["status"] == "no_fill_reconciled"
    assert reconciliation["mechanism_status"] == "pass"
    assert reconciliation["authoritative_cancel_success_observed"] is True
    assert reconciliation["ambiguous_redundant_cancel_count"] == 1
    assert reconciliation["ambiguous_redundant_cancel_tolerated"] is True
    cancel_reconciliation = reconciliation["cancel_reference_reconciliation"]
    assert cancel_reconciliation["status"] == "pass"
    assert cancel_reconciliation["proven_reference_count"] == 1
    assert cancel_reconciliation["reference_rows"][0]["ambiguous_generic_count"] == 1


def test_no_fill_reconciliation_fails_without_authoritative_cancel_success() -> None:
    reconciliation = fill_window.no_fill_reconciliation(
        real_order_endpoint_called=True,
        cancel_results=[
            {
                "method": "cancel",
                "attempt": 1,
                "oid": 123,
                "cloid": "",
                "result": {
                    "status": "ok",
                    "response": {
                        "data": {
                            "statuses": [
                                {
                                    "error": (
                                        "Order was never placed, already canceled, or filled. "
                                        "asset=0"
                                    )
                                }
                            ]
                        }
                    },
                },
            }
        ],
        tracked_refs=[{"attempt": 1, "oid": 123}],
        final_open_orders=[],
        fill_rows=[],
        fill_attribution_summary={
            "attributed_fill_count": 0,
            "unattributed_fill_count": 0,
            "fail_closed_reasons": [],
        },
        user_fills_pullbacks=[{"fill_count": 0, "fills": []}],
        post_state={"assetPositions": []},
        shutdown_status="pass",
    )

    assert reconciliation["status"] == "no_fill_unproven"
    assert reconciliation["mechanism_status"] == "fail_closed"
    assert "authoritative_cancel_success_missing_for_reference" in reconciliation["reasons"]


def _cancel_success(*, attempt: int, oid: int | None = None, cloid: str = "") -> dict:
    return {
        "method": "cancel",
        "attempt": attempt,
        "oid": oid,
        "cloid": cloid,
        "result": {
            "status": "ok",
            "response": {"data": {"statuses": ["success"]}},
        },
    }


def _ambiguous_cancel(*, attempt: int, oid: int | None = None, cloid: str = "") -> dict:
    return {
        "method": "cancel",
        "attempt": attempt,
        "oid": oid,
        "cloid": cloid,
        "result": {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "error": (
                                "Order was never placed, already canceled, or filled. "
                                "asset=0"
                            )
                        }
                    ]
                }
            },
        },
    }


def test_cancel_reconciliation_does_not_reuse_success_across_attempts() -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[
            {"attempt": 1, "oid": 101, "cloid": "a"},
            {"attempt": 2, "oid": 202, "cloid": "b"},
        ],
        cancel_results=[
            _cancel_success(attempt=1, oid=101),
            _ambiguous_cancel(attempt=2, cloid="b"),
        ],
    )

    assert reconciliation["status"] == "fail_closed"
    rows = {row["attempt"]: row for row in reconciliation["reference_rows"]}
    assert rows[1]["status"] == "pass"
    assert rows[2]["status"] == "fail_closed"
    assert rows[2]["authoritative_success_count"] == 0


def test_cancel_reconciliation_proves_two_references_by_oid_and_cloid() -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[
            {"attempt": 1, "oid": 101, "cloid": "a"},
            {"attempt": 2, "oid": 202, "cloid": "b"},
        ],
        cancel_results=[
            _cancel_success(attempt=1, oid=101),
            _cancel_success(attempt=2, cloid="b"),
        ],
    )

    assert reconciliation["status"] == "pass"
    assert reconciliation["tracked_reference_count"] == 2
    assert reconciliation["proven_reference_count"] == 2
    assert reconciliation["all_references_proven"] is True
    assert reconciliation["unmapped_cancel_evidence_count"] == 0


@pytest.mark.parametrize(
    ("tracked_refs", "cancel_results", "expected_reason"),
    [
        (
            [{"attempt": 1, "oid": 101, "cloid": "a"}],
            [_cancel_success(attempt=1, oid=999)],
            "cancel_result_unknown_target",
        ),
        (
            [{"attempt": 1, "oid": 101, "cloid": "a"}],
            [
                {
                    **_cancel_success(attempt=1, oid=101),
                    "oid": None,
                    "cloid": "",
                }
            ],
            "cancel_result_target_missing",
        ),
        (
            [
                {"attempt": 1, "oid": 101, "cloid": "a"},
                {"attempt": 1, "oid": 101, "cloid": "a"},
            ],
            [_cancel_success(attempt=1, oid=101)],
            "cancel_result_ambiguous_target",
        ),
        (
            [
                {"attempt": 1, "oid": 101, "cloid": "a"},
                {"attempt": 1, "oid": 202, "cloid": "b"},
            ],
            [_cancel_success(attempt=1, oid=101, cloid="b")],
            "cancel_result_ambiguous_target",
        ),
    ],
)
def test_cancel_reconciliation_fails_closed_on_unmapped_or_ambiguous_evidence(
    tracked_refs: list[dict],
    cancel_results: list[dict],
    expected_reason: str,
) -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=tracked_refs,
        cancel_results=cancel_results,
    )

    assert reconciliation["status"] == "fail_closed"
    assert expected_reason in reconciliation["reasons"]


def test_live_fill_ledger_fieldnames_include_attribution_contract() -> None:
    fieldnames = fill_window.live_fill_ledger_fieldnames()

    assert "attribution_status" in fieldnames
    assert "attribution_source" in fieldnames
    assert "source_oid_present" in fieldnames
    assert "source_has_liquidity_role" in fieldnames
    assert "attribution_interval_start_ms" in fieldnames
    assert "attribution_interval_end_ms" in fieldnames
    assert "duplicate_pullback_count" in fieldnames
    assert "ambiguity_reason" in fieldnames


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


def _ledger() -> fill_window.LiveFillLedger:
    return fill_window.LiveFillLedger(task_id="0717T008", window_id=1, pullback_grace_ms=100)


def _register(
    ledger: fill_window.LiveFillLedger,
    *,
    attempt_id: int,
    start_ms: int,
    end_ms: int,
    terminal_ms: int,
    oid: int | None = None,
    cloid: str | None = None,
    size_btc: float = 0.005,
) -> str:
    intent = executor.OrderIntent(
        symbol="BTC",
        is_buy=True,
        size_btc=size_btc,
        limit_px=65335.0,
        cloid=cloid or f"attempt-{attempt_id}",
    )
    refs = []
    if oid is not None:
        refs.append({"oid": oid})
    if cloid is not None:
        refs.append({"cloid": cloid})
    return ledger.register_attempt(
        attempt_id=attempt_id,
        intent=intent,
        submit_start_ms=start_ms,
        submit_end_ms=end_ms,
        tracked_refs=refs,
        terminal_end_ms=terminal_ms,
    )


def test_repeated_pullback_same_fill_is_idempotent() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    fill = {
        "fillId": "native-1",
        "coin": "BTC",
        "oid": 101,
        "side": "B",
        "sz": "0.005",
        "px": "65335",
        "fee": "0.01",
        "time": 1_200,
        "crossed": False,
    }

    ledger.ingest(fills=[fill], mark_px=65335.5, user_add_rate=0.0, pullback_phase="after_response", observed_end_ms=1_300)
    ledger.ingest(fills=[fill], mark_px=65336.5, user_add_rate=0.0, pullback_phase="finalize", observed_end_ms=1_600)

    rows = ledger.attributed_rows()
    assert len(rows) == 1
    assert rows[0]["qty_btc"] == 0.005
    assert rows[0]["fee_usdc"] == 0.01
    assert rows[0]["duplicate_pullback_count"] == 1
    assert rows[0]["pullback_phases"] == "after_response|finalize"


def test_mark_price_change_does_not_duplicate_fill() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    fill = {"fillId": "native-2", "coin": "BTC", "oid": 101, "side": "B", "sz": "0.005", "px": "65335", "time": 1_200}

    ledger.ingest(fills=[fill], mark_px=65335.5, user_add_rate=0.0, pullback_phase="a", observed_end_ms=1_300)
    ledger.ingest(fills=[fill], mark_px=65340.0, user_add_rate=0.0, pullback_phase="b", observed_end_ms=1_600)

    rows = ledger.attributed_rows()
    assert len(rows) == 1
    assert rows[0]["mark_price_usdc"] == 65340.0


def test_same_price_two_attempts_use_time_interval() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_300)
    _register(ledger, attempt_id=2, start_ms=2_000, end_ms=2_100, terminal_ms=2_300)
    fill = {"coin": "BTC", "dir": "Open Long", "sz": "0.005", "px": "65335", "time": 2_200}

    ledger.ingest(fills=[fill], mark_px=65335.5, user_add_rate=0.0, pullback_phase="finalize", observed_end_ms=2_400)

    assert ledger.attributed_rows()[0]["attempt_key"] == "0717T008:window_01:attempt_2"


def test_fill_cannot_be_claimed_by_two_attempts() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    fill = {"fillId": "native-3", "coin": "BTC", "oid": 101, "side": "B", "sz": "0.005", "px": "65335", "time": 1_200}
    ledger.ingest(fills=[fill], mark_px=65335.5, user_add_rate=0.0, pullback_phase="a", observed_end_ms=1_300)
    _register(ledger, attempt_id=2, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    ledger.ingest(fills=[fill], mark_px=65335.5, user_add_rate=0.0, pullback_phase="b", observed_end_ms=1_400)

    rows = ledger.attributed_rows()
    assert len(rows) == 1
    assert rows[0]["attempt_key"] == "0717T008:window_01:attempt_1"


def test_ambiguous_fallback_remains_unattributed() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500)
    _register(ledger, attempt_id=2, start_ms=1_050, end_ms=1_150, terminal_ms=1_550)
    fill = {"coin": "BTC", "dir": "Open Long", "sz": "0.001", "px": "65335", "time": 1_200}

    ledger.ingest(fills=[fill], mark_px=65335.5, user_add_rate=0.0, pullback_phase="finalize", observed_end_ms=1_600)

    assert ledger.attributed_rows() == []
    evidence = ledger.evidence_rows()
    assert evidence[0]["attribution_status"] == "ambiguous_unattributed_fill"
    assert evidence[0]["ambiguity_reason"] == "multiple_candidate_attempts"


def test_oid_match_precedes_fallback() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    _register(ledger, attempt_id=2, start_ms=1_050, end_ms=1_150, terminal_ms=1_550)
    fill = {"coin": "BTC", "oid": 101, "side": "B", "sz": "0.001", "px": "65335", "time": 1_200}

    ledger.ingest(fills=[fill], mark_px=65335.5, user_add_rate=0.0, pullback_phase="finalize", observed_end_ms=1_600)

    row = ledger.attributed_rows()[0]
    assert row["attempt_key"] == "0717T008:window_01:attempt_1"
    assert row["attribution_status"] == "matched_tracked_oid"


def test_cloid_match_precedes_fallback() -> None:
    ledger = _ledger()
    _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_500,
        cloid="tracked-cloid",
    )
    fill = {
        "fillId": "cloid-fill",
        "coin": "BTC",
        "cloid": "tracked-cloid",
        "side": "B",
        "sz": "0.001",
        "px": "65335",
        "time": 1_200,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65335.5,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    row = ledger.attributed_rows()[0]
    assert row["attempt_key"] == "0717T008:window_01:attempt_1"
    assert row["attribution_status"] == "matched_tracked_cloid"


def test_untracked_oid_does_not_fall_back_to_matching_price_and_time() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    fill = {
        "fillId": "foreign-order-fill",
        "coin": "BTC",
        "oid": 999,
        "side": "B",
        "sz": "0.001",
        "px": "65335",
        "time": 1_200,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65335.5,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    assert ledger.attributed_rows() == []
    assert ledger.evidence_rows()[0]["ambiguity_reason"] == "untracked_fill_oid"


def test_fallback_rejects_fills_outside_attempt_interval() -> None:
    cases = [
        ("before", 900, "fill_before_attempt_interval"),
        ("after", 1_601, "fill_after_attempt_terminal_interval"),
    ]
    for fill_id, fill_time_ms, expected_reason in cases:
        ledger = _ledger()
        _register(
            ledger,
            attempt_id=1,
            start_ms=1_000,
            end_ms=1_100,
            terminal_ms=1_500,
        )
        ledger.ingest(
            fills=[
                {
                    "fillId": fill_id,
                    "coin": "BTC",
                    "side": "B",
                    "sz": "0.001",
                    "px": "65335",
                    "time": fill_time_ms,
                }
            ],
            mark_px=65335.5,
            user_add_rate=0.0,
            pullback_phase="finalize",
            observed_end_ms=1_700,
        )

        assert ledger.attributed_rows() == []
        assert ledger.evidence_rows()[0]["ambiguity_reason"] == expected_reason


def test_partial_fills_share_attempt_without_exceeding_size() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    fills = [
        {"tradeId": "part-1", "coin": "BTC", "oid": 101, "side": "B", "sz": "0.002", "px": "65335", "time": 1_200},
        {"tradeId": "part-2", "coin": "BTC", "oid": 101, "side": "B", "sz": "0.003", "px": "65335", "time": 1_250},
        {"tradeId": "part-3", "coin": "BTC", "oid": 101, "side": "B", "sz": "0.001", "px": "65335", "time": 1_260},
    ]

    ledger.ingest(fills=fills, mark_px=65335.5, user_add_rate=0.0, pullback_phase="finalize", observed_end_ms=1_600)

    assert sum(float(row["qty_btc"]) for row in ledger.attributed_rows()) == 0.005
    assert ledger.summary()["unattributed_fill_count"] == 1
    assert ledger.evidence_rows()[-1]["ambiguity_reason"] == "attempt_quantity_cap_exceeded"


def test_conflicting_same_fill_id_fails_closed() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    first = {"fillId": "native-conflict", "coin": "BTC", "oid": 101, "side": "B", "sz": "0.005", "px": "65335", "time": 1_200}
    conflicting = {**first, "sz": "0.004"}

    ledger.ingest(fills=[first], mark_px=65335.5, user_add_rate=0.0, pullback_phase="a", observed_end_ms=1_300)
    ledger.ingest(fills=[conflicting], mark_px=65335.5, user_add_rate=0.0, pullback_phase="b", observed_end_ms=1_400)
    ledger.ingest(fills=[first], mark_px=65336.0, user_add_rate=0.0, pullback_phase="c", observed_end_ms=1_450)

    assert ledger.attributed_rows() == []
    assert ledger.summary()["attributed_fill_count"] == 0
    assert ledger.summary()["attributed_qty_btc"] == 0
    assert ledger.summary()["attributed_fee_usdc"] == 0
    assert ledger.summary()["unattributed_fill_count"] == 1
    assert ledger.summary()["fail_closed_reasons"]
    assert any(row["attribution_status"] == "conflicting_same_fill_id" for row in ledger.evidence_rows())


def test_same_pullback_synthetic_id_collision_fails_closed() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    fill = {
        "coin": "BTC",
        "oid": 101,
        "side": "B",
        "sz": "0.002",
        "px": "65335",
        "fee": "0.004",
        "time": 1_200,
    }

    ledger.ingest(
        fills=[fill, dict(fill)],
        mark_px=65335.5,
        user_add_rate=0.0,
        pullback_phase="single_pullback",
        observed_end_ms=1_300,
    )
    ledger.ingest(
        fills=[fill],
        mark_px=65336.0,
        user_add_rate=0.0,
        pullback_phase="later_pullback",
        observed_end_ms=1_400,
    )

    assert ledger.attributed_rows() == []
    evidence = ledger.evidence_rows()
    assert len(evidence) == 1
    assert evidence[0]["attribution_status"] == "ambiguous_duplicate_without_unique_fill_id"
    assert ledger.summary()["fail_closed_reasons"]


def test_later_pullback_synthetic_collision_fails_closed() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_100, terminal_ms=1_500, oid=101)
    fill = {
        "coin": "BTC",
        "oid": 101,
        "side": "B",
        "sz": "0.002",
        "px": "65335",
        "fee": "0.004",
        "time": 1_200,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65335.5,
        user_add_rate=0.0,
        pullback_phase="first_pullback",
        observed_end_ms=1_300,
    )
    ledger.ingest(
        fills=[fill, dict(fill)],
        mark_px=65336.0,
        user_add_rate=0.0,
        pullback_phase="later_pullback",
        observed_end_ms=1_400,
    )
    ledger.ingest(
        fills=[fill],
        mark_px=65336.5,
        user_add_rate=0.0,
        pullback_phase="post_quarantine_pullback",
        observed_end_ms=1_450,
    )

    assert ledger.attributed_rows() == []
    evidence = ledger.evidence_rows()
    assert len(evidence) == 1
    assert evidence[0]["attribution_status"] == "ambiguous_duplicate_without_unique_fill_id"
    assert ledger.summary()["fail_closed_reasons"]


def test_repeated_pullback_refreshes_terminal_interval() -> None:
    ledger = _ledger()
    attempt_key = _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_200,
        oid=101,
    )
    fill = {
        "fillId": "refresh-terminal",
        "coin": "BTC",
        "oid": 101,
        "side": "B",
        "sz": "0.005",
        "px": "65335",
        "time": 1_150,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65335.5,
        user_add_rate=0.0,
        pullback_phase="before_cancel",
        observed_end_ms=1_300,
    )
    ledger.update_attempt_terminal(attempt_key, terminal_end_ms=1_500)
    ledger.ingest(
        fills=[fill],
        mark_px=65336.0,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    assert ledger.attributed_rows()[0]["attribution_interval_end_ms"] == 1_600


def test_offline_two_attempt_repeated_pullback_fixture() -> None:
    ledger = _ledger()
    _register(ledger, attempt_id=1, start_ms=1_000, end_ms=1_050, terminal_ms=1_300, oid=101)
    _register(ledger, attempt_id=2, start_ms=2_000, end_ms=2_050, terminal_ms=2_300, oid=202)
    first_fill = {
        "fillId": "fixture-first",
        "coin": "BTC",
        "oid": 101,
        "side": "B",
        "sz": "0.005",
        "px": "65335",
        "fee": "0.01",
        "time": 1_200,
        "crossed": False,
    }
    later_partial = {
        "fillId": "fixture-later-partial",
        "coin": "BTC",
        "oid": 202,
        "side": "B",
        "sz": "0.002",
        "px": "65335",
        "fee": "0.004",
        "time": 2_200,
        "crossed": False,
    }
    ambiguous = {
        "fillId": "fixture-ambiguous",
        "coin": "BTC",
        "side": "B",
        "sz": "0.001",
        "px": "65335",
        "fee": "0.002",
        "time": 2_500,
    }

    ledger.ingest(
        fills=[first_fill],
        mark_px=65335.5,
        user_add_rate=0.0,
        pullback_phase="after_attempt_1",
        observed_end_ms=1_350,
    )
    ledger.ingest(
        fills=[first_fill, later_partial, ambiguous],
        mark_px=65336.0,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=2_600,
    )

    attributed = ledger.attributed_rows()
    evidence = ledger.evidence_rows()
    assert len(attributed) == 2
    assert sum(float(row["qty_btc"]) for row in attributed) == 0.007
    assert sum(float(row["fee_usdc"]) for row in attributed) == 0.014
    assert {row["attempt_id"] for row in attributed} == {1, 2}
    assert next(row for row in attributed if row["attempt_id"] == 1)["duplicate_pullback_count"] == 1
    unattributed = [row for row in evidence if row["attribution_status"] == "ambiguous_unattributed_fill"]
    assert len(unattributed) == 1
    assert unattributed[0]["fill_id"] == fill_window.stable_fill_id(ambiguous)


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
