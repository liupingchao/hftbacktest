from __future__ import annotations

import copy
import time

import pytest

from examples.hyperliquid import cross_exchange_t024_same_window_acceptance as acceptance
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


def _intent() -> executor.OrderIntent:
    return executor.OrderIntent(symbol="BTC", is_buy=True, size_btc=0.005, limit_px=65335.0)


def test_live_fill_rows_prefers_tracked_oid() -> None:
    rows = fill_window.live_fill_rows(
        fills=[
            {"coin": "BTC", "oid": 123, "side": "B", "sz": "0.005", "px": "65334", "fee": "0.01", "crossed": False},
            {"coin": "BTC", "oid": 999, "side": "B", "sz": "0.005", "px": "65335", "fee": "0.02", "crossed": False},
        ],
        tracked_oids={"123"},
        intent=_intent(),
        mark_px=65335.5,
        window_id=1,
        user_add_rate=0.0,
    )

    assert len(rows) == 1
    assert rows[0]["price_usdc"] == 65334.0
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


def _submit_response_row(
    *,
    attempt: int,
    side: str,
    cloid_token: str,
    status: dict,
    manager_action: str,
    manager_state: str,
    manager_query_status: str,
) -> dict:
    return {
        "attempt": attempt,
        "attempt_id": attempt,
        "attempt_key": f"0720T027:window_01:attempt_{attempt}",
        "side": side,
        "intent_cloid_token": cloid_token,
        "result": {
            "status": "ok",
            "side": side,
            "response": {
                "type": "order",
                "data": {"statuses": [status]},
            },
            "manager_actions": [
                {
                    "action": manager_action,
                    "state": manager_state,
                    "query_status": manager_query_status,
                    "order_endpoint_called": True,
                    "side": side,
                }
            ],
        },
    }


def test_submit_reject_and_resting_cancel_are_distinct_terminal_paths() -> None:
    buy_cloid_token = fill_window.reference_identity_token(
        "cloid",
        "buy-cloid",
    )
    sell_cloid_token = fill_window.reference_identity_token(
        "cloid",
        "sell-cloid",
    )
    sell_oid_token = fill_window.reference_identity_token("oid", 202)
    tracked_refs = [
        {
            "attempt": 1,
            "cloid": "<redacted>",
            "cloid_token": buy_cloid_token,
        },
        {
            "attempt": 2,
            "oid": "<redacted>",
            "oid_token": sell_oid_token,
            "cloid": "<redacted>",
            "cloid_token": sell_cloid_token,
        },
    ]
    cancel_results = [
        {
            **_cancel_success(attempt=2),
            "oid": "<redacted>",
            "oid_token": sell_oid_token,
            "cloid": "<redacted>",
            "cloid_token": sell_cloid_token,
        }
    ]
    submit_results = [
        _submit_response_row(
            attempt=1,
            side="buy",
            cloid_token=buy_cloid_token,
            status={"error": "Post only order would have immediately matched"},
            manager_action="rejected",
            manager_state="rejected",
            manager_query_status="rejected",
        ),
        _submit_response_row(
            attempt=2,
            side="sell",
            cloid_token=sell_cloid_token,
            status={
                "resting": {
                    "oid": "<redacted>",
                    "oid_token": sell_oid_token,
                    "cloid": "<redacted>",
                    "cloid_token": sell_cloid_token,
                }
            },
            manager_action="submitted",
            manager_state="resting",
            manager_query_status="resting",
        ),
    ]

    producer = fill_window.cancel_reference_reconciliation(
        tracked_refs=tracked_refs,
        cancel_results=cancel_results,
        submit_terminal_results=submit_results,
    )
    independent = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=tracked_refs,
        cancel_results=cancel_results,
        submit_terminal_results=submit_results,
    )

    assert producer == independent
    assert producer["schema_version"] == (
        fill_window.CANCEL_SUBMIT_TERMINAL_RECONCILIATION_SCHEMA_VERSION
    )
    assert producer["status"] == "pass"
    assert producer["submit_response_rejected_count"] == 1
    assert producer["authoritative_success_count"] == 1
    assert {
        (row["attempt"], row["submit_response_rejected_count"])
        for row in producer["reference_rows"]
    } == {(1, 1), (2, 0)}
    assert {
        row["attempt"] for row in producer["cancel_evidence_rows"]
    } == {2}


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_error",
        "multiple_statuses",
        "outer_failure",
        "attempt_mismatch",
        "side_mismatch",
        "attempt_key_mismatch",
        "cloid_mismatch",
        "forged_manager_state",
        "resting_rejected_conflict",
        "duplicate_submit_response",
        "cancel_conflict",
    ],
)
def test_submit_reject_terminal_contract_fails_closed_on_hostile_evidence(
    mutation: str,
) -> None:
    cloid_token = fill_window.reference_identity_token(
        "cloid",
        "buy-cloid",
    )
    tracked_refs = [
        {
            "attempt": 1,
            "cloid": "<redacted>",
            "cloid_token": cloid_token,
        }
    ]
    response = _submit_response_row(
        attempt=1,
        side="buy",
        cloid_token=cloid_token,
        status={"error": "Post only order would have immediately matched"},
        manager_action="rejected",
        manager_state="rejected",
        manager_query_status="rejected",
    )
    if mutation == "empty_error":
        response["result"]["response"]["data"]["statuses"][0]["error"] = ""
    elif mutation == "multiple_statuses":
        response["result"]["response"]["data"]["statuses"].append(
            {"error": "duplicate"}
        )
    elif mutation == "outer_failure":
        response["result"]["status"] = "error"
    elif mutation == "attempt_mismatch":
        response["attempt"] = 2
    elif mutation == "side_mismatch":
        response["side"] = "sell"
    elif mutation == "attempt_key_mismatch":
        response["attempt_key"] = "0720T027:window_01:attempt_2"
    elif mutation == "cloid_mismatch":
        response["intent_cloid_token"] = (
            fill_window.reference_identity_token(
                "cloid",
                "unrelated",
            )
        )
    elif mutation == "forged_manager_state":
        response["result"]["manager_actions"][0]["state"] = "resting"
    elif mutation == "resting_rejected_conflict":
        response["result"]["response"]["data"]["statuses"][0][
            "resting"
        ] = {}
    submit_results = [response]
    cancel_results: list[dict] = []
    if mutation == "duplicate_submit_response":
        submit_results.append(copy.deepcopy(response))
    elif mutation == "cancel_conflict":
        cancel_results.append(
            {
                **_cancel_success(attempt=1),
                "cloid": "<redacted>",
                "cloid_token": cloid_token,
            }
        )

    producer = fill_window.cancel_reference_reconciliation(
        tracked_refs=tracked_refs,
        cancel_results=cancel_results,
        submit_terminal_results=submit_results,
    )
    independent = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=tracked_refs,
        cancel_results=cancel_results,
        submit_terminal_results=submit_results,
    )

    assert producer["status"] == "fail_closed"
    assert independent["status"] == "fail_closed"
    expected_reject_count = (
        1
        if mutation in {"duplicate_submit_response", "cancel_conflict"}
        else 0
    )
    assert producer["submit_response_rejected_count"] == (
        expected_reject_count
    )
    assert independent["submit_response_rejected_count"] == (
        expected_reject_count
    )


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


def _terminal_query(
    *,
    attempt: int,
    status: str,
    oid: int | None = None,
    cloid: str = "",
    method: str = "query_order_by_oid",
    query_status: str | None = None,
    error: str = "",
) -> dict:
    row = {
        "attempt": attempt,
        "method": method,
        "oid": oid,
        "cloid": cloid,
        "query_status": (
            query_status
            if query_status is not None
            else fill_window.terminal_query_status_from_result(
                {"status": status}
            )
        ),
        "result": {"status": status},
    }
    if error:
        row["error"] = error
    return row


def _v4_terminal_contract() -> tuple[list[dict], list[dict], dict]:
    attempts: list[dict] = []
    for direct_round in range(1, 6):
        for method in (
            "query_order_by_oid",
            "query_order_by_cloid",
        ):
            sequence = len(attempts) + 1
            attempts.append(
                {
                    "attempt": 1,
                    "method": method,
                    "oid": 101,
                    "cloid": "a",
                    "query_sequence": sequence,
                    "direct_round": direct_round,
                    "query_started_ms": 1_000 + sequence * 2,
                    "query_ended_ms": 1_001 + sequence * 2,
                    "query_status": "unknown",
                    "result": {"status": "unknownOid"},
                }
            )
    history = {
        "attempt": 1,
        "method": "historical_orders",
        "oid": 101,
        "cloid": "a",
        "query_sequence": 11,
        "query_started_ms": 1_022,
        "query_ended_ms": 1_023,
        "query_status": "cancel_confirmed",
        "result": {
            "status": "historical_orders",
            "orders": [
                {
                    "order": {"oid": 999, "cloid": "foreign"},
                    "status": "canceled",
                },
                {
                    "order": {"oid": 101, "cloid": "a"},
                    "status": "canceled",
                },
            ],
        },
    }
    attempts.append(history)
    canonical = {**history, "source_query_sequence": 11}
    budget = {
        "budget_seconds": 5.0,
        "retry_seconds": 0.25,
        "started_monotonic": 100.0,
        "ended_monotonic": 100.5,
        "elapsed_seconds": 0.5,
        "max_direct_rounds": 5,
        "direct_rounds_used": 5,
        "direct_query_attempt_count": 10,
        "historical_fallback_attempt_count": 1,
        "historical_fallback_max_calls_per_reference": 1,
        "post_history_final_snapshot_complete": True,
    }
    return [canonical], attempts, budget


def _delayed_v4_terminal_contract() -> tuple[list[dict], list[dict], dict]:
    results, attempts, budget = _v4_terminal_contract()
    attempts[-1].update(
        {
            "history_not_before_monotonic": 104.0,
            "query_started_monotonic": 104.1,
            "query_ended_monotonic": 104.2,
            "propagation_delay_satisfied": True,
        }
    )
    results[0] = {
        **attempts[-1],
        "source_query_sequence": 11,
    }
    budget.update(
        {
            "historical_fallback_protocol_version": (
                fill_window.DELAYED_HISTORY_PROTOCOL_VERSION
            ),
            "historical_fallback_propagation_delay_seconds": (
                fill_window.DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS
            ),
            "historical_fallback_final_snapshot_reserve_seconds": (
                fill_window.DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS
            ),
            "historical_fallback_not_before_monotonic": 104.0,
            "historical_fallback_query_deadline_monotonic": 104.5,
            "historical_fallback_wait_started_monotonic": 101.0,
            "historical_fallback_wait_ended_monotonic": 104.0,
            "historical_fallback_planned_wait_seconds": 3.0,
            "historical_fallback_actual_wait_seconds": 3.0,
            "historical_fallback_deadline_remaining_before_calls_seconds": 1.0,
            "historical_fallback_call_started_after_not_before": True,
            "post_history_final_snapshot_started_monotonic": 104.3,
            "post_history_final_snapshot_ended_monotonic": 104.4,
            "ended_monotonic": 104.45,
            "elapsed_seconds": 4.45,
        }
    )
    return results, attempts, budget


@pytest.mark.parametrize("status", [[], {}, True, 1, 1.0, None])
def test_terminal_query_classifier_rejects_non_string_status(
    status: object,
) -> None:
    assert fill_window.terminal_query_status_from_result(
        {"status": status}
    ) == "unknown"


def test_terminal_query_classifier_accepts_official_and_exact_history() -> None:
    expected = {
        "oid": fill_window.reference_identity_token("oid", 101),
        "cloid": fill_window.reference_identity_token("cloid", "a"),
    }
    official = executor.redact_with_reference_tokens(
        {
            "status": "order",
            "order": {
                "order": {"oid": 101, "cloid": "a"},
                "status": "badAloPxRejected",
            },
        }
    )
    history = executor.redact_with_reference_tokens(
        {
            "status": "historical_orders",
            "orders": [
                {
                    "order": {"oid": 999, "cloid": "foreign"},
                    "status": "canceled",
                },
                {
                    "order": {"oid": 101, "cloid": "a"},
                    "status": "canceled",
                },
            ],
        }
    )

    assert fill_window.terminal_query_status_from_result(
        official,
        method="query_order_by_oid",
        expected_tokens=expected,
    ) == "rejected"
    assert fill_window.terminal_query_status_from_result(
        history,
        method="historical_orders",
        expected_tokens=expected,
    ) == "cancel_confirmed"

    history["orders"].append(copy.deepcopy(history["orders"][1]))
    assert fill_window.terminal_query_status_from_result(
        history,
        method="historical_orders",
        expected_tokens=expected,
    ) == "unknown"


@pytest.mark.parametrize(
    "conflicting_row",
    [
        {
            "order": {"oid": 101, "cloid": "other"},
            "status": "filled",
        },
        {
            "order": {"oid": 202, "cloid": "a"},
            "status": "filled",
        },
        {
            "order": {"oid": 101},
            "status": "filled",
        },
        {
            "order": {"cloid": "a"},
            "status": "filled",
        },
        {
            "order": {"oid": "0101", "cloid": "a"},
            "status": "filled",
        },
        {
            "order": {"oid": "²", "cloid": "a"},
            "status": "filled",
        },
        {
            "order": {"oid": "1" * 5000, "cloid": "a"},
            "status": "filled",
        },
        {
            "order": {
                "oid": str(executor.MAX_REFERENCE_OID + 1),
                "cloid": "a",
            },
            "status": "filled",
        },
        {
            "order": {
                "oid": 101,
                "orderId": 102,
                "cloid": "a",
            },
            "status": "filled",
        },
        {
            "order": {
                "oid": 101,
                "cloid": "a",
                "clientOrderId": "other",
            },
            "status": "filled",
        },
        {"order": [], "status": "filled"},
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": [],
        },
        {
            "order": {"oid": 999},
            "status": "canceled",
        },
        {
            "order": {"cloid": "foreign"},
            "status": "canceled",
        },
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": "",
        },
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": " ",
        },
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": "unknownOid",
        },
        {
            "order": {"oid": 101, "cloid": "a"},
            "status": "unknownOid",
        },
    ],
)
def test_historical_terminal_query_rejects_conflicting_or_malformed_rows(
    conflicting_row: dict,
) -> None:
    expected = {
        "oid": fill_window.reference_identity_token("oid", 101),
        "cloid": fill_window.reference_identity_token("cloid", "a"),
    }
    history = executor.redact_with_reference_tokens(
        {
            "status": "historical_orders",
            "orders": [
                conflicting_row,
                {
                    "order": {"oid": 101, "cloid": "a"},
                    "status": "canceled",
                },
            ],
        }
    )

    assert fill_window.terminal_query_status_from_result(
        history,
        method="historical_orders",
        expected_tokens=expected,
    ) == "unknown"


def test_historical_row_coverage_tracks_expected_token_kinds() -> None:
    oid = fill_window.reference_identity_token("oid", 101)
    cloid = fill_window.reference_identity_token("cloid", "a")
    foreign_oid = fill_window.reference_identity_token("oid", 999)
    foreign_cloid = fill_window.reference_identity_token(
        "cloid",
        "foreign",
    )

    assert fill_window.historical_reference_row_classification(
        {
            "order": {"oid": 999},
            "status": "canceled",
        },
        expected_tokens={"oid": oid},
    ) == "foreign"
    assert fill_window.historical_reference_row_classification(
        {
            "order": {"oid": 101, "cloid": "extra"},
            "status": "canceled",
        },
        expected_tokens={"oid": oid},
    ) == "conflicting"
    assert fill_window.historical_reference_row_classification(
        {
            "order": {"oid": 999, "cloid": "extra"},
            "status": "canceled",
        },
        expected_tokens={"oid": oid},
    ) == "malformed"
    assert fill_window.historical_reference_row_classification(
        {
            "order": {"cloid": "foreign"},
            "status": "canceled",
        },
        expected_tokens={"cloid": cloid},
    ) == "foreign"
    assert fill_window.historical_reference_row_classification(
        {
            "order": {"oid": 999, "cloid": "a"},
            "status": "canceled",
        },
        expected_tokens={"cloid": cloid},
    ) == "conflicting"
    assert fill_window.historical_reference_row_classification(
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": "canceled",
        },
        expected_tokens={"cloid": cloid},
    ) == "malformed"
    assert fill_window.historical_reference_row_classification(
        {
            "order": {
                "oid_token": foreign_oid,
                "cloid_token": foreign_cloid,
            },
            "status": "canceled",
        },
        expected_tokens={"oid": oid, "cloid": cloid},
    ) == "malformed"


@pytest.mark.parametrize(
    "mutation",
    [
        "bogus_redaction_marker",
        "missing_alias_map_entry",
        "string_conflict_marker",
        "numeric_invalid_marker",
        "aggregate_token_only",
    ],
)
def test_historical_terminal_query_rejects_malformed_redaction_schema(
    mutation: str,
) -> None:
    expected = {
        "oid": fill_window.reference_identity_token("oid", 101),
        "cloid": fill_window.reference_identity_token("cloid", "a"),
    }
    order = executor.redact_with_reference_tokens(
        {
            "oid": 101,
            "orderId": "101",
            "cloid": "a",
            "clientOrderId": "a",
        }
    )
    if mutation == "bogus_redaction_marker":
        order["oid"] = "<redacted_bogus>"
    elif mutation == "missing_alias_map_entry":
        del order["oid_alias_tokens"]["orderId"]
    elif mutation == "string_conflict_marker":
        order["oid_alias_conflict"] = "true"
    elif mutation == "numeric_invalid_marker":
        order["cloid_alias_invalid"] = 1
    elif mutation == "aggregate_token_only":
        for key in (
            "oid",
            "orderId",
            "cloid",
            "clientOrderId",
            "oid_alias_tokens",
            "cloid_alias_tokens",
        ):
            order.pop(key)
    history = {
        "status": "historical_orders",
        "orders": [{"order": order, "status": "canceled"}],
    }

    assert fill_window.terminal_query_status_from_result(
        history,
        method="historical_orders",
        expected_tokens=expected,
    ) == "unknown"


def test_v4_terminal_query_rejects_flat_terminal_status() -> None:
    expected = {
        "oid": fill_window.reference_identity_token("oid", 101),
        "cloid": fill_window.reference_identity_token("cloid", "a"),
    }

    assert fill_window.terminal_query_status_from_result(
        {"status": "canceled"},
        method="query_order_by_oid",
        expected_tokens=expected,
    ) == "cancel_confirmed"
    assert fill_window.terminal_query_status_from_result(
        {"status": "canceled"},
        method="query_order_by_oid",
        expected_tokens=expected,
        require_embedded_reference=True,
    ) == "unknown"


def test_v4_terminal_history_contract_reconciles_exact_reference() -> None:
    results, attempts, budget = _v4_terminal_contract()
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        cancel_results=[
            _ambiguous_cancel(attempt=1, oid=101, cloid="a")
        ],
        terminal_query_results=results,
        terminal_query_attempts=attempts,
        terminal_query_budget=budget,
        final_open_orders=[],
    )

    assert reconciliation["schema_version"] == (
        fill_window.CANCEL_BOUNDED_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
    )
    assert reconciliation["status"] == "pass"
    assert reconciliation["terminal_query_terminal_count"] == 1
    assert reconciliation["terminal_query_attempt_audit"]["status"] == "pass"
    assert reconciliation["terminal_query_attempt_audit"][
        "attempt_row_count"
    ] == 11


def test_delayed_history_audits_reject_pre_not_before_call() -> None:
    results, attempts, budget = _v4_terminal_contract()
    budget.update(
        {
            "historical_fallback_protocol_version": (
                "delayed_one_call_history_v1"
            ),
            "historical_fallback_propagation_delay_seconds": 4.0,
            "historical_fallback_final_snapshot_reserve_seconds": 0.5,
            "historical_fallback_not_before_monotonic": 104.0,
            "historical_fallback_query_deadline_monotonic": 104.5,
            "historical_fallback_wait_started_monotonic": 101.0,
            "historical_fallback_wait_ended_monotonic": 104.0,
            "historical_fallback_planned_wait_seconds": 3.0,
            "historical_fallback_actual_wait_seconds": 3.0,
            "historical_fallback_deadline_remaining_before_calls_seconds": 1.0,
            "historical_fallback_call_started_after_not_before": False,
            "post_history_final_snapshot_started_monotonic": 104.1,
            "post_history_final_snapshot_ended_monotonic": 104.15,
            "ended_monotonic": 104.2,
            "elapsed_seconds": 4.2,
        }
    )
    attempts[-1].update(
        {
            "history_not_before_monotonic": 104.0,
            "query_started_monotonic": 103.9,
            "query_ended_monotonic": 103.95,
            "propagation_delay_satisfied": False,
        }
    )
    results[0] = {
        **attempts[-1],
        "source_query_sequence": 11,
    }

    producer_audit = fill_window.terminal_query_attempt_audit(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        terminal_query_results=results,
        terminal_query_attempts=attempts,
        terminal_query_budget=budget,
    )
    independent_audit = (
        acceptance.rebuild_raw_terminal_query_attempt_audit(
            tracked_refs=[
                {"attempt": 1, "oid": 101, "cloid": "a"}
            ],
            terminal_query_results=results,
            terminal_query_attempts=attempts,
            terminal_query_budget=budget,
        )
    )

    assert producer_audit["status"] == "fail_closed"
    assert independent_audit["status"] == "fail_closed"
    assert "terminal_audit_history_started_before_not_before" in (
        producer_audit["reasons"]
    )
    assert "terminal_audit_history_call_timing_invalid" in (
        producer_audit["reasons"]
    )
    assert producer_audit == independent_audit


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (
            "call_before_wait_end",
            "terminal_audit_history_call_timing_invalid",
        ),
        (
            "snapshot_overlaps_call",
            "terminal_audit_post_history_snapshot_overlaps_call",
        ),
    ],
)
def test_delayed_history_audits_reject_impossible_time_order(
    mutation: str,
    expected_reason: str,
) -> None:
    results, attempts, budget = _v4_terminal_contract()
    budget.update(
        {
            "historical_fallback_protocol_version": (
                "delayed_one_call_history_v1"
            ),
            "historical_fallback_propagation_delay_seconds": 4.0,
            "historical_fallback_final_snapshot_reserve_seconds": 0.5,
            "historical_fallback_not_before_monotonic": 104.0,
            "historical_fallback_query_deadline_monotonic": 104.5,
            "historical_fallback_wait_started_monotonic": 101.0,
            "historical_fallback_wait_ended_monotonic": 104.0,
            "historical_fallback_planned_wait_seconds": 3.0,
            "historical_fallback_actual_wait_seconds": 3.0,
            "historical_fallback_deadline_remaining_before_calls_seconds": 1.0,
            "historical_fallback_call_started_after_not_before": True,
            "post_history_final_snapshot_started_monotonic": 104.3,
            "post_history_final_snapshot_ended_monotonic": 104.4,
            "ended_monotonic": 104.45,
            "elapsed_seconds": 4.45,
        }
    )
    attempts[-1].update(
        {
            "history_not_before_monotonic": 104.0,
            "query_started_monotonic": 104.1,
            "query_ended_monotonic": 104.2,
            "propagation_delay_satisfied": True,
        }
    )
    if mutation == "call_before_wait_end":
        budget.update(
            {
                "historical_fallback_wait_ended_monotonic": 104.15,
                "historical_fallback_actual_wait_seconds": 3.15,
                "historical_fallback_deadline_remaining_before_calls_seconds": 0.85,
            }
        )
    elif mutation == "snapshot_overlaps_call":
        attempts[-1]["query_ended_monotonic"] = 104.35
        budget[
            "post_history_final_snapshot_started_monotonic"
        ] = 104.3
    results[0] = {
        **attempts[-1],
        "source_query_sequence": 11,
    }

    producer_audit = fill_window.terminal_query_attempt_audit(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        terminal_query_results=results,
        terminal_query_attempts=attempts,
        terminal_query_budget=budget,
    )
    independent_audit = (
        acceptance.rebuild_raw_terminal_query_attempt_audit(
            tracked_refs=[
                {"attempt": 1, "oid": 101, "cloid": "a"}
            ],
            terminal_query_results=results,
            terminal_query_attempts=attempts,
            terminal_query_budget=budget,
        )
    )

    assert producer_audit["status"] == "fail_closed"
    assert expected_reason in producer_audit["reasons"]
    assert producer_audit == independent_audit


def test_delayed_history_contract_constants_match_independent_acceptance() -> None:
    assert fill_window.DELAYED_HISTORY_PROTOCOL_VERSION == (
        acceptance.DELAYED_HISTORY_PROTOCOL_VERSION
    )
    assert fill_window.DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS == (
        acceptance.DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS
    )
    assert fill_window.DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS == (
        acceptance.DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS
    )
    assert fill_window.DELAYED_HISTORY_MAX_DIRECT_ROUNDS == (
        acceptance.DELAYED_HISTORY_MAX_DIRECT_ROUNDS
    )
    assert fill_window.DELAYED_HISTORY_TOTAL_BUDGET_SECONDS == (
        acceptance.DELAYED_HISTORY_TOTAL_BUDGET_SECONDS
    )
    assert fill_window.DELAYED_HISTORY_MAX_CALLS_PER_REFERENCE == (
        acceptance.DELAYED_HISTORY_MAX_CALLS_PER_REFERENCE
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("historical_fallback_propagation_delay_seconds", 0.0),
        ("historical_fallback_propagation_delay_seconds", 3.999),
        ("historical_fallback_propagation_delay_seconds", 4.001),
        ("historical_fallback_propagation_delay_seconds", "4.0"),
        ("historical_fallback_propagation_delay_seconds", True),
        ("historical_fallback_propagation_delay_seconds", None),
        ("historical_fallback_final_snapshot_reserve_seconds", 0.0),
        ("historical_fallback_final_snapshot_reserve_seconds", 0.499),
        ("historical_fallback_final_snapshot_reserve_seconds", 0.501),
        ("historical_fallback_final_snapshot_reserve_seconds", "0.5"),
        ("max_direct_rounds", 4),
        ("budget_seconds", 4.999),
        ("historical_fallback_max_calls_per_reference", 0),
        ("historical_fallback_max_calls_per_reference", 2),
    ],
)
def test_delayed_history_audits_require_exact_production_contract(
    field: str,
    value: object,
) -> None:
    results, attempts, budget = _delayed_v4_terminal_contract()
    budget[field] = value

    producer_audit = fill_window.terminal_query_attempt_audit(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        terminal_query_results=results,
        terminal_query_attempts=attempts,
        terminal_query_budget=budget,
    )
    independent_audit = (
        acceptance.rebuild_raw_terminal_query_attempt_audit(
            tracked_refs=[
                {"attempt": 1, "oid": 101, "cloid": "a"}
            ],
            terminal_query_results=results,
            terminal_query_attempts=attempts,
            terminal_query_budget=budget,
        )
    )

    assert producer_audit["status"] == "fail_closed"
    assert "terminal_audit_history_timing_config_invalid" in (
        producer_audit["reasons"]
    )
    assert producer_audit == independent_audit


def test_delayed_history_audits_accept_exact_production_contract() -> None:
    results, attempts, budget = _delayed_v4_terminal_contract()

    producer_audit = fill_window.terminal_query_attempt_audit(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        terminal_query_results=results,
        terminal_query_attempts=attempts,
        terminal_query_budget=budget,
    )
    independent_audit = (
        acceptance.rebuild_raw_terminal_query_attempt_audit(
            tracked_refs=[
                {"attempt": 1, "oid": 101, "cloid": "a"}
            ],
            terminal_query_results=results,
            terminal_query_attempts=attempts,
            terminal_query_budget=budget,
        )
    )

    assert producer_audit["status"] == "pass"
    assert producer_audit["reasons"] == []
    assert producer_audit == independent_audit


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("missing_canonical", "terminal_audit_canonical_coverage_mismatch"),
        ("forged_source", "terminal_audit_canonical_not_final_attempt"),
        ("elapsed_mismatch", "terminal_audit_elapsed_mismatch"),
        (
            "elapsed_budget_exceeded",
            "terminal_audit_elapsed_budget_exceeded",
        ),
        ("boolean_count", "terminal_audit_direct_count_mismatch"),
        (
            "history_without_direct",
            "terminal_audit_history_without_persistent_direct_unknown",
        ),
        (
            "incomplete_direct_round",
            "terminal_audit_history_direct_rounds_incomplete",
        ),
        (
            "history_before_max_rounds",
            "terminal_audit_history_before_max_direct_rounds",
        ),
        (
            "query_times_exceed_elapsed",
            "terminal_audit_query_times_exceed_elapsed",
        ),
        (
            "query_times_not_monotonic",
            "terminal_audit_query_time_not_monotonic",
        ),
        (
            "post_history_snapshot_incomplete",
            "terminal_audit_post_history_final_snapshot_incomplete",
        ),
        ("duplicate_history_match", "terminal_audit_status_mismatch"),
    ],
)
def test_v4_terminal_history_contract_fails_closed_on_forgery(
    mutation: str,
    expected_reason: str,
) -> None:
    results, attempts, budget = _v4_terminal_contract()
    if mutation == "missing_canonical":
        results = []
    elif mutation == "forged_source":
        results[0]["source_query_sequence"] = 2
    elif mutation == "elapsed_mismatch":
        budget["elapsed_seconds"] = 0.25
    elif mutation == "elapsed_budget_exceeded":
        budget["ended_monotonic"] = 106.0
        budget["elapsed_seconds"] = 6.0
    elif mutation == "boolean_count":
        budget["direct_query_attempt_count"] = True
    elif mutation == "history_without_direct":
        attempts = [attempts[-1]]
        attempts[0]["query_sequence"] = 1
        results[0] = {
            **attempts[0],
            "source_query_sequence": 1,
        }
        budget["direct_query_attempt_count"] = 0
    elif mutation == "incomplete_direct_round":
        attempts = [attempts[0], attempts[-1]]
        attempts[-1]["query_sequence"] = 2
        results[0] = {
            **attempts[-1],
            "source_query_sequence": 2,
        }
        budget["direct_query_attempt_count"] = 1
    elif mutation == "history_before_max_rounds":
        attempts = [attempts[0], attempts[1], attempts[-1]]
        attempts[-1]["query_sequence"] = 3
        results[0] = {
            **attempts[-1],
            "source_query_sequence": 3,
        }
        budget["direct_rounds_used"] = 1
        budget["direct_query_attempt_count"] = 2
    elif mutation == "query_times_exceed_elapsed":
        attempts[-1]["query_ended_ms"] = 31_000
        results[0]["query_ended_ms"] = 31_000
    elif mutation == "query_times_not_monotonic":
        attempts[-1]["query_started_ms"] = 1_001
        attempts[-1]["query_ended_ms"] = 1_002
        results[0]["query_started_ms"] = 1_001
        results[0]["query_ended_ms"] = 1_002
    elif mutation == "post_history_snapshot_incomplete":
        budget["post_history_final_snapshot_complete"] = False
    elif mutation == "duplicate_history_match":
        duplicate = copy.deepcopy(attempts[-1]["result"]["orders"][1])
        attempts[-1]["result"]["orders"].append(duplicate)
        results[0]["result"]["orders"].append(copy.deepcopy(duplicate))

    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        cancel_results=[
            _ambiguous_cancel(attempt=1, oid=101, cloid="a")
        ],
        terminal_query_results=results,
        terminal_query_attempts=attempts,
        terminal_query_budget=budget,
        final_open_orders=[],
    )

    assert reconciliation["status"] == "fail_closed"
    assert expected_reason in reconciliation["reasons"]


@pytest.mark.parametrize(
    ("attempts", "budget"),
    [
        ([], None),
        (None, {}),
    ],
)
def test_v4_partial_contract_fails_closed(
    attempts: list[dict] | None,
    budget: dict | None,
) -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        cancel_results=[_cancel_success(attempt=1, oid=101)],
        terminal_query_results=[],
        terminal_query_attempts=attempts,
        terminal_query_budget=budget,
        final_open_orders=[],
    )

    assert reconciliation["schema_version"] == (
        fill_window.CANCEL_BOUNDED_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
    )
    assert reconciliation["status"] == "fail_closed"
    assert "terminal_query_v4_contract_incomplete" in reconciliation["reasons"]


def test_historical_result_cannot_downgrade_to_v3_without_audit() -> None:
    results, _, _ = _v4_terminal_contract()
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        cancel_results=[_ambiguous_cancel(attempt=1, oid=101, cloid="a")],
        terminal_query_results=results,
        final_open_orders=[],
    )

    assert reconciliation["schema_version"] == (
        fill_window.CANCEL_BOUNDED_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
    )
    assert reconciliation["status"] == "fail_closed"
    assert "terminal_query_v4_contract_incomplete" in reconciliation["reasons"]


def test_v4_cloid_only_direct_queries_use_one_call_slot_per_round() -> None:
    attempts = [
        {
            "attempt": 1,
            "method": "query_order_by_cloid",
            "cloid": "a",
            "query_sequence": sequence,
            "direct_round": sequence,
            "query_started_ms": 1_000 + sequence * 2,
            "query_ended_ms": 1_001 + sequence * 2,
            "query_status": "unknown",
            "result": {"status": "unknownOid"},
        }
        for sequence in range(1, 7)
    ]
    results = [
        {**attempts[-1], "source_query_sequence": 6}
    ]
    audit = fill_window.terminal_query_attempt_audit(
        tracked_refs=[{"attempt": 1, "cloid": "a"}],
        terminal_query_results=results,
        terminal_query_attempts=attempts,
        terminal_query_budget={
            "budget_seconds": 5.0,
            "retry_seconds": 0.25,
            "started_monotonic": 100.0,
            "ended_monotonic": 100.5,
            "elapsed_seconds": 0.5,
            "max_direct_rounds": 5,
            "direct_rounds_used": 5,
            "direct_query_attempt_count": 6,
            "historical_fallback_attempt_count": 0,
            "historical_fallback_max_calls_per_reference": 1,
        },
    )

    assert audit["status"] == "fail_closed"
    assert "terminal_audit_direct_budget_exceeded" in audit["reasons"]


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


def test_terminal_canceled_query_completes_per_reference_proof() -> None:
    tracked_refs = [
        {"attempt": 1, "oid": 101, "cloid": "a"},
        {"attempt": 2, "oid": 202, "cloid": "b"},
    ]
    cancellation = fill_window.cancel_reference_reconciliation(
        tracked_refs=tracked_refs,
        cancel_results=[
            _cancel_success(attempt=1, oid=101),
            _ambiguous_cancel(attempt=2, cloid="b"),
        ],
        terminal_query_results=[
            _terminal_query(
                attempt=2,
                status="canceled",
                cloid="b",
                method="query_order_by_cloid",
            )
        ],
        final_open_orders=[{"oid": 999, "cloid": "foreign"}],
    )

    assert cancellation["schema_version"] == (
        fill_window.CANCEL_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
    )
    assert cancellation["status"] == "pass"
    assert cancellation["proven_reference_count"] == 2
    assert cancellation["terminal_query_cancel_confirmed_count"] == 1
    assert (
        cancellation[
            "tracked_reference_present_in_final_open_orders_count"
        ]
        == 0
    )

    no_fill = fill_window.no_fill_reconciliation(
        real_order_endpoint_called=True,
        cancel_results=[
            _cancel_success(attempt=1, oid=101),
            _ambiguous_cancel(attempt=2, cloid="b"),
        ],
        terminal_query_results=[
            _terminal_query(
                attempt=2,
                status="canceled",
                cloid="b",
                method="query_order_by_cloid",
            )
        ],
        tracked_refs=tracked_refs,
        final_open_orders=[{"oid": 999, "cloid": "foreign"}],
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
    assert no_fill["status"] == "no_fill_reconciled"


@pytest.mark.parametrize(
    ("terminal_queries", "final_open_orders", "expected_reason"),
    [
        (
            [
                _terminal_query(
                    attempt=1,
                    status="unknownOid",
                    oid=101,
                )
            ],
            [],
            "terminal_query_status_not_cancel_confirmed",
        ),
        (
            [_terminal_query(attempt=1, status="filled", oid=101)],
            [],
            "terminal_query_filled_requires_complete_fill_proof",
        ),
        (
            [
                _terminal_query(
                    attempt=1,
                    status="canceled",
                    oid=999,
                )
            ],
            [],
            "terminal_query_target_mismatch",
        ),
        (
            [
                _terminal_query(
                    attempt=1,
                    status="unknownOid",
                    oid=101,
                    error="query failed",
                )
            ],
            [],
            "terminal_query_error_present",
        ),
        (
            [
                {
                    "attempt": 1,
                    "method": "query_order_by_oid",
                    "oid": 101,
                    "query_status": "unknown",
                    "result": {
                        "status": "ok",
                        "note": "order canceled",
                    },
                }
            ],
            [],
            "terminal_query_status_not_cancel_confirmed",
        ),
        (
            [
                {
                    "attempt": 1,
                    "method": "query_order_by_oid",
                    "oid": 101,
                    "query_status": "unknown",
                    "result": {"status": []},
                }
            ],
            [],
            "terminal_query_status_not_cancel_confirmed",
        ),
        (
            [
                _terminal_query(
                    attempt=1,
                    status="canceled",
                    oid=101,
                ),
                _terminal_query(
                    attempt=1,
                    status="canceled",
                    oid=101,
                ),
            ],
            [],
            "terminal_query_duplicate_for_reference",
        ),
        (
            [
                _terminal_query(
                    attempt=1,
                    status="canceled",
                    oid=101,
                )
            ],
            [{"oid": 101, "cloid": "a"}],
            "terminal_query_reference_present_in_final_open_orders",
        ),
    ],
)
def test_terminal_query_evidence_fails_closed(
    terminal_queries: list[dict],
    final_open_orders: list[dict],
    expected_reason: str,
) -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        cancel_results=[_ambiguous_cancel(attempt=1, oid=101)],
        terminal_query_results=terminal_queries,
        final_open_orders=final_open_orders,
    )

    assert reconciliation["status"] == "fail_closed"
    assert expected_reason in reconciliation["reasons"]


def test_empty_terminal_query_contract_does_not_upgrade_generic_cancel() -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101}],
        cancel_results=[_ambiguous_cancel(attempt=1, oid=101)],
        terminal_query_results=[],
        final_open_orders=[],
    )

    assert reconciliation["status"] == "fail_closed"
    assert (
        "authoritative_terminal_evidence_missing_for_reference"
        in reconciliation["reasons"]
    )


def test_cancel_reconciliation_accepts_consistent_oid_and_cloid_target() -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101, "cloid": "a"}],
        cancel_results=[_cancel_success(attempt=1, oid=101, cloid="a")],
    )

    assert reconciliation["status"] == "pass"


@pytest.mark.parametrize(
    "malformed_attempt",
    [
        True,
        False,
        1.0,
        1.1,
        1.9,
        0,
        -1,
        float("nan"),
        float("inf"),
        "1.0",
        "1e0",
        " 1",
        "1 ",
        "01",
        "+1",
        "",
        "1" * 5_000,
        fill_window.MAX_CANCEL_REFERENCE_ATTEMPT + 1,
        str(fill_window.MAX_CANCEL_REFERENCE_ATTEMPT + 1),
    ],
)
def test_cancel_reconciliation_rejects_malformed_attempt_identity(
    malformed_attempt: object,
) -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": malformed_attempt, "oid": 101}],
        cancel_results=[_cancel_success(attempt=malformed_attempt, oid=101)],
    )

    assert reconciliation["status"] == "fail_closed"
    assert "tracked_reference_attempt_missing" in reconciliation["reasons"]
    assert "cancel_result_attempt_missing" in reconciliation["reasons"]


def test_cancel_reconciliation_rejects_fractional_cross_attempt_alias() -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1.1, "oid": 101}],
        cancel_results=[_cancel_success(attempt=1.9, oid=101)],
    )

    assert reconciliation["status"] == "fail_closed"
    assert reconciliation["proven_reference_count"] == 0


@pytest.mark.parametrize(
    "attempt",
    [
        1,
        "1",
        fill_window.MAX_CANCEL_REFERENCE_ATTEMPT,
        str(fill_window.MAX_CANCEL_REFERENCE_ATTEMPT),
    ],
)
def test_cancel_reconciliation_accepts_canonical_attempt_identity(
    attempt: object,
) -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": attempt, "oid": 101}],
        cancel_results=[_cancel_success(attempt=attempt, oid=101)],
    )

    assert reconciliation["status"] == "pass"
    assert reconciliation["reference_rows"][0]["attempt"] == int(attempt)


@pytest.mark.parametrize(
    "statuses",
    [
        [{"success": False}],
        [{"success": None}],
        [{"success": 0}],
        [{"success": -1}],
        [{"success": ""}],
        [{"success": " "}],
        [{"success": 0.0}],
        [{"success": 1.0}],
        [{"success": {}}],
        [{"success": []}],
        [{"success": "oid-101", "extra": True}],
        ["SUCCESS"],
        ["success", "success"],
    ],
)
def test_cancel_reconciliation_rejects_malformed_success_status(
    statuses: list[object],
) -> None:
    cancel = _cancel_success(attempt=1, oid=101)
    cancel["result"]["response"]["data"]["statuses"] = statuses

    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101}],
        cancel_results=[cancel],
    )

    assert reconciliation["status"] == "fail_closed"
    assert reconciliation["authoritative_success_count"] == 0


@pytest.mark.parametrize(
    "status",
    [
        "success",
        {"success": "oid-101"},
        {"success": 101},
    ],
)
def test_cancel_reconciliation_accepts_explicit_success_status(
    status: object,
) -> None:
    cancel = _cancel_success(attempt=1, oid=101)
    cancel["result"]["response"]["data"]["statuses"] = [status]

    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101}],
        cancel_results=[cancel],
    )

    assert reconciliation["status"] == "pass"


def test_cancel_reconciliation_rejects_token_conflicting_with_raw_identity() -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[
            {
                "attempt": 1,
                "oid": 101,
                "oid_token": fill_window.reference_identity_token("oid", 999),
            }
        ],
        cancel_results=[_cancel_success(attempt=1, oid=101)],
    )

    assert reconciliation["status"] == "fail_closed"
    assert (
        "tracked_reference_oid_token_conflicts_with_raw_identity"
        in reconciliation["reasons"]
    )


def test_cancel_reconciliation_rejects_invalid_persisted_token_format() -> None:
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=[
            {
                "attempt": 1,
                "oid": "<redacted>",
                "oid_token": "oid_sha256_not-a-digest",
            }
        ],
        cancel_results=[
            {
                **_cancel_success(attempt=1),
                "oid": "<redacted>",
                "oid_token": "oid_sha256_not-a-digest",
            }
        ],
    )

    assert reconciliation["status"] == "fail_closed"
    assert "tracked_reference_oid_token_invalid" in reconciliation["reasons"]
    assert "cancel_result_oid_token_invalid" in reconciliation["reasons"]


def test_cancel_reconciliation_survives_persisted_identity_redaction() -> None:
    tracked_refs = [{"attempt": 1, "oid": 101, "cloid": "0x" + "a" * 32}]
    cancel_results = [
        _cancel_success(attempt=1, oid=101, cloid="0x" + "a" * 32)
    ]
    expected = fill_window.cancel_reference_reconciliation(
        tracked_refs=tracked_refs,
        cancel_results=cancel_results,
    )
    persisted_refs = executor.redact(
        fill_window.persisted_reference_identity_rows(tracked_refs)
    )
    persisted_cancels = executor.redact(
        fill_window.persisted_reference_identity_rows(cancel_results)
    )

    assert persisted_refs[0]["oid"] == "<redacted>"
    assert persisted_refs[0]["cloid"] == "<redacted>"
    assert fill_window.cancel_reference_reconciliation(
        tracked_refs=persisted_refs,
        cancel_results=persisted_cancels,
    ) == expected
    assert "|oid=" not in expected["reference_rows"][0]["reference_key"]
    assert "|cloid=" not in expected["reference_rows"][0]["reference_key"]


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
            [_cancel_success(attempt=1, oid=101, cloid="unknown-cloid")],
            "cancel_result_unknown_target",
        ),
        (
            [{"attempt": 1, "oid": 101, "cloid": "a"}],
            [_cancel_success(attempt=1, oid=999, cloid="a")],
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
            "cancel_result_conflicting_target",
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
    assert "source_oid_token" in fieldnames
    assert "source_cloid_token" in fieldnames
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
                "px": "65334",
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
    is_buy: bool = True,
    limit_px: float = 65335.0,
) -> str:
    intent = executor.OrderIntent(
        symbol="BTC",
        is_buy=is_buy,
        size_btc=size_btc,
        limit_px=limit_px,
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


@pytest.mark.parametrize(
    ("direction", "explicit_side", "is_buy", "expected_side"),
    [
        ("Open Long", None, True, "buy"),
        ("Close Short", None, True, "buy"),
        ("Open Short", None, False, "sell"),
        ("Close Long", None, False, "sell"),
        ("Open Long", "B", True, "buy"),
        ("Close Long", "A", False, "sell"),
    ],
)
def test_exact_hyperliquid_direction_maps_to_execution_side(
    direction: str,
    explicit_side: str | None,
    is_buy: bool,
    expected_side: str,
) -> None:
    ledger = _ledger()
    _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_500,
        oid=101,
        is_buy=is_buy,
    )
    fill = {
        "fillId": f"direction-{direction}",
        "coin": "BTC",
        "oid": 101,
        "dir": direction,
        "sz": "0.005",
        "px": "65335",
        "time": 1_200,
        "crossed": False,
    }
    if explicit_side is not None:
        fill["side"] = explicit_side

    ledger.ingest(
        fills=[fill],
        mark_px=65_335.5,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    rows = ledger.attributed_rows()
    assert len(rows) == 1
    assert rows[0]["side"] == expected_side


@pytest.mark.parametrize(
    ("fill_fields", "reason"),
    [
        (
            {"side": "B", "dir": "Close Long"},
            "fill_side_direction_conflict",
        ),
        (
            {"side": "A", "dir": "Close Short"},
            "fill_side_direction_conflict",
        ),
        (
            {"side": "B", "dir": "Increase Long"},
            "fill_direction_invalid",
        ),
        (
            {"side": "X", "dir": "Open Long"},
            "fill_explicit_side_invalid",
        ),
    ],
)
def test_invalid_or_conflicting_fill_direction_fails_closed(
    fill_fields: dict[str, str],
    reason: str,
) -> None:
    ledger = _ledger()
    _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_500,
        oid=101,
    )
    fill = {
        "fillId": f"invalid-direction-{reason}",
        "coin": "BTC",
        "oid": 101,
        "sz": "0.005",
        "px": "65335",
        "time": 1_200,
        "crossed": False,
        **fill_fields,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65_335.5,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    assert ledger.attributed_rows() == []
    assert any(
        item.startswith(f"{reason}:")
        for item in ledger.summary()["fail_closed_reasons"]
    )


def test_reference_bound_buy_fill_above_limit_fails_closed() -> None:
    ledger = _ledger()
    _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_500,
        oid=101,
    )
    fill = {
        "fillId": "buy-above-limit",
        "coin": "BTC",
        "oid": 101,
        "side": "B",
        "sz": "0.005",
        "px": "65336",
        "time": 1_200,
        "crossed": False,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65_335.5,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    assert ledger.attributed_rows() == []
    assert ledger.evidence_rows()[0]["ambiguity_reason"] == (
        "fill_price_violates_buy_limit"
    )


def test_reference_bound_sell_fill_below_limit_fails_closed() -> None:
    ledger = _ledger()
    _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_500,
        oid=101,
        is_buy=False,
    )
    fill = {
        "fillId": "sell-below-limit",
        "coin": "BTC",
        "oid": 101,
        "side": "A",
        "sz": "0.005",
        "px": "65334",
        "time": 1_200,
        "crossed": False,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65_334.5,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    assert ledger.attributed_rows() == []
    assert ledger.evidence_rows()[0]["ambiguity_reason"] == (
        "fill_price_violates_sell_limit"
    )


def test_reference_bound_sell_fill_at_or_above_limit_passes() -> None:
    ledger = _ledger()
    _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_500,
        oid=101,
        is_buy=False,
    )
    fill = {
        "fillId": "sell-valid-limit",
        "coin": "BTC",
        "oid": 101,
        "side": "A",
        "sz": "0.005",
        "px": "65336",
        "time": 1_200,
        "crossed": False,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65_335.5,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    assert ledger.attributed_rows()[0]["price_usdc"] == 65_336.0


def test_reference_bound_fill_symbol_mismatch_fails_closed() -> None:
    ledger = _ledger()
    _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_500,
        oid=101,
    )
    fill = {
        "fillId": "wrong-symbol",
        "coin": "ETH",
        "oid": 101,
        "side": "B",
        "sz": "0.005",
        "px": "65335",
        "time": 1_200,
        "crossed": False,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65_335.5,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    assert ledger.attributed_rows() == []
    assert ledger.evidence_rows()[0]["ambiguity_reason"] == (
        "fill_symbol_conflicts_with_reference_attempt"
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


def test_all_reference_tokens_must_resolve_same_attempt() -> None:
    ledger = _ledger()
    _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_500,
        oid=101,
        cloid="cloid-one",
    )
    _register(
        ledger,
        attempt_id=2,
        start_ms=1_050,
        end_ms=1_150,
        terminal_ms=1_550,
        oid=102,
        cloid="cloid-two",
    )
    fill = {
        "fillId": "conflicting-reference-fill",
        "coin": "BTC",
        "oid": 101,
        "cloid": "cloid-two",
        "side": "B",
        "sz": "0.001",
        "px": "65335",
        "time": 1_200,
        "crossed": False,
    }

    ledger.ingest(
        fills=[fill],
        mark_px=65335.5,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=1_600,
    )

    assert ledger.attributed_rows() == []
    evidence = ledger.evidence_rows()
    assert evidence[0]["ambiguity_reason"] == (
        "conflicting_fill_reference_identity"
    )


def test_all_reference_tokens_exact_match_passes() -> None:
    ledger = _ledger()
    _register(
        ledger,
        attempt_id=1,
        start_ms=1_000,
        end_ms=1_100,
        terminal_ms=1_500,
        oid=101,
        cloid="cloid-one",
    )
    fill = {
        "fillId": "all-token-fill",
        "coin": "BTC",
        "oid": 101,
        "cloid": "cloid-one",
        "side": "B",
        "sz": "0.001",
        "px": "65335",
        "time": 1_200,
        "crossed": False,
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
    assert row["attribution_status"] == "matched_tracked_all_tokens"
    assert row["attribution_source"] == "user_fills_by_time_all_tokens"


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
