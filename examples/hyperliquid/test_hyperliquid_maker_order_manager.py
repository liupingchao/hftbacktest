from __future__ import annotations

from typing import Any

import pytest

from examples.hyperliquid import hyperliquid_maker_order_manager as manager_module
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


class FakeExchange:
    """Exchange-shaped fake that keeps open orders authoritative and deterministic."""

    def __init__(
        self,
        *,
        position_btc: float = 0.0,
        response_mode: str = "resting",
        foreign_orders: list[dict[str, Any]] | None = None,
    ) -> None:
        self.position_btc = position_btc
        self.response_mode = response_mode
        self.foreign_orders = list(foreign_orders or [])
        self.open_by_cloid: dict[str, dict[str, Any]] = {}
        self.next_oid = 700001
        self.order_calls: list[executor.OrderIntent] = []
        self.cancel_calls: list[dict[str, Any]] = []
        self.query_calls: list[tuple[str, Any]] = []
        self.pending_cancels: set[str] = set()

    def order(self, intent: executor.OrderIntent) -> dict[str, Any]:
        self.order_calls.append(intent)
        if self.response_mode == "reject":
            return {
                "status": "ok",
                "response": {"data": {"statuses": [{"error": "post_only_rejected"}]}},
            }
        oid = self.next_oid
        self.next_oid += 1
        self.open_by_cloid[intent.cloid] = {
            "coin": intent.symbol,
            "side": "B" if intent.is_buy else "A",
            "sz": str(intent.size_btc),
            "limitPx": str(intent.limit_px),
            "oid": oid,
            "cloid": intent.cloid,
        }
        if self.response_mode == "ambiguous_once":
            self.response_mode = "resting"
            return {"status": "ok", "response": {"data": {"statuses": [{}]}}}
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": oid, "cloid": intent.cloid}}]}},
        }

    def cancel_tracked(
        self,
        symbol: str,
        oid: int | None = None,
        cloid: str | None = None,
    ) -> dict[str, Any]:
        ref = str(oid if oid is not None else cloid)
        self.cancel_calls.append({"symbol": symbol, "oid": oid, "cloid": cloid})
        self.pending_cancels.add(ref)
        return {"status": "ok", "response": {"data": {"statuses": [{"success": ref}]}}}

    def confirm_cancels(self) -> None:
        for ref in list(self.pending_cancels):
            for cloid, row in list(self.open_by_cloid.items()):
                if ref == str(row["oid"]) or ref == cloid:
                    del self.open_by_cloid[cloid]
            self.pending_cancels.remove(ref)

    def open_orders(self, address: str | None = None) -> list[dict[str, Any]]:
        return list(self.open_by_cloid.values()) + list(self.foreign_orders)

    def user_state(self, address: str | None = None) -> dict[str, Any]:
        positions = []
        if self.position_btc:
            positions.append({"position": {"coin": "BTC", "szi": str(self.position_btc)}})
        return {"assetPositions": positions}

    def query_order_by_oid(self, oid: int, address: str | None = None) -> dict[str, Any]:
        self.query_calls.append(("oid", oid))
        for row in self.open_by_cloid.values():
            if row["oid"] == oid:
                return {"status": "open", "order": row}
        return {"status": "unknownOid"}

    def query_order_by_cloid(self, cloid: str, address: str | None = None) -> dict[str, Any]:
        self.query_calls.append(("cloid", cloid))
        row = self.open_by_cloid.get(cloid)
        if row is None:
            return {"status": "unknownOid"}
        return {"status": "open", "order": row}


class CancelResponseInvalidExchange(FakeExchange):
    def __init__(self, *, terminal_status: str) -> None:
        super().__init__()
        self.terminal_status = terminal_status
        self.last_canceled: dict[str, Any] = {}

    def cancel_tracked(
        self,
        symbol: str,
        oid: int | None = None,
        cloid: str | None = None,
    ) -> dict[str, Any]:
        self.cancel_calls.append(
            {"symbol": symbol, "oid": oid, "cloid": cloid}
        )
        for owned_cloid, row in list(self.open_by_cloid.items()):
            if oid == row["oid"] or cloid == owned_cloid:
                self.last_canceled = dict(row)
                del self.open_by_cloid[owned_cloid]
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "error": (
                                "Order was never placed, already canceled, "
                                "or filled. asset=0"
                            )
                        }
                    ]
                }
            },
        }

    def query_order_by_oid(
        self,
        oid: int,
        address: str | None = None,
    ) -> dict[str, Any]:
        self.query_calls.append(("oid", oid))
        return {
            "status": self.terminal_status,
            "order": dict(self.last_canceled),
        }

    def query_order_by_cloid(
        self,
        cloid: str,
        address: str | None = None,
    ) -> dict[str, Any]:
        self.query_calls.append(("cloid", cloid))
        return {
            "status": self.terminal_status,
            "order": dict(self.last_canceled),
        }

    def historical_orders(
        self,
        address: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "order": dict(self.last_canceled),
                "status": self.terminal_status,
                "statusTimestamp": 2,
            }
        ]


class HistoricalRecoveryExchange(CancelResponseInvalidExchange):
    def __init__(self) -> None:
        super().__init__(terminal_status="unknownOid")

    def historical_orders(
        self,
        address: str | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "order": {
                    "oid": 999999,
                    "cloid": "foreign",
                },
                "status": "canceled",
                "statusTimestamp": 1,
            },
            {
                "order": dict(self.last_canceled),
                "status": "canceled",
                "statusTimestamp": 2,
            },
        ]


class QueryTextLeakExchange(CancelResponseInvalidExchange):
    def __init__(self) -> None:
        super().__init__(terminal_status="unknownOid")

    def query_order_by_oid(
        self,
        oid: int,
        address: str | None = None,
    ) -> dict[str, Any]:
        self.query_calls.append(("oid", oid))
        raise RuntimeError(
            f"query failed oid={oid} cloid={self.last_canceled['cloid']}"
        )

    def query_order_by_cloid(
        self,
        cloid: str,
        address: str | None = None,
    ) -> dict[str, Any]:
        self.query_calls.append(("cloid", cloid))
        raise RuntimeError(
            f"query failed oid={self.last_canceled['oid']} cloid={cloid}"
        )


class CancelTextLeakExchange(FakeExchange):
    def cancel_tracked(
        self,
        symbol: str,
        oid: int | None = None,
        cloid: str | None = None,
    ) -> dict[str, Any]:
        tracked_cloid = next(iter(self.open_by_cloid))
        raise RuntimeError(
            f"cancel failed oid={oid} cloid={tracked_cloid}"
        )


def make_manager(
    client: FakeExchange,
    *,
    task_id: str = "0718T017",
    run_id: str = "run-a",
    runtime_config: executor.TinyLiveConfig | None = None,
    **config_kwargs: Any,
) -> manager_module.MakerOrderManager:
    config_kwargs.setdefault("min_quote_age_ms", 0)
    config = manager_module.MakerOrderManagerConfig(
        task_id=task_id,
        run_id=run_id,
        **config_kwargs,
    )
    return manager_module.MakerOrderManager(
        client=client,  # type: ignore[arg-type]
        precision=executor.mock_precision(),
        config=config,
        runtime_config=runtime_config,
        now_ms=0,
    )


def quote(side: str, px: float, size: float = 0.001) -> manager_module.DesiredQuote:
    return manager_module.DesiredQuote(side=side, size_btc=size, limit_px=px)


def test_managed_cloid_is_deterministic_and_ownership_scoped() -> None:
    cloid_a = executor.generate_managed_cloid(
        task_id="0718T017",
        run_id="run-a",
        window_id=1,
        side="buy",
        canonical_price="99",
        generation=0,
    )
    cloid_b = executor.generate_managed_cloid(
        task_id="0718T017",
        run_id="run-a",
        window_id=1,
        side="buy",
        canonical_price="99",
        generation=0,
    )

    assert cloid_a == cloid_b
    assert cloid_a.startswith(executor.managed_cloid_prefix(task_id="0718T017", run_id="run-a"))
    assert len(cloid_a) == 34
    assert executor.is_owned_managed_cloid(cloid_a, task_id="0718T017", run_id="run-a")
    assert not executor.is_owned_managed_cloid(cloid_a, task_id="0718T017", run_id="run-b")
    assert not executor.is_owned_managed_cloid(
        executor.managed_cloid_prefix(task_id="0718T017", run_id="run-a") + "not-hex",
        task_id="0718T017",
        run_id="run-a",
    )
    assert executor.canonical_price_key(99.04, sz_decimals=5) == "99"


def test_single_level_two_sided_quotes_and_same_target_hold() -> None:
    client = FakeExchange()
    manager = make_manager(client)

    first = manager.reconcile_desired(
        [quote("buy", 99), quote("sell", 101)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    second = manager.reconcile_desired(
        [quote("buy", 99), quote("sell", 101)],
        now_ms=1,
        reconcile_exchange_first=False,
    )

    assert [action["action"] for action in first["actions"]] == ["submitted", "submitted"]
    assert [
        set(action["order_result"]["response"]["data"]["statuses"][0])
        for action in first["actions"]
    ] == [{"resting"}, {"resting"}]
    assert [action["action"] for action in second["actions"]] == ["hold", "hold"]
    assert len(client.order_calls) == 2
    assert len(manager.orders_by_key) == 2


def test_startup_reconcile_recovers_owned_orders_and_ignores_foreign() -> None:
    foreign = {"coin": "BTC", "side": "B", "sz": "0.001", "limitPx": "98", "oid": 999, "cloid": "0xforeign"}
    client = FakeExchange(foreign_orders=[foreign])
    manager_one = make_manager(client)
    manager_one.reconcile_desired([quote("buy", 99)], now_ms=0, reconcile_exchange_first=False)
    owned_cloid = client.order_calls[0].cloid

    manager_two = make_manager(client)
    evidence = manager_two.startup_reconcile(now_ms=100)
    result = manager_two.reconcile_desired(
        [quote("buy", 99)],
        now_ms=101,
        reconcile_exchange_first=False,
    )

    assert evidence["owned_order_count"] == 1
    assert evidence["foreign_order_count"] == 1
    assert result["actions"][0]["action"] == "hold"
    assert result["actions"][0]["cloid"] == owned_cloid
    assert len(client.order_calls) == 1


def test_duplicate_owned_logical_key_fails_closed() -> None:
    client = FakeExchange()
    cloid_a = executor.generate_managed_cloid(
        task_id="0718T017",
        run_id="run-a",
        window_id=1,
        side="buy",
        canonical_price="99",
        generation=0,
    )
    cloid_b = executor.generate_managed_cloid(
        task_id="0718T017",
        run_id="run-a",
        window_id=1,
        side="buy",
        canonical_price="99",
        generation=1,
    )
    client.open_by_cloid.update(
        {
            cloid_a: {"coin": "BTC", "side": "B", "sz": "0.001", "limitPx": "99", "oid": 1, "cloid": cloid_a},
            cloid_b: {"coin": "BTC", "side": "B", "sz": "0.001", "limitPx": "99", "oid": 2, "cloid": cloid_b},
        }
    )

    with pytest.raises(manager_module.OrderManagerError, match="duplicate_owned_logical_quote_key"):
        make_manager(client).startup_reconcile(now_ms=0)


def test_cancel_confirmed_then_readd_same_logical_key_uses_new_generation() -> None:
    client = FakeExchange()
    manager = make_manager(client)
    manager.reconcile_desired([quote("buy", 99)], now_ms=0, reconcile_exchange_first=False)
    old = next(iter(manager.orders_by_key.values()))

    manager._request_cancel(old, now_ms=1, emergency=True)
    client.confirm_cancels()
    manager.reconcile_exchange(now_ms=2)
    result = manager.reconcile_desired([quote("buy", 99)], now_ms=3, reconcile_exchange_first=False)

    new = manager.orders_by_key[manager.logical_key("buy", 99)]
    assert result["actions"][0]["action"] == "submitted"
    assert old.state == "cancel_confirmed"
    assert new.generation == 1
    assert new.cloid != old.cloid


def test_ambiguous_submit_queries_before_retry_and_does_not_duplicate() -> None:
    client = FakeExchange(response_mode="ambiguous_once")
    manager = make_manager(client)

    result = manager.reconcile_desired([quote("buy", 99)], now_ms=0, reconcile_exchange_first=False)

    assert result["actions"][0]["action"] == "submitted"
    assert result["submissions_used"] == 1
    assert len(client.order_calls) == 1
    assert client.query_calls == [("cloid", client.order_calls[0].cloid)]


def test_cancel_validation_failure_preserves_response_and_query_proves_terminal() -> None:
    client = CancelResponseInvalidExchange(terminal_status="canceled")
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))

    cancel = manager.cancel_all_owned(now_ms=1)[0]
    reconciliation = manager.reconcile_exchange(
        now_ms=2,
        reason="post_cycle_cancel_reconcile",
    )

    assert cancel["action"] == "cancel_unknown"
    assert cancel["result"]["response"]["data"]["statuses"][0]["error"]
    assert order.state == "cancel_confirmed"
    assert order.leaves_qty == pytest.approx(0.0)
    assert reconciliation["owned_order_count"] == 0
    assert manager.terminal_query_evidence == [
        {
            "phase": "post_cycle_cancel_reconcile",
            "method": "query_order_by_oid",
            "oid": order.oid,
            "cloid": order.cloid,
            "query_started_ms": manager.terminal_query_evidence[0][
                "query_started_ms"
            ],
            "query_ended_ms": manager.terminal_query_evidence[0][
                "query_ended_ms"
            ],
            "query_status": "cancel_confirmed",
            "query_sequence": 1,
            "result": executor.redact_with_reference_tokens(
                {
                    "status": "canceled",
                    "order": client.last_canceled,
                }
            ),
        }
    ]


def test_cancel_unknown_query_remains_fail_closed_and_active() -> None:
    client = CancelResponseInvalidExchange(terminal_status="unknownOid")
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))

    cancel = manager.cancel_all_owned(now_ms=1)[0]
    manager.reconcile_exchange(
        now_ms=2,
        reason="post_cycle_cancel_reconcile",
    )

    assert cancel["action"] == "cancel_unknown"
    assert order.state == "unknown"
    assert order.is_active is True
    assert order.last_query_status == "unknown"
    assert [row["query_status"] for row in manager.terminal_query_evidence] == [
        "unknown",
        "unknown",
    ]


@pytest.mark.parametrize("status", [[], {}, True, 1, 1.0, None])
def test_order_status_classifier_rejects_non_string_status(
    status: object,
) -> None:
    assert manager_module._classify_order_status_query_payload(
        {"status": status}
    ) == "unknown"


def test_order_status_classifier_rejects_keyword_only_payloads() -> None:
    assert manager_module._classify_order_status_query_payload(
        {"status": "ok", "note": "canceled"}
    ) == "unknown"
    assert manager_module._classify_order_status_query_payload(
        {"status": "canceled"}
    ) == "cancel_confirmed"
    assert manager_module._classify_order_status_query_payload(
        {"status": "canceled"},
        expected_oid=101,
        expected_cloid="abc",
        require_embedded_reference=True,
    ) == "unknown"


def test_legacy_direct_status_validates_supplied_order_identity() -> None:
    assert manager_module._classify_order_status_query_payload(
        {
            "status": "canceled",
            "order": {"oid": 101, "cloid": "abc"},
        },
        expected_oid=101,
        expected_cloid="abc",
    ) == "cancel_confirmed"
    assert manager_module._classify_order_status_query_payload(
        {
            "status": "canceled",
            "order": {"oid": 999, "cloid": "other"},
        },
        expected_oid=101,
        expected_cloid="abc",
    ) == "unknown"


def test_official_nested_order_status_requires_exact_reference() -> None:
    payload = {
        "status": "order",
        "order": {
            "order": {"oid": 101, "cloid": "abc"},
            "status": "canceled",
            "statusTimestamp": 2,
        },
    }

    assert manager_module._classify_order_status_query_payload(
        payload,
        expected_oid=101,
        expected_cloid="abc",
    ) == "cancel_confirmed"
    assert manager_module._classify_order_status_query_payload(
        payload,
        expected_oid=999,
        expected_cloid="abc",
    ) == "unknown"
    assert manager_module._classify_order_status_query_payload(
        payload,
        expected_oid=101,
        expected_cloid="other",
    ) == "unknown"


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ("open", "resting"),
        ("filled", "filled"),
        ("canceled", "cancel_confirmed"),
        ("badAloPxRejected", "rejected"),
    ],
)
def test_official_nested_order_status_classifies_supported_statuses(
    status: str,
    expected: str,
) -> None:
    payload = {
        "status": "order",
        "order": {
            "order": {"oid": 101, "cloid": "abc"},
            "status": status,
            "statusTimestamp": 2,
        },
    }

    assert manager_module._classify_order_status_query_payload(
        payload,
        expected_oid=101,
        expected_cloid="abc",
    ) == expected


@pytest.mark.parametrize(
    "order",
    [
        {"oid": True, "cloid": "abc"},
        {"oid": 101.0, "cloid": "abc"},
        {"oid": "0101", "cloid": "abc"},
        {"oid": "²", "cloid": "abc"},
        {"oid": "1" * 5000, "cloid": "abc"},
        {
            "oid": str(manager_module.MAX_REFERENCE_OID + 1),
            "cloid": "abc",
        },
        {
            "oid": manager_module.MAX_REFERENCE_OID + 1,
            "cloid": "abc",
        },
        {"oid": 101, "orderId": 102, "cloid": "abc"},
        {"oid": 101, "cloid": True},
        {"oid": 101, "cloid": "abc", "clientOrderId": "other"},
    ],
)
def test_official_nested_order_status_rejects_malformed_reference(
    order: dict[str, Any],
) -> None:
    assert manager_module._classify_order_status_query_payload(
        {
            "status": "order",
            "order": {
                "order": order,
                "status": "canceled",
            },
        },
        expected_oid=101,
        expected_cloid="abc",
    ) == "unknown"


def test_historical_order_selector_requires_one_exact_reference() -> None:
    exact = {
        "order": {"oid": 101, "cloid": "abc"},
        "status": "canceled",
        "statusTimestamp": 2,
    }
    foreign = {
        "order": {"oid": 202, "cloid": "foreign"},
        "status": "canceled",
        "statusTimestamp": 2,
    }

    assert manager_module._historical_order_status_payload(
        [foreign, exact],
        expected_oid=101,
        expected_cloid="abc",
    ) == {"status": "order", "order": exact}
    with pytest.raises(
        manager_module.OrderManagerError,
        match="historical_order_exact_match_missing",
    ):
        manager_module._historical_order_status_payload(
            [foreign],
            expected_oid=101,
            expected_cloid="abc",
        )
    with pytest.raises(
        manager_module.OrderManagerError,
        match="historical_order_exact_match_ambiguous",
    ):
        manager_module._historical_order_status_payload(
            [exact, dict(exact)],
            expected_oid=101,
            expected_cloid="abc",
        )


@pytest.mark.parametrize(
    ("conflicting_row", "expected_error"),
    [
        (
            {
                "order": {"oid": 101, "cloid": "other"},
                "status": "filled",
            },
            "historical_order_reference_conflict",
        ),
        (
            {
                "order": {"oid": 202, "cloid": "abc"},
                "status": "filled",
            },
            "historical_order_reference_conflict",
        ),
        (
            {
                "order": {"oid": 101},
                "status": "filled",
            },
            "historical_order_reference_conflict",
        ),
        (
            {
                "order": {"cloid": "abc"},
                "status": "filled",
            },
            "historical_order_reference_conflict",
        ),
        (
            {
                "order": {
                    "oid": 101,
                    "orderId": 102,
                    "cloid": "abc",
                },
                "status": "filled",
            },
            "historical_order_reference_malformed",
        ),
        (
            {
                "order": {"oid": "0101", "cloid": "abc"},
                "status": "filled",
            },
            "historical_order_reference_malformed",
        ),
        (
            {
                "order": {"oid": "²", "cloid": "abc"},
                "status": "filled",
            },
            "historical_order_reference_malformed",
        ),
        (
            {
                "order": {"oid": "1" * 5000, "cloid": "abc"},
                "status": "filled",
            },
            "historical_order_reference_malformed",
        ),
        (
            {
                "order": {
                    "oid": 101,
                    "cloid": "abc",
                    "clientOrderId": "other",
                },
                "status": "filled",
            },
            "historical_order_reference_malformed",
        ),
        (
            {"order": [], "status": "filled"},
            "historical_order_reference_malformed",
        ),
        (
            {
                "order": {"oid": 999, "cloid": "foreign"},
                "status": [],
            },
            "historical_order_reference_malformed",
        ),
    ],
)
def test_historical_order_selector_rejects_conflicting_or_malformed_rows(
    conflicting_row: dict[str, Any],
    expected_error: str,
) -> None:
    exact = {
        "order": {"oid": 101, "cloid": "abc"},
        "status": "canceled",
    }

    with pytest.raises(
        manager_module.OrderManagerError,
        match=expected_error,
    ):
        manager_module._historical_order_status_payload(
            [conflicting_row, exact],
            expected_oid=101,
            expected_cloid="abc",
        )


def test_historical_recovery_persists_complete_redacted_response() -> None:
    client = HistoricalRecoveryExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))
    manager.cancel_all_owned(now_ms=1)
    manager.reconcile_exchange(
        now_ms=2,
        reason="post_cycle_cancel_reconcile",
    )

    assert order.state == "unknown"
    assert manager.reconcile_historical_terminal(
        order,
        phase="post_cycle_historical_fallback",
        now_ms=3,
    ) == "cancel_confirmed"
    assert order.state == "cancel_confirmed"
    assert order.leaves_qty == 0.0
    evidence = manager.terminal_query_evidence[-1]
    assert evidence["method"] == "historical_orders"
    assert evidence["query_sequence"] == 3
    assert evidence["query_status"] == "cancel_confirmed"
    assert len(evidence["result"]["orders"]) == 2
    encoded = str(evidence["result"])
    assert str(order.oid) not in encoded
    assert order.cloid not in encoded
    exact_order = evidence["result"]["orders"][1]["order"]
    assert exact_order["oid"] == "<redacted>"
    assert exact_order["cloid"] == "<redacted>"
    assert exact_order["oid_token"].startswith("oid_sha256_")
    assert exact_order["cloid_token"].startswith("cloid_sha256_")


def test_terminal_query_errors_redact_known_references_from_free_text() -> None:
    client = QueryTextLeakExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))
    manager.cancel_all_owned(now_ms=1)

    manager.reconcile_exchange(
        now_ms=2,
        reason="post_cycle_cancel_reconcile",
    )

    errors = [
        str(row["error"])
        for row in manager.terminal_query_evidence
    ]
    assert all(str(order.oid) not in error for error in errors)
    assert all(order.cloid not in error for error in errors)
    assert all("<redacted_oid>" in error for error in errors)


def test_terminal_query_deadline_blocks_new_direct_and_history_calls() -> None:
    client = HistoricalRecoveryExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))
    manager.cancel_all_owned(now_ms=1)

    with pytest.raises(
        manager_module.OrderManagerError,
        match="terminal_query_deadline_exhausted_before_open_orders",
    ):
        manager.reconcile_exchange(
            now_ms=2,
            reason="post_cycle_cancel_reconcile",
            terminal_query_deadline_monotonic=0.0,
        )

    assert client.query_calls == []
    assert manager.terminal_query_evidence == []
    assert order.state == "unknown"
    assert manager.reconcile_historical_terminal(
        order,
        phase="post_cycle_historical_fallback",
        now_ms=3,
        deadline_monotonic=0.0,
    ) == "unknown"
    assert manager.historical_query_counts_by_cloid == {}
    assert manager.terminal_query_evidence == []
    assert order.last_error == "terminal_query_deadline_exhausted"


def test_terminal_reconcile_bounds_sdk_open_orders_and_defers_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Info:
        def __init__(self) -> None:
            self.timeout = 5.0
            self.open_order_timeouts: list[float] = []
            self.user_state_calls = 0

        def open_orders(self, address: str) -> list[dict[str, Any]]:
            self.open_order_timeouts.append(self.timeout)
            return []

        def user_state(self, address: str) -> dict[str, Any]:
            self.user_state_calls += 1
            return {"assetPositions": []}

    info = Info()
    client = executor.SDKHyperliquidClient(
        exchange=object(),
        info=info,
        account_address="0xaccount",
    )
    manager = make_manager(client)  # type: ignore[arg-type]
    monotonic_values = iter([100.0, 100.0])
    monkeypatch.setattr(
        manager_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    reconciliation = manager.reconcile_exchange(
        now_ms=1,
        reason="post_cycle_cancel_reconcile",
        terminal_query_deadline_monotonic=100.25,
    )

    assert info.open_order_timeouts == [pytest.approx(0.25)]
    assert info.timeout == 5.0
    assert info.user_state_calls == 0
    assert reconciliation["position_refresh_status"] == (
        "deferred_terminal_query_budget"
    )


def test_cancel_exception_redacts_known_references() -> None:
    client = CancelTextLeakExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))

    result = manager.cancel_all_owned(now_ms=1)[0]
    encoded_reason = str(result["reason"])

    assert result["action"] == "cancel_unknown"
    assert str(order.oid) not in encoded_reason
    assert order.cloid not in encoded_reason
    assert str(order.oid) not in order.last_error
    assert order.cloid not in order.last_error
    assert "<redacted_oid>" in str(result["reason"])
    assert "<redacted_" in str(result["reason"])


@pytest.mark.parametrize("cloid_value", [None, True, "different"])
def test_final_open_order_exact_oid_restores_terminal_state(
    cloid_value: object,
) -> None:
    client = HistoricalRecoveryExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))
    manager.cancel_all_owned(now_ms=1)
    manager.reconcile_exchange(
        now_ms=2,
        reason="post_cycle_cancel_reconcile",
    )
    manager.reconcile_historical_terminal(
        order,
        phase="post_cycle_historical_fallback",
        now_ms=3,
    )
    final_row = dict(client.last_canceled)
    if cloid_value is None:
        final_row.pop("cloid")
    else:
        final_row["cloid"] = cloid_value

    manager.reconcile_supplied_open_orders(
        open_orders=[final_row],
        now_ms=4,
        reason="post_history_final_open_orders",
    )

    assert order.state == "resting"
    assert order.leaves_qty == pytest.approx(order.size_btc)
    assert order.last_query_status == "resting"
    assert order.last_error == (
        "open_order_cloid_missing_or_mismatched_for_tracked_oid"
    )
    assert manager.working_exposure().working_buy_qty == pytest.approx(
        order.size_btc
    )


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        (
            {"side": "A"},
            "tracked_oid_open_order_logical_key_mismatch",
        ),
        (
            {"limitPx": "101"},
            "tracked_oid_open_order_logical_key_mismatch",
        ),
        (
            {"sz": "0.002"},
            "tracked_oid_open_order_size_exceeds_original",
        ),
        (
            {"symbol": "ETH"},
            "exchange_order_symbol_alias_conflict",
        ),
        (
            {"px": "101"},
            "exchange_order_limit_px_alias_conflict",
        ),
        (
            {"size": "0.002"},
            "exchange_order_size_alias_conflict",
        ),
        (
            {"remainingSz": "0.0006", "size": "0.002"},
            "exchange_order_size_alias_conflict",
        ),
        (
            {"remainingSz": "0.0006", "sz": "0.002"},
            "tracked_oid_open_order_size_exceeds_original",
        ),
        (
            {"limitPx": "99.04"},
            "tracked_oid_open_order_price_mismatch",
        ),
    ],
)
def test_final_open_order_exact_oid_conflict_fails_closed(
    mutation: dict[str, str],
    expected_error: str,
) -> None:
    client = HistoricalRecoveryExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))
    manager.cancel_all_owned(now_ms=1)
    manager.reconcile_exchange(
        now_ms=2,
        reason="post_cycle_cancel_reconcile",
    )
    manager.reconcile_historical_terminal(
        order,
        phase="post_cycle_historical_fallback",
        now_ms=3,
    )
    final_row = dict(client.last_canceled)
    final_row.pop("cloid")
    final_row.update(mutation)

    with pytest.raises(
        manager_module.OrderManagerError,
        match=expected_error,
    ):
        manager.reconcile_supplied_open_orders(
            open_orders=[final_row],
            now_ms=4,
            reason="post_history_final_open_orders",
        )

    assert len(manager.orders_by_key) == 1
    assert order.state == "cancel_confirmed"
    assert order.leaves_qty == pytest.approx(0.0)
    assert manager.working_exposure().working_buy_qty == pytest.approx(0.0)
    assert manager.working_exposure().working_sell_qty == pytest.approx(0.0)


def test_historical_fallback_rejects_non_unknown_reference() -> None:
    client = HistoricalRecoveryExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))

    with pytest.raises(
        manager_module.OrderManagerError,
        match="historical_fallback_requires_active_unknown_reference",
    ):
        manager.reconcile_historical_terminal(
            order,
            phase="post_cycle_historical_fallback",
            now_ms=1,
        )

    assert manager.historical_query_counts_by_cloid == {}
    assert manager.terminal_query_evidence == []


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        (
            "terminal_query_max_rounds",
            True,
            "manager_terminal_query_max_rounds_invalid",
        ),
        (
            "terminal_query_max_rounds",
            6,
            "manager_terminal_query_max_rounds_invalid",
        ),
        (
            "historical_fallback_max_calls_per_reference",
            True,
            "manager_historical_fallback_call_limit_invalid",
        ),
        (
            "historical_fallback_max_calls_per_reference",
            2,
            "manager_historical_fallback_call_limit_invalid",
        ),
    ],
)
def test_terminal_recovery_config_rejects_unbounded_or_boolean_limits(
    field: str,
    value: object,
    reason: str,
) -> None:
    with pytest.raises(ValueError, match=reason):
        manager_module.MakerOrderManagerConfig(
            task_id="0720T023",
            run_id="config-test",
            **{field: value},
        )


def test_query_classifier_exception_is_persisted_as_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = CancelResponseInvalidExchange(terminal_status="canceled")
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))
    manager.cancel_all_owned(now_ms=1)
    monkeypatch.setattr(
        manager_module,
        "_classify_order_status_query_payload",
        lambda payload, **kwargs: (_ for _ in ()).throw(
            RuntimeError("classifier exploded")
        ),
    )

    manager.reconcile_exchange(
        now_ms=2,
        reason="post_cycle_cancel_reconcile",
    )

    assert order.state == "unknown"
    assert order.last_query_status == "unknown"
    assert len(manager.terminal_query_evidence) == 2
    assert all(
        row["query_status"] == "unknown"
        and "classifier exploded" in row["error"]
        and row["result"]["status"] == "canceled"
        for row in manager.terminal_query_evidence
    )


def test_supplied_final_snapshot_restores_reappearing_order() -> None:
    client = CancelResponseInvalidExchange(terminal_status="canceled")
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))
    manager.cancel_all_owned(now_ms=1)
    manager.reconcile_exchange(
        now_ms=2,
        reason="post_cycle_cancel_reconcile",
    )
    query_count = len(client.query_calls)
    assert order.state == "cancel_confirmed"

    reconciliation = manager.reconcile_supplied_snapshot(
        open_orders=[dict(client.last_canceled)],
        user_state={"assetPositions": []},
        now_ms=3,
        reason="finalizer_account_snapshot",
    )

    assert reconciliation["owned_order_count"] == 1
    assert reconciliation["position_snapshot_status"] == "pass"
    assert reconciliation["position_snapshot_reason"] == ""
    assert "position_snapshot_error" not in reconciliation
    assert manager.current_position_btc == 0.0
    assert order.state == "resting"
    assert order.is_active is True
    assert order.leaves_qty == pytest.approx(order.size_btc)
    assert manager.working_exposure().working_buy_qty == pytest.approx(
        order.size_btc
    )
    assert len(client.query_calls) == query_count
    assert manager.position_evidence[-1]["source"] == (
        "supplied_final_user_state"
    )


def test_supplied_final_snapshot_restores_cancel_requested_order() -> None:
    client = FakeExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))
    manager.cancel_all_owned(now_ms=1)
    assert order.state == "cancel_requested"
    cancel_requested_at_ms = order.cancel_requested_at_ms

    manager.reconcile_supplied_snapshot(
        open_orders=list(client.open_by_cloid.values()),
        user_state={"assetPositions": []},
        now_ms=2,
        reason="finalizer_account_snapshot",
    )

    assert order.state == "resting"
    assert order.last_query_status == "resting"
    assert order.last_error == "cancel_requested_but_still_open"
    assert order.cancel_requested_at_ms == cancel_requested_at_ms
    assert manager.working_exposure().working_buy_qty == pytest.approx(
        order.size_btc
    )


def test_supplied_final_snapshot_keeps_orders_when_position_is_unavailable() -> None:
    client = FakeExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))

    reconciliation = manager.reconcile_supplied_snapshot(
        open_orders=list(client.open_by_cloid.values()),
        user_state=None,
        now_ms=1,
        reason="finalizer_account_snapshot",
    )

    assert reconciliation["owned_order_count"] == 1
    assert reconciliation["position_snapshot_status"] == "fail_closed"
    assert order.state == "resting"
    assert manager.working_exposure().working_buy_qty == pytest.approx(
        order.size_btc
    )


@pytest.mark.parametrize(
    "user_state",
    [
        {},
        {"assetPositions": None},
        {"assetPositions": {}},
        {"assetPositions": [None]},
        {"assetPositions": [{"position": None}]},
        {"assetPositions": [{}]},
        {"assetPositions": [{"position": {}}]},
        {"assetPositions": [{"position": {"coin": "BTC", "szi": True}}]},
        {
            "assetPositions": [
                {
                    "coin": "BTC",
                    "position": {"coin": "ETH", "szi": "0"},
                }
            ]
        },
    ],
)
def test_supplied_final_snapshot_fails_closed_for_invalid_position_shapes(
    user_state: dict,
) -> None:
    client = FakeExchange()
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))

    reconciliation = manager.reconcile_supplied_snapshot(
        open_orders=list(client.open_by_cloid.values()),
        user_state=user_state,
        now_ms=1,
        reason="finalizer_account_snapshot",
    )

    assert reconciliation["position_snapshot_status"] == "fail_closed"
    assert reconciliation["position_snapshot_reason"] == (
        "final_user_state_unavailable_or_invalid"
    )
    assert reconciliation["position_snapshot_error"]
    assert order.state == "resting"
    assert manager.working_exposure().working_buy_qty == pytest.approx(
        order.size_btc
    )


def test_query_filled_remains_unknown_without_raw_fill_proof() -> None:
    client = CancelResponseInvalidExchange(terminal_status="filled")
    manager = make_manager(client)
    manager.reconcile_desired(
        [quote("buy", 99)],
        now_ms=0,
        reconcile_exchange_first=False,
    )
    order = next(iter(manager.orders_by_key.values()))
    manager.cancel_all_owned(now_ms=1)

    manager.reconcile_exchange(
        now_ms=2,
        reason="post_cycle_cancel_reconcile",
    )
    manager.reconcile_supplied_snapshot(
        open_orders=[],
        user_state={"assetPositions": []},
        now_ms=3,
        reason="finalizer_account_snapshot",
    )

    assert order.state == "unknown"
    assert order.is_active is True
    assert order.last_query_status == "filled"
    assert order.last_error == "query_filled_requires_raw_fill_proof"
    assert order.filled_qty == pytest.approx(0.0)
    assert order.leaves_qty == pytest.approx(order.size_btc)


def test_partial_fill_and_cancel_pending_remain_in_working_exposure() -> None:
    client = FakeExchange()
    manager = make_manager(client)
    manager.reconcile_desired([quote("buy", 99)], now_ms=0, reconcile_exchange_first=False)
    order = next(iter(manager.orders_by_key.values()))

    manager.apply_fill(fill_qty=0.0004, fill_px=99, now_ms=1, oid=order.oid)
    before_cancel = manager.working_exposure()
    result = manager.reconcile_desired(
        [quote("buy", 101)],
        now_ms=2,
        reconcile_exchange_first=False,
    )

    assert order.state == "cancel_requested"
    assert before_cancel.working_buy_qty == pytest.approx(0.0006)
    assert result["working_exposure"]["working_buy_qty"] == pytest.approx(0.0006)
    client.confirm_cancels()
    manager.reconcile_exchange(now_ms=3)
    assert order.state == "cancel_confirmed"
    assert order.leaves_qty == pytest.approx(0.0)


def test_reconcile_prefers_exchange_remaining_size_for_partial_order() -> None:
    client = FakeExchange()
    manager = make_manager(client)
    manager.reconcile_desired([quote("buy", 99)], now_ms=0, reconcile_exchange_first=False)
    row = next(iter(client.open_by_cloid.values()))
    row["sz"] = "0.001"
    row["remainingSz"] = "0.0006"

    manager.reconcile_exchange(now_ms=1)

    order = next(iter(manager.orders_by_key.values()))
    assert order.leaves_qty == pytest.approx(0.0006)
    assert order.filled_qty == pytest.approx(0.0004)


def test_antichurn_guards_and_emergency_cancel() -> None:
    client = FakeExchange()
    manager = make_manager(client, min_quote_age_ms=250, max_cancel_readds_per_side_per_minute=1)
    manager.reconcile_desired([quote("buy", 99)], now_ms=0, reconcile_exchange_first=False)

    blocked = manager.reconcile_desired(
        [quote("buy", 101)],
        now_ms=100,
        reconcile_exchange_first=False,
    )
    emergency = manager.reconcile_desired(
        [quote("buy", 101)],
        now_ms=100,
        emergency=True,
        reconcile_exchange_first=False,
    )

    assert blocked["actions"][0]["reason"] == "min_quote_age_guard"
    assert emergency["actions"][0]["action"] == "cancel_requested"

    client.confirm_cancels()
    manager.reconcile_exchange(now_ms=200)
    manager.reconcile_desired([quote("buy", 101)], now_ms=201, reconcile_exchange_first=False)
    rate_blocked = manager.reconcile_desired(
        [quote("buy", 103)],
        now_ms=500,
        reconcile_exchange_first=False,
    )
    assert rate_blocked["actions"][0]["reason"] == "cancel_readd_rate_limit"


def test_post_only_reject_cooldown_blocks_immediate_retry() -> None:
    client = FakeExchange(response_mode="reject")
    manager = make_manager(client, post_only_reject_cooldown_ms=1_000)

    first = manager.reconcile_desired([quote("buy", 99)], now_ms=0, reconcile_exchange_first=False)
    rejected_order = manager.orders_by_key[
        manager.logical_key("buy", 99)
    ]
    blocked = manager.reconcile_desired([quote("buy", 99)], now_ms=500, reconcile_exchange_first=False)
    client.response_mode = "resting"
    retried = manager.reconcile_desired([quote("buy", 99)], now_ms=1_000, reconcile_exchange_first=False)

    assert first["actions"][0]["action"] == "rejected"
    assert first["actions"][0]["order_endpoint_called"] is True
    assert first["actions"][0]["order_result"]["response"]["data"]["statuses"] == [
        {"error": "post_only_rejected"}
    ]
    assert rejected_order.leaves_qty == 0.0
    assert blocked["actions"][0]["reason"] == "post_only_reject_cooldown"
    assert retried["actions"][0]["action"] == "submitted"
    assert len(client.order_calls) == 2


def test_runtime_position_cap_applies_before_submit() -> None:
    client = FakeExchange(position_btc=0.001)
    manager = make_manager(
        client,
        runtime_config=executor.TinyLiveConfig(max_position_btc=0.001),
    )

    with pytest.raises(executor.ValidationError, match="runtime_worst_long_position_cap_exceeded"):
        manager.reconcile_desired([quote("buy", 99, size=0.00001)], now_ms=0, reconcile_exchange_first=True)


def test_reconcile_missing_resting_order_becomes_unknown() -> None:
    client = FakeExchange()
    manager = make_manager(client)
    manager.reconcile_desired([quote("buy", 99)], now_ms=0, reconcile_exchange_first=False)
    client.open_by_cloid.clear()

    evidence = manager.reconnect_reconcile(now_ms=1)

    order = next(iter(manager.orders_by_key.values()))
    assert evidence["reason"] == "reconnect"
    assert order.state == "unknown"
    assert order.last_query_status == "missing_from_open_orders"
