#!/usr/bin/env python3
"""Exchange-reconciled single-level two-sided maker order manager."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Iterable

from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


ORDER_STATES = frozenset(
    {
        "desired",
        "submit_inflight",
        "resting",
        "partial_fill",
        "cancel_requested",
        "cancel_confirmed",
        "filled",
        "rejected",
        "unknown",
    }
)
ACTIVE_STATES = frozenset({"submit_inflight", "resting", "partial_fill", "cancel_requested", "unknown"})
TERMINAL_STATES = frozenset({"cancel_confirmed", "filled", "rejected"})
ORDER_STATUS_CANCEL_CONFIRMED = frozenset(
    {
        "canceled",
        "marginCanceled",
        "vaultWithdrawalCanceled",
        "openInterestCapCanceled",
        "selfTradeCanceled",
        "reduceOnlyCanceled",
        "siblingFilledCanceled",
        "delistedCanceled",
        "liquidatedCanceled",
        "scheduledCancel",
    }
)
ORDER_STATUS_REJECTED = frozenset(
    {
        "rejected",
        "tickRejected",
        "minTradeNtlRejected",
        "perpMarginRejected",
        "reduceOnlyRejected",
        "badAloPxRejected",
        "iocCancelRejected",
        "badTriggerPxRejected",
        "marketOrderNoLiquidityRejected",
        "positionIncreaseAtOpenInterestCapRejected",
        "positionFlipAtOpenInterestCapRejected",
        "tooAggressiveAtOpenInterestCapRejected",
        "openInterestIncreaseRejected",
        "insufficientSpotBalanceRejected",
        "oracleRejected",
        "perpMaxPositionRejected",
    }
)


class OrderManagerError(executor.ValidationError):
    """Raised when ownership, lifecycle, or reconciliation cannot be proven."""


@dataclass(frozen=True)
class MakerOrderManagerConfig:
    task_id: str
    run_id: str
    window_id: int = 1
    symbol: str = executor.SYMBOL
    max_levels_per_side: int = 1
    min_price_move_ticks: float = 1.0
    min_quote_age_ms: int = 250
    post_only_reject_cooldown_ms: int = 1_000
    max_cancel_readds_per_side_per_minute: int = 6
    terminal_query_max_rounds: int = 5
    historical_fallback_max_calls_per_reference: int = 1

    def __post_init__(self) -> None:
        if not self.task_id or not self.run_id:
            raise ValueError("manager_task_and_run_id_required")
        if isinstance(self.window_id, bool) or not isinstance(self.window_id, int) or self.window_id < 1:
            raise ValueError("manager_window_id_invalid")
        if self.symbol != executor.SYMBOL:
            raise ValueError("manager_symbol_must_be_btc")
        if self.max_levels_per_side != 1:
            raise ValueError("manager_only_single_level_supported")
        if self.min_price_move_ticks < 0:
            raise ValueError("manager_min_price_move_ticks_invalid")
        if self.min_quote_age_ms < 0:
            raise ValueError("manager_min_quote_age_ms_invalid")
        if self.post_only_reject_cooldown_ms < 0:
            raise ValueError("manager_reject_cooldown_invalid")
        if self.max_cancel_readds_per_side_per_minute <= 0:
            raise ValueError("manager_cancel_readd_rate_limit_invalid")
        if (
            isinstance(self.terminal_query_max_rounds, bool)
            or not isinstance(self.terminal_query_max_rounds, int)
            or not 1 <= self.terminal_query_max_rounds <= 5
        ):
            raise ValueError("manager_terminal_query_max_rounds_invalid")
        if (
            isinstance(
                self.historical_fallback_max_calls_per_reference,
                bool,
            )
            or self.historical_fallback_max_calls_per_reference != 1
        ):
            raise ValueError(
                "manager_historical_fallback_call_limit_invalid"
            )

    @property
    def ownership_prefix(self) -> str:
        return executor.managed_cloid_prefix(task_id=self.task_id, run_id=self.run_id)


@dataclass(frozen=True)
class DesiredQuote:
    side: str
    size_btc: float
    limit_px: float
    reduce_only: bool = False

    def __post_init__(self) -> None:
        if self.side not in {"buy", "sell"}:
            raise ValueError("desired_quote_side_invalid")
        if self.size_btc <= 0 or self.limit_px <= 0:
            raise ValueError("desired_quote_size_or_price_invalid")
        if self.reduce_only:
            raise ValueError("manager_reduce_only_not_enabled_in_task6")


@dataclass
class ManagedOrder:
    logical_key: tuple[str, str, str]
    symbol: str
    side: str
    canonical_price_key: str
    limit_px: float
    size_btc: float
    generation: int
    cloid: str
    oid: int | None = None
    state: str = "desired"
    leaves_qty: float = 0.0
    filled_qty: float = 0.0
    created_at_ms: int = 0
    updated_at_ms: int = 0
    cancel_requested_at_ms: int | None = None
    last_error: str = ""
    last_query_status: str = ""
    fill_evidence: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.state not in ORDER_STATES:
            raise ValueError("managed_order_state_invalid")

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_key": list(self.logical_key),
            "symbol": self.symbol,
            "side": self.side,
            "canonical_price_key": self.canonical_price_key,
            "limit_px": self.limit_px,
            "size_btc": self.size_btc,
            "generation": self.generation,
            "cloid": self.cloid,
            "oid": self.oid,
            "state": self.state,
            "leaves_qty": self.leaves_qty,
            "filled_qty": self.filled_qty,
            "created_at_ms": self.created_at_ms,
            "updated_at_ms": self.updated_at_ms,
            "cancel_requested_at_ms": self.cancel_requested_at_ms,
            "last_error": self.last_error,
            "last_query_status": self.last_query_status,
            "fill_evidence": list(self.fill_evidence),
        }


def _now_ms() -> int:
    return int(time.time() * 1000)


def _float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise OrderManagerError(f"{field}_not_numeric") from exc
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise OrderManagerError(f"{field}_not_finite")
    return parsed


def _side_from_exchange(row: dict[str, Any]) -> str:
    raw = str(row.get("side") or "").lower()
    if raw in {"b", "buy", "bid", "long"}:
        return "buy"
    if raw in {"a", "ask", "sell", "short"}:
        return "sell"
    raise OrderManagerError("exchange_order_side_unknown")


def _symbol(row: dict[str, Any]) -> str:
    aliases = [
        row[key]
        for key in ("coin", "symbol")
        if key in row and row[key] not in (None, "")
    ]
    if not aliases:
        return ""
    if any(
        isinstance(value, bool) or not isinstance(value, str)
        for value in aliases
    ):
        raise OrderManagerError("exchange_order_symbol_invalid")
    if len(set(aliases)) != 1:
        raise OrderManagerError("exchange_order_symbol_alias_conflict")
    return aliases[0]


def _numeric_alias(
    row: dict[str, Any],
    aliases: tuple[str, ...],
    field: str,
) -> float:
    raw_values = [
        row[key]
        for key in aliases
        if key in row and row[key] not in (None, "")
    ]
    if not raw_values:
        return _float(None, field)
    parsed = [_float(raw, field) for raw in raw_values]
    if len(set(parsed)) != 1:
        raise OrderManagerError(f"{field}_alias_conflict")
    return parsed[0]


def _limit_px(row: dict[str, Any]) -> float:
    return _numeric_alias(
        row,
        ("limitPx", "limit_px", "px"),
        "exchange_order_limit_px",
    )


def _oid(row: dict[str, Any]) -> int | None:
    aliases = [
        row[key]
        for key in ("oid", "orderId", "order_id")
        if key in row and row[key] not in (None, "")
    ]
    if not aliases:
        return None
    parsed: list[int] = []
    for raw in aliases:
        if isinstance(raw, bool):
            raise OrderManagerError("exchange_order_oid_invalid")
        if isinstance(raw, int):
            value = raw
        elif isinstance(raw, str) and raw.isdigit():
            value = int(raw)
        else:
            raise OrderManagerError("exchange_order_oid_invalid")
        if value < 0:
            raise OrderManagerError("exchange_order_oid_invalid")
        parsed.append(value)
    if len(set(parsed)) != 1:
        raise OrderManagerError("exchange_order_oid_alias_conflict")
    return parsed[0]


def _cloid(row: dict[str, Any]) -> str:
    aliases = [
        row[key]
        for key in ("cloid", "clientOrderId", "client_order_id")
        if key in row and row[key] not in (None, "")
    ]
    if not aliases:
        return ""
    if any(
        isinstance(value, bool) or not isinstance(value, str)
        for value in aliases
    ):
        raise OrderManagerError("exchange_order_cloid_invalid")
    if len(set(aliases)) != 1:
        raise OrderManagerError("exchange_order_cloid_alias_conflict")
    return aliases[0]


def _declared_size(row: dict[str, Any]) -> float | None:
    if not any(
        row.get(alias) not in (None, "")
        for alias in ("sz", "size")
    ):
        return None
    value = _numeric_alias(
        row,
        ("sz", "size"),
        "exchange_order_size",
    )
    if value <= 0:
        raise OrderManagerError("exchange_order_size_not_positive")
    return value


def _remaining_size(row: dict[str, Any]) -> float:
    declared_size = _declared_size(row)
    if row.get("remainingSz") not in (None, ""):
        value = _float(
            row["remainingSz"],
            "exchange_order_remaining_size",
        )
        if (
            declared_size is not None
            and value > declared_size + 1e-12
        ):
            raise OrderManagerError(
                "exchange_order_remaining_size_exceeds_declared_size"
            )
    elif declared_size is not None:
        value = declared_size
    else:
        value = _numeric_alias(
            row,
            ("sz", "size"),
            "exchange_order_size",
        )
    if value <= 0:
        raise OrderManagerError("exchange_order_size_not_positive")
    return value


def _classify_order_payload(payload: Any) -> str:
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return "unknown"
    response = payload.get("response")
    data = response.get("data") if isinstance(response, dict) else None
    statuses = data.get("statuses") if isinstance(data, dict) else None
    if not isinstance(statuses, list) or len(statuses) != 1:
        return "unknown"
    status = statuses[0]
    if not isinstance(status, dict):
        return "unknown"
    if set(status) == {"error"} and isinstance(status["error"], str):
        return "rejected"
    if set(status) == {"resting"} and isinstance(status["resting"], dict):
        return "resting"
    if set(status) == {"filled"} and isinstance(status["filled"], dict):
        return "filled"
    return "unknown"


def _status_classification(status: Any) -> str:
    if not isinstance(status, str):
        return "unknown"
    if status == "open":
        return "resting"
    if status == "filled":
        return "filled"
    if status in ORDER_STATUS_CANCEL_CONFIRMED:
        return "cancel_confirmed"
    if status in ORDER_STATUS_REJECTED:
        return "rejected"
    return "unknown"


def _order_row_matches_reference(
    row: Any,
    *,
    expected_oid: int | None,
    expected_cloid: str,
) -> bool:
    if not isinstance(row, dict):
        return False
    if expected_oid is not None:
        try:
            actual_oid = _oid(row)
        except OrderManagerError:
            return False
        if actual_oid != expected_oid:
            return False
    if expected_cloid:
        try:
            actual_cloid = _cloid(row)
        except OrderManagerError:
            return False
        if actual_cloid != expected_cloid:
            return False
    return expected_oid is not None or bool(expected_cloid)


def _classify_order_status_query_payload(
    payload: Any,
    *,
    expected_oid: int | None = None,
    expected_cloid: str = "",
    require_embedded_reference: bool = False,
) -> str:
    if not isinstance(payload, dict):
        return "unknown"
    status = payload.get("status")
    if not isinstance(status, str):
        return "unknown"
    if status != "order":
        supplied_order = payload.get("order")
        if require_embedded_reference and not isinstance(
            supplied_order,
            dict,
        ):
            return "unknown"
        if "order" in payload:
            if not _order_row_matches_reference(
                supplied_order,
                expected_oid=expected_oid,
                expected_cloid=expected_cloid,
            ):
                return "unknown"
        return _status_classification(status)
    envelope = payload.get("order")
    if not isinstance(envelope, dict):
        return "unknown"
    if not _order_row_matches_reference(
        envelope.get("order"),
        expected_oid=expected_oid,
        expected_cloid=expected_cloid,
    ):
        return "unknown"
    return _status_classification(envelope.get("status"))


def _historical_order_status_payload(
    rows: Any,
    *,
    expected_oid: int | None,
    expected_cloid: str,
) -> dict[str, Any]:
    if not isinstance(rows, list):
        raise OrderManagerError("historical_orders_payload_not_list")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and _order_row_matches_reference(
            row.get("order"),
            expected_oid=expected_oid,
            expected_cloid=expected_cloid,
        )
    ]
    if not matches:
        raise OrderManagerError("historical_order_exact_match_missing")
    if len(matches) != 1:
        raise OrderManagerError("historical_order_exact_match_ambiguous")
    return {"status": "order", "order": matches[0]}


class MakerOrderManager:
    """Single-level state machine with exchange state as lifecycle authority."""

    def __init__(
        self,
        *,
        client: executor.HyperliquidClient,
        precision: executor.PrecisionFacts,
        config: MakerOrderManagerConfig,
        runtime_config: executor.TinyLiveConfig | None = None,
        account_address: str | None = None,
        now_ms: int | None = None,
    ) -> None:
        if precision.symbol != config.symbol:
            raise OrderManagerError("manager_precision_symbol_mismatch")
        self.client = client
        self.precision = precision
        self.config = config
        self.runtime_config = runtime_config or executor.TinyLiveConfig(
            symbol=config.symbol,
            max_real_order_submissions=executor.MAX_REAL_ORDER_SUBMISSIONS,
            max_order_size_btc=executor.MAX_ORDER_SIZE_BTC,
            max_position_btc=executor.MAX_POSITION_BTC,
            max_order_notional_usdc=executor.MAX_ORDER_NOTIONAL_USDC,
            max_position_notional_usdc=executor.MAX_POSITION_NOTIONAL_USDC,
            max_notional_usdc=executor.MAX_NOTIONAL_USDC,
        )
        self.account_address = account_address
        self.orders_by_key: dict[tuple[str, str, str], ManagedOrder] = {}
        self.generation_by_key: dict[tuple[str, str, str], int] = {}
        self.cancel_events_by_side: dict[str, list[int]] = {"buy": [], "sell": []}
        self.last_rejected_at_ms: dict[str, int] = {}
        self.submissions_used = 0
        self.current_position_btc = 0.0
        self.position_evidence: list[dict[str, Any]] = []
        self.terminal_query_evidence: list[dict[str, Any]] = []
        self.historical_query_counts_by_cloid: dict[str, int] = {}
        self.terminal_query_sequence_by_phase: dict[str, int] = {}
        self.last_reconciliation: dict[str, Any] = {}
        self.last_exchange_open_orders: list[dict[str, Any]] = []
        self._default_now_ms = now_ms

    def _time(self, now_ms: int | None) -> int:
        if now_ms is not None:
            return now_ms
        if self._default_now_ms is not None:
            return self._default_now_ms
        return _now_ms()

    @staticmethod
    def multi_level_prerequisite_gate(
        *,
        requested_levels: int,
        activation_enabled: bool = False,
        single_level_lifecycle_prerequisite: bool = False,
    ) -> dict[str, Any]:
        if isinstance(requested_levels, bool) or not isinstance(requested_levels, int) or requested_levels < 1:
            raise OrderManagerError("requested_levels_must_be_positive_integer")
        if requested_levels == 1:
            return {
                "status": "single_level_authoritative",
                "reason": "multi_level_not_requested",
                "requested_levels": requested_levels,
                "activation_enabled": False,
                "single_level_lifecycle_prerequisite": single_level_lifecycle_prerequisite,
                "actual_quote_behavior_changed": False,
            }
        if not single_level_lifecycle_prerequisite:
            reason = "single_level_lifecycle_prerequisite_not_satisfied"
        elif not activation_enabled:
            reason = "multi_level_activation_disabled"
        else:
            reason = ""
        return {
            "status": "pass" if not reason else "blocked",
            "reason": reason,
            "requested_levels": requested_levels,
            "activation_enabled": bool(not reason),
            "single_level_lifecycle_prerequisite": single_level_lifecycle_prerequisite,
            "actual_quote_behavior_changed": False,
        }

    def logical_key(self, side: str, limit_px: float) -> tuple[str, str, str]:
        if side not in {"buy", "sell"}:
            raise OrderManagerError("logical_key_side_invalid")
        price_key = executor.canonical_price_key(limit_px, sz_decimals=self.precision.sz_decimals)
        return (self.config.symbol, side, price_key)

    def _next_generation(self, logical_key: tuple[str, str, str]) -> int:
        generation = self.generation_by_key.get(logical_key, -1) + 1
        self.generation_by_key[logical_key] = generation
        return generation

    def _active_for_side(self, side: str) -> ManagedOrder | None:
        active = [
            order
            for order in self.orders_by_key.values()
            if order.side == side and order.is_active
        ]
        if len(active) > 1:
            raise OrderManagerError(f"duplicate_active_owned_orders:{side}")
        return active[0] if active else None

    def _owned_exchange_order(
        self,
        row: dict[str, Any],
        *,
        restore_cancel_requested: bool = False,
    ) -> ManagedOrder | None:
        cloid = _cloid(row)
        if not executor.is_owned_managed_cloid(
            cloid,
            task_id=self.config.task_id,
            run_id=self.config.run_id,
        ):
            return None
        if _symbol(row) != self.config.symbol:
            raise OrderManagerError("owned_exchange_order_symbol_mismatch")
        side = _side_from_exchange(row)
        limit_px = _limit_px(row)
        size = _remaining_size(row)
        key = self.logical_key(side, limit_px)
        existing = self.orders_by_key.get(key)
        if existing and existing.cloid != cloid:
            raise OrderManagerError("duplicate_owned_logical_quote_key")
        if existing:
            order = existing
            prior_state = order.state
            prior_query_status = order.last_query_status
            order.oid = _oid(row) or order.oid
            order.leaves_qty = size
            order.filled_qty = max(0.0, order.size_btc - size)
            if order.state == "cancel_requested":
                if not restore_cancel_requested:
                    order.updated_at_ms = self._time(None)
                    return order
                order.state = "partial_fill" if order.filled_qty > 0 else "resting"
                order.last_query_status = "resting"
                order.last_error = "cancel_requested_but_still_open"
            else:
                order.state = "partial_fill" if order.filled_qty > 0 else "resting"
                order.last_query_status = "resting"
                if prior_state in {
                    "cancel_confirmed",
                    "filled",
                    "rejected",
                } or prior_query_status in {
                    "cancel_confirmed",
                    "filled",
                    "rejected",
                }:
                    order.last_error = (
                        "terminal_query_contradicted_by_open_order"
                    )
                else:
                    order.last_error = ""
            order.updated_at_ms = self._time(None)
            return order
        generation = self.generation_by_key.get(key, 0)
        order = ManagedOrder(
            logical_key=key,
            symbol=self.config.symbol,
            side=side,
            canonical_price_key=key[2],
            limit_px=limit_px,
            size_btc=size,
            generation=generation,
            cloid=cloid,
            oid=_oid(row),
            state="resting",
            leaves_qty=size,
            created_at_ms=self._time(None),
            updated_at_ms=self._time(None),
        )
        self.orders_by_key[key] = order
        self.generation_by_key[key] = max(self.generation_by_key.get(key, -1), generation)
        return order

    def _set_position_from_user_state(
        self,
        user_state: dict[str, Any],
        *,
        now_ms: int,
        source: str,
    ) -> None:
        if not isinstance(user_state.get("assetPositions"), list):
            raise OrderManagerError(
                "exchange_user_state_asset_positions_not_list"
            )
        self.current_position_btc = executor.extract_position_szi(user_state, symbol=self.config.symbol)
        self.position_evidence.append(
            {
                "timestamp_ms": now_ms,
                "position_btc": self.current_position_btc,
                "source": source,
            }
        )

    def _refresh_position(self, now_ms: int) -> None:
        user_state = self.client.user_state(self.account_address)
        if not isinstance(user_state, dict):
            raise OrderManagerError("exchange_user_state_not_object")
        self._set_position_from_user_state(
            user_state,
            now_ms=now_ms,
            source="exchange_user_state",
        )

    def startup_reconcile(self, *, now_ms: int | None = None) -> dict[str, Any]:
        return self.reconcile_exchange(now_ms=now_ms, reason="startup")

    def reconnect_reconcile(self, *, now_ms: int | None = None) -> dict[str, Any]:
        return self.reconcile_exchange(now_ms=now_ms, reason="reconnect")

    def _reconcile_open_orders(
        self,
        *,
        open_orders: list[dict[str, Any]],
        timestamp: int,
        reason: str,
        query_missing: bool,
        terminal_query_deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        owned_count = 0
        foreign_count = 0
        owned_cloids: set[str] = set()
        owned_oids: set[int] = set()
        seen_exchange_oids: set[int] = set()
        known_by_oid: dict[int, ManagedOrder] = {}
        known_by_cloid: dict[str, ManagedOrder] = {}
        for managed_order in self.orders_by_key.values():
            known_by_cloid[managed_order.cloid] = managed_order
            if managed_order.oid is None:
                continue
            if managed_order.oid in known_by_oid:
                raise OrderManagerError("duplicate_managed_order_oid")
            known_by_oid[managed_order.oid] = managed_order
        for row in open_orders:
            if not isinstance(row, dict):
                raise OrderManagerError("exchange_open_order_not_object")
            row_oid = _oid(row)
            if row_oid is not None:
                if row_oid in seen_exchange_oids:
                    raise OrderManagerError("duplicate_exchange_order_oid")
                seen_exchange_oids.add(row_oid)
            cloid_error = ""
            try:
                cloid = _cloid(row)
            except OrderManagerError as exc:
                cloid = ""
                cloid_error = str(exc)
            oid_match = (
                known_by_oid.get(row_oid)
                if row_oid is not None
                else None
            )
            cloid_match = known_by_cloid.get(cloid) if cloid else None
            if (
                cloid_match is not None
                and row_oid is not None
                and cloid_match.oid is not None
                and cloid_match.oid != row_oid
            ):
                raise OrderManagerError("exchange_order_oid_conflict")
            if (
                oid_match is not None
                and cloid_match is not None
                and cloid_match is not oid_match
            ):
                raise OrderManagerError("exchange_order_oid_conflict")
            if oid_match is not None:
                if _symbol(row) != oid_match.symbol:
                    raise OrderManagerError(
                        "tracked_oid_open_order_symbol_mismatch"
                    )
                row_side = _side_from_exchange(row)
                row_limit_px = _limit_px(row)
                if (
                    self.logical_key(row_side, row_limit_px)
                    != oid_match.logical_key
                ):
                    raise OrderManagerError(
                        "tracked_oid_open_order_logical_key_mismatch"
                    )
                if row_limit_px != oid_match.limit_px:
                    raise OrderManagerError(
                        "tracked_oid_open_order_price_mismatch"
                    )
                row_remaining_size = _remaining_size(row)
                row_declared_size = _declared_size(row)
                if (
                    row_remaining_size > oid_match.size_btc + 1e-12
                    or (
                        row_declared_size is not None
                        and row_declared_size
                        > oid_match.size_btc + 1e-12
                    )
                ):
                    raise OrderManagerError(
                        "tracked_oid_open_order_size_exceeds_original"
                    )
            recovery_order: ManagedOrder | None = None
            reconciled_row = row
            if oid_match is not None and cloid != oid_match.cloid:
                recovery_order = oid_match
                reconciled_row = dict(row)
                for alias in (
                    "cloid",
                    "clientOrderId",
                    "client_order_id",
                ):
                    reconciled_row.pop(alias, None)
                reconciled_row["cloid"] = oid_match.cloid
                cloid = oid_match.cloid
            elif cloid_error:
                raise OrderManagerError(cloid_error)
            if not executor.is_owned_managed_cloid(
                cloid,
                task_id=self.config.task_id,
                run_id=self.config.run_id,
            ):
                foreign_count += 1
                continue
            if cloid in owned_cloids:
                raise OrderManagerError("duplicate_owned_exchange_cloid")
            owned_cloids.add(cloid)
            if row_oid is not None:
                owned_oids.add(row_oid)
            owned_count += 1
            reconciled_order = self._owned_exchange_order(
                reconciled_row,
                restore_cancel_requested=not query_missing,
            )
            if recovery_order is not None and reconciled_order is not None:
                reconciled_order.last_error = (
                    "open_order_cloid_missing_or_mismatched_for_tracked_oid"
                )

        for order in list(self.orders_by_key.values()):
            if not order.is_active:
                continue
            if order.cloid in owned_cloids or (
                order.oid is not None and order.oid in owned_oids
            ):
                continue
            if order.state == "cancel_requested":
                order.state = "cancel_confirmed"
                order.leaves_qty = 0.0
                order.updated_at_ms = timestamp
                continue
            if order.state in {"submit_inflight", "unknown"}:
                if query_missing:
                    status = self._query_ambiguous(
                        order,
                        phase=reason,
                        deadline_monotonic=(
                            terminal_query_deadline_monotonic
                        ),
                    )
                    order.last_query_status = status
                    if status == "resting":
                        order.state = "resting"
                    elif status == "cancel_confirmed":
                        order.state = "cancel_confirmed"
                        order.leaves_qty = 0.0
                    elif status == "filled":
                        order.state = "unknown"
                        order.last_error = (
                            "query_filled_requires_raw_fill_proof"
                        )
                    elif status == "rejected":
                        order.state = "rejected"
                        order.leaves_qty = 0.0
                    else:
                        order.state = "unknown"
                else:
                    order.state = "unknown"
                    if order.last_query_status == "filled":
                        order.last_error = (
                            "query_filled_requires_raw_fill_proof"
                        )
                order.updated_at_ms = timestamp
                continue
            if order.state in {"resting", "partial_fill"}:
                order.state = "unknown"
                order.last_query_status = "missing_from_open_orders"
                order.updated_at_ms = timestamp

        self.last_reconciliation = {
            "reason": reason,
            "timestamp_ms": timestamp,
            "open_order_count": len(open_orders),
            "owned_order_count": owned_count,
            "foreign_order_count": foreign_count,
            "owned_cloids": sorted(owned_cloids),
        }
        self.last_exchange_open_orders = [
            dict(row) for row in open_orders
        ]
        return dict(self.last_reconciliation)

    def reconcile_exchange(
        self,
        *,
        now_ms: int | None = None,
        reason: str = "periodic",
        terminal_query_deadline_monotonic: float | None = None,
    ) -> dict[str, Any]:
        timestamp = self._time(now_ms)
        sdk_timeout_seconds: float | None = None
        if terminal_query_deadline_monotonic is not None:
            remaining_seconds = (
                terminal_query_deadline_monotonic - time.monotonic()
            )
            if remaining_seconds <= 0:
                raise OrderManagerError(
                    "terminal_query_deadline_exhausted_before_open_orders"
                )
            if isinstance(
                self.client,
                executor.SDKHyperliquidClient,
            ):
                sdk_timeout_seconds = remaining_seconds
        if isinstance(self.client, executor.SDKHyperliquidClient):
            open_orders = self.client.open_orders(
                self.account_address,
                timeout_seconds=sdk_timeout_seconds,
            )
        else:
            open_orders = self.client.open_orders(self.account_address)
        if not isinstance(open_orders, list):
            raise OrderManagerError("exchange_open_orders_not_list")
        reconciliation = self._reconcile_open_orders(
            open_orders=open_orders,
            timestamp=timestamp,
            reason=reason,
            query_missing=True,
            terminal_query_deadline_monotonic=(
                terminal_query_deadline_monotonic
            ),
        )
        if terminal_query_deadline_monotonic is None:
            self._refresh_position(timestamp)
        else:
            self.last_reconciliation["position_refresh_status"] = (
                "deferred_terminal_query_budget"
            )
        return dict(self.last_reconciliation)

    def reconcile_supplied_open_orders(
        self,
        *,
        open_orders: list[dict[str, Any]],
        now_ms: int | None = None,
        reason: str = "supplied_open_orders",
    ) -> dict[str, Any]:
        timestamp = self._time(now_ms)
        if not isinstance(open_orders, list):
            raise OrderManagerError("exchange_open_orders_not_list")
        return self._reconcile_open_orders(
            open_orders=open_orders,
            timestamp=timestamp,
            reason=reason,
            query_missing=False,
            terminal_query_deadline_monotonic=None,
        )

    def reconcile_supplied_snapshot(
        self,
        *,
        open_orders: list[dict[str, Any]],
        user_state: dict[str, Any] | None,
        now_ms: int | None = None,
        reason: str = "supplied_snapshot",
    ) -> dict[str, Any]:
        timestamp = self._time(now_ms)
        if not isinstance(open_orders, list):
            raise OrderManagerError("exchange_open_orders_not_list")
        reconciliation = self._reconcile_open_orders(
            open_orders=open_orders,
            timestamp=timestamp,
            reason=reason,
            query_missing=False,
            terminal_query_deadline_monotonic=None,
        )
        try:
            if not isinstance(user_state, dict):
                raise OrderManagerError("exchange_user_state_not_object")
            self._set_position_from_user_state(
                user_state,
                now_ms=timestamp,
                source="supplied_final_user_state",
            )
        except Exception as exc:
            self.last_reconciliation["position_snapshot_status"] = (
                "fail_closed"
            )
            self.last_reconciliation[
                "position_snapshot_reason"
            ] = "final_user_state_unavailable_or_invalid"
            self.last_reconciliation[
                "position_snapshot_error"
            ] = executor._redacted_error(exc)
        else:
            self.last_reconciliation["position_snapshot_status"] = "pass"
            self.last_reconciliation["position_snapshot_reason"] = ""
        return dict(self.last_reconciliation)

    def _record_order_query(
        self,
        order: ManagedOrder,
        *,
        method: str,
        phase: str,
        deadline_monotonic: float | None = None,
    ) -> str:
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            return "unknown"
        sdk_timeout_seconds: float | None = None
        if isinstance(self.client, executor.SDKHyperliquidClient):
            sdk_timeout_seconds = (
                executor.DEFAULT_INFO_REQUEST_TIMEOUT_SECONDS
                if deadline_monotonic is None
                else deadline_monotonic - time.monotonic()
            )
            if sdk_timeout_seconds <= 0:
                return "unknown"
        query_started_ms = _now_ms()
        payload: Any = None
        status_payload: Any = None
        sequence_scope = (
            "post_cycle_terminal"
            if phase.startswith("post_cycle_")
            else phase
        )
        query_sequence = (
            self.terminal_query_sequence_by_phase.get(sequence_scope, 0) + 1
        )
        self.terminal_query_sequence_by_phase[
            sequence_scope
        ] = query_sequence
        try:
            if method == "query_order_by_oid":
                if order.oid is None:
                    raise OrderManagerError("query_order_by_oid_missing_oid")
                if isinstance(
                    self.client,
                    executor.SDKHyperliquidClient,
                ):
                    payload = self.client.query_order_by_oid(
                        order.oid,
                        self.account_address,
                        timeout_seconds=sdk_timeout_seconds,
                    )
                else:
                    payload = self.client.query_order_by_oid(
                        order.oid,
                        self.account_address,
                    )
            elif method == "query_order_by_cloid":
                if isinstance(
                    self.client,
                    executor.SDKHyperliquidClient,
                ):
                    payload = self.client.query_order_by_cloid(
                        order.cloid,
                        self.account_address,
                        timeout_seconds=sdk_timeout_seconds,
                    )
                else:
                    payload = self.client.query_order_by_cloid(
                        order.cloid,
                        self.account_address,
                    )
            elif method == "historical_orders":
                historical_method = getattr(
                    self.client,
                    "historical_orders",
                    None,
                )
                if not callable(historical_method):
                    raise OrderManagerError(
                        "historical_orders_method_unavailable"
                    )
                count = self.historical_query_counts_by_cloid.get(
                    order.cloid,
                    0,
                )
                if (
                    count
                    >= self.config.historical_fallback_max_calls_per_reference
                ):
                    raise OrderManagerError(
                        "historical_orders_call_budget_exhausted"
                    )
                self.historical_query_counts_by_cloid[order.cloid] = (
                    count + 1
                )
                if isinstance(
                    self.client,
                    executor.SDKHyperliquidClient,
                ):
                    historical_rows = historical_method(
                        self.account_address,
                        timeout_seconds=sdk_timeout_seconds,
                    )
                else:
                    historical_rows = historical_method(
                        self.account_address
                    )
                payload = {
                    "status": "historical_orders",
                    "orders": historical_rows,
                }
                status_payload = _historical_order_status_payload(
                    historical_rows,
                    expected_oid=order.oid,
                    expected_cloid=order.cloid,
                )
            else:
                raise OrderManagerError("terminal_query_method_invalid")
            if status_payload is None:
                status_payload = payload
            status = _classify_order_status_query_payload(
                status_payload,
                expected_oid=order.oid,
                expected_cloid=order.cloid,
                require_embedded_reference=(
                    sequence_scope == "post_cycle_terminal"
                ),
            )
        except Exception as exc:
            status = "unknown"
            evidence = {
                "phase": phase,
                "method": method,
                "oid": order.oid,
                "cloid": order.cloid,
                "query_started_ms": query_started_ms,
                "query_ended_ms": _now_ms(),
                "query_status": status,
                "query_sequence": query_sequence,
                "error": executor.redact_with_reference_tokens(
                    str(exc),
                    known_oid=order.oid,
                    known_cloid=order.cloid,
                ),
                **(
                    {
                        "result": executor.redact_with_reference_tokens(
                            payload,
                            known_oid=order.oid,
                            known_cloid=order.cloid,
                        )
                    }
                    if payload is not None
                    else {}
                ),
            }
        else:
            evidence = {
                "phase": phase,
                "method": method,
                "oid": order.oid,
                "cloid": order.cloid,
                "query_started_ms": query_started_ms,
                "query_ended_ms": _now_ms(),
                "query_status": status,
                "query_sequence": query_sequence,
                "result": executor.redact_with_reference_tokens(
                    payload,
                    known_oid=order.oid,
                    known_cloid=order.cloid,
                ),
            }
        self.terminal_query_evidence.append(evidence)
        return status

    def _query_ambiguous(
        self,
        order: ManagedOrder,
        *,
        phase: str,
        deadline_monotonic: float | None = None,
    ) -> str:
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            return "unknown"
        if order.oid is not None and hasattr(self.client, "query_order_by_oid"):
            status = self._record_order_query(
                order,
                method="query_order_by_oid",
                phase=phase,
                deadline_monotonic=deadline_monotonic,
            )
            if status != "unknown":
                return status
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            return "unknown"
        if hasattr(self.client, "query_order_by_cloid"):
            return self._record_order_query(
                order,
                method="query_order_by_cloid",
                phase=phase,
                deadline_monotonic=deadline_monotonic,
            )
        return "unknown"

    def reconcile_historical_terminal(
        self,
        order: ManagedOrder,
        *,
        phase: str,
        now_ms: int | None = None,
        deadline_monotonic: float | None = None,
    ) -> str:
        timestamp = self._time(now_ms)
        if not order.is_active or order.last_query_status != "unknown":
            raise OrderManagerError(
                "historical_fallback_requires_active_unknown_reference"
            )
        if (
            deadline_monotonic is not None
            and time.monotonic() >= deadline_monotonic
        ):
            status = "unknown"
            order.last_error = "terminal_query_deadline_exhausted"
        else:
            status = self._record_order_query(
                order,
                method="historical_orders",
                phase=phase,
                deadline_monotonic=deadline_monotonic,
            )
        order.last_query_status = status
        if status == "cancel_confirmed":
            order.state = "cancel_confirmed"
            order.leaves_qty = 0.0
        elif status == "rejected":
            order.state = "rejected"
            order.leaves_qty = 0.0
        elif status == "filled":
            order.state = "unknown"
            order.last_error = "query_filled_requires_raw_fill_proof"
        elif status == "resting":
            order.state = "resting"
        else:
            order.state = "unknown"
        order.updated_at_ms = timestamp
        return status

    def working_exposure(self) -> executor.ProjectedExposure:
        buy = sum(order.leaves_qty for order in self.orders_by_key.values() if order.is_active and order.side == "buy")
        sell = sum(order.leaves_qty for order in self.orders_by_key.values() if order.is_active and order.side == "sell")
        return executor.projected_exposure(
            position_btc=self.current_position_btc,
            working_buy_qty=buy,
            working_sell_qty=sell,
            inflight_buy_qty=0.0,
            inflight_sell_qty=0.0,
            existing_max_quote_px=max(
                (order.limit_px for order in self.orders_by_key.values() if order.is_active),
                default=None,
            ),
        )

    def _intent(self, quote: DesiredQuote, *, generation: int) -> executor.OrderIntent:
        key = self.logical_key(quote.side, quote.limit_px)
        cloid = executor.generate_managed_cloid(
            task_id=self.config.task_id,
            run_id=self.config.run_id,
            window_id=self.config.window_id,
            side=quote.side,
            canonical_price=key[2],
            generation=generation,
        )
        return executor.OrderIntent(
            symbol=self.config.symbol,
            is_buy=quote.side == "buy",
            size_btc=quote.size_btc,
            limit_px=quote.limit_px,
            reduce_only=quote.reduce_only,
            cloid=cloid,
        )

    def _can_cancel(self, current: ManagedOrder, desired: DesiredQuote, now_ms: int, emergency: bool) -> tuple[bool, str]:
        if emergency:
            return True, ""
        if current.state in {"submit_inflight", "cancel_requested", "unknown"}:
            return False, f"state_guard:{current.state}"
        if now_ms - current.created_at_ms < self.config.min_quote_age_ms:
            return False, "min_quote_age_guard"
        price_delta_ticks = abs(desired.limit_px - current.limit_px) / self.precision.tick_size
        if price_delta_ticks < self.config.min_price_move_ticks:
            return False, "min_price_move_guard"
        events = [
            event
            for event in self.cancel_events_by_side[current.side]
            if now_ms - event < 60_000
        ]
        self.cancel_events_by_side[current.side] = events
        if len(events) >= self.config.max_cancel_readds_per_side_per_minute:
            return False, "cancel_readd_rate_limit"
        rejected_at = self.last_rejected_at_ms.get(current.side)
        if rejected_at is not None and now_ms - rejected_at < self.config.post_only_reject_cooldown_ms:
            return False, "post_only_reject_cooldown"
        return True, ""

    def _request_cancel(self, current: ManagedOrder, *, now_ms: int, emergency: bool) -> dict[str, Any]:
        if current.state == "cancel_requested":
            return {"action": "cancel_pending", "cloid": current.cloid}
        if current.state not in {"resting", "partial_fill"}:
            return {"action": "cancel_skipped", "reason": f"state:{current.state}", "cloid": current.cloid}
        cancel_request_ms = _now_ms()
        response: Any = None
        try:
            response = self.client.cancel_tracked(
                self.config.symbol,
                oid=current.oid,
                cloid=current.cloid if current.oid is None else None,
            )
            executor.assert_exchange_action_success(response, action="cancel")
        except Exception as exc:
            redacted_error = executor.redact_with_reference_tokens(
                str(exc),
                known_oid=current.oid,
                known_cloid=current.cloid,
            )
            current.state = "unknown"
            current.last_query_status = "unknown"
            current.last_error = str(redacted_error)
            current.updated_at_ms = now_ms
            return {
                "action": "cancel_unknown",
                "cloid": current.cloid,
                "oid": current.oid,
                "cancel_request_time_ms": cancel_request_ms,
                "cancel_ack_time_ms": _now_ms(),
                "reason": redacted_error,
                **(
                    {
                        "result": executor.redact_with_reference_tokens(
                            response,
                            known_oid=current.oid,
                            known_cloid=current.cloid,
                        )
                    }
                    if response is not None
                    else {}
                ),
            }
        cancel_ack_ms = _now_ms()
        current.state = "cancel_requested"
        current.cancel_requested_at_ms = now_ms
        current.updated_at_ms = cancel_ack_ms
        self.cancel_events_by_side[current.side].append(cancel_ack_ms)
        return {
            "action": "cancel_requested",
            "cloid": current.cloid,
            "oid": current.oid,
            "emergency": emergency,
            "cancel_request_time_ms": cancel_request_ms,
            "cancel_ack_time_ms": cancel_ack_ms,
            "result": executor.redact_with_reference_tokens(
                response,
                known_oid=current.oid,
                known_cloid=current.cloid,
            ),
        }

    def cancel_all_owned(
        self,
        *,
        now_ms: int | None = None,
        emergency: bool = False,
    ) -> list[dict[str, Any]]:
        timestamp = self._time(now_ms)
        actions: list[dict[str, Any]] = []
        for order in list(self.orders_by_key.values()):
            if not order.is_active:
                continue
            if order.state in {"resting", "partial_fill"}:
                actions.append(self._request_cancel(order, now_ms=timestamp, emergency=emergency))
            elif order.state == "cancel_requested":
                actions.append({"action": "cancel_pending", "cloid": order.cloid, "oid": order.oid})
            else:
                actions.append(
                    {
                        "action": "cancel_blocked",
                        "cloid": order.cloid,
                        "oid": order.oid,
                        "reason": f"state:{order.state}",
                    }
                )
        return actions

    def _submit(self, quote: DesiredQuote, *, now_ms: int) -> dict[str, Any]:
        key = self.logical_key(quote.side, quote.limit_px)
        existing = self.orders_by_key.get(key)
        if existing and existing.is_active:
            return {"action": "deduplicated", "cloid": existing.cloid, "state": existing.state}
        generation = self._next_generation(key)
        intent = self._intent(quote, generation=generation)
        executor.validate_order_intent(self.runtime_config, self.precision, intent)
        projected = self.working_exposure()
        executor.validate_runtime_envelope(
            config=self.runtime_config,
            projected=projected,
            proposed_quotes=[intent],
            submissions_used=self.submissions_used,
        )
        order = ManagedOrder(
            logical_key=key,
            symbol=self.config.symbol,
            side=quote.side,
            canonical_price_key=key[2],
            limit_px=quote.limit_px,
            size_btc=quote.size_btc,
            generation=generation,
            cloid=intent.cloid,
            state="submit_inflight",
            leaves_qty=quote.size_btc,
            created_at_ms=now_ms,
            updated_at_ms=now_ms,
        )
        self.orders_by_key[key] = order
        self.submissions_used += 1
        submit_start_ms = _now_ms()
        order_result: dict[str, Any] | None = None
        try:
            response = self.client.order(intent)
        except Exception as exc:
            redacted_error = executor.redact_with_reference_tokens(
                str(exc),
                known_cloid=order.cloid,
            )
            order_result = {
                "status": "exception",
                "error": redacted_error,
            }
            order.state = "unknown"
            order.last_error = str(redacted_error)
            order.updated_at_ms = now_ms
            status = self._query_ambiguous(
                order,
                phase="submit_exception",
            )
            submit_end_ms = _now_ms()
        else:
            order_result = (
                dict(response)
                if isinstance(response, dict)
                else {"status": "invalid_response", "response": response}
            )
            submit_end_ms = _now_ms()
            refs = executor.extract_tracked_refs(response)
            if refs:
                order.oid = refs[0].get("oid")
                order.state = "resting"
                status = "resting"
            else:
                status = _classify_order_payload(response)
                if status == "filled":
                    order.state = "filled"
                    order.leaves_qty = 0.0
                elif status == "rejected":
                    order.state = "rejected"
                    order.leaves_qty = 0.0
                    self.last_rejected_at_ms[quote.side] = now_ms
                else:
                    order.state = "unknown"
                    status = self._query_ambiguous(
                        order,
                        phase="submit_response_ambiguous",
                    )
                    if status == "resting":
                        order.state = "resting"
                    elif status == "cancel_confirmed":
                        order.state = "cancel_confirmed"
                        order.leaves_qty = 0.0
                    elif status == "filled":
                        order.state = "unknown"
                        order.last_error = (
                            "query_filled_requires_raw_fill_proof"
                        )
                    elif status == "rejected":
                        order.state = "rejected"
                        order.leaves_qty = 0.0
                        self.last_rejected_at_ms[quote.side] = now_ms
        order.last_query_status = status
        order.updated_at_ms = now_ms
        return {
            "action": "submitted" if order.state == "resting" else order.state,
            "side": quote.side,
            "cloid": order.cloid,
            "oid": order.oid,
            "state": order.state,
            "query_status": status,
            "submit_start_ms": submit_start_ms,
            "submit_end_ms": submit_end_ms,
            "order_endpoint_called": True,
            "order_result": order_result,
        }

    def reconcile_desired(
        self,
        desired_quotes: Iterable[DesiredQuote],
        *,
        now_ms: int | None = None,
        emergency: bool = False,
        reconcile_exchange_first: bool = True,
    ) -> dict[str, Any]:
        timestamp = self._time(now_ms)
        desired = list(desired_quotes)
        if len(desired) > 2 or len({quote.side for quote in desired}) != len(desired):
            raise OrderManagerError("single_level_requires_at_most_one_quote_per_side")
        if reconcile_exchange_first:
            self.reconcile_exchange(now_ms=timestamp, reason="desired_reconcile")
        actions: list[dict[str, Any]] = []
        for quote in desired:
            current = self._active_for_side(quote.side)
            desired_key = self.logical_key(quote.side, quote.limit_px)
            if current is not None and current.logical_key == desired_key:
                actions.append({"action": "hold", "side": quote.side, "cloid": current.cloid, "state": current.state})
                continue
            if current is not None:
                allowed, reason = self._can_cancel(current, quote, timestamp, emergency)
                if not allowed:
                    actions.append({"action": "blocked", "side": quote.side, "reason": reason, "cloid": current.cloid})
                    continue
                actions.append(self._request_cancel(current, now_ms=timestamp, emergency=emergency))
                continue
            rejected_at = self.last_rejected_at_ms.get(quote.side)
            if (
                not emergency
                and rejected_at is not None
                and timestamp - rejected_at < self.config.post_only_reject_cooldown_ms
            ):
                actions.append(
                    {
                        "action": "blocked",
                        "side": quote.side,
                        "reason": "post_only_reject_cooldown",
                    }
                )
                continue
            actions.append(self._submit(quote, now_ms=timestamp))
        return {
            "timestamp_ms": timestamp,
            "actions": actions,
            "orders": [order.to_dict() for order in self.orders_by_key.values()],
            "working_exposure": self.working_exposure().__dict__,
            "submissions_used": self.submissions_used,
        }

    def apply_fill(
        self,
        *,
        fill_qty: float,
        fill_px: float,
        now_ms: int | None = None,
        oid: int | None = None,
        cloid: str | None = None,
        position_btc: float | None = None,
    ) -> dict[str, Any]:
        qty = _float(fill_qty, "fill_qty")
        px = _float(fill_px, "fill_px")
        if qty <= 0 or px <= 0:
            raise OrderManagerError("fill_qty_and_price_must_be_positive")
        matches = [
            order
            for order in self.orders_by_key.values()
            if (cloid and order.cloid == cloid) or (oid is not None and order.oid == oid)
        ]
        if len(matches) != 1:
            raise OrderManagerError("fill_order_reference_not_unique")
        order = matches[0]
        if qty > order.leaves_qty + 1e-12:
            raise OrderManagerError("fill_qty_exceeds_order_leaves")
        timestamp = self._time(now_ms)
        order.leaves_qty -= qty
        order.filled_qty += qty
        order.state = "filled" if order.leaves_qty <= 1e-12 else "partial_fill"
        order.updated_at_ms = timestamp
        order.fill_evidence.append({"timestamp_ms": timestamp, "fill_qty": qty, "fill_px": px})
        if position_btc is not None:
            self.current_position_btc = _float(position_btc, "position_btc")
        else:
            signed = qty if order.side == "buy" else -qty
            self.current_position_btc += signed
        self.position_evidence.append(
            {
                "timestamp_ms": timestamp,
                "position_btc": self.current_position_btc,
                "source": "manager_fill_update",
                "cloid": order.cloid,
            }
        )
        return order.to_dict()

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "hyperliquid_maker_order_manager_v1",
            "task_id": self.config.task_id,
            "run_id": self.config.run_id,
            "window_id": self.config.window_id,
            "ownership_prefix": self.config.ownership_prefix,
            "current_position_btc": self.current_position_btc,
            "multi_level_gate": self.multi_level_prerequisite_gate(requested_levels=1),
            "submissions_used": self.submissions_used,
            "orders": [order.to_dict() for order in self.orders_by_key.values()],
            "position_evidence": list(self.position_evidence),
            "last_reconciliation": dict(self.last_reconciliation),
        }
