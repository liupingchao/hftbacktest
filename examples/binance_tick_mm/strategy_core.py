#!/usr/bin/env python3
"""Core strategy logic shared between backtest and live trading."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from audit_schema import AUDIT_FIELDS

from hftbacktest import NEW, BUY, SELL


MARKET_VIEW_AUDIT_FIELDS = [
    "market_view_source",
    "top5_source",
    "market_overlay_source",
    "top5_overlay_source",
    "book_view_ts_local",
    "book_view_ts_exch",
    "book_view_feed_latency_ns",
    "book_view_stale_ms",
    "top5_depth_best_bid_tick",
    "top5_depth_best_ask_tick",
]

for _field in MARKET_VIEW_AUDIT_FIELDS:
    if _field not in AUDIT_FIELDS:
        AUDIT_FIELDS.insert(AUDIT_FIELDS.index("greek_delta"), _field)


@dataclass
class EwmaSigma:
    tau_ns: float = 30_000_000_000.0
    prev_ts: int | None = None
    prev_mid: float | None = None
    var: float = 0.0

    def update(self, ts: int, mid: float) -> float:
        if self.prev_mid is None or self.prev_ts is None:
            self.prev_mid = mid
            self.prev_ts = ts
            self.var = 0.0
            return 0.0

        if mid <= 0.0 or self.prev_mid <= 0.0:
            self.prev_mid = mid
            self.prev_ts = ts
            return math.sqrt(max(self.var, 0.0))

        ret = math.log(mid / self.prev_mid)
        dt = max(1, ts - self.prev_ts)
        alpha = math.exp(-float(dt) / self.tau_ns)
        self.var = alpha * self.var + (1.0 - alpha) * (ret * ret)
        self.prev_mid = mid
        self.prev_ts = ts
        return math.sqrt(max(self.var, 0.0))


@dataclass
class TokenBucket:
    capacity: float
    refill_per_sec: float
    tokens: float
    last_ts: int | None = None

    @classmethod
    def create(cls, capacity: float, refill_per_sec: float) -> "TokenBucket":
        return cls(capacity=capacity, refill_per_sec=refill_per_sec, tokens=capacity)

    def allow(self, ts: int, cost: float = 1.0) -> bool:
        if self.last_ts is None:
            self.last_ts = ts
        else:
            dt = max(0, ts - self.last_ts)
            self.tokens = min(
                self.capacity,
                self.tokens + (dt / 1_000_000_000.0) * self.refill_per_sec,
            )
            self.last_ts = ts

        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False


@dataclass
class GreekValues:
    delta: float
    gamma: float
    vega: float
    theta: float


@dataclass(frozen=True)
class MarketView:
    source: str
    top5_source: str
    best_bid: float
    best_ask: float
    best_bid_tick: int
    best_ask_tick: int
    mid: float
    spread: float
    bid_size: float
    ask_size: float
    bid_top5_ticks: str
    bid_top5_qtys: str
    ask_top5_ticks: str
    ask_top5_qtys: str
    ts_local: int = 0
    ts_exch: int = 0
    feed_latency_ns: int = 0
    stale_ms: float = 0.0
    market_overlay_source: str = ""
    top5_overlay_source: str = ""


@dataclass
class LiveSafetyConfig:
    enabled: bool = True
    rest_check_interval_sec: float = 5.0
    position_tolerance: float = 0.003
    open_order_check: bool = True
    fail_on_mismatch: bool = True
    connector_config: str = ""
    position_mismatch_confirmations: int = 2
    position_mismatch_pause_trading: bool = True
    use_rest_position_for_strategy: bool = True
    open_order_grace_ns: int = 1_000_000_000
    open_order_mismatch_confirmations: int = 2

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "LiveSafetyConfig":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", True)),
            rest_check_interval_sec=max(1.0, float(cfg.get("rest_check_interval_sec", 5.0))),
            position_tolerance=max(0.0, float(cfg.get("position_tolerance", 0.003))),
            open_order_check=bool(cfg.get("open_order_check", True)),
            fail_on_mismatch=bool(cfg.get("fail_on_mismatch", True)),
            connector_config=str(cfg.get("connector_config", "")),
            position_mismatch_confirmations=max(1, int(cfg.get("position_mismatch_confirmations", 2))),
            position_mismatch_pause_trading=bool(cfg.get("position_mismatch_pause_trading", True)),
            use_rest_position_for_strategy=bool(cfg.get("use_rest_position_for_strategy", True)),
            open_order_grace_ns=int(max(0.0, float(cfg.get("open_order_grace_ms", 1000.0))) * 1_000_000),
            open_order_mismatch_confirmations=max(1, int(cfg.get("open_order_mismatch_confirmations", 2))),
        )


@dataclass
class LiveSafetyState:
    rest_position: float = 0.0
    position_mismatch: float = 0.0
    rest_open_order_count: int = 0
    local_open_order_count: int = 0
    safety_status: str = "safety_disabled"
    rest_open_orders: str = ""
    local_open_orders: str = ""
    open_order_diff: str = ""
    safety_detail: str = ""


@dataclass(frozen=True)
class OrderSnapshot:
    order_id: int
    side: str
    price: float
    price_tick: int
    qty: float
    leaves_qty: float
    exec_qty: float
    exec_price_tick: int
    status: str
    req: str
    time_in_force: str
    exch_timestamp: int
    local_timestamp: int
    cancellable: bool


@dataclass
class OrderLifecycleTracker:
    last_by_order_id: dict[int, OrderSnapshot]
    cancel_request_ts_by_order_id: dict[int, int]

    @classmethod
    def create(cls) -> "OrderLifecycleTracker":
        return cls(last_by_order_id={}, cancel_request_ts_by_order_id={})

    def mark_cancel_requested(self, order_id: int, ts_local: int) -> None:
        self.cancel_request_ts_by_order_id[int(order_id)] = int(ts_local)

    def cancel_request_ts(self, order_id: int) -> int:
        return int(self.cancel_request_ts_by_order_id.get(int(order_id), 0))

    def observe(self, order_dict: Any) -> list[tuple[str, OrderSnapshot, OrderSnapshot | None]]:
        current: dict[int, OrderSnapshot] = {}
        values = order_dict.values()
        while values.has_next():
            order = values.get()
            snap = snapshot_order(order)
            current[snap.order_id] = snap

        events: list[tuple[str, OrderSnapshot, OrderSnapshot | None]] = []
        for order_id, snap in sorted(current.items()):
            prev = self.last_by_order_id.get(order_id)
            if prev is None:
                events.append(("order_new", snap, None))
            elif snap.status != prev.status or snap.req != prev.req or snap.leaves_qty != prev.leaves_qty or snap.exec_qty != prev.exec_qty:
                events.append((_lifecycle_event_type(prev, snap), snap, prev))

        self.last_by_order_id = current
        return events


def evaluate_live_safety(
    *,
    cfg: LiveSafetyConfig,
    rest_position: float,
    local_position: float,
    rest_open_order_count: int,
    local_open_order_count: int,
    rest_error: str,
    rest_open_orders: str = "",
    local_open_orders: str = "",
    open_order_diff: str = "",
    ts_local: int | None = None,
    last_api_ts: int | None = None,
    open_order_mismatch_count: int = 0,
    position_mismatch_count: int = 0,
) -> LiveSafetyState:
    mismatch = round(abs(rest_position - local_position), 12)
    safety_detail = ""
    if not cfg.enabled:
        status = "safety_disabled"
    elif rest_error:
        status = "rest_error"
        safety_detail = rest_error
    elif mismatch > cfg.position_tolerance:
        if position_mismatch_count + 1 >= cfg.position_mismatch_confirmations:
            status = "position_mismatch"
        else:
            status = "position_mismatch_pending"
        safety_detail = f"position_mismatch={mismatch}"
    elif cfg.open_order_check and rest_open_order_count != local_open_order_count:
        in_grace = (
            ts_local is not None
            and last_api_ts is not None
            and ts_local >= last_api_ts
            and (ts_local - last_api_ts) < cfg.open_order_grace_ns
        )
        if in_grace:
            status = "open_order_grace"
        elif open_order_mismatch_count + 1 >= cfg.open_order_mismatch_confirmations:
            status = "open_order_mismatch"
        else:
            status = "open_order_mismatch_pending"
        safety_detail = open_order_diff
    else:
        status = "ok"
    return LiveSafetyState(
        rest_position=rest_position,
        position_mismatch=mismatch,
        rest_open_order_count=rest_open_order_count,
        local_open_order_count=local_open_order_count,
        safety_status=status,
        rest_open_orders=rest_open_orders,
        local_open_orders=local_open_orders,
        open_order_diff=open_order_diff,
        safety_detail=safety_detail,
    )


def is_position_limit_reached(*, position: float, position_notional: float, risk: dict[str, Any]) -> bool:
    max_position_qty = float(risk.get("max_position_qty", 0.0))
    if max_position_qty > 0.0:
        return abs(position) >= max_position_qty
    return abs(position_notional) > float(risk["max_notional_pos"])


def inventory_score_from_risk(*, position: float, position_notional: float, risk: dict[str, Any]) -> float:
    max_position_qty = float(risk.get("max_position_qty", 0.0))
    if max_position_qty > 0.0:
        return max(0.0, 1.0 - abs(position) / max_position_qty)
    return max(0.0, 1.0 - abs(position_notional) / float(risk["max_notional_pos"]))


def add_side_soft_limit_qty_from_risk(risk: dict[str, Any]) -> float:
    explicit_qty = float(risk.get("inventory_add_side_soft_limit_qty", 0.0))
    if explicit_qty > 0.0:
        return explicit_qty
    ratio = float(risk.get("inventory_add_side_soft_limit_ratio", 0.0))
    max_position_qty = float(risk.get("max_position_qty", 0.0))
    if ratio > 0.0 and max_position_qty > 0.0:
        return max_position_qty * ratio
    return 0.0


class GreekOracle:
    def __init__(
        self,
        ts_local: np.ndarray | None,
        delta: np.ndarray | None,
        gamma: np.ndarray | None,
        vega: np.ndarray | None,
        theta: np.ndarray | None,
        enabled: bool,
        use_position_as_delta: bool,
        scale_delta: float,
        scale_gamma: float,
        scale_vega: float,
        scale_theta: float,
    ):
        self.ts_local = ts_local if ts_local is not None else np.empty(0, dtype=np.int64)
        self.delta = delta if delta is not None else np.empty(0, dtype=np.float64)
        self.gamma = gamma if gamma is not None else np.empty(0, dtype=np.float64)
        self.vega = vega if vega is not None else np.empty(0, dtype=np.float64)
        self.theta = theta if theta is not None else np.empty(0, dtype=np.float64)
        self.enabled = enabled
        self.use_position_as_delta = use_position_as_delta
        self.scale_delta = scale_delta
        self.scale_gamma = scale_gamma
        self.scale_vega = scale_vega
        self.scale_theta = scale_theta
        self.i = 0

    @staticmethod
    def _row_float(row: dict[str, str], keys: list[str]) -> float:
        for k in keys:
            raw = row.get(k)
            if raw is None:
                continue
            s = str(raw).strip()
            if not s:
                continue
            try:
                return float(s)
            except ValueError:
                continue
        return 0.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], expand_path: Callable[[str], Path] | None = None) -> "GreekOracle":
        enabled = bool(cfg.get("enabled", False))
        use_position_as_delta = bool(cfg.get("use_position_as_delta", True))
        scale_delta = float(cfg.get("scale_delta", 1.0))
        scale_gamma = float(cfg.get("scale_gamma", 1.0))
        scale_vega = float(cfg.get("scale_vega", 1.0))
        scale_theta = float(cfg.get("scale_theta", 1.0))

        signal_csv = str(cfg.get("signal_csv", "")).strip()
        if not enabled or not signal_csv:
            return cls(
                ts_local=None,
                delta=None,
                gamma=None,
                vega=None,
                theta=None,
                enabled=enabled,
                use_position_as_delta=use_position_as_delta,
                scale_delta=scale_delta,
                scale_gamma=scale_gamma,
                scale_vega=scale_vega,
                scale_theta=scale_theta,
            )

        if expand_path is not None:
            path = expand_path(signal_csv)
        else:
            path = Path(signal_csv).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Greeks signal_csv not found: {path}")

        rows: list[tuple[int, float, float, float, float]] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts_raw = row.get("ts_local") or row.get("ts") or row.get("timestamp")
                if ts_raw is None:
                    continue
                try:
                    ts_local = int(float(str(ts_raw).strip()))
                except ValueError:
                    continue

                rows.append(
                    (
                        ts_local,
                        cls._row_float(row, ["delta", "net_delta", "greek_delta"]),
                        cls._row_float(row, ["gamma", "net_gamma", "greek_gamma"]),
                        cls._row_float(row, ["vega", "net_vega", "greek_vega"]),
                        cls._row_float(row, ["theta", "net_theta", "greek_theta"]),
                    )
                )

        if not rows:
            return cls(
                ts_local=None,
                delta=None,
                gamma=None,
                vega=None,
                theta=None,
                enabled=enabled,
                use_position_as_delta=use_position_as_delta,
                scale_delta=scale_delta,
                scale_gamma=scale_gamma,
                scale_vega=scale_vega,
                scale_theta=scale_theta,
            )

        rows.sort(key=lambda x: x[0])
        ts = np.asarray([r[0] for r in rows], dtype=np.int64)
        d = np.asarray([r[1] for r in rows], dtype=np.float64)
        g = np.asarray([r[2] for r in rows], dtype=np.float64)
        v = np.asarray([r[3] for r in rows], dtype=np.float64)
        t = np.asarray([r[4] for r in rows], dtype=np.float64)

        return cls(
            ts_local=ts,
            delta=d,
            gamma=g,
            vega=v,
            theta=t,
            enabled=enabled,
            use_position_as_delta=use_position_as_delta,
            scale_delta=scale_delta,
            scale_gamma=scale_gamma,
            scale_vega=scale_vega,
            scale_theta=scale_theta,
        )

    def values(self, ts_local: int, position: float) -> GreekValues:
        if not self.enabled:
            return GreekValues(0.0, 0.0, 0.0, 0.0)

        n = len(self.ts_local)
        if n > 0:
            while self.i + 1 < n and int(self.ts_local[self.i + 1]) <= ts_local:
                self.i += 1

            delta = float(self.delta[self.i])
            gamma = float(self.gamma[self.i])
            vega = float(self.vega[self.i])
            theta = float(self.theta[self.i])
        else:
            delta = position if self.use_position_as_delta else 0.0
            gamma = 0.0
            vega = 0.0
            theta = 0.0

        return GreekValues(
            delta=delta * self.scale_delta,
            gamma=gamma * self.scale_gamma,
            vega=vega * self.scale_vega,
            theta=theta * self.scale_theta,
        )


@dataclass
class ExtraOrder:
    order_id: int
    side: str
    price_tick: int
    req: str = "none"
    cancellable: bool = True
    source_order: Any | None = None


@dataclass
class PendingLocalOrder:
    order_id: int
    side: int
    price: float
    price_tick: int
    qty: float
    local_timestamp: int
    status: int = NEW
    req: int = 1
    leaves_qty: float = 0.0
    exec_qty: float = 0.0
    exec_price_tick: int = 0
    time_in_force: int = 1
    exch_timestamp: int = 0
    cancellable: bool = False
    visible_ts: int = 0
    release_ts: int = 0


@dataclass
class WorkingOrders:
    buy: Any | None
    sell: Any | None
    extras: list[ExtraOrder]

    @property
    def extra_ids(self) -> list[int]:
        return [extra.order_id for extra in self.extras]

    @property
    def extra_sides(self) -> list[str]:
        return [extra.side for extra in self.extras]

    @property
    def extra_price_ticks(self) -> list[int]:
        return [extra.price_tick for extra in self.extras]


def working_side_leaves_qty(working: WorkingOrders, side: str) -> float:
    total = 0.0
    if side == "buy" and working.buy is not None:
        total += max(0.0, float(getattr(working.buy, "leaves_qty", 0.0) or 0.0))
    if side == "sell" and working.sell is not None:
        total += max(0.0, float(getattr(working.sell, "leaves_qty", 0.0) or 0.0))
    total += sum(
        max(0.0, float(getattr(extra.raw, "leaves_qty", 0.0) or 0.0))
        for extra in working.extras
        if extra.side == side and extra.raw is not None
    )
    return total


def merge_pending_orders(
    working: WorkingOrders,
    pending_orders: list[PendingLocalOrder],
    *,
    replace_existing: bool = False,
) -> WorkingOrders:
    buy = working.buy
    sell = working.sell
    extras = list(working.extras)

    for order in sorted(pending_orders, key=lambda o: o.order_id):
        order_id = int(order.order_id)
        if any(int(extra.order_id) == order_id for extra in extras):
            continue
        if buy is not None and int(getattr(buy, "order_id", -1)) == order_id and order.side != BUY:
            continue
        if sell is not None and int(getattr(sell, "order_id", -1)) == order_id and order.side != SELL:
            continue
        if order.side == BUY:
            if buy is None:
                buy = order
            elif int(getattr(buy, "order_id", -1)) == order_id:
                if replace_existing:
                    buy = order
            else:
                extras.append(
                    ExtraOrder(
                        order.order_id,
                        "buy",
                        order.price_tick,
                        _order_req_name(int(order.req)),
                        bool(order.cancellable),
                    )
                )
        elif order.side == SELL:
            if sell is None:
                sell = order
            elif int(getattr(sell, "order_id", -1)) == order_id:
                if replace_existing:
                    sell = order
            else:
                extras.append(
                    ExtraOrder(
                        order.order_id,
                        "sell",
                        order.price_tick,
                        _order_req_name(int(order.req)),
                        bool(order.cancellable),
                    )
                )

    return WorkingOrders(buy=buy, sell=sell, extras=extras)


def _order_side_name(side: int) -> str:
    if side == BUY:
        return "buy"
    if side == SELL:
        return "sell"
    return str(side)


def order_side_name(side: Any) -> str:
    if isinstance(side, str):
        side_lower = side.strip().lower()
        if side_lower in {"buy", "sell"}:
            return side_lower
    try:
        return _order_side_name(int(side))
    except (TypeError, ValueError):
        return ""


def _order_status_name(status: int) -> str:
    return {
        0: "none",
        1: "new",
        2: "expired",
        3: "filled",
        4: "canceled",
        5: "partially_filled",
        6: "rejected",
        255: "unsupported",
    }.get(int(status), str(status))


def _order_req_name(req: int) -> str:
    return {
        0: "none",
        1: "new",
        4: "cancel",
    }.get(int(req), str(req))


def _order_time_in_force_name(tif: int) -> str:
    return {
        0: "gtc",
        1: "gtx",
        2: "fok",
        3: "ioc",
    }.get(int(tif), str(tif))


def snapshot_order(order: Any) -> OrderSnapshot:
    price_tick = int(order.price_tick)
    try:
        price = float(order.price)
    except (AttributeError, TypeError, ValueError):
        price = 0.0

    return OrderSnapshot(
        order_id=int(order.order_id),
        side=_order_side_name(int(order.side)),
        price=price,
        price_tick=price_tick,
        qty=float(order.qty),
        leaves_qty=float(getattr(order, "leaves_qty", 0.0) or 0.0),
        exec_qty=float(getattr(order, "exec_qty", 0.0) or 0.0),
        exec_price_tick=int(getattr(order, "exec_price_tick", 0) or 0),
        status=_order_status_name(int(order.status)),
        req=_order_req_name(int(order.req)),
        time_in_force=_order_time_in_force_name(int(getattr(order, "time_in_force", 0) or 0)),
        exch_timestamp=int(order.exch_timestamp),
        local_timestamp=int(order.local_timestamp),
        cancellable=bool(order.cancellable),
    )


def _lifecycle_event_type(prev: OrderSnapshot, cur: OrderSnapshot) -> str:
    if cur.status == "filled":
        return "fill"
    if cur.status == "partially_filled":
        return "partial_fill"
    if cur.status == "canceled":
        return "cancel_ack"
    if cur.status == "expired":
        return "expired"
    if cur.status == "rejected":
        return "rejected"
    if cur.req == "cancel" and prev.req != "cancel":
        return "cancel_sent"
    if cur.status == "new" and prev.status != "new":
        return "order_new"
    return "order_update"


def _format_local_order(order: Any) -> str:
    return (
        f"{int(order.order_id)}:"
        f"{_order_side_name(int(order.side))}:"
        f"{int(order.price_tick)}:"
        f"{float(order.qty):.8g}:"
        f"{_order_status_name(int(order.status))}:"
        f"req={_order_req_name(int(order.req))}:"
        f"cxl={int(bool(order.cancellable))}:"
        f"exch={int(order.exch_timestamp)}:"
        f"local={int(order.local_timestamp)}"
    )


def format_local_open_orders(order_dict: Any) -> str:
    out: list[str] = []
    values = order_dict.values()
    while values.has_next():
        order = values.get()
        if order.status == NEW:
            out.append(_format_local_order(order))
    return ";".join(sorted(out))


def format_rest_open_orders(rows: list[dict[str, Any]], tick_size: float | None = None) -> str:
    out: list[str] = []
    for row in rows:
        client_id = str(row.get("clientOrderId", "")).strip()
        local_id = ""
        if client_id:
            suffix_digits = []
            for ch in reversed(client_id):
                if not ch.isdigit():
                    break
                suffix_digits.append(ch)
            if suffix_digits:
                local_id = "".join(reversed(suffix_digits))
        order_id = str(row.get("orderId", "")).strip()
        side = str(row.get("side", "")).lower()
        price = str(row.get("price", "")).strip()
        price_key = price
        if tick_size is not None and tick_size > 0.0 and price:
            try:
                price_key = str(int(round(float(price) / float(tick_size))))
            except ValueError:
                price_key = price
        qty = str(row.get("origQty", row.get("quantity", ""))).strip()
        executed = str(row.get("executedQty", "")).strip()
        status = str(row.get("status", "")).lower()
        tif = str(row.get("timeInForce", "")).lower()
        update_time = str(row.get("updateTime", "")).strip()
        out.append(
            f"{local_id or '?'}:{side}:{price_key}:{qty}:price={price}:exec={executed}:"
            f"status={status}:tif={tif}:exchange_id={order_id}:client={client_id}:"
            f"update_ms={update_time}"
        )
    return ";".join(sorted(out))


def _normalize_qty(raw: str) -> str:
    try:
        return f"{float(raw):.8g}"
    except ValueError:
        return raw


def _open_order_key_sets(serialized: str) -> tuple[set[str], set[str]]:
    id_keys: set[str] = set()
    quote_keys: set[str] = set()
    for item in serialized.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if parts and parts[0] and parts[0] != "?":
            id_keys.add(parts[0])
        if len(parts) >= 4:
            quote_keys.add(f"{parts[1]}:{parts[2]}:{_normalize_qty(parts[3])}")
    return id_keys, quote_keys


def open_order_diff(local_open_orders: str, rest_open_orders: str) -> str:
    _, local_keys = _open_order_key_sets(local_open_orders)
    _, rest_keys = _open_order_key_sets(rest_open_orders)
    local_only = sorted(local_keys - rest_keys)
    rest_only = sorted(rest_keys - local_keys)
    out: list[str] = []
    if local_only:
        out.append("local_only=" + "|".join(local_only))
    if rest_only:
        out.append("rest_only=" + "|".join(rest_only))
    return ";".join(out)


@dataclass
class Action:
    kind: str
    side: str
    order_id: int
    price: float
    qty: float


@dataclass
class InFlightExposureOrder:
    side: str
    leaves_qty: float
    cancel_requested: bool = False


@dataclass
class InFlightExposureTracker:
    orders: dict[int, InFlightExposureOrder] = field(default_factory=dict)

    @classmethod
    def create(cls) -> "InFlightExposureTracker":
        return cls()

    def mark_submitted(self, action: Action) -> None:
        if action.kind != "submit" or action.side not in {"buy", "sell"}:
            return
        self.orders[int(action.order_id)] = InFlightExposureOrder(
            side=action.side,
            leaves_qty=max(0.0, float(action.qty)),
            cancel_requested=False,
        )

    def mark_cancel_requested(self, order_id: int) -> None:
        order = self.orders.get(int(order_id))
        if order is not None:
            order.cancel_requested = True

    def observe_lifecycle(self, event_type: str, order: OrderSnapshot) -> None:
        order_id = int(order.order_id)
        if order.side not in {"buy", "sell"}:
            self.orders.pop(order_id, None)
            return
        if event_type in {"fill", "cancel_ack", "expired", "rejected"} or order.status in {
            "filled",
            "canceled",
            "expired",
            "rejected",
        }:
            self.orders.pop(order_id, None)
            return
        if event_type == "partial_fill" or order.status == "partially_filled":
            leaves_qty = max(0.0, float(order.leaves_qty))
        else:
            leaves_qty = max(0.0, float(order.leaves_qty or order.qty))
        cancel_requested = self.orders.get(order_id, InFlightExposureOrder(order.side, leaves_qty)).cancel_requested
        self.orders[order_id] = InFlightExposureOrder(
            side=order.side,
            leaves_qty=leaves_qty,
            cancel_requested=cancel_requested or order.req == "cancel",
        )

    def side_qty(self, side: str) -> float:
        return sum(
            max(0.0, float(order.leaves_qty))
            for order in self.orders.values()
            if order.side == side
        )

    def cancel_requested_side_qty(self, side: str) -> float:
        return sum(
            max(0.0, float(order.leaves_qty))
            for order in self.orders.values()
            if order.side == side and order.cancel_requested
        )


def cancel_race_guard_side_blocks(
    *,
    enabled: bool,
    pending_cancel_block: bool,
    post_fill_cooldown_ns: int,
    ts_local: int,
    inflight_exposure: InFlightExposureTracker,
    last_buy_cancel_fill_ts: int | None = None,
    last_sell_cancel_fill_ts: int | None = None,
) -> tuple[bool, bool]:
    if not enabled:
        return False, False

    buy_block = bool(pending_cancel_block and inflight_exposure.cancel_requested_side_qty("buy") > 0.0)
    sell_block = bool(pending_cancel_block and inflight_exposure.cancel_requested_side_qty("sell") > 0.0)
    if post_fill_cooldown_ns > 0:
        if last_buy_cancel_fill_ts is not None and ts_local - last_buy_cancel_fill_ts < post_fill_cooldown_ns:
            buy_block = True
        if last_sell_cancel_fill_ts is not None and ts_local - last_sell_cancel_fill_ts < post_fill_cooldown_ns:
            sell_block = True
    return buy_block, sell_block


@dataclass(frozen=True)
class GuardBlockResult:
    buy_block: bool = False
    sell_block: bool = False
    buy_reason: str = ""
    sell_reason: str = ""
    buy_until_ts: int = 0
    sell_until_ts: int = 0
    buy_target_move_ticks: int = 0
    sell_target_move_ticks: int = 0


@dataclass(frozen=True)
class AddSideToxicTimingResult:
    buy_eligible: bool = False
    sell_eligible: bool = False
    buy_block: bool = False
    sell_block: bool = False
    buy_reason: str = ""
    sell_reason: str = ""
    buy_reduce_side_allowed: bool = False
    sell_reduce_side_allowed: bool = False
    buy_target_move_ticks: int = 0
    sell_target_move_ticks: int = 0
    buy_last_cancel_request_age_ms: float = -1.0
    sell_last_cancel_request_age_ms: float = -1.0
    buy_last_cancel_fill_age_ms: float = -1.0
    sell_last_cancel_fill_age_ms: float = -1.0
    buy_until_ts: int = 0
    sell_until_ts: int = 0


def adverse_timing_guard_side_blocks(
    *,
    enabled: bool,
    target_deterioration_enabled: bool,
    pending_cancel_enabled: bool,
    post_cancel_fill_enabled: bool,
    cooldown_ns: int,
    min_target_move_ticks: int,
    ts_local: int,
    working: WorkingOrders,
    target_bid_tick: int,
    target_ask_tick: int,
    inflight_exposure: InFlightExposureTracker,
    last_buy_cancel_fill_ts: int | None = None,
    last_sell_cancel_fill_ts: int | None = None,
) -> GuardBlockResult:
    if not enabled:
        return GuardBlockResult()

    buy_block = False
    sell_block = False
    buy_reason = ""
    sell_reason = ""
    buy_until_ts = 0
    sell_until_ts = 0
    buy_target_move_ticks = 0
    sell_target_move_ticks = 0
    min_move = max(0, int(min_target_move_ticks))

    if target_deterioration_enabled and min_move > 0:
        if working.buy is not None:
            buy_target_move_ticks = int(working.buy.price_tick) - int(target_bid_tick)
            if buy_target_move_ticks >= min_move:
                buy_block = True
                buy_reason = "target_deterioration"
                buy_until_ts = ts_local + max(0, int(cooldown_ns))
        if working.sell is not None:
            sell_target_move_ticks = int(target_ask_tick) - int(working.sell.price_tick)
            if sell_target_move_ticks >= min_move:
                sell_block = True
                sell_reason = "target_deterioration"
                sell_until_ts = ts_local + max(0, int(cooldown_ns))

    if pending_cancel_enabled:
        buy_pending = inflight_exposure.cancel_requested_side_qty("buy") > 0.0
        sell_pending = inflight_exposure.cancel_requested_side_qty("sell") > 0.0
        if buy_pending and (not target_deterioration_enabled or buy_target_move_ticks >= min_move):
            buy_block = True
            buy_reason = buy_reason or "pending_cancel_toxic_window"
            buy_until_ts = max(buy_until_ts, ts_local + max(0, int(cooldown_ns)))
        if sell_pending and (not target_deterioration_enabled or sell_target_move_ticks >= min_move):
            sell_block = True
            sell_reason = sell_reason or "pending_cancel_toxic_window"
            sell_until_ts = max(sell_until_ts, ts_local + max(0, int(cooldown_ns)))

    if post_cancel_fill_enabled and cooldown_ns > 0:
        if last_buy_cancel_fill_ts is not None and ts_local - last_buy_cancel_fill_ts < cooldown_ns:
            buy_block = True
            buy_reason = buy_reason or "cancel_requested_fill_timing"
            buy_until_ts = max(buy_until_ts, int(last_buy_cancel_fill_ts) + int(cooldown_ns))
        if last_sell_cancel_fill_ts is not None and ts_local - last_sell_cancel_fill_ts < cooldown_ns:
            sell_block = True
            sell_reason = sell_reason or "cancel_requested_fill_timing"
            sell_until_ts = max(sell_until_ts, int(last_sell_cancel_fill_ts) + int(cooldown_ns))

    return GuardBlockResult(
        buy_block=buy_block,
        sell_block=sell_block,
        buy_reason=buy_reason,
        sell_reason=sell_reason,
        buy_until_ts=buy_until_ts,
        sell_until_ts=sell_until_ts,
        buy_target_move_ticks=buy_target_move_ticks,
        sell_target_move_ticks=sell_target_move_ticks,
    )


def _age_ms(ts_local: int, event_ts: int | None) -> float:
    if event_ts is None or event_ts <= 0:
        return -1.0
    return max(0.0, (int(ts_local) - int(event_ts)) / 1_000_000.0)


def add_side_toxic_timing_guard_side_blocks(
    *,
    enabled: bool,
    pending_cancel_enabled: bool,
    post_cancel_fill_enabled: bool,
    target_move_enabled: bool,
    cooldown_ns: int,
    min_target_move_ticks: int,
    latency_threshold_ns: int,
    ts_local: int,
    position: float,
    target_bid_tick: int,
    target_ask_tick: int,
    buy_submit_eligible: bool,
    sell_submit_eligible: bool,
    buy_reduce_side_allowed: bool,
    sell_reduce_side_allowed: bool,
    inflight_exposure: InFlightExposureTracker,
    last_buy_cancel_ts: int | None = None,
    last_sell_cancel_ts: int | None = None,
    last_buy_cancel_fill_ts: int | None = None,
    last_sell_cancel_fill_ts: int | None = None,
    last_quote_or_cancel_target_bid_tick: int | None = None,
    last_quote_or_cancel_target_ask_tick: int | None = None,
    latency_signal_ns: int = 0,
) -> AddSideToxicTimingResult:
    buy_request_age = _age_ms(ts_local, last_buy_cancel_ts)
    sell_request_age = _age_ms(ts_local, last_sell_cancel_ts)
    buy_fill_age = _age_ms(ts_local, last_buy_cancel_fill_ts)
    sell_fill_age = _age_ms(ts_local, last_sell_cancel_fill_ts)

    buy_target_move_ticks = (
        int(last_quote_or_cancel_target_bid_tick) - int(target_bid_tick)
        if last_quote_or_cancel_target_bid_tick is not None
        else 0
    )
    sell_target_move_ticks = (
        int(target_ask_tick) - int(last_quote_or_cancel_target_ask_tick)
        if last_quote_or_cancel_target_ask_tick is not None
        else 0
    )

    if not enabled:
        return AddSideToxicTimingResult(
            buy_eligible=bool(buy_submit_eligible),
            sell_eligible=bool(sell_submit_eligible),
            buy_reduce_side_allowed=bool(buy_reduce_side_allowed),
            sell_reduce_side_allowed=bool(sell_reduce_side_allowed),
            buy_target_move_ticks=buy_target_move_ticks,
            sell_target_move_ticks=sell_target_move_ticks,
            buy_last_cancel_request_age_ms=buy_request_age,
            sell_last_cancel_request_age_ms=sell_request_age,
            buy_last_cancel_fill_age_ms=buy_fill_age,
            sell_last_cancel_fill_age_ms=sell_fill_age,
        )

    cooldown_ns = max(0, int(cooldown_ns))
    min_move = max(0, int(min_target_move_ticks))
    latency_threshold_ns = max(0, int(latency_threshold_ns))
    window_ms = cooldown_ns / 1_000_000.0 if cooldown_ns > 0 else 0.0

    def recent(age: float) -> bool:
        return window_ms > 0.0 and 0.0 <= age <= window_ms

    buy_pending = bool(pending_cancel_enabled and inflight_exposure.cancel_requested_side_qty("buy") > 0.0)
    sell_pending = bool(pending_cancel_enabled and inflight_exposure.cancel_requested_side_qty("sell") > 0.0)
    buy_recent_request = bool(pending_cancel_enabled and recent(buy_request_age))
    sell_recent_request = bool(pending_cancel_enabled and recent(sell_request_age))
    buy_recent_fill = bool(post_cancel_fill_enabled and recent(buy_fill_age))
    sell_recent_fill = bool(post_cancel_fill_enabled and recent(sell_fill_age))
    latency_toxic = bool(latency_threshold_ns > 0 and int(latency_signal_ns) >= latency_threshold_ns)
    buy_target_toxic = bool(target_move_enabled and min_move > 0 and buy_target_move_ticks >= min_move)
    sell_target_toxic = bool(target_move_enabled and min_move > 0 and sell_target_move_ticks >= min_move)

    buy_domain = buy_pending or buy_recent_request or buy_recent_fill
    sell_domain = sell_pending or sell_recent_request or sell_recent_fill
    buy_signal = buy_target_toxic or latency_toxic or buy_recent_fill
    sell_signal = sell_target_toxic or latency_toxic or sell_recent_fill
    buy_block = bool(buy_submit_eligible and position >= 0.0 and buy_domain and buy_signal)
    sell_block = bool(sell_submit_eligible and position <= 0.0 and sell_domain and sell_signal)

    def reason(
        pending: bool,
        recent_request: bool,
        recent_fill: bool,
        target_toxic: bool,
        latency_toxic_: bool,
    ) -> str:
        parts: list[str] = []
        if recent_fill:
            parts.append("recent_cancel_fill")
        if pending:
            parts.append("pending_cancel")
        if recent_request:
            parts.append("recent_cancel_request")
        if target_toxic:
            parts.append("target_move")
        if latency_toxic_:
            parts.append("latency")
        return "+".join(parts) or "toxic_timing"

    return AddSideToxicTimingResult(
        buy_eligible=bool(buy_submit_eligible),
        sell_eligible=bool(sell_submit_eligible),
        buy_block=buy_block,
        sell_block=sell_block,
        buy_reason=reason(
            buy_pending,
            buy_recent_request,
            buy_recent_fill,
            buy_target_toxic,
            latency_toxic,
        ) if buy_block else "",
        sell_reason=reason(
            sell_pending,
            sell_recent_request,
            sell_recent_fill,
            sell_target_toxic,
            latency_toxic,
        ) if sell_block else "",
        buy_reduce_side_allowed=bool(buy_reduce_side_allowed),
        sell_reduce_side_allowed=bool(sell_reduce_side_allowed),
        buy_target_move_ticks=buy_target_move_ticks,
        sell_target_move_ticks=sell_target_move_ticks,
        buy_last_cancel_request_age_ms=buy_request_age,
        sell_last_cancel_request_age_ms=sell_request_age,
        buy_last_cancel_fill_age_ms=buy_fill_age,
        sell_last_cancel_fill_age_ms=sell_fill_age,
        buy_until_ts=ts_local + cooldown_ns if buy_block else 0,
        sell_until_ts=ts_local + cooldown_ns if sell_block else 0,
    )


def format_actions(actions: list[Action]) -> tuple[str, str]:
    if not actions:
        return "", "keep"
    order_id = "|".join(str(action.order_id) for action in actions)
    action_name = "|".join(f"{action.kind}_{action.side}" for action in actions)
    return order_id, action_name


def is_pure_cancel_extra(actions: list[Action]) -> bool:
    return bool(actions) and all(
        action.kind == "cancel" and action.side == "extra"
        for action in actions
    )


def format_working_order_diagnostics(working: WorkingOrders) -> dict[str, str]:
    local_open_orders = []
    if working.buy is not None:
        local_open_orders.append(_format_local_order(working.buy))
    if working.sell is not None:
        local_open_orders.append(_format_local_order(working.sell))
    local_open_orders.extend(
        f"{extra.order_id}:{extra.side}:{extra.price_tick}"
        for extra in working.extras
    )
    buy_req = _order_req_name(int(working.buy.req)) if working.buy is not None else ""
    sell_req = _order_req_name(int(working.sell.req)) if working.sell is not None else ""
    return {
        "working_buy_order_id": str(int(working.buy.order_id)) if working.buy is not None else "",
        "working_sell_order_id": str(int(working.sell.order_id)) if working.sell is not None else "",
        "working_bid_qty": f"{float(working.buy.qty):.8g}" if working.buy is not None else "",
        "working_ask_qty": f"{float(working.sell.qty):.8g}" if working.sell is not None else "",
        "working_bid_status": _order_status_name(int(working.buy.status)) if working.buy is not None else "",
        "working_ask_status": _order_status_name(int(working.sell.status)) if working.sell is not None else "",
        "working_bid_req": buy_req,
        "working_ask_req": sell_req,
        "working_bid_pending_cancel": str(int(buy_req == "cancel")) if working.buy is not None else "",
        "working_ask_pending_cancel": str(int(sell_req == "cancel")) if working.sell is not None else "",
        "extra_order_ids": "|".join(str(extra.order_id) for extra in working.extras),
        "extra_order_sides": "|".join(extra.side for extra in working.extras),
        "extra_order_price_ticks": "|".join(str(extra.price_tick) for extra in working.extras),
        "local_open_orders": ";".join(local_open_orders),
    }


def _working_semantic_defaults_from_local_open_orders(local_open_orders: str) -> dict[str, str]:
    defaults = {
        "working_bid_qty": "",
        "working_ask_qty": "",
        "working_bid_status": "",
        "working_ask_status": "",
        "working_bid_req": "",
        "working_ask_req": "",
        "working_bid_pending_cancel": "",
        "working_ask_pending_cancel": "",
    }
    for raw_item in str(local_open_orders or "").split(";"):
        item = raw_item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 4:
            continue
        side = parts[1].lower()
        if side not in {"buy", "sell"}:
            continue
        status = ""
        req = ""
        for token in parts[4:]:
            if "=" in token:
                key, value = token.split("=", 1)
                key = key.strip().lower()
                value = value.strip().lower()
                if key == "req":
                    req = value
                elif key == "status" and not status:
                    status = value
            elif not status:
                status = token.strip().lower()
        if not status:
            status = "new"
        prefix = "bid" if side == "buy" else "ask"
        qty_key = f"working_{prefix}_qty"
        status_key = f"working_{prefix}_status"
        req_key = f"working_{prefix}_req"
        pending_key = f"working_{prefix}_pending_cancel"
        if not defaults[qty_key]:
            defaults[qty_key] = parts[3]
        if not defaults[status_key]:
            defaults[status_key] = status
        if not defaults[req_key]:
            defaults[req_key] = req
        if not defaults[pending_key]:
            defaults[pending_key] = "1" if req == "cancel" else "0"
    return defaults


@dataclass
class QuoteThrottleConfig:
    enabled: bool = False
    min_interval_ns: int = 100_000_000
    min_move_ticks: int = 2

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None) -> "QuoteThrottleConfig":
        cfg = cfg or {}
        enabled = bool(cfg.get("quote_throttle_enabled", False))
        min_interval_ms = max(0.0, float(cfg.get("min_quote_update_interval_ms", 100.0)))
        min_move_ticks = max(0, int(cfg.get("min_quote_move_ticks", 2)))
        return cls(
            enabled=enabled,
            min_interval_ns=int(min_interval_ms * 1_000_000),
            min_move_ticks=min_move_ticks,
        )


@dataclass
class QuoteThrottleState:
    last_sent_api_ts: int | None = None
    last_sent_target_bid_tick: int | None = None
    last_sent_target_ask_tick: int | None = None

    def mark_sent(self, ts_local: int, target_bid_tick: int, target_ask_tick: int) -> None:
        self.last_sent_api_ts = ts_local
        self.last_sent_target_bid_tick = target_bid_tick
        self.last_sent_target_ask_tick = target_ask_tick


def should_throttle_quote_update(
    *,
    cfg: QuoteThrottleConfig,
    state: QuoteThrottleState,
    ts_local: int,
    target_bid_tick: int,
    target_ask_tick: int,
    planned_actions: list[Action],
    pos_limit: bool,
) -> str:
    if not cfg.enabled or not planned_actions:
        return ""
    if pos_limit:
        return ""
    if any(action.kind == "cancel" and action.side == "extra" for action in planned_actions):
        return ""
    if state.last_sent_api_ts is None:
        return ""
    if state.last_sent_target_bid_tick is None or state.last_sent_target_ask_tick is None:
        return ""

    elapsed_ns = ts_local - state.last_sent_api_ts
    if elapsed_ns < 0 or elapsed_ns >= cfg.min_interval_ns:
        return ""

    bid_move = abs(target_bid_tick - state.last_sent_target_bid_tick)
    ask_move = abs(target_ask_tick - state.last_sent_target_ask_tick)
    if max(bid_move, ask_move) >= cfg.min_move_ticks:
        return ""

    return "min_quote_update_interval"


def quote_throttle_reason(
    *,
    actions: list[Action],
    state: QuoteThrottleState,
    ts_local: int,
    target_bid_tick: int,
    target_ask_tick: int,
    min_interval_ns: int,
    min_move_ticks: int,
    pos_limit: bool = False,
) -> str:
    cfg = QuoteThrottleConfig(
        enabled=True,
        min_interval_ns=min_interval_ns,
        min_move_ticks=min_move_ticks,
    )
    return should_throttle_quote_update(
        cfg=cfg,
        state=state,
        ts_local=ts_local,
        target_bid_tick=target_bid_tick,
        target_ask_tick=target_ask_tick,
        planned_actions=actions,
        pos_limit=pos_limit,
    )


def update_quote_throttle_state(
    state: QuoteThrottleState,
    *,
    ts_local: int,
    target_bid_tick: int,
    target_ask_tick: int,
    actions: list[Action],
) -> None:
    if any(
        action.kind == "submit"
        or (action.kind == "cancel" and action.side in {"buy", "sell"})
        for action in actions
    ):
        state.mark_sent(ts_local, target_bid_tick, target_ask_tick)


def compute_top5_size(depth: Any) -> tuple[float, float]:
    best_bid_tick = int(depth.best_bid_tick)
    best_ask_tick = int(depth.best_ask_tick)
    roi_lb_tick = int(depth.roi_lb_tick)
    roi_ub_tick = int(depth.roi_ub_tick)

    bid_size = 0.0
    ask_size = 0.0

    for i in range(5):
        bt = best_bid_tick - i
        at = best_ask_tick + i

        if roi_lb_tick <= bt <= roi_ub_tick:
            bq = depth.bid_qty_at_tick(bt)
            bid_size += bq

        if roi_lb_tick <= at <= roi_ub_tick:
            aq = depth.ask_qty_at_tick(at)
            ask_size += aq

    return bid_size, ask_size


def build_market_view_from_depth(
    depth: Any,
    *,
    source: str,
    ts_local: int = 0,
    ts_exch: int = 0,
    feed_latency_ns: int = 0,
    market_overlay_source: str = "",
    top5_overlay_source: str = "",
) -> MarketView:
    best_bid = float(depth.best_bid)
    best_ask = float(depth.best_ask)
    spread = best_ask - best_bid
    mid = 0.5 * (best_bid + best_ask)
    bid_size, ask_size = compute_top5_size(depth)
    bid_top5_ticks, bid_top5_qtys, ask_top5_ticks, ask_top5_qtys = format_top5_levels(depth)
    stale_ms = max(0.0, float(feed_latency_ns) / 1_000_000.0) if feed_latency_ns > 0 else 0.0
    return MarketView(
        source=source,
        top5_source=source,
        best_bid=best_bid,
        best_ask=best_ask,
        best_bid_tick=int(depth.best_bid_tick),
        best_ask_tick=int(depth.best_ask_tick),
        mid=mid,
        spread=spread,
        bid_size=bid_size,
        ask_size=ask_size,
        bid_top5_ticks=bid_top5_ticks,
        bid_top5_qtys=bid_top5_qtys,
        ask_top5_ticks=ask_top5_ticks,
        ask_top5_qtys=ask_top5_qtys,
        ts_local=int(ts_local),
        ts_exch=int(ts_exch),
        feed_latency_ns=int(feed_latency_ns),
        stale_ms=stale_ms,
        market_overlay_source=market_overlay_source,
        top5_overlay_source=top5_overlay_source,
    )


def format_top5_levels(depth: Any) -> tuple[str, str, str, str]:
    best_bid_tick = int(depth.best_bid_tick)
    best_ask_tick = int(depth.best_ask_tick)
    roi_lb_tick = int(depth.roi_lb_tick)
    roi_ub_tick = int(depth.roi_ub_tick)

    bid_ticks: list[str] = []
    bid_qtys: list[str] = []
    ask_ticks: list[str] = []
    ask_qtys: list[str] = []

    for i in range(5):
        bt = best_bid_tick - i
        if roi_lb_tick <= bt <= roi_ub_tick:
            bid_ticks.append(str(bt))
            bid_qtys.append(str(float(depth.bid_qty_at_tick(bt))))
        else:
            bid_ticks.append("")
            bid_qtys.append("")

        at = best_ask_tick + i
        if roi_lb_tick <= at <= roi_ub_tick:
            ask_ticks.append(str(at))
            ask_qtys.append(str(float(depth.ask_qty_at_tick(at))))
        else:
            ask_ticks.append("")
            ask_qtys.append("")

    return (
        "|".join(bid_ticks),
        "|".join(bid_qtys),
        "|".join(ask_ticks),
        "|".join(ask_qtys),
    )


def impact_cost(order_notional: float, cfg: dict[str, Any]) -> float:
    threshold = float(cfg["threshold_notional"])
    cap = float(cfg["impact_cap"])
    k1 = float(cfg["k1"])
    k2 = float(cfg["k2"])

    if order_notional <= threshold:
        impact = k1 * order_notional
    else:
        impact = k1 * threshold + k2 * (order_notional - threshold)

    return min(impact, cap)


def clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def round_to_tick(price: float, tick_size: float) -> int:
    return int(round(price / tick_size))


def empty_audit_row() -> dict[str, Any]:
    return {field: "" for field in AUDIT_FIELDS}


def _base_context_fields(
    *,
    run_id: str,
    symbol: str,
    strategy_seq: int,
    ts_local: int,
    ts_exch: int,
    replay_scheduled_ts_local: int = 0,
    bt_feed_ts_local: int = 0,
    bt_feed_ts_exch: int = 0,
    replay_lag_ns: int = 0,
    best_bid: float = 0.0,
    best_ask: float = 0.0,
    mid: float = 0.0,
    position: float = 0.0,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "symbol": symbol,
        "strategy_seq": int(strategy_seq),
        "ts_local": int(ts_local),
        "ts_exch": int(ts_exch),
        "replay_scheduled_ts_local": int(replay_scheduled_ts_local),
        "bt_feed_ts_local": int(bt_feed_ts_local),
        "bt_feed_ts_exch": int(bt_feed_ts_exch),
        "replay_lag_ns": int(replay_lag_ns),
        "replay_lag_abs_ns": abs(int(replay_lag_ns)),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "position": position,
    }


def build_lifecycle_event_row(
    *,
    run_id: str,
    symbol: str,
    strategy_seq: int,
    event_seq: int,
    event_type: str,
    event_source: str,
    ts_local: int,
    ts_exch: int = 0,
    replay_scheduled_ts_local: int = 0,
    bt_feed_ts_local: int = 0,
    bt_feed_ts_exch: int = 0,
    replay_lag_ns: int = 0,
    action: Action | None = None,
    action_order_price_tick: int | str = "",
    order: OrderSnapshot | None = None,
    linked_strategy_seq: int | None = None,
    linked_action: str = "",
    linked_order_id: str = "",
    best_bid: float = 0.0,
    best_ask: float = 0.0,
    mid: float = 0.0,
    position: float = 0.0,
    local_open_orders: str = "",
    rest_open_orders: str = "",
    open_order_diff: str = "",
    rest_position: float | str = "",
    position_mismatch: float | str = "",
    rest_open_order_count: int | str = "",
    local_open_order_count: int | str = "",
    safety_status: str = "",
    safety_detail: str = "",
    cancel_requested: bool = False,
    cancel_request_ts: int = 0,
    cancel_ack_ts: int = 0,
    fill_ts: int = 0,
    fill_qty: float = 0.0,
    fill_price: float = 0.0,
    fill_trade_id: str = "",
    fill_after_cancel_request: bool = False,
    local_order_seen: bool | str = "",
    rest_order_seen: bool | str = "",
    ws_order_seen: bool | str = "",
    client_order_id: str = "",
    exchange_order_id: str = "",
    ws_event_time: int | str = "",
    rest_update_time: int | str = "",
    lifecycle_state: str = "",
    lifecycle_detail: str = "",
) -> dict[str, Any]:
    row = empty_audit_row()
    row.update(
        _base_context_fields(
            run_id=run_id,
            symbol=symbol,
            strategy_seq=strategy_seq,
            ts_local=ts_local,
            ts_exch=ts_exch,
            replay_scheduled_ts_local=replay_scheduled_ts_local,
            bt_feed_ts_local=bt_feed_ts_local,
            bt_feed_ts_exch=bt_feed_ts_exch,
            replay_lag_ns=replay_lag_ns,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=mid,
            position=position,
        )
    )
    row.update(
        {
            "event_type": event_type,
            "event_source": event_source,
            "event_seq": int(event_seq),
            "action": linked_action or (f"{action.kind}_{action.side}" if action is not None else ""),
            "order_id": str(action.order_id if action is not None else (order.order_id if order is not None else "")),
            "linked_strategy_seq": int(linked_strategy_seq if linked_strategy_seq is not None else strategy_seq),
            "linked_action": linked_action or (f"{action.kind}_{action.side}" if action is not None else ""),
            "linked_order_id": linked_order_id or str(action.order_id if action is not None else (order.order_id if order is not None else "")),
            "cancel_requested": int(bool(cancel_requested)),
            "cancel_request_ts": int(cancel_request_ts),
            "cancel_ack_ts": int(cancel_ack_ts),
            "fill_ts": int(fill_ts),
            "fill_qty": fill_qty,
            "fill_price": fill_price,
            "fill_trade_id": fill_trade_id,
            "fill_after_cancel_request": int(bool(fill_after_cancel_request)),
            "local_order_seen": "" if local_order_seen == "" else int(bool(local_order_seen)),
            "rest_order_seen": "" if rest_order_seen == "" else int(bool(rest_order_seen)),
            "ws_order_seen": "" if ws_order_seen == "" else int(bool(ws_order_seen)),
            "client_order_id": client_order_id,
            "exchange_order_id": exchange_order_id,
            "ws_event_time": ws_event_time,
            "rest_update_time": rest_update_time,
            "lifecycle_state": lifecycle_state or event_type,
            "lifecycle_detail": lifecycle_detail,
            "local_open_orders": local_open_orders,
            "rest_open_orders": rest_open_orders,
            "open_order_diff": open_order_diff,
            "rest_position": rest_position,
            "position_mismatch": position_mismatch,
            "rest_open_order_count": rest_open_order_count,
            "local_open_order_count": local_open_order_count,
            "safety_status": safety_status,
            "safety_detail": safety_detail,
        }
    )

    if action is not None:
        row.update(
            {
                "order_side": action.side,
                "order_price": float(action.price),
                "order_price_tick": action_order_price_tick,
                "order_qty": float(action.qty),
            }
        )

    if order is not None:
        row.update(
            {
                "order_side": order.side,
                "order_price": order.price,
                "order_price_tick": order.price_tick,
                "order_qty": order.qty,
                "order_remaining_qty": order.leaves_qty,
                "order_executed_qty": order.exec_qty,
                "order_status": order.status,
                "order_time_in_force": order.time_in_force,
                "ts_exch": ts_exch or order.exch_timestamp,
            }
        )

    return row


def collect_working_orders(order_dict: Any) -> WorkingOrders:
    buy = None
    sell = None
    extras: list[ExtraOrder] = []

    values = order_dict.values()
    while values.has_next():
        order = values.get()
        if order.status != NEW:
            continue
        if order.side == BUY:
            if buy is None:
                buy = order
            else:
                extras.append(
                    ExtraOrder(
                        int(order.order_id),
                        "buy",
                        int(order.price_tick),
                        _order_req_name(int(order.req)),
                        bool(order.cancellable),
                        order,
                    )
                )
        elif order.side == SELL:
            if sell is None:
                sell = order
            else:
                extras.append(
                    ExtraOrder(
                        int(order.order_id),
                        "sell",
                        int(order.price_tick),
                        _order_req_name(int(order.req)),
                        bool(order.cancellable),
                        order,
                    )
                )

    return WorkingOrders(buy=buy, sell=sell, extras=extras)


def decide_actions(
    working: WorkingOrders,
    target_bid_tick: int,
    target_ask_tick: int,
    qty: float,
    tick_size: float,
    pos_limit: bool,
    position_notional: float,
    next_order_id: int,
    two_phase_replace_enabled: bool = False,
    position: float = 0.0,
    max_position_qty: float = 0.0,
    add_side_soft_limit_qty: float = 0.0,
    add_side_cooldown_block_buy: bool = False,
    add_side_cooldown_block_sell: bool = False,
    cancel_race_guard_block_buy: bool = False,
    cancel_race_guard_block_sell: bool = False,
    adverse_timing_guard_block_buy: bool = False,
    adverse_timing_guard_block_sell: bool = False,
    add_side_toxic_timing_guard_block_buy: bool = False,
    add_side_toxic_timing_guard_block_sell: bool = False,
    add_side_inflight_buy_qty: float | None = None,
    add_side_inflight_sell_qty: float | None = None,
) -> tuple[list[Action], int]:
    actions: list[Action] = []

    # Clean extras first to keep one-order-per-side invariant. If an extra is
    # already cancel-pending, wait for the lifecycle update instead of sending
    # duplicate cancels on every decision tick.
    if working.extra_ids:
        for extra in working.extras:
            if extra.cancellable and extra.req != "cancel":
                actions.append(Action("cancel", "extra", int(extra.order_id), 0.0, 0.0))
                return actions, next_order_id
        return actions, next_order_id

    need_reduce_sell = pos_limit and position_notional > 0
    need_reduce_buy = pos_limit and position_notional < 0

    desired_buy = not pos_limit or need_reduce_buy
    desired_sell = not pos_limit or need_reduce_sell

    if add_side_cooldown_block_buy and position >= 0.0:
        desired_buy = False
    if add_side_cooldown_block_sell and position <= 0.0:
        desired_sell = False
    if cancel_race_guard_block_buy and position >= 0.0:
        desired_buy = False
    if cancel_race_guard_block_sell and position <= 0.0:
        desired_sell = False
    if adverse_timing_guard_block_buy and position >= 0.0:
        desired_buy = False
    if adverse_timing_guard_block_sell and position <= 0.0:
        desired_sell = False

    toxic_block_buy_submit = bool(add_side_toxic_timing_guard_block_buy and position >= 0.0)
    toxic_block_sell_submit = bool(add_side_toxic_timing_guard_block_sell and position <= 0.0)

    buy_diff = 0
    sell_diff = 0
    if desired_buy and working.buy is not None:
        buy_diff = abs(int(working.buy.price_tick) - target_bid_tick)
    if desired_sell and working.sell is not None:
        sell_diff = abs(int(working.sell.price_tick) - target_ask_tick)

    will_submit_buy = desired_buy and (
        working.buy is None
        or (
            working.buy is not None
            and buy_diff > 1
            and working.buy.cancellable
            and not two_phase_replace_enabled
        )
    )
    will_submit_sell = desired_sell and (
        working.sell is None
        or (
            working.sell is not None
            and sell_diff > 1
            and working.sell.cancellable
            and not two_phase_replace_enabled
        )
    )

    # When a quantity cap is active, account for outstanding add-side leaves
    # before submitting another maker order. This prevents a strategy that is
    # still below the cap from placing one more same-side order that can push
    # the next fill beyond the intended inventory envelope.
    if max_position_qty > 0.0:
        if desired_buy and position >= 0.0:
            buy_exposure = working_side_leaves_qty(working, "buy") + max(
                0.0,
                float(add_side_inflight_buy_qty or 0.0),
            )
            projected = position + buy_exposure + (qty if will_submit_buy else 0.0)
            if projected > max_position_qty:
                desired_buy = False
        if desired_sell and position <= 0.0:
            sell_exposure = working_side_leaves_qty(working, "sell") + max(
                0.0,
                float(add_side_inflight_sell_qty or 0.0),
            )
            projected = abs(position) + sell_exposure + (qty if will_submit_sell else 0.0)
            if projected > max_position_qty:
                desired_sell = False

    if add_side_soft_limit_qty > 0.0:
        if desired_buy and position >= 0.0:
            buy_exposure = working_side_leaves_qty(working, "buy") + max(
                0.0,
                float(add_side_inflight_buy_qty or 0.0),
            )
            projected = position + buy_exposure + (qty if will_submit_buy else 0.0)
            if projected > add_side_soft_limit_qty:
                desired_buy = False
        if desired_sell and position <= 0.0:
            sell_exposure = working_side_leaves_qty(working, "sell") + max(
                0.0,
                float(add_side_inflight_sell_qty or 0.0),
            )
            projected = abs(position) + sell_exposure + (qty if will_submit_sell else 0.0)
            if projected > add_side_soft_limit_qty:
                desired_sell = False

    # Remove undesired side first.
    if not desired_buy and working.buy is not None and working.buy.cancellable:
        actions.append(Action("cancel", "buy", int(working.buy.order_id), 0.0, 0.0))
    if not desired_sell and working.sell is not None and working.sell.cancellable:
        actions.append(Action("cancel", "sell", int(working.sell.order_id), 0.0, 0.0))

    # > 1 tick: cancel first then submit.
    if desired_buy and working.buy is not None and buy_diff > 1 and working.buy.cancellable:
        actions.append(Action("cancel", "buy", int(working.buy.order_id), 0.0, 0.0))
        if not two_phase_replace_enabled and not toxic_block_buy_submit:
            oid = next_order_id
            next_order_id += 1
            actions.append(Action("submit", "buy", oid, target_bid_tick * tick_size, qty))
    if desired_sell and working.sell is not None and sell_diff > 1 and working.sell.cancellable:
        actions.append(Action("cancel", "sell", int(working.sell.order_id), 0.0, 0.0))
        if not two_phase_replace_enabled and not toxic_block_sell_submit:
            oid = next_order_id
            next_order_id += 1
            actions.append(Action("submit", "sell", oid, target_ask_tick * tick_size, qty))

    if desired_buy and working.buy is None and not toxic_block_buy_submit:
        oid = next_order_id
        next_order_id += 1
        actions.append(Action("submit", "buy", oid, target_bid_tick * tick_size, qty))
    if desired_sell and working.sell is None and not toxic_block_sell_submit:
        oid = next_order_id
        next_order_id += 1
        actions.append(Action("submit", "sell", oid, target_ask_tick * tick_size, qty))

    return actions, next_order_id

def build_audit_row(
    *,
    run_id: str,
    symbol: str,
    strategy_seq: int,
    ts_local: int,
    ts_exch: int,
    action_order_id: str,
    action_name: str,
    planned_order_id: str,
    planned_action: str,
    throttle_reason: str,
    reject_reason: str,
    req_ts: int,
    exch_ts: int,
    resp_ts: int,
    entry_latency_ns: int,
    resp_latency_ns: int,
    predicted_entry_ns: int,
    best_bid: float,
    best_ask: float,
    mid: float,
    fair: float,
    reservation: float,
    half_spread: float,
    position: float,
    auditlatency_ms: float,
    dropped_by_latency: bool,
    dropped_by_api_limit: bool,
    pos_limit: bool,
    impact_cost_val: float,
    spread_bps: float,
    vol_bps: float,
    inventory_score: float,
    feed_latency_ns: int,
    latency_signal_ns: int,
    bid_size: float,
    ask_size: float,
    bid_top5_ticks: str = "",
    bid_top5_qtys: str = "",
    ask_top5_ticks: str = "",
    ask_top5_qtys: str = "",
    market_view_source: str = "",
    top5_source: str = "",
    market_overlay_source: str = "",
    top5_overlay_source: str = "",
    book_view_ts_local: int = 0,
    book_view_ts_exch: int = 0,
    book_view_feed_latency_ns: int = 0,
    book_view_stale_ms: float = 0.0,
    top5_depth_best_bid_tick: int = 0,
    top5_depth_best_ask_tick: int = 0,
    greek_values: GreekValues,
    greek_adjustment: float,
    target_bid_tick: int,
    target_ask_tick: int,
    working_bid_tick: int,
    working_ask_tick: int,
    working_buy_order_id: str,
    working_sell_order_id: str,
    working_bid_qty: str = "",
    working_ask_qty: str = "",
    working_bid_status: str = "",
    working_ask_status: str = "",
    working_bid_req: str = "",
    working_ask_req: str = "",
    working_bid_pending_cancel: str = "",
    working_ask_pending_cancel: str = "",
    cancel_race_guard_buy_active: bool = False,
    cancel_race_guard_sell_active: bool = False,
    adverse_timing_guard_buy_active: bool = False,
    adverse_timing_guard_sell_active: bool = False,
    adverse_timing_guard_buy_reason: str = "",
    adverse_timing_guard_sell_reason: str = "",
    adverse_timing_guard_buy_until_ts: int = 0,
    adverse_timing_guard_sell_until_ts: int = 0,
    adverse_timing_guard_target_move_ticks_buy: int = 0,
    adverse_timing_guard_target_move_ticks_sell: int = 0,
    add_side_submit_eligible_buy: bool = False,
    add_side_submit_eligible_sell: bool = False,
    add_side_submit_blocked_buy: bool = False,
    add_side_submit_blocked_sell: bool = False,
    add_side_submit_block_reason_buy: str = "",
    add_side_submit_block_reason_sell: str = "",
    add_side_submit_reduce_side_allowed_buy: bool = False,
    add_side_submit_reduce_side_allowed_sell: bool = False,
    target_move_since_last_quote_or_cancel_buy: int = 0,
    target_move_since_last_quote_or_cancel_sell: int = 0,
    last_cancel_request_age_ms_buy: float = -1.0,
    last_cancel_request_age_ms_sell: float = -1.0,
    last_cancel_fill_age_ms_buy: float = -1.0,
    last_cancel_fill_age_ms_sell: float = -1.0,
    toxic_timing_guard_until_ts_buy: int = 0,
    toxic_timing_guard_until_ts_sell: int = 0,
    extra_order_ids: str = "",
    extra_order_sides: str = "",
    extra_order_price_ticks: str = "",
    replay_scheduled_ts_local: int = 0,
    bt_feed_ts_local: int = 0,
    bt_feed_ts_exch: int = 0,
    replay_lag_ns: int = 0,
    rest_position: float = 0.0,
    position_mismatch: float = 0.0,
    rest_open_order_count: int = 0,
    local_open_order_count: int = 0,
    safety_status: str = "",
    local_open_orders: str = "",
    rest_open_orders: str = "",
    open_order_diff: str = "",
    safety_detail: str = "",
) -> dict[str, Any]:
    row = empty_audit_row()
    working_defaults = _working_semantic_defaults_from_local_open_orders(local_open_orders)
    row.update({
        "run_id": run_id,
        "symbol": symbol,
        "strategy_seq": strategy_seq,
        "event_type": "decision",
        "event_source": "strategy",
        "event_seq": "",
        "ts_local": ts_local,
        "ts_exch": ts_exch,
        "replay_scheduled_ts_local": replay_scheduled_ts_local,
        "bt_feed_ts_local": bt_feed_ts_local,
        "bt_feed_ts_exch": bt_feed_ts_exch,
        "replay_lag_ns": replay_lag_ns,
        "replay_lag_abs_ns": abs(int(replay_lag_ns)),
        "order_id": action_order_id,
        "action": action_name,
        "planned_order_id": planned_order_id,
        "planned_action": planned_action,
        "throttle_reason": throttle_reason,
        "reject_reason": reject_reason,
        "req_ts": req_ts,
        "exch_ts": exch_ts,
        "resp_ts": resp_ts,
        "entry_latency_ns": entry_latency_ns,
        "resp_latency_ns": resp_latency_ns,
        "predicted_entry_ns": predicted_entry_ns,
        "spike_flag": int(max(predicted_entry_ns, entry_latency_ns, resp_latency_ns) >= 8_000_000),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid": mid,
        "fair": fair,
        "reservation": reservation,
        "half_spread": half_spread,
        "position": position,
        "auditlatency_ms": auditlatency_ms,
        "dropped_by_latency": int(dropped_by_latency),
        "dropped_by_api_limit": int(dropped_by_api_limit),
        "pos_limit": int(pos_limit),
        "impact_cost": impact_cost_val,
        "spread_bps": spread_bps,
        "vol_bps": vol_bps,
        "inventory_score": inventory_score,
        "feed_latency_ns": feed_latency_ns,
        "latency_signal_ms": latency_signal_ns / 1_000_000.0,
        "bid_size": bid_size,
        "ask_size": ask_size,
        "bid_top5_ticks": bid_top5_ticks,
        "bid_top5_qtys": bid_top5_qtys,
        "ask_top5_ticks": ask_top5_ticks,
        "ask_top5_qtys": ask_top5_qtys,
        "market_view_source": market_view_source,
        "top5_source": top5_source,
        "market_overlay_source": market_overlay_source,
        "top5_overlay_source": top5_overlay_source,
        "book_view_ts_local": int(book_view_ts_local),
        "book_view_ts_exch": int(book_view_ts_exch),
        "book_view_feed_latency_ns": int(book_view_feed_latency_ns),
        "book_view_stale_ms": float(book_view_stale_ms),
        "top5_depth_best_bid_tick": int(top5_depth_best_bid_tick),
        "top5_depth_best_ask_tick": int(top5_depth_best_ask_tick),
        "greek_delta": greek_values.delta,
        "greek_gamma": greek_values.gamma,
        "greek_vega": greek_values.vega,
        "greek_theta": greek_values.theta,
        "greek_adjustment": greek_adjustment,
        "target_bid_tick": target_bid_tick,
        "target_ask_tick": target_ask_tick,
        "working_bid_tick": working_bid_tick,
        "working_ask_tick": working_ask_tick,
        "working_buy_order_id": working_buy_order_id,
        "working_sell_order_id": working_sell_order_id,
        "working_bid_qty": working_bid_qty or working_defaults["working_bid_qty"],
        "working_ask_qty": working_ask_qty or working_defaults["working_ask_qty"],
        "working_bid_status": working_bid_status or working_defaults["working_bid_status"],
        "working_ask_status": working_ask_status or working_defaults["working_ask_status"],
        "working_bid_req": working_bid_req or working_defaults["working_bid_req"],
        "working_ask_req": working_ask_req or working_defaults["working_ask_req"],
        "working_bid_pending_cancel": working_bid_pending_cancel or working_defaults["working_bid_pending_cancel"],
        "working_ask_pending_cancel": working_ask_pending_cancel or working_defaults["working_ask_pending_cancel"],
        "cancel_race_guard_buy_active": int(bool(cancel_race_guard_buy_active)),
        "cancel_race_guard_sell_active": int(bool(cancel_race_guard_sell_active)),
        "adverse_timing_guard_buy_active": int(bool(adverse_timing_guard_buy_active)),
        "adverse_timing_guard_sell_active": int(bool(adverse_timing_guard_sell_active)),
        "adverse_timing_guard_buy_reason": adverse_timing_guard_buy_reason,
        "adverse_timing_guard_sell_reason": adverse_timing_guard_sell_reason,
        "adverse_timing_guard_buy_until_ts": int(adverse_timing_guard_buy_until_ts),
        "adverse_timing_guard_sell_until_ts": int(adverse_timing_guard_sell_until_ts),
        "adverse_timing_guard_target_move_ticks_buy": int(adverse_timing_guard_target_move_ticks_buy),
        "adverse_timing_guard_target_move_ticks_sell": int(adverse_timing_guard_target_move_ticks_sell),
        "add_side_submit_eligible_buy": int(bool(add_side_submit_eligible_buy)),
        "add_side_submit_eligible_sell": int(bool(add_side_submit_eligible_sell)),
        "add_side_submit_blocked_buy": int(bool(add_side_submit_blocked_buy)),
        "add_side_submit_blocked_sell": int(bool(add_side_submit_blocked_sell)),
        "add_side_submit_block_reason_buy": add_side_submit_block_reason_buy,
        "add_side_submit_block_reason_sell": add_side_submit_block_reason_sell,
        "add_side_submit_reduce_side_allowed_buy": int(bool(add_side_submit_reduce_side_allowed_buy)),
        "add_side_submit_reduce_side_allowed_sell": int(bool(add_side_submit_reduce_side_allowed_sell)),
        "target_move_since_last_quote_or_cancel_buy": int(target_move_since_last_quote_or_cancel_buy),
        "target_move_since_last_quote_or_cancel_sell": int(target_move_since_last_quote_or_cancel_sell),
        "last_cancel_request_age_ms_buy": float(last_cancel_request_age_ms_buy),
        "last_cancel_request_age_ms_sell": float(last_cancel_request_age_ms_sell),
        "last_cancel_fill_age_ms_buy": float(last_cancel_fill_age_ms_buy),
        "last_cancel_fill_age_ms_sell": float(last_cancel_fill_age_ms_sell),
        "toxic_timing_guard_until_ts_buy": int(toxic_timing_guard_until_ts_buy),
        "toxic_timing_guard_until_ts_sell": int(toxic_timing_guard_until_ts_sell),
        "extra_order_ids": extra_order_ids,
        "extra_order_sides": extra_order_sides,
        "extra_order_price_ticks": extra_order_price_ticks,
        "local_open_orders": local_open_orders,
        "rest_position": rest_position,
        "position_mismatch": position_mismatch,
        "rest_open_order_count": rest_open_order_count,
        "local_open_order_count": local_open_order_count,
        "rest_open_orders": rest_open_orders,
        "open_order_diff": open_order_diff,
        "safety_status": safety_status,
        "safety_detail": safety_detail,
    })
    return row
