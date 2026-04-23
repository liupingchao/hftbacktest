#!/usr/bin/env python3
"""Core strategy logic shared between backtest and live trading."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hftbacktest import NEW, BUY, SELL

from audit_schema import AUDIT_FIELDS


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
    def from_config(cls, cfg: dict[str, Any], expand_path: Any = None) -> "GreekOracle":
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
class WorkingOrders:
    buy: Any | None
    sell: Any | None
    extra_ids: list[int]


@dataclass
class Action:
    kind: str
    side: str
    order_id: int
    price: float
    qty: float


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


def collect_working_orders(order_dict: Any) -> WorkingOrders:
    buy = None
    sell = None
    extra_ids: list[int] = []

    values = order_dict.values()
    while values.has_next():
        order = values.get()
        if order.status != NEW:
            continue
        if order.side == BUY:
            if buy is None:
                buy = order
            else:
                extra_ids.append(int(order.order_id))
        elif order.side == SELL:
            if sell is None:
                sell = order
            else:
                extra_ids.append(int(order.order_id))

    return WorkingOrders(buy=buy, sell=sell, extra_ids=extra_ids)


def decide_actions(
    working: WorkingOrders,
    target_bid_tick: int,
    target_ask_tick: int,
    qty: float,
    tick_size: float,
    pos_limit: bool,
    position_notional: float,
    next_order_id: int,
) -> tuple[list[Action], int]:
    actions: list[Action] = []

    # Clean extras first to keep one-order-per-side invariant.
    if working.extra_ids:
        oid = int(working.extra_ids[0])
        actions.append(Action("cancel", "extra", oid, 0.0, 0.0))
        return actions, next_order_id

    need_reduce_sell = pos_limit and position_notional > 0
    need_reduce_buy = pos_limit and position_notional < 0

    desired_buy = not pos_limit or need_reduce_buy
    desired_sell = not pos_limit or need_reduce_sell

    # Remove undesired side first.
    if not desired_buy and working.buy is not None and working.buy.cancellable:
        actions.append(Action("cancel", "buy", int(working.buy.order_id), 0.0, 0.0))
    if not desired_sell and working.sell is not None and working.sell.cancellable:
        actions.append(Action("cancel", "sell", int(working.sell.order_id), 0.0, 0.0))

    buy_diff = 0
    sell_diff = 0
    if desired_buy and working.buy is not None:
        buy_diff = abs(int(working.buy.price_tick) - target_bid_tick)
    if desired_sell and working.sell is not None:
        sell_diff = abs(int(working.sell.price_tick) - target_ask_tick)

    # > 1 tick: cancel first then submit.
    if desired_buy and working.buy is not None and buy_diff > 1 and working.buy.cancellable:
        actions.append(Action("cancel", "buy", int(working.buy.order_id), 0.0, 0.0))
        oid = next_order_id
        next_order_id += 1
        actions.append(Action("submit", "buy", oid, target_bid_tick * tick_size, qty))
    if desired_sell and working.sell is not None and sell_diff > 1 and working.sell.cancellable:
        actions.append(Action("cancel", "sell", int(working.sell.order_id), 0.0, 0.0))
        oid = next_order_id
        next_order_id += 1
        actions.append(Action("submit", "sell", oid, target_ask_tick * tick_size, qty))

    if desired_buy and working.buy is None:
        oid = next_order_id
        next_order_id += 1
        actions.append(Action("submit", "buy", oid, target_bid_tick * tick_size, qty))
    if desired_sell and working.sell is None:
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
    greek_values: GreekValues,
    greek_adjustment: float,
    target_bid_tick: int,
    target_ask_tick: int,
    working_bid_tick: int,
    working_ask_tick: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "symbol": symbol,
        "strategy_seq": strategy_seq,
        "event_type": "decision",
        "ts_local": ts_local,
        "ts_exch": ts_exch,
        "order_id": action_order_id,
        "action": action_name,
        "reject_reason": reject_reason,
        "req_ts": req_ts,
        "exch_ts": exch_ts,
        "resp_ts": resp_ts,
        "entry_latency_ns": entry_latency_ns,
        "resp_latency_ns": resp_latency_ns,
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
        "greek_delta": greek_values.delta,
        "greek_gamma": greek_values.gamma,
        "greek_vega": greek_values.vega,
        "greek_theta": greek_values.theta,
        "greek_adjustment": greek_adjustment,
        "target_bid_tick": target_bid_tick,
        "target_ask_tick": target_ask_tick,
        "working_bid_tick": working_bid_tick,
        "working_ask_tick": working_ask_tick,
    }
