#!/usr/bin/env python3
"""Run Binance tick market-making backtest with audit logging."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Ensure local py-hftbacktest package is importable when running from repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PY_HFTBACKTEST = PROJECT_ROOT / "py-hftbacktest"
if (
    os.environ.get("HFTBACKTEST_USE_LOCAL_PY", "0") == "1"
    and PY_HFTBACKTEST.exists()
    and str(PY_HFTBACKTEST) not in sys.path
):
    sys.path.insert(0, str(PY_HFTBACKTEST))

from hftbacktest import (
    ALL_ASSETS,
    BUY,
    GTX,
    LIMIT,
    NEW,
    SELL,
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
)

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


class LatencyOracle:
    def __init__(self, data: np.ndarray):
        self.req = data["req_ts"]
        self.exch = data["exch_ts"]
        self.resp = data["resp_ts"]
        self.i = 0

    def entry_latency_ns(self, ts: int) -> int:
        n = len(self.req)
        if n == 0:
            return 0
        while self.i + 1 < n and self.req[self.i + 1] <= ts:
            self.i += 1

        req_ts = int(self.req[self.i])
        exch_ts = int(self.exch[self.i])
        resp_ts = int(self.resp[self.i])

        if exch_ts > req_ts > 0:
            return exch_ts - req_ts
        if resp_ts > req_ts > 0:
            return resp_ts - req_ts
        return 0


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
    def from_config(cls, cfg: dict[str, Any]) -> "GreekOracle":
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

        path = _expand(signal_csv)
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


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _window_ns(window: str) -> int | None:
    if window == "full_day":
        return None
    if window == "first_5m":
        return 5 * 60 * 1_000_000_000
    if window == "first_2h":
        return 2 * 60 * 60 * 1_000_000_000
    if window == "first_6h":
        return 6 * 60 * 60 * 1_000_000_000
    raise ValueError(f"Unsupported window: {window}")


def _slice_data_by_window(data: np.ndarray, window: str) -> np.ndarray:
    if len(data) == 0:
        return data

    duration_ns = _window_ns(window)
    if duration_ns is None:
        return data

    start_ts = int(data["local_ts"][0])
    end_ts = start_ts + duration_ns
    mask = data["local_ts"] <= end_ts
    return data[mask]


def _compute_top5_size(depth: Any) -> tuple[float, float]:
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


def _impact_cost(order_notional: float, cfg: dict[str, Any]) -> float:
    threshold = float(cfg["threshold_notional"])
    cap = float(cfg["impact_cap"])
    k1 = float(cfg["k1"])
    k2 = float(cfg["k2"])

    if order_notional <= threshold:
        impact = k1 * order_notional
    else:
        impact = k1 * threshold + k2 * (order_notional - threshold)

    return min(impact, cap)


def _clamp(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _round_to_tick(price: float, tick_size: float) -> int:
    return int(round(price / tick_size))


def _load_order_latency_array(config: dict[str, Any]) -> np.ndarray:
    path = str(config["latency"].get("order_latency_npz", "")).strip()
    if not path:
        raise ValueError("config.latency.order_latency_npz is required")
    npz = np.load(_expand(path))
    return npz["data"]


def _collect_working_orders(order_dict: Any) -> WorkingOrders:
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


def _decide_actions(
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


def run_backtest(config: dict[str, Any], manifest: dict[str, Any], window_override: str | None = None) -> dict[str, Any]:
    symbol = str(config["symbol"]["name"])
    market = config["market"]
    risk = config["risk"]
    fair_cfg = config["fair"]
    greek_cfg = config.get("greeks", {})
    latency_cfg = config["latency"]
    api_cfg = config["api_limit"]
    fee_cfg = config["fee"]
    queue_cfg = config["queue"]

    output_root = _expand(str(config["paths"]["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{symbol.lower()}_{manifest['start_day']}_to_{manifest['end_day']}"
    audit_name = str(config["audit"]["output_csv"])
    audit_path = output_root / audit_name

    data_files = [str(_expand(p)) for p in manifest["data_files"]]
    initial_snapshot = manifest.get("initial_snapshot")

    window = window_override or str(config["backtest"]["window"])

    data_for_asset: list[Any]
    if window == "full_day" and len(data_files) >= 1:
        data_for_asset = data_files
    else:
        first_data = np.load(data_files[0])["data"]
        sliced = _slice_data_by_window(first_data, window)
        data_for_asset = [sliced]

    latency_data = _load_order_latency_array(config)
    latency_oracle = LatencyOracle(latency_data)
    greek_oracle = GreekOracle.from_config(greek_cfg)

    asset = (
        BacktestAsset()
        .data(data_for_asset)
        .linear_asset(float(market.get("contract_size", 1.0)))
        .intp_order_latency(latency_data, 0)
        .power_prob_queue_model3(float(queue_cfg["power_prob_n"]))
        .no_partial_fill_exchange()
        .trading_value_fee_model(float(fee_cfg["maker"]), float(fee_cfg["taker"]))
        .tick_size(float(market["tick_size"]))
        .lot_size(float(market["lot_size"]))
        .roi_lb(float(market.get("roi_lb", 0.0)))
        .roi_ub(float(market.get("roi_ub", 1_000_000.0)))
    )

    if initial_snapshot:
        asset.initial_snapshot(str(_expand(initial_snapshot)))

    hbt = ROIVectorMarketDepthBacktest([asset])

    sigma_est = EwmaSigma()
    bucket = TokenBucket.create(float(api_cfg["capacity"]), float(api_cfg["refill_per_sec"]))
    min_interval_ns = int(float(api_cfg["min_interval_ms"]) * 1_000_000)
    latency_guard_ns = int(float(latency_cfg["latency_guard_ms"]) * 1_000_000)

    next_order_id = 1
    strategy_seq = 0
    last_api_ts: int | None = None
    timeout_ns = int(config["backtest"]["wait_timeout_ns"])

    rows_written = 0
    with audit_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()

        while True:
            rc = hbt.wait_next_feed(True, timeout_ns)
            if rc == 1:
                break
            if rc == 0:
                continue

            ts_local = int(hbt.current_timestamp)
            depth = hbt.depth(0)
            best_bid = float(depth.best_bid)
            best_ask = float(depth.best_ask)

            if not (math.isfinite(best_bid) and math.isfinite(best_ask)):
                continue
            if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
                continue

            strategy_seq += 1

            tick_size = float(depth.tick_size)
            lot_size = float(depth.lot_size)
            spread = best_ask - best_bid
            mid = 0.5 * (best_bid + best_ask)

            sigma = sigma_est.update(ts_local, mid)
            bid_size, ask_size = _compute_top5_size(depth)

            position = float(hbt.position(0))
            greek_values = greek_oracle.values(ts_local=ts_local, position=position)
            greek_adjustment = (
                float(greek_cfg.get("w_delta", 0.0)) * greek_values.delta
                + float(greek_cfg.get("w_gamma", 0.0)) * greek_values.gamma
                + float(greek_cfg.get("w_vega", 0.0)) * greek_values.vega
                + float(greek_cfg.get("w_theta", 0.0)) * greek_values.theta
            )

            fair = (
                mid
                + float(fair_cfg["w_imb"]) * (bid_size - ask_size)
                + float(fair_cfg["w_spread"]) * spread
                + float(fair_cfg["w_vol"]) * sigma
                + greek_adjustment
            )

            position_notional = position * mid
            pos_limit = abs(position_notional) > float(risk["max_notional_pos"])

            order_notional = float(risk["order_notional"])
            impact_cost = _impact_cost(order_notional, config["impact"])

            reservation = fair - float(risk["k_inv"]) * position
            half_spread = (
                float(risk["base_spread"])
                + float(risk["k_vol"]) * sigma
                + float(risk["k_pos"]) * abs(position)
                + impact_cost
                + min(0.05, sigma * 0.1)
            )

            target_bid = _clamp(reservation - half_spread, best_bid * 0.999, best_bid)
            target_ask = _clamp(reservation + half_spread, best_ask, best_ask * 1.001)
            target_bid_tick = _round_to_tick(target_bid, tick_size)
            target_ask_tick = _round_to_tick(target_ask, tick_size)

            qty = max(lot_size, round((order_notional / mid) / lot_size) * lot_size)

            feed_lat = hbt.feed_latency(0)
            order_lat = hbt.order_latency(0)
            feed_latency_ns = int(feed_lat[1] - feed_lat[0]) if feed_lat is not None else 0

            predicted_entry_ns = int(latency_oracle.entry_latency_ns(ts_local))
            last_entry_ns = int(order_lat[1] - order_lat[0]) if order_lat is not None else 0
            last_resp_ns = int(order_lat[2] - order_lat[1]) if order_lat is not None else 0
            latency_signal_ns = max(feed_latency_ns, last_entry_ns, last_resp_ns, predicted_entry_ns)

            dropped_by_latency = latency_signal_ns > latency_guard_ns
            dropped_by_api_limit = False

            working = _collect_working_orders(hbt.orders(0))
            working_bid_tick = int(working.buy.price_tick) if working.buy is not None else -1
            working_ask_tick = int(working.sell.price_tick) if working.sell is not None else -1

            planned_actions: list[Action] = []
            executed_actions: list[Action] = []
            reject_reason = ""
            action_order_id = ""
            action_name = "keep"
            sent_api = False

            if dropped_by_latency:
                reject_reason = "latency_guard"
            else:
                planned_actions, next_order_id = _decide_actions(
                    working=working,
                    target_bid_tick=target_bid_tick,
                    target_ask_tick=target_ask_tick,
                    qty=qty,
                    tick_size=tick_size,
                    pos_limit=pos_limit,
                    position_notional=position_notional,
                    next_order_id=next_order_id,
                )

                if planned_actions:
                    if last_api_ts is not None and (ts_local - last_api_ts) < min_interval_ns:
                        dropped_by_api_limit = True
                        reject_reason = "api_interval_guard"
                    else:
                        for action in planned_actions:
                            if bool(api_cfg.get("enabled", True)) and not bucket.allow(ts_local, 1.0):
                                dropped_by_api_limit = True
                                reject_reason = "token_bucket"
                                break

                            if action.kind == "cancel":
                                hbt.cancel(0, int(action.order_id), False)
                            elif action.kind == "submit" and action.side == "buy":
                                hbt.submit_buy_order(0, int(action.order_id), action.price, action.qty, GTX, LIMIT, False)
                            elif action.kind == "submit" and action.side == "sell":
                                hbt.submit_sell_order(0, int(action.order_id), action.price, action.qty, GTX, LIMIT, False)

                            executed_actions.append(action)
                            sent_api = True
                            last_api_ts = ts_local

                        if executed_actions:
                            action_order_id = "|".join(str(a.order_id) for a in executed_actions)
                            action_name = "|".join(f"{a.kind}_{a.side}" for a in executed_actions)
                        elif not reject_reason:
                            dropped_by_api_limit = True
                            reject_reason = "api_limit"

            order_lat_after = hbt.order_latency(0)
            req_ts = int(order_lat_after[0]) if order_lat_after is not None else 0
            exch_ts = int(order_lat_after[1]) if order_lat_after is not None else 0
            resp_ts = int(order_lat_after[2]) if order_lat_after is not None else 0
            entry_latency_ns = int(exch_ts - req_ts) if exch_ts > 0 and req_ts > 0 else 0
            resp_latency_ns = int(resp_ts - exch_ts) if exch_ts > 0 and resp_ts > exch_ts else 0
            if sent_api and entry_latency_ns > latency_guard_ns:
                dropped_by_latency = True
                if not reject_reason:
                    reject_reason = "latency_guard"

            # event2order delay; when we send in same cycle it is zero in this model.
            if sent_api and req_ts > 0:
                auditlatency_ms = max(0.0, (req_ts - ts_local) / 1_000_000.0)
            else:
                auditlatency_ms = 0.0

            spread_bps = (spread / mid) * 1e4 if mid > 0 else 0.0
            vol_bps = sigma * 1e4
            inventory_score = max(0.0, 1.0 - abs(position_notional) / float(risk["max_notional_pos"]))

            if dropped_by_latency and not reject_reason:
                reject_reason = "latency_guard"
            if dropped_by_api_limit and not reject_reason:
                reject_reason = "api_limit"

            row = {
                "run_id": run_id,
                "symbol": symbol,
                "strategy_seq": strategy_seq,
                "event_type": "decision",
                "ts_local": ts_local,
                "ts_exch": int(feed_lat[0]) if feed_lat is not None else 0,
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
                "impact_cost": impact_cost,
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
            writer.writerow(row)
            rows_written += 1

            if rows_written % int(config["audit"].get("flush_every", 1000)) == 0:
                f.flush()

            hbt.clear_inactive_orders(ALL_ASSETS)

    return {
        "run_id": run_id,
        "audit_csv": str(audit_path),
        "rows": rows_written,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Binance tick MM backtest with audit")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--manifest", required=True, help="Path to prepared data manifest JSON")
    parser.add_argument("--window", default=None, help="Optional override: first_5m|first_2h|first_6h|full_day")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_toml(_expand(args.config))
    manifest = _load_manifest(_expand(args.manifest))

    result = run_backtest(config=config, manifest=manifest, window_override=args.window)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
