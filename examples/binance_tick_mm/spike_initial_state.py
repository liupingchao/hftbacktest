#!/usr/bin/env python3
"""Spike test for initial position and working-order injection support.

This script intentionally uses only the public Python hftbacktest API. It checks
whether P1 can inject live state directly into ROIVectorMarketDepthBacktest, or
whether we need an alignment-only override / execution replay path.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from hftbacktest import (
    BUY_EVENT,
    DEPTH_EVENT,
    EXCH_EVENT,
    GTX,
    GTC,
    LIMIT,
    LOCAL_EVENT,
    ROIVectorMarketDepthBacktest,
    SELL_EVENT,
    BacktestAsset,
    event_dtype,
)


def _event(ts: int, ev: int, px: float, qty: float) -> tuple[int, int, int, float, float, int, int, float]:
    return (ev | EXCH_EVENT | LOCAL_EVENT, ts, ts, px, qty, 0, 0, 0.0)


def _market_data() -> np.ndarray:
    rows = [
        _event(1_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 10.0),
        _event(1_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 10.0),
        _event(2_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 10.0),
        _event(2_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 10.0),
        _event(3_000_000_000, DEPTH_EVENT | BUY_EVENT, 100.0, 10.0),
        _event(3_000_000_000, DEPTH_EVENT | SELL_EVENT, 101.0, 10.0),
    ]
    return np.asarray(rows, dtype=event_dtype)


def _make_backtest() -> Any:
    asset = (
        BacktestAsset()
        .linear_asset(1.0)
        .data(_market_data())
        .no_partial_fill_exchange()
        .constant_order_latency(0, 0)
        .power_prob_queue_model3(3.0)
        .tick_size(0.1)
        .lot_size(0.001)
        .roi_lb(90.0)
        .roi_ub(110.0)
    )
    return ROIVectorMarketDepthBacktest([asset])


def _orders_snapshot(hbt: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    orders = hbt.orders(0)
    values = orders.values()
    while True:
        order = values.next()
        if order is None:
            break
        out.append(
            {
                "order_id": int(order.order_id),
                "side": int(order.side),
                "price_tick": int(order.price_tick),
                "qty": float(order.qty),
                "status": int(order.status),
                "cancellable": bool(order.cancellable),
            }
        )
    return out


def _load_first_feed(hbt: Any) -> dict[str, Any]:
    rc = hbt.wait_next_feed(True, 1_000_000_000)
    depth = hbt.depth(0)
    return {
        "rc": int(rc),
        "ts": int(hbt.current_timestamp),
        "best_bid": float(depth.best_bid),
        "best_ask": float(depth.best_ask),
        "position": float(hbt.position(0)),
        "orders": _orders_snapshot(hbt),
    }


def _public_api_probe() -> dict[str, Any]:
    hbt = _make_backtest()
    asset_methods = [name for name in dir(BacktestAsset) if not name.startswith("__")]
    hbt_methods = [name for name in dir(hbt) if not name.startswith("__")]
    return {
        "asset_initial_position_api": [name for name in asset_methods if "position" in name.lower()],
        "asset_order_injection_api": [
            name for name in asset_methods if "order" in name.lower() and "latency" not in name.lower()
        ],
        "hbt_position_mutation_api": [
            name for name in hbt_methods if "position" in name.lower() and name != "position"
        ],
        "hbt_order_mutation_api": [
            name
            for name in hbt_methods
            if "order" in name.lower()
            and name not in {"orders", "submit_buy_order", "submit_sell_order", "wait_order_response"}
        ],
    }


def _submit_resting_orders_probe() -> dict[str, Any]:
    hbt = _make_backtest()
    first = _load_first_feed(hbt)
    buy_rc = int(hbt.submit_buy_order(0, 101, 99.5, 0.001, GTX, LIMIT, True))
    sell_rc = int(hbt.submit_sell_order(0, 102, 101.5, 0.001, GTX, LIMIT, True))
    after_submit = {
        "position": float(hbt.position(0)),
        "orders": _orders_snapshot(hbt),
    }
    hbt.wait_next_feed(True, 1_000_000_000)
    after_next_feed = {
        "position": float(hbt.position(0)),
        "orders": _orders_snapshot(hbt),
    }
    hbt.close()
    return {
        "first_feed": first,
        "submit_buy_rc": buy_rc,
        "submit_sell_rc": sell_rc,
        "after_submit": after_submit,
        "after_next_feed": after_next_feed,
    }


def _synthetic_position_probe() -> dict[str, Any]:
    hbt = _make_backtest()
    first = _load_first_feed(hbt)
    buy_rc = int(hbt.submit_buy_order(0, 201, 101.0, 0.001, GTC, LIMIT, True))
    after_cross_buy = {
        "position": float(hbt.position(0)),
        "orders": _orders_snapshot(hbt),
    }
    sell_rc = int(hbt.submit_sell_order(0, 202, 100.0, 0.001, GTC, LIMIT, True))
    after_cross_sell = {
        "position": float(hbt.position(0)),
        "orders": _orders_snapshot(hbt),
    }
    hbt.close()
    return {
        "first_feed": first,
        "cross_buy_rc": buy_rc,
        "after_cross_buy": after_cross_buy,
        "cross_sell_rc": sell_rc,
        "after_cross_sell": after_cross_sell,
    }


def main() -> None:
    report = {
        "public_api": _public_api_probe(),
        "resting_orders_probe": _submit_resting_orders_probe(),
        "synthetic_position_probe": _synthetic_position_probe(),
    }
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
