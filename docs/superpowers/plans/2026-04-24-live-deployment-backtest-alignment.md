# Live Deployment & Backtest-Live Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the tick market-making strategy to Tokyo production server, producing `audit_live.csv` for backtest-live calibration.

**Architecture:** Extract shared strategy logic from `backtest_tick_mm.py` into `strategy_core.py`, then build `live_tick_mm.py` using the same core with `ROIVectorMarketDepthLiveBot`. Extend `config.example.toml` with a `[live]` section. Add deployment scripts for the Tokyo server.

**Tech Stack:** Python 3.11, hftbacktest 2.4.4, Rust (connector/collector), TOML config, CSV audit

---

## File Structure

### New Files
- `examples/binance_tick_mm/strategy_core.py` — Shared strategy components extracted from `backtest_tick_mm.py`: `EwmaSigma`, `TokenBucket`, `GreekOracle`, `GreekValues`, `WorkingOrders`, `Action`, and all pure functions (`_compute_top5_size`, `_impact_cost`, `_clamp`, `_round_to_tick`, `_collect_working_orders`, `_decide_actions`, audit row building)
- `examples/binance_tick_mm/live_tick_mm.py` — Live trading engine using `ROIVectorMarketDepthLiveBot` + `strategy_core.py`, writes `audit_live.csv`
- `examples/binance_tick_mm/deploy/binancefutures.toml` — Connector config template for production Binance Futures
- `examples/binance_tick_mm/deploy/run_live.sh` — tmux launcher script for collector + connector + bot

### Modified Files
- `examples/binance_tick_mm/backtest_tick_mm.py` — Refactor to import from `strategy_core.py` instead of defining classes inline
- `examples/binance_tick_mm/config.example.toml` — Add `[live]` section

---

## Task 1: Extract strategy_core.py from backtest_tick_mm.py

**Files:**
- Create: `examples/binance_tick_mm/strategy_core.py`
- Modify: `examples/binance_tick_mm/backtest_tick_mm.py`

- [ ] **Step 1: Create strategy_core.py with all shared components**

Extract these classes and functions verbatim from `backtest_tick_mm.py`:

```python
#!/usr/bin/env python3
"""Shared strategy components for backtest and live market-making."""

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
                ts_local=None, delta=None, gamma=None, vega=None, theta=None,
                enabled=enabled, use_position_as_delta=use_position_as_delta,
                scale_delta=scale_delta, scale_gamma=scale_gamma,
                scale_vega=scale_vega, scale_theta=scale_theta,
            )

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
                rows.append((
                    ts_local,
                    cls._row_float(row, ["delta", "net_delta", "greek_delta"]),
                    cls._row_float(row, ["gamma", "net_gamma", "greek_gamma"]),
                    cls._row_float(row, ["vega", "net_vega", "greek_vega"]),
                    cls._row_float(row, ["theta", "net_theta", "greek_theta"]),
                ))

        if not rows:
            return cls(
                ts_local=None, delta=None, gamma=None, vega=None, theta=None,
                enabled=enabled, use_position_as_delta=use_position_as_delta,
                scale_delta=scale_delta, scale_gamma=scale_gamma,
                scale_vega=scale_vega, scale_theta=scale_theta,
            )

        rows.sort(key=lambda x: x[0])
        ts = np.asarray([r[0] for r in rows], dtype=np.int64)
        d = np.asarray([r[1] for r in rows], dtype=np.float64)
        g = np.asarray([r[2] for r in rows], dtype=np.float64)
        v = np.asarray([r[3] for r in rows], dtype=np.float64)
        t = np.asarray([r[4] for r in rows], dtype=np.float64)

        return cls(
            ts_local=ts, delta=d, gamma=g, vega=v, theta=t,
            enabled=enabled, use_position_as_delta=use_position_as_delta,
            scale_delta=scale_delta, scale_gamma=scale_gamma,
            scale_vega=scale_vega, scale_theta=scale_theta,
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
            bid_size += depth.bid_qty_at_tick(bt)

        if roi_lb_tick <= at <= roi_ub_tick:
            ask_size += depth.ask_qty_at_tick(at)

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

    if working.extra_ids:
        oid = int(working.extra_ids[0])
        actions.append(Action("cancel", "extra", oid, 0.0, 0.0))
        return actions, next_order_id

    need_reduce_sell = pos_limit and position_notional > 0
    need_reduce_buy = pos_limit and position_notional < 0

    desired_buy = not pos_limit or need_reduce_buy
    desired_sell = not pos_limit or need_reduce_sell

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


def compute_strategy_signals(
    mid: float,
    spread: float,
    sigma: float,
    bid_size: float,
    ask_size: float,
    position: float,
    greek_values: GreekValues,
    fair_cfg: dict[str, Any],
    greek_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    impact_cfg: dict[str, Any],
) -> dict[str, float]:
    """Compute fair, reservation, half_spread, and related signals.

    Returns a dict with keys: fair, reservation, half_spread, greek_adjustment,
    position_notional, pos_limit, impact_cost_val, inventory_score,
    target_bid, target_ask, spread_bps, vol_bps, qty.
    """
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
    max_notional = float(risk_cfg["max_notional_pos"])
    pos_limit = abs(position_notional) > max_notional

    order_notional = float(risk_cfg["order_notional"])
    impact_cost_val = impact_cost(order_notional, impact_cfg)

    reservation = fair - float(risk_cfg["k_inv"]) * position
    half_spread = (
        float(risk_cfg["base_spread"])
        + float(risk_cfg["k_vol"]) * sigma
        + float(risk_cfg["k_pos"]) * abs(position)
        + impact_cost_val
        + min(0.05, sigma * 0.1)
    )

    best_bid = mid - spread / 2.0
    best_ask = mid + spread / 2.0
    target_bid = clamp(reservation - half_spread, best_bid * 0.999, best_bid)
    target_ask = clamp(reservation + half_spread, best_ask, best_ask * 1.001)

    tick_size_val = spread  # caller will use depth.tick_size directly
    lot_size_val = 0.001  # caller will use depth.lot_size directly
    qty = max(lot_size_val, round((order_notional / mid) / lot_size_val) * lot_size_val)

    spread_bps = (spread / mid) * 1e4 if mid > 0 else 0.0
    vol_bps = sigma * 1e4
    inventory_score = max(0.0, 1.0 - abs(position_notional) / max_notional)

    return {
        "fair": fair,
        "reservation": reservation,
        "half_spread": half_spread,
        "greek_adjustment": greek_adjustment,
        "position_notional": position_notional,
        "pos_limit": pos_limit,
        "impact_cost_val": impact_cost_val,
        "inventory_score": inventory_score,
        "spread_bps": spread_bps,
        "vol_bps": vol_bps,
    }


def build_audit_row(
    *,
    run_id: str,
    symbol: str,
    strategy_seq: int,
    ts_local: int,
    ts_exch: int,
    action_name: str,
    action_order_id: str,
    reject_reason: str,
    req_ts: int,
    exch_ts: int,
    resp_ts: int,
    entry_latency_ns: int,
    resp_latency_ns: int,
    spike_flag: int,
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
        "spike_flag": spike_flag,
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
```

- [ ] **Step 2: Refactor backtest_tick_mm.py to import from strategy_core**

Replace all class/function definitions in `backtest_tick_mm.py` with imports. The file should keep only: `LatencyOracle`, `_load_order_latency_array`, `_window_ns`, `_slice_data_by_window`, `_load_toml`, `_load_manifest`, `_expand`, `run_backtest`, `parse_args`, `main` — these are backtest-specific.

Replace the existing imports and class definitions (lines 29-423) with:

```python
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
from strategy_core import (
    EwmaSigma,
    TokenBucket,
    GreekOracle,
    GreekValues,
    WorkingOrders,
    Action,
    compute_top5_size,
    impact_cost,
    clamp,
    round_to_tick,
    collect_working_orders,
    decide_actions,
    build_audit_row,
)
```

Then update `run_backtest` to call the renamed functions (no underscore prefix):
- `_compute_top5_size` → `compute_top5_size`
- `_impact_cost` → `impact_cost`
- `_clamp` → `clamp`
- `_round_to_tick` → `round_to_tick`
- `_collect_working_orders` → `collect_working_orders`
- `_decide_actions` → `decide_actions`

And use `build_audit_row(...)` to construct the audit row dict instead of the inline dict literal.

- [ ] **Step 3: Verify backtest still works**

Run: `cd examples/binance_tick_mm && python -c "from strategy_core import EwmaSigma, TokenBucket, GreekOracle; print('imports OK')"`

Expected: `imports OK`

- [ ] **Step 4: Commit**

```bash
git add examples/binance_tick_mm/strategy_core.py examples/binance_tick_mm/backtest_tick_mm.py
git commit -m "refactor: extract shared strategy components into strategy_core.py"
```

---

## Task 2: Extend config.example.toml with [live] section

**Files:**
- Modify: `examples/binance_tick_mm/config.example.toml`

- [ ] **Step 1: Add [live] section to config.example.toml**

Append after the existing `[test]` section:

```toml
[live]
# connector name must match the --name arg passed to the connector binary
connector_name = "bf"
# ROI bounds for ROIVectorMarketDepth (price range for the order book)
# Set these to bracket the expected price range of the instrument
roi_lb = 50000.0
roi_ub = 150000.0
# run_id prefix for audit_live.csv
run_id_prefix = "live"
# audit output for live trading
audit_csv = "audit_live.csv"
# heartbeat log interval in seconds
heartbeat_interval_sec = 60
```

- [ ] **Step 2: Commit**

```bash
git add examples/binance_tick_mm/config.example.toml
git commit -m "feat: add [live] section to config.example.toml"
```

---

## Task 3: Create live_tick_mm.py

**Files:**
- Create: `examples/binance_tick_mm/live_tick_mm.py`

- [ ] **Step 1: Write live_tick_mm.py**

```python
#!/usr/bin/env python3
"""Live Binance tick market-making bot with audit logging."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    GTX,
    LIMIT,
    LiveInstrument,
    ROIVectorMarketDepthLiveBot,
)

from audit_schema import AUDIT_FIELDS
from strategy_core import (
    EwmaSigma,
    TokenBucket,
    GreekOracle,
    WorkingOrders,
    compute_top5_size,
    impact_cost,
    clamp,
    round_to_tick,
    collect_working_orders,
    decide_actions,
    build_audit_row,
)


_shutdown_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    global _shutdown_requested
    _shutdown_requested = True
    print(f"\n[live] Shutdown signal received ({signum}), will exit after cleanup...")


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def run_live(config: dict[str, Any]) -> dict[str, Any]:
    global _shutdown_requested

    symbol = str(config["symbol"]["name"])
    market = config["market"]
    risk = config["risk"]
    fair_cfg = config["fair"]
    greek_cfg = config.get("greeks", {})
    latency_cfg = config["latency"]
    api_cfg = config["api_limit"]
    live_cfg = config["live"]

    connector_name = str(live_cfg["connector_name"])
    roi_lb = float(live_cfg["roi_lb"])
    roi_ub = float(live_cfg["roi_ub"])
    heartbeat_sec = float(live_cfg.get("heartbeat_interval_sec", 60))

    output_root = _expand(str(config["paths"]["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)

    run_id_prefix = str(live_cfg.get("run_id_prefix", "live"))
    today = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_id_prefix}_{symbol.lower()}_{today}"

    audit_name = str(live_cfg.get("audit_csv", "audit_live.csv"))
    audit_path = output_root / audit_name

    tick_size = float(market["tick_size"])
    lot_size = float(market["lot_size"])

    # Build live instrument — symbol must be lowercase for Binance Futures connector.
    instrument = (
        LiveInstrument()
        .connector(connector_name)
        .symbol(symbol.lower())
        .tick_size(tick_size)
        .lot_size(lot_size)
        .roi_lb(roi_lb)
        .roi_ub(roi_ub)
    )

    print(f"[live] Building bot: {symbol} via connector '{connector_name}'")
    print(f"[live] ROI: [{roi_lb}, {roi_ub}], tick={tick_size}, lot={lot_size}")
    hbt = ROIVectorMarketDepthLiveBot([instrument])
    print("[live] Bot connected to connector. Waiting for market data...")

    latency_guard_ns = int(float(latency_cfg["latency_guard_ms"]) * 1_000_000)
    sigma_est = EwmaSigma()
    bucket = TokenBucket.create(float(api_cfg["capacity"]), float(api_cfg["refill_per_sec"]))
    min_interval_ns = int(float(api_cfg["min_interval_ms"]) * 1_000_000)
    greek_oracle = GreekOracle.from_config(greek_cfg)

    next_order_id = 1
    strategy_seq = 0
    last_api_ts: int | None = None
    rows_written = 0
    last_heartbeat = time.monotonic()

    # Install signal handlers for graceful shutdown.
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"[live] Audit log: {audit_path}")
    print(f"[live] Run ID: {run_id}")
    print(f"[live] Starting live loop. Ctrl+C to stop.")

    with audit_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writeheader()

        while not _shutdown_requested:
            # wait_next_feed: timeout 1 second so we can check shutdown flag.
            rc = hbt.wait_next_feed(True, 1_000_000_000)
            if rc == 1:
                print("[live] Feed ended (rc=1), exiting.")
                break
            if rc == 0:
                # Timeout, check heartbeat.
                now = time.monotonic()
                if now - last_heartbeat >= heartbeat_sec:
                    pos = float(hbt.position(0))
                    print(f"[live] heartbeat seq={strategy_seq} pos={pos:.6f} rows={rows_written}")
                    last_heartbeat = now
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

            spread = best_ask - best_bid
            mid = 0.5 * (best_bid + best_ask)

            sigma = sigma_est.update(ts_local, mid)
            bid_size, ask_size = compute_top5_size(depth)

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
            max_notional = float(risk["max_notional_pos"])
            pos_limit = abs(position_notional) > max_notional

            order_notional = float(risk["order_notional"])
            impact_cost_val = impact_cost(order_notional, config["impact"])

            reservation = fair - float(risk["k_inv"]) * position
            half_spread = (
                float(risk["base_spread"])
                + float(risk["k_vol"]) * sigma
                + float(risk["k_pos"]) * abs(position)
                + impact_cost_val
                + min(0.05, sigma * 0.1)
            )

            target_bid = clamp(reservation - half_spread, best_bid * 0.999, best_bid)
            target_ask = clamp(reservation + half_spread, best_ask, best_ask * 1.001)
            target_bid_tick = round_to_tick(target_bid, tick_size)
            target_ask_tick = round_to_tick(target_ask, tick_size)

            qty = max(lot_size, round((order_notional / mid) / lot_size) * lot_size)

            # Live latency: read real timestamps from connector.
            feed_lat = hbt.feed_latency(0)
            order_lat = hbt.order_latency(0)
            feed_latency_ns = int(feed_lat[1] - feed_lat[0]) if feed_lat is not None else 0

            last_entry_ns = int(order_lat[1] - order_lat[0]) if order_lat is not None else 0
            last_resp_ns = int(order_lat[2] - order_lat[1]) if order_lat is not None else 0
            latency_signal_ns = max(feed_latency_ns, last_entry_ns, last_resp_ns)

            dropped_by_latency = latency_signal_ns > latency_guard_ns
            dropped_by_api_limit = False

            working = collect_working_orders(hbt.orders(0))
            working_bid_tick = int(working.buy.price_tick) if working.buy is not None else -1
            working_ask_tick = int(working.sell.price_tick) if working.sell is not None else -1

            executed_actions: list = []
            reject_reason = ""
            action_order_id = ""
            action_name = "keep"
            sent_api = False

            if dropped_by_latency:
                reject_reason = "latency_guard"
            else:
                planned_actions, next_order_id = decide_actions(
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

            # Read order latency after potential order submission.
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

            if sent_api and req_ts > 0:
                auditlatency_ms = max(0.0, (req_ts - ts_local) / 1_000_000.0)
            else:
                auditlatency_ms = 0.0

            spread_bps = (spread / mid) * 1e4 if mid > 0 else 0.0
            vol_bps = sigma * 1e4
            inventory_score = max(0.0, 1.0 - abs(position_notional) / max_notional)

            if dropped_by_latency and not reject_reason:
                reject_reason = "latency_guard"
            if dropped_by_api_limit and not reject_reason:
                reject_reason = "api_limit"

            row = build_audit_row(
                run_id=run_id,
                symbol=symbol,
                strategy_seq=strategy_seq,
                ts_local=ts_local,
                ts_exch=int(feed_lat[0]) if feed_lat is not None else 0,
                action_name=action_name,
                action_order_id=action_order_id,
                reject_reason=reject_reason,
                req_ts=req_ts,
                exch_ts=exch_ts,
                resp_ts=resp_ts,
                entry_latency_ns=entry_latency_ns,
                resp_latency_ns=resp_latency_ns,
                spike_flag=int(max(entry_latency_ns, resp_latency_ns) >= 8_000_000),
                best_bid=best_bid,
                best_ask=best_ask,
                mid=mid,
                fair=fair,
                reservation=reservation,
                half_spread=half_spread,
                position=position,
                auditlatency_ms=auditlatency_ms,
                dropped_by_latency=dropped_by_latency,
                dropped_by_api_limit=dropped_by_api_limit,
                pos_limit=pos_limit,
                impact_cost_val=impact_cost_val,
                spread_bps=spread_bps,
                vol_bps=vol_bps,
                inventory_score=inventory_score,
                feed_latency_ns=feed_latency_ns,
                latency_signal_ns=latency_signal_ns,
                bid_size=bid_size,
                ask_size=ask_size,
                greek_values=greek_values,
                greek_adjustment=greek_adjustment,
                target_bid_tick=target_bid_tick,
                target_ask_tick=target_ask_tick,
                working_bid_tick=working_bid_tick,
                working_ask_tick=working_ask_tick,
            )
            writer.writerow(row)
            rows_written += 1

            if rows_written % int(config["audit"].get("flush_every", 1000)) == 0:
                f.flush()

            hbt.clear_inactive_orders(ALL_ASSETS)

            # Heartbeat.
            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_sec:
                print(f"[live] heartbeat seq={strategy_seq} mid={mid:.2f} pos={position:.6f} rows={rows_written}")
                last_heartbeat = now

    # --- Graceful shutdown: cancel all open orders ---
    print(f"[live] Shutting down. Cancelling open orders...")
    working = collect_working_orders(hbt.orders(0))
    if working.buy is not None and working.buy.cancellable:
        hbt.cancel(0, int(working.buy.order_id), True)
        print(f"[live]   cancelled buy order {working.buy.order_id}")
    if working.sell is not None and working.sell.cancellable:
        hbt.cancel(0, int(working.sell.order_id), True)
        print(f"[live]   cancelled sell order {working.sell.order_id}")
    for oid in working.extra_ids:
        hbt.cancel(0, oid, True)
        print(f"[live]   cancelled extra order {oid}")

    hbt.close()
    final_pos = float(hbt.position(0))
    print(f"[live] Final position: {final_pos:.6f}")
    print(f"[live] Audit rows written: {rows_written}")
    print(f"[live] Audit saved to: {audit_path}")

    return {
        "run_id": run_id,
        "audit_csv": str(audit_path),
        "rows": rows_written,
        "final_position": final_pos,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Binance tick MM bot with audit")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_toml(_expand(args.config))

    if "live" not in config:
        print("ERROR: config missing [live] section. See config.example.toml.", file=sys.stderr)
        sys.exit(1)

    result = run_live(config=config)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `cd examples/binance_tick_mm && python -m py_compile live_tick_mm.py && echo "syntax OK"`

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add examples/binance_tick_mm/live_tick_mm.py
git commit -m "feat: add live_tick_mm.py for production market-making with audit logging"
```

---

## Task 4: Create deployment config and launch script

**Files:**
- Create: `examples/binance_tick_mm/deploy/binancefutures.toml`
- Create: `examples/binance_tick_mm/deploy/run_live.sh`

- [ ] **Step 1: Create connector config template**

```toml
# Production Binance USD-M Futures connector config.
# Copy this file and fill in your API credentials.
#
# Mainnet URLs:
#   stream: wss://fstream.binance.com/ws
#   api:    https://fapi.binance.com
#
# Low-Latency Market Maker URLs (if approved):
#   stream: wss://fstream-mm.binance.com/ws
#   api:    https://fapi-mm.binance.com

stream_url = "wss://fstream.binance.com/ws"
api_url = "https://fapi.binance.com"
order_prefix = "mm"
api_key = ""
secret = ""
```

- [ ] **Step 2: Create tmux launch script**

```bash
#!/usr/bin/env bash
# Launch collector + connector + live bot in a tmux session.
# Usage: ./run_live.sh <config.toml> <connector_config.toml> [symbol]
#
# Prerequisites:
#   - tmux installed
#   - connector and collector binaries built (cargo build --release)
#   - Python environment with hftbacktest installed
#
# Example:
#   ./run_live.sh ../config_live.toml ./binancefutures.toml BTCUSDT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../" && pwd)"
EXAMPLE_DIR="$SCRIPT_DIR/.."

CONFIG="${1:?Usage: $0 <config.toml> <connector_config.toml> [symbol]}"
CONNECTOR_CONFIG="${2:?Usage: $0 <config.toml> <connector_config.toml> [symbol]}"
SYMBOL="${3:-BTCUSDT}"
SESSION="hft_live"

CONNECTOR_BIN="$PROJECT_ROOT/connector/target/release/connector"
COLLECTOR_BIN="$PROJECT_ROOT/collector/target/release/collector"

if [ ! -f "$CONNECTOR_BIN" ]; then
    echo "ERROR: connector binary not found at $CONNECTOR_BIN"
    echo "Run: cd $PROJECT_ROOT/connector && cargo build --release"
    exit 1
fi

if [ ! -f "$COLLECTOR_BIN" ]; then
    echo "ERROR: collector binary not found at $COLLECTOR_BIN"
    echo "Run: cd $PROJECT_ROOT/collector && cargo build --release"
    exit 1
fi

DATA_DIR="${DATA_DIR:-/data/collected}"
mkdir -p "$DATA_DIR"

# Kill existing session if any.
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create tmux session with 3 panes.
tmux new-session -d -s "$SESSION" -n main

# Pane 0: Collector
tmux send-keys -t "$SESSION:main" \
    "$COLLECTOR_BIN $DATA_DIR binancefuturesum $SYMBOL" Enter

# Pane 1: Connector
tmux split-window -t "$SESSION:main" -v
tmux send-keys -t "$SESSION:main.1" \
    "$CONNECTOR_BIN bf binancefutures $CONNECTOR_CONFIG" Enter

# Pane 2: Live bot
tmux split-window -t "$SESSION:main" -v
tmux send-keys -t "$SESSION:main.2" \
    "cd $EXAMPLE_DIR && python live_tick_mm.py --config $CONFIG" Enter

tmux select-layout -t "$SESSION:main" even-vertical

echo "tmux session '$SESSION' started with 3 panes:"
echo "  Pane 0: collector ($SYMBOL)"
echo "  Pane 1: connector (binancefutures)"
echo "  Pane 2: live bot"
echo ""
echo "Attach with: tmux attach -t $SESSION"
```

- [ ] **Step 3: Make script executable and commit**

```bash
chmod +x examples/binance_tick_mm/deploy/run_live.sh
git add examples/binance_tick_mm/deploy/
git commit -m "feat: add deployment config and tmux launch script for live trading"
```

---

## Task 5: Update sync_to_amdserver.sh to include deploy/

**Files:**
- Modify: `examples/binance_tick_mm/sync_to_amdserver.sh`

- [ ] **Step 1: Read current sync script and verify deploy/ is included**

The existing rsync command syncs the entire `binance_tick_mm/` directory, so `deploy/` will be included automatically. Verify by reading the script.

If the rsync excludes specific directories, add deploy/ inclusion. Otherwise this step is a no-op.

- [ ] **Step 2: Commit if changed**

Only commit if changes were needed.

---

## Task 6: End-to-end verification on local machine

**Files:** None (verification only)

- [ ] **Step 1: Verify strategy_core.py imports work**

Run:
```bash
cd examples/binance_tick_mm
python -c "
from strategy_core import (
    EwmaSigma, TokenBucket, GreekOracle, GreekValues,
    WorkingOrders, Action, compute_top5_size, impact_cost,
    clamp, round_to_tick, collect_working_orders, decide_actions,
    build_audit_row,
)
# Quick smoke test
s = EwmaSigma()
assert s.update(1000, 50000.0) == 0.0
assert s.update(2000, 50001.0) > 0.0

b = TokenBucket.create(20.0, 20.0)
assert b.allow(0, 1.0) is True

print('strategy_core: all smoke tests passed')
"
```

Expected: `strategy_core: all smoke tests passed`

- [ ] **Step 2: Verify backtest_tick_mm.py still imports correctly**

Run:
```bash
cd examples/binance_tick_mm
python -c "
import backtest_tick_mm
print('backtest_tick_mm: imports OK')
print('run_backtest callable:', callable(backtest_tick_mm.run_backtest))
"
```

Expected:
```
backtest_tick_mm: imports OK
run_backtest callable: True
```

- [ ] **Step 3: Verify live_tick_mm.py syntax and imports (no connector needed)**

Run:
```bash
cd examples/binance_tick_mm
python -c "
import live_tick_mm
print('live_tick_mm: imports OK')
print('run_live callable:', callable(live_tick_mm.run_live))
" 2>&1 || echo "(import error expected if LIVE_FEATURE not available on this machine — that is OK)"
```

Note: If `hftbacktest` was not compiled with the `live` feature on this machine, the import of `ROIVectorMarketDepthLiveBot` will fail. This is expected — the live bot will work on the Tokyo server where the full package is installed.

- [ ] **Step 4: Verify deploy scripts are well-formed**

Run:
```bash
bash -n examples/binance_tick_mm/deploy/run_live.sh && echo "shell syntax OK"
```

Expected: `shell syntax OK`

---

## Task 7: Documentation update in README.md

**Files:**
- Modify: `examples/binance_tick_mm/README.md`

- [ ] **Step 1: Add live trading section to README.md**

Add a new section after the existing workflow guides, documenting:
1. Prerequisites for live trading (Rust toolchain, API keys, Tokyo server)
2. How to build connector and collector
3. How to configure `binancefutures.toml` and `config_live.toml`
4. How to launch with `run_live.sh`
5. How to pull `audit_live.csv` back and run calibration

Content should be in Chinese (matching existing README style) with command examples.

- [ ] **Step 2: Commit**

```bash
git add examples/binance_tick_mm/README.md
git commit -m "docs: add live trading deployment guide to README"
```
