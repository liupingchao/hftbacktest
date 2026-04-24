#!/usr/bin/env python3
"""Live Binance Futures BTCUSDT tick market-making engine.

Mirrors the backtest logic in backtest_tick_mm.py but uses
ROIVectorMarketDepthLiveBot connected to a Rust connector process
via iceoryx2 shared memory IPC.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import signal
import sys
import time
import tomllib
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("live_tick_mm")

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_shutdown = False


def _handle_signal(signum: int, _frame: Any) -> None:
    global _shutdown
    log.warning("Received signal %d, initiating graceful shutdown ...", signum)
    _shutdown = True


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Live trading loop
# ---------------------------------------------------------------------------

def run_live(config: dict[str, Any]) -> dict[str, Any]:
    global _shutdown

    # ---- Config sections ---------------------------------------------------
    symbol = str(config["symbol"]["name"])
    symbol_lower = symbol.lower()  # Binance Futures connector needs lowercase
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
    run_id_prefix = str(live_cfg.get("run_id_prefix", "live"))
    audit_csv_name = str(live_cfg.get("audit_csv", "audit_live.csv"))
    heartbeat_interval_ns = int(float(live_cfg.get("heartbeat_interval_sec", 60))) * 1_000_000_000

    tick_size = float(market["tick_size"])
    lot_size = float(market["lot_size"])

    run_id = f"{run_id_prefix}_{symbol_lower}_{int(time.time())}"
    audit_path = _expand(audit_csv_name)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Greeks oracle (live: typically position-as-delta, no csv) ----------
    greek_oracle = GreekOracle.from_config(greek_cfg, expand_path=_expand)

    # ---- Build live instrument + bot ---------------------------------------
    instrument = (
        LiveInstrument()
        .connector(connector_name)
        .symbol(symbol_lower)
        .tick_size(tick_size)
        .lot_size(lot_size)
        .roi_lb(roi_lb)
        .roi_ub(roi_ub)
    )

    hbt = ROIVectorMarketDepthLiveBot([instrument])
    log.info(
        "Live bot created: connector=%s symbol=%s roi=[%.1f, %.1f] run_id=%s",
        connector_name, symbol_lower, roi_lb, roi_ub, run_id,
    )

    # ---- Strategy state ----------------------------------------------------
    sigma_est = EwmaSigma()
    bucket = TokenBucket.create(float(api_cfg["capacity"]), float(api_cfg["refill_per_sec"]))
    min_interval_ns = int(float(api_cfg["min_interval_ms"]) * 1_000_000)
    latency_guard_ns = int(float(latency_cfg["latency_guard_ms"]) * 1_000_000)

    next_order_id = 1
    strategy_seq = 0
    last_api_ts: int | None = None
    last_heartbeat_ts: int = 0

    # 1-second timeout so we can check the shutdown flag periodically
    wait_timeout_ns = 1_000_000_000

    rows_written = 0
    position_before_close = 0.0

    try:
        with audit_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
            writer.writeheader()

            while not _shutdown:
                rc = hbt.wait_next_feed(True, wait_timeout_ns)
                if rc == 1:
                    # End-of-feed (connector disconnected)
                    log.warning("wait_next_feed returned 1 (end of feed), shutting down")
                    break
                if rc == 0:
                    # Timeout, no new data -- loop back to check _shutdown
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
                pos_limit = abs(position_notional) > float(risk["max_notional_pos"])

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

                # ---- Latency (live: real observed, no prediction) ----------
                feed_lat = hbt.feed_latency(0)
                order_lat = hbt.order_latency(0)
                feed_latency_ns = int(feed_lat[1] - feed_lat[0]) if feed_lat is not None else 0

                last_entry_ns = int(order_lat[1] - order_lat[0]) if order_lat is not None else 0
                last_resp_ns = int(order_lat[2] - order_lat[1]) if order_lat is not None else 0
                # Live gating uses feed latency only. hbt.order_latency() can report
                # delayed or mismatched REST/WebSocket order timestamps, which are
                # useful for audit but not safe as a forward-looking guard signal.
                latency_signal_ns = feed_latency_ns

                dropped_by_latency = latency_signal_ns > latency_guard_ns
                dropped_by_api_limit = False

                working = collect_working_orders(hbt.orders(0))
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
                inventory_score = max(0.0, 1.0 - abs(position_notional) / float(risk["max_notional_pos"]))

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
                    action_order_id=action_order_id,
                    action_name=action_name,
                    reject_reason=reject_reason,
                    req_ts=req_ts,
                    exch_ts=exch_ts,
                    resp_ts=resp_ts,
                    entry_latency_ns=entry_latency_ns,
                    resp_latency_ns=resp_latency_ns,
                    predicted_entry_ns=0,  # Live: no prediction
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

                # Flush audit periodically
                if rows_written % 100 == 0:
                    f.flush()

                hbt.clear_inactive_orders(ALL_ASSETS)

                # ---- Heartbeat logging ------------------------------------
                if ts_local - last_heartbeat_ts >= heartbeat_interval_ns:
                    last_heartbeat_ts = ts_local
                    log.info(
                        "HEARTBEAT seq=%d mid=%.2f pos=%.4f pos_notional=%.2f "
                        "spread_bps=%.2f vol_bps=%.2f feed_lat_ms=%.2f rows=%d",
                        strategy_seq,
                        mid,
                        position,
                        position_notional,
                        spread_bps,
                        vol_bps,
                        feed_latency_ns / 1_000_000.0,
                        rows_written,
                    )

            # Final flush
            f.flush()

        log.info("Event loop exited. rows_written=%d", rows_written)

    finally:
        # ---- Graceful shutdown: cancel all open orders --------------------
        log.info("Cancelling all open orders ...")
        try:
            working_final = collect_working_orders(hbt.orders(0))
            cancelled = 0
            if working_final.buy is not None and working_final.buy.cancellable:
                hbt.cancel(0, int(working_final.buy.order_id), False)
                cancelled += 1
            if working_final.sell is not None and working_final.sell.cancellable:
                hbt.cancel(0, int(working_final.sell.order_id), False)
                cancelled += 1
            for oid in working_final.extra_ids:
                hbt.cancel(0, oid, False)
                cancelled += 1
            log.info("Cancelled %d orders", cancelled)
        except Exception:
            log.exception("Error cancelling orders during shutdown")

        # Read position before closing
        try:
            position_before_close = float(hbt.position(0))
            log.info("Final position: %.6f", position_before_close)
        except Exception:
            log.exception("Error reading final position")

        # Close bot
        try:
            hbt.close()
            log.info("Bot closed")
        except Exception:
            log.exception("Error closing bot")

    return {
        "run_id": run_id,
        "audit_csv": str(audit_path),
        "rows": rows_written,
        "final_position": position_before_close,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Binance tick MM engine")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    return parser.parse_args()


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    args = parse_args()
    config = _load_toml(_expand(args.config))

    log.info("Starting live tick MM engine ...")
    result = run_live(config)
    log.info("Finished: %s", result)


if __name__ == "__main__":
    main()
