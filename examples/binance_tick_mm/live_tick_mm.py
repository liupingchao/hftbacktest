#!/usr/bin/env python3
"""Live Binance Futures BTCUSDT tick market-making engine.

Mirrors the backtest logic in backtest_tick_mm.py but uses
ROIVectorMarketDepthLiveBot connected to a Rust connector process
via iceoryx2 shared memory IPC.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import hmac
import json
import logging
import math
import os
import signal
import sys
import time
import tomllib
import urllib.parse
import urllib.request
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
    OrderLifecycleTracker,
    WorkingOrders,
    LiveSafetyConfig,
    LiveSafetyState,
    evaluate_live_safety,
    QuoteThrottleConfig,
    QuoteThrottleState,
    compute_top5_size,
    impact_cost,
    clamp,
    round_to_tick,
    collect_working_orders,
    decide_actions,
    format_actions,
    format_rest_open_orders,
    format_working_order_diagnostics,
    inventory_score_from_risk,
    is_position_limit_reached,
    is_pure_cancel_extra,
    open_order_diff,
    should_throttle_quote_update,
    update_quote_throttle_state,
    build_audit_row,
    build_lifecycle_event_row,
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
# REST safety helpers
# ---------------------------------------------------------------------------

class BinanceFuturesRestClient:
    def __init__(self, config_path: str) -> None:
        path = Path(config_path).expanduser().resolve()
        cfg = tomllib.loads(path.read_text())
        self.api_url = str(cfg.get("api_url", "https://fapi.binance.com")).rstrip("/")
        self.api_key = str(cfg.get("api_key", ""))
        self.secret = str(cfg.get("secret", ""))
        if not self.api_key or not self.secret:
            raise ValueError("Binance REST credentials are missing")

    def _signed_get(self, path: str, params: dict[str, Any]) -> Any:
        params = dict(params)
        params["timestamp"] = int(time.time() * 1000)
        params["recvWindow"] = 5000
        query = urllib.parse.urlencode(params)
        signature = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        url = f"{self.api_url}{path}?{query}&signature={signature}"
        request = urllib.request.Request(url, headers={"X-MBX-APIKEY": self.api_key})
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode())

    def position(self, symbol: str) -> float:
        rows = self._signed_get("/fapi/v2/positionRisk", {"symbol": symbol.upper()})
        for row in rows:
            if str(row.get("symbol", "")).upper() == symbol.upper():
                return float(row.get("positionAmt", 0.0))
        return 0.0

    def open_order_count(self, symbol: str) -> int:
        rows = self._signed_get("/fapi/v1/openOrders", {"symbol": symbol.upper()})
        return len(rows)

    def open_orders(self, symbol: str) -> list[dict[str, Any]]:
        rows = self._signed_get("/fapi/v1/openOrders", {"symbol": symbol.upper()})
        if not isinstance(rows, list):
            return []
        return [row for row in rows if isinstance(row, dict)]


def _local_open_order_count(working: WorkingOrders) -> int:
    count = 0
    if working.buy is not None:
        count += 1
    if working.sell is not None:
        count += 1
    count += len(working.extras)
    return count


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
    throttle_cfg = QuoteThrottleConfig.from_config(config.get("strategy", {}))
    throttle_state = QuoteThrottleState()
    strategy_cfg = config.get("strategy", {})
    two_phase_replace_enabled = bool(strategy_cfg.get("two_phase_replace_enabled", False))
    safety_cfg = LiveSafetyConfig.from_config(config.get("live_safety", {}))
    rest_client = None
    safety_state = LiveSafetyState(safety_status="safety_disabled")
    next_safety_check_ns = 0
    open_order_mismatch_count = 0
    position_mismatch_count = 0
    if safety_cfg.enabled:
        connector_config = safety_cfg.connector_config or str(live_cfg.get("connector_config", ""))
        if not connector_config:
            raise ValueError("live_safety.connector_config must be set when live safety is enabled")
        rest_client = BinanceFuturesRestClient(connector_config)

    next_order_id = 1
    strategy_seq = 0
    last_api_ts: int | None = None
    last_heartbeat_ts: int = 0
    lifecycle_tracker = OrderLifecycleTracker.create()
    lifecycle_event_seq = 0

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

                local_position = float(hbt.position(0))
                position = local_position
                safety_checked = False

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
                working_diagnostics = format_working_order_diagnostics(working)

                if safety_cfg.enabled and rest_client is not None and ts_local >= next_safety_check_ns:
                    rest_error = ""
                    rest_position = 0.0
                    rest_open_order_count = 0
                    rest_open_orders = ""
                    local_open_orders = working_diagnostics["local_open_orders"]
                    try:
                        rest_position = rest_client.position(symbol)
                        rest_open_order_rows = rest_client.open_orders(symbol)
                        rest_open_order_count = len(rest_open_order_rows)
                        rest_open_orders = format_rest_open_orders(rest_open_order_rows, tick_size=tick_size)
                    except Exception as exc:
                        rest_error = str(exc)
                    open_order_diff_value = open_order_diff(local_open_orders, rest_open_orders)
                    safety_state = evaluate_live_safety(
                        cfg=safety_cfg,
                        rest_position=rest_position,
                        local_position=local_position,
                        rest_open_order_count=rest_open_order_count,
                        local_open_order_count=_local_open_order_count(working),
                        rest_error=rest_error,
                        rest_open_orders=rest_open_orders,
                        local_open_orders=local_open_orders,
                        open_order_diff=open_order_diff_value,
                        ts_local=ts_local,
                        last_api_ts=last_api_ts,
                        open_order_mismatch_count=open_order_mismatch_count,
                        position_mismatch_count=position_mismatch_count,
                    )
                    safety_checked = rest_error == ""
                    if safety_state.safety_status in {"open_order_mismatch_pending", "open_order_mismatch"}:
                        open_order_mismatch_count += 1
                    elif safety_state.safety_status != "open_order_grace":
                        open_order_mismatch_count = 0
                    if safety_state.safety_status in {"position_mismatch_pending", "position_mismatch"}:
                        position_mismatch_count += 1
                    else:
                        position_mismatch_count = 0
                    next_safety_check_ns = ts_local + int(safety_cfg.rest_check_interval_sec * 1_000_000_000)
                    lifecycle_event_seq += 1
                    writer.writerow(
                        build_lifecycle_event_row(
                            run_id=run_id,
                            symbol=symbol,
                            strategy_seq=strategy_seq,
                            event_seq=lifecycle_event_seq,
                            event_type="safety_check",
                            event_source="rest",
                            ts_local=ts_local,
                            ts_exch=int(feed_lat[0]) if feed_lat is not None else 0,
                            best_bid=best_bid,
                            best_ask=best_ask,
                            mid=mid,
                            position=local_position,
                            local_open_orders=safety_state.local_open_orders,
                            rest_open_orders=safety_state.rest_open_orders,
                            open_order_diff=safety_state.open_order_diff,
                            rest_position=safety_state.rest_position,
                            position_mismatch=safety_state.position_mismatch,
                            rest_open_order_count=safety_state.rest_open_order_count,
                            local_open_order_count=safety_state.local_open_order_count,
                            safety_status=safety_state.safety_status,
                            safety_detail=safety_state.safety_detail,
                            local_order_seen=bool(safety_state.local_open_order_count),
                            rest_order_seen=bool(safety_state.rest_open_order_count),
                            lifecycle_detail=rest_error or safety_state.open_order_diff,
                        )
                    )
                    rows_written += 1
                    if safety_cfg.fail_on_mismatch and safety_state.safety_status not in {
                        "ok",
                        "safety_disabled",
                        "open_order_grace",
                        "open_order_mismatch_pending",
                        "position_mismatch_pending",
                    }:
                        log.critical(
                            "Live safety mismatch: status=%s rest_position=%.6f local_position=%.6f mismatch=%.6f rest_open_orders=%d local_open_orders=%d diff=%s local_detail=%s rest_detail=%s",
                            safety_state.safety_status,
                            safety_state.rest_position,
                            local_position,
                            safety_state.position_mismatch,
                            safety_state.rest_open_order_count,
                            safety_state.local_open_order_count,
                            safety_state.open_order_diff,
                            safety_state.local_open_orders,
                            safety_state.rest_open_orders,
                        )
                        break

                if safety_cfg.use_rest_position_for_strategy and safety_checked:
                    position = safety_state.rest_position

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
                pos_limit = is_position_limit_reached(position=position, position_notional=position_notional, risk=risk)

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

                planned_actions: list[Action] = []
                executed_actions: list[Action] = []
                reject_reason = ""
                action_order_id = ""
                action_name = "keep"
                planned_order_id = ""
                planned_action = "keep"
                throttle_reason = ""
                sent_api = False

                if dropped_by_latency:
                    reject_reason = "latency_guard"
                elif safety_cfg.position_mismatch_pause_trading and safety_state.safety_status == "position_mismatch_pending":
                    dropped_by_api_limit = True
                    reject_reason = "safety_pause"
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
                        two_phase_replace_enabled=two_phase_replace_enabled,
                    )

                    if planned_actions:
                        planned_order_id, planned_action = format_actions(planned_actions)
                        throttle_reason = should_throttle_quote_update(
                            cfg=throttle_cfg,
                            state=throttle_state,
                            ts_local=ts_local,
                            target_bid_tick=target_bid_tick,
                            target_ask_tick=target_ask_tick,
                            planned_actions=planned_actions,
                            pos_limit=pos_limit,
                        )
                        if throttle_reason:
                            dropped_by_api_limit = True
                            reject_reason = "quote_throttle"
                        elif (
                            last_api_ts is not None
                            and (ts_local - last_api_ts) < min_interval_ns
                            and not is_pure_cancel_extra(planned_actions)
                        ):
                            dropped_by_api_limit = True
                            reject_reason = "api_interval_guard"
                            throttle_reason = "api_interval"
                        else:
                            for action in planned_actions:
                                if bool(api_cfg.get("enabled", True)) and not bucket.allow(ts_local, 1.0):
                                    dropped_by_api_limit = True
                                    reject_reason = "token_bucket"
                                    break

                                if action.kind == "cancel":
                                    lifecycle_tracker.mark_cancel_requested(action.order_id, ts_local)
                                    hbt.cancel(0, int(action.order_id), False)
                                elif action.kind == "submit" and action.side == "buy":
                                    hbt.submit_buy_order(0, int(action.order_id), action.price, action.qty, GTX, LIMIT, False)
                                elif action.kind == "submit" and action.side == "sell":
                                    hbt.submit_sell_order(0, int(action.order_id), action.price, action.qty, GTX, LIMIT, False)

                                executed_actions.append(action)
                                sent_api = True
                                last_api_ts = ts_local
                                lifecycle_event_seq += 1
                                writer.writerow(
                                    build_lifecycle_event_row(
                                        run_id=run_id,
                                        symbol=symbol,
                                        strategy_seq=strategy_seq,
                                        event_seq=lifecycle_event_seq,
                                        event_type="cancel_sent" if action.kind == "cancel" else "order_submit_sent",
                                        event_source="live_local",
                                        ts_local=ts_local,
                                        ts_exch=int(feed_lat[0]) if feed_lat is not None else 0,
                                        action=action,
                                        action_order_price_tick=round_to_tick(action.price, tick_size) if action.price > 0.0 else "",
                                        best_bid=best_bid,
                                        best_ask=best_ask,
                                        mid=mid,
                                        position=position,
                                        cancel_requested=action.kind == "cancel",
                                        cancel_request_ts=ts_local if action.kind == "cancel" else 0,
                                        local_order_seen=True,
                                        lifecycle_detail="api_action_sent",
                                    )
                                )
                                rows_written += 1

                            if executed_actions:
                                action_order_id, action_name = format_actions(executed_actions)
                                update_quote_throttle_state(
                                    throttle_state,
                                    ts_local=ts_local,
                                    target_bid_tick=target_bid_tick,
                                    target_ask_tick=target_ask_tick,
                                    actions=executed_actions,
                                )
                            elif not reject_reason:
                                dropped_by_api_limit = True
                                reject_reason = "api_limit"

                order_lat_after = hbt.order_latency(0)
                req_ts = int(order_lat_after[0]) if order_lat_after is not None else 0
                exch_ts = int(order_lat_after[1]) if order_lat_after is not None else 0
                resp_ts = int(order_lat_after[2]) if order_lat_after is not None else 0
                entry_latency_ns = int(exch_ts - req_ts) if exch_ts > 0 and req_ts > 0 else 0
                resp_latency_ns = int(resp_ts - exch_ts) if exch_ts > 0 and resp_ts > exch_ts else 0

                if sent_api and req_ts > 0:
                    auditlatency_ms = max(0.0, (req_ts - ts_local) / 1_000_000.0)
                else:
                    auditlatency_ms = 0.0

                spread_bps = (spread / mid) * 1e4 if mid > 0 else 0.0
                vol_bps = sigma * 1e4
                inventory_score = inventory_score_from_risk(position=position, position_notional=position_notional, risk=risk)

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
                    planned_order_id=planned_order_id,
                    planned_action=planned_action,
                    throttle_reason=throttle_reason,
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
                    working_buy_order_id=working_diagnostics["working_buy_order_id"],
                    working_sell_order_id=working_diagnostics["working_sell_order_id"],
                    working_bid_qty=working_diagnostics["working_bid_qty"],
                    working_ask_qty=working_diagnostics["working_ask_qty"],
                    working_bid_status=working_diagnostics["working_bid_status"],
                    working_ask_status=working_diagnostics["working_ask_status"],
                    working_bid_req=working_diagnostics["working_bid_req"],
                    working_ask_req=working_diagnostics["working_ask_req"],
                    working_bid_pending_cancel=working_diagnostics["working_bid_pending_cancel"],
                    working_ask_pending_cancel=working_diagnostics["working_ask_pending_cancel"],
                    extra_order_ids=working_diagnostics["extra_order_ids"],
                    extra_order_sides=working_diagnostics["extra_order_sides"],
                    extra_order_price_ticks=working_diagnostics["extra_order_price_ticks"],
                    rest_position=safety_state.rest_position,
                    position_mismatch=safety_state.position_mismatch,
                    rest_open_order_count=safety_state.rest_open_order_count,
                    local_open_order_count=_local_open_order_count(working),
                    local_open_orders=working_diagnostics["local_open_orders"],
                    rest_open_orders=safety_state.rest_open_orders,
                    open_order_diff=safety_state.open_order_diff,
                    safety_status=safety_state.safety_status,
                    safety_detail=safety_state.safety_detail,
                )
                writer.writerow(row)
                rows_written += 1

                current_working = collect_working_orders(hbt.orders(0))
                current_order_diagnostics = format_working_order_diagnostics(current_working)
                for lifecycle_type, order_snapshot, _prev_snapshot in lifecycle_tracker.observe(hbt.orders(0)):
                    cancel_request_ts = lifecycle_tracker.cancel_request_ts(order_snapshot.order_id)
                    is_fill_event = lifecycle_type in {"fill", "partial_fill"}
                    lifecycle_event_seq += 1
                    writer.writerow(
                        build_lifecycle_event_row(
                            run_id=run_id,
                            symbol=symbol,
                            strategy_seq=strategy_seq,
                            event_seq=lifecycle_event_seq,
                            event_type=lifecycle_type,
                            event_source="live_order",
                            ts_local=ts_local,
                            ts_exch=order_snapshot.exch_timestamp,
                            order=order_snapshot,
                            linked_action=lifecycle_type,
                            linked_order_id=str(order_snapshot.order_id),
                            best_bid=best_bid,
                            best_ask=best_ask,
                            mid=mid,
                            position=position,
                            local_open_orders=current_order_diagnostics["local_open_orders"],
                            rest_open_orders=safety_state.rest_open_orders,
                            open_order_diff=safety_state.open_order_diff,
                            rest_position=safety_state.rest_position,
                            position_mismatch=safety_state.position_mismatch,
                            rest_open_order_count=safety_state.rest_open_order_count,
                            local_open_order_count=safety_state.local_open_order_count,
                            safety_status=safety_state.safety_status,
                            safety_detail=safety_state.safety_detail,
                            cancel_requested=cancel_request_ts > 0,
                            cancel_request_ts=cancel_request_ts,
                            cancel_ack_ts=ts_local if lifecycle_type == "cancel_ack" else 0,
                            fill_ts=order_snapshot.exch_timestamp if is_fill_event else 0,
                            fill_qty=order_snapshot.exec_qty if is_fill_event else 0.0,
                            fill_price=(
                                order_snapshot.exec_price_tick * tick_size
                                if is_fill_event and order_snapshot.exec_price_tick != 0
                                else 0.0
                            ),
                            fill_after_cancel_request=is_fill_event and cancel_request_ts > 0,
                            local_order_seen=True,
                            ws_order_seen=True,
                            lifecycle_detail="fill_after_cancel_request" if is_fill_event and cancel_request_ts > 0 else "",
                        )
                    )
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
