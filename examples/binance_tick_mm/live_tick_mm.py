#!/usr/bin/env python3
"""Live Binance Futures BTCUSDT tick market-making engine.

Mirrors the backtest logic in backtest_tick_mm.py but uses
ROIVectorMarketDepthLiveBot connected to a Rust connector process
via iceoryx2 shared memory IPC.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
    CANCELED,
    EXPIRED,
    FILLED,
    GTX,
    LIMIT,
    LiveInstrument,
    NEW,
)
from hftbacktest.order import PARTIALLY_FILLED, REJECTED
try:
    from hftbacktest import ROIVectorMarketDepthLiveBot
except ImportError:  # pragma: no cover - depends on live-extension build availability.
    ROIVectorMarketDepthLiveBot = None

from audit_schema import AUDIT_FIELDS

from strategy_core import (
    add_side_soft_limit_qty_from_risk,
    build_market_view_from_depth,
    EwmaSigma,
    InFlightExposureTracker,
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
    order_side_name,
    should_throttle_quote_update,
    update_quote_throttle_state,
    build_audit_row,
    build_lifecycle_event_row,
    build_quote_update_audit_fields,
    add_side_toxic_timing_guard_side_blocks,
    adverse_timing_guard_side_blocks,
    cancel_race_guard_side_blocks,
    working_side_leaves_qty,
)
from quote_anchor_safety import QuoteAnchorSafetyConfig, apply_quote_anchor_safety

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


SHUTDOWN_CANCEL_ACK_TIMEOUT_NS = 5_000_000_000
SHUTDOWN_WAIT_OUTCOMES = frozenset(
    {
        "order_response_received",
        "ok_unknown_or_timeout",
        "wait_error",
        "not_requested",
    }
)
SHUTDOWN_TERMINAL_CONFIRMATION_SOURCES = frozenset(
    {
        "local_orders",
        "rest_open_orders",
        "none",
    }
)
SHUTDOWN_ACTIVE_LOCAL_ORDER_STATUSES = frozenset({NEW, PARTIALLY_FILLED})
SHUTDOWN_TERMINAL_LOCAL_ORDER_STATUSES = frozenset({EXPIRED, FILLED, CANCELED, REJECTED})
SHUTDOWN_LOCAL_ORDER_STATUS_NAMES = {
    NEW: "new",
    EXPIRED: "expired",
    FILLED: "filled",
    CANCELED: "canceled",
    PARTIALLY_FILLED: "partially_filled",
    REJECTED: "rejected",
}


@dataclass
class ShutdownCancelResult:
    order_id: int
    source: str
    side: str
    cancel_sent: bool
    wait_requested: bool
    wait_result_raw: int | None = None
    order_response_received: bool = False
    wait_outcome: str = "not_requested"
    terminal_confirmed: bool = False
    terminal_confirmation_source: str = "none"
    final_order_status: str = "not_checked"
    error: str = ""

    @property
    def wait_result(self) -> int | None:
        return self.wait_result_raw


def _iter_local_shutdown_orders(order_dict: Any) -> list[Any]:
    orders: list[Any] = []
    values = order_dict.values()
    while values.has_next():
        orders.append(values.get())
    return orders


def _local_shutdown_order_status_name(status: int) -> str:
    return SHUTDOWN_LOCAL_ORDER_STATUS_NAMES.get(int(status), f"unknown:{int(status)}")


def _confirm_shutdown_terminal_state_from_local_orders(
    hbt: Any,
    order_id: int,
    asset_no: int,
) -> tuple[bool, str, str]:
    try:
        local_orders = _iter_local_shutdown_orders(hbt.orders(asset_no))
    except Exception as exc:
        return False, "none", f"local_orders_unavailable:{type(exc).__name__}:{exc}"

    for order in local_orders:
        if int(getattr(order, "order_id", -1)) != order_id:
            continue
        try:
            status = int(order.status)
        except (AttributeError, TypeError, ValueError) as exc:
            return False, "local_orders", f"local_status_unavailable:{type(exc).__name__}:{exc}"
        status_name = _local_shutdown_order_status_name(status)
        if status in SHUTDOWN_ACTIVE_LOCAL_ORDER_STATUSES:
            return False, "local_orders", f"local_active_order:{status_name}"
        if status in SHUTDOWN_TERMINAL_LOCAL_ORDER_STATUSES:
            return True, "local_orders", f"local_terminal_order:{status_name}"
        return False, "local_orders", f"local_unknown_order_status:{status_name}"
    return True, "local_orders", "local_absent_from_local_orders"


def _classify_shutdown_wait_result(result: ShutdownCancelResult) -> None:
    if not result.wait_requested:
        result.wait_outcome = "not_requested"
        result.order_response_received = False
        return
    if result.error.startswith("wait_error:"):
        result.wait_outcome = "wait_error"
        result.order_response_received = False
        return
    if result.wait_result_raw == 3:
        result.wait_outcome = "order_response_received"
        result.order_response_received = True
        return
    result.wait_outcome = "ok_unknown_or_timeout"
    result.order_response_received = False


def cancel_working_orders_for_shutdown(
    hbt: Any,
    working: WorkingOrders,
    *,
    asset_no: int = 0,
    wait_timeout_ns: int = SHUTDOWN_CANCEL_ACK_TIMEOUT_NS,
) -> list[ShutdownCancelResult]:
    """Cancel shutdown candidates and wait for bounded exchange acknowledgement."""
    candidates: list[tuple[int, str, str]] = []
    if working.buy is not None and bool(getattr(working.buy, "cancellable", False)):
        candidates.append((int(working.buy.order_id), "primary_buy", "buy"))
    if working.sell is not None and bool(getattr(working.sell, "cancellable", False)):
        candidates.append((int(working.sell.order_id), "primary_sell", "sell"))
    for extra in working.extras:
        if extra.cancellable and extra.req != "cancel":
            candidates.append((int(extra.order_id), f"extra_{extra.side}", extra.side))

    results: list[ShutdownCancelResult] = []
    for order_id, source, side in candidates:
        result = ShutdownCancelResult(
            order_id=order_id,
            source=source,
            side=side,
            cancel_sent=False,
            wait_requested=False,
        )
        try:
            hbt.cancel(asset_no, order_id, False)
            result.cancel_sent = True
        except Exception as exc:
            result.error = f"cancel_error:{type(exc).__name__}:{exc}"
            results.append(result)
            continue

        try:
            result.wait_requested = True
            result.wait_result_raw = hbt.wait_order_response(asset_no, order_id, wait_timeout_ns)
        except Exception as exc:
            result.error = f"wait_error:{type(exc).__name__}:{exc}"
        _classify_shutdown_wait_result(result)
        (
            result.terminal_confirmed,
            result.terminal_confirmation_source,
            result.final_order_status,
        ) = _confirm_shutdown_terminal_state_from_local_orders(hbt, order_id, asset_no)
        results.append(result)

    return results


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

    if ROIVectorMarketDepthLiveBot is None:
        raise ImportError("ROIVectorMarketDepthLiveBot is unavailable in this hftbacktest build")
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
    quote_anchor_safety_cfg = QuoteAnchorSafetyConfig.from_config(
        config.get("quote_anchor_safety") or strategy_cfg.get("quote_anchor_safety")
    )
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
    inflight_exposure_enabled = bool(risk.get("inventory_inflight_exposure_enabled", False))
    inflight_exposure = InFlightExposureTracker.create()
    add_side_cancel_cooldown_ns = int(float(risk.get("inventory_add_side_cancel_cooldown_ms", 0.0)) * 1_000_000)
    cancel_race_guard_enabled = bool(risk.get("cancel_race_guard_enabled", False))
    cancel_race_guard_pending_cancel_block = bool(risk.get("cancel_race_guard_pending_cancel_block", True))
    cancel_race_guard_post_fill_cooldown_ns = int(
        float(risk.get("cancel_race_guard_post_fill_cooldown_ms", 0.0)) * 1_000_000
    )
    adverse_timing_guard_enabled = bool(risk.get("adverse_timing_guard_enabled", False))
    adverse_timing_guard_target_deterioration_enabled = bool(
        risk.get("adverse_timing_guard_target_deterioration_enabled", True)
    )
    adverse_timing_guard_pending_cancel_enabled = bool(
        risk.get("adverse_timing_guard_pending_cancel_enabled", True)
    )
    adverse_timing_guard_post_cancel_fill_enabled = bool(
        risk.get("adverse_timing_guard_post_cancel_fill_enabled", True)
    )
    adverse_timing_guard_cooldown_ns = int(
        float(risk.get("adverse_timing_guard_cooldown_ms", 100.0)) * 1_000_000
    )
    adverse_timing_guard_min_target_move_ticks = int(
        risk.get("adverse_timing_guard_min_target_move_ticks", 2)
    )
    adverse_timing_guard_block_mode = str(risk.get("adverse_timing_guard_block_mode", "add_side_only"))
    if adverse_timing_guard_enabled and adverse_timing_guard_block_mode != "add_side_only":
        raise ValueError("risk.adverse_timing_guard_block_mode must be 'add_side_only'")
    add_side_toxic_timing_guard_enabled = bool(risk.get("add_side_toxic_timing_guard_enabled", False))
    add_side_toxic_timing_guard_window_ns = int(
        float(risk.get("add_side_toxic_timing_guard_window_ms", 100.0)) * 1_000_000
    )
    add_side_toxic_timing_guard_min_target_move_ticks = int(
        risk.get("add_side_toxic_timing_guard_min_target_move_ticks", 2)
    )
    add_side_toxic_timing_guard_latency_threshold_ns = int(
        float(risk.get("add_side_toxic_timing_guard_latency_threshold_ms", 0.0)) * 1_000_000
    )
    add_side_toxic_timing_guard_pending_cancel_enabled = bool(
        risk.get("add_side_toxic_timing_guard_pending_cancel_enabled", True)
    )
    add_side_toxic_timing_guard_post_cancel_fill_enabled = bool(
        risk.get("add_side_toxic_timing_guard_post_cancel_fill_enabled", True)
    )
    add_side_toxic_timing_guard_target_move_enabled = bool(
        risk.get("add_side_toxic_timing_guard_target_move_enabled", True)
    )
    add_side_toxic_timing_guard_block_mode = str(
        risk.get("add_side_toxic_timing_guard_block_mode", "add_side_submit_only")
    )
    if add_side_toxic_timing_guard_enabled and add_side_toxic_timing_guard_block_mode != "add_side_submit_only":
        raise ValueError("risk.add_side_toxic_timing_guard_block_mode must be 'add_side_submit_only'")
    last_buy_cancel_ts: int | None = None
    last_sell_cancel_ts: int | None = None
    last_buy_cancel_fill_ts: int | None = None
    last_sell_cancel_fill_ts: int | None = None
    last_quote_or_cancel_target_bid_tick: int | None = None
    last_quote_or_cancel_target_ask_tick: int | None = None

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
                feed_lat = hbt.feed_latency(0)
                feed_latency_ns = int(feed_lat[1] - feed_lat[0]) if feed_lat is not None else 0
                market_view = build_market_view_from_depth(
                    depth,
                    source="live_depth",
                    ts_local=ts_local,
                    ts_exch=int(feed_lat[0]) if feed_lat is not None else 0,
                    feed_latency_ns=feed_latency_ns,
                )
                best_bid = market_view.best_bid
                best_ask = market_view.best_ask

                if not (math.isfinite(best_bid) and math.isfinite(best_ask)):
                    continue
                if best_bid <= 0.0 or best_ask <= 0.0 or best_ask <= best_bid:
                    continue

                strategy_seq += 1

                spread = market_view.spread
                mid = market_view.mid

                sigma = sigma_est.update(ts_local, mid)
                bid_size = market_view.bid_size
                ask_size = market_view.ask_size
                bid_top5_ticks = market_view.bid_top5_ticks
                bid_top5_qtys = market_view.bid_top5_qtys
                ask_top5_ticks = market_view.ask_top5_ticks
                ask_top5_qtys = market_view.ask_top5_qtys

                local_position = float(hbt.position(0))
                position = local_position
                safety_checked = False

                # ---- Latency (live: real observed, no prediction) ----------
                order_lat = hbt.order_latency(0)

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
                quote_anchor_safety = apply_quote_anchor_safety(
                    cfg=quote_anchor_safety_cfg,
                    target_bid_tick=target_bid_tick,
                    target_ask_tick=target_ask_tick,
                    target_bid_price=target_bid,
                    target_ask_price=target_ask,
                    tick_size=tick_size,
                    fast_bid_tick=None,
                    fast_ask_tick=None,
                    fast_anchor_age_ms=None,
                    depth_bid_tick=market_view.best_bid_tick,
                    depth_ask_tick=market_view.best_ask_tick,
                    depth_anchor_age_ms=market_view.stale_ms,
                )
                target_bid_tick = int(quote_anchor_safety.safe_bid_tick or target_bid_tick)
                target_ask_tick = int(quote_anchor_safety.safe_ask_tick or target_ask_tick)

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
                throttle_state_snapshot = throttle_state.snapshot()
                bucket_snapshot = bucket.snapshot()
                buy_cooldown_active = (
                    add_side_cancel_cooldown_ns > 0
                    and last_buy_cancel_ts is not None
                    and (ts_local - last_buy_cancel_ts) < add_side_cancel_cooldown_ns
                )
                sell_cooldown_active = (
                    add_side_cancel_cooldown_ns > 0
                    and last_sell_cancel_ts is not None
                    and (ts_local - last_sell_cancel_ts) < add_side_cancel_cooldown_ns
                )
                inflight_buy_qty = (
                    max(0.0, inflight_exposure.side_qty("buy") - working_side_leaves_qty(working, "buy"))
                    if inflight_exposure_enabled
                    else None
                )
                inflight_sell_qty = (
                    max(0.0, inflight_exposure.side_qty("sell") - working_side_leaves_qty(working, "sell"))
                    if inflight_exposure_enabled
                    else None
                )
                cancel_race_guard_buy_active, cancel_race_guard_sell_active = cancel_race_guard_side_blocks(
                    enabled=cancel_race_guard_enabled,
                    pending_cancel_block=cancel_race_guard_pending_cancel_block,
                    post_fill_cooldown_ns=cancel_race_guard_post_fill_cooldown_ns,
                    ts_local=ts_local,
                    inflight_exposure=inflight_exposure,
                    last_buy_cancel_fill_ts=last_buy_cancel_fill_ts,
                    last_sell_cancel_fill_ts=last_sell_cancel_fill_ts,
                )
                adverse_timing_guard = adverse_timing_guard_side_blocks(
                    enabled=adverse_timing_guard_enabled,
                    target_deterioration_enabled=adverse_timing_guard_target_deterioration_enabled,
                    pending_cancel_enabled=adverse_timing_guard_pending_cancel_enabled,
                    post_cancel_fill_enabled=adverse_timing_guard_post_cancel_fill_enabled,
                    cooldown_ns=adverse_timing_guard_cooldown_ns,
                    min_target_move_ticks=adverse_timing_guard_min_target_move_ticks,
                    ts_local=ts_local,
                    working=working,
                    target_bid_tick=target_bid_tick,
                    target_ask_tick=target_ask_tick,
                    inflight_exposure=inflight_exposure,
                    last_buy_cancel_fill_ts=last_buy_cancel_fill_ts,
                    last_sell_cancel_fill_ts=last_sell_cancel_fill_ts,
                )
                desired_buy_probe = (not pos_limit or position_notional < 0)
                desired_sell_probe = (not pos_limit or position_notional > 0)
                buy_diff_probe = (
                    abs(int(working.buy.price_tick) - target_bid_tick)
                    if desired_buy_probe and working.buy is not None
                    else 0
                )
                sell_diff_probe = (
                    abs(int(working.sell.price_tick) - target_ask_tick)
                    if desired_sell_probe and working.sell is not None
                    else 0
                )
                buy_submit_eligible = desired_buy_probe and (
                    working.buy is None
                    or (
                        working.buy is not None
                        and buy_diff_probe > 1
                        and working.buy.cancellable
                        and not two_phase_replace_enabled
                    )
                )
                sell_submit_eligible = desired_sell_probe and (
                    working.sell is None
                    or (
                        working.sell is not None
                        and sell_diff_probe > 1
                        and working.sell.cancellable
                        and not two_phase_replace_enabled
                    )
                )
                add_side_toxic_timing_guard = add_side_toxic_timing_guard_side_blocks(
                    enabled=add_side_toxic_timing_guard_enabled,
                    pending_cancel_enabled=add_side_toxic_timing_guard_pending_cancel_enabled,
                    post_cancel_fill_enabled=add_side_toxic_timing_guard_post_cancel_fill_enabled,
                    target_move_enabled=add_side_toxic_timing_guard_target_move_enabled,
                    cooldown_ns=add_side_toxic_timing_guard_window_ns,
                    min_target_move_ticks=add_side_toxic_timing_guard_min_target_move_ticks,
                    latency_threshold_ns=add_side_toxic_timing_guard_latency_threshold_ns,
                    ts_local=ts_local,
                    position=position,
                    target_bid_tick=target_bid_tick,
                    target_ask_tick=target_ask_tick,
                    buy_submit_eligible=buy_submit_eligible,
                    sell_submit_eligible=sell_submit_eligible,
                    buy_reduce_side_allowed=bool(buy_submit_eligible and position < 0.0),
                    sell_reduce_side_allowed=bool(sell_submit_eligible and position > 0.0),
                    inflight_exposure=inflight_exposure,
                    last_buy_cancel_ts=last_buy_cancel_ts,
                    last_sell_cancel_ts=last_sell_cancel_ts,
                    last_buy_cancel_fill_ts=last_buy_cancel_fill_ts,
                    last_sell_cancel_fill_ts=last_sell_cancel_fill_ts,
                    last_quote_or_cancel_target_bid_tick=last_quote_or_cancel_target_bid_tick,
                    last_quote_or_cancel_target_ask_tick=last_quote_or_cancel_target_ask_tick,
                    latency_signal_ns=latency_signal_ns,
                )

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
                        position=position,
                        max_position_qty=float(risk.get("max_position_qty", 0.0)),
                        add_side_soft_limit_qty=add_side_soft_limit_qty_from_risk(risk),
                        add_side_cooldown_block_buy=buy_cooldown_active,
                        add_side_cooldown_block_sell=sell_cooldown_active,
                        cancel_race_guard_block_buy=cancel_race_guard_buy_active,
                        cancel_race_guard_block_sell=cancel_race_guard_sell_active,
                        adverse_timing_guard_block_buy=adverse_timing_guard.buy_block,
                        adverse_timing_guard_block_sell=adverse_timing_guard.sell_block,
                        add_side_toxic_timing_guard_block_buy=(
                            add_side_toxic_timing_guard.buy_block or quote_anchor_safety.suppress_buy
                        ),
                        add_side_toxic_timing_guard_block_sell=(
                            add_side_toxic_timing_guard.sell_block or quote_anchor_safety.suppress_sell
                        ),
                        add_side_inflight_buy_qty=inflight_buy_qty,
                        add_side_inflight_sell_qty=inflight_sell_qty,
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
                                    if action.side == "buy" and position >= 0.0:
                                        last_buy_cancel_ts = ts_local
                                        last_quote_or_cancel_target_bid_tick = target_bid_tick
                                    if action.side == "sell" and position <= 0.0:
                                        last_sell_cancel_ts = ts_local
                                        last_quote_or_cancel_target_ask_tick = target_ask_tick
                                    lifecycle_tracker.mark_cancel_requested(action.order_id, ts_local)
                                    if inflight_exposure_enabled:
                                        inflight_exposure.mark_cancel_requested(action.order_id)
                                    hbt.cancel(0, int(action.order_id), False)
                                elif action.kind == "submit" and action.side == "buy":
                                    hbt.submit_buy_order(0, int(action.order_id), action.price, action.qty, GTX, LIMIT, False)
                                    last_quote_or_cancel_target_bid_tick = target_bid_tick
                                    if inflight_exposure_enabled:
                                        inflight_exposure.mark_submitted(action)
                                elif action.kind == "submit" and action.side == "sell":
                                    hbt.submit_sell_order(0, int(action.order_id), action.price, action.qty, GTX, LIMIT, False)
                                    last_quote_or_cancel_target_ask_tick = target_ask_tick
                                    if inflight_exposure_enabled:
                                        inflight_exposure.mark_submitted(action)

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
                    bid_top5_ticks=bid_top5_ticks,
                    bid_top5_qtys=bid_top5_qtys,
                    ask_top5_ticks=ask_top5_ticks,
                    ask_top5_qtys=ask_top5_qtys,
                    market_view_source=market_view.source,
                    top5_source=market_view.top5_source,
                    market_overlay_source=market_view.market_overlay_source,
                    top5_overlay_source=market_view.top5_overlay_source,
                    book_view_ts_local=market_view.ts_local,
                    book_view_ts_exch=market_view.ts_exch,
                    book_view_feed_latency_ns=market_view.feed_latency_ns,
                    book_view_stale_ms=market_view.stale_ms,
                    top5_depth_best_bid_tick=market_view.best_bid_tick,
                    top5_depth_best_ask_tick=market_view.best_ask_tick,
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
                    cancel_race_guard_buy_active=cancel_race_guard_buy_active,
                    cancel_race_guard_sell_active=cancel_race_guard_sell_active,
                    adverse_timing_guard_buy_active=adverse_timing_guard.buy_block,
                    adverse_timing_guard_sell_active=adverse_timing_guard.sell_block,
                    adverse_timing_guard_buy_reason=adverse_timing_guard.buy_reason,
                    adverse_timing_guard_sell_reason=adverse_timing_guard.sell_reason,
                    adverse_timing_guard_buy_until_ts=adverse_timing_guard.buy_until_ts,
                    adverse_timing_guard_sell_until_ts=adverse_timing_guard.sell_until_ts,
                    adverse_timing_guard_target_move_ticks_buy=adverse_timing_guard.buy_target_move_ticks,
                    adverse_timing_guard_target_move_ticks_sell=adverse_timing_guard.sell_target_move_ticks,
                    add_side_submit_eligible_buy=add_side_toxic_timing_guard.buy_eligible,
                    add_side_submit_eligible_sell=add_side_toxic_timing_guard.sell_eligible,
                    add_side_submit_blocked_buy=add_side_toxic_timing_guard.buy_block,
                    add_side_submit_blocked_sell=add_side_toxic_timing_guard.sell_block,
                    add_side_submit_block_reason_buy=add_side_toxic_timing_guard.buy_reason,
                    add_side_submit_block_reason_sell=add_side_toxic_timing_guard.sell_reason,
                    add_side_submit_reduce_side_allowed_buy=add_side_toxic_timing_guard.buy_reduce_side_allowed,
                    add_side_submit_reduce_side_allowed_sell=add_side_toxic_timing_guard.sell_reduce_side_allowed,
                    target_move_since_last_quote_or_cancel_buy=add_side_toxic_timing_guard.buy_target_move_ticks,
                    target_move_since_last_quote_or_cancel_sell=add_side_toxic_timing_guard.sell_target_move_ticks,
                    last_cancel_request_age_ms_buy=add_side_toxic_timing_guard.buy_last_cancel_request_age_ms,
                    last_cancel_request_age_ms_sell=add_side_toxic_timing_guard.sell_last_cancel_request_age_ms,
                    last_cancel_fill_age_ms_buy=add_side_toxic_timing_guard.buy_last_cancel_fill_age_ms,
                    last_cancel_fill_age_ms_sell=add_side_toxic_timing_guard.sell_last_cancel_fill_age_ms,
                    toxic_timing_guard_until_ts_buy=add_side_toxic_timing_guard.buy_until_ts,
                    toxic_timing_guard_until_ts_sell=add_side_toxic_timing_guard.sell_until_ts,
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
                    quote_update_fields=build_quote_update_audit_fields(
                        planned_actions=planned_actions,
                        executed_actions=executed_actions,
                        quote_throttle_cfg=throttle_cfg,
                        quote_throttle_state=throttle_state_snapshot,
                        token_bucket=bucket_snapshot,
                        ts_local=ts_local,
                        target_bid_tick=target_bid_tick,
                        target_ask_tick=target_ask_tick,
                        quote_anchor_safety=quote_anchor_safety,
                        book_view_stale_ms=market_view.stale_ms,
                        auditlatency_ms=auditlatency_ms,
                        feed_latency_ns=feed_latency_ns,
                        latency_signal_ns=latency_signal_ns,
                        reject_reason=reject_reason,
                        throttle_reason=throttle_reason,
                        dropped_by_latency=dropped_by_latency,
                        dropped_by_api_limit=dropped_by_api_limit,
                        pos_limit=pos_limit,
                        working_bid_req=working_diagnostics["working_bid_req"],
                        working_ask_req=working_diagnostics["working_ask_req"],
                        last_cancel_request_age_ms_buy=add_side_toxic_timing_guard.buy_last_cancel_request_age_ms,
                        last_cancel_request_age_ms_sell=add_side_toxic_timing_guard.sell_last_cancel_request_age_ms,
                        last_cancel_fill_age_ms_buy=add_side_toxic_timing_guard.buy_last_cancel_fill_age_ms,
                        last_cancel_fill_age_ms_sell=add_side_toxic_timing_guard.sell_last_cancel_fill_age_ms,
                        inventory_request_id="",
                        api_enabled=bool(api_cfg.get("enabled", True)),
                    ),
                )
                writer.writerow(row)
                rows_written += 1

                current_working = collect_working_orders(hbt.orders(0))
                current_order_diagnostics = format_working_order_diagnostics(current_working)
                for lifecycle_type, order_snapshot, _prev_snapshot in lifecycle_tracker.observe(hbt.orders(0)):
                    if inflight_exposure_enabled:
                        inflight_exposure.observe_lifecycle(lifecycle_type, order_snapshot)
                    cancel_request_ts = lifecycle_tracker.cancel_request_ts(order_snapshot.order_id)
                    is_fill_event = lifecycle_type in {"fill", "partial_fill"}
                    if is_fill_event and cancel_request_ts > 0:
                        fill_side = order_side_name(order_snapshot.side)
                        if fill_side == "buy":
                            last_buy_cancel_fill_ts = ts_local
                        elif fill_side == "sell":
                            last_sell_cancel_fill_ts = ts_local
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
            cancel_results = cancel_working_orders_for_shutdown(hbt, working_final)
            cancelled = sum(1 for result in cancel_results if result.cancel_sent)
            waited = sum(1 for result in cancel_results if result.wait_requested)
            order_responses = sum(1 for result in cancel_results if result.order_response_received)
            terminal_confirmed = sum(1 for result in cancel_results if result.terminal_confirmed)
            unknown_or_timeout = sum(
                1 for result in cancel_results if result.wait_outcome == "ok_unknown_or_timeout"
            )
            failed = sum(1 for result in cancel_results if result.error)
            log.info(
                "Shutdown cancel attempts=%d sent=%d wait_requests=%d order_responses=%d "
                "terminal_confirmed=%d unknown_or_timeout=%d failed=%d",
                len(cancel_results),
                cancelled,
                waited,
                order_responses,
                terminal_confirmed,
                unknown_or_timeout,
                failed,
            )
            for result in cancel_results:
                if result.error:
                    log.warning(
                        "Shutdown cancel issue order_id=%d source=%s side=%s "
                        "wait_outcome=%s terminal_confirmed=%s "
                        "terminal_confirmation_source=%s final_order_status=%s error=%s",
                        result.order_id,
                        result.source,
                        result.side,
                        result.wait_outcome,
                        result.terminal_confirmed,
                        result.terminal_confirmation_source,
                        result.final_order_status,
                        result.error,
                    )
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
