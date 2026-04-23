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
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
)

from audit_schema import AUDIT_FIELDS

from strategy_core import (
    EwmaSigma,
    TokenBucket,
    GreekOracle,
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


def _load_order_latency_array(config: dict[str, Any]) -> np.ndarray:
    path = str(config["latency"].get("order_latency_npz", "")).strip()
    if not path:
        raise ValueError("config.latency.order_latency_npz is required")
    npz = np.load(_expand(path))
    return npz["data"]


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
    greek_oracle = GreekOracle.from_config(greek_cfg, expand_path=_expand)

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

            feed_lat = hbt.feed_latency(0)
            order_lat = hbt.order_latency(0)
            feed_latency_ns = int(feed_lat[1] - feed_lat[0]) if feed_lat is not None else 0

            predicted_entry_ns = int(latency_oracle.entry_latency_ns(ts_local))
            last_entry_ns = int(order_lat[1] - order_lat[0]) if order_lat is not None else 0
            last_resp_ns = int(order_lat[2] - order_lat[1]) if order_lat is not None else 0
            latency_signal_ns = max(feed_latency_ns, last_entry_ns, last_resp_ns, predicted_entry_ns)

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
                predicted_entry_ns=predicted_entry_ns,
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
