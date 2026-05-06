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
from decimal import Decimal, InvalidOperation
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
    GTC,
    GTX,
    LIMIT,
    SELL,
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
)

from audit_schema import AUDIT_FIELDS

from strategy_core import (
    EwmaSigma,
    TokenBucket,
    QuoteThrottleState,
    OrderLifecycleTracker,
    PendingLocalOrder,
    GreekOracle,
    WorkingOrders,
    Action,
    compute_top5_size,
    impact_cost,
    clamp,
    round_to_tick,
    collect_working_orders,
    decide_actions,
    format_actions,
    inventory_score_from_risk,
    is_position_limit_reached,
    is_pure_cancel_extra,
    quote_throttle_reason,
    update_quote_throttle_state,
    build_audit_row,
    build_lifecycle_event_row,
    format_working_order_diagnostics,
    merge_pending_orders,
)


DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
DEPTH_SNAPSHOT_EVENT = 4
EXCH_EVENT = 1 << 31
LOCAL_EVENT = 1 << 30
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28
AUDIT_REPLAY_DECISION_MARKER_EVENT_KIND = 0xFF
AUDIT_REPLAY_DECISION_MARKER_EVENT = LOCAL_EVENT | AUDIT_REPLAY_DECISION_MARKER_EVENT_KIND

from backtest_metrics import (
    AuditPolicy,
    MetricAccumulator,
    flatten_summary,
    utc_day_from_ns,
    write_daily_csv,
    write_json,
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


class FeedLatencyOracle:
    def __init__(self, rows: list[tuple[int, int]]):
        self.rows = rows
        self.i = 0

    @classmethod
    def disabled(cls) -> "FeedLatencyOracle":
        return cls([])

    @classmethod
    def from_audit_csv(cls, path: Path, run_id: str = "") -> "FeedLatencyOracle":
        rows: list[tuple[int, int]] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
            if "ts_local" not in fields or "feed_latency_ns" not in fields:
                raise KeyError(f"missing ts_local/feed_latency_ns in {path}")
            for row in reader:
                if run_id and row.get("run_id", "") != run_id:
                    continue
                try:
                    ts_local = _parse_int_timestamp(str(row.get("ts_local", "")))
                    latency_ns = int(float(str(row.get("feed_latency_ns", "") or "0")))
                except ValueError:
                    continue
                if ts_local > 0 and latency_ns >= 0:
                    rows.append((ts_local, latency_ns))
        rows = sorted(set(rows))
        if not rows:
            raise ValueError(f"no feed latency rows loaded from {path}")
        return cls(rows)

    def feed_latency_ns(self, ts: int, fallback: int) -> int:
        if not self.rows:
            return fallback
        n = len(self.rows)
        while self.i + 1 < n and self.rows[self.i + 1][0] <= ts:
            self.i += 1
        return int(self.rows[self.i][1])


def _latency_guard_signal_ns(feed_latency_ns: int) -> int:
    return feed_latency_ns


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _distribution(vals: list[int]) -> dict[str, float]:
    if not vals:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
    s = sorted(vals)
    return {
        "count": float(len(vals)),
        "mean": float(sum(vals) / len(vals)),
        "p50": float(np.quantile(s, 0.50)),
        "p90": float(np.quantile(s, 0.90)),
        "p99": float(np.quantile(s, 0.99)),
        "min": float(s[0]),
        "max": float(s[-1]),
    }


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _validate_manifest_paths(manifest: dict[str, Any]) -> tuple[list[str], str | None]:
    raw_data_files = manifest.get("data_files")
    if not raw_data_files:
        raise ValueError("manifest.data_files must contain at least one file")

    data_files = [str(_expand(str(path))) for path in raw_data_files]
    seen: set[str] = set()
    for path in data_files:
        if path in seen:
            raise ValueError(f"manifest.data_files contains duplicate path: {path}")
        seen.add(path)
        if not Path(path).exists():
            raise FileNotFoundError(f"manifest.data_files does not exist: {path}")

    raw_initial_snapshot = manifest.get("initial_snapshot")
    initial_snapshot = None
    if raw_initial_snapshot:
        initial_snapshot = str(_expand(str(raw_initial_snapshot)))
        if not Path(initial_snapshot).exists():
            raise FileNotFoundError(f"manifest.initial_snapshot does not exist: {initial_snapshot}")

    return data_files, initial_snapshot


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


def _slice_data_by_absolute_local_ts(data: np.ndarray, start_ts_local: int, end_ts_local: int) -> np.ndarray:
    if start_ts_local > end_ts_local:
        raise ValueError("slice_ts_local_start must be <= slice_ts_local_end")
    if len(data) == 0:
        return data
    mask = (data["local_ts"] >= start_ts_local) & (data["local_ts"] <= end_ts_local)
    return data[mask]


def _select_data_for_asset(
    data_files: list[str],
    window: str,
    slice_ts_local_start: int | None = None,
    slice_ts_local_end: int | None = None,
) -> list[Any]:
    absolute_slice_requested = slice_ts_local_start is not None or slice_ts_local_end is not None
    if absolute_slice_requested:
        if slice_ts_local_start is None or slice_ts_local_end is None:
            raise ValueError("slice_ts_local_start and slice_ts_local_end must be provided together")
        if window != "full_day":
            raise ValueError("absolute ts_local slicing requires window='full_day'")
        if len(data_files) != 1:
            raise ValueError("absolute ts_local slicing currently supports exactly one data file")
        data = np.load(data_files[0])["data"]
        sliced = _slice_data_by_absolute_local_ts(data, slice_ts_local_start, slice_ts_local_end)
        if len(sliced) == 0:
            first_ts = int(data["local_ts"][0]) if len(data) else None
            last_ts = int(data["local_ts"][-1]) if len(data) else None
            raise ValueError(
                "absolute ts_local slice selected zero rows "
                f"for [{slice_ts_local_start}, {slice_ts_local_end}], "
                f"available local_ts [{first_ts}, {last_ts}]"
            )
        return [sliced]

    if window == "full_day":
        return data_files

    first_data = np.load(data_files[0])["data"]
    sliced = _slice_data_by_window(first_data, window)
    return [sliced]


def _market_data_replay_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("market_data_replay", {})
    return {
        "live_local_feed_compat": bool(cfg.get("live_local_feed_compat", False)),
        "insert_audit_replay_decision_markers": bool(
            cfg.get("insert_audit_replay_decision_markers", True)
        ),
        "tick_size": float(config["market"]["tick_size"]),
        "lot_size": float(config["market"]["lot_size"]),
    }


def _event_seen_timestamps(data: np.ndarray) -> np.ndarray:
    return np.where(
        (data["ev"] & LOCAL_EVENT) == LOCAL_EVENT,
        data["local_ts"],
        data["exch_ts"],
    )


@dataclass
class _QtyTimestamp:
    qty: float
    ts: int


class _LiveLocalFeedFuser:
    def __init__(self, tick_size: float, lot_size: float) -> None:
        self.tick_size = float(tick_size)
        self.lot_size = float(lot_size)
        self.bid_depth: dict[int, _QtyTimestamp] = {}
        self.ask_depth: dict[int, _QtyTimestamp] = {}
        self.best_bid_tick: int | None = None
        self.best_ask_tick: int | None = None
        self.best_bid_timestamp = 0
        self.best_ask_timestamp = 0
        self.low_bid_tick: int | None = None
        self.high_ask_tick: int | None = None

    def _price_tick(self, px: float) -> int:
        return int(round(float(px) / self.tick_size))

    def _qty_lot(self, qty: float) -> int:
        return int(round(float(qty) / self.lot_size))

    def _depth_below(self, start_tick: int) -> int | None:
        candidates = [tick for tick in self.bid_depth if tick <= start_tick]
        return max(candidates) if candidates else None

    def _depth_above(self, start_tick: int) -> int | None:
        candidates = [tick for tick in self.ask_depth if tick >= start_tick]
        return min(candidates) if candidates else None

    def update_bid(self, row: np.void) -> bool:
        price_tick = self._price_tick(float(row["px"]))
        qty_lot = self._qty_lot(float(row["qty"]))
        exch_ts = int(row["exch_ts"])

        if (
            self.best_bid_tick is not None
            and price_tick >= self.best_bid_tick
            and exch_ts < self.best_bid_timestamp
        ) or (
            self.best_ask_tick is not None
            and price_tick >= self.best_ask_tick
            and exch_ts < self.best_ask_timestamp
        ):
            return False

        emitted = False
        current = self.bid_depth.get(price_tick)
        if current is not None:
            if exch_ts >= current.ts:
                if qty_lot > 0:
                    self.bid_depth[price_tick] = _QtyTimestamp(float(row["qty"]), exch_ts)
                else:
                    self.bid_depth.pop(price_tick, None)
                emitted = True
        elif qty_lot > 0:
            self.bid_depth[price_tick] = _QtyTimestamp(float(row["qty"]), exch_ts)
            emitted = True

        if qty_lot == 0:
            if self.best_bid_tick == price_tick:
                self.best_bid_tick = self._depth_below(price_tick)
                self.best_bid_timestamp = exch_ts
                if self.best_bid_tick is None:
                    self.low_bid_tick = None
        else:
            if self.best_bid_tick is None or price_tick >= self.best_bid_tick:
                self.best_bid_tick = price_tick
                self.best_bid_timestamp = exch_ts
                if self.best_ask_tick is not None and price_tick >= self.best_ask_tick:
                    new_best_ask = self._depth_above(price_tick)
                    prev_best_ask = self.best_ask_tick
                    self.best_ask_tick = new_best_ask
                    self.best_ask_timestamp = exch_ts
                    for tick in list(self.ask_depth):
                        if prev_best_ask <= tick < (new_best_ask or price_tick + 1):
                            self.ask_depth.pop(tick, None)
                    if self.best_ask_tick is None:
                        self.high_ask_tick = None
            self.low_bid_tick = (
                price_tick
                if self.low_bid_tick is None
                else min(self.low_bid_tick, price_tick)
            )
        return emitted

    def update_ask(self, row: np.void) -> bool:
        price_tick = self._price_tick(float(row["px"]))
        qty_lot = self._qty_lot(float(row["qty"]))
        exch_ts = int(row["exch_ts"])

        if (
            self.best_ask_tick is not None
            and price_tick <= self.best_ask_tick
            and exch_ts < self.best_ask_timestamp
        ) or (
            self.best_bid_tick is not None
            and price_tick <= self.best_bid_tick
            and exch_ts < self.best_bid_timestamp
        ):
            return False

        emitted = False
        current = self.ask_depth.get(price_tick)
        if current is not None:
            if exch_ts >= current.ts:
                if qty_lot > 0:
                    self.ask_depth[price_tick] = _QtyTimestamp(float(row["qty"]), exch_ts)
                else:
                    self.ask_depth.pop(price_tick, None)
                emitted = True
        elif qty_lot > 0:
            self.ask_depth[price_tick] = _QtyTimestamp(float(row["qty"]), exch_ts)
            emitted = True

        if qty_lot == 0:
            if self.best_ask_tick == price_tick:
                self.best_ask_tick = self._depth_above(price_tick)
                self.best_ask_timestamp = exch_ts
                if self.best_ask_tick is None:
                    self.high_ask_tick = None
        else:
            if self.best_ask_tick is None or price_tick <= self.best_ask_tick:
                self.best_ask_tick = price_tick
                self.best_ask_timestamp = exch_ts
                if self.best_bid_tick is not None and price_tick <= self.best_bid_tick:
                    new_best_bid = self._depth_below(price_tick)
                    prev_best_bid = self.best_bid_tick
                    self.best_bid_tick = new_best_bid
                    self.best_bid_timestamp = exch_ts
                    lower = (new_best_bid if new_best_bid is not None else price_tick - 1) + 1
                    for tick in list(self.bid_depth):
                        if lower <= tick <= prev_best_bid:
                            self.bid_depth.pop(tick, None)
                    if self.best_bid_tick is None:
                        self.low_bid_tick = None
            self.high_ask_tick = (
                price_tick
                if self.high_ask_tick is None
                else max(self.high_ask_tick, price_tick)
            )
        return emitted


def _local_feed_row_from(row: np.void, event_kind: int) -> tuple[Any, ...]:
    ev = int(row["ev"])
    side = BUY_EVENT if ev & BUY_EVENT == BUY_EVENT else SELL_EVENT if ev & SELL_EVENT == SELL_EVENT else 0
    return (
        LOCAL_EVENT | side | event_kind,
        int(row["exch_ts"]),
        int(row["local_ts"]),
        float(row["px"]),
        float(row["qty"]),
        int(row["order_id"]) if "order_id" in (row.dtype.names or ()) else 0,
        int(row["ival"]) if "ival" in (row.dtype.names or ()) else 0,
        float(row["fval"]) if "fval" in (row.dtype.names or ()) else 0.0,
    )


def _live_local_feed_compat_data(
    data: np.ndarray,
    *,
    tick_size: float,
    lot_size: float,
) -> tuple[np.ndarray, dict[str, int]]:
    if len(data) == 0 or "ev" not in (data.dtype.names or ()):
        return data, {
            "input_rows": int(len(data)),
            "output_rows": int(len(data)),
            "local_input_rows": 0,
            "local_output_rows": 0,
            "dropped_local_clear_rows": 0,
            "dropped_outdated_depth_rows": 0,
        }

    base = data[(data["ev"] & EXCH_EVENT) == EXCH_EVENT].copy()
    base["ev"] = base["ev"] & ~np.asarray(LOCAL_EVENT, dtype=base["ev"].dtype)

    fuser = _LiveLocalFeedFuser(tick_size=tick_size, lot_size=lot_size)
    local_rows: list[tuple[Any, ...]] = []
    local_input_rows = 0
    dropped_local_clear_rows = 0
    dropped_outdated_depth_rows = 0

    def process_depth_row(row: np.void) -> None:
        nonlocal dropped_outdated_depth_rows
        ev = int(row["ev"])
        emitted = (
            fuser.update_bid(row)
            if ev & BUY_EVENT == BUY_EVENT
            else fuser.update_ask(row)
            if ev & SELL_EVENT == SELL_EVENT
            else False
        )
        if emitted:
            local_rows.append(_local_feed_row_from(row, DEPTH_EVENT))
        else:
            dropped_outdated_depth_rows += 1

    def process_local_row(row: np.void) -> None:
        nonlocal dropped_local_clear_rows
        event_kind = int(row["ev"]) & 0xff
        if event_kind == DEPTH_CLEAR_EVENT:
            dropped_local_clear_rows += 1
        elif event_kind in {DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT}:
            process_depth_row(row)
        elif event_kind == TRADE_EVENT:
            local_rows.append(_local_feed_row_from(row, TRADE_EVENT))
        else:
            local_rows.append(_local_feed_row_from(row, event_kind))

    pending_snapshot_rows: list[np.void] = []
    current_group_ts: int | None = None
    current_group: list[np.void] = []

    def flush_group(group: list[np.void]) -> None:
        nonlocal pending_snapshot_rows
        if not group:
            return
        regular_depth_rows = [
            row
            for row in group
            if (int(row["ev"]) & 0xff) == DEPTH_EVENT
        ]
        snapshot_or_clear_rows = [
            row
            for row in group
            if (int(row["ev"]) & 0xff) in {DEPTH_CLEAR_EVENT, DEPTH_SNAPSHOT_EVENT}
        ]
        other_rows = [
            row
            for row in group
            if (int(row["ev"]) & 0xff) not in {DEPTH_EVENT, DEPTH_CLEAR_EVENT, DEPTH_SNAPSHOT_EVENT}
        ]

        for row in regular_depth_rows:
            process_local_row(row)
        for row in other_rows:
            process_local_row(row)
        if regular_depth_rows and pending_snapshot_rows:
            for row in pending_snapshot_rows:
                process_local_row(row)
            pending_snapshot_rows = []
        pending_snapshot_rows.extend(snapshot_or_clear_rows)

    for row in data:
        ev = int(row["ev"])
        if ev & LOCAL_EVENT != LOCAL_EVENT:
            continue
        local_input_rows += 1
        row_ts = int(row["local_ts"])
        if current_group_ts is None:
            current_group_ts = row_ts
        if row_ts != current_group_ts:
            flush_group(current_group)
            current_group = []
            current_group_ts = row_ts
        current_group.append(row)

    flush_group(current_group)
    for row in pending_snapshot_rows:
        process_local_row(row)

    if local_rows:
        local = np.asarray(local_rows, dtype=data.dtype)
        combined = np.concatenate([base, local])
    else:
        combined = base

    seen_ts = _event_seen_timestamps(combined)
    order = np.argsort(seen_ts, kind="mergesort")
    out = combined[order]
    return out, {
        "input_rows": int(len(data)),
        "output_rows": int(len(out)),
        "local_input_rows": int(local_input_rows),
        "local_output_rows": int(len(local_rows)),
        "dropped_local_clear_rows": int(dropped_local_clear_rows),
        "dropped_outdated_depth_rows": int(dropped_outdated_depth_rows),
    }


def _strip_local_snapshot_events(data: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    if len(data) == 0 or "ev" not in (data.dtype.names or ()):
        return data, {
            "input_rows": int(len(data)),
            "output_rows": int(len(data)),
            "dropped_local_only_snapshot_rows": 0,
            "local_flag_removed_snapshot_rows": 0,
        }

    ev = data["ev"]
    event_kind = ev & 0xff
    is_snapshot_or_clear = (event_kind == DEPTH_CLEAR_EVENT) | (event_kind == DEPTH_SNAPSHOT_EVENT)
    has_local = (ev & LOCAL_EVENT) == LOCAL_EVENT
    has_exch = (ev & EXCH_EVENT) == EXCH_EVENT
    drop_mask = is_snapshot_or_clear & has_local & ~has_exch
    remove_local_mask = is_snapshot_or_clear & has_local & has_exch

    if not bool(drop_mask.any() or remove_local_mask.any()):
        return data, {
            "input_rows": int(len(data)),
            "output_rows": int(len(data)),
            "dropped_local_only_snapshot_rows": 0,
            "local_flag_removed_snapshot_rows": 0,
        }

    filtered = data[~drop_mask].copy()
    filtered_ev = filtered["ev"]
    filtered_event_kind = filtered_ev & 0xff
    filtered_remove_mask = (
        ((filtered_event_kind == DEPTH_CLEAR_EVENT) | (filtered_event_kind == DEPTH_SNAPSHOT_EVENT))
        & ((filtered_ev & LOCAL_EVENT) == LOCAL_EVENT)
        & ((filtered_ev & EXCH_EVENT) == EXCH_EVENT)
    )
    filtered["ev"][filtered_remove_mask] = filtered_ev[filtered_remove_mask] - LOCAL_EVENT

    return filtered, {
        "input_rows": int(len(data)),
        "output_rows": int(len(filtered)),
        "dropped_local_only_snapshot_rows": int(drop_mask.sum()),
        "local_flag_removed_snapshot_rows": int(remove_local_mask.sum()),
    }


def _apply_market_data_replay_filters(
    data_items: list[Any],
    replay_cfg: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    if not bool(replay_cfg.get("live_local_feed_compat", False)):
        return data_items, {
            "live_local_feed_compat": False,
            "input_rows": 0,
            "output_rows": 0,
            "local_input_rows": 0,
            "local_output_rows": 0,
            "dropped_local_clear_rows": 0,
            "dropped_outdated_depth_rows": 0,
        }

    filtered_items: list[Any] = []
    stats = {
        "live_local_feed_compat": True,
        "input_rows": 0,
        "output_rows": 0,
        "local_input_rows": 0,
        "local_output_rows": 0,
        "dropped_local_clear_rows": 0,
        "dropped_outdated_depth_rows": 0,
    }
    for item in data_items:
        data = np.load(item)["data"] if isinstance(item, str) else item
        filtered, item_stats = _live_local_feed_compat_data(
            data,
            tick_size=float(replay_cfg["tick_size"]),
            lot_size=float(replay_cfg["lot_size"]),
        )
        filtered_items.append(filtered)
        for key in (
            "input_rows",
            "output_rows",
            "local_input_rows",
            "local_output_rows",
            "dropped_local_clear_rows",
            "dropped_outdated_depth_rows",
        ):
            stats[key] += int(item_stats[key])
    return filtered_items, stats


def _empty_audit_replay_marker_stats() -> dict[str, int]:
    return {
        "enabled": False,
        "scheduled_count": 0,
        "inserted_count": 0,
        "duplicate_timestamp_count": 0,
        "item_count": 0,
    }


def _audit_replay_decision_marker_rows(
    schedule: list["AuditReplayScheduleEntry"],
    dtype: np.dtype,
) -> np.ndarray:
    rows = []
    names = dtype.names or ()
    for entry in schedule:
        if entry.ts_local <= 0:
            continue
        values: list[Any] = []
        for name in names:
            if name == "ev":
                values.append(AUDIT_REPLAY_DECISION_MARKER_EVENT)
            elif name == "exch_ts":
                values.append(int(entry.ts_exch))
            elif name == "local_ts":
                values.append(int(entry.ts_local))
            elif name == "px":
                values.append(0.0)
            elif name == "qty":
                values.append(0.0)
            elif name in {"order_id", "ival"}:
                values.append(0)
            elif name == "fval":
                values.append(0.0)
            else:
                field_dtype = dtype.fields[name][0]
                values.append(0.0 if field_dtype.kind == "f" else 0)
        rows.append(tuple(values))
    if not rows:
        return np.empty(0, dtype=dtype)
    return np.asarray(rows, dtype=dtype)


def _insert_audit_replay_decision_markers(
    data_items: list[Any],
    schedule: list["AuditReplayScheduleEntry"],
) -> tuple[list[Any], dict[str, int]]:
    stats = _empty_audit_replay_marker_stats()
    stats["scheduled_count"] = len(schedule)
    if not schedule:
        return data_items, stats

    out: list[Any] = []
    remaining_markers = schedule
    for item in data_items:
        data = np.load(item)["data"] if isinstance(item, str) else item
        stats["item_count"] += 1
        if len(data) == 0:
            out.append(data)
            continue

        seen_ts = _event_seen_timestamps(data)
        first_ts = int(seen_ts[0])
        last_ts = int(seen_ts[-1])
        item_schedule = [
            entry
            for entry in remaining_markers
            if first_ts <= entry.ts_local <= last_ts
        ]
        remaining_markers = [
            entry
            for entry in remaining_markers
            if entry.ts_local > last_ts
        ]
        if not item_schedule:
            out.append(data)
            continue

        existing_local_ts = {
            int(ts)
            for ts in data["local_ts"][
                ((data["ev"] & LOCAL_EVENT) == LOCAL_EVENT)
                & np.isin(data["local_ts"], [entry.ts_local for entry in item_schedule])
            ]
        }
        markers = _audit_replay_decision_marker_rows(
            [
                entry
                for entry in item_schedule
                if entry.ts_local not in existing_local_ts
            ],
            data.dtype,
        )
        duplicate_count = len(item_schedule) - len(markers)
        stats["duplicate_timestamp_count"] += int(duplicate_count)
        stats["inserted_count"] += int(len(markers))
        if len(markers) == 0:
            out.append(data)
            continue

        combined = np.concatenate([data, markers])
        order = np.argsort(_event_seen_timestamps(combined), kind="mergesort")
        out.append(combined[order])

    if not out:
        out = data_items
    stats["enabled"] = True
    return out, stats


def _parse_int_timestamp(raw: str) -> int:
    text = raw.strip()
    if not text:
        return 0
    try:
        return int(text)
    except ValueError:
        try:
            value = Decimal(text)
        except InvalidOperation as exc:
            raise ValueError(f"invalid timestamp: {raw}") from exc
        if not value.is_finite() or value != value.to_integral_value():
            raise ValueError(f"invalid timestamp: {raw}")
        return int(value)


def _feed_latency_config(config: dict[str, Any]) -> dict[str, str]:
    cfg = config.get("feed_latency", {})
    return {
        "audit_csv": str(cfg.get("audit_csv", "")),
        "run_id": str(cfg.get("run_id", "")),
    }


DECISION_EVENT_TYPES = {"", "0", "decision"}


@dataclass(frozen=True)
class AuditReplayScheduleEntry:
    # Replay is gated on the feed-local timestamp used by the live decision.
    # The strategy row still uses decision_ts_local to preserve live API/throttle
    # timing semantics.
    ts_local: int
    ts_exch: int = 0
    decision_ts_local: int = 0


@dataclass(frozen=True)
class LiveMarketState:
    best_bid: float
    best_ask: float
    mid: float
    bid_size: float
    ask_size: float
    fair: float
    reservation: float
    half_spread: float
    target_bid_tick: int
    target_ask_tick: int


def _is_decision_audit_event_type(raw: str) -> bool:
    return raw.strip().lower() in DECISION_EVENT_TYPES


def _empty_audit_cadence_schedule_stats() -> dict[str, Any]:
    return {
        "ts_column": "",
        "ts_exch_column": "",
        "feed_ts_local_source": "",
        "raw_row_count": 0,
        "run_id_match_row_count": 0,
        "decision_row_count": 0,
        "ignored_non_decision_row_count": 0,
        "empty_timestamp_row_count": 0,
        "invalid_timestamp_row_count": 0,
        "valid_decision_timestamp_count": 0,
        "deduped_decision_timestamp_count": 0,
        "unique_schedule_count": 0,
        "has_event_type_column": False,
        "has_ts_exch_column": False,
        "empty_ts_exch_row_count": 0,
        "invalid_ts_exch_row_count": 0,
        "valid_ts_exch_timestamp_count": 0,
        "missing_ts_exch_timestamp_count": 0,
        "valid_feed_ts_local_count": 0,
        "derived_feed_ts_local_count": 0,
        "fallback_decision_ts_local_count": 0,
        "first_decision_ts_local": 0,
        "last_decision_ts_local": 0,
        "first_feed_ts_local": 0,
        "last_feed_ts_local": 0,
    }


def _empty_replay_lag_gate_stats(
    max_lag_ns: int = 0,
    max_exch_lag_ns: int = 0,
    strict: bool = False,
    action: str = "report",
) -> dict[str, Any]:
    enabled = bool(max_lag_ns > 0 or max_exch_lag_ns > 0)
    return {
        "enabled": enabled,
        "strict": bool(strict and enabled),
        "passed": True,
        "action": action,
        "max_lag_ns": int(max_lag_ns),
        "max_exch_lag_ns": int(max_exch_lag_ns),
        "scheduled_count": 0,
        "due_count": 0,
        "accepted_count": 0,
        "drop_count": 0,
        "fail_count": 0,
        "breach_count": 0,
        "local_breach_count": 0,
        "exch_breach_count": 0,
        "missing_exch_lag_count": 0,
        "drop_ratio": 0.0,
        "breach_ratio": 0.0,
        "accepted_lag_ns": _distribution([]),
        "accepted_exch_lag_ns": _distribution([]),
        "all_due_lag_ns": _distribution([]),
        "all_due_exch_lag_ns": _distribution([]),
        "dropped_lag_ns": _distribution([]),
        "dropped_exch_lag_ns": _distribution([]),
    }


def _load_audit_replay_schedule_with_stats(
    audit_csv: Path,
    run_id: str = "",
    ts_column: str = "ts_local",
    ts_exch_column: str = "ts_exch",
    trigger_ts_source: str = "feed_local",
    feed_latency_column: str = "feed_latency_ns",
) -> tuple[list[AuditReplayScheduleEntry], dict[str, Any]]:
    if trigger_ts_source not in {"ts_local", "feed_local"}:
        raise ValueError(f"Unsupported audit replay trigger_ts_source: {trigger_ts_source}")
    stats = _empty_audit_cadence_schedule_stats()
    stats["ts_column"] = ts_column
    stats["ts_exch_column"] = ts_exch_column
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if ts_column not in fieldnames:
            raise KeyError(f"missing cadence timestamp column: {ts_column}")
        has_event_type_column = "event_type" in fieldnames
        has_ts_exch_column = ts_exch_column in fieldnames
        stats["has_event_type_column"] = has_event_type_column
        stats["has_ts_exch_column"] = has_ts_exch_column
        has_feed_latency_column = feed_latency_column in fieldnames
        has_feed_ts_column = "feed_ts_local" in fieldnames
        stats["feed_ts_local_source"] = (
            "decision_ts_local"
            if trigger_ts_source == "ts_local"
            else
            "feed_ts_local"
            if has_feed_ts_column
            else "ts_exch_plus_feed_latency_ns"
            if has_ts_exch_column and has_feed_latency_column
            else "decision_ts_local"
        )
        out: dict[int, AuditReplayScheduleEntry] = {}
        for row in reader:
            stats["raw_row_count"] += 1
            if run_id and row.get("run_id", "") != run_id:
                continue
            stats["run_id_match_row_count"] += 1
            if has_event_type_column and not _is_decision_audit_event_type(row.get("event_type", "")):
                stats["ignored_non_decision_row_count"] += 1
                continue
            stats["decision_row_count"] += 1
            raw = row.get(ts_column, "")
            if not raw:
                stats["empty_timestamp_row_count"] += 1
                continue
            try:
                ts = _parse_int_timestamp(raw)
            except ValueError:
                stats["invalid_timestamp_row_count"] += 1
                continue
            if ts > 0:
                stats["valid_decision_timestamp_count"] += 1
                ts_exch = 0
                if has_ts_exch_column:
                    raw_ts_exch = str(row.get(ts_exch_column, "") or "").strip()
                    if not raw_ts_exch:
                        stats["empty_ts_exch_row_count"] += 1
                    else:
                        try:
                            ts_exch = _parse_int_timestamp(raw_ts_exch)
                        except ValueError:
                            stats["invalid_ts_exch_row_count"] += 1
                            ts_exch = 0
                    if ts_exch > 0:
                        stats["valid_ts_exch_timestamp_count"] += 1
                    else:
                        stats["missing_ts_exch_timestamp_count"] += 1
                else:
                    stats["missing_ts_exch_timestamp_count"] += 1

                feed_ts_local = 0
                if trigger_ts_source == "ts_local":
                    feed_ts_local = ts
                    stats["fallback_decision_ts_local_count"] += 1
                elif has_feed_ts_column:
                    raw_feed_ts = str(row.get("feed_ts_local", "") or "").strip()
                    if raw_feed_ts:
                        try:
                            feed_ts_local = _parse_int_timestamp(raw_feed_ts)
                        except ValueError:
                            feed_ts_local = 0
                if feed_ts_local <= 0 and ts_exch > 0 and has_feed_latency_column:
                    raw_feed_latency = str(row.get(feed_latency_column, "") or "").strip()
                    if raw_feed_latency:
                        try:
                            feed_latency_ns = _parse_int_timestamp(raw_feed_latency)
                        except ValueError:
                            feed_latency_ns = 0
                        if feed_latency_ns >= 0:
                            feed_ts_local = ts_exch + feed_latency_ns
                            stats["derived_feed_ts_local_count"] += 1
                if feed_ts_local <= 0:
                    feed_ts_local = ts
                    stats["fallback_decision_ts_local_count"] += 1
                else:
                    stats["valid_feed_ts_local_count"] += 1

                out.setdefault(
                    ts,
                    AuditReplayScheduleEntry(
                        ts_local=feed_ts_local,
                        ts_exch=ts_exch,
                        decision_ts_local=ts,
                    ),
                )
    schedule = sorted(out.values(), key=lambda entry: (entry.ts_local, entry.decision_ts_local or entry.ts_local))
    stats["unique_schedule_count"] = len(schedule)
    stats["deduped_decision_timestamp_count"] = max(
        0,
        int(stats["valid_decision_timestamp_count"]) - len(schedule),
    )
    if schedule:
        decision_ts_values = [entry.decision_ts_local or entry.ts_local for entry in schedule]
        feed_ts_values = [entry.ts_local for entry in schedule]
        stats["first_decision_ts_local"] = min(decision_ts_values)
        stats["last_decision_ts_local"] = max(decision_ts_values)
        stats["first_feed_ts_local"] = min(feed_ts_values)
        stats["last_feed_ts_local"] = max(feed_ts_values)
    if not schedule:
        raise ValueError(f"no cadence timestamps loaded from {audit_csv}")
    return schedule, stats


def _load_audit_cadence_schedule_with_stats(
    audit_csv: Path,
    run_id: str = "",
    ts_column: str = "ts_local",
) -> tuple[list[int], dict[str, Any]]:
    schedule, stats = _load_audit_replay_schedule_with_stats(
        audit_csv,
        run_id=run_id,
        ts_column=ts_column,
        trigger_ts_source="ts_local",
    )
    return [entry.ts_local for entry in schedule], stats


def _load_audit_cadence_schedule(audit_csv: Path, run_id: str = "", ts_column: str = "ts_local") -> list[int]:
    schedule, _ = _load_audit_cadence_schedule_with_stats(audit_csv, run_id=run_id, ts_column=ts_column)
    return schedule


def _parse_local_order_tokens(serialized: str) -> list[dict[str, str]]:
    orders: list[dict[str, str]] = []
    for item in str(serialized or "").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 5:
            continue
        extras: dict[str, str] = {}
        for token in parts[5:]:
            if "=" in token:
                key, value = token.split("=", 1)
                extras[key.strip()] = value.strip()
        orders.append(
            {
                "order_id": parts[0].strip(),
                "side": parts[1].strip(),
                "price_tick": parts[2].strip(),
                "qty": parts[3].strip(),
                "status": parts[4].strip(),
                "req": extras.get("req", ""),
                "cxl": extras.get("cxl", ""),
            }
        )
    return orders


def _load_live_order_release_ts(audit_csv: Path, run_id: str = "") -> dict[int, int]:
    release_ts: dict[int, int] = {}
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "ts_local" not in fields or "local_open_orders" not in fields:
            return release_ts
        for row in reader:
            if run_id and row.get("run_id", "") != run_id:
                continue
            if not _is_decision_audit_event_type(row.get("event_type", "")):
                continue
            try:
                ts_local = _parse_int_timestamp(str(row.get("ts_local", "") or ""))
            except ValueError:
                continue
            if ts_local <= 0:
                continue
            for order in _parse_local_order_tokens(str(row.get("local_open_orders", "") or "")):
                try:
                    order_id = int(order["order_id"])
                except ValueError:
                    continue
                if order_id in release_ts:
                    continue
                req = order.get("req", "")
                cxl = order.get("cxl", "")
                if req == "none" or cxl == "1":
                    release_ts[order_id] = ts_local
    return release_ts


def _load_live_order_cancel_pending_ts(audit_csv: Path, run_id: str = "") -> dict[int, int]:
    cancel_pending_ts: dict[int, int] = {}
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "ts_local" not in fields or "local_open_orders" not in fields:
            return cancel_pending_ts
        for row in reader:
            if run_id and row.get("run_id", "") != run_id:
                continue
            if not _is_decision_audit_event_type(row.get("event_type", "")):
                continue
            try:
                ts_local = _parse_int_timestamp(str(row.get("ts_local", "") or ""))
            except ValueError:
                continue
            if ts_local <= 0:
                continue
            for order in _parse_local_order_tokens(str(row.get("local_open_orders", "") or "")):
                try:
                    order_id = int(order["order_id"])
                except ValueError:
                    continue
                if order_id in cancel_pending_ts:
                    continue
                if order.get("req", "") == "cancel":
                    cancel_pending_ts[order_id] = ts_local
    return cancel_pending_ts


def _load_live_order_absent_after_seen_ts(audit_csv: Path, run_id: str = "") -> dict[int, int]:
    absent_ts: dict[int, int] = {}
    seen: set[int] = set()
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "ts_local" not in fields or "local_open_orders" not in fields:
            return absent_ts
        for row in reader:
            if run_id and row.get("run_id", "") != run_id:
                continue
            if not _is_decision_audit_event_type(row.get("event_type", "")):
                continue
            try:
                ts_local = _parse_int_timestamp(str(row.get("ts_local", "") or ""))
            except ValueError:
                continue
            if ts_local <= 0:
                continue
            current: set[int] = set()
            for order in _parse_local_order_tokens(str(row.get("local_open_orders", "") or "")):
                try:
                    current.add(int(order["order_id"]))
                except ValueError:
                    continue
            for order_id in sorted(seen - current):
                absent_ts.setdefault(order_id, ts_local)
            seen.update(current)
    return absent_ts


def _load_live_strategy_position_by_decision_ts(audit_csv: Path, run_id: str = "") -> dict[int, float]:
    positions: dict[int, float] = {}
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "ts_local" not in fields or "position" not in fields:
            return positions
        for row in reader:
            if run_id and row.get("run_id", "") != run_id:
                continue
            if not _is_decision_audit_event_type(row.get("event_type", "")):
                continue
            raw_position = str(row.get("position", "") or "").strip()
            if not raw_position:
                continue
            try:
                ts_local = _parse_int_timestamp(str(row.get("ts_local", "") or ""))
                position = float(raw_position)
            except ValueError:
                continue
            if ts_local <= 0:
                continue
            positions.setdefault(ts_local, position)
    return positions


def _load_live_market_state_by_decision_ts(audit_csv: Path, run_id: str = "") -> dict[int, LiveMarketState]:
    states: dict[int, LiveMarketState] = {}
    required = {
        "ts_local",
        "best_bid",
        "best_ask",
        "mid",
        "bid_size",
        "ask_size",
        "fair",
        "reservation",
        "half_spread",
        "target_bid_tick",
        "target_ask_tick",
    }
    with audit_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if not required.issubset(fields):
            return states
        for row in reader:
            if run_id and row.get("run_id", "") != run_id:
                continue
            if not _is_decision_audit_event_type(row.get("event_type", "")):
                continue
            try:
                ts_local = _parse_int_timestamp(str(row.get("ts_local", "") or ""))
                state = LiveMarketState(
                    best_bid=float(str(row.get("best_bid", "") or "0")),
                    best_ask=float(str(row.get("best_ask", "") or "0")),
                    mid=float(str(row.get("mid", "") or "0")),
                    bid_size=float(str(row.get("bid_size", "") or "0")),
                    ask_size=float(str(row.get("ask_size", "") or "0")),
                    fair=float(str(row.get("fair", "") or "0")),
                    reservation=float(str(row.get("reservation", "") or "0")),
                    half_spread=float(str(row.get("half_spread", "") or "0")),
                    target_bid_tick=_parse_int_timestamp(str(row.get("target_bid_tick", "") or "0")),
                    target_ask_tick=_parse_int_timestamp(str(row.get("target_ask_tick", "") or "0")),
                )
            except ValueError:
                continue
            if ts_local <= 0:
                continue
            if (
                state.best_bid <= 0.0
                or state.best_ask <= 0.0
                or state.best_ask <= state.best_bid
                or state.mid <= 0.0
                or state.target_bid_tick <= 0
                or state.target_ask_tick <= 0
            ):
                continue
            states.setdefault(ts_local, state)
    return states


def _order_side_int(order: Any) -> int:
    side = getattr(order, "side")
    if isinstance(side, str):
        side_lower = side.strip().lower()
        if side_lower == "buy":
            return BUY
        if side_lower == "sell":
            return SELL
    return int(side)


def _pending_cancel_overlay_from_order(
    order: Any,
    *,
    visible_ts: int = 0,
    release_ts: int,
    tick_size: float,
    decision_ts: int,
    req: int = 0,
    cancellable: bool = True,
) -> PendingLocalOrder:
    price_tick = int(getattr(order, "price_tick", 0) or 0)
    raw_price = getattr(order, "price", 0.0)
    try:
        price = float(raw_price)
    except (TypeError, ValueError):
        price = 0.0
    if price <= 0.0 and price_tick > 0:
        price = float(price_tick) * float(tick_size)
    qty = float(getattr(order, "qty", 0.0) or 0.0)
    leaves_qty = float(getattr(order, "leaves_qty", qty) or qty)
    return PendingLocalOrder(
        order_id=int(getattr(order, "order_id")),
        side=_order_side_int(order),
        price=price,
        price_tick=price_tick,
        qty=qty,
        leaves_qty=leaves_qty,
        local_timestamp=int(getattr(order, "local_timestamp", decision_ts) or decision_ts),
        req=int(req),
        exch_timestamp=int(getattr(order, "exch_timestamp", 0) or 0),
        cancellable=bool(cancellable),
        visible_ts=int(visible_ts),
        release_ts=int(release_ts),
    )


def _find_visible_order_for_action(working: WorkingOrders, action: Action) -> Any | None:
    if action.side == "buy":
        return working.buy
    if action.side == "sell":
        return working.sell
    return None


def _hide_orders_absent_in_live(
    working: WorkingOrders,
    absent_after_seen_ts: dict[int, int],
    decision_ts: int,
    order_overlays_by_id: dict[int, PendingLocalOrder] | None = None,
) -> WorkingOrders:
    order_overlays_by_id = order_overlays_by_id or {}

    def keep(order: Any | None) -> Any | None:
        if order is None:
            return None
        absent_ts = int(absent_after_seen_ts.get(int(getattr(order, "order_id")), 0) or 0)
        if absent_ts > 0 and int(decision_ts) >= absent_ts:
            return None
        return order

    def visible_extra(extra: Any) -> bool:
        absent_ts = int(absent_after_seen_ts.get(int(extra.order_id), 0) or 0)
        return not (absent_ts > 0 and int(decision_ts) >= absent_ts)

    buy = keep(working.buy)
    sell = keep(working.sell)
    extras = [extra for extra in working.extras if visible_extra(extra)]

    if buy is None:
        for idx, extra in enumerate(list(extras)):
            if getattr(extra, "side", "") == "buy":
                promoted = order_overlays_by_id.get(int(extra.order_id)) or getattr(extra, "source_order", None)
                if promoted is not None:
                    buy = promoted
                    extras.pop(idx)
                break
    if sell is None:
        for idx, extra in enumerate(list(extras)):
            if getattr(extra, "side", "") == "sell":
                promoted = order_overlays_by_id.get(int(extra.order_id)) or getattr(extra, "source_order", None)
                if promoted is not None:
                    sell = promoted
                    extras.pop(idx)
                break

    return WorkingOrders(buy=buy, sell=sell, extras=extras)


def _prune_released_pending_orders(pending_orders: dict[int, PendingLocalOrder], decision_ts: int) -> None:
    for order_id, order in list(pending_orders.items()):
        release_ts = int(getattr(order, "release_ts", 0) or 0)
        if release_ts > 0 and int(decision_ts) >= release_ts:
            pending_orders.pop(order_id, None)


def _prune_absent_pending_submit_orders(
    pending_orders: dict[int, PendingLocalOrder],
    absent_after_seen_ts: dict[int, int],
    decision_ts: int,
) -> None:
    for order_id in list(pending_orders):
        absent_ts = int(absent_after_seen_ts.get(int(order_id), 0) or 0)
        if absent_ts > 0 and int(decision_ts) >= absent_ts:
            pending_orders.pop(order_id, None)


def _submit_overlay_for_decision(order: PendingLocalOrder, decision_ts: int) -> PendingLocalOrder:
    release_ts = int(getattr(order, "release_ts", 0) or 0)
    if release_ts <= 0 or int(decision_ts) < release_ts:
        return order
    return PendingLocalOrder(
        order_id=int(order.order_id),
        side=int(order.side),
        price=float(order.price),
        price_tick=int(order.price_tick),
        qty=float(order.qty),
        leaves_qty=float(order.leaves_qty),
        local_timestamp=int(order.local_timestamp),
        req=0,
        exch_timestamp=int(order.exch_timestamp),
        cancellable=True,
        visible_ts=int(getattr(order, "visible_ts", 0) or 0),
        release_ts=int(release_ts),
    )


def _cancel_overlay_for_decision(order: PendingLocalOrder, decision_ts: int) -> PendingLocalOrder:
    visible_ts = int(getattr(order, "visible_ts", 0) or 0)
    if visible_ts <= 0 or int(decision_ts) < visible_ts:
        return order
    return PendingLocalOrder(
        order_id=int(order.order_id),
        side=int(order.side),
        price=float(order.price),
        price_tick=int(order.price_tick),
        qty=float(order.qty),
        leaves_qty=float(order.leaves_qty),
        local_timestamp=int(order.local_timestamp),
        req=4,
        exch_timestamp=int(order.exch_timestamp),
        cancellable=False,
        visible_ts=int(visible_ts),
        release_ts=int(getattr(order, "release_ts", 0) or 0),
    )


def _live_visible_working_orders(
    working: WorkingOrders,
    pending_submit_orders: list[PendingLocalOrder],
    pending_cancel_overlays: list[PendingLocalOrder] | None,
    pending_cancel_retention_overlays: list[PendingLocalOrder] | None = None,
    *,
    decision_ts: int,
    absent_after_seen_ts: dict[int, int] | None = None,
) -> WorkingOrders:
    submit_overlays = [_submit_overlay_for_decision(order, decision_ts) for order in pending_submit_orders]
    order_overlays_by_id = {int(order.order_id): order for order in submit_overlays}
    if absent_after_seen_ts:
        working = _hide_orders_absent_in_live(
            working,
            absent_after_seen_ts,
            decision_ts,
            order_overlays_by_id=order_overlays_by_id,
        )

    cancel_overlays = [
        _cancel_overlay_for_decision(order, decision_ts)
        for order in (pending_cancel_overlays or [])
        if int(getattr(order, "release_ts", 0) or 0) > int(decision_ts)
    ]
    cancel_retention_overlays = [
        order
        for order in (pending_cancel_retention_overlays or [])
        if int(getattr(order, "visible_ts", 0) or 0) <= int(decision_ts)
        and int(getattr(order, "release_ts", 0) or 0) > int(decision_ts)
    ]

    merged = merge_pending_orders(working, submit_overlays, replace_existing=True)
    merged = merge_pending_orders(merged, cancel_overlays, replace_existing=True)
    return merge_pending_orders(merged, cancel_retention_overlays, replace_existing=True)


def _audit_replay_decision_due(
    ts_local: int,
    schedule: list[int],
    schedule_idx: int,
    tolerance_ns: int,
    replay_mode: str = "single",
    max_lag_ns: int = 0,
) -> tuple[bool, int, int, int, bool, int]:
    if schedule_idx >= len(schedule):
        return False, schedule_idx, 0, 0, False, 0
    due_cutoff = ts_local + tolerance_ns
    if due_cutoff < schedule[schedule_idx]:
        return False, schedule_idx, 0, 0, False, 0

    if replay_mode == "single":
        scheduled_ts = schedule[schedule_idx]
        lag = ts_local - scheduled_ts
        breach = max_lag_ns > 0 and abs(lag) > max_lag_ns
        return True, schedule_idx + 1, lag, 0, breach, scheduled_ts

    if replay_mode != "drain_due":
        raise ValueError(f"Unsupported audit replay_mode: {replay_mode}")

    next_idx = schedule_idx + 1
    while next_idx < len(schedule) and schedule[next_idx] <= due_cutoff:
        next_idx += 1

    scheduled_ts = schedule[next_idx - 1]
    lag = ts_local - scheduled_ts
    skipped_due = max(0, next_idx - schedule_idx - 1)
    breach = max_lag_ns > 0 and abs(lag) > max_lag_ns
    return True, next_idx, lag, skipped_due, breach, scheduled_ts


def _exchange_lag_ns(bt_feed_ts_exch: int, scheduled_ts_exch: int) -> int | None:
    if bt_feed_ts_exch <= 0 or scheduled_ts_exch <= 0:
        return None
    return bt_feed_ts_exch - scheduled_ts_exch


def _lag_breached(lag_ns: int | None, max_lag_ns: int) -> bool:
    if max_lag_ns <= 0:
        return False
    return lag_ns is None or abs(lag_ns) > max_lag_ns


def _audit_replay_schedule_decision_due(
    ts_local: int,
    bt_feed_ts_exch: int,
    schedule: list[AuditReplayScheduleEntry],
    schedule_idx: int,
    tolerance_ns: int,
    replay_mode: str = "single",
    max_lag_ns: int = 0,
    max_exch_lag_ns: int = 0,
) -> tuple[bool, int, int, int | None, int, bool, bool, int, int, int]:
    if schedule_idx >= len(schedule):
        return False, schedule_idx, 0, None, 0, False, False, 0, 0, 0
    due_cutoff = ts_local + tolerance_ns
    if due_cutoff < schedule[schedule_idx].ts_local:
        return False, schedule_idx, 0, None, 0, False, False, 0, 0, 0

    if replay_mode == "single":
        entry = schedule[schedule_idx]
        next_idx = schedule_idx + 1
        skipped_due = 0
    elif replay_mode == "drain_due":
        next_idx = schedule_idx + 1
        while next_idx < len(schedule) and schedule[next_idx].ts_local <= due_cutoff:
            next_idx += 1
        entry = schedule[next_idx - 1]
        skipped_due = max(0, next_idx - schedule_idx - 1)
    else:
        raise ValueError(f"Unsupported audit replay_mode: {replay_mode}")

    local_lag_ns = ts_local - entry.ts_local
    exch_lag_ns = _exchange_lag_ns(bt_feed_ts_exch, entry.ts_exch)
    local_breach = _lag_breached(local_lag_ns, max_lag_ns)
    exch_breach = _lag_breached(exch_lag_ns, max_exch_lag_ns)
    decision_ts_local = entry.decision_ts_local or entry.ts_local
    return (
        True,
        next_idx,
        local_lag_ns,
        exch_lag_ns,
        skipped_due,
        local_breach,
        exch_breach,
        decision_ts_local,
        entry.ts_local,
        entry.ts_exch,
    )


def _backtest_cadence_config(config: dict[str, Any]) -> dict[str, Any]:
    cadence_cfg = config.get("backtest_cadence", {})
    mode = str(cadence_cfg.get("mode", "")).strip()
    if not mode:
        mode = "fixed_interval"
    if mode not in {"fixed_interval", "audit_replay"}:
        raise ValueError(f"Unsupported backtest_cadence.mode: {mode}")

    enabled = bool(cadence_cfg.get("enabled", mode == "audit_replay"))
    min_interval_ns = 0
    if enabled and mode == "fixed_interval":
        min_interval_ns = int(float(cadence_cfg.get("min_decision_interval_ms", 0.0)) * 1_000_000)

    max_lag_ns = int(float(cadence_cfg.get("max_lag_ms", 0.0)) * 1_000_000)
    max_exch_lag_ns = int(float(cadence_cfg.get("max_exch_lag_ms", cadence_cfg.get("max_lag_ms", 0.0))) * 1_000_000)
    strict_lag_gate = bool(cadence_cfg.get("strict_lag_gate", max_lag_ns > 0 or max_exch_lag_ns > 0))
    lag_gate_action = str(cadence_cfg.get("lag_gate_action", "fail" if strict_lag_gate else "report"))
    if lag_gate_action not in {"report", "drop", "fail"}:
        raise ValueError(f"Unsupported backtest_cadence.lag_gate_action: {lag_gate_action}")
    replay_mode = str(cadence_cfg.get("replay_mode", "single"))
    if replay_mode not in {"single", "drain_due", "emit_due"}:
        raise ValueError(f"Unsupported audit replay_mode: {replay_mode}")
    trigger_ts_source = str(cadence_cfg.get("trigger_ts_source", "feed_local" if mode == "audit_replay" else "ts_local")).strip()
    if trigger_ts_source not in {"ts_local", "feed_local"}:
        raise ValueError(f"Unsupported backtest_cadence.trigger_ts_source: {trigger_ts_source}")
    market_state_overlay = str(cadence_cfg.get("market_state_overlay", "off")).strip()
    if market_state_overlay not in {"off", "audit"}:
        raise ValueError(f"Unsupported backtest_cadence.market_state_overlay: {market_state_overlay}")
    return {
        "mode": mode,
        "min_interval_ns": min_interval_ns,
        "audit_csv": str(cadence_cfg.get("audit_csv", "")),
        "run_id": str(cadence_cfg.get("run_id", "")),
        "ts_column": str(cadence_cfg.get("ts_column", "ts_local")),
        "tolerance_ns": int(float(cadence_cfg.get("tolerance_ms", 0.0)) * 1_000_000),
        "replay_mode": replay_mode,
        "max_lag_ns": max_lag_ns,
        "max_exch_lag_ns": max_exch_lag_ns,
        "strict_lag_gate": strict_lag_gate,
        "lag_gate_action": lag_gate_action,
        "trigger_ts_source": trigger_ts_source,
        "feed_latency_column": str(cadence_cfg.get("feed_latency_column", "feed_latency_ns")),
        "market_state_overlay": market_state_overlay,
    }


def _backtest_cadence_interval_ns(config: dict[str, Any]) -> int:
    return int(_backtest_cadence_config(config)["min_interval_ns"])


def _should_skip_strategy_decision(ts_local: int, last_decision_ts: int | None, min_interval_ns: int) -> bool:
    return min_interval_ns > 0 and last_decision_ts is not None and (ts_local - last_decision_ts) < min_interval_ns


def _alignment_initial_position_disabled_metadata() -> dict[str, Any]:
    return {
        "enabled": False,
        "position_mode": "off",
        "target_position": 0.0,
        "applied_position": 0.0,
        "source": "",
        "ts_local": 0,
        "order_id": 0,
        "side": "",
        "price": 0.0,
        "qty": 0.0,
        "rc": 0,
        "success": True,
        "tolerance": 0.0,
        "reason": "",
    }


def _local_open_order_count(working: WorkingOrders) -> int:
    count = 0
    if working.buy is not None:
        count += 1
    if working.sell is not None:
        count += 1
    count += len(working.extras)
    return count


def _alignment_init_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("alignment_init", {})
    enabled = bool(raw.get("enabled", False))
    position_mode = str(raw.get("position_mode", "synthetic_fill" if enabled else "off")).strip()
    if not enabled:
        position_mode = "off"
    if position_mode not in {"off", "synthetic_fill"}:
        raise ValueError(f"Unsupported alignment_init.position_mode: {position_mode}")

    return {
        "enabled": enabled,
        "position_mode": position_mode,
        "position": float(raw.get("position", 0.0)),
        "source": str(raw.get("source", "")),
        "ts_local": int(raw.get("ts_local", 0)),
        "order_id_start": int(raw.get("order_id_start", 900_000_000_000)),
    }


def _round_position_qty(position: float, lot_size: float) -> float:
    if lot_size <= 0.0:
        raise ValueError("lot_size must be positive")
    raw_lots = abs(float(position)) / float(lot_size)
    lots = math.floor(raw_lots + 0.5)
    if lots <= 0:
        return 0.0
    return lots * float(lot_size)


def _apply_alignment_initial_position(
    hbt: Any,
    asset_no: int,
    *,
    target_position: float,
    order_id: int,
    best_bid: float,
    best_ask: float,
    lot_size: float,
    source: str = "",
    source_ts_local: int = 0,
) -> dict[str, Any]:
    qty = _round_position_qty(target_position, lot_size)
    tolerance = max(float(lot_size) * 0.5, 1e-12)
    metadata = {
        "enabled": abs(float(target_position)) > 0.0,
        "position_mode": "synthetic_fill",
        "target_position": float(target_position),
        "applied_position": float(hbt.position(asset_no)),
        "source": source,
        "ts_local": int(source_ts_local),
        "order_id": int(order_id),
        "side": "",
        "price": 0.0,
        "qty": float(qty),
        "rc": 0,
        "success": True,
        "tolerance": tolerance,
        "reason": "",
    }
    if qty <= 0.0:
        return metadata

    if target_position > 0.0:
        metadata["side"] = "buy"
        metadata["price"] = float(best_ask)
        rc = hbt.submit_buy_order(asset_no, int(order_id), float(best_ask), qty, GTC, LIMIT, True)
    else:
        metadata["side"] = "sell"
        metadata["price"] = float(best_bid)
        rc = hbt.submit_sell_order(asset_no, int(order_id), float(best_bid), qty, GTC, LIMIT, True)

    applied_position = float(hbt.position(asset_no))
    metadata["rc"] = int(rc)
    metadata["applied_position"] = applied_position
    metadata["success"] = abs(applied_position - float(target_position)) <= tolerance
    if not metadata["success"]:
        metadata["reason"] = "position_mismatch"
    return metadata


def _apply_initial_snapshot(asset: Any, initial_snapshot: str | None) -> None:
    if initial_snapshot:
        asset.initial_snapshot(initial_snapshot)


def _continuous_run_metadata(window: str, data_files: list[str], initial_snapshot: str | None) -> dict[str, Any]:
    return {
        "continuous_run": window == "full_day" and len(data_files) > 1,
        "initial_snapshot": initial_snapshot,
        "data_file_count": len(data_files),
        "data_files": data_files,
    }


def _load_order_latency_array(config: dict[str, Any]) -> np.ndarray:
    path = str(config["latency"].get("order_latency_npz", "")).strip()
    if not path:
        raise ValueError("config.latency.order_latency_npz is required")
    npz = np.load(_expand(path))
    return npz["data"]


def run_backtest(
    config: dict[str, Any],
    manifest: dict[str, Any],
    window_override: str | None = None,
    slice_ts_local_start: int | None = None,
    slice_ts_local_end: int | None = None,
) -> dict[str, Any]:
    symbol = str(config["symbol"]["name"])
    market = config["market"]
    risk = config["risk"]
    fair_cfg = config["fair"]
    greek_cfg = config.get("greeks", {})
    latency_cfg = config["latency"]
    api_cfg = config["api_limit"]
    fee_cfg = config["fee"]
    queue_cfg = config["queue"]
    strategy_cfg = config.get("strategy", {})

    output_root = _expand(str(config["paths"]["output_root"]))
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = f"{symbol.lower()}_{manifest['start_day']}_to_{manifest['end_day']}"
    audit_name = str(config["audit"]["output_csv"])
    audit_path = output_root / audit_name

    audit_cfg = config.get("audit", {})
    audit_policy = AuditPolicy.from_config(audit_cfg)
    summary_cfg = config.get("summary", {})
    summary_enabled = bool(summary_cfg.get("enabled", False))
    summary_json_path = output_root / str(summary_cfg.get("output_json", "summary.json"))
    daily_csv_path = output_root / str(summary_cfg.get("daily_csv", "daily_summary.csv"))

    data_files, initial_snapshot = _validate_manifest_paths(manifest)

    window = window_override or str(config["backtest"]["window"])

    cadence_cfg = _backtest_cadence_config(config)
    cadence_interval_ns = int(cadence_cfg["min_interval_ns"])
    cadence_mode = str(cadence_cfg["mode"])
    audit_replay_schedule: list[AuditReplayScheduleEntry] = []
    audit_replay_schedule_stats = _empty_audit_cadence_schedule_stats()
    if cadence_mode == "audit_replay":
        audit_csv = str(cadence_cfg["audit_csv"]).strip()
        if not audit_csv:
            raise ValueError("backtest_cadence.audit_csv is required for audit_replay mode")
        audit_csv_path = _expand(audit_csv)
        audit_replay_schedule, audit_replay_schedule_stats = _load_audit_replay_schedule_with_stats(
            audit_csv_path,
            run_id=str(cadence_cfg["run_id"]),
            ts_column=str(cadence_cfg["ts_column"]),
            trigger_ts_source=str(cadence_cfg["trigger_ts_source"]),
            feed_latency_column=str(cadence_cfg["feed_latency_column"]),
        )
        live_order_release_ts = _load_live_order_release_ts(
            audit_csv_path,
            run_id=str(cadence_cfg["run_id"]),
        )
        live_order_cancel_pending_ts = _load_live_order_cancel_pending_ts(
            audit_csv_path,
            run_id=str(cadence_cfg["run_id"]),
        )
        live_order_absent_after_seen_ts = _load_live_order_absent_after_seen_ts(
            audit_csv_path,
            run_id=str(cadence_cfg["run_id"]),
        )
        live_strategy_position_by_decision_ts = _load_live_strategy_position_by_decision_ts(
            audit_csv_path,
            run_id=str(cadence_cfg["run_id"]),
        )
        live_market_state_by_decision_ts = (
            _load_live_market_state_by_decision_ts(
                audit_csv_path,
                run_id=str(cadence_cfg["run_id"]),
            )
            if str(cadence_cfg["market_state_overlay"]) == "audit"
            else {}
        )
    else:
        live_order_release_ts = {}
        live_order_cancel_pending_ts = {}
        live_order_absent_after_seen_ts = {}
        live_strategy_position_by_decision_ts = {}
        live_market_state_by_decision_ts = {}

    data_for_asset = _select_data_for_asset(
        data_files,
        window,
        slice_ts_local_start=slice_ts_local_start,
        slice_ts_local_end=slice_ts_local_end,
    )
    market_data_replay_cfg = _market_data_replay_config(config)
    data_for_asset, market_data_replay_stats = _apply_market_data_replay_filters(
        data_for_asset,
        market_data_replay_cfg,
    )
    audit_replay_marker_stats = _empty_audit_replay_marker_stats()
    if (
        cadence_mode == "audit_replay"
        and bool(market_data_replay_cfg.get("insert_audit_replay_decision_markers", True))
    ):
        data_for_asset, audit_replay_marker_stats = _insert_audit_replay_decision_markers(
            data_for_asset,
            audit_replay_schedule,
        )
    market_data_replay_stats["audit_replay_decision_markers"] = audit_replay_marker_stats

    continuous_metadata = _continuous_run_metadata(window, data_files, initial_snapshot)

    latency_data = _load_order_latency_array(config)
    latency_oracle = LatencyOracle(latency_data)
    feed_latency_cfg = _feed_latency_config(config)
    feed_latency_oracle = (
        FeedLatencyOracle.from_audit_csv(
            _expand(feed_latency_cfg["audit_csv"]),
            run_id=feed_latency_cfg["run_id"],
        )
        if feed_latency_cfg["audit_csv"]
        else FeedLatencyOracle.disabled()
    )
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

    _apply_initial_snapshot(asset, initial_snapshot)

    hbt = ROIVectorMarketDepthBacktest([asset])

    sigma_est = EwmaSigma()
    bucket = TokenBucket.create(float(api_cfg["capacity"]), float(api_cfg["refill_per_sec"]))
    min_interval_ns = int(float(api_cfg["min_interval_ms"]) * 1_000_000)
    latency_guard_ns = int(float(latency_cfg["latency_guard_ms"]) * 1_000_000)
    quote_throttle_enabled = bool(strategy_cfg.get("quote_throttle_enabled", False))
    quote_min_interval_ns = int(float(strategy_cfg.get("min_quote_update_interval_ms", 0.0)) * 1_000_000)
    quote_min_move_ticks = int(strategy_cfg.get("min_quote_move_ticks", 0))
    two_phase_replace_enabled = bool(strategy_cfg.get("two_phase_replace_enabled", False))
    quote_throttle = QuoteThrottleState()
    lifecycle_tracker = OrderLifecycleTracker.create()
    pending_local_orders: dict[int, PendingLocalOrder] = {}
    pending_cancel_overlays: dict[int, PendingLocalOrder] = {}
    pending_cancel_retention_overlays: dict[int, PendingLocalOrder] = {}
    lifecycle_event_seq = 0
    alignment_init_cfg = _alignment_init_config(config)
    alignment_init_checked = False
    alignment_initial_position = _alignment_initial_position_disabled_metadata()
    if alignment_init_cfg["enabled"]:
        alignment_initial_position.update(
            {
                "enabled": True,
                "position_mode": str(alignment_init_cfg["position_mode"]),
                "target_position": float(alignment_init_cfg["position"]),
                "source": str(alignment_init_cfg["source"]),
                "ts_local": int(alignment_init_cfg["ts_local"]),
                "order_id": int(alignment_init_cfg["order_id_start"]),
                "success": False,
                "reason": "not_applied",
            }
        )
    audit_replay_idx = 0
    cadence_skipped_feed_events = 0
    cadence_lags_ns: list[int] = []
    cadence_exch_lags_ns: list[int] = []
    audit_replay_due_lags_ns: list[int] = []
    audit_replay_due_exch_lags_ns: list[int] = []
    audit_replay_dropped_lags_ns: list[int] = []
    audit_replay_dropped_exch_lags_ns: list[int] = []
    audit_replay_skipped_due_count = 0
    audit_replay_max_lag_breaches = 0
    audit_replay_local_lag_breaches = 0
    audit_replay_exch_lag_breaches = 0
    audit_replay_missing_exch_lag_count = 0
    audit_replay_lag_gate_drops = 0
    audit_replay_lag_gate_failures = 0
    audit_replay_strategy_position_overlay_count = 0
    audit_replay_market_state_overlay_count = 0

    next_order_id = 1
    strategy_seq = 0
    last_api_ts: int | None = None
    last_decision_ts: int | None = None
    timeout_ns = int(config["backtest"]["wait_timeout_ns"])

    rows_written = 0
    audit_file = None
    writer = None
    if audit_policy.mode != "off":
        audit_file = audit_path.open("w", newline="")
        writer = csv.DictWriter(audit_file, fieldnames=AUDIT_FIELDS)
        writer.writeheader()

    metrics = MetricAccumulator()
    day_metrics = MetricAccumulator()
    current_day: str | None = None
    daily_rows: list[dict[str, Any]] = []
    pending_replay_decisions: list[tuple[int, int, int, int, int]] = []

    try:
        while True:
            if not pending_replay_decisions:
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

            feed_lat = hbt.feed_latency(0)
            order_lat = hbt.order_latency(0)
            current_feed_ts_exch = int(feed_lat[0]) if feed_lat is not None else 0

            if not alignment_init_checked:
                alignment_init_checked = True
                if (
                    alignment_init_cfg["enabled"]
                    and alignment_init_cfg["position_mode"] == "synthetic_fill"
                    and abs(float(alignment_init_cfg["position"])) > 0.0
                ):
                    alignment_initial_position = _apply_alignment_initial_position(
                        hbt,
                        0,
                        target_position=float(alignment_init_cfg["position"]),
                        order_id=int(alignment_init_cfg["order_id_start"]),
                        best_bid=best_bid,
                        best_ask=best_ask,
                        lot_size=float(depth.lot_size),
                        source=str(alignment_init_cfg["source"]),
                        source_ts_local=int(alignment_init_cfg["ts_local"]),
                    )
                    if not bool(alignment_initial_position["success"]):
                        raise RuntimeError(
                            "failed to apply alignment initial position: "
                            + json.dumps(alignment_initial_position, sort_keys=True)
                        )
                    continue

            if cadence_mode == "audit_replay":
                if not pending_replay_decisions:
                    had_due = False
                    emit_due = str(cadence_cfg["replay_mode"]) == "emit_due"
                    schedule_replay_mode = "single" if emit_due else str(cadence_cfg["replay_mode"])
                    while True:
                        (
                            due,
                            audit_replay_idx,
                            lag_ns,
                            exch_lag_ns,
                            skipped_due,
                            local_lag_breach,
                            exch_lag_breach,
                            consumed_decision_ts,
                            consumed_feed_ts,
                            _consumed_live_ts_exch,
                        ) = _audit_replay_schedule_decision_due(
                            ts_local,
                            current_feed_ts_exch,
                            audit_replay_schedule,
                            audit_replay_idx,
                            int(cadence_cfg["tolerance_ns"]),
                            schedule_replay_mode,
                            int(cadence_cfg["max_lag_ns"]),
                            int(cadence_cfg["max_exch_lag_ns"]),
                        )
                        if not due:
                            break
                        had_due = True
                        audit_replay_due_lags_ns.append(lag_ns)
                        if exch_lag_ns is None:
                            audit_replay_missing_exch_lag_count += 1
                        else:
                            audit_replay_due_exch_lags_ns.append(exch_lag_ns)
                        audit_replay_skipped_due_count += skipped_due
                        lag_breach = local_lag_breach or exch_lag_breach
                        if local_lag_breach:
                            audit_replay_local_lag_breaches += 1
                        if exch_lag_breach:
                            audit_replay_exch_lag_breaches += 1
                        if lag_breach:
                            audit_replay_max_lag_breaches += 1
                            if str(cadence_cfg["lag_gate_action"]) == "fail":
                                audit_replay_lag_gate_failures += 1
                            elif str(cadence_cfg["lag_gate_action"]) == "drop":
                                audit_replay_lag_gate_drops += 1
                                audit_replay_dropped_lags_ns.append(lag_ns)
                                if exch_lag_ns is not None:
                                    audit_replay_dropped_exch_lags_ns.append(exch_lag_ns)
                                if not emit_due:
                                    break
                                continue
                        cadence_lags_ns.append(lag_ns)
                        if exch_lag_ns is not None:
                            cadence_exch_lags_ns.append(exch_lag_ns)
                        pending_replay_decisions.append(
                            (
                                consumed_decision_ts,
                                consumed_feed_ts,
                                lag_ns,
                                ts_local,
                                current_feed_ts_exch,
                            )
                        )
                        if not emit_due:
                            break
                    if not pending_replay_decisions:
                        if not had_due:
                            cadence_skipped_feed_events += 1
                        continue
                (
                    decision_ts,
                    replay_scheduled_ts_local,
                    replay_lag_ns,
                    bt_feed_ts_local,
                    bt_feed_ts_exch,
                ) = pending_replay_decisions.pop(0)
            elif _should_skip_strategy_decision(ts_local, last_decision_ts, cadence_interval_ns):
                cadence_skipped_feed_events += 1
                continue
            else:
                decision_ts = ts_local
                replay_scheduled_ts_local = 0
                replay_lag_ns = 0
                bt_feed_ts_local = 0
                bt_feed_ts_exch = 0

            strategy_seq += 1
            last_decision_ts = decision_ts

            tick_size = float(depth.tick_size)
            lot_size = float(depth.lot_size)
            spread = best_ask - best_bid
            mid = 0.5 * (best_bid + best_ask)

            bid_size, ask_size = compute_top5_size(depth)
            live_market_state = live_market_state_by_decision_ts.get(decision_ts)
            if live_market_state is not None:
                best_bid = float(live_market_state.best_bid)
                best_ask = float(live_market_state.best_ask)
                mid = float(live_market_state.mid)
                spread = best_ask - best_bid
                bid_size = float(live_market_state.bid_size)
                ask_size = float(live_market_state.ask_size)
                audit_replay_market_state_overlay_count += 1

            sigma = sigma_est.update(decision_ts, mid)
            engine_position = float(hbt.position(0))
            if decision_ts in live_strategy_position_by_decision_ts:
                position = float(live_strategy_position_by_decision_ts[decision_ts])
                audit_replay_strategy_position_overlay_count += 1
            else:
                position = engine_position
            greek_values = greek_oracle.values(ts_local=decision_ts, position=position)
            greek_adjustment = (
                float(greek_cfg.get("w_delta", 0.0)) * greek_values.delta
                + float(greek_cfg.get("w_gamma", 0.0)) * greek_values.gamma
                + float(greek_cfg.get("w_vega", 0.0)) * greek_values.vega
                + float(greek_cfg.get("w_theta", 0.0)) * greek_values.theta
            )
            order_notional = float(risk["order_notional"])
            impact_cost_val = impact_cost(order_notional, config["impact"])

            if live_market_state is not None:
                fair = float(live_market_state.fair)
                reservation = float(live_market_state.reservation)
                half_spread = float(live_market_state.half_spread)
                target_bid_tick = int(live_market_state.target_bid_tick)
                target_ask_tick = int(live_market_state.target_ask_tick)
            else:
                fair = (
                    mid
                    + float(fair_cfg["w_imb"]) * (bid_size - ask_size)
                    + float(fair_cfg["w_spread"]) * spread
                    + float(fair_cfg["w_vol"]) * sigma
                    + greek_adjustment
                )

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

            position_notional = position * mid
            pos_limit = is_position_limit_reached(position=position, position_notional=position_notional, risk=risk)

            qty = max(lot_size, round((order_notional / mid) / lot_size) * lot_size)

            raw_feed_latency_ns = int(feed_lat[1] - feed_lat[0]) if feed_lat is not None else 0
            feed_latency_ns = feed_latency_oracle.feed_latency_ns(decision_ts, raw_feed_latency_ns)

            predicted_entry_ns = int(latency_oracle.entry_latency_ns(decision_ts))
            last_entry_ns = int(order_lat[1] - order_lat[0]) if order_lat is not None else 0
            last_resp_ns = int(order_lat[2] - order_lat[1]) if order_lat is not None else 0
            latency_signal_ns = _latency_guard_signal_ns(feed_latency_ns)

            dropped_by_latency = latency_signal_ns > latency_guard_ns
            dropped_by_api_limit = False

            _prune_absent_pending_submit_orders(pending_local_orders, live_order_absent_after_seen_ts, decision_ts)
            _prune_released_pending_orders(pending_cancel_overlays, decision_ts)
            _prune_released_pending_orders(pending_cancel_retention_overlays, decision_ts)
            engine_working = collect_working_orders(hbt.orders(0))
            working = _live_visible_working_orders(
                engine_working,
                list(pending_local_orders.values()),
                list(pending_cancel_overlays.values()),
                list(pending_cancel_retention_overlays.values()),
                decision_ts=decision_ts,
                absent_after_seen_ts=live_order_absent_after_seen_ts,
            )
            working_bid_tick = int(working.buy.price_tick) if working.buy is not None else -1
            working_ask_tick = int(working.sell.price_tick) if working.sell is not None else -1
            working_diagnostics = format_working_order_diagnostics(working)

            planned_actions: list[Action] = []
            executed_actions: list[Action] = []
            reject_reason = ""
            throttle_reason = ""
            planned_order_id = ""
            planned_action = "keep"
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
                    two_phase_replace_enabled=two_phase_replace_enabled,
                )
                planned_order_id, planned_action = format_actions(planned_actions)

                if planned_actions:
                    if quote_throttle_enabled:
                        throttle_reason = quote_throttle_reason(
                            actions=planned_actions,
                            state=quote_throttle,
                            ts_local=decision_ts,
                            target_bid_tick=target_bid_tick,
                            target_ask_tick=target_ask_tick,
                            min_interval_ns=quote_min_interval_ns,
                            min_move_ticks=quote_min_move_ticks,
                        )
                    if throttle_reason:
                        dropped_by_api_limit = True
                        reject_reason = "quote_throttle"
                    elif (
                        last_api_ts is not None
                        and (decision_ts - last_api_ts) < min_interval_ns
                        and not is_pure_cancel_extra(planned_actions)
                    ):
                        dropped_by_api_limit = True
                        reject_reason = "api_interval_guard"
                        throttle_reason = "api_interval"
                    else:
                        for action in planned_actions:
                            if bool(api_cfg.get("enabled", True)) and not bucket.allow(decision_ts, 1.0):
                                dropped_by_api_limit = True
                                reject_reason = "token_bucket"
                                break

                            if action.kind == "cancel":
                                lifecycle_tracker.mark_cancel_requested(action.order_id, decision_ts)
                                cancel_visible_ts = int(live_order_cancel_pending_ts.get(int(action.order_id), 0))
                                if cancel_visible_ts > decision_ts:
                                    visible_order = _find_visible_order_for_action(working, action)
                                    if visible_order is not None:
                                        absent_ts = int(live_order_absent_after_seen_ts.get(int(action.order_id), 0))
                                        pending_cancel_overlays[int(action.order_id)] = _pending_cancel_overlay_from_order(
                                            visible_order,
                                            visible_ts=cancel_visible_ts,
                                            release_ts=absent_ts if absent_ts > cancel_visible_ts else cancel_visible_ts,
                                            tick_size=tick_size,
                                            decision_ts=decision_ts,
                                        )
                                hbt.cancel(0, int(action.order_id), False)
                            elif action.kind == "submit" and action.side == "buy":
                                hbt.submit_buy_order(0, int(action.order_id), action.price, action.qty, GTX, LIMIT, False)
                                pending_local_orders[int(action.order_id)] = PendingLocalOrder(
                                    order_id=int(action.order_id),
                                    side=BUY,
                                    price=float(action.price),
                                    price_tick=round_to_tick(action.price, tick_size),
                                    qty=float(action.qty),
                                    leaves_qty=float(action.qty),
                                    local_timestamp=decision_ts,
                                    release_ts=int(live_order_release_ts.get(int(action.order_id), 0)),
                                )
                            elif action.kind == "submit" and action.side == "sell":
                                hbt.submit_sell_order(0, int(action.order_id), action.price, action.qty, GTX, LIMIT, False)
                                pending_local_orders[int(action.order_id)] = PendingLocalOrder(
                                    order_id=int(action.order_id),
                                    side=SELL,
                                    price=float(action.price),
                                    price_tick=round_to_tick(action.price, tick_size),
                                    qty=float(action.qty),
                                    leaves_qty=float(action.qty),
                                    local_timestamp=decision_ts,
                                    release_ts=int(live_order_release_ts.get(int(action.order_id), 0)),
                                )

                            executed_actions.append(action)
                            sent_api = True
                            last_api_ts = decision_ts
                            if writer is not None and audit_policy.should_write({"action": f"{action.kind}_{action.side}", "reject_reason": ""}, strategy_seq):
                                lifecycle_event_seq += 1
                                writer.writerow(
                                    build_lifecycle_event_row(
                                        run_id=run_id,
                                        symbol=symbol,
                                        strategy_seq=strategy_seq,
                                        event_seq=lifecycle_event_seq,
                                        event_type="cancel_sent" if action.kind == "cancel" else "order_submit_sent",
                                        event_source="backtest_local",
                                        ts_local=decision_ts,
                                        ts_exch=int(feed_lat[0]) if feed_lat is not None else 0,
                                        replay_scheduled_ts_local=replay_scheduled_ts_local,
                                        bt_feed_ts_local=bt_feed_ts_local,
                                        bt_feed_ts_exch=bt_feed_ts_exch,
                                        replay_lag_ns=replay_lag_ns,
                                        action=action,
                                        action_order_price_tick=round_to_tick(action.price, tick_size) if action.price > 0.0 else "",
                                        best_bid=best_bid,
                                        best_ask=best_ask,
                                        mid=mid,
                                        position=position,
                                        cancel_requested=action.kind == "cancel",
                                        cancel_request_ts=decision_ts if action.kind == "cancel" else 0,
                                        local_order_seen=True,
                                        lifecycle_detail="api_action_sent",
                                    )
                                )
                                rows_written += 1

                        if executed_actions:
                            action_order_id, action_name = format_actions(executed_actions)
                            update_quote_throttle_state(
                                quote_throttle,
                                ts_local=decision_ts,
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

            # event2order delay; when we send in same cycle it is zero in this model.
            if sent_api and req_ts > 0:
                auditlatency_ms = max(0.0, (req_ts - decision_ts) / 1_000_000.0)
            else:
                auditlatency_ms = 0.0

            spread_bps = (spread / mid) * 1e4 if mid > 0 else 0.0
            vol_bps = sigma * 1e4
            inventory_score = inventory_score_from_risk(position=position, position_notional=position_notional, risk=risk)

            if dropped_by_latency and not reject_reason:
                reject_reason = "latency_guard"
            if dropped_by_api_limit and not reject_reason:
                reject_reason = "api_limit"

            row = build_audit_row(
                run_id=run_id,
                symbol=symbol,
                strategy_seq=strategy_seq,
                ts_local=decision_ts,
                ts_exch=int(feed_lat[0]) if feed_lat is not None else 0,
                replay_scheduled_ts_local=replay_scheduled_ts_local,
                bt_feed_ts_local=bt_feed_ts_local,
                bt_feed_ts_exch=bt_feed_ts_exch,
                replay_lag_ns=replay_lag_ns,
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
                local_open_order_count=_local_open_order_count(working),
                local_open_orders=working_diagnostics["local_open_orders"],
            )

            metrics.update(row)

            row_day = utc_day_from_ns(decision_ts)
            if current_day is None:
                current_day = row_day
            elif row_day != current_day:
                daily_rows.append({"day": current_day, **flatten_summary("", day_metrics.summary())})
                day_metrics = MetricAccumulator()
                current_day = row_day
            day_metrics.update(row)

            if writer is not None and audit_policy.should_write(row, strategy_seq):
                writer.writerow(row)
                rows_written += 1
                if rows_written % int(audit_cfg.get("flush_every", 1000)) == 0 and audit_file is not None:
                    audit_file.flush()

            current_working = merge_pending_orders(
                collect_working_orders(hbt.orders(0)),
                list(pending_local_orders.values()),
            )
            current_working = _live_visible_working_orders(
                current_working,
                [],
                list(pending_cancel_overlays.values()),
                list(pending_cancel_retention_overlays.values()),
                decision_ts=decision_ts,
                absent_after_seen_ts=live_order_absent_after_seen_ts,
            )
            current_order_diagnostics = format_working_order_diagnostics(current_working)
            for lifecycle_type, order_snapshot, _prev_snapshot in lifecycle_tracker.observe(hbt.orders(0)):
                pending_order = pending_local_orders.get(order_snapshot.order_id)
                pending_release_ts = int(getattr(pending_order, "release_ts", 0) or 0) if pending_order is not None else 0
                if (
                    pending_order is not None
                    and pending_release_ts <= decision_ts
                    and (
                        order_snapshot.req != "new"
                        or order_snapshot.cancellable
                        or order_snapshot.status in {"filled", "expired", "rejected", "canceled"}
                    )
                ):
                    pending_local_orders.pop(order_snapshot.order_id, None)

                if order_snapshot.status in {"canceled", "expired", "rejected", "filled"}:
                    absent_ts = int(live_order_absent_after_seen_ts.get(order_snapshot.order_id, 0))
                    cancel_visible_ts = int(live_order_cancel_pending_ts.get(order_snapshot.order_id, 0))
                    if absent_ts > decision_ts and cancel_visible_ts > 0:
                        pending_cancel_retention_overlays[order_snapshot.order_id] = _pending_cancel_overlay_from_order(
                            order_snapshot,
                            visible_ts=cancel_visible_ts,
                            release_ts=absent_ts,
                            tick_size=tick_size,
                            decision_ts=decision_ts,
                            req=4,
                            cancellable=False,
                        )
                    else:
                        pending_cancel_retention_overlays.pop(order_snapshot.order_id, None)
                if writer is None:
                    continue
                if not audit_policy.should_write({"action": lifecycle_type, "reject_reason": ""}, strategy_seq):
                    continue
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
                        event_source="backtest_exchange",
                        ts_local=decision_ts,
                        ts_exch=order_snapshot.exch_timestamp,
                        replay_scheduled_ts_local=replay_scheduled_ts_local,
                        bt_feed_ts_local=bt_feed_ts_local,
                        bt_feed_ts_exch=bt_feed_ts_exch,
                        replay_lag_ns=replay_lag_ns,
                        order=order_snapshot,
                        linked_action=lifecycle_type,
                        linked_order_id=str(order_snapshot.order_id),
                        best_bid=best_bid,
                        best_ask=best_ask,
                        mid=mid,
                        position=position,
                        local_open_orders=current_order_diagnostics["local_open_orders"],
                        cancel_requested=cancel_request_ts > 0,
                        cancel_request_ts=cancel_request_ts,
                        cancel_ack_ts=decision_ts if lifecycle_type == "cancel_ack" else 0,
                        fill_ts=order_snapshot.exch_timestamp if is_fill_event else 0,
                        fill_qty=order_snapshot.exec_qty if is_fill_event else 0.0,
                        fill_price=(
                            order_snapshot.exec_price_tick * tick_size
                            if is_fill_event and order_snapshot.exec_price_tick != 0
                            else 0.0
                        ),
                        fill_after_cancel_request=is_fill_event and cancel_request_ts > 0,
                        local_order_seen=True,
                        lifecycle_detail="fill_after_cancel_request" if is_fill_event and cancel_request_ts > 0 else "",
                    )
                )
                rows_written += 1

            hbt.clear_inactive_orders(ALL_ASSETS)
    finally:
        if current_day is not None and day_metrics.rows > 0:
            daily_rows.append({"day": current_day, **flatten_summary("", day_metrics.summary())})
        if audit_file is not None:
            audit_file.flush()
            audit_file.close()

    summary = metrics.summary()
    replay_gate_due_count = len(audit_replay_due_lags_ns)
    replay_gate_accepted_count = len(cadence_lags_ns)
    replay_lag_gate = _empty_replay_lag_gate_stats(
        max_lag_ns=int(cadence_cfg["max_lag_ns"]),
        max_exch_lag_ns=int(cadence_cfg["max_exch_lag_ns"]),
        strict=bool(cadence_cfg["strict_lag_gate"]),
        action=str(cadence_cfg["lag_gate_action"]),
    )
    if cadence_mode == "audit_replay":
        replay_lag_gate.update(
            {
                "scheduled_count": len(audit_replay_schedule),
                "due_count": replay_gate_due_count,
                "accepted_count": replay_gate_accepted_count,
                "drop_count": audit_replay_lag_gate_drops,
                "fail_count": audit_replay_lag_gate_failures,
                "breach_count": audit_replay_max_lag_breaches,
                "local_breach_count": audit_replay_local_lag_breaches,
                "exch_breach_count": audit_replay_exch_lag_breaches,
                "missing_exch_lag_count": audit_replay_missing_exch_lag_count,
                "passed": audit_replay_lag_gate_failures == 0,
                "drop_ratio": (
                    float(audit_replay_lag_gate_drops / replay_gate_due_count)
                    if replay_gate_due_count
                    else 0.0
                ),
                "breach_ratio": (
                    float(audit_replay_max_lag_breaches / replay_gate_due_count)
                    if replay_gate_due_count
                    else 0.0
                ),
                "accepted_lag_ns": _distribution(cadence_lags_ns),
                "accepted_exch_lag_ns": _distribution(cadence_exch_lags_ns),
                "all_due_lag_ns": _distribution(audit_replay_due_lags_ns),
                "all_due_exch_lag_ns": _distribution(audit_replay_due_exch_lags_ns),
                "dropped_lag_ns": _distribution(audit_replay_dropped_lags_ns),
                "dropped_exch_lag_ns": _distribution(audit_replay_dropped_exch_lags_ns),
            }
        )
    result = {
        "run_id": run_id,
        "audit_csv": str(audit_path) if audit_policy.mode != "off" else "",
        "audit_rows": rows_written,
        "rows": summary["rows"],
        "summary": summary,
        "daily_summary_csv": str(daily_csv_path) if summary_enabled else "",
        "summary_json": str(summary_json_path) if summary_enabled else "",
        "slice_ts_local_start": slice_ts_local_start,
        "slice_ts_local_end": slice_ts_local_end,
        "backtest_cadence_mode": cadence_mode,
        "backtest_cadence_interval_ns": cadence_interval_ns,
        "audit_replay_scheduled_count": len(audit_replay_schedule),
        "audit_replay_schedule_stats": audit_replay_schedule_stats,
        "audit_replay_consumed_count": audit_replay_idx,
        "audit_replay_unconsumed_count": max(0, len(audit_replay_schedule) - audit_replay_idx),
        "audit_replay_skipped_due_count": audit_replay_skipped_due_count,
        "audit_replay_max_lag_breaches": audit_replay_max_lag_breaches,
        "audit_replay_lag_gate_drops": audit_replay_lag_gate_drops,
        "audit_replay_lag_gate_failures": audit_replay_lag_gate_failures,
        "audit_replay_local_lag_breaches": audit_replay_local_lag_breaches,
        "audit_replay_exch_lag_breaches": audit_replay_exch_lag_breaches,
        "audit_replay_missing_exch_lag_count": audit_replay_missing_exch_lag_count,
        "audit_replay_lag_gate": replay_lag_gate,
        "cadence_skipped_feed_events": cadence_skipped_feed_events,
        "audit_replay_lag_ns": _distribution(cadence_lags_ns),
        "audit_replay_exch_lag_ns": _distribution(cadence_exch_lags_ns),
        "audit_replay_due_lag_ns": _distribution(audit_replay_due_lags_ns),
        "audit_replay_due_exch_lag_ns": _distribution(audit_replay_due_exch_lags_ns),
        "audit_replay_strategy_position_overlay_count": audit_replay_strategy_position_overlay_count,
        "audit_replay_strategy_position_loaded_count": len(live_strategy_position_by_decision_ts),
        "audit_replay_market_state_overlay_mode": str(cadence_cfg.get("market_state_overlay", "off")),
        "audit_replay_market_state_overlay_count": audit_replay_market_state_overlay_count,
        "audit_replay_market_state_loaded_count": len(live_market_state_by_decision_ts),
        "alignment_initial_position": alignment_initial_position,
        "alignment_init_affects_pnl": bool(
            alignment_initial_position.get("enabled", False)
            and alignment_initial_position.get("success", False)
        ),
        "market_data_replay": market_data_replay_stats,
        **continuous_metadata,
    }

    if summary_enabled:
        write_daily_csv(daily_csv_path, daily_rows)
        write_json(summary_json_path, result)

    if audit_replay_lag_gate_failures > 0:
        raise RuntimeError(
            "audit replay lag gate failed: "
            f"{audit_replay_lag_gate_failures} breaches over "
            f"local={int(cadence_cfg['max_lag_ns'])}ns "
            f"exchange={int(cadence_cfg['max_exch_lag_ns'])}ns"
        )

    return result



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Binance tick MM backtest with audit")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    parser.add_argument("--manifest", required=True, help="Path to prepared data manifest JSON")
    parser.add_argument("--window", default=None, help="Optional override: first_5m|first_2h|first_6h|full_day")
    parser.add_argument("--slice-ts-local-start", type=int, default=None, help="Inclusive absolute local_ts lower bound for same-window replay")
    parser.add_argument("--slice-ts-local-end", type=int, default=None, help="Inclusive absolute local_ts upper bound for same-window replay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_toml(_expand(args.config))
    manifest = _load_manifest(_expand(args.manifest))

    result = run_backtest(
        config=config,
        manifest=manifest,
        window_override=args.window,
        slice_ts_local_start=args.slice_ts_local_start,
        slice_ts_local_end=args.slice_ts_local_end,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
