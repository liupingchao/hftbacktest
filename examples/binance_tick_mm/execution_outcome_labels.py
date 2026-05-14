#!/usr/bin/env python3
"""Read-only Stage 5 maker execution outcome label runner."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TASK_ID = "0514T005"
DEFAULT_HORIZONS_MS = (100, 500, 1_000, 5_000)
DEFAULT_TICK_SIZE = 0.1
DEFAULT_MAX_FUTURE_GAP_MS = 250.0
DEFAULT_MAKER_FEE_BPS = 0.0
LOW_SAMPLE_FILL_ROWS = 75
STAT_SIGNALS = (
    "edge_vs_fair_ticks",
    "distance_to_bbo_ticks",
    "top5_imbalance",
    "inventory_score",
)


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _ts_ns(value: Any) -> int | None:
    parsed = _int(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_div(num: float, denom: float) -> float:
    if not _finite(num) or not _finite(denom) or denom == 0.0:
        return math.nan
    return float(num) / float(denom)


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return math.nan
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _quantile(values: Iterable[float], q: float) -> float:
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return math.nan
    return float(np.quantile(np.asarray(finite, dtype=np.float64), q))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return math.nan
    return _corr(_rank(x), _rank(y))


def _parse_pipe_floats(value: Any) -> tuple[float, ...]:
    if value is None:
        return ()
    out: list[float] = []
    for item in str(value).split("|"):
        item = item.strip()
        if not item:
            continue
        number = _float(item)
        if _finite(number):
            out.append(number)
    return tuple(out)


def _parse_pipe_ints(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    out: list[int] = []
    for item in str(value).split("|"):
        number = _int(item)
        if number is not None:
            out.append(number)
    return tuple(out)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _side_sign_from_text(*values: Any) -> int:
    text = "|".join(str(value or "").lower() for value in values)
    has_buy = "buy" in text
    has_sell = "sell" in text
    if has_buy and not has_sell:
        return 1
    if has_sell and not has_buy:
        return -1
    return 0


def _asof_index(ts_values: list[int], ts: int) -> int | None:
    idx = bisect.bisect_right(ts_values, ts) - 1
    return idx if idx >= 0 else None


def _future_index(ts_values: list[int], ts: int) -> int | None:
    idx = bisect.bisect_left(ts_values, ts)
    return idx if idx < len(ts_values) else None


def _bucketize(values: list[float], buckets: int = 5) -> tuple[np.ndarray, np.ndarray]:
    if not values:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.int32)
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, np.linspace(0.0, 1.0, buckets + 1))
    bucket_ids = np.ones(len(array), dtype=np.int32)
    for bucket in range(1, buckets):
        bucket_ids[array >= quantiles[bucket]] = bucket + 1
    return quantiles, bucket_ids


def _monotonic_score(bucket_means: list[float]) -> float:
    finite = [value for value in bucket_means if _finite(value)]
    if len(finite) < 2:
        return math.nan
    diffs = np.diff(np.asarray(finite, dtype=np.float64))
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero) == 0:
        return 1.0
    return float(max(np.mean(nonzero > 0), np.mean(nonzero < 0)))


def _fee_ticks(mid_price: float, tick_size: float, maker_fee_bps: float) -> float:
    if not (_finite(mid_price) and _finite(tick_size) and _finite(maker_fee_bps)):
        return math.nan
    if tick_size <= 0.0:
        return math.nan
    fee_price = mid_price * maker_fee_bps / 10_000.0
    return fee_price / tick_size


def _price_to_tick(price: float, tick_size: float) -> int | None:
    if not (_finite(price) and _finite(tick_size)) or tick_size <= 0.0:
        return None
    return int(round(price / tick_size))


def _first_tick(side: str, bid_ticks: tuple[int, ...], ask_ticks: tuple[int, ...], best_price: float, tick_size: float) -> int | None:
    if side == "buy":
        if bid_ticks:
            return bid_ticks[0]
        return _price_to_tick(best_price, tick_size)
    if ask_ticks:
        return ask_ticks[0]
    return _price_to_tick(best_price, tick_size)


def _count_window(prefix: list[int], ts_values: list[int], start_ts: int, end_ts: int) -> int:
    left = bisect.bisect_left(ts_values, start_ts)
    right = bisect.bisect_right(ts_values, end_ts)
    return prefix[right] - prefix[left]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_audit_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("audit_live_*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one audit_live_*.csv under {run_dir}, found {len(candidates)}")
    return candidates[0]


def load_joined_decisions(path: Path) -> dict[int, dict[str, str]]:
    joined: dict[int, dict[str, str]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seq = _int(row.get("strategy_seq"))
            if seq is not None:
                joined[seq] = row
    return joined


def load_audit_rows(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _select_submit_row(events: list[dict[str, str]]) -> dict[str, str] | None:
    for row in events:
        if row.get("event_type") == "order_submit_sent":
            return row
    for row in events:
        if row.get("event_type") == "decision":
            continue
        action = str(row.get("action") or "").strip()
        if not action.startswith("submit_"):
            continue
        if not str(row.get("order_side") or "").strip():
            continue
        if _int(row.get("order_price_tick")) is None and not _finite(_float(row.get("order_price"))):
            continue
        return row
    return None


def build_decision_index(
    audit_rows: list[dict[str, str]],
    joined_rows: dict[int, dict[str, str]],
    *,
    tick_size: float,
) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    for row in audit_rows:
        if row.get("event_type") != "decision":
            continue
        strategy_seq = _int(row.get("strategy_seq"))
        ts_local = _ts_ns(row.get("ts_local"))
        if strategy_seq is None or ts_local is None:
            continue
        joined = joined_rows.get(strategy_seq, {})
        bid_qtys = _parse_pipe_floats(row.get("bid_top5_qtys"))
        ask_qtys = _parse_pipe_floats(row.get("ask_top5_qtys"))
        bid_ticks = _parse_pipe_ints(row.get("bid_top5_ticks"))
        ask_ticks = _parse_pipe_ints(row.get("ask_top5_ticks"))
        top1_bid_qty = bid_qtys[0] if bid_qtys else math.nan
        top1_ask_qty = ask_qtys[0] if ask_qtys else math.nan
        top5_bid_qty = float(sum(bid_qtys[:5])) if bid_qtys else math.nan
        top5_ask_qty = float(sum(ask_qtys[:5])) if ask_qtys else math.nan
        top1_imbalance = _safe_div(top1_bid_qty - top1_ask_qty, top1_bid_qty + top1_ask_qty)
        top5_imbalance = _safe_div(top5_bid_qty - top5_ask_qty, top5_bid_qty + top5_ask_qty)
        decision = {
            "strategy_seq": strategy_seq,
            "ts_local": ts_local,
            "mid": _float(row.get("mid")),
            "fair": _float(row.get("fair")),
            "reservation": _float(row.get("reservation")),
            "best_bid": _float(row.get("best_bid")),
            "best_ask": _float(row.get("best_ask")),
            "position": _float(row.get("position")),
            "inventory_score": _float(row.get("inventory_score")),
            "feed_latency_ms": _safe_div(_float(row.get("feed_latency_ns")), 1_000_000.0),
            "latency_signal_ms": _float(row.get("latency_signal_ms")),
            "book_view_stale_ms": _float(row.get("book_view_stale_ms")),
            "spread_bps": _float(row.get("spread_bps")),
            "vol_bps": _float(row.get("vol_bps")),
            "bid_size": _float(row.get("bid_size")),
            "ask_size": _float(row.get("ask_size")),
            "bid_top5_ticks": bid_ticks,
            "ask_top5_ticks": ask_ticks,
            "bid_top5_qtys": bid_qtys,
            "ask_top5_qtys": ask_qtys,
            "top1_bid_qty": top1_bid_qty,
            "top1_ask_qty": top1_ask_qty,
            "top5_bid_qty": top5_bid_qty,
            "top5_ask_qty": top5_ask_qty,
            "top1_imbalance": top1_imbalance,
            "top5_imbalance": top5_imbalance,
            "reject_reason": str(row.get("reject_reason") or "").strip(),
            "throttle_reason": str(row.get("throttle_reason") or "").strip(),
            "action": str(row.get("action") or "").strip(),
            "planned_action": str(row.get("planned_action") or "").strip(),
            "target_bid_tick": _int(row.get("target_bid_tick")),
            "target_ask_tick": _int(row.get("target_ask_tick")),
            "working_bid_tick": _int(row.get("working_bid_tick")),
            "working_ask_tick": _int(row.get("working_ask_tick")),
            "top5_join_age_ms": _float(joined.get("top5_join_age_ms")),
            "depth_join_age_ms": _float(joined.get("depth_join_age_ms")),
            "bookticker_join_age_ms": _float(joined.get("bookticker_join_age_ms")),
            "max_join_age_ms": _float(joined.get("max_join_age_ms")),
            "join_stale": _parse_bool(joined.get("join_stale")),
            "join_gap_crossed": _parse_bool(joined.get("join_gap_crossed")),
            "join_missing": _parse_bool(joined.get("join_missing")) or not bool(joined),
            "join_used_future": _parse_bool(joined.get("join_used_future")),
            "market_view_source": str(row.get("market_view_source") or "").strip(),
            "top5_source": str(row.get("top5_source") or "").strip(),
        }
        decisions.append(decision)
    decisions.sort(key=lambda item: (item["ts_local"], item["strategy_seq"]))
    ts_values = [int(item["ts_local"]) for item in decisions]
    by_seq = {int(item["strategy_seq"]): item for item in decisions}
    reject_prefix = [0]
    throttle_prefix = [0]
    for item in decisions:
        reject_prefix.append(reject_prefix[-1] + (1 if item["reject_reason"] else 0))
        throttle_prefix.append(throttle_prefix[-1] + (1 if item["throttle_reason"] else 0))
    return {
        "decisions": decisions,
        "ts_values": ts_values,
        "by_seq": by_seq,
        "reject_prefix": reject_prefix,
        "throttle_prefix": throttle_prefix,
    }


def _match_decision_context(
    submit_row: dict[str, str],
    decision_index: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    linked_seq = _int(submit_row.get("linked_strategy_seq"))
    strategy_seq = _int(submit_row.get("strategy_seq"))
    ts_local = _ts_ns(submit_row.get("ts_local"))
    by_seq = decision_index["by_seq"]
    if linked_seq is not None and linked_seq in by_seq:
        return by_seq[linked_seq], "linked_strategy_seq"
    if strategy_seq is not None and strategy_seq in by_seq:
        return by_seq[strategy_seq], "strategy_seq"
    if ts_local is None:
        return None, "missing"
    idx = _asof_index(decision_index["ts_values"], ts_local)
    if idx is None:
        return None, "missing"
    return decision_index["decisions"][idx], "asof_previous"


def _future_decision(
    decision_index: dict[str, Any],
    target_ts: int,
    *,
    max_future_gap_ns: int,
) -> dict[str, Any] | None:
    idx = _future_index(decision_index["ts_values"], target_ts)
    if idx is None:
        return None
    decision = decision_index["decisions"][idx]
    if int(decision["ts_local"]) - target_ts > max_future_gap_ns:
        return None
    return decision


def _recent_counts(decision_index: dict[str, Any], ts_local: int, window_ms: int) -> tuple[int, int]:
    start_ts = ts_local - int(window_ms * 1_000_000)
    end_ts = ts_local
    reject_count = _count_window(decision_index["reject_prefix"], decision_index["ts_values"], start_ts, end_ts)
    throttle_count = _count_window(decision_index["throttle_prefix"], decision_index["ts_values"], start_ts, end_ts)
    return reject_count, throttle_count


def _inventory_cycle(
    decision_index: dict[str, Any],
    *,
    fill_ts_local: int | None,
    position_after_fill: float,
) -> tuple[float, float, int, int]:
    if fill_ts_local is None or not _finite(position_after_fill):
        return math.nan, math.nan, 0, 0
    if abs(position_after_fill) <= 1e-12:
        return 0.0, abs(position_after_fill), 0, 1
    idx = _future_index(decision_index["ts_values"], fill_ts_local)
    if idx is None:
        return math.nan, abs(position_after_fill), 0, 0
    sign_after = 1 if position_after_fill > 0 else -1
    max_abs_position = abs(position_after_fill)
    crossed_zero = 0
    time_to_flat_ms = math.nan
    found_flat = 0
    for item in decision_index["decisions"][idx:]:
        position = _float(item.get("position"))
        if not _finite(position):
            continue
        max_abs_position = max(max_abs_position, abs(position))
        if position * sign_after <= 0.0:
            crossed_zero = 1
        if abs(position) <= 1e-12:
            found_flat = 1
            time_to_flat_ms = (int(item["ts_local"]) - fill_ts_local) / 1_000_000.0
            break
    return time_to_flat_ms, max_abs_position, crossed_zero, found_flat


def _final_state(
    *,
    full_fill: int,
    partial_fill: int,
    cancel_ack_count: int,
    has_expired: bool,
    missing_lifecycle: int,
) -> str:
    if full_fill:
        return "filled"
    if partial_fill and cancel_ack_count:
        return "partial_canceled"
    if partial_fill:
        return "partial_open"
    if has_expired:
        return "expired"
    if cancel_ack_count:
        return "canceled"
    if missing_lifecycle:
        return "open_or_missing"
    return "open_or_missing"


def build_execution_labels(
    *,
    audit_rows: list[dict[str, str]],
    decision_index: dict[str, Any],
    horizons_ms: Iterable[int],
    tick_size: float,
    max_future_gap_ms: float,
    maker_fee_bps: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    order_events: dict[str, list[dict[str, str]]] = defaultdict(list)
    sample_end_ts = 0
    for row in audit_rows:
        row_ts = _ts_ns(row.get("ts_local")) or 0
        sample_end_ts = max(sample_end_ts, row_ts)
        order_id = str(row.get("order_id") or "").strip()
        if order_id:
            order_events[order_id].append(row)

    max_future_gap_ns = int(max_future_gap_ms * 1_000_000)
    horizons = [int(horizon) for horizon in horizons_ms]
    execution_rows: list[dict[str, Any]] = []
    fill_horizon_rows: list[dict[str, Any]] = []
    fill_markout_rows: list[dict[str, Any]] = []

    order_ids = sorted(order_events, key=lambda item: min(_ts_ns(row.get("ts_local")) or 0 for row in order_events[item]))

    for order_id in order_ids:
        events = sorted(order_events[order_id], key=lambda row: (_ts_ns(row.get("ts_local")) or 0, row.get("event_type") or ""))
        submit_row = _select_submit_row(events)
        if submit_row is None:
            continue

        submit_ts_local = _ts_ns(submit_row.get("ts_local"))
        if submit_ts_local is None:
            continue
        decision_ctx, context_match = _match_decision_context(submit_row, decision_index)
        side_text = str(submit_row.get("order_side") or "").strip().lower()
        if side_text not in {"buy", "sell"}:
            side_sign = _side_sign_from_text(
                side_text,
                submit_row.get("action"),
                submit_row.get("linked_action"),
            )
            side_text = "buy" if side_sign > 0 else "sell" if side_sign < 0 else ""
        side_sign = 1 if side_text == "buy" else -1 if side_text == "sell" else 0
        order_price = _float(submit_row.get("order_price"))
        order_price_tick = _int(submit_row.get("order_price_tick"))
        order_qty = _float(submit_row.get("order_qty"))

        cancel_request_ts_values = sorted(
            {
                ts
                for ts in (
                    _ts_ns(row.get("cancel_request_ts")) or (
                        _ts_ns(row.get("ts_local"))
                        if row.get("event_type") == "cancel_sent"
                        else None
                    )
                    for row in events
                )
                if ts is not None
            }
        )
        cancel_ack_ts_values = sorted(
            {
                ts
                for ts in (
                    _ts_ns(row.get("cancel_ack_ts")) or (
                        _ts_ns(row.get("ts_local"))
                        if row.get("event_type") == "cancel_ack"
                        else None
                    )
                    for row in events
                )
                if ts is not None
            }
        )
        fill_events = [row for row in events if row.get("event_type") == "fill"]
        fill_events.sort(key=lambda row: _ts_ns(row.get("ts_local")) or 0)
        first_fill = fill_events[0] if fill_events else None
        fill_ts_local = _ts_ns(first_fill.get("ts_local")) if first_fill else None
        fill_qtys = [_float(row.get("fill_qty")) for row in fill_events]
        fill_prices = [_float(row.get("fill_price")) for row in fill_events]
        total_fill_qty = float(sum(value for value in fill_qtys if _finite(value)))
        total_fill_notional = float(
            sum(
                qty * price
                for qty, price in zip(fill_qtys, fill_prices, strict=False)
                if _finite(qty) and _finite(price)
            )
        )
        avg_fill_price = _safe_div(total_fill_notional, total_fill_qty)
        executed_qty_candidates = [_float(row.get("order_executed_qty")) for row in events if _finite(_float(row.get("order_executed_qty")))]
        executed_qty = max(executed_qty_candidates, default=total_fill_qty if total_fill_qty > 0.0 else 0.0)
        remaining_qty_candidates = [_float(row.get("order_remaining_qty")) for row in events if _finite(_float(row.get("order_remaining_qty")))]
        remaining_qty = remaining_qty_candidates[-1] if remaining_qty_candidates else max(order_qty - executed_qty, 0.0) if _finite(order_qty) and _finite(executed_qty) else math.nan
        fill_ratio = _safe_div(executed_qty, order_qty)
        full_fill = 1 if _finite(fill_ratio) and fill_ratio >= 0.999 else 0
        partial_fill = 1 if _finite(fill_ratio) and 0.0 < fill_ratio < 0.999 else 0
        no_fill = 1 if executed_qty <= 0.0 else 0
        fill_after_cancel_request = 1 if any(_parse_bool(row.get("fill_after_cancel_request")) for row in fill_events) else 0
        cancel_request_ts_local = cancel_request_ts_values[0] if cancel_request_ts_values else None
        cancel_ack_ts_local = cancel_ack_ts_values[0] if cancel_ack_ts_values else None
        cancel_to_fill_delay_ms = (
            (fill_ts_local - cancel_request_ts_local) / 1_000_000.0
            if fill_after_cancel_request and fill_ts_local is not None and cancel_request_ts_local is not None
            else math.nan
        )
        cancel_ack_before_fill = 1 if cancel_ack_ts_local is not None and fill_ts_local is not None and cancel_ack_ts_local <= fill_ts_local else 0
        cancel_ack_after_fill = 1 if cancel_ack_ts_local is not None and fill_ts_local is not None and cancel_ack_ts_local > fill_ts_local else 0
        order_new_count = sum(1 for row in events if row.get("event_type") == "order_new")
        order_update_count = sum(1 for row in events if row.get("event_type") == "order_update")
        cancel_request_count = len(cancel_request_ts_values)
        cancel_ack_count = len(cancel_ack_ts_values)
        fill_count = len(fill_events)
        has_expired = any(row.get("event_type") == "expired" or str(row.get("order_status") or "").strip().lower() == "expired" for row in events)
        terminal_candidates = [ts for ts in (fill_ts_local, cancel_ack_ts_local) if ts is not None]
        if has_expired:
            terminal_candidates.extend(
                ts for ts in (_ts_ns(row.get("ts_local")) for row in events if row.get("event_type") == "expired") if ts is not None
            )
        explicit_terminal_ts_local = max(terminal_candidates) if terminal_candidates else None
        final_event_ts = _ts_ns(events[-1].get("ts_local")) if events else None
        terminal_ts_local = explicit_terminal_ts_local if explicit_terminal_ts_local is not None else final_event_ts
        missing_lifecycle = 1 if not terminal_candidates and not has_expired and full_fill == 0 and cancel_ack_count == 0 else 0
        final_order_state = _final_state(
            full_fill=full_fill,
            partial_fill=partial_fill,
            cancel_ack_count=cancel_ack_count,
            has_expired=has_expired,
            missing_lifecycle=missing_lifecycle,
        )
        lifetime_end = terminal_ts_local if terminal_ts_local is not None else sample_end_ts
        working_lifetime_ms = (lifetime_end - submit_ts_local) / 1_000_000.0 if lifetime_end >= submit_ts_local else math.nan
        fast_cancel_churn = 1 if cancel_request_ts_local is not None and cancel_request_ts_local - submit_ts_local <= 1_000_000_000 else 0

        submit_mid = _float(decision_ctx.get("mid")) if decision_ctx else _float(submit_row.get("mid"))
        submit_fair = _float(decision_ctx.get("fair")) if decision_ctx else _float(submit_row.get("fair"))
        submit_reservation = _float(decision_ctx.get("reservation")) if decision_ctx else _float(submit_row.get("reservation"))
        submit_best_bid = _float(decision_ctx.get("best_bid")) if decision_ctx else _float(submit_row.get("best_bid"))
        submit_best_ask = _float(decision_ctx.get("best_ask")) if decision_ctx else _float(submit_row.get("best_ask"))
        submit_position = _float(decision_ctx.get("position")) if decision_ctx else _float(submit_row.get("position"))
        inventory_score = _float(decision_ctx.get("inventory_score")) if decision_ctx else _float(submit_row.get("inventory_score"))
        feed_latency_ms = _float(decision_ctx.get("feed_latency_ms")) if decision_ctx else _safe_div(_float(submit_row.get("feed_latency_ns")), 1_000_000.0)
        latency_signal_ms = _float(decision_ctx.get("latency_signal_ms")) if decision_ctx else _float(submit_row.get("latency_signal_ms"))
        book_view_stale_ms = _float(decision_ctx.get("book_view_stale_ms")) if decision_ctx else _float(submit_row.get("book_view_stale_ms"))
        top1_bid_qty = _float(decision_ctx.get("top1_bid_qty")) if decision_ctx else math.nan
        top1_ask_qty = _float(decision_ctx.get("top1_ask_qty")) if decision_ctx else math.nan
        top5_bid_qty = _float(decision_ctx.get("top5_bid_qty")) if decision_ctx else math.nan
        top5_ask_qty = _float(decision_ctx.get("top5_ask_qty")) if decision_ctx else math.nan
        top5_imbalance = _float(decision_ctx.get("top5_imbalance")) if decision_ctx else math.nan
        best_bid_tick = _first_tick(
            "buy",
            tuple(decision_ctx.get("bid_top5_ticks", ())) if decision_ctx else (),
            tuple(decision_ctx.get("ask_top5_ticks", ())) if decision_ctx else (),
            submit_best_bid,
            tick_size,
        )
        best_ask_tick = _first_tick(
            "sell",
            tuple(decision_ctx.get("bid_top5_ticks", ())) if decision_ctx else (),
            tuple(decision_ctx.get("ask_top5_ticks", ())) if decision_ctx else (),
            submit_best_ask,
            tick_size,
        )
        target_tick_same_side = None
        working_tick_same_side = None
        if decision_ctx:
            if side_text == "buy":
                target_tick_same_side = decision_ctx.get("target_bid_tick")
                working_tick_same_side = decision_ctx.get("working_bid_tick")
            elif side_text == "sell":
                target_tick_same_side = decision_ctx.get("target_ask_tick")
                working_tick_same_side = decision_ctx.get("working_ask_tick")
        distance_to_bbo_ticks = math.nan
        if order_price_tick is not None:
            if side_text == "buy" and best_bid_tick is not None:
                distance_to_bbo_ticks = float(best_bid_tick - order_price_tick)
            elif side_text == "sell" and best_ask_tick is not None:
                distance_to_bbo_ticks = float(order_price_tick - best_ask_tick)
        if _finite(distance_to_bbo_ticks):
            if distance_to_bbo_ticks < 0.0:
                placement_bucket = "inside_or_crossed"
            elif distance_to_bbo_ticks == 0.0:
                placement_bucket = "touch"
            elif distance_to_bbo_ticks == 1.0:
                placement_bucket = "step_back_1"
            else:
                placement_bucket = "step_back_gt1"
        else:
            placement_bucket = "unknown"
        post_only_risk = 0
        if order_price_tick is not None:
            if side_text == "buy" and best_ask_tick is not None and order_price_tick >= best_ask_tick:
                post_only_risk = 1
            if side_text == "sell" and best_bid_tick is not None and order_price_tick <= best_bid_tick:
                post_only_risk = 1
        target_offset_ticks = math.nan
        if order_price_tick is not None and target_tick_same_side is not None:
            if side_text == "buy":
                target_offset_ticks = float(target_tick_same_side - order_price_tick)
            elif side_text == "sell":
                target_offset_ticks = float(order_price_tick - target_tick_same_side)
        if _finite(target_offset_ticks):
            if target_offset_ticks < 0.0:
                target_offset_bucket = "improved_inside"
            elif target_offset_ticks == 0.0:
                target_offset_bucket = "target_match"
            elif target_offset_ticks == 1.0:
                target_offset_bucket = "backoff_1"
            else:
                target_offset_bucket = "backoff_gt1"
        else:
            target_offset_bucket = "unknown"
        edge_vs_mid_ticks = side_sign * _safe_div(submit_mid - order_price, tick_size)
        edge_vs_fair_ticks = side_sign * _safe_div(submit_fair - order_price, tick_size)
        edge_vs_reservation_ticks = side_sign * _safe_div(submit_reservation - order_price, tick_size)
        same_side_top1_qty = top1_bid_qty if side_text == "buy" else top1_ask_qty
        opposite_side_top1_qty = top1_ask_qty if side_text == "buy" else top1_bid_qty
        same_side_top5_qty = top5_bid_qty if side_text == "buy" else top5_ask_qty
        opposite_side_top5_qty = top5_ask_qty if side_text == "buy" else top5_bid_qty
        recent_reject_count_500ms, recent_throttle_count_500ms = _recent_counts(decision_index, submit_ts_local, 500)
        recent_reject_count_5000ms, recent_throttle_count_5000ms = _recent_counts(decision_index, submit_ts_local, 5_000)

        fill_mid_at_fill = _float(first_fill.get("mid")) if first_fill else math.nan
        if not _finite(fill_mid_at_fill) and fill_ts_local is not None:
            asof_idx = _asof_index(decision_index["ts_values"], fill_ts_local)
            if asof_idx is not None:
                fill_mid_at_fill = _float(decision_index["decisions"][asof_idx].get("mid"))
        realized_spread_proxy_ticks = side_sign * _safe_div(fill_mid_at_fill - avg_fill_price, tick_size) if fill_ts_local is not None else math.nan
        fee_ticks_at_fill = _fee_ticks(fill_mid_at_fill, tick_size, maker_fee_bps) if fill_ts_local is not None else math.nan
        fee_adjusted_realized_spread_ticks = realized_spread_proxy_ticks - fee_ticks_at_fill if fill_ts_local is not None else math.nan
        position_after_fill = _float(first_fill.get("position")) if first_fill else math.nan
        if fill_ts_local is not None and not _finite(position_after_fill) and _finite(submit_position) and _finite(total_fill_qty):
            position_after_fill = submit_position + side_sign * total_fill_qty
        position_delta_fill = position_after_fill - submit_position if _finite(position_after_fill) and _finite(submit_position) else math.nan
        inventory_increasing_fill = (
            1
            if _finite(position_after_fill) and _finite(submit_position) and abs(position_after_fill) > abs(submit_position) + 1e-12
            else 0
        )
        inventory_reducing_fill = (
            1
            if _finite(position_after_fill) and _finite(submit_position) and abs(position_after_fill) + 1e-12 < abs(submit_position)
            else 0
        )
        time_to_flat_ms, max_abs_position_until_flat, crossed_zero_after_fill, found_flat = _inventory_cycle(
            decision_index,
            fill_ts_local=fill_ts_local,
            position_after_fill=position_after_fill,
        )

        order_row: dict[str, Any] = {
            "order_id": order_id,
            "submit_strategy_seq": _int(submit_row.get("strategy_seq")),
            "linked_strategy_seq": _int(submit_row.get("linked_strategy_seq")),
            "submit_ts_local": submit_ts_local,
            "order_side": side_text,
            "side_sign": side_sign,
            "order_price": order_price,
            "order_price_tick": order_price_tick,
            "order_qty": order_qty,
            "decision_context_match": context_match,
            "decision_context_strategy_seq": decision_ctx.get("strategy_seq") if decision_ctx else "",
            "submit_mid": submit_mid,
            "submit_fair": submit_fair,
            "submit_reservation": submit_reservation,
            "submit_best_bid": submit_best_bid,
            "submit_best_ask": submit_best_ask,
            "submit_spread_ticks": _safe_div(submit_best_ask - submit_best_bid, tick_size),
            "edge_vs_mid_ticks": edge_vs_mid_ticks,
            "edge_vs_fair_ticks": edge_vs_fair_ticks,
            "edge_vs_reservation_ticks": edge_vs_reservation_ticks,
            "position_before_submit": submit_position,
            "inventory_score": inventory_score,
            "feed_latency_ms": feed_latency_ms,
            "latency_signal_ms": latency_signal_ms,
            "book_view_stale_ms": book_view_stale_ms,
            "top1_bid_qty": top1_bid_qty,
            "top1_ask_qty": top1_ask_qty,
            "top5_bid_qty": top5_bid_qty,
            "top5_ask_qty": top5_ask_qty,
            "top5_imbalance": top5_imbalance,
            "same_side_top1_qty": same_side_top1_qty,
            "opposite_side_top1_qty": opposite_side_top1_qty,
            "same_side_top5_qty": same_side_top5_qty,
            "opposite_side_top5_qty": opposite_side_top5_qty,
            "top5_join_age_ms": _float(decision_ctx.get("top5_join_age_ms")) if decision_ctx else math.nan,
            "max_join_age_ms": _float(decision_ctx.get("max_join_age_ms")) if decision_ctx else math.nan,
            "join_stale": int(_parse_bool(decision_ctx.get("join_stale"))) if decision_ctx else 0,
            "join_gap_crossed": int(_parse_bool(decision_ctx.get("join_gap_crossed"))) if decision_ctx else 0,
            "market_view_source": decision_ctx.get("market_view_source") if decision_ctx else "",
            "top5_source": decision_ctx.get("top5_source") if decision_ctx else "",
            "best_bid_tick": best_bid_tick,
            "best_ask_tick": best_ask_tick,
            "target_tick_same_side": target_tick_same_side,
            "working_tick_same_side": working_tick_same_side,
            "distance_to_bbo_ticks": distance_to_bbo_ticks,
            "placement_bucket": placement_bucket,
            "post_only_risk": post_only_risk,
            "target_offset_ticks": target_offset_ticks,
            "target_offset_bucket": target_offset_bucket,
            "order_new_count": order_new_count,
            "order_update_count": order_update_count,
            "cancel_request_count": cancel_request_count,
            "cancel_ack_count": cancel_ack_count,
            "fill_count": fill_count,
            "total_fill_qty": total_fill_qty,
            "avg_fill_price": avg_fill_price,
            "fill_ratio": fill_ratio,
            "full_fill": full_fill,
            "partial_fill": partial_fill,
            "no_fill": no_fill,
            "first_fill_ts_local": fill_ts_local,
            "cancel_request_ts_local": cancel_request_ts_local,
            "cancel_ack_ts_local": cancel_ack_ts_local,
            "terminal_ts_local": terminal_ts_local,
            "working_lifetime_ms": working_lifetime_ms,
            "final_order_state": final_order_state,
            "missing_lifecycle": missing_lifecycle,
            "fill_after_cancel_request": fill_after_cancel_request,
            "cancel_to_fill_delay_ms": cancel_to_fill_delay_ms,
            "cancel_ack_before_fill": cancel_ack_before_fill,
            "cancel_ack_after_fill": cancel_ack_after_fill,
            "recent_reject_count_500ms": recent_reject_count_500ms,
            "recent_throttle_count_500ms": recent_throttle_count_500ms,
            "recent_reject_count_5000ms": recent_reject_count_5000ms,
            "recent_throttle_count_5000ms": recent_throttle_count_5000ms,
            "fast_cancel_churn": fast_cancel_churn,
            "fill_mid_at_fill": fill_mid_at_fill,
            "realized_spread_proxy_ticks": realized_spread_proxy_ticks,
            "fee_adjusted_realized_spread_ticks": fee_adjusted_realized_spread_ticks,
            "position_after_fill": position_after_fill,
            "position_delta_fill": position_delta_fill,
            "inventory_increasing_fill": inventory_increasing_fill,
            "inventory_reducing_fill": inventory_reducing_fill,
            "time_to_flat_ms": time_to_flat_ms,
            "max_abs_position_until_flat": max_abs_position_until_flat,
            "crossed_zero_after_fill": crossed_zero_after_fill,
            "inventory_cycle_flat_observed": found_flat,
        }

        for horizon_ms in horizons:
            horizon_ts = submit_ts_local + horizon_ms * 1_000_000
            horizon_known = horizon_ts <= sample_end_ts or (
                explicit_terminal_ts_local is not None and explicit_terminal_ts_local <= horizon_ts
            )
            fill_by_horizon = 1 if fill_ts_local is not None and fill_ts_local <= horizon_ts else 0
            future_decision = _future_decision(decision_index, horizon_ts, max_future_gap_ns=max_future_gap_ns)
            future_mid = _float(future_decision.get("mid")) if future_decision else math.nan
            opportunity_cost_proxy_ticks = side_sign * _safe_div(future_mid - order_price, tick_size) if future_decision else math.nan
            missed_favorable_mid_move = 1 if future_decision and not fill_by_horizon and _finite(opportunity_cost_proxy_ticks) and opportunity_cost_proxy_ticks > 0.0 else 0
            observed_tradeoff_proxy_ticks = math.nan
            if fill_by_horizon and fill_ts_local is not None:
                observed_tradeoff_proxy_ticks = math.nan
            elif future_decision and _finite(opportunity_cost_proxy_ticks):
                observed_tradeoff_proxy_ticks = -max(opportunity_cost_proxy_ticks, 0.0)
            fill_horizon_rows.append(
                {
                    "order_id": order_id,
                    "submit_ts_local": submit_ts_local,
                    "order_side": side_text,
                    "placement_bucket": placement_bucket,
                    "distance_to_bbo_ticks": distance_to_bbo_ticks,
                    "edge_vs_fair_ticks": edge_vs_fair_ticks,
                    "horizon_ms": horizon_ms,
                    "fill_by_horizon": fill_by_horizon,
                    "time_to_fill_ms": (fill_ts_local - submit_ts_local) / 1_000_000.0 if fill_by_horizon and fill_ts_local is not None else math.nan,
                    "horizon_observable": 1 if horizon_known else 0,
                    "right_censored": 1 if not horizon_known and not fill_by_horizon else 0,
                    "tail_truncated": 1 if not horizon_known and horizon_ts > sample_end_ts else 0,
                    "missing_lifecycle": missing_lifecycle,
                    "future_mid": future_mid,
                    "future_decision_ts_local": future_decision.get("ts_local") if future_decision else "",
                    "future_gap_ms": ((int(future_decision["ts_local"]) - horizon_ts) / 1_000_000.0) if future_decision else math.nan,
                    "opportunity_cost_proxy_ticks": opportunity_cost_proxy_ticks,
                    "missed_favorable_mid_move": missed_favorable_mid_move,
                    "missed_spread_capture_proxy_ticks": max(opportunity_cost_proxy_ticks, 0.0) if _finite(opportunity_cost_proxy_ticks) and not fill_by_horizon else math.nan,
                    "conservative_quote_bucket": 1 if _finite(distance_to_bbo_ticks) and distance_to_bbo_ticks > 0.0 else 0,
                    "observed_tradeoff_proxy_ticks": observed_tradeoff_proxy_ticks,
                    "final_order_state": final_order_state,
                }
            )
            order_row[f"fill_by_{horizon_ms}ms"] = fill_by_horizon
            order_row[f"horizon_observable_{horizon_ms}ms"] = 1 if horizon_known else 0
            order_row[f"opportunity_cost_proxy_{horizon_ms}ms_ticks"] = opportunity_cost_proxy_ticks
            order_row[f"missed_favorable_mid_move_{horizon_ms}ms"] = missed_favorable_mid_move

        for horizon_ms in horizons:
            if fill_ts_local is None:
                order_row[f"fill_markout_{horizon_ms}ms_ticks"] = math.nan
                order_row[f"markout_observable_{horizon_ms}ms"] = 0
                continue
            target_ts = fill_ts_local + horizon_ms * 1_000_000
            future_decision = _future_decision(decision_index, target_ts, max_future_gap_ns=max_future_gap_ns)
            future_mid = _float(future_decision.get("mid")) if future_decision else math.nan
            inventory_mtm_component_ticks = side_sign * _safe_div(future_mid - fill_mid_at_fill, tick_size) if future_decision else math.nan
            side_adjusted_markout_ticks = side_sign * _safe_div(future_mid - avg_fill_price, tick_size) if future_decision else math.nan
            net_ev_proxy_ticks = (
                fee_adjusted_realized_spread_ticks + inventory_mtm_component_ticks
                if _finite(fee_adjusted_realized_spread_ticks) and _finite(inventory_mtm_component_ticks)
                else math.nan
            )
            fill_markout_rows.append(
                {
                    "order_id": order_id,
                    "submit_ts_local": submit_ts_local,
                    "fill_ts_local": fill_ts_local,
                    "order_side": side_text,
                    "placement_bucket": placement_bucket,
                    "distance_to_bbo_ticks": distance_to_bbo_ticks,
                    "edge_vs_fair_ticks": edge_vs_fair_ticks,
                    "top5_imbalance": top5_imbalance,
                    "inventory_score": inventory_score,
                    "horizon_ms": horizon_ms,
                    "avg_fill_price": avg_fill_price,
                    "fill_mid_at_fill": fill_mid_at_fill,
                    "future_mid": future_mid,
                    "future_decision_ts_local": future_decision.get("ts_local") if future_decision else "",
                    "future_gap_ms": ((int(future_decision["ts_local"]) - target_ts) / 1_000_000.0) if future_decision else math.nan,
                    "horizon_observable": 1 if future_decision else 0,
                    "right_censored": 1 if future_decision is None else 0,
                    "tail_truncated": 1 if future_decision is None and target_ts > sample_end_ts else 0,
                    "fill_after_cancel_request": fill_after_cancel_request,
                    "cancel_to_fill_delay_ms": cancel_to_fill_delay_ms,
                    "side_adjusted_markout_ticks": side_adjusted_markout_ticks,
                    "realized_spread_proxy_ticks": realized_spread_proxy_ticks,
                    "fee_adjusted_realized_spread_ticks": fee_adjusted_realized_spread_ticks,
                    "inventory_mtm_component_ticks_proxy": inventory_mtm_component_ticks,
                    "net_ev_proxy_ticks": net_ev_proxy_ticks,
                }
            )
            order_row[f"fill_markout_{horizon_ms}ms_ticks"] = side_adjusted_markout_ticks
            order_row[f"markout_observable_{horizon_ms}ms"] = 1 if future_decision else 0

        execution_rows.append(order_row)

    summary = {
        "submit_orders": len(execution_rows),
        "filled_orders": sum(1 for row in execution_rows if int(row["fill_count"]) > 0),
        "fill_after_cancel_orders": sum(int(row["fill_after_cancel_request"]) for row in execution_rows),
        "partial_fill_orders": sum(int(row["partial_fill"]) for row in execution_rows),
        "missing_lifecycle_orders": sum(int(row["missing_lifecycle"]) for row in execution_rows),
        "sample_end_ts_local": sample_end_ts,
    }
    for horizon_ms in horizons:
        summary[f"fill_by_{horizon_ms}ms_orders"] = sum(int(row[f"fill_by_{horizon_ms}ms"]) for row in execution_rows)
        summary[f"fill_observable_{horizon_ms}ms_orders"] = sum(int(row[f"horizon_observable_{horizon_ms}ms"]) for row in execution_rows)
        summary[f"markout_observable_{horizon_ms}ms_orders"] = sum(int(row.get(f"markout_observable_{horizon_ms}ms", 0)) for row in execution_rows)
    return execution_rows, fill_horizon_rows, fill_markout_rows, summary


def build_lifecycle_summary(execution_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    state_counts = Counter(str(row["final_order_state"]) for row in execution_rows)
    total = len(execution_rows)
    rows: list[dict[str, Any]] = []
    for state, count in sorted(state_counts.items()):
        rows.append(
            {
                "category": "final_order_state",
                "label": state,
                "count": count,
                "rate": _safe_div(count, total),
                "mean_value": math.nan,
            }
        )
    metrics = [
        ("full_fill", sum(int(row["full_fill"]) for row in execution_rows)),
        ("partial_fill", sum(int(row["partial_fill"]) for row in execution_rows)),
        ("no_fill", sum(int(row["no_fill"]) for row in execution_rows)),
        ("fill_after_cancel_request", sum(int(row["fill_after_cancel_request"]) for row in execution_rows)),
        ("cancel_ack_before_fill", sum(int(row["cancel_ack_before_fill"]) for row in execution_rows)),
        ("cancel_ack_after_fill", sum(int(row["cancel_ack_after_fill"]) for row in execution_rows)),
        ("fast_cancel_churn", sum(int(row["fast_cancel_churn"]) for row in execution_rows)),
    ]
    for label, count in metrics:
        rows.append(
            {
                "category": "binary_metric",
                "label": label,
                "count": count,
                "rate": _safe_div(count, total),
                "mean_value": math.nan,
            }
        )
    rows.append(
        {
            "category": "continuous_metric",
            "label": "working_lifetime_ms",
            "count": total,
            "rate": math.nan,
            "mean_value": _mean(_float(row["working_lifetime_ms"]) for row in execution_rows),
        }
    )
    rows.append(
        {
            "category": "continuous_metric",
            "label": "fill_ratio",
            "count": total,
            "rate": math.nan,
            "mean_value": _mean(_float(row["fill_ratio"]) for row in execution_rows),
        }
    )
    return rows


def build_censoring_summary(
    execution_rows: list[dict[str, Any]],
    fill_horizon_rows: list[dict[str, Any]],
    fill_markout_rows: list[dict[str, Any]],
    horizons_ms: Iterable[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_ms in horizons_ms:
        horizon_fill = [row for row in fill_horizon_rows if int(row["horizon_ms"]) == int(horizon_ms)]
        horizon_markout = [row for row in fill_markout_rows if int(row["horizon_ms"]) == int(horizon_ms)]
        rows.append(
            {
                "label_family": "fill_probability",
                "horizon_ms": int(horizon_ms),
                "total_rows": len(horizon_fill),
                "observable_rows": sum(int(row["horizon_observable"]) for row in horizon_fill),
                "right_censored_rows": sum(int(row["right_censored"]) for row in horizon_fill),
                "tail_truncated_rows": sum(int(row["tail_truncated"]) for row in horizon_fill),
                "missing_lifecycle_rows": sum(int(row["missing_lifecycle"]) for row in horizon_fill),
                "positive_rows": sum(int(row["fill_by_horizon"]) for row in horizon_fill),
            }
        )
        rows.append(
            {
                "label_family": "fill_markout",
                "horizon_ms": int(horizon_ms),
                "total_rows": len(horizon_markout),
                "observable_rows": sum(int(row["horizon_observable"]) for row in horizon_markout),
                "right_censored_rows": sum(int(row["right_censored"]) for row in horizon_markout),
                "tail_truncated_rows": sum(int(row["tail_truncated"]) for row in horizon_markout),
                "missing_lifecycle_rows": 0,
                "positive_rows": sum(
                    1
                    for row in horizon_markout
                    if _finite(row["side_adjusted_markout_ticks"]) and float(row["side_adjusted_markout_ticks"]) > 0.0
                ),
            }
        )
    rows.append(
        {
            "label_family": "inventory_cycle",
            "horizon_ms": "",
            "total_rows": len(execution_rows),
            "observable_rows": sum(int(row["inventory_cycle_flat_observed"]) for row in execution_rows),
            "right_censored_rows": sum(1 for row in execution_rows if int(row["fill_count"]) > 0 and not int(row["inventory_cycle_flat_observed"])),
            "tail_truncated_rows": 0,
            "missing_lifecycle_rows": sum(int(row["missing_lifecycle"]) for row in execution_rows),
            "positive_rows": sum(int(row["crossed_zero_after_fill"]) for row in execution_rows),
        }
    )
    return rows


def _append_stat(
    stats: list[dict[str, Any]],
    *,
    label_class: str,
    label_name: str,
    label_kind: str,
    signal_name: str = "",
    conditioning: str = "",
    horizon_ms: int | str = "",
    stat_name: str,
    bucket_or_level: str = "",
    rows: int = 0,
    positive_rows: int | str = "",
    exposure_ms: float | str = "",
    value: float | str = "",
    baseline_value: float | str = "",
    comparison_value: float | str = "",
    notes: str = "",
) -> None:
    stats.append(
        {
            "label_class": label_class,
            "label_name": label_name,
            "label_kind": label_kind,
            "signal_name": signal_name,
            "conditioning": conditioning,
            "horizon_ms": horizon_ms,
            "stat_name": stat_name,
            "bucket_or_level": bucket_or_level,
            "rows": rows,
            "positive_rows": positive_rows,
            "exposure_ms": exposure_ms,
            "value": value,
            "baseline_value": baseline_value,
            "comparison_value": comparison_value,
            "notes": notes,
        }
    )


def _continuous_stats(
    stats: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    label_class: str,
    label_name: str,
    signal_name: str,
    horizon_ms: int | str,
) -> None:
    signal_values: list[float] = []
    label_values: list[float] = []
    selected: list[dict[str, Any]] = []
    for row in rows:
        signal = _float(row.get(signal_name))
        label = _float(row.get(label_name))
        if _finite(signal) and _finite(label):
            signal_values.append(signal)
            label_values.append(label)
            selected.append(row)
    if len(selected) < 2:
        return
    x = np.asarray(signal_values, dtype=np.float64)
    y = np.asarray(label_values, dtype=np.float64)
    _append_stat(
        stats,
        label_class=label_class,
        label_name=label_name,
        label_kind="continuous",
        signal_name=signal_name,
        horizon_ms=horizon_ms,
        stat_name="pearson",
        rows=len(selected),
        value=_corr(x, y),
    )
    _append_stat(
        stats,
        label_class=label_class,
        label_name=label_name,
        label_kind="continuous",
        signal_name=signal_name,
        horizon_ms=horizon_ms,
        stat_name="spearman",
        rows=len(selected),
        value=_spearman(x, y),
    )
    _, bucket_ids = _bucketize(signal_values, buckets=5)
    bucket_means: list[float] = []
    first_half_cut = int(np.median(np.asarray([int(row["submit_ts_local"]) for row in selected], dtype=np.int64)))
    for bucket in range(1, 6):
        bucket_rows = [label_values[idx] for idx, bucket_id in enumerate(bucket_ids) if int(bucket_id) == bucket]
        bucket_mean = _mean(bucket_rows)
        bucket_means.append(bucket_mean)
        _append_stat(
            stats,
            label_class=label_class,
            label_name=label_name,
            label_kind="continuous",
            signal_name=signal_name,
            horizon_ms=horizon_ms,
            stat_name="bucket_mean",
            bucket_or_level=str(bucket),
            rows=len(bucket_rows),
            value=bucket_mean,
            comparison_value=_quantile(bucket_rows, 0.5),
        )
    _append_stat(
        stats,
        label_class=label_class,
        label_name=label_name,
        label_kind="continuous",
        signal_name=signal_name,
        horizon_ms=horizon_ms,
        stat_name="top_bottom_spread",
        rows=len(selected),
        value=(bucket_means[-1] - bucket_means[0]) if _finite(bucket_means[-1]) and _finite(bucket_means[0]) else math.nan,
        comparison_value=_monotonic_score(bucket_means),
    )
    first_half_rows = [row for row in selected if int(row["submit_ts_local"]) <= first_half_cut]
    second_half_rows = [row for row in selected if int(row["submit_ts_local"]) > first_half_cut]
    first_half_signal = np.asarray([_float(row[signal_name]) for row in first_half_rows if _finite(_float(row[signal_name])) and _finite(_float(row[label_name]))], dtype=np.float64)
    first_half_label = np.asarray([_float(row[label_name]) for row in first_half_rows if _finite(_float(row[signal_name])) and _finite(_float(row[label_name]))], dtype=np.float64)
    second_half_signal = np.asarray([_float(row[signal_name]) for row in second_half_rows if _finite(_float(row[signal_name])) and _finite(_float(row[label_name]))], dtype=np.float64)
    second_half_label = np.asarray([_float(row[label_name]) for row in second_half_rows if _finite(_float(row[signal_name])) and _finite(_float(row[label_name]))], dtype=np.float64)
    first_half_spearman = _spearman(first_half_signal, first_half_label) if len(first_half_signal) >= 2 else math.nan
    second_half_spearman = _spearman(second_half_signal, second_half_label) if len(second_half_signal) >= 2 else math.nan
    _append_stat(
        stats,
        label_class=label_class,
        label_name=label_name,
        label_kind="continuous",
        signal_name=signal_name,
        horizon_ms=horizon_ms,
        stat_name="time_split_stability",
        rows=len(selected),
        value=first_half_spearman,
        baseline_value=second_half_spearman,
        comparison_value=abs(first_half_spearman - second_half_spearman) if _finite(first_half_spearman) and _finite(second_half_spearman) else math.nan,
        notes="first_half=first baseline_value, second_half=second baseline_value, comparison_value=abs_diff",
    )


def _binary_stats(
    stats: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    label_class: str,
    label_name: str,
    signal_name: str,
    horizon_ms: int | str,
) -> None:
    selected: list[dict[str, Any]] = []
    signal_values: list[float] = []
    label_values: list[int] = []
    for row in rows:
        signal = _float(row.get(signal_name))
        label = _int(row.get(label_name))
        if _finite(signal) and label in {0, 1}:
            selected.append(row)
            signal_values.append(signal)
            label_values.append(label)
    if len(selected) < 2:
        return
    baseline_rate = float(np.mean(np.asarray(label_values, dtype=np.float64)))
    _, bucket_ids = _bucketize(signal_values, buckets=5)
    bucket_rates: dict[int, float] = {}
    bucket_counts: dict[int, tuple[int, int]] = {}
    for bucket in range(1, 6):
        positives = sum(label_values[idx] for idx, bucket_id in enumerate(bucket_ids) if int(bucket_id) == bucket)
        count = sum(1 for bucket_id in bucket_ids if int(bucket_id) == bucket)
        rate = _safe_div(positives, count)
        bucket_rates[bucket] = rate
        bucket_counts[bucket] = (positives, count)
        _append_stat(
            stats,
            label_class=label_class,
            label_name=label_name,
            label_kind="binary",
            signal_name=signal_name,
            horizon_ms=horizon_ms,
            stat_name="event_rate",
            bucket_or_level=str(bucket),
            rows=count,
            positive_rows=positives,
            value=rate,
            baseline_value=baseline_rate,
            comparison_value=_safe_div(rate, baseline_rate),
        )
    top_pos, top_count = bucket_counts[5]
    bottom_pos, bottom_count = bucket_counts[1]
    top_rate = _safe_div(top_pos, top_count)
    bottom_rate = _safe_div(bottom_pos, bottom_count)
    top_odds = _safe_div(top_pos, max(top_count - top_pos, 0))
    bottom_odds = _safe_div(bottom_pos, max(bottom_count - bottom_pos, 0))
    odds_ratio = _safe_div(top_odds, bottom_odds)
    _append_stat(
        stats,
        label_class=label_class,
        label_name=label_name,
        label_kind="binary",
        signal_name=signal_name,
        horizon_ms=horizon_ms,
        stat_name="top_bucket_lift",
        rows=len(selected),
        positive_rows=sum(label_values),
        value=_safe_div(top_rate, baseline_rate),
        baseline_value=baseline_rate,
        comparison_value=odds_ratio,
        notes="comparison_value=top_vs_bottom_odds_ratio",
    )


def _count_rate_stats(
    stats: list[dict[str, Any]],
    *,
    rows: list[dict[str, Any]],
    label_class: str,
    label_name: str,
    signal_name: str,
    exposure_name: str,
) -> None:
    selected: list[dict[str, Any]] = []
    signal_values: list[float] = []
    count_values: list[float] = []
    exposure_values: list[float] = []
    for row in rows:
        signal = _float(row.get(signal_name))
        count = _float(row.get(label_name))
        exposure = _float(row.get(exposure_name))
        if _finite(signal) and _finite(count) and _finite(exposure) and exposure > 0.0:
            selected.append(row)
            signal_values.append(signal)
            count_values.append(count)
            exposure_values.append(exposure)
    if len(selected) < 2:
        return
    _, bucket_ids = _bucketize(signal_values, buckets=5)
    bucket_rates: dict[int, float] = {}
    for bucket in range(1, 6):
        bucket_count = sum(count_values[idx] for idx, bucket_id in enumerate(bucket_ids) if int(bucket_id) == bucket)
        bucket_exposure = sum(exposure_values[idx] for idx, bucket_id in enumerate(bucket_ids) if int(bucket_id) == bucket)
        rate = _safe_div(bucket_count, bucket_exposure)
        bucket_rates[bucket] = rate
        _append_stat(
            stats,
            label_class=label_class,
            label_name=label_name,
            label_kind="count_rate",
            signal_name=signal_name,
            stat_name="exposure_normalized_rate",
            bucket_or_level=str(bucket),
            rows=sum(1 for bucket_id in bucket_ids if int(bucket_id) == bucket),
            exposure_ms=bucket_exposure,
            value=rate,
            baseline_value=_safe_div(sum(count_values), sum(exposure_values)),
        )
    _append_stat(
        stats,
        label_class=label_class,
        label_name=label_name,
        label_kind="count_rate",
        signal_name=signal_name,
        stat_name="top_bottom_rate_ratio",
        rows=len(selected),
        value=_safe_div(bucket_rates[5], bucket_rates[1]),
    )


def _time_to_event_stats(
    stats: list[dict[str, Any]],
    *,
    execution_rows: list[dict[str, Any]],
    horizons_ms: Iterable[int],
) -> None:
    previous_horizon = 0
    at_risk_order_ids = {str(row["order_id"]) for row in execution_rows}
    survival = 1.0
    for horizon_ms in horizons_ms:
        interval_end = int(horizon_ms)
        interval_rows = [row for row in execution_rows if str(row["order_id"]) in at_risk_order_ids]
        fills_in_interval = 0
        observed_at_end = 0
        newly_censored = 0
        next_at_risk: set[str] = set()
        for row in interval_rows:
            order_id = str(row["order_id"])
            time_to_fill = (
                (_ts_ns(row.get("first_fill_ts_local")) - int(row["submit_ts_local"])) / 1_000_000.0
                if _ts_ns(row.get("first_fill_ts_local")) is not None
                else math.nan
            )
            observable = int(row.get(f"horizon_observable_{horizon_ms}ms", 0))
            if observable:
                observed_at_end += 1
            elif _int(row.get(f"fill_by_{horizon_ms}ms")) == 0:
                newly_censored += 1
            if _finite(time_to_fill) and previous_horizon < float(time_to_fill) <= interval_end:
                fills_in_interval += 1
            elif (not _finite(time_to_fill) or float(time_to_fill) > interval_end) and observable:
                next_at_risk.add(order_id)
        hazard = _safe_div(fills_in_interval, observed_at_end)
        if _finite(hazard):
            survival *= 1.0 - hazard
        _append_stat(
            stats,
            label_class="time_to_fill",
            label_name="time_to_fill_ms",
            label_kind="time_to_event",
            horizon_ms=horizon_ms,
            stat_name="discrete_hazard",
            bucket_or_level=f"{previous_horizon}-{interval_end}",
            rows=len(interval_rows),
            positive_rows=fills_in_interval,
            value=hazard,
            baseline_value=survival,
            comparison_value=newly_censored,
            notes="baseline_value=survival_end, comparison_value=newly_censored_count; Cox-style model remains later work",
        )
        previous_horizon = interval_end
        at_risk_order_ids = next_at_risk


def _multiclass_stats(
    stats: list[dict[str, Any]],
    *,
    execution_rows: list[dict[str, Any]],
) -> None:
    total = len(execution_rows)
    overall = Counter(str(row["final_order_state"]) for row in execution_rows)
    by_bucket: dict[str, Counter[str]] = defaultdict(Counter)
    for row in execution_rows:
        by_bucket[str(row["placement_bucket"])][str(row["final_order_state"])] += 1
    for bucket, counter in sorted(by_bucket.items()):
        bucket_total = sum(counter.values())
        for state, count in sorted(counter.items()):
            overall_rate = _safe_div(overall[state], total)
            conditional_rate = _safe_div(count, bucket_total)
            _append_stat(
                stats,
                label_class="partial_fill_lifecycle",
                label_name="final_order_state",
                label_kind="multiclass",
                conditioning="placement_bucket",
                stat_name="conditional_probability",
                bucket_or_level=f"{bucket}:{state}",
                rows=bucket_total,
                positive_rows=count,
                value=conditional_rate,
                baseline_value=overall_rate,
                comparison_value=_safe_div(conditional_rate, overall_rate),
            )


def _tail_stats(
    stats: list[dict[str, Any]],
    *,
    fill_markout_rows: list[dict[str, Any]],
    horizons_ms: Iterable[int],
) -> None:
    for horizon_ms in horizons_ms:
        horizon_rows = [
            row
            for row in fill_markout_rows
            if int(row["horizon_ms"]) == int(horizon_ms) and _finite(row.get("side_adjusted_markout_ticks"))
        ]
        if not horizon_rows:
            continue
        markouts = [float(row["side_adjusted_markout_ticks"]) for row in horizon_rows]
        p01 = _quantile(markouts, 0.01)
        p05 = _quantile(markouts, 0.05)
        tail_mean = _mean(value for value in markouts if _finite(p05) and value <= p05)
        exceedance_rate = _safe_div(sum(1 for value in markouts if value <= 0.0), len(markouts))
        _append_stat(
            stats,
            label_class="tail_risk",
            label_name="side_adjusted_markout_ticks",
            label_kind="tail",
            horizon_ms=horizon_ms,
            stat_name="tail_quantile_p01",
            rows=len(horizon_rows),
            value=p01,
            baseline_value=p05,
            comparison_value=tail_mean,
            notes="baseline_value=p05, comparison_value=tail_mean_at_or_below_p05",
        )
        _append_stat(
            stats,
            label_class="tail_risk",
            label_name="side_adjusted_markout_ticks",
            label_kind="tail",
            horizon_ms=horizon_ms,
            stat_name="exceedance_rate_nonpositive",
            rows=len(horizon_rows),
            value=exceedance_rate,
        )
        worst_count = max(1, int(math.ceil(len(horizon_rows) * 0.1)))
        worst_rows = sorted(horizon_rows, key=lambda row: float(row["side_adjusted_markout_ticks"]))[:worst_count]
        bucket_counter = Counter(str(row["placement_bucket"]) for row in worst_rows)
        for bucket, count in sorted(bucket_counter.items()):
            _append_stat(
                stats,
                label_class="tail_risk",
                label_name="side_adjusted_markout_ticks",
                label_kind="tail",
                horizon_ms=horizon_ms,
                stat_name="worst_bucket_concentration",
                bucket_or_level=bucket,
                rows=len(worst_rows),
                positive_rows=count,
                value=_safe_div(count, len(worst_rows)),
            )


def build_placement_tradeoff(execution_rows: list[dict[str, Any]], fill_markout_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    signal_values = [_float(row.get("edge_vs_fair_ticks")) for row in execution_rows if _finite(_float(row.get("edge_vs_fair_ticks")))]
    _, bucket_ids = _bucketize(signal_values, buckets=5)
    signal_bucket_by_order: dict[str, int] = {}
    signal_rows = [row for row in execution_rows if _finite(_float(row.get("edge_vs_fair_ticks")))]
    for idx, row in enumerate(signal_rows):
        signal_bucket_by_order[str(row["order_id"])] = int(bucket_ids[idx])
    markout_500 = {
        str(row["order_id"]): row
        for row in fill_markout_rows
        if int(row["horizon_ms"]) == 500 and _finite(row.get("side_adjusted_markout_ticks"))
    }
    horizon_5000 = {str(row["order_id"]): int(row["fill_by_5000ms"]) for row in execution_rows}
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in execution_rows:
        signal_bucket = signal_bucket_by_order.get(str(row["order_id"]))
        if signal_bucket is None:
            continue
        groups[(str(row["placement_bucket"]), signal_bucket)].append(row)
    for (placement_bucket, signal_bucket), group_rows in sorted(groups.items()):
        filled_500 = [row for row in group_rows if int(row.get("fill_by_500ms", 0)) == 1]
        missed_proxy = [
            _float(row.get("opportunity_cost_proxy_500ms_ticks"))
            for row in group_rows
            if int(row.get("fill_by_500ms", 0)) == 0 and _finite(_float(row.get("opportunity_cost_proxy_500ms_ticks")))
        ]
        markouts = [
            _float(markout_500[str(row["order_id"])]["side_adjusted_markout_ticks"])
            for row in group_rows
            if str(row["order_id"]) in markout_500
        ]
        net_ev_proxy = [
            _float(markout_500[str(row["order_id"])]["net_ev_proxy_ticks"])
            for row in group_rows
            if str(row["order_id"]) in markout_500 and _finite(_float(markout_500[str(row["order_id"])]["net_ev_proxy_ticks"]))
        ]
        rows.append(
            {
                "placement_bucket": placement_bucket,
                "signal_bucket": signal_bucket,
                "rows": len(group_rows),
                "fill_500ms_rate": _safe_div(len(filled_500), len(group_rows)),
                "fill_5000ms_rate": _safe_div(sum(horizon_5000.get(str(row["order_id"]), 0) for row in group_rows), len(group_rows)),
                "missed_favorable_mid_move_500ms_rate": _safe_div(
                    sum(int(row.get("missed_favorable_mid_move_500ms", 0)) for row in group_rows),
                    len(group_rows),
                ),
                "missed_opportunity_cost_proxy_500ms_ticks_mean": _mean(missed_proxy),
                "filled_markout_500ms_ticks_mean": _mean(markouts),
                "net_ev_proxy_500ms_ticks_mean": _mean(net_ev_proxy),
            }
        )
    return rows


def build_label_coverage(
    execution_rows: list[dict[str, Any]],
    fill_horizon_rows: list[dict[str, Any]],
    fill_markout_rows: list[dict[str, Any]],
    *,
    horizons_ms: Iterable[int],
) -> list[dict[str, Any]]:
    submit_rows = len(execution_rows)
    filled_orders = sum(1 for row in execution_rows if int(row["fill_count"]) > 0)
    fill_horizon_observed = {
        int(horizon): sum(
            int(row["horizon_observable"])
            for row in fill_horizon_rows
            if int(row["horizon_ms"]) == int(horizon)
        )
        for horizon in horizons_ms
    }
    markout_observed = {
        int(horizon): sum(
            int(row["horizon_observable"])
            for row in fill_markout_rows
            if int(row["horizon_ms"]) == int(horizon)
        )
        for horizon in horizons_ms
    }
    fill_after_cancel_count = sum(int(row["fill_after_cancel_request"]) for row in execution_rows)
    tail_status = "available" if filled_orders >= LOW_SAMPLE_FILL_ROWS else "low_sample"
    tail_reason = (
        f"{filled_orders} filled orders available for tail quantiles"
        if tail_status == "available"
        else f"{filled_orders} filled orders; tail estimates are implemented but sample is thin"
    )
    rows = [
        {
            "label_class": "fill_probability",
            "status": "available",
            "rows": submit_rows,
            "observed_rows": min(fill_horizon_observed.values()) if fill_horizon_observed else 0,
            "positive_rows": sum(int(row["fill_by_5000ms"]) for row in execution_rows),
            "reason": "Observed directly from submit to first fill over 100/500/1000/5000ms horizons",
            "notes": ",".join(f"{h}ms={fill_horizon_observed[h]}" for h in sorted(fill_horizon_observed)),
        },
        {
            "label_class": "time_to_fill",
            "status": "available",
            "rows": submit_rows,
            "observed_rows": filled_orders,
            "positive_rows": filled_orders,
            "reason": "Observed first-fill elapsed time is available for every filled order",
            "notes": "Discrete hazard and survival summary use right-censoring counts",
        },
        {
            "label_class": "adverse_selection_after_fill",
            "status": "available",
            "rows": filled_orders,
            "observed_rows": min(markout_observed.values()) if markout_observed else 0,
            "positive_rows": sum(
                1
                for row in fill_markout_rows
                if int(row["horizon_ms"]) == 500 and _finite(row["side_adjusted_markout_ticks"]) and float(row["side_adjusted_markout_ticks"]) <= 0.0
            ),
            "reason": "Observed side-adjusted future-mid markout after fill",
            "notes": ",".join(f"{h}ms={markout_observed[h]}" for h in sorted(markout_observed)),
        },
        {
            "label_class": "spread_capture",
            "status": "available",
            "rows": filled_orders,
            "observed_rows": filled_orders,
            "positive_rows": sum(
                1 for row in execution_rows if _finite(row["realized_spread_proxy_ticks"]) and float(row["realized_spread_proxy_ticks"]) > 0.0
            ),
            "reason": "Observed realized spread proxy from fill price vs fill-time mid, plus future-mid retained spread",
            "notes": "Future retained spread appears in fill_markout_labels.csv",
        },
        {
            "label_class": "queue_priority_proxy",
            "status": "observed_only_proxy",
            "rows": submit_rows,
            "observed_rows": submit_rows,
            "positive_rows": "",
            "reason": "Only top1/top5 size, join age, stale age, and latency proxies are available; exact queue position is not observable",
            "notes": "No exact queue proof or fill calibration",
        },
        {
            "label_class": "cancel_to_fill_race",
            "status": "available",
            "rows": submit_rows,
            "observed_rows": sum(1 for row in execution_rows if _ts_ns(row.get("cancel_request_ts_local")) is not None),
            "positive_rows": fill_after_cancel_count,
            "reason": "Observed cancel-request and fill timestamps allow direct fill-after-cancel labels",
            "notes": "cancel_to_fill_delay_ms and cancel-fill markout are included",
        },
        {
            "label_class": "post_only_reject_throttle_churn",
            "status": "observed_only_proxy",
            "rows": submit_rows,
            "observed_rows": submit_rows,
            "positive_rows": sum(int(row["fast_cancel_churn"]) for row in execution_rows),
            "reason": "Actual submit orders expose churn directly; reject/throttle are only available as nearby decision-level context counts",
            "notes": "Reject/throttle are not a complete quote-attempt universe here",
        },
        {
            "label_class": "inventory_impact",
            "status": "available",
            "rows": submit_rows,
            "observed_rows": sum(1 for row in execution_rows if _finite(row["position_after_fill"])),
            "positive_rows": sum(int(row["inventory_increasing_fill"]) for row in execution_rows),
            "reason": "Position before/after fill and inventory cycle fields are observed from audit rows",
            "notes": "Inventory-cycle subfields may still be censored near sample end",
        },
        {
            "label_class": "quote_placement_distance",
            "status": "available",
            "rows": submit_rows,
            "observed_rows": sum(1 for row in execution_rows if _finite(row["distance_to_bbo_ticks"])),
            "positive_rows": sum(1 for row in execution_rows if str(row["placement_bucket"]) == "touch"),
            "reason": "Distance-to-BBO ticks, placement bucket, post-only risk, and target-offset bucket are derived directly from submit context",
            "notes": "",
        },
        {
            "label_class": "missed_fill_opportunity_cost",
            "status": "observed_only_proxy",
            "rows": submit_rows,
            "observed_rows": sum(1 for row in fill_horizon_rows if int(row["horizon_ms"]) == 500 and _finite(row["opportunity_cost_proxy_ticks"])),
            "positive_rows": sum(int(row["missed_favorable_mid_move_500ms"]) for row in execution_rows),
            "reason": "Only future-mid observed opportunity-cost proxies are available; no counterfactual queue/fill proof",
            "notes": "Uses unfilled-then-favorable-mid proxy at each horizon",
        },
        {
            "label_class": "realized_pnl_decomposition",
            "status": "observed_only_proxy",
            "rows": filled_orders,
            "observed_rows": filled_orders,
            "positive_rows": sum(
                1 for row in execution_rows if _finite(row["fee_adjusted_realized_spread_ticks"]) and float(row["fee_adjusted_realized_spread_ticks"]) > 0.0
            ),
            "reason": "Only observed spread/markout decomposition proxies are available; fees are parameterized, inventory MTM is horizon-based",
            "notes": "No realized strategy PnL proof and no counterfactual decomposition",
        },
        {
            "label_class": "tail_risk",
            "status": tail_status,
            "rows": filled_orders,
            "observed_rows": min(markout_observed.values()) if markout_observed else 0,
            "positive_rows": "",
            "reason": tail_reason,
            "notes": "Tail quantiles, tail mean, exceedance rate, and worst placement-bucket concentration are still emitted",
        },
        {
            "label_class": "partial_fill_lifecycle",
            "status": "available",
            "rows": submit_rows,
            "observed_rows": submit_rows,
            "positive_rows": sum(int(row["partial_fill"]) for row in execution_rows),
            "reason": "Lifecycle state, fill ratio, fill count, remaining qty, and terminal state are observed directly",
            "notes": "",
        },
        {
            "label_class": "inventory_cycle",
            "status": "available",
            "rows": filled_orders,
            "observed_rows": sum(int(row["inventory_cycle_flat_observed"]) for row in execution_rows),
            "positive_rows": sum(int(row["crossed_zero_after_fill"]) for row in execution_rows),
            "reason": "Observed from subsequent decision positions after fill, with explicit censoring when flat is not seen",
            "notes": "",
        },
        {
            "label_class": "sample_validity_censoring",
            "status": "available",
            "rows": submit_rows,
            "observed_rows": sum(int(row["horizon_observable_5000ms"]) for row in execution_rows),
            "positive_rows": sum(int(row["missing_lifecycle"]) for row in execution_rows),
            "reason": "Each horizon includes observable/right-censored/tail-truncated flags and missing-lifecycle flags",
            "notes": "",
        },
    ]
    return rows


def build_label_statistics(
    execution_rows: list[dict[str, Any]],
    fill_horizon_rows: list[dict[str, Any]],
    fill_markout_rows: list[dict[str, Any]],
    *,
    horizons_ms: Iterable[int],
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for signal_name in STAT_SIGNALS:
        for horizon_ms in horizons_ms:
            horizon_fill_rows = [row for row in fill_horizon_rows if int(row["horizon_ms"]) == int(horizon_ms)]
            _binary_stats(
                stats,
                rows=horizon_fill_rows,
                label_class="fill_probability",
                label_name="fill_by_horizon",
                signal_name=signal_name,
                horizon_ms=horizon_ms,
            )
    for row in execution_rows:
        if _ts_ns(row.get("first_fill_ts_local")) is not None:
            row["time_to_fill_ms"] = (_ts_ns(row.get("first_fill_ts_local")) - int(row["submit_ts_local"])) / 1_000_000.0
        else:
            row["time_to_fill_ms"] = math.nan
    _continuous_stats(
        stats,
        rows=execution_rows,
        label_class="time_to_fill",
        label_name="time_to_fill_ms",
        signal_name="distance_to_bbo_ticks",
        horizon_ms="",
    )
    for signal_name in STAT_SIGNALS:
        _continuous_stats(
            stats,
            rows=execution_rows,
            label_class="spread_capture",
            label_name="realized_spread_proxy_ticks",
            signal_name=signal_name,
            horizon_ms="fill",
        )
    for signal_name in STAT_SIGNALS:
        for horizon_ms in horizons_ms:
            horizon_fill_rows = [row for row in fill_horizon_rows if int(row["horizon_ms"]) == int(horizon_ms)]
            _binary_stats(
                stats,
                rows=horizon_fill_rows,
                label_class="missed_fill_opportunity_cost",
                label_name="missed_favorable_mid_move",
                signal_name=signal_name,
                horizon_ms=horizon_ms,
            )
            _continuous_stats(
                stats,
                rows=horizon_fill_rows,
                label_class="missed_fill_opportunity_cost",
                label_name="opportunity_cost_proxy_ticks",
                signal_name=signal_name,
                horizon_ms=horizon_ms,
            )
    for signal_name in STAT_SIGNALS:
        for horizon_ms in horizons_ms:
            horizon_markout_rows = [row for row in fill_markout_rows if int(row["horizon_ms"]) == int(horizon_ms)]
            _continuous_stats(
                stats,
                rows=horizon_markout_rows,
                label_class="adverse_selection_after_fill",
                label_name="side_adjusted_markout_ticks",
                signal_name=signal_name,
                horizon_ms=horizon_ms,
            )
            _continuous_stats(
                stats,
                rows=horizon_markout_rows,
                label_class="realized_pnl_decomposition",
                label_name="net_ev_proxy_ticks",
                signal_name=signal_name,
                horizon_ms=horizon_ms,
            )
    _binary_stats(
        stats,
        rows=execution_rows,
        label_class="cancel_to_fill_race",
        label_name="fill_after_cancel_request",
        signal_name="distance_to_bbo_ticks",
        horizon_ms="",
    )
    _binary_stats(
        stats,
        rows=execution_rows,
        label_class="inventory_impact",
        label_name="inventory_increasing_fill",
        signal_name="inventory_score",
        horizon_ms="",
    )
    _continuous_stats(
        stats,
        rows=execution_rows,
        label_class="inventory_cycle",
        label_name="time_to_flat_ms",
        signal_name="inventory_score",
        horizon_ms="",
    )
    _continuous_stats(
        stats,
        rows=execution_rows,
        label_class="inventory_cycle",
        label_name="max_abs_position_until_flat",
        signal_name="inventory_score",
        horizon_ms="",
    )
    _count_rate_stats(
        stats,
        rows=execution_rows,
        label_class="post_only_reject_throttle_churn",
        label_name="cancel_request_count",
        signal_name="distance_to_bbo_ticks",
        exposure_name="working_lifetime_ms",
    )
    _count_rate_stats(
        stats,
        rows=execution_rows,
        label_class="post_only_reject_throttle_churn",
        label_name="order_update_count",
        signal_name="distance_to_bbo_ticks",
        exposure_name="working_lifetime_ms",
    )
    _time_to_event_stats(
        stats,
        execution_rows=execution_rows,
        horizons_ms=horizons_ms,
    )
    for horizon_ms in horizons_ms:
        horizon_fill_rows = [row for row in fill_horizon_rows if int(row["horizon_ms"]) == int(horizon_ms)]
        if horizon_fill_rows:
            total_rows = len(horizon_fill_rows)
            right_censored = sum(int(row["right_censored"]) for row in horizon_fill_rows)
            tail_truncated = sum(int(row["tail_truncated"]) for row in horizon_fill_rows)
            observable = sum(int(row["horizon_observable"]) for row in horizon_fill_rows)
            _append_stat(
                stats,
                label_class="sample_validity_censoring",
                label_name="fill_horizon_observable",
                label_kind="time_to_event",
                horizon_ms=horizon_ms,
                stat_name="censoring_rate",
                rows=total_rows,
                positive_rows=right_censored,
                value=_safe_div(right_censored, total_rows),
                baseline_value=_safe_div(observable, total_rows),
                comparison_value=_safe_div(tail_truncated, total_rows),
                notes="baseline_value=observable_rate, comparison_value=tail_truncated_rate",
            )
    _multiclass_stats(
        stats,
        execution_rows=execution_rows,
    )
    _tail_stats(
        stats,
        fill_markout_rows=fill_markout_rows,
        horizons_ms=horizons_ms,
    )
    tradeoff_rows = build_placement_tradeoff(execution_rows, fill_markout_rows)
    for row in tradeoff_rows:
        _append_stat(
            stats,
            label_class="quote_placement_distance",
            label_name="placement_opportunity_tradeoff",
            label_kind="placement_opportunity",
            signal_name="edge_vs_fair_ticks",
            conditioning="placement_bucket",
            horizon_ms=500,
            stat_name="tradeoff_cell",
            bucket_or_level=f"{row['placement_bucket']}|signal_bucket_{row['signal_bucket']}",
            rows=int(row["rows"]),
            value=row["fill_500ms_rate"],
            baseline_value=row["missed_opportunity_cost_proxy_500ms_ticks_mean"],
            comparison_value=row["filled_markout_500ms_ticks_mean"],
            notes="value=fill_500ms_rate, baseline_value=missed_opportunity_cost_proxy_500ms_ticks_mean, comparison_value=filled_markout_500ms_ticks_mean",
        )
    return stats


def write_summary_markdown(
    path: Path,
    *,
    run_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    coverage_rows: list[dict[str, Any]],
    lifecycle_rows: list[dict[str, Any]],
    censoring_rows: list[dict[str, Any]],
) -> None:
    coverage_lines = [
        f"- `{row['label_class']}`: `{row['status']}` - {row['reason']}"
        for row in coverage_rows
    ]
    lifecycle_lines = [
        f"- `{row['label']}`: count={row['count']}, rate={row['rate']}"
        for row in lifecycle_rows
        if row["category"] != "continuous_metric"
    ]
    censoring_lines = [
        f"- `{row['label_family']}` horizon `{row['horizon_ms']}`: observable={row['observable_rows']}, right_censored={row['right_censored_rows']}, tail_truncated={row['tail_truncated_rows']}"
        for row in censoring_rows
        if str(row["horizon_ms"]) != ""
    ]
    text = "\n".join(
        [
            f"# {TASK_ID} execution outcome labels",
            "",
            "## Dataset",
            f"- run_dir: `{run_dir}`",
            f"- output_dir: `{output_dir}`",
            f"- stage3 classification: `{manifest['stage3_classification']}`",
            f"- submit_orders: `{manifest['row_counts']['submit_orders']}`",
            f"- filled_orders: `{manifest['row_counts']['filled_orders']}`",
            f"- fill_after_cancel_orders: `{manifest['row_counts']['fill_after_cancel_orders']}`",
            "",
            "## Coverage",
            *coverage_lines,
            "",
            "## Lifecycle",
            *lifecycle_lines,
            "",
            "## Censoring",
            *censoring_lines,
            "",
            "## Method Notes",
            "- queue / priority, missed opportunity, and realized PnL decomposition remain observed-only proxy labels.",
            "- hazard summary is discrete horizon-based survival reporting; Cox-style modeling remains later work.",
            "- observed correlations and proxies are not counterfactual queue/fill proof and are not strategy PnL proof.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def run_execution_outcome_labels(
    *,
    run_dir: Path,
    output_dir: Path,
    horizons_ms: Iterable[int] = DEFAULT_HORIZONS_MS,
    tick_size: float = DEFAULT_TICK_SIZE,
    max_future_gap_ms: float = DEFAULT_MAX_FUTURE_GAP_MS,
    maker_fee_bps: float = DEFAULT_MAKER_FEE_BPS,
) -> dict[str, Any]:
    run_dir = _expand(run_dir)
    output_dir = _expand(output_dir)
    audit_csv = _find_audit_csv(run_dir)
    joined_csv = run_dir / "t009_fixed_sidecar" / "joined_decisions.csv"
    stage3_json = run_dir / "maker_acceptance_stage3.json"
    sidecar_metrics_json = run_dir / "t009_fixed_sidecar" / "metrics.json"
    join_metrics_json = run_dir / "t009_fixed_sidecar" / "joined_decisions.metrics.json"

    audit_rows = load_audit_rows(audit_csv)
    joined_rows = load_joined_decisions(joined_csv)
    decision_index = build_decision_index(audit_rows, joined_rows, tick_size=tick_size)
    execution_rows, fill_horizon_rows, fill_markout_rows, row_counts = build_execution_labels(
        audit_rows=audit_rows,
        decision_index=decision_index,
        horizons_ms=horizons_ms,
        tick_size=tick_size,
        max_future_gap_ms=max_future_gap_ms,
        maker_fee_bps=maker_fee_bps,
    )
    coverage_rows = build_label_coverage(
        execution_rows,
        fill_horizon_rows,
        fill_markout_rows,
        horizons_ms=horizons_ms,
    )
    lifecycle_rows = build_lifecycle_summary(execution_rows)
    censoring_rows = build_censoring_summary(execution_rows, fill_horizon_rows, fill_markout_rows, horizons_ms)
    statistics_rows = build_label_statistics(
        execution_rows,
        fill_horizon_rows,
        fill_markout_rows,
        horizons_ms=horizons_ms,
    )
    tradeoff_rows = build_placement_tradeoff(execution_rows, fill_markout_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "execution_outcome_labels.csv",
        execution_rows,
        fieldnames=list(execution_rows[0].keys()) if execution_rows else [],
    )
    _write_csv(
        output_dir / "fill_horizon_labels.csv",
        fill_horizon_rows,
        fieldnames=list(fill_horizon_rows[0].keys()) if fill_horizon_rows else [],
    )
    _write_csv(
        output_dir / "fill_markout_labels.csv",
        fill_markout_rows,
        fieldnames=list(fill_markout_rows[0].keys()) if fill_markout_rows else [],
    )
    _write_csv(
        output_dir / "label_coverage.csv",
        coverage_rows,
        fieldnames=["label_class", "status", "rows", "observed_rows", "positive_rows", "reason", "notes"],
    )
    _write_csv(
        output_dir / "label_statistics.csv",
        statistics_rows,
        fieldnames=[
            "label_class",
            "label_name",
            "label_kind",
            "signal_name",
            "conditioning",
            "horizon_ms",
            "stat_name",
            "bucket_or_level",
            "rows",
            "positive_rows",
            "exposure_ms",
            "value",
            "baseline_value",
            "comparison_value",
            "notes",
        ],
    )
    _write_csv(
        output_dir / "placement_opportunity_tradeoff.csv",
        tradeoff_rows,
        fieldnames=list(tradeoff_rows[0].keys()) if tradeoff_rows else [
            "placement_bucket",
            "signal_bucket",
            "rows",
            "fill_500ms_rate",
            "fill_5000ms_rate",
            "missed_favorable_mid_move_500ms_rate",
            "missed_opportunity_cost_proxy_500ms_ticks_mean",
            "filled_markout_500ms_ticks_mean",
            "net_ev_proxy_500ms_ticks_mean",
        ],
    )
    _write_csv(
        output_dir / "lifecycle_label_summary.csv",
        lifecycle_rows,
        fieldnames=["category", "label", "count", "rate", "mean_value"],
    )
    _write_csv(
        output_dir / "censoring_summary.csv",
        censoring_rows,
        fieldnames=[
            "label_family",
            "horizon_ms",
            "total_rows",
            "observable_rows",
            "right_censored_rows",
            "tail_truncated_rows",
            "missing_lifecycle_rows",
            "positive_rows",
        ],
    )

    stage3_payload = _load_json(stage3_json) if stage3_json.exists() else {}
    sidecar_metrics = _load_json(sidecar_metrics_json) if sidecar_metrics_json.exists() else {}
    join_metrics = _load_json(join_metrics_json) if join_metrics_json.exists() else {}
    manifest = {
        "task_id": TASK_ID,
        "generated_at": _generated_at(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "input_hashes": {
            "audit_csv": _hash_file(audit_csv),
            "joined_decisions_csv": _hash_file(joined_csv),
            "maker_acceptance_stage3_json": _hash_file(stage3_json) if stage3_json.exists() else "",
            "sidecar_metrics_json": _hash_file(sidecar_metrics_json) if sidecar_metrics_json.exists() else "",
            "joined_decisions_metrics_json": _hash_file(join_metrics_json) if join_metrics_json.exists() else "",
        },
        "parameters": {
            "horizons_ms": [int(value) for value in horizons_ms],
            "tick_size": tick_size,
            "max_future_gap_ms": max_future_gap_ms,
            "maker_fee_bps": maker_fee_bps,
        },
        "stage3_classification": (
            stage3_payload.get("market_view", {}).get("classification")
            or stage3_payload.get("classification")
            or "unknown"
        ),
        "sidecar_metrics": sidecar_metrics,
        "joined_decisions_metrics": join_metrics,
        "row_counts": row_counts,
        "artifacts": [
            "execution_outcome_label_summary.md",
            "execution_outcome_labels.csv",
            "label_coverage.csv",
            "label_statistics.csv",
            "fill_horizon_labels.csv",
            "fill_markout_labels.csv",
            "placement_opportunity_tradeoff.csv",
            "lifecycle_label_summary.csv",
            "censoring_summary.csv",
            "run_manifest.json",
        ],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    write_summary_markdown(
        output_dir / "execution_outcome_label_summary.md",
        run_dir=run_dir,
        output_dir=output_dir,
        manifest=manifest,
        coverage_rows=coverage_rows,
        lifecycle_rows=lifecycle_rows,
        censoring_rows=censoring_rows,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory containing audit_live_*.csv and T009 artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        help=f"Output directory. Default: <run-dir>/stage5_execution_outcome_labels_{TASK_ID}",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=DEFAULT_TICK_SIZE,
        help=f"Tick size used for tick-normalized labels. Default: {DEFAULT_TICK_SIZE}",
    )
    parser.add_argument(
        "--maker-fee-bps",
        type=float,
        default=DEFAULT_MAKER_FEE_BPS,
        help=f"Maker fee assumption in bps for fee-adjusted proxy labels. Default: {DEFAULT_MAKER_FEE_BPS}",
    )
    parser.add_argument(
        "--max-future-gap-ms",
        type=float,
        default=DEFAULT_MAX_FUTURE_GAP_MS,
        help=f"Maximum allowed future-decision gap for markout labels. Default: {DEFAULT_MAX_FUTURE_GAP_MS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _expand(args.run_dir)
    output_dir = _expand(args.output_dir) if args.output_dir else run_dir / f"stage5_execution_outcome_labels_{TASK_ID}"
    manifest = run_execution_outcome_labels(
        run_dir=run_dir,
        output_dir=output_dir,
        tick_size=float(args.tick_size),
        max_future_gap_ms=float(args.max_future_gap_ms),
        maker_fee_bps=float(args.maker_fee_bps),
    )
    print(json.dumps({"task_id": TASK_ID, "output_dir": str(output_dir), "row_counts": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
