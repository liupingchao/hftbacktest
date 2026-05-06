#!/usr/bin/env python3
"""Compare audit_bt and audit_live with shared schema and alignment metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import mean
from typing import Any

from audit_schema import REQUIRED_ALIGNMENT_FIELDS


DECISION_EVENT_TYPES = {"", "0", "decision"}

LEGACY_OPTIONAL_ALIGNMENT_FIELDS = {
    "local_open_orders",
    "rest_open_orders",
    "open_order_diff",
    "safety_detail",
    "working_bid_qty",
    "working_ask_qty",
    "working_bid_status",
    "working_ask_status",
    "working_bid_req",
    "working_ask_req",
    "working_bid_pending_cancel",
    "working_ask_pending_cancel",
    "replay_scheduled_ts_local",
    "bt_feed_ts_local",
    "bt_feed_ts_exch",
    "replay_lag_ns",
    "replay_lag_abs_ns",
    "working_bid_qty",
    "working_ask_qty",
    "working_bid_status",
    "working_ask_status",
    "working_bid_req",
    "working_ask_req",
    "working_bid_pending_cancel",
    "working_ask_pending_cancel",
}

FIRST_DIVERGENCE_CONTEXT_ROWS = 20


def _expand(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        x = float(v)
        if math.isfinite(x):
            return x
        return default
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = [c for c in REQUIRED_ALIGNMENT_FIELDS if c not in cols]
        missing_required = [c for c in missing if c not in LEGACY_OPTIONAL_ALIGNMENT_FIELDS]
        if missing_required:
            raise KeyError(f"Missing required columns in {path}: {missing_required}")
        rows = list(reader)
        for row in rows:
            for field in missing:
                row.setdefault(field, "")
        return rows


def _decision_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if _is_decision_row(row)]


def _is_decision_row(row: dict[str, str]) -> bool:
    return str(row.get("event_type", "")).strip() in DECISION_EVENT_TYPES


def _series(rows: list[dict[str, str]], key: str, scale: float = 1.0, positive_only: bool = False) -> list[float]:
    out: list[float] = []
    for r in rows:
        x = _safe_float(r.get(key, 0.0), 0.0) * scale
        if positive_only and x <= 0:
            continue
        out.append(x)
    return out


def _distribution(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
    s = sorted(vals)
    return {
        "count": float(len(vals)),
        "mean": float(mean(vals)),
        "p50": float(_quantile(s, 0.50)),
        "p90": float(_quantile(s, 0.90)),
        "p99": float(_quantile(s, 0.99)),
    }


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


def _series_int(rows: list[dict[str, str]], key: str) -> list[int]:
    out = []
    for r in rows:
        try:
            v = _parse_int_timestamp(r.get(key, "") or "0")
        except ValueError:
            continue
        if v > 0:
            out.append(v)
    return out


def _dist(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    s = sorted(vals)
    return {
        "count": float(len(vals)),
        "mean": float(mean(vals)),
        "p50": float(_quantile(s, 0.50)),
        "p90": float(_quantile(s, 0.90)),
        "p99": float(_quantile(s, 0.99)),
        "max": float(s[-1]),
    }


def _cadence_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    ts = _series_int(rows, "ts_local")
    raw_deltas = [b - a for a, b in zip(ts, ts[1:])]
    deltas = [float(delta) for delta in raw_deltas if delta >= 0]
    return {
        "rows": len(rows),
        "ts_local_count": len(ts),
        "ts_local_first": ts[0] if ts else 0,
        "ts_local_last": ts[-1] if ts else 0,
        "negative_delta_count": sum(delta < 0 for delta in raw_deltas),
        "delta_ns": _dist(deltas),
    }


def _nearest_lag_stats(bt_rows: list[dict[str, str]], live_rows: list[dict[str, str]]) -> dict[str, Any]:
    import bisect

    bt_ts = sorted(_series_int(bt_rows, "ts_local"))
    live_ts = _series_int(live_rows, "ts_local")
    signed = []
    for ts in live_ts:
        i = bisect.bisect_left(bt_ts, ts)
        candidates = []
        if i < len(bt_ts):
            candidates.append(bt_ts[i])
        if i > 0:
            candidates.append(bt_ts[i - 1])
        if candidates:
            nearest = min(candidates, key=lambda x: abs(x - ts))
            signed.append(float(nearest - ts))
    return {
        "count": len(signed),
        "abs_ns": _dist([abs(v) for v in signed]),
        "signed_ns": _dist(signed),
    }


def _summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {
            "rows": 0,
            "drop_latency_rate": 0.0,
            "drop_api_rate": 0.0,
            "entry_latency_ms": _distribution([]),
            "auditlatency_ms": _distribution([]),
            "inventory_score": _distribution([]),
            "spread_bps": _distribution([]),
            "vol_bps": _distribution([]),
        }

    drop_latency = sum(_safe_int(r.get("dropped_by_latency", 0)) > 0 for r in rows)
    drop_api = sum(_safe_int(r.get("dropped_by_api_limit", 0)) > 0 for r in rows)

    entry_ms = _series(rows, "entry_latency_ns", scale=1.0 / 1_000_000.0, positive_only=True)
    audit_ms = _series(rows, "auditlatency_ms", scale=1.0, positive_only=False)
    inv = _series(rows, "inventory_score")
    spread_bps = _series(rows, "spread_bps")
    vol_bps = _series(rows, "vol_bps")

    return {
        "rows": n,
        "drop_latency_rate": float(drop_latency / n),
        "drop_api_rate": float(drop_api / n),
        "entry_latency_ms": _distribution(entry_ms),
        "auditlatency_ms": _distribution(audit_ms),
        "inventory_score": _distribution(inv),
        "spread_bps": _distribution(spread_bps),
        "vol_bps": _distribution(vol_bps),
    }


def _align_by_seq(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for r in rows:
        seq = _safe_int(r.get("strategy_seq", 0), 0)
        if seq <= 0:
            continue
        out[seq] = r
    return out


def _mae(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(abs(x - y) for x, y in zip(a, b)) / len(a))


def _counter_top(counter: Counter[tuple[str, str]], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"bt": bt, "live": live, "count": count}
        for (bt, live), count in counter.most_common(limit)
    ]


def _counter_key_top(counter: Counter[str], limit: int = 20) -> list[dict[str, Any]]:
    return [
        {"key": key, "count": count}
        for key, count in counter.most_common(limit)
    ]


def _ratio(num: int, den: int) -> float:
    return float(num / den) if den else 0.0


def _parse_optional_int(raw: str) -> int | None:
    try:
        return _parse_int_timestamp(raw)
    except ValueError:
        return None


def _abs_tick_diff(b: dict[str, str], l: dict[str, str]) -> int | None:
    b_bid = _parse_optional_int(str(b.get("target_bid_tick", "") or ""))
    b_ask = _parse_optional_int(str(b.get("target_ask_tick", "") or ""))
    l_bid = _parse_optional_int(str(l.get("target_bid_tick", "") or ""))
    l_ask = _parse_optional_int(str(l.get("target_ask_tick", "") or ""))
    if b_bid is None or b_ask is None or l_bid is None or l_ask is None:
        return None
    return max(abs(b_bid - l_bid), abs(b_ask - l_ask))


def _abs_ts_exch_diff_ns(b: dict[str, str], l: dict[str, str]) -> int | None:
    b_ts = _parse_optional_int(str(b.get("ts_exch", "") or ""))
    l_ts = _parse_optional_int(str(l.get("ts_exch", "") or ""))
    if b_ts is None or l_ts is None or b_ts <= 0 or l_ts <= 0:
        return None
    return abs(b_ts - l_ts)


def _same_target_ticks(b: dict[str, str], l: dict[str, str]) -> bool:
    return (
        str(b.get("target_bid_tick", "")) == str(l.get("target_bid_tick", ""))
        and str(b.get("target_ask_tick", "")) == str(l.get("target_ask_tick", ""))
    )


def _same_working_ticks(b: dict[str, str], l: dict[str, str]) -> bool:
    return (
        str(b.get("working_bid_tick", "")) == str(l.get("working_bid_tick", ""))
        and str(b.get("working_ask_tick", "")) == str(l.get("working_ask_tick", ""))
    )


def _same_working_order_ids(b: dict[str, str], l: dict[str, str]) -> bool:
    return (
        str(b.get("working_buy_order_id", "")) == str(l.get("working_buy_order_id", ""))
        and str(b.get("working_sell_order_id", "")) == str(l.get("working_sell_order_id", ""))
    )


def _same_extra_orders(b: dict[str, str], l: dict[str, str]) -> bool:
    return (
        str(b.get("extra_order_ids", "")) == str(l.get("extra_order_ids", ""))
        and str(b.get("extra_order_sides", "")) == str(l.get("extra_order_sides", ""))
        and str(b.get("extra_order_price_ticks", "")) == str(l.get("extra_order_price_ticks", ""))
    )


def _same_open_order_counts(b: dict[str, str], l: dict[str, str]) -> bool:
    return str(b.get("local_open_order_count", "")) == str(l.get("local_open_order_count", ""))


def _same_open_order_detail(b: dict[str, str], l: dict[str, str]) -> bool:
    return str(b.get("local_open_orders", "")) == str(l.get("local_open_orders", ""))


def _status_empty_or_ok(raw: str) -> bool:
    return raw in {"", "0", "ok", "safety_disabled"}


def _normalize_decimal_text(raw: str) -> str:
    text = str(raw).strip()
    if not text:
        return ""
    try:
        value = Decimal(text)
    except InvalidOperation:
        return text
    if not value.is_finite():
        return text
    if value == 0:
        return "0"
    normalized = format(value.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized


def _parse_open_order_item(item: str) -> dict[str, str] | None:
    parts = [part.strip() for part in item.split(":")]
    if len(parts) < 3:
        return None

    extras: dict[str, str] = {}
    status = ""
    for token in parts[4:]:
        if "=" in token:
            key, value = token.split("=", 1)
            extras[key.strip().lower()] = value.strip().lower()
        elif not status:
            status = token.strip().lower()

    if not status:
        status = extras.get("status", "")

    req = extras.get("req", "")
    cancel_requested = extras.get("cancel_requested", "")
    pending_cancel = (
        req == "cancel"
        or status == "pending_cancel"
        or cancel_requested in {"1", "true", "yes"}
    )
    return {
        "id": parts[0],
        "side": parts[1].lower(),
        "price_tick": _normalize_decimal_text(parts[2]),
        "qty": _normalize_decimal_text(parts[3]) if len(parts) >= 4 else "",
        "status": status,
        "req": req,
        "pending_cancel": "1" if pending_cancel else "0",
    }


def _parse_open_orders(serialized: str) -> list[dict[str, str]]:
    orders: list[dict[str, str]] = []
    for item in str(serialized or "").split(";"):
        item = item.strip()
        if not item:
            continue
        parsed = _parse_open_order_item(item)
        if parsed is not None:
            orders.append(parsed)
    return orders


def _order_semantic_key(order: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(order.get("side", "")),
        str(order.get("price_tick", "")),
        str(order.get("qty", "")),
    )


def _order_side_price_key(order: dict[str, str]) -> tuple[str, str]:
    return (
        str(order.get("side", "")),
        str(order.get("price_tick", "")),
    )


def _order_side_qty_key(order: dict[str, str]) -> tuple[str, str]:
    return (
        str(order.get("side", "")),
        str(order.get("qty", "")),
    )


def _semantic_counter(
    orders: list[dict[str, str]],
    key_fn: Any = _order_semantic_key,
) -> Counter[tuple[str, ...]]:
    return Counter(key_fn(order) for order in orders)


def _append_unique(out: list[str], category: str) -> None:
    if category not in out:
        out.append(category)


def _has_explicit_working_semantics(row: dict[str, str]) -> bool:
    keys = [
        "working_bid_qty",
        "working_ask_qty",
        "working_bid_status",
        "working_ask_status",
        "working_bid_req",
        "working_ask_req",
        "working_bid_pending_cancel",
        "working_ask_pending_cancel",
    ]
    return any(str(row.get(key, "") or "").strip() for key in keys)


def _explicit_side_order(row: dict[str, str], side: str) -> dict[str, str] | None:
    prefix = "bid" if side == "buy" else "ask"
    tick_key = "working_bid_tick" if side == "buy" else "working_ask_tick"
    tick = str(row.get(tick_key, "") or "").strip()
    if tick in {"", "-1"}:
        return None
    req = str(row.get(f"working_{prefix}_req", "") or "").strip().lower()
    pending = str(row.get(f"working_{prefix}_pending_cancel", "") or "").strip()
    return {
        "id": str(row.get("working_buy_order_id" if side == "buy" else "working_sell_order_id", "") or ""),
        "side": side,
        "price_tick": _normalize_decimal_text(tick),
        "qty": _normalize_decimal_text(str(row.get(f"working_{prefix}_qty", "") or "")),
        "status": str(row.get(f"working_{prefix}_status", "") or "").strip().lower(),
        "req": req,
        "pending_cancel": "1" if pending in {"1", "true", "yes"} or req == "cancel" else "0",
    }


def _explicit_working_orders(row: dict[str, str]) -> list[dict[str, str]]:
    orders: list[dict[str, str]] = []
    for side in ("buy", "sell"):
        order = _explicit_side_order(row, side)
        if order is not None:
            orders.append(order)
    return orders


def _semantic_orders(row: dict[str, str]) -> list[dict[str, str]]:
    if _has_explicit_working_semantics(row):
        return _explicit_working_orders(row)
    return _parse_open_orders(str(row.get("local_open_orders", "") or ""))


def _local_order_semantic_categories(b: dict[str, str], l: dict[str, str]) -> list[str]:
    categories: list[str] = []

    if not _same_working_ticks(b, l):
        _append_unique(categories, "price_tick_mismatch")

    b_orders = _semantic_orders(b)
    l_orders = _semantic_orders(l)
    b_keys = _semantic_counter(b_orders)
    l_keys = _semantic_counter(l_orders)

    if b_keys != l_keys:
        if l_keys - b_keys:
            _append_unique(categories, "missing_local_order")
        if b_keys - l_keys:
            _append_unique(categories, "extra_local_order")
        if _semantic_counter(b_orders, _order_side_price_key) != _semantic_counter(l_orders, _order_side_price_key):
            _append_unique(categories, "price_tick_mismatch")
        if _semantic_counter(b_orders, _order_side_qty_key) != _semantic_counter(l_orders, _order_side_qty_key):
            _append_unique(categories, "qty_mismatch")
    else:
        b_state = Counter(
            (_order_semantic_key(order), str(order.get("status", "")))
            for order in b_orders
        )
        l_state = Counter(
            (_order_semantic_key(order), str(order.get("status", "")))
            for order in l_orders
        )
        if b_state != l_state:
            _append_unique(categories, "state_mismatch")

        b_pending = Counter(
            (_order_semantic_key(order), str(order.get("pending_cancel", "")))
            for order in b_orders
        )
        l_pending = Counter(
            (_order_semantic_key(order), str(order.get("pending_cancel", "")))
            for order in l_orders
        )
        if b_pending != l_pending:
            _append_unique(categories, "pending_cancel_mismatch")

    if not _same_open_order_counts(b, l):
        _append_unique(categories, "active_order_count_mismatch")

    return categories


def _identity_lifecycle_categories(b: dict[str, str], l: dict[str, str]) -> list[str]:
    categories: list[str] = []
    if not _same_working_order_ids(b, l):
        categories.append("working_order_id")
    if not _same_extra_orders(b, l):
        categories.append("extra_order_identity")
    if not _same_open_order_detail(b, l):
        categories.append("local_open_order_detail")
    return categories


def _row_has_rest_local_divergence(row: dict[str, str]) -> bool:
    open_order_diff_value = str(row.get("open_order_diff", "") or "").strip()
    if open_order_diff_value:
        return True
    status = str(row.get("safety_status", "") or "").strip()
    return bool(status and not _status_empty_or_ok(status))


def _pair_has_rest_local_divergence(b: dict[str, str], l: dict[str, str]) -> bool:
    return _row_has_rest_local_divergence(b) or _row_has_rest_local_divergence(l)


def _pair_has_working_order_semantic_mismatch(b: dict[str, str], l: dict[str, str]) -> bool:
    return bool(_local_order_semantic_categories(b, l))


def _pair_has_identity_only_mismatch(b: dict[str, str], l: dict[str, str]) -> bool:
    return (
        not _pair_has_working_order_semantic_mismatch(b, l)
        and bool(_identity_lifecycle_categories(b, l))
    )


def _replay_lag_ns(row: dict[str, str]) -> int | None:
    raw_abs = str(row.get("replay_lag_abs_ns", "") or "")
    parsed_abs = _parse_optional_int(raw_abs)
    if parsed_abs is not None and parsed_abs > 0:
        return parsed_abs
    raw = str(row.get("replay_lag_ns", "") or "")
    parsed = _parse_optional_int(raw)
    if parsed is None:
        return None
    return abs(parsed)


def _pair_replay_lag_ns(b: dict[str, str], _l: dict[str, str]) -> int | None:
    return _replay_lag_ns(b)


def _pair_exchange_replay_lag_ns(b: dict[str, str], l: dict[str, str]) -> int | None:
    bt_ts = _parse_optional_int(str(b.get("bt_feed_ts_exch", "") or ""))
    if bt_ts is None or bt_ts <= 0:
        bt_ts = _parse_optional_int(str(b.get("ts_exch", "") or ""))
    live_ts = _parse_optional_int(str(l.get("ts_exch", "") or ""))
    if bt_ts is None or live_ts is None or bt_ts <= 0 or live_ts <= 0:
        return None
    return abs(bt_ts - live_ts)


def _pair_in_replay_lag_gate(b: dict[str, str], l: dict[str, str], max_lag_ns: int) -> bool:
    lag = _pair_replay_lag_ns(b, l)
    if lag is None:
        return False
    return lag <= max_lag_ns


def _pair_in_exchange_replay_lag_gate(b: dict[str, str], l: dict[str, str], max_lag_ns: int) -> bool:
    lag = _pair_exchange_replay_lag_ns(b, l)
    if lag is None:
        return False
    return lag <= max_lag_ns


def _pair_in_dual_replay_lag_gate(b: dict[str, str], l: dict[str, str], max_lag_ns: int) -> bool:
    return _pair_in_replay_lag_gate(b, l, max_lag_ns) and _pair_in_exchange_replay_lag_gate(b, l, max_lag_ns)


def _api_throttle_pair_mismatched(b: dict[str, str], l: dict[str, str]) -> bool:
    return (
        (_safe_int(b.get("dropped_by_api_limit", 0)) > 0)
        != (_safe_int(l.get("dropped_by_api_limit", 0)) > 0)
        or str(b.get("throttle_reason", "")) != str(l.get("throttle_reason", ""))
    )


def _ts_exch_abs_diff_le(b: dict[str, str], l: dict[str, str], threshold_ns: int) -> bool:
    diff = _abs_ts_exch_diff_ns(b, l)
    return diff is not None and diff <= threshold_ns


def _ts_exch_abs_diff_gt(b: dict[str, str], l: dict[str, str], threshold_ns: int) -> bool:
    diff = _abs_ts_exch_diff_ns(b, l)
    return diff is not None and diff > threshold_ns


def _working_order_lifecycle_categories(b: dict[str, str], l: dict[str, str]) -> list[str]:
    categories: list[str] = []
    if not _same_working_ticks(b, l):
        categories.append("working_tick")
    if not _same_working_order_ids(b, l):
        categories.append("working_order_id")
    if not _same_extra_orders(b, l):
        categories.append("extra_order")
    if not _same_open_order_counts(b, l):
        categories.append("local_open_order_count")
    if not _same_open_order_detail(b, l):
        categories.append("local_open_order_detail")
    if str(b.get("rest_open_order_count", "")) != str(l.get("rest_open_order_count", "")):
        categories.append("rest_open_order_count")
    if str(b.get("rest_open_orders", "")) != str(l.get("rest_open_orders", "")):
        categories.append("rest_open_order_detail")
    if str(b.get("open_order_diff", "")) != str(l.get("open_order_diff", "")):
        categories.append("open_order_diff")
    b_status = str(b.get("safety_status", ""))
    l_status = str(l.get("safety_status", ""))
    if b_status != l_status and not (_status_empty_or_ok(b_status) and _status_empty_or_ok(l_status)):
        categories.append("safety_status")
    if not categories:
        categories.append("matched")
    return categories


def _pair_has_working_order_lifecycle_mismatch(b: dict[str, str], l: dict[str, str]) -> bool:
    return _working_order_lifecycle_categories(b, l) != ["matched"]


def _compact_pair_row(
    b: dict[str, str],
    l: dict[str, str],
    categories: list[str],
    identity_categories: list[str],
) -> dict[str, Any]:
    return {
        "strategy_seq": _safe_int(b.get("strategy_seq", 0), 0),
        "bt_ts_local": str(b.get("ts_local", "")),
        "live_ts_local": str(l.get("ts_local", "")),
        "bt_ts_exch": str(b.get("ts_exch", "")),
        "live_ts_exch": str(l.get("ts_exch", "")),
        "replay_scheduled_ts_local": str(b.get("replay_scheduled_ts_local", "")),
        "bt_feed_ts_local": str(b.get("bt_feed_ts_local", "")),
        "bt_feed_ts_exch": str(b.get("bt_feed_ts_exch", "")),
        "bt_replay_lag_abs_ns": str(b.get("replay_lag_abs_ns", "")),
        "bt_exchange_replay_lag_abs_ns": (
            str(_pair_exchange_replay_lag_ns(b, l))
            if _pair_exchange_replay_lag_ns(b, l) is not None
            else ""
        ),
        "categories": categories,
        "identity_categories": identity_categories,
        "bt_action": str(b.get("action", "")),
        "live_action": str(l.get("action", "")),
        "bt_planned_action": str(b.get("planned_action", "")),
        "live_planned_action": str(l.get("planned_action", "")),
        "bt_target_bid_tick": str(b.get("target_bid_tick", "")),
        "live_target_bid_tick": str(l.get("target_bid_tick", "")),
        "bt_target_ask_tick": str(b.get("target_ask_tick", "")),
        "live_target_ask_tick": str(l.get("target_ask_tick", "")),
        "bt_working_bid_tick": str(b.get("working_bid_tick", "")),
        "live_working_bid_tick": str(l.get("working_bid_tick", "")),
        "bt_working_ask_tick": str(b.get("working_ask_tick", "")),
        "live_working_ask_tick": str(l.get("working_ask_tick", "")),
        "bt_working_bid_qty": str(b.get("working_bid_qty", "")),
        "live_working_bid_qty": str(l.get("working_bid_qty", "")),
        "bt_working_ask_qty": str(b.get("working_ask_qty", "")),
        "live_working_ask_qty": str(l.get("working_ask_qty", "")),
        "bt_working_bid_status": str(b.get("working_bid_status", "")),
        "live_working_bid_status": str(l.get("working_bid_status", "")),
        "bt_working_ask_status": str(b.get("working_ask_status", "")),
        "live_working_ask_status": str(l.get("working_ask_status", "")),
        "bt_working_bid_req": str(b.get("working_bid_req", "")),
        "live_working_bid_req": str(l.get("working_bid_req", "")),
        "bt_working_ask_req": str(b.get("working_ask_req", "")),
        "live_working_ask_req": str(l.get("working_ask_req", "")),
        "bt_working_bid_pending_cancel": str(b.get("working_bid_pending_cancel", "")),
        "live_working_bid_pending_cancel": str(l.get("working_bid_pending_cancel", "")),
        "bt_working_ask_pending_cancel": str(b.get("working_ask_pending_cancel", "")),
        "live_working_ask_pending_cancel": str(l.get("working_ask_pending_cancel", "")),
        "bt_local_open_order_count": str(b.get("local_open_order_count", "")),
        "live_local_open_order_count": str(l.get("local_open_order_count", "")),
        "bt_local_open_orders": str(b.get("local_open_orders", ""))[:500],
        "live_local_open_orders": str(l.get("local_open_orders", ""))[:500],
        "bt_rest_open_order_count": str(b.get("rest_open_order_count", "")),
        "live_rest_open_order_count": str(l.get("rest_open_order_count", "")),
        "bt_open_order_diff": str(b.get("open_order_diff", ""))[:500],
        "live_open_order_diff": str(l.get("open_order_diff", ""))[:500],
        "bt_safety_status": str(b.get("safety_status", "")),
        "live_safety_status": str(l.get("safety_status", "")),
    }


def _compact_event_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "strategy_seq": _safe_int(row.get("strategy_seq", 0), 0),
        "event_seq": _safe_int(row.get("event_seq", 0), 0),
        "event_type": str(row.get("event_type", "")),
        "event_source": str(row.get("event_source", "")),
        "ts_local": str(row.get("ts_local", "")),
        "ts_exch": str(row.get("ts_exch", "")),
        "replay_scheduled_ts_local": str(row.get("replay_scheduled_ts_local", "")),
        "bt_feed_ts_local": str(row.get("bt_feed_ts_local", "")),
        "bt_feed_ts_exch": str(row.get("bt_feed_ts_exch", "")),
        "replay_lag_abs_ns": str(row.get("replay_lag_abs_ns", "")),
        "action": str(row.get("action", "")),
        "planned_action": str(row.get("planned_action", "")),
        "reject_reason": str(row.get("reject_reason", "")),
        "throttle_reason": str(row.get("throttle_reason", "")),
        "order_id": str(row.get("order_id", "")),
        "linked_order_id": str(row.get("linked_order_id", "")),
        "linked_action": str(row.get("linked_action", "")),
        "order_side": str(row.get("order_side", "")),
        "order_price_tick": str(row.get("order_price_tick", "")),
        "order_qty": str(row.get("order_qty", "")),
        "order_status": str(row.get("order_status", "")),
        "lifecycle_state": str(row.get("lifecycle_state", "")),
        "cancel_requested": str(row.get("cancel_requested", "")),
        "cancel_request_ts": str(row.get("cancel_request_ts", "")),
        "cancel_ack_ts": str(row.get("cancel_ack_ts", "")),
        "fill_ts": str(row.get("fill_ts", "")),
        "fill_qty": str(row.get("fill_qty", "")),
        "fill_price": str(row.get("fill_price", "")),
        "local_order_seen": str(row.get("local_order_seen", "")),
        "rest_order_seen": str(row.get("rest_order_seen", "")),
        "ws_order_seen": str(row.get("ws_order_seen", "")),
        "working_bid_tick": str(row.get("working_bid_tick", "")),
        "working_ask_tick": str(row.get("working_ask_tick", "")),
        "working_buy_order_id": str(row.get("working_buy_order_id", "")),
        "working_sell_order_id": str(row.get("working_sell_order_id", "")),
        "working_bid_qty": str(row.get("working_bid_qty", "")),
        "working_ask_qty": str(row.get("working_ask_qty", "")),
        "working_bid_status": str(row.get("working_bid_status", "")),
        "working_ask_status": str(row.get("working_ask_status", "")),
        "working_bid_req": str(row.get("working_bid_req", "")),
        "working_ask_req": str(row.get("working_ask_req", "")),
        "working_bid_pending_cancel": str(row.get("working_bid_pending_cancel", "")),
        "working_ask_pending_cancel": str(row.get("working_ask_pending_cancel", "")),
        "local_open_order_count": str(row.get("local_open_order_count", "")),
        "rest_open_order_count": str(row.get("rest_open_order_count", "")),
        "safety_status": str(row.get("safety_status", "")),
    }
    detail_fields = [
        "local_open_orders",
        "rest_open_orders",
        "open_order_diff",
        "lifecycle_detail",
        "safety_detail",
    ]
    for field in detail_fields:
        value = str(row.get(field, "") or "")
        if value:
            out[field] = value[:500]
    return out


def _rows_around_seq(
    rows: list[dict[str, str]],
    seq: int,
    *,
    before: int = FIRST_DIVERGENCE_CONTEXT_ROWS,
    after: int = FIRST_DIVERGENCE_CONTEXT_ROWS,
) -> list[dict[str, Any]]:
    if seq <= 0:
        return []
    indexed: list[tuple[int, dict[str, str]]] = []
    for row in rows:
        row_seq = _safe_int(row.get("strategy_seq", 0), 0)
        if seq - before <= row_seq <= seq + after:
            indexed.append((row_seq, row))
    indexed.sort(
        key=lambda item: (
            item[0],
            _safe_int(item[1].get("event_seq", 0), 0),
            0 if _is_decision_row(item[1]) else 1,
            str(item[1].get("event_type", "")),
        )
    )
    return [_compact_event_row(row) for _, row in indexed]


def _first_semantic_divergence_context(
    first: dict[str, Any],
    bt_rows: list[dict[str, str]],
    live_rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not first:
        return {}
    seq = _safe_int(first.get("strategy_seq", 0), 0)
    return {
        "divergence": first,
        "context_before_rows": FIRST_DIVERGENCE_CONTEXT_ROWS,
        "context_after_rows": FIRST_DIVERGENCE_CONTEXT_ROWS,
        "bt_rows": _rows_around_seq(bt_rows, seq),
        "live_rows": _rows_around_seq(live_rows, seq),
    }


def _working_order_lifecycle_breakdown(pairs: list[tuple[dict[str, str], dict[str, str]]]) -> dict[str, Any]:
    n = len(pairs)
    category_rows: Counter[str] = Counter()
    semantic_category_rows: Counter[str] = Counter()
    identity_category_rows: Counter[str] = Counter()
    category_pair_top: Counter[tuple[str, str]] = Counter()
    planned_mismatch_category_rows: Counter[str] = Counter()
    planned_mismatch_semantic_category_rows: Counter[str] = Counter()
    api_throttle_mismatch_category_rows: Counter[str] = Counter()
    api_throttle_mismatch_semantic_category_rows: Counter[str] = Counter()

    mismatch_rows = 0
    semantic_mismatch_rows = 0
    identity_only_mismatch_rows = 0
    rest_local_divergence_rows = 0
    planned_mismatch_rows = 0
    api_throttle_mismatch_rows = 0
    first_semantic_divergence: dict[str, Any] | None = None
    first_identity_only_divergence: dict[str, Any] | None = None
    for b, l in pairs:
        categories = _working_order_lifecycle_categories(b, l)
        semantic_categories = _local_order_semantic_categories(b, l)
        identity_categories = _identity_lifecycle_categories(b, l)
        has_mismatch = categories != ["matched"]
        has_semantic_mismatch = bool(semantic_categories)
        has_identity_only_mismatch = (
            has_mismatch
            and not has_semantic_mismatch
            and bool(identity_categories)
        )
        if has_mismatch:
            mismatch_rows += 1
        if has_semantic_mismatch:
            semantic_mismatch_rows += 1
            if first_semantic_divergence is None:
                first_semantic_divergence = _compact_pair_row(
                    b,
                    l,
                    semantic_categories,
                    identity_categories,
                )
        if has_identity_only_mismatch:
            identity_only_mismatch_rows += 1
            if first_identity_only_divergence is None:
                first_identity_only_divergence = _compact_pair_row(
                    b,
                    l,
                    categories,
                    identity_categories,
                )
        if _pair_has_rest_local_divergence(b, l):
            rest_local_divergence_rows += 1
        if str(b.get("planned_action", "")) != str(l.get("planned_action", "")):
            planned_mismatch_rows += 1
            for category in categories:
                if category != "matched":
                    planned_mismatch_category_rows[category] += 1
            for category in semantic_categories:
                planned_mismatch_semantic_category_rows[category] += 1
        if _api_throttle_pair_mismatched(b, l):
            api_throttle_mismatch_rows += 1
            for category in categories:
                if category != "matched":
                    api_throttle_mismatch_category_rows[category] += 1
            for category in semantic_categories:
                api_throttle_mismatch_semantic_category_rows[category] += 1
        for category in categories:
            category_rows[category] += 1
        for category in semantic_categories:
            semantic_category_rows[category] += 1
        for category in identity_categories:
            identity_category_rows[category] += 1
        category_pair_top[(
            str(b.get("local_open_orders", ""))[:160],
            str(l.get("local_open_orders", ""))[:160],
        )] += 1

    semantic_mismatch_pairs = [
        (b, l)
        for b, l in pairs
        if _pair_has_working_order_semantic_mismatch(b, l)
    ]
    identity_only_pairs = [
        (b, l)
        for b, l in pairs
        if _pair_has_identity_only_mismatch(b, l)
    ]
    semantic_matched_pairs = [
        (b, l)
        for b, l in pairs
        if not _pair_has_working_order_semantic_mismatch(b, l)
    ]

    return {
        "rows": n,
        "mismatch_rows": mismatch_rows,
        "mismatch_rate": _ratio(mismatch_rows, n),
        "semantic_mismatch_rows": semantic_mismatch_rows,
        "semantic_mismatch_rate": _ratio(semantic_mismatch_rows, n),
        "identity_only_mismatch_rows": identity_only_mismatch_rows,
        "identity_only_mismatch_rate": _ratio(identity_only_mismatch_rows, n),
        "rest_local_divergence_rows": rest_local_divergence_rows,
        "rest_local_divergence_rate": _ratio(rest_local_divergence_rows, n),
        "category_rows": _counter_key_top(category_rows),
        "semantic_category_rows": _counter_key_top(semantic_category_rows),
        "identity_category_rows": _counter_key_top(identity_category_rows),
        "planned_action_mismatch_rows": planned_mismatch_rows,
        "planned_action_mismatch_with_lifecycle_mismatch_rows": sum(
            1
            for b, l in pairs
            if str(b.get("planned_action", "")) != str(l.get("planned_action", ""))
            and _pair_has_working_order_lifecycle_mismatch(b, l)
        ),
        "planned_action_mismatch_with_semantic_mismatch_rows": sum(
            1
            for b, l in pairs
            if str(b.get("planned_action", "")) != str(l.get("planned_action", ""))
            and _pair_has_working_order_semantic_mismatch(b, l)
        ),
        "planned_action_mismatch_category_rows": _counter_key_top(planned_mismatch_category_rows),
        "planned_action_mismatch_semantic_category_rows": _counter_key_top(
            planned_mismatch_semantic_category_rows
        ),
        "api_throttle_mismatch_rows": api_throttle_mismatch_rows,
        "api_throttle_mismatch_with_lifecycle_mismatch_rows": sum(
            1
            for b, l in pairs
            if _api_throttle_pair_mismatched(b, l)
            and _pair_has_working_order_lifecycle_mismatch(b, l)
        ),
        "api_throttle_mismatch_with_semantic_mismatch_rows": sum(
            1
            for b, l in pairs
            if _api_throttle_pair_mismatched(b, l)
            and _pair_has_working_order_semantic_mismatch(b, l)
        ),
        "api_throttle_mismatch_category_rows": _counter_key_top(api_throttle_mismatch_category_rows),
        "api_throttle_mismatch_semantic_category_rows": _counter_key_top(
            api_throttle_mismatch_semantic_category_rows
        ),
        "semantic_mismatch_subset": _api_throttle_subset_summary(semantic_mismatch_pairs),
        "semantic_matched_subset": _api_throttle_subset_summary(semantic_matched_pairs),
        "identity_only_mismatch_subset": _api_throttle_subset_summary(identity_only_pairs),
        "first_semantic_divergence": first_semantic_divergence or {},
        "first_identity_only_divergence": first_identity_only_divergence or {},
        "local_open_orders_pair_top": _counter_top(category_pair_top, limit=10),
    }


def _stateful_replay_gate_summary(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    max_lag_ns: int,
) -> dict[str, Any]:
    first_outside_dual_gate: dict[str, Any] | None = None
    clean_prefix_rows = 0
    current_streak = 0
    longest_streak = 0

    for b, l in pairs:
        in_dual_gate = _pair_in_dual_replay_lag_gate(b, l, max_lag_ns)
        if in_dual_gate:
            current_streak += 1
            longest_streak = max(longest_streak, current_streak)
            if first_outside_dual_gate is None:
                clean_prefix_rows += 1
            continue

        current_streak = 0
        if first_outside_dual_gate is None:
            first_outside_dual_gate = _compact_pair_row(
                b,
                l,
                ["outside_dual_replay_lag_gate"],
                [],
            )

    state_contaminated_rows = len(pairs) - clean_prefix_rows
    return {
        "strict_full_window_passed": first_outside_dual_gate is None,
        "clean_prefix_rows": clean_prefix_rows,
        "state_contaminated_rows": state_contaminated_rows,
        "clean_prefix_rate": _ratio(clean_prefix_rows, len(pairs)),
        "longest_contiguous_dual_gate_rows": longest_streak,
        "first_outside_dual_gate": first_outside_dual_gate or {},
    }


def _replay_lag_summary(pairs: list[tuple[dict[str, str], dict[str, str]]], max_lag_ns: int) -> dict[str, Any]:
    lags = [
        float(lag)
        for b, l in pairs
        if (lag := _pair_replay_lag_ns(b, l)) is not None
    ]
    exch_lags = [
        float(lag)
        for b, l in pairs
        if (lag := _pair_exchange_replay_lag_ns(b, l)) is not None
    ]
    in_gate = [
        (b, l)
        for b, l in pairs
        if _pair_in_replay_lag_gate(b, l, max_lag_ns)
    ]
    in_exchange_gate = [
        (b, l)
        for b, l in pairs
        if _pair_in_exchange_replay_lag_gate(b, l, max_lag_ns)
    ]
    in_dual_gate = [
        (b, l)
        for b, l in pairs
        if _pair_in_dual_replay_lag_gate(b, l, max_lag_ns)
    ]
    outside_gate = [
        (b, l)
        for b, l in pairs
        if (lag := _pair_replay_lag_ns(b, l)) is not None and lag > max_lag_ns
    ]
    outside_exchange_gate = [
        (b, l)
        for b, l in pairs
        if (lag := _pair_exchange_replay_lag_ns(b, l)) is not None and lag > max_lag_ns
    ]
    outside_dual_gate = [
        (b, l)
        for b, l in pairs
        if not _pair_in_dual_replay_lag_gate(b, l, max_lag_ns)
    ]
    missing_lag = len(pairs) - len(lags)
    missing_exchange_lag = len(pairs) - len(exch_lags)
    buckets = {
        "le_50ms": [
            (b, l)
            for b, l in pairs
            if (lag := _pair_replay_lag_ns(b, l)) is not None and lag <= 50_000_000
        ],
        "le_250ms": [
            (b, l)
            for b, l in pairs
            if (lag := _pair_replay_lag_ns(b, l)) is not None and lag <= 250_000_000
        ],
        "le_1000ms": [
            (b, l)
            for b, l in pairs
            if (lag := _pair_replay_lag_ns(b, l)) is not None and lag <= 1_000_000_000
        ],
        "gt_1000ms": [
            (b, l)
            for b, l in pairs
            if (lag := _pair_replay_lag_ns(b, l)) is not None and lag > 1_000_000_000
        ],
    }
    return {
        "max_lag_ns": int(max_lag_ns),
        "rows": len(pairs),
        "rows_with_lag": len(lags),
        "missing_lag_rows": missing_lag,
        "in_gate_rows": len(in_gate),
        "outside_gate_rows": len(outside_gate),
        "in_gate_rate": _ratio(len(in_gate), len(pairs)),
        "rows_with_exchange_lag": len(exch_lags),
        "missing_exchange_lag_rows": missing_exchange_lag,
        "in_exchange_gate_rows": len(in_exchange_gate),
        "outside_exchange_gate_rows": len(outside_exchange_gate),
        "in_exchange_gate_rate": _ratio(len(in_exchange_gate), len(pairs)),
        "in_dual_gate_rows": len(in_dual_gate),
        "outside_dual_gate_rows": len(outside_dual_gate),
        "in_dual_gate_rate": _ratio(len(in_dual_gate), len(pairs)),
        "abs_ns": _dist(lags),
        "exchange_abs_ns": _dist(exch_lags),
        "buckets": {
            name: _api_throttle_subset_summary(bucket_pairs)
            for name, bucket_pairs in buckets.items()
        },
        "local_gate_subset": _api_throttle_subset_summary(in_gate),
        "exchange_gate_subset": _api_throttle_subset_summary(in_exchange_gate),
        "dual_gate_subset": _api_throttle_subset_summary(in_dual_gate),
        "stateful_gate": _stateful_replay_gate_summary(pairs, max_lag_ns),
        "lifecycle": {
            "local_gate_subset": _working_order_lifecycle_breakdown(in_gate),
            "exchange_gate_subset": _working_order_lifecycle_breakdown(in_exchange_gate),
            "dual_gate_subset": _working_order_lifecycle_breakdown(in_dual_gate),
            "outside_dual_gate_subset": _working_order_lifecycle_breakdown(outside_dual_gate),
        },
    }


def _api_throttle_subset_summary(pairs: list[tuple[dict[str, str], dict[str, str]]]) -> dict[str, Any]:
    n = len(pairs)
    if n == 0:
        return {
            "rows": 0,
            "api_drop_rate_bt": 0.0,
            "api_drop_rate_live": 0.0,
            "api_drop_abs_diff": 0.0,
            "reject_reason_match_rate": 0.0,
            "throttle_reason_match_rate": 0.0,
            "planned_action_match_rate": 0.0,
            "same_target_ticks_rate": 0.0,
            "same_working_ticks_rate": 0.0,
            "target_tick_abs_diff": _dist([]),
            "ts_exch_abs_diff_ms": _dist([]),
            "reject_reason_mismatch_top": [],
            "throttle_reason_mismatch_top": [],
        }

    bt_api = 0
    live_api = 0
    reject_match = 0
    throttle_match = 0
    planned_match = 0
    same_targets = 0
    same_working = 0
    target_diffs: list[float] = []
    ts_exch_diffs_ms: list[float] = []
    reject_mismatches: Counter[tuple[str, str]] = Counter()
    throttle_mismatches: Counter[tuple[str, str]] = Counter()

    for b, l in pairs:
        if _safe_int(b.get("dropped_by_api_limit", 0)) > 0:
            bt_api += 1
        if _safe_int(l.get("dropped_by_api_limit", 0)) > 0:
            live_api += 1

        b_reject = str(b.get("reject_reason", ""))
        l_reject = str(l.get("reject_reason", ""))
        b_throttle = str(b.get("throttle_reason", ""))
        l_throttle = str(l.get("throttle_reason", ""))
        b_planned = str(b.get("planned_action", ""))
        l_planned = str(l.get("planned_action", ""))

        if b_reject == l_reject:
            reject_match += 1
        else:
            reject_mismatches[(b_reject, l_reject)] += 1
        if b_throttle == l_throttle:
            throttle_match += 1
        else:
            throttle_mismatches[(b_throttle, l_throttle)] += 1
        if b_planned == l_planned:
            planned_match += 1
        if _same_target_ticks(b, l):
            same_targets += 1
        if _same_working_ticks(b, l):
            same_working += 1

        target_diff = _abs_tick_diff(b, l)
        if target_diff is not None:
            target_diffs.append(float(target_diff))
        ts_exch_diff = _abs_ts_exch_diff_ns(b, l)
        if ts_exch_diff is not None:
            ts_exch_diffs_ms.append(float(ts_exch_diff / 1_000_000.0))

    bt_rate = _ratio(bt_api, n)
    live_rate = _ratio(live_api, n)
    return {
        "rows": n,
        "api_drop_rate_bt": bt_rate,
        "api_drop_rate_live": live_rate,
        "api_drop_abs_diff": abs(bt_rate - live_rate),
        "reject_reason_match_rate": _ratio(reject_match, n),
        "throttle_reason_match_rate": _ratio(throttle_match, n),
        "planned_action_match_rate": _ratio(planned_match, n),
        "same_target_ticks_rate": _ratio(same_targets, n),
        "same_working_ticks_rate": _ratio(same_working, n),
        "target_tick_abs_diff": _dist(target_diffs),
        "ts_exch_abs_diff_ms": _dist(ts_exch_diffs_ms),
        "reject_reason_mismatch_top": _counter_top(reject_mismatches),
        "throttle_reason_mismatch_top": _counter_top(throttle_mismatches),
    }


def _api_throttle_breakdown(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    *,
    max_replay_lag_ns: int,
) -> dict[str, Any]:
    planned_match = [
        (b, l)
        for b, l in pairs
        if str(b.get("planned_action", "")) == str(l.get("planned_action", ""))
    ]
    planned_mismatch = [
        (b, l)
        for b, l in pairs
        if str(b.get("planned_action", "")) != str(l.get("planned_action", ""))
    ]
    same_guard_inputs = [
        (b, l)
        for b, l in planned_match
        if _same_target_ticks(b, l) and _same_working_ticks(b, l)
    ]
    low_exchange_lag_250ms = [
        (b, l)
        for b, l in pairs
        if _ts_exch_abs_diff_le(b, l, 250_000_000)
    ]
    low_exchange_lag_1000ms = [
        (b, l)
        for b, l in pairs
        if _ts_exch_abs_diff_le(b, l, 1_000_000_000)
    ]
    in_replay_lag_gate = [
        (b, l)
        for b, l in pairs
        if _pair_in_replay_lag_gate(b, l, max_replay_lag_ns)
    ]
    in_exchange_replay_lag_gate = [
        (b, l)
        for b, l in pairs
        if _pair_in_exchange_replay_lag_gate(b, l, max_replay_lag_ns)
    ]
    in_dual_replay_lag_gate = [
        (b, l)
        for b, l in pairs
        if _pair_in_dual_replay_lag_gate(b, l, max_replay_lag_ns)
    ]
    outside_replay_lag_gate = [
        (b, l)
        for b, l in pairs
        if (lag := _pair_replay_lag_ns(b, l)) is not None and lag > max_replay_lag_ns
    ]
    outside_exchange_replay_lag_gate = [
        (b, l)
        for b, l in pairs
        if (lag := _pair_exchange_replay_lag_ns(b, l)) is not None and lag > max_replay_lag_ns
    ]
    outside_dual_replay_lag_gate = [
        (b, l)
        for b, l in pairs
        if not _pair_in_dual_replay_lag_gate(b, l, max_replay_lag_ns)
    ]

    api_throttle_mismatches = [(b, l) for b, l in pairs if _api_throttle_pair_mismatched(b, l)]
    attribution = {
        "mismatch_rows": len(api_throttle_mismatches),
        "planned_action_mismatch_rows": sum(
            str(b.get("planned_action", "")) != str(l.get("planned_action", ""))
            for b, l in api_throttle_mismatches
        ),
        "target_tick_mismatch_rows": sum(
            not _same_target_ticks(b, l) for b, l in api_throttle_mismatches
        ),
        "working_tick_mismatch_rows": sum(
            not _same_working_ticks(b, l) for b, l in api_throttle_mismatches
        ),
        "working_order_lifecycle_mismatch_rows": sum(
            _pair_has_working_order_lifecycle_mismatch(b, l)
            for b, l in api_throttle_mismatches
        ),
        "working_order_semantic_mismatch_rows": sum(
            _pair_has_working_order_semantic_mismatch(b, l)
            for b, l in api_throttle_mismatches
        ),
        "working_order_identity_only_mismatch_rows": sum(
            _pair_has_identity_only_mismatch(b, l)
            for b, l in api_throttle_mismatches
        ),
        "rest_local_divergence_rows": sum(
            _pair_has_rest_local_divergence(b, l)
            for b, l in api_throttle_mismatches
        ),
        "ts_exch_abs_diff_gt_250ms_rows": sum(
            _ts_exch_abs_diff_gt(b, l, 250_000_000)
            for b, l in api_throttle_mismatches
        ),
        "ts_exch_abs_diff_gt_1000ms_rows": sum(
            _ts_exch_abs_diff_gt(b, l, 1_000_000_000)
            for b, l in api_throttle_mismatches
        ),
        "replay_dual_gate_outside_rows": sum(
            not _pair_in_dual_replay_lag_gate(b, l, max_replay_lag_ns)
            for b, l in api_throttle_mismatches
        ),
        "replay_exchange_lag_gt_gate_rows": sum(
            (lag := _pair_exchange_replay_lag_ns(b, l)) is not None and lag > max_replay_lag_ns
            for b, l in api_throttle_mismatches
        ),
        "replay_exchange_lag_missing_rows": sum(
            _pair_exchange_replay_lag_ns(b, l) is None
            for b, l in api_throttle_mismatches
        ),
    }

    return {
        "all": _api_throttle_subset_summary(pairs),
        "planned_action_match": _api_throttle_subset_summary(planned_match),
        "planned_action_mismatch": _api_throttle_subset_summary(planned_mismatch),
        "same_guard_inputs": _api_throttle_subset_summary(same_guard_inputs),
        "ts_exch_abs_diff_le_250ms": _api_throttle_subset_summary(low_exchange_lag_250ms),
        "ts_exch_abs_diff_le_1000ms": _api_throttle_subset_summary(low_exchange_lag_1000ms),
        "replay_lag_abs_diff_le_gate": _api_throttle_subset_summary(in_replay_lag_gate),
        "replay_lag_abs_diff_gt_gate": _api_throttle_subset_summary(outside_replay_lag_gate),
        "replay_exchange_lag_abs_diff_le_gate": _api_throttle_subset_summary(in_exchange_replay_lag_gate),
        "replay_exchange_lag_abs_diff_gt_gate": _api_throttle_subset_summary(outside_exchange_replay_lag_gate),
        "replay_dual_lag_abs_diff_le_gate": _api_throttle_subset_summary(in_dual_replay_lag_gate),
        "replay_dual_lag_abs_diff_gt_gate": _api_throttle_subset_summary(outside_dual_replay_lag_gate),
        "mismatch_attribution": attribution,
    }


def _alignment_from_pairs(
    pairs: list[tuple[dict[str, str], dict[str, str]]],
    *,
    max_replay_lag_ns: int = 250_000_000,
    bt_all_rows: list[dict[str, str]] | None = None,
    live_all_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not pairs:
        return {
            "common_rows": 0,
            "matched_rows": 0,
            "action_match_rate": 0.0,
            "reject_reason_match_rate": 0.0,
            "planned_action_match_rate": 0.0,
            "throttle_reason_match_rate": 0.0,
            "mae": {},
            "action_mismatch_top": [],
            "reject_reason_mismatch_top": [],
            "planned_action_mismatch_top": [],
            "throttle_reason_mismatch_top": [],
            "replay_lag": _replay_lag_summary([], max_replay_lag_ns),
            "working_order_lifecycle": _working_order_lifecycle_breakdown([]),
            "api_throttle": _api_throttle_breakdown([], max_replay_lag_ns=max_replay_lag_ns),
        }

    metrics = ["fair", "reservation", "half_spread", "position", "inventory_score", "spread_bps", "vol_bps"]
    bt_series: dict[str, list[float]] = {k: [] for k in metrics}
    live_series: dict[str, list[float]] = {k: [] for k in metrics}

    action_match = 0
    reject_match = 0
    planned_match = 0
    throttle_match = 0
    action_mismatches: Counter[tuple[str, str]] = Counter()
    reject_mismatches: Counter[tuple[str, str]] = Counter()
    planned_mismatches: Counter[tuple[str, str]] = Counter()
    throttle_mismatches: Counter[tuple[str, str]] = Counter()

    for b, l in pairs:
        b_action = str(b.get("action", ""))
        l_action = str(l.get("action", ""))
        b_reject = str(b.get("reject_reason", ""))
        l_reject = str(l.get("reject_reason", ""))
        b_planned = str(b.get("planned_action", ""))
        l_planned = str(l.get("planned_action", ""))
        b_throttle = str(b.get("throttle_reason", ""))
        l_throttle = str(l.get("throttle_reason", ""))

        if b_action == l_action:
            action_match += 1
        else:
            action_mismatches[(b_action, l_action)] += 1
        if b_reject == l_reject:
            reject_match += 1
        else:
            reject_mismatches[(b_reject, l_reject)] += 1
        if b_planned == l_planned:
            planned_match += 1
        else:
            planned_mismatches[(b_planned, l_planned)] += 1
        if b_throttle == l_throttle:
            throttle_match += 1
        else:
            throttle_mismatches[(b_throttle, l_throttle)] += 1

        for k in metrics:
            bt_series[k].append(_safe_float(b.get(k, 0.0)))
            live_series[k].append(_safe_float(l.get(k, 0.0)))

    mae = {k: _mae(bt_series[k], live_series[k]) for k in metrics}
    n = len(pairs)
    lifecycle_breakdown = _working_order_lifecycle_breakdown(pairs)
    first_divergence = lifecycle_breakdown.get("first_semantic_divergence", {})
    if bt_all_rows is not None and live_all_rows is not None:
        lifecycle_breakdown["first_semantic_divergence_context"] = (
            _first_semantic_divergence_context(
                first_divergence,
                bt_all_rows,
                live_all_rows,
            )
        )

    return {
        "common_rows": n,
        "matched_rows": n,
        "action_match_rate": float(action_match / n),
        "reject_reason_match_rate": float(reject_match / n),
        "planned_action_match_rate": float(planned_match / n),
        "throttle_reason_match_rate": float(throttle_match / n),
        "mae": mae,
        "action_mismatch_top": _counter_top(action_mismatches),
        "reject_reason_mismatch_top": _counter_top(reject_mismatches),
        "planned_action_mismatch_top": _counter_top(planned_mismatches),
        "throttle_reason_mismatch_top": _counter_top(throttle_mismatches),
        "replay_lag": _replay_lag_summary(pairs, max_replay_lag_ns),
        "working_order_lifecycle": lifecycle_breakdown,
        "api_throttle": _api_throttle_breakdown(pairs, max_replay_lag_ns=max_replay_lag_ns),
    }


def _alignment_by_seq_rows(
    bt_rows: list[dict[str, str]],
    live_rows: list[dict[str, str]],
    *,
    max_replay_lag_ns: int,
    bt_all_rows: list[dict[str, str]] | None = None,
    live_all_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    bt_by_seq = _align_by_seq(bt_rows)
    live_by_seq = _align_by_seq(live_rows)
    common = sorted(set(bt_by_seq.keys()) & set(live_by_seq.keys()))
    return _alignment_from_pairs(
        [(bt_by_seq[seq], live_by_seq[seq]) for seq in common],
        max_replay_lag_ns=max_replay_lag_ns,
        bt_all_rows=bt_all_rows,
        live_all_rows=live_all_rows,
    )


def _alignment_by_nearest_ts(
    bt_rows: list[dict[str, str]],
    live_rows: list[dict[str, str]],
    max_lag_ns: int,
    max_replay_lag_ns: int,
    bt_all_rows: list[dict[str, str]] | None = None,
    live_all_rows: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    import bisect

    bt_ts_rows = sorted((_parse_int_timestamp(r.get("ts_local", "") or "0"), r) for r in bt_rows)
    bt_ts = [ts for ts, _ in bt_ts_rows if ts > 0]
    bt_by_ts = [r for ts, r in bt_ts_rows if ts > 0]
    pairs: list[tuple[dict[str, str], dict[str, str]]] = []
    lags: list[float] = []
    unmatched = 0

    for live in live_rows:
        try:
            live_ts = _parse_int_timestamp(live.get("ts_local", "") or "0")
        except ValueError:
            unmatched += 1
            continue
        if live_ts <= 0 or not bt_ts:
            unmatched += 1
            continue
        i = bisect.bisect_left(bt_ts, live_ts)
        candidates: list[tuple[int, dict[str, str]]] = []
        if i < len(bt_ts):
            candidates.append((bt_ts[i], bt_by_ts[i]))
        if i > 0:
            candidates.append((bt_ts[i - 1], bt_by_ts[i - 1]))
        nearest_ts, nearest = min(candidates, key=lambda x: abs(x[0] - live_ts))
        lag = nearest_ts - live_ts
        if max_lag_ns > 0 and abs(lag) > max_lag_ns:
            unmatched += 1
            continue
        pairs.append((nearest, live))
        lags.append(float(lag))

    result = _alignment_from_pairs(
        pairs,
        max_replay_lag_ns=max_replay_lag_ns,
        bt_all_rows=bt_all_rows,
        live_all_rows=live_all_rows,
    )
    result["nearest_ts_matched_rows"] = len(pairs)
    result["nearest_ts_unmatched_rows"] = unmatched
    result["nearest_ts_lag_ns"] = {
        "abs_ns": _dist([abs(v) for v in lags]),
        "signed_ns": _dist(lags),
    }
    return result


def compare(
    bt_csv: Path,
    live_csv: Path,
    *,
    align_mode: str = "both",
    max_lag_ms: float = 250.0,
    max_replay_lag_ms: float = 250.0,
) -> dict[str, Any]:
    bt_rows = _read_csv_rows(bt_csv)
    live_rows = _read_csv_rows(live_csv)
    bt_decision_rows = _decision_rows(bt_rows)
    live_decision_rows = _decision_rows(live_rows)
    if align_mode not in {"seq", "nearest_ts", "both"}:
        raise ValueError(f"unsupported align_mode: {align_mode}")

    max_replay_lag_ns = int(max_replay_lag_ms * 1_000_000)
    seq_alignment = _alignment_by_seq_rows(
        bt_decision_rows,
        live_decision_rows,
        max_replay_lag_ns=max_replay_lag_ns,
        bt_all_rows=bt_rows,
        live_all_rows=live_rows,
    )
    nearest_alignment = _alignment_by_nearest_ts(
        bt_decision_rows,
        live_decision_rows,
        int(max_lag_ms * 1_000_000),
        max_replay_lag_ns,
        bt_all_rows=bt_rows,
        live_all_rows=live_rows,
    )
    primary = nearest_alignment if align_mode == "nearest_ts" else seq_alignment

    return {
        "bt_file": str(bt_csv),
        "live_file": str(live_csv),
        "bt_summary": _summary(bt_decision_rows),
        "live_summary": _summary(live_decision_rows),
        "bt_event_rows": len(bt_rows),
        "live_event_rows": len(live_rows),
        "alignment": primary,
        "alignment_seq": seq_alignment,
        "alignment_nearest_ts": nearest_alignment,
        "cadence": {
            "bt": _cadence_stats(bt_decision_rows),
            "live": _cadence_stats(live_decision_rows),
            "nearest_lag_live_to_bt": _nearest_lag_stats(bt_decision_rows, live_decision_rows),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare backtest and live audit csv")
    parser.add_argument("--bt", required=True, help="audit_bt.csv path")
    parser.add_argument("--live", required=True, help="audit_live.csv path")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    parser.add_argument("--align-mode", choices=["seq", "nearest_ts", "both"], default="both")
    parser.add_argument("--max-lag-ms", type=float, default=250.0)
    parser.add_argument("--max-replay-lag-ms", type=float, default=250.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bt_csv = _expand(args.bt)
    live_csv = _expand(args.live)
    report = compare(
        bt_csv,
        live_csv,
        align_mode=args.align_mode,
        max_lag_ms=args.max_lag_ms,
        max_replay_lag_ms=args.max_replay_lag_ms,
    )

    print(json.dumps(report, indent=2, ensure_ascii=True))

    if args.out:
        out = _expand(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
