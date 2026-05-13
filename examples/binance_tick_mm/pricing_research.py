#!/usr/bin/env python3
"""Read-only Stage 4 pricing signal research over accepted Binance MM samples."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TASK_ID = "0514T003"
DEFAULT_HORIZONS_MS = (100, 500, 1_000, 5_000)
DEFAULT_BUCKETS = 5
DEFAULT_MIN_ROWS = 100
DEFAULT_MIN_ABS_SPEARMAN = 0.10
DEFAULT_MIN_ABS_SPREAD_TICKS = 1.0
DEFAULT_MAX_FUTURE_GAP_MS = 250.0


@dataclass
class Top5Snapshot:
    raw_seq: int
    local_ts: int
    exch_ts: int
    bid_px: tuple[float, ...]
    bid_ticks: tuple[int, ...]
    bid_qtys: tuple[float, ...]
    ask_px: tuple[float, ...]
    ask_ticks: tuple[int, ...]
    ask_qtys: tuple[float, ...]
    bookticker_bid_px: float = math.nan
    bookticker_ask_px: float = math.nan
    bookticker_depth_age_ms: float = math.nan
    startup_excluded: bool = False
    sync_waiting_snapshot: bool = False
    sync_gap: bool = False

    @property
    def usable(self) -> bool:
        return (
            len(self.bid_px) > 0
            and len(self.ask_px) > 0
            and not self.startup_excluded
            and not self.sync_waiting_snapshot
            and not self.sync_gap
        )


@dataclass
class DecisionFeatureRow:
    strategy_seq: int
    decision_ts_local: int
    audit_ts_local: int
    side_sign: int
    join_used_future: bool
    join_missing: bool
    join_stale: bool
    join_gap_crossed: bool
    startup_excluded: bool
    joined_raw_seq: int | None
    top5_join_age_ms: float
    depth_join_age_ms: float
    bookticker_join_age_ms: float
    max_join_age_ms: float
    signals: dict[str, float] = field(default_factory=dict)
    markouts: dict[int, dict[str, float]] = field(default_factory=dict)

    @property
    def accepted_with_stale(self) -> bool:
        return (
            not self.join_used_future
            and not self.join_missing
            and not self.join_gap_crossed
            and not self.startup_excluded
        )

    @property
    def primary_non_stale(self) -> bool:
        return self.accepted_with_stale and not self.join_stale


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


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def _parse_pipe_floats(value: str) -> tuple[float, ...]:
    if not value:
        return ()
    out: list[float] = []
    for item in str(value).split("|"):
        item = item.strip()
        if item == "":
            continue
        number = _float(item)
        if _finite(number):
            out.append(number)
    return tuple(out)


def _parse_pipe_ints(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    out: list[int] = []
    for item in str(value).split("|"):
        number = _int(item)
        if number is not None:
            out.append(number)
    return tuple(out)


def _safe_div(num: float, denom: float) -> float:
    if not _finite(num) or not _finite(denom) or denom == 0.0:
        return math.nan
    return num / denom


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(float(value))]
    if not finite:
        return math.nan
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _quantile(values: list[float], q: float) -> float:
    finite = np.asarray([value for value in values if _finite(value)], dtype=np.float64)
    if len(finite) == 0:
        return math.nan
    return float(np.quantile(finite, q))


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


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_top5_snapshots(top5_sidecar_csv: Path) -> list[Top5Snapshot]:
    snapshots: list[Top5Snapshot] = []
    with top5_sidecar_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            bid_px = _parse_pipe_floats(row.get("bid_top5_px", ""))
            ask_px = _parse_pipe_floats(row.get("ask_top5_px", ""))
            bid_qtys = _parse_pipe_floats(row.get("bid_top5_qtys", ""))
            ask_qtys = _parse_pipe_floats(row.get("ask_top5_qtys", ""))
            if not bid_px or not ask_px or not bid_qtys or not ask_qtys:
                continue
            raw_seq = _int(row.get("raw_seq"), 0)
            local_ts = _int(row.get("local_ts"), 0)
            exch_ts = _int(row.get("exch_ts"), 0)
            if raw_seq is None or local_ts is None or exch_ts is None:
                continue
            snapshots.append(
                Top5Snapshot(
                    raw_seq=raw_seq,
                    local_ts=local_ts,
                    exch_ts=exch_ts,
                    bid_px=bid_px,
                    bid_ticks=_parse_pipe_ints(row.get("bid_top5_ticks", "")),
                    bid_qtys=bid_qtys,
                    ask_px=ask_px,
                    ask_ticks=_parse_pipe_ints(row.get("ask_top5_ticks", "")),
                    ask_qtys=ask_qtys,
                    bookticker_bid_px=_float(row.get("bookticker_bid_px")),
                    bookticker_ask_px=_float(row.get("bookticker_ask_px")),
                    bookticker_depth_age_ms=_float(row.get("bookticker_depth_age_ms")),
                    startup_excluded=_parse_bool(row.get("startup_excluded")),
                    sync_waiting_snapshot=_parse_bool(row.get("sync_waiting_snapshot")),
                    sync_gap=_parse_bool(row.get("sync_gap")),
                )
            )
    snapshots.sort(key=lambda item: (item.local_ts, item.raw_seq))
    return snapshots


def _read_joined_decisions(joined_decisions_csv: Path) -> dict[int, dict[str, str]]:
    joined: dict[int, dict[str, str]] = {}
    with joined_decisions_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seq = _int(row.get("strategy_seq"))
            if seq is not None:
                joined[seq] = row
    return joined


def _read_audit_decisions(audit_csv: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with audit_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("event_type") == "decision":
                rows.append(row)
    return rows


def _side_sign(row: dict[str, str]) -> int:
    text = "|".join(
        str(row.get(name, "")).lower()
        for name in ("action", "planned_action", "order_side", "linked_action")
    )
    has_buy = "buy" in text
    has_sell = "sell" in text
    if has_buy and not has_sell:
        return 1
    if has_sell and not has_buy:
        return -1
    return 0


def _top_mid(snapshot: Top5Snapshot) -> float:
    return (snapshot.bid_px[0] + snapshot.ask_px[0]) / 2.0


def _bookticker_mid(snapshot: Top5Snapshot) -> float:
    if _finite(snapshot.bookticker_bid_px) and _finite(snapshot.bookticker_ask_px):
        return (snapshot.bookticker_bid_px + snapshot.bookticker_ask_px) / 2.0
    return math.nan


def _microprice_top1(snapshot: Top5Snapshot) -> float:
    bid_qty = snapshot.bid_qtys[0]
    ask_qty = snapshot.ask_qtys[0]
    return _safe_div(snapshot.ask_px[0] * bid_qty + snapshot.bid_px[0] * ask_qty, bid_qty + ask_qty)


def _microprice_top5(snapshot: Top5Snapshot) -> float:
    levels = min(len(snapshot.bid_px), len(snapshot.ask_px), len(snapshot.bid_qtys), len(snapshot.ask_qtys), 5)
    if levels <= 0:
        return math.nan
    bid_qty = np.asarray(snapshot.bid_qtys[:levels], dtype=np.float64)
    ask_qty = np.asarray(snapshot.ask_qtys[:levels], dtype=np.float64)
    bid_px = np.asarray(snapshot.bid_px[:levels], dtype=np.float64)
    ask_px = np.asarray(snapshot.ask_px[:levels], dtype=np.float64)
    denom = float(np.sum(bid_qty) + np.sum(ask_qty))
    if denom == 0.0:
        return math.nan
    return float((np.sum(ask_px * bid_qty) + np.sum(bid_px * ask_qty)) / denom)


def _imbalance(bid_qty: Iterable[float], ask_qty: Iterable[float]) -> float:
    bid_total = float(np.sum(np.asarray(list(bid_qty), dtype=np.float64)))
    ask_total = float(np.sum(np.asarray(list(ask_qty), dtype=np.float64)))
    return _safe_div(bid_total - ask_total, bid_total + ask_total)


def _spread_ticks(snapshot: Top5Snapshot, tick_size: float) -> float:
    if snapshot.bid_ticks and snapshot.ask_ticks:
        return float(snapshot.ask_ticks[0] - snapshot.bid_ticks[0])
    return _safe_div(snapshot.ask_px[0] - snapshot.bid_px[0], tick_size)


def _snapshot_signals(
    *,
    audit_row: dict[str, str],
    snapshot: Top5Snapshot,
    previous: Top5Snapshot | None,
    joined_row: dict[str, str] | None,
    tick_size: float,
) -> dict[str, float]:
    top_mid = _top_mid(snapshot)
    book_mid = _bookticker_mid(snapshot)
    top1_micro = _microprice_top1(snapshot)
    top5_micro = _microprice_top5(snapshot)
    bid_top5_qty = float(np.sum(np.asarray(snapshot.bid_qtys[:5], dtype=np.float64)))
    ask_top5_qty = float(np.sum(np.asarray(snapshot.ask_qtys[:5], dtype=np.float64)))
    top5_qty_total = bid_top5_qty + ask_top5_qty
    top1_qty_total = snapshot.bid_qtys[0] + snapshot.ask_qtys[0]

    prev_top1_ofi = math.nan
    prev_top5_ofi = math.nan
    prev_mid_move_ticks = math.nan
    if previous is not None:
        prev_top1_ofi = (snapshot.bid_qtys[0] - previous.bid_qtys[0]) - (
            snapshot.ask_qtys[0] - previous.ask_qtys[0]
        )
        prev_top5_ofi = (
            float(np.sum(np.asarray(snapshot.bid_qtys[:5], dtype=np.float64)))
            - float(np.sum(np.asarray(previous.bid_qtys[:5], dtype=np.float64)))
        ) - (
            float(np.sum(np.asarray(snapshot.ask_qtys[:5], dtype=np.float64)))
            - float(np.sum(np.asarray(previous.ask_qtys[:5], dtype=np.float64)))
        )
        prev_mid_move_ticks = _safe_div(top_mid - _top_mid(previous), tick_size)

    audit_mid = _float(audit_row.get("mid"))
    fair = _float(audit_row.get("fair"))
    reservation = _float(audit_row.get("reservation"))
    best_bid = _float(audit_row.get("best_bid"))
    best_ask = _float(audit_row.get("best_ask"))
    audit_bbo_mid = (best_bid + best_ask) / 2.0 if _finite(best_bid) and _finite(best_ask) else math.nan
    joined = joined_row or {}

    signals = {
        "audit_mid_edge_ticks": _safe_div(audit_mid - top_mid, tick_size),
        "audit_bbo_mid_edge_ticks": _safe_div(audit_bbo_mid - top_mid, tick_size),
        "bookticker_mid_edge_ticks": _safe_div(book_mid - top_mid, tick_size),
        "top1_microprice_edge_ticks": _safe_div(top1_micro - top_mid, tick_size),
        "top5_microprice_edge_ticks": _safe_div(top5_micro - top_mid, tick_size),
        "fair_edge_ticks": _safe_div(fair - top_mid, tick_size),
        "reservation_edge_ticks": _safe_div(reservation - top_mid, tick_size),
        "top1_imbalance": _imbalance(snapshot.bid_qtys[:1], snapshot.ask_qtys[:1]),
        "top5_imbalance": _imbalance(snapshot.bid_qtys[:5], snapshot.ask_qtys[:5]),
        "top1_ofi_proxy": prev_top1_ofi,
        "top5_ofi_proxy": prev_top5_ofi,
        "recent_mid_move_ticks": prev_mid_move_ticks,
        "spread_ticks": _spread_ticks(snapshot, tick_size),
        "top5_bid_qty": bid_top5_qty,
        "top5_ask_qty": ask_top5_qty,
        "top5_depth_imbalance_qty": bid_top5_qty - ask_top5_qty,
        "top5_liquidity_concentration": _safe_div(top1_qty_total, top5_qty_total),
        "bookticker_depth_age_ms": snapshot.bookticker_depth_age_ms,
        "top5_join_age_ms": _float(joined.get("top5_join_age_ms")),
        "depth_join_age_ms": _float(joined.get("depth_join_age_ms")),
        "bookticker_join_age_ms": _float(joined.get("bookticker_join_age_ms")),
        "max_join_age_ms": _float(joined.get("max_join_age_ms")),
        "audit_feed_latency_ms": _safe_div(_float(audit_row.get("feed_latency_ns")), 1_000_000.0),
        "latency_signal_ms": _float(audit_row.get("latency_signal_ms")),
        "book_view_stale_ms": _float(audit_row.get("book_view_stale_ms")),
        "spread_bps": _float(audit_row.get("spread_bps")),
        "vol_bps": _float(audit_row.get("vol_bps")),
        "inventory_score": _float(audit_row.get("inventory_score")),
    }
    return signals


def _asof_index(local_ts: list[int], ts: int) -> int | None:
    idx = bisect.bisect_right(local_ts, ts) - 1
    return idx if idx >= 0 else None


def _future_index(local_ts: list[int], ts: int) -> int | None:
    idx = bisect.bisect_left(local_ts, ts)
    return idx if idx < len(local_ts) else None


def build_decision_features(
    *,
    audit_csv: Path,
    joined_decisions_csv: Path,
    top5_sidecar_csv: Path,
    tick_size: float = 0.1,
    horizons_ms: Iterable[int] = DEFAULT_HORIZONS_MS,
    max_future_gap_ms: float = DEFAULT_MAX_FUTURE_GAP_MS,
) -> tuple[list[DecisionFeatureRow], dict[str, Any]]:
    audit_rows = _read_audit_decisions(audit_csv)
    joined_rows = _read_joined_decisions(joined_decisions_csv)
    snapshots = load_top5_snapshots(top5_sidecar_csv)
    usable_snapshots = [item for item in snapshots if item.usable]
    usable_ts = [item.local_ts for item in usable_snapshots]

    feature_rows: list[DecisionFeatureRow] = []
    excluded_missing_asof = 0
    max_future_gap_ns = int(max_future_gap_ms * 1_000_000)

    for audit_row in audit_rows:
        seq = _int(audit_row.get("strategy_seq"))
        audit_ts = _int(audit_row.get("ts_local"))
        if seq is None or audit_ts is None:
            continue
        joined = joined_rows.get(seq, {})
        joined_ts = _int(joined.get("decision_ts_local"), audit_ts) or audit_ts
        idx = _asof_index(usable_ts, joined_ts)
        if idx is None:
            excluded_missing_asof += 1
            continue
        snapshot = usable_snapshots[idx]
        previous = usable_snapshots[idx - 1] if idx > 0 else None
        row = DecisionFeatureRow(
            strategy_seq=seq,
            decision_ts_local=joined_ts,
            audit_ts_local=audit_ts,
            side_sign=_side_sign(audit_row),
            join_used_future=_parse_bool(joined.get("join_used_future")),
            join_missing=_parse_bool(joined.get("join_missing")) or not bool(joined),
            join_stale=_parse_bool(joined.get("join_stale")),
            join_gap_crossed=_parse_bool(joined.get("join_gap_crossed")),
            startup_excluded=snapshot.startup_excluded or snapshot.sync_waiting_snapshot or snapshot.sync_gap,
            joined_raw_seq=_int(joined.get("joined_raw_seq")),
            top5_join_age_ms=_float(joined.get("top5_join_age_ms")),
            depth_join_age_ms=_float(joined.get("depth_join_age_ms")),
            bookticker_join_age_ms=_float(joined.get("bookticker_join_age_ms")),
            max_join_age_ms=_float(joined.get("max_join_age_ms")),
            signals=_snapshot_signals(
                audit_row=audit_row,
                snapshot=snapshot,
                previous=previous,
                joined_row=joined,
                tick_size=tick_size,
            ),
        )
        current_mid = _top_mid(snapshot)
        for horizon_ms in horizons_ms:
            target_ts = joined_ts + int(horizon_ms) * 1_000_000
            future_idx = _future_index(usable_ts, target_ts)
            future_mid = math.nan
            future_gap_ms = math.nan
            raw_markout = math.nan
            raw_markout_ticks = math.nan
            side_adjusted_ticks = math.nan
            if future_idx is not None:
                future = usable_snapshots[future_idx]
                future_gap_ns = future.local_ts - target_ts
                if future_gap_ns <= max_future_gap_ns:
                    future_mid = _top_mid(future)
                    future_gap_ms = future_gap_ns / 1_000_000.0
                    raw_markout = future_mid - current_mid
                    raw_markout_ticks = _safe_div(raw_markout, tick_size)
                    if row.side_sign != 0:
                        side_adjusted_ticks = raw_markout_ticks * row.side_sign
            row.markouts[int(horizon_ms)] = {
                "future_mid": future_mid,
                "future_gap_ms": future_gap_ms,
                "raw_mid_markout": raw_markout,
                "raw_mid_markout_ticks": raw_markout_ticks,
                "side_adjusted_markout_ticks": side_adjusted_ticks,
            }
        feature_rows.append(row)

    row_counts = {
        "audit_decision_rows": len(audit_rows),
        "joined_decision_rows": len(joined_rows),
        "top5_snapshot_rows": len(snapshots),
        "usable_top5_snapshot_rows": len(usable_snapshots),
        "feature_rows": len(feature_rows),
        "excluded_missing_asof_rows": excluded_missing_asof,
        "accepted_with_stale_rows": sum(1 for row in feature_rows if row.accepted_with_stale),
        "primary_non_stale_rows": sum(1 for row in feature_rows if row.primary_non_stale),
        "join_used_future_rows": sum(1 for row in feature_rows if row.join_used_future),
        "join_missing_rows": sum(1 for row in feature_rows if row.join_missing),
        "join_stale_rows": sum(1 for row in feature_rows if row.join_stale),
        "join_gap_crossed_rows": sum(1 for row in feature_rows if row.join_gap_crossed),
        "startup_excluded_rows": sum(1 for row in feature_rows if row.startup_excluded),
    }
    for horizon_ms in horizons_ms:
        row_counts[f"markout_rows_{int(horizon_ms)}ms"] = sum(
            1 for row in feature_rows if _finite(row.markouts[int(horizon_ms)]["raw_mid_markout_ticks"])
        )
    return feature_rows, row_counts


def _universe_filter(row: DecisionFeatureRow, universe: str) -> bool:
    if universe == "accepted_with_stale":
        return row.accepted_with_stale
    if universe == "primary_non_stale":
        return row.primary_non_stale
    raise ValueError(f"Unknown universe: {universe}")


def _bucket_table(
    x: np.ndarray,
    y: np.ndarray,
    side_y: np.ndarray,
    *,
    buckets: int,
) -> list[dict[str, Any]]:
    if len(x) == 0:
        return []
    quantiles = np.quantile(x, np.linspace(0.0, 1.0, buckets + 1))
    rows: list[dict[str, Any]] = []
    for bucket in range(buckets):
        low = quantiles[bucket]
        high = quantiles[bucket + 1]
        if bucket == buckets - 1:
            mask = (x >= low) & (x <= high)
        else:
            mask = (x >= low) & (x < high)
        if not np.any(mask) and bucket == 0:
            mask = x == low
        signal_values = x[mask]
        markout_values = y[mask]
        side_values = side_y[mask]
        finite_side = side_values[np.isfinite(side_values)]
        rows.append(
            {
                "bucket": bucket + 1,
                "rows": int(len(signal_values)),
                "signal_min": float(np.min(signal_values)) if len(signal_values) else math.nan,
                "signal_max": float(np.max(signal_values)) if len(signal_values) else math.nan,
                "signal_mean": float(np.mean(signal_values)) if len(signal_values) else math.nan,
                "raw_markout_mean_ticks": float(np.mean(markout_values)) if len(markout_values) else math.nan,
                "raw_markout_p50_ticks": float(np.quantile(markout_values, 0.5)) if len(markout_values) else math.nan,
                "side_adjusted_rows": int(len(finite_side)),
                "side_adjusted_mean_ticks": float(np.mean(finite_side)) if len(finite_side) else math.nan,
            }
        )
    return rows


def _monotonic_score(bucket_rows: list[dict[str, Any]]) -> tuple[float, float]:
    means = [float(row["raw_markout_mean_ticks"]) for row in bucket_rows if int(row["rows"]) > 0]
    means = [value for value in means if _finite(value)]
    if len(means) < 2:
        return math.nan, math.nan
    diffs = np.diff(np.asarray(means, dtype=np.float64))
    nonzero = diffs[np.abs(diffs) > 1e-12]
    if len(nonzero) == 0:
        return 1.0, 0.0
    positive = float(np.mean(nonzero > 0))
    negative = float(np.mean(nonzero < 0))
    return max(positive, negative), means[-1] - means[0]


def _signal_arrays(
    rows: list[DecisionFeatureRow],
    *,
    signal: str,
    horizon_ms: int,
    universe: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[DecisionFeatureRow]]:
    selected: list[DecisionFeatureRow] = []
    x_values: list[float] = []
    y_values: list[float] = []
    side_y_values: list[float] = []
    for row in rows:
        if not _universe_filter(row, universe):
            continue
        signal_value = row.signals.get(signal, math.nan)
        markout = row.markouts[horizon_ms]["raw_mid_markout_ticks"]
        if not (_finite(signal_value) and _finite(markout)):
            continue
        selected.append(row)
        x_values.append(float(signal_value))
        y_values.append(float(markout))
        side_y_values.append(float(row.markouts[horizon_ms]["side_adjusted_markout_ticks"]))
    return (
        np.asarray(x_values, dtype=np.float64),
        np.asarray(y_values, dtype=np.float64),
        np.asarray(side_y_values, dtype=np.float64),
        selected,
    )


def evaluate_signals(
    rows: list[DecisionFeatureRow],
    *,
    horizons_ms: Iterable[int] = DEFAULT_HORIZONS_MS,
    buckets: int = DEFAULT_BUCKETS,
    min_rows: int = DEFAULT_MIN_ROWS,
    min_abs_spearman: float = DEFAULT_MIN_ABS_SPEARMAN,
    min_abs_spread_ticks: float = DEFAULT_MIN_ABS_SPREAD_TICKS,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    signal_names = sorted({name for row in rows for name in row.signals})
    metric_rows: list[dict[str, Any]] = []
    bucket_tables: dict[str, list[dict[str, Any]]] = {name: [] for name in signal_names}

    for signal in signal_names:
        for universe in ("accepted_with_stale", "primary_non_stale"):
            for horizon_ms in horizons_ms:
                x, y, side_y, selected = _signal_arrays(
                    rows,
                    signal=signal,
                    horizon_ms=int(horizon_ms),
                    universe=universe,
                )
                bucket_rows = _bucket_table(x, y, side_y, buckets=buckets)
                monotonic_score, top_bottom_spread = _monotonic_score(bucket_rows)
                pearson = _corr(x, y)
                spearman = _spearman(x, y)
                split = len(x) // 2
                first_corr = _corr(x[:split], y[:split]) if split >= 2 else math.nan
                second_corr = _corr(x[split:], y[split:]) if len(x) - split >= 2 else math.nan
                first_spread = math.nan
                second_spread = math.nan
                if split >= max(2, buckets):
                    _, first_spread = _monotonic_score(_bucket_table(x[:split], y[:split], side_y[:split], buckets=buckets))
                if len(x) - split >= max(2, buckets):
                    _, second_spread = _monotonic_score(_bucket_table(x[split:], y[split:], side_y[split:], buckets=buckets))
                stable_sign = (
                    _finite(first_spread)
                    and _finite(second_spread)
                    and first_spread != 0.0
                    and second_spread != 0.0
                    and math.copysign(1.0, first_spread) == math.copysign(1.0, second_spread)
                )
                status = "rejected"
                reason = []
                if len(x) < min_rows:
                    reason.append("too_few_rows")
                if not _finite(spearman) or abs(spearman) < min_abs_spearman:
                    reason.append("weak_rank_correlation")
                if not _finite(top_bottom_spread) or abs(top_bottom_spread) < min_abs_spread_ticks:
                    reason.append("weak_top_bottom_spread")
                if _finite(first_spread) and _finite(second_spread) and not stable_sign:
                    reason.append("unstable_split_sign")
                if not reason:
                    status = "candidate_for_followup"
                metric = {
                    "signal": signal,
                    "universe": universe,
                    "horizon_ms": int(horizon_ms),
                    "rows": int(len(x)),
                    "signal_mean": float(np.mean(x)) if len(x) else math.nan,
                    "signal_std": float(np.std(x)) if len(x) else math.nan,
                    "raw_markout_mean_ticks": float(np.mean(y)) if len(y) else math.nan,
                    "raw_markout_p50_ticks": float(np.quantile(y, 0.5)) if len(y) else math.nan,
                    "pearson": pearson,
                    "spearman": spearman,
                    "first_half_pearson": first_corr,
                    "second_half_pearson": second_corr,
                    "first_half_top_bottom_spread_ticks": first_spread,
                    "second_half_top_bottom_spread_ticks": second_spread,
                    "split_sign_stable": stable_sign,
                    "bucket_monotonic_score": monotonic_score,
                    "top_bottom_spread_ticks": top_bottom_spread,
                    "status": status,
                    "reason": ";".join(reason) if reason else "",
                    "first_strategy_seq": selected[0].strategy_seq if selected else "",
                    "last_strategy_seq": selected[-1].strategy_seq if selected else "",
                }
                metric_rows.append(metric)
                for bucket_row in bucket_rows:
                    bucket_tables[signal].append(
                        {
                            "signal": signal,
                            "universe": universe,
                            "horizon_ms": int(horizon_ms),
                            **bucket_row,
                        }
                    )

    rejected = summarize_signal_status(metric_rows)
    return metric_rows, bucket_tables, rejected


def summarize_signal_status(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in metric_rows:
        if row["universe"] != "primary_non_stale":
            continue
        grouped.setdefault(str(row["signal"]), []).append(row)

    summary: list[dict[str, Any]] = []
    for signal, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda item: (
                abs(float(item["spearman"])) if _finite(float(item["spearman"])) else -1.0,
                abs(float(item["top_bottom_spread_ticks"]))
                if _finite(float(item["top_bottom_spread_ticks"]))
                else -1.0,
            ),
            reverse=True,
        )
        best = ranked[0] if ranked else {}
        candidate_rows = [row for row in rows if row.get("status") == "candidate_for_followup"]
        status = "candidate_for_followup" if candidate_rows else "rejected"
        reasons = sorted({reason for row in rows for reason in str(row.get("reason", "")).split(";") if reason})
        summary.append(
            {
                "signal": signal,
                "status": status,
                "best_horizon_ms": best.get("horizon_ms", ""),
                "best_abs_spearman": abs(float(best["spearman"])) if best and _finite(float(best["spearman"])) else math.nan,
                "best_spearman": float(best["spearman"]) if best and _finite(float(best["spearman"])) else math.nan,
                "best_top_bottom_spread_ticks": float(best["top_bottom_spread_ticks"])
                if best and _finite(float(best["top_bottom_spread_ticks"]))
                else math.nan,
                "best_bucket_monotonic_score": float(best["bucket_monotonic_score"])
                if best and _finite(float(best["bucket_monotonic_score"]))
                else math.nan,
                "best_rows": int(best["rows"]) if best else 0,
                "reason": "" if status == "candidate_for_followup" else ";".join(reasons),
            }
        )
    seen_candidate_signatures: dict[tuple[Any, ...], str] = {}
    for row in summary:
        if row["status"] != "candidate_for_followup":
            continue
        signature = (
            row["best_horizon_ms"],
            round(float(row["best_spearman"]), 6) if _finite(float(row["best_spearman"])) else None,
            round(float(row["best_top_bottom_spread_ticks"]), 6)
            if _finite(float(row["best_top_bottom_spread_ticks"]))
            else None,
            row["best_rows"],
        )
        duplicate_of = seen_candidate_signatures.get(signature)
        if duplicate_of is not None:
            row["status"] = "rejected"
            row["reason"] = f"duplicates:{duplicate_of}"
            continue
        seen_candidate_signatures[signature] = str(row["signal"])
    summary.sort(
        key=lambda item: (
            item["status"] != "candidate_for_followup",
            -(float(item["best_abs_spearman"]) if _finite(float(item["best_abs_spearman"])) else -1.0),
        )
    )
    return summary


def _csv_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        fieldnames = names
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row.get(name, "")) for name in fieldnames})


def _json_clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_clean(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_clean(item) for item in value]
    if isinstance(value, tuple):
        return [_json_clean(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_clean(payload), indent=2, sort_keys=True, ensure_ascii=True), encoding="utf-8")


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def markout_rows_for_csv(rows: list[DecisionFeatureRow], horizons_ms: Iterable[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        for horizon_ms in horizons_ms:
            markout = row.markouts[int(horizon_ms)]
            out.append(
                {
                    "strategy_seq": row.strategy_seq,
                    "decision_ts_local": row.decision_ts_local,
                    "horizon_ms": int(horizon_ms),
                    "universe_accepted_with_stale": row.accepted_with_stale,
                    "universe_primary_non_stale": row.primary_non_stale,
                    "join_stale": row.join_stale,
                    "side_sign": row.side_sign,
                    "future_mid": markout["future_mid"],
                    "future_gap_ms": markout["future_gap_ms"],
                    "raw_mid_markout": markout["raw_mid_markout"],
                    "raw_mid_markout_ticks": markout["raw_mid_markout_ticks"],
                    "side_adjusted_markout_ticks": markout["side_adjusted_markout_ticks"],
                }
            )
    return out


def _top_metric_rows(metric_rows: list[dict[str, Any]], limit: int = 10) -> list[dict[str, Any]]:
    primary = [row for row in metric_rows if row.get("universe") == "primary_non_stale"]
    primary.sort(
        key=lambda row: abs(float(row["spearman"])) if _finite(float(row["spearman"])) else -1.0,
        reverse=True,
    )
    return primary[:limit]


def _stage3_classification(stage3_json: dict[str, Any]) -> str:
    market_view = stage3_json.get("market_view", {})
    return str(market_view.get("classification", "unknown"))


def build_summary_markdown(
    *,
    sample_dir: Path,
    output_dir: Path,
    stage3_json: dict[str, Any],
    sidecar_metrics: dict[str, Any],
    joined_metrics: dict[str, Any],
    row_counts: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
) -> str:
    top_rows = _top_metric_rows(metric_rows, limit=12)
    candidates = [row for row in rejected_rows if row["status"] == "candidate_for_followup"]
    rejected = [row for row in rejected_rows if row["status"] != "candidate_for_followup"]
    lines = [
        "# Stage 4 Pricing Research Summary",
        "",
        "## Scope",
        "",
        f"- Task: `{TASK_ID}`",
        f"- Sample: `{sample_dir}`",
        f"- Output: `{output_dir}`",
        f"- Stage 3 market-view classification: `{_stage3_classification(stage3_json)}`",
        "- Mode: read-only research; no strategy, live, core, connector, or schema changes.",
        "",
        "## Input Gate Snapshot",
        "",
        f"- first_valid_update_aligned: `{sidecar_metrics.get('first_valid_update_aligned')}`",
        f"- depth_pu_mismatch_count: `{sidecar_metrics.get('depth_pu_mismatch_count')}`",
        f"- decision_join_coverage: `{joined_metrics.get('decision_join_coverage')}`",
        f"- future_join_count: `{joined_metrics.get('future_join_count')}`",
        f"- join_missing_count: `{joined_metrics.get('join_missing_count')}`",
        f"- gap_crossed_join_count: `{joined_metrics.get('gap_crossed_join_count')}`",
        f"- stale_join_count: `{joined_metrics.get('stale_join_count')}`",
        "",
        "## Row Counts",
        "",
    ]
    for key in sorted(row_counts):
        lines.append(f"- {key}: `{row_counts[key]}`")
    lines.extend(["", "## Top Primary Non-Stale Signal Rows", ""])
    if top_rows:
        lines.append("| signal | horizon_ms | rows | spearman | top_bottom_spread_ticks | status | reason |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        for row in top_rows:
            lines.append(
                "| {signal} | {horizon_ms} | {rows} | {spearman:.6g} | {spread:.6g} | {status} | {reason} |".format(
                    signal=row["signal"],
                    horizon_ms=row["horizon_ms"],
                    rows=row["rows"],
                    spearman=float(row["spearman"]) if _finite(float(row["spearman"])) else math.nan,
                    spread=float(row["top_bottom_spread_ticks"])
                    if _finite(float(row["top_bottom_spread_ticks"]))
                    else math.nan,
                    status=row["status"],
                    reason=row.get("reason", ""),
                )
            )
    else:
        lines.append("- No primary non-stale metrics were produced.")
    lines.extend(["", "## Candidate Signals", ""])
    if candidates:
        for row in candidates:
            lines.append(
                f"- `{row['signal']}`: best horizon `{row['best_horizon_ms']}ms`, "
                f"abs spearman `{row['best_abs_spearman']:.6g}`, "
                f"top-bottom spread `{row['best_top_bottom_spread_ticks']:.6g}` ticks."
            )
    else:
        lines.append("- No signal met the conservative follow-up threshold on this single sample.")
    lines.extend(["", "## Rejected Or Weak Signals", ""])
    for row in rejected[:12]:
        lines.append(
            f"- `{row['signal']}`: `{row['reason'] or 'not_selected'}` "
            f"(best horizon `{row['best_horizon_ms']}ms`, abs spearman `{row['best_abs_spearman']}`)"
        )
    lines.extend(
        [
            "",
            "## Interpretation Boundary",
            "",
            "- This research can only support a later design task for fair/reservation adjustment candidates.",
            "- It does not prove strategy PnL, full L2 equivalence, exact queue position, queue/fill calibration, or live readiness.",
        ]
    )
    return "\n".join(lines) + "\n"


def _generated_at() -> str:
    epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if epoch:
        return datetime.fromtimestamp(int(epoch), tz=timezone.utc).isoformat()
    return "1970-01-01T00:00:00+00:00"


def run_research(
    *,
    sample_dir: Path,
    output_dir: Path,
    tick_size: float = 0.1,
    horizons_ms: Iterable[int] = DEFAULT_HORIZONS_MS,
    buckets: int = DEFAULT_BUCKETS,
    min_rows: int = DEFAULT_MIN_ROWS,
    min_abs_spearman: float = DEFAULT_MIN_ABS_SPEARMAN,
    min_abs_spread_ticks: float = DEFAULT_MIN_ABS_SPREAD_TICKS,
    max_future_gap_ms: float = DEFAULT_MAX_FUTURE_GAP_MS,
) -> dict[str, Any]:
    sample_dir = _expand(sample_dir)
    output_dir = _expand(output_dir)
    audit_csv = sample_dir / f"audit_live_{sample_dir.name}.csv"
    stage3_path = sample_dir / "maker_acceptance_stage3.json"
    sidecar_dir = sample_dir / "t009_fixed_sidecar"
    top5_sidecar_csv = sidecar_dir / "top5_sidecar.csv"
    joined_decisions_csv = sidecar_dir / "joined_decisions.csv"
    sidecar_metrics_path = sidecar_dir / "metrics.json"
    joined_metrics_path = sidecar_dir / "joined_decisions.metrics.json"

    required = [
        audit_csv,
        stage3_path,
        top5_sidecar_csv,
        joined_decisions_csv,
        sidecar_metrics_path,
        joined_metrics_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    horizons = tuple(int(item) for item in horizons_ms)
    stage3_json = _read_json(stage3_path)
    sidecar_metrics = _read_json(sidecar_metrics_path)
    joined_metrics = _read_json(joined_metrics_path)

    rows, row_counts = build_decision_features(
        audit_csv=audit_csv,
        joined_decisions_csv=joined_decisions_csv,
        top5_sidecar_csv=top5_sidecar_csv,
        tick_size=tick_size,
        horizons_ms=horizons,
        max_future_gap_ms=max_future_gap_ms,
    )
    metric_rows, bucket_tables, rejected_rows = evaluate_signals(
        rows,
        horizons_ms=horizons,
        buckets=buckets,
        min_rows=min_rows,
        min_abs_spearman=min_abs_spearman,
        min_abs_spread_ticks=min_abs_spread_ticks,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_dir = output_dir / "bucket_tables"
    _write_csv(output_dir / "candidate_signal_metrics.csv", metric_rows)
    _write_json(output_dir / "candidate_signal_metrics.json", metric_rows)
    _write_csv(output_dir / "markout_by_horizon.csv", markout_rows_for_csv(rows, horizons))
    _write_csv(output_dir / "rejected_signals.csv", rejected_rows)
    for signal, table_rows in bucket_tables.items():
        _write_csv(bucket_dir / f"{_safe_filename(signal)}.csv", table_rows)

    manifest = {
        "task_id": TASK_ID,
        "schema_version": "pricing_research_v1",
        "generated_at_utc": _generated_at(),
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "tick_size": tick_size,
        "horizons_ms": list(horizons),
        "buckets": buckets,
        "min_rows": min_rows,
        "min_abs_spearman": min_abs_spearman,
        "min_abs_spread_ticks": min_abs_spread_ticks,
        "max_future_gap_ms": max_future_gap_ms,
        "stage3_classification": _stage3_classification(stage3_json),
        "row_counts": row_counts,
        "input_hashes": {str(path.relative_to(sample_dir)): _hash_file(path) for path in required},
        "outputs": {
            "pricing_research_summary": str(output_dir / "pricing_research_summary.md"),
            "candidate_signal_metrics_csv": str(output_dir / "candidate_signal_metrics.csv"),
            "candidate_signal_metrics_json": str(output_dir / "candidate_signal_metrics.json"),
            "markout_by_horizon": str(output_dir / "markout_by_horizon.csv"),
            "rejected_signals": str(output_dir / "rejected_signals.csv"),
            "bucket_tables": str(bucket_dir),
            "run_manifest": str(output_dir / "run_manifest.json"),
        },
        "boundary": {
            "read_only": True,
            "strategy_changes": False,
            "live_changes": False,
            "core_connector_schema_changes": False,
            "queue_fill_proof": False,
            "live_promotion": False,
        },
    }
    summary = build_summary_markdown(
        sample_dir=sample_dir,
        output_dir=output_dir,
        stage3_json=stage3_json,
        sidecar_metrics=sidecar_metrics,
        joined_metrics=joined_metrics,
        row_counts=row_counts,
        metric_rows=metric_rows,
        rejected_rows=rejected_rows,
    )
    (output_dir / "pricing_research_summary.md").write_text(summary, encoding="utf-8")
    _write_json(output_dir / "run_manifest.json", manifest)

    return {
        "row_counts": row_counts,
        "metric_count": len(metric_rows),
        "signal_count": len(rejected_rows),
        "candidate_count": sum(1 for row in rejected_rows if row["status"] == "candidate_for_followup"),
        "output_dir": str(output_dir),
        "manifest": manifest,
        "top_metric_rows": _top_metric_rows(metric_rows, limit=10),
        "signal_status": rejected_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run read-only Stage 4 pricing signal research")
    parser.add_argument("--sample-dir", required=True, help="Accepted sample directory, e.g. local_live_analysis/5-13-day-control-30min")
    parser.add_argument("--output-dir", required=True, help="Output directory for Stage 4 research artifacts")
    parser.add_argument("--tick-size", type=float, default=0.1)
    parser.add_argument("--horizons-ms", default="100,500,1000,5000")
    parser.add_argument("--buckets", type=int, default=DEFAULT_BUCKETS)
    parser.add_argument("--min-rows", type=int, default=DEFAULT_MIN_ROWS)
    parser.add_argument("--min-abs-spearman", type=float, default=DEFAULT_MIN_ABS_SPEARMAN)
    parser.add_argument("--min-abs-spread-ticks", type=float, default=DEFAULT_MIN_ABS_SPREAD_TICKS)
    parser.add_argument("--max-future-gap-ms", type=float, default=DEFAULT_MAX_FUTURE_GAP_MS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = tuple(int(item.strip()) for item in args.horizons_ms.split(",") if item.strip())
    result = run_research(
        sample_dir=Path(args.sample_dir),
        output_dir=Path(args.output_dir),
        tick_size=args.tick_size,
        horizons_ms=horizons,
        buckets=args.buckets,
        min_rows=args.min_rows,
        min_abs_spearman=args.min_abs_spearman,
        min_abs_spread_ticks=args.min_abs_spread_ticks,
        max_future_gap_ms=args.max_future_gap_ms,
    )
    print(json.dumps(_json_clean(result), indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    main()
