#!/usr/bin/env python3
"""Streaming metrics and audit-output helpers for binance_tick_mm backtests."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_AUDIT_MODES = {"full", "actions_only", "sampled", "off"}


@dataclass
class AuditPolicy:
    mode: str = "full"
    sample_every: int = 1000

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "AuditPolicy":
        mode = str(cfg.get("mode", "full"))
        sample_every = int(cfg.get("sample_every", 1000))
        if mode not in VALID_AUDIT_MODES:
            raise ValueError(f"Unsupported audit.mode={mode!r}; expected one of {sorted(VALID_AUDIT_MODES)}")
        if mode == "sampled" and sample_every <= 0:
            raise ValueError("audit.sample_every must be > 0 when audit.mode='sampled'")
        return cls(mode=mode, sample_every=sample_every)

    def should_write(self, row: dict[str, Any], seq: int) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "full":
            return True

        action = str(row.get("action", "keep"))
        reject_reason = str(row.get("reject_reason", ""))
        is_action_or_reject = action != "keep" or bool(reject_reason)

        if self.mode == "actions_only":
            return is_action_or_reject
        if self.mode == "sampled":
            return is_action_or_reject or (seq % self.sample_every == 0)
        raise AssertionError(f"unreachable audit mode: {self.mode}")


@dataclass
class LatencyHistogram:
    """Fixed-memory latency histogram in milliseconds."""

    max_ms: float = 100.0
    bucket_ms: float = 0.1
    buckets: list[int] = field(default_factory=list)
    count: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        if not self.buckets:
            n = int(self.max_ms / self.bucket_ms) + 1
            self.buckets = [0] * n

    def add(self, value_ms: float) -> None:
        if not math.isfinite(value_ms) or value_ms < 0:
            return
        idx = int(value_ms / self.bucket_ms)
        if idx >= len(self.buckets):
            idx = len(self.buckets) - 1
        self.buckets[idx] += 1
        self.count += 1
        self.total += value_ms

    def percentile(self, q: float) -> float:
        if self.count == 0:
            return 0.0
        target = max(1, int(math.ceil(self.count * q)))
        seen = 0
        for idx, n in enumerate(self.buckets):
            seen += n
            if seen >= target:
                return idx * self.bucket_ms
        return self.max_ms

    def summary(self) -> dict[str, float]:
        if self.count == 0:
            return {"count": 0.0, "mean": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}
        return {
            "count": float(self.count),
            "mean": self.total / self.count,
            "p50": self.percentile(0.50),
            "p90": self.percentile(0.90),
            "p99": self.percentile(0.99),
        }


@dataclass
class MetricAccumulator:
    rows: int = 0
    pnl_mtm: float = 0.0
    peak_pnl: float = 0.0
    max_drawdown_mtm: float = 0.0
    prev_mid: float | None = None
    prev_position: float | None = None
    inventory_score_sum: float = 0.0
    spread_bps_sum: float = 0.0
    vol_bps_sum: float = 0.0
    abs_position_notional_sum: float = 0.0
    max_abs_position_notional: float = 0.0
    drop_latency_count: int = 0
    drop_api_count: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    reject_counts: Counter[str] = field(default_factory=Counter)
    feed_latency_ms: LatencyHistogram = field(default_factory=LatencyHistogram)
    entry_latency_ms: LatencyHistogram = field(default_factory=LatencyHistogram)
    latency_signal_ms: LatencyHistogram = field(default_factory=LatencyHistogram)

    def update(self, row: dict[str, Any]) -> None:
        mid = float(row["mid"])
        position = float(row["position"])

        if self.prev_mid is not None and self.prev_position is not None:
            self.pnl_mtm += self.prev_position * (mid - self.prev_mid)
            if self.pnl_mtm > self.peak_pnl:
                self.peak_pnl = self.pnl_mtm
            drawdown = self.peak_pnl - self.pnl_mtm
            if drawdown > self.max_drawdown_mtm:
                self.max_drawdown_mtm = drawdown

        self.prev_mid = mid
        self.prev_position = position
        self.rows += 1

        self.inventory_score_sum += float(row["inventory_score"])
        self.spread_bps_sum += float(row["spread_bps"])
        self.vol_bps_sum += float(row["vol_bps"])

        abs_notional = abs(position * mid)
        self.abs_position_notional_sum += abs_notional
        if abs_notional > self.max_abs_position_notional:
            self.max_abs_position_notional = abs_notional

        if int(row["dropped_by_latency"]) > 0:
            self.drop_latency_count += 1
        if int(row["dropped_by_api_limit"]) > 0:
            self.drop_api_count += 1

        self.action_counts[str(row["action"])] += 1
        reject_reason = str(row.get("reject_reason", ""))
        if reject_reason:
            self.reject_counts[reject_reason] += 1

        self.feed_latency_ms.add(float(row["feed_latency_ns"]) / 1_000_000.0)
        self.entry_latency_ms.add(float(row["entry_latency_ns"]) / 1_000_000.0)
        self.latency_signal_ms.add(float(row["latency_signal_ms"]))

    def summary(self) -> dict[str, Any]:
        n = float(self.rows)
        if self.rows == 0:
            return {
                "rows": 0,
                "pnl_mtm": 0.0,
                "max_drawdown_mtm": 0.0,
                "avg_inventory_score": 0.0,
                "avg_abs_position_notional": 0.0,
                "max_abs_position_notional": 0.0,
                "avg_spread_bps": 0.0,
                "avg_vol_bps": 0.0,
                "drop_latency_rate": 0.0,
                "drop_api_rate": 0.0,
                "actions": {},
                "reject_reasons": {},
                "feed_latency_ms": self.feed_latency_ms.summary(),
                "entry_latency_ms": self.entry_latency_ms.summary(),
                "latency_signal_ms": self.latency_signal_ms.summary(),
            }
        return {
            "rows": self.rows,
            "pnl_mtm": self.pnl_mtm,
            "max_drawdown_mtm": self.max_drawdown_mtm,
            "avg_inventory_score": self.inventory_score_sum / n,
            "avg_abs_position_notional": self.abs_position_notional_sum / n,
            "max_abs_position_notional": self.max_abs_position_notional,
            "avg_spread_bps": self.spread_bps_sum / n,
            "avg_vol_bps": self.vol_bps_sum / n,
            "drop_latency_rate": self.drop_latency_count / n,
            "drop_api_rate": self.drop_api_count / n,
            "actions": dict(self.action_counts),
            "reject_reasons": dict(self.reject_counts),
            "feed_latency_ms": self.feed_latency_ms.summary(),
            "entry_latency_ms": self.entry_latency_ms.summary(),
            "latency_signal_ms": self.latency_signal_ms.summary(),
        }


def utc_day_from_ns(ts_ns: int) -> str:
    return datetime.fromtimestamp(ts_ns / 1_000_000_000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def flatten_summary(prefix: str, summary: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in summary.items():
        name = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, dict):
                    for leaf_key, leaf_val in sub_val.items():
                        out[f"{name}_{sub_key}_{leaf_key}"] = leaf_val
                else:
                    out[f"{name}_{sub_key}"] = sub_val
        else:
            out[name] = value
    return out


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))


def write_daily_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
