#!/usr/bin/env python3
"""Compare audit_bt and audit_live with shared schema and alignment metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from audit_schema import REQUIRED_ALIGNMENT_FIELDS


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
        if missing:
            raise KeyError(f"Missing required columns in {path}: {missing}")
        return list(reader)


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


def _alignment(bt_rows: list[dict[str, str]], live_rows: list[dict[str, str]]) -> dict[str, Any]:
    bt_by_seq = _align_by_seq(bt_rows)
    live_by_seq = _align_by_seq(live_rows)

    common = sorted(set(bt_by_seq.keys()) & set(live_by_seq.keys()))
    if not common:
        return {
            "common_rows": 0,
            "action_match_rate": 0.0,
            "reject_reason_match_rate": 0.0,
            "mae": {},
        }

    metrics = ["fair", "reservation", "half_spread", "position", "inventory_score", "spread_bps", "vol_bps"]
    bt_series: dict[str, list[float]] = {k: [] for k in metrics}
    live_series: dict[str, list[float]] = {k: [] for k in metrics}

    action_match = 0
    reject_match = 0

    for seq in common:
        b = bt_by_seq[seq]
        l = live_by_seq[seq]

        if str(b.get("action", "")) == str(l.get("action", "")):
            action_match += 1
        if str(b.get("reject_reason", "")) == str(l.get("reject_reason", "")):
            reject_match += 1

        for k in metrics:
            bt_series[k].append(_safe_float(b.get(k, 0.0)))
            live_series[k].append(_safe_float(l.get(k, 0.0)))

    mae = {k: _mae(bt_series[k], live_series[k]) for k in metrics}

    return {
        "common_rows": len(common),
        "action_match_rate": float(action_match / len(common)),
        "reject_reason_match_rate": float(reject_match / len(common)),
        "mae": mae,
    }


def compare(bt_csv: Path, live_csv: Path) -> dict[str, Any]:
    bt_rows = _read_csv_rows(bt_csv)
    live_rows = _read_csv_rows(live_csv)

    return {
        "bt_file": str(bt_csv),
        "live_file": str(live_csv),
        "bt_summary": _summary(bt_rows),
        "live_summary": _summary(live_rows),
        "alignment": _alignment(bt_rows, live_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare backtest and live audit csv")
    parser.add_argument("--bt", required=True, help="audit_bt.csv path")
    parser.add_argument("--live", required=True, help="audit_live.csv path")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bt_csv = _expand(args.bt)
    live_csv = _expand(args.live)
    report = compare(bt_csv, live_csv)

    print(json.dumps(report, indent=2, ensure_ascii=True))

    if args.out:
        out = _expand(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
