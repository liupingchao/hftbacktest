#!/usr/bin/env python3
"""Build Binance-lead / Hyperliquid-lag as-of joined feature artifacts.

This task-scoped runner reads only the accepted local synchronized public
sample from ``0602T001``. Cross-venue time comparison uses local controller
capture timestamps only; venue event timestamps are preserved as diagnostics.
The output is joined-feature input for later analysis, not a lead-lag
statistical conclusion, strategy signal, live path, or promotion artifact.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


getcontext().prec = 28

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0601T002"
SOURCE_TASK_ID = "0602T001"
SCHEMA_VERSION = "cross_exchange_lead_lag_join_v1"
DEFAULT_SAMPLE_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_public_sample_0602T001"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_lead_lag_join_0601T002"
DEFAULT_BINANCE_SYMBOL = "BTCUSDT"
DEFAULT_HYPERLIQUID_COIN = "BTC"
DEFAULT_TICK_SIZE = Decimal("0.1")
TRADE_PRESSURE_STATUS = "disabled_unverified_side_semantics"
TRADE_PRESSURE_BUCKET = "disabled_diagnostic_only"
CONTRACT_BASIS_CAVEAT = "diagnostic_only_binance_usdm_futures_BTCUSDT_vs_hyperliquid_BTC_contract_basis"
BOUNDARY_FLAGS = {
    "no_private_keys": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_process": True,
    "no_live_trading_bot": True,
    "no_parameter_search": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_promotion": True,
}


@dataclass(frozen=True)
class Paths:
    sample_dir: Path
    sample_manifest: Path
    source_run_manifest: Path
    synchronization_quality_summary: Path
    binance_top5_sidecar: Path
    binance_metrics: Path
    hyperliquid_synthetic_joined_views: Path
    hyperliquid_topn_sidecar: Path
    hyperliquid_metrics: Path


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _to_decimal(value: str | int | float | Decimal | None) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(str(value))


def _decimal_text(value: Decimal | float | None, places: int = 8) -> str:
    if value is None:
        return ""
    decimal_value = Decimal(str(value)) if not isinstance(value, Decimal) else value
    quant = Decimal(1).scaleb(-places)
    text = format(decimal_value.quantize(quant), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _int_text(value: int | None) -> str:
    return "" if value is None else str(value)


def _split_decimals(value: str) -> list[Decimal]:
    if not value:
        return []
    return [Decimal(item) for item in value.split("|") if item != ""]


def _best(values: list[Decimal]) -> Decimal | None:
    return values[0] if values else None


def _mid(bid_px: Decimal | None, ask_px: Decimal | None) -> Decimal | None:
    if bid_px is None or ask_px is None:
        return None
    return (bid_px + ask_px) / Decimal(2)


def _spread_ticks(bid_px: Decimal | None, ask_px: Decimal | None, tick_size: Decimal) -> int | None:
    if bid_px is None or ask_px is None:
        return None
    return int(((ask_px - bid_px) / tick_size).to_integral_value())


def _sum_first(values: list[Decimal], n: int) -> Decimal | None:
    subset = values[:n]
    if not subset:
        return None
    return sum(subset, Decimal(0))


def _imbalance(bid_qty: list[Decimal], ask_qty: list[Decimal], n: int) -> Decimal | None:
    bid = _sum_first(bid_qty, n)
    ask = _sum_first(ask_qty, n)
    if bid is None or ask is None or bid + ask == 0:
        return None
    return (bid - ask) / (bid + ask)


def _microprice(
    bid_px: Decimal | None,
    ask_px: Decimal | None,
    bid_qty: list[Decimal],
    ask_qty: list[Decimal],
    n: int,
) -> Decimal | None:
    bid = _sum_first(bid_qty, n)
    ask = _sum_first(ask_qty, n)
    if bid_px is None or ask_px is None or bid is None or ask is None or bid + ask == 0:
        return None
    return (ask_px * bid + bid_px * ask) / (bid + ask)


def _ns_age_ms(source_ts: int | None, decision_ts: int | None) -> Decimal | None:
    if source_ts is None or decision_ts is None:
        return None
    return Decimal(decision_ts - source_ts) / Decimal(1_000_000)


def _age_bucket(age_ms: Decimal | None, stale_source_age_ms: Decimal) -> str:
    if age_ms is None:
        return "missing"
    if age_ms < 0:
        return "future"
    if age_ms <= Decimal("50"):
        return "fresh_0_50ms"
    if age_ms <= Decimal("250"):
        return "warm_50_250ms"
    if age_ms <= stale_source_age_ms:
        return "watch_250ms_to_stale_limit"
    return "stale_over_limit"


def _bookticker_depth_age_bucket(value: str) -> str:
    age = _to_decimal(value)
    if age is None:
        return "missing"
    if age <= Decimal("1"):
        return "fresh_0_1ms"
    if age <= Decimal("10"):
        return "warm_1_10ms"
    if age <= Decimal("50"):
        return "watch_10_50ms"
    return "stale_over_50ms"


def _quality_from_age(age_ms: Decimal | None, missing: bool, future: bool, stale_source_age_ms: Decimal) -> str:
    if missing or future:
        return "diagnostic_only"
    if age_ms is not None and age_ms > stale_source_age_ms:
        return "watch_only_stale_source"
    return "primary_usable"


def _quantiles(values: list[Decimal]) -> dict[str, float]:
    if not values:
        return {"count": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    return {
        "count": len(ordered),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": max(ordered),
    }


def _resolve_paths(sample_dir: Path) -> Paths:
    return Paths(
        sample_dir=sample_dir,
        sample_manifest=sample_dir / "sample_manifest.json",
        source_run_manifest=sample_dir / "run_manifest.json",
        synchronization_quality_summary=sample_dir / "synchronization_quality_summary.json",
        binance_top5_sidecar=sample_dir / "binance_alignment" / "top5_sidecar.csv",
        binance_metrics=sample_dir / "binance_alignment" / "metrics.json",
        hyperliquid_synthetic_joined_views=sample_dir / "hyperliquid_public_sample" / "alignment" / "synthetic_joined_views.csv",
        hyperliquid_topn_sidecar=sample_dir / "hyperliquid_public_sample" / "alignment" / "topn_sidecar.csv",
        hyperliquid_metrics=sample_dir / "hyperliquid_public_sample" / "alignment" / "metrics.json",
    )


def _ensure_required(paths: Paths) -> None:
    missing = [f"{name}: {path}" for name, path in paths.__dict__.items() if isinstance(path, Path) and not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required local 0602T001 artifacts: " + "; ".join(missing))


def build_binance_lead_features(
    rows: list[dict[str, str]],
    *,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
) -> list[dict[str, str]]:
    features: list[dict[str, str]] = []
    previous_mid: Decimal | None = None
    recent_moves: list[Decimal] = []
    for feature_seq, row in enumerate(sorted(rows, key=lambda item: int(item.get("local_ts") or 0))):
        bid_px_values = _split_decimals(row.get("bid_top5_px", ""))
        ask_px_values = _split_decimals(row.get("ask_top5_px", ""))
        bid_qty_values = _split_decimals(row.get("bid_top5_qtys", ""))
        ask_qty_values = _split_decimals(row.get("ask_top5_qtys", ""))
        best_bid = _to_decimal(row.get("bookticker_bid_px")) or _best(bid_px_values)
        best_ask = _to_decimal(row.get("bookticker_ask_px")) or _best(ask_px_values)
        mid_px = _mid(best_bid, best_ask)
        spread_ticks = _spread_ticks(best_bid, best_ask, tick_size)
        top5_imbalance = _imbalance(bid_qty_values, ask_qty_values, 5)
        top5_microprice = _microprice(best_bid, best_ask, bid_qty_values, ask_qty_values, 5)
        microprice_minus_mid_ticks = None
        if top5_microprice is not None and mid_px is not None:
            microprice_minus_mid_ticks = (top5_microprice - mid_px) / tick_size

        mid_move_ticks: Decimal | None = None
        if previous_mid is not None and mid_px is not None:
            mid_move_ticks = (mid_px - previous_mid) / tick_size
            recent_moves.append(mid_move_ticks)
            if len(recent_moves) > 20:
                recent_moves.pop(0)
        previous_mid = mid_px if mid_px is not None else previous_mid

        rolling_abs_5 = None
        rolling_rv_20 = None
        if recent_moves:
            last5 = recent_moves[-5:]
            rolling_abs_5 = sum((abs(value) for value in last5), Decimal(0)) / Decimal(len(last5))
            rolling_rv_20 = Decimal(str(math.sqrt(sum(float(value * value) for value in recent_moves) / len(recent_moves))))

        features.append(
            {
                "binance_feature_seq": str(feature_seq),
                "binance_source_raw_seq": row.get("raw_seq", ""),
                "binance_event_type": row.get("event_type", ""),
                "binance_local_ts": row.get("local_ts", ""),
                "binance_exch_ts": row.get("exch_ts", ""),
                "binance_last_u": row.get("last_u", ""),
                "binance_prev_u": row.get("prev_u", ""),
                "binance_sync_aligned": row.get("sync_aligned", ""),
                "binance_sync_gap": row.get("sync_gap", ""),
                "binance_startup_excluded": row.get("startup_excluded", ""),
                "binance_first_valid_update_aligned": row.get("first_valid_update_aligned", ""),
                "binance_best_bid_px": _decimal_text(best_bid),
                "binance_best_ask_px": _decimal_text(best_ask),
                "binance_mid_px": _decimal_text(mid_px),
                "binance_spread_ticks": _int_text(spread_ticks),
                "binance_bid_top5_px": row.get("bid_top5_px", ""),
                "binance_ask_top5_px": row.get("ask_top5_px", ""),
                "binance_bid_top5_qtys": row.get("bid_top5_qtys", ""),
                "binance_ask_top5_qtys": row.get("ask_top5_qtys", ""),
                "binance_top5_bid_qty": _decimal_text(_sum_first(bid_qty_values, 5)),
                "binance_top5_ask_qty": _decimal_text(_sum_first(ask_qty_values, 5)),
                "binance_top5_imbalance": _decimal_text(top5_imbalance),
                "binance_top5_microprice_px": _decimal_text(top5_microprice),
                "binance_microprice_minus_mid_ticks": _decimal_text(microprice_minus_mid_ticks),
                "binance_mid_move_ticks_from_prev": _decimal_text(mid_move_ticks),
                "binance_rolling_abs_mid_move_ticks_5": _decimal_text(rolling_abs_5),
                "binance_rolling_rv_ticks_20": _decimal_text(rolling_rv_20),
                "binance_bookticker_u": row.get("bookticker_u", ""),
                "binance_bookticker_local_ts": row.get("bookticker_local_ts", ""),
                "binance_bookticker_depth_age_ms": row.get("bookticker_depth_age_ms", ""),
                "binance_bookticker_depth_age_bucket": _bookticker_depth_age_bucket(row.get("bookticker_depth_age_ms", "")),
                "binance_bookticker_bbo_match": row.get("bookticker_bbo_match", ""),
                "binance_trade_pressure_status": TRADE_PRESSURE_STATUS,
                "binance_trade_pressure_bucket": TRADE_PRESSURE_BUCKET,
            }
        )
    return features


def build_hyperliquid_lag_context(
    join_rows: list[dict[str, str]],
    topn_rows: list[dict[str, str]],
    *,
    coin: str,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
    stale_source_age_ms: Decimal = Decimal("1000"),
) -> list[dict[str, str]]:
    topn_by_raw_seq = {row.get("raw_seq", ""): row for row in topn_rows}
    contexts: list[dict[str, str]] = []
    for join_row in sorted(join_rows, key=lambda item: int(item.get("decision_ts") or 0)):
        topn_row = topn_by_raw_seq.get(join_row.get("joined_raw_seq", ""))
        bid_px_values = _split_decimals((topn_row or {}).get("bid_topn_px", ""))
        ask_px_values = _split_decimals((topn_row or {}).get("ask_topn_px", ""))
        bid_qty_values = _split_decimals((topn_row or {}).get("bid_topn_qtys", ""))
        ask_qty_values = _split_decimals((topn_row or {}).get("ask_topn_qtys", ""))
        best_bid = _to_decimal(join_row.get("best_bid_px")) or _best(bid_px_values)
        best_ask = _to_decimal(join_row.get("best_ask_px")) or _best(ask_px_values)
        mid_px = _mid(best_bid, best_ask)
        spread_ticks = _spread_ticks(best_bid, best_ask, tick_size)
        top5_imbalance = _imbalance(bid_qty_values, ask_qty_values, 5)
        top5_microprice = _microprice(best_bid, best_ask, bid_qty_values, ask_qty_values, 5)
        microprice_minus_mid_ticks = None
        if top5_microprice is not None and mid_px is not None:
            microprice_minus_mid_ticks = (top5_microprice - mid_px) / tick_size

        join_age_ms = _to_decimal(join_row.get("join_age_ms"))
        future = join_row.get("future_join", "false") == "true"
        missing = join_row.get("missing_join", "false") == "true"
        contexts.append(
            {
                "hyperliquid_decision_seq": join_row.get("decision_seq", ""),
                "hyperliquid_decision_ts": join_row.get("decision_ts", ""),
                "hyperliquid_joined_raw_seq": join_row.get("joined_raw_seq", ""),
                "hyperliquid_l2book_local_ts": join_row.get("joined_l2book_local_ts", ""),
                "hyperliquid_l2book_event_ts": join_row.get("joined_l2book_event_ts", ""),
                "hyperliquid_join_age_ms": join_row.get("join_age_ms", ""),
                "hyperliquid_join_age_bucket": _age_bucket(join_age_ms, stale_source_age_ms),
                "hyperliquid_future_join": join_row.get("future_join", ""),
                "hyperliquid_missing_join": join_row.get("missing_join", ""),
                "hyperliquid_reconnect_recovery_crossed": join_row.get("reconnect_recovery_crossed", ""),
                "hyperliquid_coin": (topn_row or {}).get("coin", coin) or coin,
                "hyperliquid_best_bid_px": _decimal_text(best_bid),
                "hyperliquid_best_ask_px": _decimal_text(best_ask),
                "hyperliquid_mid_px": _decimal_text(mid_px),
                "hyperliquid_spread_ticks": _int_text(spread_ticks),
                "hyperliquid_bid_topn_px": (topn_row or {}).get("bid_topn_px", ""),
                "hyperliquid_ask_topn_px": (topn_row or {}).get("ask_topn_px", ""),
                "hyperliquid_bid_topn_qtys": (topn_row or {}).get("bid_topn_qtys", ""),
                "hyperliquid_ask_topn_qtys": (topn_row or {}).get("ask_topn_qtys", ""),
                "hyperliquid_bid_topn_order_count": (topn_row or {}).get("bid_topn_n", ""),
                "hyperliquid_ask_topn_order_count": (topn_row or {}).get("ask_topn_n", ""),
                "hyperliquid_top5_bid_qty": _decimal_text(_sum_first(bid_qty_values, 5)),
                "hyperliquid_top5_ask_qty": _decimal_text(_sum_first(ask_qty_values, 5)),
                "hyperliquid_top5_imbalance": _decimal_text(top5_imbalance),
                "hyperliquid_top5_microprice_px": _decimal_text(top5_microprice),
                "hyperliquid_microprice_minus_mid_ticks": _decimal_text(microprice_minus_mid_ticks),
                "hyperliquid_context_quality": _quality_from_age(join_age_ms, missing, future, stale_source_age_ms),
                "hyperliquid_trade_pressure_status": TRADE_PRESSURE_STATUS,
                "hyperliquid_trade_pressure_bucket": TRADE_PRESSURE_BUCKET,
            }
        )
    return contexts


def asof_join_binance_to_hyperliquid(
    binance_features: list[dict[str, str]],
    hyperliquid_context: list[dict[str, str]],
    *,
    stale_source_age_ms: Decimal = Decimal("1000"),
) -> list[dict[str, str]]:
    sorted_binance = sorted(binance_features, key=lambda row: int(row.get("binance_local_ts") or 0))
    binance_ts = [int(row.get("binance_local_ts") or 0) for row in sorted_binance]
    joined: list[dict[str, str]] = []
    for join_seq, lag_row in enumerate(sorted(hyperliquid_context, key=lambda row: int(row.get("hyperliquid_decision_ts") or 0))):
        decision_ts = int(lag_row.get("hyperliquid_decision_ts") or 0)
        idx = bisect.bisect_right(binance_ts, decision_ts) - 1
        lead_row = sorted_binance[idx] if idx >= 0 else None
        source_ts = int(lead_row["binance_local_ts"]) if lead_row is not None and lead_row.get("binance_local_ts") else None
        source_age_ms = _ns_age_ms(source_ts, decision_ts)
        future_join = bool(source_ts is not None and source_ts > decision_ts)
        missing_join = lead_row is None
        binance_quality = _quality_from_age(source_age_ms, missing_join, future_join, stale_source_age_ms)
        hl_quality = lag_row.get("hyperliquid_context_quality", "diagnostic_only")
        primary_usable = binance_quality == "primary_usable" and hl_quality == "primary_usable"

        basis_mid = None
        basis_micro = None
        basis_mid_ticks = None
        binance_mid = _to_decimal((lead_row or {}).get("binance_mid_px"))
        binance_micro = _to_decimal((lead_row or {}).get("binance_top5_microprice_px"))
        hl_mid = _to_decimal(lag_row.get("hyperliquid_mid_px"))
        if binance_mid is not None and hl_mid is not None:
            basis_mid = binance_mid - hl_mid
            basis_mid_ticks = basis_mid / DEFAULT_TICK_SIZE
        if binance_micro is not None and hl_mid is not None:
            basis_micro = binance_micro - hl_mid

        output: dict[str, str] = {
            "join_seq": str(join_seq),
            "timestamp_clock_policy": "local_controller_capture_ts_ns",
            "hyperliquid_decision_ts": lag_row.get("hyperliquid_decision_ts", ""),
            "binance_source_found": "true" if lead_row is not None else "false",
            "binance_source_age_ms": _decimal_text(source_age_ms, places=6),
            "binance_source_age_bucket": _age_bucket(source_age_ms, stale_source_age_ms),
            "cross_exchange_future_join": "true" if future_join else "false",
            "cross_exchange_missing_binance_join": "true" if missing_join else "false",
            "binance_lead_quality": binance_quality,
            "hyperliquid_lag_quality": hl_quality,
            "joined_row_quality": "primary_usable" if primary_usable else "watch_or_diagnostic",
            "basis_mid_px": _decimal_text(basis_mid),
            "basis_mid_ticks": _decimal_text(basis_mid_ticks),
            "basis_microprice_px": _decimal_text(basis_micro),
            "basis_contract_caveat": CONTRACT_BASIS_CAVEAT,
            "lead_lag_statistical_conclusion": "not_calculated_in_0601T002",
        }
        if lead_row is not None:
            output.update(lead_row)
        else:
            output.update({field: "" for field in BINANCE_FIELDNAMES})
        output.update(lag_row)
        joined.append(output)
    return joined


BINANCE_FIELDNAMES = [
    "binance_feature_seq",
    "binance_source_raw_seq",
    "binance_event_type",
    "binance_local_ts",
    "binance_exch_ts",
    "binance_last_u",
    "binance_prev_u",
    "binance_sync_aligned",
    "binance_sync_gap",
    "binance_startup_excluded",
    "binance_first_valid_update_aligned",
    "binance_best_bid_px",
    "binance_best_ask_px",
    "binance_mid_px",
    "binance_spread_ticks",
    "binance_bid_top5_px",
    "binance_ask_top5_px",
    "binance_bid_top5_qtys",
    "binance_ask_top5_qtys",
    "binance_top5_bid_qty",
    "binance_top5_ask_qty",
    "binance_top5_imbalance",
    "binance_top5_microprice_px",
    "binance_microprice_minus_mid_ticks",
    "binance_mid_move_ticks_from_prev",
    "binance_rolling_abs_mid_move_ticks_5",
    "binance_rolling_rv_ticks_20",
    "binance_bookticker_u",
    "binance_bookticker_local_ts",
    "binance_bookticker_depth_age_ms",
    "binance_bookticker_depth_age_bucket",
    "binance_bookticker_bbo_match",
    "binance_trade_pressure_status",
    "binance_trade_pressure_bucket",
]

HYPERLIQUID_FIELDNAMES = [
    "hyperliquid_decision_seq",
    "hyperliquid_decision_ts",
    "hyperliquid_joined_raw_seq",
    "hyperliquid_l2book_local_ts",
    "hyperliquid_l2book_event_ts",
    "hyperliquid_join_age_ms",
    "hyperliquid_join_age_bucket",
    "hyperliquid_future_join",
    "hyperliquid_missing_join",
    "hyperliquid_reconnect_recovery_crossed",
    "hyperliquid_coin",
    "hyperliquid_best_bid_px",
    "hyperliquid_best_ask_px",
    "hyperliquid_mid_px",
    "hyperliquid_spread_ticks",
    "hyperliquid_bid_topn_px",
    "hyperliquid_ask_topn_px",
    "hyperliquid_bid_topn_qtys",
    "hyperliquid_ask_topn_qtys",
    "hyperliquid_bid_topn_order_count",
    "hyperliquid_ask_topn_order_count",
    "hyperliquid_top5_bid_qty",
    "hyperliquid_top5_ask_qty",
    "hyperliquid_top5_imbalance",
    "hyperliquid_top5_microprice_px",
    "hyperliquid_microprice_minus_mid_ticks",
    "hyperliquid_context_quality",
    "hyperliquid_trade_pressure_status",
    "hyperliquid_trade_pressure_bucket",
]

JOIN_FIELDNAMES = [
    "join_seq",
    "timestamp_clock_policy",
    "hyperliquid_decision_ts",
    "binance_source_found",
    "binance_source_age_ms",
    "binance_source_age_bucket",
    "cross_exchange_future_join",
    "cross_exchange_missing_binance_join",
    "binance_lead_quality",
    "hyperliquid_lag_quality",
    "joined_row_quality",
    "basis_mid_px",
    "basis_mid_ticks",
    "basis_microprice_px",
    "basis_contract_caveat",
    "lead_lag_statistical_conclusion",
] + BINANCE_FIELDNAMES + HYPERLIQUID_FIELDNAMES

BASIS_FIELDNAMES = [
    "metric",
    "row_count",
    "mean_px",
    "min_px",
    "max_px",
    "caveat",
]


def _quality_summary(
    *,
    paths: Paths,
    source_sync_summary: dict[str, Any],
    binance_features: list[dict[str, str]],
    hyperliquid_context: list[dict[str, str]],
    joined_rows: list[dict[str, str]],
) -> dict[str, Any]:
    source_ages = [_to_decimal(row.get("binance_source_age_ms")) for row in joined_rows if row.get("binance_source_age_ms")]
    source_ages = [value for value in source_ages if value is not None]
    binance_age_buckets = Counter(row.get("binance_source_age_bucket", "missing") for row in joined_rows)
    joined_quality = Counter(row.get("joined_row_quality", "unknown") for row in joined_rows)
    future_join_count = sum(row.get("cross_exchange_future_join") == "true" for row in joined_rows)
    missing_join_count = sum(row.get("cross_exchange_missing_binance_join") == "true" for row in joined_rows)
    stale_join_count = sum(row.get("binance_source_age_bucket") == "stale_over_limit" for row in joined_rows)
    if future_join_count:
        raise ValueError(f"future_join_count must be zero, got {future_join_count}")
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "source_sample_dir": str(paths.sample_dir),
        "input_row_counts": {
            "binance_top5_sidecar_rows": _count_csv_rows(paths.binance_top5_sidecar),
            "hyperliquid_synthetic_join_rows": _count_csv_rows(paths.hyperliquid_synthetic_joined_views),
            "hyperliquid_topn_sidecar_rows": _count_csv_rows(paths.hyperliquid_topn_sidecar),
        },
        "output_row_counts": {
            "binance_lead_feature_rows": len(binance_features),
            "hyperliquid_lag_context_rows": len(hyperliquid_context),
            "joined_feature_rows": len(joined_rows),
        },
        "cross_exchange_join": {
            "timestamp_clock_policy": "local_controller_capture_ts_ns",
            "join_policy": "bisect_right as-of where binance_local_ts <= hyperliquid_decision_ts",
            "future_join_count": future_join_count,
            "missing_binance_join_count": missing_join_count,
            "stale_binance_source_count": stale_join_count,
            "primary_usable_row_count": joined_quality.get("primary_usable", 0),
            "watch_or_diagnostic_row_count": joined_quality.get("watch_or_diagnostic", 0),
            "binance_source_age_ms": _quantiles(source_ages),
            "binance_source_age_bucket_counts": dict(sorted(binance_age_buckets.items())),
        },
        "hyperliquid_lag_context": {
            "future_join_count": sum(row.get("hyperliquid_future_join") == "true" for row in hyperliquid_context),
            "missing_join_count": sum(row.get("hyperliquid_missing_join") == "true" for row in hyperliquid_context),
            "context_quality_counts": dict(Counter(row.get("hyperliquid_context_quality", "") for row in hyperliquid_context)),
        },
        "disabled_features": {
            "binance_trade_pressure_status": TRADE_PRESSURE_STATUS,
            "binance_trade_pressure_disabled_rows": len(binance_features),
            "hyperliquid_trade_pressure_status": TRADE_PRESSURE_STATUS,
            "hyperliquid_trade_pressure_disabled_rows": len(hyperliquid_context),
        },
        "source_overlap": source_sync_summary.get("overlap", {}),
        "boundary_flags": BOUNDARY_FLAGS,
        "lead_lag_statistical_conclusion": "not_calculated_in_0601T002",
        "basis_dislocation_caveat": CONTRACT_BASIS_CAVEAT,
    }


def _basis_summary(joined_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metric in ["basis_mid_px", "basis_microprice_px"]:
        values = [_to_decimal(row.get(metric)) for row in joined_rows if row.get(metric)]
        values = [value for value in values if value is not None]
        if values:
            mean_value = sum(values, Decimal(0)) / Decimal(len(values))
            rows.append(
                {
                    "metric": metric,
                    "row_count": str(len(values)),
                    "mean_px": _decimal_text(mean_value),
                    "min_px": _decimal_text(min(values)),
                    "max_px": _decimal_text(max(values)),
                    "caveat": CONTRACT_BASIS_CAVEAT,
                }
            )
        else:
            rows.append({"metric": metric, "row_count": "0", "mean_px": "", "min_px": "", "max_px": "", "caveat": CONTRACT_BASIS_CAVEAT})
    return rows


def _write_report(path: Path, quality: dict[str, Any]) -> None:
    join = quality["cross_exchange_join"]
    text = f"""# Cross-Exchange Lead/Lag Join Report

Task: `{TASK_ID}`

Source sample: `{quality["source_sample_dir"]}`

## Scope

- Reads only accepted synchronized public-data artifacts from `{SOURCE_TASK_ID}`.
- Joins Binance USD-M Futures `BTCUSDT` lead rows to Hyperliquid `BTC` lag decision timestamps.
- Uses local controller capture timestamps as the cross-venue clock: `binance_local_ts <= hyperliquid_decision_ts`.
- Does not calculate lead-lag stability, predictive edge, strategy readiness, tiny-live readiness, or promotion.

## Row Counts

- Binance lead feature rows: `{quality["output_row_counts"]["binance_lead_feature_rows"]}`
- Hyperliquid lag context rows: `{quality["output_row_counts"]["hyperliquid_lag_context_rows"]}`
- Joined feature rows: `{quality["output_row_counts"]["joined_feature_rows"]}`
- Primary usable joined rows: `{join["primary_usable_row_count"]}`
- Watch/diagnostic joined rows: `{join["watch_or_diagnostic_row_count"]}`

## Join Quality

- Future cross-exchange joins: `{join["future_join_count"]}`
- Missing Binance joins: `{join["missing_binance_join_count"]}`
- Stale Binance source rows: `{join["stale_binance_source_count"]}`
- Binance source age p99 ms: `{join["binance_source_age_ms"]["p99"]}`

## Disabled Features

- Binance trade pressure: `{TRADE_PRESSURE_STATUS}`
- Hyperliquid trade pressure: `{TRADE_PRESSURE_STATUS}`

## Caveats

- Basis/dislocation fields are diagnostic only because they compare Binance USD-M Futures `BTCUSDT` with Hyperliquid `BTC` contract context.
- Output is synchronized public-data joined-feature input for `0601T003` only.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_join_artifacts(
    *,
    sample_dir: str | Path,
    output_dir: str | Path,
    binance_symbol: str = DEFAULT_BINANCE_SYMBOL,
    hyperliquid_coin: str = DEFAULT_HYPERLIQUID_COIN,
    tick_size: Decimal = DEFAULT_TICK_SIZE,
    stale_source_age_ms: Decimal = Decimal("1000"),
) -> dict[str, Any]:
    resolved_sample_dir = _expand(sample_dir)
    resolved_output_dir = _expand(output_dir)
    paths = _resolve_paths(resolved_sample_dir)
    _ensure_required(paths)

    source_sample_manifest = _read_json(paths.sample_manifest)
    source_run_manifest = _read_json(paths.source_run_manifest)
    source_sync_summary = _read_json(paths.synchronization_quality_summary)
    binance_metrics = _read_json(paths.binance_metrics)
    hyperliquid_metrics = _read_json(paths.hyperliquid_metrics)

    binance_features = build_binance_lead_features(_read_csv(paths.binance_top5_sidecar), tick_size=tick_size)
    hyperliquid_context = build_hyperliquid_lag_context(
        _read_csv(paths.hyperliquid_synthetic_joined_views),
        _read_csv(paths.hyperliquid_topn_sidecar),
        coin=hyperliquid_coin,
        tick_size=tick_size,
        stale_source_age_ms=stale_source_age_ms,
    )
    joined_rows = asof_join_binance_to_hyperliquid(
        binance_features,
        hyperliquid_context,
        stale_source_age_ms=stale_source_age_ms,
    )
    quality = _quality_summary(
        paths=paths,
        source_sync_summary=source_sync_summary,
        binance_features=binance_features,
        hyperliquid_context=hyperliquid_context,
        joined_rows=joined_rows,
    )
    basis_summary = _basis_summary(joined_rows)

    sample_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_sample_dir": str(resolved_sample_dir),
        "source_sample_manifest": str(paths.sample_manifest),
        "source_run_manifest": str(paths.source_run_manifest),
        "source_synchronization_quality_summary": str(paths.synchronization_quality_summary),
        "venues": {
            "binance_lead": {
                "exchange": "binance_usdm_futures",
                "symbol": binance_symbol,
                "role": "lead_public_market_data",
                "artifact": str(paths.binance_top5_sidecar),
                "timestamp_field": "local_ts",
                "event_timestamp_field": "exch_ts_diagnostic_only",
            },
            "hyperliquid_lag": {
                "exchange": "hyperliquid",
                "coin": hyperliquid_coin,
                "role": "lag_public_market_data",
                "decision_grid_artifact": str(paths.hyperliquid_synthetic_joined_views),
                "topn_artifact": str(paths.hyperliquid_topn_sidecar),
                "timestamp_field": "decision_ts",
                "event_timestamp_field": "joined_l2book_event_ts_diagnostic_only",
            },
        },
        "timestamp_policy": {
            "cross_venue_clock": "local_controller_capture_ts_ns",
            "asof_rule": "binance_local_ts <= hyperliquid_decision_ts",
            "venue_event_timestamps": "diagnostic_only",
        },
        "boundary_flags": BOUNDARY_FLAGS,
        "lead_lag_statistical_conclusion": "not_calculated_in_0601T002",
        "basis_dislocation_caveat": CONTRACT_BASIS_CAVEAT,
    }
    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "generated_at": sample_manifest["generated_at"],
        "git_commit": _git_commit(),
        "sample_dir": str(resolved_sample_dir),
        "output_dir": str(resolved_output_dir),
        "source_schema_versions": {
            "source_sample": source_sample_manifest.get("schema_version", ""),
            "source_run": source_run_manifest.get("schema_version", ""),
            "binance_metrics": binance_metrics.get("schema_version", ""),
            "hyperliquid_metrics": hyperliquid_metrics.get("schema_version", ""),
        },
        "artifacts": {
            "sample_manifest": str(resolved_output_dir / "sample_manifest.json"),
            "run_manifest": str(resolved_output_dir / "run_manifest.json"),
            "binance_lead_features": str(resolved_output_dir / "binance_lead_features.csv"),
            "hyperliquid_lag_context": str(resolved_output_dir / "hyperliquid_lag_context.csv"),
            "cross_exchange_joined_features": str(resolved_output_dir / "cross_exchange_joined_features.csv"),
            "join_quality_summary": str(resolved_output_dir / "join_quality_summary.json"),
            "basis_dislocation_summary": str(resolved_output_dir / "basis_dislocation_summary.csv"),
            "cross_exchange_join_report": str(resolved_output_dir / "cross_exchange_join_report.md"),
        },
        "row_counts": quality["output_row_counts"],
        "boundary_flags": BOUNDARY_FLAGS,
        "quality": quality["cross_exchange_join"],
    }

    _write_csv(resolved_output_dir / "binance_lead_features.csv", binance_features, BINANCE_FIELDNAMES)
    _write_csv(resolved_output_dir / "hyperliquid_lag_context.csv", hyperliquid_context, HYPERLIQUID_FIELDNAMES)
    _write_csv(resolved_output_dir / "cross_exchange_joined_features.csv", joined_rows, JOIN_FIELDNAMES)
    _write_csv(resolved_output_dir / "basis_dislocation_summary.csv", basis_summary, BASIS_FIELDNAMES)
    _write_json(resolved_output_dir / "sample_manifest.json", sample_manifest)
    _write_json(resolved_output_dir / "run_manifest.json", run_manifest)
    _write_json(resolved_output_dir / "join_quality_summary.json", quality)
    _write_report(resolved_output_dir / "cross_exchange_join_report.md", quality)
    return {
        "sample_manifest": sample_manifest,
        "run_manifest": run_manifest,
        "join_quality_summary": quality,
        "basis_dislocation_summary": basis_summary,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Binance-lead / Hyperliquid-lag local as-of joined features.")
    parser.add_argument("--sample-dir", type=Path, default=DEFAULT_SAMPLE_DIR, help="Accepted 0602T001 synchronized public sample directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for 0601T002 join artifacts.")
    parser.add_argument("--binance-symbol", default=DEFAULT_BINANCE_SYMBOL)
    parser.add_argument("--hyperliquid-coin", default=DEFAULT_HYPERLIQUID_COIN)
    parser.add_argument("--tick-size", default=str(DEFAULT_TICK_SIZE))
    parser.add_argument("--stale-source-age-ms", default="1000")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_join_artifacts(
        sample_dir=args.sample_dir,
        output_dir=args.output_dir,
        binance_symbol=args.binance_symbol,
        hyperliquid_coin=args.hyperliquid_coin,
        tick_size=Decimal(str(args.tick_size)),
        stale_source_age_ms=Decimal(str(args.stale_source_age_ms)),
    )
    quality = result["join_quality_summary"]["cross_exchange_join"]
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "output_dir": str(_expand(args.output_dir)),
                "joined_feature_rows": result["run_manifest"]["row_counts"]["joined_feature_rows"],
                "future_join_count": quality["future_join_count"],
                "missing_binance_join_count": quality["missing_binance_join_count"],
                "primary_usable_row_count": quality["primary_usable_row_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
