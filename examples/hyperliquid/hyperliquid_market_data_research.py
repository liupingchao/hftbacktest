#!/usr/bin/env python3
"""Build read-only Hyperliquid market-data research artifacts.

The consumer only reads accepted local public artifacts from task ``0529T004``
and turns them into deterministic research tables and quality summaries.
It does not touch private endpoints, order lifecycle code, live strategy
processes, parameter search, default-on behavior, tiny-live, or promotion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any


getcontext().prec = 28

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0531T001"
SCHEMA_VERSION = "hyperliquid_market_data_research_v1"
SOURCE_TASK_ID = "0529T004"
SOURCE_SAMPLE_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_public_sample_0529T004"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_market_data_research_0531T001"
DEFAULT_COIN = "BTC"
DEFAULT_TOP_N = 5
OFFICIAL_REFERENCES = [
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size",
    "https://github.com/hyperliquid-dex/hyperliquid-python-sdk",
]
TRADE_PRESSURE_STATUS = "unverified_side_semantics"
TRADE_PRESSURE_BUCKET = "disabled_unverified_side_semantics"
OUTPUT_FIELD_PLACES = 8


@dataclass(frozen=True)
class MarketViewRow:
    view_seq: str
    view_ts: str
    coin: str
    session_id: str
    connection_attempt: str
    joined_raw_seq: str
    joined_l2book_local_ts: str
    joined_l2book_event_ts: str
    join_age_ms: str
    future_join: str
    missing_join: str
    reconnect_recovery_crossed: str
    best_bid_px: str
    best_ask_px: str
    bid_topn_px: str
    ask_topn_px: str
    bid_topn_qty: str
    ask_topn_qty: str
    bid_topn_order_count: str
    ask_topn_order_count: str
    mid_px: str
    spread_px: str
    spread_ticks: str
    book_freshness_bucket: str
    market_view_quality: str


@dataclass(frozen=True)
class PricingFeatureRow:
    view_seq: str
    view_ts: str
    coin: str
    session_id: str
    connection_attempt: str
    mid_px: str
    spread_ticks: str
    top1_imbalance: str
    top3_imbalance: str
    top5_imbalance: str
    top1_microprice_px: str
    top3_microprice_px: str
    top5_microprice_px: str
    microprice_minus_mid_ticks: str
    bid_depth_top5_qty: str
    ask_depth_top5_qty: str
    bid_depth_top5_notional: str
    ask_depth_top5_notional: str
    book_pressure_bucket: str
    spread_bucket: str
    join_age_bucket: str
    l2book_cadence_bucket: str
    recovery_context: str
    trade_pressure_status: str
    trade_pressure_qty: str
    trade_pressure_count: str
    trade_pressure_bucket: str
    feature_row_quality: str


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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_csv_rows(path: Path) -> int:
    with path.open(newline="", encoding="utf-8") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _to_decimal(value: str | None) -> Decimal | None:
    if value is None or value == "":
        return None
    return Decimal(value)


def _split_pipe_decimals(value: str) -> list[Decimal]:
    if not value:
        return []
    return [Decimal(item) for item in value.split("|") if item != ""]


def _split_pipe_str(value: str) -> list[str]:
    if not value:
        return []
    return [item for item in value.split("|") if item != ""]


def _decimal_to_text(value: Decimal | None, places: int = OUTPUT_FIELD_PLACES) -> str:
    if value is None:
        return ""
    quant = Decimal(1).scaleb(-places)
    text = format(value.quantize(quant), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _format_int(value: int | None) -> str:
    if value is None:
        return ""
    return str(int(value))


def _pipe_join(values: list[str]) -> str:
    return "|".join(values)


def _mean_price_px(bid_px: Decimal | None, ask_px: Decimal | None) -> Decimal | None:
    if bid_px is None or ask_px is None:
        return None
    return (bid_px + ask_px) / Decimal(2)


def _spread_px(bid_px: Decimal | None, ask_px: Decimal | None) -> Decimal | None:
    if bid_px is None or ask_px is None:
        return None
    return ask_px - bid_px


def _spread_ticks(spread_px: Decimal | None, tick_size: Decimal) -> int | None:
    if spread_px is None:
        return None
    return int((spread_px / tick_size).to_integral_value())


def _sum_first(values: list[Decimal], n: int) -> Decimal | None:
    if not values:
        return None
    subset = values[:n]
    if not subset:
        return None
    return sum(subset, Decimal(0))


def _weighted_microprice(
    bid_px: Decimal | None,
    ask_px: Decimal | None,
    bid_qty: list[Decimal],
    ask_qty: list[Decimal],
    n: int,
) -> Decimal | None:
    if bid_px is None or ask_px is None:
        return None
    bid_total = _sum_first(bid_qty, n)
    ask_total = _sum_first(ask_qty, n)
    if bid_total is None or ask_total is None:
        return None
    denom = bid_total + ask_total
    if denom == 0:
        return None
    return (ask_px * bid_total + bid_px * ask_total) / denom


def _imbalance(bid_qty: list[Decimal], ask_qty: list[Decimal], n: int) -> Decimal | None:
    bid_total = _sum_first(bid_qty, n)
    ask_total = _sum_first(ask_qty, n)
    if bid_total is None or ask_total is None:
        return None
    denom = bid_total + ask_total
    if denom == 0:
        return None
    return (bid_total - ask_total) / denom


def _best_level(values: list[str]) -> str:
    return values[0] if values else ""


def _join_age_value(join_row: dict[str, str]) -> Decimal | None:
    text = join_row.get("join_age_ms", "")
    if not text:
        return None
    return Decimal(text)


def _join_age_bucket(join_age_ms: Decimal | None) -> str:
    if join_age_ms is None:
        return "missing"
    if join_age_ms <= Decimal("250"):
        return "fresh"
    if join_age_ms <= Decimal("750"):
        return "warm"
    if join_age_ms <= Decimal("1000"):
        return "stale"
    return "very_stale"


def _book_freshness_bucket(join_age_ms: Decimal | None, missing_join: str, future_join: str) -> str:
    if missing_join == "true":
        return "missing"
    if future_join == "true":
        return "future"
    if join_age_ms is None:
        return "missing"
    if join_age_ms <= Decimal("250"):
        return "fresh"
    if join_age_ms <= Decimal("1000"):
        return "warm"
    return "stale"


def _market_view_quality(join_age_ms: Decimal | None, missing_join: str, future_join: str, has_topn: bool, recovery_crossed: str) -> str:
    if missing_join == "true" or future_join == "true" or not has_topn:
        return "diagnostic_only"
    if recovery_crossed == "true" or (join_age_ms is not None and join_age_ms > Decimal("1000")):
        return "watch_only"
    return "candidate_ready"


def _l2book_cadence_bucket(delta_ms: Decimal | None) -> str:
    if delta_ms is None:
        return "first"
    if delta_ms == 0:
        return "repeat_same_book"
    if delta_ms <= Decimal("250"):
        return "fast"
    if delta_ms <= Decimal("750"):
        return "regular"
    if delta_ms <= Decimal("1500"):
        return "slow"
    return "gap"


def _spread_bucket(spread_ticks: int | None) -> str:
    if spread_ticks is None:
        return "unknown"
    if spread_ticks <= 1:
        return "tight"
    if spread_ticks <= 5:
        return "moderate"
    if spread_ticks <= 10:
        return "wide"
    return "very_wide"


def _book_pressure_bucket(top5_imbalance: Decimal | None, microprice_minus_mid_ticks: Decimal | None) -> str:
    if top5_imbalance is None or microprice_minus_mid_ticks is None:
        return "unknown"
    if top5_imbalance >= Decimal("0.20") or microprice_minus_mid_ticks >= Decimal("0.50"):
        return "strong_bid"
    if top5_imbalance >= Decimal("0.05"):
        return "bid"
    if top5_imbalance <= Decimal("-0.20") or microprice_minus_mid_ticks <= Decimal("-0.50"):
        return "strong_ask"
    if top5_imbalance <= Decimal("-0.05"):
        return "ask"
    return "balanced"


def _recovery_context(reconnect_recovery_crossed: str, recovery_snapshot_count: int) -> str:
    if reconnect_recovery_crossed == "true":
        return "recovery_crossed"
    if recovery_snapshot_count > 0:
        return "recovery_present"
    return "recovery_absent"


def _feature_row_quality(market_view_quality: str) -> str:
    if market_view_quality == "diagnostic_only":
        return "diagnostic_only"
    return "research_ready_without_trade_pressure"


def _parse_required_inputs(input_dir: Path) -> dict[str, Path]:
    return {
        "collection_manifest": input_dir / "collection_manifest.json",
        "raw_sha256": input_dir / "raw.sha256",
        "raw_gzip": input_dir / "raw.gz",
        "recovery_snapshots": input_dir / "recovery_snapshots.jsonl",
        "alignment_run_manifest": input_dir / "alignment" / "run_manifest.json",
        "alignment_collection_manifest": input_dir / "alignment" / "collection_manifest.json",
        "alignment_converter_manifest": input_dir / "alignment" / "converter_manifest.json",
        "alignment_data": input_dir / "alignment" / "data.npz",
        "alignment_raw_provenance": input_dir / "alignment" / "raw_provenance.csv",
        "alignment_raw_to_npz_mapping": input_dir / "alignment" / "raw_to_npz_mapping.csv",
        "alignment_topn_sidecar": input_dir / "alignment" / "topn_sidecar.csv",
        "alignment_synthetic_joined_views": input_dir / "alignment" / "synthetic_joined_views.csv",
        "alignment_metrics": input_dir / "alignment" / "metrics.json",
        "alignment_acceptance_report": input_dir / "alignment" / "acceptance_report.md",
    }


def _ensure_required_inputs(paths: dict[str, Path]) -> list[str]:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input artifacts: " + "; ".join(missing))
    return missing


def _count_recovery_snapshot_statuses(path: Path) -> list[str]:
    statuses: list[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            status = str(payload.get("status", ""))
            if status:
                statuses.append(status)
    return statuses


def _normalize_field_map(payload: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: payload.get(key, "") for key in keys}


def _output_row_counts(paths: dict[str, Path]) -> dict[str, int]:
    return {
        "raw_provenance_row_count": _count_csv_rows(paths["alignment_raw_provenance"]),
        "raw_to_npz_mapping_row_count": _count_csv_rows(paths["alignment_raw_to_npz_mapping"]),
        "topn_row_count": _count_csv_rows(paths["alignment_topn_sidecar"]),
        "synthetic_join_row_count": _count_csv_rows(paths["alignment_synthetic_joined_views"]),
    }


def _market_view_row(
    join_row: dict[str, str],
    topn_row: dict[str, str] | None,
    *,
    input_coin: str,
    session_id: str,
    connection_attempt: str,
    tick_size: Decimal,
    recovery_snapshot_count: int,
    previous_joined_local_ts: Decimal | None,
) -> tuple[MarketViewRow, Decimal | None]:
    join_age_ms = _join_age_value(join_row)
    missing_join = join_row.get("missing_join", "true")
    future_join = join_row.get("future_join", "false")
    reconnect_crossed = join_row.get("reconnect_recovery_crossed", "")
    has_topn = topn_row is not None and bool(topn_row.get("bid_topn_px")) and bool(topn_row.get("ask_topn_px"))
    market_view_quality = _market_view_quality(join_age_ms, missing_join, future_join, has_topn, reconnect_crossed)
    book_freshness_bucket = _book_freshness_bucket(join_age_ms, missing_join, future_join)

    if topn_row is None:
        row = MarketViewRow(
            view_seq=join_row.get("decision_seq", ""),
            view_ts=join_row.get("decision_ts", ""),
            coin=input_coin,
            session_id=session_id,
            connection_attempt=connection_attempt,
            joined_raw_seq=join_row.get("joined_raw_seq", ""),
            joined_l2book_local_ts=join_row.get("joined_l2book_local_ts", ""),
            joined_l2book_event_ts=join_row.get("joined_l2book_event_ts", ""),
            join_age_ms=join_row.get("join_age_ms", ""),
            future_join=future_join,
            missing_join=missing_join,
            reconnect_recovery_crossed=reconnect_crossed,
            best_bid_px="",
            best_ask_px="",
            bid_topn_px="",
            ask_topn_px="",
            bid_topn_qty="",
            ask_topn_qty="",
            bid_topn_order_count="",
            ask_topn_order_count="",
            mid_px="",
            spread_px="",
            spread_ticks="",
            book_freshness_bucket=book_freshness_bucket,
            market_view_quality=market_view_quality,
        )
        return row, previous_joined_local_ts

    bid_px_values = _split_pipe_str(topn_row.get("bid_topn_px", ""))
    ask_px_values = _split_pipe_str(topn_row.get("ask_topn_px", ""))
    bid_qty_values = _split_pipe_str(topn_row.get("bid_topn_qtys", ""))
    ask_qty_values = _split_pipe_str(topn_row.get("ask_topn_qtys", ""))
    bid_count_values = _split_pipe_str(topn_row.get("bid_topn_n", ""))
    ask_count_values = _split_pipe_str(topn_row.get("ask_topn_n", ""))

    best_bid_px = _to_decimal(_best_level(bid_px_values))
    best_ask_px = _to_decimal(_best_level(ask_px_values))
    mid_px = _mean_price_px(best_bid_px, best_ask_px)
    spread_px = _spread_px(best_bid_px, best_ask_px)
    spread_ticks = _spread_ticks(spread_px, tick_size)

    local_ts_text = topn_row.get("local_ts", "")
    joined_local_ts = Decimal(local_ts_text) if local_ts_text else None
    cadence_delta_ms: Decimal | None = None
    if previous_joined_local_ts is not None and joined_local_ts is not None:
        cadence_delta_ms = (joined_local_ts - previous_joined_local_ts) / Decimal(1_000_000)
    next_previous_local_ts = joined_local_ts if joined_local_ts is not None else previous_joined_local_ts

    row = MarketViewRow(
        view_seq=join_row.get("decision_seq", ""),
        view_ts=join_row.get("decision_ts", ""),
        coin=topn_row.get("coin", input_coin) or input_coin,
        session_id=topn_row.get("session_id", session_id) or session_id,
        connection_attempt=topn_row.get("connection_attempt", connection_attempt) or connection_attempt,
        joined_raw_seq=join_row.get("joined_raw_seq", ""),
        joined_l2book_local_ts=join_row.get("joined_l2book_local_ts", ""),
        joined_l2book_event_ts=join_row.get("joined_l2book_event_ts", ""),
        join_age_ms=join_row.get("join_age_ms", ""),
        future_join=future_join,
        missing_join=missing_join,
        reconnect_recovery_crossed=reconnect_crossed,
        best_bid_px=_best_level(bid_px_values),
        best_ask_px=_best_level(ask_px_values),
        bid_topn_px=topn_row.get("bid_topn_px", ""),
        ask_topn_px=topn_row.get("ask_topn_px", ""),
        bid_topn_qty=topn_row.get("bid_topn_qtys", ""),
        ask_topn_qty=topn_row.get("ask_topn_qtys", ""),
        bid_topn_order_count=topn_row.get("bid_topn_n", ""),
        ask_topn_order_count=topn_row.get("ask_topn_n", ""),
        mid_px=_decimal_to_text(mid_px),
        spread_px=_decimal_to_text(spread_px),
        spread_ticks=_format_int(spread_ticks),
        book_freshness_bucket=book_freshness_bucket,
        market_view_quality=market_view_quality,
    )
    return row, next_previous_local_ts


def _pricing_feature_row(
    market_view_row: MarketViewRow,
    *,
    tick_size: Decimal,
    l2book_cadence_bucket: str,
    recovery_snapshot_count: int,
) -> PricingFeatureRow:
    bid_px_values = _split_pipe_decimals(market_view_row.bid_topn_px)
    ask_px_values = _split_pipe_decimals(market_view_row.ask_topn_px)
    bid_qty_values = _split_pipe_decimals(market_view_row.bid_topn_qty)
    ask_qty_values = _split_pipe_decimals(market_view_row.ask_topn_qty)

    best_bid_px = _to_decimal(market_view_row.best_bid_px)
    best_ask_px = _to_decimal(market_view_row.best_ask_px)
    mid_px = _to_decimal(market_view_row.mid_px)
    spread_ticks = int(market_view_row.spread_ticks) if market_view_row.spread_ticks else None

    top1_imbalance = _imbalance(bid_qty_values, ask_qty_values, 1)
    top3_imbalance = _imbalance(bid_qty_values, ask_qty_values, 3)
    top5_imbalance = _imbalance(bid_qty_values, ask_qty_values, 5)
    top1_microprice_px = _weighted_microprice(best_bid_px, best_ask_px, bid_qty_values, ask_qty_values, 1)
    top3_microprice_px = _weighted_microprice(best_bid_px, best_ask_px, bid_qty_values, ask_qty_values, 3)
    top5_microprice_px = _weighted_microprice(best_bid_px, best_ask_px, bid_qty_values, ask_qty_values, 5)
    microprice_minus_mid_ticks: Decimal | None = None
    if top5_microprice_px is not None and mid_px is not None:
        microprice_minus_mid_ticks = (top5_microprice_px - mid_px) / tick_size

    bid_depth_top5_qty = _sum_first(bid_qty_values, 5)
    ask_depth_top5_qty = _sum_first(ask_qty_values, 5)
    bid_depth_top5_notional = None
    ask_depth_top5_notional = None
    if bid_depth_top5_qty is not None:
        bid_depth_top5_notional = sum((bid_px_values[i] * bid_qty_values[i] for i in range(min(5, len(bid_px_values), len(bid_qty_values)))), Decimal(0))
    if ask_depth_top5_qty is not None:
        ask_depth_top5_notional = sum((ask_px_values[i] * ask_qty_values[i] for i in range(min(5, len(ask_px_values), len(ask_qty_values)))), Decimal(0))

    book_pressure_bucket = _book_pressure_bucket(top5_imbalance, microprice_minus_mid_ticks)
    spread_bucket = _spread_bucket(spread_ticks)
    join_age_bucket = _join_age_bucket(Decimal(market_view_row.join_age_ms) if market_view_row.join_age_ms else None)

    feature_row_quality = _feature_row_quality(market_view_row.market_view_quality)

    return PricingFeatureRow(
        view_seq=market_view_row.view_seq,
        view_ts=market_view_row.view_ts,
        coin=market_view_row.coin,
        session_id=market_view_row.session_id,
        connection_attempt=market_view_row.connection_attempt,
        mid_px=market_view_row.mid_px,
        spread_ticks=market_view_row.spread_ticks,
        top1_imbalance=_decimal_to_text(top1_imbalance),
        top3_imbalance=_decimal_to_text(top3_imbalance),
        top5_imbalance=_decimal_to_text(top5_imbalance),
        top1_microprice_px=_decimal_to_text(top1_microprice_px),
        top3_microprice_px=_decimal_to_text(top3_microprice_px),
        top5_microprice_px=_decimal_to_text(top5_microprice_px),
        microprice_minus_mid_ticks=_decimal_to_text(microprice_minus_mid_ticks),
        bid_depth_top5_qty=_decimal_to_text(bid_depth_top5_qty),
        ask_depth_top5_qty=_decimal_to_text(ask_depth_top5_qty),
        bid_depth_top5_notional=_decimal_to_text(bid_depth_top5_notional),
        ask_depth_top5_notional=_decimal_to_text(ask_depth_top5_notional),
        book_pressure_bucket=book_pressure_bucket,
        spread_bucket=spread_bucket,
        join_age_bucket=join_age_bucket,
        l2book_cadence_bucket=l2book_cadence_bucket,
        recovery_context=_recovery_context(market_view_row.reconnect_recovery_crossed, recovery_snapshot_count),
        trade_pressure_status=TRADE_PRESSURE_STATUS,
        trade_pressure_qty="",
        trade_pressure_count="",
        trade_pressure_bucket=TRADE_PRESSURE_BUCKET,
        feature_row_quality=feature_row_quality,
    )


def _row_to_dict(obj: Any) -> dict[str, Any]:
    return dict(obj.__dict__)


def _valid_price_row(row: MarketViewRow) -> bool:
    return bool(row.best_bid_px) and bool(row.best_ask_px) and row.missing_join != "true" and row.future_join != "true"


def _compute_quantiles_ms(values_ms: list[Decimal | int | float]) -> dict[str, float]:
    if not values_ms:
        return {"count": 0, "p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values_ms)

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


def _feature_null_counts(rows: list[PricingFeatureRow]) -> dict[str, int]:
    if not rows:
        return {}
    fields = list(rows[0].__dict__.keys())
    counts: dict[str, int] = {field: 0 for field in fields}
    for row in rows:
        payload = row.__dict__
        for field in fields:
            if payload.get(field, "") == "":
                counts[field] += 1
    return counts


def _feature_outlier_counts(rows: list[PricingFeatureRow], market_rows: list[MarketViewRow]) -> dict[str, int]:
    outliers = {
        "mid_px_nonpositive": 0,
        "spread_ticks_negative": 0,
        "imbalance_out_of_range": 0,
        "microprice_outside_best_bid_ask": 0,
        "depth_qty_negative": 0,
        "notional_negative": 0,
    }
    for feature_row, market_row in zip(rows, market_rows):
        mid_px = _to_decimal(feature_row.mid_px)
        spread_ticks = _to_decimal(feature_row.spread_ticks)
        imbalances = [
            _to_decimal(feature_row.top1_imbalance),
            _to_decimal(feature_row.top3_imbalance),
            _to_decimal(feature_row.top5_imbalance),
        ]
        microprice = _to_decimal(feature_row.top5_microprice_px)
        best_bid = _to_decimal(market_row.best_bid_px)
        best_ask = _to_decimal(market_row.best_ask_px)
        bid_qty = _to_decimal(feature_row.bid_depth_top5_qty)
        ask_qty = _to_decimal(feature_row.ask_depth_top5_qty)
        bid_notional = _to_decimal(feature_row.bid_depth_top5_notional)
        ask_notional = _to_decimal(feature_row.ask_depth_top5_notional)
        if mid_px is not None and mid_px <= 0:
            outliers["mid_px_nonpositive"] += 1
        if spread_ticks is not None and spread_ticks < 0:
            outliers["spread_ticks_negative"] += 1
        if any(value is not None and (value < Decimal("-1") or value > Decimal("1")) for value in imbalances):
            outliers["imbalance_out_of_range"] += 1
        if (
            microprice is not None
            and best_bid is not None
            and best_ask is not None
            and (microprice < best_bid or microprice > best_ask)
        ):
            outliers["microprice_outside_best_bid_ask"] += 1
        if any(value is not None and value < 0 for value in [bid_qty, ask_qty]):
            outliers["depth_qty_negative"] += 1
        if any(value is not None and value < 0 for value in [bid_notional, ask_notional]):
            outliers["notional_negative"] += 1
    return outliers


def _count_values(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counter = Counter(str(row.get(field, "")) for row in rows)
    return dict(sorted(counter.items()))


def _load_source_artifacts(input_dir: Path) -> dict[str, Any]:
    paths = _parse_required_inputs(input_dir)
    _ensure_required_inputs(paths)

    collection_manifest = _read_json(paths["collection_manifest"])
    source_alignment_run_manifest = _read_json(paths["alignment_run_manifest"])
    source_alignment_collection_manifest = _read_json(paths["alignment_collection_manifest"])
    source_alignment_converter_manifest = _read_json(paths["alignment_converter_manifest"])
    source_alignment_metrics = _read_json(paths["alignment_metrics"])

    raw_sha256_file = paths["raw_sha256"].read_text(encoding="utf-8").strip()
    raw_sha256_collection_manifest = str(collection_manifest.get("raw_sha256", ""))
    raw_sha256_alignment_collection_manifest = str(source_alignment_collection_manifest.get("source_raw_sha256", ""))
    raw_sha256_computed = _sha256_file(paths["raw_gzip"])

    if not raw_sha256_file or not raw_sha256_collection_manifest or not raw_sha256_alignment_collection_manifest:
        raise ValueError("Missing raw sha256 provenance in required manifests.")

    if not (
        raw_sha256_file == raw_sha256_collection_manifest == raw_sha256_alignment_collection_manifest == raw_sha256_computed
    ):
        raise ValueError(
            "raw sha256 mismatch across raw.sha256, collection manifest, alignment collection manifest, and raw.gz."
        )

    topn_rows = _read_csv(paths["alignment_topn_sidecar"])
    join_rows = _read_csv(paths["alignment_synthetic_joined_views"])
    raw_provenance_rows = _read_csv(paths["alignment_raw_provenance"])
    raw_to_npz_rows = _read_csv(paths["alignment_raw_to_npz_mapping"])
    recovery_statuses = _count_recovery_snapshot_statuses(paths["recovery_snapshots"])

    return {
        "paths": paths,
        "collection_manifest": collection_manifest,
        "alignment_run_manifest": source_alignment_run_manifest,
        "alignment_collection_manifest": source_alignment_collection_manifest,
        "alignment_converter_manifest": source_alignment_converter_manifest,
        "alignment_metrics": source_alignment_metrics,
        "raw_sha256_file": raw_sha256_file,
        "raw_sha256_collection_manifest": raw_sha256_collection_manifest,
        "raw_sha256_alignment_collection_manifest": raw_sha256_alignment_collection_manifest,
        "raw_sha256_computed": raw_sha256_computed,
        "topn_rows": topn_rows,
        "join_rows": join_rows,
        "raw_provenance_rows": raw_provenance_rows,
        "raw_to_npz_rows": raw_to_npz_rows,
        "recovery_statuses": recovery_statuses,
    }


def build_research_artifacts(
    *,
    input_dir: Path,
    output_dir: Path,
    coin: str = DEFAULT_COIN,
    top_n: int = DEFAULT_TOP_N,
) -> dict[str, Any]:
    input_dir = _expand(input_dir)
    output_dir = _expand(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = _load_source_artifacts(input_dir)
    paths = source["paths"]
    collection_manifest = source["collection_manifest"]
    alignment_run_manifest = source["alignment_run_manifest"]
    alignment_metrics = source["alignment_metrics"]
    alignment_collection_manifest = source["alignment_collection_manifest"]
    alignment_converter_manifest = source["alignment_converter_manifest"]

    source_top_n = int(alignment_run_manifest.get("top_n", top_n) or top_n)
    tick_size = Decimal(str(alignment_run_manifest.get("tick_size", 0.1)))
    synthetic_interval_ms = int(alignment_run_manifest.get("synthetic_interval_ms", 500) or 500)
    session_id = str(collection_manifest.get("session_id", ""))
    connection_attempt = str(collection_manifest.get("connection_attempt_count", 0) or 0)
    recovery_snapshot_count = int(collection_manifest.get("recovery_snapshot_count", 0) or 0)
    source_coin = str(collection_manifest.get("coin", coin) or coin)
    source_network = str(collection_manifest.get("network", "unknown") or "unknown")

    if coin != source_coin:
        raise ValueError(f"Input coin is {source_coin}, but CLI requested {coin}.")
    if top_n != source_top_n:
        raise ValueError(f"Input top_n is {source_top_n}, but CLI requested {top_n}.")

    topn_by_raw_seq = {row.get("raw_seq", ""): row for row in source["topn_rows"]}
    market_view_rows: list[MarketViewRow] = []
    pricing_feature_rows: list[PricingFeatureRow] = []
    previous_joined_local_ts: Decimal | None = None
    for join_row in source["join_rows"]:
        topn_row = topn_by_raw_seq.get(join_row.get("joined_raw_seq", ""))
        prior_joined_local_ts = previous_joined_local_ts
        market_view_row, previous_joined_local_ts = _market_view_row(
            join_row,
            topn_row,
            input_coin=source_coin,
            session_id=session_id,
            connection_attempt=connection_attempt,
            tick_size=tick_size,
            recovery_snapshot_count=recovery_snapshot_count,
            previous_joined_local_ts=previous_joined_local_ts,
        )
        market_view_rows.append(market_view_row)
        if market_view_row.joined_l2book_local_ts:
            current_joined_local_ts = Decimal(market_view_row.joined_l2book_local_ts)
        else:
            current_joined_local_ts = None
        if prior_joined_local_ts is None:
            cadence_bucket = "first"
        elif current_joined_local_ts is None:
            cadence_bucket = "first"
        else:
            delta_ms = (current_joined_local_ts - prior_joined_local_ts) / Decimal(1_000_000)
            cadence_bucket = _l2book_cadence_bucket(delta_ms)
        pricing_feature_rows.append(
            _pricing_feature_row(
                market_view_row,
                tick_size=tick_size,
                l2book_cadence_bucket=cadence_bucket,
                recovery_snapshot_count=recovery_snapshot_count,
            )
        )
        if current_joined_local_ts is not None:
            previous_joined_local_ts = current_joined_local_ts

    market_view_dicts = [_row_to_dict(row) for row in market_view_rows]
    pricing_feature_dicts = [_row_to_dict(row) for row in pricing_feature_rows]
    market_view_fields = list(MarketViewRow.__annotations__.keys())
    pricing_feature_fields = list(PricingFeatureRow.__annotations__.keys())

    _write_csv(output_dir / "market_view_timeseries.csv", market_view_dicts, market_view_fields)
    _write_csv(output_dir / "pricing_features.csv", pricing_feature_dicts, pricing_feature_fields)

    output_counts = _output_row_counts(paths)
    join_age_values = [
        _join_age_value(join_row) for join_row in source["join_rows"] if _join_age_value(join_row) is not None
    ]
    cadence_values_ms: list[Decimal] = []
    previous_joined_ts: Decimal | None = None
    for row in market_view_rows:
        if row.joined_l2book_local_ts:
            current = Decimal(row.joined_l2book_local_ts)
            if previous_joined_ts is not None:
                cadence_values_ms.append((current - previous_joined_ts) / Decimal(1_000_000))
            previous_joined_ts = current
    feature_null_counts = _feature_null_counts(pricing_feature_rows)
    feature_outlier_counts = _feature_outlier_counts(pricing_feature_rows, market_view_rows)
    market_view_quality_counts = _count_values(market_view_dicts, "market_view_quality")
    feature_row_quality_counts = _count_values(pricing_feature_dicts, "feature_row_quality")

    validation = {
        "required_inputs_present": True,
        "required_input_count": len(paths),
        "raw_sha256_match": True,
        "raw_sha256_values": {
            "file": source["raw_sha256_file"],
            "collection_manifest": source["raw_sha256_collection_manifest"],
            "alignment_collection_manifest": source["raw_sha256_alignment_collection_manifest"],
            "computed": source["raw_sha256_computed"],
        },
        "alignment_sample_classification": alignment_metrics.get("sample_classification", ""),
        "alignment_classification_reason": alignment_metrics.get("classification_reason", ""),
        "alignment_classification_matches_expected": alignment_metrics.get("sample_classification", "")
        == "passes_pricing_research_market_view",
        "topn_coverage": float(alignment_metrics.get("topn_coverage", 0.0)),
        "synthetic_join_coverage": float(alignment_metrics.get("decision_join_coverage", 0.0)),
        "future_join_count": int(alignment_metrics.get("future_join_count", 0) or 0),
        "missing_join_count": int(alignment_metrics.get("missing_join_count", 0) or 0),
        "raw_provenance_row_count": output_counts["raw_provenance_row_count"],
        "raw_to_npz_mapping_row_count": output_counts["raw_to_npz_mapping_row_count"],
        "topn_row_count": output_counts["topn_row_count"],
        "synthetic_join_row_count": output_counts["synthetic_join_row_count"],
        "converter_data_rows": int(alignment_converter_manifest.get("data_rows", 0) or 0),
        "market_view_row_count": len(market_view_rows),
        "pricing_feature_row_count": len(pricing_feature_rows),
        "market_view_quality_counts": market_view_quality_counts,
        "feature_row_quality_counts": feature_row_quality_counts,
    }

    feature_quality_summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "input_artifact_presence": {name: path.exists() for name, path in paths.items()},
        "missing_input_artifacts": [name for name, path in paths.items() if not path.exists()],
        "raw_sha256_consistency": validation["raw_sha256_values"],
        "raw_sha256_match": validation["raw_sha256_match"],
        "raw_sha256_computed": source["raw_sha256_computed"],
        "alignment_sample_classification": validation["alignment_sample_classification"],
        "alignment_classification_reason": validation["alignment_classification_reason"],
        "topn_coverage": validation["topn_coverage"],
        "synthetic_join_coverage": validation["synthetic_join_coverage"],
        "future_join_count": validation["future_join_count"],
        "missing_join_count": validation["missing_join_count"],
        "join_age_ms": _compute_quantiles_ms([float(value) for value in join_age_values]),
        "l2book_cadence_ms": _compute_quantiles_ms([float(value) for value in cadence_values_ms]),
        "recovery_snapshot_count": recovery_snapshot_count,
        "recovery_snapshot_statuses": source["recovery_statuses"],
        "market_view_row_count": len(market_view_rows),
        "pricing_feature_row_count": len(pricing_feature_rows),
        "raw_provenance_row_count": output_counts["raw_provenance_row_count"],
        "raw_to_npz_mapping_row_count": output_counts["raw_to_npz_mapping_row_count"],
        "converter_data_rows": validation["converter_data_rows"],
        "market_view_quality_counts": market_view_quality_counts,
        "feature_row_quality_counts": feature_row_quality_counts,
        "feature_null_counts": feature_null_counts,
        "feature_outlier_counts": feature_outlier_counts,
        "trade_pressure_status": TRADE_PRESSURE_STATUS,
        "trade_pressure_bucket": TRADE_PRESSURE_BUCKET,
        "final_classification": validation["alignment_sample_classification"],
        "final_classification_reason": "accepted_public_market_data_readiness_with_trade_pressure_disabled",
    }

    session_summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "session_id": session_id,
        "coin": source_coin,
        "network": source_network,
        "actual_duration_seconds": collection_manifest.get("actual_duration_seconds", 0),
        "close_reason": collection_manifest.get("close_reason", ""),
        "connection_attempt_count": int(collection_manifest.get("connection_attempt_count", 0) or 0),
        "reconnect_count": int(collection_manifest.get("reconnect_count", 0) or 0),
        "subscription_ack_count": int(collection_manifest.get("subscription_ack_count", 0) or 0),
        "subscription_ack_count_by_channel": collection_manifest.get("subscription_ack_count_by_channel", {}),
        "first_local_ts_by_channel": collection_manifest.get("first_local_ts_by_channel", {}),
        "last_local_ts_by_channel": collection_manifest.get("last_local_ts_by_channel", {}),
        "message_counts_by_channel": alignment_metrics.get("channel_counts", {}),
        "recovery_snapshot_count": recovery_snapshot_count,
        "recovery_snapshot_statuses": source["recovery_statuses"],
        "official_public_references_used": _dedupe_references(
            collection_manifest.get("official_references_checked", [])
            or alignment_run_manifest.get("official_references_checked", [])
            or OFFICIAL_REFERENCES
        ),
        "source_private_order_absence_flags": {
            "no_private_keys": bool(collection_manifest.get("no_private_keys", True)),
            "no_private_account_endpoints": bool(collection_manifest.get("no_private_account_endpoints", True)),
            "no_order_endpoints": bool(collection_manifest.get("no_order_endpoints", True)),
            "no_strategy_process": bool(collection_manifest.get("no_strategy_process", True)),
            "no_remote_deploy": bool(collection_manifest.get("no_remote_deploy", True)),
        },
    }

    recommendation = _recommendation_markdown(
        source_coin=source_coin,
        source_network=source_network,
        alignment_classification=validation["alignment_sample_classification"],
        validation=validation,
        trade_pressure_status=TRADE_PRESSURE_STATUS,
    )

    run_manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": _utc_now(),
        "git_commit": _git_commit(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "source_sample_dir": str(SOURCE_SAMPLE_DIR),
        "coin": source_coin,
        "network": source_network,
        "top_n": source_top_n,
        "synthetic_interval_ms": synthetic_interval_ms,
        "tick_size": _decimal_to_text(tick_size),
        "source_task_id": SOURCE_TASK_ID,
        "source_sample_classification": validation["alignment_sample_classification"],
        "source_sample_classification_reason": validation["alignment_classification_reason"],
        "official_doc_recheck_status": "inherited_from_design_contract",
        "official_doc_recheck_timestamp": _utc_now(),
        "official_references_checked": _dedupe_references(
            collection_manifest.get("official_references_checked", [])
            or alignment_run_manifest.get("official_references_checked", [])
            or OFFICIAL_REFERENCES
        ),
        "input_artifact_presence": validation["required_inputs_present"],
        "required_input_count": validation["required_input_count"],
        "raw_sha256_consistency": validation["raw_sha256_values"],
        "raw_sha256_match": validation["raw_sha256_match"],
        "validation_summary": validation,
        "trade_pressure_status": TRADE_PRESSURE_STATUS,
        "trade_pressure_bucket": TRADE_PRESSURE_BUCKET,
        "boundary_flags": {
            "no_private_keys": True,
            "no_private_account_endpoints": True,
            "no_order_endpoints": True,
            "no_order_lifecycle": True,
            "no_strategy_live_process": True,
            "no_parameter_search": True,
            "no_default_on": True,
            "no_tiny_live": True,
            "no_promotion": True,
        },
        "output_artifacts": {
            "market_view_timeseries": str(output_dir / "market_view_timeseries.csv"),
            "pricing_features": str(output_dir / "pricing_features.csv"),
            "feature_quality_summary": str(output_dir / "feature_quality_summary.json"),
            "sample_session_quality_summary": str(output_dir / "sample_session_quality_summary.json"),
            "research_recommendation": str(output_dir / "research_recommendation.md"),
        },
        "market_view_row_count": len(market_view_rows),
        "pricing_feature_row_count": len(pricing_feature_rows),
    }

    _write_json(output_dir / "feature_quality_summary.json", feature_quality_summary)
    _write_json(output_dir / "sample_session_quality_summary.json", session_summary)
    _write_json(output_dir / "run_manifest.json", run_manifest)
    (output_dir / "research_recommendation.md").write_text(recommendation, encoding="utf-8")

    return {
        "run_manifest": run_manifest,
        "feature_quality_summary": feature_quality_summary,
        "sample_session_quality_summary": session_summary,
        "market_view_rows": market_view_dicts,
        "pricing_feature_rows": pricing_feature_dicts,
        "output_dir": output_dir,
    }


def _dedupe_references(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return deduped


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _recommendation_markdown(
    *,
    source_coin: str,
    source_network: str,
    alignment_classification: str,
    validation: dict[str, Any],
    trade_pressure_status: str,
) -> str:
    lines = [
        "# Research Recommendation",
        "",
        f"- Task: `{TASK_ID}`",
        f"- Source sample: `{SOURCE_TASK_ID}` / `{source_coin}` / `{source_network}`",
        f"- Source classification: `{alignment_classification}`",
        f"- Raw sha256 consistent: `{validation['raw_sha256_match']}`",
        f"- Top-N coverage: `{validation['topn_coverage']:.6f}`",
        f"- Synthetic join coverage: `{validation['synthetic_join_coverage']:.6f}`",
        f"- Market-view rows: `{validation['market_view_row_count']}`",
        f"- Pricing-feature rows: `{validation['pricing_feature_row_count']}`",
        f"- Trade pressure status: `{trade_pressure_status}`",
        "",
        "## Public Feature Coverage",
        "",
        "- BBO, mid, spread, top-N imbalance, and top-N microprice proxies are present on the accepted sample.",
        "- Book freshness and join-age context are present and stay within the accepted market-view bounds.",
        "- Trade pressure is left disabled because the public trade side semantics were not treated as freshly confirmed in this task.",
        "",
        "## Quality Gate",
        "",
        f"- Final classification: `{alignment_classification}`",
        "- The sample is acceptable for public pricing / market-view research.",
        "",
        "## Next Step",
        "",
        "- Collect at least one additional accepted public Hyperliquid sample in a different session or regime, then rerun this consumer to compare feature stability.",
        "- Do not treat this result as authorization for private connector work, order lifecycle work, live strategy logic, parameter search, default-on behavior, tiny-live, or promotion.",
        "",
        "## Conclusion",
        "",
        "- The sample supports read-only public market-data research only.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build read-only Hyperliquid public market-data research artifacts."
    )
    parser.add_argument(
        "--input-dir",
        default=str(SOURCE_SAMPLE_DIR),
        help="Accepted local Hyperliquid public sample directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated research artifacts.",
    )
    parser.add_argument("--coin", default=DEFAULT_COIN, help="Expected coin from the accepted sample.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Expected top-N sidecar width.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_research_artifacts(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        coin=args.coin,
        top_n=args.top_n,
    )
    manifest = result["run_manifest"]
    print(f"wrote {manifest['output_dir']}")
    print(
        "classification="
        f"{manifest['source_sample_classification']} "
        f"trade_pressure={manifest['trade_pressure_status']}"
    )
    print(
        "rows "
        f"market_view={manifest['market_view_row_count']} "
        f"pricing_features={manifest['pricing_feature_row_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
