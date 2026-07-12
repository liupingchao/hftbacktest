#!/usr/bin/env python3
"""Offline resting-interval public-flow artifact repair/design for 0712T001.

The runner consumes accepted local T011/0710 artifacts only. It does not call
live, remote, credential, private, account, order, cancel, or market-data APIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0712T001"
SCHEMA_VERSION = "cross_exchange_public_flow_interval_artifact_repair_v1"
CONTRACT_VERSION = "cross_exchange_resting_interval_public_flow_contract_v1"
DEFAULT_QFP_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_quote_fill_probability_evidence_0710T001"
DEFAULT_T011_ROOT = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t011_multi_window_live_evidence_0709T001_20260709T064251Z"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_public_flow_interval_artifact_repair_0712T001"

OFFLINE_SUFFICIENT_ROUTE = "offline_repair_sufficient"
CONTROLLED_CAPTURE_ROUTE = "route_to_controlled_same_envelope_live_evidence_with_resting_interval_public_flow_artifacts"
RECOMMENDATIONS = {OFFLINE_SUFFICIENT_ROUTE, CONTROLLED_CAPTURE_ROUTE}
NOT_RECONSTRUCTABLE = "not_reconstructable_from_current_artifact"

CONTRACT_FIELDS = [
    "window_id",
    "attempt_id",
    "source_kind",
    "source_path",
    "live_classification",
    "order_status_type",
    "reconstruction_status",
    "resting_start_ts",
    "resting_start_ts_source",
    "resting_start_ts_status",
    "cancel_or_shutdown_ts",
    "cancel_or_shutdown_ts_source",
    "cancel_or_shutdown_ts_status",
    "hold_elapsed_seconds",
    "side",
    "quote_px",
    "size_btc",
    "same_side_visible_depth_at_resting_start_btc",
    "same_side_visible_depth_order_count_at_resting_start",
    "depth_multiple_of_order_at_resting_start",
    "depth_reconstruction_status",
    "public_trades_during_resting_interval",
    "public_trades_reconstruction_status",
    "trade_through_count_during_interval",
    "touch_trade_qty_btc",
    "strict_trade_through_qty_btc",
    "at_or_through_trade_qty_btc",
    "depletion_estimate_status",
    "depletion_estimate_qty_btc",
    "queue_depletion_multiple",
    "censoring_status",
    "not_reconstructable_reason",
    "route_signal",
]

PUBLIC_TRADE_FIELDS = [
    "window_id",
    "attempt_id",
    "interval_start_ts",
    "interval_end_ts",
    "interval_start_ms_proxy",
    "interval_end_ms_proxy",
    "source_artifact",
    "public_stream_interval_coverage_status",
    "public_stream_last_exchange_time_ms",
    "rolling_flow_rows_in_interval",
    "trade_event_rows_in_interval",
    "public_trade_count_during_interval",
    "public_trades_during_interval",
    "trade_through_status",
    "not_reconstructable_reason",
]

DEPTH_DEPLETION_FIELDS = [
    "window_id",
    "attempt_id",
    "side",
    "quote_px",
    "size_btc",
    "start_depth_source",
    "same_side_depth_status",
    "same_side_visible_depth_btc",
    "same_side_visible_order_count",
    "top_depth_multiple_of_order",
    "required_depletion_qty_btc",
    "rolling_proxy_public_depletion_status",
    "actual_interval_depletion_status",
    "depletion_estimate_qty_btc",
    "queue_depletion_multiple",
    "not_reconstructable_reason",
]

GAP_FIELDS = [
    "window_id",
    "attempt_id",
    "field_name",
    "reconstruction_status",
    "current_source",
    "reason",
    "required_future_artifact_field",
]


def git_commit() -> str:
    try:
        runner_path = Path(__file__).resolve().relative_to(PROJECT_ROOT)
        value = subprocess.run(
            ["git", "log", "-n", "1", "--format=%h", "--", str(runner_path)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if value:
            return value
    except Exception:
        pass
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_sha256_manifest(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append({"artifact": str(path.relative_to(output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256", "bytes"])


def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        out[value] = out.get(value, 0) + 1
    return out


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", NOT_RECONSTRUCTABLE, "accepted_by_0708T002_QA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.9f}".rstrip("0").rstrip(".")


def fmt_ms_from_seconds(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(round(value * 1000)))


def row_key(row: dict[str, str]) -> tuple[str, str]:
    return row.get("window_id", ""), row.get("attempt_id", "")


def by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {row_key(row): row for row in rows}


def rows_by_attempt(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        attempt = row.get("attempt", "")
        if attempt:
            out.setdefault(attempt, []).append(row)
    return out


def source_path_to_path(source_path: str, t011_root: Path) -> Path:
    path = Path(source_path)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    if source_path.startswith("window_"):
        return t011_root / source_path
    return candidate


def selected_quote_row(window_dir: Path, attempt_id: str) -> dict[str, str]:
    rows = read_csv_rows(window_dir / "quote_attempt_matrix.csv")
    for row in rows:
        if row.get("attempt") == attempt_id and str(row.get("order_endpoint_called", "")).lower() == "true":
            return row
    return {}


def selected_exchange_response_row(window_dir: Path, quote_row: dict[str, str]) -> dict[str, str]:
    event_sequence = quote_row.get("event_sequence", "")
    rows = read_csv_rows(window_dir / "event_driven_latency_matrix.csv")
    matching = [
        row
        for row in rows
        if row.get("attempt") == quote_row.get("attempt")
        and row.get("phase") == "exchange_order_response"
        and (not event_sequence or row.get("event_sequence") == event_sequence)
    ]
    return matching[-1] if matching else {}


def last_exchange_time_ms(rows: list[dict[str, str]]) -> str:
    values = [as_float(row.get("source_event_exchange_time_ms")) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return ""
    return str(int(max(values)))


def interval_rolling_rows(rows: list[dict[str, str]], start_ms: str, end_ms: str) -> list[dict[str, str]]:
    start = as_float(start_ms)
    end = as_float(end_ms)
    if start is None or end is None:
        return []
    out: list[dict[str, str]] = []
    for row in rows:
        ts = as_float(row.get("source_event_exchange_time_ms"))
        if ts is not None and start <= ts <= end:
            out.append(row)
    return out


def trade_px(row: dict[str, str]) -> float | None:
    for key in ("trade_px", "px", "price", "price_usdc"):
        value = as_float(row.get(key))
        if value is not None:
            return value
    return None


def trade_qty(row: dict[str, str]) -> float:
    for key in ("size_btc", "qty_btc", "sz", "qty"):
        value = as_float(row.get(key))
        if value is not None:
            return value
    return 0.0


def public_trade_rows(window_dir: Path, attempt_id: str, start_ms: str, end_ms: str) -> tuple[Path, list[dict[str, str]]]:
    path = window_dir / "resting_interval_public_trades.csv"
    rows = read_csv_rows(path)
    if not rows:
        return path, []
    start = as_float(start_ms)
    end = as_float(end_ms)
    if start is None or end is None:
        return path, []
    filtered: list[dict[str, str]] = []
    for row in rows:
        if row.get("attempt") not in {"", attempt_id}:
            continue
        ts = as_float(row.get("exchange_time_ms"))
        if ts is not None and start <= ts <= end:
            filtered.append(row)
    return path, filtered


def trade_through_quantities(side: str, quote_px: str, rows: list[dict[str, str]]) -> tuple[int, float, float, float]:
    quote = as_float(quote_px)
    if quote is None:
        return 0, 0.0, 0.0, 0.0
    touch = strict = at_or_through = 0.0
    through_count = 0
    for row in rows:
        px = trade_px(row)
        qty = trade_qty(row)
        if px is None or qty <= 0:
            continue
        if side == "buy":
            at_quote = px == quote
            through = px < quote
            at_or = px <= quote
        elif side == "sell":
            at_quote = px == quote
            through = px > quote
            at_or = px >= quote
        else:
            at_quote = through = at_or = False
        if at_quote:
            touch += qty
        if through:
            strict += qty
        if at_or:
            through_count += 1
            at_or_through += qty
    return through_count, touch, strict, at_or_through


def gap_rows_for_attempt(row: dict[str, Any]) -> list[dict[str, str]]:
    base = {"window_id": row["window_id"], "attempt_id": row["attempt_id"]}
    gaps: list[dict[str, str]] = []
    if row["resting_start_ts_status"] != "exact_exchange_resting_timestamp":
        gaps.append(
            {
                **base,
                "field_name": "resting_start_ts",
                "reconstruction_status": row["resting_start_ts_status"],
                "current_source": row["resting_start_ts_source"],
                "reason": "current artifact records local exchange response timing, not exchange-side order-resting timestamp",
                "required_future_artifact_field": "order_resting_exchange_time_ms",
            }
        )
    if row["cancel_or_shutdown_ts_status"] != "exact_cancel_or_shutdown_ack_timestamp":
        gaps.append(
            {
                **base,
                "field_name": "cancel_or_shutdown_ts",
                "reconstruction_status": row["cancel_or_shutdown_ts_status"],
                "current_source": row["cancel_or_shutdown_ts_source"],
                "reason": "current artifact records hold duration/cancel proof without cancel acknowledgement timestamp",
                "required_future_artifact_field": "cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms",
            }
        )
    if row["depth_reconstruction_status"] != "exact_resting_start_l2_depth":
        gaps.append(
            {
                **base,
                "field_name": "same_side_visible_depth_at_resting_start",
                "reconstruction_status": row["depth_reconstruction_status"],
                "current_source": "same_side_depth_proxy_matrix.csv",
                "reason": "current depth is post-open-orders inline-reprice/pre-submit proxy, not a snapshot after exchange resting acknowledgement",
                "required_future_artifact_field": "resting_start_l2_book_snapshot_at_or_after_order_resting",
            }
        )
    if row["public_trades_reconstruction_status"] != "exact_interval_public_trades_present":
        gaps.append(
            {
                **base,
                "field_name": "public_trades_during_resting_interval",
                "reconstruction_status": row["public_trades_reconstruction_status"],
                "current_source": "rolling_flow_state.csv",
                "reason": row["not_reconstructable_reason"],
                "required_future_artifact_field": "resting_interval_public_trades.csv: exchange_time_ms, local_receive_ts_ns, side/aggressor, px, size_btc",
            }
        )
    if row["depletion_estimate_status"] != "visible_depletion_proxy_from_interval_public_trades":
        gaps.append(
            {
                **base,
                "field_name": "depletion_trade_through_estimate",
                "reconstruction_status": row["depletion_estimate_status"],
                "current_source": "current_candidate_audit.csv / rolling_flow_state.csv",
                "reason": "rolling decision-time proxy is not actual resting-interval trade-through/depletion",
                "required_future_artifact_field": "interval trade-through totals plus resting_start/cancel l2 top-depth snapshots",
            }
        )
    return gaps


def build_prior_row(attempt: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reason = "prior_reference_has_no_local_resting_interval_public_flow_artifact"
    contract = {
        "window_id": attempt.get("window_id", ""),
        "attempt_id": attempt.get("attempt_id", ""),
        "source_kind": attempt.get("source_kind", ""),
        "source_path": attempt.get("source_path", ""),
        "live_classification": attempt.get("live_classification", ""),
        "order_status_type": attempt.get("order_status_type", ""),
        "reconstruction_status": NOT_RECONSTRUCTABLE,
        "resting_start_ts": NOT_RECONSTRUCTABLE,
        "resting_start_ts_source": "missing",
        "resting_start_ts_status": NOT_RECONSTRUCTABLE,
        "cancel_or_shutdown_ts": NOT_RECONSTRUCTABLE,
        "cancel_or_shutdown_ts_source": "missing",
        "cancel_or_shutdown_ts_status": NOT_RECONSTRUCTABLE,
        "hold_elapsed_seconds": attempt.get("hold_elapsed_seconds", ""),
        "side": attempt.get("side", ""),
        "quote_px": attempt.get("limit_px", ""),
        "size_btc": attempt.get("size_btc", ""),
        "same_side_visible_depth_at_resting_start_btc": NOT_RECONSTRUCTABLE,
        "same_side_visible_depth_order_count_at_resting_start": NOT_RECONSTRUCTABLE,
        "depth_multiple_of_order_at_resting_start": NOT_RECONSTRUCTABLE,
        "depth_reconstruction_status": NOT_RECONSTRUCTABLE,
        "public_trades_during_resting_interval": NOT_RECONSTRUCTABLE,
        "public_trades_reconstruction_status": NOT_RECONSTRUCTABLE,
        "trade_through_count_during_interval": NOT_RECONSTRUCTABLE,
        "touch_trade_qty_btc": NOT_RECONSTRUCTABLE,
        "strict_trade_through_qty_btc": NOT_RECONSTRUCTABLE,
        "at_or_through_trade_qty_btc": NOT_RECONSTRUCTABLE,
        "depletion_estimate_status": NOT_RECONSTRUCTABLE,
        "depletion_estimate_qty_btc": NOT_RECONSTRUCTABLE,
        "queue_depletion_multiple": NOT_RECONSTRUCTABLE,
        "censoring_status": attempt.get("censoring_status", ""),
        "not_reconstructable_reason": reason,
        "route_signal": "requires_new_interval_public_flow_artifact",
    }
    public_trade = {
        "window_id": contract["window_id"],
        "attempt_id": contract["attempt_id"],
        "public_trades_during_interval": NOT_RECONSTRUCTABLE,
        "trade_through_status": NOT_RECONSTRUCTABLE,
        "not_reconstructable_reason": reason,
    }
    depth = {
        "window_id": contract["window_id"],
        "attempt_id": contract["attempt_id"],
        "same_side_depth_status": NOT_RECONSTRUCTABLE,
        "actual_interval_depletion_status": NOT_RECONSTRUCTABLE,
        "not_reconstructable_reason": reason,
    }
    return contract, public_trade, depth


def build_live_row(
    *,
    attempt: dict[str, str],
    depth_proxy: dict[str, str],
    rolling_proxy: dict[str, str],
    horizon: dict[str, str],
    t011_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    window_dir = source_path_to_path(attempt.get("source_path", ""), t011_root)
    quote = selected_quote_row(window_dir, attempt.get("attempt_id", ""))
    response = selected_exchange_response_row(window_dir, quote)
    start_ts = as_float(response.get("end_unix_seconds"))
    hold_seconds = as_float(horizon.get("hold_elapsed_seconds"))
    end_ts = start_ts + hold_seconds if start_ts is not None and hold_seconds is not None else None
    start_ms = fmt_ms_from_seconds(start_ts)
    end_ms = fmt_ms_from_seconds(end_ts)
    rolling_rows = read_csv_rows(window_dir / "rolling_flow_state.csv")
    rolling_interval_rows = interval_rolling_rows(rolling_rows, start_ms, end_ms)
    trade_event_rows = [row for row in rolling_interval_rows if row.get("source_channel") == "trades"]
    interval_trade_path, interval_trade_rows = public_trade_rows(window_dir, attempt.get("attempt_id", ""), start_ms, end_ms)
    last_public_ms = last_exchange_time_ms(rolling_rows)
    side = attempt.get("side", "")
    quote_px = attempt.get("limit_px", "")
    size_btc = attempt.get("size_btc", "")
    same_side_depth = depth_proxy.get("same_side_top_qty_btc", "")
    top_depth_multiple = depth_proxy.get("top_depth_multiple_of_order", "")
    same_side_order_count = depth_proxy.get("same_side_top_order_count", "")
    through_count, touch_qty, strict_qty, at_or_through_qty = trade_through_quantities(side, quote_px, interval_trade_rows)
    required_depletion = (as_float(same_side_depth, 0.0) or 0.0) + (as_float(size_btc, 0.0) or 0.0)
    queue_depletion_multiple = at_or_through_qty / required_depletion if interval_trade_rows and required_depletion > 0 else None

    if interval_trade_rows:
        public_trade_status = "exact_interval_public_trades_present"
        coverage_status = "interval_public_trade_rows_present"
        public_trades_value = str(len(interval_trade_rows))
        trade_through_status = "interval_trade_through_present" if through_count else "interval_public_trades_present_no_trade_through"
        depletion_status = "visible_depletion_proxy_from_interval_public_trades"
        depletion_qty = fmt_float(at_or_through_qty)
        route_signal = "offline_repair_possible"
        reason = ""
        reconstruction_status = "resting_interval_reconstructed_from_contract_artifacts"
    else:
        public_trade_status = NOT_RECONSTRUCTABLE
        public_trades_value = NOT_RECONSTRUCTABLE
        if last_public_ms and start_ms and as_float(last_public_ms) is not None and as_float(start_ms) is not None and as_float(last_public_ms) < as_float(start_ms):
            coverage_status = "public_stream_ended_before_resting_interval"
            reason = "rolling/public state artifacts end before or at the order-response proxy; no individual public trade events cover the resting interval"
        elif rolling_interval_rows:
            coverage_status = "rolling_aggregate_rows_present_no_individual_public_trade_events"
            reason = "only rolling aggregate rows are present during the interval; individual public trades cannot be reconstructed"
        else:
            coverage_status = "no_public_flow_rows_cover_resting_interval"
            reason = "current artifacts contain no public-flow rows bounded by the resting interval"
        trade_through_status = NOT_RECONSTRUCTABLE
        depletion_status = NOT_RECONSTRUCTABLE
        depletion_qty = NOT_RECONSTRUCTABLE
        route_signal = "requires_new_interval_public_flow_artifact"
        reconstruction_status = "partial_proxy_only"

    resting_status = "local_exchange_response_end_proxy_not_exact_exchange_resting_timestamp" if start_ts is not None else NOT_RECONSTRUCTABLE
    cancel_status = "derived_from_response_end_plus_hold_elapsed_not_exact_cancel_ack" if end_ts is not None else NOT_RECONSTRUCTABLE
    depth_status = "pre_submit_reprice_l2_proxy_not_exact_resting_start_l2" if same_side_depth else NOT_RECONSTRUCTABLE

    contract = {
        "window_id": attempt.get("window_id", ""),
        "attempt_id": attempt.get("attempt_id", ""),
        "source_kind": attempt.get("source_kind", ""),
        "source_path": display_path(window_dir),
        "live_classification": attempt.get("live_classification", ""),
        "order_status_type": attempt.get("order_status_type", ""),
        "reconstruction_status": reconstruction_status,
        "resting_start_ts": fmt_float(start_ts) if start_ts is not None else NOT_RECONSTRUCTABLE,
        "resting_start_ts_source": "event_driven_latency_matrix.exchange_order_response.end_unix_seconds" if start_ts is not None else "missing",
        "resting_start_ts_status": resting_status,
        "cancel_or_shutdown_ts": fmt_float(end_ts) if end_ts is not None else NOT_RECONSTRUCTABLE,
        "cancel_or_shutdown_ts_source": "resting_start_ts_proxy_plus_quote_aging_guard_matrix.hold_elapsed_seconds" if end_ts is not None else "missing",
        "cancel_or_shutdown_ts_status": cancel_status,
        "hold_elapsed_seconds": horizon.get("hold_elapsed_seconds", ""),
        "side": side,
        "quote_px": quote_px,
        "size_btc": size_btc,
        "same_side_visible_depth_at_resting_start_btc": same_side_depth or NOT_RECONSTRUCTABLE,
        "same_side_visible_depth_order_count_at_resting_start": same_side_order_count or NOT_RECONSTRUCTABLE,
        "depth_multiple_of_order_at_resting_start": top_depth_multiple or NOT_RECONSTRUCTABLE,
        "depth_reconstruction_status": depth_status,
        "public_trades_during_resting_interval": public_trades_value,
        "public_trades_reconstruction_status": public_trade_status,
        "trade_through_count_during_interval": str(through_count) if interval_trade_rows else NOT_RECONSTRUCTABLE,
        "touch_trade_qty_btc": fmt_float(touch_qty) if interval_trade_rows else NOT_RECONSTRUCTABLE,
        "strict_trade_through_qty_btc": fmt_float(strict_qty) if interval_trade_rows else NOT_RECONSTRUCTABLE,
        "at_or_through_trade_qty_btc": fmt_float(at_or_through_qty) if interval_trade_rows else NOT_RECONSTRUCTABLE,
        "depletion_estimate_status": depletion_status,
        "depletion_estimate_qty_btc": depletion_qty,
        "queue_depletion_multiple": fmt_float(queue_depletion_multiple) if interval_trade_rows else NOT_RECONSTRUCTABLE,
        "censoring_status": attempt.get("censoring_status", ""),
        "not_reconstructable_reason": reason,
        "route_signal": route_signal,
    }
    public_trade = {
        "window_id": contract["window_id"],
        "attempt_id": contract["attempt_id"],
        "interval_start_ts": contract["resting_start_ts"],
        "interval_end_ts": contract["cancel_or_shutdown_ts"],
        "interval_start_ms_proxy": start_ms,
        "interval_end_ms_proxy": end_ms,
        "source_artifact": display_path(interval_trade_path) if interval_trade_path.exists() else display_path(window_dir / "rolling_flow_state.csv"),
        "public_stream_interval_coverage_status": coverage_status,
        "public_stream_last_exchange_time_ms": last_public_ms,
        "rolling_flow_rows_in_interval": str(len(rolling_interval_rows)),
        "trade_event_rows_in_interval": str(len(trade_event_rows)),
        "public_trade_count_during_interval": str(len(interval_trade_rows)) if interval_trade_rows else NOT_RECONSTRUCTABLE,
        "public_trades_during_interval": public_trades_value,
        "trade_through_status": trade_through_status,
        "not_reconstructable_reason": reason,
    }
    depth = {
        "window_id": contract["window_id"],
        "attempt_id": contract["attempt_id"],
        "side": side,
        "quote_px": quote_px,
        "size_btc": size_btc,
        "start_depth_source": "same_side_depth_proxy_matrix.csv/post_open_orders_inline_reprice" if same_side_depth else "missing",
        "same_side_depth_status": depth_status,
        "same_side_visible_depth_btc": same_side_depth or NOT_RECONSTRUCTABLE,
        "same_side_visible_order_count": same_side_order_count or NOT_RECONSTRUCTABLE,
        "top_depth_multiple_of_order": top_depth_multiple or NOT_RECONSTRUCTABLE,
        "required_depletion_qty_btc": fmt_float(required_depletion) if same_side_depth and size_btc else NOT_RECONSTRUCTABLE,
        "rolling_proxy_public_depletion_status": rolling_proxy.get("public_depletion_status", ""),
        "actual_interval_depletion_status": depletion_status,
        "depletion_estimate_qty_btc": depletion_qty,
        "queue_depletion_multiple": contract["queue_depletion_multiple"],
        "not_reconstructable_reason": reason if depletion_status == NOT_RECONSTRUCTABLE else "",
    }
    return contract, public_trade, depth


def contract_definition() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "purpose": "Capture enough public-flow evidence to reconstruct each accepted resting order from resting acknowledgement through cancel or shutdown.",
        "required_attempt_keys": ["window_id", "attempt_id", "side", "quote_px", "size_btc"],
        "required_lifecycle_fields": [
            "order_resting_exchange_time_ms",
            "order_resting_local_receive_ts_ns",
            "cancel_request_time_ms",
            "cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms",
        ],
        "required_public_trade_fields": [
            "attempt",
            "exchange_time_ms",
            "local_receive_ts_ns",
            "side_or_aggressor",
            "px",
            "size_btc",
            "raw_event_sequence",
        ],
        "required_depth_fields": [
            "attempt",
            "snapshot_role",
            "exchange_time_ms",
            "local_receive_ts_ns",
            "bid_px",
            "ask_px",
            "same_side_levels_at_or_ahead_of_quote",
            "same_side_visible_qty_at_or_ahead_of_quote_btc",
            "same_side_visible_order_count_at_or_ahead_of_quote",
        ],
        "generated_matrices": {
            "resting_interval_contract_matrix.csv": CONTRACT_FIELDS,
            "resting_interval_public_trades_matrix.csv": PUBLIC_TRADE_FIELDS,
            "resting_interval_depth_depletion_matrix.csv": DEPTH_DEPLETION_FIELDS,
            "artifact_gap_matrix.csv": GAP_FIELDS,
        },
        "status_values": {
            "not_reconstructable": NOT_RECONSTRUCTABLE,
            "timestamp_proxy": "local_exchange_response_end_proxy_not_exact_exchange_resting_timestamp",
            "depth_proxy": "pre_submit_reprice_l2_proxy_not_exact_resting_start_l2",
            "offline_sufficient_route": OFFLINE_SUFFICIENT_ROUTE,
            "controlled_capture_route": CONTROLLED_CAPTURE_ROUTE,
        },
        "forbidden_claims": ["fill_probability", "queue_priority", "fee", "rebate", "realized_pnl", "maker_viability", "t012", "promotion", "final_mvp_pass"],
    }


def choose_route(contract_rows: list[dict[str, Any]]) -> str:
    if contract_rows and all(row.get("public_trades_reconstruction_status") == "exact_interval_public_trades_present" for row in contract_rows):
        return OFFLINE_SUFFICIENT_ROUTE
    return CONTROLLED_CAPTURE_ROUTE


def run_analysis(*, qfp_dir: Path, t011_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    qfp_manifest = read_json(qfp_dir / "quote_fill_probability_manifest.json")
    attempts = read_csv_rows(qfp_dir / "attempt_level_fill_probability_matrix.csv")
    depth_by_key = by_key(read_csv_rows(qfp_dir / "same_side_depth_proxy_matrix.csv"))
    trade_by_key = by_key(read_csv_rows(qfp_dir / "trade_through_depletion_matrix.csv"))
    horizon_by_key = by_key(read_csv_rows(qfp_dir / "censoring_and_horizon_matrix.csv"))
    resting_attempts = [
        row
        for row in attempts
        if row.get("order_status_type") == "resting" and row.get("no_fill_state") == "resting_no_fill_observed"
    ]

    contract_rows: list[dict[str, Any]] = []
    public_trade_rows_out: list[dict[str, Any]] = []
    depth_rows_out: list[dict[str, Any]] = []
    gap_rows: list[dict[str, str]] = []

    for attempt in resting_attempts:
        key = row_key(attempt)
        if attempt.get("source_kind") == "prior_accepted_replay_reference":
            contract, public_trade, depth = build_prior_row(attempt)
        else:
            contract, public_trade, depth = build_live_row(
                attempt=attempt,
                depth_proxy=depth_by_key.get(key, {}),
                rolling_proxy=trade_by_key.get(key, {}),
                horizon=horizon_by_key.get(key, {}),
                t011_root=t011_root,
            )
        contract_rows.append(contract)
        public_trade_rows_out.append(public_trade)
        depth_rows_out.append(depth)
        gap_rows.extend(gap_rows_for_attempt(contract))

    final_route = choose_route(contract_rows)
    if final_route not in RECOMMENDATIONS:
        raise ValueError(f"invalid_final_route:{final_route}")

    write_csv(output_dir / "resting_interval_contract_matrix.csv", contract_rows, CONTRACT_FIELDS)
    write_csv(output_dir / "resting_interval_public_trades_matrix.csv", public_trade_rows_out, PUBLIC_TRADE_FIELDS)
    write_csv(output_dir / "resting_interval_depth_depletion_matrix.csv", depth_rows_out, DEPTH_DEPLETION_FIELDS)
    write_csv(output_dir / "artifact_gap_matrix.csv", gap_rows, GAP_FIELDS)
    write_json(output_dir / "resting_interval_public_flow_artifact_contract.json", contract_definition())

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "git_commit": git_commit(),
        "qfp_dir": display_path(qfp_dir),
        "t011_root": display_path(t011_root),
        "qfp_source_task_id": qfp_manifest.get("task_id"),
        "qfp_final_recommendation": qfp_manifest.get("final_recommendation"),
        "accepted_resting_no_fill_attempt_count": len(resting_attempts),
        "prior_reference_count": counts(contract_rows, "source_kind").get("prior_accepted_replay_reference", 0),
        "live_resting_no_fill_attempt_count": counts(contract_rows, "source_kind").get("t011_live_window_artifact", 0),
        "reconstruction_status_counts": counts(contract_rows, "reconstruction_status"),
        "resting_start_ts_status_counts": counts(contract_rows, "resting_start_ts_status"),
        "cancel_or_shutdown_ts_status_counts": counts(contract_rows, "cancel_or_shutdown_ts_status"),
        "depth_reconstruction_status_counts": counts(contract_rows, "depth_reconstruction_status"),
        "public_trades_reconstruction_status_counts": counts(contract_rows, "public_trades_reconstruction_status"),
        "depletion_estimate_status_counts": counts(contract_rows, "depletion_estimate_status"),
        "route_signal_counts": counts(contract_rows, "route_signal"),
        "gap_field_counts": counts(gap_rows, "field_name"),
        "offline_repair_sufficient": final_route == OFFLINE_SUFFICIENT_ROUTE,
        "final_route": final_route,
        "route_rationale": "Current accepted artifacts bind lifecycle/depth proxies for live resting attempts but do not contain individual public trades or exact L2 depth over the actual resting interval; a later separately authorized same-envelope evidence collection should capture the interval public-flow contract before any fill-probability claim.",
        "does_not_claim": ["fill_probability", "synthetic_fill", "queue_priority", "fee", "rebate", "realized_pnl", "maker_viability", "promotion", "t012", "final_mvp_pass"],
    }
    write_json(output_dir / "public_flow_interval_repair_manifest.json", manifest)

    boundary = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "boundary_status": "pass",
        "offline_only": True,
        "network_called": False,
        "remote_called": False,
        "aws_called": False,
        "credentials_read": False,
        "private_endpoint_called": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "live_submit_executed": False,
        "market_data_collected": False,
        "threshold_changed": False,
        "quote_envelope_changed": False,
        "order_size_changed": False,
        "max_submissions_changed": False,
        "strategy_changed": False,
        "fill_probability_claim": False,
        "synthetic_fill_claim": False,
        "fee_claim": False,
        "rebate_claim": False,
        "realized_pnl_claim": False,
        "queue_priority_claim": False,
        "maker_viability_claim": False,
        "promotion_authorized": False,
        "t012_claim": False,
        "final_mvp_claim": False,
    }
    write_json(output_dir / "boundary_manifest.json", boundary)

    report = [
        "# 0712T001 Public-Flow Interval Artifact Repair",
        "",
        f"Final route: `{final_route}`",
        "",
        f"- Accepted resting/no-fill attempts: `{len(resting_attempts)}`",
        f"- Reconstruction statuses: `{manifest['reconstruction_status_counts']}`",
        f"- Public-trades statuses: `{manifest['public_trades_reconstruction_status_counts']}`",
        f"- Depletion statuses: `{manifest['depletion_estimate_status_counts']}`",
        f"- Gap fields: `{manifest['gap_field_counts']}`",
        "",
        "Current artifacts can bind live resting attempts to response-time and hold-duration proxies plus pre-submit depth proxy, but they do not reconstruct individual public trades or actual same-side depletion during the resting interval.",
        "",
        "No fill probability, queue priority, fee/rebate, realized PnL, maker viability, T012, promotion, or final MVP claim is made.",
        "",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    build_sha256_manifest(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qfp-dir", type=Path, default=DEFAULT_QFP_DIR)
    parser.add_argument("--t011-root", type=Path, default=DEFAULT_T011_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_analysis(qfp_dir=args.qfp_dir, t011_root=args.t011_root, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
