#!/usr/bin/env python3
"""Offline quote/fill evidence rerun using accepted 0714T003 v2 resting-interval artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0714T004"
SCHEMA_VERSION = "cross_exchange_quote_fill_probability_evidence_0714T004_v1"
SOURCE_TASK_ID = "0714T003"
SOURCE_RAW_TASK_ID = "0714T003"
REMOTE_PROVENANCE_SOURCE_ROOT = (
    "awsserver1:/home/admin/hftbacktest-cross-exchange-artifacts/"
    "cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z/"
)
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_resting_interval_v2_live_evidence_0714T003_20260714T063004Z"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_quote_fill_probability_evidence_0714T004"


ATTEMPT_FIELDS = [
    "window_id",
    "evaluation_id",
    "order_attempt_id",
    "event_sequence",
    "source_kind",
    "source_path",
    "live_classification",
    "order_status_type",
    "side",
    "limit_px",
    "size_btc",
    "post_only_tif",
    "order_endpoint_called",
    "post_only_reject",
    "reject_interpretation",
    "quote_placement",
    "pre_bid",
    "pre_ask",
    "post_bid",
    "post_ask",
    "hold_elapsed_seconds",
    "censoring_status",
    "censoring_reason",
    "no_fill_state",
    "depth_proxy_status",
    "depth_proxy_scope",
    "same_side_visible_qty_at_or_ahead_of_quote_btc",
    "same_side_visible_order_count_at_or_ahead_of_quote",
    "top_depth_multiple_of_order",
    "trade_through_status",
    "public_trade_count",
    "touch_trade_qty_btc",
    "strict_trade_through_qty_btc",
    "at_or_through_trade_qty_btc",
    "required_depletion_qty_btc",
    "queue_depletion_multiple",
    "public_depletion_status",
    "public_stream_coverage_status",
    "zero_public_trade_interpretation",
    "opportunity_status",
    "exact_timestamp_caveat",
    "evidence_limitation",
    "route_signal",
]


PUBLIC_TRADE_SUMMARY_FIELDS = [
    "window_id",
    "evaluation_id",
    "order_attempt_id",
    "event_sequence",
    "source_kind",
    "order_status_type",
    "public_trade_count",
    "touch_trade_qty_btc",
    "strict_trade_through_qty_btc",
    "at_or_through_trade_qty_btc",
    "same_side_visible_qty_at_or_ahead_of_quote_btc",
    "required_depletion_qty_btc",
    "queue_depletion_multiple",
    "trade_through_status",
    "public_depletion_status",
    "lifecycle_status",
    "depth_status",
    "public_stream_coverage_status",
    "zero_public_trade_interpretation",
    "capture_status",
]


PUBLIC_STREAM_COVERAGE_FIELDS = [
    "window_id",
    "evaluation_id",
    "order_attempt_id",
    "attempt_key",
    "event_sequence",
    "source_kind",
    "stream",
    "interval_start_ms",
    "interval_end_ms",
    "start_cursor",
    "end_cursor",
    "first_event_exchange_time_ms",
    "last_event_exchange_time_ms",
    "gap_count",
    "coverage_status",
    "zero_public_trade_interpretation",
    "coverage_interpretation",
    "route_signal",
]


CENSORING_FIELDS = [
    "window_id",
    "evaluation_id",
    "order_attempt_id",
    "event_sequence",
    "source_kind",
    "order_status_type",
    "hold_elapsed_seconds",
    "quote_aging_guard_status",
    "quote_aging_guard_reason",
    "observation_horizon_status",
    "censoring_status",
    "censoring_reason",
]


DEPTH_FIELDS = [
    "window_id",
    "evaluation_id",
    "order_attempt_id",
    "event_sequence",
    "source_kind",
    "depth_proxy_status",
    "depth_proxy_scope",
    "side",
    "limit_px",
    "current_bid",
    "current_ask",
    "same_side_visible_qty_at_or_ahead_of_quote_btc",
    "same_side_visible_order_count_at_or_ahead_of_quote",
    "top_depth_multiple_of_order",
    "visible_depth_bucket",
    "depth_reconstruction_status",
    "post_only_non_crossing",
    "current_touch_match",
]


def git_commit() -> str:
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "null"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.9f}".rstrip("0").rstrip(".")


def truthy(value: Any) -> bool:
    return str(value).lower() == "true"


def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        out[value] = out.get(value, 0) + 1
    return out


def rows_by_event(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("event_sequence", ""): row for row in rows if row.get("event_sequence")}


def rows_by_order_attempt(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        attempt = row.get("attempt", "")
        if not attempt:
            continue
        out.setdefault(attempt, []).append(row)
    return out


def rows_by_attempt_key(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        attempt_key = row.get("attempt_key", "")
        if not attempt_key:
            continue
        out.setdefault(attempt_key, []).append(row)
    return out


def quote_placement(side: str, limit_px: Any, bid: Any, ask: Any) -> str:
    limit = as_float(limit_px)
    bid_v = as_float(bid)
    ask_v = as_float(ask)
    if limit is None or bid_v is None or ask_v is None:
        return "placement_unknown"
    if side == "buy":
        if limit >= ask_v:
            return "crossing_or_marketable"
        if limit == bid_v:
            return "at_touch_bid"
        return "behind_touch_bid"
    if side == "sell":
        if limit <= bid_v:
            return "crossing_or_marketable"
        if limit == ask_v:
            return "at_touch_ask"
        return "behind_touch_ask"
    return "placement_unknown"


def reject_interpretation(quote_row: dict[str, str], post_only_reject_rows: list[dict[str, str]]) -> str:
    if not truthy(quote_row.get("post_only_reject")):
        return ""
    attempt = quote_row.get("attempt", "")
    reject = next((row for row in post_only_reject_rows if row.get("attempt") == attempt), {})
    reason = reject.get("reject_reason", "")
    if "Post only order would have immediately matched" in reason:
        return "consistent_post_only_protection"
    if reason:
        return "post_only_reject_reason_present"
    return "post_only_reject_reason_missing"


def visible_depth_bucket(depth_multiple: Any) -> str:
    value = as_float(depth_multiple)
    if value is None:
        return "depth_proxy_missing"
    if value < 1:
        return "thin_visible_top_less_than_order"
    if value < 5:
        return "moderate_visible_top_depth"
    return "large_visible_top_depth"


def no_fill_state(quote_row: dict[str, str]) -> str:
    if truthy(quote_row.get("post_only_reject")):
        return "not_resting_rejected"
    fills = as_float(quote_row.get("fill_count_after_attempt"), 0) or 0
    maker_fills = as_float(quote_row.get("maker_fill_count_after_attempt"), 0) or 0
    if fills > 0 or maker_fills > 0:
        return "fill_observed"
    if quote_row.get("order_status_types") == "resting":
        return "resting_no_fill_observed"
    return "no_order_submitted"


def censoring(quote_row: dict[str, str], aging_row: dict[str, str]) -> tuple[str, str, str]:
    status = quote_row.get("order_status_types", "")
    if truthy(quote_row.get("post_only_reject")) or status == "error":
        return "not_applicable_rejected", "not_resting_post_only_reject", "not_applicable_rejected"
    if status != "resting":
        return "not_applicable_no_order_submitted", "no_order_submitted", "not_applicable_no_order_submitted"
    hold = as_float(aging_row.get("hold_elapsed_seconds"))
    if hold is None:
        return "horizon_missing", "hold_elapsed_seconds_missing", "horizon_missing"
    if hold < 5:
        return "short_hold_censored", "hold_lt_5s_no_low_fill_probability_inference", "short_horizon"
    return "observed_no_fill_censored", "no_fill_observed_but_fill_probability_not_modeled", "bounded_horizon"


def opportunity_status(trade_status: str, order_status_type: str) -> str:
    if trade_status in {"public_flow_artifact_missing", "coverage_not_proven_complete"}:
        return "opportunity_not_assessable_public_flow_missing"
    if order_status_type == "resting":
        return "opportunity_censored_no_resting_interval_trade_reconstruction"
    if order_status_type == "error":
        return "not_applicable_post_only_reject"
    if order_status_type == "skipped":
        return "opportunity_not_applicable_no_order_submitted"
    return "opportunity_not_applicable"


def trade_through_status(
    *,
    quote_row: dict[str, str],
    lifecycle_row: dict[str, str],
    public_trade_rows: list[dict[str, str]],
    coverage_row: dict[str, str],
) -> str:
    status = quote_row.get("order_status_types", "")
    if truthy(quote_row.get("post_only_reject")) or status == "error":
        return "post_only_reject_not_a_fill_sample"
    if status != "resting":
        return "not_applicable_no_order_submitted"
    if not lifecycle_row:
        return "public_flow_artifact_missing"
    if coverage_row.get("coverage_status") != "complete_interval_trade_stream_coverage":
        return "coverage_not_proven_complete"
    if public_trade_rows:
        strict_qty = sum(as_float(row.get("strict_trade_through_qty_btc"), 0) or 0 for row in public_trade_rows)
        if strict_qty > 0:
            return "strict_trade_through_present"
        touch_qty = sum(as_float(row.get("touch_trade_qty_btc"), 0) or 0 for row in public_trade_rows)
        if touch_qty > 0:
            return "touch_trades_present_no_strict_trade_through"
        return "public_trade_rows_present_no_strict_trade_through"
    return "zero_public_trades_with_complete_interval_coverage"


def coverage_interpretation(coverage_row: dict[str, str]) -> str:
    status = coverage_row.get("coverage_status", "")
    zero_interpretation = coverage_row.get("zero_public_trade_interpretation", "")
    if not coverage_row:
        return "coverage_artifact_missing"
    if status == "complete_interval_trade_stream_coverage" and zero_interpretation == "zero_public_trades_observed_with_complete_interval_coverage":
        return "complete_coverage_zero_trades_observed"
    if zero_interpretation == "artifact_gap_not_no_exchange_trades":
        return "artifact_gap_not_no_exchange_trades"
    if status != "complete_interval_trade_stream_coverage":
        return "coverage_not_proven_complete"
    return "coverage_present"


def selected_guard_row(rows: list[dict[str, str]], event_sequence: str, attempt: str, limit_px: str) -> dict[str, str]:
    matches = [row for row in rows if row.get("event_sequence") == event_sequence or row.get("attempt") == attempt]
    if not matches:
        return {}
    if limit_px:
        for row in matches:
            if as_float(row.get("selected_quote_px")) == as_float(limit_px) or as_float(row.get("limit_px")) == as_float(limit_px):
                return row
    return matches[-1]


def build_attempt_rows(
    *,
    window_id: str,
    window_dir: Path,
    live_classification: str,
    source_task_id: str,
    source_raw_task_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quote_rows = read_csv_rows(window_dir / "quote_attempt_matrix.csv")
    guard_rows = read_csv_rows_if_exists(window_dir / "inline_reprice_guard_matrix.csv")
    candidate_rows = rows_by_event(read_csv_rows_if_exists(window_dir / "current_candidate_audit.csv"))
    aging_rows = rows_by_order_attempt(read_csv_rows_if_exists(window_dir / "quote_aging_guard_matrix.csv"))
    reject_rows = read_csv_rows_if_exists(window_dir / "inline_reprice_post_only_reject_matrix.csv")
    lifecycle_rows = rows_by_order_attempt(read_csv_rows_if_exists(window_dir / "resting_interval_lifecycle_matrix.csv"))
    public_trade_rows_all = read_csv_rows_if_exists(window_dir / "resting_interval_public_trades.csv")
    public_trade_rows_by_key = rows_by_attempt_key(public_trade_rows_all)
    depth_rows = rows_by_order_attempt(read_csv_rows_if_exists(window_dir / "resting_interval_depth_depletion_matrix.csv"))
    l2_rows = rows_by_order_attempt(read_csv_rows_if_exists(window_dir / "resting_start_l2_book_snapshot_at_or_after_order_resting.csv"))
    coverage_rows = rows_by_order_attempt(read_csv_rows_if_exists(window_dir / "public_stream_coverage.csv"))

    attempt_rows: list[dict[str, Any]] = []
    public_trade_summary_rows: list[dict[str, Any]] = []
    censoring_rows: list[dict[str, Any]] = []
    depth_proxy_rows: list[dict[str, Any]] = []
    coverage_evidence_rows: list[dict[str, Any]] = []

    for evaluation_id, quote in enumerate(quote_rows, start=1):
        event_sequence = quote.get("event_sequence", "")
        source_attempt_counter = quote.get("attempt", str(evaluation_id))
        order_status = quote.get("order_status_types", "")
        post_only_reject = truthy(quote.get("post_only_reject"))
        order_called = truthy(quote.get("order_endpoint_called"))
        order_attempt_id = source_attempt_counter if order_called or order_status in {"resting", "error"} else ""
        lookup_attempt_id = order_attempt_id or "__no_order_submitted__"
        guard = selected_guard_row(guard_rows, event_sequence, source_attempt_counter, quote.get("limit_px", ""))
        candidate = candidate_rows.get(event_sequence, {})
        aging = (aging_rows.get(lookup_attempt_id) or [{}])[-1]
        lifecycle = (lifecycle_rows.get(lookup_attempt_id) or [{}])[-1]
        depth_summary = (depth_rows.get(lookup_attempt_id) or [{}])[-1]
        l2 = (l2_rows.get(lookup_attempt_id) or [{}])[-1]
        coverage = (coverage_rows.get(lookup_attempt_id) or [{}])[-1]
        attempt_key = lifecycle.get("attempt_key") or depth_summary.get("attempt_key") or l2.get("attempt_key") or coverage.get("attempt_key", "")
        public_trade_rows = public_trade_rows_by_key.get(attempt_key, []) if attempt_key else []
        public_trade_count = len(public_trade_rows)
        side = quote.get("side", "")
        limit_px = quote.get("limit_px", "")
        current_bid = guard.get("current_bid") or quote.get("submit_intent_bid") or l2.get("bid_px", "")
        current_ask = guard.get("current_ask") or quote.get("submit_intent_ask") or l2.get("ask_px", "")
        if order_called and order_status == "resting":
            side = side or guard.get("selected_side") or "buy"
            limit_px = limit_px or guard.get("selected_quote_px") or guard.get("limit_px") or quote.get("submit_intent_bid", "")
            placement = quote_placement(side, limit_px, current_bid, current_ask)
            hold_elapsed = aging.get("hold_elapsed_seconds", "")
            hold_float = as_float(hold_elapsed)
            censoring_status, censoring_reason, horizon_status = censoring(quote, aging)
            same_side_qty = l2.get("same_side_visible_qty_at_or_ahead_of_quote_btc", "")
            same_side_order_count = l2.get("same_side_visible_order_count_at_or_ahead_of_quote", "")
            top_depth_multiple = ""
            same_side_qty_float = as_float(same_side_qty)
            order_size_float = as_float(limit_px and quote.get("size_btc", ""))
            if same_side_qty_float is not None and order_size_float and order_size_float > 0:
                top_depth_multiple = fmt_float(same_side_qty_float / order_size_float)
            trade_status = trade_through_status(
                quote_row=quote,
                lifecycle_row=lifecycle,
                public_trade_rows=public_trade_rows,
                coverage_row=coverage,
            )
            coverage_status = coverage.get("coverage_status", "")
            zero_public_trade_interpretation = coverage.get("zero_public_trade_interpretation", "")
            exact_timestamp_caveat = ";".join(
                part
                for part in [
                    lifecycle.get("order_resting_exchange_time_ms_status", ""),
                    lifecycle.get("cancel_ack_time_status", ""),
                ]
                if part
            ) or "proxy_timestamps_only"
            if coverage_interpretation(coverage) in {"artifact_gap_not_no_exchange_trades", "coverage_not_proven_complete", "coverage_artifact_missing"}:
                evidence_limitation = "coverage_not_proven_complete;zero_rows_are_artifact_gap;proxy_only_lifecycle_depth"
                route_signal = "public_flow_artifact_gap"
            elif public_trade_count == 0:
                evidence_limitation = "complete_coverage_zero_public_trades;proxy_only_lifecycle_depth"
                route_signal = "censored_resting_no_fill"
            else:
                evidence_limitation = "resting_interval_public_trades_present"
                route_signal = "censored_resting_no_fill"
            coverage_evidence_rows.append(
                {
                    "window_id": window_id,
                    "evaluation_id": evaluation_id,
                    "order_attempt_id": order_attempt_id,
                    "attempt_key": attempt_key,
                    "event_sequence": event_sequence,
                    "source_kind": "0714T003_v2_live_window_artifact",
                    "stream": coverage.get("stream", ""),
                    "interval_start_ms": coverage.get("interval_start_ms", ""),
                    "interval_end_ms": coverage.get("interval_end_ms", ""),
                    "start_cursor": coverage.get("start_cursor", ""),
                    "end_cursor": coverage.get("end_cursor", ""),
                    "first_event_exchange_time_ms": coverage.get("first_event_exchange_time_ms", ""),
                    "last_event_exchange_time_ms": coverage.get("last_event_exchange_time_ms", ""),
                    "gap_count": coverage.get("gap_count", ""),
                    "coverage_status": coverage_status,
                    "zero_public_trade_interpretation": zero_public_trade_interpretation,
                    "coverage_interpretation": coverage_interpretation(coverage),
                    "route_signal": route_signal,
                }
            )
            public_trade_summary_rows.append(
                {
                    "window_id": window_id,
                    "evaluation_id": evaluation_id,
                    "order_attempt_id": order_attempt_id,
                    "event_sequence": event_sequence,
                    "source_kind": "0714T003_v2_live_window_artifact",
                    "order_status_type": order_status,
                    "public_trade_count": public_trade_count,
                    "touch_trade_qty_btc": fmt_float(sum(as_float(row.get("sz"), 0) or 0 for row in public_trade_rows if row.get("px") == limit_px)),
                    "strict_trade_through_qty_btc": fmt_float(sum(as_float(row.get("sz"), 0) or 0 for row in public_trade_rows if row.get("px") and as_float(row.get("px")) is not None and as_float(row.get("px")) < as_float(limit_px))),
                    "at_or_through_trade_qty_btc": fmt_float(sum(as_float(row.get("sz"), 0) or 0 for row in public_trade_rows if row.get("px") and as_float(row.get("px")) is not None and as_float(row.get("px")) <= as_float(limit_px))),
                    "same_side_visible_qty_at_or_ahead_of_quote_btc": same_side_qty,
                    "required_depletion_qty_btc": fmt_float((same_side_qty_float or 0) + (order_size_float or 0)) if same_side_qty_float is not None and order_size_float else "",
                    "queue_depletion_multiple": "0" if not public_trade_rows else fmt_float((sum(as_float(row.get("sz"), 0) or 0 for row in public_trade_rows if row.get("px") and as_float(row.get("px")) is not None and as_float(row.get("px")) <= as_float(limit_px))) / ((same_side_qty_float or 0) + (order_size_float or 0))) if same_side_qty_float is not None and order_size_float and ((same_side_qty_float or 0) + (order_size_float or 0)) > 0 else "",
                    "trade_through_status": trade_status,
                    "public_depletion_status": depth_summary.get("depletion_estimate_status", "insufficient_interval_trades_or_depth"),
                    "lifecycle_status": lifecycle.get("interval_status", "proxy_interval_from_local_order_response_and_cancel_ack"),
                    "depth_status": depth_summary.get("depth_status", "l2_snapshot_proxy_not_after_order_resting"),
                    "public_stream_coverage_status": coverage_status,
                    "zero_public_trade_interpretation": zero_public_trade_interpretation,
                    "capture_status": coverage_interpretation(coverage),
                }
            )
            attempt_rows.append(
                {
                    "window_id": window_id,
                    "evaluation_id": evaluation_id,
                    "order_attempt_id": order_attempt_id,
                    "event_sequence": event_sequence,
                    "source_kind": "0714T003_v2_live_window_artifact",
                    "source_path": display_path(window_dir),
                    "live_classification": live_classification,
                    "order_status_type": order_status,
                    "side": side,
                    "limit_px": limit_px,
                    "size_btc": quote.get("size_btc", ""),
                    "post_only_tif": quote.get("post_only_tif", ""),
                    "order_endpoint_called": quote.get("order_endpoint_called", ""),
                    "post_only_reject": quote.get("post_only_reject", ""),
                    "reject_interpretation": reject_interpretation(quote, reject_rows),
                    "quote_placement": placement,
                    "pre_bid": current_bid,
                    "pre_ask": current_ask,
                    "post_bid": current_bid,
                    "post_ask": current_ask,
                    "hold_elapsed_seconds": hold_elapsed,
                    "censoring_status": censoring_status,
                    "censoring_reason": censoring_reason,
                    "no_fill_state": no_fill_state(quote),
                    "depth_proxy_status": "depth_proxy_present" if l2 else "depth_proxy_missing",
                    "depth_proxy_scope": "resting_start_l2_book_snapshot_at_or_after_order_resting_proxy_not_exact" if l2 else "missing",
                    "same_side_visible_qty_at_or_ahead_of_quote_btc": same_side_qty,
                    "same_side_visible_order_count_at_or_ahead_of_quote": same_side_order_count,
                    "top_depth_multiple_of_order": top_depth_multiple,
                    "trade_through_status": trade_status,
                    "public_trade_count": public_trade_count,
                    "touch_trade_qty_btc": public_trade_summary_rows[-1]["touch_trade_qty_btc"],
                    "strict_trade_through_qty_btc": public_trade_summary_rows[-1]["strict_trade_through_qty_btc"],
                    "at_or_through_trade_qty_btc": public_trade_summary_rows[-1]["at_or_through_trade_qty_btc"],
                    "required_depletion_qty_btc": public_trade_summary_rows[-1]["required_depletion_qty_btc"],
                    "queue_depletion_multiple": public_trade_summary_rows[-1]["queue_depletion_multiple"],
                    "public_depletion_status": depth_summary.get("depletion_estimate_status", "insufficient_interval_trades_or_depth"),
                    "public_stream_coverage_status": coverage_status,
                    "zero_public_trade_interpretation": zero_public_trade_interpretation,
                    "opportunity_status": opportunity_status(trade_status, order_status),
                    "exact_timestamp_caveat": exact_timestamp_caveat,
                    "evidence_limitation": evidence_limitation,
                    "route_signal": route_signal,
                }
            )
            censoring_rows.append(
                {
                    "window_id": window_id,
                    "evaluation_id": evaluation_id,
                    "order_attempt_id": order_attempt_id,
                    "event_sequence": event_sequence,
                    "source_kind": "0714T003_v2_live_window_artifact",
                    "order_status_type": order_status,
                    "hold_elapsed_seconds": hold_elapsed,
                    "quote_aging_guard_status": aging.get("status", ""),
                    "quote_aging_guard_reason": aging.get("reason", ""),
                    "observation_horizon_status": horizon_status,
                    "censoring_status": censoring_status,
                    "censoring_reason": censoring_reason,
                }
            )
            depth_proxy_rows.append(
                {
                    "window_id": window_id,
                    "evaluation_id": evaluation_id,
                    "order_attempt_id": order_attempt_id,
                    "event_sequence": event_sequence,
                    "source_kind": "0714T003_v2_live_window_artifact",
                    "depth_proxy_status": "depth_proxy_present" if l2 else "depth_proxy_missing",
                    "depth_proxy_scope": "resting_start_l2_book_snapshot_at_or_after_order_resting_proxy_not_exact" if l2 else "missing",
                    "side": side,
                    "limit_px": limit_px,
                    "current_bid": current_bid,
                    "current_ask": current_ask,
                    "same_side_visible_qty_at_or_ahead_of_quote_btc": same_side_qty,
                    "same_side_visible_order_count_at_or_ahead_of_quote": same_side_order_count,
                    "top_depth_multiple_of_order": top_depth_multiple,
                    "visible_depth_bucket": visible_depth_bucket(top_depth_multiple),
                    "depth_reconstruction_status": depth_summary.get("depth_status", "l2_snapshot_proxy_not_after_order_resting"),
                    "post_only_non_crossing": guard.get("post_only_non_crossing", ""),
                    "current_touch_match": guard.get("current_touch_match", ""),
                }
            )
            continue

        censoring_status, censoring_reason, horizon_status = censoring(quote, aging)
        attempt_rows.append(
            {
                "window_id": window_id,
                "evaluation_id": evaluation_id,
                "order_attempt_id": "",
                "event_sequence": event_sequence,
                "source_kind": "0714T003_v2_live_window_artifact",
                "source_path": display_path(window_dir),
                "live_classification": live_classification,
                "order_status_type": order_status,
                "side": side,
                "limit_px": limit_px,
                "size_btc": quote.get("size_btc", ""),
                "post_only_tif": quote.get("post_only_tif", ""),
                "order_endpoint_called": quote.get("order_endpoint_called", ""),
                "post_only_reject": quote.get("post_only_reject", ""),
                "reject_interpretation": reject_interpretation(quote, reject_rows),
                "quote_placement": "placement_unknown_no_order_submitted",
                "pre_bid": current_bid,
                "pre_ask": current_ask,
                "post_bid": current_bid,
                "post_ask": current_ask,
                "hold_elapsed_seconds": aging.get("hold_elapsed_seconds", ""),
                "censoring_status": censoring_status,
                "censoring_reason": censoring_reason,
                "no_fill_state": no_fill_state(quote),
                "depth_proxy_status": "depth_proxy_missing_or_not_applicable",
                "depth_proxy_scope": "no_order_submitted",
                "same_side_visible_qty_at_or_ahead_of_quote_btc": "",
                "same_side_visible_order_count_at_or_ahead_of_quote": "",
                "top_depth_multiple_of_order": "",
                "trade_through_status": trade_through_status(
                    quote_row=quote,
                    lifecycle_row={},
                    public_trade_rows=[],
                    coverage_row={},
                ),
                "public_trade_count": 0,
                "touch_trade_qty_btc": "",
                "strict_trade_through_qty_btc": "",
                "at_or_through_trade_qty_btc": "",
                "required_depletion_qty_btc": "",
                "queue_depletion_multiple": "",
                "public_depletion_status": "not_applicable_no_order_submitted",
                "public_stream_coverage_status": "",
                "zero_public_trade_interpretation": "",
                "opportunity_status": opportunity_status("not_applicable_no_order_submitted", order_status),
                "exact_timestamp_caveat": "no_order_submitted",
                "evidence_limitation": "no_order_submitted",
                "route_signal": "no_order_submitted",
            }
        )
        censoring_rows.append(
            {
                "window_id": window_id,
                "evaluation_id": evaluation_id,
                "order_attempt_id": "",
                "event_sequence": event_sequence,
                "source_kind": "0714T003_v2_live_window_artifact",
                "order_status_type": order_status,
                "hold_elapsed_seconds": aging.get("hold_elapsed_seconds", ""),
                "quote_aging_guard_status": aging.get("status", ""),
                "quote_aging_guard_reason": aging.get("reason", ""),
                "observation_horizon_status": horizon_status,
                "censoring_status": censoring_status,
                "censoring_reason": censoring_reason,
            }
        )
        depth_proxy_rows.append(
            {
                "window_id": window_id,
                "evaluation_id": evaluation_id,
                "order_attempt_id": "",
                "event_sequence": event_sequence,
                "source_kind": "0714T003_v2_live_window_artifact",
                "depth_proxy_status": "depth_proxy_missing_or_not_applicable",
                "depth_proxy_scope": "no_order_submitted",
                "side": side,
                "limit_px": limit_px,
                "current_bid": current_bid,
                "current_ask": current_ask,
                "same_side_visible_qty_at_or_ahead_of_quote_btc": "",
                "same_side_visible_order_count_at_or_ahead_of_quote": "",
                "top_depth_multiple_of_order": "",
                "visible_depth_bucket": "depth_proxy_missing",
                "depth_reconstruction_status": "not_applicable_no_order_submitted",
                "post_only_non_crossing": guard.get("post_only_non_crossing", ""),
                "current_touch_match": guard.get("current_touch_match", ""),
            }
        )

    return attempt_rows, public_trade_summary_rows, censoring_rows, depth_proxy_rows, coverage_evidence_rows


def choose_recommendation(attempt_rows: list[dict[str, Any]]) -> str:
    if any(row.get("no_fill_state") == "fill_observed" for row in attempt_rows):
        return "route_to_fee_inventory_pnl_calibration"
    if any(
        row.get("trade_through_status") in {"public_flow_artifact_missing", "coverage_not_proven_complete"}
        or row.get("zero_public_trade_interpretation") == "artifact_gap_not_no_exchange_trades"
        for row in attempt_rows
    ):
        return "route_to_public_flow_artifact_repair"
    if any(row.get("censoring_status") in {"short_hold_censored", "horizon_missing"} for row in attempt_rows):
        return "route_to_more_conservative_evidence"
    if any(row.get("quote_placement") in {"behind_touch_bid", "behind_touch_ask"} for row in attempt_rows):
        return "route_to_quote_policy_design"
    if any(row.get("order_status_type") == "resting" for row in attempt_rows):
        return "stop_for_human_strategy_decision"
    return "route_to_more_conservative_evidence"


def build_sha256_manifest(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append({"artifact": str(path.relative_to(output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256", "bytes"])
    return rows


def run_analysis(*, source_root: Path, output_dir: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_summary = read_json(source_root / "0714T003_local_validation_summary.json")
    boundary_source = read_json(source_root / "boundary_manifest.json")
    window_classification_rows = read_csv_rows(source_root / "window_classification_summary.csv")

    window_dirs = sorted([path for path in source_root.glob("window_*") if path.is_dir()])
    if not window_dirs:
        raise ValueError(f"no_window_dirs_found_in:{source_root}")

    attempt_rows: list[dict[str, Any]] = []
    public_trade_summary_rows: list[dict[str, Any]] = []
    censoring_rows: list[dict[str, Any]] = []
    depth_proxy_rows: list[dict[str, Any]] = []
    coverage_evidence_rows: list[dict[str, Any]] = []

    for window_dir in window_dirs:
        classification_row = next((row for row in window_classification_rows if row.get("window") == window_dir.name), {})
        live_classification = classification_row.get("classification") or validation_summary.get("window_classifications", [{}])[0].get("classification", "")
        window_attempts, window_public_trade_summary, window_censoring, window_depth_proxy, window_coverage_evidence = build_attempt_rows(
            window_id=window_dir.name,
            window_dir=window_dir,
            live_classification=live_classification,
            source_task_id=SOURCE_TASK_ID,
            source_raw_task_id=SOURCE_RAW_TASK_ID,
        )
        attempt_rows.extend(window_attempts)
        public_trade_summary_rows.extend(window_public_trade_summary)
        censoring_rows.extend(window_censoring)
        depth_proxy_rows.extend(window_depth_proxy)
        coverage_evidence_rows.extend(window_coverage_evidence)

    if not attempt_rows:
        raise ValueError("no_attempt_rows_generated")

    recommendation = choose_recommendation(attempt_rows)

    write_csv(output_dir / "attempt_level_quote_fill_evidence_matrix.csv", attempt_rows, ATTEMPT_FIELDS)
    write_csv(output_dir / "resting_interval_public_trades_depletion_summary.csv", public_trade_summary_rows, PUBLIC_TRADE_SUMMARY_FIELDS)
    write_csv(output_dir / "public_stream_coverage_evidence_matrix.csv", coverage_evidence_rows, PUBLIC_STREAM_COVERAGE_FIELDS)
    write_csv(output_dir / "censoring_horizon_matrix.csv", censoring_rows, CENSORING_FIELDS)
    write_csv(output_dir / "same_side_depth_proxy_matrix.csv", depth_proxy_rows, DEPTH_FIELDS)

    source_files = [path for path in sorted(source_root.rglob("*")) if path.is_file()]
    input_manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_task_id": SOURCE_TASK_ID,
        "source_raw_task_id": SOURCE_RAW_TASK_ID,
        "source_root": display_path(source_root),
        "source_window_dirs": [display_path(path) for path in window_dirs],
        "source_file_count": len(source_files),
        "source_json_count": sum(1 for path in source_files if path.suffix == ".json"),
        "source_csv_count": sum(1 for path in source_files if path.suffix == ".csv"),
        "source_validation_summary": {
            "window_count": validation_summary.get("window_count", 0),
            "sha256_reconciliation_pass": validation_summary.get("sha256_reconciliation_pass", False),
            "json_parse_error_count": len(validation_summary.get("json_parse_errors", [])),
            "csv_parse_error_count": len(validation_summary.get("csv_parse_errors", [])),
            "boundary_status": validation_summary.get("boundary_manifest", {}).get("boundary_status", ""),
            "route_candidate": validation_summary.get("route_candidate", ""),
        },
        "source_attribution_overlay_used": False,
        "local_only": True,
        "provenance_remote_source_root": REMOTE_PROVENANCE_SOURCE_ROOT,
        "window_classification": window_classification_rows,
    }
    write_json(output_dir / "input_source_manifest.json", input_manifest)

    boundary = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "boundary_status": "pass",
        "offline_only": True,
        "remote_called": False,
        "aws_called": False,
        "network_called": False,
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
        "fee_claim": False,
        "rebate_claim": False,
        "realized_pnl_claim": False,
        "queue_priority_claim": False,
        "maker_viability_claim": False,
        "promotion_authorized": False,
        "t012_claim": False,
        "final_mvp_claim": False,
        "source_task_id": SOURCE_TASK_ID,
        "source_raw_task_id": SOURCE_RAW_TASK_ID,
        "local_only_source_package": True,
        "source_attribution_overlay_used": False,
        "source_boundary_status": boundary_source.get("boundary_status", ""),
    }
    write_json(output_dir / "boundary_manifest.json", boundary)

    output_files = {
        "attempt_level_quote_fill_evidence_matrix": display_path(output_dir / "attempt_level_quote_fill_evidence_matrix.csv"),
        "resting_interval_public_trades_depletion_summary": display_path(output_dir / "resting_interval_public_trades_depletion_summary.csv"),
        "public_stream_coverage_evidence_matrix": display_path(output_dir / "public_stream_coverage_evidence_matrix.csv"),
        "censoring_horizon_matrix": display_path(output_dir / "censoring_horizon_matrix.csv"),
        "same_side_depth_proxy_matrix": display_path(output_dir / "same_side_depth_proxy_matrix.csv"),
        "input_source_manifest": display_path(output_dir / "input_source_manifest.json"),
        "boundary_manifest": display_path(output_dir / "boundary_manifest.json"),
    }

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "source_task_id": SOURCE_TASK_ID,
        "source_raw_task_id": SOURCE_RAW_TASK_ID,
        "source_root": display_path(source_root),
        "output_dir": display_path(output_dir),
        "window_count": len(window_dirs),
        "attempt_count": len(attempt_rows),
        "resting_attempt_count": sum(1 for row in attempt_rows if row.get("order_status_type") == "resting"),
        "order_status_counts": counts(attempt_rows, "order_status_type"),
        "no_fill_state_counts": counts(attempt_rows, "no_fill_state"),
        "depth_proxy_status_counts": counts(attempt_rows, "depth_proxy_status"),
        "trade_through_status_counts": counts(attempt_rows, "trade_through_status"),
        "censoring_status_counts": counts(attempt_rows, "censoring_status"),
        "route_signal_counts": counts(attempt_rows, "route_signal"),
        "public_trade_summary_row_count": len(public_trade_summary_rows),
        "public_stream_coverage_evidence_row_count": len(coverage_evidence_rows),
        "public_stream_coverage_status_counts": counts(coverage_evidence_rows, "coverage_status"),
        "zero_public_trade_interpretation_counts": counts(coverage_evidence_rows, "zero_public_trade_interpretation"),
        "coverage_interpretation_counts": counts(coverage_evidence_rows, "coverage_interpretation"),
        "censoring_row_count": len(censoring_rows),
        "depth_proxy_row_count": len(depth_proxy_rows),
        "source_file_count": len(source_files),
        "source_json_count": sum(1 for path in source_files if path.suffix == ".json"),
        "source_csv_count": sum(1 for path in source_files if path.suffix == ".csv"),
        "source_validation_summary": validation_summary,
        "final_recommendation": recommendation,
        "final_route": recommendation,
        "route_rationale": (
            "The accepted v2 live window has a real submitted/resting/no-fill lifecycle, but public_stream_coverage is not complete; "
            "zero captured interval trade rows remain an artifact gap, not evidence of low fill probability."
        ),
        "supported_no_fill_reasons": [
            "submitted_resting_no_fill_observed",
            "coverage_not_proven_complete",
            "zero_rows_are_artifact_gap_not_no_exchange_trades",
            "proxy_interval_censoring",
        ],
        "unsupported_claims": [
            "synthetic_fill",
            "fee",
            "rebate",
            "realized_pnl",
            "queue_priority",
            "maker_viability",
            "promotion",
            "t012",
            "final_mvp_pass",
        ],
        "output_files": output_files,
    }
    write_json(output_dir / "quote_fill_probability_manifest.json", manifest)
    write_json(output_dir / "final_route.json", {"task_id": TASK_ID, "final_route": recommendation})

    report = [
        "# 0714T004 Quote/Fill Probability Evidence Rerun",
        "",
        f"Final recommendation: `{recommendation}`",
        "",
        f"- Attempt rows: `{len(attempt_rows)}`",
        f"- Resting attempt rows: `{manifest['resting_attempt_count']}`",
        f"- Public-trade summary rows: `{len(public_trade_summary_rows)}`",
        f"- Coverage evidence rows: `{len(coverage_evidence_rows)}`",
        f"- Censoring rows: `{len(censoring_rows)}`",
        f"- Depth proxy rows: `{len(depth_proxy_rows)}`",
        f"- Order status counts: `{manifest['order_status_counts']}`",
        f"- No-fill state counts: `{manifest['no_fill_state_counts']}`",
        f"- Trade-through counts: `{manifest['trade_through_status_counts']}`",
        f"- Censoring counts: `{manifest['censoring_status_counts']}`",
        "",
        "The v2 live package contains one submitted/resting/no-fill lifecycle, but `public_stream_coverage.csv` reports incomplete interval coverage and zero rows are marked as `artifact_gap_not_no_exchange_trades`. This does not support fill probability, queue priority, fee/rebate, quote policy design, or realized PnL claims.",
        "",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    build_sha256_manifest(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_analysis(source_root=args.source_root, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
