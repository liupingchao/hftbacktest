#!/usr/bin/env python3
"""Offline quote/fill probability evidence analysis for the accepted T011 route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0710T001"
SCHEMA_VERSION = "cross_exchange_quote_fill_probability_evidence_v1"
DEFAULT_T011_ROOT = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t011_multi_window_live_evidence_0709T001_20260709T064251Z"
DEFAULT_T002_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t011_batch_same_window_replay_acceptance_0709T002"
DEFAULT_T003_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t011_multi_window_robustness_synthesis_0709T003"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_quote_fill_probability_evidence_0710T001"

RECOMMENDATIONS = {
    "route_to_public_flow_artifact_repair",
    "route_to_more_conservative_evidence",
    "route_to_quote_policy_design",
    "route_to_controlled_same_envelope_live_evidence",
    "route_to_fee_inventory_pnl_calibration",
    "stop_for_human_strategy_decision",
}

ATTEMPT_FIELDS = [
    "window_id",
    "attempt_id",
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
    "no_fill_state",
    "depth_proxy_status",
    "trade_through_status",
    "opportunity_status",
    "evidence_limitation",
    "route_signal",
]

DEPTH_FIELDS = [
    "window_id",
    "attempt_id",
    "source_kind",
    "depth_proxy_status",
    "side",
    "limit_px",
    "current_bid",
    "current_ask",
    "same_side_top_qty_btc",
    "same_side_top_order_count",
    "top_depth_multiple_of_order",
    "visible_depth_bucket",
    "post_only_non_crossing",
    "current_touch_match",
    "depth_proxy_scope",
]

TRADE_FIELDS = [
    "window_id",
    "attempt_id",
    "source_kind",
    "trade_through_status",
    "flow_proxy_event_sequence",
    "rolling_trade_count_last_3s",
    "touch_trade_qty_btc",
    "strict_trade_through_qty_btc",
    "at_or_through_trade_qty_btc",
    "required_depletion_qty_btc",
    "queue_depletion_multiple",
    "public_depletion_status",
    "inference_scope",
]

CENSORING_FIELDS = [
    "window_id",
    "attempt_id",
    "source_kind",
    "order_status_type",
    "hold_elapsed_seconds",
    "quote_aging_guard_status",
    "quote_aging_guard_reason",
    "observation_horizon_status",
    "censoring_status",
    "censoring_reason",
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


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "accepted_by_0708T002_QA"):
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


def by_attempt(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("attempt", ""): row for row in rows if row.get("attempt")}


def rows_by_event(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("event_sequence", ""): row for row in rows if row.get("event_sequence")}


def selected_guard_row(rows: list[dict[str, str]], attempt: str, limit_px: str) -> dict[str, str]:
    matching = [row for row in rows if row.get("attempt") == attempt and row.get("status") == "pass"]
    if limit_px:
        for row in matching:
            if as_float(row.get("selected_quote_px")) == as_float(limit_px):
                return row
    return matching[-1] if matching else {}


def visible_depth_bucket(depth_multiple: str) -> str:
    value = as_float(depth_multiple)
    if value is None:
        return "depth_proxy_missing"
    if value < 1:
        return "thin_visible_top_less_than_order"
    if value < 5:
        return "moderate_visible_top_depth"
    return "large_visible_top_depth"


def quote_placement(side: str, limit_px: str, bid: str, ask: str) -> str:
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


def no_fill_state(quote_row: dict[str, str]) -> str:
    if truthy(quote_row.get("post_only_reject")):
        return "not_resting_rejected"
    fills = as_float(quote_row.get("fill_count_after_attempt"), 0) or 0
    maker_fills = as_float(quote_row.get("maker_fill_count_after_attempt"), 0) or 0
    if fills > 0 or maker_fills > 0:
        return "fill_observed"
    if quote_row.get("order_status_types") == "resting":
        return "resting_no_fill_observed"
    return "no_fill_state_unknown"


def trade_through_status(candidate_row: dict[str, str], order_status_type: str) -> str:
    if not candidate_row:
        return "public_flow_artifact_missing"
    if order_status_type == "resting":
        # Current artifacts expose rolling public-flow proxies around decision time, not a
        # full trade-through/depletion reconstruction over the actual resting interval.
        return "rolling_proxy_present_resting_interval_missing"
    strict_qty = as_float(candidate_row.get("strict_trade_through_qty_btc"), 0) or 0
    if strict_qty > 0:
        return "rolling_proxy_strict_trade_through_present"
    return "rolling_proxy_no_strict_trade_through"


def opportunity_status(candidate_row: dict[str, str], order_status_type: str) -> str:
    if not candidate_row:
        return "opportunity_not_assessable_public_flow_missing"
    if order_status_type == "resting":
        return "opportunity_censored_no_resting_interval_trade_reconstruction"
    if order_status_type == "error":
        return "not_applicable_post_only_reject"
    return "opportunity_not_assessable"


def censoring(quote_row: dict[str, str], aging_row: dict[str, str]) -> tuple[str, str, str]:
    status = quote_row.get("order_status_types", "")
    if truthy(quote_row.get("post_only_reject")) or status == "error":
        return "not_applicable_rejected", "not_resting_post_only_reject", ""
    hold = as_float(aging_row.get("hold_elapsed_seconds"))
    if hold is None:
        return "horizon_missing", "hold_elapsed_seconds_missing", "horizon_missing"
    if hold < 5:
        return "short_hold_censored", "hold_lt_5s_no_low_fill_probability_inference", "short_horizon"
    return "observed_no_fill_censored", "no_fill_observed_but_fill_probability_not_modeled", "bounded_horizon"


def market_l2(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    payload = read_json(path)
    out: dict[str, str] = {}
    for prefix, key in [("pre", "pre_submit_current_l2"), ("post", "post_submit_current_l2")]:
        levels = payload.get(key, {}).get("levels", [])
        try:
            bid = levels[0][0]
            ask = levels[1][0]
        except (IndexError, TypeError):
            continue
        out[f"{prefix}_bid"] = str(bid.get("px", ""))
        out[f"{prefix}_ask"] = str(ask.get("px", ""))
    return out


def prior_reference_attempt(t002_row: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    attempt = {
        "window_id": t002_row.get("window_id", "0708T001"),
        "attempt_id": "1",
        "source_kind": "prior_accepted_replay_reference",
        "source_path": t002_row.get("source_path", ".workflow/reports/0708T002-qa.md"),
        "live_classification": t002_row.get("live_classification", "submitted_resting_no_fill"),
        "order_status_type": t002_row.get("order_status_types", "resting"),
        "side": "",
        "limit_px": "",
        "size_btc": t002_row.get("max_order_size_btc", ""),
        "post_only_tif": "Alo",
        "order_endpoint_called": t002_row.get("real_order_endpoint_called", ""),
        "post_only_reject": "False",
        "reject_interpretation": "",
        "quote_placement": "placement_unknown_prior_reference",
        "pre_bid": "",
        "pre_ask": "",
        "post_bid": "",
        "post_ask": "",
        "hold_elapsed_seconds": "",
        "censoring_status": "horizon_missing",
        "no_fill_state": "resting_no_fill_observed",
        "depth_proxy_status": "depth_proxy_missing",
        "trade_through_status": "public_flow_artifact_missing",
        "opportunity_status": "opportunity_not_assessable_public_flow_missing",
        "evidence_limitation": "prior_reference_no_local_quote_fill_artifact",
        "route_signal": "public_flow_artifact_gap",
    }
    depth = {
        "window_id": attempt["window_id"],
        "attempt_id": "1",
        "source_kind": attempt["source_kind"],
        "depth_proxy_status": "depth_proxy_missing",
        "depth_proxy_scope": "prior_reference_no_local_artifact",
    }
    trade = {
        "window_id": attempt["window_id"],
        "attempt_id": "1",
        "source_kind": attempt["source_kind"],
        "trade_through_status": "public_flow_artifact_missing",
        "inference_scope": "prior_reference_no_local_artifact",
    }
    horizon = {
        "window_id": attempt["window_id"],
        "attempt_id": "1",
        "source_kind": attempt["source_kind"],
        "order_status_type": attempt["order_status_type"],
        "hold_elapsed_seconds": "",
        "quote_aging_guard_status": "",
        "quote_aging_guard_reason": "",
        "observation_horizon_status": "horizon_missing",
        "censoring_status": "horizon_missing",
        "censoring_reason": "prior_reference_no_local_quote_aging_artifact",
    }
    return attempt, depth, trade, horizon


def live_attempts_for_window(window_id: str, window_dir: Path, live_classification: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    quote_rows = [row for row in read_csv_rows(window_dir / "quote_attempt_matrix.csv") if truthy(row.get("order_endpoint_called"))]
    guard_rows = read_csv_rows_if_exists(window_dir / "inline_reprice_guard_matrix.csv")
    candidate_rows = rows_by_event(read_csv_rows_if_exists(window_dir / "current_candidate_audit.csv"))
    aging_rows = by_attempt(read_csv_rows_if_exists(window_dir / "quote_aging_guard_matrix.csv"))
    reject_rows = read_csv_rows_if_exists(window_dir / "inline_reprice_post_only_reject_matrix.csv")
    l2 = market_l2(window_dir / "market_markout_snapshot.json")
    attempts: list[dict[str, Any]] = []
    depths: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    horizons: list[dict[str, Any]] = []

    for quote in quote_rows:
        attempt_id = quote.get("attempt", "")
        guard = selected_guard_row(guard_rows, attempt_id, quote.get("limit_px", ""))
        candidate = candidate_rows.get(quote.get("event_sequence", ""), {})
        aging = aging_rows.get(attempt_id, {})
        order_status = quote.get("order_status_types", "")
        depth_status = "depth_proxy_present" if guard else "depth_proxy_missing"
        trade_status = trade_through_status(candidate, order_status)
        censor_status, censor_reason, horizon_status = censoring(quote, aging)
        placement = quote_placement(
            quote.get("side", ""),
            quote.get("limit_px", ""),
            guard.get("current_bid") or l2.get("pre_bid", ""),
            guard.get("current_ask") or l2.get("pre_ask", ""),
        )
        route_signal = "fill_supported" if no_fill_state(quote) == "fill_observed" else ""
        if trade_status == "public_flow_artifact_missing" or trade_status == "rolling_proxy_present_resting_interval_missing":
            route_signal = "public_flow_artifact_gap"
        elif order_status == "resting":
            route_signal = "censored_resting_no_fill"
        elif truthy(quote.get("post_only_reject")):
            route_signal = "post_only_reject"
        evidence_limitation = ""
        if trade_status == "rolling_proxy_present_resting_interval_missing":
            evidence_limitation = "rolling_public_flow_proxy_not_resting_interval_reconstruction"
        elif trade_status == "public_flow_artifact_missing":
            evidence_limitation = "public_flow_artifact_missing"
        elif truthy(quote.get("post_only_reject")):
            evidence_limitation = "reject_not_fill_probability_sample"

        attempts.append(
            {
                "window_id": window_id,
                "attempt_id": attempt_id,
                "source_kind": "t011_live_window_artifact",
                "source_path": str(window_dir),
                "live_classification": live_classification,
                "order_status_type": order_status,
                "side": quote.get("side", ""),
                "limit_px": quote.get("limit_px", ""),
                "size_btc": quote.get("size_btc", ""),
                "post_only_tif": quote.get("post_only_tif", ""),
                "order_endpoint_called": quote.get("order_endpoint_called", ""),
                "post_only_reject": quote.get("post_only_reject", ""),
                "reject_interpretation": reject_interpretation(quote, reject_rows),
                "quote_placement": placement,
                "pre_bid": guard.get("current_bid") or l2.get("pre_bid", ""),
                "pre_ask": guard.get("current_ask") or l2.get("pre_ask", ""),
                "post_bid": guard.get("current_bid") or l2.get("post_bid", ""),
                "post_ask": guard.get("current_ask") or l2.get("post_ask", ""),
                "hold_elapsed_seconds": aging.get("hold_elapsed_seconds", ""),
                "censoring_status": censor_status,
                "no_fill_state": no_fill_state(quote),
                "depth_proxy_status": depth_status,
                "trade_through_status": trade_status,
                "opportunity_status": opportunity_status(candidate, order_status),
                "evidence_limitation": evidence_limitation,
                "route_signal": route_signal,
            }
        )
        depths.append(
            {
                "window_id": window_id,
                "attempt_id": attempt_id,
                "source_kind": "t011_live_window_artifact",
                "depth_proxy_status": depth_status,
                "side": quote.get("side", ""),
                "limit_px": quote.get("limit_px", ""),
                "current_bid": guard.get("current_bid", ""),
                "current_ask": guard.get("current_ask", ""),
                "same_side_top_qty_btc": guard.get("current_same_side_top_qty_btc", ""),
                "same_side_top_order_count": guard.get("current_same_side_top_order_count", ""),
                "top_depth_multiple_of_order": guard.get("current_top_depth_multiple_of_order", ""),
                "visible_depth_bucket": visible_depth_bucket(guard.get("current_top_depth_multiple_of_order", "")),
                "post_only_non_crossing": guard.get("post_only_non_crossing", ""),
                "current_touch_match": guard.get("current_touch_match", ""),
                "depth_proxy_scope": "post_open_orders_inline_reprice_public_l2_proxy_not_exact_queue_priority" if guard else "missing",
            }
        )
        trades.append(
            {
                "window_id": window_id,
                "attempt_id": attempt_id,
                "source_kind": "t011_live_window_artifact",
                "trade_through_status": trade_status,
                "flow_proxy_event_sequence": candidate.get("event_sequence", ""),
                "rolling_trade_count_last_3s": candidate.get("rolling_trade_count_last_3s", ""),
                "touch_trade_qty_btc": candidate.get("touch_trade_qty_btc", ""),
                "strict_trade_through_qty_btc": candidate.get("strict_trade_through_qty_btc", ""),
                "at_or_through_trade_qty_btc": candidate.get("at_or_through_trade_qty_btc", ""),
                "required_depletion_qty_btc": candidate.get("required_depletion_qty_btc", ""),
                "queue_depletion_multiple": candidate.get("queue_depletion_multiple", ""),
                "public_depletion_status": candidate.get("public_depletion_status", ""),
                "inference_scope": candidate.get("inference_scope", "missing"),
            }
        )
        horizons.append(
            {
                "window_id": window_id,
                "attempt_id": attempt_id,
                "source_kind": "t011_live_window_artifact",
                "order_status_type": order_status,
                "hold_elapsed_seconds": aging.get("hold_elapsed_seconds", ""),
                "quote_aging_guard_status": aging.get("status", ""),
                "quote_aging_guard_reason": aging.get("reason", ""),
                "observation_horizon_status": horizon_status,
                "censoring_status": censor_status,
                "censoring_reason": censor_reason,
            }
        )
    return attempts, depths, trades, horizons


def build_sha256_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append({"artifact": str(path.relative_to(output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256", "bytes"])


def choose_recommendation(attempts: list[dict[str, Any]]) -> str:
    if any(row.get("no_fill_state") == "fill_observed" for row in attempts):
        return "route_to_fee_inventory_pnl_calibration"
    if any(row.get("trade_through_status") in {"public_flow_artifact_missing", "rolling_proxy_present_resting_interval_missing"} for row in attempts):
        return "route_to_public_flow_artifact_repair"
    if any(row.get("censoring_status") in {"short_hold_censored", "horizon_missing"} for row in attempts):
        return "route_to_more_conservative_evidence"
    if any(row.get("quote_placement") in {"behind_touch_bid", "behind_touch_ask"} for row in attempts):
        return "route_to_quote_policy_design"
    return "route_to_controlled_same_envelope_live_evidence"


def run_analysis(*, t011_root: Path, t002_dir: Path, t003_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    t002_rows = read_csv_rows(t002_dir / "batch_replay_acceptance_matrix.csv")
    t003_manifest = read_json(t003_dir / "multi_window_synthesis_manifest.json")
    attempts: list[dict[str, Any]] = []
    depths: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    horizons: list[dict[str, Any]] = []

    for row in t002_rows:
        window_id = row.get("window_id", "")
        if row.get("source_kind") == "prior_accepted_replay_reference":
            attempt, depth, trade, horizon = prior_reference_attempt(row)
            attempts.append(attempt)
            depths.append(depth)
            trades.append(trade)
            horizons.append(horizon)
            continue
        if not window_id.startswith("0709T001_window_"):
            continue
        suffix = window_id.rsplit("_", 1)[-1]
        window_dir = t011_root / f"window_{suffix}"
        window_attempts, window_depths, window_trades, window_horizons = live_attempts_for_window(window_id, window_dir, row.get("live_classification", ""))
        attempts.extend(window_attempts)
        depths.extend(window_depths)
        trades.extend(window_trades)
        horizons.extend(window_horizons)

    recommendation = choose_recommendation(attempts)
    if recommendation not in RECOMMENDATIONS:
        raise ValueError(f"invalid_recommendation:{recommendation}")

    write_csv(output_dir / "attempt_level_fill_probability_matrix.csv", attempts, ATTEMPT_FIELDS)
    write_csv(output_dir / "same_side_depth_proxy_matrix.csv", depths, DEPTH_FIELDS)
    write_csv(output_dir / "trade_through_depletion_matrix.csv", trades, TRADE_FIELDS)
    write_csv(output_dir / "censoring_and_horizon_matrix.csv", horizons, CENSORING_FIELDS)

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "git_commit": git_commit(),
        "t011_root": str(t011_root),
        "t002_dir": str(t002_dir),
        "t003_dir": str(t003_dir),
        "t003_final_recommendation": t003_manifest.get("final_recommendation"),
        "accepted_reference_count": len(t002_rows),
        "attempt_count": len(attempts),
        "source_kind_counts": counts(attempts, "source_kind"),
        "live_classification_counts": counts(attempts, "live_classification"),
        "order_status_counts": counts(attempts, "order_status_type"),
        "no_fill_state_counts": counts(attempts, "no_fill_state"),
        "depth_proxy_status_counts": counts(attempts, "depth_proxy_status"),
        "trade_through_status_counts": counts(attempts, "trade_through_status"),
        "censoring_status_counts": counts(attempts, "censoring_status"),
        "route_signal_counts": counts(attempts, "route_signal"),
        "final_recommendation": recommendation,
        "route_rationale": "Accepted T011 attempts expose post-only rejects and short-horizon resting no-fill rows, but current artifacts do not reconstruct public trade-through/depletion over the actual resting interval; repair public-flow interval evidence before quoting fill probability.",
        "does_not_claim": ["synthetic_fill", "fee", "rebate", "realized_pnl", "queue_priority", "maker_viability", "promotion", "t012", "final_mvp_pass"],
    }
    write_json(output_dir / "quote_fill_probability_manifest.json", manifest)

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
        "# 0710T001 Quote/Fill Probability Evidence",
        "",
        f"Final recommendation: `{recommendation}`",
        "",
        f"- Attempt count: `{len(attempts)}`",
        f"- Order status counts: `{manifest['order_status_counts']}`",
        f"- No-fill state counts: `{manifest['no_fill_state_counts']}`",
        f"- Depth proxy counts: `{manifest['depth_proxy_status_counts']}`",
        f"- Trade-through status counts: `{manifest['trade_through_status_counts']}`",
        f"- Censoring counts: `{manifest['censoring_status_counts']}`",
        "",
        "The available evidence supports post-only reject and censored no-fill classification only. It does not support a fitted fill-probability model, exact queue priority, fee/rebate, realized PnL, maker viability, T012, promotion, or final MVP claims.",
        "",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    build_sha256_manifest(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t011-root", type=Path, default=DEFAULT_T011_ROOT)
    parser.add_argument("--t002-dir", type=Path, default=DEFAULT_T002_DIR)
    parser.add_argument("--t003-dir", type=Path, default=DEFAULT_T003_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_analysis(t011_root=args.t011_root, t002_dir=args.t002_dir, t003_dir=args.t003_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
