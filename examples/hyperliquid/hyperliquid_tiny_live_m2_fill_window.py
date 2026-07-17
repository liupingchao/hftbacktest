#!/usr/bin/env python3
"""Single-window M2B Hyperliquid maker-only fill attempt.

This script is intended to run on awsserver1 inside the approved T009 envelope.
It may place one real post-only Alo order, wait briefly for passive fills,
tracked-cancel the order, and write redacted artifacts for local reconciliation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import hyperliquid_public_sample
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_flow_diagnosis as public_flow
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


TASK_ID = "0622T004"
READY_RECOMMENDATION = "hyperliquid_tiny_live_m2_fill_window_ready_for_qa"
BLOCKED_RECOMMENDATION = "hyperliquid_tiny_live_m2_fill_window_blocked"
OPERATOR_ACK = executor.LIVE_OPERATOR_ACK
MAX_WAIT_SECONDS = 600
FLOW_AWARE_POLICY_VERSION = "m2_flow_aware_v1"
FRESH_TOUCH_POLICY_VERSION = "m2_fresh_touch_size_by_throughput_session_gate_v1"
DEFAULT_FLOW_MAX_TOP_DEPTH_MULTIPLE = 500.0
DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS = 0.0
FLOW_SAFE_HOLD_SECONDS = 15
DEFAULT_FLOW_PRECHECK_SECONDS = 20.0
FRESH_TOUCH_HARD_CAP_BTC = 0.005
FRESH_TOUCH_QUALITY_A_BUCKET_CAP_BTC = 0.005
FRESH_TOUCH_QUALITY_B_BUCKET_CAP_BTC = 0.002
FRESH_TOUCH_QUALITY_A_MAX_DEPTH_MULTIPLE = 20.0
FRESH_TOUCH_QUALITY_B_MAX_DEPTH_MULTIPLE = 100.0
FRESH_TOUCH_QUALITY_A_MAX_ORDER_COUNT = 6
FRESH_TOUCH_QUALITY_B_MAX_ORDER_COUNT = 12
FRESH_TOUCH_QUALITY_A_HOLD_SECONDS = 3
FRESH_TOUCH_QUALITY_B_HOLD_SECONDS = 1
FRESH_TOUCH_THROUGHPUT_LOOKBACK_SECONDS = 3.0
FRESH_TOUCH_MAX_PRECHECK_AGE_SECONDS = 20.0
FRESH_TOUCH_MAX_IMMEDIATE_GUARD_AGE_SECONDS = 3.0
DEFAULT_FRESH_TOUCH_PRECHECK_SECONDS = 20.0
DEFAULT_FRESH_TOUCH_CANDIDATE_STRIDE_SECONDS = 1.0


def policy_version_for_side_policy(side_policy: str) -> str:
    if side_policy == "fresh_touch":
        return FRESH_TOUCH_POLICY_VERSION
    if side_policy == "flow_aware":
        return FLOW_AWARE_POLICY_VERSION
    return "legacy_side_policy"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(executor.redact(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: executor.redact(row.get(field, "")) for field in fieldnames})


def floor_to_lot(size: float, lot_size: float) -> float:
    if lot_size <= 0:
        return 0.0
    steps = math.floor((size + lot_size * 1e-9) / lot_size)
    return round(steps * lot_size, 10)


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value in ("", None):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def safe_int(value: Any, default: int | None = None) -> int | None:
    if value in ("", None):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def artifact_window_label(window_id: int) -> str:
    parsed = safe_int(window_id)
    if parsed is None or parsed <= 0:
        raise executor.ValidationError("artifact_window_id_must_be_positive")
    return f"window_{parsed:02d}"


def artifact_attempt_key(*, task_id: str, window_id: int, attempt_id: int) -> str:
    normalized_task_id = str(task_id).strip()
    parsed_attempt_id = safe_int(attempt_id)
    if not normalized_task_id:
        raise executor.ValidationError("artifact_task_id_must_be_nonempty")
    if parsed_attempt_id is None or parsed_attempt_id <= 0:
        raise executor.ValidationError("attempt_id_must_be_positive")
    return f"{normalized_task_id}:{artifact_window_label(window_id)}:attempt_{parsed_attempt_id}"


def bind_attempt_identity(
    attempt_rows: list[dict[str, Any]],
    *,
    task_id: str,
    window_id: int,
) -> None:
    window_label = artifact_window_label(window_id)
    for row in attempt_rows:
        attempt_id = safe_int(row.get("attempt"))
        row["window_id"] = window_label
        row["attempt_id"] = "" if attempt_id is None else attempt_id
        row["attempt_key"] = (
            ""
            if attempt_id is None
            else artifact_attempt_key(task_id=task_id, window_id=window_id, attempt_id=attempt_id)
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def best_bid_ask(l2_snapshot: dict[str, Any]) -> tuple[float, float]:
    levels = l2_snapshot.get("levels", [])
    if len(levels) < 2 or not levels[0] or not levels[1]:
        raise executor.ValidationError("l2_snapshot_missing_bid_ask")
    bid = float(levels[0][0]["px"])
    ask = float(levels[1][0]["px"])
    if bid <= 0 or ask <= 0 or bid >= ask:
        raise executor.ValidationError(f"invalid_bid_ask:{bid}:{ask}")
    return bid, ask


def top_qty_order_count(l2_snapshot: dict[str, Any], *, is_buy: bool) -> tuple[float, int | None]:
    levels = l2_snapshot.get("levels", [])
    side_index = 0 if is_buy else 1
    if len(levels) <= side_index or not levels[side_index]:
        raise executor.ValidationError("l2_snapshot_missing_same_side_top")
    top = levels[side_index][0]
    qty = float(top.get("sz", 0.0))
    order_count = top.get("n")
    try:
        parsed_order_count = int(order_count) if order_count not in ("", None) else None
    except (TypeError, ValueError):
        parsed_order_count = None
    if qty <= 0:
        raise executor.ValidationError("same_side_top_qty_nonpositive")
    return qty, parsed_order_count


def precision_from_l2_public_snapshot(l2_snapshot: dict[str, Any]) -> executor.PrecisionFacts:
    bid, ask = best_bid_ask(l2_snapshot)
    return executor.PrecisionFacts(
        symbol=executor.SYMBOL,
        sz_decimals=5,
        tick_size=1.0,
        lot_size=0.00001,
        mid_px=(bid + ask) / 2.0,
        source="public_l2_immediate_guard_default_btc_precision_no_private_meta",
    )


def build_top_of_book_maker_intent(
    *,
    precision: executor.PrecisionFacts,
    bid: float,
    ask: float,
    quote_offset_ticks: int,
    window_id: int,
    is_buy: bool = True,
    attempt_id: int = 1,
) -> executor.OrderIntent:
    ticks = max(0, quote_offset_ticks)
    if is_buy:
        limit_px = bid + ticks * precision.tick_size
        if limit_px >= ask:
            limit_px = bid
    else:
        limit_px = ask - ticks * precision.tick_size
        if limit_px <= bid:
            limit_px = ask
    limit_px = executor.round_hyperliquid_perp_price(limit_px, precision.sz_decimals)
    if is_buy and limit_px >= ask:
        raise executor.ValidationError("post_only_buy_would_cross_ask")
    if not is_buy and limit_px <= bid:
        raise executor.ValidationError("post_only_sell_would_cross_bid")
    size_cap = min(executor.MAX_ORDER_SIZE_BTC, executor.MAX_ORDER_NOTIONAL_USDC / limit_px)
    size = floor_to_lot(size_cap, precision.lot_size)
    if size <= 0:
        raise executor.ValidationError("computed_order_size_nonpositive")
    return executor.OrderIntent(
        symbol=executor.SYMBOL,
        is_buy=is_buy,
        size_btc=size,
        limit_px=limit_px,
        time_in_force=executor.POST_ONLY_TIF,
        reduce_only=False,
        cloid=executor.generate_cloid(f"{TASK_ID}_w{window_id}_a{attempt_id}"),
    )


def side_for_attempt(side_policy: str, attempt_id: int, flow_decision: dict[str, Any] | None = None) -> bool:
    if side_policy == "flow_aware":
        if not flow_decision or flow_decision.get("allowed") is not True:
            raise executor.ValidationError("flow_aware_side_unavailable_or_blocked")
        side = flow_decision.get("selected_side")
        if side == "buy":
            return True
        if side == "sell":
            return False
        raise executor.ValidationError(f"flow_aware_bad_selected_side:{side}")
    if side_policy == "buy":
        return True
    if side_policy == "sell":
        return False
    if side_policy == "alternate":
        return attempt_id % 2 == 1
    raise executor.ValidationError(f"unsupported_side_policy:{side_policy}")


def flow_side_scores(
    *,
    l2_snapshot: dict[str, Any],
    order_size_btc: float,
    max_top_depth_multiple: float = DEFAULT_FLOW_MAX_TOP_DEPTH_MULTIPLE,
    public_flow_summary: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    by_side = (public_flow_summary or {}).get("by_side", {})
    for side, is_buy in (("buy", True), ("sell", False)):
        top_qty, order_count = top_qty_order_count(l2_snapshot, is_buy=is_buy)
        top_depth_multiple = top_qty / order_size_btc if order_size_btc > 0 else math.inf
        crowd_penalty = min(1.0, top_depth_multiple / max_top_depth_multiple) if max_top_depth_multiple > 0 else 1.0
        order_count_penalty = min(0.25, max(0, (order_count or 0) - 1) * 0.02)
        side_prior = 0.15 if side == "buy" else -0.15
        public_side = by_side.get(side, {})
        candidate_count = int(public_side.get("candidate_count", 0) or 0)
        strict_rate = float(public_side.get("strict_trade_through_candidate_count", 0) or 0) / candidate_count if candidate_count else 0.0
        depletion_rate = float(public_side.get("public_depletion_candidate_count", 0) or 0) / candidate_count if candidate_count else 0.0
        aging_rate = float(public_side.get("adverse_lost_touch_candidate_count", 0) or 0) / candidate_count if candidate_count else 0.0
        public_flow_boost = 0.20 * strict_rate + 0.35 * depletion_rate - 0.20 * aging_rate
        score = max(0.0, min(1.0, 1.0 - crowd_penalty - order_count_penalty + side_prior + public_flow_boost))
        scores[side] = {
            "side": side,
            "same_side_top_qty_btc": top_qty,
            "same_side_top_order_count": "" if order_count is None else order_count,
            "top_depth_multiple_of_order": top_depth_multiple,
            "public_flow_candidate_count": candidate_count,
            "public_flow_strict_rate": round(strict_rate, 8),
            "public_flow_depletion_rate": round(depletion_rate, 8),
            "public_flow_aging_rate": round(aging_rate, 8),
            "crowd_penalty": round(crowd_penalty, 8),
            "side_prior": side_prior,
            "public_flow_boost": round(public_flow_boost, 8),
            "score": round(score, 8),
            "guard_status": "pass" if top_depth_multiple <= max_top_depth_multiple else "skip_crowded_touch",
            "evidence_scope": "public_l2_top_depth_plus_public_trades_proxy" if candidate_count else "public_l2_top_depth_proxy_only",
        }
    return scores


def same_side_at_or_through_qty_from_candidate(candidate: dict[str, Any], lookback_seconds: float = FRESH_TOUCH_THROUGHPUT_LOOKBACK_SECONDS) -> float | None:
    qty = safe_float(candidate.get("at_or_through_trade_qty_btc"))
    if qty is None:
        return None
    hold = safe_float(candidate.get("hold_seconds"))
    if hold is None or hold <= 0:
        return qty
    scale = min(1.0, lookback_seconds / hold)
    return qty * scale


def dynamic_fresh_touch_size(
    *,
    bucket: str,
    recent_same_side_at_or_through_qty_btc: float | None,
    lot_size: float,
    hard_cap_btc: float = FRESH_TOUCH_HARD_CAP_BTC,
) -> dict[str, Any]:
    if bucket == "quality_a":
        bucket_cap = min(FRESH_TOUCH_QUALITY_A_BUCKET_CAP_BTC, hard_cap_btc)
    elif bucket == "quality_b":
        bucket_cap = min(FRESH_TOUCH_QUALITY_B_BUCKET_CAP_BTC, hard_cap_btc)
    else:
        return {
            "bucket": bucket,
            "bucket_cap_btc": "",
            "hard_cap_btc": hard_cap_btc,
            "recent_same_side_at_or_through_qty_btc_last_3s": "" if recent_same_side_at_or_through_qty_btc is None else recent_same_side_at_or_through_qty_btc,
            "raw_size_btc": "",
            "floored_size_btc": 0.0,
            "status": "skip",
            "reason": "unsupported_quality_bucket",
        }
    if recent_same_side_at_or_through_qty_btc is None or recent_same_side_at_or_through_qty_btc <= 0:
        raw_size = 0.0
        floored = 0.0
        status = "skip"
        reason = "missing_or_zero_recent_same_side_at_or_through_trade_qty"
    else:
        raw_size = min(bucket_cap, 0.25 * recent_same_side_at_or_through_qty_btc, hard_cap_btc)
        floored = floor_to_lot(raw_size, lot_size)
        status = "pass" if floored > 0 else "skip"
        reason = "" if floored > 0 else "dynamic_size_floors_to_zero"
    return {
        "bucket": bucket,
        "bucket_cap_btc": bucket_cap,
        "hard_cap_btc": hard_cap_btc,
        "recent_same_side_at_or_through_qty_btc_last_3s": "" if recent_same_side_at_or_through_qty_btc is None else round(recent_same_side_at_or_through_qty_btc, 10),
        "raw_size_btc": round(raw_size, 10),
        "floored_size_btc": floored,
        "status": status,
        "reason": reason,
    }


def classify_fresh_touch_quality(
    *,
    side: str,
    top_depth_multiple: float | None,
    same_side_top_order_count: int | None,
    strict_through_supported: bool,
    touch_freshness_present: bool,
    recent_same_side_at_or_through_qty_btc: float | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if side != "buy":
        reasons.append("sell_disabled_by_default_buy_only_gate")
    if top_depth_multiple is None:
        reasons.append("missing_same_side_top_depth_multiple")
    if same_side_top_order_count is None:
        reasons.append("missing_same_side_top_order_count")
    if not strict_through_supported:
        reasons.append("missing_same_side_strict_through_support")
    if not touch_freshness_present:
        reasons.append("missing_touch_freshness_or_queue_reset_evidence")
    if recent_same_side_at_or_through_qty_btc is None or recent_same_side_at_or_through_qty_btc <= 0:
        reasons.append("missing_recent_same_side_at_or_through_throughput")
    if reasons:
        return {"allowed": False, "quality_bucket": "", "hold_seconds": "", "skip_reason": ";".join(reasons)}
    assert top_depth_multiple is not None
    assert same_side_top_order_count is not None
    if top_depth_multiple <= FRESH_TOUCH_QUALITY_A_MAX_DEPTH_MULTIPLE and same_side_top_order_count <= FRESH_TOUCH_QUALITY_A_MAX_ORDER_COUNT:
        return {"allowed": True, "quality_bucket": "quality_a", "hold_seconds": FRESH_TOUCH_QUALITY_A_HOLD_SECONDS, "skip_reason": ""}
    if (
        top_depth_multiple > FRESH_TOUCH_QUALITY_A_MAX_DEPTH_MULTIPLE
        and top_depth_multiple <= FRESH_TOUCH_QUALITY_B_MAX_DEPTH_MULTIPLE
        and same_side_top_order_count <= FRESH_TOUCH_QUALITY_B_MAX_ORDER_COUNT
    ):
        return {"allowed": True, "quality_bucket": "quality_b", "hold_seconds": FRESH_TOUCH_QUALITY_B_HOLD_SECONDS, "skip_reason": ""}
    return {
        "allowed": False,
        "quality_bucket": "",
        "hold_seconds": "",
        "skip_reason": "outside_quality_a_b_queue_bands",
    }


def candidate_freshness_status(row: dict[str, Any], *, summary: dict[str, Any], require_real_bbo_history: bool | None = None) -> dict[str, Any]:
    candidate_start_ms = safe_int(row.get("start_exchange_time_ms"))
    collection_end_ms = safe_int((summary or {}).get("last_book_exchange_time_ms"))
    if candidate_start_ms is None:
        return {"status": "missing", "age_seconds": "", "reason": "missing_candidate_start_time"}
    if collection_end_ms is None:
        age_seconds = 0.0
    else:
        age_seconds = max(0.0, (collection_end_ms - candidate_start_ms) / 1000.0)
    if age_seconds > FRESH_TOUCH_MAX_PRECHECK_AGE_SECONDS:
        return {"status": "stale", "age_seconds": round(age_seconds, 6), "reason": "candidate_older_than_session_gate_max_age"}
    freshness_source = str(row.get("freshness_source", ""))
    evidence_status = str(row.get("fresh_touch_evidence_status", ""))
    synthetic_only_sources = {"synthetic_current_event_only", "synthetic_stayed_touch"}
    if require_real_bbo_history is None:
        require_real_bbo_history = bool(row.get("event_driven_current_candidate")) or freshness_source in synthetic_only_sources
    if require_real_bbo_history:
        if not freshness_source:
            return {"status": "missing", "age_seconds": round(age_seconds, 6), "reason": "missing_real_bbo_history_freshness_source"}
        if freshness_source in synthetic_only_sources:
            return {"status": "missing", "age_seconds": round(age_seconds, 6), "reason": "synthetic_current_event_not_fresh_touch_proof"}
        if evidence_status != "pass":
            return {
                "status": "missing",
                "age_seconds": round(age_seconds, 6),
                "reason": str(row.get("top_reset_reason") or row.get("fresh_touch_evidence_reason") or "real_bbo_history_fresh_touch_evidence_not_pass"),
            }
        return {"status": "fresh_or_reset_supported", "age_seconds": round(age_seconds, 6), "reason": ""}
    first_touch_ms = safe_int(row.get("first_touch_trade_ms"))
    first_strict_ms = safe_int(row.get("first_strict_trade_through_ms"))
    quote_aging_status = str(row.get("quote_aging_status", ""))
    if first_touch_ms is not None or first_strict_ms is not None or quote_aging_status == "stayed_touch":
        return {"status": "fresh_or_reset_supported", "age_seconds": round(age_seconds, 6), "reason": ""}
    return {"status": "missing", "age_seconds": round(age_seconds, 6), "reason": "no_touch_or_reset_proxy_in_public_window"}


def load_fresh_touch_candidates(public_flow_precheck: dict[str, Any]) -> list[dict[str, str]]:
    inline_rows = public_flow_precheck.get("candidate_rows_inline")
    if isinstance(inline_rows, list):
        return [dict(row) for row in inline_rows if isinstance(row, dict)]
    diagnosis_files = public_flow_precheck.get("diagnosis_manifest", {}).get("output_files", {})
    candidates_path = diagnosis_files.get("candidate_flow_diagnostics")
    if not candidates_path:
        return []
    return read_csv_rows(Path(candidates_path))


def select_fresh_touch_candidate(
    *,
    l2_snapshot: dict[str, Any],
    precision: executor.PrecisionFacts,
    window_id: int,
    attempt_id: int,
    public_flow_precheck: dict[str, Any],
    max_order_size_btc: float,
) -> dict[str, Any]:
    bid, ask = best_bid_ask(l2_snapshot)
    buy_top_qty, buy_order_count = top_qty_order_count(l2_snapshot, is_buy=True)
    rows = load_fresh_touch_candidates(public_flow_precheck)
    summary = public_flow_precheck.get("summary", {})
    require_real_bbo_history = bool(public_flow_precheck.get("event_driven_inline_candidate"))
    buy_rows = [row for row in rows if row.get("side") == "buy"]
    decisions: list[dict[str, Any]] = []
    for index, row in enumerate(buy_rows, start=1):
        freshness = candidate_freshness_status(row, summary=summary, require_real_bbo_history=require_real_bbo_history)
        strict_qty = safe_float(row.get("strict_trade_through_qty_btc"), 0.0) or 0.0
        recent_qty = same_side_at_or_through_qty_from_candidate(row)
        common_reasons: list[str] = []
        if buy_order_count is None:
            common_reasons.append("missing_same_side_top_order_count")
        if strict_qty <= 0:
            common_reasons.append("missing_same_side_strict_through_support")
        if freshness.get("status") != "fresh_or_reset_supported":
            common_reasons.append("missing_touch_freshness_or_queue_reset_evidence")
        if recent_qty is None or recent_qty <= 0:
            common_reasons.append("missing_recent_same_side_at_or_through_throughput")

        size_decision = dynamic_fresh_touch_size(
            bucket="quality_a",
            recent_same_side_at_or_through_qty_btc=recent_qty,
            lot_size=precision.lot_size,
            hard_cap_btc=min(FRESH_TOUCH_HARD_CAP_BTC, max_order_size_btc),
        )
        candidate_size = float(size_decision.get("floored_size_btc") or 0.0)
        top_depth_multiple = buy_top_qty / candidate_size if candidate_size > 0 else math.inf
        quality = {"allowed": False, "quality_bucket": "", "hold_seconds": "", "skip_reason": ";".join(common_reasons)}
        if not common_reasons:
            assert buy_order_count is not None
            if (
                size_decision.get("status") == "pass"
                and top_depth_multiple <= FRESH_TOUCH_QUALITY_A_MAX_DEPTH_MULTIPLE
                and buy_order_count <= FRESH_TOUCH_QUALITY_A_MAX_ORDER_COUNT
            ):
                quality = {"allowed": True, "quality_bucket": "quality_a", "hold_seconds": FRESH_TOUCH_QUALITY_A_HOLD_SECONDS, "skip_reason": ""}
            else:
                size_decision = dynamic_fresh_touch_size(
                    bucket="quality_b",
                    recent_same_side_at_or_through_qty_btc=recent_qty,
                    lot_size=precision.lot_size,
                    hard_cap_btc=min(FRESH_TOUCH_HARD_CAP_BTC, max_order_size_btc),
                )
                candidate_size = float(size_decision.get("floored_size_btc") or 0.0)
                top_depth_multiple = buy_top_qty / candidate_size if candidate_size > 0 else math.inf
                if (
                    size_decision.get("status") == "pass"
                    and top_depth_multiple > FRESH_TOUCH_QUALITY_A_MAX_DEPTH_MULTIPLE
                    and top_depth_multiple <= FRESH_TOUCH_QUALITY_B_MAX_DEPTH_MULTIPLE
                    and buy_order_count <= FRESH_TOUCH_QUALITY_B_MAX_ORDER_COUNT
                ):
                    quality = {"allowed": True, "quality_bucket": "quality_b", "hold_seconds": FRESH_TOUCH_QUALITY_B_HOLD_SECONDS, "skip_reason": ""}
                else:
                    quality = {"allowed": False, "quality_bucket": "", "hold_seconds": "", "skip_reason": "outside_quality_a_b_queue_bands"}
        allowed = bool(quality.get("allowed")) and size_decision.get("status") == "pass"
        skip_reason = str(quality.get("skip_reason") or size_decision.get("reason") or "")
        decision = {
            "candidate_index": index,
            "side": "buy",
            "allowed": allowed,
            "selected": False,
            "quality_bucket": quality.get("quality_bucket", ""),
            "hold_seconds": quality.get("hold_seconds", ""),
            "skip_reason": skip_reason,
            "bid": bid,
            "ask": ask,
            "quote_px": bid,
            "same_side_top_qty_btc": buy_top_qty,
            "same_side_top_order_count": "" if buy_order_count is None else buy_order_count,
            "top_depth_multiple_of_order": round(top_depth_multiple, 8),
            "strict_trade_through_qty_btc": strict_qty,
            "recent_same_side_at_or_through_qty_btc_last_3s": size_decision.get("recent_same_side_at_or_through_qty_btc_last_3s", ""),
            "raw_size_btc": size_decision.get("raw_size_btc", ""),
            "dynamic_size_btc": size_decision.get("floored_size_btc", 0.0),
            "bucket_cap_btc": size_decision.get("bucket_cap_btc", ""),
            "dynamic_size_status": size_decision.get("status", ""),
            "dynamic_size_reason": size_decision.get("reason", ""),
            "freshness_status": freshness.get("status", ""),
            "freshness_age_seconds": freshness.get("age_seconds", ""),
            "freshness_reason": freshness.get("reason", ""),
            "source_start_exchange_time_ms": row.get("start_exchange_time_ms", ""),
            "source_quote_aging_status": row.get("quote_aging_status", ""),
            "freshness_source": row.get("freshness_source", ""),
            "touch_stability_ms": row.get("touch_stability_ms", ""),
            "last_touch_change_ms": row.get("last_touch_change_ms", ""),
            "top_reset_status": row.get("top_reset_status", ""),
            "top_reset_reason": row.get("top_reset_reason", ""),
            "fresh_touch_evidence_status": row.get("fresh_touch_evidence_status", ""),
            "source_first_touch_trade_ms": row.get("first_touch_trade_ms", ""),
            "source_first_strict_trade_through_ms": row.get("first_strict_trade_through_ms", ""),
            "inference_scope": "public_flow_proxy_plus_current_l2_top_depth_not_exact_queue_or_fill_probability",
        }
        decisions.append(decision)
    allowed_decisions = [row for row in decisions if row.get("allowed")]
    if allowed_decisions:
        selected = max(allowed_decisions, key=lambda row: safe_int(row.get("source_start_exchange_time_ms"), -1) or -1)
    else:
        selected = max(decisions, key=lambda row: safe_int(row.get("source_start_exchange_time_ms"), -1) or -1) if decisions else None
    if selected:
        selected["selected"] = bool(selected.get("allowed"))
    if not selected:
        return {
            "policy_version": FRESH_TOUCH_POLICY_VERSION,
            "allowed": False,
            "selected_side": "",
            "skip_reason": "no_buy_public_flow_candidates",
            "bid": bid,
            "ask": ask,
            "candidate_rows": [],
            "intent_limit_px": "",
            "intent_size_btc": "",
            "hold_seconds": "",
        }
    intent: executor.OrderIntent | None = None
    allowed = bool(selected.get("allowed"))
    skip_reason = str(selected.get("skip_reason", ""))
    if allowed:
        try:
            intent = executor.OrderIntent(
                symbol=executor.SYMBOL,
                is_buy=True,
                size_btc=float(selected.get("dynamic_size_btc") or 0.0),
                limit_px=executor.round_hyperliquid_perp_price(bid, precision.sz_decimals),
                time_in_force=executor.POST_ONLY_TIF,
                reduce_only=False,
                cloid=executor.generate_cloid(f"{TASK_ID}_w{window_id}_a{attempt_id}"),
            )
            if intent.limit_px >= ask:
                raise executor.ValidationError("post_only_buy_would_cross_ask")
        except Exception as exc:
            allowed = False
            skip_reason = executor._redacted_error(exc)
    return {
        "policy_version": FRESH_TOUCH_POLICY_VERSION,
        "allowed": allowed,
        "selected_side": "buy" if allowed else "",
        "skip_reason": skip_reason,
        "bid": bid,
        "ask": ask,
        "candidate_rows": decisions,
        "intent_limit_px": "" if intent is None else intent.limit_px,
        "intent_size_btc": "" if intent is None else intent.size_btc,
        "hold_seconds": selected.get("hold_seconds", ""),
        "quality_bucket": selected.get("quality_bucket", ""),
        "selected_candidate": selected,
        "inference_scope": "public_flow_proxy_plus_current_l2_top_depth_not_exact_queue_or_fill_probability",
    }


def immediate_fresh_touch_guard(
    *,
    selected_candidate: dict[str, Any],
    decision: dict[str, Any],
    l2_snapshot: dict[str, Any],
    precision: executor.PrecisionFacts,
    max_order_size_btc: float,
    max_age_seconds: float = FRESH_TOUCH_MAX_IMMEDIATE_GUARD_AGE_SECONDS,
    trigger_candidate: dict[str, Any] | None = None,
    handoff_phase: str = "",
) -> dict[str, Any]:
    bid, ask = best_bid_ask(l2_snapshot)
    buy_top_qty, buy_order_count = top_qty_order_count(l2_snapshot, is_buy=True)
    current_candidate = dict(selected_candidate or decision.get("selected_candidate") or {})
    trigger = dict(trigger_candidate or {})
    current_candidate_source_ms = safe_int(
        current_candidate.get("source_start_exchange_time_ms")
        or selected_candidate.get("source_start_exchange_time_ms")
        or decision.get("source_start_exchange_time_ms")
    )
    trigger_candidate_source_ms = safe_int(
        trigger.get("source_start_exchange_time_ms")
        or trigger.get("source_event_exchange_time_ms")
        or trigger.get("start_exchange_time_ms")
    )
    candidate_source_ms = trigger_candidate_source_ms if trigger else current_candidate_source_ms
    now_ms = int(time.time() * 1000)
    age_seconds = "" if candidate_source_ms is None else max(0.0, (now_ms - candidate_source_ms) / 1000.0)
    current_candidate_age_seconds = (
        "" if current_candidate_source_ms is None else max(0.0, (now_ms - current_candidate_source_ms) / 1000.0)
    )
    trigger_candidate_age_seconds = (
        "" if trigger_candidate_source_ms is None else max(0.0, (now_ms - trigger_candidate_source_ms) / 1000.0)
    )
    reasons: list[str] = []
    handoff_latency_exceeded = (
        handoff_phase == "post_open_orders_inline_reprice"
        and age_seconds != ""
        and float(age_seconds) > max_age_seconds
    )
    if decision.get("allowed") is not True:
        decision_reason = str(decision.get("skip_reason") or "fresh_touch_decision_not_allowed")
        if not handoff_latency_exceeded:
            reasons.append(decision_reason)
    if age_seconds == "":
        reasons.append("missing_selected_candidate_source_time")
    elif handoff_latency_exceeded:
        reasons.append("post_open_orders_handoff_latency_exceeded")
    elif float(age_seconds) > max_age_seconds:
        reasons.append("trigger_candidate_stale_before_order")
    intent_limit_px = safe_float(decision.get("intent_limit_px"))
    intent_size_btc = safe_float(decision.get("intent_size_btc"))
    quality_bucket = str(decision.get("quality_bucket") or current_candidate.get("quality_bucket") or "")
    current_top_depth_multiple = math.inf
    if not handoff_latency_exceeded:
        if intent_limit_px is None:
            reasons.append("missing_intent_limit_px")
        elif intent_limit_px != bid:
            reasons.append("selected_quote_not_current_touch")
        elif intent_limit_px >= ask:
            reasons.append("post_only_buy_would_cross_current_ask")
        if intent_size_btc is None or intent_size_btc <= 0:
            reasons.append("missing_or_nonpositive_intent_size")
        elif intent_size_btc > max_order_size_btc or intent_size_btc > FRESH_TOUCH_HARD_CAP_BTC:
            reasons.append("intent_size_exceeds_fresh_touch_cap")
        else:
            current_top_depth_multiple = buy_top_qty / intent_size_btc
        if buy_order_count is None:
            reasons.append("missing_current_same_side_top_order_count")
        elif quality_bucket == "quality_a":
            if current_top_depth_multiple > FRESH_TOUCH_QUALITY_A_MAX_DEPTH_MULTIPLE:
                reasons.append("current_top_depth_outside_quality_a_band")
            if buy_order_count > FRESH_TOUCH_QUALITY_A_MAX_ORDER_COUNT:
                reasons.append("current_top_order_count_outside_quality_a_band")
        elif quality_bucket == "quality_b":
            if current_top_depth_multiple <= FRESH_TOUCH_QUALITY_A_MAX_DEPTH_MULTIPLE or current_top_depth_multiple > FRESH_TOUCH_QUALITY_B_MAX_DEPTH_MULTIPLE:
                reasons.append("current_top_depth_outside_quality_b_band")
            if buy_order_count > FRESH_TOUCH_QUALITY_B_MAX_ORDER_COUNT:
                reasons.append("current_top_order_count_outside_quality_b_band")
        else:
            reasons.append("missing_quality_bucket")
    tif = executor.POST_ONLY_TIF
    status = "pass" if not reasons else "fail_closed"
    reason = ";".join(reasons)
    return {
        "attempt": "",
        "status": status,
        "reason": reason,
        "handoff_phase": handoff_phase,
        "candidate_source_exchange_time_ms": "" if candidate_source_ms is None else candidate_source_ms,
        "candidate_age_seconds": "" if age_seconds == "" else round(float(age_seconds), 6),
        "max_age_seconds": max_age_seconds,
        "trigger_candidate_source_exchange_time_ms": "" if trigger_candidate_source_ms is None else trigger_candidate_source_ms,
        "trigger_candidate_age_seconds": "" if trigger_candidate_age_seconds == "" else round(float(trigger_candidate_age_seconds), 6),
        "trigger_candidate_side": trigger.get("side", ""),
        "trigger_candidate_quote_px": trigger.get("quote_px", ""),
        "trigger_candidate_size_btc": trigger.get("dynamic_size_btc", "") or trigger.get("order_size_btc", ""),
        "trigger_candidate_quality_bucket": trigger.get("quality_bucket", ""),
        "trigger_candidate_freshness_status": trigger.get("freshness_status", "") or trigger.get("fresh_touch_evidence_status", ""),
        "trigger_candidate_skip_reason": trigger.get("skip_reason", ""),
        "current_reprice_allowed": decision.get("allowed", ""),
        "current_reprice_skip_reason": decision.get("skip_reason", ""),
        "current_reprice_candidate_source_exchange_time_ms": "" if current_candidate_source_ms is None else current_candidate_source_ms,
        "current_reprice_candidate_age_seconds": "" if current_candidate_age_seconds == "" else round(float(current_candidate_age_seconds), 6),
        "selected_side": decision.get("selected_side", ""),
        "selected_quote_px": "" if intent_limit_px is None else intent_limit_px,
        "current_bid": bid,
        "current_ask": ask,
        "selected_size_btc": "" if intent_size_btc is None else intent_size_btc,
        "max_order_size_btc": max_order_size_btc,
        "quality_bucket": quality_bucket,
        "current_same_side_top_qty_btc": buy_top_qty,
        "current_same_side_top_order_count": "" if buy_order_count is None else buy_order_count,
        "current_top_depth_multiple_of_order": "" if not math.isfinite(current_top_depth_multiple) else round(current_top_depth_multiple, 8),
        "post_only_tif": tif,
        "post_only_non_crossing": intent_limit_px is not None and intent_limit_px < ask,
        "current_touch_match": intent_limit_px is not None and intent_limit_px == bid,
        "source": "same_process_public_watcher_current_candidate_guard",
    }


def select_flow_aware_side(
    *,
    l2_snapshot: dict[str, Any],
    precision: executor.PrecisionFacts,
    quote_offset_ticks: int,
    window_id: int,
    attempt_id: int,
    max_order_size_btc: float,
    max_top_depth_multiple: float = DEFAULT_FLOW_MAX_TOP_DEPTH_MULTIPLE,
    public_flow_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bid, ask = best_bid_ask(l2_snapshot)
    size_cap = min(max_order_size_btc, executor.MAX_ORDER_NOTIONAL_USDC / max(bid, 1.0))
    size = floor_to_lot(size_cap, precision.lot_size)
    if size <= 0:
        return {
            "policy_version": FLOW_AWARE_POLICY_VERSION,
            "allowed": False,
            "selected_side": "",
            "skip_reason": "computed_order_size_nonpositive",
            "scores": {},
        }
    scores = flow_side_scores(
        l2_snapshot=l2_snapshot,
        order_size_btc=size,
        max_top_depth_multiple=max_top_depth_multiple,
        public_flow_summary=public_flow_summary,
    )
    buy = scores["buy"]
    sell = scores["sell"]
    selected = "buy" if buy["score"] >= sell["score"] else "sell"
    selected_score = scores[selected]
    skip_reason = ""
    allowed = selected_score["guard_status"] == "pass"
    if selected == "sell" and int(sell.get("public_flow_candidate_count", 0) or 0) <= 0:
        allowed = False
        skip_reason = "sell_without_public_flow_support"
    elif selected == "sell" and sell["score"] <= buy["score"] + 0.20:
        allowed = False
        skip_reason = "sell_not_materially_better_than_buy"
    elif not allowed:
        skip_reason = str(selected_score["guard_status"])
    intent: executor.OrderIntent | None = None
    if allowed:
        try:
            intent = build_top_of_book_maker_intent(
                precision=precision,
                bid=bid,
                ask=ask,
                quote_offset_ticks=quote_offset_ticks,
                window_id=window_id,
                is_buy=selected == "buy",
                attempt_id=attempt_id,
            )
        except Exception as exc:
            allowed = False
            skip_reason = executor._redacted_error(exc)
    return {
        "policy_version": FLOW_AWARE_POLICY_VERSION,
        "allowed": allowed,
        "selected_side": selected if allowed else "",
        "skip_reason": skip_reason,
        "bid": bid,
        "ask": ask,
        "spread": ask - bid,
        "candidate_order_size_btc": size,
        "max_top_depth_multiple": max_top_depth_multiple,
        "scores": scores,
        "intent_limit_px": "" if intent is None else intent.limit_px,
        "intent_size_btc": "" if intent is None else intent.size_btc,
        "inference_scope": "public_l2_top_depth_proxy_not_exact_queue_or_fill_probability",
    }


def run_public_flow_precheck(
    *,
    output_dir: Path,
    order_size_btc: float,
    quote_hold_seconds: int,
    duration_seconds: float,
    candidate_stride_seconds: float | None = None,
) -> dict[str, Any]:
    manifest_path = output_dir / "public_flow_precheck_manifest.json"
    if duration_seconds <= 0:
        manifest = {
            "status": "not_requested",
            "reason": "duration_seconds_lte_zero",
            "public_market_data_only": True,
            "no_private_or_order_endpoint": True,
        }
        write_json(manifest_path, manifest)
        return manifest
    try:
        collection_dir = output_dir / "public_flow_precheck_collection"
        collection = hyperliquid_public_sample.collect_sample(
            coin=executor.SYMBOL,
            channels=["l2Book", "trades"],
            duration_seconds=duration_seconds,
            output_dir=collection_dir,
            network="mainnet",
            ws_url=hyperliquid_public_sample.MAINNET_WS_URL,
            info_url=hyperliquid_public_sample.MAINNET_INFO_URL,
            request_timeout=10.0,
            websocket_timeout=5.0,
            max_reconnects=2,
            task_id=TASK_ID,
        )
        raw_path = Path(str(collection.get("raw_file", ""))).resolve()
        diagnosis_dir = output_dir / "public_flow_precheck_diagnosis"
        diagnosis = public_flow.run_diagnosis(
            output_dir=diagnosis_dir,
            raw_input=raw_path,
            order_size=Decimal(str(order_size_btc)),
            quote_hold_seconds=float(quote_hold_seconds),
            candidate_stride_seconds=candidate_stride_seconds if candidate_stride_seconds is not None else max(1.0, min(5.0, duration_seconds / 4.0)),
        )
        summary = diagnosis.get("summary", {})
        manifest = {
            "status": "pass" if int(summary.get("candidate_count", 0) or 0) > 0 else "inconclusive",
            "reason": "" if int(summary.get("candidate_count", 0) or 0) > 0 else "no_public_flow_candidates",
            "collection_manifest": collection,
            "diagnosis_manifest": diagnosis,
            "summary": summary,
            "public_market_data_only": True,
            "no_private_or_order_endpoint": True,
        }
    except Exception as exc:
        manifest = {
            "status": "fail_closed",
            "reason": executor._redacted_error(exc),
            "public_market_data_only": True,
            "no_private_or_order_endpoint": True,
        }
    write_json(manifest_path, manifest)
    return manifest


def write_preorder_blocked_artifacts(
    *,
    output_dir: Path,
    env_file: Path,
    window_id: int,
    side_policy: str,
    blocking_reasons: list[str],
    public_flow_precheck: dict[str, Any],
    max_order_size: float,
    quote_hold_seconds: int,
    requote_attempts: int,
    flow_max_top_depth_multiple: float,
    flow_max_lost_touch_ticks: float,
) -> dict[str, Any]:
    window_label = artifact_window_label(window_id)
    write_json(
        output_dir / "run_intent_marker.json",
        {
            "task_id": TASK_ID,
            "window_id": window_label,
            "artifact_window_id": window_id,
            "real_orders_allowed": False,
            "post_only_required": True,
        },
    )
    write_json(output_dir / "credential_source_manifest.json", {"env_file": str(env_file), "env_file_keys_loaded": [], "candidate_keys_present": [], "secret_values_written": False})
    write_json(output_dir / "private_preflight_summary.json", {"preflight_summary": {}, "open_orders_before": [], "endpoint_called": False})
    write_csv(output_dir / "precision_tick_lot_snapshot.csv", [], ["symbol", "sz_decimals", "tick_size", "lot_size", "mid_px", "source"])
    write_csv(output_dir / "order_intent_audit.csv", [], ["symbol", "side", "size_btc", "limit_px", "notional_usdc", "time_in_force", "order_type", "reduce_only", "endpoint_called", "cloid_redacted"])
    write_csv(
        output_dir / "quote_attempt_matrix.csv",
        [],
        ["attempt", "window_id", "attempt_id", "attempt_key", "side", "limit_px", "size_btc", "bid", "ask", "post_only_tif", "order_status_types", "fill_count_after_attempt", "crossing_guard_status", "flow_guard_status", "fresh_touch_quality_bucket", "dynamic_size_btc", "quote_hold_seconds", "skip_reason", "quote_aging_guard_status", "quote_aging_guard_reason"],
    )
    write_csv(
        output_dir / "flow_side_score_matrix.csv",
        [],
        ["attempt", "side", "score", "same_side_top_qty_btc", "same_side_top_order_count", "top_depth_multiple_of_order", "public_flow_candidate_count", "public_flow_strict_rate", "public_flow_depletion_rate", "public_flow_aging_rate", "guard_status", "selected", "allowed", "skip_reason", "evidence_scope"],
    )
    write_csv(output_dir / "quote_aging_guard_matrix.csv", [], ["attempt", "status", "reason", "side", "pre_bid", "pre_ask", "post_bid", "post_ask", "limit_px", "lost_touch_ticks", "max_lost_touch_ticks"])
    write_csv(
        output_dir / "touch_freshness_matrix.csv",
        [],
        ["attempt", "candidate_index", "side", "status", "age_seconds", "reason", "source_start_exchange_time_ms", "source_quote_aging_status", "source_first_touch_trade_ms", "source_first_strict_trade_through_ms", "selected"],
    )
    write_csv(
        output_dir / "dynamic_size_decision_matrix.csv",
        [],
        ["attempt", "candidate_index", "side", "quality_bucket", "bucket_cap_btc", "hard_cap_btc", "recent_same_side_at_or_through_qty_btc_last_3s", "raw_size_btc", "floored_size_btc", "status", "reason", "candidate_allowed", "selected"],
    )
    write_csv(
        output_dir / "session_side_eligibility.csv",
        [],
        ["side", "default_policy", "eligible", "same_window_public_support", "materially_favors_sell", "reason"],
    )
    write_csv(
        output_dir / "time_gate_decision_matrix.csv",
        [],
        ["gate", "utc_hour", "eligible", "fixed_hour_allowlist_used", "precheck_status", "reason"],
    )
    write_json(output_dir / "private_order_response_audit.json", {"real_order_endpoint_called": False, "order_submission_attempted": False, "order_status_rows": [], "order_result": None, "blocking_reasons": blocking_reasons})
    write_json(output_dir / "account_inventory_snapshots.json", {"pre_state": {}, "post_state": {}, "user_fees": {}})
    write_json(output_dir / "user_fills_pullback_audit.json", {"pullbacks": [], "pullback_count": 0, "raw_payload_redacted": True})
    write_json(output_dir / "market_markout_snapshot.json", {"pre_l2": {}, "post_l2": {}})
    write_csv(output_dir / "live_fill_ledger.csv", [], live_fill_ledger_fieldnames())
    write_csv(output_dir / "fill_liquidity_role_evidence.csv", [], fill_liquidity_role_evidence_fieldnames())
    write_json(output_dir / "cancel_shutdown_proof.json", {"real_cancel_endpoint_called": False, "tracked_refs": [], "cancel_results": [], "final_open_orders": [], "proof_status": "no_order_submitted"})
    write_json(output_dir / "max_loss_monitor_summary.json", {"status": "not_evaluated", "reason": "blocked_before_order"})
    manifest = {
        "task_id": TASK_ID,
        "policy_version": policy_version_for_side_policy(side_policy),
        "window_id": window_label,
        "artifact_window_id": window_id,
        "requote_attempts_requested": requote_attempts,
        "requote_attempts_completed": 0,
        "side_policy": side_policy,
        "max_order_size_btc": max_order_size,
        "quote_hold_seconds": quote_hold_seconds,
        "flow_max_top_depth_multiple": flow_max_top_depth_multiple,
        "flow_max_lost_touch_ticks": flow_max_lost_touch_ticks,
        "public_flow_precheck_status": public_flow_precheck.get("status", ""),
        "final_recommendation": BLOCKED_RECOMMENDATION,
        "blocking_reasons": blocking_reasons,
        "order_status_types": [],
        "fill_count": 0,
        "maker_fill_count": 0,
        "ledger_fill_rows": 0,
        "real_order_endpoint_called": False,
        "private_endpoint_called": False,
        "real_cancel_endpoint_called": False,
        "final_open_orders_count": 0,
        "shutdown_proof_status": "no_order_submitted",
        "post_only_tif": executor.POST_ONLY_TIF,
        "crossing_guard_status": "not_submitted",
        "flow_guard_status": "public_flow_precheck_blocked",
        "fresh_touch_guard_status": "public_flow_precheck_blocked" if side_policy == "fresh_touch" else "not_applicable",
        "credentials_written": False,
        "secret_values_written": False,
        "raw_signatures_written": False,
        "git_commit": executor.git_commit(),
    }
    write_json(output_dir / "m2_fill_window_manifest.json", manifest)
    write_json(
        output_dir / "executor_manifest.json",
        {
            "task_id": TASK_ID,
            "window_id": window_label,
            "artifact_window_id": window_id,
            "order_submission_attempted": False,
            "private_endpoint_called": False,
            "real_order_endpoint_called": False,
            "real_cancel_endpoint_called": False,
            "shutdown_proof_status": "no_order_submitted",
            "credentials_written": False,
            "secret_values_written": False,
            "raw_signatures_written": False,
            "final_recommendation": BLOCKED_RECOMMENDATION,
        },
    )
    (output_dir / "README.md").write_text("# Hyperliquid M2 Flow-Aware Fill Window\n\nBlocked before live order submission.\n", encoding="utf-8")
    return manifest


def quote_aging_guard(
    *,
    intent: executor.OrderIntent,
    pre_bid: float,
    pre_ask: float,
    post_bid: float,
    post_ask: float,
    tick_size: float,
    max_lost_touch_ticks: float = DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS,
) -> dict[str, Any]:
    lost_touch_ticks = 0.0
    adverse = False
    if intent.is_buy:
        lost_touch_ticks = max(0.0, (intent.limit_px - post_bid) / tick_size) if tick_size > 0 else math.inf
        adverse = post_bid < pre_bid or intent.limit_px > post_bid
    else:
        lost_touch_ticks = max(0.0, (post_ask - intent.limit_px) / tick_size) if tick_size > 0 else math.inf
        adverse = post_ask > pre_ask or intent.limit_px < post_ask
    status = "pass"
    reason = ""
    if lost_touch_ticks > max_lost_touch_ticks:
        status = "cancel_requote"
        reason = "lost_touch"
    if adverse:
        status = "cancel_requote"
        reason = "adverse_drift" if not reason else reason + "+adverse_drift"
    return {
        "status": status,
        "reason": reason,
        "side": "buy" if intent.is_buy else "sell",
        "pre_bid": pre_bid,
        "pre_ask": pre_ask,
        "post_bid": post_bid,
        "post_ask": post_ask,
        "limit_px": intent.limit_px,
        "lost_touch_ticks": round(lost_touch_ticks, 8),
        "max_lost_touch_ticks": max_lost_touch_ticks,
    }


def extract_tracked_oids(order_result: dict[str, Any]) -> set[str]:
    refs = executor.extract_tracked_refs(order_result)
    return {str(ref.get("oid")) for ref in refs if ref.get("oid") is not None}


def live_fill_ledger_fieldnames() -> list[str]:
    return [
        "source_window",
        "window_id",
        "attempt_id",
        "attempt_key",
        "fill_id",
        "side",
        "qty_btc",
        "price_usdc",
        "intent_price_usdc",
        "mark_price_usdc",
        "fee_usdc",
        "rebate_usdc",
        "liquidity",
        "attribution_status",
        "attribution_source",
        "source_oid_present",
        "source_has_liquidity_role",
        "fill_time_ms",
    ]


def fill_liquidity_role_evidence_fieldnames() -> list[str]:
    return [
        "source_window",
        "window_id",
        "attempt_id",
        "attempt_key",
        "fill_id",
        "liquidity",
        "liquidity_role_status",
        "liquidity_role_source",
        "source_has_liquidity_role",
        "source_oid_present",
        "attribution_status",
        "fee_pnl_role_gate",
    ]


def fill_liquidity_role_evidence_rows(fill_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fill in fill_rows:
        liquidity = str(fill.get("liquidity") or "unknown").lower()
        has_role = fill.get("source_has_liquidity_role") is True or str(fill.get("source_has_liquidity_role", "")).lower() == "true"
        if liquidity == "maker" and has_role:
            role_status = "confirmed_maker"
            role_gate = "pass_role_known"
        elif liquidity == "taker" and has_role:
            role_status = "confirmed_taker"
            role_gate = "pass_role_known_but_not_maker"
        else:
            role_status = "unknown_liquidity_role"
            role_gate = "block_unknown_liquidity_role"
        rows.append(
            {
                "source_window": fill.get("source_window", ""),
                "window_id": fill.get("window_id", ""),
                "attempt_id": fill.get("attempt_id", ""),
                "attempt_key": fill.get("attempt_key", ""),
                "fill_id": fill.get("fill_id", ""),
                "liquidity": liquidity,
                "liquidity_role_status": role_status,
                "liquidity_role_source": "user_fills_by_time_crossed_or_liquidity_field" if has_role else "missing_in_source_payload",
                "source_has_liquidity_role": has_role,
                "source_oid_present": fill.get("source_oid_present", ""),
                "attribution_status": fill.get("attribution_status", ""),
                "fee_pnl_role_gate": role_gate,
            }
        )
    return rows


def side_from_fill(fill: dict[str, Any]) -> str:
    side = str(fill.get("side", "")).upper()
    if side == "B":
        return "buy"
    if side == "A":
        return "sell"
    direction = str(fill.get("dir", "")).lower()
    if "buy" in direction or "long" in direction:
        return "buy"
    if "sell" in direction or "short" in direction:
        return "sell"
    return "unknown"


def symbol_from_fill(fill: dict[str, Any]) -> str:
    return str(fill.get("coin") or fill.get("symbol") or "").upper()


def liquidity_from_fill(fill: dict[str, Any]) -> tuple[str, bool]:
    if "crossed" in fill:
        return ("taker" if bool(fill.get("crossed")) else "maker"), True
    if "liquidity" in fill:
        value = str(fill.get("liquidity") or "").lower()
        if value in {"maker", "taker"}:
            return value, True
    return "unknown", False


def fill_matches_intent_without_oid(
    fill: dict[str, Any],
    *,
    intent: executor.OrderIntent,
    side: str,
    qty: float,
    price: float,
    attributed_qty: float,
) -> bool:
    fill_symbol = symbol_from_fill(fill)
    if fill_symbol and fill_symbol != intent.symbol.upper():
        return False
    if side != ("buy" if intent.is_buy else "sell"):
        return False
    if abs(price - intent.limit_px) > 1e-9:
        return False
    return attributed_qty + qty <= intent.size_btc + 1e-12


def live_fill_rows(
    *,
    fills: list[dict[str, Any]],
    tracked_oids: set[str],
    intent: executor.OrderIntent,
    mark_px: float,
    window_id: int,
    user_add_rate: float,
    attempt_id: int = 1,
    task_id: str = TASK_ID,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    attributed_qty = 0.0
    window_label = artifact_window_label(window_id)
    attempt_key = artifact_attempt_key(task_id=task_id, window_id=window_id, attempt_id=attempt_id)
    for idx, fill in enumerate(fills, start=1):
        side = side_from_fill(fill)
        if side not in {"buy", "sell"}:
            continue
        qty = float(fill.get("sz", 0.0))
        price = float(fill.get("px", 0.0))
        if qty <= 0 or price <= 0:
            continue
        fill_oid = fill.get("oid")
        oid_matches = fill_oid is not None and str(fill_oid) in tracked_oids
        fallback_matches = fill_matches_intent_without_oid(
            fill,
            intent=intent,
            side=side,
            qty=qty,
            price=price,
            attributed_qty=attributed_qty,
        )
        if tracked_oids and not oid_matches and not fallback_matches:
            continue
        if not tracked_oids and not fallback_matches:
            continue
        attribution_status = "matched_tracked_oid" if oid_matches else "matched_price_size_without_oid"
        attribution_source = "user_fills_by_time_oid" if oid_matches else "user_fills_by_time_price_size_fallback"
        attributed_qty += qty
        liquidity, has_liquidity_role = liquidity_from_fill(fill)
        fee = abs(float(fill.get("fee", 0.0))) if fill.get("fee") not in ("", None) else abs(qty * price * user_add_rate)
        rows.append(
            {
                "source_window": window_label,
                "window_id": window_label,
                "attempt_id": attempt_id,
                "attempt_key": attempt_key,
                "fill_id": "fill_sha256_" + hashlib.sha256(str(fill).encode()).hexdigest()[:12] if fill.get("hash") else f"{window_label}_attempt_{attempt_id}_fill_{idx}",
                "side": side,
                "qty_btc": qty,
                "price_usdc": price,
                "intent_price_usdc": intent.limit_px,
                "mark_price_usdc": mark_px,
                "fee_usdc": fee,
                "rebate_usdc": 0.0,
                "liquidity": liquidity,
                "attribution_status": attribution_status,
                "attribution_source": attribution_source,
                "source_oid_present": fill_oid is not None,
                "source_has_liquidity_role": has_liquidity_role,
                "fill_time_ms": fill.get("time", ""),
            }
        )
    return rows


def cancel_result_mentions_filled(cancel_results: list[dict[str, Any]]) -> bool:
    for result in cancel_results:
        text = json.dumps(result, sort_keys=True).lower()
        if "already canceled, or filled" in text or "already cancelled, or filled" in text:
            return True
    return False


def run_window(
    *,
    output_dir: Path,
    env_file: Path,
    window_id: int,
    wait_seconds: int,
    quote_offset_ticks: int,
    requote_attempts: int = 1,
    quote_hold_seconds: int | None = None,
    side_policy: str = "buy",
    max_order_size: float = 0.00999,
    flow_max_top_depth_multiple: float = DEFAULT_FLOW_MAX_TOP_DEPTH_MULTIPLE,
    flow_max_lost_touch_ticks: float = DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS,
    fresh_touch_precheck_seconds: float = DEFAULT_FRESH_TOUCH_PRECHECK_SECONDS,
    public_flow_precheck_override: dict[str, Any] | None = None,
    selected_candidate_context: dict[str, Any] | None = None,
    same_process_trigger: bool = False,
    immediate_guard_max_age_seconds: float = FRESH_TOUCH_MAX_IMMEDIATE_GUARD_AGE_SECONDS,
    fast_event_driven_submit: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if wait_seconds < 1 or wait_seconds > MAX_WAIT_SECONDS:
        raise executor.ValidationError("wait_seconds_outside_approved_duration")
    if requote_attempts < 1:
        raise executor.ValidationError("requote_attempts_must_be_positive")
    hold_seconds = quote_hold_seconds if quote_hold_seconds is not None else wait_seconds
    if hold_seconds < 1:
        raise executor.ValidationError("quote_hold_seconds_must_be_positive")
    if requote_attempts * hold_seconds > MAX_WAIT_SECONDS:
        raise executor.ValidationError("adaptive_window_exceeds_approved_duration")
    if max_order_size <= 0 or max_order_size > executor.MAX_ORDER_SIZE_BTC:
        raise executor.ValidationError("max_order_size_outside_approved_cap")
    if side_policy == "flow_aware" and hold_seconds > FLOW_SAFE_HOLD_SECONDS:
        raise executor.ValidationError("flow_aware_quote_hold_seconds_too_long")
    if side_policy == "fresh_touch":
        if quote_offset_ticks != 0:
            raise executor.ValidationError("fresh_touch_requires_quote_offset_ticks_zero")
        if max_order_size > FRESH_TOUCH_HARD_CAP_BTC:
            raise executor.ValidationError("fresh_touch_max_order_size_exceeds_0_005_btc")
        if hold_seconds > FRESH_TOUCH_QUALITY_A_HOLD_SECONDS:
            raise executor.ValidationError("fresh_touch_quote_hold_seconds_too_long")
        if requote_attempts > 2:
            raise executor.ValidationError("fresh_touch_requote_attempts_exceeds_two_submission_cap")
    public_flow_precheck: dict[str, Any] = {"status": "not_applicable", "summary": {}}
    if public_flow_precheck_override is not None:
        public_flow_precheck = dict(public_flow_precheck_override)
        public_flow_precheck["same_process_override_used"] = True
    elif side_policy in {"flow_aware", "fresh_touch"}:
        public_flow_precheck = run_public_flow_precheck(
            output_dir=output_dir,
            order_size_btc=max_order_size,
            quote_hold_seconds=FRESH_TOUCH_QUALITY_A_HOLD_SECONDS if side_policy == "fresh_touch" else hold_seconds,
            duration_seconds=fresh_touch_precheck_seconds if side_policy == "fresh_touch" else DEFAULT_FLOW_PRECHECK_SECONDS,
            candidate_stride_seconds=DEFAULT_FRESH_TOUCH_CANDIDATE_STRIDE_SECONDS if side_policy == "fresh_touch" else None,
        )
        if public_flow_precheck.get("status") != "pass":
            return write_preorder_blocked_artifacts(
                output_dir=output_dir,
                env_file=env_file,
                window_id=window_id,
                side_policy=side_policy,
                blocking_reasons=[f"public_flow_precheck_{public_flow_precheck.get('status')}:{public_flow_precheck.get('reason','')}"],
                public_flow_precheck=public_flow_precheck,
                max_order_size=max_order_size,
                quote_hold_seconds=hold_seconds,
                requote_attempts=requote_attempts,
                flow_max_top_depth_multiple=flow_max_top_depth_multiple,
                flow_max_lost_touch_ticks=flow_max_lost_touch_ticks,
            )

    blocking_reasons: list[str] = []
    order_result: dict[str, Any] | None = None
    cancel_results: list[dict[str, Any]] = []
    tracked_refs: list[dict[str, Any]] = []
    final_open_orders: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    order_status_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    side_score_rows: list[dict[str, Any]] = []
    quote_guard_rows: list[dict[str, Any]] = []
    immediate_guard_rows: list[dict[str, Any]] = []
    touch_freshness_rows: list[dict[str, Any]] = []
    dynamic_size_rows: list[dict[str, Any]] = []
    session_side_rows: list[dict[str, Any]] = []
    time_gate_rows: list[dict[str, Any]] = []
    pre_state: dict[str, Any] = {}
    post_state: dict[str, Any] = {}
    user_fees: dict[str, Any] = {}
    pre_open_orders: list[dict[str, Any]] = []
    post_l2: dict[str, Any] = {}
    intent: executor.OrderIntent | None = None
    loss: dict[str, Any] = {"status": "not_evaluated", "reason": "no_order_submitted"}
    endpoint_flags = {
        "private_endpoint_called": False,
        "real_order_endpoint_called": False,
        "real_cancel_endpoint_called": False,
    }
    user_fills_pullbacks: list[dict[str, Any]] = []
    pre_user_state_deferred = False
    user_fees_deferred = False
    user_fees_pullback_attempted = False
    last_submitted_attempt_id = 1

    def pull_user_fees_after_submit_once() -> None:
        nonlocal user_fees, user_add_rate, user_fees_pullback_attempted
        if not fast_event_driven_submit or user_fees or user_fees_pullback_attempted:
            return
        if endpoint_flags["real_order_endpoint_called"] is not True:
            return
        user_fees_pullback_attempted = True
        try:
            user_fees = client.info.user_fees(client.account_address)
            user_add_rate = float(user_fees.get("userAddRate", 0.0) or 0.0)
        except Exception as exc:
            blocking_reasons.append(f"post_submit_user_fees_pullback_failed:{executor._redacted_error(exc)}")

    env_load = executor.load_env_file(env_file)
    client = executor.build_live_client_from_env()
    if client is None:
        raise executor.ValidationError("live_client_unavailable")
    endpoint_flags["private_endpoint_called"] = True
    start_ms = int(time.time() * 1000) - 2_000
    pre_l2 = client.info.l2_snapshot(executor.SYMBOL)
    if fast_event_driven_submit:
        precision = precision_from_l2_public_snapshot(pre_l2)
        pre_user_state_deferred = True
        user_fees_deferred = True
    else:
        precision = executor.fetch_live_precision(client)
    config = executor.TinyLiveConfig(
        artifact_dir=output_dir,
        live_mode=True,
        operator_ack=OPERATOR_ACK,
        use_schedule_cancel=False,
        max_order_size_btc=max_order_size,
    )
    executor.assert_config_valid(config, precision)
    pre_open_orders = client.open_orders()
    if pre_open_orders:
        raise executor.ValidationError("pre_existing_open_orders_present")
    if fast_event_driven_submit:
        user_add_rate = 0.0
    else:
        pre_state = client.user_state()
        user_fees = client.info.user_fees(client.account_address)
        user_add_rate = float(user_fees.get("userAddRate", 0.0) or 0.0)
    if side_policy == "fresh_touch":
        precheck_summary = public_flow_precheck.get("summary", {})
        by_side = precheck_summary.get("by_side", {})
        buy_support = by_side.get("buy", {})
        sell_support = by_side.get("sell", {})
        buy_candidate_count = int(buy_support.get("candidate_count", 0) or 0)
        buy_strict_count = int(buy_support.get("strict_trade_through_candidate_count", 0) or 0)
        sell_candidate_count = int(sell_support.get("candidate_count", 0) or 0)
        sell_depletion = int(sell_support.get("public_depletion_candidate_count", 0) or 0)
        buy_depletion = int(buy_support.get("public_depletion_candidate_count", 0) or 0)
        session_side_rows.extend(
            [
                {
                    "side": "buy",
                    "default_policy": "buy_only",
                    "eligible": buy_candidate_count > 0 and buy_strict_count > 0,
                    "same_window_public_support": f"candidate_count={buy_candidate_count};strict_through={buy_strict_count}",
                    "materially_favors_sell": False,
                    "reason": "" if buy_candidate_count > 0 and buy_strict_count > 0 else "missing_buy_same_window_strict_through_support",
                },
                {
                    "side": "sell",
                    "default_policy": "buy_only",
                    "eligible": False,
                    "same_window_public_support": f"candidate_count={sell_candidate_count};public_depletion={sell_depletion};buy_public_depletion={buy_depletion}",
                    "materially_favors_sell": False,
                    "reason": "sell_disabled_in_0622T001_without_later_material_scorecard",
                },
            ]
        )
        time_gate_rows.append(
            {
                "gate": "current_public_precheck_micro_window",
                "utc_hour": ",".join(str(value) for value in precheck_summary.get("utc_hours", [])),
                "eligible": public_flow_precheck.get("status") == "pass",
                "fixed_hour_allowlist_used": False,
                "precheck_status": public_flow_precheck.get("status", ""),
                "reason": "" if public_flow_precheck.get("status") == "pass" else str(public_flow_precheck.get("reason", "")),
            }
        )

    try:
        for attempt_id in range(1, requote_attempts + 1):
            attempt_l2 = client.info.l2_snapshot(executor.SYMBOL)
            bid, ask = best_bid_ask(attempt_l2)
            flow_decision = None
            fresh_touch_decision = None
            skip_reason = ""
            if side_policy == "flow_aware":
                flow_decision = select_flow_aware_side(
                    l2_snapshot=attempt_l2,
                    precision=precision,
                    quote_offset_ticks=quote_offset_ticks,
                    window_id=window_id,
                    attempt_id=attempt_id,
                    max_order_size_btc=max_order_size,
                    max_top_depth_multiple=flow_max_top_depth_multiple,
                    public_flow_summary=public_flow_precheck.get("summary", {}),
                )
                for side, score in flow_decision.get("scores", {}).items():
                    side_score_rows.append(
                        {
                            "attempt": attempt_id,
                            "side": side,
                            "score": score.get("score", ""),
                            "same_side_top_qty_btc": score.get("same_side_top_qty_btc", ""),
                            "same_side_top_order_count": score.get("same_side_top_order_count", ""),
                            "top_depth_multiple_of_order": score.get("top_depth_multiple_of_order", ""),
                            "public_flow_candidate_count": score.get("public_flow_candidate_count", ""),
                            "public_flow_strict_rate": score.get("public_flow_strict_rate", ""),
                            "public_flow_depletion_rate": score.get("public_flow_depletion_rate", ""),
                            "public_flow_aging_rate": score.get("public_flow_aging_rate", ""),
                            "guard_status": score.get("guard_status", ""),
                            "selected": side == flow_decision.get("selected_side"),
                            "allowed": flow_decision.get("allowed", False),
                            "skip_reason": flow_decision.get("skip_reason", ""),
                            "evidence_scope": score.get("evidence_scope", ""),
                        }
                    )
                if flow_decision.get("allowed") is not True:
                    skip_reason = str(flow_decision.get("skip_reason") or "flow_guard_rejected_candidate")
                    attempt_rows.append(
                        {
                            "attempt": attempt_id,
                            "side": "",
                            "limit_px": "",
                            "size_btc": "",
                            "bid": bid,
                            "ask": ask,
                            "post_only_tif": executor.POST_ONLY_TIF,
                            "order_status_types": "skipped",
                            "fill_count_after_attempt": len(fill_rows),
                            "crossing_guard_status": "not_submitted",
                            "flow_guard_status": "skip",
                            "fresh_touch_quality_bucket": "",
                            "dynamic_size_btc": "",
                            "quote_hold_seconds": "",
                            "skip_reason": skip_reason,
                            "quote_aging_guard_status": "not_submitted",
                            "quote_aging_guard_reason": "",
                        }
                    )
                    continue
            if side_policy == "fresh_touch":
                fresh_touch_decision = select_fresh_touch_candidate(
                    l2_snapshot=attempt_l2,
                    precision=precision,
                    window_id=window_id,
                    attempt_id=attempt_id,
                    public_flow_precheck=public_flow_precheck,
                    max_order_size_btc=max_order_size,
                )
                for candidate in fresh_touch_decision.get("candidate_rows", []):
                    candidate_index = candidate.get("candidate_index", "")
                    touch_freshness_rows.append(
                        {
                            "attempt": attempt_id,
                            "candidate_index": candidate_index,
                            "side": candidate.get("side", ""),
                            "status": candidate.get("freshness_status", ""),
                            "age_seconds": candidate.get("freshness_age_seconds", ""),
                            "reason": candidate.get("freshness_reason", ""),
                            "source_start_exchange_time_ms": candidate.get("source_start_exchange_time_ms", ""),
                            "source_quote_aging_status": candidate.get("source_quote_aging_status", ""),
                            "source_first_touch_trade_ms": candidate.get("source_first_touch_trade_ms", ""),
                            "source_first_strict_trade_through_ms": candidate.get("source_first_strict_trade_through_ms", ""),
                            "selected": candidate.get("selected", False),
                        }
                    )
                    dynamic_size_rows.append(
                        {
                            "attempt": attempt_id,
                            "candidate_index": candidate_index,
                            "side": candidate.get("side", ""),
                            "quality_bucket": candidate.get("quality_bucket", ""),
                            "bucket_cap_btc": candidate.get("bucket_cap_btc", ""),
                            "hard_cap_btc": min(FRESH_TOUCH_HARD_CAP_BTC, max_order_size),
                            "recent_same_side_at_or_through_qty_btc_last_3s": candidate.get("recent_same_side_at_or_through_qty_btc_last_3s", ""),
                            "raw_size_btc": candidate.get("raw_size_btc", ""),
                            "floored_size_btc": candidate.get("dynamic_size_btc", ""),
                            "status": candidate.get("dynamic_size_status", "pass" if candidate.get("allowed") else "skip"),
                            "reason": candidate.get("dynamic_size_reason", "") or candidate.get("skip_reason", ""),
                            "candidate_allowed": candidate.get("allowed", False),
                            "selected": candidate.get("selected", False),
                        }
                    )
                if fresh_touch_decision.get("allowed") is not True:
                    skip_reason = str(fresh_touch_decision.get("skip_reason") or "fresh_touch_session_gate_rejected_candidate")
                    attempt_rows.append(
                        {
                            "attempt": attempt_id,
                            "side": "",
                            "limit_px": "",
                            "size_btc": "",
                            "bid": bid,
                            "ask": ask,
                            "post_only_tif": executor.POST_ONLY_TIF,
                            "order_status_types": "skipped",
                            "fill_count_after_attempt": len(fill_rows),
                            "crossing_guard_status": "not_submitted",
                            "flow_guard_status": "skip",
                            "fresh_touch_quality_bucket": "",
                            "dynamic_size_btc": "",
                            "quote_hold_seconds": "",
                            "skip_reason": skip_reason,
                            "quote_aging_guard_status": "not_submitted",
                            "quote_aging_guard_reason": "",
                        }
                    )
                    continue
                selected_context = (
                    selected_candidate_context.get("fresh_touch_decision", {}).get("selected_candidate", {})
                    if selected_candidate_context
                    else {}
                )
                if not selected_context and selected_candidate_context:
                    selected_context = selected_candidate_context.get("candidate_source_row", {})
                immediate_guard = immediate_fresh_touch_guard(
                    selected_candidate=selected_context,
                    decision=fresh_touch_decision,
                    l2_snapshot=attempt_l2,
                    precision=precision,
                    max_order_size_btc=max_order_size,
                    max_age_seconds=immediate_guard_max_age_seconds,
                )
                immediate_guard["attempt"] = attempt_id
                immediate_guard_rows.append(immediate_guard)
                if same_process_trigger and immediate_guard["status"] != "pass":
                    skip_reason = str(immediate_guard["reason"] or "immediate_pre_submit_guard_failed")
                    attempt_rows.append(
                        {
                            "attempt": attempt_id,
                            "side": "",
                            "limit_px": "",
                            "size_btc": "",
                            "bid": bid,
                            "ask": ask,
                            "post_only_tif": executor.POST_ONLY_TIF,
                            "order_status_types": "skipped",
                            "fill_count_after_attempt": len(fill_rows),
                            "crossing_guard_status": "not_submitted",
                            "flow_guard_status": "skip",
                            "fresh_touch_quality_bucket": "",
                            "dynamic_size_btc": "",
                            "quote_hold_seconds": "",
                            "skip_reason": skip_reason,
                            "quote_aging_guard_status": "not_submitted",
                            "quote_aging_guard_reason": "",
                        }
                    )
                    continue
            attempt_hold_seconds = hold_seconds
            fresh_touch_quality_bucket = ""
            fresh_touch_dynamic_size = ""
            if side_policy == "fresh_touch":
                intent = executor.OrderIntent(
                    symbol=executor.SYMBOL,
                    is_buy=True,
                    size_btc=float(fresh_touch_decision.get("intent_size_btc") or 0.0),
                    limit_px=float(fresh_touch_decision.get("intent_limit_px") or bid),
                    time_in_force=executor.POST_ONLY_TIF,
                    reduce_only=False,
                    cloid=executor.generate_cloid(f"{TASK_ID}_w{window_id}_a{attempt_id}"),
                )
                attempt_hold_seconds = int(fresh_touch_decision.get("hold_seconds") or hold_seconds)
                fresh_touch_quality_bucket = str(fresh_touch_decision.get("quality_bucket", ""))
                fresh_touch_dynamic_size = intent.size_btc
            else:
                intent = build_top_of_book_maker_intent(
                    precision=precision,
                    bid=bid,
                    ask=ask,
                    quote_offset_ticks=quote_offset_ticks,
                    window_id=window_id,
                    is_buy=side_for_attempt(side_policy, attempt_id, flow_decision),
                    attempt_id=attempt_id,
                )
            if intent.size_btc > max_order_size:
                intent = executor.OrderIntent(
                    symbol=intent.symbol,
                    is_buy=intent.is_buy,
                    size_btc=floor_to_lot(max_order_size, precision.lot_size),
                    limit_px=intent.limit_px,
                    time_in_force=intent.time_in_force,
                    reduce_only=intent.reduce_only,
                    cloid=intent.cloid,
                )
            executor.validate_order_intent(config, precision, intent)
            loss = executor.loss_status(config, executor.LossSnapshot(intent.limit_px, intent.limit_px, intent.size_btc))
            if loss["status"] != "pass":
                raise executor.ValidationError(f"max_loss_check_failed:{loss['reason']}")
            endpoint_flags["real_order_endpoint_called"] = True
            last_submitted_attempt_id = attempt_id
            order_result = executor.run_order_once(
                config=config,
                precision=precision,
                intent=intent,
                loss_snapshot=executor.LossSnapshot(intent.limit_px, intent.limit_px, intent.size_btc),
                client=client,
            )
            current_status_rows = executor.extract_status_rows(order_result)
            order_status_rows.extend(current_status_rows)
            tracked_refs = executor.canary_tracked_refs(order_result, intent)
            aging_guard = {
                "status": "pass",
                "reason": "",
                "side": "buy" if intent.is_buy else "sell",
                "pre_bid": bid,
                "pre_ask": ask,
                "post_bid": bid,
                "post_ask": ask,
                "limit_px": intent.limit_px,
                "lost_touch_ticks": 0.0,
                "max_lost_touch_ticks": flow_max_lost_touch_ticks,
                "hold_elapsed_seconds": 0.0,
            }
            hold_started = time.monotonic()
            hold_deadline = hold_started + attempt_hold_seconds
            while time.monotonic() < hold_deadline:
                time.sleep(min(1.0, max(0.0, hold_deadline - time.monotonic())))
                guard_l2 = client.info.l2_snapshot(executor.SYMBOL)
                guard_bid, guard_ask = best_bid_ask(guard_l2)
                aging_guard = quote_aging_guard(
                    intent=intent,
                    pre_bid=bid,
                    pre_ask=ask,
                    post_bid=guard_bid,
                    post_ask=guard_ask,
                    tick_size=precision.tick_size,
                    max_lost_touch_ticks=flow_max_lost_touch_ticks,
                )
                aging_guard["hold_elapsed_seconds"] = round(time.monotonic() - hold_started, 6)
                if side_policy in {"flow_aware", "fresh_touch"} and aging_guard["status"] != "pass":
                    break
            end_ms = int(time.time() * 1000) + 2_000
            fills = client.info.user_fills_by_time(client.account_address, start_ms, end_ms, aggregate_by_time=False)
            user_fills_pullbacks.append(
                {
                    "phase": "after_attempt_hold",
                    "attempt": attempt_id,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "fill_count": len(fills),
                    "fills": fills,
                }
            )
            pull_user_fees_after_submit_once()
            post_l2 = client.info.l2_snapshot(executor.SYMBOL)
            post_bid, post_ask = best_bid_ask(post_l2)
            mark_px = (post_bid + post_ask) / 2.0
            quote_guard_rows.append({"attempt": attempt_id, **aging_guard})
            attempt_fill_rows = live_fill_rows(
                fills=fills,
                tracked_oids=extract_tracked_oids(order_result or {}),
                intent=intent,
                mark_px=mark_px,
                window_id=window_id,
                user_add_rate=user_add_rate,
                attempt_id=attempt_id,
                task_id=TASK_ID,
            )
            fill_rows.extend(row for row in attempt_fill_rows if row not in fill_rows)
            attempt_rows.append(
                {
                    "attempt": attempt_id,
                    "side": "buy" if intent.is_buy else "sell",
                    "limit_px": intent.limit_px,
                    "size_btc": intent.size_btc,
                    "bid": bid,
                    "ask": ask,
                    "post_only_tif": intent.time_in_force,
                    "order_status_types": ",".join(row.get("status_type", "") for row in current_status_rows),
                    "fill_count_after_attempt": len(fill_rows),
                    "crossing_guard_status": "pass",
                    "flow_guard_status": "pass" if side_policy in {"flow_aware", "fresh_touch"} else "not_applicable",
                    "fresh_touch_quality_bucket": fresh_touch_quality_bucket,
                    "dynamic_size_btc": fresh_touch_dynamic_size,
                    "quote_hold_seconds": attempt_hold_seconds,
                    "skip_reason": "",
                    "quote_aging_guard_status": aging_guard.get("status", ""),
                    "quote_aging_guard_reason": aging_guard.get("reason", ""),
                }
            )
            for ref in tracked_refs:
                oid = ref.get("oid")
                if oid is not None:
                    endpoint_flags["real_cancel_endpoint_called"] = True
                    try:
                        cancel_results.append({"method": "cancel", "attempt": attempt_id, "result": executor.redact(client.cancel_tracked(executor.SYMBOL, oid=int(oid)))})
                    except Exception as exc:
                        cancel_results.append({"method": "cancel", "attempt": attempt_id, "error": executor._redacted_error(exc)})
            endpoint_flags["real_cancel_endpoint_called"] = True
            try:
                cancel_results.append({"method": "cancel_by_cloid", "attempt": attempt_id, "result": executor.redact(client.cancel_tracked(executor.SYMBOL, cloid=intent.cloid))})
            except Exception as exc:
                cancel_results.append({"method": "cancel_by_cloid", "attempt": attempt_id, "error": executor._redacted_error(exc)})
            if fill_rows:
                break
    except Exception as exc:
        blocking_reasons.append(executor._redacted_error(exc))
    finally:
        for ref in tracked_refs:
            oid = ref.get("oid")
            if oid is not None:
                endpoint_flags["real_cancel_endpoint_called"] = True
                try:
                    cancel_results.append({"method": "cancel", "result": executor.redact(client.cancel_tracked(executor.SYMBOL, oid=int(oid)))})
                except Exception as exc:
                    cancel_results.append({"method": "cancel", "error": executor._redacted_error(exc)})
        if intent is not None:
            endpoint_flags["real_cancel_endpoint_called"] = True
            try:
                cancel_results.append({"method": "cancel_by_cloid", "result": executor.redact(client.cancel_tracked(executor.SYMBOL, cloid=intent.cloid))})
            except Exception as exc:
                cancel_results.append({"method": "cancel_by_cloid", "error": executor._redacted_error(exc)})
        end_ms = int(time.time() * 1000) + 2_000
        fills = client.info.user_fills_by_time(client.account_address, start_ms, end_ms, aggregate_by_time=False)
        user_fills_pullbacks.append(
            {
                "phase": "finalize",
                "attempt": len(attempt_rows) or "",
                "start_ms": start_ms,
                "end_ms": end_ms,
                "fill_count": len(fills),
                "fills": fills,
            }
        )
        pull_user_fees_after_submit_once()
        post_state = client.user_state()
        post_l2 = client.info.l2_snapshot(executor.SYMBOL)
        final_open_orders = client.open_orders()
        post_bid, post_ask = best_bid_ask(post_l2)
        mark_px = (post_bid + post_ask) / 2.0
        if intent is not None:
            final_fill_rows = live_fill_rows(
                fills=fills,
                tracked_oids=extract_tracked_oids(order_result or {}),
                intent=intent,
                mark_px=mark_px,
                window_id=window_id,
                user_add_rate=user_add_rate,
                attempt_id=last_submitted_attempt_id,
                task_id=TASK_ID,
            )
            fill_rows.extend(row for row in final_fill_rows if row not in fill_rows)

    if fill_rows:
        order_status_rows = order_status_rows + [{"status_type": "filled", "payload": {"source": "user_fills_by_time"}}]
    remaining_tracked = []
    tracked_oids = extract_tracked_oids(order_result or {})
    for order in final_open_orders:
        if str(order.get("oid")) in tracked_oids:
            remaining_tracked.append(order)
    shutdown_status = "pass" if not remaining_tracked else "fail_closed"
    if shutdown_status != "pass":
        blocking_reasons.append("tracked_order_still_open")
    maker_fill_count = sum(1 for row in fill_rows if row.get("liquidity") == "maker")
    if any(row.get("liquidity") not in {"maker", "unknown"} for row in fill_rows):
        blocking_reasons.append("non_maker_fill_detected")
    if not fill_rows:
        if endpoint_flags["real_order_endpoint_called"] and cancel_result_mentions_filled(cancel_results):
            blocking_reasons.append("fill_reconciliation_required_no_fill_unproven")
        else:
            blocking_reasons.append("no_fill_observed")
    if side_policy == "flow_aware" and endpoint_flags["real_order_endpoint_called"] is False:
        blocking_reasons.append("flow_guard_no_safe_candidate")
    if side_policy == "fresh_touch" and endpoint_flags["real_order_endpoint_called"] is False:
        blocking_reasons.append("fresh_touch_session_gate_no_eligible_candidate")

    final_recommendation = READY_RECOMMENDATION if fill_rows and maker_fill_count == len(fill_rows) and shutdown_status == "pass" and not blocking_reasons else BLOCKED_RECOMMENDATION

    window_label = artifact_window_label(window_id)
    bind_attempt_identity(attempt_rows, task_id=TASK_ID, window_id=window_id)
    write_json(
        output_dir / "run_intent_marker.json",
        {
            "task_id": TASK_ID,
            "window_id": window_label,
            "artifact_window_id": window_id,
            "real_orders_allowed": True,
            "post_only_required": True,
        },
    )
    write_json(output_dir / "approved_config_snapshot.json", executor.config_snapshot(config))
    write_json(output_dir / "credential_source_manifest.json", executor.credential_source_snapshot(env_file=env_file, env_load=env_load))
    write_json(
        output_dir / "private_preflight_summary.json",
        {
            "preflight_summary": {
                "asset_position_count_before": len(pre_state.get("assetPositions", [])),
                "open_order_count_before": len(pre_open_orders),
                "user_fill_query_start_ms": start_ms,
                "user_add_rate": user_add_rate,
                "fast_event_driven_submit": fast_event_driven_submit,
                "pre_user_state_deferred_until_post_submit": pre_user_state_deferred,
                "user_fees_deferred_until_post_submit": user_fees_deferred,
                "open_orders_checked_before_submit": bool(pre_open_orders == []),
            },
            "open_orders_before": pre_open_orders,
            "endpoint_called": endpoint_flags["private_endpoint_called"],
        },
    )
    write_csv(output_dir / "precision_tick_lot_snapshot.csv", [executor.precision_to_row(precision)], list(executor.precision_to_row(precision)))
    intent_fieldnames = ["symbol", "side", "size_btc", "limit_px", "notional_usdc", "time_in_force", "order_type", "reduce_only", "endpoint_called", "cloid_redacted"]
    write_csv(
        output_dir / "order_intent_audit.csv",
        [executor.order_intent_row(intent, endpoint_called=endpoint_flags["real_order_endpoint_called"])] if intent is not None else [],
        intent_fieldnames,
    )
    write_csv(
        output_dir / "quote_attempt_matrix.csv",
        attempt_rows,
        [
            "attempt",
            "window_id",
            "attempt_id",
            "attempt_key",
            "side",
            "limit_px",
            "size_btc",
            "bid",
            "ask",
            "post_only_tif",
            "order_status_types",
            "fill_count_after_attempt",
            "crossing_guard_status",
            "flow_guard_status",
            "fresh_touch_quality_bucket",
            "dynamic_size_btc",
            "quote_hold_seconds",
            "skip_reason",
            "quote_aging_guard_status",
            "quote_aging_guard_reason",
        ],
    )
    write_csv(
        output_dir / "flow_side_score_matrix.csv",
        side_score_rows,
        [
            "attempt",
            "side",
            "score",
            "same_side_top_qty_btc",
            "same_side_top_order_count",
            "top_depth_multiple_of_order",
            "public_flow_candidate_count",
            "public_flow_strict_rate",
            "public_flow_depletion_rate",
            "public_flow_aging_rate",
            "guard_status",
            "selected",
            "allowed",
            "skip_reason",
            "evidence_scope",
        ],
    )
    write_csv(
        output_dir / "quote_aging_guard_matrix.csv",
        quote_guard_rows,
        [
            "attempt",
            "status",
            "reason",
            "side",
            "pre_bid",
            "pre_ask",
            "post_bid",
            "post_ask",
            "limit_px",
            "lost_touch_ticks",
            "max_lost_touch_ticks",
            "hold_elapsed_seconds",
        ],
    )
    write_csv(
        output_dir / "immediate_pre_submit_guard_matrix.csv",
        immediate_guard_rows,
        [
            "attempt",
            "status",
            "reason",
            "candidate_source_exchange_time_ms",
            "candidate_age_seconds",
            "max_age_seconds",
            "selected_side",
            "selected_quote_px",
            "current_bid",
            "current_ask",
            "selected_size_btc",
            "max_order_size_btc",
            "quality_bucket",
            "current_same_side_top_qty_btc",
            "current_same_side_top_order_count",
            "current_top_depth_multiple_of_order",
            "post_only_tif",
            "post_only_non_crossing",
            "current_touch_match",
            "source",
        ],
    )
    write_csv(
        output_dir / "touch_freshness_matrix.csv",
        touch_freshness_rows,
        [
            "attempt",
            "candidate_index",
            "side",
            "status",
            "age_seconds",
            "reason",
            "source_start_exchange_time_ms",
            "source_quote_aging_status",
            "source_first_touch_trade_ms",
            "source_first_strict_trade_through_ms",
            "selected",
        ],
    )
    write_csv(
        output_dir / "dynamic_size_decision_matrix.csv",
        dynamic_size_rows,
        [
            "attempt",
            "candidate_index",
            "side",
            "quality_bucket",
            "bucket_cap_btc",
            "hard_cap_btc",
            "recent_same_side_at_or_through_qty_btc_last_3s",
            "raw_size_btc",
            "floored_size_btc",
            "status",
            "reason",
            "candidate_allowed",
            "selected",
        ],
    )
    write_csv(
        output_dir / "session_side_eligibility.csv",
        session_side_rows,
        ["side", "default_policy", "eligible", "same_window_public_support", "materially_favors_sell", "reason"],
    )
    write_csv(
        output_dir / "time_gate_decision_matrix.csv",
        time_gate_rows,
        ["gate", "utc_hour", "eligible", "fixed_hour_allowlist_used", "precheck_status", "reason"],
    )
    write_json(
        output_dir / "private_order_response_audit.json",
        {
            "real_order_endpoint_called": endpoint_flags["real_order_endpoint_called"],
            "order_submission_attempted": endpoint_flags["real_order_endpoint_called"],
            "order_status_rows": order_status_rows,
            "order_result": order_result,
            "blocking_reasons": blocking_reasons,
        },
    )
    write_json(
        output_dir / "account_inventory_snapshots.json",
        {
            "pre_state": pre_state,
            "post_state": post_state,
            "user_fees": user_fees,
        },
    )
    write_json(
        output_dir / "user_fills_pullback_audit.json",
        {
            "pullbacks": user_fills_pullbacks,
            "pullback_count": len(user_fills_pullbacks),
            "raw_payload_redacted": True,
        },
    )
    write_json(output_dir / "market_markout_snapshot.json", {"pre_l2": pre_l2, "post_l2": post_l2})
    write_csv(output_dir / "live_fill_ledger.csv", fill_rows, live_fill_ledger_fieldnames())
    write_csv(
        output_dir / "fill_liquidity_role_evidence.csv",
        fill_liquidity_role_evidence_rows(fill_rows),
        fill_liquidity_role_evidence_fieldnames(),
    )
    write_json(
        output_dir / "cancel_shutdown_proof.json",
        {
            "real_cancel_endpoint_called": endpoint_flags["real_cancel_endpoint_called"],
            "tracked_refs": tracked_refs,
            "cancel_results": cancel_results,
            "final_open_orders": final_open_orders,
            "proof_status": shutdown_status,
        },
    )
    write_json(output_dir / "max_loss_monitor_summary.json", loss)
    manifest = {
        "task_id": TASK_ID,
        "policy_version": policy_version_for_side_policy(side_policy),
        "window_id": window_label,
        "artifact_window_id": window_id,
        "requote_attempts_requested": requote_attempts,
        "requote_attempts_completed": len(attempt_rows),
        "side_policy": side_policy,
        "max_order_size_btc": max_order_size,
        "flow_max_top_depth_multiple": flow_max_top_depth_multiple,
        "flow_max_lost_touch_ticks": flow_max_lost_touch_ticks,
        "fresh_touch_hard_cap_btc": FRESH_TOUCH_HARD_CAP_BTC if side_policy == "fresh_touch" else "",
        "public_flow_precheck_status": public_flow_precheck.get("status", ""),
        "public_flow_precheck_reason": public_flow_precheck.get("reason", ""),
        "flow_safe_candidate_count": sum(1 for row in attempt_rows if row.get("flow_guard_status") in {"pass", "not_applicable"}),
        "flow_skipped_candidate_count": sum(1 for row in attempt_rows if row.get("flow_guard_status") == "skip"),
        "fresh_touch_candidate_count": len(touch_freshness_rows),
        "fresh_touch_allowed_candidate_count": sum(1 for row in dynamic_size_rows if row.get("candidate_allowed") is True),
        "fresh_touch_submitted_count": sum(1 for row in attempt_rows if row.get("flow_guard_status") == "pass" and row.get("fresh_touch_quality_bucket")),
        "fresh_touch_buy_only": side_policy == "fresh_touch",
        "same_process_trigger": same_process_trigger,
        "fast_event_driven_submit": fast_event_driven_submit,
        "public_flow_precheck_override_used": public_flow_precheck_override is not None,
        "immediate_pre_submit_guard_status": (
            immediate_guard_rows[-1].get("status", "") if immediate_guard_rows else "not_evaluated"
        ),
        "immediate_pre_submit_guard_reason": (
            immediate_guard_rows[-1].get("reason", "") if immediate_guard_rows else ""
        ),
        "final_recommendation": final_recommendation,
        "blocking_reasons": blocking_reasons,
        "order_status_types": [row.get("status_type", "") for row in order_status_rows],
        "fill_count": len(fill_rows),
        "maker_fill_count": maker_fill_count,
        "ledger_fill_rows": len(fill_rows),
        "real_order_endpoint_called": endpoint_flags["real_order_endpoint_called"],
        "private_endpoint_called": endpoint_flags["private_endpoint_called"],
        "real_cancel_endpoint_called": endpoint_flags["real_cancel_endpoint_called"],
        "final_open_orders_count": len(final_open_orders),
        "shutdown_proof_status": shutdown_status,
        "post_only_tif": executor.POST_ONLY_TIF,
        "crossing_guard_status": "pass",
        "flow_guard_status": "pass" if endpoint_flags["real_order_endpoint_called"] else "no_safe_candidate",
        "fresh_touch_guard_status": (
            "pass"
            if side_policy == "fresh_touch" and endpoint_flags["real_order_endpoint_called"]
            else ("no_eligible_candidate" if side_policy == "fresh_touch" else "not_applicable")
        ),
        "credentials_written": False,
        "secret_values_written": False,
        "raw_signatures_written": False,
        "git_commit": executor.git_commit(),
    }
    write_json(output_dir / "m2_fill_window_manifest.json", manifest)
    write_json(
        output_dir / "executor_manifest.json",
        {
            "task_id": TASK_ID,
            "window_id": window_label,
            "artifact_window_id": window_id,
            "order_submission_attempted": endpoint_flags["real_order_endpoint_called"],
            "order_status_types": manifest["order_status_types"],
            "private_endpoint_called": endpoint_flags["private_endpoint_called"],
            "real_order_endpoint_called": endpoint_flags["real_order_endpoint_called"],
            "real_cancel_endpoint_called": endpoint_flags["real_cancel_endpoint_called"],
            "shutdown_proof_status": shutdown_status,
            "credentials_written": False,
            "secret_values_written": False,
            "raw_signatures_written": False,
            "final_recommendation": final_recommendation,
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Hyperliquid M2B Fill Window",
                "",
                f"Final recommendation: `{final_recommendation}`",
                "",
                "The window uses one real post-only Alo order attempt, tracked cancel, and private/economics pullback.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--window-id", type=int, required=True)
    parser.add_argument("--wait-seconds", type=int, default=45)
    parser.add_argument("--quote-offset-ticks", type=int, default=1)
    parser.add_argument("--requote-attempts", type=int, default=1)
    parser.add_argument("--quote-hold-seconds", type=int, default=None)
    parser.add_argument("--side-policy", choices=["buy", "sell", "alternate", "flow_aware", "fresh_touch"], default="buy")
    parser.add_argument("--max-order-size", type=float, default=0.00999)
    parser.add_argument("--flow-max-top-depth-multiple", type=float, default=DEFAULT_FLOW_MAX_TOP_DEPTH_MULTIPLE)
    parser.add_argument("--flow-max-lost-touch-ticks", type=float, default=DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS)
    parser.add_argument("--fresh-touch-precheck-seconds", type=float, default=DEFAULT_FRESH_TOUCH_PRECHECK_SECONDS)
    parser.add_argument("--operator-ack", default="")
    args = parser.parse_args()
    if args.operator_ack != OPERATOR_ACK:
        raise SystemExit("M2B live fill window requires exact operator acknowledgement")
    manifest = run_window(
        output_dir=args.output_dir,
        env_file=args.env_file,
        window_id=args.window_id,
        wait_seconds=args.wait_seconds,
        quote_offset_ticks=args.quote_offset_ticks,
        requote_attempts=args.requote_attempts,
        quote_hold_seconds=args.quote_hold_seconds,
        side_policy=args.side_policy,
        max_order_size=args.max_order_size,
        flow_max_top_depth_multiple=args.flow_max_top_depth_multiple,
        flow_max_lost_touch_ticks=args.flow_max_lost_touch_ticks,
        fresh_touch_precheck_seconds=args.fresh_touch_precheck_seconds,
    )
    print(json.dumps(executor.redact(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
