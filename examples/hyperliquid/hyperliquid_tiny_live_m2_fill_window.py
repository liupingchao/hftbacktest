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


TASK_ID = "0619T001"
READY_RECOMMENDATION = "hyperliquid_tiny_live_m2_fill_window_ready_for_qa"
BLOCKED_RECOMMENDATION = "hyperliquid_tiny_live_m2_fill_window_blocked"
OPERATOR_ACK = executor.LIVE_OPERATOR_ACK
MAX_WAIT_SECONDS = 600
FLOW_AWARE_POLICY_VERSION = "m2_flow_aware_v1"
DEFAULT_FLOW_MAX_TOP_DEPTH_MULTIPLE = 500.0
DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS = 0.0
FLOW_SAFE_HOLD_SECONDS = 15
DEFAULT_FLOW_PRECHECK_SECONDS = 20.0


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
    steps = math.floor(size / lot_size)
    return round(steps * lot_size, 10)


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
            candidate_stride_seconds=max(1.0, min(5.0, duration_seconds / 4.0)),
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
    write_json(output_dir / "run_intent_marker.json", {"task_id": TASK_ID, "window_id": window_id, "real_orders_allowed": False, "post_only_required": True})
    write_json(output_dir / "credential_source_manifest.json", {"env_file": str(env_file), "env_file_keys_loaded": [], "candidate_keys_present": [], "secret_values_written": False})
    write_json(output_dir / "private_preflight_summary.json", {"preflight_summary": {}, "open_orders_before": [], "endpoint_called": False})
    write_csv(output_dir / "precision_tick_lot_snapshot.csv", [], ["symbol", "sz_decimals", "tick_size", "lot_size", "mid_px", "source"])
    write_csv(output_dir / "order_intent_audit.csv", [], ["symbol", "side", "size_btc", "limit_px", "notional_usdc", "time_in_force", "order_type", "reduce_only", "endpoint_called", "cloid_redacted"])
    write_csv(
        output_dir / "quote_attempt_matrix.csv",
        [],
        ["attempt", "side", "limit_px", "size_btc", "bid", "ask", "post_only_tif", "order_status_types", "fill_count_after_attempt", "crossing_guard_status", "flow_guard_status", "skip_reason", "quote_aging_guard_status", "quote_aging_guard_reason"],
    )
    write_csv(
        output_dir / "flow_side_score_matrix.csv",
        [],
        ["attempt", "side", "score", "same_side_top_qty_btc", "same_side_top_order_count", "top_depth_multiple_of_order", "public_flow_candidate_count", "public_flow_strict_rate", "public_flow_depletion_rate", "public_flow_aging_rate", "guard_status", "selected", "allowed", "skip_reason", "evidence_scope"],
    )
    write_csv(output_dir / "quote_aging_guard_matrix.csv", [], ["attempt", "status", "reason", "side", "pre_bid", "pre_ask", "post_bid", "post_ask", "limit_px", "lost_touch_ticks", "max_lost_touch_ticks"])
    write_json(output_dir / "private_order_response_audit.json", {"real_order_endpoint_called": False, "order_submission_attempted": False, "order_status_rows": [], "order_result": None, "blocking_reasons": blocking_reasons})
    write_json(output_dir / "account_inventory_snapshots.json", {"pre_state": {}, "post_state": {}, "user_fees": {}})
    write_json(output_dir / "market_markout_snapshot.json", {"pre_l2": {}, "post_l2": {}})
    write_csv(output_dir / "live_fill_ledger.csv", [], ["source_window", "fill_id", "side", "qty_btc", "price_usdc", "intent_price_usdc", "mark_price_usdc", "fee_usdc", "rebate_usdc", "liquidity"])
    write_json(output_dir / "cancel_shutdown_proof.json", {"real_cancel_endpoint_called": False, "tracked_refs": [], "cancel_results": [], "final_open_orders": [], "proof_status": "no_order_submitted"})
    write_json(output_dir / "max_loss_monitor_summary.json", {"status": "not_evaluated", "reason": "blocked_before_order"})
    manifest = {
        "task_id": TASK_ID,
        "policy_version": FLOW_AWARE_POLICY_VERSION,
        "window_id": window_id,
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
        "credentials_written": False,
        "secret_values_written": False,
        "raw_signatures_written": False,
        "git_commit": executor.git_commit(),
    }
    write_json(output_dir / "m2_fill_window_manifest.json", manifest)
    write_json(output_dir / "executor_manifest.json", {"task_id": TASK_ID, "order_submission_attempted": False, "private_endpoint_called": False, "real_order_endpoint_called": False, "real_cancel_endpoint_called": False, "shutdown_proof_status": "no_order_submitted", "credentials_written": False, "secret_values_written": False, "raw_signatures_written": False, "final_recommendation": BLOCKED_RECOMMENDATION})
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


def live_fill_rows(
    *,
    fills: list[dict[str, Any]],
    tracked_oids: set[str],
    intent: executor.OrderIntent,
    mark_px: float,
    window_id: int,
    user_add_rate: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, fill in enumerate(fills, start=1):
        if tracked_oids and str(fill.get("oid")) not in tracked_oids:
            continue
        side = side_from_fill(fill)
        if side not in {"buy", "sell"}:
            continue
        qty = float(fill.get("sz", 0.0))
        price = float(fill.get("px", 0.0))
        if qty <= 0 or price <= 0:
            continue
        crossed = bool(fill.get("crossed"))
        liquidity = "taker" if crossed else "maker"
        fee = abs(float(fill.get("fee", 0.0))) if fill.get("fee") not in ("", None) else abs(qty * price * user_add_rate)
        rows.append(
            {
                "source_window": f"window_{window_id}",
                "fill_id": "fill_sha256_" + hashlib.sha256(str(fill).encode()).hexdigest()[:12] if fill.get("hash") else f"window_{window_id}_fill_{idx}",
                "side": side,
                "qty_btc": qty,
                "price_usdc": price,
                "intent_price_usdc": intent.limit_px,
                "mark_price_usdc": mark_px,
                "fee_usdc": fee,
                "rebate_usdc": 0.0,
                "liquidity": liquidity,
            }
        )
    return rows


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
    public_flow_precheck: dict[str, Any] = {"status": "not_applicable", "summary": {}}
    if side_policy == "flow_aware":
        public_flow_precheck = run_public_flow_precheck(
            output_dir=output_dir,
            order_size_btc=max_order_size,
            quote_hold_seconds=hold_seconds,
            duration_seconds=DEFAULT_FLOW_PRECHECK_SECONDS,
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

    env_load = executor.load_env_file(env_file)
    client = executor.build_live_client_from_env()
    if client is None:
        raise executor.ValidationError("live_client_unavailable")
    endpoint_flags["private_endpoint_called"] = True
    start_ms = int(time.time() * 1000) - 2_000
    precision = executor.fetch_live_precision(client)
    pre_l2 = client.info.l2_snapshot(executor.SYMBOL)
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
    pre_state = client.user_state()
    user_fees = client.info.user_fees(client.account_address)
    user_add_rate = float(user_fees.get("userAddRate", 0.0) or 0.0)

    try:
        for attempt_id in range(1, requote_attempts + 1):
            attempt_l2 = client.info.l2_snapshot(executor.SYMBOL)
            bid, ask = best_bid_ask(attempt_l2)
            flow_decision = None
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
                            "skip_reason": skip_reason,
                            "quote_aging_guard_status": "not_submitted",
                            "quote_aging_guard_reason": "",
                        }
                    )
                    continue
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
            hold_deadline = hold_started + hold_seconds
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
                if side_policy == "flow_aware" and aging_guard["status"] != "pass":
                    break
            end_ms = int(time.time() * 1000) + 2_000
            fills = client.info.user_fills_by_time(client.account_address, start_ms, end_ms, aggregate_by_time=False)
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
                    "flow_guard_status": "pass" if side_policy == "flow_aware" else "not_applicable",
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
    if any(row.get("liquidity") != "maker" for row in fill_rows):
        blocking_reasons.append("non_maker_fill_detected")
    if not fill_rows:
        blocking_reasons.append("no_fill_observed")
    if side_policy == "flow_aware" and endpoint_flags["real_order_endpoint_called"] is False:
        blocking_reasons.append("flow_guard_no_safe_candidate")

    final_recommendation = READY_RECOMMENDATION if fill_rows and maker_fill_count == len(fill_rows) and shutdown_status == "pass" and not blocking_reasons else BLOCKED_RECOMMENDATION

    write_json(output_dir / "run_intent_marker.json", {"task_id": TASK_ID, "window_id": window_id, "real_orders_allowed": True, "post_only_required": True})
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
    write_json(output_dir / "market_markout_snapshot.json", {"pre_l2": pre_l2, "post_l2": post_l2})
    write_csv(
        output_dir / "live_fill_ledger.csv",
        fill_rows,
        [
            "source_window",
            "fill_id",
            "side",
            "qty_btc",
            "price_usdc",
            "intent_price_usdc",
            "mark_price_usdc",
            "fee_usdc",
            "rebate_usdc",
            "liquidity",
        ],
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
        "policy_version": FLOW_AWARE_POLICY_VERSION if side_policy == "flow_aware" else "legacy_side_policy",
        "window_id": window_id,
        "requote_attempts_requested": requote_attempts,
        "requote_attempts_completed": len(attempt_rows),
        "side_policy": side_policy,
        "max_order_size_btc": max_order_size,
        "flow_max_top_depth_multiple": flow_max_top_depth_multiple,
        "flow_max_lost_touch_ticks": flow_max_lost_touch_ticks,
        "public_flow_precheck_status": public_flow_precheck.get("status", ""),
        "public_flow_precheck_reason": public_flow_precheck.get("reason", ""),
        "flow_safe_candidate_count": sum(1 for row in attempt_rows if row.get("flow_guard_status") in {"pass", "not_applicable"}),
        "flow_skipped_candidate_count": sum(1 for row in attempt_rows if row.get("flow_guard_status") == "skip"),
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
    parser.add_argument("--side-policy", choices=["buy", "sell", "alternate", "flow_aware"], default="buy")
    parser.add_argument("--max-order-size", type=float, default=0.00999)
    parser.add_argument("--flow-max-top-depth-multiple", type=float, default=DEFAULT_FLOW_MAX_TOP_DEPTH_MULTIPLE)
    parser.add_argument("--flow-max-lost-touch-ticks", type=float, default=DEFAULT_FLOW_MAX_LOST_TOUCH_TICKS)
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
    )
    print(json.dumps(executor.redact(manifest), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
