#!/usr/bin/env python3
"""Read-only inventory-aware quote placement request runner.

This runner implements the fixed 0526T006 policy contract as an offline
classifier. It does not change strategy behavior, does not run live, and does
not search parameters.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from quote_adjustment_replay import (
    REQUIRED_T006_FIELDS,
    _bool,
    _find_first,
    _float,
    _generated_at,
    _int,
    _read_csv,
    _safe_num,
    _write_csv,
    _write_json,
)


TASK_ID = "0528T001"
RUNNER_MODE = "stage9i_inventory_aware_quote_placement"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/stage9i_inventory_aware_quote_placement_0528T001")
DEFAULT_RUN_DIRS = [
    Path("local_live_analysis/5-19-day-control-30min"),
    Path("local_live_analysis/5-19-night-active-30min-a"),
    Path("local_live_analysis/5-19-night-active-30min-b"),
    Path("local_live_analysis/5-19-night-active-30min-c"),
    Path("local_live_analysis/5-21-day-control-60min"),
    Path("local_live_analysis/5-26-active-minmove-control-30min-a"),
    Path("local_live_analysis/5-26-active-minmove-control-60min-a"),
    Path("local_live_analysis/5-26-active-makeredge-control-180min-a"),
    Path("local_live_analysis/5-26-active-minmove-control-30min-b"),
]
DEFAULT_CAVEATED_SAMPLE_IDS = {"5-19-night-active-30min-a", "5-26-active-minmove-control-60min-a"}

VERDICT_TAXONOMY = (
    "focused_design_promising",
    "promising_but_needs_parameter_sweep",
    "needs_more_clean_fills",
    "too_conservative_fill_loss",
    "caveated_only",
    "reject",
    "not_decisionable",
)

POSITION_FLAT_THRESHOLD = 0.0005
POSITION_LARGE_THRESHOLD = 0.0015
LOW_INVENTORY_SCORE_THRESHOLD = 0.35
STRONG_EDGE_TICKS = 5.0
STALE_MS_THRESHOLD = 250.0
LATENCY_MS_THRESHOLD = 50.0
MIN_CLEAN_REQUEST_FILLS = 75


@dataclass(frozen=True)
class SampleArtifacts:
    sample_id: str
    run_dir: Path
    is_caveated: bool
    stage5_labels: Path | None
    fill_markouts: Path | None
    audit_live: Path | None
    stage5c_safety: Path | None
    status: str
    notes: tuple[str, ...]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else math.nan


def _sample_id(run_dir: Path) -> str:
    return run_dir.name


def _stage5_path(run_dir: Path, name: str) -> Path:
    return run_dir / "stage5_execution_outcome_labels_0514T005" / name


def _discover_stage5c(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("stage5c_quote_anchor_safety_*/quote_anchor_safety_rows.csv"))
    return matches[-1] if matches else None


def _discover_audit(run_dir: Path) -> Path | None:
    try:
        return _find_first(run_dir, "audit_live*.csv", "**/audit_live*.csv")
    except FileNotFoundError:
        return None


def discover_sample(run_dir: Path, caveated_ids: set[str]) -> SampleArtifacts:
    notes: list[str] = []
    labels = _stage5_path(run_dir, "execution_outcome_labels.csv")
    markouts = _stage5_path(run_dir, "fill_markout_labels.csv")
    audit = _discover_audit(run_dir)
    safety = _discover_stage5c(run_dir)
    if not run_dir.exists():
        notes.append("run_dir_missing")
    if not labels.exists():
        notes.append("stage5_execution_labels_missing")
        labels_path: Path | None = None
    else:
        labels_path = labels
    markout_path: Path | None = markouts if markouts.exists() else None
    if markout_path is None:
        notes.append("fill_markout_labels_missing")
    if audit is None:
        notes.append("audit_live_missing")
    if safety is None:
        notes.append("stage5c_safety_missing")
    status = "usable" if labels_path is not None else "missing_required_artifact"
    return SampleArtifacts(
        sample_id=_sample_id(run_dir),
        run_dir=run_dir,
        is_caveated=_sample_id(run_dir) in caveated_ids,
        stage5_labels=labels_path,
        fill_markouts=markout_path,
        audit_live=audit,
        stage5c_safety=safety,
        status=status,
        notes=tuple(notes),
    )


def classify_inventory_bucket(position: float, inventory_score: float) -> str:
    abs_position = abs(position)
    if not _finite(position) or abs_position <= POSITION_FLAT_THRESHOLD:
        return "flat"
    if abs_position >= POSITION_LARGE_THRESHOLD or (_finite(inventory_score) and inventory_score <= LOW_INVENTORY_SCORE_THRESHOLD):
        return "large_skew_or_low_score"
    return "mild_skew"


def classify_side_class(order_side: str, position: float) -> str:
    side = str(order_side or "").strip().lower()
    if not _finite(position) or abs(position) <= POSITION_FLAT_THRESHOLD:
        return "flat_side"
    if side == "buy" and position > POSITION_FLAT_THRESHOLD:
        return "add_side"
    if side == "sell" and position < -POSITION_FLAT_THRESHOLD:
        return "add_side"
    if side == "buy" and position < -POSITION_FLAT_THRESHOLD:
        return "reduce_side"
    if side == "sell" and position > POSITION_FLAT_THRESHOLD:
        return "reduce_side"
    return "flat_side"


def classify_edge_bucket(edge_vs_fair_ticks: float, edge_vs_reservation_ticks: float) -> str:
    edges = [value for value in (edge_vs_fair_ticks, edge_vs_reservation_ticks) if _finite(value)]
    if not edges:
        return "edge_unknown"
    conservative_edge = min(edges)
    if conservative_edge < 0.0:
        return "edge_adverse"
    if conservative_edge >= STRONG_EDGE_TICKS:
        return "edge_strong_favorable"
    return "edge_weak_or_neutral"


def classify_quote_distance_bucket(placement_bucket: str, distance_to_bbo_ticks: float) -> str:
    placement = str(placement_bucket or "").strip().lower()
    if placement in {"touch", "one_tick_tight", "step_back_gt1", "outside_or_no_quote"}:
        return placement
    if "touch" in placement:
        return "touch"
    if "one" in placement or "1" in placement:
        return "one_tick_tight"
    if "outside" in placement or "no_quote" in placement:
        return "outside_or_no_quote"
    if "step" in placement:
        return "step_back_gt1"
    distance = abs(distance_to_bbo_ticks)
    if not _finite(distance):
        return "outside_or_no_quote"
    if distance <= 0.5:
        return "touch"
    if distance <= 1.5:
        return "one_tick_tight"
    return "step_back_gt1"


def _seq(row: dict[str, str]) -> int | None:
    for key in ("decision_context_strategy_seq", "submit_strategy_seq", "strategy_seq"):
        value = _int(row.get(key))
        if value is not None:
            return value
    return None


def _order_filled(row: dict[str, str]) -> bool:
    return _float(row.get("fill_count"), 0.0) > 0.0 or _bool(row.get("full_fill")) or _bool(row.get("partial_fill"))


def _markouts_by_order(path: Path | None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if path is None:
        return out
    for row in _read_csv(path):
        if int(_float(row.get("horizon_ms"), -1.0)) != 5000:
            continue
        if not _bool(row.get("horizon_observable")):
            continue
        order_id = str(row.get("order_id", "")).strip()
        if order_id:
            out[order_id] = row
    return out


def _read_stage5c_by_seq(path: Path | None, seqs: set[int]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    if path is None:
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seq = _int(row.get("strategy_seq"))
            if seq in seqs:
                out[int(seq)] = row
    return out


def _read_audit_by_seq(path: Path | None, seqs: set[int]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    if path is None:
        return out
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            seq = _int(row.get("strategy_seq"))
            if seq is None or seq not in seqs:
                continue
            event_type = str(row.get("event_type", "")).strip().lower()
            if event_type and event_type not in {"decision", "quote_decision"}:
                continue
            out[int(seq)] = row
            if len(out) >= len(seqs):
                break
    return out


def _unsafe_reasons(label: dict[str, str], audit: dict[str, str], safety: dict[str, str], side: str) -> list[str]:
    reasons: list[str] = []
    if _bool(label.get("post_only_risk")) or _bool(audit.get("post_only_pre_check")) or _bool(audit.get("post_only_post_check")):
        reasons.append("post_only_risk")
    if _bool(safety.get("post_only_risk_after_recheck")):
        reasons.append("post_only_risk_after_recheck")
    if _bool(safety.get("missing_anchor")):
        reasons.append("missing_anchor")
    if _bool(safety.get("stale_anchor")):
        reasons.append("stale_anchor")
    if _bool(label.get("join_gap_crossed")) or _bool(safety.get("join_gap_crossed")):
        reasons.append("join_gap_crossed")
    if side == "buy" and _bool(safety.get("suppress_buy")):
        reasons.append("suppress_buy")
    if side == "sell" and _bool(safety.get("suppress_sell")):
        reasons.append("suppress_sell")
    if _float(label.get("book_view_stale_ms"), 0.0) > STALE_MS_THRESHOLD:
        reasons.append("book_view_stale")
    if _float(label.get("top5_join_age_ms"), 0.0) > STALE_MS_THRESHOLD:
        reasons.append("top5_join_stale")
    if _float(label.get("latency_signal_ms"), 0.0) > LATENCY_MS_THRESHOLD:
        reasons.append("latency_stale")
    if _float(audit.get("anchor_age_ms"), 0.0) > STALE_MS_THRESHOLD or _float(safety.get("anchor_age_ms"), 0.0) > STALE_MS_THRESHOLD:
        reasons.append("anchor_age_stale")
    drop_cause = str(audit.get("reject_throttle_drop_cause", "")).strip()
    if drop_cause:
        reasons.append("reject_throttle_drop_cause")
    return reasons


def classify_request(label: dict[str, str], audit: dict[str, str] | None = None, safety: dict[str, str] | None = None) -> dict[str, Any]:
    audit = audit or {}
    safety = safety or {}
    side = str(label.get("order_side", "")).strip().lower()
    position = _float(label.get("position_before_submit"))
    inventory_score = _float(label.get("inventory_score"))
    edge_fair = _float(label.get("edge_vs_fair_ticks"))
    edge_reservation = _float(label.get("edge_vs_reservation_ticks"))
    distance = _float(label.get("distance_to_bbo_ticks"))

    missing = []
    if side not in {"buy", "sell"}:
        missing.append("order_side")
    if not _finite(position):
        missing.append("position_before_submit")
    if not _finite(inventory_score):
        missing.append("inventory_score")
    if not _finite(edge_fair) and not _finite(edge_reservation):
        missing.append("edge")

    inventory_bucket = classify_inventory_bucket(position, inventory_score)
    side_class = classify_side_class(side, position)
    edge_bucket = classify_edge_bucket(edge_fair, edge_reservation)
    quote_distance_bucket = classify_quote_distance_bucket(label.get("placement_bucket", ""), distance)
    unsafe_reasons = _unsafe_reasons(label, audit, safety, side)

    side_preference_request = "allow_both_sides"
    quote_distance_request = "keep_baseline"
    size_request = "baseline_size"
    candidate_action = "no_change"
    no_change_reason = ""

    if missing:
        no_change_reason = "missing_required_fields:" + "|".join(missing)
    elif unsafe_reasons:
        no_change_reason = "unsafe_safety_context:" + "|".join(sorted(set(unsafe_reasons)))
    elif side_class == "reduce_side" and edge_bucket == "edge_weak_or_neutral":
        no_change_reason = "reduce_side_weak_or_neutral_non_adverse"
    elif edge_bucket == "edge_unknown":
        no_change_reason = "edge_unknown"
    elif side_class == "reduce_side" and edge_bucket == "edge_strong_favorable":
        side_preference_request = "prefer_reduce_side"
        size_request = "preserve_reduce_side_size"
        quote_distance_request = "allow_touch"
        candidate_action = "request_side_priority"
    elif side_class == "add_side" and inventory_bucket == "large_skew_or_low_score":
        side_preference_request = "discourage_add_side"
        if edge_bucket == "edge_adverse":
            size_request = "suppress_add_side"
            quote_distance_request = "outside_or_no_quote"
            candidate_action = "request_size_adjustment"
        else:
            size_request = "reduce_add_side_size"
            quote_distance_request = "prefer_step_back_gt1"
            candidate_action = "request_size_adjustment"
    elif side_class == "add_side" and inventory_bucket == "mild_skew" and edge_bucket != "edge_strong_favorable":
        side_preference_request = "discourage_add_side"
        size_request = "reduce_add_side_size"
        quote_distance_request = "prefer_one_tick_tight"
        candidate_action = "request_quote_adjustment"
    elif side_class == "flat_side" and edge_bucket == "edge_adverse":
        quote_distance_request = "prefer_step_back_gt1"
        candidate_action = "request_quote_adjustment"
    else:
        no_change_reason = "baseline_context_supported"

    return {
        "inventory_bucket": inventory_bucket,
        "side_class": side_class,
        "edge_bucket": edge_bucket,
        "quote_distance_bucket": quote_distance_bucket,
        "safety_context": "unsafe" if unsafe_reasons else "safe",
        "safety_reasons": "|".join(sorted(set(unsafe_reasons))),
        "side_preference_request": side_preference_request,
        "quote_distance_request": quote_distance_request,
        "size_request": size_request,
        "candidate_action": candidate_action,
        "no_change_reason": no_change_reason,
        "missing_required_fields": "|".join(missing),
    }


def _candidate_row(
    sample: SampleArtifacts,
    label: dict[str, str],
    audit: dict[str, str],
    safety: dict[str, str],
    markout: dict[str, str],
) -> dict[str, Any]:
    request = classify_request(label, audit, safety)
    filled = _order_filled(label)
    order_id = str(label.get("order_id", "")).strip()
    seq = _seq(label)
    spread = _float(markout.get("realized_spread_proxy_ticks"), _float(label.get("realized_spread_proxy_ticks")))
    fee_spread = _float(label.get("fee_adjusted_realized_spread_ticks"), spread)
    markout_5s = _float(markout.get("side_adjusted_markout_ticks"), _float(label.get("fill_markout_5000ms_ticks")))
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "is_caveated": int(sample.is_caveated),
        "order_id": order_id,
        "strategy_seq": "" if seq is None else seq,
        "submit_ts_local": label.get("submit_ts_local", ""),
        "order_side": label.get("order_side", ""),
        "position_before_submit": _safe_num(_float(label.get("position_before_submit"))),
        "inventory_score": _safe_num(_float(label.get("inventory_score"))),
        "edge_vs_fair_ticks": _safe_num(_float(label.get("edge_vs_fair_ticks"))),
        "edge_vs_reservation_ticks": _safe_num(_float(label.get("edge_vs_reservation_ticks"))),
        "distance_to_bbo_ticks": _safe_num(_float(label.get("distance_to_bbo_ticks"))),
        "placement_bucket": label.get("placement_bucket", ""),
        **request,
        "filled": int(filled),
        "fill_count": _safe_num(_float(label.get("fill_count"), 0.0)),
        "fill_rate_denominator": 1,
        "fill_after_cancel_request": int(_bool(label.get("fill_after_cancel_request"))),
        "time_to_fill_ms": _safe_num(_float(label.get("time_to_fill_ms"))),
        "fill_markout_5000ms_ticks": _safe_num(markout_5s),
        "realized_spread_proxy_ticks": _safe_num(spread),
        "fee_adjusted_realized_spread_ticks": _safe_num(fee_spread),
        "inventory_increasing_fill": int(_bool(label.get("inventory_increasing_fill"))),
        "inventory_reducing_fill": int(_bool(label.get("inventory_reducing_fill"))),
        "position_after_fill": _safe_num(_float(label.get("position_after_fill"))),
        "time_to_flat_ms": _safe_num(_float(label.get("time_to_flat_ms"))),
        "quote_update_intent": audit.get("quote_update_intent", ""),
        "quote_update_action": audit.get("quote_update_action", ""),
        "quote_update_reason": audit.get("quote_update_reason", ""),
        "min_move_passed": audit.get("min_move_passed", ""),
        "quote_age_ms": _safe_num(_float(audit.get("quote_age_ms"))),
        "join_age_ms": _safe_num(_float(audit.get("join_age_ms"), _float(label.get("top5_join_age_ms")))),
        "anchor_age_ms": _safe_num(_float(audit.get("anchor_age_ms"), _float(safety.get("anchor_age_ms")))),
        "latency_bucket": audit.get("latency_bucket", ""),
        "throttle_state": audit.get("throttle_state", ""),
        "token_bucket_state": audit.get("token_bucket_state", ""),
        "cancel_readd_bucket": audit.get("cancel_readd_bucket", ""),
        "reject_throttle_drop_cause": audit.get("reject_throttle_drop_cause", ""),
        "post_only_pre_check": audit.get("post_only_pre_check", ""),
        "post_only_post_check": audit.get("post_only_post_check", ""),
        "inventory_request_id": audit.get("inventory_request_id", ""),
        "stage5c_join_missing": safety.get("join_missing", ""),
        "stage5c_join_gap_crossed": safety.get("join_gap_crossed", ""),
        "stage5c_join_stale": safety.get("join_stale", ""),
        "stage5c_post_only_risk_after_recheck": safety.get("post_only_risk_after_recheck", ""),
        "stage5c_diagnostic_reason": safety.get("diagnostic_reason", ""),
        "evaluation_status": "observed_submit_proxy",
    }
    return row


def _load_candidate_rows(sample: SampleArtifacts) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sample.stage5_labels is None:
        return [], {"sample_id": sample.sample_id, "status": sample.status, "rows": 0, "notes": list(sample.notes)}
    labels = _read_csv(sample.stage5_labels)
    seqs = {seq for row in labels if (seq := _seq(row)) is not None}
    audit_by_seq = _read_audit_by_seq(sample.audit_live, seqs)
    safety_by_seq = _read_stage5c_by_seq(sample.stage5c_safety, seqs)
    markouts = _markouts_by_order(sample.fill_markouts)
    candidate_rows = []
    for label in labels:
        seq = _seq(label)
        order_id = str(label.get("order_id", "")).strip()
        candidate_rows.append(
            _candidate_row(
                sample,
                label,
                audit_by_seq.get(int(seq), {}) if seq is not None else {},
                safety_by_seq.get(int(seq), {}) if seq is not None else {},
                markouts.get(order_id, {}),
            )
        )
    return candidate_rows, {
        "sample_id": sample.sample_id,
        "status": sample.status,
        "is_caveated": sample.is_caveated,
        "rows": len(candidate_rows),
        "stage5_labels": str(sample.stage5_labels),
        "fill_markouts": str(sample.fill_markouts) if sample.fill_markouts else "",
        "audit_live": str(sample.audit_live) if sample.audit_live else "",
        "stage5c_safety": str(sample.stage5c_safety) if sample.stage5c_safety else "",
        "audit_seq_join_coverage": _safe_num(_rate(len(audit_by_seq), len(seqs))),
        "stage5c_seq_join_coverage": _safe_num(_rate(len(safety_by_seq), len(seqs))),
        "notes": list(sample.notes),
    }


def _sum_bool(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if _bool(row.get(key)))


def _num(row: dict[str, Any], key: str) -> float:
    return _float(row.get(key))


def _metric_row(group: tuple[str, ...], rows: list[dict[str, Any]], prefix_fields: list[str]) -> dict[str, Any]:
    filled = [row for row in rows if _bool(row.get("filled"))]
    request_rows = [row for row in rows if row.get("candidate_action") != "no_change"]
    out: dict[str, Any] = {field: value for field, value in zip(prefix_fields, group)}
    out.update(
        {
            "rows": len(rows),
            "fills": len(filled),
            "fill_rate": _safe_num(_rate(len(filled), len(rows))),
            "candidate_action_rows": len(request_rows),
            "candidate_action_coverage": _safe_num(_rate(len(request_rows), len(rows))),
            "markout_5000ms_mean": _safe_num(_mean(_num(row, "fill_markout_5000ms_ticks") for row in filled)),
            "spread_capture_mean": _safe_num(_mean(_num(row, "realized_spread_proxy_ticks") for row in filled)),
            "fee_adjusted_spread_mean": _safe_num(_mean(_num(row, "fee_adjusted_realized_spread_ticks") for row in filled)),
            "fill_after_cancel_rate": _safe_num(_rate(_sum_bool(filled, "fill_after_cancel_request"), len(filled))),
            "inventory_increasing_fill_rate": _safe_num(_rate(_sum_bool(filled, "inventory_increasing_fill"), len(filled))),
            "inventory_reducing_fill_rate": _safe_num(_rate(_sum_bool(filled, "inventory_reducing_fill"), len(filled))),
            "post_only_risk_after_recheck_rate": _safe_num(_rate(_sum_bool(rows, "stage5c_post_only_risk_after_recheck"), len(rows))),
            "reject_throttle_drop_rate": _safe_num(_rate(sum(1 for row in rows if str(row.get("reject_throttle_drop_cause", "")).strip()), len(rows))),
            "fast_cancel_or_readd_rate": _safe_num(_rate(sum(1 for row in rows if str(row.get("cancel_readd_bucket", "")).strip()), len(rows))),
        }
    )
    return out


def _group_metrics(rows: list[dict[str, Any]], keys: list[str], scope: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in keys)].append(row)
    out = []
    for group, group_rows in sorted(groups.items()):
        metric = _metric_row(group, group_rows, keys)
        metric["scope"] = scope
        out.append(metric)
    return out


def _scope_rows(rows: list[dict[str, Any]], include_caveated: bool) -> list[dict[str, Any]]:
    if include_caveated:
        return rows
    return [row for row in rows if not _bool(row.get("is_caveated"))]


def _summary_for_rows(rows: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    request_rows = [row for row in rows if row.get("candidate_action") != "no_change"]
    request_fills = [row for row in request_rows if _bool(row.get("filled"))]
    no_change_rows = [row for row in rows if row.get("candidate_action") == "no_change"]
    no_change_fills = [row for row in no_change_rows if _bool(row.get("filled"))]
    action_samples = {row.get("sample_id") for row in request_rows}
    direction_rows = [
        row
        for row in request_fills
        if row.get("side_class") == "reduce_side" or row.get("size_request") in {"suppress_add_side", "reduce_add_side_size"}
    ]
    return {
        "scope": scope,
        "sample_count": len({row.get("sample_id") for row in rows}),
        "rows": len(rows),
        "fills": _sum_bool(rows, "filled"),
        "fill_rate": _safe_num(_rate(_sum_bool(rows, "filled"), len(rows))),
        "request_rows": len(request_rows),
        "request_row_share": _safe_num(_rate(len(request_rows), len(rows))),
        "request_fills": len(request_fills),
        "request_fill_rate": _safe_num(_rate(len(request_fills), len(request_rows))),
        "request_sample_count": len(action_samples),
        "no_change_rows": len(no_change_rows),
        "no_change_fill_rate": _safe_num(_rate(len(no_change_fills), len(no_change_rows))),
        "request_markout_5000ms_mean": _safe_num(_mean(_num(row, "fill_markout_5000ms_ticks") for row in request_fills)),
        "no_change_markout_5000ms_mean": _safe_num(_mean(_num(row, "fill_markout_5000ms_ticks") for row in no_change_fills)),
        "request_spread_capture_mean": _safe_num(_mean(_num(row, "realized_spread_proxy_ticks") for row in request_fills)),
        "no_change_spread_capture_mean": _safe_num(_mean(_num(row, "realized_spread_proxy_ticks") for row in no_change_fills)),
        "fill_after_cancel_rate": _safe_num(_rate(_sum_bool(request_fills, "fill_after_cancel_request"), len(request_fills))),
        "inventory_reducing_fill_rate": _safe_num(_rate(_sum_bool(direction_rows, "inventory_reducing_fill"), len(direction_rows))),
        "inventory_increasing_fill_rate": _safe_num(_rate(_sum_bool(request_fills, "inventory_increasing_fill"), len(request_fills))),
        "post_only_risk_after_recheck_rows": _sum_bool(request_rows, "stage5c_post_only_risk_after_recheck"),
        "direction_consistency_proxy": _safe_num(_rate(_sum_bool(direction_rows, "inventory_reducing_fill"), len(direction_rows))),
        "fill_mass_sufficient": int(len(request_fills) >= MIN_CLEAN_REQUEST_FILLS),
        "metric_status": "observed_only_proxy",
    }


def _verdict(clean: dict[str, Any], caveated: dict[str, Any]) -> tuple[str, str]:
    if int(clean.get("rows", 0)) == 0 or int(clean.get("request_rows", 0)) == 0:
        if int(caveated.get("request_fills", 0)) >= MIN_CLEAN_REQUEST_FILLS:
            return "caveated_only", "Only caveated samples have enough candidate support; clean-only evidence is not decisionable."
        return "not_decisionable", "No clean candidate action coverage."
    if int(clean.get("post_only_risk_after_recheck_rows", 0)) > 0:
        return "reject", "Candidate rows overlap post-only recheck risk in clean samples."
    request_fills = int(clean.get("request_fills", 0))
    if request_fills < MIN_CLEAN_REQUEST_FILLS:
        return "needs_more_clean_fills", "Clean request fill mass is below the fixed sufficiency threshold."
    request_fill_rate = _float(clean.get("request_fill_rate"))
    no_change_fill_rate = _float(clean.get("no_change_fill_rate"))
    request_markout = _float(clean.get("request_markout_5000ms_mean"))
    no_change_markout = _float(clean.get("no_change_markout_5000ms_mean"))
    request_spread = _float(clean.get("request_spread_capture_mean"))
    no_change_spread = _float(clean.get("no_change_spread_capture_mean"))
    fill_loss_large = _finite(request_fill_rate) and _finite(no_change_fill_rate) and request_fill_rate < no_change_fill_rate * 0.65
    quality_better = _finite(request_markout) and _finite(no_change_markout) and request_markout >= no_change_markout
    quality_worse = (
        _finite(request_markout)
        and _finite(no_change_markout)
        and _finite(request_spread)
        and _finite(no_change_spread)
        and request_markout < no_change_markout
        and request_spread < no_change_spread
    )
    if fill_loss_large and not quality_better:
        return "too_conservative_fill_loss", "Observed request buckets would risk too much fill participation without a quality offset."
    if quality_worse:
        return "reject", "Clean request buckets have enough fill mass but worse 5s markout and spread capture than no-change buckets."
    if quality_better:
        return "promising_but_needs_parameter_sweep", "Clean observed buckets are distinguishable, but thresholds and size multipliers still require a separate bounded design."
    return "needs_more_clean_fills", "Clean fill mass is present, but the quality direction is too mixed for a design task."


def build_outputs(rows: list[dict[str, Any]], include_caveated: bool) -> dict[str, list[dict[str, Any]] | dict[str, Any]]:
    accepted_rows = _scope_rows(rows, include_caveated)
    clean_rows = _scope_rows(rows, False)
    caveated_rows = [row for row in rows if _bool(row.get("is_caveated"))]

    bucket_keys = ["inventory_bucket", "side_class", "edge_bucket", "quote_distance_bucket", "candidate_action"]
    bucket_metrics = _group_metrics(clean_rows, bucket_keys, "clean_only")
    if include_caveated:
        bucket_metrics.extend(_group_metrics(accepted_rows, bucket_keys, "accepted_set"))

    clean_summary = _summary_for_rows(clean_rows, "clean_only")
    caveated_summary = _summary_for_rows(caveated_rows, "caveated_only")
    accepted_summary = _summary_for_rows(accepted_rows, "accepted_set")
    verdict, reason = _verdict(clean_summary, caveated_summary)
    caveated_verdict, caveated_reason = _verdict(caveated_summary, caveated_summary)
    clean_summary["clean_only_verdict"] = verdict
    clean_summary["verdict_reason"] = reason
    clean_summary["verdict_taxonomy"] = "|".join(VERDICT_TAXONOMY)

    sensitivity = [
        accepted_summary | {"sensitivity_verdict": verdict, "notes": "accepted_set_includes_caveated_when_enabled"},
        caveated_summary | {"sensitivity_verdict": caveated_verdict, "notes": caveated_reason},
    ]

    participation = _group_metrics(accepted_rows, ["candidate_action", "quote_distance_request", "size_request"], "accepted_set")
    for row in participation:
        fill_rate = _float(row.get("fill_rate"))
        clean_baseline = _float(clean_summary.get("no_change_fill_rate"))
        row["fill_rate_delta_vs_clean_no_change"] = _safe_num(fill_rate - clean_baseline if _finite(fill_rate) and _finite(clean_baseline) else math.nan)
        row["observed_only_proxy_note"] = "participation/fill-loss is based on observed submits, not counterfactual skipped orders"

    inventory = _group_metrics(accepted_rows, ["inventory_bucket", "side_class", "candidate_action"], "accepted_set")
    mechanics = _group_metrics(accepted_rows, ["candidate_action", "quote_distance_request", "safety_context"], "accepted_set")

    return {
        "bucket_metrics": bucket_metrics,
        "clean_summary": [clean_summary],
        "sensitivity": sensitivity,
        "participation": participation,
        "inventory": inventory,
        "mechanics": mechanics,
        "recommendation": {
            "clean_only_verdict": verdict,
            "clean_only_reason": reason,
            "caveated_sensitivity_verdict": caveated_verdict,
            "caveated_sensitivity_reason": caveated_reason,
            "supports_later_parameter_design": verdict in {"focused_design_promising", "promising_but_needs_parameter_sweep"},
        },
    }


def _write_recommendation(path: Path, recommendation: dict[str, Any], sample_notes: list[dict[str, Any]]) -> None:
    lines = [
        "# 0528T001 Inventory-Aware Quote Placement Recommendation",
        "",
        f"- clean_only_verdict: `{recommendation['clean_only_verdict']}`",
        f"- clean_only_reason: {recommendation['clean_only_reason']}",
        f"- caveated_sensitivity_verdict: `{recommendation['caveated_sensitivity_verdict']}`",
        f"- caveated_sensitivity_reason: {recommendation['caveated_sensitivity_reason']}",
        f"- supports_later_parameter_design: `{recommendation['supports_later_parameter_design']}`",
        "",
        "Boundary: read-only/default-off offline runner only. No live run, no strategy change, no default-on behavior, no parameter search, and no promotion claim.",
        "",
        "Proxy limitations: participation/fill-loss, queue effects, opportunity cost, and PnL decomposition are observed-only proxies from submitted orders. They are not counterfactual simulation of unsubmitted quotes.",
        "",
        "Samples:",
    ]
    for note in sample_notes:
        lines.append(f"- {note['sample_id']}: status={note['status']}, caveated={note.get('is_caveated', False)}, notes={','.join(note.get('notes', [])) or 'none'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_inventory_aware_quote_placement(
    run_dirs: list[Path],
    output_dir: Path,
    include_caveated: bool = True,
    caveated_ids: set[str] | None = None,
) -> dict[str, Any]:
    caveated_ids = set(caveated_ids or DEFAULT_CAVEATED_SAMPLE_IDS)
    samples = [discover_sample(run_dir, caveated_ids) for run_dir in run_dirs]
    rows: list[dict[str, Any]] = []
    sample_notes: list[dict[str, Any]] = []
    for sample in samples:
        sample_rows, note = _load_candidate_rows(sample)
        rows.extend(sample_rows)
        sample_notes.append(note)

    outputs = build_outputs(rows, include_caveated)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "candidate_decision_rows.csv", rows)
    _write_csv(output_dir / "bucket_metrics_by_inventory_side_edge_distance.csv", outputs["bucket_metrics"])  # type: ignore[arg-type]
    _write_csv(output_dir / "clean_only_stability_summary.csv", outputs["clean_summary"])  # type: ignore[arg-type]
    _write_csv(output_dir / "caveated_sample_sensitivity.csv", outputs["sensitivity"])  # type: ignore[arg-type]
    _write_csv(output_dir / "participation_and_fill_loss.csv", outputs["participation"])  # type: ignore[arg-type]
    _write_csv(output_dir / "inventory_recovery_quality.csv", outputs["inventory"])  # type: ignore[arg-type]
    _write_csv(output_dir / "quote_mechanics_safety.csv", outputs["mechanics"])  # type: ignore[arg-type]
    _write_recommendation(output_dir / "candidate_recommendation.md", outputs["recommendation"], sample_notes)  # type: ignore[arg-type]
    manifest = {
        "task_id": TASK_ID,
        "runner_mode": RUNNER_MODE,
        "generated_at": _generated_at(),
        "runner_path": "examples/binance_tick_mm/inventory_aware_quote_placement.py",
        "output_dir": str(output_dir),
        "include_caveated": include_caveated,
        "verdict_taxonomy": list(VERDICT_TAXONOMY),
        "policy_contract": "inventory_aware_quote_placement_request",
        "decision_inputs": [
            "order_side",
            "position_before_submit",
            "inventory_score",
            "edge_vs_fair_ticks",
            "edge_vs_reservation_ticks",
            "placement_bucket",
            "distance_to_bbo_ticks",
            "post_only/anchor/latency/stale safety context gates",
            *REQUIRED_T006_FIELDS,
        ],
        "forbidden_inputs_not_used_for_request_generation": [
            "future_fill",
            "future_markout",
            "future_spread_capture",
            "fill_after_cancel_outcome",
            "same_sample_pnl_feedback",
            "exact_queue_position",
            "hidden_queue_assumptions",
        ],
        "boundary": {
            "read_only": True,
            "default_off": True,
            "live_run": False,
            "strategy_change": False,
            "parameter_search": False,
            "promotion": False,
        },
        "proxy_limitations": [
            "participation_and_fill_loss is observed-only from existing submitted orders",
            "queue / opportunity cost / pnl decomposition are observed-only proxies",
            "caveated samples are sensitivity only, not strict-clean proof",
        ],
        "sample_notes": sample_notes,
        "recommendation": outputs["recommendation"],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for Stage 9I output artifacts.")
    parser.add_argument("--sample", type=Path, action="append", default=None, help="Sample run directory. Repeatable. Defaults to the accepted current-format sample set.")
    parser.add_argument("--include-caveated", action=argparse.BooleanOptionalAction, default=True, help="Include caveated samples in accepted-set sensitivity outputs. Clean-only remains primary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dirs = args.sample if args.sample else DEFAULT_RUN_DIRS
    manifest = run_inventory_aware_quote_placement(run_dirs, args.output_dir, include_caveated=args.include_caveated)
    recommendation = manifest["recommendation"]
    print(f"wrote {args.output_dir}")
    print(f"clean_only_verdict={recommendation['clean_only_verdict']}")
    print(f"caveated_sensitivity_verdict={recommendation['caveated_sensitivity_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
