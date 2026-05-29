#!/usr/bin/env python3
"""Read-only Stage 9K fill-quality bucket synthesis.

This runner consumes existing current-format live-run artifacts and summarizes
observed fill quality by decision-visible buckets. It does not change strategy
behavior, run replay, infer counterfactual fills, search parameters, or make
promotion claims.
"""

from __future__ import annotations

import argparse
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


TASK_ID = "0529T002"
RUNNER_MODE = "stage9k_fill_quality_bucket_synthesis"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/stage9k_fill_quality_bucket_synthesis_0529T002")
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
    "ready_for_policy_design",
    "needs_more_clean_fills",
    "reject_quality_negative",
    "not_decisionable",
)

MIN_CLEAN_FILLS = 40
MIN_CLEAN_SAMPLES = 3
MIN_SHAPE_FILLS = 60
MIN_SAMPLE_FILL_COVERAGE = 2
MARKOUT_FLOOR_TICKS = -35.0
SPREAD_FLOOR_TICKS = 5.0
FILL_AFTER_CANCEL_MAX = 0.55
UNSAFE_RATE_MAX = 0.0
REJECT_THROTTLE_CHURN_MAX = 0.05
POSITION_FLAT_THRESHOLD = 0.0005
POSITION_LARGE_THRESHOLD = 0.0015
LOW_INVENTORY_SCORE_THRESHOLD = 0.35
STRONG_EDGE_TICKS = 5.0
NEGATIVE_MARKOUT_TICKS = -50.0
HIGH_SPREAD_TICKS = 15.0

BUCKET_KEYS = [
    "inventory_bucket",
    "side_class",
    "quote_distance_bucket",
    "fair_or_reservation_edge_bucket",
    "latency_stale_bucket",
    "post_only_safety_bucket",
    "reject_throttle_churn_bucket",
    "fill_after_cancel_bucket",
]
TRIGGER_BUCKET_KEYS = [
    "inventory_bucket",
    "side_class",
    "quote_distance_bucket",
    "fair_or_reservation_edge_bucket",
    "latency_stale_bucket",
    "post_only_safety_bucket",
    "reject_throttle_churn_bucket",
]
METRIC_FIELDS = [
    "bucket_level",
    "scope",
    "sample_count",
    "fill_sample_count",
    "rows",
    "fills",
    "fill_rate",
    "clean_only_fill_mass",
    "side_adjusted_markout_5000ms_mean",
    "spread_capture_ticks_mean",
    "fee_adjusted_spread_capture_ticks_mean",
    "fill_after_cancel_rate",
    "inventory_increasing_fill_rate",
    "inventory_reducing_fill_rate",
    "post_only_risk_rows",
    "post_only_risk_rate",
    "reject_throttle_churn_rows",
    "reject_throttle_churn_rate",
    "verdict",
    "verdict_reason",
]


@dataclass(frozen=True)
class SampleArtifacts:
    sample_id: str
    run_dir: Path
    is_caveated: bool
    stage5_labels: Path | None
    fill_markouts: Path | None
    audit_live: Path | None
    stage5c_safety: Path | None
    stage6_manifest: Path | None
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


def _max_finite(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return max(finite) if finite else math.nan


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


def _discover_stage6_manifest(run_dir: Path) -> Path | None:
    matches = sorted(run_dir.glob("stage6*/*run_manifest.json"))
    return matches[-1] if matches else None


def discover_sample(run_dir: Path, caveated_ids: set[str]) -> SampleArtifacts:
    notes: list[str] = []
    labels = _stage5_path(run_dir, "execution_outcome_labels.csv")
    markouts = _stage5_path(run_dir, "fill_markout_labels.csv")
    audit = _discover_audit(run_dir)
    safety = _discover_stage5c(run_dir)
    stage6_manifest = _discover_stage6_manifest(run_dir)
    if not run_dir.exists():
        notes.append("run_dir_missing")
    labels_path = labels if labels.exists() else None
    if labels_path is None:
        notes.append("stage5_execution_labels_missing")
    markout_path = markouts if markouts.exists() else None
    if markout_path is None:
        notes.append("fill_markout_labels_missing")
    if audit is None:
        notes.append("audit_live_missing")
    if safety is None:
        notes.append("stage5c_safety_missing")
    if stage6_manifest is None:
        notes.append("stage6_manifest_missing")
    status = "usable" if labels_path is not None else "missing_required_artifact"
    return SampleArtifacts(
        sample_id=_sample_id(run_dir),
        run_dir=run_dir,
        is_caveated=_sample_id(run_dir) in caveated_ids,
        stage5_labels=labels_path,
        fill_markouts=markout_path,
        audit_live=audit,
        stage5c_safety=safety,
        stage6_manifest=stage6_manifest,
        status=status,
        notes=tuple(notes),
    )


def seq_from_label(row: dict[str, str]) -> int | None:
    for key in ("decision_context_strategy_seq", "submit_strategy_seq", "strategy_seq", "linked_strategy_seq"):
        value = _int(row.get(key))
        if value is not None:
            return value
    return None


def order_id(row: dict[str, str]) -> str:
    return str(row.get("order_id", "")).strip()


def is_filled(row: dict[str, Any]) -> bool:
    return _float(row.get("fill_count"), 0.0) > 0.0 or _bool(row.get("full_fill")) or _bool(row.get("partial_fill"))


def classify_inventory_bucket(position: float, inventory_score: float) -> str:
    if not _finite(position):
        return "inventory_unknown"
    abs_position = abs(position)
    if abs_position <= POSITION_FLAT_THRESHOLD:
        return "flat"
    if abs_position >= POSITION_LARGE_THRESHOLD or (_finite(inventory_score) and inventory_score <= LOW_INVENTORY_SCORE_THRESHOLD):
        return "large_skew_or_low_score"
    return "mild_skew"


def classify_side_class(order_side: str, position: float) -> str:
    side = str(order_side or "").strip().lower()
    if side not in {"buy", "sell"} or not _finite(position) or abs(position) <= POSITION_FLAT_THRESHOLD:
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
    conservative = min(edges)
    if conservative < 0.0:
        return "edge_adverse"
    if conservative >= STRONG_EDGE_TICKS:
        return "edge_strong_favorable"
    return "edge_weak_or_neutral"


def classify_quote_distance_bucket(placement_bucket: str, distance_to_bbo_ticks: float) -> str:
    placement = str(placement_bucket or "").strip().lower()
    if placement in {"touch", "one_tick_tight", "step_back_gt1", "outside_or_no_quote"}:
        return placement
    if "touch" in placement:
        return "touch"
    if "one" in placement or placement == "1":
        return "one_tick_tight"
    if "outside" in placement or "no_quote" in placement:
        return "outside_or_no_quote"
    if "step" in placement:
        return "step_back_gt1"
    distance = abs(distance_to_bbo_ticks)
    if not _finite(distance):
        return "quote_distance_unknown"
    if distance <= 0.5:
        return "touch"
    if distance <= 1.5:
        return "one_tick_tight"
    return "step_back_gt1"


def classify_latency_stale_bucket(label: dict[str, str], audit: dict[str, str], safety: dict[str, str]) -> str:
    latency = _max_finite(
        [
            _float(label.get("latency_signal_ms")),
            _float(label.get("feed_latency_ms")),
            _float(audit.get("join_age_ms")),
        ]
    )
    stale = _max_finite(
        [
            _float(label.get("book_view_stale_ms")),
            _float(label.get("top5_join_age_ms")),
            _float(audit.get("quote_age_ms")),
            _float(audit.get("anchor_age_ms")),
            _float(safety.get("anchor_age_ms")),
        ]
    )
    if _bool(label.get("join_gap_crossed")) or _bool(safety.get("join_gap_crossed")):
        return "stale_latency_gap_crossed"
    if _bool(label.get("join_stale")) or _bool(safety.get("join_stale")):
        return "stale_latency_join_stale"
    if (_finite(latency) and latency >= 50.0) or (_finite(stale) and stale >= 250.0):
        return "stale_latency_high"
    if (_finite(latency) and latency >= 5.0) or (_finite(stale) and stale >= 50.0):
        return "stale_latency_medium"
    return "fresh_low_latency"


def classify_post_only_safety_bucket(label: dict[str, str], audit: dict[str, str], safety: dict[str, str]) -> str:
    if _bool(label.get("post_only_risk")) or _bool(audit.get("post_only_pre_check")) or _bool(audit.get("post_only_post_check")):
        return "post_only_risk"
    if _bool(safety.get("post_only_risk_after_recheck")):
        return "post_only_risk_after_recheck"
    if _bool(safety.get("missing_anchor")):
        return "missing_anchor"
    if _bool(safety.get("stale_anchor")):
        return "stale_anchor"
    if _bool(safety.get("suppress_buy")) or _bool(safety.get("suppress_sell")):
        return "suppressed"
    if _bool(safety.get("bid_clamped")) or _bool(safety.get("ask_clamped")):
        return "clamped"
    if _bool(safety.get("depth_fallback_used")):
        return "guarded_depth_fallback"
    return "post_only_clean"


def classify_reject_throttle_churn_bucket(label: dict[str, str], audit: dict[str, str]) -> str:
    if str(audit.get("reject_throttle_drop_cause", "")).strip():
        return "reject_throttle_or_drop"
    if _float(label.get("recent_reject_count_500ms"), 0.0) > 0.0 or _float(label.get("recent_throttle_count_500ms"), 0.0) > 0.0:
        return "recent_reject_or_throttle"
    if _bool(label.get("fast_cancel_churn")):
        return "fast_cancel_churn"
    cancel_bucket = str(audit.get("cancel_readd_bucket", "")).strip().lower()
    if cancel_bucket and cancel_bucket not in {"none", "normal", "0"}:
        return "cancel_readd_pressure"
    if audit.get("min_move_passed") != "" and not _bool(audit.get("min_move_passed")):
        return "min_move_failed"
    return "churn_normal"


def classify_fill_after_cancel_bucket(label: dict[str, str], filled: bool) -> str:
    if not filled:
        return "no_fill"
    if _bool(label.get("fill_after_cancel_request")):
        delay = _float(label.get("cancel_to_fill_delay_ms"))
        if _finite(delay) and delay <= 250.0:
            return "fill_after_cancel_fast"
        return "fill_after_cancel_slow_or_unknown"
    return "filled_no_cancel_race"


def _markouts_by_order(path: Path | None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    if path is None:
        return out
    for row in _read_csv(path):
        if _int(row.get("horizon_ms")) != 5000 or not _bool(row.get("horizon_observable")):
            continue
        oid = order_id(row)
        if oid:
            out[oid] = row
    return out


def _read_stage5c_by_seq(path: Path | None, seqs: set[int]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    if path is None:
        return out
    for row in _read_csv(path):
        seq = _int(row.get("strategy_seq"))
        if seq in seqs:
            out[int(seq)] = row
    return out


def _read_audit_by_seq(path: Path | None, seqs: set[int]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    if path is None:
        return out
    for row in _read_csv(path):
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


def synthesize_label_row(
    sample: SampleArtifacts,
    label: dict[str, str],
    audit: dict[str, str] | None = None,
    safety: dict[str, str] | None = None,
    markout: dict[str, str] | None = None,
) -> dict[str, Any]:
    audit = audit or {}
    safety = safety or {}
    markout = markout or {}
    position = _float(label.get("position_before_submit"))
    inventory_score = _float(label.get("inventory_score"))
    edge_fair = _float(label.get("edge_vs_fair_ticks"))
    edge_reservation = _float(label.get("edge_vs_reservation_ticks"))
    distance = _float(label.get("distance_to_bbo_ticks"))
    filled = is_filled(label)
    markout_5s = _float(markout.get("side_adjusted_markout_ticks"), _float(label.get("fill_markout_5000ms_ticks")))
    spread = _float(markout.get("realized_spread_proxy_ticks"), _float(label.get("realized_spread_proxy_ticks")))
    fee_spread = _float(markout.get("fee_adjusted_realized_spread_ticks"), _float(label.get("fee_adjusted_realized_spread_ticks"), spread))
    post_only_bucket = classify_post_only_safety_bucket(label, audit, safety)
    churn_bucket = classify_reject_throttle_churn_bucket(label, audit)
    row: dict[str, Any] = {
        "sample_id": sample.sample_id,
        "is_caveated": int(sample.is_caveated),
        "order_id": order_id(label),
        "strategy_seq": "" if seq_from_label(label) is None else seq_from_label(label),
        "order_side": label.get("order_side", ""),
        "inventory_bucket": classify_inventory_bucket(position, inventory_score),
        "side_class": classify_side_class(label.get("order_side", ""), position),
        "quote_distance_bucket": classify_quote_distance_bucket(label.get("placement_bucket", ""), distance),
        "fair_or_reservation_edge_bucket": classify_edge_bucket(edge_fair, edge_reservation),
        "latency_stale_bucket": classify_latency_stale_bucket(label, audit, safety),
        "post_only_safety_bucket": post_only_bucket,
        "reject_throttle_churn_bucket": churn_bucket,
        "fill_after_cancel_bucket": classify_fill_after_cancel_bucket(label, filled),
        "filled": int(filled),
        "fill_count": _safe_num(_float(label.get("fill_count"), 0.0)),
        "fill_rate_denominator": 1,
        "side_adjusted_markout_5000ms_ticks": _safe_num(markout_5s),
        "spread_capture_ticks": _safe_num(spread),
        "fee_adjusted_spread_capture_ticks": _safe_num(fee_spread),
        "fill_after_cancel_request": int(_bool(label.get("fill_after_cancel_request"))),
        "inventory_increasing_fill": int(_bool(label.get("inventory_increasing_fill"))),
        "inventory_reducing_fill": int(_bool(label.get("inventory_reducing_fill"))),
        "post_only_risk": int(post_only_bucket in {"post_only_risk", "post_only_risk_after_recheck"}),
        "reject_throttle_churn_risk": int(churn_bucket != "churn_normal"),
        "evaluation_status": "observed_submit_fill_quality",
    }
    return row


def _load_fill_quality_rows(sample: SampleArtifacts) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if sample.stage5_labels is None:
        return [], {"sample_id": sample.sample_id, "status": sample.status, "rows": 0, "notes": list(sample.notes)}
    labels = _read_csv(sample.stage5_labels)
    seqs = {seq for row in labels if (seq := seq_from_label(row)) is not None}
    audit_by_seq = _read_audit_by_seq(sample.audit_live, seqs)
    safety_by_seq = _read_stage5c_by_seq(sample.stage5c_safety, seqs)
    markouts = _markouts_by_order(sample.fill_markouts)
    rows: list[dict[str, Any]] = []
    for label in labels:
        seq = seq_from_label(label)
        rows.append(
            synthesize_label_row(
                sample,
                label,
                audit_by_seq.get(int(seq), {}) if seq is not None else {},
                safety_by_seq.get(int(seq), {}) if seq is not None else {},
                markouts.get(order_id(label), {}),
            )
        )
    return rows, {
        "sample_id": sample.sample_id,
        "status": sample.status,
        "is_caveated": sample.is_caveated,
        "rows": len(rows),
        "stage5_labels": str(sample.stage5_labels),
        "fill_markouts": str(sample.fill_markouts) if sample.fill_markouts else "",
        "audit_live": str(sample.audit_live) if sample.audit_live else "",
        "stage5c_safety": str(sample.stage5c_safety) if sample.stage5c_safety else "",
        "stage6_manifest": str(sample.stage6_manifest) if sample.stage6_manifest else "",
        "audit_seq_join_coverage": _safe_num(_rate(len(audit_by_seq), len(seqs))),
        "stage5c_seq_join_coverage": _safe_num(_rate(len(safety_by_seq), len(seqs))),
        "notes": list(sample.notes),
    }


def _num(row: dict[str, Any], key: str) -> float:
    return _float(row.get(key))


def _sum_bool(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if _bool(row.get(key)))


def _metric_row(
    group: tuple[str, ...],
    rows: list[dict[str, Any]],
    keys: list[str],
    scope: str,
    *,
    bucket_level: str,
) -> dict[str, Any]:
    filled = [row for row in rows if _bool(row.get("filled"))]
    sample_ids = {str(row.get("sample_id", "")) for row in rows}
    fill_sample_ids = {str(row.get("sample_id", "")) for row in filled}
    out: dict[str, Any] = {key: value for key, value in zip(keys, group)}
    if "fill_after_cancel_bucket" not in out:
        out["fill_after_cancel_bucket"] = "all_fill_after_cancel_outcomes"
    fill_count = len(filled)
    post_only_risk_rows = _sum_bool(rows, "post_only_risk")
    reject_throttle_churn_rows = _sum_bool(rows, "reject_throttle_churn_risk")
    out.update(
        {
            "bucket_level": bucket_level,
            "scope": scope,
            "sample_count": len(sample_ids),
            "fill_sample_count": len(fill_sample_ids),
            "rows": len(rows),
            "fills": fill_count,
            "fill_rate": _safe_num(_rate(fill_count, len(rows))),
            "clean_only_fill_mass": fill_count if scope == "clean_only" else "",
            "side_adjusted_markout_5000ms_mean": _safe_num(_mean(_num(row, "side_adjusted_markout_5000ms_ticks") for row in filled)),
            "spread_capture_ticks_mean": _safe_num(_mean(_num(row, "spread_capture_ticks") for row in filled)),
            "fee_adjusted_spread_capture_ticks_mean": _safe_num(_mean(_num(row, "fee_adjusted_spread_capture_ticks") for row in filled)),
            "fill_after_cancel_rate": _safe_num(_rate(_sum_bool(filled, "fill_after_cancel_request"), fill_count)),
            "inventory_increasing_fill_rate": _safe_num(_rate(_sum_bool(filled, "inventory_increasing_fill"), fill_count)),
            "inventory_reducing_fill_rate": _safe_num(_rate(_sum_bool(filled, "inventory_reducing_fill"), fill_count)),
            "post_only_risk_rows": post_only_risk_rows,
            "post_only_risk_rate": _safe_num(_rate(post_only_risk_rows, len(rows))),
            "reject_throttle_churn_rows": reject_throttle_churn_rows,
            "reject_throttle_churn_rate": _safe_num(_rate(reject_throttle_churn_rows, len(rows))),
        }
    )
    verdict, reason = assign_bucket_verdict(out)
    out["verdict"] = verdict
    out["verdict_reason"] = reason
    return out


def build_bucket_metrics(rows: list[dict[str, Any]], *, scope: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in BUCKET_KEYS)].append(row)
    return [
        _metric_row(group, group_rows, BUCKET_KEYS, scope, bucket_level="fill_after_cancel_sensitivity")
        for group, group_rows in sorted(groups.items())
    ]


def build_trigger_metrics(rows: list[dict[str, Any]], *, scope: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row.get(key, "")) for key in TRIGGER_BUCKET_KEYS)].append(row)
    return [
        _metric_row(group, group_rows, TRIGGER_BUCKET_KEYS, scope, bucket_level="decision_visible_trigger")
        for group, group_rows in sorted(groups.items())
    ]


def assign_bucket_verdict(metric: dict[str, Any]) -> tuple[str, str]:
    rows = int(_float(metric.get("rows"), 0.0))
    fills = int(_float(metric.get("fills"), 0.0))
    sample_count = int(_float(metric.get("sample_count"), 0.0))
    fill_sample_count = int(_float(metric.get("fill_sample_count"), 0.0))
    markout = _float(metric.get("side_adjusted_markout_5000ms_mean"))
    spread = _float(metric.get("spread_capture_ticks_mean"))
    fac = _float(metric.get("fill_after_cancel_rate"))
    post_only = _float(metric.get("post_only_risk_rate"), 0.0)
    churn = _float(metric.get("reject_throttle_churn_rate"), 0.0)

    if rows == 0:
        return "not_decisionable", "bucket has no observed submit rows"
    if post_only > UNSAFE_RATE_MAX or churn > REJECT_THROTTLE_CHURN_MAX:
        return "reject_quality_negative", "bucket overlaps post-only or reject/throttle/churn risk"
    if fills < MIN_CLEAN_FILLS or sample_count < MIN_CLEAN_SAMPLES or fill_sample_count < MIN_SAMPLE_FILL_COVERAGE:
        return "needs_more_clean_fills", "clean fill mass or sample coverage is below threshold"
    if _finite(markout) and markout <= NEGATIVE_MARKOUT_TICKS:
        return "reject_quality_negative", "5s side-adjusted markout is materially adverse"
    if _finite(spread) and spread < 0.0:
        return "reject_quality_negative", "spread capture is negative"
    if _finite(fac) and fac > FILL_AFTER_CANCEL_MAX:
        return "reject_quality_negative", "fill-after-cancel sensitivity is elevated"
    if _finite(markout) and markout >= MARKOUT_FLOOR_TICKS and _finite(spread) and spread >= SPREAD_FLOOR_TICKS:
        return "ready_for_policy_design", "clean bucket has enough fills with acceptable markout, spread, and safety"
    return "not_decisionable", "quality metrics are mixed or missing despite enough fills"


def _shape_a_candidate(metric: dict[str, Any]) -> bool:
    if metric.get("verdict") != "ready_for_policy_design":
        return False
    if int(_float(metric.get("fills"), 0.0)) < MIN_SHAPE_FILLS:
        return False
    if metric.get("quote_distance_bucket") not in {"one_tick_tight", "step_back_gt1"}:
        return False
    if str(metric.get("latency_stale_bucket")) not in {"fresh_low_latency", "stale_latency_medium"}:
        return False
    if metric.get("post_only_safety_bucket") in {"post_only_risk", "post_only_risk_after_recheck", "missing_anchor", "stale_anchor"}:
        return False
    return True


def _shape_b_candidate(metric: dict[str, Any]) -> bool:
    if metric.get("verdict") != "ready_for_policy_design":
        return False
    if int(_float(metric.get("fills"), 0.0)) < MIN_SHAPE_FILLS:
        return False
    if metric.get("side_class") != "reduce_side":
        return False
    if metric.get("quote_distance_bucket") not in {"touch", "one_tick_tight", "step_back_gt1"}:
        return False
    spread = _float(metric.get("spread_capture_ticks_mean"))
    return _finite(spread) and spread >= SPREAD_FLOOR_TICKS


def shape_candidate_rows(metrics: list[dict[str, Any]], *, shape: str) -> list[dict[str, Any]]:
    predicate = _shape_a_candidate if shape == "shape_a" else _shape_b_candidate
    out: list[dict[str, Any]] = []
    for row in metrics:
        if not predicate(row):
            continue
        row_out = dict(row)
        row_out["candidate_shape"] = (
            "passive_quality_gate_with_inventory_sizing"
            if shape == "shape_a"
            else "reduce_side_participation_gate_with_spread_capture_floor"
        )
        row_out["policy_design_note"] = "candidate trigger bucket only; no strategy behavior or parameters are implemented"
        out.append(row_out)
    return out


def rejected_bucket_rows(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in metrics:
        fills = int(_float(row.get("fills"), 0.0))
        markout = _float(row.get("side_adjusted_markout_5000ms_mean"))
        spread = _float(row.get("spread_capture_ticks_mean"))
        verdict = str(row.get("verdict", ""))
        reasons: list[str] = []
        if verdict == "reject_quality_negative":
            reasons.append(str(row.get("verdict_reason", "reject_quality_negative")))
        if fills >= MIN_CLEAN_FILLS and _finite(markout) and markout <= NEGATIVE_MARKOUT_TICKS:
            reasons.append("high_fill_but_poor_markout")
        if _finite(spread) and spread >= HIGH_SPREAD_TICKS and fills < MIN_CLEAN_FILLS:
            reasons.append("high_spread_but_too_low_fill")
        if _float(row.get("post_only_risk_rate"), 0.0) > UNSAFE_RATE_MAX:
            reasons.append("unsafe_post_only_context")
        if _float(row.get("reject_throttle_churn_rate"), 0.0) > REJECT_THROTTLE_CHURN_MAX:
            reasons.append("unsafe_reject_throttle_churn_context")
        if not reasons:
            continue
        out = dict(row)
        out["rejection_reasons"] = "|".join(sorted(set(reasons)))
        rows.append(out)
    return rows


def _summary_from_metrics(metrics: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in metrics:
        counts[str(row.get("verdict", "not_decisionable"))] += 1
    fills = sum(int(_float(row.get("fills"), 0.0)) for row in metrics)
    rows = sum(int(_float(row.get("rows"), 0.0)) for row in metrics)
    return {
        "scope": scope,
        "bucket_count": len(metrics),
        "rows": rows,
        "fills": fills,
        "ready_for_policy_design": counts["ready_for_policy_design"],
        "needs_more_clean_fills": counts["needs_more_clean_fills"],
        "reject_quality_negative": counts["reject_quality_negative"],
        "not_decisionable": counts["not_decisionable"],
        "verdict_taxonomy": "|".join(VERDICT_TAXONOMY),
    }


def _caveated_sensitivity_rows(clean_metrics: list[dict[str, Any]], accepted_metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_by_key = {tuple(str(row.get(key, "")) for key in BUCKET_KEYS): row for row in clean_metrics}
    out: list[dict[str, Any]] = []
    for row in accepted_metrics:
        key = tuple(str(row.get(field, "")) for field in BUCKET_KEYS)
        clean = clean_by_key.get(key, {})
        out_row = {field: row.get(field, "") for field in BUCKET_KEYS}
        out_row.update(
            {
                "clean_verdict": clean.get("verdict", "not_decisionable"),
                "accepted_verdict": row.get("verdict", "not_decisionable"),
                "clean_fills": clean.get("fills", 0),
                "accepted_fills": row.get("fills", 0),
                "fill_delta_from_caveated": int(_float(row.get("fills"), 0.0)) - int(_float(clean.get("fills"), 0.0)),
                "clean_markout": clean.get("side_adjusted_markout_5000ms_mean", ""),
                "accepted_markout": row.get("side_adjusted_markout_5000ms_mean", ""),
                "clean_spread": clean.get("spread_capture_ticks_mean", ""),
                "accepted_spread": row.get("spread_capture_ticks_mean", ""),
                "stability": "same_verdict" if clean.get("verdict") == row.get("verdict") else "changed_with_caveated",
            }
        )
        out.append(out_row)
    return out


def _overall_recommendation(
    clean_metrics: list[dict[str, Any]],
    shape_a: list[dict[str, Any]],
    shape_b: list[dict[str, Any]],
) -> dict[str, Any]:
    ready = [row for row in clean_metrics if row.get("verdict") == "ready_for_policy_design"]
    needs = [row for row in clean_metrics if row.get("verdict") == "needs_more_clean_fills"]
    reject = [row for row in clean_metrics if row.get("verdict") == "reject_quality_negative"]
    if shape_a or shape_b:
        verdict = "ready_for_policy_design"
        reason = "at least one clean bucket satisfies a candidate shape gate"
    elif ready:
        verdict = "ready_for_policy_design"
        reason = "clean quality-positive buckets exist, but shape filters need design refinement"
    elif needs:
        verdict = "needs_more_clean_fills"
        reason = "candidate axes exist but clean fill mass is below bucket thresholds"
    elif reject:
        verdict = "reject_quality_negative"
        reason = "available decisionable buckets are quality-negative or unsafe"
    else:
        verdict = "not_decisionable"
        reason = "no bucket is decisionable under the fixed thresholds"
    return {
        "overall_verdict": verdict,
        "overall_reason": reason,
        "shape_a_candidate_count": len(shape_a),
        "shape_b_candidate_count": len(shape_b),
        "ready_bucket_count": len(ready),
        "needs_more_clean_fills_bucket_count": len(needs),
        "reject_quality_negative_bucket_count": len(reject),
        "next_recommended_task": (
            "design_only_policy_contract_for_stage9k_ready_buckets"
            if verdict == "ready_for_policy_design"
            else "collect_or_refine_read_only_evidence_before_policy_design"
        ),
        "boundary": "read-only synthesis only; no strategy behavior, live behavior, replay semantics, parameter search, default-on, guard relaxation, tiny-live, or promotion changed",
    }


def _write_recommendation(path: Path, recommendation: dict[str, Any], sample_notes: list[dict[str, Any]]) -> None:
    lines = [
        "# 0529T002 Fill-Quality Bucket Recommendation",
        "",
        f"- overall_verdict: `{recommendation['overall_verdict']}`",
        f"- overall_reason: {recommendation['overall_reason']}",
        f"- shape_a_candidate_count: `{recommendation['shape_a_candidate_count']}`",
        f"- shape_b_candidate_count: `{recommendation['shape_b_candidate_count']}`",
        f"- ready_bucket_count: `{recommendation['ready_bucket_count']}`",
        f"- needs_more_clean_fills_bucket_count: `{recommendation['needs_more_clean_fills_bucket_count']}`",
        f"- reject_quality_negative_bucket_count: `{recommendation['reject_quality_negative_bucket_count']}`",
        f"- next_recommended_task: `{recommendation['next_recommended_task']}`",
        "",
        "Boundary: read-only/default-off synthesis only. No live run, no replay run, no strategy behavior change, no parameter search, no guard relaxation, no default-on behavior, no tiny-live, and no promotion claim.",
        "",
        "Proxy limitations: metrics are observed submitted-order labels from existing artifacts. The runner does not infer counterfactual fills, exact queue position, or hidden queue behavior.",
        "",
        "Samples:",
    ]
    for note in sample_notes:
        lines.append(f"- {note['sample_id']}: status={note['status']}, caveated={note.get('is_caveated', False)}, rows={note.get('rows', 0)}, notes={','.join(note.get('notes', [])) or 'none'}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fill_quality_bucket_synthesis(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    include_caveated: bool = True,
    caveated_ids: set[str] | None = None,
) -> dict[str, Any]:
    caveated_ids = set(caveated_ids or DEFAULT_CAVEATED_SAMPLE_IDS)
    samples = [discover_sample(run_dir, caveated_ids) for run_dir in run_dirs]
    rows: list[dict[str, Any]] = []
    sample_notes: list[dict[str, Any]] = []
    for sample in samples:
        sample_rows, note = _load_fill_quality_rows(sample)
        rows.extend(sample_rows)
        sample_notes.append(note)

    clean_rows = [row for row in rows if not _bool(row.get("is_caveated"))]
    accepted_rows = rows if include_caveated else clean_rows
    caveated_rows = [row for row in rows if _bool(row.get("is_caveated"))]
    clean_metrics = build_bucket_metrics(clean_rows, scope="clean_only")
    accepted_metrics = build_bucket_metrics(accepted_rows, scope="accepted_set")
    trigger_clean_metrics = build_trigger_metrics(clean_rows, scope="clean_only")
    trigger_accepted_metrics = build_trigger_metrics(accepted_rows, scope="accepted_set")
    trigger_caveated_metrics = build_trigger_metrics(caveated_rows, scope="caveated_only")
    shape_a = shape_candidate_rows(trigger_clean_metrics, shape="shape_a")
    shape_b = shape_candidate_rows(trigger_clean_metrics, shape="shape_b")
    rejected = rejected_bucket_rows(trigger_clean_metrics)
    clean_summary = [_summary_from_metrics(trigger_clean_metrics, "clean_only")]
    caveated_summary = [_summary_from_metrics(trigger_caveated_metrics, "caveated_only")]
    recommendation = _overall_recommendation(trigger_clean_metrics, shape_a, shape_b)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "bucket_fill_quality_metrics.csv", trigger_clean_metrics + trigger_accepted_metrics + clean_metrics + accepted_metrics)
    _write_csv(output_dir / "clean_only_stability_summary.csv", clean_summary)
    _write_csv(output_dir / "caveated_sensitivity_summary.csv", _caveated_sensitivity_rows(trigger_clean_metrics, trigger_accepted_metrics) + caveated_summary)
    _write_csv(output_dir / "shape_a_passive_quality_gate_candidates.csv", shape_a, fieldnames=BUCKET_KEYS + METRIC_FIELDS + ["candidate_shape", "policy_design_note"])
    _write_csv(output_dir / "shape_b_reduce_side_participation_candidates.csv", shape_b, fieldnames=BUCKET_KEYS + METRIC_FIELDS + ["candidate_shape", "policy_design_note"])
    _write_csv(output_dir / "rejected_bucket_reasons.csv", rejected)
    _write_recommendation(output_dir / "fill_quality_bucket_recommendation.md", recommendation, sample_notes)
    manifest = {
        "task_id": TASK_ID,
        "runner_mode": RUNNER_MODE,
        "generated_at": _generated_at(),
        "runner_path": "examples/binance_tick_mm/fill_quality_bucket_synthesis.py",
        "output_dir": str(output_dir),
        "include_caveated": include_caveated,
        "verdict_taxonomy": list(VERDICT_TAXONOMY),
        "bucket_keys": BUCKET_KEYS,
        "thresholds": {
            "min_clean_fills": MIN_CLEAN_FILLS,
            "min_clean_samples": MIN_CLEAN_SAMPLES,
            "min_shape_fills": MIN_SHAPE_FILLS,
            "markout_floor_ticks": MARKOUT_FLOOR_TICKS,
            "spread_floor_ticks": SPREAD_FLOOR_TICKS,
            "fill_after_cancel_max": FILL_AFTER_CANCEL_MAX,
            "reject_throttle_churn_max": REJECT_THROTTLE_CHURN_MAX,
        },
        "decision_visible_bucket_inputs": [
            "inventory_bucket",
            "side_class",
            "quote_distance_bucket",
            "fair_or_reservation_edge_bucket",
            "latency_stale_bucket",
            "post_only_safety_bucket",
            "reject_throttle_churn_bucket",
            *REQUIRED_T006_FIELDS,
        ],
        "outcome_sensitivity_buckets": ["fill_after_cancel_bucket"],
        "forbidden_inputs_not_used_as_triggers": [
            "future_fill",
            "future_markout",
            "future_spread_capture",
            "same_sample_pnl_feedback",
            "exact_queue_position",
            "hidden_queue_assumptions",
        ],
        "boundary": {
            "read_only": True,
            "default_off": True,
            "live_run": False,
            "replay_run": False,
            "strategy_change": False,
            "parameter_search": False,
            "promotion": False,
        },
        "sample_notes": sample_notes,
        "recommendation": recommendation,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=None, help="local_live_analysis/<run_id> to include. Repeatable.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for Stage 9K output artifacts.")
    parser.add_argument("--include-caveated", action=argparse.BooleanOptionalAction, default=True, help="Include caveated samples in accepted-set sensitivity. Clean-only remains primary.")
    parser.add_argument("--caveated-sample-id", action="append", default=sorted(DEFAULT_CAVEATED_SAMPLE_IDS), help="Sample id to treat as caveated. Repeatable.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_dirs = args.run_dir if args.run_dir else DEFAULT_RUN_DIRS
    manifest = run_fill_quality_bucket_synthesis(
        run_dirs=run_dirs,
        output_dir=args.output_dir,
        include_caveated=args.include_caveated,
        caveated_ids=set(args.caveated_sample_id),
    )
    recommendation = manifest["recommendation"]
    print(f"wrote {args.output_dir}")
    print(f"overall_verdict={recommendation['overall_verdict']}")
    print(f"shape_a_candidate_count={recommendation['shape_a_candidate_count']}")
    print(f"shape_b_candidate_count={recommendation['shape_b_candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
