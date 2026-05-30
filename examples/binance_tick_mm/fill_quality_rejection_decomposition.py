#!/usr/bin/env python3
"""Read-only Stage 9L fill-quality rejection decomposition.

This runner consumes the accepted Stage 9K manifest, reconstructs the same
observed submit rows from existing artifacts, and evaluates rejection reasons,
churn-gate sensitivity, and decision-visible bucket coarsening. It does not
change strategy behavior, run live, run replay, search parameters, relax guards,
or make promotion claims.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from fill_quality_bucket_synthesis import (
    FILL_AFTER_CANCEL_MAX,
    HIGH_SPREAD_TICKS,
    MARKOUT_FLOOR_TICKS,
    METRIC_FIELDS,
    MIN_CLEAN_FILLS,
    MIN_CLEAN_SAMPLES,
    MIN_SAMPLE_FILL_COVERAGE,
    MIN_SHAPE_FILLS,
    NEGATIVE_MARKOUT_TICKS,
    SPREAD_FLOOR_TICKS,
    SampleArtifacts,
    _finite,
    _float,
    _generated_at,
    _load_fill_quality_rows,
    _mean,
    _rate,
    _safe_num,
    _write_csv,
    _write_json,
    discover_sample,
)
from quote_adjustment_replay import _bool, _int, _read_csv


TASK_ID = "0529T005"
RUNNER_MODE = "stage9l_fill_quality_rejection_decomposition"
DEFAULT_STAGE9K_INPUT_DIR = Path("local_live_analysis/stage9k_fill_quality_bucket_synthesis_0529T002")
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/stage9l_fill_quality_rejection_decomposition_0529T005")

TRIGGER_KEYS = [
    "inventory_bucket",
    "side_class",
    "quote_distance_bucket",
    "fair_or_reservation_edge_bucket",
    "latency_stale_bucket",
    "post_only_safety_bucket",
    "reject_throttle_churn_bucket",
]

COARSENED_METRIC_FIELDS = [
    "coarsening_variant",
    *TRIGGER_KEYS,
    "bucket_level",
    "scope",
    "sample_count",
    "fill_sample_count",
    "rows",
    "fills",
    "fill_rate",
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
    "warning_churn_rows",
    "warning_churn_rate",
    "hard_reject_drop_rows",
    "hard_reject_drop_rate",
    "verdict",
    "verdict_reason",
]

CHURN_VARIANTS = (
    "stage9k_original_hard_gate",
    "recent_reject_throttle_warning",
    "fast_cancel_cancel_readd_warning",
    "non_true_reject_churn_warning",
)

COARSENING_VARIANTS = (
    "identity",
    "edge_latency_coarsened",
    "quote_distance_family_coarsened",
    "churn_warning_coarsened",
)

FINAL_CLASSIFICATIONS = (
    "ready_for_policy_design_after_coarsening",
    "needs_targeted_clean_fills",
    "reject_current_fill_quality_direction",
    "not_decisionable",
)


@dataclass(frozen=True)
class Stage9KContext:
    input_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]
    run_dirs: list[Path]
    caveated_ids: set[str]
    thresholds: dict[str, Any]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_stage9k_context(input_dir: Path) -> Stage9KContext:
    manifest_path = input_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    sample_notes = manifest.get("sample_notes", [])
    run_dirs = [Path(note["stage5_labels"]).parents[1] for note in sample_notes if note.get("stage5_labels")]
    caveated_ids = {str(note["sample_id"]) for note in sample_notes if _bool(note.get("is_caveated"))}
    return Stage9KContext(
        input_dir=input_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        run_dirs=run_dirs,
        caveated_ids=caveated_ids,
        thresholds=dict(manifest.get("thresholds", {})),
    )


def load_stage9l_rows(context: Stage9KContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sample_notes: list[dict[str, Any]] = []
    for run_dir in context.run_dirs:
        sample = discover_sample(run_dir, context.caveated_ids)
        sample_rows, note = _load_fill_quality_rows(sample)
        rows.extend(sample_rows)
        sample_notes.append(note)
    return rows, sample_notes


def _sum_bool(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for row in rows if _bool(row.get(key)))


def _num(row: dict[str, Any], key: str) -> float:
    return _float(row.get(key))


def _count_filled(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _bool(row.get("filled"))]


def _metric_row(
    group: tuple[str, ...],
    rows: list[dict[str, Any]],
    keys: list[str],
    *,
    scope: str,
    coarsening_variant: str,
    churn_variant: str,
) -> dict[str, Any]:
    filled = _count_filled(rows)
    sample_ids = {str(row.get("sample_id", "")) for row in rows}
    fill_sample_ids = {str(row.get("sample_id", "")) for row in filled}
    post_only_risk_rows = _sum_bool(rows, "post_only_risk")
    reject_throttle_churn_rows = _sum_bool(rows, "reject_throttle_churn_risk")
    warning_churn_rows = sum(1 for row in rows if _churn_policy(row, churn_variant) == "warning")
    hard_reject_drop_rows = sum(1 for row in rows if _churn_policy(row, churn_variant) == "hard")
    out: dict[str, Any] = {"coarsening_variant": coarsening_variant}
    out.update({key: value for key, value in zip(keys, group)})
    fill_count = len(filled)
    out.update(
        {
            "bucket_level": "coarsened_decision_visible_trigger",
            "scope": scope,
            "sample_count": len(sample_ids),
            "fill_sample_count": len(fill_sample_ids),
            "rows": len(rows),
            "fills": fill_count,
            "fill_rate": _safe_num(_rate(fill_count, len(rows))),
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
            "warning_churn_rows": warning_churn_rows,
            "warning_churn_rate": _safe_num(_rate(warning_churn_rows, len(rows))),
            "hard_reject_drop_rows": hard_reject_drop_rows,
            "hard_reject_drop_rate": _safe_num(_rate(hard_reject_drop_rows, len(rows))),
        }
    )
    verdict, reason = assign_stage9l_verdict(out, churn_variant=churn_variant)
    out["verdict"] = verdict
    out["verdict_reason"] = reason
    return out


def _churn_policy(row: dict[str, Any], churn_variant: str) -> str:
    bucket = str(row.get("reject_throttle_churn_bucket", "churn_normal"))
    if bucket == "churn_normal":
        return "normal"
    if churn_variant == "stage9k_original_hard_gate":
        return "hard"
    if bucket == "reject_throttle_or_drop":
        return "hard"
    if churn_variant == "recent_reject_throttle_warning" and bucket == "recent_reject_or_throttle":
        return "warning"
    if churn_variant == "fast_cancel_cancel_readd_warning" and bucket in {"fast_cancel_churn", "cancel_readd_pressure"}:
        return "warning"
    if churn_variant == "non_true_reject_churn_warning":
        return "warning"
    return "hard"


def assign_stage9l_verdict(metric: dict[str, Any], *, churn_variant: str) -> tuple[str, str]:
    rows = int(_float(metric.get("rows"), 0.0))
    fills = int(_float(metric.get("fills"), 0.0))
    sample_count = int(_float(metric.get("sample_count"), 0.0))
    fill_sample_count = int(_float(metric.get("fill_sample_count"), 0.0))
    markout = _float(metric.get("side_adjusted_markout_5000ms_mean"))
    spread = _float(metric.get("spread_capture_ticks_mean"))
    fac = _float(metric.get("fill_after_cancel_rate"))
    post_only = _float(metric.get("post_only_risk_rate"), 0.0)
    hard_churn = _float(metric.get("hard_reject_drop_rate"), 0.0)

    if rows == 0:
        return "not_decisionable", "bucket has no observed submit rows"
    if post_only > 0.0:
        return "reject_quality_negative", "bucket overlaps post-only risk"
    if hard_churn > 0.0:
        return "reject_quality_negative", f"bucket overlaps hard reject/throttle/churn risk under {churn_variant}"
    if fills < MIN_CLEAN_FILLS or sample_count < MIN_CLEAN_SAMPLES or fill_sample_count < MIN_SAMPLE_FILL_COVERAGE:
        return "needs_more_clean_fills", "clean fill mass or sample coverage is below threshold"
    if _finite(markout) and markout <= NEGATIVE_MARKOUT_TICKS:
        return "reject_quality_negative", "5s side-adjusted markout is materially adverse"
    if _finite(spread) and spread < 0.0:
        return "reject_quality_negative", "spread capture is negative"
    if _finite(fac) and fac > FILL_AFTER_CANCEL_MAX:
        return "reject_quality_negative", "fill-after-cancel sensitivity is elevated"
    if _finite(markout) and markout >= MARKOUT_FLOOR_TICKS and _finite(spread) and spread >= SPREAD_FLOOR_TICKS:
        return "ready_for_policy_design", "coarsened bucket has enough fills with acceptable markout, spread, and safety"
    return "not_decisionable", "quality metrics are mixed or missing despite enough fills"


def rejection_components_for_metric(row: dict[str, Any]) -> list[str]:
    components: list[str] = []
    fills = int(_float(row.get("fills"), 0.0))
    markout = _float(row.get("side_adjusted_markout_5000ms_mean"))
    spread = _float(row.get("spread_capture_ticks_mean"))
    fac = _float(row.get("fill_after_cancel_rate"))
    churn_bucket = str(row.get("reject_throttle_churn_bucket", ""))
    post_bucket = str(row.get("post_only_safety_bucket", ""))
    latency = str(row.get("latency_stale_bucket", ""))

    if fills >= MIN_CLEAN_FILLS and _finite(markout) and markout <= NEGATIVE_MARKOUT_TICKS:
        components.append("adverse_5s_markout")
    if _finite(spread) and spread < 0.0:
        components.append("negative_spread_capture")
    elif _finite(spread) and spread < SPREAD_FLOOR_TICKS:
        components.append("weak_spread_capture")
    if _finite(fac) and fac > FILL_AFTER_CANCEL_MAX:
        components.append("high_fill_after_cancel_sensitivity")
    if churn_bucket == "reject_throttle_or_drop":
        components.append("reject_throttle_drop")
    if churn_bucket == "fast_cancel_churn":
        components.append("fast_cancel_churn")
    if churn_bucket == "cancel_readd_pressure":
        components.append("cancel_readd_pressure")
    if churn_bucket == "recent_reject_or_throttle":
        components.append("recent_reject_or_throttle")
    if post_bucket in {"post_only_risk", "post_only_risk_after_recheck"}:
        components.append("post_only_risk")
    if post_bucket in {"missing_anchor", "stale_anchor", "guarded_depth_fallback", "clamped", "suppressed"}:
        components.append(f"post_only_safety_context_{post_bucket}")
    if latency != "fresh_low_latency":
        components.append(f"latency_stale_context_{latency}")
    if not components:
        components.append("other_or_threshold_ordering")
    return components


def build_rejection_reason_decomposition(stage9k_input_dir: Path) -> list[dict[str, Any]]:
    metrics_path = stage9k_input_dir / "bucket_fill_quality_metrics.csv"
    rows = [
        row
        for row in _read_csv(metrics_path)
        if row.get("scope") == "clean_only"
        and row.get("bucket_level") == "decision_visible_trigger"
        and row.get("verdict") == "reject_quality_negative"
    ]
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for component in rejection_components_for_metric(row):
            buckets[component].append(row)

    out: list[dict[str, Any]] = []
    for component, component_rows in sorted(buckets.items()):
        out.append(
            {
                "rejection_component": component,
                "bucket_count": len(component_rows),
                "rows": sum(int(_float(row.get("rows"), 0.0)) for row in component_rows),
                "fills": sum(int(_float(row.get("fills"), 0.0)) for row in component_rows),
                "mean_markout_5s_ticks": _safe_num(_mean(_float(row.get("side_adjusted_markout_5000ms_mean")) for row in component_rows)),
                "mean_spread_capture_ticks": _safe_num(_mean(_float(row.get("spread_capture_ticks_mean")) for row in component_rows)),
                "mean_fill_after_cancel_rate": _safe_num(_mean(_float(row.get("fill_after_cancel_rate")) for row in component_rows)),
                "source": "stage9k_clean_decision_visible_trigger_metrics",
            }
        )
    return out


def _coarsen_value(key: str, value: str, variant: str) -> str:
    if variant == "identity":
        return value
    if key == "fair_or_reservation_edge_bucket" and variant in {"edge_latency_coarsened", "churn_warning_coarsened"}:
        if value in {"edge_strong_favorable", "edge_weak_or_neutral"}:
            return "edge_non_adverse"
    if key == "latency_stale_bucket" and variant in {"edge_latency_coarsened", "churn_warning_coarsened"}:
        if value in {"fresh_low_latency", "stale_latency_medium"}:
            return "market_view_usable"
        return value
    if key == "quote_distance_bucket" and variant == "quote_distance_family_coarsened":
        if value in {"touch", "one_tick_tight"}:
            return "near_touch"
        if value == "step_back_gt1":
            return "passive_step_back"
    if key == "reject_throttle_churn_bucket" and variant == "churn_warning_coarsened":
        if value == "reject_throttle_or_drop":
            return "hard_reject_drop"
        if value == "churn_normal":
            return "churn_normal"
        return "warning_churn_context"
    return value


def _group_key(row: dict[str, Any], variant: str) -> tuple[str, ...]:
    return tuple(_coarsen_value(key, str(row.get(key, "")), variant) for key in TRIGGER_KEYS)


def build_coarsened_metrics(rows: list[dict[str, Any]], *, scope: str, coarsening_variant: str, churn_variant: str) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_group_key(row, coarsening_variant)].append(row)
    return [
        _metric_row(group, group_rows, TRIGGER_KEYS, scope=scope, coarsening_variant=coarsening_variant, churn_variant=churn_variant)
        for group, group_rows in sorted(groups.items())
    ]


def _shape_a_candidate(metric: dict[str, Any]) -> bool:
    if metric.get("verdict") != "ready_for_policy_design":
        return False
    if int(_float(metric.get("fills"), 0.0)) < MIN_SHAPE_FILLS:
        return False
    if metric.get("quote_distance_bucket") not in {"one_tick_tight", "step_back_gt1", "near_touch", "passive_step_back"}:
        return False
    if metric.get("latency_stale_bucket") not in {"fresh_low_latency", "stale_latency_medium", "market_view_usable"}:
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
    if metric.get("quote_distance_bucket") not in {"touch", "one_tick_tight", "step_back_gt1", "near_touch", "passive_step_back"}:
        return False
    spread = _float(metric.get("spread_capture_ticks_mean"))
    return _finite(spread) and spread >= SPREAD_FLOOR_TICKS


def build_shape_candidates(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in metrics:
        for shape, predicate, label in (
            ("shape_a", _shape_a_candidate, "passive_quality_gate_with_inventory_sizing"),
            ("shape_b", _shape_b_candidate, "reduce_side_participation_gate_with_spread_capture_floor"),
        ):
            if not predicate(row):
                continue
            candidate = dict(row)
            candidate["candidate_shape"] = shape
            candidate["policy_design_label"] = label
            candidate["policy_design_note"] = "coarsened trigger bucket only; no strategy behavior or parameters are implemented"
            out.append(candidate)
    return out


def build_churn_gate_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    clean_rows = [row for row in rows if not _bool(row.get("is_caveated"))]
    for variant in CHURN_VARIANTS:
        metrics = build_coarsened_metrics(
            clean_rows,
            scope="clean_only",
            coarsening_variant="identity",
            churn_variant=variant,
        )
        counts: dict[str, int] = defaultdict(int)
        for row in metrics:
            counts[str(row.get("verdict"))] += 1
        candidates = build_shape_candidates(metrics)
        out.append(
            {
                "churn_gate_variant": variant,
                "coarsening_variant": "identity",
                "bucket_count": len(metrics),
                "rows": sum(int(_float(row.get("rows"), 0.0)) for row in metrics),
                "fills": sum(int(_float(row.get("fills"), 0.0)) for row in metrics),
                "ready_for_policy_design": counts["ready_for_policy_design"],
                "needs_more_clean_fills": counts["needs_more_clean_fills"],
                "reject_quality_negative": counts["reject_quality_negative"],
                "not_decisionable": counts["not_decisionable"],
                "shape_candidate_count": len(candidates),
                "churn_warning_alone_creates_ready_candidates": int(variant != "stage9k_original_hard_gate" and len(candidates) > 0),
                "hard_gate_policy": "true reject/drop and post-only risk stay hard outside original hard-gate baseline",
            }
        )
    return out


def build_sample_gap_by_regime(metrics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for row in metrics:
        if row.get("verdict") != "needs_more_clean_fills":
            continue
        fills = int(_float(row.get("fills"), 0.0))
        sample_count = int(_float(row.get("sample_count"), 0.0))
        fill_sample_count = int(_float(row.get("fill_sample_count"), 0.0))
        gaps.append(
            {
                "coarsening_variant": row.get("coarsening_variant", ""),
                "inventory_bucket": row.get("inventory_bucket", ""),
                "side_class": row.get("side_class", ""),
                "quote_distance_bucket": row.get("quote_distance_bucket", ""),
                "fair_or_reservation_edge_bucket": row.get("fair_or_reservation_edge_bucket", ""),
                "latency_stale_bucket": row.get("latency_stale_bucket", ""),
                "post_only_safety_bucket": row.get("post_only_safety_bucket", ""),
                "reject_throttle_churn_bucket": row.get("reject_throttle_churn_bucket", ""),
                "rows": row.get("rows", 0),
                "fills": fills,
                "fills_needed_for_min_clean_fills": max(0, MIN_CLEAN_FILLS - fills),
                "samples_needed_for_min_clean_samples": max(0, MIN_CLEAN_SAMPLES - sample_count),
                "fill_samples_needed_for_min_coverage": max(0, MIN_SAMPLE_FILL_COVERAGE - fill_sample_count),
                "sample_count": sample_count,
                "fill_sample_count": fill_sample_count,
                "recommendation": "target clean observed fills in this decision-visible regime",
            }
        )
    return sorted(
        gaps,
        key=lambda row: (
            int(row["fills_needed_for_min_clean_fills"]),
            int(row["fill_samples_needed_for_min_coverage"]),
            -int(_float(row["fills"], 0.0)),
        ),
    )


def final_recommendation(metrics: list[dict[str, Any]], candidates: list[dict[str, Any]], gaps: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in metrics:
        counts[str(row.get("verdict"))] += 1
    if candidates:
        classification = "ready_for_policy_design_after_coarsening"
        next_task = "design_only_policy_contract_for_stage9l_coarsened_buckets"
        reason = "at least one coarsened clean bucket satisfies Shape A or Shape B gates"
    elif gaps:
        classification = "needs_targeted_clean_fills"
        next_task = "targeted_clean_fill_collection_or_read_only_evidence_refinement"
        reason = "coarsening leaves potentially useful regimes under-filled before a policy contract can be justified"
    elif counts["reject_quality_negative"] > 0 and counts["ready_for_policy_design"] == 0:
        classification = "reject_current_fill_quality_direction"
        next_task = "reject_current_fill_quality_direction_or_reframe_hypothesis"
        reason = "coarsened decisionable buckets remain quality-negative after churn sensitivity"
    else:
        classification = "not_decisionable"
        next_task = "controller_review_of_stage9l_evidence"
        reason = "coarsened evidence is mixed or missing without a clear collection target"
    return {
        "final_classification": classification,
        "classification_taxonomy": list(FINAL_CLASSIFICATIONS),
        "next_recommended_task": next_task,
        "reason": reason,
        "coarsened_ready_bucket_count": counts["ready_for_policy_design"],
        "coarsened_needs_more_clean_fills_bucket_count": counts["needs_more_clean_fills"],
        "coarsened_reject_quality_negative_bucket_count": counts["reject_quality_negative"],
        "coarsened_not_decisionable_bucket_count": counts["not_decisionable"],
        "shape_candidate_count": len(candidates),
        "boundary": "read-only Stage 9L evidence refinement only; no strategy behavior, live behavior, replay semantics, parameter search, default-on, guard relaxation, tiny-live, or promotion changed",
    }


def write_recommendation(path: Path, recommendation: dict[str, Any], churn_rows: list[dict[str, Any]], candidates: list[dict[str, Any]]) -> None:
    lines = [
        "# 0529T005 Stage 9L Recommendation",
        "",
        f"- final_classification: `{recommendation['final_classification']}`",
        f"- reason: {recommendation['reason']}",
        f"- next_recommended_task: `{recommendation['next_recommended_task']}`",
        f"- shape_candidate_count: `{recommendation['shape_candidate_count']}`",
        f"- coarsened_ready_bucket_count: `{recommendation['coarsened_ready_bucket_count']}`",
        f"- coarsened_needs_more_clean_fills_bucket_count: `{recommendation['coarsened_needs_more_clean_fills_bucket_count']}`",
        f"- coarsened_reject_quality_negative_bucket_count: `{recommendation['coarsened_reject_quality_negative_bucket_count']}`",
        f"- coarsened_not_decisionable_bucket_count: `{recommendation['coarsened_not_decisionable_bucket_count']}`",
        "",
        "Churn sensitivity:",
    ]
    for row in churn_rows:
        lines.append(
            f"- {row['churn_gate_variant']}: ready={row['ready_for_policy_design']}, "
            f"needs_more={row['needs_more_clean_fills']}, reject={row['reject_quality_negative']}, "
            f"shape_candidates={row['shape_candidate_count']}"
        )
    lines.extend(
        [
            "",
            f"Any Shape A / Shape B candidate after coarsening: `{'yes' if candidates else 'no'}`.",
            "",
            "Boundary: read-only/default-off analysis only. No strategy behavior, live run, replay run, fill/cancel replay semantic change, parameter search, guard relaxation, default-on behavior, tiny-live, or promotion claim.",
            "",
            "Proxy limitations: this uses observed submitted-order labels from existing artifacts. It does not infer counterfactual fills, exact queue position, hidden queue behavior, future markout triggers, or same-sample PnL feedback.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fill_quality_rejection_decomposition(stage9k_input_dir: Path, output_dir: Path) -> dict[str, Any]:
    context = load_stage9k_context(stage9k_input_dir)
    rows, sample_notes = load_stage9l_rows(context)
    clean_rows = [row for row in rows if not _bool(row.get("is_caveated"))]
    accepted_rows = rows

    rejection_rows = build_rejection_reason_decomposition(stage9k_input_dir)
    churn_rows = build_churn_gate_sensitivity(rows)

    coarsened_metrics: list[dict[str, Any]] = []
    for variant in COARSENING_VARIANTS:
        churn_variant = "non_true_reject_churn_warning" if variant == "churn_warning_coarsened" else "stage9k_original_hard_gate"
        coarsened_metrics.extend(
            build_coarsened_metrics(
                clean_rows,
                scope="clean_only",
                coarsening_variant=variant,
                churn_variant=churn_variant,
            )
        )
        coarsened_metrics.extend(
            build_coarsened_metrics(
                accepted_rows,
                scope="accepted_set",
                coarsening_variant=variant,
                churn_variant=churn_variant,
            )
        )

    primary_metrics = [row for row in coarsened_metrics if row.get("scope") == "clean_only"]
    candidates = build_shape_candidates(primary_metrics)
    gaps = build_sample_gap_by_regime(primary_metrics)
    recommendation = final_recommendation(primary_metrics, candidates, gaps)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "rejection_reason_decomposition.csv", rejection_rows)
    _write_csv(output_dir / "churn_gate_sensitivity.csv", churn_rows)
    _write_csv(output_dir / "coarsened_trigger_bucket_metrics.csv", coarsened_metrics, fieldnames=COARSENED_METRIC_FIELDS)
    _write_csv(
        output_dir / "coarsened_shape_candidates.csv",
        candidates,
        fieldnames=COARSENED_METRIC_FIELDS + ["candidate_shape", "policy_design_label", "policy_design_note"],
    )
    _write_csv(output_dir / "sample_gap_by_regime.csv", gaps)
    write_recommendation(output_dir / "stage9l_recommendation.md", recommendation, churn_rows, candidates)

    manifest = {
        "task_id": TASK_ID,
        "runner_mode": RUNNER_MODE,
        "generated_at": _generated_at(),
        "runner_path": "examples/binance_tick_mm/fill_quality_rejection_decomposition.py",
        "stage9k_input_dir": str(stage9k_input_dir),
        "stage9k_manifest_path": str(context.manifest_path),
        "output_dir": str(output_dir),
        "sample_ids": [note.get("sample_id") for note in sample_notes],
        "caveated_sample_ids": sorted(context.caveated_ids),
        "thresholds": context.thresholds,
        "coarsening_variants": list(COARSENING_VARIANTS),
        "churn_gate_variants": list(CHURN_VARIANTS),
        "decision_visible_trigger_keys": list(TRIGGER_KEYS),
        "row_level_reaggregation": True,
        "coarsened_sample_coverage_source": "row_level_sample_id",
        "forbidden_inputs_not_used_as_triggers": [
            "future_fill",
            "future_markout",
            "future_spread_capture",
            "fill_after_cancel_bucket",
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
            "guard_relaxation": False,
            "promotion": False,
        },
        "sample_notes": sample_notes,
        "recommendation": recommendation,
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_STAGE9K_INPUT_DIR, help="Stage 9K output directory to consume.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for Stage 9L output artifacts.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = run_fill_quality_rejection_decomposition(args.input_dir, args.output_dir)
    recommendation = manifest["recommendation"]
    print(f"wrote {args.output_dir}")
    print(f"final_classification={recommendation['final_classification']}")
    print(f"shape_candidate_count={recommendation['shape_candidate_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
