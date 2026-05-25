#!/usr/bin/env python3
"""Step 9D fine-bucket refinement for keep-for-research quote candidates.

This runner is read-only. It reuses the Step 9B candidate triggers and existing
audit / Stage 5 / Stage 5C artifacts, then compares each candidate with the
sample-local baseline inside finer scenario buckets.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from quote_adjustment_replay import (
    TINY_LIVE_DESIGN_THRESHOLDS,
    _artifact_paths,
    _bool,
    _float,
    _generated_at,
    _int,
    _mean,
    _order_id,
    _order_seq,
    _read_csv,
    _read_csv_with_header,
    _read_json,
    _safe_num,
    _seq,
    _stage5c_by_seq,
    _write_csv,
    _write_json,
    build_candidate_runtimes,
    candidate_definitions,
)


TASK_ID = "0525T001"
RUNNER_MODE = "step9d_fine_bucket_refinement"
KEEP_CANDIDATE_IDS = (
    "min_move_quote_age_churn_guard",
    "inventory_reservation_shift_band",
    "size_reduction_or_add_side_suppression_pressure",
)
BASELINE_ID = "baseline_control"
DEFAULT_OUTPUT_DIR = "local_live_analysis/stage9d_candidate_bucket_refinement_0525T001"
DEFAULT_CAVEATED_SAMPLE_IDS = {"5-19-night-active-30min-a"}
MIN_BUCKET_FILLS = 30
MIN_BUCKET_FILL_SAMPLES = 2
PARAMETER_SEED_BUCKET_FAMILIES = {
    "volatility_markout_dispersion",
    "spread_quote_distance",
    "latency_stale_age",
    "inventory_state",
    "api_churn",
    "post_only_safety",
}


@dataclass(frozen=True)
class SampleContext:
    sample_id: str
    run_dir: Path
    market_view_quality: str
    decisions: list[dict[str, str]]
    decisions_by_seq: dict[int, dict[str, str]]
    labels: list[dict[str, str]]
    labels_by_seq: dict[int, list[dict[str, str]]]
    markouts_by_order_id: dict[str, list[dict[str, str]]]
    safety_by_seq: dict[int, dict[str, str]]
    candidate_seqs: dict[str, set[int]]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _nanmedian(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return median(finite) if finite else math.nan


def _nanmin(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return min(finite) if finite else math.nan


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else math.nan


def _decision_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in audit_rows if row.get("event_type") == "decision"]


def _sample_quality(run_dir: Path) -> str:
    metrics = run_dir / "t009_fixed_sidecar" / "metrics.json"
    joined = run_dir / "t009_fixed_sidecar" / "joined_decisions.metrics.json"
    if not metrics.exists() or not joined.exists():
        return "unknown_market_view"
    sidecar = _read_json(metrics)
    join = _read_json(joined)
    if (
        _bool(sidecar.get("first_valid_update_aligned"))
        and int(_float(sidecar.get("depth_pu_mismatch_count"), 0.0)) == 0
        and int(_float(join.get("future_join_count"), 0.0)) == 0
        and int(_float(join.get("gap_crossed_join_count"), 0.0)) == 0
    ):
        return "strict_clean_market_view"
    return "caveated_market_view"


def _load_context(run_dir: Path) -> SampleContext:
    paths = _artifact_paths(run_dir)
    for name, path in paths.items():
        if name == "stage6_summary":
            continue
        if not path.exists():
            raise FileNotFoundError(f"required input {name} not found: {path}")

    audit_rows, _ = _read_csv_with_header(paths["audit_csv"])
    decisions = _decision_rows(audit_rows)
    decisions_by_seq = {seq: row for row in decisions if (seq := _seq(row, "strategy_seq")) is not None}
    labels = _read_csv(paths["execution_labels"])
    safety_by_seq = _stage5c_by_seq(_read_csv(paths["stage5c_safety"]))
    markout_rows = _read_csv(paths["fill_markout"])

    labels_by_seq: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in labels:
        seq = _order_seq(row)
        if seq is not None:
            labels_by_seq[seq].append(row)

    markouts_by_order_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in markout_rows:
        order_id = _order_id(row)
        if order_id:
            markouts_by_order_id[order_id].append(row)

    definitions = [candidate for candidate in candidate_definitions() if candidate.candidate_id in (BASELINE_ID, *KEEP_CANDIDATE_IDS)]
    runtimes = build_candidate_runtimes(decisions=decisions, safety_by_seq=safety_by_seq, candidates=definitions)
    candidate_seqs = {runtime.definition.candidate_id: set(runtime.decision_seqs) for runtime in runtimes}

    return SampleContext(
        sample_id=run_dir.name,
        run_dir=run_dir,
        market_view_quality=_sample_quality(run_dir),
        decisions=decisions,
        decisions_by_seq=decisions_by_seq,
        labels=labels,
        labels_by_seq=labels_by_seq,
        markouts_by_order_id=markouts_by_order_id,
        safety_by_seq=safety_by_seq,
        candidate_seqs=candidate_seqs,
    )


def _quantile_thresholds(values: Iterable[float]) -> tuple[float, float] | None:
    finite = sorted(float(value) for value in values if _finite(value))
    if len(finite) < 3:
        return None
    low_idx = max(0, min(len(finite) - 1, int(len(finite) / 3)))
    high_idx = max(0, min(len(finite) - 1, int(len(finite) * 2 / 3)))
    return finite[low_idx], finite[high_idx]


def _tercile(value: float, thresholds: tuple[float, float] | None, prefix: str) -> str:
    if not _finite(value) or thresholds is None:
        return f"{prefix}_unknown"
    low, high = thresholds
    if value <= low:
        return f"{prefix}_low"
    if value <= high:
        return f"{prefix}_medium"
    return f"{prefix}_high"


def _spread_bucket(row: dict[str, str]) -> str:
    best_bid = _float(row.get("best_bid"))
    best_ask = _float(row.get("best_ask"))
    if not (_finite(best_bid) and _finite(best_ask)):
        return "spread_unknown"
    spread_ticks = (best_ask - best_bid) / 0.1
    if spread_ticks <= 1.5:
        return "one_tick_tight"
    if spread_ticks <= 3.5:
        return "two_to_three_ticks"
    return "wide_spread"


def _latency_bucket(row: dict[str, str]) -> str:
    latency = _float(row.get("latency_signal_ms"), 0.0)
    stale = _float(row.get("book_view_stale_ms"), 0.0)
    quote_age = _float(row.get("quote_age_ms"), 0.0)
    join_age = _float(row.get("join_age_ms"), 0.0)
    anchor_age = _float(row.get("anchor_age_ms"), 0.0)
    max_age = max(value for value in (stale, quote_age, join_age, anchor_age) if _finite(value))
    if latency >= 5.0 or max_age >= 50.0:
        return "stale_latency_high"
    if latency >= 3.0 or max_age >= 10.0:
        return "stale_latency_medium"
    return "fresh_low_latency"


def _inventory_bucket(row: dict[str, str]) -> str:
    position = abs(_float(row.get("position"), 0.0))
    score = _float(row.get("inventory_score"), 1.0)
    if position <= 0.0005:
        return "flat"
    if position >= 0.0015 or score <= 0.35:
        return "large_skew_or_low_score"
    if score <= 0.5:
        return "mild_skew_recovery_zone"
    return "mild_skew"


def _api_churn_bucket(row: dict[str, str]) -> str:
    reject = str(row.get("reject_reason", "") or row.get("reject_throttle_drop_cause", "")).strip()
    throttle = str(row.get("throttle_reason", "")).strip()
    if reject or throttle or _bool(row.get("dropped_by_api_limit")) or _bool(row.get("dropped_by_latency")):
        return "reject_throttle_or_drop"
    if row.get("min_move_passed") != "" and not _bool(row.get("min_move_passed")):
        return "min_move_failed"
    cancel_bucket = str(row.get("cancel_readd_bucket", "")).strip().lower()
    if cancel_bucket and cancel_bucket not in {"none", "normal", "0"}:
        return "cancel_readd_pressure"
    quote_age = _float(row.get("quote_age_ms"))
    if _finite(quote_age) and quote_age < 100.0:
        return "young_quote_churn"
    return "api_churn_normal"


def _post_only_bucket(safety: dict[str, str]) -> str:
    if not safety:
        return "post_only_unknown"
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


def _outcome_bucket(seq: int, ctx: SampleContext) -> str:
    labels = ctx.labels_by_seq.get(seq, [])
    if not labels:
        return "no_submit_no_fill"
    if any(_bool(row.get("fill_after_cancel_request")) for row in labels):
        return "fill_after_cancel"
    order_ids = {_order_id(row) for row in labels if _order_id(row)}
    five_second_markouts = [
        _float(row.get("side_adjusted_markout_ticks"))
        for order_id in order_ids
        for row in ctx.markouts_by_order_id.get(order_id, [])
        if _int(row.get("horizon_ms")) == 5000 and _bool(row.get("horizon_observable"))
    ]
    if any(_finite(value) and value < -2.0 for value in five_second_markouts):
        return "adverse_5s_markout"
    spread_values = [_float(row.get("realized_spread_proxy_ticks")) for row in labels]
    if any(_finite(value) and value > 0.0 for value in spread_values):
        return "positive_spread_capture"
    if not any(_float(row.get("fill_count"), 0.0) > 0.0 or _bool(row.get("full_fill")) for row in labels):
        return "no_fill"
    return "neutral_fill_outcome"


def _build_bucket_maps(ctx: SampleContext) -> dict[str, dict[int, str]]:
    vol_thresholds = _quantile_thresholds(_float(row.get("vol_bps")) for row in ctx.decisions)
    minute_submit_counts: dict[int, int] = defaultdict(int)
    for row in ctx.labels:
        ts = _int(row.get("submit_ts_local"))
        if ts is not None:
            minute_submit_counts[int(ts // 60_000_000_000)] += 1
    density_thresholds = _quantile_thresholds(minute_submit_counts.values())
    out: dict[str, dict[int, str]] = {
        "volatility_markout_dispersion": {},
        "spread_quote_distance": {},
        "trade_intensity_fill_opportunity": {},
        "latency_stale_age": {},
        "inventory_state": {},
        "api_churn": {},
        "post_only_safety": {},
        "cancel_fill_adverse_markout": {},
        "market_view_quality": {},
    }
    for seq, row in ctx.decisions_by_seq.items():
        out["volatility_markout_dispersion"][seq] = _tercile(_float(row.get("vol_bps")), vol_thresholds, "volatility")
        out["spread_quote_distance"][seq] = _spread_bucket(row)
        ts = _int(row.get("ts_local"))
        density = minute_submit_counts.get(int(ts // 60_000_000_000), 0) if ts is not None else math.nan
        out["trade_intensity_fill_opportunity"][seq] = _tercile(density, density_thresholds, "trade_intensity")
        out["latency_stale_age"][seq] = _latency_bucket(row)
        out["inventory_state"][seq] = _inventory_bucket(row)
        out["api_churn"][seq] = _api_churn_bucket(row)
        out["post_only_safety"][seq] = _post_only_bucket(ctx.safety_by_seq.get(seq, {}))
        out["cancel_fill_adverse_markout"][seq] = _outcome_bucket(seq, ctx)
        if ctx.market_view_quality != "strict_clean_market_view":
            out["market_view_quality"][seq] = "caveated_sample"
        elif _bool(ctx.safety_by_seq.get(seq, {}).get("join_gap_crossed")) or _bool(ctx.safety_by_seq.get(seq, {}).get("join_stale")):
            out["market_view_quality"][seq] = "stale_or_gap_caveat"
        else:
            out["market_view_quality"][seq] = "strict_clean_market_view"
    return out


def _labels_for_seqs(ctx: SampleContext, seqs: set[int]) -> list[dict[str, str]]:
    return [row for seq in seqs for row in ctx.labels_by_seq.get(seq, [])]


def _markouts_for_labels(ctx: SampleContext, labels: list[dict[str, str]]) -> list[dict[str, str]]:
    order_ids = {_order_id(row) for row in labels if _order_id(row)}
    return [
        row
        for order_id in order_ids
        for row in ctx.markouts_by_order_id.get(order_id, [])
        if _int(row.get("horizon_ms")) == 5000 and _bool(row.get("horizon_observable"))
    ]


def _metrics_for_seqs(ctx: SampleContext, seqs: set[int]) -> dict[str, Any]:
    labels = _labels_for_seqs(ctx, seqs)
    filled = [row for row in labels if _float(row.get("fill_count"), 0.0) > 0.0 or _bool(row.get("full_fill"))]
    markouts = _markouts_for_labels(ctx, labels)
    safety_rows = [ctx.safety_by_seq[seq] for seq in seqs if seq in ctx.safety_by_seq]
    spread_capture_values = [_float(row.get("realized_spread_proxy_ticks")) for row in markouts]
    if not any(_finite(value) for value in spread_capture_values):
        spread_capture_values = [_float(row.get("realized_spread_proxy_ticks")) for row in labels]
    return {
        "decision_rows": len(seqs),
        "submit_orders": len(labels),
        "filled_orders": len(filled),
        "fill_rate": _safe_num(_rate(len(filled), len(labels))),
        "markout_5000ms_ticks": _safe_num(_mean(_float(row.get("side_adjusted_markout_ticks")) for row in markouts)),
        "spread_capture_ticks_mean": _safe_num(_mean(spread_capture_values)),
        "fill_after_cancel_rate": _safe_num(_rate(sum(1 for row in labels if _bool(row.get("fill_after_cancel_request"))), len(labels))),
        "fast_cancel_churn_rate": _safe_num(_rate(sum(1 for row in labels if _bool(row.get("fast_cancel_churn"))), len(labels))),
        "inventory_increasing_fill_rate": _safe_num(_rate(sum(1 for row in filled if _bool(row.get("inventory_increasing_fill"))), len(filled))),
        "inventory_reducing_fill_rate": _safe_num(_rate(sum(1 for row in filled if _bool(row.get("inventory_reducing_fill"))), len(filled))),
        "post_only_risk_after_recheck_rows": sum(1 for row in safety_rows if _bool(row.get("post_only_risk_after_recheck"))),
        "net_pnl_proxy_ticks": _safe_num(sum(_float(row.get("fee_adjusted_realized_spread_ticks"), 0.0) for row in labels if _finite(_float(row.get("fee_adjusted_realized_spread_ticks"))))),
    }


def _delta(candidate: dict[str, Any], baseline: dict[str, Any], key: str) -> float | str:
    cand = _float(candidate.get(key))
    base = _float(baseline.get(key))
    if not (_finite(cand) and _finite(base)):
        return ""
    return cand - base


def _fine_bucket_metric_rows(ctx: SampleContext) -> list[dict[str, Any]]:
    bucket_maps = _build_bucket_maps(ctx)
    rows: list[dict[str, Any]] = []
    all_seqs = set(ctx.decisions_by_seq)
    for candidate_id in KEEP_CANDIDATE_IDS:
        candidate_seqs = ctx.candidate_seqs.get(candidate_id, set())
        for family, seq_to_bucket in bucket_maps.items():
            for bucket in sorted(set(seq_to_bucket.values())):
                bucket_seqs = {seq for seq, value in seq_to_bucket.items() if value == bucket}
                base_seqs = all_seqs & bucket_seqs
                cand_seqs = candidate_seqs & bucket_seqs
                base_metrics = _metrics_for_seqs(ctx, base_seqs)
                cand_metrics = _metrics_for_seqs(ctx, cand_seqs)
                row = {
                    "sample_id": ctx.sample_id,
                    "market_view_quality": ctx.market_view_quality,
                    "candidate_id": candidate_id,
                    "fine_bucket_family": family,
                    "fine_bucket": bucket,
                    **cand_metrics,
                    "baseline_decision_rows": base_metrics["decision_rows"],
                    "baseline_submit_orders": base_metrics["submit_orders"],
                    "baseline_filled_orders": base_metrics["filled_orders"],
                    "baseline_fill_rate": base_metrics["fill_rate"],
                    "baseline_markout_5000ms_ticks": base_metrics["markout_5000ms_ticks"],
                    "baseline_spread_capture_ticks_mean": base_metrics["spread_capture_ticks_mean"],
                    "baseline_fill_after_cancel_rate": base_metrics["fill_after_cancel_rate"],
                    "baseline_fast_cancel_churn_rate": base_metrics["fast_cancel_churn_rate"],
                    "fill_rate_delta_vs_bucket_baseline": _delta(cand_metrics, base_metrics, "fill_rate"),
                    "markout_5000ms_delta_vs_bucket_baseline": _delta(cand_metrics, base_metrics, "markout_5000ms_ticks"),
                    "spread_capture_delta_vs_bucket_baseline": _delta(cand_metrics, base_metrics, "spread_capture_ticks_mean"),
                    "fill_after_cancel_delta_vs_bucket_baseline": _delta(cand_metrics, base_metrics, "fill_after_cancel_rate"),
                    "fast_cancel_churn_delta_vs_bucket_baseline": _delta(cand_metrics, base_metrics, "fast_cancel_churn_rate"),
                    "net_pnl_proxy_delta_vs_bucket_baseline": _delta(cand_metrics, base_metrics, "net_pnl_proxy_ticks"),
                    "metric_status": (
                        "diagnostic_outcome_bucket"
                        if family == "cancel_fill_adverse_markout"
                        else "decision_time_visible_bucket"
                    ),
                }
                rows.append(row)
    return rows


def _bucket_verdict(rows: list[dict[str, Any]]) -> tuple[str, str]:
    total_decision = sum(int(_float(row.get("decision_rows"), 0.0)) for row in rows)
    total_submit = sum(int(_float(row.get("submit_orders"), 0.0)) for row in rows)
    total_fill = sum(int(_float(row.get("filled_orders"), 0.0)) for row in rows)
    fill_samples = sum(1 for row in rows if int(_float(row.get("filled_orders"), 0.0)) > 0)
    post_only_risk = sum(int(_float(row.get("post_only_risk_after_recheck_rows"), 0.0)) for row in rows)
    markout_deltas = [_float(row.get("markout_5000ms_delta_vs_bucket_baseline")) for row in rows]
    fill_deltas = [_float(row.get("fill_rate_delta_vs_bucket_baseline")) for row in rows]
    churn_deltas = [_float(row.get("fast_cancel_churn_delta_vs_bucket_baseline")) for row in rows]
    fac_deltas = [_float(row.get("fill_after_cancel_delta_vs_bucket_baseline")) for row in rows]
    wins = sum(1 for value in markout_deltas if _finite(value) and value >= 0.0)
    observed = sum(1 for value in markout_deltas if _finite(value))
    win_rate = _rate(wins, observed)
    median_markout = _nanmedian(markout_deltas)
    median_fill = _nanmedian(fill_deltas)
    median_churn = _nanmedian(churn_deltas)
    median_fac = _nanmedian(fac_deltas)
    worst_markout = _nanmin(markout_deltas)
    if total_decision == 0 or total_submit == 0:
        return "not_decisionable", "candidate has no decision or submit coverage in this fine bucket"
    if post_only_risk > 0:
        return "reject_bucket", "post-only risk after recheck is nonzero"
    if _finite(median_markout) and _finite(median_fill) and median_markout < -2.0 and median_fill < -0.001:
        return "reject_bucket", "5s markout worsens while fill rate also drops"
    if _finite(worst_markout) and worst_markout < -5.0:
        return "reject_bucket", "worst-sample 5s markout delta is materially adverse"
    if total_fill < MIN_BUCKET_FILLS or fill_samples < MIN_BUCKET_FILL_SAMPLES:
        return "needs_more_fills", "insufficient filled-order mass for stable bucket interpretation"
    if (
        _finite(median_markout)
        and median_markout >= 0.0
        and (not _finite(worst_markout) or worst_markout > -2.0)
        and _finite(win_rate)
        and win_rate >= 0.6
        and (not _finite(median_fill) or median_fill > -0.005)
        and (not _finite(median_churn) or median_churn <= 0.02)
        and (not _finite(median_fac) or median_fac <= 0.01)
    ):
        return "stable_promising_bucket", "non-worse markout with acceptable fill/churn/cancel-fill behavior"
    return "mixed_keep_for_research", "mixed or weak evidence after fine-bucket split"


def _stability_rows(metric_rows: list[dict[str, Any]], caveated_sample_ids: set[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        groups[(row["candidate_id"], row["fine_bucket_family"], row["fine_bucket"])].append(row)
    out: list[dict[str, Any]] = []
    for (candidate_id, family, bucket), rows in sorted(groups.items()):
        verdict, reason = _bucket_verdict(rows)
        clean = [row for row in rows if row.get("sample_id") not in caveated_sample_ids]
        clean_verdict, clean_reason = _bucket_verdict(clean) if clean else ("not_decisionable", "no clean-only rows")
        markout_deltas = [_float(row.get("markout_5000ms_delta_vs_bucket_baseline")) for row in rows]
        fill_deltas = [_float(row.get("fill_rate_delta_vs_bucket_baseline")) for row in rows]
        churn_deltas = [_float(row.get("fast_cancel_churn_delta_vs_bucket_baseline")) for row in rows]
        out.append(
            {
                "candidate_id": candidate_id,
                "fine_bucket_family": family,
                "fine_bucket": bucket,
                "sample_count": len(rows),
                "clean_only_sample_count": len(clean),
                "decision_coverage_sample_count": sum(1 for row in rows if int(_float(row.get("decision_rows"), 0.0)) > 0),
                "submit_coverage_sample_count": sum(1 for row in rows if int(_float(row.get("submit_orders"), 0.0)) > 0),
                "fill_coverage_sample_count": sum(1 for row in rows if int(_float(row.get("filled_orders"), 0.0)) > 0),
                "total_decision_rows": sum(int(_float(row.get("decision_rows"), 0.0)) for row in rows),
                "total_submit_orders": sum(int(_float(row.get("submit_orders"), 0.0)) for row in rows),
                "total_filled_orders": sum(int(_float(row.get("filled_orders"), 0.0)) for row in rows),
                "median_fill_rate_delta_vs_bucket_baseline": _safe_num(_nanmedian(fill_deltas)),
                "median_markout_5000ms_delta_vs_bucket_baseline": _safe_num(_nanmedian(markout_deltas)),
                "worst_markout_5000ms_delta_vs_bucket_baseline": _safe_num(_nanmin(markout_deltas)),
                "markout_win_rate": _safe_num(
                    _rate(
                        sum(1 for value in markout_deltas if _finite(value) and value >= 0.0),
                        sum(1 for value in markout_deltas if _finite(value)),
                    )
                ),
                "median_fast_cancel_churn_delta_vs_bucket_baseline": _safe_num(_nanmedian(churn_deltas)),
                "accepted_verdict": verdict,
                "accepted_reason": reason,
                "clean_only_verdict": clean_verdict,
                "clean_only_reason": clean_reason,
                "recommendation": _recommendation(row_family=family, rows=rows, verdict=verdict, clean_verdict=clean_verdict),
            }
        )
    return out


def _has_parameter_seed_signal(rows: list[dict[str, Any]]) -> bool:
    markout = _nanmedian(_float(row.get("markout_5000ms_delta_vs_bucket_baseline")) for row in rows)
    fill = _nanmedian(_float(row.get("fill_rate_delta_vs_bucket_baseline")) for row in rows)
    churn = _nanmedian(_float(row.get("fast_cancel_churn_delta_vs_bucket_baseline")) for row in rows)
    cancel_fill = _nanmedian(_float(row.get("fill_after_cancel_delta_vs_bucket_baseline")) for row in rows)
    return (
        (_finite(markout) and markout > 0.25)
        or (_finite(fill) and fill > 0.001)
        or (_finite(churn) and churn < -0.005)
        or (_finite(cancel_fill) and cancel_fill < -0.002)
    )


def _recommendation(
    *,
    row_family: str,
    rows: list[dict[str, Any]],
    verdict: str,
    clean_verdict: str,
) -> str:
    if verdict == "stable_promising_bucket" and clean_verdict == "stable_promising_bucket":
        if row_family not in PARAMETER_SEED_BUCKET_FAMILIES:
            return "diagnostic_only_not_sweep_seed"
        if _has_parameter_seed_signal(rows):
            return "parameter_sweep_seed"
        return "stable_but_no_incremental_signal"
    if verdict == "reject_bucket" or clean_verdict == "reject_bucket":
        return "reject_bucket"
    if verdict == "needs_more_fills" or clean_verdict == "needs_more_fills":
        return "collect_more_targeted_fills"
    if verdict == "not_decisionable":
        return "not_decisionable"
    return "keep_for_research"


def _sample_gap_rows(stability_rows: list[dict[str, Any]], *, aggregate_filled_orders: int) -> list[dict[str, Any]]:
    rows = []
    for row in stability_rows:
        if row["accepted_verdict"] not in {"needs_more_fills", "not_decisionable"}:
            continue
        filled = int(_float(row.get("total_filled_orders"), 0.0))
        rows.append(
            {
                "candidate_id": row["candidate_id"],
                "fine_bucket_family": row["fine_bucket_family"],
                "fine_bucket": row["fine_bucket"],
                "gap_type": row["accepted_verdict"],
                "total_filled_orders": filled,
                "additional_fills_to_min_bucket": max(0, MIN_BUCKET_FILLS - filled),
                "target_regime": f"{row['fine_bucket_family']}={row['fine_bucket']}",
                "note": "collect targeted current-format no-rule/default-off samples; do not just extend calm duration",
            }
        )
    rows.append(
        {
            "candidate_id": "all",
            "fine_bucket_family": "global_tiny_live_threshold",
            "fine_bucket": "filled_orders",
            "gap_type": "global_filled_order_gap",
            "total_filled_orders": aggregate_filled_orders,
            "additional_fills_to_min_bucket": max(0, TINY_LIVE_DESIGN_THRESHOLDS["filled_orders"] - aggregate_filled_orders),
            "target_regime": "non-calm regimes with natural fill opportunity",
            "note": "future promotion-style work still needs at least 500 aggregate fills",
        }
    )
    return rows


def _write_recommendations(path: Path, stability_rows: list[dict[str, Any]]) -> None:
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        by_candidate[row["candidate_id"]].append(row)

    lines = [
        "# 0525T001 Step 9D Fine-Bucket Recommendations",
        "",
        "## Boundary",
        "",
        "- Mode: read-only fine-bucket refinement.",
        "- No live, default-on, guard relaxation, strategy behavior change, or parameter search was performed.",
        "- PnL proxy remains diagnostic and is not a promotion metric.",
        "",
        "## Candidate Summary",
        "",
    ]
    for candidate_id in KEEP_CANDIDATE_IDS:
        rows = by_candidate.get(candidate_id, [])
        counts = defaultdict(int)
        for row in rows:
            counts[row["accepted_verdict"]] += 1
        seeds = [row for row in rows if row["recommendation"] == "parameter_sweep_seed"]
        lines.extend(
            [
                f"### `{candidate_id}`",
                "",
                f"- stable promising buckets: `{counts['stable_promising_bucket']}`",
                f"- mixed keep-for-research buckets: `{counts['mixed_keep_for_research']}`",
                f"- needs-more-fills buckets: `{counts['needs_more_fills']}`",
                f"- reject buckets: `{counts['reject_bucket']}`",
                f"- not-decisionable buckets: `{counts['not_decisionable']}`",
            ]
        )
        if seeds:
            rendered = ", ".join(f"{row['fine_bucket_family']}={row['fine_bucket']}" for row in seeds[:8])
            lines.append(f"- parameter-sweep seed buckets: {rendered}")
        else:
            lines.append("- parameter-sweep seed buckets: none")
        diagnostic_stable = [row for row in rows if row["recommendation"] == "diagnostic_only_not_sweep_seed"]
        if diagnostic_stable:
            rendered = ", ".join(f"{row['fine_bucket_family']}={row['fine_bucket']}" for row in diagnostic_stable[:6])
            lines.append(f"- diagnostic-only stable buckets: {rendered}")
        lines.append("")

    lines.extend(
        [
            "## Required Judgments",
            "",
            "- `min_move_quote_age_churn_guard`: judge whether stable buckets concentrate in API/churn, quote-age, or min-move regimes before treating it as a sweep seed.",
            "- `inventory_reservation_shift_band`: judge whether stable buckets concentrate in inventory skew or recovery-side regimes instead of aggregate sample mixing.",
            "- `size_reduction_or_add_side_suppression_pressure`: judge whether favorable markout buckets coincide with toxic-fill reduction rather than broad participation loss.",
            "- Outcome-defined and market-view-quality buckets may explain behavior or sample gaps, but they are not parameter-sweep seeds because they are not live decision inputs.",
            "",
            "## Next Step Rule",
            "",
            "- If at least one clean-only stable promising bucket exists, it can seed a later multi-sample parameter-sweep design task.",
            "- If buckets are mostly `needs_more_fills`, collect targeted current-format samples in the listed regimes.",
            "- If buckets are mostly `reject_bucket`, do not spend parameter-search budget on that candidate family.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_candidate_bucket_refinement(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    caveated_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    caveated = set(caveated_sample_ids or set())
    contexts = [_load_context(path) for path in run_dirs]
    metric_rows: list[dict[str, Any]] = []
    for ctx in contexts:
        metric_rows.extend(_fine_bucket_metric_rows(ctx))
    stability = _stability_rows(metric_rows, caveated)
    aggregate_filled_orders = sum(
        1
        for ctx in contexts
        for row in ctx.labels
        if _float(row.get("fill_count"), 0.0) > 0.0 or _bool(row.get("full_fill"))
    )
    gaps = _sample_gap_rows(stability, aggregate_filled_orders=aggregate_filled_orders)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "fine_bucket_metrics.csv", metric_rows)
    _write_csv(output_dir / "fine_bucket_stability_summary.csv", stability)
    _write_json(output_dir / "fine_bucket_stability_summary.json", stability)
    _write_csv(output_dir / "sample_gap_recommendations.csv", gaps)
    _write_recommendations(output_dir / "candidate_bucket_recommendations.md", stability)
    summary = {
        "task_id": TASK_ID,
        "runner_mode": RUNNER_MODE,
        "generated_at": _generated_at(),
        "sample_ids": [ctx.sample_id for ctx in contexts],
        "caveated_sample_ids": sorted(caveated),
        "candidate_ids": list(KEEP_CANDIDATE_IDS),
        "stable_promising_bucket_count": sum(1 for row in stability if row["accepted_verdict"] == "stable_promising_bucket"),
        "parameter_sweep_seed_count": sum(1 for row in stability if row["recommendation"] == "parameter_sweep_seed"),
        "not_authorized": ["live", "default-on", "guard relaxation", "strategy behavior change", "parameter search", "promotion"],
    }
    _write_json(output_dir / "run_manifest.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="local_live_analysis/<run_id> to include")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--caveated-sample-id", action="append", default=sorted(DEFAULT_CAVEATED_SAMPLE_IDS))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_candidate_bucket_refinement(
        run_dirs=args.run_dir,
        output_dir=args.output_dir,
        caveated_sample_ids=set(args.caveated_sample_id),
    )


if __name__ == "__main__":
    main()
