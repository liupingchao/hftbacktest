#!/usr/bin/env python3
"""Step 9G narrow min-move / quote-age / churn read-only parameter sweep.

This runner is diagnostic only. It projects how a narrow
``min_move_quote_age_churn_guard`` would have suppressed decision rows inside
two accepted seed regimes, then compares the projected kept rows with the
sample-local seed baseline. It never changes live strategy behavior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from candidate_bucket_refinement import (
    SampleContext,
    _build_bucket_maps,
    _finite,
    _load_context,
    _metrics_for_seqs,
    _nanmedian,
    _nanmin,
)
from quote_adjustment_replay import (
    _bool,
    _float,
    _generated_at,
    _int,
    _safe_num,
    _write_csv,
    _write_json,
)


TASK_ID = "0526T004"
RUNNER_MODE = "step9g_narrow_min_move_parameter_sweep"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/stage9g_min_move_parameter_sweep_0526T004")
DEFAULT_RUN_DIRS = [
    Path("local_live_analysis/5-19-day-control-30min"),
    Path("local_live_analysis/5-19-night-active-30min-a"),
    Path("local_live_analysis/5-19-night-active-30min-b"),
    Path("local_live_analysis/5-19-night-active-30min-c"),
    Path("local_live_analysis/5-21-day-control-60min"),
    Path("local_live_analysis/5-26-active-minmove-control-30min-a"),
    Path("local_live_analysis/5-26-active-minmove-control-60min-a"),
]
DEFAULT_CAVEATED_SAMPLE_IDS = {"5-19-night-active-30min-a", "5-26-active-minmove-control-60min-a"}
SEED_A_FAMILY = "inventory_state"
SEED_A_BUCKET = "large_skew_or_low_score"
SEED_B_FAMILY = "latency_stale_age"
SEED_B_BUCKET = "stale_latency_medium"
SEED_SLICES = ("inventory_only", "stale_latency_only", "intersection")
FULL_GRID_VALUES = {
    "min_move_ticks": [1, 2, 3],
    "min_quote_age_ms": [250, 500, 1000, 2000],
    "churn_window_ms": [1000, 3000, 5000],
    "max_readd_count_in_window": [1, 2, 3],
    "stale_latency_guard_ms": [50, 100, 200],
}
SMOKE_GRID_VALUES = {
    "min_move_ticks": [1],
    "min_quote_age_ms": [500],
    "churn_window_ms": [1000],
    "max_readd_count_in_window": [1],
    "stale_latency_guard_ms": [100],
}
MIN_CLEAN_SAMPLES_WITH_FILLS = 2
MIN_CLEAN_FILLED_ORDERS = 30
MAX_FILL_RATE_LOSS_FOR_PROMISING = -0.01


@dataclass(frozen=True)
class ParameterSet:
    param_hash: str
    min_move_ticks: int
    min_quote_age_ms: int
    churn_window_ms: int
    max_readd_count_in_window: int
    stale_latency_guard_ms: int | None


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else math.nan


def _nanmax(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return max(finite) if finite else math.nan


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def build_parameter_grid(*, smoke: bool = False) -> list[ParameterSet]:
    values = SMOKE_GRID_VALUES if smoke else FULL_GRID_VALUES
    params: list[ParameterSet] = []
    seen: set[str] = set()
    for min_move in values["min_move_ticks"]:
        for quote_age in values["min_quote_age_ms"]:
            for churn_window in values["churn_window_ms"]:
                for max_readd in values["max_readd_count_in_window"]:
                    base = {
                        "min_move_ticks": min_move,
                        "min_quote_age_ms": quote_age,
                        "churn_window_ms": churn_window,
                        "max_readd_count_in_window": max_readd,
                        "stale_latency_guard_ms": None,
                    }
                    param_hash = _stable_hash(base)
                    if param_hash not in seen:
                        seen.add(param_hash)
                        params.append(ParameterSet(param_hash=param_hash, **base))
                    for stale_guard in values["stale_latency_guard_ms"]:
                        with_stale = {
                            **base,
                            "stale_latency_guard_ms": stale_guard,
                        }
                        param_hash = _stable_hash(with_stale)
                        if param_hash not in seen:
                            seen.add(param_hash)
                            params.append(ParameterSet(param_hash=param_hash, **with_stale))
    return sorted(params, key=lambda item: item.param_hash)


def _parameter_applies_to_slice(param: ParameterSet, seed_slice: str) -> bool:
    if seed_slice == "inventory_only":
        return param.stale_latency_guard_ms is None
    return param.stale_latency_guard_ms is not None


def _parameter_row(param: ParameterSet) -> dict[str, Any]:
    return {
        "param_hash": param.param_hash,
        "min_move_ticks": param.min_move_ticks,
        "min_quote_age_ms": param.min_quote_age_ms,
        "churn_window_ms": param.churn_window_ms,
        "max_readd_count_in_window": param.max_readd_count_in_window,
        "stale_latency_guard_ms": "" if param.stale_latency_guard_ms is None else param.stale_latency_guard_ms,
    }


def _seq_timestamp(row: dict[str, str]) -> int | None:
    return _int(row.get("ts_local"))


def _min_abs_move_ticks(row: dict[str, str]) -> float:
    values = [
        abs(_float(row.get("target_move_since_last_quote_or_cancel_buy"))),
        abs(_float(row.get("target_move_since_last_quote_or_cancel_sell"))),
    ]
    finite = [value for value in values if _finite(value)]
    if finite:
        return min(finite)
    if row.get("min_move_passed") != "" and not _bool(row.get("min_move_passed")):
        return 0.0
    return math.inf


def _quote_age_ms(row: dict[str, str]) -> float:
    values = [
        _float(row.get("quote_age_ms")),
        _float(row.get("join_age_ms")),
        _float(row.get("anchor_age_ms")),
    ]
    finite = [value for value in values if _finite(value)]
    return min(finite) if finite else math.inf


def _stale_age_ms(row: dict[str, str]) -> float:
    values = [
        _float(row.get("book_view_stale_ms")),
        _float(row.get("join_age_ms")),
        _float(row.get("anchor_age_ms")),
    ]
    finite = [value for value in values if _finite(value)]
    return max(finite) if finite else 0.0


def _is_submit_like(row: dict[str, str]) -> bool:
    return "submit" in str(row.get("action", "")).lower() or "submit" in str(row.get("planned_action", "")).lower()


def _is_safety_or_inventory_bypass(row: dict[str, str], safety: dict[str, str] | None) -> bool:
    if safety and (
        _bool(safety.get("post_only_risk_after_recheck"))
        or _bool(safety.get("missing_anchor"))
        or _bool(safety.get("stale_anchor"))
    ):
        return True
    action = str(row.get("action", "")).lower()
    position = _float(row.get("position"), 0.0)
    if position > 0.0 and "sell" in action:
        return True
    if position < 0.0 and "buy" in action:
        return True
    return False


def _precomputed_features(ctx: SampleContext, seed_seqs: set[int]) -> dict[int, dict[str, Any]]:
    rows: list[tuple[int, int, dict[str, str]]] = []
    for seq in seed_seqs:
        row = ctx.decisions_by_seq.get(seq)
        if not row:
            continue
        ts = _seq_timestamp(row)
        if ts is None:
            continue
        rows.append((ts, seq, row))
    rows.sort()
    active_timestamps = [
        ts
        for ts, _seq, row in rows
        if _is_submit_like(row) or str(row.get("cancel_readd_bucket", "")).strip().lower() not in {"", "none", "normal", "0"}
    ]
    features: dict[int, dict[str, Any]] = {}
    for ts, seq, row in rows:
        recent_counts: dict[int, int] = {}
        for window_ms in FULL_GRID_VALUES["churn_window_ms"]:
            lower = ts - int(window_ms * 1_000_000)
            left = _lower_bound(active_timestamps, lower)
            right = _lower_bound(active_timestamps, ts)
            recent_counts[int(window_ms)] = max(0, right - left)
        features[seq] = {
            "is_submit_like": _is_submit_like(row),
            "safety_or_inventory_bypass": _is_safety_or_inventory_bypass(row, ctx.safety_by_seq.get(seq)),
            "min_abs_move_ticks": _min_abs_move_ticks(row),
            "quote_age_ms": _quote_age_ms(row),
            "stale_age_ms": _stale_age_ms(row),
            "recent_counts": recent_counts,
        }
    return features


def _lower_bound(values: list[int], target: int) -> int:
    lo = 0
    hi = len(values)
    while lo < hi:
        mid = (lo + hi) // 2
        if values[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _should_suppress(feature: dict[str, Any], param: ParameterSet) -> bool:
    if not feature.get("is_submit_like"):
        return False
    if feature.get("safety_or_inventory_bypass"):
        return False
    move_small = _float(feature.get("min_abs_move_ticks")) < float(param.min_move_ticks)
    age_young = _float(feature.get("quote_age_ms")) < float(param.min_quote_age_ms)
    recent_counts = feature.get("recent_counts") or {}
    churn_count = int(recent_counts.get(int(param.churn_window_ms), 0))
    churn_active = churn_count >= int(param.max_readd_count_in_window)
    stale_ok = True
    if param.stale_latency_guard_ms is not None:
        stale_ok = _float(feature.get("stale_age_ms")) >= float(param.stale_latency_guard_ms)
    return bool(move_small and age_young and churn_active and stale_ok)


def _seed_seqs(ctx: SampleContext, seed_slice: str) -> set[int]:
    bucket_maps = _build_bucket_maps(ctx)
    inventory = {
        seq
        for seq, bucket in bucket_maps[SEED_A_FAMILY].items()
        if bucket == SEED_A_BUCKET
    }
    latency = {
        seq
        for seq, bucket in bucket_maps[SEED_B_FAMILY].items()
        if bucket == SEED_B_BUCKET
    }
    if seed_slice == "inventory_only":
        return inventory - latency
    if seed_slice == "stale_latency_only":
        return latency - inventory
    if seed_slice == "intersection":
        return inventory & latency
    raise ValueError(f"unknown seed slice: {seed_slice}")


def _metrics_delta(kept: dict[str, Any], baseline: dict[str, Any], key: str) -> float | str:
    value = _float(kept.get(key))
    base = _float(baseline.get(key))
    if not (_finite(value) and _finite(base)):
        return ""
    return value - base


def _sample_param_metric(
    ctx: SampleContext,
    seed_slice: str,
    param: ParameterSet,
    *,
    worker_id: str,
    seed_seqs: set[int] | None = None,
    features: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    seed_seqs = seed_seqs if seed_seqs is not None else _seed_seqs(ctx, seed_slice)
    features = features if features is not None else _precomputed_features(ctx, seed_seqs)
    suppressed = {seq for seq, feature in features.items() if _should_suppress(feature, param)}
    kept = seed_seqs - suppressed
    baseline_metrics = _metrics_for_seqs(ctx, seed_seqs)
    kept_metrics = _metrics_for_seqs(ctx, kept)
    suppressed_metrics = _metrics_for_seqs(ctx, suppressed)
    return {
        "sample_id": ctx.sample_id,
        "market_view_quality": ctx.market_view_quality,
        "is_caveated_sample": "",
        "seed_slice": seed_slice,
        **_parameter_row(param),
        "worker_id": worker_id,
        "seed_decision_rows": baseline_metrics["decision_rows"],
        "kept_decision_rows": kept_metrics["decision_rows"],
        "suppressed_decision_rows": suppressed_metrics["decision_rows"],
        "suppressed_decision_rate": _safe_num(_rate(int(suppressed_metrics["decision_rows"]), int(baseline_metrics["decision_rows"]))),
        "baseline_submit_orders": baseline_metrics["submit_orders"],
        "kept_submit_orders": kept_metrics["submit_orders"],
        "suppressed_submit_orders": suppressed_metrics["submit_orders"],
        "submit_reduction": int(baseline_metrics["submit_orders"]) - int(kept_metrics["submit_orders"]),
        "submit_reduction_rate": _safe_num(_rate(int(baseline_metrics["submit_orders"]) - int(kept_metrics["submit_orders"]), int(baseline_metrics["submit_orders"]))),
        "baseline_filled_orders": baseline_metrics["filled_orders"],
        "kept_filled_orders": kept_metrics["filled_orders"],
        "suppressed_filled_orders": suppressed_metrics["filled_orders"],
        "fill_reduction": int(baseline_metrics["filled_orders"]) - int(kept_metrics["filled_orders"]),
        "baseline_fill_rate": baseline_metrics["fill_rate"],
        "kept_fill_rate": kept_metrics["fill_rate"],
        "fill_rate_delta": _metrics_delta(kept_metrics, baseline_metrics, "fill_rate"),
        "baseline_markout_5000ms_ticks": baseline_metrics["markout_5000ms_ticks"],
        "kept_markout_5000ms_ticks": kept_metrics["markout_5000ms_ticks"],
        "suppressed_markout_5000ms_ticks": suppressed_metrics["markout_5000ms_ticks"],
        "markout_5000ms_delta": _metrics_delta(kept_metrics, baseline_metrics, "markout_5000ms_ticks"),
        "baseline_spread_capture_ticks_mean": baseline_metrics["spread_capture_ticks_mean"],
        "kept_spread_capture_ticks_mean": kept_metrics["spread_capture_ticks_mean"],
        "spread_capture_delta": _metrics_delta(kept_metrics, baseline_metrics, "spread_capture_ticks_mean"),
        "baseline_fill_after_cancel_rate": baseline_metrics["fill_after_cancel_rate"],
        "kept_fill_after_cancel_rate": kept_metrics["fill_after_cancel_rate"],
        "fill_after_cancel_delta": _metrics_delta(kept_metrics, baseline_metrics, "fill_after_cancel_rate"),
        "baseline_fast_cancel_churn_rate": baseline_metrics["fast_cancel_churn_rate"],
        "kept_fast_cancel_churn_rate": kept_metrics["fast_cancel_churn_rate"],
        "fast_cancel_churn_delta": _metrics_delta(kept_metrics, baseline_metrics, "fast_cancel_churn_rate"),
        "baseline_inventory_increasing_fill_rate": baseline_metrics["inventory_increasing_fill_rate"],
        "kept_inventory_increasing_fill_rate": kept_metrics["inventory_increasing_fill_rate"],
        "inventory_increasing_fill_rate_delta": _metrics_delta(kept_metrics, baseline_metrics, "inventory_increasing_fill_rate"),
        "baseline_inventory_reducing_fill_rate": baseline_metrics["inventory_reducing_fill_rate"],
        "kept_inventory_reducing_fill_rate": kept_metrics["inventory_reducing_fill_rate"],
        "inventory_reducing_fill_rate_delta": _metrics_delta(kept_metrics, baseline_metrics, "inventory_reducing_fill_rate"),
        "post_only_risk_after_recheck_rows": kept_metrics["post_only_risk_after_recheck_rows"],
        "net_pnl_proxy_delta": _metrics_delta(kept_metrics, baseline_metrics, "net_pnl_proxy_ticks"),
        "metric_status": "read_only_projected_suppression",
    }


def _chunked(items: list[ParameterSet], chunk_size: int) -> list[list[ParameterSet]]:
    return [items[idx : idx + chunk_size] for idx in range(0, len(items), chunk_size)]


def _run_shard(
    *,
    run_dir: Path,
    seed_slice: str,
    params: list[ParameterSet],
    worker_id: str,
    caveated_sample_ids: set[str],
) -> list[dict[str, Any]]:
    ctx = _load_context(run_dir)
    seed_seqs = _seed_seqs(ctx, seed_slice)
    features = _precomputed_features(ctx, seed_seqs)
    rows = [
        _sample_param_metric(
            ctx,
            seed_slice,
            param,
            worker_id=worker_id,
            seed_seqs=seed_seqs,
            features=features,
        )
        for param in params
    ]
    for row in rows:
        row["is_caveated_sample"] = "true" if row["sample_id"] in caveated_sample_ids else "false"
    return rows


def _write_shard(path: Path, rows: list[dict[str, Any]]) -> None:
    _write_csv(path, rows)


def _load_context_for_coverage(run_dir: Path, caveated: set[str]) -> list[dict[str, Any]]:
    ctx = _load_context(run_dir)
    rows = []
    for seed_slice in SEED_SLICES:
        seed = _seed_seqs(ctx, seed_slice)
        metrics = _metrics_for_seqs(ctx, seed)
        rows.append(
            {
                "sample_id": ctx.sample_id,
                "market_view_quality": ctx.market_view_quality,
                "is_caveated_sample": "true" if ctx.sample_id in caveated else "false",
                "seed_slice": seed_slice,
                "seed_decision_rows": metrics["decision_rows"],
                "seed_submit_orders": metrics["submit_orders"],
                "seed_filled_orders": metrics["filled_orders"],
                "seed_fill_rate": metrics["fill_rate"],
            }
        )
    return rows


def _classify_parameter(rows: list[dict[str, Any]], *, caveated_sample_ids: set[str]) -> tuple[str, str]:
    clean = [row for row in rows if row.get("sample_id") not in caveated_sample_ids]
    all_filled = sum(int(_float(row.get("kept_filled_orders"), 0.0)) for row in rows)
    clean_filled = sum(int(_float(row.get("kept_filled_orders"), 0.0)) for row in clean)
    clean_fill_samples = sum(1 for row in clean if int(_float(row.get("kept_filled_orders"), 0.0)) > 0)
    post_only_risk = sum(int(_float(row.get("post_only_risk_after_recheck_rows"), 0.0)) for row in rows)
    clean_markout = [_float(row.get("markout_5000ms_delta")) for row in clean]
    all_markout = [_float(row.get("markout_5000ms_delta")) for row in rows]
    clean_churn = [_float(row.get("fast_cancel_churn_delta")) for row in clean]
    clean_fill_delta = [_float(row.get("fill_rate_delta")) for row in clean]
    clean_submit_reduction = [_float(row.get("submit_reduction_rate")) for row in clean]
    clean_fac = [_float(row.get("fill_after_cancel_delta")) for row in clean]
    clean_spread = [_float(row.get("spread_capture_delta")) for row in clean]
    median_markout = _nanmedian(clean_markout)
    worst_markout = _nanmin(clean_markout)
    median_churn = _nanmedian(clean_churn)
    median_fill = _nanmedian(clean_fill_delta)
    median_submit_reduction = _nanmedian(clean_submit_reduction)
    median_fac = _nanmedian(clean_fac)
    median_spread = _nanmedian(clean_spread)
    clean_observed = sum(1 for value in clean_markout if _finite(value))
    clean_wins = sum(1 for value in clean_markout if _finite(value) and value >= 0.0)
    all_observed = sum(1 for value in all_markout if _finite(value))
    all_wins = sum(1 for value in all_markout if _finite(value) and value >= 0.0)
    clean_win_rate = _rate(clean_wins, clean_observed)
    all_win_rate = _rate(all_wins, all_observed)
    if not clean:
        return "not_decisionable", "no clean-only samples available"
    if post_only_risk > 0:
        return "reject", "post-only risk after recheck is nonzero"
    if clean_filled < MIN_CLEAN_FILLED_ORDERS or clean_fill_samples < MIN_CLEAN_SAMPLES_WITH_FILLS:
        if _finite(median_churn) and median_churn < 0.0:
            return "stable_but_low_fill", "directional churn improvement but insufficient clean filled-order mass"
        return "not_decisionable", "insufficient clean filled-order mass"
    if _finite(median_fill) and median_fill < MAX_FILL_RATE_LOSS_FOR_PROMISING:
        if _finite(median_markout) and median_markout > 0.0:
            return "too_conservative_fill_loss", "markout improves but fill-rate loss is too large"
        return "reject", "fill-rate loss is too large without markout improvement"
    if _finite(worst_markout) and worst_markout < -5.0:
        return "reject", "worst clean-sample 5s markout delta is materially adverse"
    if (
        _finite(median_churn)
        and median_churn < 0.0
        and (not _finite(median_markout) or median_markout >= 0.0)
        and (not _finite(median_fac) or median_fac <= 0.0)
        and (not _finite(median_fill) or median_fill >= MAX_FILL_RATE_LOSS_FOR_PROMISING)
        and (not _finite(all_win_rate) or all_win_rate >= 0.5)
        and (not _finite(clean_win_rate) or clean_win_rate >= 0.5)
    ):
        if not _finite(median_spread) or median_spread >= -0.25:
            return "sweep_seed_promising", "clean samples show churn reduction without material fill/safety degradation"
    if _finite(median_submit_reduction) and median_submit_reduction > 0.05 and (
        not _finite(median_churn) or median_churn >= 0.0
    ):
        return "churn_only_no_execution_benefit", "participation changes without execution-quality improvement"
    if all_filled > clean_filled and clean_observed == 0 and all_observed > 0:
        return "caveated_only", "effect is only observable in caveated samples"
    return "not_decisionable", "mixed or weak clean-only evidence"


def _stability_rows(metric_rows: list[dict[str, Any]], caveated_sample_ids: set[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in metric_rows:
        groups.setdefault((row["seed_slice"], row["param_hash"]), []).append(row)
    out: list[dict[str, Any]] = []
    for (seed_slice, param_hash), rows in sorted(groups.items()):
        verdict, reason = _classify_parameter(rows, caveated_sample_ids=caveated_sample_ids)
        clean = [row for row in rows if row.get("sample_id") not in caveated_sample_ids]
        all_markout = [_float(row.get("markout_5000ms_delta")) for row in rows]
        clean_markout = [_float(row.get("markout_5000ms_delta")) for row in clean]
        clean_churn = [_float(row.get("fast_cancel_churn_delta")) for row in clean]
        clean_fill = [_float(row.get("fill_rate_delta")) for row in clean]
        clean_submit_reduction = [_float(row.get("submit_reduction_rate")) for row in clean]
        clean_fac = [_float(row.get("fill_after_cancel_delta")) for row in clean]
        clean_spread = [_float(row.get("spread_capture_delta")) for row in clean]
        first = rows[0]
        out.append(
            {
                "candidate_id": "min_move_quote_age_churn_guard",
                "seed_slice": seed_slice,
                "param_hash": param_hash,
                "min_move_ticks": first["min_move_ticks"],
                "min_quote_age_ms": first["min_quote_age_ms"],
                "churn_window_ms": first["churn_window_ms"],
                "max_readd_count_in_window": first["max_readd_count_in_window"],
                "stale_latency_guard_ms": first["stale_latency_guard_ms"],
                "sample_count": len(rows),
                "clean_only_sample_count": len(clean),
                "total_seed_decision_rows": sum(int(_float(row.get("seed_decision_rows"), 0.0)) for row in rows),
                "total_suppressed_decision_rows": sum(int(_float(row.get("suppressed_decision_rows"), 0.0)) for row in rows),
                "total_seed_submit_orders": sum(int(_float(row.get("baseline_submit_orders"), 0.0)) for row in rows),
                "total_kept_submit_orders": sum(int(_float(row.get("kept_submit_orders"), 0.0)) for row in rows),
                "total_seed_filled_orders": sum(int(_float(row.get("baseline_filled_orders"), 0.0)) for row in rows),
                "total_kept_filled_orders": sum(int(_float(row.get("kept_filled_orders"), 0.0)) for row in rows),
                "clean_only_kept_filled_orders": sum(int(_float(row.get("kept_filled_orders"), 0.0)) for row in clean),
                "clean_only_fill_sample_count": sum(1 for row in clean if int(_float(row.get("kept_filled_orders"), 0.0)) > 0),
                "clean_median_submit_reduction_rate": _safe_num(_nanmedian(clean_submit_reduction)),
                "clean_median_fill_rate_delta": _safe_num(_nanmedian(clean_fill)),
                "clean_median_markout_5000ms_delta": _safe_num(_nanmedian(clean_markout)),
                "clean_worst_markout_5000ms_delta": _safe_num(_nanmin(clean_markout)),
                "clean_best_markout_5000ms_delta": _safe_num(_nanmax(clean_markout)),
                "clean_markout_win_rate": _safe_num(
                    _rate(
                        sum(1 for value in clean_markout if _finite(value) and value >= 0.0),
                        sum(1 for value in clean_markout if _finite(value)),
                    )
                ),
                "all_sample_markout_win_rate": _safe_num(
                    _rate(
                        sum(1 for value in all_markout if _finite(value) and value >= 0.0),
                        sum(1 for value in all_markout if _finite(value)),
                    )
                ),
                "clean_median_fast_cancel_churn_delta": _safe_num(_nanmedian(clean_churn)),
                "clean_median_fill_after_cancel_delta": _safe_num(_nanmedian(clean_fac)),
                "clean_median_spread_capture_delta": _safe_num(_nanmedian(clean_spread)),
                "verdict": verdict,
                "verdict_reason": reason,
                "effect_type": _effect_type(verdict, clean_churn, clean_markout, clean_fill),
            }
        )
    return out


def _effect_type(verdict: str, churn: list[float], markout: list[float], fill: list[float]) -> str:
    median_churn = _nanmedian(churn)
    median_markout = _nanmedian(markout)
    median_fill = _nanmedian(fill)
    if verdict == "too_conservative_fill_loss" or (_finite(median_fill) and median_fill < MAX_FILL_RATE_LOSS_FOR_PROMISING):
        return "simple_fill_suppression"
    if _finite(median_churn) and median_churn < 0.0 and (not _finite(median_markout) or median_markout >= 0.0):
        return "churn_reduction"
    if _finite(median_markout) and median_markout > 0.0:
        return "toxic_fill_reduction_proxy"
    return "mixed_or_insufficient"


def _recommendation_lines(stability: list[dict[str, Any]], coverage: list[dict[str, Any]]) -> list[str]:
    verdict_counts: dict[str, int] = {}
    for row in stability:
        verdict_counts[row["verdict"]] = verdict_counts.get(row["verdict"], 0) + 1
    promising = [row for row in stability if row["verdict"] == "sweep_seed_promising"]
    low_fill = [row for row in stability if row["verdict"] == "stable_but_low_fill"]
    rejected = [row for row in stability if row["verdict"] == "reject"]
    lines = [
        "# 0526T004 Narrow Min-Move Parameter Sweep Recommendations",
        "",
        "## Boundary",
        "",
        "- Mode: read-only offline parameter sweep.",
        "- Candidate: `min_move_quote_age_churn_guard` only.",
        "- Seed scope: `inventory_state=large_skew_or_low_score` and `latency_stale_age=stale_latency_medium` only.",
        "- No live, default-on, guard relaxation, strategy behavior change, tiny-live design, or promotion was performed.",
        "- Net PnL proxy remains diagnostic only.",
        "",
        "## Verdict Counts",
        "",
    ]
    for verdict in (
        "sweep_seed_promising",
        "stable_but_low_fill",
        "churn_only_no_execution_benefit",
        "too_conservative_fill_loss",
        "caveated_only",
        "reject",
        "not_decisionable",
    ):
        lines.append(f"- `{verdict}`: `{verdict_counts.get(verdict, 0)}`")
    lines.extend(["", "## Promising Parameters", ""])
    if promising:
        for row in promising[:10]:
            lines.append(
                "- "
                f"`{row['seed_slice']}` `{row['param_hash']}` "
                f"min_move={row['min_move_ticks']} quote_age={row['min_quote_age_ms']} "
                f"churn_window={row['churn_window_ms']} max_readd={row['max_readd_count_in_window']} "
                f"stale_guard={row['stale_latency_guard_ms']} effect={row['effect_type']}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Low-Fill Parameters", ""])
    if low_fill:
        for row in low_fill[:10]:
            lines.append(f"- `{row['seed_slice']}` `{row['param_hash']}`: {row['verdict_reason']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Rejected Parameters", ""])
    if rejected:
        for row in rejected[:10]:
            lines.append(f"- `{row['seed_slice']}` `{row['param_hash']}`: {row['verdict_reason']}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Seed Coverage",
            "",
        ]
    )
    for row in coverage:
        lines.append(
            f"- `{row['sample_id']}` `{row['seed_slice']}`: "
            f"decisions `{row['seed_decision_rows']}`, submits `{row['seed_submit_orders']}`, fills `{row['seed_filled_orders']}`"
        )
    lines.extend(
        [
            "",
            "## Next-Step Interpretation",
            "",
            "- If QA accepts this runner, controller may decide between a narrower implementation-design task for the best parameter region or more strict-clean active sampling.",
            "- This result still does not authorize Step 10 tiny-live design unless existing Step 9C hard gates are explicitly satisfied and QA accepts that interpretation.",
        ]
    )
    return lines


def run_min_move_parameter_sweep(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    caveated_sample_ids: set[str] | None = None,
    smoke: bool = False,
    workers: int = 1,
    chunk_size: int = 50,
) -> dict[str, Any]:
    caveated = set(caveated_sample_ids or set())
    params = build_parameter_grid(smoke=smoke)
    valid_eval_units: list[tuple[Path, str, list[ParameterSet], str]] = []
    for run_dir in run_dirs:
        for seed_slice in SEED_SLICES:
            slice_params = [param for param in params if _parameter_applies_to_slice(param, seed_slice)]
            for index, chunk in enumerate(_chunked(slice_params, max(1, chunk_size))):
                worker_id = f"{run_dir.name}:{seed_slice}:{index:04d}"
                valid_eval_units.append((run_dir, seed_slice, chunk, worker_id))

    metric_rows: list[dict[str, Any]] = []
    shard_dir = output_dir / "sweep_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    if workers > 1 and len(valid_eval_units) > 1:
        max_workers = max(1, min(workers, os.cpu_count() or 1))
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    _run_shard,
                    run_dir=run_dir,
                    seed_slice=seed_slice,
                    params=chunk,
                    worker_id=worker_id,
                    caveated_sample_ids=caveated,
                ): (seed_slice, worker_id)
                for run_dir, seed_slice, chunk, worker_id in valid_eval_units
            }
            for future in as_completed(futures):
                seed_slice, worker_id = futures[future]
                rows = future.result()
                metric_rows.extend(rows)
                _write_shard(shard_dir / f"{worker_id.replace(':', '__')}.csv", rows)
    else:
        for run_dir, seed_slice, chunk, worker_id in valid_eval_units:
            rows = _run_shard(
                run_dir=run_dir,
                seed_slice=seed_slice,
                params=chunk,
                worker_id=worker_id,
                caveated_sample_ids=caveated,
            )
            metric_rows.extend(rows)
            _write_shard(shard_dir / f"{worker_id.replace(':', '__')}.csv", rows)

    metric_rows.sort(key=lambda row: (row["sample_id"], row["seed_slice"], row["param_hash"]))
    parameter_rows = [_parameter_row(param) for param in params]
    stability = _stability_rows(metric_rows, caveated)
    stability.sort(key=lambda row: (row["seed_slice"], row["verdict"], row["param_hash"]))
    coverage: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        coverage.extend(_load_context_for_coverage(run_dir, caveated))
    coverage.sort(key=lambda row: (row["sample_id"], row["seed_slice"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "parameter_grid.csv", parameter_rows)
    _write_json(output_dir / "parameter_grid.json", parameter_rows)
    _write_csv(output_dir / "sweep_metrics.csv", metric_rows)
    _write_csv(output_dir / "sweep_stability_summary.csv", stability)
    _write_json(output_dir / "sweep_stability_summary.json", stability)
    _write_csv(output_dir / "seed_slice_coverage.csv", coverage)
    (output_dir / "candidate_recommendations.md").write_text(
        "\n".join(_recommendation_lines(stability, coverage)) + "\n",
        encoding="utf-8",
    )
    summary = {
        "task_id": TASK_ID,
        "runner_mode": RUNNER_MODE,
        "generated_at": _generated_at(),
        "smoke": smoke,
        "sample_ids": [path.name for path in run_dirs],
        "caveated_sample_ids": sorted(caveated),
        "candidate_id": "min_move_quote_age_churn_guard",
        "seed_slices": list(SEED_SLICES),
        "parameter_set_count": len(parameter_rows),
        "seed_slice_evaluation_count": len({(row["seed_slice"], row["param_hash"]) for row in metric_rows}),
        "metric_row_count": len(metric_rows),
        "worker_count_requested": workers,
        "chunk_size": chunk_size,
        "verdict_counts": {
            verdict: sum(1 for row in stability if row["verdict"] == verdict)
            for verdict in sorted({row["verdict"] for row in stability})
        },
        "promising_count": sum(1 for row in stability if row["verdict"] == "sweep_seed_promising"),
        "not_authorized": ["live", "default-on", "guard relaxation", "strategy behavior change", "tiny-live", "promotion"],
    }
    _write_json(output_dir / "run_manifest.json", summary)
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=None, help="local_live_analysis/<run_id> to include")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--caveated-sample-id", action="append", default=sorted(DEFAULT_CAVEATED_SAMPLE_IDS))
    parser.add_argument("--smoke", action="store_true", help="run a tiny deterministic grid")
    parser.add_argument("--workers", type=int, default=1, help="parallel workers for shard execution")
    parser.add_argument("--chunk-size", type=int, default=50, help="parameter sets per shard")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dirs = args.run_dir or DEFAULT_RUN_DIRS
    run_min_move_parameter_sweep(
        run_dirs=run_dirs,
        output_dir=args.output_dir,
        caveated_sample_ids=set(args.caveated_sample_id),
        smoke=args.smoke,
        workers=args.workers,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
