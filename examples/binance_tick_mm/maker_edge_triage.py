#!/usr/bin/env python3
"""Read-only maker edge family triage over current-format samples.

This runner ranks broad maker-edge research families without parameter search or
strategy changes. It uses existing Stage 5 execution labels as the low-cost
evidence layer and treats caveated samples as sensitivity only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from quote_adjustment_replay import _bool, _float, _generated_at, _read_csv, _safe_num, _write_csv, _write_json


TASK_ID = "0526T005"
RUNNER_MODE = "stage9h_maker_edge_family_triage"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/stage9h_maker_edge_triage_0526T005")
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
FAMILIES = ("fair_price", "reservation", "inventory", "quote_distance", "size_side")


@dataclass(frozen=True)
class TriageRow:
    sample_id: str
    is_caveated: bool
    family: str
    bucket: str
    rows: int
    fills: int
    fill_rate: float
    markout_5000ms_mean: float
    spread_capture_mean: float
    fill_after_cancel_rate: float
    inventory_increasing_fill_rate: float
    inventory_reducing_fill_rate: float


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


def _stage5_path(run_dir: Path, name: str) -> Path:
    return run_dir / "stage5_execution_outcome_labels_0514T005" / name


def _labels(run_dir: Path) -> list[dict[str, str]]:
    path = _stage5_path(run_dir, "execution_outcome_labels.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    return _read_csv(path)


def _markouts_by_order(run_dir: Path) -> dict[str, list[dict[str, str]]]:
    path = _stage5_path(run_dir, "fill_markout_labels.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(path):
        order_id = str(row.get("order_id", "")).strip()
        if order_id:
            out[order_id].append(row)
    return out


def _filled(row: dict[str, str]) -> bool:
    return _float(row.get("fill_count"), 0.0) > 0.0 or _bool(row.get("full_fill"))


def _order_side(row: dict[str, str]) -> str:
    return str(row.get("order_side", "")).strip().lower()


def _side_sign(row: dict[str, str]) -> int:
    side = _order_side(row)
    if side == "buy":
        return 1
    if side == "sell":
        return -1
    return int(_float(row.get("side_sign"), 0.0))


def _markout_5s(row: dict[str, str], markouts_by_order: dict[str, list[dict[str, str]]]) -> float:
    order_id = str(row.get("order_id", "")).strip()
    for markout in markouts_by_order.get(order_id, []):
        if int(_float(markout.get("horizon_ms"), -1)) == 5000 and _bool(markout.get("horizon_observable")):
            return _float(markout.get("side_adjusted_markout_ticks"))
    return _float(row.get("fill_markout_5000ms_ticks"))


def _spread_capture(row: dict[str, str], markouts_by_order: dict[str, list[dict[str, str]]]) -> float:
    order_id = str(row.get("order_id", "")).strip()
    for markout in markouts_by_order.get(order_id, []):
        if int(_float(markout.get("horizon_ms"), -1)) == 5000 and _bool(markout.get("horizon_observable")):
            return _float(markout.get("realized_spread_proxy_ticks"))
    return _float(row.get("realized_spread_proxy_ticks"))


def _metric_for_rows(rows: list[dict[str, str]], markouts_by_order: dict[str, list[dict[str, str]]]) -> dict[str, Any]:
    filled_rows = [row for row in rows if _filled(row)]
    markouts = [_markout_5s(row, markouts_by_order) for row in filled_rows]
    spreads = [_spread_capture(row, markouts_by_order) for row in filled_rows]
    return {
        "rows": len(rows),
        "fills": len(filled_rows),
        "fill_rate": _safe_num(_rate(len(filled_rows), len(rows))),
        "markout_5000ms_mean": _safe_num(_mean(markouts)),
        "spread_capture_mean": _safe_num(_mean(spreads)),
        "fill_after_cancel_rate": _safe_num(_rate(sum(1 for row in filled_rows if _bool(row.get("fill_after_cancel_request"))), len(filled_rows))),
        "inventory_increasing_fill_rate": _safe_num(_rate(sum(1 for row in filled_rows if _bool(row.get("inventory_increasing_fill"))), len(filled_rows))),
        "inventory_reducing_fill_rate": _safe_num(_rate(sum(1 for row in filled_rows if _bool(row.get("inventory_reducing_fill"))), len(filled_rows))),
    }


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def _bucket_by_quantiles(value: float, low: float, high: float, prefix: str) -> str:
    if not _finite(value):
        return f"{prefix}_unknown"
    if value <= low:
        return f"{prefix}_low"
    if value <= high:
        return f"{prefix}_medium"
    return f"{prefix}_high"


def _quantiles(values: Iterable[float]) -> tuple[float, float]:
    finite = sorted(float(value) for value in values if _finite(value))
    if len(finite) < 3:
        return math.nan, math.nan
    return finite[len(finite) // 3], finite[(len(finite) * 2) // 3]


def _fair_price_bucket(row: dict[str, str], low: float, high: float) -> str:
    edge = abs(_float(row.get("edge_vs_fair_ticks")))
    return _bucket_by_quantiles(edge, low, high, "fair_abs_edge")


def _reservation_bucket(row: dict[str, str], low: float, high: float) -> str:
    edge = abs(_float(row.get("edge_vs_reservation_ticks")))
    return _bucket_by_quantiles(edge, low, high, "reservation_abs_edge")


def _inventory_bucket(row: dict[str, str]) -> str:
    position = abs(_float(row.get("position_before_submit"), 0.0))
    score = _float(row.get("inventory_score"), 1.0)
    if position <= 0.0005:
        return "flat"
    if position >= 0.0015 or score <= 0.35:
        return "large_skew_or_low_score"
    if score <= 0.5:
        return "mild_skew_recovery_zone"
    return "mild_skew"


def _quote_distance_bucket(row: dict[str, str]) -> str:
    placement = str(row.get("placement_bucket", "")).strip()
    if placement:
        return placement
    distance = abs(_float(row.get("distance_to_bbo_ticks")))
    if not _finite(distance):
        return "quote_distance_unknown"
    if distance <= 0.5:
        return "touch"
    if distance <= 2.0:
        return "one_to_two_ticks"
    return "step_back_gt1"


def _size_side_bucket(row: dict[str, str]) -> str:
    side = _order_side(row)
    position = _float(row.get("position_before_submit"), 0.0)
    if side == "buy" and position < -0.0005:
        return "reduce_side_buy"
    if side == "sell" and position > 0.0005:
        return "reduce_side_sell"
    if side == "buy" and position > 0.0005:
        return "add_side_buy"
    if side == "sell" and position < -0.0005:
        return "add_side_sell"
    return f"flat_{side or 'unknown'}"


def _sample_rows(run_dir: Path, caveated: set[str]) -> list[dict[str, Any]]:
    labels = _labels(run_dir)
    markouts_by_order = _markouts_by_order(run_dir)
    fair_low, fair_high = _quantiles(abs(_float(row.get("edge_vs_fair_ticks"))) for row in labels)
    reservation_low, reservation_high = _quantiles(abs(_float(row.get("edge_vs_reservation_ticks"))) for row in labels)
    family_to_bucketed: dict[str, dict[str, list[dict[str, str]]]] = {family: defaultdict(list) for family in FAMILIES}
    for row in labels:
        family_to_bucketed["fair_price"][_fair_price_bucket(row, fair_low, fair_high)].append(row)
        family_to_bucketed["reservation"][_reservation_bucket(row, reservation_low, reservation_high)].append(row)
        family_to_bucketed["inventory"][_inventory_bucket(row)].append(row)
        family_to_bucketed["quote_distance"][_quote_distance_bucket(row)].append(row)
        family_to_bucketed["size_side"][_size_side_bucket(row)].append(row)

    out: list[dict[str, Any]] = []
    for family, buckets in family_to_bucketed.items():
        for bucket, bucket_rows in sorted(buckets.items()):
            metrics = _metric_for_rows(bucket_rows, markouts_by_order)
            out.append(
                {
                    "sample_id": run_dir.name,
                    "is_caveated_sample": "true" if run_dir.name in caveated else "false",
                    "family": family,
                    "bucket": bucket,
                    **metrics,
                }
            )
    return out


def _family_score(rows: list[dict[str, Any]], caveated: set[str]) -> dict[str, Any]:
    clean = [row for row in rows if row["sample_id"] not in caveated]
    caveated_rows = [row for row in rows if row["sample_id"] in caveated]
    clean_fills = sum(int(_float(row.get("fills"), 0.0)) for row in clean)
    clean_samples = len({row["sample_id"] for row in clean})
    bucket_count = len({row["bucket"] for row in clean})
    markouts = [_float(row.get("markout_5000ms_mean")) for row in clean]
    spreads = [_float(row.get("spread_capture_mean")) for row in clean]
    fill_rates = [_float(row.get("fill_rate")) for row in clean]
    inv_inc = [_float(row.get("inventory_increasing_fill_rate")) for row in clean]
    inv_red = [_float(row.get("inventory_reducing_fill_rate")) for row in clean]
    markout_range = _range(markouts)
    spread_range = _range(spreads)
    fill_rate_range = _range(fill_rates)
    inventory_direction_gap = _nanmedian(inv_red) - _nanmedian(inv_inc) if _finite(_nanmedian(inv_red)) and _finite(_nanmedian(inv_inc)) else math.nan
    caveated_markout_range = _range(_float(row.get("markout_5000ms_mean")) for row in caveated_rows)
    caveated_influence = ""
    if _finite(caveated_markout_range) and _finite(markout_range) and abs(caveated_markout_range) > abs(markout_range) * 2.0 + 1.0:
        caveated_influence = "high"
    elif caveated_rows:
        caveated_influence = "present"
    else:
        caveated_influence = "none"
    score = 0.0
    if clean_fills >= 100:
        score += 2.0
    elif clean_fills >= 50:
        score += 1.0
    if _finite(markout_range) and abs(markout_range) >= 5.0:
        score += 2.0
    elif _finite(markout_range) and abs(markout_range) >= 2.0:
        score += 1.0
    if _finite(spread_range) and abs(spread_range) >= 2.0:
        score += 1.0
    if _finite(fill_rate_range) and abs(fill_rate_range) >= 0.01:
        score += 1.0
    if _finite(inventory_direction_gap) and abs(inventory_direction_gap) >= 0.05:
        score += 1.0
    if caveated_influence == "high":
        score -= 1.0
    verdict, reason = _verdict(
        score=score,
        clean_fills=clean_fills,
        clean_samples=clean_samples,
        bucket_count=bucket_count,
        markout_range=markout_range,
        caveated_influence=caveated_influence,
    )
    return {
        "family": rows[0]["family"] if rows else "",
        "verdict": verdict,
        "score": _safe_num(score),
        "reason": reason,
        "clean_sample_count": clean_samples,
        "clean_filled_orders": clean_fills,
        "clean_bucket_count": bucket_count,
        "clean_markout_5000ms_range": _safe_num(markout_range),
        "clean_spread_capture_range": _safe_num(spread_range),
        "clean_fill_rate_range": _safe_num(fill_rate_range),
        "inventory_direction_gap": _safe_num(inventory_direction_gap),
        "caveated_influence": caveated_influence,
        "next_step": _next_step(verdict, rows[0]["family"] if rows else ""),
    }


def _range(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return max(finite) - min(finite) if finite else math.nan


def _verdict(
    *,
    score: float,
    clean_fills: int,
    clean_samples: int,
    bucket_count: int,
    markout_range: float,
    caveated_influence: str,
) -> tuple[str, str]:
    if clean_samples == 0 or bucket_count < 2:
        return "blocked_by_artifact_gap", "insufficient clean samples or buckets"
    if clean_fills < 50:
        return "needs_more_clean_fills", "clean filled-order mass is too low for family triage"
    if score >= 5.0 and caveated_influence != "high":
        return "strong_next_candidate", "clean evidence shows multi-metric separation"
    if score >= 3.0 and caveated_influence != "high":
        return "promising_needs_parameter_design", "clean evidence is directional enough for a focused design task"
    if _finite(markout_range) and abs(markout_range) < 1.0 and clean_fills >= 100:
        return "reject", "enough clean fills but weak markout separation"
    return "weak_or_mixed", "evidence is present but not strong enough for immediate parameter design"


def _next_step(verdict: str, family: str) -> str:
    if verdict in {"strong_next_candidate", "promising_needs_parameter_design"}:
        return f"create focused {family} design task"
    if verdict == "needs_more_clean_fills":
        return f"collect targeted clean fills for {family} only if controller prioritizes this family"
    if verdict == "reject":
        return "deprioritize"
    if verdict == "blocked_by_artifact_gap":
        return "repair artifact coverage before further analysis"
    return "keep as background context; do not parameter-search yet"


def _write_report(path: Path, family_rows: list[dict[str, Any]]) -> None:
    ranked = sorted(
        family_rows,
        key=lambda row: (
            -_float(row.get("score"), 0.0),
            row.get("verdict") not in {"strong_next_candidate", "promising_needs_parameter_design"},
            row.get("family", ""),
        ),
    )
    lines = [
        "# 0526T005 Maker Edge Family Triage",
        "",
        "## Boundary",
        "",
        "- Mode: read-only family triage.",
        "- No parameter search, strategy implementation, live, default-on, guard relaxation, or promotion.",
        "- Caveated samples are sensitivity only.",
        "",
        "## Ranking",
        "",
        "| rank | family | verdict | score | clean fills | reason | next step |",
        "|---:|---|---|---:|---:|---|---|",
    ]
    for idx, row in enumerate(ranked, start=1):
        lines.append(
            f"| {idx} | `{row['family']}` | `{row['verdict']}` | `{row['score']}` | "
            f"`{row['clean_filled_orders']}` | {row['reason']} | {row['next_step']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Families with `strong_next_candidate` or `promising_needs_parameter_design` can justify a focused design task.",
            "- `needs_more_clean_fills` means sample collection should be targeted to that family, not generic duration.",
            "- `weak_or_mixed` / `reject` should not enter parameter search by default.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_maker_edge_triage(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    caveated_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    caveated = set(caveated_sample_ids or set())
    metric_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        metric_rows.extend(_sample_rows(run_dir, caveated))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_family[row["family"]].append(row)
    family_summary = [_family_score(rows, caveated) for _family, rows in sorted(by_family.items())]
    family_summary.sort(key=lambda row: (-_float(row.get("score"), 0.0), row["family"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "family_bucket_metrics.csv", metric_rows)
    _write_csv(output_dir / "family_triage_summary.csv", family_summary)
    _write_json(output_dir / "family_triage_summary.json", family_summary)
    _write_report(output_dir / "maker_edge_triage_recommendations.md", family_summary)
    manifest = {
        "task_id": TASK_ID,
        "runner_mode": RUNNER_MODE,
        "generated_at": _generated_at(),
        "sample_ids": [path.name for path in run_dirs],
        "caveated_sample_ids": sorted(caveated),
        "families": list(FAMILIES),
        "top_family": family_summary[0]["family"] if family_summary else "",
        "top_verdict": family_summary[0]["verdict"] if family_summary else "",
        "verdict_counts": {
            verdict: sum(1 for row in family_summary if row["verdict"] == verdict)
            for verdict in sorted({row["verdict"] for row in family_summary})
        },
        "not_authorized": ["live", "default-on", "parameter search", "strategy behavior change", "tiny-live", "promotion"],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--caveated-sample-id", action="append", default=sorted(DEFAULT_CAVEATED_SAMPLE_IDS))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    run_maker_edge_triage(
        run_dirs=args.run_dir or DEFAULT_RUN_DIRS,
        output_dir=args.output_dir,
        caveated_sample_ids=set(args.caveated_sample_id),
    )


if __name__ == "__main__":
    main()
