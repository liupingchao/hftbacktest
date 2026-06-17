#!/usr/bin/env python3
"""Read-only Hyperliquid tiny-live signal / quote replay.

This task-scoped runner uses local public/read-only artifacts only. It never
reads credentials, calls private endpoints, queries accounts, places orders,
cancels orders, amends orders, starts a live bot, or claims PnL / fills /
maker viability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0617T005"
FINAL_RECOMMENDATION = "hyperliquid_tiny_live_signal_quote_replay_ready_for_qa"
DEFAULT_PRICING_ROWS = PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_pricing_signal_0601T005" / "pricing_signal_rows.csv"
DEFAULT_ROW_LEVEL = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_generator_0609T008" / "row_level_read_only_cases.csv"
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_generator_0609T008" / "source_artifact_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_signal_quote_replay_0617T005"
TICK_SIZE = 0.1
ORDER_SIZE_BTC = 0.01
MAX_POSITION_BTC = 0.04
THRESHOLDS_TICKS = [10, 20, 30, 40, 50, 75, 100]
PERSISTENCE_COUNTS = [1, 2, 3]
LOCAL_ANALYSIS_MARKER = "local_live_analysis"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: float | None, places: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{places}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _side(basis_ticks: float, threshold: float) -> str:
    if basis_ticks >= threshold:
        return "buy"
    if basis_ticks <= -threshold:
        return "sell"
    return "none"


def _sample_row(row: dict[str, str]) -> str:
    return row.get("sample_id") or row.get("source_sample_id") or "unknown_sample"


def _resolve_local_artifact_path(raw_path: str) -> Path | None:
    if not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.exists():
        return path
    parts = path.parts
    if LOCAL_ANALYSIS_MARKER in parts:
        marker_index = parts.index(LOCAL_ANALYSIS_MARKER)
        relocated = PROJECT_ROOT.joinpath(*parts[marker_index:])
        if relocated.exists():
            return relocated
    return None


def _pricing_paths_from_manifest(source_manifest_rows: list[dict[str, str]], primary_pricing_rows: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    primary = primary_pricing_rows.resolve()
    if primary.exists():
        paths.append(primary)
        seen.add(primary)
    for row in source_manifest_rows:
        resolved = _resolve_local_artifact_path(row["source_artifact_path"])
        if resolved is None:
            continue
        real = resolved.resolve()
        if real not in seen:
            paths.append(real)
            seen.add(real)
    return paths


def _read_pricing_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        for row in _read_csv(path):
            row = dict(row)
            row["replay_input_path"] = str(path)
            rows.append(row)
    return rows


def _primary(row: dict[str, str]) -> bool:
    return (
        row.get("joined_row_quality") == "primary_usable"
        and row.get("label_row_quality") == "primary_label_available"
        and row.get("context_hyperliquid_context_quality") == "primary_usable"
    )


def _fresh(row: dict[str, str]) -> bool:
    join_bucket = row.get("context_hyperliquid_join_age_bucket", "")
    source_age = _float(row, "binance_source_age_ms")
    return join_bucket == "fresh_0_50ms" and source_age is not None and source_age <= 50


def _replay_pricing_rows(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    # De-duplicate horizon-expanded pricing rows at the decision row level.
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (_sample_row(row), row.get("source_row_index") or row.get("hyperliquid_decision_ts", ""))
        if key not in unique or row.get("horizon_ms") == "1000":
            unique[key] = row
    ordered = sorted(unique.values(), key=lambda item: (_sample_row(item), int(item.get("hyperliquid_decision_ts") or 0)))

    threshold_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    sample_rows: dict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)

    for threshold in THRESHOLDS_TICKS:
        for persistence in PERSISTENCE_COUNTS:
            last_sample = ""
            last_side = "none"
            run_length = 0
            position_by_sample: defaultdict[str, float] = defaultdict(float)
            counters: Counter[str] = Counter()
            distance_values: list[float] = []
            for row in ordered:
                sample_id = _sample_row(row)
                if sample_id != last_sample:
                    last_side = "none"
                    run_length = 0
                    last_sample = sample_id
                basis_ticks = _float(row, "context_basis_mid_ticks")
                mid = _float(row, "context_hyperliquid_mid_px")
                spread_ticks = _float(row, "context_hyperliquid_spread_ticks")
                if basis_ticks is None:
                    counters["reject_missing_basis"] += 1
                    continue
                raw_side = _side(basis_ticks, threshold)
                if raw_side == "none":
                    counters["no_signal"] += 1
                    last_side = "none"
                    run_length = 0
                    continue
                if raw_side == last_side:
                    run_length += 1
                else:
                    last_side = raw_side
                    run_length = 1
                if run_length < persistence:
                    counters["reject_persistence"] += 1
                    continue
                if not _primary(row):
                    counters["reject_quality"] += 1
                    continue
                if not _fresh(row):
                    counters["reject_stale_or_data_gap"] += 1
                    continue
                if mid is None or spread_ticks is None or spread_ticks <= 0:
                    counters["reject_missing_book"] += 1
                    continue

                best_bid = mid - (spread_ticks * TICK_SIZE / 2)
                best_ask = mid + (spread_ticks * TICK_SIZE / 2)
                if raw_side == "buy":
                    quote = best_bid
                    quote_distance_ticks = (mid - quote) / TICK_SIZE
                    crossing = quote >= best_ask
                    projected_position = position_by_sample[sample_id] + ORDER_SIZE_BTC
                else:
                    quote = best_ask
                    quote_distance_ticks = (quote - mid) / TICK_SIZE
                    crossing = quote <= best_bid
                    projected_position = position_by_sample[sample_id] - ORDER_SIZE_BTC
                distance_values.append(quote_distance_ticks)

                if crossing:
                    counters["reject_crossing_post_only"] += 1
                    continue
                reduce_only = abs(position_by_sample[sample_id]) >= MAX_POSITION_BTC - ORDER_SIZE_BTC
                if abs(projected_position) > MAX_POSITION_BTC:
                    counters["cap_reduce_side_only"] += 1
                    continue

                position_by_sample[sample_id] = projected_position
                counters[f"intent_{raw_side}"] += 1
                sample_rows[(sample_id, threshold, persistence)][f"intent_{raw_side}"] += 1
                if len(audit_rows) < 200:
                    audit_rows.append(
                        {
                            "sample_id": sample_id,
                            "source_row_index": row.get("source_row_index", ""),
                            "hyperliquid_decision_ts": row.get("hyperliquid_decision_ts", ""),
                            "threshold_ticks": threshold,
                            "persistence_count": persistence,
                            "basis_mid_ticks": _fmt(basis_ticks),
                            "side_intent": raw_side,
                            "quote_price": _fmt(quote),
                            "quote_distance_ticks": _fmt(quote_distance_ticks),
                            "post_only": "true",
                            "crossing": str(crossing).lower(),
                            "size_btc": _fmt(ORDER_SIZE_BTC, 4),
                            "position_after_btc": _fmt(position_by_sample[sample_id], 4),
                            "cap_check": "pass",
                            "cancel_reason": "",
                        }
                    )

            threshold_rows.append(
                {
                    "threshold_ticks": threshold,
                    "persistence_count": persistence,
                    "rows_evaluated": len(ordered),
                    "intent_buy": counters["intent_buy"],
                    "intent_sell": counters["intent_sell"],
                    "no_signal": counters["no_signal"],
                    "reject_persistence": counters["reject_persistence"],
                    "reject_quality": counters["reject_quality"],
                    "reject_stale_or_data_gap": counters["reject_stale_or_data_gap"],
                    "reject_missing_book": counters["reject_missing_book"],
                    "reject_crossing_post_only": counters["reject_crossing_post_only"],
                    "cap_reduce_side_only": counters["cap_reduce_side_only"],
                    "quote_distance_ticks_min": _fmt(min(distance_values) if distance_values else None),
                    "quote_distance_ticks_mean": _fmt(sum(distance_values) / len(distance_values) if distance_values else None),
                    "quote_distance_ticks_max": _fmt(max(distance_values) if distance_values else None),
                }
            )

    sample_summary = [
        {
            "sample_id": sample,
            "threshold_ticks": threshold,
            "persistence_count": persistence,
            "intent_buy": counts["intent_buy"],
            "intent_sell": counts["intent_sell"],
        }
        for (sample, threshold, persistence), counts in sorted(sample_rows.items())
    ]
    return threshold_rows, sample_summary, audit_rows


def _row_level_calibration(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_sample: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = _float(row, "context_basis_mid_ticks")
        if value is not None:
            by_sample[_sample_row(row)].append(value)
    for sample_id, values in sorted(by_sample.items()):
        positives = [value for value in values if value > 0]
        negatives = [value for value in values if value < 0]
        abs_values = sorted(abs(value) for value in values)

        def percentile(q: float) -> float:
            if not abs_values:
                return 0.0
            idx = min(int((len(abs_values) - 1) * q), len(abs_values) - 1)
            return abs_values[idx]

        result.append(
            {
                "sample_id": sample_id,
                "row_level_rows": len(values),
                "positive_rows": len(positives),
                "negative_rows": len(negatives),
                "abs_basis_ticks_p50": _fmt(percentile(0.50)),
                "abs_basis_ticks_p75": _fmt(percentile(0.75)),
                "abs_basis_ticks_p90": _fmt(percentile(0.90)),
                "abs_basis_ticks_p95": _fmt(percentile(0.95)),
                "calibration_use": "basis_distribution_cross_check_full_quote_replay_available",
            }
        )
    return result


def _calibration_summary(threshold_rows: list[dict[str, Any]], sample_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    coverage: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)
    for row in sample_rows:
        threshold = int(row["threshold_ticks"])
        persistence = int(row["persistence_count"])
        buy = int(row["intent_buy"])
        sell = int(row["intent_sell"])
        if buy or sell:
            coverage[(threshold, persistence)]["samples_with_intent"] += 1
        if buy:
            coverage[(threshold, persistence)]["samples_with_buy"] += 1
        if sell:
            coverage[(threshold, persistence)]["samples_with_sell"] += 1

    summaries: list[dict[str, Any]] = []
    for row in threshold_rows:
        threshold = int(row["threshold_ticks"])
        persistence = int(row["persistence_count"])
        rows_evaluated = int(row["rows_evaluated"])
        intent_buy = int(row["intent_buy"])
        intent_sell = int(row["intent_sell"])
        total_intents = intent_buy + intent_sell
        label = ""
        if threshold == 75 and persistence == 2:
            label = "primary_candidate"
        elif threshold == 75 and persistence == 3:
            label = "stricter_low_activity_fallback"

        def rate(field: str) -> float:
            return int(row[field]) / rows_evaluated if rows_evaluated else 0.0

        summaries.append(
            {
                "threshold_ticks": threshold,
                "persistence_count": persistence,
                "candidate_label": label,
                "rows_evaluated": rows_evaluated,
                "total_intents": total_intents,
                "intent_rate": _fmt(total_intents / rows_evaluated if rows_evaluated else 0.0, 6),
                "intent_buy": intent_buy,
                "intent_sell": intent_sell,
                "no_signal_rate": _fmt(rate("no_signal"), 6),
                "stale_or_data_gap_rate": _fmt(rate("reject_stale_or_data_gap"), 6),
                "cap_reduce_side_only_rate": _fmt(rate("cap_reduce_side_only"), 6),
                "samples_with_intent": coverage[(threshold, persistence)]["samples_with_intent"],
                "samples_with_buy": coverage[(threshold, persistence)]["samples_with_buy"],
                "samples_with_sell": coverage[(threshold, persistence)]["samples_with_sell"],
            }
        )
    return summaries


def _source_availability(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    availability = []
    for row in rows:
        source_path = Path(row["source_artifact_path"])
        resolved = _resolve_local_artifact_path(row["source_artifact_path"])
        manifest_exists = source_path.exists()
        local_direct = resolved is not None
        availability.append(
            {
                "sample_id": row["sample_id"],
                "source_row_count": row["source_row_count"],
                "manifest_source_path": row["source_artifact_path"],
                "manifest_path_exists_on_this_host": str(manifest_exists).lower(),
                "local_direct_file_available": str(local_direct).lower(),
                "resolved_pricing_signal_path": str(resolved) if resolved is not None else "",
                "replay_source_used": "pricing_signal_rows" if resolved is not None else "row_level_read_only_cases",
                "availability_note": "available_for_full_quote_replay" if resolved is not None else "pricing_signal_rows not present on this host",
            }
        )
    return availability


def run(pricing_rows_path: Path, row_level_path: Path, source_manifest_path: Path, output_dir: Path) -> None:
    row_level_rows = _read_csv(row_level_path)
    source_manifest_rows = _read_csv(source_manifest_path)
    pricing_paths = _pricing_paths_from_manifest(source_manifest_rows, pricing_rows_path)
    pricing_rows = _read_pricing_rows(pricing_paths)

    threshold_rows, sample_summary, audit_rows = _replay_pricing_rows(pricing_rows)
    calibration_summary = _calibration_summary(threshold_rows, sample_summary)
    row_level_summary = _row_level_calibration(row_level_rows)
    source_availability = _source_availability(source_manifest_rows)

    _write_csv(
        output_dir / "threshold_sensitivity.csv",
        threshold_rows,
        [
            "threshold_ticks",
            "persistence_count",
            "rows_evaluated",
            "intent_buy",
            "intent_sell",
            "no_signal",
            "reject_persistence",
            "reject_quality",
            "reject_stale_or_data_gap",
            "reject_missing_book",
            "reject_crossing_post_only",
            "cap_reduce_side_only",
            "quote_distance_ticks_min",
            "quote_distance_ticks_mean",
            "quote_distance_ticks_max",
        ],
    )
    _write_csv(output_dir / "sample_trigger_summary.csv", sample_summary, ["sample_id", "threshold_ticks", "persistence_count", "intent_buy", "intent_sell"])
    _write_csv(
        output_dir / "calibration_summary.csv",
        calibration_summary,
        [
            "threshold_ticks",
            "persistence_count",
            "candidate_label",
            "rows_evaluated",
            "total_intents",
            "intent_rate",
            "intent_buy",
            "intent_sell",
            "no_signal_rate",
            "stale_or_data_gap_rate",
            "cap_reduce_side_only_rate",
            "samples_with_intent",
            "samples_with_buy",
            "samples_with_sell",
        ],
    )
    _write_csv(
        output_dir / "row_level_basis_distribution_by_sample.csv",
        row_level_summary,
        [
            "sample_id",
            "row_level_rows",
            "positive_rows",
            "negative_rows",
            "abs_basis_ticks_p50",
            "abs_basis_ticks_p75",
            "abs_basis_ticks_p90",
            "abs_basis_ticks_p95",
            "calibration_use",
        ],
    )
    _write_csv(
        output_dir / "source_availability.csv",
        source_availability,
        [
            "sample_id",
            "source_row_count",
            "manifest_source_path",
            "manifest_path_exists_on_this_host",
            "local_direct_file_available",
            "resolved_pricing_signal_path",
            "replay_source_used",
            "availability_note",
        ],
    )
    _write_csv(
        output_dir / "row_level_audit_sample.csv",
        audit_rows,
        [
            "sample_id",
            "source_row_index",
            "hyperliquid_decision_ts",
            "threshold_ticks",
            "persistence_count",
            "basis_mid_ticks",
            "side_intent",
            "quote_price",
            "quote_distance_ticks",
            "post_only",
            "crossing",
            "size_btc",
            "position_after_btc",
            "cap_check",
            "cancel_reason",
        ],
    )
    _write_json(
        output_dir / "replay_manifest.json",
        {
            "boundary_flags": {
                "account_query_called": False,
                "credentials_read": False,
                "live_bot_started": False,
                "order_amendment_called": False,
                "order_cancellation_called": False,
                "order_placement_called": False,
                "private_endpoint_called": False,
                "pnl_claimed": False,
                "real_fill_claimed": False,
            },
            "calibration_interpretation": {
                "primary_candidate": "threshold_75_ticks_persistence_2",
                "stricter_low_activity_fallback": "threshold_75_ticks_persistence_3",
                "candidate_status": "read_only_replay_candidate_requires_qa_and_controller_ratification",
                "live_execution_authorized": False,
            },
            "final_recommendation": FINAL_RECOMMENDATION,
            "git_commit": _git_commit(),
            "order_size_btc": ORDER_SIZE_BTC,
            "max_position_btc": MAX_POSITION_BTC,
            "pricing_rows_input": str(pricing_rows_path),
            "pricing_rows_inputs": [str(path) for path in pricing_paths],
            "pricing_rows_input_count": len(pricing_paths),
            "pricing_rows_replayed": len(pricing_rows),
            "pricing_decision_rows_evaluated": threshold_rows[0]["rows_evaluated"] if threshold_rows else 0,
            "row_level_input": str(row_level_path),
            "row_level_rows": len(row_level_rows),
            "source_manifest_input": str(source_manifest_path),
            "source_manifest_samples": len(source_manifest_rows),
            "task_id": TASK_ID,
            "threshold_grid_ticks": THRESHOLDS_TICKS,
            "persistence_grid": PERSISTENCE_COUNTS,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pricing-rows", type=Path, default=DEFAULT_PRICING_ROWS)
    parser.add_argument("--row-level", type=Path, default=DEFAULT_ROW_LEVEL)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.pricing_rows.resolve(), args.row_level.resolve(), args.source_manifest.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
