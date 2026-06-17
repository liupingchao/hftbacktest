#!/usr/bin/env python3
"""Read-only optimistic Hyperliquid tiny-live PnL proxy.

This task-scoped runner consumes local public/read-only pricing-signal rows. It
estimates a theoretical upper bound under an explicit 100% theoretical maker
intent fill assumption. It does not read credentials, call private endpoints,
query accounts, place/cancel/amend orders, start a live bot, model fill
probability, or claim real PnL / fills / maker viability.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0617T006"
FINAL_RECOMMENDATION = "hyperliquid_tiny_live_optimistic_pnl_proxy_ready_for_qa"
OFFICIAL_SAMPLE_SET_ID = "canonical_7"
SAMPLE_SET_CLARIFICATION = "user_selected_canonical_7_as_official_sample_set_on_2026_06_17"
DEFAULT_REPLAY_MANIFEST = (
    PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_signal_quote_replay_0617T005" / "replay_manifest.json"
)
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_generator_0609T008" / "source_artifact_manifest.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006"
TICK_SIZE = 0.1
ORDER_SIZE_BTC = 0.01
PNL_USDC_PER_TICK = TICK_SIZE * ORDER_SIZE_BTC
THRESHOLDS_TICKS = [50, 75, 100]
PERSISTENCE_COUNTS = [1, 2, 3]
PRIMARY_THRESHOLD_TICKS = 75
PRIMARY_PERSISTENCE_COUNT = 2
FALLBACK_PERSISTENCE_COUNT = 3
PRIMARY_INTERPRETATION_HORIZON_MS = 1000
LOCAL_ANALYSIS_MARKER = "local_live_analysis"


@dataclass(frozen=True)
class SampleSet:
    sample_set_id: str
    paths: list[Path]
    compute_status: str
    reconciliation_status: str
    note: str


@dataclass
class Decision:
    sample_id: str
    source_row_index: str
    decision_ts: int
    context_row: dict[str, str]
    horizon_rows: list[dict[str, str]]


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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _float(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _int_value(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _fmt(value: float | None, places: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{places}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _sample_row(row: dict[str, str]) -> str:
    return row.get("sample_id") or row.get("source_sample_id") or "unknown_sample"


def _resolve_local_artifact_path(raw_path: str) -> Path | None:
    if not raw_path.strip():
        return None
    path = Path(raw_path)
    if path.exists():
        return path.resolve()
    parts = path.parts
    if LOCAL_ANALYSIS_MARKER in parts:
        marker_index = parts.index(LOCAL_ANALYSIS_MARKER)
        relocated = PROJECT_ROOT.joinpath(*parts[marker_index:])
        if relocated.exists():
            return relocated.resolve()
    return None


def _unique_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        real = path.resolve()
        if real not in seen:
            out.append(real)
            seen.add(real)
    return out


def _side(basis_ticks: float, threshold: float) -> str:
    if basis_ticks >= threshold:
        return "buy"
    if basis_ticks <= -threshold:
        return "sell"
    return "none"


def _primary(row: dict[str, str]) -> bool:
    return (
        row.get("joined_row_quality") == "primary_usable"
        and row.get("label_row_quality") == "primary_label_available"
        and row.get("context_hyperliquid_context_quality") == "primary_usable"
    )


def _fresh(row: dict[str, str]) -> bool:
    source_age = _float(row, "binance_source_age_ms")
    return row.get("context_hyperliquid_join_age_bucket") == "fresh_0_50ms" and source_age is not None and source_age <= 50


def _sample_sets(replay_manifest: dict[str, Any], source_manifest_rows: list[dict[str, str]]) -> tuple[list[SampleSet], list[dict[str, Any]]]:
    canonical_paths: list[Path] = []
    for row in source_manifest_rows:
        resolved = _resolve_local_artifact_path(row.get("source_artifact_path", ""))
        if resolved is not None:
            canonical_paths.append(resolved)
    replay_paths = [
        resolved
        for raw in replay_manifest.get("pricing_rows_inputs", [])
        if (resolved := _resolve_local_artifact_path(str(raw))) is not None
    ]
    canonical_paths = _unique_paths(canonical_paths)
    replay_paths = _unique_paths(replay_paths)

    reconciliation_rows = [
        {
            "sample_set_id": "requested_six",
            "requested_by_user": "true",
            "official_sample_set": "false",
            "requested_count": 6,
            "manifest_sample_count": "",
            "resolved_pricing_input_count": "",
            "compute_status": "not_computed",
            "reconciliation_status": "superseded_by_user_selected_canonical_7",
            "note": "The user clarified that canonical_7 is the formal sample-set policy, so the earlier exact six-sample wording no longer blocks QA.",
        },
        {
            "sample_set_id": "canonical_7",
            "requested_by_user": "true",
            "official_sample_set": "true",
            "requested_count": "",
            "manifest_sample_count": len(source_manifest_rows),
            "resolved_pricing_input_count": len(canonical_paths),
            "compute_status": "computed" if canonical_paths else "blocked",
            "reconciliation_status": "controller_selected_official_sample_set",
            "note": "Canonical event-mode samples from 0609T008 source_artifact_manifest.csv; selected by the user as the formal sample-set policy.",
        },
        {
            "sample_set_id": "0617T005_8_input",
            "requested_by_user": "false",
            "official_sample_set": "false",
            "requested_count": "",
            "manifest_sample_count": replay_manifest.get("pricing_rows_input_count", ""),
            "resolved_pricing_input_count": len(replay_paths),
            "compute_status": "computed" if replay_paths else "blocked",
            "reconciliation_status": "diagnostic_full_replay_set",
            "note": "All pricing_signal_rows inputs consumed by accepted 0617T005 replay.",
        },
    ]
    return (
        [
            SampleSet("requested_six", [], "not_computed", "superseded_by_user_selected_canonical_7", reconciliation_rows[0]["note"]),
            SampleSet("canonical_7", canonical_paths, reconciliation_rows[1]["compute_status"], "controller_selected_official_sample_set", reconciliation_rows[1]["note"]),
            SampleSet("0617T005_8_input", replay_paths, reconciliation_rows[2]["compute_status"], "diagnostic_full_replay_set", reconciliation_rows[2]["note"]),
        ],
        reconciliation_rows,
    )


def _load_sample_set_rows(sample_set: SampleSet) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    membership_rows: list[dict[str, Any]] = []
    for path in sample_set.paths:
        path_rows = _read_csv(path)
        sample_ids = sorted({_sample_row(row) for row in path_rows})
        for row in path_rows:
            next_row = dict(row)
            next_row["pricing_input_path"] = str(path)
            rows.append(next_row)
        membership_rows.append(
            {
                "sample_set_id": sample_set.sample_set_id,
                "pricing_signal_path": str(path),
                "pricing_row_count": len(path_rows),
                "sample_ids": "|".join(sample_ids),
                "sample_id_count": len(sample_ids),
                "membership_note": sample_set.note,
            }
        )
    return rows, membership_rows


def _decision_rows(rows: list[dict[str, str]]) -> list[Decision]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        sample_id = _sample_row(row)
        source_row_index = row.get("source_row_index") or row.get("hyperliquid_decision_ts", "")
        key = (sample_id, source_row_index)
        item = grouped.setdefault(
            key,
            {
                "sample_id": sample_id,
                "source_row_index": source_row_index,
                "decision_ts": _int_value(row.get("hyperliquid_decision_ts")) or 0,
                "context_row": row,
                "horizon_rows": [],
            },
        )
        horizon = _int_value(row.get("horizon_ms"))
        if horizon == PRIMARY_INTERPRETATION_HORIZON_MS:
            item["context_row"] = row
        item["horizon_rows"].append(row)
    decisions = [
        Decision(
            sample_id=str(item["sample_id"]),
            source_row_index=str(item["source_row_index"]),
            decision_ts=int(item["decision_ts"]),
            context_row=item["context_row"],
            horizon_rows=sorted(item["horizon_rows"], key=lambda row: _int_value(row.get("horizon_ms")) or -1),
        )
        for item in grouped.values()
    ]
    return sorted(decisions, key=lambda item: (item.sample_id, item.decision_ts, _int_value(item.source_row_index) or 0))


def _pnl_rows_for_grid(sample_set_id: str, decisions: list[Decision]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pnl_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    for threshold in THRESHOLDS_TICKS:
        for persistence in PERSISTENCE_COUNTS:
            last_sample = ""
            last_side = "none"
            run_length = 0
            counters: Counter[str] = Counter()
            for decision in decisions:
                row = decision.context_row
                if decision.sample_id != last_sample:
                    last_sample = decision.sample_id
                    last_side = "none"
                    run_length = 0
                basis_ticks = _float(row, "context_basis_mid_ticks")
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
                mid_px = _float(row, "context_hyperliquid_mid_px")
                spread_ticks = _float(row, "context_hyperliquid_spread_ticks")
                if mid_px is None or spread_ticks is None or spread_ticks <= 0:
                    counters["reject_missing_book"] += 1
                    continue
                quote_distance_ticks = spread_ticks / 2.0
                quote_price = mid_px - quote_distance_ticks * TICK_SIZE if raw_side == "buy" else mid_px + quote_distance_ticks * TICK_SIZE
                counters[f"eligible_{raw_side}"] += 1
                counters["eligible_decisions"] += 1
                for horizon_row in decision.horizon_rows:
                    future_move_ticks = _float(horizon_row, "hyperliquid_future_mid_move_ticks")
                    horizon_ms = _int_value(horizon_row.get("horizon_ms"))
                    if future_move_ticks is None or horizon_ms is None:
                        counters["reject_missing_future_label"] += 1
                        continue
                    pnl_ticks = (
                        future_move_ticks + quote_distance_ticks
                        if raw_side == "buy"
                        else -future_move_ticks + quote_distance_ticks
                    )
                    pnl_rows.append(
                        {
                            "sample_set_id": sample_set_id,
                            "sample_id": decision.sample_id,
                            "source_row_index": decision.source_row_index,
                            "hyperliquid_decision_ts": decision.decision_ts,
                            "threshold_ticks": threshold,
                            "persistence_count": persistence,
                            "quote_side": raw_side,
                            "horizon_ms": horizon_ms,
                            "basis_mid_ticks": basis_ticks,
                            "quote_price": quote_price,
                            "quote_distance_ticks": quote_distance_ticks,
                            "future_mid_move_ticks": future_move_ticks,
                            "optimistic_mid_pnl_ticks": pnl_ticks,
                            "optimistic_mid_pnl_usdc": pnl_ticks * PNL_USDC_PER_TICK,
                            "basis_future_mid_response_ticks": _float(horizon_row, "basis_future_mid_response_ticks"),
                            "assumption_set": "unconstrained_all_intents",
                            "formula_version": "future_mid_move_plus_half_spread_v1",
                        }
                    )
            eligibility_rows.append(
                {
                    "sample_set_id": sample_set_id,
                    "threshold_ticks": threshold,
                    "persistence_count": persistence,
                    "decision_rows_evaluated": len(decisions),
                    "eligible_decisions": counters["eligible_decisions"],
                    "eligible_buy": counters["eligible_buy"],
                    "eligible_sell": counters["eligible_sell"],
                    "no_signal": counters["no_signal"],
                    "reject_missing_basis": counters["reject_missing_basis"],
                    "reject_persistence": counters["reject_persistence"],
                    "reject_quality": counters["reject_quality"],
                    "reject_stale_or_data_gap": counters["reject_stale_or_data_gap"],
                    "reject_missing_book": counters["reject_missing_book"],
                    "reject_missing_future_label": counters["reject_missing_future_label"],
                }
            )
    return pnl_rows, eligibility_rows


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _summary_row(key: tuple[Any, ...], rows: list[dict[str, Any]], key_fields: list[str]) -> dict[str, Any]:
    values_ticks = [float(row["optimistic_mid_pnl_ticks"]) for row in rows]
    values_usdc = [float(row["optimistic_mid_pnl_usdc"]) for row in rows]
    out = {field: value for field, value in zip(key_fields, key)}
    out.update(
        {
            "intent_count": len(rows),
            "positive_intent_count": sum(1 for value in values_ticks if value > 0),
            "positive_intent_rate": _fmt(sum(1 for value in values_ticks if value > 0) / len(values_ticks), 6),
            "total_optimistic_mid_pnl_ticks": _fmt(sum(values_ticks), 6),
            "mean_optimistic_mid_pnl_ticks": _fmt(statistics.fmean(values_ticks), 6),
            "median_optimistic_mid_pnl_ticks": _fmt(_median(values_ticks), 6),
            "min_optimistic_mid_pnl_ticks": _fmt(min(values_ticks), 6),
            "max_optimistic_mid_pnl_ticks": _fmt(max(values_ticks), 6),
            "total_optimistic_mid_pnl_usdc": _fmt(sum(values_usdc), 8),
            "mean_optimistic_mid_pnl_usdc": _fmt(statistics.fmean(values_usdc), 8),
            "assumption_set": rows[0]["assumption_set"],
        }
    )
    return out


def _summarize_fixed(pnl_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    key_fields = ["sample_set_id", "sample_id", "quote_side", "threshold_ticks", "persistence_count", "horizon_ms"]
    for row in pnl_rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(row)
    return [_summary_row(key, rows, key_fields) for key, rows in sorted(grouped.items())]


def _summarize_aggregate_fixed(pnl_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    key_fields = ["sample_set_id", "quote_side", "threshold_ticks", "persistence_count", "horizon_ms"]
    for row in pnl_rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(row)
    return [_summary_row(key, rows, key_fields) for key, rows in sorted(grouped.items())]


def _summarize_oracle(pnl_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_decision: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    decision_fields = ["sample_set_id", "sample_id", "source_row_index", "threshold_ticks", "persistence_count", "quote_side"]
    for row in pnl_rows:
        key = tuple(row[field] for field in decision_fields)
        by_decision[key].append(row)
    best_rows: list[dict[str, Any]] = []
    for rows in by_decision.values():
        best = max(rows, key=lambda row: (float(row["optimistic_mid_pnl_ticks"]), -int(row["horizon_ms"])))
        best_rows.append(best)

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    key_fields = ["sample_set_id", "sample_id", "quote_side", "threshold_ticks", "persistence_count"]
    for row in best_rows:
        key = tuple(row[field] for field in key_fields)
        grouped[key].append(row)

    out: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        summary = _summary_row(key, rows, key_fields)
        horizon_counts = Counter(str(row["horizon_ms"]) for row in rows)
        summary["oracle_label"] = "non_tradeable_oracle_upper_bound"
        summary["best_horizon_ms_counts"] = "|".join(f"{h}:{count}" for h, count in sorted(horizon_counts.items(), key=lambda item: int(item[0])))
        out.append(summary)
    return out


def _interpretation_rows(pnl_rows: list[dict[str, Any]], sample_sets: list[SampleSet]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "sample_set_id": "requested_six",
            "threshold_ticks": PRIMARY_THRESHOLD_TICKS,
            "persistence_count": PRIMARY_PERSISTENCE_COUNT,
            "horizon_ms": PRIMARY_INTERPRETATION_HORIZON_MS,
            "classification": "superseded",
            "sample_count": "",
            "samples_positive_total": "",
            "aggregate_total_optimistic_mid_pnl_usdc": "",
            "mean_optimistic_mid_pnl_ticks": "",
            "note": "Superseded by user-selected official sample set canonical_7.",
        }
    ]
    computed_sets = {sample_set.sample_set_id for sample_set in sample_sets if sample_set.compute_status == "computed"}
    for sample_set_id in sorted(computed_sets):
        if sample_set_id == "requested_six":
            continue
        selected = [
            row
            for row in pnl_rows
            if row["sample_set_id"] == sample_set_id
            and row["threshold_ticks"] == PRIMARY_THRESHOLD_TICKS
            and row["persistence_count"] == PRIMARY_PERSISTENCE_COUNT
            and row["horizon_ms"] == PRIMARY_INTERPRETATION_HORIZON_MS
        ]
        if not selected:
            rows.append(
                {
                    "sample_set_id": sample_set_id,
                    "threshold_ticks": PRIMARY_THRESHOLD_TICKS,
                    "persistence_count": PRIMARY_PERSISTENCE_COUNT,
                    "horizon_ms": PRIMARY_INTERPRETATION_HORIZON_MS,
                    "classification": "blocked",
                    "sample_count": 0,
                    "samples_positive_total": 0,
                    "aggregate_total_optimistic_mid_pnl_usdc": "",
                    "mean_optimistic_mid_pnl_ticks": "",
                    "note": "No eligible primary-candidate optimistic PnL rows.",
                }
            )
            continue
        by_sample: dict[str, float] = defaultdict(float)
        for row in selected:
            by_sample[str(row["sample_id"])] += float(row["optimistic_mid_pnl_usdc"])
        sample_count = len(by_sample)
        positive_samples = sum(1 for value in by_sample.values() if value > 0)
        positive_share = positive_samples / sample_count if sample_count else 0.0
        total_usdc = sum(by_sample.values())
        mean_ticks = statistics.fmean(float(row["optimistic_mid_pnl_ticks"]) for row in selected)
        if total_usdc <= 0 or mean_ticks <= 0:
            classification = "weak"
        elif positive_share >= 0.8 and mean_ticks >= 5:
            classification = "materially_positive"
        elif positive_share >= 0.5:
            classification = "weak"
        else:
            classification = "unstable"
        rows.append(
            {
                "sample_set_id": sample_set_id,
                "threshold_ticks": PRIMARY_THRESHOLD_TICKS,
                "persistence_count": PRIMARY_PERSISTENCE_COUNT,
                "horizon_ms": PRIMARY_INTERPRETATION_HORIZON_MS,
                "classification": classification,
                "sample_count": sample_count,
                "samples_positive_total": positive_samples,
                "aggregate_total_optimistic_mid_pnl_usdc": _fmt(total_usdc, 8),
                "mean_optimistic_mid_pnl_ticks": _fmt(mean_ticks, 6),
                "note": "Diagnostic optimistic public-data upper bound; execution layer and real PnL remain unproven.",
            }
        )
    return rows


def _audit_rows(pnl_rows: list[dict[str, Any]], limit: int = 500) -> list[dict[str, Any]]:
    preferred = [
        row
        for row in pnl_rows
        if row["threshold_ticks"] == PRIMARY_THRESHOLD_TICKS
        and row["persistence_count"] in {PRIMARY_PERSISTENCE_COUNT, FALLBACK_PERSISTENCE_COUNT}
    ]
    rows = preferred[:limit] if preferred else pnl_rows[:limit]
    return [
        {
            "sample_set_id": row["sample_set_id"],
            "sample_id": row["sample_id"],
            "source_row_index": row["source_row_index"],
            "hyperliquid_decision_ts": row["hyperliquid_decision_ts"],
            "threshold_ticks": row["threshold_ticks"],
            "persistence_count": row["persistence_count"],
            "quote_side": row["quote_side"],
            "horizon_ms": row["horizon_ms"],
            "basis_mid_ticks": _fmt(float(row["basis_mid_ticks"]), 6),
            "quote_price": _fmt(float(row["quote_price"]), 6),
            "quote_distance_ticks": _fmt(float(row["quote_distance_ticks"]), 6),
            "future_mid_move_ticks": _fmt(float(row["future_mid_move_ticks"]), 6),
            "optimistic_mid_pnl_ticks": _fmt(float(row["optimistic_mid_pnl_ticks"]), 6),
            "optimistic_mid_pnl_usdc": _fmt(float(row["optimistic_mid_pnl_usdc"]), 8),
            "basis_future_mid_response_ticks": _fmt(row["basis_future_mid_response_ticks"], 6),
            "assumption_set": row["assumption_set"],
            "proof_boundary": "optimistic_proxy_not_real_pnl_not_real_fill",
        }
        for row in rows
    ]


def run(replay_manifest_path: Path, source_manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    replay_manifest = _read_json(replay_manifest_path)
    source_manifest_rows = _read_csv(source_manifest_path)
    sample_sets, reconciliation_rows = _sample_sets(replay_manifest, source_manifest_rows)

    all_pnl_rows: list[dict[str, Any]] = []
    all_eligibility_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    decision_counts: dict[str, int] = {}
    pricing_row_counts: dict[str, int] = {}

    for sample_set in sample_sets:
        if sample_set.compute_status != "computed":
            continue
        pricing_rows, next_membership = _load_sample_set_rows(sample_set)
        decisions = _decision_rows(pricing_rows)
        pnl_rows, eligibility_rows = _pnl_rows_for_grid(sample_set.sample_set_id, decisions)
        all_pnl_rows.extend(pnl_rows)
        all_eligibility_rows.extend(eligibility_rows)
        membership_rows.extend(next_membership)
        decision_counts[sample_set.sample_set_id] = len(decisions)
        pricing_row_counts[sample_set.sample_set_id] = len(pricing_rows)

    fixed_summary = _summarize_fixed(all_pnl_rows)
    aggregate_fixed_summary = _summarize_aggregate_fixed(all_pnl_rows)
    oracle_summary = _summarize_oracle(all_pnl_rows)
    interpretation = _interpretation_rows(all_pnl_rows, sample_sets)
    audit_rows = _audit_rows(all_pnl_rows)

    _write_csv(
        output_dir / "sample_set_reconciliation.csv",
        reconciliation_rows,
        [
            "sample_set_id",
            "requested_by_user",
            "official_sample_set",
            "requested_count",
            "manifest_sample_count",
            "resolved_pricing_input_count",
            "compute_status",
            "reconciliation_status",
            "note",
        ],
    )
    _write_csv(
        output_dir / "sample_set_membership.csv",
        membership_rows,
        ["sample_set_id", "pricing_signal_path", "pricing_row_count", "sample_ids", "sample_id_count", "membership_note"],
    )
    _write_csv(
        output_dir / "eligibility_summary.csv",
        all_eligibility_rows,
        [
            "sample_set_id",
            "threshold_ticks",
            "persistence_count",
            "decision_rows_evaluated",
            "eligible_decisions",
            "eligible_buy",
            "eligible_sell",
            "no_signal",
            "reject_missing_basis",
            "reject_persistence",
            "reject_quality",
            "reject_stale_or_data_gap",
            "reject_missing_book",
            "reject_missing_future_label",
        ],
    )
    fixed_fields = [
        "sample_set_id",
        "sample_id",
        "quote_side",
        "threshold_ticks",
        "persistence_count",
        "horizon_ms",
        "intent_count",
        "positive_intent_count",
        "positive_intent_rate",
        "total_optimistic_mid_pnl_ticks",
        "mean_optimistic_mid_pnl_ticks",
        "median_optimistic_mid_pnl_ticks",
        "min_optimistic_mid_pnl_ticks",
        "max_optimistic_mid_pnl_ticks",
        "total_optimistic_mid_pnl_usdc",
        "mean_optimistic_mid_pnl_usdc",
        "assumption_set",
    ]
    _write_csv(output_dir / "fixed_horizon_pnl_summary.csv", fixed_summary, fixed_fields)
    _write_csv(
        output_dir / "aggregate_fixed_horizon_pnl_summary.csv",
        aggregate_fixed_summary,
        [field for field in fixed_fields if field != "sample_id"],
    )
    _write_csv(
        output_dir / "oracle_best_horizon_summary.csv",
        oracle_summary,
        [
            "sample_set_id",
            "sample_id",
            "quote_side",
            "threshold_ticks",
            "persistence_count",
            "oracle_label",
            "intent_count",
            "positive_intent_count",
            "positive_intent_rate",
            "total_optimistic_mid_pnl_ticks",
            "mean_optimistic_mid_pnl_ticks",
            "median_optimistic_mid_pnl_ticks",
            "min_optimistic_mid_pnl_ticks",
            "max_optimistic_mid_pnl_ticks",
            "total_optimistic_mid_pnl_usdc",
            "mean_optimistic_mid_pnl_usdc",
            "best_horizon_ms_counts",
            "assumption_set",
        ],
    )
    _write_csv(
        output_dir / "diagnostic_interpretation.csv",
        interpretation,
        [
            "sample_set_id",
            "threshold_ticks",
            "persistence_count",
            "horizon_ms",
            "classification",
            "sample_count",
            "samples_positive_total",
            "aggregate_total_optimistic_mid_pnl_usdc",
            "mean_optimistic_mid_pnl_ticks",
            "note",
        ],
    )
    _write_csv(
        output_dir / "row_level_audit_sample.csv",
        audit_rows,
        [
            "sample_set_id",
            "sample_id",
            "source_row_index",
            "hyperliquid_decision_ts",
            "threshold_ticks",
            "persistence_count",
            "quote_side",
            "horizon_ms",
            "basis_mid_ticks",
            "quote_price",
            "quote_distance_ticks",
            "future_mid_move_ticks",
            "optimistic_mid_pnl_ticks",
            "optimistic_mid_pnl_usdc",
            "basis_future_mid_response_ticks",
            "assumption_set",
            "proof_boundary",
        ],
    )

    manifest = {
        "assumption_set": "unconstrained_all_intents",
        "boundary_flags": {
            "account_inventory_modeled": False,
            "account_query_called": False,
            "credentials_read": False,
            "fee_rebate_settlement_claimed": False,
            "fill_probability_modeled": False,
            "live_authorized": False,
            "live_bot_started": False,
            "maker_viability_claimed": False,
            "order_amendment_called": False,
            "order_cancellation_called": False,
            "order_placement_called": False,
            "private_endpoint_called": False,
            "queue_priority_modeled": False,
            "real_fill_claimed": False,
            "real_pnl_claimed": False,
        },
        "decision_counts": decision_counts,
        "final_recommendation": FINAL_RECOMMENDATION,
        "formula": {
            "buy": "hyperliquid_future_mid_move_ticks + context_hyperliquid_spread_ticks / 2",
            "sell": "-hyperliquid_future_mid_move_ticks + context_hyperliquid_spread_ticks / 2",
            "usdc": "optimistic_mid_pnl_ticks * 0.1 * 0.01",
        },
        "git_commit": _git_commit(),
        "horizon_ms_observed": sorted({int(row["horizon_ms"]) for row in all_pnl_rows}),
        "input_replay_manifest": str(replay_manifest_path),
        "input_source_manifest": str(source_manifest_path),
        "order_size_btc": ORDER_SIZE_BTC,
        "official_sample_set": OFFICIAL_SAMPLE_SET_ID,
        "output_files": {
            "aggregate_fixed_horizon_pnl_summary": str(output_dir / "aggregate_fixed_horizon_pnl_summary.csv"),
            "diagnostic_interpretation": str(output_dir / "diagnostic_interpretation.csv"),
            "eligibility_summary": str(output_dir / "eligibility_summary.csv"),
            "fixed_horizon_pnl_summary": str(output_dir / "fixed_horizon_pnl_summary.csv"),
            "oracle_best_horizon_summary": str(output_dir / "oracle_best_horizon_summary.csv"),
            "row_level_audit_sample": str(output_dir / "row_level_audit_sample.csv"),
            "sample_set_membership": str(output_dir / "sample_set_membership.csv"),
            "sample_set_reconciliation": str(output_dir / "sample_set_reconciliation.csv"),
        },
        "pnl_rows_evaluated": len(all_pnl_rows),
        "pricing_row_counts": pricing_row_counts,
        "sample_set_clarification": SAMPLE_SET_CLARIFICATION,
        "sample_sets": {
            sample_set.sample_set_id: {
                "compute_status": sample_set.compute_status,
                "path_count": len(sample_set.paths),
                "reconciliation_status": sample_set.reconciliation_status,
                "note": sample_set.note,
            }
            for sample_set in sample_sets
        },
        "task_id": TASK_ID,
        "threshold_grid_ticks": THRESHOLDS_TICKS,
        "tick_size": TICK_SIZE,
    }
    _write_json(output_dir / "optimistic_pnl_proxy_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-manifest", type=Path, default=DEFAULT_REPLAY_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.replay_manifest.resolve(), args.source_manifest.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
