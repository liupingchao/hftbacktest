#!/usr/bin/env python3
"""Read-only basis-positive filtered context viability assessment."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import canonical_basis_positive_wrong_way_decomposition as wrong_way
import canonical_event_mode_evidence as canonical_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0609T003"
SCHEMA_VERSION = "basis_positive_filtered_context_viability_v1"
DEFAULT_INPUT_DIR = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "basis_positive_targeted_public_collection_0609T002"
    / "event_mode_canonical_pricing_signal_0609T002"
)
DEFAULT_T002_COLLECTION_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_targeted_public_collection_0609T002"
DEFAULT_T002_DECOMPOSITION_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_basis_positive_wrong_way_decomposition_0609T002"
DEFAULT_T001_DECOMPOSITION_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_basis_positive_wrong_way_decomposition_0609T001"
DEFAULT_T006_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_basis_positive_robustness_0608T006"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_filtered_context_viability_0609T003"
PRIMARY_HORIZON_MS = 1000

FINAL_RECOMMENDATIONS = {
    "candidate_for_read_only_case_design",
    "keep_as_context_only_signal",
    "needs_more_filtered_public_samples",
    "reject_filtered_context_viability",
}
LABELS = {
    "basis_positive_raw_context",
    "basis_positive_clean_context",
    "basis_positive_tail_risk_context",
}
BOUNDARY_FLAGS = {
    "read_only_public_artifact_assessment": True,
    "observation_layer_only": True,
    "no_new_data_collection": True,
    "no_remote_run": True,
    "no_private_keys": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_implementation": True,
    "no_case_library_implementation": True,
    "no_case_library_trigger": True,
    "no_shadow_decision_generation": True,
    "no_executable_trading_instruction": True,
    "no_actual_order_side_output": True,
    "no_quote_price_or_size_output": True,
    "no_leverage_output": True,
    "no_stop_or_take_profit_rule": True,
    "no_live_trading_bot": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_parameter_search": True,
    "no_deployment_recommendation": True,
    "no_promotion": True,
}


class FilteredContextInputError(ValueError):
    """Raised when inputs violate the T003 read-only viability contract."""


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FilteredContextInputError(f"{path} must contain a JSON object")
    return payload


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


def _as_float(value: Any) -> float | None:
    return wrong_way._as_float(value)


def _as_int(value: Any, default: int = 0) -> int:
    return wrong_way._as_int(value, default=default)


def _fmt(value: float | None, places: int = 8) -> str:
    return wrong_way._fmt(value, places=places)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _percentile(values: list[float], pct: float) -> float | None:
    return wrong_way._percentile(values, pct)


def _sample_count(rows: list[dict[str, Any]]) -> int:
    return len({str(row.get("sample_id", "")) for row in rows if row.get("sample_id", "")})


def _max_sample_share(rows: list[dict[str, Any]]) -> float | None:
    counts = Counter(str(row.get("sample_id", "")) for row in rows)
    total = sum(counts.values())
    return max(counts.values()) / total if total else None


def _load_all_horizon_rows(loaded: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for sample in loaded["source_manifest"].get("samples", []):
        if not isinstance(sample, dict):
            continue
        if sample.get("decision_mode") != "event" or sample.get("canonical_status") != "canonical_event_mode":
            raise FilteredContextInputError("formal input contains non-canonical sample in manifest")
        pricing_path = Path(str(sample.get("pricing_signal_rows", "")))
        if not pricing_path.exists():
            raise FilteredContextInputError(
                f"missing manifest samples[].pricing_signal_rows for sample {sample.get('sample_id')}: {pricing_path}"
            )
        rows.extend(_read_csv(pricing_path))
    return rows


def _validate_prerequisites(t002_collection_dir: Path, t002_decomposition_dir: Path, t001_decomposition_dir: Path, t006_dir: Path) -> dict[str, Any]:
    collection = _read_json(t002_collection_dir / "local_processing_manifest.json")
    if collection.get("t002_final_recommendation") != "tail_filter_hypothesis_validated_for_read_only_research":
        raise FilteredContextInputError("T003 requires T002 final recommendation tail_filter_hypothesis_validated_for_read_only_research")
    if collection.get("aggregate_canonical_sample_count") != 7 or collection.get("decomposition_canonical_sample_count") != 7:
        raise FilteredContextInputError("T003 requires T002 7-sample aggregate and decomposition")
    t002 = _read_json(t002_decomposition_dir / "basis_positive_wrong_way_manifest.json")
    if t002.get("canonical_sample_count") != 7:
        raise FilteredContextInputError("T003 requires T002 decomposition canonical_sample_count=7")
    t001 = _read_json(t001_decomposition_dir / "basis_positive_wrong_way_manifest.json")
    if t001.get("final_recommendation") != "targeted_collection_ready":
        raise FilteredContextInputError("T003 requires T001 targeted_collection_ready prerequisite")
    t006 = _read_json(t006_dir / "basis_positive_robustness_manifest.json")
    if t006.get("final_recommendation") != "needs_more_samples":
        raise FilteredContextInputError("T003 requires T006 needs_more_samples prerequisite")
    return {"t002_collection": collection, "t002_decomposition": t002, "t001_decomposition": t001, "t006": t006}


def _basis_thresholds(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    return wrong_way._basis_thresholds(rows)


def _basis_magnitude_bucket(value: float | None, low: float | None, high: float | None) -> str:
    return wrong_way._basis_magnitude_bucket(value, low, high)


def _signed_magnitude_bucket(value: float | None, prefix: str) -> str:
    return wrong_way._signed_magnitude_bucket(value, prefix)


def _spread_bucket(value: float | None) -> str:
    return wrong_way._spread_bucket(value)


def _join_age_bucket(value: float | None) -> str:
    return wrong_way._join_age_bucket(value)


def _vol_thresholds(rows: list[dict[str, Any]]) -> tuple[float | None, float | None]:
    return wrong_way._vol_thresholds(rows)


def _visible_movement_bucket(value: float | None, low: float | None, high: float | None) -> str:
    return wrong_way._volatility_bucket(value, low, high)


def _enrich_rows(rows: list[dict[str, Any]], *, threshold_rows: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    threshold_source = threshold_rows if threshold_rows is not None else rows
    basis_low, basis_high = _basis_thresholds(threshold_source)
    vol_low, vol_high = _vol_thresholds(threshold_source)
    out: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        enriched["_basis_magnitude_bucket"] = _basis_magnitude_bucket(row["_basis"], basis_low, basis_high)
        enriched["_hl_top5_imbalance_bucket"] = _signed_magnitude_bucket(
            _as_float(row.get("context_hyperliquid_top5_imbalance")),
            "hl_top5_imbalance",
        )
        enriched["_hl_microprice_minus_mid_bucket"] = _signed_magnitude_bucket(
            _as_float(row.get("context_hyperliquid_microprice_minus_mid_ticks")),
            "hl_microprice_minus_mid",
        )
        enriched["_spread_bucket"] = _spread_bucket(_as_float(row.get("context_hyperliquid_spread_ticks")))
        enriched["_join_age_bucket"] = _join_age_bucket(_as_float(row.get("context_hyperliquid_join_age_ms")))
        enriched["_binance_momentum_bucket"] = _signed_magnitude_bucket(
            _as_float(row.get("input_binance_mid_move_ticks_from_prev")),
            "binance_momentum",
        )
        enriched["_visible_movement_bucket"] = _visible_movement_bucket(
            _as_float(row.get("input_binance_mid_move_ticks_from_prev")),
            vol_low,
            vol_high,
        )
        enriched["_hl_book_state_bucket"] = "|".join(
            [enriched["_hl_top5_imbalance_bucket"], enriched["_hl_microprice_minus_mid_bucket"]]
        )
        enriched["_tail_basis_small"] = enriched["_basis_magnitude_bucket"] == "basis_positive_small"
        enriched["_tail_hl_top5_negative_small"] = (
            enriched["_hl_top5_imbalance_bucket"] == "hl_top5_imbalance_negative_small"
        )
        enriched["_tail_hl_microprice_negative_small"] = (
            enriched["_hl_microprice_minus_mid_bucket"] == "hl_microprice_minus_mid_negative_small"
        )
        enriched["_tail_combined"] = (
            bool(enriched["_tail_basis_small"])
            or bool(enriched["_tail_hl_top5_negative_small"])
            or bool(enriched["_tail_hl_microprice_negative_small"])
        )
        out.append(enriched)
    return out


def _stats(rows: list[dict[str, Any]], *, label: str) -> dict[str, Any]:
    moves = [float(row["_future"]) for row in rows]
    positives = sum(1 for value in moves if value > 0)
    negatives = sum(1 for value in moves if value < 0)
    nonzero = positives + negatives
    wrong_losses = [abs(value) for value in moves if value < 0]
    return {
        "context_label": label,
        "row_count": len(rows),
        "sample_count": _sample_count(rows),
        "positive_future_count": positives,
        "negative_future_count": negatives,
        "direction_hit_rate": positives / nonzero if nonzero else None,
        "mean_future_mid_move_ticks": _mean(moves),
        "median_future_mid_move_ticks": statistics.median(moves) if moves else None,
        "wrong_way_count": negatives,
        "wrong_way_rate": negatives / len(rows) if rows else None,
        "p95_wrong_way_loss_ticks": _percentile(wrong_losses, 0.95),
        "max_wrong_way_loss_ticks": max(wrong_losses) if wrong_losses else None,
        "max_sample_row_share": _max_sample_share(rows),
    }


def _stats_row(label: str, rows: list[dict[str, Any]], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    stats = _stats(rows, label=label)
    p95 = stats["p95_wrong_way_loss_ticks"]
    base_p95 = baseline.get("p95_wrong_way_loss_ticks") if baseline else None
    return {
        "context_label": label,
        "row_count": stats["row_count"],
        "sample_count": stats["sample_count"],
        "positive_future_count": stats["positive_future_count"],
        "negative_future_count": stats["negative_future_count"],
        "direction_hit_rate": _fmt(stats["direction_hit_rate"]),
        "mean_future_mid_move_ticks": _fmt(stats["mean_future_mid_move_ticks"]),
        "median_future_mid_move_ticks": _fmt(stats["median_future_mid_move_ticks"]),
        "wrong_way_count": stats["wrong_way_count"],
        "wrong_way_rate": _fmt(stats["wrong_way_rate"]),
        "p95_wrong_way_loss_ticks": _fmt(p95),
        "max_wrong_way_loss_ticks": _fmt(stats["max_wrong_way_loss_ticks"]),
        "max_sample_row_share": _fmt(stats["max_sample_row_share"]),
        "p95_wrong_way_improvement_vs_raw_ticks": _fmt((base_p95 - p95) if base_p95 is not None and p95 is not None else None),
    }


SUMMARY_FIELDS = [
    "context_label",
    "row_count",
    "sample_count",
    "positive_future_count",
    "negative_future_count",
    "direction_hit_rate",
    "mean_future_mid_move_ticks",
    "median_future_mid_move_ticks",
    "wrong_way_count",
    "wrong_way_rate",
    "p95_wrong_way_loss_ticks",
    "max_wrong_way_loss_ticks",
    "max_sample_row_share",
    "p95_wrong_way_improvement_vs_raw_ticks",
]


def _subset_rows(primary_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    raw = [row for row in primary_rows if row["_basis"] > 0]
    rejected = [row for row in raw if row["_tail_combined"]]
    clean = [row for row in raw if not row["_tail_combined"]]
    return {
        "basis_positive_raw_context": raw,
        "basis_positive_clean_context": clean,
        "basis_positive_tail_risk_context": rejected,
    }


def _summary_rows(primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    subsets = _subset_rows(primary_rows)
    raw_stats = _stats(subsets["basis_positive_raw_context"], label="basis_positive_raw_context")
    return [_stats_row(label, rows, raw_stats) for label, rows in subsets.items()]


def _tail_subset_rows(primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = [row for row in primary_rows if row["_basis"] > 0]
    definitions = {
        "basis_positive_small": lambda row: row["_tail_basis_small"],
        "hl_top5_imbalance_negative_small": lambda row: row["_tail_hl_top5_negative_small"],
        "hl_microprice_minus_mid_negative_small": lambda row: row["_tail_hl_microprice_negative_small"],
        "combined_tail_risk_mask": lambda row: row["_tail_combined"],
    }
    raw_stats = _stats(raw, label="basis_positive_raw_context")
    out: list[dict[str, Any]] = []
    for label, predicate in definitions.items():
        subset = [row for row in raw if predicate(row)]
        stats = _stats_row(label, subset, raw_stats)
        stats["tail_condition"] = label
        stats["condition_definition"] = {
            "basis_positive_small": "basis_magnitude_bucket=basis_positive_small",
            "hl_top5_imbalance_negative_small": "hl_top5_imbalance_bucket=hl_top5_imbalance_negative_small",
            "hl_microprice_minus_mid_negative_small": "hl_microprice_minus_mid_bucket=hl_microprice_minus_mid_negative_small",
            "combined_tail_risk_mask": "OR of the three T002 validated visible tail hypotheses",
        }[label]
        out.append(stats)
    return out


def _stability_rows(primary_rows: list[dict[str, Any]], *, group_key: str, group_name: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in [row for row in primary_rows if row["_basis"] > 0]:
        groups[str(row.get(group_key, ""))].append(row)
    raw_stats = _stats([row for row in primary_rows if row["_basis"] > 0], label="basis_positive_raw_context")
    out: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        subsets = _subset_rows(rows)
        for label, subset in subsets.items():
            record = _stats_row(label, subset, raw_stats)
            record["group_name"] = group_name
            record["group_value"] = key
            out.append(record)
    return out


def _horizon_rows(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        if row["_basis"] > 0:
            groups[_as_int(row.get("horizon_ms"))].append(row)
    primary_raw = [row for row in all_rows if row["_basis"] > 0 and _as_int(row.get("horizon_ms")) == PRIMARY_HORIZON_MS]
    raw_stats = _stats(primary_raw, label="basis_positive_raw_context")
    out: list[dict[str, Any]] = []
    for horizon, rows in sorted(groups.items()):
        subsets = _subset_rows(rows)
        for label, subset in subsets.items():
            record = _stats_row(label, subset, raw_stats)
            record["horizon_ms"] = horizon
            record["effective_future_row_delta_count"] = len({str(row.get("effective_future_row_delta", "")) for row in subset})
            out.append(record)
    return out


def _conditioning_rows(primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw_stats = _stats([row for row in primary_rows if row["_basis"] > 0], label="basis_positive_raw_context")
    dimensions = {
        "spread_bucket": "_spread_bucket",
        "join_age_bucket": "_join_age_bucket",
        "visible_movement_bucket": "_visible_movement_bucket",
        "binance_momentum_bucket": "_binance_momentum_bucket",
        "hl_book_state_bucket": "_hl_book_state_bucket",
        "hl_top5_imbalance_bucket": "_hl_top5_imbalance_bucket",
        "hl_microprice_minus_mid_bucket": "_hl_microprice_minus_mid_bucket",
    }
    out: list[dict[str, Any]] = []
    for dimension, field in dimensions.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in [row for row in primary_rows if row["_basis"] > 0]:
            groups[str(row.get(field, ""))].append(row)
        for bucket, rows in sorted(groups.items()):
            subsets = _subset_rows(rows)
            for label, subset in subsets.items():
                record = _stats_row(label, subset, raw_stats)
                record["conditioning_dimension"] = dimension
                record["conditioning_bucket"] = bucket
                out.append(record)
    return out


def _label_rows(primary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw = _subset_rows(primary_rows)
    rows = [
        {
            "research_context_label": "basis_positive_raw_context",
            "read_only_context_only": True,
            "definition": "context_basis_mid_ticks > 0 after excluding missing/non-numeric basis rows",
            "row_count": len(raw["basis_positive_raw_context"]),
            "sample_count": _sample_count(raw["basis_positive_raw_context"]),
            "allowed_use": "observation-layer read-only research label only",
            "forbidden_use": "no order side, quote price, size, leverage, stop/take-profit, case-library trigger, shadow decision, or executable trading instruction",
        },
        {
            "research_context_label": "basis_positive_clean_context",
            "read_only_context_only": True,
            "definition": "basis_positive_raw_context AND NOT combined_tail_risk_mask",
            "row_count": len(raw["basis_positive_clean_context"]),
            "sample_count": _sample_count(raw["basis_positive_clean_context"]),
            "allowed_use": "observation-layer filtered context research label only",
            "forbidden_use": "no order side, quote price, size, leverage, stop/take-profit, case-library trigger, shadow decision, or executable trading instruction",
        },
        {
            "research_context_label": "basis_positive_tail_risk_context",
            "read_only_context_only": True,
            "definition": "basis_positive_raw_context AND combined_tail_risk_mask",
            "row_count": len(raw["basis_positive_tail_risk_context"]),
            "sample_count": _sample_count(raw["basis_positive_tail_risk_context"]),
            "allowed_use": "observation-layer rejected-tail diagnostic label only",
            "forbidden_use": "no order side, quote price, size, leverage, stop/take-profit, case-library trigger, shadow decision, or executable trading instruction",
        },
    ]
    return rows


def _gap_register(path: Path) -> None:
    lines = [
        "# Execution Evidence Gap Register",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "This task uses public observation-layer artifacts only. It does not prove maker execution viability.",
        "",
        "## Unproven Execution-Layer Items",
        "",
        "- Fill probability is unproven because no private/order lifecycle or live maker orders are used.",
        "- Queue position and queue-ahead are unproven because public book observations do not establish actual maker priority.",
        "- Post-only reject behavior is unproven because no order submission path is exercised.",
        "- Cancel-fill race behavior is unproven because no cancel/order lifecycle events are observed.",
        "- Fees, rebates, and spread capture are unproven because public future-mid movement is not realized execution PnL.",
        "- Inventory lifecycle is unproven because there are no positions, fills, or account state.",
        "- Real order lifecycle is unproven because private/account/order endpoints, user streams, signing, and nonce handling are out of scope.",
        "",
        "## Boundary",
        "",
        "- No new collection, no remote run, no private/order endpoint, no strategy implementation, no case-library, no shadow decision, no live/default-on/tiny-live, no parameter search, and no promotion is authorized.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _recommendation(
    summary: list[dict[str, Any]],
    per_sample: list[dict[str, Any]],
    horizon: list[dict[str, Any]],
) -> tuple[str, str]:
    raw = next(row for row in summary if row["context_label"] == "basis_positive_raw_context")
    clean = next(row for row in summary if row["context_label"] == "basis_positive_clean_context")
    clean_samples = int(clean["sample_count"])
    clean_share = _as_float(clean["max_sample_row_share"])
    clean_improvement = _as_float(clean["p95_wrong_way_improvement_vs_raw_ticks"])
    clean_mean = _as_float(clean["mean_future_mid_move_ticks"])
    sample_clean = [row for row in per_sample if row["context_label"] == "basis_positive_clean_context" and int(row["row_count"]) > 0]
    sample_reversals = [row for row in sample_clean if (_as_float(row["mean_future_mid_move_ticks"]) or 0.0) < 0]
    horizon_clean = [row for row in horizon if row["context_label"] == "basis_positive_clean_context" and int(row["row_count"]) > 0]
    horizon_reversals = [row for row in horizon_clean if (_as_float(row["mean_future_mid_move_ticks"]) or 0.0) < 0]
    raw_wrong = int(raw["wrong_way_count"])
    clean_wrong = int(clean["wrong_way_count"])
    if clean_samples >= 5 and clean_share is not None and clean_share < 0.40 and clean_improvement is not None and clean_improvement > 0 and not sample_reversals and not horizon_reversals:
        return "candidate_for_read_only_case_design", "clean filtered basis-positive context improves wrong-way p95 loss and shows no sample/horizon reversal"
    if clean_samples >= 5 and clean_mean is not None and clean_mean > 0 and clean_wrong < raw_wrong:
        return "keep_as_context_only_signal", "filtered context remains positive but promotion to case-design is blocked by concentration, tail, or conditioning caveats"
    if clean_samples < 5 or (clean_share is not None and clean_share >= 0.40):
        return "needs_more_filtered_public_samples", "clean filtered context lacks enough independent sample coverage"
    return "reject_filtered_context_viability", "filtered context does not retain stable observation-layer edge"


def _recommendation_doc(path: Path, manifest: dict[str, Any], summary: list[dict[str, Any]]) -> None:
    clean = next(row for row in summary if row["context_label"] == "basis_positive_clean_context")
    rejected = next(row for row in summary if row["context_label"] == "basis_positive_tail_risk_context")
    lines = [
        "# Filtered Context Next Step Recommendation",
        "",
        f"Task: `{TASK_ID}`",
        "",
        f"- Final recommendation: `{manifest['final_recommendation']}`",
        f"- Reason: {manifest['final_recommendation_reason']}",
        f"- Clean context rows: `{clean['row_count']}`, samples: `{clean['sample_count']}`, mean future move: `{clean['mean_future_mid_move_ticks']}` ticks.",
        f"- Clean context p95 wrong-way loss improvement vs raw: `{clean['p95_wrong_way_improvement_vs_raw_ticks']}` ticks.",
        f"- Rejected tail-risk rows: `{rejected['row_count']}`, wrong-way count: `{rejected['wrong_way_count']}`.",
        "",
        "## Boundary",
        "",
        "- This recommendation is read-only public observation-layer research only.",
        "- It does not authorize strategy implementation, private/order endpoint use, order lifecycle, case-library implementation, shadow decision generation, live/default-on/tiny-live, parameter search, deployment recommendation, or promotion.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_filtered_context_viability(
    *,
    input_dir: str | Path,
    t002_collection_dir: str | Path,
    t002_decomposition_dir: str | Path,
    t001_decomposition_dir: str | Path,
    t006_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_t002_collection = _expand(t002_collection_dir)
    resolved_t002_decomposition = _expand(t002_decomposition_dir)
    resolved_t001_decomposition = _expand(t001_decomposition_dir)
    resolved_t006 = _expand(t006_dir)
    resolved_output = _expand(output_dir)
    prerequisites = _validate_prerequisites(
        resolved_t002_collection,
        resolved_t002_decomposition,
        resolved_t001_decomposition,
        resolved_t006,
    )
    guard = canonical_loader.guard_canonical_event_mode_evidence(
        input_dir=resolved_input,
        require_formal_evidence=True,
        allow_diagnostic_validation=False,
    )
    canonical_loader.validate_canonical_source_lock_manifest(
        guard["canonical_source_lock_manifest"],
        require_formal_evidence=True,
    )
    if guard["canonical_sample_count"] < 5 or guard["diagnostic_rejection_count"] != 0:
        raise FilteredContextInputError("T003 requires at least 5 formal canonical event-mode samples and no diagnostic synthetic samples")
    eligible_all_base = wrong_way._eligible_rows(_load_all_horizon_rows(guard["loaded_evidence"]))
    primary_base = [row for row in eligible_all_base if _as_int(row.get("horizon_ms")) == PRIMARY_HORIZON_MS]
    if not primary_base:
        raise FilteredContextInputError("no eligible primary horizon rows with numeric basis")
    eligible_all = _enrich_rows(eligible_all_base)
    primary = _enrich_rows(primary_base, threshold_rows=primary_base)
    summary = _summary_rows(primary)
    tail = _tail_subset_rows(primary)
    per_sample = _stability_rows(primary, group_key="sample_id", group_name="sample_id")
    horizon = _horizon_rows(eligible_all)
    conditioning = _conditioning_rows(primary)
    labels = _label_rows(primary)
    final_recommendation, reason = _recommendation(summary, per_sample, horizon)
    if final_recommendation not in FINAL_RECOMMENDATIONS:
        raise AssertionError(f"unexpected recommendation: {final_recommendation}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "t002_collection_dir": str(resolved_t002_collection),
        "t002_decomposition_dir": str(resolved_t002_decomposition),
        "t001_decomposition_dir": str(resolved_t001_decomposition),
        "t006_dir": str(resolved_t006),
        "output_dir": str(resolved_output),
        "canonical_sample_count": guard["canonical_sample_count"],
        "diagnostic_rejection_count": guard["diagnostic_rejection_count"],
        "row_level_input_policy": "multi_sample_manifest.samples[].pricing_signal_rows",
        "primary_horizon_ms": PRIMARY_HORIZON_MS,
        "eligible_all_horizon_row_count": len(eligible_all),
        "eligible_primary_row_count": len(primary),
        "assessed_pattern": "context_basis_mid_ticks > 0",
        "raw_context_definition": "context_basis_mid_ticks > 0 excluding missing/non-numeric basis rows",
        "clean_context_definition": "raw basis-positive context AND NOT combined_tail_risk_mask",
        "tail_risk_definition": "basis_positive_small OR hl_top5_imbalance_negative_small OR hl_microprice_minus_mid_negative_small",
        "t002_final_recommendation": prerequisites["t002_collection"].get("t002_final_recommendation"),
        "t002_decomposition_final_recommendation": prerequisites["t002_decomposition"].get("final_recommendation"),
        "t001_final_recommendation": prerequisites["t001_decomposition"].get("final_recommendation"),
        "t006_final_recommendation": prerequisites["t006"].get("final_recommendation"),
        "final_recommendation": final_recommendation,
        "final_recommendation_reason": reason,
        "allowed_final_recommendations": sorted(FINAL_RECOMMENDATIONS),
        "research_context_label_taxonomy": sorted(LABELS),
        "output_artifacts": {
            "filtered_context_viability_manifest": str(resolved_output / "filtered_context_viability_manifest.json"),
            "raw_vs_filtered_basis_positive_summary": str(resolved_output / "raw_vs_filtered_basis_positive_summary.csv"),
            "tail_risk_reject_subset_summary": str(resolved_output / "tail_risk_reject_subset_summary.csv"),
            "per_sample_filtered_context_stability": str(resolved_output / "per_sample_filtered_context_stability.csv"),
            "horizon_filtered_context_stability": str(resolved_output / "horizon_filtered_context_stability.csv"),
            "conditioning_filtered_context_summary": str(resolved_output / "conditioning_filtered_context_summary.csv"),
            "research_context_labels": str(resolved_output / "research_context_labels.csv"),
            "execution_evidence_gap_register": str(resolved_output / "execution_evidence_gap_register.md"),
            "filtered_context_next_step_recommendation": str(resolved_output / "filtered_context_next_step_recommendation.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }
    _write_csv(resolved_output / "raw_vs_filtered_basis_positive_summary.csv", summary, SUMMARY_FIELDS)
    _write_csv(
        resolved_output / "tail_risk_reject_subset_summary.csv",
        tail,
        ["tail_condition", "condition_definition"] + SUMMARY_FIELDS,
    )
    _write_csv(resolved_output / "per_sample_filtered_context_stability.csv", per_sample, ["group_name", "group_value"] + SUMMARY_FIELDS)
    _write_csv(
        resolved_output / "horizon_filtered_context_stability.csv",
        horizon,
        ["horizon_ms", "effective_future_row_delta_count"] + SUMMARY_FIELDS,
    )
    _write_csv(
        resolved_output / "conditioning_filtered_context_summary.csv",
        conditioning,
        ["conditioning_dimension", "conditioning_bucket"] + SUMMARY_FIELDS,
    )
    _write_csv(
        resolved_output / "research_context_labels.csv",
        labels,
        [
            "research_context_label",
            "read_only_context_only",
            "definition",
            "row_count",
            "sample_count",
            "allowed_use",
            "forbidden_use",
        ],
    )
    _write_json(resolved_output / "filtered_context_viability_manifest.json", manifest)
    _gap_register(resolved_output / "execution_evidence_gap_register.md")
    _recommendation_doc(resolved_output / "filtered_context_next_step_recommendation.md", manifest, summary)
    return {
        "manifest": manifest,
        "summary_rows": summary,
        "tail_rows": tail,
        "per_sample_rows": per_sample,
        "horizon_rows": horizon,
        "conditioning_rows": conditioning,
        "label_rows": labels,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--t002-collection-dir", type=Path, default=DEFAULT_T002_COLLECTION_DIR)
    parser.add_argument("--t002-decomposition-dir", type=Path, default=DEFAULT_T002_DECOMPOSITION_DIR)
    parser.add_argument("--t001-decomposition-dir", type=Path, default=DEFAULT_T001_DECOMPOSITION_DIR)
    parser.add_argument("--t006-dir", type=Path, default=DEFAULT_T006_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_filtered_context_viability(
        input_dir=args.input_dir,
        t002_collection_dir=args.t002_collection_dir,
        t002_decomposition_dir=args.t002_decomposition_dir,
        t001_decomposition_dir=args.t001_decomposition_dir,
        t006_dir=args.t006_dir,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "task_id": TASK_ID,
                "final_recommendation": result["manifest"]["final_recommendation"],
                "output_dir": str(_expand(args.output_dir)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
