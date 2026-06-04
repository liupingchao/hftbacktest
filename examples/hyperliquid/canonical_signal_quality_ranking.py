#!/usr/bin/env python3
"""Rank canonical event-mode Binance-led Hyperliquid pricing signals.

This task-scoped runner consumes only accepted local canonical event-mode
evidence. It produces read-only research rankings and does not collect data,
select regimes, construct cases, generate shadow decisions, touch private/order
endpoints, implement strategy behavior, run parameter search, enable live or
default-on behavior, or make promotion claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import canonical_event_mode_evidence as canonical_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0604T006"
SCHEMA_VERSION = "canonical_signal_quality_ranking_v1"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_CONTRACT_DIR = PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_data_contract_0601T004"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_signal_quality_ranking_0604T006"

EXPECTED_PRIMARY_ALLOWLIST = [
    "binance_top5_imbalance",
    "binance_microprice_minus_mid_ticks",
    "binance_mid_move_ticks_from_prev",
    "binance_top5_bid_qty",
]

ALLOWED_BUCKETS = {
    "keep_for_read_only_research",
    "watch_regime_dependent",
    "reject_for_canonical_signal_ranking",
}

BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "no_regime_selection": True,
    "no_case_library": True,
    "no_shadow_decision_generation": True,
    "no_private_keys": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_implementation": True,
    "no_live_trading_bot": True,
    "no_parameter_search": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_promotion": True,
    "no_schema_or_connector_or_core_api_change": True,
}

CONTROLLER_INTERPRETATION = {
    "binance_mid_move_ticks_from_prev": "most stable global signal candidate",
    "binance_top5_imbalance": "strong book-pressure candidate",
    "binance_top5_bid_qty": "useful liquidity/context signal, not a simple global directional signal",
    "binance_microprice_minus_mid_ticks": "more regime-dependent and not a simple global signal",
}


class SignalQualityRankingError(ValueError):
    """Raised when ranking inputs are outside the canonical evidence boundary."""


@dataclass
class FeatureSummary:
    feature: str
    evidence_row_count: int
    canonical_sample_count: int
    usable_primary_row_count: int
    total_label_row_count: int
    stable_row_count: int
    stable_row_ratio: float
    direction_consistency_mean: float
    direction_consistency_long_mean: float
    effect_size_mean_abs: float
    effect_size_long_mean_abs: float
    mean_abs_corr: float
    mean_abs_corr_long: float
    independent_future_row_delta_mean: float
    independent_future_row_delta_long_mean: float
    mature_horizon_row_ratio: float
    very_long_horizon_row_ratio: float
    stable_mature_row_ratio: float
    stable_very_long_row_ratio: float
    short_horizon_reliance_ratio: float
    sample_balance_score: float
    max_single_sample_dominance: float
    single_sample_concentration_penalty: float
    short_horizon_reliance_penalty: float
    normalized_effect_score: float
    normalized_corr_score: float
    ranking_score: float
    bucket: str
    rank: int
    primary_reason: str
    controller_interpretation: str
    controller_alignment: str


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


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


def _float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else 0.0


def _fmt(value: float, places: int = 8) -> str:
    if not math.isfinite(value):
        return ""
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _load_primary_allowlist(contract_dir: str | Path) -> list[str]:
    resolved = _expand(contract_dir)
    rows = _read_csv(resolved / "feature_decision_table.csv")
    allowlist = [
        row["feature"]
        for row in rows
        if row.get("decision") == "allow" and row.get("status") == "primary_allowlist"
    ]
    if allowlist != EXPECTED_PRIMARY_ALLOWLIST:
        raise SignalQualityRankingError(
            "primary allowlist must match the accepted 0601T004 contract exactly: "
            f"expected={EXPECTED_PRIMARY_ALLOWLIST}, actual={allowlist}"
        )
    return allowlist


def _sample_effect_dominance(sample_effects: str) -> float | None:
    effects: list[float] = []
    for part in sample_effects.split("|"):
        if not part:
            continue
        fields = part.rsplit(":", 1)
        if len(fields) != 2:
            continue
        effects.append(abs(_float(fields[1])))
    total = sum(effects)
    if not effects or total <= 0:
        return None
    return max(effects) / total


def _balance_from_dominance(dominance: float | None, sample_count: int) -> float:
    if dominance is None or sample_count <= 1:
        return 0.0
    ideal = 1.0 / sample_count
    if dominance <= ideal:
        return 1.0
    return max(0.0, min(1.0, (1.0 - dominance) / (1.0 - ideal)))


def _evidence_strength(row: dict[str, str]) -> float:
    direction = _float(row.get("direction_consistency_ratio"))
    corr = abs(_float(row.get("mean_abs_corr")))
    effect = math.log1p(abs(_float(row.get("mean_high_minus_low_effect"))))
    stable_bonus = 1.0 if row.get("stability_verdict") == "stable_across_samples" else 0.35
    return stable_bonus * max(0.05, direction) * (corr + 0.08 * effect)


def _controller_alignment(feature: str, bucket: str, rank: int, stable_row_ratio: float) -> str:
    if feature == "binance_mid_move_ticks_from_prev":
        if bucket == "keep_for_read_only_research" and rank <= 2 and stable_row_ratio == 1.0:
            return "matches_current_controller_interpretation"
        return "partly_contradicts_current_controller_interpretation"
    if feature == "binance_top5_imbalance":
        if bucket != "reject_for_canonical_signal_ranking":
            return "matches_current_controller_interpretation"
        return "contradicts_current_controller_interpretation"
    if feature == "binance_top5_bid_qty":
        if bucket == "watch_regime_dependent":
            return "matches_current_controller_interpretation"
        return "partly_contradicts_current_controller_interpretation"
    if feature == "binance_microprice_minus_mid_ticks":
        if bucket == "watch_regime_dependent":
            return "matches_current_controller_interpretation"
        return "partly_contradicts_current_controller_interpretation"
    return "not_in_controller_interpretation"


def _classify_feature(summary: dict[str, float], feature: str) -> tuple[str, str]:
    reasons: list[str] = []
    if summary["canonical_sample_count"] < 2:
        reasons.append("fewer than two canonical samples")
    if summary["stable_mature_row_ratio"] < 0.35:
        reasons.append("weak stable 500ms+ canonical evidence")
    if summary["ranking_score"] < 0.45:
        reasons.append("low aggregate ranking score")
    if reasons:
        return "reject_for_canonical_signal_ranking", "; ".join(reasons)

    watch_reasons: list[str] = []
    if summary["stable_mature_row_ratio"] < 0.80:
        watch_reasons.append("500ms+ stability is below keep threshold")
    if summary["stable_very_long_row_ratio"] < 0.70:
        watch_reasons.append("1000ms+ stability is below keep threshold")
    if summary["short_horizon_reliance_ratio"] > 0.38:
        watch_reasons.append("evidence has material 100/250ms reliance")
    if summary["max_single_sample_dominance"] > 0.78:
        watch_reasons.append("sample effect is materially single-sample dominated")
    if summary["ranking_score"] < 0.76:
        watch_reasons.append("aggregate score remains below keep threshold")
    if feature == "binance_top5_bid_qty":
        watch_reasons.append("contract/controller treats bid quantity as liquidity/context")
    if feature == "binance_microprice_minus_mid_ticks":
        watch_reasons.append("controller interpretation is regime-dependent microprice dislocation")
    if watch_reasons:
        return "watch_regime_dependent", "; ".join(dict.fromkeys(watch_reasons))
    return "keep_for_read_only_research", "stable multi-sample canonical evidence at 500ms+ and 1000ms+"


def _summarize_feature_rows(
    *,
    feature_rows: list[dict[str, str]],
    allowlist: list[str],
    canonical_sample_count: int,
    usable_primary_row_count: int,
) -> list[FeatureSummary]:
    rows_by_feature: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in feature_rows:
        feature = row.get("feature", "")
        if feature in allowlist:
            if _int(row.get("diagnostic_synthetic_sample_count")) != 0:
                raise SignalQualityRankingError(
                    f"feature row for {feature} contains diagnostic synthetic sample count"
                )
            rows_by_feature[feature].append(row)

    missing = [feature for feature in allowlist if not rows_by_feature.get(feature)]
    if missing:
        raise SignalQualityRankingError(f"canonical evidence is missing allowlist features: {missing}")

    raw: list[dict[str, Any]] = []
    for feature in allowlist:
        rows = rows_by_feature[feature]
        mature_rows = [row for row in rows if _int(row.get("horizon_ms")) >= 500]
        very_long_rows = [row for row in rows if _int(row.get("horizon_ms")) >= 1000]
        short_rows = [row for row in rows if _int(row.get("horizon_ms")) in {100, 250}]
        stable_rows = [row for row in rows if row.get("stability_verdict") == "stable_across_samples"]
        stable_mature_rows = [row for row in mature_rows if row.get("stability_verdict") == "stable_across_samples"]
        stable_very_long_rows = [
            row for row in very_long_rows if row.get("stability_verdict") == "stable_across_samples"
        ]
        dominance_values = [
            dominance
            for dominance in (_sample_effect_dominance(row.get("sample_effects", "")) for row in rows)
            if dominance is not None
        ]
        max_dominance = max(dominance_values) if dominance_values else 1.0
        balance_values = [
            _balance_from_dominance(dominance, canonical_sample_count) for dominance in dominance_values
        ]
        short_strength = sum(_evidence_strength(row) for row in short_rows)
        mature_strength = sum(_evidence_strength(row) for row in mature_rows)
        total_strength = short_strength + mature_strength
        raw.append(
            {
                "feature": feature,
                "evidence_row_count": len(rows),
                "canonical_sample_count": canonical_sample_count,
                "usable_primary_row_count": usable_primary_row_count,
                "total_label_row_count": sum(_int(row.get("total_row_count")) for row in rows),
                "stable_row_count": len(stable_rows),
                "stable_row_ratio": len(stable_rows) / len(rows),
                "direction_consistency_mean": _mean(
                    [_float(row.get("direction_consistency_ratio")) for row in rows]
                ),
                "direction_consistency_long_mean": _mean(
                    [_float(row.get("direction_consistency_ratio")) for row in mature_rows]
                ),
                "effect_size_mean_abs": _mean(
                    [abs(_float(row.get("mean_high_minus_low_effect"))) for row in rows]
                ),
                "effect_size_long_mean_abs": _mean(
                    [abs(_float(row.get("mean_high_minus_low_effect"))) for row in mature_rows]
                ),
                "mean_abs_corr": _mean([abs(_float(row.get("mean_abs_corr"))) for row in rows]),
                "mean_abs_corr_long": _mean([abs(_float(row.get("mean_abs_corr"))) for row in mature_rows]),
                "independent_future_row_delta_mean": _mean(
                    [_float(row.get("canonical_independent_future_row_delta_count")) for row in rows]
                ),
                "independent_future_row_delta_long_mean": _mean(
                    [_float(row.get("canonical_independent_future_row_delta_count")) for row in mature_rows]
                ),
                "mature_horizon_row_ratio": len(mature_rows) / len(rows),
                "very_long_horizon_row_ratio": len(very_long_rows) / len(rows),
                "stable_mature_row_ratio": len(stable_mature_rows) / max(1, len(mature_rows)),
                "stable_very_long_row_ratio": len(stable_very_long_rows) / max(1, len(very_long_rows)),
                "short_horizon_reliance_ratio": short_strength / total_strength if total_strength else 1.0,
                "sample_balance_score": _mean(balance_values),
                "max_single_sample_dominance": max_dominance,
            }
        )

    max_effect = max(math.log1p(row["effect_size_long_mean_abs"]) for row in raw) or 1.0
    max_corr = max(row["mean_abs_corr_long"] for row in raw) or 1.0

    summaries: list[FeatureSummary] = []
    for row in raw:
        normalized_effect = math.log1p(row["effect_size_long_mean_abs"]) / max_effect
        normalized_corr = row["mean_abs_corr_long"] / max_corr
        concentration_penalty = max(0.0, row["max_single_sample_dominance"] - 0.58) * 0.55
        short_penalty = max(0.0, row["short_horizon_reliance_ratio"] - 0.28) * 0.45
        score = (
            0.25 * row["stable_row_ratio"]
            + 0.18 * row["direction_consistency_long_mean"]
            + 0.16 * normalized_corr
            + 0.10 * normalized_effect
            + 0.12 * min(1.0, row["independent_future_row_delta_long_mean"] / 3.0)
            + 0.12 * row["stable_very_long_row_ratio"]
            + 0.07 * row["sample_balance_score"]
            - concentration_penalty
            - short_penalty
        )
        row["normalized_effect_score"] = normalized_effect
        row["normalized_corr_score"] = normalized_corr
        row["single_sample_concentration_penalty"] = concentration_penalty
        row["short_horizon_reliance_penalty"] = short_penalty
        row["ranking_score"] = max(0.0, min(1.0, score))

    raw.sort(
        key=lambda item: (
            item["ranking_score"],
            item["stable_row_ratio"],
            item["direction_consistency_long_mean"],
            item["mean_abs_corr_long"],
        ),
        reverse=True,
    )

    for rank, row in enumerate(raw, start=1):
        bucket, reason = _classify_feature(row, row["feature"])
        summaries.append(
            FeatureSummary(
                feature=row["feature"],
                evidence_row_count=row["evidence_row_count"],
                canonical_sample_count=row["canonical_sample_count"],
                usable_primary_row_count=row["usable_primary_row_count"],
                total_label_row_count=row["total_label_row_count"],
                stable_row_count=row["stable_row_count"],
                stable_row_ratio=row["stable_row_ratio"],
                direction_consistency_mean=row["direction_consistency_mean"],
                direction_consistency_long_mean=row["direction_consistency_long_mean"],
                effect_size_mean_abs=row["effect_size_mean_abs"],
                effect_size_long_mean_abs=row["effect_size_long_mean_abs"],
                mean_abs_corr=row["mean_abs_corr"],
                mean_abs_corr_long=row["mean_abs_corr_long"],
                independent_future_row_delta_mean=row["independent_future_row_delta_mean"],
                independent_future_row_delta_long_mean=row["independent_future_row_delta_long_mean"],
                mature_horizon_row_ratio=row["mature_horizon_row_ratio"],
                very_long_horizon_row_ratio=row["very_long_horizon_row_ratio"],
                stable_mature_row_ratio=row["stable_mature_row_ratio"],
                stable_very_long_row_ratio=row["stable_very_long_row_ratio"],
                short_horizon_reliance_ratio=row["short_horizon_reliance_ratio"],
                sample_balance_score=row["sample_balance_score"],
                max_single_sample_dominance=row["max_single_sample_dominance"],
                single_sample_concentration_penalty=row["single_sample_concentration_penalty"],
                short_horizon_reliance_penalty=row["short_horizon_reliance_penalty"],
                normalized_effect_score=row["normalized_effect_score"],
                normalized_corr_score=row["normalized_corr_score"],
                ranking_score=row["ranking_score"],
                bucket=bucket,
                rank=rank,
                primary_reason=reason,
                controller_interpretation=CONTROLLER_INTERPRETATION.get(row["feature"], ""),
                controller_alignment=_controller_alignment(
                    row["feature"],
                    bucket,
                    rank,
                    row["stable_row_ratio"],
                ),
            )
        )
    return summaries


def _summary_to_row(summary: FeatureSummary) -> dict[str, Any]:
    return {
        "rank": summary.rank,
        "feature": summary.feature,
        "bucket": summary.bucket,
        "ranking_score": _fmt(summary.ranking_score),
        "canonical_sample_count": summary.canonical_sample_count,
        "usable_primary_row_count": summary.usable_primary_row_count,
        "evidence_row_count": summary.evidence_row_count,
        "stable_row_count": summary.stable_row_count,
        "stable_row_ratio": _fmt(summary.stable_row_ratio),
        "direction_consistency_mean": _fmt(summary.direction_consistency_mean),
        "direction_consistency_500ms_plus_mean": _fmt(summary.direction_consistency_long_mean),
        "effect_size_mean_abs": _fmt(summary.effect_size_mean_abs),
        "effect_size_500ms_plus_mean_abs": _fmt(summary.effect_size_long_mean_abs),
        "mean_abs_corr": _fmt(summary.mean_abs_corr),
        "mean_abs_corr_500ms_plus": _fmt(summary.mean_abs_corr_long),
        "independent_future_row_delta_mean": _fmt(summary.independent_future_row_delta_mean),
        "independent_future_row_delta_500ms_plus_mean": _fmt(
            summary.independent_future_row_delta_long_mean
        ),
        "stable_500ms_plus_row_ratio": _fmt(summary.stable_mature_row_ratio),
        "stable_1000ms_plus_row_ratio": _fmt(summary.stable_very_long_row_ratio),
        "short_horizon_100_250ms_reliance_ratio": _fmt(summary.short_horizon_reliance_ratio),
        "sample_balance_score": _fmt(summary.sample_balance_score),
        "max_single_sample_dominance": _fmt(summary.max_single_sample_dominance),
        "single_sample_concentration_penalty": _fmt(summary.single_sample_concentration_penalty),
        "short_horizon_reliance_penalty": _fmt(summary.short_horizon_reliance_penalty),
        "primary_reason": summary.primary_reason,
        "controller_interpretation": summary.controller_interpretation,
        "controller_alignment": summary.controller_alignment,
    }


RANKING_FIELDS = [
    "rank",
    "feature",
    "bucket",
    "ranking_score",
    "canonical_sample_count",
    "usable_primary_row_count",
    "evidence_row_count",
    "stable_row_count",
    "stable_row_ratio",
    "direction_consistency_mean",
    "direction_consistency_500ms_plus_mean",
    "effect_size_mean_abs",
    "effect_size_500ms_plus_mean_abs",
    "mean_abs_corr",
    "mean_abs_corr_500ms_plus",
    "independent_future_row_delta_mean",
    "independent_future_row_delta_500ms_plus_mean",
    "stable_500ms_plus_row_ratio",
    "stable_1000ms_plus_row_ratio",
    "short_horizon_100_250ms_reliance_ratio",
    "sample_balance_score",
    "max_single_sample_dominance",
    "single_sample_concentration_penalty",
    "short_horizon_reliance_penalty",
    "primary_reason",
    "controller_interpretation",
    "controller_alignment",
]

LIST_FIELDS = [
    "feature",
    "bucket",
    "rank",
    "ranking_score",
    "primary_reason",
    "controller_interpretation",
    "controller_alignment",
]


def _write_report(path: Path, *, input_dir: Path, output_dir: Path, summaries: list[FeatureSummary]) -> None:
    lines = [
        "# Canonical Signal Quality Ranking Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Output directory: `{output_dir}`",
        "- Inputs are existing local `0604T003` canonical event-mode artifacts loaded through `0604T004` foundation.",
        "- Ranking is limited to the four `0601T004` primary Binance lead allowlist features.",
        "",
        "## Ranking Method",
        "",
        "- Score dimensions: direction consistency, effect size, mean absolute correlation, usable row/sample count, independent future-row-delta count, sample concentration, and 100/250ms versus 500ms+/1000ms+ reliance.",
        "- `100/250ms` evidence is tracked as weakly independent context; keep decisions require stable 500ms+ and 1000ms+ canonical evidence.",
        "- Buckets are limited to `keep_for_read_only_research`, `watch_regime_dependent`, and `reject_for_canonical_signal_ranking`.",
        "",
        "## Feature Ranking",
        "",
        "| Rank | Feature | Bucket | Score | Reason | Controller Alignment |",
        "|---:|---|---|---:|---|---|",
    ]
    for summary in summaries:
        lines.append(
            "| "
            f"{summary.rank} | `{summary.feature}` | `{summary.bucket}` | {_fmt(summary.ranking_score, 4)} | "
            f"{summary.primary_reason} | `{summary.controller_alignment}` |"
        )
    lines.extend(
        [
            "",
            "## Keep / Watch / Reject",
            "",
        ]
    )
    for bucket in [
        "keep_for_read_only_research",
        "watch_regime_dependent",
        "reject_for_canonical_signal_ranking",
    ]:
        members = [summary for summary in summaries if summary.bucket == bucket]
        lines.append(f"### {bucket}")
        lines.append("")
        if not members:
            lines.append("- None")
        for summary in members:
            lines.append(f"- `{summary.feature}`: {summary.primary_reason}")
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "- This is read-only signal quality ranking evidence only.",
            "- It does not authorize regime selection, case-library construction, shadow decisions, strategy implementation, private/order endpoints, order lifecycle, parameter search, live/default-on/tiny-live, or promotion.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_signal_quality_ranking_artifacts(
    *,
    input_dir: str | Path,
    contract_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_contract = _expand(contract_dir)
    resolved_output = _expand(output_dir)
    allowlist = _load_primary_allowlist(resolved_contract)
    evidence = canonical_loader.load_canonical_event_mode_evidence(input_dir=resolved_input)
    if evidence["diagnostic_rejections"]:
        rejected_ids = [row["sample_id"] for row in evidence["diagnostic_rejections"]]
        raise SignalQualityRankingError(
            "ranking input contains non-canonical or diagnostic-only samples: "
            f"{rejected_ids}"
        )
    canonical_samples = evidence["canonical_samples"]
    if not canonical_samples:
        raise SignalQualityRankingError("ranking input has zero canonical event-mode samples")
    usable_primary_row_count = sum(
        _int(row.get("primary_usable_row_count")) for row in evidence["canonical_quality_summary"]
    )
    summaries = _summarize_feature_rows(
        feature_rows=evidence["feature_horizon_stability_rows"],
        allowlist=allowlist,
        canonical_sample_count=len(canonical_samples),
        usable_primary_row_count=usable_primary_row_count,
    )
    ranking_rows = [_summary_to_row(summary) for summary in summaries]
    reject_watch_rows = [
        {
            "feature": summary.feature,
            "bucket": summary.bucket,
            "rank": summary.rank,
            "ranking_score": _fmt(summary.ranking_score),
            "primary_reason": summary.primary_reason,
            "controller_interpretation": summary.controller_interpretation,
            "controller_alignment": summary.controller_alignment,
        }
        for summary in summaries
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "contract_dir": str(resolved_contract),
        "output_dir": str(resolved_output),
        "canonical_loader_schema_version": canonical_loader.SCHEMA_VERSION,
        "canonical_sample_count": len(canonical_samples),
        "diagnostic_rejection_count": len(evidence["diagnostic_rejections"]),
        "primary_allowlist": allowlist,
        "bucket_counts": {
            bucket: sum(1 for summary in summaries if summary.bucket == bucket)
            for bucket in sorted(ALLOWED_BUCKETS)
        },
        "ranked_features": [
            {
                "rank": summary.rank,
                "feature": summary.feature,
                "bucket": summary.bucket,
                "ranking_score": summary.ranking_score,
                "controller_alignment": summary.controller_alignment,
            }
            for summary in summaries
        ],
        "source_artifacts": {name: str(path) for name, path in evidence["source_paths"].items()},
        "output_artifacts": {
            "signal_quality_ranking": str(resolved_output / "signal_quality_ranking.csv"),
            "signal_quality_reject_watch_list": str(
                resolved_output / "signal_quality_reject_watch_list.csv"
            ),
            "signal_quality_ranking_manifest": str(
                resolved_output / "signal_quality_ranking_manifest.json"
            ),
            "signal_quality_ranking_report": str(
                resolved_output / "signal_quality_ranking_report.md"
            ),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }
    _write_csv(resolved_output / "signal_quality_ranking.csv", ranking_rows, RANKING_FIELDS)
    _write_csv(
        resolved_output / "signal_quality_reject_watch_list.csv",
        reject_watch_rows,
        LIST_FIELDS,
    )
    _write_json(resolved_output / "signal_quality_ranking_manifest.json", manifest)
    _write_report(
        resolved_output / "signal_quality_ranking_report.md",
        input_dir=resolved_input,
        output_dir=resolved_output,
        summaries=summaries,
    )
    return {
        "manifest": manifest,
        "ranking_rows": ranking_rows,
        "reject_watch_rows": reject_watch_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank canonical event-mode signal quality.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--contract-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_signal_quality_ranking_artifacts(
        input_dir=args.input_dir,
        contract_dir=args.contract_dir,
        output_dir=args.output_dir,
    )
    print(f"ranked_features={len(result['ranking_rows'])}")
    for row in result["ranking_rows"]:
        print(f"{row['rank']} {row['feature']} {row['bucket']} score={row['ranking_score']}")
    print(f"wrote {Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
