#!/usr/bin/env python3
"""Read-only signal quality ranking for canonical event-mode evidence.

This task-scoped runner consumes only the canonical event-mode aggregate guarded
by ``canonical_event_mode_evidence.py``. Outputs are research rankings only; they
do not select final regimes, define maker actions, implement strategy behavior,
touch private/order flows, enable live/default-on/tiny-live behavior, run
parameter search, or make promotion claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import canonical_event_mode_evidence as canonical_loader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0604T006"
SCHEMA_VERSION = "canonical_signal_quality_ranking_v1"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_signal_quality_ranking_0604T006"
DEFAULT_ALLOWLIST_CSV = (
    PROJECT_ROOT / "local_live_analysis" / "binance_led_hyperliquid_data_contract_0601T004" / "feature_decision_table.csv"
)

DEFAULT_FEATURES = [
    "binance_top5_imbalance",
    "binance_microprice_minus_mid_ticks",
    "binance_mid_move_ticks_from_prev",
    "binance_top5_bid_qty",
]
PREFERRED_HORIZON_MS = 1000
MIN_FORMAL_HORIZON_MS = 500
ALLOWED_FINAL_BUCKETS = {
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


class RankingInputError(ValueError):
    """Raised when ranking inputs are outside the canonical read-only boundary."""


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _format_float(value: float | None, places: int = 8) -> str:
    if value is None or not math.isfinite(value):
        return ""
    text = f"{value:.{places}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _mean(values: list[float]) -> float | None:
    clean = [value for value in values if math.isfinite(value)]
    return statistics.fmean(clean) if clean else None


def _feature_allowlist(allowlist_csv: Path) -> list[str]:
    if not allowlist_csv.exists():
        return list(DEFAULT_FEATURES)
    rows = _read_csv(allowlist_csv)
    feature_field = "feature" if rows and "feature" in rows[0] else "feature_name"
    out = [
        row.get(feature_field, "")
        for row in rows
        if row.get("decision") == "allow" and row.get("status") == "primary_allowlist"
    ]
    filtered = [feature for feature in out if feature in DEFAULT_FEATURES]
    return filtered or list(DEFAULT_FEATURES)


def _sample_effect_concentration(sample_effects: str) -> float:
    effects: list[float] = []
    for part in (sample_effects or "").split("|"):
        if ":" not in part:
            continue
        raw = part.rsplit(":", 1)[-1]
        value = _as_float(raw)
        if value is not None:
            effects.append(abs(value))
    total = sum(effects)
    return max(effects) / total if total > 0 else 0.0


def _require_canonical_only(guard_result: dict[str, Any]) -> dict[str, Any]:
    manifest = guard_result["canonical_source_lock_manifest"]
    canonical_loader.validate_canonical_source_lock_manifest(manifest, require_formal_evidence=True)
    loaded = guard_result["loaded_evidence"]
    if guard_result["canonical_sample_count"] <= 0:
        raise RankingInputError("signal quality ranking requires canonical_sample_count > 0")
    if guard_result["diagnostic_rejection_count"] != 0:
        raise RankingInputError("signal quality ranking refuses mixed or diagnostic-only inputs")
    for sample in loaded["samples"]:
        if not sample.get("canonical_sample"):
            raise RankingInputError(f"non-canonical sample in formal ranking input: {sample.get('sample_id', '')}")
    return loaded


def _aggregate_feature_rows(rows: list[dict[str, str]], *, features: list[str]) -> list[dict[str, Any]]:
    by_feature: dict[str, list[dict[str, str]]] = {
        feature: [row for row in rows if row.get("feature") == feature]
        for feature in features
    }
    max_effect_1000 = max(
        (
            abs(_as_float(row.get("mean_high_minus_low_effect")) or 0.0)
            for rowset in by_feature.values()
            for row in rowset
            if _as_int(row.get("horizon_ms")) >= PREFERRED_HORIZON_MS
        ),
        default=1.0,
    )
    max_corr_1000 = max(
        (
            _as_float(row.get("mean_abs_corr")) or 0.0
            for rowset in by_feature.values()
            for row in rowset
            if _as_int(row.get("horizon_ms")) >= PREFERRED_HORIZON_MS
        ),
        default=1.0,
    )
    ranking_rows: list[dict[str, Any]] = []
    for feature, feature_rows in by_feature.items():
        if not feature_rows:
            ranking_rows.append(
                {
                    "feature": feature,
                    "final_bucket": "reject_for_canonical_signal_ranking",
                    "ranking_score": "0",
                    "reason": "feature_missing_from_canonical_evidence",
                }
            )
            continue
        rows_500 = [row for row in feature_rows if _as_int(row.get("horizon_ms")) >= MIN_FORMAL_HORIZON_MS]
        rows_1000 = [row for row in feature_rows if _as_int(row.get("horizon_ms")) >= PREFERRED_HORIZON_MS]
        rows_short = [row for row in feature_rows if _as_int(row.get("horizon_ms")) < MIN_FORMAL_HORIZON_MS]
        stable_500 = sum(row.get("stability_verdict") == "stable_across_samples" for row in rows_500)
        stable_1000 = sum(row.get("stability_verdict") == "stable_across_samples" for row in rows_1000)
        consistency_1000 = _mean([_as_float(row.get("direction_consistency_ratio")) or 0.0 for row in rows_1000]) or 0.0
        stable_ratio_1000 = stable_1000 / len(rows_1000) if rows_1000 else 0.0
        stable_ratio_500 = stable_500 / len(rows_500) if rows_500 else 0.0
        mean_effect_1000 = _mean(
            [abs(_as_float(row.get("mean_high_minus_low_effect")) or 0.0) for row in rows_1000]
        ) or 0.0
        mean_corr_1000 = _mean([_as_float(row.get("mean_abs_corr")) or 0.0 for row in rows_1000]) or 0.0
        independent_1000 = _mean(
            [_as_float(row.get("canonical_independent_future_row_delta_count")) or 0.0 for row in rows_1000]
        ) or 0.0
        short_stable = sum(row.get("stability_verdict") == "stable_across_samples" for row in rows_short)
        short_reliance_ratio = short_stable / (short_stable + stable_500) if (short_stable + stable_500) else 0.0
        concentration_max = max((_sample_effect_concentration(row.get("sample_effects", "")) for row in feature_rows), default=0.0)
        direction_counts = Counter(row.get("majority_direction", "") for row in rows_1000 if row.get("majority_direction"))

        effect_component = mean_effect_1000 / max_effect_1000 if max_effect_1000 > 0 else 0.0
        corr_component = mean_corr_1000 / max_corr_1000 if max_corr_1000 > 0 else 0.0
        independent_component = min(independent_1000 / 4.0, 1.0)
        concentration_penalty = max(0.0, concentration_max - 0.70) * 0.60
        short_penalty = short_reliance_ratio * 0.05
        score = (
            0.45 * consistency_1000
            + 0.25 * stable_ratio_1000
            + 0.15 * effect_component
            + 0.10 * corr_component
            + 0.05 * independent_component
            - concentration_penalty
            - short_penalty
        )
        score = max(0.0, min(score, 1.0))

        if stable_ratio_1000 >= 0.95 and consistency_1000 >= 0.95 and independent_1000 >= 3:
            final_bucket = "keep_for_read_only_research"
        elif stable_ratio_500 >= 0.50 and consistency_1000 >= 0.75:
            final_bucket = "watch_regime_dependent"
        else:
            final_bucket = "reject_for_canonical_signal_ranking"

        reason_parts = [
            f"stable_1000_plus={stable_1000}/{len(rows_1000)}",
            f"consistency_1000_plus={_format_float(consistency_1000)}",
            f"independent_future_row_delta_1000_plus={_format_float(independent_1000, places=2)}",
        ]
        if concentration_max > 0.70:
            reason_parts.append(f"sample_concentration_watch={_format_float(concentration_max)}")
        if short_reliance_ratio > 0.35:
            reason_parts.append(f"short_horizon_reliance_watch={_format_float(short_reliance_ratio)}")

        ranking_rows.append(
            {
                "feature": feature,
                "final_bucket": final_bucket,
                "ranking_score": _format_float(score),
                "rank": "",
                "rows_total": len(feature_rows),
                "rows_500ms_plus": len(rows_500),
                "rows_1000ms_plus": len(rows_1000),
                "stable_rows_500ms_plus": stable_500,
                "stable_rows_1000ms_plus": stable_1000,
                "stable_ratio_500ms_plus": _format_float(stable_ratio_500),
                "stable_ratio_1000ms_plus": _format_float(stable_ratio_1000),
                "direction_consistency_1000ms_plus": _format_float(consistency_1000),
                "majority_directions_1000ms_plus": "|".join(f"{key}:{value}" for key, value in sorted(direction_counts.items())),
                "mean_abs_effect_1000ms_plus": _format_float(mean_effect_1000),
                "mean_abs_corr_1000ms_plus": _format_float(mean_corr_1000),
                "independent_future_row_delta_1000ms_plus": _format_float(independent_1000, places=2),
                "short_horizon_reliance_ratio": _format_float(short_reliance_ratio),
                "max_sample_effect_concentration": _format_float(concentration_max),
                "reason": "; ".join(reason_parts),
            }
        )
    ranking_rows.sort(key=lambda row: float(row.get("ranking_score") or 0.0), reverse=True)
    for index, row in enumerate(ranking_rows, start=1):
        row["rank"] = index
    return ranking_rows


def _build_watch_rows(ranking_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in ranking_rows:
        if row["final_bucket"] == "keep_for_read_only_research":
            disposition = "keep"
        elif row["final_bucket"] == "watch_regime_dependent":
            disposition = "watch"
        else:
            disposition = "reject"
        rows.append(
            {
                "feature": row["feature"],
                "rank": row["rank"],
                "disposition": disposition,
                "final_bucket": row["final_bucket"],
                "reason": row["reason"],
            }
        )
    return rows


def _write_report(path: Path, *, ranking_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    interpretation_labels = {
        "keep_for_read_only_research": "keep for read-only research",
        "watch_regime_dependent": "watch/regime-dependent",
        "reject_for_canonical_signal_ranking": "reject for canonical signal ranking",
    }
    interpretation_notes = {
        "binance_mid_move_ticks_from_prev": "current strongest global signal candidate under preferred canonical horizons",
        "binance_top5_imbalance": "book-pressure candidate with concentration/short-horizon caveats when present",
        "binance_top5_bid_qty": "liquidity/context candidate behind stronger directional candidates",
        "binance_microprice_minus_mid_ticks": "regime-dependent microprice context candidate",
    }
    lines = [
        "# Canonical Signal Quality Ranking Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        f"- Input directory: `{manifest['input_dir']}`",
        f"- Output directory: `{manifest['output_dir']}`",
        "- Inputs are existing `0604T003` canonical event-mode aggregate artifacts only.",
        "- `100/250ms` evidence is treated as weakly independent; ranking prioritizes `500ms+` and especially `1000ms+` rows.",
        "",
        "## Ranking",
        "",
    ]
    for row in ranking_rows:
        lines.append(
            f"- Rank `{row['rank']}` `{row['feature']}`: `{row['final_bucket']}`, "
            f"score `{row['ranking_score']}`; {row['reason']}"
        )
    lines.extend(
        [
            "",
            "## Controller Interpretation Check",
            "",
        ]
    )
    for row in ranking_rows:
        bucket = str(row["final_bucket"])
        bucket_label = interpretation_labels.get(bucket, bucket)
        note = interpretation_notes.get(str(row["feature"]), "canonical allowlist feature")
        lines.append(
            f"- `{row['feature']}` is `{bucket}` ({bucket_label}); "
            f"rank `{row['rank']}`, score `{row['ranking_score']}`; {note}; {row['reason']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is read-only signal quality ranking only.",
            "- No new data collection, regime selection, case-library construction, shadow decision generation, strategy implementation, private/order endpoints, order lifecycle, parameter search, live/default-on/tiny-live, or promotion is authorized.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_signal_quality_ranking(
    *,
    input_dir: str | Path,
    output_dir: str | Path,
    allowlist_csv: str | Path = DEFAULT_ALLOWLIST_CSV,
) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_output = _expand(output_dir)
    features = _feature_allowlist(_expand(allowlist_csv))
    if set(features) != set(DEFAULT_FEATURES):
        raise RankingInputError(f"primary allowlist must contain exactly {DEFAULT_FEATURES}, got {features}")
    guard_result = canonical_loader.guard_canonical_event_mode_evidence(
        input_dir=resolved_input,
        require_formal_evidence=True,
        allow_diagnostic_validation=False,
    )
    loaded = _require_canonical_only(guard_result)
    unknown_features = sorted({row.get("feature", "") for row in loaded["feature_horizon_stability_rows"]} - set(features))
    ranking_rows = _aggregate_feature_rows(loaded["feature_horizon_stability_rows"], features=features)
    watch_rows = _build_watch_rows(ranking_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "output_dir": str(resolved_output),
        "canonical_sample_count": guard_result["canonical_sample_count"],
        "diagnostic_rejection_count": guard_result["diagnostic_rejection_count"],
        "ranked_feature_count": len(ranking_rows),
        "features_ranked": [row["feature"] for row in ranking_rows],
        "primary_allowlist_features": features,
        "unknown_non_allowlist_features_seen": unknown_features,
        "preferred_horizon_ms": PREFERRED_HORIZON_MS,
        "minimum_formal_horizon_ms": MIN_FORMAL_HORIZON_MS,
        "allowed_final_buckets": sorted(ALLOWED_FINAL_BUCKETS),
        "source_lock_task_id": canonical_loader.SOURCE_LOCK_TASK_ID,
        "boundary_flags": BOUNDARY_FLAGS,
    }
    resolved_output.mkdir(parents=True, exist_ok=True)
    ranking_fields = [
        "rank",
        "feature",
        "final_bucket",
        "ranking_score",
        "rows_total",
        "rows_500ms_plus",
        "rows_1000ms_plus",
        "stable_rows_500ms_plus",
        "stable_rows_1000ms_plus",
        "stable_ratio_500ms_plus",
        "stable_ratio_1000ms_plus",
        "direction_consistency_1000ms_plus",
        "majority_directions_1000ms_plus",
        "mean_abs_effect_1000ms_plus",
        "mean_abs_corr_1000ms_plus",
        "independent_future_row_delta_1000ms_plus",
        "short_horizon_reliance_ratio",
        "max_sample_effect_concentration",
        "reason",
    ]
    _write_csv(resolved_output / "signal_quality_ranking.csv", ranking_rows, ranking_fields)
    _write_csv(
        resolved_output / "signal_quality_reject_watch_list.csv",
        watch_rows,
        ["feature", "rank", "disposition", "final_bucket", "reason"],
    )
    _write_json(resolved_output / "signal_quality_ranking_manifest.json", manifest)
    _write_report(resolved_output / "signal_quality_ranking_report.md", ranking_rows=ranking_rows, manifest=manifest)
    return {
        "manifest": manifest,
        "ranking_rows": ranking_rows,
        "watch_rows": watch_rows,
        "output_dir": resolved_output,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--allowlist-csv", default=str(DEFAULT_ALLOWLIST_CSV))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = build_signal_quality_ranking(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        allowlist_csv=args.allowlist_csv,
    )
    manifest = result["manifest"]
    print(
        "canonical signal quality ranking complete: "
        f"canonical_sample_count={manifest['canonical_sample_count']} "
        f"ranked_feature_count={manifest['ranked_feature_count']} "
        f"output_dir={result['output_dir']}"
    )


if __name__ == "__main__":
    main()
