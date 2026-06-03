#!/usr/bin/env python3
"""Validate canonical event-mode evidence artifacts.

This task-scoped loader is offline/read-only. It consumes accepted aggregate
artifacts produced from Binance-led Hyperliquid public data and writes a small
canonical-only manifest for later analysis runners. It does not collect data,
touch private/order endpoints, implement strategy behavior, run parameter
search, enable live/default-on/tiny-live behavior, or make promotion claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0604T004"
SCHEMA_VERSION = "canonical_event_mode_evidence_v1"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "event_mode_canonical_pricing_signal_0604T003"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "canonical_event_mode_evidence_0604T004"

CANONICAL_DECISION_MODE = "event"
CANONICAL_EVENT_STATUS = "canonical_event_mode"
DIAGNOSTIC_SYNTHETIC_STATUS = "diagnostic_only_synthetic_decision_grid"

REQUIRED_FILES = {
    "multi_sample_manifest": "multi_sample_manifest.json",
    "sample_quality_matrix": "sample_quality_matrix.csv",
    "feature_horizon_stability_across_samples": "feature_horizon_stability_across_samples.csv",
    "effective_horizon_aliasing_by_sample": "effective_horizon_aliasing_by_sample.csv",
    "venue_state_conditioning_across_samples": "venue_state_conditioning_across_samples.csv",
}
REQUIRED_COLUMNS = {
    "sample_quality_matrix": {
        "sample_id",
        "decision_mode",
        "canonical_status",
        "pricing_signal_dir",
        "source_sample_dir",
        "input_rows",
        "primary_rows",
        "pricing_signal_rows",
        "future_join_count",
        "missing_binance_join_count",
        "primary_usable_row_count",
        "horizon_count",
        "independent_future_row_delta_count",
    },
    "feature_horizon_stability_across_samples": {
        "feature",
        "horizon_ms",
        "label",
        "sample_count",
        "eligible_sample_count",
        "canonical_eligible_sample_count",
        "canonical_independent_future_row_delta_count",
        "diagnostic_synthetic_sample_count",
        "stability_verdict",
    },
    "effective_horizon_aliasing_by_sample": {
        "sample_id",
        "decision_mode",
        "canonical_status",
        "horizon_ms",
        "row_count",
        "effective_future_age_ms_mean",
        "effective_future_row_delta_mean",
        "distinct_future_row_delta_count",
        "aliasing_status",
    },
    "venue_state_conditioning_across_samples": {
        "horizon_ms",
        "hyperliquid_context_quality",
        "hyperliquid_join_age_bucket",
        "hyperliquid_spread_bucket",
        "sample_count",
        "row_count",
        "conditioning_status",
    },
}
BOUNDARY_FLAGS = {
    "no_new_data_collection": True,
    "no_signal_ranking": True,
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


class EvidenceValidationError(ValueError):
    """Raised when canonical evidence artifacts fail structural validation."""


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceValidationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceValidationError(f"{path} must contain a JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(
    path: Path,
    required_columns: set[str],
    *,
    allow_empty_without_header: bool = False,
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if allow_empty_without_header and (reader.fieldnames is None or not any(reader.fieldnames)):
            return []
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required_columns - fieldnames)
        if missing:
            raise EvidenceValidationError(f"{path} missing required columns: {', '.join(missing)}")
        return list(reader)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _required_paths(input_dir: Path) -> dict[str, Path]:
    return {name: input_dir / filename for name, filename in REQUIRED_FILES.items()}


def _ensure_required_files(paths: dict[str, Path]) -> None:
    missing = [f"{name}: {path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required canonical evidence artifacts: " + "; ".join(missing))


def _as_int(value: Any, *, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise EvidenceValidationError(f"{field} must be an integer, got {value!r}") from exc


def _validate_sample_metadata(sample: dict[str, Any], *, source: str) -> dict[str, Any]:
    sample_id = str(sample.get("sample_id", "")).strip()
    decision_mode = str(sample.get("decision_mode", "")).strip()
    canonical_status = str(sample.get("canonical_status", "")).strip()
    if not sample_id:
        raise EvidenceValidationError(f"{source} sample is missing sample_id")
    canonical_sample = decision_mode == CANONICAL_DECISION_MODE and canonical_status == CANONICAL_EVENT_STATUS
    if decision_mode == CANONICAL_DECISION_MODE and canonical_status != CANONICAL_EVENT_STATUS:
        raise EvidenceValidationError(
            f"{source} sample {sample_id} has decision_mode=event but canonical_status={canonical_status!r}"
        )
    if canonical_status == CANONICAL_EVENT_STATUS and decision_mode != CANONICAL_DECISION_MODE:
        raise EvidenceValidationError(
            f"{source} sample {sample_id} has canonical_event_mode status but decision_mode={decision_mode!r}"
        )
    normalized = dict(sample)
    normalized["sample_id"] = sample_id
    normalized["decision_mode"] = decision_mode
    normalized["canonical_status"] = canonical_status
    normalized["canonical_sample"] = canonical_sample
    return normalized


def _manifest_samples(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    raw_samples = manifest.get("samples")
    if not isinstance(raw_samples, list):
        raise EvidenceValidationError("multi_sample_manifest.json must contain a samples list")
    samples = [
        _validate_sample_metadata(sample, source="manifest")
        for sample in raw_samples
        if isinstance(sample, dict)
    ]
    if len(samples) != len(raw_samples):
        raise EvidenceValidationError("multi_sample_manifest.json samples must all be JSON objects")
    duplicates = [sample_id for sample_id, count in Counter(sample["sample_id"] for sample in samples).items() if count > 1]
    if duplicates:
        raise EvidenceValidationError(f"duplicate manifest sample_id values are not allowed: {duplicates}")
    return samples


def _sample_quality_by_id(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    normalized_rows = [_validate_sample_metadata(row, source="sample_quality_matrix") for row in rows]
    duplicates = [
        sample_id
        for sample_id, count in Counter(row["sample_id"] for row in normalized_rows).items()
        if count > 1
    ]
    if duplicates:
        raise EvidenceValidationError(f"duplicate sample_quality_matrix sample_id values are not allowed: {duplicates}")
    return {row["sample_id"]: row for row in normalized_rows}


def _validate_cross_file_consistency(
    *,
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
    quality_by_id: dict[str, dict[str, str]],
    aliasing_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> None:
    manifest_ids = {sample["sample_id"] for sample in samples}
    quality_ids = set(quality_by_id)
    missing_quality = sorted(manifest_ids - quality_ids)
    extra_quality = sorted(quality_ids - manifest_ids)
    if missing_quality or extra_quality:
        raise EvidenceValidationError(
            "manifest/sample_quality sample mismatch: "
            f"missing_quality={missing_quality}, extra_quality={extra_quality}"
        )
    for sample in samples:
        quality = quality_by_id[sample["sample_id"]]
        for field in ("decision_mode", "canonical_status"):
            if str(quality.get(field, "")) != str(sample.get(field, "")):
                raise EvidenceValidationError(
                    f"sample {sample['sample_id']} has inconsistent {field}: "
                    f"manifest={sample.get(field)!r}, sample_quality={quality.get(field)!r}"
                )
    aliasing_sample_ids = {row["sample_id"] for row in aliasing_rows if row.get("sample_id")}
    unknown_aliasing = sorted(aliasing_sample_ids - manifest_ids)
    if unknown_aliasing:
        raise EvidenceValidationError(f"effective_horizon_aliasing_by_sample has unknown samples: {unknown_aliasing}")
    for row in aliasing_rows:
        _validate_sample_metadata(row, source="effective_horizon_aliasing_by_sample")
    manifest_sample_count = _as_int(manifest.get("sample_count"), field="manifest.sample_count")
    manifest_canonical_count = _as_int(
        manifest.get("canonical_sample_count"),
        field="manifest.canonical_sample_count",
    )
    manifest_diagnostic_count = _as_int(
        manifest.get("diagnostic_synthetic_sample_count", 0),
        field="manifest.diagnostic_synthetic_sample_count",
    )
    canonical_count = sum(1 for sample in samples if sample["canonical_sample"])
    diagnostic_synthetic_count = sum(
        1 for sample in samples if sample["canonical_status"] == DIAGNOSTIC_SYNTHETIC_STATUS
    )
    if manifest_sample_count != len(samples):
        raise EvidenceValidationError(
            f"manifest.sample_count={manifest_sample_count} does not match samples={len(samples)}"
        )
    if manifest_canonical_count != canonical_count:
        raise EvidenceValidationError(
            f"manifest.canonical_sample_count={manifest_canonical_count} does not match canonical samples={canonical_count}"
        )
    if manifest_diagnostic_count != diagnostic_synthetic_count:
        raise EvidenceValidationError(
            "manifest.diagnostic_synthetic_sample_count="
            f"{manifest_diagnostic_count} does not match diagnostic samples={diagnostic_synthetic_count}"
        )
    for row in feature_rows:
        canonical_eligible = _as_int(
            row.get("canonical_eligible_sample_count", 0),
            field="feature_horizon_stability_across_samples.canonical_eligible_sample_count",
        )
        diagnostic_eligible = _as_int(
            row.get("diagnostic_synthetic_sample_count", 0),
            field="feature_horizon_stability_across_samples.diagnostic_synthetic_sample_count",
        )
        if canonical_count == 0 and canonical_eligible != 0:
            raise EvidenceValidationError("feature stability row reports canonical eligibility when canonical sample count is zero")
        if diagnostic_synthetic_count == 0 and diagnostic_eligible != 0:
            raise EvidenceValidationError("feature stability row reports diagnostic synthetic eligibility when diagnostic sample count is zero")


def _rejection_reason(sample: dict[str, Any]) -> str:
    if sample["canonical_sample"]:
        return ""
    if sample["canonical_status"] == DIAGNOSTIC_SYNTHETIC_STATUS:
        return "diagnostic_only_synthetic_decision_grid_excluded_from_canonical_evidence"
    if sample["decision_mode"] != CANONICAL_DECISION_MODE:
        return f"non_event_decision_mode_excluded:{sample['decision_mode']}"
    return f"non_canonical_status_excluded:{sample['canonical_status']}"


def _build_quality_summary(canonical_samples: list[dict[str, Any]], quality_by_id: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in canonical_samples:
        quality = quality_by_id[sample["sample_id"]]
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "decision_mode": sample["decision_mode"],
                "canonical_status": sample["canonical_status"],
                "pricing_signal_dir": quality.get("pricing_signal_dir", sample.get("pricing_signal_dir", "")),
                "source_sample_dir": quality.get("source_sample_dir", sample.get("source_sample_dir", "")),
                "input_rows": quality.get("input_rows", ""),
                "primary_rows": quality.get("primary_rows", ""),
                "pricing_signal_rows": quality.get("pricing_signal_rows", ""),
                "future_join_count": quality.get("future_join_count", ""),
                "missing_binance_join_count": quality.get("missing_binance_join_count", ""),
                "primary_usable_row_count": quality.get("primary_usable_row_count", ""),
                "horizon_count": quality.get("horizon_count", ""),
                "independent_future_row_delta_count": quality.get("independent_future_row_delta_count", ""),
                "horizon_future_row_delta_groups": quality.get("horizon_future_row_delta_groups", ""),
                "quality_status": quality.get("quality_status", sample["canonical_status"]),
                "validation_status": "accepted_canonical_event_mode",
            }
        )
    return rows


def _build_rejection_rows(samples: list[dict[str, Any]], quality_by_id: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        reason = _rejection_reason(sample)
        if not reason:
            continue
        quality = quality_by_id.get(sample["sample_id"], {})
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "decision_mode": sample["decision_mode"],
                "canonical_status": sample["canonical_status"],
                "pricing_signal_dir": quality.get("pricing_signal_dir", sample.get("pricing_signal_dir", "")),
                "source_sample_dir": quality.get("source_sample_dir", sample.get("source_sample_dir", "")),
                "rejection_reason": reason,
                "canonical_output_policy": "excluded_from_canonical_sample_manifest",
            }
        )
    return rows


def _write_report(
    path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    canonical_sample_count: int,
    diagnostic_rejection_count: int,
    source_sample_count: int,
) -> None:
    lines = [
        "# Canonical Evidence Validation Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Output directory: `{output_dir}`",
        "- Inputs are existing local `0604T003` aggregate artifacts only.",
        "",
        "## Result",
        "",
        f"- Source sample count: `{source_sample_count}`",
        f"- Canonical event-mode sample count: `{canonical_sample_count}`",
        f"- Diagnostic rejection count: `{diagnostic_rejection_count}`",
        "- Required files and columns passed validation.",
        "- Canonical evidence requires `decision_mode=event` and `canonical_status=canonical_event_mode`.",
        "- Synthetic fixed-grid samples are parseable diagnostics only and are excluded from canonical outputs.",
        "",
        "## Boundary",
        "",
        "- No new data collection, signal ranking, regime selection, case-library construction, or shadow decision generation was performed.",
        "- No private/order endpoints, order lifecycle, strategy implementation, parameter search, live/default-on/tiny-live, or promotion is authorized.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_canonical_event_mode_evidence(*, input_dir: str | Path) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    paths = _required_paths(resolved_input)
    _ensure_required_files(paths)
    manifest = _read_json(paths["multi_sample_manifest"])
    samples = _manifest_samples(manifest)
    has_canonical_samples = any(sample["canonical_sample"] for sample in samples)
    sample_quality_rows = _read_csv(paths["sample_quality_matrix"], REQUIRED_COLUMNS["sample_quality_matrix"])
    feature_rows = _read_csv(
        paths["feature_horizon_stability_across_samples"],
        REQUIRED_COLUMNS["feature_horizon_stability_across_samples"],
    )
    aliasing_rows = _read_csv(
        paths["effective_horizon_aliasing_by_sample"],
        REQUIRED_COLUMNS["effective_horizon_aliasing_by_sample"],
    )
    venue_rows = _read_csv(
        paths["venue_state_conditioning_across_samples"],
        REQUIRED_COLUMNS["venue_state_conditioning_across_samples"],
        allow_empty_without_header=not has_canonical_samples,
    )
    quality_by_id = _sample_quality_by_id(sample_quality_rows)
    _validate_cross_file_consistency(
        manifest=manifest,
        samples=samples,
        quality_by_id=quality_by_id,
        aliasing_rows=aliasing_rows,
        feature_rows=feature_rows,
    )
    canonical_samples = [sample for sample in samples if sample["canonical_sample"]]
    diagnostic_rejections = _build_rejection_rows(samples, quality_by_id)
    return {
        "input_dir": resolved_input,
        "source_manifest": manifest,
        "source_paths": paths,
        "samples": samples,
        "canonical_samples": canonical_samples,
        "diagnostic_rejections": diagnostic_rejections,
        "sample_quality_rows": sample_quality_rows,
        "canonical_quality_summary": _build_quality_summary(canonical_samples, quality_by_id),
        "feature_horizon_stability_rows": feature_rows,
        "effective_horizon_aliasing_rows": aliasing_rows,
        "venue_state_conditioning_rows": venue_rows,
    }


def build_canonical_event_mode_evidence_artifacts(*, input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    loaded = load_canonical_event_mode_evidence(input_dir=input_dir)
    resolved_input = loaded["input_dir"]
    resolved_output = _expand(output_dir)
    canonical_samples = loaded["canonical_samples"]
    diagnostic_rejections = loaded["diagnostic_rejections"]
    source_manifest = loaded["source_manifest"]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "input_dir": str(resolved_input),
        "output_dir": str(resolved_output),
        "source_schema_version": source_manifest.get("schema_version", ""),
        "source_task_id": source_manifest.get("task_id", ""),
        "source_sample_count": len(loaded["samples"]),
        "canonical_sample_count": len(canonical_samples),
        "diagnostic_rejection_count": len(diagnostic_rejections),
        "canonical_decision_mode": CANONICAL_DECISION_MODE,
        "canonical_status": CANONICAL_EVENT_STATUS,
        "diagnostic_synthetic_status": DIAGNOSTIC_SYNTHETIC_STATUS,
        "samples": [
            {
                key: sample.get(key, "")
                for key in [
                    "sample_id",
                    "decision_mode",
                    "canonical_status",
                    "pricing_signal_dir",
                    "source_sample_dir",
                    "source_join_dir",
                    "source_analysis_dir",
                    "alignment_run_manifest",
                    "alignment_metrics",
                    "event_decision_count",
                    "synthetic_decision_count",
                    "run_manifest",
                    "pricing_signal_rows",
                ]
            }
            for sample in canonical_samples
        ],
        "source_artifacts": {name: str(path) for name, path in loaded["source_paths"].items()},
        "output_artifacts": {
            "canonical_sample_manifest": str(resolved_output / "canonical_sample_manifest.json"),
            "canonical_sample_quality_summary": str(resolved_output / "canonical_sample_quality_summary.csv"),
            "diagnostic_rejection_report": str(resolved_output / "diagnostic_rejection_report.csv"),
            "canonical_evidence_validation_report": str(resolved_output / "canonical_evidence_validation_report.md"),
        },
        "boundary_flags": BOUNDARY_FLAGS,
    }
    _write_json(resolved_output / "canonical_sample_manifest.json", manifest)
    _write_csv(
        resolved_output / "canonical_sample_quality_summary.csv",
        loaded["canonical_quality_summary"],
        [
            "sample_id",
            "decision_mode",
            "canonical_status",
            "pricing_signal_dir",
            "source_sample_dir",
            "input_rows",
            "primary_rows",
            "pricing_signal_rows",
            "future_join_count",
            "missing_binance_join_count",
            "primary_usable_row_count",
            "horizon_count",
            "independent_future_row_delta_count",
            "horizon_future_row_delta_groups",
            "quality_status",
            "validation_status",
        ],
    )
    _write_csv(
        resolved_output / "diagnostic_rejection_report.csv",
        diagnostic_rejections,
        [
            "sample_id",
            "decision_mode",
            "canonical_status",
            "pricing_signal_dir",
            "source_sample_dir",
            "rejection_reason",
            "canonical_output_policy",
        ],
    )
    _write_report(
        resolved_output / "canonical_evidence_validation_report.md",
        input_dir=resolved_input,
        output_dir=resolved_output,
        canonical_sample_count=len(canonical_samples),
        diagnostic_rejection_count=len(diagnostic_rejections),
        source_sample_count=len(loaded["samples"]),
    )
    return {
        "canonical_sample_manifest": manifest,
        "canonical_sample_quality_summary": loaded["canonical_quality_summary"],
        "diagnostic_rejection_report": diagnostic_rejections,
        "canonical_sample_count": len(canonical_samples),
        "diagnostic_rejection_count": len(diagnostic_rejections),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical event-mode evidence artifacts.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_canonical_event_mode_evidence_artifacts(input_dir=args.input_dir, output_dir=args.output_dir)
    print(
        "canonical_sample_count="
        f"{result['canonical_sample_count']} diagnostic_rejection_count={result['diagnostic_rejection_count']}"
    )
    print(f"wrote {Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
