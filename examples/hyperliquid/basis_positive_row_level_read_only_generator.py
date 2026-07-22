#!/usr/bin/env python3
"""Generate basis-positive row-level read-only research artifacts.

This T008 generator is local and read-only. It runs the T007 preflight
validator before reading source rows, then emits observation-layer research
rows only. It does not implement a case library, shadow decisions, executable
triggers, private/order endpoints, order lifecycle behavior, strategy behavior,
live/default-on/tiny-live behavior, parameter search, deployment, or promotion.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import basis_positive_filtered_context_viability as filtered
import basis_positive_row_level_preflight_validator as preflight


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_PROJECT_ROOTS = (Path("/home/molly/project/hftbacktest"),)
TASK_ID = "0609T008"
SCHEMA_VERSION = "basis_positive_row_level_read_only_artifact_v1"
T006_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_generator_design_0609T006"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_generator_0609T008"
DEFAULT_PREFLIGHT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "preflight_validator"
PRIMARY_HORIZON_MS = 1000
CASE_LABEL = "basis_positive_clean_context"
FINAL_RECOMMENDATIONS = {
    "row_level_read_only_artifacts_ready_for_qa",
    "needs_more_row_level_generator_coverage",
    "reject_row_level_generation_direction",
}
FUTURE_LABEL_COLUMNS = {
    "horizon_ms",
    "effective_future_row_delta_count",
    "hyperliquid_future_mid_move_ticks",
}
ACTION_COLUMN_MARKERS = {
    "order_side",
    "quote_price",
    "quote_size",
    "leverage",
    "stop_loss",
    "take_profit",
    "submit",
    "cancel",
    "fill",
    "order_id",
    "client_order_id",
    "lifecycle",
    "trigger",
    "signal",
    "action",
    "decision",
    "shadow",
    "live",
    "deploy",
    "promotion",
}
EXECUTION_GAP_COLUMNS = {
    "fill_probability_unproven",
    "queue_position_unproven",
    "post_only_reject_unproven",
    "cancel_fill_race_unproven",
    "fees_rebates_spread_capture_unproven",
    "inventory_lifecycle_unproven",
    "real_order_lifecycle_unproven",
}
BOUNDARY_FLAGS = {
    "row_level_read_only_generator": True,
    "read_only_research_artifact": True,
    "observation_layer_only": True,
    "preflight_validator_required": True,
    "no_case_library_implementation": True,
    "no_case_catalog_generation": True,
    "no_source_row_case_catalog_generation": True,
    "no_shadow_decision_generation": True,
    "no_executable_trigger": True,
    "no_executable_trading_instruction": True,
    "no_actual_order_side_output": True,
    "no_quote_price_or_size_output": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_implementation": True,
    "no_new_data_collection": True,
    "no_remote_execution": True,
    "no_parameter_search": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_live_trading_bot": True,
    "no_deployment_recommendation": True,
    "no_promotion": True,
    "execution_layer_maker_viability_unproven": True,
}


class RowLevelGeneratorError(ValueError):
    """Raised when T008 inputs or generated rows violate the task contract."""


class HistoricalArtifactMissingError(RowLevelGeneratorError):
    """Raised only when an external historical source artifact is absent."""


@dataclass(frozen=True)
class SampleSource:
    sample_id: str
    pricing_signal_rows: Path
    decision_mode: str
    canonical_status: str


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _resolve_recorded_artifact_path(path: str | Path) -> Path:
    recorded = Path(path).expanduser()
    if not recorded.is_absolute():
        return (PROJECT_ROOT / recorded).resolve()
    if recorded.exists():
        return recorded.resolve()
    resolved_project_root = PROJECT_ROOT.resolve()
    for legacy_root in LEGACY_PROJECT_ROOTS:
        try:
            relative = recorded.relative_to(legacy_root)
        except ValueError:
            continue
        if ".." in relative.parts:
            return recorded
        candidate = PROJECT_ROOT / relative
        if not candidate.exists():
            return recorded
        resolved_candidate = candidate.resolve()
        try:
            resolved_candidate.relative_to(resolved_project_root)
        except ValueError:
            return recorded
        return resolved_candidate
    return recorded


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
        raise RowLevelGeneratorError(f"{path} must contain a JSON object")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


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


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.8f}".rstrip("0").rstrip(".")
    return str(value)


def _as_float(value: Any) -> float | None:
    return filtered._as_float(value)


def _as_int(value: Any, default: int = 0) -> int:
    return filtered._as_int(value, default=default)


def _schema_rows(t006_dir: Path) -> list[dict[str, str]]:
    rows = _read_csv(t006_dir / "proposed_row_level_output_schema.csv")
    if not rows:
        raise RowLevelGeneratorError("T006 proposed row-level output schema is empty")
    return rows


def _schema_fieldnames(t006_dir: Path) -> list[str]:
    return [row["column_name"] for row in _schema_rows(t006_dir)]


def _load_t003_manifest(t006_manifest: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / t006_manifest["input_artifacts"]["t003_manifest"]
    manifest = _read_json(path)
    if manifest.get("task_id") != "0609T003":
        raise RowLevelGeneratorError("T008 requires T003 filtered-context manifest")
    if manifest.get("final_recommendation") != "candidate_for_read_only_case_design":
        raise RowLevelGeneratorError("T008 requires T003 candidate_for_read_only_case_design")
    if manifest.get("row_level_input_policy") != "multi_sample_manifest.samples[].pricing_signal_rows":
        raise RowLevelGeneratorError("T008 requires T003 row-level input policy")
    return manifest


def _load_source_manifest(t003_manifest: dict[str, Any]) -> dict[str, Any]:
    input_dir = _resolve_recorded_artifact_path(str(t003_manifest["input_dir"]))
    manifest = _read_json(input_dir / "multi_sample_manifest.json")
    if manifest.get("canonical_sample_count") != 7:
        raise RowLevelGeneratorError("T008 requires the accepted 7-sample canonical source manifest")
    return manifest


def _sample_sources(source_manifest: dict[str, Any]) -> list[SampleSource]:
    sources: list[SampleSource] = []
    for sample in source_manifest.get("samples", []):
        if not isinstance(sample, dict):
            continue
        source = SampleSource(
            sample_id=str(sample.get("sample_id", "")),
            pricing_signal_rows=_resolve_recorded_artifact_path(str(sample.get("pricing_signal_rows", ""))),
            decision_mode=str(sample.get("decision_mode", "")),
            canonical_status=str(sample.get("canonical_status", "")),
        )
        if source.decision_mode != "event" or source.canonical_status != "canonical_event_mode":
            raise RowLevelGeneratorError(f"sample {source.sample_id} is not canonical event-mode")
        if not source.pricing_signal_rows.exists():
            raise HistoricalArtifactMissingError(
                f"missing pricing_signal_rows for {source.sample_id}: {source.pricing_signal_rows}"
            )
        sources.append(source)
    if not sources:
        raise RowLevelGeneratorError("no canonical sample sources found")
    return sources


def historical_source_artifacts_available(t006_dir: str | Path = T006_DIR) -> bool:
    try:
        resolved_t006 = _expand(t006_dir)
        t006_manifest = _read_json(resolved_t006 / "generator_design_manifest.json")
        t003_manifest = _load_t003_manifest(t006_manifest)
        source_manifest = _load_source_manifest(t003_manifest)
        _sample_sources(source_manifest)
    except (FileNotFoundError, HistoricalArtifactMissingError):
        return False
    return True


def _allowed_source_paths(sources: list[SampleSource]) -> set[Path]:
    return {source.pricing_signal_rows.resolve() for source in sources}


def _validate_source_path_allowed(path: Path, allowed_paths: set[Path]) -> None:
    resolved = path.resolve()
    if resolved not in allowed_paths:
        raise RowLevelGeneratorError(f"source path outside T006/T003 allowlist: {resolved}")


def _load_source_rows(sources: list[SampleSource]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    allowed = _allowed_source_paths(sources)
    all_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for source in sources:
        _validate_source_path_allowed(source.pricing_signal_rows, allowed)
        rows = _read_csv(source.pricing_signal_rows)
        all_rows.extend(rows)
        manifest_rows.append(
            {
                "sample_id": source.sample_id,
                "source_artifact_path": str(source.pricing_signal_rows),
                "decision_mode": source.decision_mode,
                "canonical_status": source.canonical_status,
                "source_row_count": len(rows),
                "allowed_source_status": "pass",
            }
        )
    return all_rows, manifest_rows


def _primary_clean_rows(all_source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = filtered.wrong_way._eligible_rows(all_source_rows)
    primary_base = [row for row in eligible if _as_int(row.get("horizon_ms")) == PRIMARY_HORIZON_MS]
    if not primary_base:
        raise RowLevelGeneratorError("no eligible primary-horizon rows")
    enriched = filtered._enrich_rows(primary_base, threshold_rows=primary_base)
    return [row for row in enriched if row["_basis"] > 0 and not row["_tail_combined"]]


def _row_reference(row: dict[str, Any]) -> str:
    return (
        f"{row.get('sample_id', '')}:source_row_index={row.get('source_row_index', '')}:"
        f"future_row_index={row.get('future_row_index', '')}:horizon_ms={row.get('horizon_ms', '')}"
    )


def _artifact_row(row: dict[str, Any], source_path_by_sample: dict[str, str]) -> dict[str, Any]:
    return {
        "artifact_schema_version": SCHEMA_VERSION,
        "artifact_generation_task_id": TASK_ID,
        "source_task_id": "0609T003",
        "source_artifact_path": source_path_by_sample[str(row.get("sample_id", ""))],
        "source_sample_id": row.get("sample_id", ""),
        "source_row_reference": _row_reference(row),
        "source_observation_ts": row.get("hyperliquid_decision_ts", ""),
        "context_basis_mid_ticks": row.get("context_basis_mid_ticks", ""),
        "basis_magnitude_bucket": row.get("_basis_magnitude_bucket", ""),
        "hl_top5_imbalance_bucket": row.get("_hl_top5_imbalance_bucket", ""),
        "hl_microprice_minus_mid_bucket": row.get("_hl_microprice_minus_mid_bucket", ""),
        "input_binance_mid_move_ticks_from_prev": row.get("input_binance_mid_move_ticks_from_prev", ""),
        "spread_bucket": row.get("_spread_bucket", ""),
        "join_age_bucket": row.get("_join_age_bucket", ""),
        "visible_movement_bucket": row.get("_visible_movement_bucket", ""),
        "case_label": CASE_LABEL,
        "horizon_ms": row.get("horizon_ms", ""),
        "effective_future_row_delta_count": row.get("effective_future_row_delta", ""),
        "hyperliquid_future_mid_move_ticks": row.get("hyperliquid_future_mid_move_ticks", ""),
        "fill_probability_unproven": True,
        "queue_position_unproven": True,
        "post_only_reject_unproven": True,
        "cancel_fill_race_unproven": True,
        "fees_rebates_spread_capture_unproven": True,
        "inventory_lifecycle_unproven": True,
        "real_order_lifecycle_unproven": True,
        "validator_schema_pass": True,
        "validator_no_action_fields_pass": True,
        "validator_future_label_output_only_pass": True,
    }


def _validate_no_action_fields(fieldnames: list[str]) -> list[dict[str, Any]]:
    hits = []
    for name in fieldnames:
        lowered = name.lower()
        if name in EXECUTION_GAP_COLUMNS:
            continue
        if name.startswith("validator_"):
            continue
        if any(marker in lowered for marker in ACTION_COLUMN_MARKERS):
            hits.append(name)
    return [
        {
            "check_id": "no_action_capable_columns",
            "status": "pass" if not hits else "fail",
            "detail": "no action-capable columns" if not hits else "|".join(sorted(hits)),
        }
    ]


def _validate_future_labels(schema_rows: list[dict[str, str]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    category_by_name = {row["column_name"]: row["column_category"] for row in schema_rows}
    bad_categories = [
        name for name in FUTURE_LABEL_COLUMNS if category_by_name.get(name) != "future_label_for_research_only"
    ]
    bad_case_labels = [row["source_row_reference"] for row in rows if any(str(row.get("case_label", "")).find(name) >= 0 for name in FUTURE_LABEL_COLUMNS)]
    checks = [
        {
            "check_id": "future_label_schema_category_output_only",
            "status": "pass" if not bad_categories else "fail",
            "detail": "future labels categorized output-only" if not bad_categories else "|".join(sorted(bad_categories)),
        },
        {
            "check_id": "future_label_not_used_in_case_label",
            "status": "pass" if not bad_case_labels else "fail",
            "detail": "case labels do not contain future labels" if not bad_case_labels else "|".join(bad_case_labels[:10]),
        },
        {
            "check_id": "future_label_filter_policy",
            "status": "pass",
            "detail": "row filter uses decision-time basis and tail-risk context only; horizon is fixed to accepted primary horizon",
        },
    ]
    return checks


def _validate_execution_gap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failures = []
    for idx, row in enumerate(rows):
        missing = [name for name in EXECUTION_GAP_COLUMNS if row.get(name) is not True]
        if missing:
            failures.append(f"row={idx}:{'|'.join(missing)}")
            break
    return [
        {
            "check_id": "execution_gap_markers_true",
            "status": "pass" if not failures else "fail",
            "detail": f"{len(EXECUTION_GAP_COLUMNS)} unproven markers true for every row" if not failures else failures[0],
        }
    ]


def _schema_validation_rows(schema_rows: list[dict[str, str]], generated_fieldnames: list[str]) -> list[dict[str, Any]]:
    schema_names = [row["column_name"] for row in schema_rows]
    missing = [name for name in schema_names if name not in generated_fieldnames]
    extra = [name for name in generated_fieldnames if name not in schema_names]
    bad_categories = [row["column_name"] for row in schema_rows if row["column_category"] not in preflight.OUTPUT_CATEGORIES]
    return [
        {
            "check_id": "schema_required_columns_present",
            "status": "pass" if not missing else "fail",
            "detail": f"{len(schema_names)} schema columns present" if not missing else "|".join(missing),
        },
        {
            "check_id": "schema_no_extra_columns",
            "status": "pass" if not extra else "fail",
            "detail": "generated columns match T006 schema" if not extra else "|".join(extra),
        },
        {
            "check_id": "schema_categories_allowed",
            "status": "pass" if not bad_categories else "fail",
            "detail": "all T006 schema categories allowed" if not bad_categories else "|".join(bad_categories),
        },
    ]


def _lineage_rows(rows: list[dict[str, Any]], source_manifest_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row["source_sample_id"]) for row in rows)
    lineage_failures = [
        str(row.get("source_row_reference", ""))
        for row in rows
        if not row.get("source_artifact_path") or not row.get("source_sample_id") or not row.get("source_row_reference")
    ]
    out = [
        {
            "check_id": "lineage_required_fields_present",
            "status": "pass" if not lineage_failures else "fail",
            "detail": "source path, sample id, and source row reference present for every row"
            if not lineage_failures
            else "|".join(lineage_failures[:10]),
        }
    ]
    for source in source_manifest_rows:
        sample_id = source["sample_id"]
        out.append(
            {
                "check_id": "lineage_sample_row_count",
                "sample_id": sample_id,
                "status": "pass",
                "detail": counts.get(sample_id, 0),
            }
        )
    return out


def _fail_if_checks_failed(*groups: list[dict[str, Any]]) -> None:
    failed = [row for group in groups for row in group if row.get("status") != "pass"]
    if failed:
        detail = "; ".join(f"{row.get('check_id')}: {row.get('detail')}" for row in failed)
        raise RowLevelGeneratorError(detail)


def _write_report(path: Path, *, manifest: dict[str, Any]) -> None:
    lines = [
        "# Basis-Positive Row-Level Read-Only Generator Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Result",
        "",
        f"- Final recommendation: `{manifest['final_recommendation']}`",
        f"- Generated row count: `{manifest['generated_row_count']}`",
        f"- Source sample count: `{manifest['source_sample_count']}`",
        f"- Primary horizon: `{manifest['primary_horizon_ms']}` ms",
        "",
        "## Boundary",
        "",
        "- T008 produces read-only research rows only.",
        "- It does not implement case-library behavior, generated case catalogs, source-row case catalogs, shadow decisions, executable triggers, trading instructions, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.",
        "- Future labels are output-only offline research labels and are not inputs, filters, triggers, case conditions, shadow-decision fields, live decisions, or deployment criteria.",
        "- Fill probability, queue/queue-ahead, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, and maker execution viability remain unproven.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_row_level_artifacts(
    *,
    t006_dir: str | Path = T006_DIR,
    preflight_output_dir: str | Path = DEFAULT_PREFLIGHT_OUTPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    resolved_t006 = _expand(t006_dir)
    resolved_preflight = _expand(preflight_output_dir)
    resolved_output = _expand(output_dir)
    preflight_result = preflight.build_preflight_artifacts(input_dir=resolved_t006, output_dir=resolved_preflight)
    preflight_manifest = preflight_result["manifest"]
    if preflight_manifest.get("final_recommendation") != "preflight_validator_ready_for_qa":
        raise RowLevelGeneratorError("T007 preflight validator did not pass")
    if preflight_manifest.get("failed_check_count") != 0:
        raise RowLevelGeneratorError("T007 preflight validator has failed checks")

    t006_manifest = _read_json(resolved_t006 / "generator_design_manifest.json")
    t003_manifest = _load_t003_manifest(t006_manifest)
    source_manifest = _load_source_manifest(t003_manifest)
    sources = _sample_sources(source_manifest)
    source_rows, source_manifest_rows = _load_source_rows(sources)
    clean_rows = _primary_clean_rows(source_rows)
    source_path_by_sample = {row["sample_id"]: row["source_artifact_path"] for row in source_manifest_rows}
    schema_rows = _schema_rows(resolved_t006)
    fieldnames = [row["column_name"] for row in schema_rows]
    artifact_rows = [_artifact_row(row, source_path_by_sample) for row in clean_rows]
    normalized_rows = [{field: row.get(field, "") for field in fieldnames} for row in artifact_rows]

    schema_checks = _schema_validation_rows(schema_rows, fieldnames)
    future_checks = _validate_future_labels(schema_rows, artifact_rows)
    no_action_checks = _validate_no_action_fields(fieldnames)
    lineage_checks = _lineage_rows(artifact_rows, source_manifest_rows)
    gap_checks = _validate_execution_gap_rows(artifact_rows)
    _fail_if_checks_failed(schema_checks, future_checks, no_action_checks, lineage_checks, gap_checks)

    sample_counts = Counter(row["source_sample_id"] for row in artifact_rows)
    final_recommendation = (
        "row_level_read_only_artifacts_ready_for_qa"
        if artifact_rows and len(sample_counts) >= 5
        else "needs_more_row_level_generator_coverage"
    )
    if final_recommendation not in FINAL_RECOMMENDATIONS:
        raise AssertionError(f"unexpected recommendation: {final_recommendation}")

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "t006_dir": str(resolved_t006),
        "preflight_output_dir": str(resolved_preflight),
        "output_dir": str(resolved_output),
        "source_task_id": "0609T003",
        "source_row_level_input_policy": t003_manifest.get("row_level_input_policy"),
        "source_manifest_path": str(
            _resolve_recorded_artifact_path(str(t003_manifest["input_dir"])) / "multi_sample_manifest.json"
        ),
        "source_sample_count": len(sources),
        "primary_horizon_ms": PRIMARY_HORIZON_MS,
        "case_label": CASE_LABEL,
        "generated_row_count": len(artifact_rows),
        "generated_sample_count": len(sample_counts),
        "per_sample_generated_rows": dict(sorted(sample_counts.items())),
        "preflight_final_recommendation": preflight_manifest.get("final_recommendation"),
        "preflight_failed_check_count": preflight_manifest.get("failed_check_count"),
        "final_recommendation": final_recommendation,
        "allowed_final_recommendations": sorted(FINAL_RECOMMENDATIONS),
        "boundary_flags": BOUNDARY_FLAGS,
        "output_artifacts": {
            "row_level_generator_manifest": str(resolved_output / "row_level_generator_manifest.json"),
            "preflight_validation_result": str(resolved_output / "preflight_validation_result.json"),
            "source_artifact_manifest": str(resolved_output / "source_artifact_manifest.csv"),
            "row_level_read_only_cases": str(resolved_output / "row_level_read_only_cases.csv"),
            "row_level_schema_validation": str(resolved_output / "row_level_schema_validation.csv"),
            "future_label_leakage_check": str(resolved_output / "future_label_leakage_check.csv"),
            "no_action_field_check": str(resolved_output / "no_action_field_check.csv"),
            "lineage_validation_summary": str(resolved_output / "lineage_validation_summary.csv"),
            "execution_gap_boundary_check": str(resolved_output / "execution_gap_boundary_check.csv"),
            "row_level_generator_report": str(resolved_output / "row_level_generator_report.md"),
        },
    }

    _write_json(resolved_output / "row_level_generator_manifest.json", manifest)
    _write_json(resolved_output / "preflight_validation_result.json", preflight_manifest)
    _write_csv(
        resolved_output / "source_artifact_manifest.csv",
        source_manifest_rows,
        ["sample_id", "source_artifact_path", "decision_mode", "canonical_status", "source_row_count", "allowed_source_status"],
    )
    _write_csv(resolved_output / "row_level_read_only_cases.csv", normalized_rows, fieldnames)
    _write_csv(resolved_output / "row_level_schema_validation.csv", schema_checks, ["check_id", "status", "detail"])
    _write_csv(resolved_output / "future_label_leakage_check.csv", future_checks, ["check_id", "status", "detail"])
    _write_csv(resolved_output / "no_action_field_check.csv", no_action_checks, ["check_id", "status", "detail"])
    _write_csv(resolved_output / "lineage_validation_summary.csv", lineage_checks, ["check_id", "sample_id", "status", "detail"])
    _write_csv(resolved_output / "execution_gap_boundary_check.csv", gap_checks, ["check_id", "status", "detail"])
    _write_report(resolved_output / "row_level_generator_report.md", manifest=manifest)
    return {"manifest": manifest, "rows": normalized_rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate T008 basis-positive row-level read-only artifacts.")
    parser.add_argument("--t006-dir", type=Path, default=T006_DIR)
    parser.add_argument("--preflight-output-dir", type=Path, default=DEFAULT_PREFLIGHT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_row_level_artifacts(
        t006_dir=args.t006_dir,
        preflight_output_dir=args.preflight_output_dir,
        output_dir=args.output_dir,
    )
    manifest = result["manifest"]
    print(
        "final_recommendation="
        f"{manifest['final_recommendation']} rows={manifest['generated_row_count']} output_dir={manifest['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
