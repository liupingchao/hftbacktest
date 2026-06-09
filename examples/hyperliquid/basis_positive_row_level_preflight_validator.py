#!/usr/bin/env python3
"""Preflight validator for the basis-positive row-level generator design.

This is a local read-only validator for T006 design artifacts. It validates
schemas, guard requirements, and reject-condition coverage. It does not
implement a row-level generator, read source rows, create case catalogs, create
shadow decisions, touch private/order endpoints, implement strategy behavior,
run live/default-on/tiny-live behavior, run parameter search, or make promotion
claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0609T007"
SCHEMA_VERSION = "basis_positive_row_level_preflight_validator_v1"
SOURCE_TASK_ID = "0609T006"
SOURCE_SCHEMA_VERSION = "basis_positive_row_level_read_only_generator_design_v1"
DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_generator_design_0609T006"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_preflight_validator_0609T007"
FINAL_RECOMMENDATIONS = {
    "preflight_validator_ready_for_qa",
    "needs_more_validator_coverage",
    "reject_validator_direction",
}

REQUIRED_FILES = {
    "manifest": "generator_design_manifest.json",
    "input_allowlist": "input_manifest_allowlist.csv",
    "output_schema": "proposed_row_level_output_schema.csv",
    "future_label_guards": "future_label_leakage_guard_requirements.csv",
    "no_action_guards": "no_action_field_guard_requirements.csv",
    "reject_conditions": "generator_reject_conditions.csv",
    "acceptance_gate": "generator_acceptance_gate.md",
    "lineage_requirements": "lineage_and_provenance_requirements.md",
}
REQUIRED_COLUMNS = {
    "input_allowlist": {
        "input_class",
        "allowed_status",
        "required_source",
        "required_qa_status",
        "allowed_use",
        "explicit_reject",
    },
    "output_schema": {
        "column_name",
        "column_category",
        "required_status",
        "source_contract",
        "definition",
        "allowed_use",
        "forbidden_use",
        "validator_reference",
    },
    "future_label_guards": {
        "requirement_id",
        "requirement_type",
        "scope",
        "required_check",
        "reject_if",
        "notes",
    },
    "no_action_guards": {
        "requirement_id",
        "requirement_type",
        "scope",
        "required_check",
        "reject_if",
        "notes",
    },
    "reject_conditions": {
        "condition_id",
        "condition_category",
        "reject_if",
        "required_response",
        "notes",
    },
}
REQUIRED_BOUNDARY_FLAGS = {
    "generator_design_only",
    "no_generator_implementation",
    "no_row_level_generation",
    "no_case_catalog_generation",
    "read_only_research_artifact",
    "observation_layer_only",
    "no_new_data_collection",
    "no_remote_execution",
    "no_private_account_endpoints",
    "no_order_endpoints",
    "no_order_lifecycle",
    "no_strategy_implementation",
    "no_shadow_decision_generation",
    "no_executable_trigger",
    "no_executable_trading_instruction",
    "no_actual_order_side_output",
    "no_quote_price_or_size_output",
    "no_parameter_search",
    "no_default_on",
    "no_tiny_live",
    "no_live_trading_bot",
    "no_deployment_recommendation",
    "no_promotion",
    "execution_layer_maker_viability_unproven",
}
OUTPUT_CATEGORIES = {
    "row_identity",
    "lineage",
    "decision_time_visible_context",
    "diagnostic_context",
    "read_only_label",
    "future_label_for_research_only",
    "execution_gap_reference",
    "validation_trace",
}
REQUIRED_OUTPUT_COLUMNS = {
    "artifact_schema_version",
    "artifact_generation_task_id",
    "source_task_id",
    "source_artifact_path",
    "source_row_reference",
    "context_basis_mid_ticks",
    "case_label",
    "horizon_ms",
    "effective_future_row_delta_count",
    "hyperliquid_future_mid_move_ticks",
    "fill_probability_unproven",
    "queue_position_unproven",
    "post_only_reject_unproven",
    "cancel_fill_race_unproven",
    "fees_rebates_spread_capture_unproven",
    "inventory_lifecycle_unproven",
    "real_order_lifecycle_unproven",
    "validator_no_action_fields_pass",
    "validator_future_label_output_only_pass",
}
FUTURE_LABEL_COLUMNS = {
    "horizon_ms",
    "effective_future_row_delta_count",
    "hyperliquid_future_mid_move_ticks",
}
REQUIRED_FUTURE_LABEL_GUARDS = {
    "reject_future_label_input_section",
    "reject_future_label_filter_condition",
    "reject_future_label_trigger",
    "reject_future_label_shadow_decision",
    "reject_future_label_live_decision",
    "reject_future_label_join_back_to_context",
    "reject_future_label_aliases",
    "reject_future_label_in_case_label",
    "reject_future_label_in_free_text",
    "require_future_label_output_only_trace",
}
REQUIRED_NO_ACTION_GUARDS = {
    "reject_order_side_fields",
    "reject_quote_price_fields",
    "reject_quote_size_fields",
    "reject_leverage_stop_take_profit_fields",
    "reject_submit_cancel_fill_fields",
    "reject_executable_trigger_fields",
    "reject_shadow_decision_fields",
    "reject_private_order_live_fields",
    "reject_parameter_search_fields",
    "reject_deployment_promotion_fields",
    "reject_action_aliases_in_text",
    "require_no_action_validation_trace",
}
REQUIRED_REJECT_CONDITIONS = {
    "reject_generator_implementation_in_t006",
    "reject_row_level_generation_in_t006",
    "reject_case_catalog_generation",
    "reject_shadow_decision_generation",
    "reject_executable_trigger_fields",
    "reject_action_field_definitions",
    "reject_private_order_or_collection_scope_expansion",
    "reject_strategy_private_order_live_or_promotion",
    "reject_future_label_as_input",
    "reject_label_as_trigger_or_shadow_decision",
    "reject_diagnostic_to_strategy_upgrade",
    "reject_ad_hoc_source_expansion",
    "reject_implicit_generator_authorization",
    "reject_execution_layer_proof_overclaim",
    "reject_t004_t005_boundary_weakening",
    "reject_missing_required_artifact",
    "reject_missing_validation_trace_requirement",
    "reject_deployment_or_promotion_language",
}
REQUIRED_ALLOWLIST_CLASSES = {
    "t003_filtered_context_manifest",
    "t004_field_contract",
    "t004_label_contract",
    "t005_schema_manifest",
    "t005_validator_requirements",
    "t005_row_level_prerequisite_register",
    "t005_execution_gap_boundary_register",
    "upstream_public_pricing_signal_rows_reference",
    "remote_or_private_sources",
}
FORBIDDEN_SOURCE_MARKERS = {
    "private",
    "account",
    "order",
    "user_stream",
    "position",
    "signing",
    "nonce",
    "live_bot",
    "production_config",
    "remote",
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
    "client_order_id",
    "shadow_action",
    "shadow_side",
    "shadow_quote",
    "live_instruction",
    "deployment_recommendation",
    "promotion",
}
BOUNDARY_FLAGS = {
    "validator_preflight_only": True,
    "read_only_research_artifact": True,
    "no_generator_implementation": True,
    "no_row_level_generation": True,
    "no_case_catalog_generation": True,
    "no_source_row_catalog_generation": True,
    "no_shadow_decision_generation": True,
    "no_executable_trigger": True,
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


class PreflightValidationError(ValueError):
    """Raised when T006 design artifacts fail preflight validation."""


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    check_category: str
    status: str
    detail: str


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
        raise PreflightValidationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PreflightValidationError(f"{path} must contain a JSON object")
    return payload


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(required_columns - fieldnames)
        if missing:
            raise PreflightValidationError(f"{path} missing required columns: {', '.join(missing)}")
        return list(reader)


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


def _required_paths(input_dir: Path) -> dict[str, Path]:
    return {name: input_dir / filename for name, filename in REQUIRED_FILES.items()}


def _ensure_required_files(paths: dict[str, Path]) -> list[CheckResult]:
    missing = [f"{name}:{path}" for name, path in paths.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required T006 design artifacts: " + "; ".join(missing))
    return [CheckResult("required_files_present", "artifact_scope", "pass", f"{len(paths)} files present")]


def _pass(check_id: str, category: str, detail: str) -> CheckResult:
    return CheckResult(check_id, category, "pass", detail)


def _fail(check_id: str, category: str, detail: str) -> CheckResult:
    return CheckResult(check_id, category, "fail", detail)


def _contains_any(text: str, markers: set[str]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def _validate_manifest(manifest: dict[str, Any]) -> list[CheckResult]:
    rows: list[CheckResult] = []
    if manifest.get("task_id") == SOURCE_TASK_ID:
        rows.append(_pass("manifest_task_id", "manifest", f"task_id={SOURCE_TASK_ID}"))
    else:
        rows.append(_fail("manifest_task_id", "manifest", f"unexpected task_id={manifest.get('task_id')!r}"))
    if manifest.get("schema_version") == SOURCE_SCHEMA_VERSION:
        rows.append(_pass("manifest_schema_version", "manifest", SOURCE_SCHEMA_VERSION))
    else:
        rows.append(_fail("manifest_schema_version", "manifest", f"unexpected schema_version={manifest.get('schema_version')!r}"))
    if manifest.get("final_recommendation") == "row_level_read_only_generator_design_ready":
        rows.append(_pass("manifest_final_recommendation", "manifest", "row_level_read_only_generator_design_ready"))
    else:
        rows.append(
            _fail(
                "manifest_final_recommendation",
                "manifest",
                f"unexpected final_recommendation={manifest.get('final_recommendation')!r}",
            )
        )
    for key, expected in (
        ("source_schema_qa_status", "passed"),
        ("source_case_contract_qa_status", "passed"),
        ("evidence_source_qa_status", "passed"),
    ):
        if manifest.get(key) == expected:
            rows.append(_pass(f"manifest_{key}", "manifest", f"{key}=passed"))
        else:
            rows.append(_fail(f"manifest_{key}", "manifest", f"{key}={manifest.get(key)!r}"))
    flags = manifest.get("boundary_flags")
    if not isinstance(flags, dict):
        rows.append(_fail("manifest_boundary_flags_object", "boundary_flags", "boundary_flags is not an object"))
        return rows
    missing = sorted(flag for flag in REQUIRED_BOUNDARY_FLAGS if flags.get(flag) is not True)
    if missing:
        rows.append(_fail("manifest_required_boundary_flags", "boundary_flags", "missing_or_false=" + "|".join(missing)))
    else:
        rows.append(_pass("manifest_required_boundary_flags", "boundary_flags", f"{len(REQUIRED_BOUNDARY_FLAGS)} flags true"))
    categories = set(manifest.get("designed_output_categories") or [])
    missing_categories = sorted(OUTPUT_CATEGORIES - categories)
    if missing_categories:
        rows.append(_fail("manifest_output_categories", "manifest", "missing=" + "|".join(missing_categories)))
    else:
        rows.append(_pass("manifest_output_categories", "manifest", f"{len(categories)} categories declared"))
    return rows


def _validate_allowlist(rows: list[dict[str, str]]) -> list[CheckResult]:
    out: list[CheckResult] = []
    if not rows:
        return [_fail("allowlist_nonempty", "input_allowlist", "no rows")]
    classes = {row["input_class"] for row in rows}
    missing_classes = sorted(REQUIRED_ALLOWLIST_CLASSES - classes)
    if missing_classes:
        out.append(_fail("allowlist_required_classes", "input_allowlist", "missing=" + "|".join(missing_classes)))
    else:
        out.append(_pass("allowlist_required_classes", "input_allowlist", f"{len(REQUIRED_ALLOWLIST_CLASSES)} classes present"))
    forbidden_rows = [row for row in rows if row["allowed_status"] == "forbidden"]
    if any(row["input_class"] == "remote_or_private_sources" for row in forbidden_rows):
        out.append(_pass("allowlist_forbidden_source_class", "input_allowlist", "remote_or_private_sources forbidden"))
    else:
        out.append(_fail("allowlist_forbidden_source_class", "input_allowlist", "remote_or_private_sources not forbidden"))
    bad_allowed = []
    for row in rows:
        status = row["allowed_status"]
        if status in {"allowed", "conditional_allowed"}:
            if row["required_qa_status"] != "passed":
                bad_allowed.append(f"{row['input_class']}:qa={row['required_qa_status']}")
            if _contains_any(row["required_source"], FORBIDDEN_SOURCE_MARKERS):
                bad_allowed.append(f"{row['input_class']}:forbidden_source_marker")
            if "reject" not in row["explicit_reject"].lower():
                bad_allowed.append(f"{row['input_class']}:missing_reject_text")
    if bad_allowed:
        out.append(_fail("allowlist_allowed_rows", "input_allowlist", "|".join(bad_allowed)))
    else:
        out.append(_pass("allowlist_allowed_rows", "input_allowlist", "allowed rows are QA-passed and reject-scoped"))
    return out


def _validate_output_schema(rows: list[dict[str, str]]) -> list[CheckResult]:
    out: list[CheckResult] = []
    if not rows:
        return [_fail("schema_nonempty", "output_schema", "no rows")]
    names = {row["column_name"] for row in rows}
    missing_columns = sorted(REQUIRED_OUTPUT_COLUMNS - names)
    if missing_columns:
        out.append(_fail("schema_required_columns", "output_schema", "missing=" + "|".join(missing_columns)))
    else:
        out.append(_pass("schema_required_columns", "output_schema", f"{len(REQUIRED_OUTPUT_COLUMNS)} required columns present"))
    bad_categories = sorted(
        f"{row['column_name']}:{row['column_category']}"
        for row in rows
        if row["column_category"] not in OUTPUT_CATEGORIES
    )
    if bad_categories:
        out.append(_fail("schema_allowed_categories", "output_schema", "|".join(bad_categories)))
    else:
        out.append(_pass("schema_allowed_categories", "output_schema", "all categories allowed by T006"))
    future_bad = []
    for row in rows:
        if row["column_name"] in FUTURE_LABEL_COLUMNS:
            if row["column_category"] != "future_label_for_research_only":
                future_bad.append(f"{row['column_name']}:category={row['column_category']}")
            if "input" not in row["forbidden_use"].lower():
                future_bad.append(f"{row['column_name']}:missing_input_forbidden_use")
    if future_bad:
        out.append(_fail("schema_future_labels_output_only", "future_label_guard", "|".join(future_bad)))
    else:
        out.append(_pass("schema_future_labels_output_only", "future_label_guard", "future labels are output-only"))
    action_bad = []
    for row in rows:
        name = row["column_name"].lower()
        category = row["column_category"]
        if category in {"validation_trace", "execution_gap_reference"}:
            continue
        if _contains_any(name, ACTION_COLUMN_MARKERS):
            action_bad.append(row["column_name"])
    if action_bad:
        out.append(_fail("schema_no_action_columns", "no_action_guard", "|".join(sorted(action_bad))))
    else:
        out.append(_pass("schema_no_action_columns", "no_action_guard", "no action-capable columns outside validation/gap categories"))
    real_row_markers = [name for name in names if name in {"row_level_case_entry", "case_catalog_row", "shadow_decision"}]
    if real_row_markers:
        out.append(_fail("schema_no_row_generation_columns", "artifact_scope", "|".join(real_row_markers)))
    else:
        out.append(_pass("schema_no_row_generation_columns", "artifact_scope", "no row-level generation columns"))
    return out


def _validate_required_ids(
    rows: list[dict[str, str]],
    *,
    file_key: str,
    id_field: str,
    required_ids: set[str],
    category: str,
) -> tuple[list[CheckResult], list[dict[str, Any]]]:
    ids = {row[id_field] for row in rows}
    missing = sorted(required_ids - ids)
    result = []
    if missing:
        result.append(_fail(f"{file_key}_required_coverage", category, "missing=" + "|".join(missing)))
    else:
        result.append(_pass(f"{file_key}_required_coverage", category, f"{len(required_ids)} required ids present"))
    coverage_rows = [
        {
            "required_id": item,
            "source_file": file_key,
            "status": "pass" if item in ids else "fail",
            "detail": "present" if item in ids else "missing",
        }
        for item in sorted(required_ids)
    ]
    return result, coverage_rows


def _validate_text_boundaries(texts: dict[str, str]) -> list[CheckResult]:
    combined = "\n".join(texts.values()).lower()
    required_phrase_groups = {
        "authorization_boundary": ["does not authorize", "not approval", "must be separately dispatched"],
        "execution_gap_boundary": ["unproven", "execution-layer gaps remain"],
        "separate_dispatch_boundary": ["separate", "separately dispatched"],
        "private_boundary": ["no private", "private/account/order", "private/order"],
        "strategy_boundary": ["no strategy", "strategy/live", "strategy implementation"],
        "row_level_boundary": ["no row-level", "does not generate row-level", "must not read or generate row-level"],
    }
    missing = [
        group
        for group, phrases in required_phrase_groups.items()
        if not any(phrase in combined for phrase in phrases)
    ]
    if missing:
        return [_fail("text_boundary_required_phrases", "boundary_text", "missing=" + "|".join(missing))]
    positive_phrase_tokens = [
        ("authorizes", "generator implementation"),
        ("authorizes", "row generation"),
        ("authorizes", "live"),
        ("ready for", "deployment"),
        ("ready for", "promotion"),
        ("maker viability", "is proven"),
        ("fill probability", "is proven"),
        ("queue position", "is proven"),
    ]
    positive_hits = [
        " ".join(tokens)
        for tokens in positive_phrase_tokens
        if all(token in combined for token in tokens) and f"not {' '.join(tokens)}" not in combined
    ]
    if positive_hits:
        return [_fail("text_boundary_positive_authorization", "boundary_text", "hits=" + "|".join(positive_hits))]
    return [_pass("text_boundary_language", "boundary_text", "required prohibition/boundary text present")]


def load_design_artifacts(input_dir: Path) -> dict[str, Any]:
    paths = _required_paths(input_dir)
    checks = _ensure_required_files(paths)
    manifest = _read_json(paths["manifest"])
    input_allowlist = _read_csv(paths["input_allowlist"], REQUIRED_COLUMNS["input_allowlist"])
    output_schema = _read_csv(paths["output_schema"], REQUIRED_COLUMNS["output_schema"])
    future_label_guards = _read_csv(paths["future_label_guards"], REQUIRED_COLUMNS["future_label_guards"])
    no_action_guards = _read_csv(paths["no_action_guards"], REQUIRED_COLUMNS["no_action_guards"])
    reject_conditions = _read_csv(paths["reject_conditions"], REQUIRED_COLUMNS["reject_conditions"])
    texts = {
        "acceptance_gate": paths["acceptance_gate"].read_text(encoding="utf-8"),
        "lineage_requirements": paths["lineage_requirements"].read_text(encoding="utf-8"),
    }
    return {
        "paths": paths,
        "initial_checks": checks,
        "manifest": manifest,
        "input_allowlist": input_allowlist,
        "output_schema": output_schema,
        "future_label_guards": future_label_guards,
        "no_action_guards": no_action_guards,
        "reject_conditions": reject_conditions,
        "texts": texts,
    }


def validate_design_artifacts(input_dir: Path) -> tuple[list[CheckResult], list[dict[str, Any]], dict[str, Any]]:
    loaded = load_design_artifacts(input_dir)
    checks: list[CheckResult] = list(loaded["initial_checks"])
    checks.extend(_validate_manifest(loaded["manifest"]))
    checks.extend(_validate_allowlist(loaded["input_allowlist"]))
    checks.extend(_validate_output_schema(loaded["output_schema"]))
    reject_rows: list[dict[str, Any]] = []
    for file_key, rows, required_ids, category in (
        ("future_label_leakage_guard_requirements", loaded["future_label_guards"], REQUIRED_FUTURE_LABEL_GUARDS, "future_label_guard"),
        ("no_action_field_guard_requirements", loaded["no_action_guards"], REQUIRED_NO_ACTION_GUARDS, "no_action_guard"),
        ("generator_reject_conditions", loaded["reject_conditions"], REQUIRED_REJECT_CONDITIONS, "reject_conditions"),
    ):
        field = "condition_id" if file_key == "generator_reject_conditions" else "requirement_id"
        result, coverage = _validate_required_ids(
            rows,
            file_key=file_key,
            id_field=field,
            required_ids=required_ids,
            category=category,
        )
        checks.extend(result)
        reject_rows.extend(coverage)
    checks.extend(_validate_text_boundaries(loaded["texts"]))
    return checks, reject_rows, loaded


def _write_report(path: Path, *, input_dir: Path, output_dir: Path, checks: list[CheckResult], final_recommendation: str) -> None:
    failed = [row for row in checks if row.status != "pass"]
    lines = [
        "# Basis-Positive Row-Level Preflight Validation Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Scope",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Output directory: `{output_dir}`",
        "- This validator checks T006 design artifacts only.",
        "- It does not implement a row-level generator or read/generate row-level case entries.",
        "",
        "## Result",
        "",
        f"- Final recommendation: `{final_recommendation}`",
        f"- Total checks: `{len(checks)}`",
        f"- Failed checks: `{len(failed)}`",
        "",
        "## Boundary",
        "",
        "- No generator implementation, row-level generation, case catalog, source-row catalog, or shadow decision generation was performed.",
        "- No executable trigger, order side, quote price/size, leverage, stop/take-profit, submit/cancel/fill, private/order endpoint, strategy/live/default-on/tiny-live, parameter search, deployment recommendation, or promotion is authorized.",
        "- Execution-layer maker viability remains unproven.",
    ]
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        for row in failed:
            lines.append(f"- `{row.check_id}`: {row.detail}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_preflight_artifacts(input_dir: Path = DEFAULT_INPUT_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    resolved_input = _expand(input_dir)
    resolved_output = _expand(output_dir)
    checks, reject_rows, loaded = validate_design_artifacts(resolved_input)
    failed = [row for row in checks if row.status != "pass"]
    final_recommendation = "preflight_validator_ready_for_qa" if not failed else "needs_more_validator_coverage"
    check_rows = [
        {
            "check_id": row.check_id,
            "check_category": row.check_category,
            "status": row.status,
            "detail": row.detail,
        }
        for row in checks
    ]
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "source_task_id": SOURCE_TASK_ID,
        "source_schema_version": loaded["manifest"].get("schema_version"),
        "source_final_recommendation": loaded["manifest"].get("final_recommendation"),
        "input_dir": str(resolved_input),
        "output_dir": str(resolved_output),
        "final_recommendation": final_recommendation,
        "allowed_final_recommendations": sorted(FINAL_RECOMMENDATIONS),
        "check_count": len(checks),
        "failed_check_count": len(failed),
        "reject_coverage_count": len(reject_rows),
        "boundary_flags": BOUNDARY_FLAGS,
        "output_artifacts": {
            "preflight_validator_manifest": str(resolved_output / "preflight_validator_manifest.json"),
            "preflight_validation_summary": str(resolved_output / "preflight_validation_summary.csv"),
            "preflight_reject_check_results": str(resolved_output / "preflight_reject_check_results.csv"),
            "preflight_validation_report": str(resolved_output / "preflight_validation_report.md"),
        },
    }
    _write_json(resolved_output / "preflight_validator_manifest.json", manifest)
    _write_csv(
        resolved_output / "preflight_validation_summary.csv",
        check_rows,
        ["check_id", "check_category", "status", "detail"],
    )
    _write_csv(
        resolved_output / "preflight_reject_check_results.csv",
        reject_rows,
        ["required_id", "source_file", "status", "detail"],
    )
    _write_report(
        resolved_output / "preflight_validation_report.md",
        input_dir=resolved_input,
        output_dir=resolved_output,
        checks=checks,
        final_recommendation=final_recommendation,
    )
    if failed:
        details = "; ".join(f"{row.check_id}: {row.detail}" for row in failed)
        raise PreflightValidationError(details)
    return {
        "manifest": manifest,
        "checks": check_rows,
        "reject_rows": reject_rows,
        "final_recommendation": final_recommendation,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate T006 row-level read-only generator design artifacts.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_preflight_artifacts(input_dir=args.input_dir, output_dir=args.output_dir)
    print(f"final_recommendation={result['final_recommendation']} checks={len(result['checks'])}")
    print(f"wrote {Path(args.output_dir).expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
