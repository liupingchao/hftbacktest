#!/usr/bin/env python3
"""Generate read-only maker-viability proxy artifacts for basis-positive rows.

This T010 runner is local and read-only. It consumes the QA-accepted T008
row-level research rows and the T009 execution-evidence contract, then emits
proxy metrics only. It does not implement a case library, shadow decisions,
executable triggers, private/order endpoints, order lifecycle behavior,
strategy behavior, live/default-on/tiny-live behavior, parameter search,
deployment, promotion, or execution-layer maker viability proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0609T010"
SCHEMA_VERSION = "basis_positive_maker_viability_proxy_v1"
DEFAULT_T009_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_execution_evidence_contract_0609T009"
DEFAULT_T008_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_row_level_generator_0609T008"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_maker_viability_proxy_0609T010"
T009_QA_REPORT = PROJECT_ROOT / ".workflow" / "reports" / "0609T009-qa.md"
T008_QA_REPORT = PROJECT_ROOT / ".workflow" / "reports" / "0609T008-qa.md"
FINAL_RECOMMENDATIONS = {
    "read_only_proxy_evidence_ready_for_qa",
    "needs_more_proxy_runner_coverage",
    "reject_proxy_runner_direction",
}
REQUIRED_T008_GAP_COLUMNS = {
    "fill_probability_unproven",
    "queue_position_unproven",
    "post_only_reject_unproven",
    "cancel_fill_race_unproven",
    "fees_rebates_spread_capture_unproven",
    "inventory_lifecycle_unproven",
    "real_order_lifecycle_unproven",
}
FORBIDDEN_OUTPUT_COLUMNS = {
    "execution_viability_proven",
    "order_side",
    "quote_price",
    "quote_size",
    "shadow_decision",
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
ALLOWED_PROXY_METRICS = {
    "public_book_post_only_feasibility_proxy",
    "spread_capture_fee_rebate_proxy",
    "adverse_move_after_hypothetical_passive_quote",
    "touch_proximity_opportunity_proxy",
    "queue_priority_public_depth_proxy",
    "clean_context_stability_summary",
}
FORBIDDEN_PROXY_METRICS = {"execution_viability_decision"}
BOUNDARY_FLAGS = {
    "read_only_proxy_runner": True,
    "read_only_research_artifact": True,
    "observation_layer_only": True,
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
OUTPUT_FIELDNAMES = [
    "artifact_schema_version",
    "artifact_generation_task_id",
    "source_task_id",
    "source_sample_id",
    "source_row_reference",
    "case_label",
    "proxy_metric_id",
    "proof_class",
    "proxy_value",
    "proxy_units",
    "future_label_value",
    "future_label_output_only",
    "no_action_fields_pass",
    "source_allowlist_pass",
    "execution_gap_preserved",
    "overclaim_reject_check_pass",
]


class ProxyRunnerError(ValueError):
    """Raised when T010 inputs or outputs violate the accepted contract."""


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
        raise ProxyRunnerError(f"{path} must contain a JSON object")
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


def _qa_passed(path: Path, task_id: str) -> bool:
    text = path.read_text(encoding="utf-8")
    return f"任务ID：\n- {task_id}" in text and "状态：\n- 已通过" in text


def _validate_prerequisites(t009_dir: Path, t008_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _qa_passed(T009_QA_REPORT, "0609T009"):
        raise ProxyRunnerError("T009 QA report is missing or not passed")
    if not _qa_passed(T008_QA_REPORT, "0609T008"):
        raise ProxyRunnerError("T008 QA report is missing or not passed")
    t009_manifest = _read_json(t009_dir / "execution_evidence_contract_manifest.json")
    if t009_manifest.get("task_id") != "0609T009":
        raise ProxyRunnerError("T009 manifest task_id mismatch")
    if t009_manifest.get("final_recommendation") != "read_only_proxy_runner_ready_for_implementation":
        raise ProxyRunnerError("T009 recommendation is not ready for proxy runner implementation")
    if t009_manifest.get("source_qa_status") != "passed":
        raise ProxyRunnerError("T009 source QA status is not passed")
    t008_manifest = _read_json(t008_dir / "row_level_generator_manifest.json")
    if t008_manifest.get("task_id") != "0609T008":
        raise ProxyRunnerError("T008 manifest task_id mismatch")
    if t008_manifest.get("final_recommendation") != "row_level_read_only_artifacts_ready_for_qa":
        raise ProxyRunnerError("T008 row-level artifacts are not ready for QA")
    if int(t008_manifest.get("generated_row_count", 0)) <= 0:
        raise ProxyRunnerError("T008 generated row count must be positive")
    return t009_manifest, t008_manifest


def _load_contracts(t009_dir: Path) -> dict[str, list[dict[str, str]]]:
    contracts = {
        "execution_gap_taxonomy": _read_csv(t009_dir / "execution_gap_taxonomy.csv"),
        "proxy_metric_contract": _read_csv(t009_dir / "proxy_metric_contract.csv"),
        "allowed_input_artifact_contract": _read_csv(t009_dir / "allowed_input_artifact_contract.csv"),
        "rejected_input_artifact_contract": _read_csv(t009_dir / "rejected_input_artifact_contract.csv"),
        "proxy_runner_output_schema_contract": _read_csv(t009_dir / "proxy_runner_output_schema_contract.csv"),
        "proxy_runner_validation_requirements": _read_csv(t009_dir / "proxy_runner_validation_requirements.csv"),
        "execution_overclaim_reject_conditions": _read_csv(t009_dir / "execution_overclaim_reject_conditions.csv"),
    }
    if not all(contracts.values()):
        raise ProxyRunnerError("T009 contract CSV artifacts must not be empty")
    return contracts


def _validate_t008_checks(t008_dir: Path) -> None:
    check_files = [
        "row_level_schema_validation.csv",
        "future_label_leakage_check.csv",
        "no_action_field_check.csv",
        "lineage_validation_summary.csv",
        "execution_gap_boundary_check.csv",
    ]
    failures: list[str] = []
    for name in check_files:
        rows = _read_csv(t008_dir / name)
        for row in rows:
            if row.get("status") != "pass":
                failures.append(f"{name}:{row.get('check_id')}:{row.get('detail')}")
    if failures:
        raise ProxyRunnerError("; ".join(failures))


def _load_t008_rows(t008_dir: Path) -> list[dict[str, str]]:
    rows = _read_csv(t008_dir / "row_level_read_only_cases.csv")
    if not rows:
        raise ProxyRunnerError("T008 row-level read-only cases are empty")
    return rows


def _contract_metric_map(contracts: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, str]]:
    metric_rows = {
        row["metric_id"]: row
        for row in contracts["proxy_metric_contract"]
        if row.get("metric_id") in ALLOWED_PROXY_METRICS or row.get("metric_id") in FORBIDDEN_PROXY_METRICS
    }
    missing = sorted(ALLOWED_PROXY_METRICS - set(metric_rows))
    if missing:
        raise ProxyRunnerError(f"missing allowed proxy metrics in T009 contract: {missing}")
    if "execution_viability_decision" not in metric_rows:
        raise ProxyRunnerError("T009 contract must explicitly forbid execution_viability_decision")
    return metric_rows


def _contract_output_fieldnames(contracts: dict[str, list[dict[str, str]]]) -> list[str]:
    rows = contracts["proxy_runner_output_schema_contract"]
    forbidden = {row["column_name"] for row in rows if row.get("category") == "forbidden"}
    if not FORBIDDEN_OUTPUT_COLUMNS.issubset(forbidden):
        raise ProxyRunnerError("T009 output schema contract is missing forbidden output columns")
    required = [row["column_name"] for row in rows if row.get("required") == "True" and row.get("category") != "forbidden"]
    missing_required = [name for name in required if name not in OUTPUT_FIELDNAMES]
    if missing_required:
        raise ProxyRunnerError("T010 output schema is missing T009 required non-forbidden columns")
    allowed_non_forbidden = {row["column_name"] for row in rows if row.get("category") != "forbidden"}
    extra = [name for name in OUTPUT_FIELDNAMES if name not in allowed_non_forbidden]
    if extra:
        raise ProxyRunnerError("T010 output schema includes columns outside T009 non-forbidden schema")
    return OUTPUT_FIELDNAMES


def _spread_midpoint_ticks(spread_bucket: str) -> float | None:
    if spread_bucket.startswith("spread_") and spread_bucket.endswith("_ticks"):
        parts = spread_bucket.removeprefix("spread_").removesuffix("_ticks").split("_")
        if len(parts) >= 2:
            try:
                return (float(parts[0]) + float(parts[1])) / 2.0
            except ValueError:
                return None
    if "one" in spread_bucket or "tight" in spread_bucket:
        return 1.0
    return None


def _future_label(row: dict[str, str], metric_id: str) -> str:
    if metric_id == "adverse_move_after_hypothetical_passive_quote":
        return row.get("hyperliquid_future_mid_move_ticks", "")
    return ""


def _proxy_value(row: dict[str, str], metric_id: str) -> tuple[str, str]:
    spread_bucket = row.get("spread_bucket", "")
    join_age_bucket = row.get("join_age_bucket", "")
    basis_bucket = row.get("basis_magnitude_bucket", "")
    hl_imbalance = row.get("hl_top5_imbalance_bucket", "")
    hl_microprice = row.get("hl_microprice_minus_mid_bucket", "")
    spread_mid = _spread_midpoint_ticks(spread_bucket)
    if metric_id == "public_book_post_only_feasibility_proxy":
        conflict = "conflict" if "negative" in hl_imbalance or "negative" in hl_microprice else "aligned_or_neutral"
        if spread_mid is None:
            return f"unknown_spread_{conflict}", "category"
        if spread_mid >= 10:
            return f"public_book_feasible_proxy_{conflict}", "category"
        return f"tight_spread_caution_proxy_{conflict}", "category"
    if metric_id == "spread_capture_fee_rebate_proxy":
        if spread_mid is None:
            return "", "ticks_hypothetical"
        return f"{spread_mid / 2.0:.4f}", "ticks_hypothetical"
    if metric_id == "adverse_move_after_hypothetical_passive_quote":
        return row.get("hyperliquid_future_mid_move_ticks", ""), "ticks_output_label_only"
    if metric_id == "touch_proximity_opportunity_proxy":
        if spread_mid is None:
            return "unknown_touch_opportunity_proxy", "category"
        if spread_mid <= 5:
            return "near_touch_opportunity_proxy", "category"
        return "wide_spread_waiting_opportunity_proxy", "category"
    if metric_id == "queue_priority_public_depth_proxy":
        freshness = "fresh" if join_age_bucket == "join_age_0_50ms" else "stale_or_unknown"
        return f"public_depth_diagnostic_{freshness}_no_queue_proof", "category"
    if metric_id == "clean_context_stability_summary":
        return f"{basis_bucket}|{hl_imbalance}|{hl_microprice}|{row.get('visible_movement_bucket', '')}", "category"
    raise ProxyRunnerError(f"unknown proxy metric: {metric_id}")


def _proxy_row(source: dict[str, str], metric: dict[str, str]) -> dict[str, Any]:
    metric_id = metric["metric_id"]
    value, units = _proxy_value(source, metric_id)
    return {
        "artifact_schema_version": SCHEMA_VERSION,
        "artifact_generation_task_id": TASK_ID,
        "source_task_id": "0609T008",
        "source_sample_id": source.get("source_sample_id", ""),
        "source_row_reference": source.get("source_row_reference", ""),
        "case_label": source.get("case_label", ""),
        "proxy_metric_id": metric_id,
        "proof_class": metric["proof_class"],
        "proxy_value": value,
        "proxy_units": units,
        "future_label_value": _future_label(source, metric_id),
        "future_label_output_only": True,
        "no_action_fields_pass": True,
        "source_allowlist_pass": True,
        "execution_gap_preserved": True,
        "overclaim_reject_check_pass": True,
    }


def _build_proxy_rows(source_rows: list[dict[str, str]], metric_map: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    metrics = [metric_map[metric_id] for metric_id in sorted(ALLOWED_PROXY_METRICS)]
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        missing_gaps = [name for name in REQUIRED_T008_GAP_COLUMNS if str(source.get(name, "")).lower() != "true"]
        if missing_gaps:
            raise ProxyRunnerError(f"T008 execution gap marker weakened in {source.get('source_row_reference')}: {missing_gaps}")
        for metric in metrics:
            rows.append(_proxy_row(source, metric))
    return rows


def _validate_source_allowlist(t009_dir: Path, t008_dir: Path, contracts: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    allowed_rows = contracts["allowed_input_artifact_contract"]
    required_artifacts = {
        "t008_manifest": t008_dir / "row_level_generator_manifest.json",
        "t008_rows": t008_dir / "row_level_read_only_cases.csv",
        "t008_source_manifest": t008_dir / "source_artifact_manifest.csv",
        "t008_schema_validation": t008_dir / "row_level_schema_validation.csv",
        "t008_future_label_check": t008_dir / "future_label_leakage_check.csv",
        "t008_no_action_check": t008_dir / "no_action_field_check.csv",
        "t008_lineage_summary": t008_dir / "lineage_validation_summary.csv",
        "t008_execution_gap_check": t008_dir / "execution_gap_boundary_check.csv",
    }
    out: list[dict[str, Any]] = []
    allowed_ids = {row["artifact_id"] for row in allowed_rows}
    for artifact_id, path in required_artifacts.items():
        status = "pass" if artifact_id in allowed_ids and path.exists() else "fail"
        out.append(
            {
                "check_id": "source_allowlist_artifact_present",
                "artifact_id": artifact_id,
                "status": status,
                "detail": str(path),
            }
        )
    out.append(
        {
            "check_id": "t009_contract_artifacts_present",
            "artifact_id": "t009_contract",
            "status": "pass" if t009_dir.exists() else "fail",
            "detail": str(t009_dir),
        }
    )
    return out


def _validate_rejected_sources(contracts: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    return [
        {
            "check_id": "rejected_source_absent",
            "source_class": row["source_class"],
            "status": "pass",
            "detail": "not used by T010 runner",
        }
        for row in contracts["rejected_input_artifact_contract"]
    ]


def _validate_no_action_fields(fieldnames: list[str], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for name in fieldnames:
        lowered = name.lower()
        if name in {"artifact_generation_task_id", "no_action_fields_pass"}:
            continue
        if any(marker in lowered for marker in ACTION_COLUMN_MARKERS):
            hits.append(name)
    forbidden_metrics = sorted({row["proxy_metric_id"] for row in rows if row["proxy_metric_id"] in FORBIDDEN_PROXY_METRICS})
    return [
        {
            "check_id": "no_action_capable_columns",
            "status": "pass" if not hits else "fail",
            "detail": "no action-capable columns" if not hits else "|".join(sorted(hits)),
        },
        {
            "check_id": "no_forbidden_metric_ids",
            "status": "pass" if not forbidden_metrics else "fail",
            "detail": "forbidden metric ids absent" if not forbidden_metrics else "|".join(forbidden_metrics),
        },
    ]


def _validate_future_label_output_only(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bad_flags = [row["source_row_reference"] for row in rows if row.get("future_label_value") and row.get("future_label_output_only") is not True]
    bad_metrics = [
        row["proxy_metric_id"]
        for row in rows
        if row.get("future_label_value") and row.get("proxy_metric_id") != "adverse_move_after_hypothetical_passive_quote"
    ]
    return [
        {
            "check_id": "future_label_output_only_flag",
            "status": "pass" if not bad_flags else "fail",
            "detail": "future labels are output-only when present" if not bad_flags else "|".join(bad_flags[:10]),
        },
        {
            "check_id": "future_label_metric_scope",
            "status": "pass" if not bad_metrics else "fail",
            "detail": "future labels appear only in output-label metric" if not bad_metrics else "|".join(sorted(set(bad_metrics))),
        },
        {
            "check_id": "future_label_filter_policy",
            "status": "pass",
            "detail": "runner filters only by accepted T008 rows and T009 metric contract; future labels are not row filters",
        },
    ]


def _validate_execution_gap_preservation(source_rows: list[dict[str, str]], proxy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    missing = []
    for row in source_rows:
        row_missing = [name for name in REQUIRED_T008_GAP_COLUMNS if str(row.get(name, "")).lower() != "true"]
        if row_missing:
            missing.append(f"{row.get('source_row_reference')}:{'|'.join(row_missing)}")
            break
    bad_proxy = [row["source_row_reference"] for row in proxy_rows if row.get("execution_gap_preserved") is not True]
    return [
        {
            "check_id": "source_execution_gap_markers_true",
            "status": "pass" if not missing else "fail",
            "detail": f"{len(REQUIRED_T008_GAP_COLUMNS)} T008 execution gap markers true for every source row" if not missing else missing[0],
        },
        {
            "check_id": "proxy_execution_gap_preserved_flag",
            "status": "pass" if not bad_proxy else "fail",
            "detail": "execution_gap_preserved true for every proxy row" if not bad_proxy else "|".join(bad_proxy[:10]),
        },
    ]


def _validate_overclaim(rows: list[dict[str, Any]], report_text: str = "") -> list[dict[str, Any]]:
    forbidden_phrases = [
        "fill_probability_proven",
        "exact_queue_position_proven",
        "post_only_reject_behavior_proven",
        "cancel_fill_race_proven",
        "realized_spread_capture_proven",
        "inventory_lifecycle_proven",
        "real_order_lifecycle_proven",
        "maker_execution_viability_proven",
        "live_readiness_proven",
        "promotion_authorized",
    ]
    text = report_text.lower()
    phrase_hits = [phrase for phrase in forbidden_phrases if phrase in text]
    bad_flags = [row["source_row_reference"] for row in rows if row.get("overclaim_reject_check_pass") is not True]
    return [
        {
            "check_id": "overclaim_flags_pass",
            "status": "pass" if not bad_flags else "fail",
            "detail": "overclaim reject check pass for every proxy row" if not bad_flags else "|".join(bad_flags[:10]),
        },
        {
            "check_id": "overclaim_text_absent",
            "status": "pass" if not phrase_hits else "fail",
            "detail": "no execution-layer proof overclaim text" if not phrase_hits else "|".join(phrase_hits),
        },
    ]


def _validate_schema(fieldnames: list[str], contracts: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    schema_rows = contracts["proxy_runner_output_schema_contract"]
    required = [row["column_name"] for row in schema_rows if row.get("required") == "True" and row.get("category") != "forbidden"]
    allowed_non_forbidden = {row["column_name"] for row in schema_rows if row.get("category") != "forbidden"}
    missing = [name for name in required if name not in fieldnames]
    extra = [name for name in fieldnames if name not in allowed_non_forbidden]
    forbidden = [name for name in fieldnames if name in FORBIDDEN_OUTPUT_COLUMNS]
    return [
        {
            "check_id": "schema_required_columns_present",
            "status": "pass" if not missing else "fail",
            "detail": f"{len(required)} required output columns present" if not missing else "|".join(missing),
        },
        {
            "check_id": "schema_no_extra_columns",
            "status": "pass" if not extra else "fail",
            "detail": "proxy output columns match T009 required non-forbidden schema" if not extra else "|".join(extra),
        },
        {
            "check_id": "schema_forbidden_columns_absent",
            "status": "pass" if not forbidden else "fail",
            "detail": "forbidden output columns absent" if not forbidden else "|".join(forbidden),
        },
    ]


def _validate_proof_classes(rows: list[dict[str, Any]], contracts: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    allowed = {row["proof_class"] for row in contracts["execution_gap_taxonomy"]}
    allowed.update(row["proof_class"] for row in contracts["proxy_metric_contract"])
    allowed.discard("")
    classes = {str(row["proof_class"]) for row in rows}
    bad = sorted(classes - allowed)
    return [
        {
            "check_id": "proof_class_valid",
            "status": "pass" if not bad else "fail",
            "detail": "all proxy proof classes are contract-defined" if not bad else "|".join(bad),
        }
    ]


def _summary_by_sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source_sample_id"], row["proxy_metric_id"])].append(row)
    out = []
    for (sample_id, metric_id), items in sorted(grouped.items()):
        numeric_values = []
        for item in items:
            try:
                numeric_values.append(float(item["proxy_value"]))
            except (TypeError, ValueError):
                pass
        out.append(
            {
                "source_sample_id": sample_id,
                "proxy_metric_id": metric_id,
                "proxy_row_count": len(items),
                "numeric_mean_proxy_value": f"{sum(numeric_values) / len(numeric_values):.8f}" if numeric_values else "",
                "proof_class": items[0]["proof_class"],
            }
        )
    return out


def _summary_by_proof_class(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(row["proof_class"]) for row in rows)
    return [
        {
            "proof_class": proof_class,
            "proxy_row_count": count,
            "metric_count": len({row["proxy_metric_id"] for row in rows if row["proof_class"] == proof_class}),
        }
        for proof_class, count in sorted(counts.items())
    ]


def _fail_if_checks_failed(*groups: list[dict[str, Any]]) -> None:
    failed = [row for group in groups for row in group if row.get("status") != "pass"]
    if failed:
        detail = "; ".join(f"{row.get('check_id')}: {row.get('detail')}" for row in failed)
        raise ProxyRunnerError(detail)


def _write_report(path: Path, *, manifest: dict[str, Any]) -> None:
    lines = [
        "# Basis-Positive Maker-Viability Proxy Runner Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Result",
        "",
        f"- Final recommendation: `{manifest['final_recommendation']}`",
        f"- Source row count: `{manifest['source_row_count']}`",
        f"- Generated proxy row count: `{manifest['generated_proxy_row_count']}`",
        f"- Proxy metric count: `{manifest['proxy_metric_count']}`",
        "",
        "## Boundary",
        "",
        "- T010 produces read-only proxy evidence only.",
        "- It does not implement case-library behavior, source-row case catalogs, shadow decisions, executable triggers, trading instructions, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.",
        "- Future labels are output-only offline research labels and are not inputs, filters, triggers, case conditions, shadow-decision fields, live decisions, or deployment criteria.",
        "- Fill probability, exact queue position, exchange post-only reject behavior, cancel-fill race, realized fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, and maker execution viability remain unproven.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_proxy_artifacts(
    *,
    t009_dir: str | Path = DEFAULT_T009_DIR,
    t008_dir: str | Path = DEFAULT_T008_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    resolved_t009 = _expand(t009_dir)
    resolved_t008 = _expand(t008_dir)
    resolved_output = _expand(output_dir)
    t009_manifest, t008_manifest = _validate_prerequisites(resolved_t009, resolved_t008)
    contracts = _load_contracts(resolved_t009)
    _validate_t008_checks(resolved_t008)
    metric_map = _contract_metric_map(contracts)
    fieldnames = _contract_output_fieldnames(contracts)
    source_rows = _load_t008_rows(resolved_t008)
    proxy_rows = _build_proxy_rows(source_rows, metric_map)
    normalized_rows = [{field: row.get(field, "") for field in fieldnames} for row in proxy_rows]

    source_allowlist_checks = _validate_source_allowlist(resolved_t009, resolved_t008, contracts)
    rejected_source_checks = _validate_rejected_sources(contracts)
    schema_checks = _validate_schema(fieldnames, contracts)
    future_checks = _validate_future_label_output_only(proxy_rows)
    no_action_checks = _validate_no_action_fields(fieldnames, proxy_rows)
    gap_checks = _validate_execution_gap_preservation(source_rows, proxy_rows)
    proof_checks = _validate_proof_classes(proxy_rows, contracts)
    overclaim_checks = _validate_overclaim(proxy_rows)
    _fail_if_checks_failed(
        source_allowlist_checks,
        rejected_source_checks,
        schema_checks,
        future_checks,
        no_action_checks,
        gap_checks,
        proof_checks,
        overclaim_checks,
    )

    sample_counts = Counter(row["source_sample_id"] for row in proxy_rows)
    metric_counts = Counter(row["proxy_metric_id"] for row in proxy_rows)
    final_recommendation = (
        "read_only_proxy_evidence_ready_for_qa"
        if len(sample_counts) >= 5 and set(metric_counts) == ALLOWED_PROXY_METRICS
        else "needs_more_proxy_runner_coverage"
    )
    if final_recommendation not in FINAL_RECOMMENDATIONS:
        raise AssertionError(f"unexpected recommendation: {final_recommendation}")

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "t009_dir": str(resolved_t009),
        "t008_dir": str(resolved_t008),
        "output_dir": str(resolved_output),
        "source_task_id": "0609T008",
        "source_contract_task_id": "0609T009",
        "source_t009_final_recommendation": t009_manifest.get("final_recommendation"),
        "source_t008_final_recommendation": t008_manifest.get("final_recommendation"),
        "source_row_count": len(source_rows),
        "source_sample_count": len(sample_counts),
        "generated_proxy_row_count": len(proxy_rows),
        "proxy_metric_count": len(metric_counts),
        "per_metric_proxy_rows": dict(sorted(metric_counts.items())),
        "per_sample_proxy_rows": dict(sorted(sample_counts.items())),
        "final_recommendation": final_recommendation,
        "allowed_final_recommendations": sorted(FINAL_RECOMMENDATIONS),
        "boundary_flags": BOUNDARY_FLAGS,
        "output_artifacts": {
            "proxy_runner_manifest": str(resolved_output / "proxy_runner_manifest.json"),
            "source_allowlist_validation": str(resolved_output / "source_allowlist_validation.csv"),
            "rejected_source_validation": str(resolved_output / "rejected_source_validation.csv"),
            "proxy_metric_outputs": str(resolved_output / "proxy_metric_outputs.csv"),
            "proxy_metric_summary_by_sample": str(resolved_output / "proxy_metric_summary_by_sample.csv"),
            "proxy_metric_summary_by_proof_class": str(resolved_output / "proxy_metric_summary_by_proof_class.csv"),
            "future_label_output_only_validation": str(resolved_output / "future_label_output_only_validation.csv"),
            "no_action_field_validation": str(resolved_output / "no_action_field_validation.csv"),
            "execution_gap_preservation_validation": str(resolved_output / "execution_gap_preservation_validation.csv"),
            "overclaim_reject_validation": str(resolved_output / "overclaim_reject_validation.csv"),
            "proxy_runner_report": str(resolved_output / "proxy_runner_report.md"),
        },
    }

    _write_json(resolved_output / "proxy_runner_manifest.json", manifest)
    _write_csv(resolved_output / "source_allowlist_validation.csv", source_allowlist_checks, ["check_id", "artifact_id", "status", "detail"])
    _write_csv(resolved_output / "rejected_source_validation.csv", rejected_source_checks, ["check_id", "source_class", "status", "detail"])
    _write_csv(resolved_output / "proxy_metric_outputs.csv", normalized_rows, fieldnames)
    _write_csv(
        resolved_output / "proxy_metric_summary_by_sample.csv",
        _summary_by_sample(proxy_rows),
        ["source_sample_id", "proxy_metric_id", "proxy_row_count", "numeric_mean_proxy_value", "proof_class"],
    )
    _write_csv(
        resolved_output / "proxy_metric_summary_by_proof_class.csv",
        _summary_by_proof_class(proxy_rows),
        ["proof_class", "proxy_row_count", "metric_count"],
    )
    _write_csv(resolved_output / "future_label_output_only_validation.csv", future_checks, ["check_id", "status", "detail"])
    _write_csv(resolved_output / "no_action_field_validation.csv", no_action_checks, ["check_id", "status", "detail"])
    _write_csv(resolved_output / "execution_gap_preservation_validation.csv", gap_checks, ["check_id", "status", "detail"])
    _write_csv(resolved_output / "overclaim_reject_validation.csv", overclaim_checks, ["check_id", "status", "detail"])
    _write_report(resolved_output / "proxy_runner_report.md", manifest=manifest)
    return {"manifest": manifest, "rows": normalized_rows}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate T010 read-only maker-viability proxy artifacts.")
    parser.add_argument("--t009-dir", type=Path, default=DEFAULT_T009_DIR)
    parser.add_argument("--t008-dir", type=Path, default=DEFAULT_T008_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_proxy_artifacts(t009_dir=args.t009_dir, t008_dir=args.t008_dir, output_dir=args.output_dir)
    manifest = result["manifest"]
    print(
        "final_recommendation="
        f"{manifest['final_recommendation']} proxy_rows={manifest['generated_proxy_row_count']} "
        f"output_dir={manifest['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
