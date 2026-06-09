#!/usr/bin/env python3
"""Synthesize T010 read-only proxy evidence into next evidence decisions.

This T011 runner consumes only QA-accepted T010 local proxy artifacts and
reports. It emits aggregate decision matrices, execution-evidence gap
requirements, boundary validation, a manifest, and a report. It does not create
case libraries, source-row case catalogs, shadow decisions, executable triggers,
strategy/private/order/live/default-on/tiny-live behavior, parameter search,
deployment recommendations, promotion, or execution-layer maker viability proof.
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
TASK_ID = "0609T011"
SCHEMA_VERSION = "basis_positive_proxy_evidence_synthesis_v1"
DEFAULT_T010_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_maker_viability_proxy_0609T010"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "basis_positive_proxy_evidence_synthesis_0609T011"
DEFAULT_T010_QA_REPORT = PROJECT_ROOT / ".workflow" / "reports" / "0609T010-qa.md"
DEFAULT_T010_BUSINESS_REPORT = PROJECT_ROOT / ".workflow" / "reports" / "0609T010-business.md"

T010_READY_RECOMMENDATION = "read_only_proxy_evidence_ready_for_qa"
FINAL_RECOMMENDATIONS = {
    "continue_to_execution_evidence_design",
    "needs_more_proxy_synthesis",
    "reject_current_maker_direction_from_proxy",
}
T010_REQUIRED_FILES = {
    "proxy_runner_manifest": "proxy_runner_manifest.json",
    "proxy_metric_outputs": "proxy_metric_outputs.csv",
    "proxy_metric_summary_by_sample": "proxy_metric_summary_by_sample.csv",
    "proxy_metric_summary_by_proof_class": "proxy_metric_summary_by_proof_class.csv",
    "future_label_output_only_validation": "future_label_output_only_validation.csv",
    "no_action_field_validation": "no_action_field_validation.csv",
    "execution_gap_preservation_validation": "execution_gap_preservation_validation.csv",
    "overclaim_reject_validation": "overclaim_reject_validation.csv",
}
ALLOWED_PROXY_METRICS = {
    "adverse_move_after_hypothetical_passive_quote",
    "clean_context_stability_summary",
    "public_book_post_only_feasibility_proxy",
    "queue_priority_public_depth_proxy",
    "spread_capture_fee_rebate_proxy",
    "touch_proximity_opportunity_proxy",
}
PUBLIC_OBSERVATION_PROOF_CLASS = "proxy_available_from_public_observation_rows"
STRICT_CAVEAT_PROOF_CLASS = "proxy_available_with_strict_caveat"
ACTION_CAPABLE_COLUMN_MARKERS = {
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
    "shadow",
    "live",
    "deploy",
    "promotion",
}
ALLOWED_BOUNDARY_COLUMN_EXCEPTIONS = {
    "boundary_no_action_capable_outputs",
    "current_t010_status",
    "minimum_next_requirement",
    "signal_direction",
}
FORBIDDEN_POSITIVE_AUTHORIZATION_FRAGMENTS = [
    ("case_library", "authorized"),
    ("case_catalog", "authorized"),
    ("shadow_decision", "authorized"),
    ("executable_trigger", "authorized"),
    ("trading_action", "authorized"),
    ("strategy", "authorized"),
    ("private_order", "authorized"),
    ("live", "authorized"),
    ("default_on", "authorized"),
    ("tiny_live", "authorized"),
    ("parameter_search", "authorized"),
    ("deployment", "authorized"),
    ("promotion", "authorized"),
    ("maker_execution_viability", "proven"),
    ("fill_probability", "proven"),
    ("queue_position", "proven"),
    ("post_only_reject", "proven"),
    ("cancel_fill_race", "proven"),
    ("fees_rebates_spread_capture", "proven"),
    ("inventory_lifecycle", "proven"),
    ("real_order_lifecycle", "proven"),
    ("pnl", "proven"),
]

OUTPUT_ARTIFACTS = {
    "proxy_evidence_synthesis_manifest": "proxy_evidence_synthesis_manifest.json",
    "metric_decision_matrix": "metric_decision_matrix.csv",
    "sample_decision_matrix": "sample_decision_matrix.csv",
    "proof_class_decision_matrix": "proof_class_decision_matrix.csv",
    "execution_evidence_gap_next_requirements": "execution_evidence_gap_next_requirements.csv",
    "boundary_validation": "boundary_validation.csv",
    "proxy_evidence_synthesis_report": "proxy_evidence_synthesis_report.md",
}

METRIC_CAVEATS = {
    "adverse_move_after_hypothetical_passive_quote": (
        "future label is output-only offline observation; not a fill, queue, order, PnL, or maker viability proof"
    ),
    "clean_context_stability_summary": (
        "decision-time public context summary only; does not define case-library conditions or executable filters"
    ),
    "public_book_post_only_feasibility_proxy": (
        "public book category proxy only; not exchange post-only reject behavior or order acceptance proof"
    ),
    "queue_priority_public_depth_proxy": (
        "public depth freshness proxy only; not exact queue position, priority, or fill probability proof"
    ),
    "spread_capture_fee_rebate_proxy": (
        "hypothetical half-spread proxy only; excludes realized fees, rebates, spread capture, and PnL"
    ),
    "touch_proximity_opportunity_proxy": (
        "public spread/touch opportunity proxy only; not an order placement instruction or execution lifecycle proof"
    ),
}

METRIC_BLOCKER_GAPS = {
    "adverse_move_after_hypothetical_passive_quote": "fill_probability|queue_priority|post_only_reject_behavior|cancel_fill_race|fees_rebates_spread_capture|inventory_lifecycle|real_order_lifecycle",
    "clean_context_stability_summary": "fill_probability|queue_priority|post_only_reject_behavior|cancel_fill_race|fees_rebates_spread_capture|inventory_lifecycle|real_order_lifecycle",
    "public_book_post_only_feasibility_proxy": "post_only_reject_behavior|real_order_lifecycle|fill_probability",
    "queue_priority_public_depth_proxy": "queue_priority|fill_probability|real_order_lifecycle",
    "spread_capture_fee_rebate_proxy": "fees_rebates_spread_capture|fill_probability|real_order_lifecycle|PnL",
    "touch_proximity_opportunity_proxy": "fill_probability|queue_priority|post_only_reject_behavior|real_order_lifecycle",
}

GAP_REQUIREMENTS = [
    (
        "fill_probability",
        "not_provable_from_t010_proxy",
        "T010 has observation rows and output-only future labels but no submitted passive orders or fill outcomes.",
        "separately scoped execution-evidence design defining allowed labels, observation unit, sample policy, and validation for real or accepted simulated fill outcomes",
    ),
    (
        "queue_priority",
        "not_provable_from_t010_proxy",
        "T010 public-depth freshness cannot identify exact exchange queue position or priority.",
        "separately scoped queue/priority evidence design with explicit source contract, timestamp policy, and proof limits",
    ),
    (
        "post_only_reject_behavior",
        "not_provable_from_t010_proxy",
        "T010 public book categories do not observe exchange post-only checks or rejection responses.",
        "separately scoped post-only evidence design requiring response-level labels before any execution interpretation",
    ),
    (
        "cancel_fill_race",
        "not_provable_from_t010_proxy",
        "T010 does not contain submitted orders, cancel requests, acknowledgements, fills, or race timing.",
        "separately scoped lifecycle evidence design with cancel/request/fill timing labels and overclaim guards",
    ),
    (
        "fees_rebates_spread_capture",
        "not_provable_from_t010_proxy",
        "T010 spread proxy is hypothetical and excludes realized fees, rebates, spread capture, and PnL.",
        "separately scoped economics evidence design defining fee/rebate source, spread-capture labels, and PnL exclusion or inclusion rules",
    ),
    (
        "inventory_lifecycle",
        "not_provable_from_t010_proxy",
        "T010 has no positions, inventory transitions, inventory risk controls, or lifecycle state.",
        "separately scoped inventory evidence design defining inventory state source, transitions, and non-live validation boundaries",
    ),
    (
        "real_order_lifecycle",
        "not_provable_from_t010_proxy",
        "T010 is public/local proxy evidence and has no real order submit/ack/reject/cancel/fill lifecycle.",
        "separately scoped real-order lifecycle evidence design before any private/order endpoint or live consideration",
    ),
]


class ProxyEvidenceSynthesisError(ValueError):
    """Raised when T011 inputs or boundary checks fail closed."""


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
        raise ProxyEvidenceSynthesisError(f"{path} must contain a JSON object")
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


def _status_rows_all_pass(rows: list[dict[str, str]]) -> bool:
    return bool(rows) and all(row.get("status") == "pass" for row in rows)


def _load_t010_inputs(
    *,
    t010_dir: Path,
    t010_qa_report: Path,
    t010_business_report: Path,
) -> dict[str, Any]:
    if not _qa_passed(t010_qa_report, "0609T010"):
        raise ProxyEvidenceSynthesisError("T010 QA report is missing or not passed")
    if not t010_business_report.exists():
        raise ProxyEvidenceSynthesisError("T010 business report is missing")

    missing = [name for name in T010_REQUIRED_FILES.values() if not (t010_dir / name).exists()]
    if missing:
        raise ProxyEvidenceSynthesisError(f"T010 required artifacts missing: {missing}")

    manifest = _read_json(t010_dir / "proxy_runner_manifest.json")
    if manifest.get("task_id") != "0609T010":
        raise ProxyEvidenceSynthesisError("T010 manifest task_id mismatch")
    if manifest.get("final_recommendation") != T010_READY_RECOMMENDATION:
        raise ProxyEvidenceSynthesisError("T010 final recommendation is not read-only proxy evidence ready")
    if int(manifest.get("source_row_count", 0)) <= 0:
        raise ProxyEvidenceSynthesisError("T010 source row count must be positive")
    if int(manifest.get("proxy_metric_count", 0)) <= 0:
        raise ProxyEvidenceSynthesisError("T010 proxy metric count must be positive")

    outputs = _read_csv(t010_dir / "proxy_metric_outputs.csv")
    sample_summary = _read_csv(t010_dir / "proxy_metric_summary_by_sample.csv")
    proof_summary = _read_csv(t010_dir / "proxy_metric_summary_by_proof_class.csv")
    future_checks = _read_csv(t010_dir / "future_label_output_only_validation.csv")
    no_action_checks = _read_csv(t010_dir / "no_action_field_validation.csv")
    gap_checks = _read_csv(t010_dir / "execution_gap_preservation_validation.csv")
    overclaim_checks = _read_csv(t010_dir / "overclaim_reject_validation.csv")
    validation_groups = [future_checks, no_action_checks, gap_checks, overclaim_checks]
    if not all(_status_rows_all_pass(group) for group in validation_groups):
        raise ProxyEvidenceSynthesisError("T010 validation artifacts must all pass")

    metric_ids = {row.get("proxy_metric_id", "") for row in outputs}
    if metric_ids != ALLOWED_PROXY_METRICS:
        raise ProxyEvidenceSynthesisError(f"T010 proxy metric set mismatch: {sorted(metric_ids)}")
    if any(str(row.get("execution_gap_preserved", "")).lower() != "true" for row in outputs):
        raise ProxyEvidenceSynthesisError("T010 proxy rows must preserve execution gaps")
    if any(str(row.get("overclaim_reject_check_pass", "")).lower() != "true" for row in outputs):
        raise ProxyEvidenceSynthesisError("T010 proxy rows must pass overclaim checks")

    return {
        "manifest": manifest,
        "outputs": outputs,
        "sample_summary": sample_summary,
        "proof_summary": proof_summary,
        "future_checks": future_checks,
        "no_action_checks": no_action_checks,
        "gap_checks": gap_checks,
        "overclaim_checks": overclaim_checks,
    }


def _as_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _numeric_mean(rows: list[dict[str, str]], field: str) -> float | None:
    values = [value for row in rows if (value := _as_float(row.get(field, ""))) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _numeric_min_max(rows: list[dict[str, str]], field: str) -> tuple[float | None, float | None]:
    values = [value for row in rows if (value := _as_float(row.get(field, ""))) is not None]
    if not values:
        return None, None
    return min(values), max(values)


def _top_categories(rows: list[dict[str, str]], field: str, limit: int = 3) -> str:
    counts = Counter(row.get(field, "") for row in rows if row.get(field, "") != "")
    if not counts:
        return ""
    return "|".join(f"{name}:{count}" for name, count in counts.most_common(limit))


def _signal_direction(metric_id: str, rows: list[dict[str, str]]) -> str:
    if metric_id == "adverse_move_after_hypothetical_passive_quote":
        mean_value = _numeric_mean(rows, "proxy_value")
        if mean_value is None:
            return "not_numeric"
        if mean_value > 0:
            return "positive_future_mid_move_output_label"
        if mean_value < 0:
            return "negative_future_mid_move_output_label"
        return "flat_future_mid_move_output_label"
    if metric_id == "spread_capture_fee_rebate_proxy":
        mean_value = _numeric_mean(rows, "proxy_value")
        if mean_value is None:
            return "not_numeric"
        if mean_value > 0:
            return "positive_hypothetical_half_spread_proxy"
        return "non_positive_hypothetical_half_spread_proxy"
    return "categorical_public_context_proxy"


def _metric_decision(metric_id: str, proof_class: str, rows: list[dict[str, str]], sample_count: int) -> str:
    if proof_class == PUBLIC_OBSERVATION_PROOF_CLASS and sample_count >= 3:
        return "supports_execution_evidence_design_input"
    if proof_class == STRICT_CAVEAT_PROOF_CLASS and sample_count >= 3:
        return "supports_execution_evidence_design_with_strict_caveat"
    return "needs_more_proxy_synthesis"


def _build_metric_decision_matrix(outputs: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in outputs:
        grouped[row["proxy_metric_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for metric_id in sorted(grouped):
        items = grouped[metric_id]
        proof_classes = sorted({row.get("proof_class", "") for row in items})
        proof_class = proof_classes[0] if len(proof_classes) == 1 else "|".join(proof_classes)
        sample_count = len({row.get("source_sample_id", "") for row in items})
        mean_value = _numeric_mean(items, "proxy_value")
        min_value, max_value = _numeric_min_max(items, "proxy_value")
        decision = _metric_decision(metric_id, proof_class, items, sample_count)
        rows.append(
            {
                "proxy_metric_id": metric_id,
                "proof_class": proof_class,
                "proxy_row_count": len(items),
                "source_sample_count": sample_count,
                "numeric_mean_proxy_value": f"{mean_value:.8f}" if mean_value is not None else "",
                "numeric_min_proxy_value": f"{min_value:.8f}" if min_value is not None else "",
                "numeric_max_proxy_value": f"{max_value:.8f}" if max_value is not None else "",
                "top_proxy_values": _top_categories(items, "proxy_value"),
                "signal_direction": _signal_direction(metric_id, items),
                "synthesis_classification": decision,
                "caveat": METRIC_CAVEATS[metric_id],
                "blocker_gaps": METRIC_BLOCKER_GAPS[metric_id],
                "next_evidence_implication": (
                    "eligible only as input to a later execution-evidence requirements design; no implementation/live authorization"
                ),
            }
        )
    return rows


def _support_label(adverse_mean: float | None, strict_caveat_metric_count: int) -> str:
    if adverse_mean is not None and adverse_mean > 0 and strict_caveat_metric_count >= 4:
        return "proxy_support_present_with_execution_gaps"
    if adverse_mean is not None and adverse_mean <= 0:
        return "proxy_weakness_negative_or_flat_output_label"
    return "proxy_support_incomplete"


def _build_sample_decision_matrix(outputs: list[dict[str, str]]) -> list[dict[str, Any]]:
    by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in outputs:
        by_sample[row["source_sample_id"]].append(row)

    rows: list[dict[str, Any]] = []
    for sample_id in sorted(by_sample):
        items = by_sample[sample_id]
        metric_ids = sorted({row["proxy_metric_id"] for row in items})
        adverse_rows = [row for row in items if row["proxy_metric_id"] == "adverse_move_after_hypothetical_passive_quote"]
        spread_rows = [row for row in items if row["proxy_metric_id"] == "spread_capture_fee_rebate_proxy"]
        adverse_mean = _numeric_mean(adverse_rows, "proxy_value")
        spread_mean = _numeric_mean(spread_rows, "proxy_value")
        strict_caveat_metric_count = len(
            {row["proxy_metric_id"] for row in items if row.get("proof_class") == STRICT_CAVEAT_PROOF_CLASS}
        )
        source_row_count = len({row.get("source_row_reference", "") for row in items})
        rows.append(
            {
                "source_sample_id": sample_id,
                "source_row_count": source_row_count,
                "proxy_row_count": len(items),
                "metric_count": len(metric_ids),
                "proof_class_count": len({row.get("proof_class", "") for row in items}),
                "mean_adverse_move_output_label_ticks": f"{adverse_mean:.8f}" if adverse_mean is not None else "",
                "mean_hypothetical_half_spread_ticks": f"{spread_mean:.8f}" if spread_mean is not None else "",
                "strict_caveat_metric_count": strict_caveat_metric_count,
                "support_summary": _support_label(adverse_mean, strict_caveat_metric_count),
                "weakness_summary": (
                    "sample remains proxy-only; execution-layer gaps are unchanged and no source-row case catalog is emitted"
                ),
                "next_evidence_implication": (
                    "use only for later aggregate execution-evidence design scoping; not for cases, shadows, or trading instructions"
                ),
            }
        )
    return rows


def _build_proof_class_decision_matrix(
    outputs: list[dict[str, str]],
    proof_summary: list[dict[str, str]],
) -> list[dict[str, Any]]:
    summary_by_class = {row["proof_class"]: row for row in proof_summary}
    rows: list[dict[str, Any]] = []
    for proof_class in sorted({row.get("proof_class", "") for row in outputs}):
        items = [row for row in outputs if row.get("proof_class") == proof_class]
        metrics = sorted({row["proxy_metric_id"] for row in items})
        summary = summary_by_class.get(proof_class, {})
        if proof_class == PUBLIC_OBSERVATION_PROOF_CLASS:
            decision = "public_observation_proxy_supports_next_design_input"
            caveat = "public observation proxy only; still no execution lifecycle, order response, fill, PnL, or maker viability proof"
        elif proof_class == STRICT_CAVEAT_PROOF_CLASS:
            decision = "strict_caveat_proxy_supports_gap_requirements_only"
            caveat = "strict caveat proxy only; cannot prove exchange execution mechanics or economics"
        else:
            decision = "unknown_proof_class_needs_review"
            caveat = "proof class is not recognized by T011"
        rows.append(
            {
                "proof_class": proof_class,
                "metric_ids": "|".join(metrics),
                "proxy_row_count": summary.get("proxy_row_count", len(items)),
                "metric_count": summary.get("metric_count", len(metrics)),
                "source_sample_count": len({row.get("source_sample_id", "") for row in items}),
                "decision": decision,
                "caveat": caveat,
                "next_evidence_implication": (
                    "separate execution-evidence requirements design only; no strategy/private/order/live/default-on/tiny-live authorization"
                ),
            }
        )
    return rows


def _build_gap_requirements() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gap_id, status, why_not_proof, requirement in GAP_REQUIREMENTS:
        rows.append(
            {
                "execution_gap_id": gap_id,
                "current_t010_status": status,
                "why_t010_proxy_is_not_proof": why_not_proof,
                "minimum_next_requirement": requirement,
                "allowed_next_step": "later separately scoped execution-evidence design task",
                "forbidden_interpretation": (
                    "do not treat T010 or T011 proxy artifacts as execution-layer maker viability, live readiness, or promotion proof"
                ),
            }
        )
    return rows


def _scan_action_capable_columns(artifact_rows: dict[str, tuple[list[dict[str, Any]], list[str]]]) -> list[str]:
    hits: list[str] = []
    for artifact_name, (_, fieldnames) in artifact_rows.items():
        for field in fieldnames:
            if field in ALLOWED_BOUNDARY_COLUMN_EXCEPTIONS:
                continue
            lowered = field.lower()
            if any(marker in lowered for marker in ACTION_CAPABLE_COLUMN_MARKERS):
                hits.append(f"{artifact_name}.{field}")
    return sorted(hits)


def _scan_forbidden_positive_authorization(texts: list[str]) -> list[str]:
    haystack = "\n".join(texts).lower()
    phrases = {f"{left}_{right}" for left, right in FORBIDDEN_POSITIVE_AUTHORIZATION_FRAGMENTS}
    return sorted(phrase for phrase in phrases if phrase in haystack)


def _build_boundary_validation(
    *,
    t010_inputs: dict[str, Any],
    final_recommendation: str,
    artifact_rows: dict[str, tuple[list[dict[str, Any]], list[str]]],
    report_text: str,
) -> list[dict[str, Any]]:
    manifest = t010_inputs["manifest"]
    action_column_hits = _scan_action_capable_columns(artifact_rows)
    positive_authorization_hits = _scan_forbidden_positive_authorization([report_text, json.dumps(manifest, sort_keys=True)])
    rows = [
        {
            "check_id": "t010_final_recommendation_ready",
            "status": "pass" if manifest.get("final_recommendation") == T010_READY_RECOMMENDATION else "fail",
            "detail": str(manifest.get("final_recommendation")),
        },
        {
            "check_id": "t010_validation_artifacts_all_pass",
            "status": "pass",
            "detail": "future-label, no-action, execution-gap, and overclaim validations are all pass",
        },
        {
            "check_id": "t011_final_recommendation_taxonomy",
            "status": "pass" if final_recommendation in FINAL_RECOMMENDATIONS else "fail",
            "detail": final_recommendation,
        },
        {
            "check_id": "t011_no_action_capable_output_columns",
            "status": "pass" if not action_column_hits else "fail",
            "detail": "no action-capable output columns" if not action_column_hits else "|".join(action_column_hits),
        },
        {
            "check_id": "t011_no_source_row_case_catalog",
            "status": "pass",
            "detail": "T011 emits aggregate matrices only and does not emit source_row_reference outputs",
        },
        {
            "check_id": "t011_execution_gaps_preserved",
            "status": "pass",
            "detail": "fill probability, queue/priority, post-only reject, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, and real order lifecycle remain unproven",
        },
        {
            "check_id": "t011_no_positive_authorization_text",
            "status": "pass" if not positive_authorization_hits else "fail",
            "detail": "no positive authorization phrases found"
            if not positive_authorization_hits
            else "|".join(positive_authorization_hits),
        },
        {
            "check_id": "t011_continue_semantics_limited_to_design",
            "status": "pass",
            "detail": "`continue_to_execution_evidence_design` means only a later separately scoped design task",
        },
    ]
    return rows


def _fail_if_boundary_failed(boundary_rows: list[dict[str, Any]]) -> None:
    failed = [row for row in boundary_rows if row.get("status") != "pass"]
    if failed:
        detail = "; ".join(f"{row['check_id']}={row['detail']}" for row in failed)
        raise ProxyEvidenceSynthesisError(detail)


def _choose_final_recommendation(
    metric_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    proof_rows: list[dict[str, Any]],
) -> str:
    if len(metric_rows) < len(ALLOWED_PROXY_METRICS) or len(sample_rows) < 3:
        return "needs_more_proxy_synthesis"
    if any(row["synthesis_classification"] == "needs_more_proxy_synthesis" for row in metric_rows):
        return "needs_more_proxy_synthesis"
    if any(row["decision"] == "unknown_proof_class_needs_review" for row in proof_rows):
        return "needs_more_proxy_synthesis"
    if all(row["support_summary"] == "proxy_weakness_negative_or_flat_output_label" for row in sample_rows):
        return "reject_current_maker_direction_from_proxy"
    return "continue_to_execution_evidence_design"


def _report_text(
    *,
    manifest: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    sample_rows: list[dict[str, Any]],
    proof_rows: list[dict[str, Any]],
) -> str:
    lines = [
        "# Basis-Positive Proxy Evidence Synthesis Report",
        "",
        f"Task: `{TASK_ID}`",
        "",
        "## Result",
        "",
        f"- Final recommendation: `{manifest['final_recommendation']}`",
        f"- T010 source rows: `{manifest['t010_source_row_count']}`",
        f"- T010 proxy rows: `{manifest['t010_proxy_row_count']}`",
        f"- Metric decision rows: `{len(metric_rows)}`",
        f"- Sample decision rows: `{len(sample_rows)}`",
        f"- Proof-class decision rows: `{len(proof_rows)}`",
        "",
        "## Interpretation",
        "",
        "- T011 is read-only proxy synthesis only.",
        "- `continue_to_execution_evidence_design` means only that a later separately scoped design task can define execution-layer evidence requirements.",
        "- T011 does not authorize case-library implementation, source-row case catalogs, shadow decisions, executable triggers, trading instructions, order side, quote price/size, strategy/private/order/live/default-on/tiny-live behavior, parameter search, deployment recommendation, promotion, or execution-layer maker viability proof.",
        "- Fill probability, exact queue position, exchange post-only reject behavior, cancel-fill race, realized fees/rebates/spread capture, inventory lifecycle, real order lifecycle, PnL, live readiness, default-on readiness, tiny-live readiness, deployment readiness, and promotion remain unproven.",
        "",
        "## Next Evidence",
        "",
        "- The next useful step is an execution-evidence requirements design task, not implementation or live behavior.",
        "- Required evidence areas: fill probability, queue/priority, post-only reject behavior, cancel-fill race, fees/rebates/spread capture, inventory lifecycle, and real order lifecycle.",
    ]
    return "\n".join(lines) + "\n"


def build_synthesis_artifacts(
    *,
    t010_dir: str | Path = DEFAULT_T010_DIR,
    t010_qa_report: str | Path = DEFAULT_T010_QA_REPORT,
    t010_business_report: str | Path = DEFAULT_T010_BUSINESS_REPORT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    resolved_t010_dir = _expand(t010_dir)
    resolved_qa_report = _expand(t010_qa_report)
    resolved_business_report = _expand(t010_business_report)
    resolved_output = _expand(output_dir)

    t010_inputs = _load_t010_inputs(
        t010_dir=resolved_t010_dir,
        t010_qa_report=resolved_qa_report,
        t010_business_report=resolved_business_report,
    )
    outputs = t010_inputs["outputs"]
    metric_rows = _build_metric_decision_matrix(outputs)
    sample_rows = _build_sample_decision_matrix(outputs)
    proof_rows = _build_proof_class_decision_matrix(outputs, t010_inputs["proof_summary"])
    gap_rows = _build_gap_requirements()
    final_recommendation = _choose_final_recommendation(metric_rows, sample_rows, proof_rows)
    if final_recommendation not in FINAL_RECOMMENDATIONS:
        raise AssertionError(f"unexpected final recommendation: {final_recommendation}")

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "t010_dir": str(resolved_t010_dir),
        "t010_qa_report": str(resolved_qa_report),
        "t010_business_report": str(resolved_business_report),
        "output_dir": str(resolved_output),
        "source_task_id": "0609T010",
        "source_final_recommendation": t010_inputs["manifest"].get("final_recommendation"),
        "t010_source_row_count": t010_inputs["manifest"].get("source_row_count"),
        "t010_source_sample_count": t010_inputs["manifest"].get("source_sample_count"),
        "t010_proxy_row_count": t010_inputs["manifest"].get("generated_proxy_row_count"),
        "t010_proxy_metric_count": t010_inputs["manifest"].get("proxy_metric_count"),
        "metric_decision_row_count": len(metric_rows),
        "sample_decision_row_count": len(sample_rows),
        "proof_class_decision_row_count": len(proof_rows),
        "execution_evidence_gap_count": len(gap_rows),
        "final_recommendation": final_recommendation,
        "allowed_final_recommendations": sorted(FINAL_RECOMMENDATIONS),
        "boundary_flags": {
            "read_only_proxy_synthesis": True,
            "consumes_only_t010_local_artifacts_and_reports": True,
            "no_case_library_implementation": True,
            "no_source_row_case_catalog": True,
            "no_shadow_decisions": True,
            "no_executable_triggers": True,
            "no_trading_instructions": True,
            "no_order_side_quote_price_or_quote_size": True,
            "no_strategy_private_order_live_default_on_tiny_live": True,
            "no_parameter_search_deployment_or_promotion": True,
            "execution_layer_maker_viability_unproven": True,
        },
        "output_artifacts": {name: str(resolved_output / filename) for name, filename in OUTPUT_ARTIFACTS.items()},
    }

    report = _report_text(manifest=manifest, metric_rows=metric_rows, sample_rows=sample_rows, proof_rows=proof_rows)
    artifact_rows = {
        "metric_decision_matrix": (
            metric_rows,
            [
                "proxy_metric_id",
                "proof_class",
                "proxy_row_count",
                "source_sample_count",
                "numeric_mean_proxy_value",
                "numeric_min_proxy_value",
                "numeric_max_proxy_value",
                "top_proxy_values",
                "signal_direction",
                "synthesis_classification",
                "caveat",
                "blocker_gaps",
                "next_evidence_implication",
            ],
        ),
        "sample_decision_matrix": (
            sample_rows,
            [
                "source_sample_id",
                "source_row_count",
                "proxy_row_count",
                "metric_count",
                "proof_class_count",
                "mean_adverse_move_output_label_ticks",
                "mean_hypothetical_half_spread_ticks",
                "strict_caveat_metric_count",
                "support_summary",
                "weakness_summary",
                "next_evidence_implication",
            ],
        ),
        "proof_class_decision_matrix": (
            proof_rows,
            [
                "proof_class",
                "metric_ids",
                "proxy_row_count",
                "metric_count",
                "source_sample_count",
                "decision",
                "caveat",
                "next_evidence_implication",
            ],
        ),
        "execution_evidence_gap_next_requirements": (
            gap_rows,
            [
                "execution_gap_id",
                "current_t010_status",
                "why_t010_proxy_is_not_proof",
                "minimum_next_requirement",
                "allowed_next_step",
                "forbidden_interpretation",
            ],
        ),
    }
    boundary_rows = _build_boundary_validation(
        t010_inputs=t010_inputs,
        final_recommendation=final_recommendation,
        artifact_rows=artifact_rows,
        report_text=report,
    )
    _fail_if_boundary_failed(boundary_rows)
    manifest["boundary_validation_passed"] = True
    manifest["boundary_validation_row_count"] = len(boundary_rows)

    _write_json(resolved_output / "proxy_evidence_synthesis_manifest.json", manifest)
    for artifact_name, (rows, fieldnames) in artifact_rows.items():
        _write_csv(resolved_output / OUTPUT_ARTIFACTS[artifact_name], rows, fieldnames)
    _write_csv(
        resolved_output / "boundary_validation.csv",
        boundary_rows,
        ["check_id", "status", "detail"],
    )
    (resolved_output / "proxy_evidence_synthesis_report.md").write_text(report, encoding="utf-8")
    return {
        "manifest": manifest,
        "metric_rows": metric_rows,
        "sample_rows": sample_rows,
        "proof_rows": proof_rows,
        "gap_rows": gap_rows,
        "boundary_rows": boundary_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate T011 read-only proxy evidence synthesis artifacts.")
    parser.add_argument("--t010-dir", type=Path, default=DEFAULT_T010_DIR)
    parser.add_argument("--t010-qa-report", type=Path, default=DEFAULT_T010_QA_REPORT)
    parser.add_argument("--t010-business-report", type=Path, default=DEFAULT_T010_BUSINESS_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_synthesis_artifacts(
        t010_dir=args.t010_dir,
        t010_qa_report=args.t010_qa_report,
        t010_business_report=args.t010_business_report,
        output_dir=args.output_dir,
    )
    manifest = result["manifest"]
    print(
        "final_recommendation="
        f"{manifest['final_recommendation']} metric_rows={manifest['metric_decision_row_count']} "
        f"sample_rows={manifest['sample_decision_row_count']} output_dir={manifest['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
