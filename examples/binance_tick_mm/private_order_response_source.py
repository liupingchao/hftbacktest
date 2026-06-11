#!/usr/bin/env python3
"""Local private-order response artifact skeleton and validator.

This module is deliberately local-only. It validates task-local CSV/JSON
artifacts against the accepted 0610T006 private-order response contract. It
does not connect to exchange endpoints, read credentials, run user streams,
collect private data, compute execution metrics, or authorize strategy use.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "0611T002"
SOURCE_TASK_ID = "0610T006"
SOURCE_FINAL_RECOMMENDATION = "private_order_response_contract_ready_for_qa"
SYNTHESIS_TASK_ID = "0611T001"
SYNTHESIS_FINAL_RECOMMENDATION = "source_line_synthesis_gate_ready_for_qa"
SOURCE_LINE_ID = "private_order_response_source_line"
SCHEMA_VERSION = "basis_positive_private_order_response_artifact_skeleton_v1"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/basis_positive_private_order_response_source_artifact_skeleton_0611T002")
FIXED_GENERATED_AT = "2026-06-11T00:00:00Z"

EXPECTED_ARTIFACT_SOURCE_CLASS = "accepted_private_order_response_artifact_contract"
EXPECTED_SOURCE_POLICY = "design_only_no_endpoint_no_collector_no_runner"

REQUIRED_FIELDS = [
    "task_id",
    "source_task_id",
    "source_line_id",
    "venue",
    "instrument",
    "artifact_source_class",
    "source_policy",
    "artifact_event_id",
    "event_sequence_index",
    "local_receive_time",
    "artifact_generated_time",
    "validation_or_reconciliation_time",
    "response_category",
    "lifecycle_state_label",
    "post_only_reject_class",
    "terminal_state_marker",
    "unknown_missing_conflicting_flag",
    "validation_gate_id",
    "validation_status",
    "overclaim_rejection_id",
    "allowed_future_use",
    "forbidden_current_interpretation",
]
CONDITIONAL_FIELDS = [
    "client_order_ref_hash",
    "exchange_order_ref_hash",
    "exchange_event_time",
    "fail_closed_reason",
]
ALL_FIELDS = REQUIRED_FIELDS + CONDITIONAL_FIELDS

ALLOWED_RESPONSE_CATEGORIES = {
    "accepted",
    "rejected",
    "filled",
    "partially_filled",
    "canceled",
    "expired",
    "terminal",
    "unknown",
    "missing",
    "conflicting",
}
ALLOWED_LIFECYCLE_STATES = {
    "new_or_submitted_design_label",
    "accepted_ack_design_label",
    "rejected_terminal_design_label",
    "filled_terminal_design_label",
    "partially_filled_active_design_label",
    "partially_filled_terminal_design_label",
    "canceled_terminal_design_label",
    "expired_terminal_design_label",
    "unknown_state_fail_closed",
    "conflicting_state_fail_closed",
}
ALLOWED_POST_ONLY_CLASSES = {
    "post_only_reject",
    "non_post_only_reject",
    "unknown",
    "missing",
    "conflicting",
    "unsupported",
}
ALLOWED_TERMINAL_MARKERS = {"terminal", "non_terminal", "unknown", "conflicting"}
ALLOWED_VALIDATION_STATUSES = {
    "accepted_design_label",
    "fail_closed_design_label",
    "unsupported_design_label",
}
TERMINAL_RESPONSE_CATEGORIES = {"rejected", "filled", "canceled", "expired", "terminal"}
TERMINAL_LIFECYCLE_STATES = {
    "rejected_terminal_design_label",
    "filled_terminal_design_label",
    "partially_filled_terminal_design_label",
    "canceled_terminal_design_label",
    "expired_terminal_design_label",
}
FAIL_CLOSED_REASON_FIELDS = {
    "missing_required_field",
    "unknown_enum_value",
    "unsupported_evidence_source",
    "missing_timestamp",
    "timestamp_order_invalid",
    "duplicate_event_identity",
    "terminal_state_conflict",
    "terminal_consistency_invalid",
    "incomplete_lifecycle_evidence",
    "accepted_status_with_fail_closed_flag",
}


@dataclass(frozen=True)
class ValidationIssue:
    artifact_event_id: str
    row_index: int
    reason_code: str
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    artifact_path: Path
    rows: list[dict[str, str]]
    issues: list[ValidationIssue]

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail_closed"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _row_id(row: dict[str, str], row_index: int) -> str:
    return row.get("artifact_event_id") or f"row_{row_index}"


def _issue(row: dict[str, str], row_index: int, reason_code: str, detail: str) -> ValidationIssue:
    return ValidationIssue(
        artifact_event_id=_row_id(row, row_index),
        row_index=row_index,
        reason_code=reason_code,
        detail=detail,
    )


def _check_required_fields(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for field in REQUIRED_FIELDS:
        if not str(row.get(field, "")).strip():
            issues.append(_issue(row, row_index, "missing_required_field", field))
    return issues


def _check_enums(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    checks = [
        ("source_line_id", {SOURCE_LINE_ID}),
        ("artifact_source_class", {EXPECTED_ARTIFACT_SOURCE_CLASS}),
        ("source_policy", {EXPECTED_SOURCE_POLICY}),
        ("response_category", ALLOWED_RESPONSE_CATEGORIES),
        ("lifecycle_state_label", ALLOWED_LIFECYCLE_STATES),
        ("post_only_reject_class", ALLOWED_POST_ONLY_CLASSES),
        ("terminal_state_marker", ALLOWED_TERMINAL_MARKERS),
        ("validation_status", ALLOWED_VALIDATION_STATUSES),
    ]
    issues: list[ValidationIssue] = []
    for field, allowed in checks:
        value = str(row.get(field, "")).strip()
        if value and value not in allowed:
            reason = "unsupported_evidence_source" if field in {"artifact_source_class", "source_policy"} else "unknown_enum_value"
            issues.append(_issue(row, row_index, reason, f"{field}={value}"))
    return issues


def _check_timestamps(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    fields = [
        "exchange_event_time",
        "local_receive_time",
        "artifact_generated_time",
        "validation_or_reconciliation_time",
    ]
    parsed: list[datetime] = []
    issues: list[ValidationIssue] = []
    for field in fields:
        value = row.get(field, "")
        timestamp = _parse_time(value)
        if timestamp is None:
            issues.append(_issue(row, row_index, "missing_timestamp", field))
        else:
            parsed.append(timestamp)
    if len(parsed) == len(fields) and any(parsed[i] > parsed[i + 1] for i in range(len(parsed) - 1)):
        issues.append(
            _issue(
                row,
                row_index,
                "timestamp_order_invalid",
                "exchange_event_time<=local_receive_time<=artifact_generated_time<=validation_or_reconciliation_time",
            )
        )
    return issues


def _check_terminal_consistency(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    response_category = row.get("response_category", "")
    lifecycle_state = row.get("lifecycle_state_label", "")
    terminal_marker = row.get("terminal_state_marker", "")
    validation_status = row.get("validation_status", "")
    fail_flag = _truthy(row.get("unknown_missing_conflicting_flag", ""))
    issues: list[ValidationIssue] = []

    if response_category in TERMINAL_RESPONSE_CATEGORIES and terminal_marker != "terminal":
        issues.append(_issue(row, row_index, "terminal_consistency_invalid", "terminal response without terminal marker"))
    if terminal_marker == "terminal" and lifecycle_state not in TERMINAL_LIFECYCLE_STATES:
        issues.append(_issue(row, row_index, "terminal_consistency_invalid", "terminal marker incompatible with lifecycle state"))
    if terminal_marker in {"unknown", "conflicting"} and validation_status == "accepted_design_label":
        issues.append(_issue(row, row_index, "accepted_status_with_fail_closed_flag", "accepted status with non-positive terminal marker"))
    if fail_flag and validation_status == "accepted_design_label":
        issues.append(_issue(row, row_index, "accepted_status_with_fail_closed_flag", "accepted status with fail-closed flag"))
    if terminal_marker == "terminal" and not row.get("client_order_ref_hash") and not row.get("exchange_order_ref_hash"):
        issues.append(_issue(row, row_index, "incomplete_lifecycle_evidence", "terminal row missing opaque order reference"))
    return issues


def _check_cross_row(rows: list[dict[str, str]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    event_ids = Counter(row.get("artifact_event_id", "") for row in rows)
    for index, row in enumerate(rows, start=1):
        event_id = row.get("artifact_event_id", "")
        if event_id and event_ids[event_id] > 1:
            issues.append(_issue(row, index, "duplicate_event_identity", event_id))

    terminal_by_order: dict[str, set[str]] = defaultdict(set)
    rows_by_order: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        order_ref = row.get("exchange_order_ref_hash") or row.get("client_order_ref_hash")
        if not order_ref:
            continue
        rows_by_order[order_ref].append((index, row))
        if row.get("terminal_state_marker") == "terminal":
            terminal_by_order[order_ref].add(row.get("response_category", ""))

    for order_ref, terminal_categories in terminal_by_order.items():
        positive_terminals = terminal_categories & TERMINAL_RESPONSE_CATEGORIES
        if len(positive_terminals) > 1:
            for index, row in rows_by_order[order_ref]:
                if row.get("terminal_state_marker") == "terminal":
                    issues.append(
                        _issue(
                            row,
                            index,
                            "terminal_state_conflict",
                            f"{order_ref}:{'|'.join(sorted(positive_terminals))}",
                        )
                    )
    return issues


def validate_rows(rows: list[dict[str, str]], artifact_path: Path | None = None) -> ValidationResult:
    issues: list[ValidationIssue] = []
    for index, row in enumerate(rows, start=1):
        issues.extend(_check_required_fields(row, index))
        issues.extend(_check_enums(row, index))
        issues.extend(_check_timestamps(row, index))
        issues.extend(_check_terminal_consistency(row, index))
    issues.extend(_check_cross_row(rows))
    return ValidationResult(artifact_path or Path("<memory>"), rows, issues)


def validate_artifact(path: Path) -> ValidationResult:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            rows = [{str(key): str(value) for key, value in row.items()} for row in payload["rows"]]
        elif isinstance(payload, list):
            rows = [{str(key): str(value) for key, value in row.items()} for row in payload]
        else:
            rows = []
    else:
        rows = _read_csv(path)
    return validate_rows(rows, artifact_path=path)


def _base_row(case_id: str, event_id: str, seq: int, order_ref: str) -> dict[str, str]:
    base_time = f"2026-06-11T00:00:{seq:02d}Z"
    return {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "source_line_id": SOURCE_LINE_ID,
        "venue": "synthetic_binance_context",
        "instrument": "BTCUSDT",
        "artifact_source_class": EXPECTED_ARTIFACT_SOURCE_CLASS,
        "source_policy": EXPECTED_SOURCE_POLICY,
        "artifact_event_id": event_id,
        "client_order_ref_hash": order_ref,
        "exchange_order_ref_hash": f"exchange_{order_ref}",
        "event_sequence_index": str(seq),
        "exchange_event_time": base_time,
        "local_receive_time": f"2026-06-11T00:00:{seq + 1:02d}Z",
        "artifact_generated_time": f"2026-06-11T00:00:{seq + 2:02d}Z",
        "validation_or_reconciliation_time": f"2026-06-11T00:00:{seq + 3:02d}Z",
        "response_category": "accepted",
        "lifecycle_state_label": "accepted_ack_design_label",
        "post_only_reject_class": "non_post_only_reject",
        "terminal_state_marker": "non_terminal",
        "unknown_missing_conflicting_flag": "0",
        "validation_gate_id": "local_artifact_skeleton_gate",
        "validation_status": "accepted_design_label",
        "fail_closed_reason": "",
        "overclaim_rejection_id": "reject_real_order_lifecycle_proof",
        "allowed_future_use": "later_separately_scoped_metric_design_only",
        "forbidden_current_interpretation": "endpoint_or_metric_proof",
        "fixture_case_id": case_id,
    }


def synthetic_fixture_cases() -> dict[str, list[dict[str, str]]]:
    valid_accepted = [
        _base_row("valid_accepted_lifecycle", "evt_valid_accept_1", 1, "client_valid_accept"),
        {
            **_base_row("valid_accepted_lifecycle", "evt_valid_accept_2", 4, "client_valid_accept"),
            "response_category": "filled",
            "lifecycle_state_label": "filled_terminal_design_label",
            "terminal_state_marker": "terminal",
            "overclaim_rejection_id": "reject_fill_probability_proof",
        },
    ]

    valid_rejected = [
        {
            **_base_row("valid_post_only_reject", "evt_valid_reject_1", 8, "client_valid_reject"),
            "response_category": "rejected",
            "lifecycle_state_label": "rejected_terminal_design_label",
            "post_only_reject_class": "post_only_reject",
            "terminal_state_marker": "terminal",
            "overclaim_rejection_id": "reject_post_only_behavior_proof",
        }
    ]

    missing_required = [{**_base_row("missing_required_field", "evt_missing_required", 12, "client_missing"), "venue": ""}]
    unknown_enum = [{**_base_row("unknown_enum_value", "evt_unknown_enum", 16, "client_unknown"), "response_category": "accepted_live"}]
    conflicting_terminal = [
        {
            **_base_row("conflicting_terminal_state", "evt_conflict_1", 20, "client_conflict"),
            "response_category": "filled",
            "lifecycle_state_label": "filled_terminal_design_label",
            "terminal_state_marker": "terminal",
        },
        {
            **_base_row("conflicting_terminal_state", "evt_conflict_2", 24, "client_conflict"),
            "response_category": "canceled",
            "lifecycle_state_label": "canceled_terminal_design_label",
            "terminal_state_marker": "terminal",
        },
    ]
    bad_timestamp = [
        {
            **_base_row("bad_timestamp", "evt_bad_timestamp", 28, "client_bad_time"),
            "local_receive_time": "",
        }
    ]
    duplicate_event = [
        _base_row("duplicate_event_identity", "evt_duplicate", 32, "client_duplicate_a"),
        _base_row("duplicate_event_identity", "evt_duplicate", 36, "client_duplicate_b"),
    ]
    unsupported_source = [
        {
            **_base_row("unsupported_evidence_source", "evt_unsupported", 40, "client_unsupported"),
            "artifact_source_class": "signed_exchange_endpoint_response",
        }
    ]
    incomplete_lifecycle = [
        {
            **_base_row("incomplete_lifecycle_evidence", "evt_incomplete", 44, "client_incomplete"),
            "response_category": "filled",
            "lifecycle_state_label": "accepted_ack_design_label",
            "terminal_state_marker": "terminal",
        }
    ]

    return {
        "valid_accepted_lifecycle": valid_accepted,
        "valid_post_only_reject": valid_rejected,
        "missing_required_field": missing_required,
        "unknown_enum_value": unknown_enum,
        "conflicting_terminal_state": conflicting_terminal,
        "bad_timestamp": bad_timestamp,
        "duplicate_event_identity": duplicate_event,
        "unsupported_evidence_source": unsupported_source,
        "incomplete_lifecycle_evidence": incomplete_lifecycle,
    }


def schema_rows() -> list[dict[str, str]]:
    rows = []
    for field in REQUIRED_FIELDS:
        rows.append(
            {
                "field_name": field,
                "required": "yes",
                "field_type": "string",
                "source_contract": SOURCE_TASK_ID,
                "current_use": "local_validation_only",
                "forbidden_current_interpretation": "endpoint_or_metric_authorization",
            }
        )
    for field in CONDITIONAL_FIELDS:
        rows.append(
            {
                "field_name": field,
                "required": "conditional",
                "field_type": "string",
                "source_contract": SOURCE_TASK_ID,
                "current_use": "local_validation_only",
                "forbidden_current_interpretation": "endpoint_or_metric_authorization",
            }
        )
    return rows


def _issue_rows(result: ValidationResult, case_id: str) -> list[dict[str, Any]]:
    if result.passed:
        return [
            {
                "fixture_case_id": case_id,
                "artifact_event_id": "",
                "row_index": "",
                "reason_code": "",
                "detail": "pass",
            }
        ]
    return [
        {
            "fixture_case_id": case_id,
            "artifact_event_id": issue.artifact_event_id,
            "row_index": issue.row_index,
            "reason_code": issue.reason_code,
            "detail": issue.detail,
        }
        for issue in result.issues
    ]


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = output_dir / "fixtures"
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    cases = synthetic_fixture_cases()

    for case_id, rows in cases.items():
        fixture_path = fixture_dir / f"{case_id}.csv"
        _write_csv(fixture_path, rows, ALL_FIELDS + ["fixture_case_id"])
        result = validate_artifact(fixture_path)
        expected_status = "pass" if case_id in {"valid_accepted_lifecycle", "valid_post_only_reject"} else "fail_closed"
        reasons = sorted({issue.reason_code for issue in result.issues})
        summary_rows.append(
            {
                "fixture_case_id": case_id,
                "fixture_path": str(fixture_path),
                "row_count": len(rows),
                "expected_status": expected_status,
                "actual_status": result.status,
                "issue_count": len(result.issues),
                "reason_codes": "|".join(reasons),
            }
        )
        case_rows.append(
            {
                "fixture_case_id": case_id,
                "purpose": case_id,
                "expected_status": expected_status,
                "covered_validation_area": "|".join(reasons) if reasons else "valid_local_artifact",
                "forbidden_current_interpretation": "metric_or_endpoint_proof",
            }
        )
        detail_rows.extend(_issue_rows(result, case_id))

    _write_csv(
        output_dir / "schema_columns.csv",
        schema_rows(),
        [
            "field_name",
            "required",
            "field_type",
            "source_contract",
            "current_use",
            "forbidden_current_interpretation",
        ],
    )
    _write_csv(
        output_dir / "fixture_case_catalog.csv",
        case_rows,
        [
            "fixture_case_id",
            "purpose",
            "expected_status",
            "covered_validation_area",
            "forbidden_current_interpretation",
        ],
    )
    _write_csv(
        output_dir / "validator_result_summary.csv",
        summary_rows,
        [
            "fixture_case_id",
            "fixture_path",
            "row_count",
            "expected_status",
            "actual_status",
            "issue_count",
            "reason_codes",
        ],
    )
    _write_csv(
        output_dir / "validator_result_details.csv",
        detail_rows,
        ["fixture_case_id", "artifact_event_id", "row_index", "reason_code", "detail"],
    )

    all_expected = all(row["expected_status"] == row["actual_status"] for row in summary_rows)
    boundary_rows = boundary_validation_rows(all_expected)
    _write_csv(output_dir / "boundary_validation.csv", boundary_rows, ["check_id", "status", "detail"])
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_task_id": SOURCE_TASK_ID,
        "source_final_recommendation": SOURCE_FINAL_RECOMMENDATION,
        "synthesis_task_id": SYNTHESIS_TASK_ID,
        "synthesis_final_recommendation": SYNTHESIS_FINAL_RECOMMENDATION,
        "source_line_id": SOURCE_LINE_ID,
        "generated_at": FIXED_GENERATED_AT,
        "final_recommendation": "private_order_response_artifact_skeleton_ready_for_qa" if all_expected else "private_order_response_artifact_skeleton_needs_revision",
        "fixture_case_count": len(summary_rows),
        "fixture_pass_count": sum(1 for row in summary_rows if row["actual_status"] == "pass"),
        "fixture_fail_closed_count": sum(1 for row in summary_rows if row["actual_status"] == "fail_closed"),
        "all_expected_statuses_matched": all_expected,
        "boundary_flags": {
            "local_artifact_skeleton_only": True,
            "no_endpoint_implementation": True,
            "no_credentials_signing_nonce_user_stream": True,
            "no_private_order_account_live_data_read": True,
            "no_remote_execution_or_collection": True,
            "no_runner_consumption": True,
            "no_real_execution_metrics": True,
            "no_real_economics_metrics": True,
            "no_pnl_proof": True,
            "no_strategy_live_default_tiny": True,
            "no_case_library_or_shadow_decisions": True,
            "no_parameter_search": True,
            "no_deployment_or_promotion": True,
            "execution_layer_maker_viability_unproven": True,
        },
        "output_artifacts": {
            "schema_columns": str(output_dir / "schema_columns.csv"),
            "fixture_case_catalog": str(output_dir / "fixture_case_catalog.csv"),
            "validator_result_summary": str(output_dir / "validator_result_summary.csv"),
            "validator_result_details": str(output_dir / "validator_result_details.csv"),
            "boundary_validation": str(output_dir / "boundary_validation.csv"),
        },
    }
    _write_json(output_dir / "private_order_response_skeleton_manifest.json", manifest)
    return manifest


def boundary_validation_rows(validation_cases_matched: bool) -> list[dict[str, str]]:
    checks = [
        ("prerequisite_0611T001_qa_passed", "0611T001 QA is passed and synthesis gate ready"),
        ("prerequisite_0610T006_qa_passed", "0610T006 QA is passed and private order response contract ready"),
        ("local_artifact_skeleton_only", "local fixtures parser validator and artifacts only"),
        ("no_endpoint_implementation", "no endpoint URL REST websocket exchange client or connector code"),
        ("no_credentials_signing_nonce_user_stream", "no credentials signing nonce handling or user stream"),
        ("no_private_order_account_live_data_read", "only synthetic task-local fixtures are read"),
        ("no_remote_execution_or_collection", "no remote execution or collection"),
        ("no_runner_consumption", "no execution runner consumption implemented"),
        ("no_real_execution_metrics", "no fill probability post-only real lifecycle or other execution metrics"),
        ("no_real_economics_metrics_or_pnl", "no economics metrics or PnL proof"),
        ("no_strategy_live_default_tiny", "no strategy live default-on or tiny-live behavior"),
        ("no_case_library_shadow_parameter", "no case-library shadow decisions or parameter search"),
        ("no_deployment_promotion_viability", "no deployment promotion or execution-layer maker viability proof"),
        ("future_use_requires_separate_task_and_qa", "metrics runner endpoint collector or strategy use requires later task and QA"),
        ("validation_cases_fail_closed", "synthetic invalid cases fail closed" if validation_cases_matched else "validation case mismatch"),
    ]
    return [
        {
            "check_id": check_id,
            "status": "pass" if check_id != "validation_cases_fail_closed" or validation_cases_matched else "fail",
            "detail": detail,
        }
        for check_id, detail in checks
    ]


def validate_command(args: argparse.Namespace) -> int:
    result = validate_artifact(Path(args.input))
    summary = {
        "artifact_path": str(result.artifact_path),
        "row_count": len(result.rows),
        "status": result.status,
        "issue_count": len(result.issues),
        "reason_codes": sorted({issue.reason_code for issue in result.issues}),
    }
    if args.summary_out:
        _write_json(Path(args.summary_out), summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if result.passed else 2


def generate_artifacts_command(args: argparse.Namespace) -> int:
    manifest = generate_artifacts(Path(args.output_dir))
    print(json.dumps({"output_dir": args.output_dir, "final_recommendation": manifest["final_recommendation"]}, sort_keys=True))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate local private-order response artifacts without endpoints, credentials, user streams, or metrics."
    )
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate", help="validate one local CSV/JSON artifact")
    validate.add_argument("--input", required=True, help="local CSV or JSON artifact path")
    validate.add_argument("--summary-out", help="optional JSON summary path")
    validate.set_defaults(func=validate_command)

    generate = subparsers.add_parser("generate-artifacts", help="write 0611T002 synthetic fixtures and validation artifacts")
    generate.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="task-local output directory")
    generate.set_defaults(func=generate_artifacts_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
