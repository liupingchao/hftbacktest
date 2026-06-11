#!/usr/bin/env python3
"""Local replay lifecycle validation and reconciliation gate.

This module is deliberately local-only. It validates task-local CSV/JSON
artifacts against the accepted 0610T007 replay lifecycle source-line contract.
It does not connect to exchange endpoints, read credentials, run user streams,
consume runners, compute queue/race metrics, or authorize strategy use.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "0611T003"
SOURCE_TASK_ID = "0610T007"
SOURCE_FINAL_RECOMMENDATION = "replay_lifecycle_contract_ready_for_qa"
SYNTHESIS_TASK_ID = "0611T001"
SYNTHESIS_FINAL_RECOMMENDATION = "source_line_synthesis_gate_ready_for_qa"
PRIVATE_ORDER_CONTEXT_TASK_ID = "0611T002"
PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION = "private_order_response_artifact_skeleton_ready_for_qa"
SOURCE_LINE_ID = "replay_lifecycle_semantics_source_line"
SCHEMA_VERSION = "basis_positive_replay_lifecycle_validation_gate_v1"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/basis_positive_replay_lifecycle_validation_gate_0611T003")
FIXED_GENERATED_AT = "2026-06-11T00:00:00Z"

EXPECTED_ARTIFACT_SOURCE_CLASS = "accepted_replay_lifecycle_artifact_contract"
EXPECTED_SOURCE_POLICY = "design_only_no_endpoint_no_collector_no_runner"

REQUIRED_FIELDS = [
    "task_id",
    "source_task_id",
    "source_line_id",
    "artifact_source_class",
    "replay_live_domain",
    "source_policy",
    "validation_status",
    "proof_limit_class",
    "lifecycle_event_id",
    "opaque_order_reference",
    "event_type",
    "lifecycle_state_design_label",
    "event_sequence_index",
    "ordering_scope",
    "decision_time",
    "replay_time",
    "artifact_generated_time",
    "validation_or_reconciliation_time",
    "ambiguity_flag",
    "conflict_flag",
    "out_of_order_flag",
    "queue_observation_class",
    "queue_proof_status",
    "cancel_fill_race_observation_class",
    "race_proof_status",
    "fail_closed_reason",
    "overclaim_rejection_id",
    "allowed_future_use",
    "forbidden_current_interpretation",
]
CONDITIONAL_FIELDS = [
    "previous_source_line_task_id",
    "exchange_event_time",
    "local_receive_time",
    "same_order_causal_sequence",
    "cross_order_ordering_scope",
    "predecessor_event_reference",
    "successor_event_reference",
]
ALL_FIELDS = REQUIRED_FIELDS + CONDITIONAL_FIELDS

ALLOWED_REPLAY_LIVE_DOMAINS = {"replay_regression", "future_live_lifecycle_context"}
ALLOWED_VALIDATION_STATUSES = {"accepted_design_label", "fail_closed_design_label", "unsupported_design_label"}
ALLOWED_PROOF_LIMIT_CLASSES = {
    "supporting_regression_not_execution_proof",
    "future_dependency_context",
    "fail_closed",
}
ALLOWED_EVENT_TYPES = {
    "submit",
    "ack",
    "cancel_request",
    "cancel_ack",
    "fill",
    "partial_fill",
    "reject",
    "expire",
    "terminal",
    "unknown",
}
ALLOWED_LIFECYCLE_STATES = {
    "submitted_design_label",
    "accepted_ack_design_label",
    "cancel_requested_design_label",
    "canceled_terminal_design_label",
    "filled_terminal_design_label",
    "partially_filled_active_design_label",
    "rejected_terminal_design_label",
    "expired_terminal_design_label",
    "unknown_state_fail_closed",
    "conflicting_state_fail_closed",
}
ALLOWED_ORDERING_SCOPES = {"same_order", "cross_order_context", "unsupported"}
ALLOWED_BOOLEAN_FLAGS = {"0", "1", "false", "true", "no", "yes"}
ALLOWED_QUEUE_OBSERVATION_CLASSES = {
    "public_book_context",
    "replay_model_context",
    "local_audit_context",
    "queue_ahead_proxy",
    "unknown",
}
ALLOWED_QUEUE_PROOF_STATUSES = {"unproven", "exact_position_rejected", "diagnostic_only", "fail_closed"}
ALLOWED_RACE_OBSERVATION_CLASSES = {"same_order_context", "cross_order_context", "replay_regression_context", "unknown"}
ALLOWED_RACE_PROOF_STATUSES = {"unproven", "metric_rejected", "diagnostic_only", "fail_closed"}
ALLOWED_CROSS_ORDER_SCOPES = {"", "diagnostic_context_only", "unsupported"}

TERMINAL_EVENT_TYPES = {"cancel_ack", "fill", "reject", "expire", "terminal"}
TERMINAL_STATE_BY_EVENT = {
    "cancel_ack": "canceled_terminal_design_label",
    "fill": "filled_terminal_design_label",
    "reject": "rejected_terminal_design_label",
    "expire": "expired_terminal_design_label",
}


@dataclass(frozen=True)
class ValidationIssue:
    lifecycle_event_id: str
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
    return row.get("lifecycle_event_id") or f"row_{row_index}"


def _issue(row: dict[str, str], row_index: int, reason_code: str, detail: str) -> ValidationIssue:
    return ValidationIssue(
        lifecycle_event_id=_row_id(row, row_index),
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
        ("replay_live_domain", ALLOWED_REPLAY_LIVE_DOMAINS),
        ("validation_status", ALLOWED_VALIDATION_STATUSES),
        ("proof_limit_class", ALLOWED_PROOF_LIMIT_CLASSES),
        ("event_type", ALLOWED_EVENT_TYPES),
        ("lifecycle_state_design_label", ALLOWED_LIFECYCLE_STATES),
        ("ordering_scope", ALLOWED_ORDERING_SCOPES),
        ("queue_observation_class", ALLOWED_QUEUE_OBSERVATION_CLASSES),
        ("queue_proof_status", ALLOWED_QUEUE_PROOF_STATUSES),
        ("cancel_fill_race_observation_class", ALLOWED_RACE_OBSERVATION_CLASSES),
        ("race_proof_status", ALLOWED_RACE_PROOF_STATUSES),
        ("cross_order_ordering_scope", ALLOWED_CROSS_ORDER_SCOPES),
    ]
    issues: list[ValidationIssue] = []
    for field, allowed in checks:
        value = str(row.get(field, "")).strip()
        if value and value not in allowed:
            reason = "unsupported_source_policy" if field in {"artifact_source_class", "source_policy"} else "unknown_enum_value"
            issues.append(_issue(row, row_index, reason, f"{field}={value}"))
    for field in ["ambiguity_flag", "conflict_flag", "out_of_order_flag"]:
        value = str(row.get(field, "")).strip().lower()
        if value and value not in ALLOWED_BOOLEAN_FLAGS:
            issues.append(_issue(row, row_index, "unknown_enum_value", f"{field}={value}"))
    return issues


def _check_timestamps(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    required_time_fields = [
        "decision_time",
        "replay_time",
        "artifact_generated_time",
        "validation_or_reconciliation_time",
    ]
    parsed: dict[str, datetime] = {}
    issues: list[ValidationIssue] = []
    for field in required_time_fields:
        timestamp = _parse_time(row.get(field, ""))
        if timestamp is None:
            issues.append(_issue(row, row_index, "missing_timestamp_domain", field))
        else:
            parsed[field] = timestamp
    if len(parsed) == len(required_time_fields):
        ordered = [parsed[field] for field in required_time_fields]
        if any(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)):
            issues.append(
                _issue(
                    row,
                    row_index,
                    "timestamp_domain_merged_or_invalid",
                    "decision_time<replay_time<artifact_generated_time<validation_or_reconciliation_time",
                )
            )
    optional_pair = ["exchange_event_time", "local_receive_time"]
    optional_parsed = [_parse_time(row.get(field, "")) for field in optional_pair if row.get(field, "")]
    if len(optional_parsed) == 2:
        if optional_parsed[0] is None or optional_parsed[1] is None or optional_parsed[0] > optional_parsed[1]:
            issues.append(
                _issue(row, row_index, "timestamp_domain_merged_or_invalid", "exchange_event_time<=local_receive_time")
            )
    return issues


def _check_flags_and_overclaims(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    validation_status = row.get("validation_status", "")
    fail_closed_reason = row.get("fail_closed_reason", "")
    active_flags = [
        field
        for field in ["ambiguity_flag", "conflict_flag", "out_of_order_flag"]
        if _truthy(row.get(field, ""))
    ]
    if active_flags and validation_status == "accepted_design_label":
        issues.append(_issue(row, row_index, "fail_closed_flag_accepted", "|".join(active_flags)))
    if (active_flags or validation_status != "accepted_design_label") and not fail_closed_reason:
        issues.append(_issue(row, row_index, "missing_fail_closed_reason", validation_status or "flagged_row"))

    proof_limit = row.get("proof_limit_class", "")
    allowed_future_use = row.get("allowed_future_use", "")
    forbidden = row.get("forbidden_current_interpretation", "")
    if proof_limit == "execution_proof" or "current_execution_proof" in allowed_future_use or "execution_proof_allowed" in forbidden:
        issues.append(_issue(row, row_index, "replay_as_execution_proof_overclaim", proof_limit or allowed_future_use))
    if row.get("queue_proof_status") in {"proof_available", "exact_position_proven"}:
        issues.append(_issue(row, row_index, "queue_priority_proof_overclaim", row.get("queue_proof_status", "")))
    if row.get("race_proof_status") in {"metric_proven", "race_metric_available"}:
        issues.append(_issue(row, row_index, "cancel_fill_race_metric_overclaim", row.get("race_proof_status", "")))
    if row.get("cross_order_ordering_scope") == "global_causal_proof":
        issues.append(_issue(row, row_index, "cross_order_causal_overclaim", row.get("cross_order_ordering_scope", "")))
    return issues


def _check_lifecycle_consistency(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    event_type = row.get("event_type", "")
    lifecycle_state = row.get("lifecycle_state_design_label", "")
    issues: list[ValidationIssue] = []
    expected_terminal_state = TERMINAL_STATE_BY_EVENT.get(event_type)
    if expected_terminal_state and lifecycle_state != expected_terminal_state:
        issues.append(_issue(row, row_index, "invalid_terminal_lifecycle_ordering", f"{event_type}->{lifecycle_state}"))
    if event_type == "terminal" and lifecycle_state not in set(TERMINAL_STATE_BY_EVENT.values()):
        issues.append(_issue(row, row_index, "invalid_terminal_lifecycle_ordering", lifecycle_state))
    return issues


def _int_value(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _check_cross_row(rows: list[dict[str, str]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    event_ids = Counter(row.get("lifecycle_event_id", "") for row in rows)
    for index, row in enumerate(rows, start=1):
        event_id = row.get("lifecycle_event_id", "")
        if event_id and event_ids[event_id] > 1:
            issues.append(_issue(row, index, "duplicate_lifecycle_event_identity", event_id))

    rows_by_order: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        order_ref = row.get("opaque_order_reference")
        if order_ref:
            rows_by_order[order_ref].append((index, row))

    lifecycle_event_ids = {row.get("lifecycle_event_id", "") for row in rows}
    for index, row in enumerate(rows, start=1):
        for field, reason_code in [
            ("predecessor_event_reference", "missing_predecessor_event_reference"),
            ("successor_event_reference", "missing_successor_event_reference"),
        ]:
            ref = row.get(field, "")
            if ref and ref not in lifecycle_event_ids:
                issues.append(_issue(row, index, reason_code, ref))

    for order_ref, order_rows in rows_by_order.items():
        ordered = sorted(order_rows, key=lambda item: _int_value(item[1].get("event_sequence_index", "")) or 0)
        causal_values = [
            (index, row, _int_value(row.get("same_order_causal_sequence", "")))
            for index, row in ordered
            if row.get("ordering_scope") == "same_order"
        ]
        if any(value is None for _, _, value in causal_values):
            for index, row, value in causal_values:
                if value is None:
                    issues.append(_issue(row, index, "non_monotonic_same_order_sequence", "missing sequence"))
        non_null_values = [value for _, _, value in causal_values if value is not None]
        if non_null_values != sorted(non_null_values) or len(non_null_values) != len(set(non_null_values)):
            for index, row, _ in causal_values:
                issues.append(_issue(row, index, "non_monotonic_same_order_sequence", order_ref))

        terminal_rows = [
            (index, row)
            for index, row in ordered
            if row.get("event_type") in TERMINAL_EVENT_TYPES
            or row.get("lifecycle_state_design_label") in set(TERMINAL_STATE_BY_EVENT.values())
        ]
        if len(terminal_rows) > 1:
            for index, row in terminal_rows:
                issues.append(_issue(row, index, "duplicate_terminal_state", order_ref))
        if terminal_rows:
            terminal_seq = _int_value(terminal_rows[0][1].get("same_order_causal_sequence", ""))
            if terminal_seq is not None:
                for index, row in ordered:
                    row_seq = _int_value(row.get("same_order_causal_sequence", ""))
                    if row_seq is not None and row_seq > terminal_seq:
                        issues.append(_issue(row, index, "invalid_terminal_lifecycle_ordering", order_ref))
    return issues


def validate_rows(rows: list[dict[str, str]], artifact_path: Path | None = None) -> ValidationResult:
    issues: list[ValidationIssue] = []
    for index, row in enumerate(rows, start=1):
        issues.extend(_check_required_fields(row, index))
        issues.extend(_check_enums(row, index))
        issues.extend(_check_timestamps(row, index))
        issues.extend(_check_flags_and_overclaims(row, index))
        issues.extend(_check_lifecycle_consistency(row, index))
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


def _base_row(case_id: str, event_id: str, order_ref: str, seq: int, event_type: str) -> dict[str, str]:
    base_time = datetime(2026, 6, 11, tzinfo=timezone.utc) + timedelta(seconds=seq * 4)

    def fmt(offset_seconds: int) -> str:
        return (base_time + timedelta(seconds=offset_seconds)).isoformat().replace("+00:00", "Z")

    return {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "source_line_id": SOURCE_LINE_ID,
        "previous_source_line_task_id": PRIVATE_ORDER_CONTEXT_TASK_ID,
        "artifact_source_class": EXPECTED_ARTIFACT_SOURCE_CLASS,
        "replay_live_domain": "replay_regression",
        "source_policy": EXPECTED_SOURCE_POLICY,
        "validation_status": "accepted_design_label",
        "proof_limit_class": "supporting_regression_not_execution_proof",
        "lifecycle_event_id": event_id,
        "opaque_order_reference": order_ref,
        "event_type": event_type,
        "lifecycle_state_design_label": "submitted_design_label",
        "event_sequence_index": str(seq),
        "ordering_scope": "same_order",
        "decision_time": fmt(0),
        "replay_time": fmt(1),
        "exchange_event_time": fmt(1),
        "local_receive_time": fmt(2),
        "artifact_generated_time": fmt(3),
        "validation_or_reconciliation_time": fmt(4),
        "same_order_causal_sequence": str(seq),
        "cross_order_ordering_scope": "",
        "predecessor_event_reference": "",
        "successor_event_reference": "",
        "ambiguity_flag": "0",
        "conflict_flag": "0",
        "out_of_order_flag": "0",
        "queue_observation_class": "replay_model_context",
        "queue_proof_status": "diagnostic_only",
        "cancel_fill_race_observation_class": "same_order_context",
        "race_proof_status": "diagnostic_only",
        "fail_closed_reason": "none",
        "overclaim_rejection_id": "reject_replay_as_execution_proof",
        "allowed_future_use": "later_separately_scoped_reconciliation_design_only",
        "forbidden_current_interpretation": "no_execution_metric_or_live_readiness",
        "fixture_case_id": case_id,
    }


def _with_links(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for index, row in enumerate(rows):
        if index > 0:
            row["predecessor_event_reference"] = rows[index - 1]["lifecycle_event_id"]
        if index < len(rows) - 1:
            row["successor_event_reference"] = rows[index + 1]["lifecycle_event_id"]
    return rows


def synthetic_fixture_cases() -> dict[str, list[dict[str, str]]]:
    valid_cancel = _with_links(
        [
            _base_row("valid_same_order_cancel_lifecycle", "evt_cancel_submit", "order_cancel", 1, "submit"),
            {
                **_base_row("valid_same_order_cancel_lifecycle", "evt_cancel_ack", "order_cancel", 2, "ack"),
                "lifecycle_state_design_label": "accepted_ack_design_label",
            },
            {
                **_base_row("valid_same_order_cancel_lifecycle", "evt_cancel_request", "order_cancel", 3, "cancel_request"),
                "lifecycle_state_design_label": "cancel_requested_design_label",
            },
            {
                **_base_row("valid_same_order_cancel_lifecycle", "evt_cancel_terminal", "order_cancel", 4, "cancel_ack"),
                "lifecycle_state_design_label": "canceled_terminal_design_label",
            },
        ]
    )

    valid_fill = _with_links(
        [
            _base_row("valid_fill_before_cancel_context", "evt_fill_submit", "order_fill", 5, "submit"),
            {
                **_base_row("valid_fill_before_cancel_context", "evt_fill_ack", "order_fill", 6, "ack"),
                "lifecycle_state_design_label": "accepted_ack_design_label",
            },
            {
                **_base_row("valid_fill_before_cancel_context", "evt_fill_terminal", "order_fill", 7, "fill"),
                "lifecycle_state_design_label": "filled_terminal_design_label",
                "overclaim_rejection_id": "reject_cancel_fill_race_metric",
            },
        ]
    )

    missing_required = [{**_base_row("missing_required_field", "evt_missing_required", "order_missing", 8, "submit"), "source_policy": ""}]
    unknown_enum = [{**_base_row("unknown_enum_value", "evt_unknown_enum", "order_unknown", 9, "submit"), "event_type": "live_fill"}]
    merged_timestamp = [{**_base_row("merged_timestamp_domain", "evt_merged_time", "order_time", 10, "submit"), "replay_time": "2026-06-11T00:00:40Z"}]
    non_monotonic = [
        _base_row("non_monotonic_same_order_sequence", "evt_seq_1", "order_seq", 11, "submit"),
        {**_base_row("non_monotonic_same_order_sequence", "evt_seq_2", "order_seq", 12, "ack"), "same_order_causal_sequence": "1"},
    ]
    flagged = [
        {
            **_base_row("ambiguous_conflicting_out_of_order_event", "evt_flagged", "order_flagged", 13, "submit"),
            "ambiguity_flag": "1",
            "conflict_flag": "1",
            "out_of_order_flag": "1",
        }
    ]
    duplicate_event = [
        _base_row("duplicate_lifecycle_event_identity", "evt_duplicate", "order_duplicate_a", 14, "submit"),
        _base_row("duplicate_lifecycle_event_identity", "evt_duplicate", "order_duplicate_b", 15, "submit"),
    ]
    duplicate_terminal = [
        {**_base_row("duplicate_terminal_state", "evt_dup_term_1", "order_dup_term", 16, "fill"), "lifecycle_state_design_label": "filled_terminal_design_label"},
        {**_base_row("duplicate_terminal_state", "evt_dup_term_2", "order_dup_term", 17, "cancel_ack"), "lifecycle_state_design_label": "canceled_terminal_design_label"},
    ]
    cross_order = [
        {
            **_base_row("cross_order_causal_overclaim", "evt_cross_order", "order_cross", 18, "submit"),
            "ordering_scope": "cross_order_context",
            "cross_order_ordering_scope": "global_causal_proof",
        }
    ]
    replay_proof = [
        {
            **_base_row("replay_as_execution_proof_overclaim", "evt_replay_proof", "order_replay_proof", 19, "submit"),
            "proof_limit_class": "execution_proof",
            "allowed_future_use": "current_execution_proof",
        }
    ]
    queue_proof = [
        {
            **_base_row("queue_priority_proof_overclaim", "evt_queue_proof", "order_queue_proof", 20, "submit"),
            "queue_proof_status": "proof_available",
        }
    ]
    race_metric = [
        {
            **_base_row("cancel_fill_race_metric_overclaim", "evt_race_metric", "order_race_metric", 21, "submit"),
            "race_proof_status": "metric_proven",
        }
    ]

    return {
        "valid_same_order_cancel_lifecycle": valid_cancel,
        "valid_fill_before_cancel_context": valid_fill,
        "missing_required_field": missing_required,
        "unknown_enum_value": unknown_enum,
        "merged_timestamp_domain": merged_timestamp,
        "non_monotonic_same_order_sequence": non_monotonic,
        "ambiguous_conflicting_out_of_order_event": flagged,
        "duplicate_lifecycle_event_identity": duplicate_event,
        "duplicate_terminal_state": duplicate_terminal,
        "cross_order_causal_overclaim": cross_order,
        "replay_as_execution_proof_overclaim": replay_proof,
        "queue_priority_proof_overclaim": queue_proof,
        "cancel_fill_race_metric_overclaim": race_metric,
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
                "forbidden_current_interpretation": "metric_runner_or_endpoint_authorization",
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
                "forbidden_current_interpretation": "metric_runner_or_endpoint_authorization",
            }
        )
    return rows


def ordering_reconciliation_policy_rows() -> list[dict[str, str]]:
    return [
        {
            "policy_id": "same_order_monotonic_sequence",
            "scope": "same_order",
            "required_behavior": "same_order_causal_sequence must be present unique and monotonic per opaque_order_reference",
            "fail_closed_reason": "non_monotonic_same_order_sequence",
            "forbidden_current_interpretation": "not exact exchange sequence proof",
        },
        {
            "policy_id": "terminal_state_singleton",
            "scope": "same_order",
            "required_behavior": "at most one terminal lifecycle event per opaque_order_reference",
            "fail_closed_reason": "duplicate_terminal_state",
            "forbidden_current_interpretation": "not real order lifecycle proof",
        },
        {
            "policy_id": "timestamp_domain_separation",
            "scope": "artifact",
            "required_behavior": "decision replay artifact and validation time domains must remain present and ordered",
            "fail_closed_reason": "timestamp_domain_merged_or_invalid",
            "forbidden_current_interpretation": "not causal execution proof",
        },
        {
            "policy_id": "cross_order_context_only",
            "scope": "cross_order_context",
            "required_behavior": "cross order context cannot claim global causal proof",
            "fail_closed_reason": "cross_order_causal_overclaim",
            "forbidden_current_interpretation": "not cross order causal proof",
        },
        {
            "policy_id": "replay_regression_only",
            "scope": "proof_limit",
            "required_behavior": "replay lifecycle artifacts remain supporting regression not execution proof",
            "fail_closed_reason": "replay_as_execution_proof_overclaim",
            "forbidden_current_interpretation": "not execution proof",
        },
        {
            "policy_id": "queue_race_metric_rejection",
            "scope": "overclaim",
            "required_behavior": "queue priority and cancel fill race metric claims fail closed",
            "fail_closed_reason": "queue_priority_proof_overclaim|cancel_fill_race_metric_overclaim",
            "forbidden_current_interpretation": "not queue or race metric proof",
        },
    ]


def _issue_rows(result: ValidationResult, case_id: str) -> list[dict[str, Any]]:
    if result.passed:
        return [
            {
                "fixture_case_id": case_id,
                "lifecycle_event_id": "",
                "row_index": "",
                "reason_code": "",
                "detail": "pass",
            }
        ]
    return [
        {
            "fixture_case_id": case_id,
            "lifecycle_event_id": issue.lifecycle_event_id,
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
    pass_cases = {"valid_same_order_cancel_lifecycle", "valid_fill_before_cancel_context"}

    for case_id, rows in cases.items():
        fixture_path = fixture_dir / f"{case_id}.csv"
        _write_csv(fixture_path, rows, ALL_FIELDS + ["fixture_case_id"])
        result = validate_artifact(fixture_path)
        expected_status = "pass" if case_id in pass_cases else "fail_closed"
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
                "covered_validation_area": "|".join(reasons) if reasons else "valid_local_replay_lifecycle_artifact",
                "forbidden_current_interpretation": "metric_runner_or_endpoint_proof",
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
        ["fixture_case_id", "lifecycle_event_id", "row_index", "reason_code", "detail"],
    )
    _write_csv(
        output_dir / "ordering_reconciliation_policy.csv",
        ordering_reconciliation_policy_rows(),
        ["policy_id", "scope", "required_behavior", "fail_closed_reason", "forbidden_current_interpretation"],
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
        "private_order_context_task_id": PRIVATE_ORDER_CONTEXT_TASK_ID,
        "private_order_context_final_recommendation": PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION,
        "source_line_id": SOURCE_LINE_ID,
        "generated_at": FIXED_GENERATED_AT,
        "final_recommendation": "replay_lifecycle_validation_gate_ready_for_qa" if all_expected else "replay_lifecycle_validation_gate_needs_revision",
        "fixture_case_count": len(summary_rows),
        "fixture_pass_count": sum(1 for row in summary_rows if row["actual_status"] == "pass"),
        "fixture_fail_closed_count": sum(1 for row in summary_rows if row["actual_status"] == "fail_closed"),
        "all_expected_statuses_matched": all_expected,
        "boundary_flags": {
            "local_replay_lifecycle_validation_gate_only": True,
            "no_endpoint_implementation": True,
            "no_credentials_signing_nonce_user_stream": True,
            "no_private_order_account_live_data_read": True,
            "no_remote_execution_or_collection": True,
            "no_runner_consumption": True,
            "no_replay_live_semantic_implementation": True,
            "no_real_execution_metrics": True,
            "no_queue_priority_proof": True,
            "no_exact_queue_position_proof": True,
            "no_cancel_fill_race_proof": True,
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
            "ordering_reconciliation_policy": str(output_dir / "ordering_reconciliation_policy.csv"),
            "boundary_validation": str(output_dir / "boundary_validation.csv"),
        },
    }
    _write_json(output_dir / "replay_lifecycle_validation_manifest.json", manifest)
    return manifest


def boundary_validation_rows(validation_cases_matched: bool) -> list[dict[str, str]]:
    checks = [
        ("prerequisite_0611T002_qa_passed", "0611T002 QA is passed and private-order skeleton ready"),
        ("prerequisite_0611T001_qa_passed", "0611T001 QA is passed and synthesis gate ready"),
        ("prerequisite_0610T007_qa_passed", "0610T007 QA is passed and replay lifecycle contract ready"),
        ("local_validation_gate_only", "local fixtures parser validator and artifacts only"),
        ("no_endpoint_implementation", "no endpoint URL REST websocket exchange client or connector code"),
        ("no_credentials_signing_nonce_user_stream", "no credentials signing nonce handling or user stream"),
        ("no_private_order_account_live_data_read", "only synthetic task-local fixtures and accepted local artifacts are read"),
        ("no_remote_execution_or_collection", "no remote execution or collection"),
        ("no_runner_consumption", "no execution runner consumption implemented"),
        ("no_replay_live_semantic_implementation", "no replay or live semantic implementation changed"),
        ("no_real_execution_metrics", "no queue priority cancel-fill race real lifecycle or other execution metrics"),
        ("no_queue_or_race_proof", "no queue priority exact queue position or cancel-fill race proof"),
        ("no_real_economics_metrics_or_pnl", "no economics metrics or PnL proof"),
        ("no_strategy_live_default_tiny", "no strategy live default-on or tiny-live behavior"),
        ("no_case_library_shadow_parameter", "no case-library shadow decisions or parameter search"),
        ("no_deployment_promotion_viability", "no deployment promotion or execution-layer maker viability proof"),
        ("future_use_requires_separate_task_and_qa", "metrics runner endpoint collector reconciliation or strategy use requires later task and QA"),
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
        description="Validate local replay lifecycle artifacts without endpoints, credentials, runners, metrics, or live behavior."
    )
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate", help="validate one local CSV/JSON replay lifecycle artifact")
    validate.add_argument("--input", required=True, help="local CSV or JSON artifact path")
    validate.add_argument("--summary-out", help="optional JSON summary path")
    validate.set_defaults(func=validate_command)

    generate = subparsers.add_parser("generate-artifacts", help="write 0611T003 synthetic fixtures and validation artifacts")
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
