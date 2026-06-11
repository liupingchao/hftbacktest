#!/usr/bin/env python3
"""Local account/inventory artifact skeleton and validator.

This module is deliberately local-only. It validates task-local CSV/JSON
artifacts against the accepted 0610T008 account inventory source-line
contract. It does not connect to exchange endpoints, read credentials, run
user streams, collect account data, consume runners, compute inventory/PnL
metrics, or authorize strategy use.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "0611T004"
SOURCE_TASK_ID = "0610T008"
SOURCE_FINAL_RECOMMENDATION = "account_inventory_contract_ready_for_qa"
SYNTHESIS_TASK_ID = "0611T001"
SYNTHESIS_FINAL_RECOMMENDATION = "source_line_synthesis_gate_ready_for_qa"
PRIVATE_ORDER_CONTEXT_TASK_ID = "0611T002"
PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION = "private_order_response_artifact_skeleton_ready_for_qa"
REPLAY_LIFECYCLE_CONTEXT_TASK_ID = "0611T003"
REPLAY_LIFECYCLE_CONTEXT_FINAL_RECOMMENDATION = "replay_lifecycle_validation_gate_ready_for_qa"
SOURCE_LINE_ID = "account_inventory_source_line"
SCHEMA_VERSION = "basis_positive_account_inventory_artifact_skeleton_v1"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/basis_positive_account_inventory_source_artifact_skeleton_0611T004")
FIXED_GENERATED_AT = "2026-06-11T00:00:00Z"

EXPECTED_ARTIFACT_SOURCE_CLASS = "accepted_account_inventory_artifact_contract"
EXPECTED_SOURCE_POLICY = "design_only_no_endpoint_no_collector_no_runner"

REQUIRED_FIELDS = [
    "task_id",
    "source_task_id",
    "source_line_id",
    "schema_version",
    "source_artifact_id",
    "artifact_source_class",
    "source_policy",
    "validation_gate_id",
    "venue_id",
    "account_scope_id",
    "asset_id",
    "instrument_id",
    "quantity_unit",
    "precision_policy_id",
    "snapshot_type",
    "state_completeness",
    "transition_type",
    "transition_id",
    "transition_validation_status",
    "available_quantity",
    "locked_quantity",
    "total_quantity",
    "position_quantity",
    "before_quantity",
    "after_quantity",
    "delta_quantity",
    "delta_attribution",
    "account_state_time",
    "artifact_generated_time",
    "validation_or_reconciliation_time",
    "freshness_policy_id",
    "ordering_policy_id",
    "current_proof_status",
    "overclaim_rejection_id",
    "allowed_future_use",
    "forbidden_current_interpretation",
    "fail_closed_reason",
]
CONDITIONAL_FIELDS = [
    "exchange_event_time",
    "local_receive_time",
    "related_future_opaque_order_ref_hash",
    "row_sequence_index",
]
ALL_FIELDS = REQUIRED_FIELDS + CONDITIONAL_FIELDS

FORBIDDEN_FIELDS = {
    "endpoint_url",
    "api_key",
    "api_secret",
    "secret",
    "signature",
    "signing_payload",
    "nonce",
    "user_stream_subscription",
    "listen_key",
    "account_id",
    "order_side",
    "quote_price",
    "quote_size",
    "strategy_signal",
    "live_gate",
    "deployment_flag",
    "promotion_flag",
}

ACCEPTED_SNAPSHOT_TYPES = {
    "initial_snapshot_design_label",
    "periodic_snapshot_design_label",
    "pre_transition_snapshot_design_label",
    "post_transition_snapshot_design_label",
    "reconciliation_snapshot_design_label",
}
FAIL_CLOSED_SNAPSHOT_TYPES = {
    "missing_snapshot_fail_closed",
    "stale_snapshot_fail_closed",
    "partial_snapshot_fail_closed",
    "conflicting_snapshot_fail_closed",
    "unsupported_snapshot_fail_closed",
}
ALLOWED_SNAPSHOT_TYPES = ACCEPTED_SNAPSHOT_TYPES | FAIL_CLOSED_SNAPSHOT_TYPES

ACCEPTED_TRANSITION_TYPES = {
    "fill_derived_candidate_transition",
    "account_observed_transition",
    "transfer_adjustment_transition",
    "funding_settlement_adjustment_transition",
    "fee_rebate_inventory_adjustment_transition",
    "manual_external_adjustment_transition",
}
FAIL_CLOSED_TRANSITION_TYPES = {
    "unknown_transition_fail_closed",
    "ambiguous_transition_fail_closed",
    "conflicting_transition_fail_closed",
    "unsupported_transition_fail_closed",
}
ALLOWED_TRANSITION_TYPES = ACCEPTED_TRANSITION_TYPES | FAIL_CLOSED_TRANSITION_TYPES

ALLOWED_STATE_COMPLETENESS = {"complete", "partial", "missing", "conflicting", "unsupported"}
ALLOWED_TRANSITION_STATUSES = {"accepted_design_label", "fail_closed_design_label", "unsupported_design_label"}
ALLOWED_CURRENT_PROOF_STATUSES = {
    "unproven_design_label",
    "future_account_state_context_only",
    "fail_closed_design_label",
}
ALLOWED_PRECISION_POLICIES = {
    "spot_base_precision_8": 8,
    "contract_unit_precision_3": 3,
}
ALLOWED_FRESHNESS_POLICIES = {"local_fixture_not_live_freshness", "unsupported_freshness_fail_closed"}
ALLOWED_ORDERING_POLICIES = {"account_state_time_then_sequence", "unsupported_ordering_fail_closed"}


@dataclass(frozen=True)
class ValidationIssue:
    source_artifact_id: str
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


def _decimal_value(value: str) -> Decimal | None:
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None


def _int_value(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except ValueError:
        return None


def _scale(value: Decimal) -> int:
    return max(0, -value.as_tuple().exponent)


def _row_id(row: dict[str, str], row_index: int) -> str:
    return row.get("source_artifact_id") or row.get("transition_id") or f"row_{row_index}"


def _issue(row: dict[str, str], row_index: int, reason_code: str, detail: str) -> ValidationIssue:
    return ValidationIssue(
        source_artifact_id=_row_id(row, row_index),
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


def _check_forbidden_fields(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    return [
        _issue(row, row_index, "forbidden_endpoint_or_action_field", field)
        for field in sorted(FORBIDDEN_FIELDS & set(row))
        if str(row.get(field, "")).strip()
    ]


def _check_enums(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    checks = [
        ("source_task_id", {SOURCE_TASK_ID}),
        ("source_line_id", {SOURCE_LINE_ID}),
        ("schema_version", {SCHEMA_VERSION}),
        ("artifact_source_class", {EXPECTED_ARTIFACT_SOURCE_CLASS}),
        ("source_policy", {EXPECTED_SOURCE_POLICY}),
        ("snapshot_type", ALLOWED_SNAPSHOT_TYPES),
        ("state_completeness", ALLOWED_STATE_COMPLETENESS),
        ("transition_type", ALLOWED_TRANSITION_TYPES),
        ("transition_validation_status", ALLOWED_TRANSITION_STATUSES),
        ("current_proof_status", ALLOWED_CURRENT_PROOF_STATUSES),
        ("precision_policy_id", set(ALLOWED_PRECISION_POLICIES)),
        ("freshness_policy_id", ALLOWED_FRESHNESS_POLICIES),
        ("ordering_policy_id", ALLOWED_ORDERING_POLICIES),
    ]
    issues: list[ValidationIssue] = []
    for field, allowed in checks:
        value = str(row.get(field, "")).strip()
        if value and value not in allowed:
            reason = "unsupported_source_policy" if field in {"artifact_source_class", "source_policy"} else "unknown_enum_value"
            issues.append(_issue(row, row_index, reason, f"{field}={value}"))
    return issues


def _check_timestamps(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    required = ["account_state_time", "artifact_generated_time", "validation_or_reconciliation_time"]
    parsed: dict[str, datetime] = {}
    issues: list[ValidationIssue] = []
    for field in required:
        timestamp = _parse_time(row.get(field, ""))
        if timestamp is None:
            issues.append(_issue(row, row_index, "missing_timestamp_domain", field))
        else:
            parsed[field] = timestamp
    if len(parsed) == len(required):
        ordered = [parsed[field] for field in required]
        if any(ordered[i] >= ordered[i + 1] for i in range(len(ordered) - 1)):
            issues.append(
                _issue(
                    row,
                    row_index,
                    "timestamp_domain_merged_or_invalid",
                    "account_state_time<artifact_generated_time<validation_or_reconciliation_time",
                )
            )
    exchange_time = _parse_time(row.get("exchange_event_time", ""))
    local_receive = _parse_time(row.get("local_receive_time", ""))
    if row.get("exchange_event_time") and row.get("local_receive_time"):
        if exchange_time is None or local_receive is None or exchange_time > local_receive:
            issues.append(_issue(row, row_index, "timestamp_domain_merged_or_invalid", "exchange_event_time<=local_receive_time"))
    return issues


def _check_snapshot_transition_fail_closed(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    snapshot_type = row.get("snapshot_type", "")
    transition_type = row.get("transition_type", "")
    state_completeness = row.get("state_completeness", "")
    status = row.get("transition_validation_status", "")
    fail_reason = row.get("fail_closed_reason", "")
    if snapshot_type in FAIL_CLOSED_SNAPSHOT_TYPES and status == "accepted_design_label":
        issues.append(_issue(row, row_index, "snapshot_fail_closed_label_accepted", snapshot_type))
    if transition_type in FAIL_CLOSED_TRANSITION_TYPES and status == "accepted_design_label":
        issues.append(_issue(row, row_index, "transition_fail_closed_label_accepted", transition_type))
    if state_completeness != "complete" and status == "accepted_design_label":
        issues.append(_issue(row, row_index, "incomplete_state_accepted", state_completeness))
    if status != "accepted_design_label" and not fail_reason:
        issues.append(_issue(row, row_index, "missing_fail_closed_reason", status))
    return issues


def _check_quantities(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    quantity_fields = [
        "available_quantity",
        "locked_quantity",
        "total_quantity",
        "position_quantity",
        "before_quantity",
        "after_quantity",
        "delta_quantity",
    ]
    values: dict[str, Decimal] = {}
    issues: list[ValidationIssue] = []
    for field in quantity_fields:
        parsed = _decimal_value(row.get(field, ""))
        if parsed is None:
            issues.append(_issue(row, row_index, "quantity_parse_invalid", field))
        else:
            values[field] = parsed

    policy = row.get("precision_policy_id", "")
    max_scale = ALLOWED_PRECISION_POLICIES.get(policy)
    if max_scale is not None:
        for field, value in values.items():
            if _scale(value) > max_scale:
                issues.append(_issue(row, row_index, "precision_policy_invalid", f"{field} scale>{max_scale}"))

    if {"available_quantity", "locked_quantity", "total_quantity"} <= set(values):
        if values["available_quantity"] + values["locked_quantity"] != values["total_quantity"]:
            issues.append(_issue(row, row_index, "available_locked_total_nonconserving", "available+locked must equal total"))

    if {"before_quantity", "after_quantity", "delta_quantity"} <= set(values):
        if values["before_quantity"] + values["delta_quantity"] != values["after_quantity"]:
            issues.append(_issue(row, row_index, "before_after_delta_nonconserving", "before+delta must equal after"))

    transition_type = row.get("transition_type", "")
    delta = values.get("delta_quantity")
    if transition_type == "transfer_adjustment_transition" and delta is not None and delta == 0:
        issues.append(_issue(row, row_index, "sign_policy_invalid", "transfer adjustment delta cannot be zero"))
    if row.get("quantity_unit", "").strip() in {"unknown", "unsupported", ""}:
        issues.append(_issue(row, row_index, "unit_policy_invalid", row.get("quantity_unit", "")))
    return issues


def _check_overclaims(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    proof_status = row.get("current_proof_status", "")
    allowed = row.get("allowed_future_use", "")
    forbidden = row.get("forbidden_current_interpretation", "")
    transition_type = row.get("transition_type", "")
    artifact_source = row.get("artifact_source_class", "")

    if transition_type == "fill_derived_candidate_transition" and (
        "inventory_lifecycle_proof" in allowed or proof_status == "inventory_lifecycle_proven"
    ):
        issues.append(_issue(row, row_index, "order_fills_alone_inventory_proof_overclaim", transition_type))
    if proof_status in {"inventory_lifecycle_proven", "realized_inventory_proven", "exposure_proven"}:
        issues.append(_issue(row, row_index, "inventory_lifecycle_proof_overclaim", proof_status))
    if "pnl_proof" in allowed or "economics_proof" in allowed or "pnl_allowed" in forbidden:
        issues.append(_issue(row, row_index, "pnl_or_economics_overclaim", allowed or forbidden))
    if any(token in allowed for token in ["live_readiness", "deployment_ready", "promotion_ready", "strategy_decision"]):
        issues.append(_issue(row, row_index, "live_deployment_promotion_overclaim", allowed))
    if artifact_source and artifact_source != EXPECTED_ARTIFACT_SOURCE_CLASS:
        issues.append(_issue(row, row_index, "non_account_authority_inventory_overclaim", artifact_source))
    return issues


def _check_cross_row(rows: list[dict[str, str]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    transition_ids = Counter(row.get("transition_id", "") for row in rows if row.get("transition_id"))
    for index, row in enumerate(rows, start=1):
        transition_id = row.get("transition_id", "")
        if transition_id and transition_ids[transition_id] > 1:
            issues.append(_issue(row, index, "duplicate_transition_identity", transition_id))

    grouped: dict[tuple[str, str, str], list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        grouped[(row.get("account_scope_id", ""), row.get("asset_id", ""), row.get("instrument_id", ""))].append((index, row))
    for _, group_rows in grouped.items():
        sortable = []
        for index, row in group_rows:
            state_time = _parse_time(row.get("account_state_time", ""))
            sequence = _int_value(row.get("row_sequence_index", ""))
            if state_time is not None and sequence is not None:
                sortable.append((index, row, state_time, sequence))
        if len(sortable) < 2:
            continue
        original_order = [(time, seq) for _, _, time, seq in sortable]
        sorted_order = sorted(original_order)
        if original_order != sorted_order:
            for index, row, _, _ in sortable:
                issues.append(_issue(row, index, "out_of_order_transition", row.get("transition_id", "")))
    return issues


def validate_rows(rows: list[dict[str, str]], artifact_path: Path | None = None) -> ValidationResult:
    issues: list[ValidationIssue] = []
    for index, row in enumerate(rows, start=1):
        issues.extend(_check_required_fields(row, index))
        issues.extend(_check_forbidden_fields(row, index))
        issues.extend(_check_enums(row, index))
        issues.extend(_check_timestamps(row, index))
        issues.extend(_check_snapshot_transition_fail_closed(row, index))
        issues.extend(_check_quantities(row, index))
        issues.extend(_check_overclaims(row, index))
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


def _base_row(case_id: str, artifact_id: str, transition_id: str, seq: int) -> dict[str, str]:
    base_time = datetime(2026, 6, 11, tzinfo=timezone.utc) + timedelta(seconds=seq * 10)

    def fmt(offset_seconds: int) -> str:
        return (base_time + timedelta(seconds=offset_seconds)).isoformat().replace("+00:00", "Z")

    return {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "source_line_id": SOURCE_LINE_ID,
        "schema_version": SCHEMA_VERSION,
        "source_artifact_id": artifact_id,
        "artifact_source_class": EXPECTED_ARTIFACT_SOURCE_CLASS,
        "source_policy": EXPECTED_SOURCE_POLICY,
        "validation_gate_id": "account_inventory_local_fixture_gate",
        "venue_id": "binance_design_label",
        "account_scope_id": "opaque_account_scope_hash",
        "asset_id": "BTC",
        "instrument_id": "BTCUSDT",
        "quantity_unit": "base_asset_BTC",
        "precision_policy_id": "spot_base_precision_8",
        "snapshot_type": "reconciliation_snapshot_design_label",
        "state_completeness": "complete",
        "transition_type": "account_observed_transition",
        "transition_id": transition_id,
        "transition_validation_status": "accepted_design_label",
        "available_quantity": "1.10000000",
        "locked_quantity": "0.10000000",
        "total_quantity": "1.20000000",
        "position_quantity": "1.20000000",
        "before_quantity": "1.00000000",
        "after_quantity": "1.20000000",
        "delta_quantity": "0.20000000",
        "delta_attribution": "account_observed_delta",
        "related_future_opaque_order_ref_hash": "",
        "account_state_time": fmt(0),
        "exchange_event_time": fmt(1),
        "local_receive_time": fmt(2),
        "artifact_generated_time": fmt(3),
        "validation_or_reconciliation_time": fmt(4),
        "freshness_policy_id": "local_fixture_not_live_freshness",
        "ordering_policy_id": "account_state_time_then_sequence",
        "current_proof_status": "future_account_state_context_only",
        "overclaim_rejection_id": "reject_current_inventory_lifecycle_proof",
        "allowed_future_use": "later_separately_scoped_account_inventory_reconciliation_design_only",
        "forbidden_current_interpretation": "no_inventory_lifecycle_pnl_live_or_promotion_proof",
        "fail_closed_reason": "none",
        "row_sequence_index": str(seq),
        "fixture_case_id": case_id,
    }


def synthetic_fixture_cases() -> dict[str, list[dict[str, str]]]:
    valid_snapshot = [
        {
            **_base_row("valid_complete_snapshot", "acct_snapshot_valid", "transition_snapshot_context", 1),
            "snapshot_type": "initial_snapshot_design_label",
            "transition_type": "account_observed_transition",
            "delta_quantity": "0.00000000",
            "before_quantity": "1.20000000",
            "after_quantity": "1.20000000",
            "allowed_future_use": "future_initial_inventory_state_design_input_only",
        }
    ]
    valid_transition = [
        _base_row("valid_account_observed_transition", "acct_transition_valid", "transition_valid", 2)
    ]

    missing_required = [{**_base_row("missing_required_field", "acct_missing_required", "transition_missing", 3), "account_scope_id": ""}]
    unknown_enum = [{**_base_row("unknown_enum_value", "acct_unknown_enum", "transition_unknown_enum", 4), "snapshot_type": "live_snapshot"}]
    forbidden_field = [{**_base_row("forbidden_endpoint_field", "acct_forbidden", "transition_forbidden", 5), "endpoint_url": "https://example.invalid/account"}]
    snapshot_fail_closed = [
        {
            **_base_row("snapshot_fail_closed_label_accepted", "acct_snapshot_fail", "transition_snapshot_fail", 6),
            "snapshot_type": "partial_snapshot_fail_closed",
        }
    ]
    transition_fail_closed = [
        {
            **_base_row("transition_fail_closed_label_accepted", "acct_transition_fail", "transition_fail", 7),
            "transition_type": "unknown_transition_fail_closed",
        }
    ]
    total_nonconserving = [
        {**_base_row("available_locked_total_nonconserving", "acct_total_bad", "transition_total_bad", 8), "total_quantity": "1.30000000"}
    ]
    delta_nonconserving = [
        {**_base_row("before_after_delta_nonconserving", "acct_delta_bad", "transition_delta_bad", 9), "after_quantity": "1.10000000"}
    ]
    precision_invalid = [
        {**_base_row("precision_policy_invalid", "acct_precision_bad", "transition_precision_bad", 10), "available_quantity": "1.123456789"}
    ]
    duplicate_transition = [
        _base_row("duplicate_transition_identity", "acct_dup_a", "transition_duplicate", 11),
        _base_row("duplicate_transition_identity", "acct_dup_b", "transition_duplicate", 12),
    ]
    out_of_order = [
        _base_row("out_of_order_transition", "acct_order_a", "transition_order_a", 14),
        _base_row("out_of_order_transition", "acct_order_b", "transition_order_b", 13),
    ]
    fill_overclaim = [
        {
            **_base_row("order_fills_alone_inventory_proof_overclaim", "acct_fill_overclaim", "transition_fill_candidate", 15),
            "transition_type": "fill_derived_candidate_transition",
            "allowed_future_use": "inventory_lifecycle_proof",
        }
    ]
    inventory_overclaim = [
        {
            **_base_row("inventory_lifecycle_proof_overclaim", "acct_inventory_overclaim", "transition_inventory_overclaim", 16),
            "current_proof_status": "inventory_lifecycle_proven",
        }
    ]
    pnl_overclaim = [
        {
            **_base_row("pnl_or_economics_overclaim", "acct_pnl_overclaim", "transition_pnl_overclaim", 17),
            "allowed_future_use": "pnl_proof",
        }
    ]
    non_account_authority = [
        {
            **_base_row("non_account_authority_inventory_overclaim", "acct_non_account", "transition_non_account", 18),
            "artifact_source_class": "private_order_response_artifact",
        }
    ]

    return {
        "valid_complete_snapshot": valid_snapshot,
        "valid_account_observed_transition": valid_transition,
        "missing_required_field": missing_required,
        "unknown_enum_value": unknown_enum,
        "forbidden_endpoint_field": forbidden_field,
        "snapshot_fail_closed_label_accepted": snapshot_fail_closed,
        "transition_fail_closed_label_accepted": transition_fail_closed,
        "available_locked_total_nonconserving": total_nonconserving,
        "before_after_delta_nonconserving": delta_nonconserving,
        "precision_policy_invalid": precision_invalid,
        "duplicate_transition_identity": duplicate_transition,
        "out_of_order_transition": out_of_order,
        "order_fills_alone_inventory_proof_overclaim": fill_overclaim,
        "inventory_lifecycle_proof_overclaim": inventory_overclaim,
        "pnl_or_economics_overclaim": pnl_overclaim,
        "non_account_authority_inventory_overclaim": non_account_authority,
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
                "forbidden_current_interpretation": "endpoint_runner_metric_or_strategy_authorization",
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
                "forbidden_current_interpretation": "endpoint_runner_metric_or_strategy_authorization",
            }
        )
    return rows


def conservation_policy_rows() -> list[dict[str, str]]:
    return [
        {
            "policy_id": "available_locked_total_relationship",
            "scope": "snapshot",
            "required_behavior": "available_quantity plus locked_quantity must equal total_quantity",
            "fail_closed_reason": "available_locked_total_nonconserving",
            "forbidden_current_interpretation": "not realized exposure proof",
        },
        {
            "policy_id": "before_after_delta_relationship",
            "scope": "transition",
            "required_behavior": "before_quantity plus delta_quantity must equal after_quantity",
            "fail_closed_reason": "before_after_delta_nonconserving",
            "forbidden_current_interpretation": "not inventory lifecycle proof",
        },
        {
            "policy_id": "unit_precision_sign_policy",
            "scope": "quantity",
            "required_behavior": "quantity unit and precision policy must be known and all quantities must fit precision",
            "fail_closed_reason": "unit_policy_invalid|precision_policy_invalid|sign_policy_invalid",
            "forbidden_current_interpretation": "not PnL or conversion proof",
        },
        {
            "policy_id": "duplicate_out_of_order_transition_policy",
            "scope": "transition",
            "required_behavior": "transition identity must be unique and account-state rows must remain ordered",
            "fail_closed_reason": "duplicate_transition_identity|out_of_order_transition",
            "forbidden_current_interpretation": "not exchange lifecycle proof",
        },
    ]


def reconciliation_policy_rows() -> list[dict[str, str]]:
    return [
        {
            "policy_id": "order_fills_context_only",
            "evidence_area": "private_order_response_source_line",
            "required_behavior": "order fills remain candidate transition inputs and cannot prove inventory lifecycle alone",
            "fail_closed_reason": "order_fills_alone_inventory_proof_overclaim",
            "forbidden_current_interpretation": "inventory lifecycle proof from fills",
        },
        {
            "policy_id": "account_state_authority_required",
            "evidence_area": "account_inventory_source_line",
            "required_behavior": "accepted account snapshot or account-observed transition plus conservation checks are required",
            "fail_closed_reason": "inventory_lifecycle_proof_overclaim",
            "forbidden_current_interpretation": "current realized inventory proof",
        },
        {
            "policy_id": "economics_separate_source_line",
            "evidence_area": "economics_fee_rebate_source_line",
            "required_behavior": "fees rebates funding settlement and PnL remain separate future authority",
            "fail_closed_reason": "pnl_or_economics_overclaim",
            "forbidden_current_interpretation": "PnL or economics proof",
        },
        {
            "policy_id": "endpoint_collector_forbidden",
            "evidence_area": "future_endpoint_collector",
            "required_behavior": "endpoint reader collector user stream signing nonce and live account data are out of scope",
            "fail_closed_reason": "forbidden_endpoint_or_action_field",
            "forbidden_current_interpretation": "endpoint or collector readiness",
        },
    ]


def _issue_rows(result: ValidationResult, case_id: str) -> list[dict[str, Any]]:
    if result.passed:
        return [
            {
                "fixture_case_id": case_id,
                "source_artifact_id": "",
                "row_index": "",
                "reason_code": "",
                "detail": "pass",
            }
        ]
    return [
        {
            "fixture_case_id": case_id,
            "source_artifact_id": issue.source_artifact_id,
            "row_index": issue.row_index,
            "reason_code": issue.reason_code,
            "detail": issue.detail,
        }
        for issue in result.issues
    ]


def boundary_validation_rows(validation_cases_matched: bool) -> list[dict[str, str]]:
    checks = [
        ("prerequisite_0611T003_qa_passed", "0611T003 QA is passed and replay lifecycle validation gate ready"),
        ("prerequisite_0611T002_qa_passed", "0611T002 QA is passed and private-order skeleton ready"),
        ("prerequisite_0611T001_qa_passed", "0611T001 QA is passed and synthesis gate ready"),
        ("prerequisite_0610T008_qa_passed", "0610T008 QA is passed and account inventory contract ready"),
        ("local_artifact_skeleton_only", "local fixtures parser validator and artifacts only"),
        ("no_endpoint_implementation", "no endpoint URL REST websocket exchange client or connector code"),
        ("no_credentials_signing_nonce_user_stream", "no credentials signing nonce handling or user stream"),
        ("no_private_order_account_live_data_read", "only synthetic task-local fixtures and accepted local artifacts are read"),
        ("no_remote_execution_or_collection", "no remote execution or collection"),
        ("no_runner_consumption", "no execution runner consumption implemented"),
        ("no_real_inventory_or_execution_metrics", "no inventory lifecycle real inventory exposure or execution metrics"),
        ("order_fills_alone_rejected", "order fills alone cannot prove inventory lifecycle"),
        ("no_economics_metrics_or_pnl", "no fees rebates spread capture economics or PnL proof"),
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


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = output_dir / "fixtures"
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    cases = synthetic_fixture_cases()
    pass_cases = {"valid_complete_snapshot", "valid_account_observed_transition"}

    for case_id, rows in cases.items():
        fixture_path = fixture_dir / f"{case_id}.csv"
        fields = ALL_FIELDS + ["fixture_case_id"] + sorted(FORBIDDEN_FIELDS & set().union(*(row.keys() for row in rows)))
        _write_csv(fixture_path, rows, fields)
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
                "covered_validation_area": "|".join(reasons) if reasons else "valid_local_account_inventory_artifact",
                "forbidden_current_interpretation": "endpoint_runner_metric_strategy_or_inventory_proof",
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
        ["fixture_case_id", "source_artifact_id", "row_index", "reason_code", "detail"],
    )
    _write_csv(
        output_dir / "conservation_check_policy.csv",
        conservation_policy_rows(),
        ["policy_id", "scope", "required_behavior", "fail_closed_reason", "forbidden_current_interpretation"],
    )
    _write_csv(
        output_dir / "reconciliation_boundary_policy.csv",
        reconciliation_policy_rows(),
        ["policy_id", "evidence_area", "required_behavior", "fail_closed_reason", "forbidden_current_interpretation"],
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
        "replay_lifecycle_context_task_id": REPLAY_LIFECYCLE_CONTEXT_TASK_ID,
        "replay_lifecycle_context_final_recommendation": REPLAY_LIFECYCLE_CONTEXT_FINAL_RECOMMENDATION,
        "source_line_id": SOURCE_LINE_ID,
        "generated_at": FIXED_GENERATED_AT,
        "final_recommendation": "account_inventory_artifact_skeleton_ready_for_qa"
        if all_expected
        else "account_inventory_artifact_skeleton_needs_revision",
        "fixture_case_count": len(summary_rows),
        "fixture_pass_count": sum(1 for row in summary_rows if row["actual_status"] == "pass"),
        "fixture_fail_closed_count": sum(1 for row in summary_rows if row["actual_status"] == "fail_closed"),
        "all_expected_statuses_matched": all_expected,
        "boundary_flags": {
            "local_account_inventory_artifact_skeleton_only": True,
            "disk_only_csv_json_parser": True,
            "no_endpoint_implementation": True,
            "no_credentials_signing_nonce_user_stream": True,
            "no_private_order_account_live_data_read": True,
            "no_remote_execution_or_collection": True,
            "no_runner_consumption": True,
            "no_real_inventory_metrics": True,
            "no_real_execution_metrics": True,
            "order_fills_alone_cannot_prove_inventory_lifecycle": True,
            "no_economics_metrics_or_pnl": True,
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
            "conservation_check_policy": str(output_dir / "conservation_check_policy.csv"),
            "reconciliation_boundary_policy": str(output_dir / "reconciliation_boundary_policy.csv"),
            "boundary_validation": str(output_dir / "boundary_validation.csv"),
        },
    }
    _write_json(output_dir / "account_inventory_validation_manifest.json", manifest)
    return manifest


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
        description="Validate local account/inventory artifacts without endpoints, credentials, runners, metrics, or live behavior."
    )
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate", help="validate one local CSV/JSON account inventory artifact")
    validate.add_argument("--input", required=True, help="local CSV or JSON artifact path")
    validate.add_argument("--summary-out", help="optional JSON summary path")
    validate.set_defaults(func=validate_command)

    generate = subparsers.add_parser("generate-artifacts", help="write 0611T004 synthetic fixtures and validation artifacts")
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
