#!/usr/bin/env python3
"""Local economics fee/rebate settlement artifact skeleton and validator.

This module is deliberately local-only. It validates task-local CSV/JSON
artifacts against the accepted 0610T009 economics fee/rebate source-line
contract. It does not connect to exchange endpoints, read credentials, run
user streams, collect economics/account/order data, consume runners, compute
real economics metrics or PnL, or authorize strategy use.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "0612T001"
SOURCE_TASK_ID = "0610T009"
SOURCE_FINAL_RECOMMENDATION = "economics_fee_rebate_contract_ready_for_qa"
SYNTHESIS_TASK_ID = "0611T001"
SYNTHESIS_FINAL_RECOMMENDATION = "source_line_synthesis_gate_ready_for_qa"
PRIVATE_ORDER_CONTEXT_TASK_ID = "0611T002"
PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION = "private_order_response_artifact_skeleton_ready_for_qa"
REPLAY_LIFECYCLE_CONTEXT_TASK_ID = "0611T003"
REPLAY_LIFECYCLE_CONTEXT_FINAL_RECOMMENDATION = "replay_lifecycle_validation_gate_ready_for_qa"
ACCOUNT_INVENTORY_CONTEXT_TASK_ID = "0611T004"
ACCOUNT_INVENTORY_CONTEXT_FINAL_RECOMMENDATION = "account_inventory_artifact_skeleton_ready_for_qa"
SOURCE_LINE_ID = "economics_fee_rebate_source_line"
SCHEMA_VERSION = "basis_positive_economics_fee_rebate_artifact_skeleton_v1"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/basis_positive_economics_fee_rebate_source_artifact_skeleton_0612T001")
FIXED_GENERATED_AT = "2026-06-12T00:00:00Z"

EXPECTED_ARTIFACT_SOURCE_CLASS = "accepted_economics_fee_rebate_artifact_contract"
EXPECTED_SOURCE_POLICY = "design_only_no_endpoint_no_collector_no_runner"
EXPECTED_SETTLEMENT_AUTHORITY = "economics_settlement_record_authority_design_label"

REQUIRED_FIELDS = [
    "task_id",
    "source_task_id",
    "source_line_id",
    "schema_version",
    "settlement_record_id",
    "settlement_version",
    "venue_id",
    "instrument_id",
    "account_scope_id",
    "settlement_scope",
    "source_artifact_id",
    "artifact_source_class",
    "source_policy",
    "settlement_source_authority",
    "prior_source_line_context",
    "validation_gate_id",
    "opaque_future_fill_ref",
    "fill_dependency_status",
    "fill_dependency_source_line",
    "fill_validation_status",
    "forbidden_fill_only_interpretation",
    "maker_taker_label",
    "classification_evidence_type",
    "venue_rule_dependency_status",
    "classification_validation_status",
    "settlement_label",
    "settlement_validation_status",
    "fee_amount",
    "rebate_amount",
    "net_fee_amount",
    "amount_sign_convention",
    "fee_currency",
    "rebate_currency",
    "settlement_currency",
    "base_asset",
    "quote_asset",
    "fee_asset",
    "rebate_asset",
    "precision_policy_id",
    "rounding_policy_id",
    "conversion_source_provenance",
    "conversion_timestamp",
    "conversion_rate",
    "conversion_tolerance",
    "net_fee_in_settlement_currency",
    "tick_size",
    "tick_value",
    "net_fee_ticks",
    "conversion_validation_status",
    "quoted_spread_ticks",
    "filled_spread_ticks",
    "realized_spread_ticks",
    "fee_adjusted_spread_ticks",
    "spread_capture_label",
    "markout_context_design_label",
    "hypothetical_spread_flag",
    "spread_capture_proof_status",
    "spread_validation_status",
    "fill_time",
    "exchange_settlement_time",
    "local_receive_time",
    "account_economics_reconciliation_time",
    "artifact_generated_time",
    "validation_time",
    "current_proof_status",
    "allowed_future_use",
    "forbidden_current_interpretation",
    "overclaim_rejection_id",
    "fail_closed_reason",
]
CONDITIONAL_FIELDS = [
    "row_sequence_index",
    "public_markout_context_ref",
    "account_inventory_context_ref",
    "replay_lifecycle_context_ref",
]
ALL_FIELDS = REQUIRED_FIELDS + CONDITIONAL_FIELDS

FORBIDDEN_FIELDS = {
    "endpoint_url",
    "api_key",
    "api_secret",
    "credential",
    "secret",
    "signature",
    "signing_payload",
    "nonce",
    "listen_key",
    "user_stream_subscription",
    "account_id",
    "order_side",
    "quote_price",
    "quote_size",
    "executable_action",
    "strategy_signal",
    "live_gate",
    "deployment_flag",
    "promotion_flag",
}

ACCEPTED_SETTLEMENT_LABELS = {
    "maker_fee_settlement_design_label",
    "maker_rebate_settlement_design_label",
    "taker_fee_settlement_design_label",
    "zero_fee_settlement_design_label",
    "funding_commission_adjustment_design_label",
}
FAIL_CLOSED_SETTLEMENT_LABELS = {
    "missing_settlement_fail_closed",
    "delayed_settlement_fail_closed",
    "partial_settlement_fail_closed",
    "unknown_settlement_fail_closed",
    "ambiguous_settlement_fail_closed",
    "conflicting_settlement_fail_closed",
    "unsupported_settlement_fail_closed",
}
ALLOWED_SETTLEMENT_LABELS = ACCEPTED_SETTLEMENT_LABELS | FAIL_CLOSED_SETTLEMENT_LABELS

ACCEPTED_SPREAD_LABELS = {
    "quoted_spread_context_design_label",
    "filled_spread_context_design_label",
    "realized_spread_design_label",
    "markout_context_design_label",
}
FAIL_CLOSED_SPREAD_LABELS = {
    "hypothetical_spread_context_design_label",
    "missing_spread_capture_fail_closed",
    "ambiguous_spread_capture_fail_closed",
    "conflicting_spread_capture_fail_closed",
    "unsupported_spread_capture_fail_closed",
}
ALLOWED_SPREAD_LABELS = ACCEPTED_SPREAD_LABELS | FAIL_CLOSED_SPREAD_LABELS

ALLOWED_MAKER_TAKER_LABELS = {"maker", "taker", "zero_fee", "funding_adjustment"}
FAIL_CLOSED_MAKER_TAKER_LABELS = {
    "maker_taker_missing_fail_closed",
    "maker_taker_unknown_fail_closed",
    "maker_taker_ambiguous_fail_closed",
    "maker_taker_conflicting_fail_closed",
    "venue_rule_dependency_unaccepted_fail_closed",
    "unsupported_classification_fail_closed",
}
ALLOWED_CLASSIFICATION_LABELS = ALLOWED_MAKER_TAKER_LABELS | FAIL_CLOSED_MAKER_TAKER_LABELS

ALLOWED_VALIDATION_STATUSES = {"accepted_design_label", "fail_closed_design_label", "unsupported_design_label"}
ALLOWED_FILL_DEPENDENCY_STATUSES = {
    "future_fill_dependency_context_only",
    "missing_fill_dependency_fail_closed",
    "unsupported_fill_dependency_fail_closed",
}
ALLOWED_CLASSIFICATION_EVIDENCE = {
    "accepted_economics_settlement_artifact",
    "accepted_exchange_settlement_label",
    "accepted_venue_rule_classification_record",
    "private_fill_label_only_insufficient",
    "public_book_context_only_insufficient",
    "replay_lifecycle_label_only_insufficient",
    "unsupported_classification_evidence",
}
ALLOWED_VENUE_RULE_STATUSES = {"accepted_rule_provenance", "not_required", "unaccepted_rule_dependency_fail_closed"}
ALLOWED_SIGN_CONVENTIONS = {"fees_positive_rebates_negative_net_fee_equals_fee_plus_rebate"}
ALLOWED_PRECISION_POLICIES = {"quote_precision_8": 8, "settlement_precision_8": 8}
ALLOWED_ROUNDING_POLICIES = {"decimal_exact_no_rounding", "round_half_even_8dp"}
ALLOWED_CONVERSION_PROVENANCE = {
    "same_currency_no_conversion",
    "accepted_local_conversion_context",
    "missing_conversion_fail_closed",
    "conflicting_conversion_fail_closed",
    "unsupported_conversion_fail_closed",
}
ALLOWED_CONVERSION_STATUSES = {"accepted_design_label", "fail_closed_design_label", "unsupported_design_label"}
ALLOWED_PROOF_STATUSES = {"unproven_design_label", "future_economics_context_only", "fail_closed_design_label"}
ALLOWED_MARKOUT_LABELS = {"markout_context_only", "none", "public_markout_context_only_insufficient"}
ALLOWED_HYPOTHETICAL_FLAGS = {"0", "1", "false", "true", "no", "yes"}


@dataclass(frozen=True)
class ValidationIssue:
    settlement_record_id: str
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


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _scale(value: Decimal) -> int:
    return max(0, -value.as_tuple().exponent)


def _within_tolerance(left: Decimal, right: Decimal, tolerance: Decimal) -> bool:
    return abs(left - right) <= tolerance


def _row_id(row: dict[str, str], row_index: int) -> str:
    return row.get("settlement_record_id") or row.get("source_artifact_id") or f"row_{row_index}"


def _issue(row: dict[str, str], row_index: int, reason_code: str, detail: str) -> ValidationIssue:
    return ValidationIssue(
        settlement_record_id=_row_id(row, row_index),
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
        ("fill_dependency_status", ALLOWED_FILL_DEPENDENCY_STATUSES),
        ("fill_dependency_source_line", {"private_order_response_source_line"}),
        ("fill_validation_status", ALLOWED_VALIDATION_STATUSES),
        ("maker_taker_label", ALLOWED_CLASSIFICATION_LABELS),
        ("classification_evidence_type", ALLOWED_CLASSIFICATION_EVIDENCE),
        ("venue_rule_dependency_status", ALLOWED_VENUE_RULE_STATUSES),
        ("classification_validation_status", ALLOWED_VALIDATION_STATUSES),
        ("settlement_label", ALLOWED_SETTLEMENT_LABELS),
        ("settlement_validation_status", ALLOWED_VALIDATION_STATUSES),
        ("amount_sign_convention", ALLOWED_SIGN_CONVENTIONS),
        ("precision_policy_id", set(ALLOWED_PRECISION_POLICIES)),
        ("rounding_policy_id", ALLOWED_ROUNDING_POLICIES),
        ("conversion_source_provenance", ALLOWED_CONVERSION_PROVENANCE),
        ("conversion_validation_status", ALLOWED_CONVERSION_STATUSES),
        ("spread_capture_label", ALLOWED_SPREAD_LABELS),
        ("markout_context_design_label", ALLOWED_MARKOUT_LABELS),
        ("hypothetical_spread_flag", ALLOWED_HYPOTHETICAL_FLAGS),
        ("spread_capture_proof_status", ALLOWED_PROOF_STATUSES),
        ("spread_validation_status", ALLOWED_VALIDATION_STATUSES),
        ("current_proof_status", ALLOWED_PROOF_STATUSES),
    ]
    issues: list[ValidationIssue] = []
    for field, allowed in checks:
        value = str(row.get(field, "")).strip()
        if value and value not in allowed:
            reason = "unsupported_source_policy" if field in {"artifact_source_class", "source_policy"} else "unknown_enum_value"
            issues.append(_issue(row, row_index, reason, f"{field}={value}"))
    return issues


def _check_settlement_authority(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    authority = row.get("settlement_source_authority", "").strip()
    if not authority:
        return [_issue(row, row_index, "missing_settlement_authority", "settlement_source_authority")]
    if authority == "conflicting_settlement_authority":
        return [_issue(row, row_index, "conflicting_settlement_authority", authority)]
    if authority != EXPECTED_SETTLEMENT_AUTHORITY:
        return [_issue(row, row_index, "unsupported_source_policy", f"settlement_source_authority={authority}")]
    return []


def _check_timestamps(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    order = [
        "fill_time",
        "exchange_settlement_time",
        "local_receive_time",
        "account_economics_reconciliation_time",
        "artifact_generated_time",
        "validation_time",
    ]
    parsed: dict[str, datetime] = {}
    issues: list[ValidationIssue] = []
    for field in order:
        timestamp = _parse_time(row.get(field, ""))
        if timestamp is None:
            issues.append(_issue(row, row_index, "missing_timestamp_domain", field))
        else:
            parsed[field] = timestamp
    if len(parsed) == len(order):
        ordered = [parsed[field] for field in order]
        if any(ordered[index] >= ordered[index + 1] for index in range(len(ordered) - 1)):
            issues.append(
                _issue(
                    row,
                    row_index,
                    "timestamp_domain_merged_or_invalid",
                    "fill<exchange_settlement<local_receive<reconciliation<artifact_generated<validation",
                )
            )
    conversion_time = _parse_time(row.get("conversion_timestamp", ""))
    if conversion_time is None:
        issues.append(_issue(row, row_index, "missing_timestamp_domain", "conversion_timestamp"))
    elif "fill_time" in parsed and "exchange_settlement_time" in parsed:
        if not (parsed["fill_time"] < conversion_time < parsed["exchange_settlement_time"]):
            issues.append(_issue(row, row_index, "timestamp_domain_merged_or_invalid", "conversion_timestamp"))
    return issues


def _check_fail_closed_labels(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    settlement_label = row.get("settlement_label", "")
    spread_label = row.get("spread_capture_label", "")
    maker_taker = row.get("maker_taker_label", "")
    settlement_status = row.get("settlement_validation_status", "")
    classification_status = row.get("classification_validation_status", "")
    spread_status = row.get("spread_validation_status", "")
    fail_reason = row.get("fail_closed_reason", "")

    if settlement_label in FAIL_CLOSED_SETTLEMENT_LABELS and settlement_status == "accepted_design_label":
        issues.append(_issue(row, row_index, "settlement_fail_closed_label_accepted", settlement_label))
    if spread_label in FAIL_CLOSED_SPREAD_LABELS and spread_status == "accepted_design_label":
        issues.append(_issue(row, row_index, "spread_fail_closed_label_accepted", spread_label))
    if maker_taker in FAIL_CLOSED_MAKER_TAKER_LABELS and classification_status == "accepted_design_label":
        issues.append(_issue(row, row_index, "unsupported_maker_taker_classification", maker_taker))
    if row.get("classification_evidence_type") in {
        "private_fill_label_only_insufficient",
        "public_book_context_only_insufficient",
        "replay_lifecycle_label_only_insufficient",
        "unsupported_classification_evidence",
    } and classification_status == "accepted_design_label":
        issues.append(_issue(row, row_index, "unsupported_maker_taker_classification", row.get("classification_evidence_type", "")))
    if row.get("venue_rule_dependency_status") == "unaccepted_rule_dependency_fail_closed" and classification_status == "accepted_design_label":
        issues.append(_issue(row, row_index, "unsupported_maker_taker_classification", "venue_rule_dependency_unaccepted"))
    if any(
        status != "accepted_design_label"
        for status in [settlement_status, classification_status, row.get("conversion_validation_status", ""), spread_status]
    ) and not fail_reason:
        issues.append(_issue(row, row_index, "missing_fail_closed_reason", "fail_closed_or_unsupported_status"))
    return issues


def _check_arithmetic(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    decimal_fields = [
        "fee_amount",
        "rebate_amount",
        "net_fee_amount",
        "conversion_rate",
        "conversion_tolerance",
        "net_fee_in_settlement_currency",
        "tick_size",
        "tick_value",
        "net_fee_ticks",
        "quoted_spread_ticks",
        "filled_spread_ticks",
        "realized_spread_ticks",
        "fee_adjusted_spread_ticks",
    ]
    values: dict[str, Decimal] = {}
    issues: list[ValidationIssue] = []
    for field in decimal_fields:
        parsed = _decimal_value(row.get(field, ""))
        if parsed is None:
            issues.append(_issue(row, row_index, "arithmetic_parse_invalid", field))
        else:
            values[field] = parsed

    max_scale = ALLOWED_PRECISION_POLICIES.get(row.get("precision_policy_id", ""))
    if max_scale is not None:
        for field in ["fee_amount", "rebate_amount", "net_fee_amount", "net_fee_in_settlement_currency"]:
            value = values.get(field)
            if value is not None and _scale(value) > max_scale:
                issues.append(_issue(row, row_index, "precision_policy_invalid", f"{field} scale>{max_scale}"))

    if {"fee_amount", "rebate_amount", "net_fee_amount"} <= set(values):
        if values["fee_amount"] + values["rebate_amount"] != values["net_fee_amount"]:
            issues.append(_issue(row, row_index, "fee_rebate_arithmetic_nonconserving", "fee_amount+rebate_amount must equal net_fee_amount"))
        if values["fee_amount"] < 0 or values["rebate_amount"] > 0:
            issues.append(_issue(row, row_index, "amount_sign_policy_invalid", "fee must be >=0 and rebate must be <=0"))

    tolerance = values.get("conversion_tolerance", Decimal("0.00000001"))
    if {"net_fee_amount", "conversion_rate", "net_fee_in_settlement_currency"} <= set(values):
        expected = values["net_fee_amount"] * values["conversion_rate"]
        if not _within_tolerance(expected, values["net_fee_in_settlement_currency"], tolerance):
            issues.append(_issue(row, row_index, "currency_conversion_mismatch", "net_fee_amount*conversion_rate"))

    if {"net_fee_in_settlement_currency", "tick_value", "net_fee_ticks"} <= set(values):
        if values["tick_value"] <= 0:
            issues.append(_issue(row, row_index, "tick_value_arithmetic_mismatch", "tick_value must be positive"))
        else:
            expected_ticks = values["net_fee_in_settlement_currency"] / values["tick_value"]
            if not _within_tolerance(expected_ticks, values["net_fee_ticks"], tolerance):
                issues.append(_issue(row, row_index, "tick_value_arithmetic_mismatch", "net_fee_in_settlement_currency/tick_value"))

    if {"realized_spread_ticks", "net_fee_ticks", "fee_adjusted_spread_ticks"} <= set(values):
        expected_adjusted = values["realized_spread_ticks"] - values["net_fee_ticks"]
        if not _within_tolerance(expected_adjusted, values["fee_adjusted_spread_ticks"], tolerance):
            issues.append(_issue(row, row_index, "spread_capture_consistency_mismatch", "realized_spread_ticks-net_fee_ticks"))
    return issues


def _check_overclaims(row: dict[str, str], row_index: int) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    allowed = row.get("allowed_future_use", "")
    forbidden = row.get("forbidden_current_interpretation", "")
    proof_status = row.get("current_proof_status", "")
    spread_proof = row.get("spread_capture_proof_status", "")
    text = "|".join([allowed, forbidden, proof_status, spread_proof])

    if _truthy(row.get("hypothetical_spread_flag", "")) or "hypothetical_spread_proof" in text:
        issues.append(_issue(row, row_index, "hypothetical_spread_overclaim", row.get("hypothetical_spread_flag", "")))
    if "fill_notional_proof" in text:
        issues.append(_issue(row, row_index, "fill_notional_alone_overclaim", text))
    if "order_fills_alone_proof" in text or "order_fill_proof" in text:
        issues.append(_issue(row, row_index, "order_fills_alone_overclaim", text))
    if "public_markout_proof" in text:
        issues.append(_issue(row, row_index, "public_markout_alone_overclaim", text))
    if "account_inventory_proof" in text:
        issues.append(_issue(row, row_index, "account_inventory_alone_overclaim", text))
    if "replay_lifecycle_proof" in text:
        issues.append(_issue(row, row_index, "replay_lifecycle_alone_overclaim", text))
    if any(token in text for token in ["pnl_proof", "realized_economics_proof", "fees_rebates_spread_capture_proven"]):
        issues.append(_issue(row, row_index, "pnl_proof_overclaim", text))
    if any(token in text for token in ["live_readiness", "default_on", "tiny_live", "deployment_ready", "promotion_ready", "strategy_decision"]):
        issues.append(_issue(row, row_index, "live_deployment_promotion_overclaim", text))
    if proof_status not in {"unproven_design_label", "future_economics_context_only", "fail_closed_design_label"}:
        issues.append(_issue(row, row_index, "pnl_proof_overclaim", proof_status))
    return issues


def _check_cross_row(rows: list[dict[str, str]]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    settlement_ids = Counter(row.get("settlement_record_id", "") for row in rows if row.get("settlement_record_id"))
    for index, row in enumerate(rows, start=1):
        settlement_id = row.get("settlement_record_id", "")
        if settlement_id and settlement_ids[settlement_id] > 1:
            issues.append(_issue(row, index, "duplicate_settlement_identity", settlement_id))
    return issues


def validate_rows(rows: list[dict[str, str]], artifact_path: Path | None = None) -> ValidationResult:
    issues: list[ValidationIssue] = []
    for index, row in enumerate(rows, start=1):
        issues.extend(_check_required_fields(row, index))
        issues.extend(_check_forbidden_fields(row, index))
        issues.extend(_check_enums(row, index))
        issues.extend(_check_settlement_authority(row, index))
        issues.extend(_check_timestamps(row, index))
        issues.extend(_check_fail_closed_labels(row, index))
        issues.extend(_check_arithmetic(row, index))
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


def _base_row(case_id: str, settlement_id: str, seq: int) -> dict[str, str]:
    base_time = datetime(2026, 6, 12, tzinfo=timezone.utc) + timedelta(seconds=seq * 10)

    def fmt(offset_seconds: int) -> str:
        return (base_time + timedelta(seconds=offset_seconds)).isoformat().replace("+00:00", "Z")

    return {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "source_line_id": SOURCE_LINE_ID,
        "schema_version": SCHEMA_VERSION,
        "settlement_record_id": settlement_id,
        "settlement_version": "1",
        "venue_id": "binance_design_label",
        "instrument_id": "BTCUSDT",
        "account_scope_id": "opaque_account_scope_hash",
        "settlement_scope": "single_fill_settlement_context",
        "source_artifact_id": f"econ_artifact_{settlement_id}",
        "artifact_source_class": EXPECTED_ARTIFACT_SOURCE_CLASS,
        "source_policy": EXPECTED_SOURCE_POLICY,
        "settlement_source_authority": EXPECTED_SETTLEMENT_AUTHORITY,
        "prior_source_line_context": "0611T002|0611T003|0611T004_context_only",
        "validation_gate_id": "economics_fee_rebate_local_fixture_gate",
        "opaque_future_fill_ref": "opaque_future_fill_ref_hash",
        "fill_dependency_status": "future_fill_dependency_context_only",
        "fill_dependency_source_line": "private_order_response_source_line",
        "fill_validation_status": "accepted_design_label",
        "forbidden_fill_only_interpretation": "fill_notional_or_order_fill_alone_cannot_prove_fees_rebates_spread_capture_or_pnl",
        "maker_taker_label": "maker",
        "classification_evidence_type": "accepted_economics_settlement_artifact",
        "venue_rule_dependency_status": "not_required",
        "classification_validation_status": "accepted_design_label",
        "settlement_label": "maker_fee_settlement_design_label",
        "settlement_validation_status": "accepted_design_label",
        "fee_amount": "2.00000000",
        "rebate_amount": "-0.50000000",
        "net_fee_amount": "1.50000000",
        "amount_sign_convention": "fees_positive_rebates_negative_net_fee_equals_fee_plus_rebate",
        "fee_currency": "USDT",
        "rebate_currency": "USDT",
        "settlement_currency": "USDT",
        "base_asset": "BTC",
        "quote_asset": "USDT",
        "fee_asset": "USDT",
        "rebate_asset": "USDT",
        "precision_policy_id": "settlement_precision_8",
        "rounding_policy_id": "decimal_exact_no_rounding",
        "conversion_source_provenance": "same_currency_no_conversion",
        "conversion_timestamp": fmt(1),
        "conversion_rate": "1.00000000",
        "conversion_tolerance": "0.00000001",
        "net_fee_in_settlement_currency": "1.50000000",
        "tick_size": "0.10000000",
        "tick_value": "0.10000000",
        "net_fee_ticks": "15.00000000",
        "conversion_validation_status": "accepted_design_label",
        "quoted_spread_ticks": "30.00000000",
        "filled_spread_ticks": "28.00000000",
        "realized_spread_ticks": "25.00000000",
        "fee_adjusted_spread_ticks": "10.00000000",
        "spread_capture_label": "realized_spread_design_label",
        "markout_context_design_label": "markout_context_only",
        "hypothetical_spread_flag": "0",
        "spread_capture_proof_status": "future_economics_context_only",
        "spread_validation_status": "accepted_design_label",
        "fill_time": fmt(0),
        "exchange_settlement_time": fmt(2),
        "local_receive_time": fmt(3),
        "account_economics_reconciliation_time": fmt(4),
        "artifact_generated_time": fmt(5),
        "validation_time": fmt(6),
        "current_proof_status": "future_economics_context_only",
        "allowed_future_use": "later_separately_scoped_economics_reconciliation_design_input_only",
        "forbidden_current_interpretation": "no_real_fees_rebates_spread_capture_pnl_live_deployment_or_promotion_proof",
        "overclaim_rejection_id": "reject_current_real_economics_pnl_proof",
        "fail_closed_reason": "none",
        "row_sequence_index": str(seq),
        "public_markout_context_ref": "public_markout_context_only",
        "account_inventory_context_ref": "0611T004_context_only_not_economics_proof",
        "replay_lifecycle_context_ref": "0611T003_context_only_not_economics_proof",
        "fixture_case_id": case_id,
    }


def synthetic_fixture_cases() -> dict[str, list[dict[str, str]]]:
    valid_maker_fee = [_base_row("valid_maker_fee_settlement", "settlement_valid_fee", 1)]
    valid_rebate = [
        {
            **_base_row("valid_maker_rebate_settlement", "settlement_valid_rebate", 2),
            "settlement_label": "maker_rebate_settlement_design_label",
            "fee_amount": "0.00000000",
            "rebate_amount": "-0.70000000",
            "net_fee_amount": "-0.70000000",
            "net_fee_in_settlement_currency": "-0.70000000",
            "net_fee_ticks": "-7.00000000",
            "fee_adjusted_spread_ticks": "32.00000000",
        }
    ]
    return {
        "valid_maker_fee_settlement": valid_maker_fee,
        "valid_maker_rebate_settlement": valid_rebate,
        "missing_required_field": [{**_base_row("missing_required_field", "settlement_missing_required", 3), "settlement_record_id": ""}],
        "unknown_enum_value": [{**_base_row("unknown_enum_value", "settlement_unknown_enum", 4), "settlement_label": "real_fee_proof"}],
        "forbidden_endpoint_field": [
            {**_base_row("forbidden_endpoint_field", "settlement_forbidden", 5), "endpoint_url": "https://example.invalid/economics"}
        ],
        "missing_settlement_authority": [
            {**_base_row("missing_settlement_authority", "settlement_missing_authority", 6), "settlement_source_authority": ""}
        ],
        "conflicting_settlement_authority": [
            {
                **_base_row("conflicting_settlement_authority", "settlement_conflicting_authority", 7),
                "settlement_source_authority": "conflicting_settlement_authority",
            }
        ],
        "unsupported_maker_taker_classification": [
            {
                **_base_row("unsupported_maker_taker_classification", "settlement_bad_classification", 8),
                "maker_taker_label": "maker_taker_unknown_fail_closed",
            }
        ],
        "settlement_fail_closed_label_accepted": [
            {
                **_base_row("settlement_fail_closed_label_accepted", "settlement_fail_closed_accepted", 9),
                "settlement_label": "partial_settlement_fail_closed",
            }
        ],
        "spread_fail_closed_label_accepted": [
            {
                **_base_row("spread_fail_closed_label_accepted", "settlement_spread_fail_closed", 10),
                "spread_capture_label": "missing_spread_capture_fail_closed",
            }
        ],
        "fee_rebate_arithmetic_nonconserving": [
            {**_base_row("fee_rebate_arithmetic_nonconserving", "settlement_bad_fee_math", 11), "net_fee_amount": "1.60000000"}
        ],
        "currency_conversion_mismatch": [
            {**_base_row("currency_conversion_mismatch", "settlement_bad_conversion", 12), "net_fee_in_settlement_currency": "1.40000000"}
        ],
        "tick_value_arithmetic_mismatch": [
            {**_base_row("tick_value_arithmetic_mismatch", "settlement_bad_tick", 13), "net_fee_ticks": "14.00000000"}
        ],
        "timestamp_domain_merged_or_invalid": [
            {
                **_base_row("timestamp_domain_merged_or_invalid", "settlement_bad_time", 14),
                "exchange_settlement_time": _base_row("timestamp_domain_merged_or_invalid", "settlement_bad_time", 14)["fill_time"],
            }
        ],
        "hypothetical_spread_overclaim": [
            {**_base_row("hypothetical_spread_overclaim", "settlement_hypothetical", 15), "hypothetical_spread_flag": "1"}
        ],
        "fill_notional_alone_overclaim": [
            {
                **_base_row("fill_notional_alone_overclaim", "settlement_fill_notional", 16),
                "allowed_future_use": "fill_notional_proof",
            }
        ],
        "order_fills_alone_overclaim": [
            {
                **_base_row("order_fills_alone_overclaim", "settlement_order_fill", 17),
                "allowed_future_use": "order_fills_alone_proof",
            }
        ],
        "public_markout_alone_overclaim": [
            {
                **_base_row("public_markout_alone_overclaim", "settlement_markout", 18),
                "allowed_future_use": "public_markout_proof",
            }
        ],
        "account_inventory_alone_overclaim": [
            {
                **_base_row("account_inventory_alone_overclaim", "settlement_inventory", 19),
                "allowed_future_use": "account_inventory_proof",
            }
        ],
        "replay_lifecycle_alone_overclaim": [
            {
                **_base_row("replay_lifecycle_alone_overclaim", "settlement_replay", 20),
                "allowed_future_use": "replay_lifecycle_proof",
            }
        ],
        "pnl_proof_overclaim": [
            {**_base_row("pnl_proof_overclaim", "settlement_pnl", 21), "allowed_future_use": "pnl_proof"}
        ],
        "live_deployment_promotion_overclaim": [
            {
                **_base_row("live_deployment_promotion_overclaim", "settlement_live", 22),
                "allowed_future_use": "live_readiness",
            }
        ],
        "duplicate_settlement_identity": [
            _base_row("duplicate_settlement_identity", "settlement_duplicate", 23),
            _base_row("duplicate_settlement_identity", "settlement_duplicate", 24),
        ],
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
                "forbidden_current_interpretation": "endpoint_runner_metric_pnl_or_strategy_authorization",
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
                "forbidden_current_interpretation": "endpoint_runner_metric_pnl_or_strategy_authorization",
            }
        )
    return rows


def arithmetic_policy_rows() -> list[dict[str, str]]:
    return [
        {
            "policy_id": "fee_rebate_net_fee_relationship",
            "scope": "fee_rebate_amounts",
            "required_behavior": "fee_amount plus rebate_amount must equal net_fee_amount under explicit sign convention",
            "fail_closed_reason": "fee_rebate_arithmetic_nonconserving",
            "forbidden_current_interpretation": "not real fee rebate or PnL proof",
        },
        {
            "policy_id": "currency_conversion_relationship",
            "scope": "currency_conversion",
            "required_behavior": "net_fee_amount times conversion_rate must equal net_fee_in_settlement_currency within tolerance",
            "fail_closed_reason": "currency_conversion_mismatch",
            "forbidden_current_interpretation": "not accepted venue conversion proof",
        },
        {
            "policy_id": "tick_value_relationship",
            "scope": "tick_value",
            "required_behavior": "net_fee_in_settlement_currency divided by tick_value must equal net_fee_ticks within tolerance",
            "fail_closed_reason": "tick_value_arithmetic_mismatch",
            "forbidden_current_interpretation": "not executable price or quote proof",
        },
        {
            "policy_id": "fee_adjusted_spread_relationship",
            "scope": "spread_capture_context",
            "required_behavior": "realized_spread_ticks minus net_fee_ticks must equal fee_adjusted_spread_ticks within tolerance",
            "fail_closed_reason": "spread_capture_consistency_mismatch",
            "forbidden_current_interpretation": "not current spread capture or PnL proof",
        },
    ]


def reconciliation_policy_rows() -> list[dict[str, str]]:
    return [
        {
            "policy_id": "economics_settlement_authority_required",
            "evidence_area": "economics_fee_rebate_source_line",
            "required_behavior": "accepted economics settlement authority and arithmetic gates are required for local candidate validation",
            "fail_closed_reason": "missing_settlement_authority|conflicting_settlement_authority",
            "forbidden_current_interpretation": "current realized economics proof",
        },
        {
            "policy_id": "private_order_context_only",
            "evidence_area": "private_order_response_source_line",
            "required_behavior": "private order fills remain future dependency context and cannot prove fees rebates spread capture or PnL alone",
            "fail_closed_reason": "order_fills_alone_overclaim|fill_notional_alone_overclaim",
            "forbidden_current_interpretation": "economics proof from order fills",
        },
        {
            "policy_id": "account_inventory_context_only",
            "evidence_area": "account_inventory_source_line",
            "required_behavior": "account inventory can be future reconciliation context only and cannot prove economics or PnL alone",
            "fail_closed_reason": "account_inventory_alone_overclaim",
            "forbidden_current_interpretation": "economics proof from account inventory",
        },
        {
            "policy_id": "replay_public_context_only",
            "evidence_area": "replay_lifecycle_or_public_markout",
            "required_behavior": "replay lifecycle and public markout remain context only and cannot prove realized spread capture or PnL",
            "fail_closed_reason": "replay_lifecycle_alone_overclaim|public_markout_alone_overclaim|hypothetical_spread_overclaim",
            "forbidden_current_interpretation": "realized spread capture proof",
        },
        {
            "policy_id": "endpoint_collector_forbidden",
            "evidence_area": "future_endpoint_collector",
            "required_behavior": "endpoint reader collector user stream signing nonce and live account data are out of scope",
            "fail_closed_reason": "forbidden_endpoint_or_action_field",
            "forbidden_current_interpretation": "endpoint collector or live readiness",
        },
    ]


def _issue_rows(result: ValidationResult, case_id: str) -> list[dict[str, Any]]:
    if result.passed:
        return [
            {
                "fixture_case_id": case_id,
                "settlement_record_id": "",
                "row_index": "",
                "reason_code": "",
                "detail": "pass",
            }
        ]
    return [
        {
            "fixture_case_id": case_id,
            "settlement_record_id": issue.settlement_record_id,
            "row_index": issue.row_index,
            "reason_code": issue.reason_code,
            "detail": issue.detail,
        }
        for issue in result.issues
    ]


def boundary_validation_rows(validation_cases_matched: bool) -> list[dict[str, str]]:
    checks = [
        ("prerequisite_0611T004_qa_passed", "0611T004 QA is passed and account inventory skeleton ready"),
        ("prerequisite_0611T003_qa_passed", "0611T003 QA is passed and replay lifecycle gate ready"),
        ("prerequisite_0611T002_qa_passed", "0611T002 QA is passed and private-order skeleton ready"),
        ("prerequisite_0611T001_qa_passed", "0611T001 QA is passed and synthesis gate ready"),
        ("prerequisite_0610T009_qa_passed", "0610T009 QA is passed and economics contract ready"),
        ("local_economics_artifact_skeleton_only", "local synthetic fixtures parser validator and artifacts only"),
        ("disk_only_csv_json_parser", "CSV and JSON are read from local paths only"),
        ("no_endpoint_implementation", "no endpoint URL REST websocket exchange client or connector code"),
        ("no_credentials_signing_nonce_user_stream", "no credentials signing nonce handling or user stream"),
        ("no_private_order_account_live_economics_data_read", "only synthetic task-local fixtures and accepted local artifacts are read"),
        ("no_remote_execution_or_collection", "no remote execution or collection"),
        ("no_runner_consumption", "no execution runner consumption implemented"),
        ("no_real_economics_or_execution_metrics", "no real fees rebates spread capture PnL or execution metrics"),
        ("context_sources_alone_rejected", "fills inventory replay lifecycle public markout and hypothetical spread alone are rejected"),
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
    pass_cases = {"valid_maker_fee_settlement", "valid_maker_rebate_settlement"}

    for case_id, rows in cases.items():
        fixture_path = fixture_dir / f"{case_id}.csv"
        extra_fields = sorted(set().union(*(row.keys() for row in rows)) - set(ALL_FIELDS) - {"fixture_case_id"})
        fields = ALL_FIELDS + ["fixture_case_id"] + extra_fields
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
                "covered_validation_area": "|".join(reasons) if reasons else "valid_local_economics_fee_rebate_artifact",
                "forbidden_current_interpretation": "endpoint_runner_metric_strategy_pnl_or_economics_proof",
            }
        )
        detail_rows.extend(_issue_rows(result, case_id))

    _write_csv(
        output_dir / "schema_columns.csv",
        schema_rows(),
        ["field_name", "required", "field_type", "source_contract", "current_use", "forbidden_current_interpretation"],
    )
    _write_csv(
        output_dir / "fixture_case_catalog.csv",
        case_rows,
        ["fixture_case_id", "purpose", "expected_status", "covered_validation_area", "forbidden_current_interpretation"],
    )
    _write_csv(
        output_dir / "validator_result_summary.csv",
        summary_rows,
        ["fixture_case_id", "fixture_path", "row_count", "expected_status", "actual_status", "issue_count", "reason_codes"],
    )
    _write_csv(
        output_dir / "validator_result_details.csv",
        detail_rows,
        ["fixture_case_id", "settlement_record_id", "row_index", "reason_code", "detail"],
    )
    _write_csv(
        output_dir / "arithmetic_validation_policy.csv",
        arithmetic_policy_rows(),
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
        "account_inventory_context_task_id": ACCOUNT_INVENTORY_CONTEXT_TASK_ID,
        "account_inventory_context_final_recommendation": ACCOUNT_INVENTORY_CONTEXT_FINAL_RECOMMENDATION,
        "source_line_id": SOURCE_LINE_ID,
        "generated_at": FIXED_GENERATED_AT,
        "final_recommendation": "economics_fee_rebate_artifact_skeleton_ready_for_qa"
        if all_expected
        else "economics_fee_rebate_artifact_skeleton_needs_revision",
        "fixture_case_count": len(summary_rows),
        "fixture_pass_count": sum(1 for row in summary_rows if row["actual_status"] == "pass"),
        "fixture_fail_closed_count": sum(1 for row in summary_rows if row["actual_status"] == "fail_closed"),
        "all_expected_statuses_matched": all_expected,
        "boundary_flags": {
            "local_economics_fee_rebate_artifact_skeleton_only": True,
            "disk_only_csv_json_parser": True,
            "no_endpoint_implementation": True,
            "no_credentials_signing_nonce_user_stream": True,
            "no_private_order_account_live_economics_data_read": True,
            "no_remote_execution_or_collection": True,
            "no_runner_consumption": True,
            "no_real_economics_metrics": True,
            "no_real_execution_metrics": True,
            "no_pnl_proof": True,
            "context_sources_alone_cannot_prove_fees_rebates_spread_capture_or_pnl": True,
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
            "arithmetic_validation_policy": str(output_dir / "arithmetic_validation_policy.csv"),
            "reconciliation_boundary_policy": str(output_dir / "reconciliation_boundary_policy.csv"),
            "boundary_validation": str(output_dir / "boundary_validation.csv"),
        },
    }
    _write_json(output_dir / "economics_fee_rebate_validation_manifest.json", manifest)
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
        description=(
            "Validate local economics fee/rebate artifacts without endpoints, credentials, runners, "
            "real economics metrics, PnL proof, or live behavior."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    validate = subparsers.add_parser("validate", help="validate one local CSV/JSON economics fee/rebate artifact")
    validate.add_argument("--input", required=True, help="local CSV or JSON artifact path")
    validate.add_argument("--summary-out", help="optional JSON summary path")
    validate.set_defaults(func=validate_command)

    generate = subparsers.add_parser("generate-artifacts", help="write 0612T001 synthetic fixtures and validation artifacts")
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
