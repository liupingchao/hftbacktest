#!/usr/bin/env python3
"""Local Hyperliquid private-order artifact validator.

This module is deliberately no-trading and fixture-only. It validates local
Hyperliquid private order artifact rows against the 0616T002 readiness boundary.
It does not call endpoints, read credentials, sign requests, manage nonces,
subscribe to user streams, query accounts, place/cancel orders, run live, or
authorize strategy behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0616T003"
BOUNDARY_TASK_ID = "0616T002"
SCHEMA_VERSION = "hyperliquid_private_order_artifact_validator_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_private_order_artifact_validator_0616T003"
FINAL_RECOMMENDATION = "hyperliquid_private_order_artifact_validator_ready_for_qa"

REQUIRED_FIELDS = [
    "task_id",
    "source_task_id",
    "venue",
    "instrument",
    "artifact_event_id",
    "event_sequence_index",
    "opaque_client_order_ref",
    "opaque_exchange_order_ref",
    "local_request_time",
    "exchange_ack_time",
    "local_response_time",
    "private_stream_receive_time",
    "validation_time",
    "intent_label",
    "post_only_semantics_label",
    "lifecycle_state",
    "terminal_state_marker",
    "validation_status",
    "overclaim_rejection_id",
    "allowed_future_use",
    "forbidden_current_interpretation",
]
OPTIONAL_FIELDS = [
    "fill_qty",
    "remaining_qty",
    "reject_code",
    "reject_reason",
    "fail_closed_reason",
]
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

ALLOWED_INTENTS = {
    "post_only_limit_intent_design_label",
    "cancel_intent_design_label",
    "status_reconciliation_design_label",
    "unknown_intent_fail_closed",
}
ALLOWED_POST_ONLY = {
    "post_only_intent_declared",
    "post_only_acceptance_observed",
    "post_only_reject_observed",
    "maker_fill_observed",
    "taker_fill_or_cross_detected_fail_closed",
    "post_only_semantics_unavailable_fail_closed",
}
ALLOWED_LIFECYCLE = {
    "submitted_design_label",
    "accepted_ack_design_label",
    "rejected_terminal_design_label",
    "cancel_requested_design_label",
    "canceled_terminal_design_label",
    "partially_filled_active_design_label",
    "filled_terminal_design_label",
    "unknown_state_fail_closed",
    "conflicting_state_fail_closed",
}
ALLOWED_TERMINAL = {"terminal", "non_terminal", "unknown", "conflicting"}
ALLOWED_STATUS = {"accepted_design_label", "fail_closed_design_label"}
TERMINAL_STATES = {
    "rejected_terminal_design_label",
    "canceled_terminal_design_label",
    "filled_terminal_design_label",
}
FORBIDDEN_FIELDS = {
    "endpoint_url",
    "api_key",
    "secret",
    "signature",
    "nonce",
    "signed_payload",
    "user_stream_endpoint",
    "place_order",
    "cancel_order",
    "account_query",
    "order_side",
    "quote_price",
    "quote_size",
    "strategy_signal",
    "live_gate",
    "deployment_flag",
    "promotion_flag",
}
BOUNDARY_FLAGS = {
    "local_fixture_only": True,
    "no_private_endpoint_calls": True,
    "no_credentials": True,
    "no_signing": True,
    "no_nonce": True,
    "no_user_stream": True,
    "no_account_query": True,
    "no_order_placement": True,
    "no_order_cancellation": True,
    "no_strategy_implementation": True,
    "no_live_process": True,
    "no_default_on": True,
    "no_tiny_live": True,
    "no_deployment": True,
    "no_promotion": True,
    "no_pnl_proof": True,
}


@dataclass(frozen=True)
class ValidationIssue:
    artifact_event_id: str
    row_index: int
    reason_code: str
    detail: str


@dataclass(frozen=True)
class ValidationResult:
    rows: List[Dict[str, str]]
    issues: List[ValidationIssue]

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail_closed"


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


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_time(value):
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _issue(row, row_index, reason, detail):
    return ValidationIssue(row.get("artifact_event_id", f"row_{row_index}"), row_index, reason, detail)


def _check_required(row, row_index):
    return [
        _issue(row, row_index, "missing_required_field", field)
        for field in REQUIRED_FIELDS
        if not str(row.get(field, "")).strip()
    ]


def _check_forbidden(row, row_index):
    return [
        _issue(row, row_index, "forbidden_field_present", field)
        for field in FORBIDDEN_FIELDS
        if str(row.get(field, "")).strip()
    ]


def _check_enums(row, row_index):
    checks = [
        ("task_id", {TASK_ID}),
        ("source_task_id", {BOUNDARY_TASK_ID}),
        ("venue", {"hyperliquid"}),
        ("intent_label", ALLOWED_INTENTS),
        ("post_only_semantics_label", ALLOWED_POST_ONLY),
        ("lifecycle_state", ALLOWED_LIFECYCLE),
        ("terminal_state_marker", ALLOWED_TERMINAL),
        ("validation_status", ALLOWED_STATUS),
    ]
    issues = []
    for field, allowed in checks:
        value = str(row.get(field, "")).strip()
        if value and value not in allowed:
            issues.append(_issue(row, row_index, "unknown_enum_value", f"{field}={value}"))
    return issues


def _check_timestamps(row, row_index):
    fields = [
        "local_request_time",
        "exchange_ack_time",
        "local_response_time",
        "private_stream_receive_time",
        "validation_time",
    ]
    parsed = []
    issues = []
    for field in fields:
        stamp = _parse_time(row.get(field, ""))
        if stamp is None:
            issues.append(_issue(row, row_index, "missing_or_invalid_timestamp", field))
        else:
            parsed.append(stamp)
    if len(parsed) == len(fields) and any(parsed[i] > parsed[i + 1] for i in range(len(parsed) - 1)):
        issues.append(_issue(row, row_index, "timestamp_order_invalid", "<=".join(fields)))
    return issues


def _check_terminal(row, row_index):
    state = row.get("lifecycle_state", "")
    marker = row.get("terminal_state_marker", "")
    status = row.get("validation_status", "")
    fail_reason = row.get("fail_closed_reason", "")
    issues = []
    if state in TERMINAL_STATES and marker != "terminal":
        issues.append(_issue(row, row_index, "terminal_consistency_invalid", state))
    if state not in TERMINAL_STATES and marker == "terminal":
        issues.append(_issue(row, row_index, "terminal_consistency_invalid", state))
    if status == "fail_closed_design_label" and not fail_reason:
        issues.append(_issue(row, row_index, "missing_fail_closed_reason", "fail_closed_reason"))
    if status == "accepted_design_label" and fail_reason:
        issues.append(_issue(row, row_index, "accepted_row_has_fail_closed_reason", fail_reason))
    return issues


def validate_rows(rows):
    issues = []
    seen = set()
    for row_index, row in enumerate(rows, start=1):
        event_id = row.get("artifact_event_id", "")
        if event_id in seen:
            issues.append(_issue(row, row_index, "duplicate_event_identity", event_id))
        seen.add(event_id)
        issues.extend(_check_required(row, row_index))
        issues.extend(_check_forbidden(row, row_index))
        issues.extend(_check_enums(row, row_index))
        issues.extend(_check_timestamps(row, row_index))
        issues.extend(_check_terminal(row, row_index))
    return ValidationResult(rows, issues)


def validate_artifact(path):
    return validate_rows(_read_csv(path))


def _base_row(case_id, seq, state, terminal, post_only, status="accepted_design_label"):
    base_time = f"2026-06-16T0{seq}:00:"
    return {
        "task_id": TASK_ID,
        "source_task_id": BOUNDARY_TASK_ID,
        "venue": "hyperliquid",
        "instrument": "BTC",
        "artifact_event_id": f"{case_id}_{seq}",
        "event_sequence_index": str(seq),
        "opaque_client_order_ref": f"cloid_sha256_fixture_{seq}",
        "opaque_exchange_order_ref": f"oid_sha256_fixture_{seq}",
        "local_request_time": base_time + "00Z",
        "exchange_ack_time": base_time + "01Z",
        "local_response_time": base_time + "02Z",
        "private_stream_receive_time": base_time + "03Z",
        "validation_time": base_time + "04Z",
        "intent_label": "post_only_limit_intent_design_label",
        "post_only_semantics_label": post_only,
        "lifecycle_state": state,
        "terminal_state_marker": terminal,
        "validation_status": status,
        "overclaim_rejection_id": "reject_fixture_as_real_execution_proof",
        "allowed_future_use": "local_fixture_validation_only",
        "forbidden_current_interpretation": "endpoint_order_strategy_live_metric_pnl_or_promotion_proof",
        "fill_qty": "",
        "remaining_qty": "",
        "reject_code": "",
        "reject_reason": "",
        "fail_closed_reason": "",
    }


def fixture_rows():
    accepted_ack = _base_row(
        "accepted_post_only_ack",
        seq=1,
        state="accepted_ack_design_label",
        terminal="non_terminal",
        post_only="post_only_acceptance_observed",
    )
    post_only_reject = _base_row(
        "post_only_reject",
        seq=2,
        state="rejected_terminal_design_label",
        terminal="terminal",
        post_only="post_only_reject_observed",
    )
    post_only_reject["reject_code"] = "post_only_would_cross_design_label"
    cancel_accepted = _base_row(
        "cancel_accepted",
        seq=3,
        state="canceled_terminal_design_label",
        terminal="terminal",
        post_only="post_only_acceptance_observed",
    )
    cancel_accepted["intent_label"] = "cancel_intent_design_label"
    partial_fill = _base_row(
        "partial_fill_active",
        seq=4,
        state="partially_filled_active_design_label",
        terminal="non_terminal",
        post_only="maker_fill_observed",
    )
    partial_fill["fill_qty"] = "0.001"
    partial_fill["remaining_qty"] = "0.002"
    filled = _base_row(
        "filled_terminal",
        seq=5,
        state="filled_terminal_design_label",
        terminal="terminal",
        post_only="maker_fill_observed",
    )
    filled["fill_qty"] = "0.003"
    filled["remaining_qty"] = "0"

    missing_timestamp = dict(accepted_ack)
    missing_timestamp["artifact_event_id"] = "missing_timestamp_6"
    missing_timestamp["exchange_ack_time"] = ""
    missing_timestamp["validation_status"] = "fail_closed_design_label"
    missing_timestamp["fail_closed_reason"] = "missing_or_invalid_timestamp"

    conflicting_terminal = dict(accepted_ack)
    conflicting_terminal["artifact_event_id"] = "conflicting_terminal_7"
    conflicting_terminal["terminal_state_marker"] = "terminal"
    conflicting_terminal["validation_status"] = "fail_closed_design_label"
    conflicting_terminal["fail_closed_reason"] = "terminal_consistency_invalid"

    forbidden_field = dict(accepted_ack)
    forbidden_field["artifact_event_id"] = "forbidden_field_8"
    forbidden_field["endpoint_url"] = "https://forbidden.invalid"
    forbidden_field["validation_status"] = "fail_closed_design_label"
    forbidden_field["fail_closed_reason"] = "forbidden_field_present"

    duplicate_a = dict(accepted_ack)
    duplicate_b = dict(accepted_ack)
    duplicate_a["artifact_event_id"] = "duplicate_event_9"
    duplicate_b["artifact_event_id"] = "duplicate_event_9"
    duplicate_a["validation_status"] = "fail_closed_design_label"
    duplicate_b["validation_status"] = "fail_closed_design_label"
    duplicate_a["fail_closed_reason"] = "duplicate_event_identity"
    duplicate_b["fail_closed_reason"] = "duplicate_event_identity"

    return {
        "accepted": [accepted_ack, post_only_reject, cancel_accepted, partial_fill, filled],
        "fail_closed": [missing_timestamp, conflicting_terminal, forbidden_field, duplicate_a, duplicate_b],
    }


def _issue_rows(result):
    return [
        {
            "artifact_event_id": issue.artifact_event_id,
            "row_index": str(issue.row_index),
            "reason_code": issue.reason_code,
            "detail": issue.detail,
        }
        for issue in result.issues
    ]


def _boundary_rows():
    return [{"check": key, "status": "pass" if value else "fail"} for key, value in BOUNDARY_FLAGS.items()]


def generate_artifacts(output_dir=DEFAULT_OUTPUT_DIR):
    output_dir = output_dir.resolve()
    fixtures = fixture_rows()
    accepted_rows = fixtures["accepted"]
    fail_closed_rows = fixtures["fail_closed"]
    all_rows = accepted_rows + fail_closed_rows
    accepted_result = validate_rows(accepted_rows)
    fail_closed_result = validate_rows(fail_closed_rows)

    _write_csv(output_dir / "fixture_private_order_events.csv", all_rows, ALL_FIELDS + sorted(FORBIDDEN_FIELDS))
    _write_csv(output_dir / "accepted_artifact_rows.csv", accepted_rows, ALL_FIELDS)
    _write_csv(output_dir / "fail_closed_artifact_rows.csv", fail_closed_rows, ALL_FIELDS + sorted(FORBIDDEN_FIELDS))
    summary_rows = [
        {
            "fixture_group": "accepted",
            "row_count": str(len(accepted_rows)),
            "actual_status": accepted_result.status,
            "issue_count": str(len(accepted_result.issues)),
            "expected_status": "pass",
        },
        {
            "fixture_group": "fail_closed",
            "row_count": str(len(fail_closed_rows)),
            "actual_status": fail_closed_result.status,
            "issue_count": str(len(fail_closed_result.issues)),
            "expected_status": "fail_closed",
        },
    ]
    _write_csv(output_dir / "validator_result_summary.csv", summary_rows, list(summary_rows[0]))
    _write_csv(output_dir / "validator_issue_details.csv", _issue_rows(fail_closed_result), ["artifact_event_id", "row_index", "reason_code", "detail"])
    _write_csv(output_dir / "boundary_validation.csv", _boundary_rows(), ["check", "status"])

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_task_id": BOUNDARY_TASK_ID,
        "git_commit": _git_commit(),
        "final_recommendation": FINAL_RECOMMENDATION,
        "accepted_fixture_status": accepted_result.status,
        "fail_closed_fixture_status": fail_closed_result.status,
        "accepted_row_count": len(accepted_rows),
        "fail_closed_row_count": len(fail_closed_rows),
        "fail_closed_issue_count": len(fail_closed_result.issues),
        "boundary_flags": BOUNDARY_FLAGS,
        "artifacts": {
            "fixture_private_order_events": str(output_dir / "fixture_private_order_events.csv"),
            "accepted_artifact_rows": str(output_dir / "accepted_artifact_rows.csv"),
            "fail_closed_artifact_rows": str(output_dir / "fail_closed_artifact_rows.csv"),
            "validator_result_summary": str(output_dir / "validator_result_summary.csv"),
            "validator_issue_details": str(output_dir / "validator_issue_details.csv"),
            "boundary_validation": str(output_dir / "boundary_validation.csv"),
        },
    }
    _write_json(output_dir / "hyperliquid_private_order_validator_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local Hyperliquid private-order fixture artifacts.")
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser("generate-artifacts", help="write task-scoped local validator artifacts")
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    validate = subparsers.add_parser("validate-artifact", help="validate a CSV artifact")
    validate.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "generate-artifacts":
        manifest = generate_artifacts(args.output_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
    elif args.command == "validate-artifact":
        result = validate_artifact(args.path)
        print(json.dumps({"status": result.status, "issue_count": len(result.issues)}, indent=2))
        raise SystemExit(0 if result.passed else 2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
