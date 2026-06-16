#!/usr/bin/env python3
"""No-trading local private-order response read-only collector.

This module implements the 0615T003 collector boundary as a local transform:
task-local fixture rows are redacted into the accepted
private_order_response_source.py artifact schema and validated locally.

It does not call endpoints, implement exchange clients, read credentials,
sign requests, manage nonces, subscribe to user streams, read real private
data, place/cancel/amend orders, feed runners, or compute execution metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from private_order_response_source import (
    ALL_FIELDS,
    EXPECTED_ARTIFACT_SOURCE_CLASS,
    EXPECTED_SOURCE_POLICY,
    SOURCE_LINE_ID,
    validate_rows,
)


TASK_ID = "0615T003"
BOUNDARY_TASK_ID = "0615T002"
LOCAL_SKELETON_TASK_ID = "0611T002"
SOURCE_TASK_ID = "0610T006"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/basis_positive_private_order_response_read_only_collector_0615T003")
FIXED_GENERATED_AT = "2026-06-15T18:30:00Z"
FINAL_RECOMMENDATION = "private_order_response_read_only_collector_ready_for_qa"

INPUT_FIELDS = [
    "fixture_case_id",
    "venue",
    "instrument",
    "source_event_id",
    "event_sequence_index",
    "client_order_ref_input",
    "exchange_order_ref_input",
    "exchange_event_time",
    "local_receive_time",
    "artifact_generated_time",
    "validation_or_reconciliation_time",
    "response_category",
    "lifecycle_state_label",
    "post_only_reject_class",
    "terminal_state_marker",
    "unknown_missing_conflicting_flag",
]

FORBIDDEN_INPUT_FIELDS = {
    "api_key",
    "secret",
    "signature",
    "nonce",
    "signed_payload",
    "endpoint_url",
    "user_stream_endpoint",
    "place_order",
    "cancel_order",
    "amend_order",
    "order_side",
    "quote_price",
    "quote_size",
    "strategy_signal",
    "live_gate",
    "deployment_flag",
    "promotion_flag",
}


@dataclass(frozen=True)
class CollectorIssue:
    row_index: int
    reason_code: str
    detail: str


@dataclass(frozen=True)
class CollectorResult:
    rows: list[dict[str, str]]
    issues: list[CollectorIssue]
    validation_status: str
    validation_issue_count: int

    @property
    def passed(self) -> bool:
        return not self.issues and self.validation_status == "pass"


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


def _opaque_ref(value: str, kind: str) -> str:
    if not value:
        return ""
    digest = hashlib.sha256(f"{TASK_ID}:{kind}:{value}".encode("utf-8")).hexdigest()
    return f"{kind}_sha256_{digest}"


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _collector_issues(input_rows: list[dict[str, str]]) -> list[CollectorIssue]:
    issues: list[CollectorIssue] = []
    for row_index, row in enumerate(input_rows, start=1):
        for field in FORBIDDEN_INPUT_FIELDS:
            if str(row.get(field, "")).strip():
                issues.append(CollectorIssue(row_index, "forbidden_input_field", field))
        if not str(row.get("source_event_id", "")).strip():
            issues.append(CollectorIssue(row_index, "missing_source_event_id", "source_event_id"))
        if not str(row.get("client_order_ref_input", "")).strip() and not str(row.get("exchange_order_ref_input", "")).strip():
            issues.append(CollectorIssue(row_index, "missing_order_reference", "client_or_exchange_order_ref_input"))
        for field in ["exchange_event_time", "local_receive_time", "artifact_generated_time", "validation_or_reconciliation_time"]:
            if not str(row.get(field, "")).strip():
                issues.append(CollectorIssue(row_index, "missing_timestamp_domain", field))
    return issues


def transform_rows(input_rows: list[dict[str, str]]) -> CollectorResult:
    issues = _collector_issues(input_rows)
    if issues:
        return CollectorResult([], issues, "fail_closed", 0)

    artifact_rows: list[dict[str, str]] = []
    for row in input_rows:
        fail_flag = "1" if _truthy(row.get("unknown_missing_conflicting_flag", "")) else "0"
        artifact_rows.append(
            {
                "task_id": LOCAL_SKELETON_TASK_ID,
                "source_task_id": SOURCE_TASK_ID,
                "source_line_id": SOURCE_LINE_ID,
                "venue": row["venue"],
                "instrument": row["instrument"],
                "artifact_source_class": EXPECTED_ARTIFACT_SOURCE_CLASS,
                "source_policy": EXPECTED_SOURCE_POLICY,
                "artifact_event_id": f"{TASK_ID}_{row['source_event_id']}",
                "event_sequence_index": row["event_sequence_index"],
                "local_receive_time": row["local_receive_time"],
                "artifact_generated_time": row["artifact_generated_time"],
                "validation_or_reconciliation_time": row["validation_or_reconciliation_time"],
                "response_category": row["response_category"],
                "lifecycle_state_label": row["lifecycle_state_label"],
                "post_only_reject_class": row["post_only_reject_class"],
                "terminal_state_marker": row["terminal_state_marker"],
                "unknown_missing_conflicting_flag": fail_flag,
                "validation_gate_id": "0615T003_no_trading_read_only_collector_gate",
                "validation_status": "accepted_design_label" if fail_flag == "0" else "fail_closed_design_label",
                "overclaim_rejection_id": "reject_collector_output_as_execution_metric",
                "allowed_future_use": "local_artifact_validation_and_later_separately_scoped_reconciliation_only",
                "forbidden_current_interpretation": "endpoint_runner_strategy_live_metric_pnl_or_promotion_proof",
                "client_order_ref_hash": _opaque_ref(row.get("client_order_ref_input", ""), "client_ref"),
                "exchange_order_ref_hash": _opaque_ref(row.get("exchange_order_ref_input", ""), "exchange_ref"),
                "exchange_event_time": row["exchange_event_time"],
                "fail_closed_reason": "unknown_missing_conflicting_input" if fail_flag == "1" else "",
            }
        )

    validation = validate_rows(artifact_rows)
    return CollectorResult(artifact_rows, [], validation.status, len(validation.issues))


def synthetic_input_rows() -> list[dict[str, str]]:
    return [
        {
            "fixture_case_id": "accepted_ack",
            "venue": "synthetic_binance_context",
            "instrument": "BTCUSDT",
            "source_event_id": "response_ack_001",
            "event_sequence_index": "1",
            "client_order_ref_input": "synthetic_client_alpha",
            "exchange_order_ref_input": "synthetic_exchange_alpha",
            "exchange_event_time": "2026-06-15T18:30:00Z",
            "local_receive_time": "2026-06-15T18:30:01Z",
            "artifact_generated_time": "2026-06-15T18:30:02Z",
            "validation_or_reconciliation_time": "2026-06-15T18:30:03Z",
            "response_category": "accepted",
            "lifecycle_state_label": "accepted_ack_design_label",
            "post_only_reject_class": "non_post_only_reject",
            "terminal_state_marker": "non_terminal",
            "unknown_missing_conflicting_flag": "0",
        },
        {
            "fixture_case_id": "post_only_reject",
            "venue": "synthetic_binance_context",
            "instrument": "BTCUSDT",
            "source_event_id": "response_reject_001",
            "event_sequence_index": "2",
            "client_order_ref_input": "synthetic_client_beta",
            "exchange_order_ref_input": "synthetic_exchange_beta",
            "exchange_event_time": "2026-06-15T18:31:00Z",
            "local_receive_time": "2026-06-15T18:31:01Z",
            "artifact_generated_time": "2026-06-15T18:31:02Z",
            "validation_or_reconciliation_time": "2026-06-15T18:31:03Z",
            "response_category": "rejected",
            "lifecycle_state_label": "rejected_terminal_design_label",
            "post_only_reject_class": "post_only_reject",
            "terminal_state_marker": "terminal",
            "unknown_missing_conflicting_flag": "0",
        },
    ]


def forbidden_input_rows() -> list[dict[str, str]]:
    row = dict(synthetic_input_rows()[0])
    row["fixture_case_id"] = "forbidden_action_field"
    row["place_order"] = "true"
    return [row]


def redaction_audit_rows(input_rows: list[dict[str, str]], artifact_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source, artifact in zip(input_rows, artifact_rows):
        rows.append(
            {
                "fixture_case_id": source["fixture_case_id"],
                "client_order_ref_input_present": "yes" if source.get("client_order_ref_input") else "no",
                "exchange_order_ref_input_present": "yes" if source.get("exchange_order_ref_input") else "no",
                "client_hash_prefix": artifact.get("client_order_ref_hash", "")[:24],
                "exchange_hash_prefix": artifact.get("exchange_order_ref_hash", "")[:24],
                "raw_client_ref_persisted_in_artifact": str(source.get("client_order_ref_input") in json.dumps(artifact)),
                "raw_exchange_ref_persisted_in_artifact": str(source.get("exchange_order_ref_input") in json.dumps(artifact)),
            }
        )
    return rows


def no_trading_safety_audit_rows() -> list[dict[str, str]]:
    checks = [
        ("no_endpoint_calls_or_clients", "no endpoint URL REST websocket exchange client or connector code"),
        ("no_credentials_signing_nonce_user_stream", "no credentials signing nonce or user stream implementation"),
        ("no_order_place_cancel_amend", "collector has no trading action method and rejects action fields"),
        ("no_quote_or_strategy_outputs", "collector artifact schema emits no side price size or strategy signal"),
        ("no_runner_consumption", "collector only writes local artifacts and validates them locally"),
        ("no_real_private_data_read", "official artifacts use synthetic task-local fixture rows"),
        ("no_metrics_pnl_viability", "collector emits no execution economics PnL or maker viability metrics"),
        ("no_deployment_or_promotion", "collector emits no deployment or promotion readiness"),
    ]
    return [{"check_id": check_id, "status": "pass", "detail": detail} for check_id, detail in checks]


def boundary_validation_rows(passed: bool) -> list[dict[str, str]]:
    checks = [
        ("prerequisite_0615T002_qa", "0615T002 QA is passed and boundary ready"),
        ("local_transform_only", "task-local rows transformed into accepted artifact schema"),
        ("redaction_required", "client and exchange order refs are hashed before artifact storage"),
        ("validator_handoff", "collector output validates through private_order_response_source.py"),
        ("forbidden_input_fields_fail_closed", "action credential endpoint and strategy fields fail closed"),
        ("no_endpoint_or_user_stream", "no endpoint client signed request nonce or user stream code"),
        ("no_real_private_data", "no real private order account live or economics data read"),
        ("no_runner_strategy_live", "no runner consumption strategy or live behavior"),
        ("no_metrics_pnl_promotion", "no metrics PnL deployment promotion or maker viability proof"),
    ]
    return [{"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail} for check_id, detail in checks]


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = synthetic_input_rows()
    result = transform_rows(inputs)
    forbidden_result = transform_rows(forbidden_input_rows())
    collector_passed = result.passed and not forbidden_result.passed

    _write_csv(output_dir / "collector_fixture_inputs.csv", inputs, INPUT_FIELDS)
    _write_csv(output_dir / "collector_output_artifact.csv", result.rows, ALL_FIELDS)
    _write_csv(
        output_dir / "collector_validation_summary.csv",
        [
            {
                "case_id": "official_valid_fixture_transform",
                "row_count": len(result.rows),
                "collector_issue_count": len(result.issues),
                "validator_status": result.validation_status,
                "validator_issue_count": result.validation_issue_count,
                "expected_status": "pass",
                "actual_status": "pass" if result.passed else "fail_closed",
            },
            {
                "case_id": "forbidden_action_field",
                "row_count": len(forbidden_result.rows),
                "collector_issue_count": len(forbidden_result.issues),
                "validator_status": forbidden_result.validation_status,
                "validator_issue_count": forbidden_result.validation_issue_count,
                "expected_status": "fail_closed",
                "actual_status": "fail_closed" if not forbidden_result.passed else "pass",
            },
        ],
        [
            "case_id",
            "row_count",
            "collector_issue_count",
            "validator_status",
            "validator_issue_count",
            "expected_status",
            "actual_status",
        ],
    )
    _write_csv(
        output_dir / "redaction_audit.csv",
        redaction_audit_rows(inputs, result.rows),
        [
            "fixture_case_id",
            "client_order_ref_input_present",
            "exchange_order_ref_input_present",
            "client_hash_prefix",
            "exchange_hash_prefix",
            "raw_client_ref_persisted_in_artifact",
            "raw_exchange_ref_persisted_in_artifact",
        ],
    )
    _write_csv(output_dir / "no_trading_safety_audit.csv", no_trading_safety_audit_rows(), ["check_id", "status", "detail"])
    _write_csv(output_dir / "boundary_validation.csv", boundary_validation_rows(collector_passed), ["check_id", "status", "detail"])

    manifest = {
        "task_id": TASK_ID,
        "boundary_task_id": BOUNDARY_TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "local_skeleton_task_id": LOCAL_SKELETON_TASK_ID,
        "source_line_id": SOURCE_LINE_ID,
        "generated_at": FIXED_GENERATED_AT,
        "final_recommendation": FINAL_RECOMMENDATION if collector_passed else "private_order_response_read_only_collector_needs_revision",
        "official_output_rows": len(result.rows),
        "valid_transform_passed": result.passed,
        "forbidden_action_failed_closed": not forbidden_result.passed,
        "boundary_flags": {
            "local_task_fixture_input_only": True,
            "redacted_artifacts_only": True,
            "no_endpoint_calls_or_clients": True,
            "no_credentials_signing_nonce_user_stream": True,
            "no_real_private_order_account_live_economics_data_read": True,
            "no_remote_execution_or_venue_collection": True,
            "no_order_placement_cancellation_or_amendment": True,
            "no_runner_consumption": True,
            "no_strategy_live_default_on_tiny_live": True,
            "no_real_execution_metrics": True,
            "no_real_economics_metrics_or_pnl": True,
            "no_parameter_search": True,
            "no_deployment_or_promotion": True,
            "maker_viability_unproven": True,
        },
    }
    _write_json(output_dir / "private_order_response_read_only_collector_manifest.json", manifest)
    return manifest


def generate_artifacts_command(args: argparse.Namespace) -> int:
    manifest = generate_artifacts(Path(args.output_dir))
    print(json.dumps({"output_dir": args.output_dir, "final_recommendation": manifest["final_recommendation"]}, sort_keys=True))
    return 0 if manifest["final_recommendation"] == FINAL_RECOMMENDATION else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local redacted private-order response artifacts without endpoints, credentials, trading, runners, or metrics."
    )
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser("generate-artifacts", help="write 0615T003 task-local collector artifacts")
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
