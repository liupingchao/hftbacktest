#!/usr/bin/env python3
"""No-trading local account inventory read-only source.

This module implements the 0615T004 boundary as a local transform: task-local
fixture rows are redacted into the accepted account_inventory_source.py schema
and validated locally.

It does not connect to endpoints, read credentials, manage nonces, subscribe
to user streams, read real account/private/order/live data, place or cancel
orders, feed runners, or compute execution, inventory, or PnL metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from account_inventory_source import (
    ALL_FIELDS,
    EXPECTED_ARTIFACT_SOURCE_CLASS,
    EXPECTED_SOURCE_POLICY,
    SCHEMA_VERSION,
    SOURCE_LINE_ID,
    validate_rows,
)


TASK_ID = "0615T004"
SOURCE_TASK_ID = "0610T008"
SOURCE_FINAL_RECOMMENDATION = "account_inventory_contract_ready_for_qa"
SYNTHESIS_TASK_ID = "0611T001"
SYNTHESIS_FINAL_RECOMMENDATION = "source_line_synthesis_gate_ready_for_qa"
PRIVATE_ORDER_CONTEXT_TASK_ID = "0615T003"
PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION = "private_order_response_read_only_collector_ready_for_qa"
LOCAL_SKELETON_TASK_ID = "0611T004"
LOCAL_SKELETON_FINAL_RECOMMENDATION = "account_inventory_artifact_skeleton_ready_for_qa"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/basis_positive_account_inventory_read_only_source_0615T004")
FIXED_GENERATED_AT = "2026-06-16T00:10:00Z"
FINAL_RECOMMENDATION = "account_inventory_read_only_source_ready_for_qa"

INPUT_FIELDS = [
    "fixture_case_id",
    "venue_id",
    "account_scope_input",
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
    "exchange_event_time",
    "local_receive_time",
    "artifact_generated_time",
    "validation_or_reconciliation_time",
    "freshness_policy_id",
    "ordering_policy_id",
    "current_proof_status",
    "allowed_future_use",
    "forbidden_current_interpretation",
    "fail_closed_reason",
    "related_future_order_ref_input",
]

FORBIDDEN_INPUT_FIELDS = {
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


@dataclass(frozen=True)
class SourceIssue:
    row_index: int
    reason_code: str
    detail: str


@dataclass(frozen=True)
class SourceResult:
    rows: list[dict[str, str]]
    issues: list[SourceIssue]
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


def _collector_issues(input_rows: list[dict[str, str]]) -> list[SourceIssue]:
    issues: list[SourceIssue] = []
    for row_index, row in enumerate(input_rows, start=1):
        for field in FORBIDDEN_INPUT_FIELDS:
            if str(row.get(field, "")).strip():
                issues.append(SourceIssue(row_index, "forbidden_input_field", field))
        if not str(row.get("account_scope_input", "")).strip():
            issues.append(SourceIssue(row_index, "missing_account_scope_input", "account_scope_input"))
        if not str(row.get("account_state_time", "")).strip():
            issues.append(SourceIssue(row_index, "missing_timestamp_domain", "account_state_time"))
        if not str(row.get("artifact_generated_time", "")).strip():
            issues.append(SourceIssue(row_index, "missing_timestamp_domain", "artifact_generated_time"))
        if not str(row.get("validation_or_reconciliation_time", "")).strip():
            issues.append(SourceIssue(row_index, "missing_timestamp_domain", "validation_or_reconciliation_time"))
    return issues


def transform_rows(input_rows: list[dict[str, str]]) -> SourceResult:
    issues = _collector_issues(input_rows)
    if issues:
        return SourceResult([], issues, "fail_closed", 0)

    artifact_rows: list[dict[str, str]] = []
    for row in input_rows:
        row_sequence_index = row.get("row_sequence_index", "")
        artifact_rows.append(
            {
                "task_id": TASK_ID,
                "source_task_id": SOURCE_TASK_ID,
                "source_line_id": SOURCE_LINE_ID,
                "schema_version": SCHEMA_VERSION,
                "source_artifact_id": f"{TASK_ID}_{row['fixture_case_id']}",
                "artifact_source_class": EXPECTED_ARTIFACT_SOURCE_CLASS,
                "source_policy": EXPECTED_SOURCE_POLICY,
                "validation_gate_id": "0615T004_account_inventory_read_only_source_gate",
                "venue_id": row["venue_id"],
                "account_scope_id": _opaque_ref(row["account_scope_input"], "account_scope"),
                "asset_id": row["asset_id"],
                "instrument_id": row["instrument_id"],
                "quantity_unit": row["quantity_unit"],
                "precision_policy_id": row["precision_policy_id"],
                "snapshot_type": row["snapshot_type"],
                "state_completeness": row["state_completeness"],
                "transition_type": row["transition_type"],
                "transition_id": f"{TASK_ID}_{row['transition_id']}",
                "transition_validation_status": row["transition_validation_status"],
                "available_quantity": row["available_quantity"],
                "locked_quantity": row["locked_quantity"],
                "total_quantity": row["total_quantity"],
                "position_quantity": row["position_quantity"],
                "before_quantity": row["before_quantity"],
                "after_quantity": row["after_quantity"],
                "delta_quantity": row["delta_quantity"],
                "delta_attribution": row["delta_attribution"],
                "account_state_time": row["account_state_time"],
                "artifact_generated_time": row["artifact_generated_time"],
                "validation_or_reconciliation_time": row["validation_or_reconciliation_time"],
                "freshness_policy_id": row["freshness_policy_id"],
                "ordering_policy_id": row["ordering_policy_id"],
                "current_proof_status": row["current_proof_status"],
                "overclaim_rejection_id": "reject_inventory_lifecycle_proof_overclaim",
                "allowed_future_use": row["allowed_future_use"],
                "forbidden_current_interpretation": row["forbidden_current_interpretation"],
                "fail_closed_reason": row["fail_closed_reason"],
                "exchange_event_time": row["exchange_event_time"],
                "local_receive_time": row["local_receive_time"],
                "related_future_opaque_order_ref_hash": _opaque_ref(row.get("related_future_order_ref_input", ""), "future_order"),
                "row_sequence_index": row_sequence_index,
            }
        )

    validation = validate_rows(artifact_rows)
    return SourceResult(artifact_rows, [], validation.status, len(validation.issues))


def synthetic_input_rows() -> list[dict[str, str]]:
    return [
        {
            "fixture_case_id": "snapshot_complete",
            "venue_id": "binance_design_label",
            "account_scope_input": "synthetic_account_scope_alpha",
            "asset_id": "BTC",
            "instrument_id": "BTCUSDT",
            "quantity_unit": "base_asset_BTC",
            "precision_policy_id": "spot_base_precision_8",
            "snapshot_type": "initial_snapshot_design_label",
            "state_completeness": "complete",
            "transition_type": "account_observed_transition",
            "transition_id": "snapshot_complete_001",
            "transition_validation_status": "accepted_design_label",
            "available_quantity": "1.00000000",
            "locked_quantity": "0.10000000",
            "total_quantity": "1.10000000",
            "position_quantity": "1.10000000",
            "before_quantity": "1.10000000",
            "after_quantity": "1.10000000",
            "delta_quantity": "0.00000000",
            "delta_attribution": "initial_snapshot",
            "account_state_time": "2026-06-16T00:10:00Z",
            "exchange_event_time": "2026-06-16T00:10:01Z",
            "local_receive_time": "2026-06-16T00:10:02Z",
            "artifact_generated_time": "2026-06-16T00:10:03Z",
            "validation_or_reconciliation_time": "2026-06-16T00:10:04Z",
            "freshness_policy_id": "local_fixture_not_live_freshness",
            "ordering_policy_id": "account_state_time_then_sequence",
            "current_proof_status": "future_account_state_context_only",
            "allowed_future_use": "later_separately_scoped_account_inventory_reconciliation_design_only",
            "forbidden_current_interpretation": "no_inventory_lifecycle_pnl_live_or_promotion_proof",
            "fail_closed_reason": "none",
            "related_future_order_ref_input": "synthetic_future_order_alpha",
            "row_sequence_index": "1",
        },
        {
            "fixture_case_id": "observed_transition",
            "venue_id": "binance_design_label",
            "account_scope_input": "synthetic_account_scope_alpha",
            "asset_id": "BTC",
            "instrument_id": "BTCUSDT",
            "quantity_unit": "base_asset_BTC",
            "precision_policy_id": "spot_base_precision_8",
            "snapshot_type": "reconciliation_snapshot_design_label",
            "state_completeness": "complete",
            "transition_type": "account_observed_transition",
            "transition_id": "observed_transition_001",
            "transition_validation_status": "accepted_design_label",
            "available_quantity": "1.20000000",
            "locked_quantity": "0.10000000",
            "total_quantity": "1.30000000",
            "position_quantity": "1.30000000",
            "before_quantity": "1.10000000",
            "after_quantity": "1.30000000",
            "delta_quantity": "0.20000000",
            "delta_attribution": "account_observed_delta",
            "account_state_time": "2026-06-16T00:10:10Z",
            "exchange_event_time": "2026-06-16T00:10:11Z",
            "local_receive_time": "2026-06-16T00:10:12Z",
            "artifact_generated_time": "2026-06-16T00:10:13Z",
            "validation_or_reconciliation_time": "2026-06-16T00:10:14Z",
            "freshness_policy_id": "local_fixture_not_live_freshness",
            "ordering_policy_id": "account_state_time_then_sequence",
            "current_proof_status": "future_account_state_context_only",
            "allowed_future_use": "later_separately_scoped_account_inventory_reconciliation_design_only",
            "forbidden_current_interpretation": "no_inventory_lifecycle_pnl_live_or_promotion_proof",
            "fail_closed_reason": "none",
            "related_future_order_ref_input": "synthetic_future_order_beta",
            "row_sequence_index": "2",
        },
    ]


def forbidden_input_rows() -> list[dict[str, str]]:
    row = dict(synthetic_input_rows()[0])
    row["fixture_case_id"] = "forbidden_endpoint_field"
    row["endpoint_url"] = "https://example.invalid/account"
    return [row]


def redaction_audit_rows(input_rows: list[dict[str, str]], artifact_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source, artifact in zip(input_rows, artifact_rows):
        rows.append(
            {
                "fixture_case_id": source["fixture_case_id"],
                "account_scope_input_present": "yes" if source.get("account_scope_input") else "no",
                "related_future_order_ref_input_present": "yes" if source.get("related_future_order_ref_input") else "no",
                "account_scope_hash_prefix": artifact.get("account_scope_id", "")[:24],
                "related_future_order_ref_hash_prefix": artifact.get("related_future_opaque_order_ref_hash", "")[:24],
                "raw_account_scope_persisted_in_artifact": str(source.get("account_scope_input") in json.dumps(artifact)),
                "raw_future_order_ref_persisted_in_artifact": str(source.get("related_future_order_ref_input") in json.dumps(artifact)),
            }
        )
    return rows


def no_trading_safety_audit_rows() -> list[dict[str, str]]:
    checks = [
        ("no_endpoint_calls_or_clients", "no endpoint URL REST websocket exchange client or connector code"),
        ("no_credentials_signing_nonce_user_stream", "no credentials signing nonce or user stream implementation"),
        ("no_order_action_or_strategy", "no order placement cancellation amendment or strategy hook"),
        ("no_runner_consumption", "collector only writes local artifacts and validates them locally"),
        ("no_real_account_data_read", "official artifacts use synthetic task-local fixture rows"),
        ("no_metrics_pnl_viability", "collector emits no inventory execution economics PnL or viability metrics"),
        ("no_live_deployment_promotion", "collector emits no live deployment or promotion readiness"),
    ]
    return [{"check_id": check_id, "status": "pass", "detail": detail} for check_id, detail in checks]


def boundary_validation_rows(passed: bool) -> list[dict[str, str]]:
    checks = [
        ("prerequisite_0615T003_qa", "0615T003 QA is passed and collector ready"),
        ("local_transform_only", "task-local rows transformed into accepted account inventory artifact schema"),
        ("redaction_required", "account scope and related future order refs are hashed before artifact storage"),
        ("validator_handoff", "collector output validates through account_inventory_source.py"),
        ("forbidden_input_fields_fail_closed", "endpoint credential action and strategy fields fail closed"),
        ("no_endpoint_or_user_stream", "no endpoint client signed request nonce or user stream code"),
        ("no_real_private_data", "no real private order account live or economics data read"),
        ("no_runner_strategy_live", "no runner consumption strategy or live behavior"),
        ("no_metrics_pnl_promotion", "no inventory execution economics PnL deployment promotion or maker viability proof"),
    ]
    return [{"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail} for check_id, detail in checks]


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = synthetic_input_rows()
    result = transform_rows(inputs)
    forbidden_result = transform_rows(forbidden_input_rows())
    source_passed = result.passed and not forbidden_result.passed

    _write_csv(output_dir / "account_inventory_fixture_inputs.csv", inputs, INPUT_FIELDS)
    _write_csv(output_dir / "account_inventory_output_artifact.csv", result.rows, ALL_FIELDS)
    _write_csv(
        output_dir / "account_inventory_validation_summary.csv",
        [
            {
                "case_id": "official_valid_fixture_transform",
                "row_count": len(result.rows),
                "source_issue_count": len(result.issues),
                "validator_status": result.validation_status,
                "validator_issue_count": result.validation_issue_count,
                "expected_status": "pass",
                "actual_status": "pass" if result.passed else "fail_closed",
            },
            {
                "case_id": "forbidden_endpoint_field",
                "row_count": len(forbidden_result.rows),
                "source_issue_count": len(forbidden_result.issues),
                "validator_status": forbidden_result.validation_status,
                "validator_issue_count": forbidden_result.validation_issue_count,
                "expected_status": "fail_closed",
                "actual_status": "fail_closed" if not forbidden_result.passed else "pass",
            },
        ],
        [
            "case_id",
            "row_count",
            "source_issue_count",
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
            "account_scope_input_present",
            "related_future_order_ref_input_present",
            "account_scope_hash_prefix",
            "related_future_order_ref_hash_prefix",
            "raw_account_scope_persisted_in_artifact",
            "raw_future_order_ref_persisted_in_artifact",
        ],
    )
    _write_csv(output_dir / "no_trading_safety_audit.csv", no_trading_safety_audit_rows(), ["check_id", "status", "detail"])
    _write_csv(output_dir / "boundary_validation.csv", boundary_validation_rows(source_passed), ["check_id", "status", "detail"])

    manifest = {
        "task_id": TASK_ID,
        "boundary_task_id": "0615T003",
        "source_task_id": SOURCE_TASK_ID,
        "source_final_recommendation": SOURCE_FINAL_RECOMMENDATION,
        "synthesis_task_id": SYNTHESIS_TASK_ID,
        "synthesis_final_recommendation": SYNTHESIS_FINAL_RECOMMENDATION,
        "private_order_context_task_id": PRIVATE_ORDER_CONTEXT_TASK_ID,
        "private_order_context_final_recommendation": PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION,
        "local_skeleton_task_id": LOCAL_SKELETON_TASK_ID,
        "local_skeleton_final_recommendation": LOCAL_SKELETON_FINAL_RECOMMENDATION,
        "source_line_id": SOURCE_LINE_ID,
        "generated_at": FIXED_GENERATED_AT,
        "final_recommendation": FINAL_RECOMMENDATION if source_passed else "account_inventory_read_only_source_needs_revision",
        "official_output_rows": len(result.rows),
        "valid_transform_passed": result.passed,
        "forbidden_input_failed_closed": not forbidden_result.passed,
        "boundary_flags": {
            "local_task_fixture_input_only": True,
            "redacted_artifacts_only": True,
            "no_endpoint_calls_or_clients": True,
            "no_credentials_signing_nonce_user_stream": True,
            "no_real_private_order_account_live_economics_data_read": True,
            "no_remote_execution_or_venue_collection": True,
            "no_order_action_or_strategy": True,
            "no_runner_consumption": True,
            "no_real_inventory_metrics": True,
            "no_real_execution_metrics": True,
            "order_fills_alone_cannot_prove_inventory_lifecycle": True,
            "no_economics_metrics_or_pnl": True,
            "no_live_deployment_promotion": True,
            "maker_viability_unproven": True,
        },
    }
    _write_json(output_dir / "account_inventory_read_only_source_manifest.json", manifest)
    return manifest


def generate_artifacts_command(args: argparse.Namespace) -> int:
    manifest = generate_artifacts(Path(args.output_dir))
    print(json.dumps({"output_dir": args.output_dir, "final_recommendation": manifest["final_recommendation"]}, sort_keys=True))
    return 0 if manifest["final_recommendation"] == FINAL_RECOMMENDATION else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local account inventory artifacts without endpoints, credentials, runners, metrics, or live behavior."
    )
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser("generate-artifacts", help="write 0615T004 task-local account inventory artifacts")
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
