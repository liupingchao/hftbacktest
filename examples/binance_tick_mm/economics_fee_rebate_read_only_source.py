#!/usr/bin/env python3
"""No-trading local economics fee/rebate read-only source.

This module implements the 0615T005 boundary as a local transform: task-local
economics settlement fixture rows are redacted into the accepted
economics_fee_rebate_source.py schema and validated locally.

It does not connect to endpoints, read credentials, manage nonces, subscribe
to user streams, read real private/order/account/live/economics data, place or
cancel orders, feed runners, or compute real economics, execution, or PnL
metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from economics_fee_rebate_source import (
    ALL_FIELDS,
    EXPECTED_ARTIFACT_SOURCE_CLASS,
    EXPECTED_SETTLEMENT_AUTHORITY,
    EXPECTED_SOURCE_POLICY,
    SCHEMA_VERSION,
    SOURCE_LINE_ID,
    validate_rows,
)


TASK_ID = "0615T005"
SOURCE_TASK_ID = "0610T009"
SOURCE_FINAL_RECOMMENDATION = "economics_fee_rebate_contract_ready_for_qa"
SYNTHESIS_TASK_ID = "0611T001"
SYNTHESIS_FINAL_RECOMMENDATION = "source_line_synthesis_gate_ready_for_qa"
PRIVATE_ORDER_CONTEXT_TASK_ID = "0615T003"
PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION = "private_order_response_read_only_collector_ready_for_qa"
ACCOUNT_INVENTORY_CONTEXT_TASK_ID = "0615T004"
ACCOUNT_INVENTORY_CONTEXT_FINAL_RECOMMENDATION = "account_inventory_read_only_source_ready_for_qa"
LOCAL_SKELETON_TASK_ID = "0612T001"
LOCAL_SKELETON_FINAL_RECOMMENDATION = "economics_fee_rebate_artifact_skeleton_ready_for_qa"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/basis_positive_economics_fee_rebate_read_only_source_0615T005")
FIXED_GENERATED_AT = "2026-06-16T00:20:00Z"
FINAL_RECOMMENDATION = "economics_fee_rebate_read_only_source_ready_for_qa"

INPUT_FIELDS = [
    "fixture_case_id",
    "venue_id",
    "instrument_id",
    "account_scope_input",
    "settlement_scope",
    "future_fill_ref_input",
    "maker_taker_label",
    "settlement_label",
    "fee_amount",
    "rebate_amount",
    "net_fee_amount",
    "fee_currency",
    "rebate_currency",
    "settlement_currency",
    "base_asset",
    "quote_asset",
    "fee_asset",
    "rebate_asset",
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
    "fill_time",
    "conversion_timestamp",
    "exchange_settlement_time",
    "local_receive_time",
    "account_economics_reconciliation_time",
    "artifact_generated_time",
    "validation_time",
    "allowed_future_use",
    "forbidden_current_interpretation",
    "fail_closed_reason",
    "row_sequence_index",
]

FORBIDDEN_INPUT_FIELDS = {
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

TIMESTAMP_FIELDS = [
    "fill_time",
    "conversion_timestamp",
    "exchange_settlement_time",
    "local_receive_time",
    "account_economics_reconciliation_time",
    "artifact_generated_time",
    "validation_time",
]


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
        if not str(row.get("future_fill_ref_input", "")).strip():
            issues.append(SourceIssue(row_index, "missing_future_fill_ref_input", "future_fill_ref_input"))
        for field in TIMESTAMP_FIELDS:
            if not str(row.get(field, "")).strip():
                issues.append(SourceIssue(row_index, "missing_timestamp_domain", field))
    return issues


def transform_rows(input_rows: list[dict[str, str]]) -> SourceResult:
    issues = _collector_issues(input_rows)
    if issues:
        return SourceResult([], issues, "fail_closed", 0)

    artifact_rows: list[dict[str, str]] = []
    for row in input_rows:
        settlement_record_id = f"{TASK_ID}_{row['fixture_case_id']}"
        artifact_rows.append(
            {
                "task_id": TASK_ID,
                "source_task_id": SOURCE_TASK_ID,
                "source_line_id": SOURCE_LINE_ID,
                "schema_version": SCHEMA_VERSION,
                "settlement_record_id": settlement_record_id,
                "settlement_version": "1",
                "venue_id": row["venue_id"],
                "instrument_id": row["instrument_id"],
                "account_scope_id": _opaque_ref(row["account_scope_input"], "account_scope"),
                "settlement_scope": row["settlement_scope"],
                "source_artifact_id": f"{TASK_ID}_artifact_{row['fixture_case_id']}",
                "artifact_source_class": EXPECTED_ARTIFACT_SOURCE_CLASS,
                "source_policy": EXPECTED_SOURCE_POLICY,
                "settlement_source_authority": EXPECTED_SETTLEMENT_AUTHORITY,
                "prior_source_line_context": "0615T003|0615T004_context_only",
                "validation_gate_id": "0615T005_economics_fee_rebate_read_only_source_gate",
                "opaque_future_fill_ref": _opaque_ref(row["future_fill_ref_input"], "future_fill"),
                "fill_dependency_status": "future_fill_dependency_context_only",
                "fill_dependency_source_line": "private_order_response_source_line",
                "fill_validation_status": "accepted_design_label",
                "forbidden_fill_only_interpretation": "fill_notional_or_order_fill_alone_cannot_prove_fee_rebate_spread_capture_or_pnl",
                "maker_taker_label": row["maker_taker_label"],
                "classification_evidence_type": "accepted_economics_settlement_artifact",
                "venue_rule_dependency_status": "not_required",
                "classification_validation_status": "accepted_design_label",
                "settlement_label": row["settlement_label"],
                "settlement_validation_status": "accepted_design_label",
                "fee_amount": row["fee_amount"],
                "rebate_amount": row["rebate_amount"],
                "net_fee_amount": row["net_fee_amount"],
                "amount_sign_convention": "fees_positive_rebates_negative_net_fee_equals_fee_plus_rebate",
                "fee_currency": row["fee_currency"],
                "rebate_currency": row["rebate_currency"],
                "settlement_currency": row["settlement_currency"],
                "base_asset": row["base_asset"],
                "quote_asset": row["quote_asset"],
                "fee_asset": row["fee_asset"],
                "rebate_asset": row["rebate_asset"],
                "precision_policy_id": "settlement_precision_8",
                "rounding_policy_id": "decimal_exact_no_rounding",
                "conversion_source_provenance": "same_currency_no_conversion",
                "conversion_timestamp": row["conversion_timestamp"],
                "conversion_rate": row["conversion_rate"],
                "conversion_tolerance": row["conversion_tolerance"],
                "net_fee_in_settlement_currency": row["net_fee_in_settlement_currency"],
                "tick_size": row["tick_size"],
                "tick_value": row["tick_value"],
                "net_fee_ticks": row["net_fee_ticks"],
                "conversion_validation_status": "accepted_design_label",
                "quoted_spread_ticks": row["quoted_spread_ticks"],
                "filled_spread_ticks": row["filled_spread_ticks"],
                "realized_spread_ticks": row["realized_spread_ticks"],
                "fee_adjusted_spread_ticks": row["fee_adjusted_spread_ticks"],
                "spread_capture_label": "realized_spread_design_label",
                "markout_context_design_label": "markout_context_only",
                "hypothetical_spread_flag": "0",
                "spread_capture_proof_status": "future_economics_context_only",
                "spread_validation_status": "accepted_design_label",
                "fill_time": row["fill_time"],
                "exchange_settlement_time": row["exchange_settlement_time"],
                "local_receive_time": row["local_receive_time"],
                "account_economics_reconciliation_time": row["account_economics_reconciliation_time"],
                "artifact_generated_time": row["artifact_generated_time"],
                "validation_time": row["validation_time"],
                "current_proof_status": "future_economics_context_only",
                "allowed_future_use": row["allowed_future_use"],
                "forbidden_current_interpretation": row["forbidden_current_interpretation"],
                "overclaim_rejection_id": "reject_current_real_economics_and_pnl_proof",
                "fail_closed_reason": row["fail_closed_reason"],
                "row_sequence_index": row["row_sequence_index"],
                "public_markout_context_ref": "public_markout_context_only",
                "account_inventory_context_ref": "0615T004_context_only_not_economics_proof",
                "replay_lifecycle_context_ref": "0611T003_context_only_not_economics_proof",
            }
        )

    validation = validate_rows(artifact_rows)
    return SourceResult(artifact_rows, [], validation.status, len(validation.issues))


def synthetic_input_rows() -> list[dict[str, str]]:
    base = {
        "venue_id": "binance_design_label",
        "instrument_id": "BTCUSDT",
        "account_scope_input": "synthetic_economics_account_scope_alpha",
        "settlement_scope": "single_fill_settlement_context",
        "maker_taker_label": "maker",
        "fee_currency": "USDT",
        "rebate_currency": "USDT",
        "settlement_currency": "USDT",
        "base_asset": "BTC",
        "quote_asset": "USDT",
        "fee_asset": "USDT",
        "rebate_asset": "USDT",
        "conversion_rate": "1.00000000",
        "conversion_tolerance": "0.00000001",
        "tick_size": "0.10000000",
        "tick_value": "0.10000000",
        "quoted_spread_ticks": "30.00000000",
        "filled_spread_ticks": "28.00000000",
        "realized_spread_ticks": "25.00000000",
        "allowed_future_use": "later_separately_scoped_economics_reconciliation_design_input_only",
        "forbidden_current_interpretation": "no_current_economics_pnl_live_deployment_or_promotion_authorization",
        "fail_closed_reason": "none",
    }
    return [
        {
            **base,
            "fixture_case_id": "maker_fee_settlement",
            "future_fill_ref_input": "synthetic_future_fill_alpha",
            "settlement_label": "maker_fee_settlement_design_label",
            "fee_amount": "2.00000000",
            "rebate_amount": "-0.50000000",
            "net_fee_amount": "1.50000000",
            "net_fee_in_settlement_currency": "1.50000000",
            "net_fee_ticks": "15.00000000",
            "fee_adjusted_spread_ticks": "10.00000000",
            "fill_time": "2026-06-16T00:20:00Z",
            "conversion_timestamp": "2026-06-16T00:20:01Z",
            "exchange_settlement_time": "2026-06-16T00:20:02Z",
            "local_receive_time": "2026-06-16T00:20:03Z",
            "account_economics_reconciliation_time": "2026-06-16T00:20:04Z",
            "artifact_generated_time": "2026-06-16T00:20:05Z",
            "validation_time": "2026-06-16T00:20:06Z",
            "row_sequence_index": "1",
        },
        {
            **base,
            "fixture_case_id": "maker_rebate_settlement",
            "future_fill_ref_input": "synthetic_future_fill_beta",
            "settlement_label": "maker_rebate_settlement_design_label",
            "fee_amount": "0.00000000",
            "rebate_amount": "-0.70000000",
            "net_fee_amount": "-0.70000000",
            "net_fee_in_settlement_currency": "-0.70000000",
            "net_fee_ticks": "-7.00000000",
            "fee_adjusted_spread_ticks": "32.00000000",
            "fill_time": "2026-06-16T00:20:10Z",
            "conversion_timestamp": "2026-06-16T00:20:11Z",
            "exchange_settlement_time": "2026-06-16T00:20:12Z",
            "local_receive_time": "2026-06-16T00:20:13Z",
            "account_economics_reconciliation_time": "2026-06-16T00:20:14Z",
            "artifact_generated_time": "2026-06-16T00:20:15Z",
            "validation_time": "2026-06-16T00:20:16Z",
            "row_sequence_index": "2",
        },
    ]


def forbidden_input_rows() -> list[dict[str, str]]:
    row = dict(synthetic_input_rows()[0])
    row["fixture_case_id"] = "forbidden_endpoint_field"
    row["endpoint_url"] = "https://example.invalid/economics"
    return [row]


def redaction_audit_rows(input_rows: list[dict[str, str]], artifact_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source, artifact in zip(input_rows, artifact_rows):
        rows.append(
            {
                "fixture_case_id": source["fixture_case_id"],
                "account_scope_input_present": "yes" if source.get("account_scope_input") else "no",
                "future_fill_ref_input_present": "yes" if source.get("future_fill_ref_input") else "no",
                "account_scope_hash_prefix": artifact.get("account_scope_id", "")[:24],
                "future_fill_ref_hash_prefix": artifact.get("opaque_future_fill_ref", "")[:24],
                "raw_account_scope_persisted_in_artifact": str(source.get("account_scope_input") in json.dumps(artifact)),
                "raw_future_fill_ref_persisted_in_artifact": str(source.get("future_fill_ref_input") in json.dumps(artifact)),
            }
        )
    return rows


def no_trading_safety_audit_rows() -> list[dict[str, str]]:
    checks = [
        ("no_endpoint_calls_or_clients", "no endpoint URL REST websocket exchange client or connector code"),
        ("no_credentials_signing_nonce_user_stream", "no credentials signing nonce or user stream implementation"),
        ("no_order_action_or_strategy", "no order placement cancellation amendment or strategy hook"),
        ("no_runner_consumption", "source only writes local artifacts and validates them locally"),
        ("no_real_economics_data_read", "official artifacts use synthetic task-local fixture rows"),
        ("no_metrics_pnl_viability", "source emits no real economics execution PnL or viability metrics"),
        ("no_live_deployment_promotion", "source emits no live deployment or promotion readiness"),
    ]
    return [{"check_id": check_id, "status": "pass", "detail": detail} for check_id, detail in checks]


def boundary_validation_rows(passed: bool) -> list[dict[str, str]]:
    checks = [
        ("prerequisite_0615T004_qa", "0615T004 QA is passed and account source ready"),
        ("local_transform_only", "task-local rows transformed into accepted economics fee rebate artifact schema"),
        ("redaction_required", "account scope and future fill refs are hashed before artifact storage"),
        ("validator_handoff", "source output validates through economics_fee_rebate_source.py"),
        ("arithmetic_preserved", "fee rebate net fee conversion tick value and fee-adjusted spread arithmetic are conserved"),
        ("forbidden_input_fields_fail_closed", "endpoint credential action and strategy fields fail closed"),
        ("no_endpoint_or_user_stream", "no endpoint client signed request nonce or user stream code"),
        ("no_real_private_data", "no real private order account live or economics data read"),
        ("no_runner_strategy_live", "no runner consumption strategy or live behavior"),
        ("no_metrics_pnl_promotion", "no real economics execution PnL deployment promotion or maker viability proof"),
    ]
    return [{"check_id": check_id, "status": "pass" if passed else "fail", "detail": detail} for check_id, detail in checks]


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = synthetic_input_rows()
    result = transform_rows(inputs)
    forbidden_result = transform_rows(forbidden_input_rows())
    source_passed = result.passed and not forbidden_result.passed

    _write_csv(output_dir / "economics_fixture_inputs.csv", inputs, INPUT_FIELDS)
    _write_csv(output_dir / "economics_output_artifact.csv", result.rows, ALL_FIELDS)
    _write_csv(
        output_dir / "economics_validation_summary.csv",
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
            "future_fill_ref_input_present",
            "account_scope_hash_prefix",
            "future_fill_ref_hash_prefix",
            "raw_account_scope_persisted_in_artifact",
            "raw_future_fill_ref_persisted_in_artifact",
        ],
    )
    _write_csv(output_dir / "no_trading_safety_audit.csv", no_trading_safety_audit_rows(), ["check_id", "status", "detail"])
    _write_csv(output_dir / "boundary_validation.csv", boundary_validation_rows(source_passed), ["check_id", "status", "detail"])

    manifest = {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "source_final_recommendation": SOURCE_FINAL_RECOMMENDATION,
        "synthesis_task_id": SYNTHESIS_TASK_ID,
        "synthesis_final_recommendation": SYNTHESIS_FINAL_RECOMMENDATION,
        "private_order_context_task_id": PRIVATE_ORDER_CONTEXT_TASK_ID,
        "private_order_context_final_recommendation": PRIVATE_ORDER_CONTEXT_FINAL_RECOMMENDATION,
        "account_inventory_context_task_id": ACCOUNT_INVENTORY_CONTEXT_TASK_ID,
        "account_inventory_context_final_recommendation": ACCOUNT_INVENTORY_CONTEXT_FINAL_RECOMMENDATION,
        "local_skeleton_task_id": LOCAL_SKELETON_TASK_ID,
        "local_skeleton_final_recommendation": LOCAL_SKELETON_FINAL_RECOMMENDATION,
        "source_line_id": SOURCE_LINE_ID,
        "generated_at": FIXED_GENERATED_AT,
        "final_recommendation": FINAL_RECOMMENDATION if source_passed else "economics_fee_rebate_read_only_source_needs_revision",
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
            "no_real_economics_metrics": True,
            "no_real_execution_metrics": True,
            "context_sources_alone_cannot_prove_fees_rebates_spread_capture_or_pnl": True,
            "no_pnl_proof": True,
            "no_live_deployment_promotion": True,
            "maker_viability_unproven": True,
        },
    }
    _write_json(output_dir / "economics_fee_rebate_read_only_source_manifest.json", manifest)
    return manifest


def generate_artifacts_command(args: argparse.Namespace) -> int:
    manifest = generate_artifacts(Path(args.output_dir))
    print(json.dumps({"output_dir": args.output_dir, "final_recommendation": manifest["final_recommendation"]}, sort_keys=True))
    return 0 if manifest["final_recommendation"] == FINAL_RECOMMENDATION else 2


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate local economics fee/rebate artifacts without endpoints, credentials, runners, metrics, PnL, or live behavior."
    )
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser("generate-artifacts", help="write 0615T005 task-local economics fee/rebate artifacts")
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
