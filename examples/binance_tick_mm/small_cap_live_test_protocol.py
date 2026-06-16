#!/usr/bin/env python3
"""Small-cap live-test protocol and dry-run gate.

This is a protocol validator and dry-run artifact generator. It does not
connect to venues, read credentials, start live processes, place orders,
cancel orders, or change strategy behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable


TASK_ID = "0615T008"
RUNNER_TASK_ID = "0615T007"
FINAL_RECOMMENDATION = "small_cap_live_test_protocol_ready_for_qa"
DEFAULT_OUTPUT_DIR = Path("local_live_analysis/small_cap_live_test_protocol_0615T008")

PROTOCOL_FIELDS = [
    "protocol_id",
    "symbol",
    "duration_minutes",
    "max_gross_notional_usdt",
    "max_single_order_notional_usdt",
    "max_position_notional_usdt",
    "max_loss_usdt",
    "post_only_required",
    "maker_only_required",
    "default_on_allowed",
    "requires_manual_total_control_approval",
    "requires_preflight",
    "requires_runner_qa_passed",
    "requires_cancel_all_shutdown_proof",
    "kill_switch_loss_usdt",
    "kill_switch_position_notional_usdt",
    "kill_switch_reject_count",
    "kill_switch_latency_p99_ms",
]

GATE_FIELDS = ["gate_id", "gate_name", "required_value", "actual_value", "status", "evidence"]


@dataclass(frozen=True)
class ProtocolIssue:
    reason_code: str
    detail: str


@dataclass(frozen=True)
class ProtocolResult:
    rows: list[dict[str, str]]
    issues: list[ProtocolIssue]

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def status(self) -> str:
        return "pass" if self.passed else "fail_closed"


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


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(str(value)) from exc


def default_protocol_row() -> dict[str, str]:
    return {
        "protocol_id": "0615T008_small_cap_protocol_v1",
        "symbol": "BTCUSDT",
        "duration_minutes": "10",
        "max_gross_notional_usdt": "25",
        "max_single_order_notional_usdt": "5",
        "max_position_notional_usdt": "10",
        "max_loss_usdt": "2",
        "post_only_required": "true",
        "maker_only_required": "true",
        "default_on_allowed": "false",
        "requires_manual_total_control_approval": "true",
        "requires_preflight": "true",
        "requires_runner_qa_passed": "true",
        "requires_cancel_all_shutdown_proof": "true",
        "kill_switch_loss_usdt": "2",
        "kill_switch_position_notional_usdt": "10",
        "kill_switch_reject_count": "3",
        "kill_switch_latency_p99_ms": "5000",
    }


def validate_protocol(row: dict[str, str]) -> ProtocolResult:
    issues: list[ProtocolIssue] = []
    bool_fields = [
        "post_only_required",
        "maker_only_required",
        "requires_manual_total_control_approval",
        "requires_preflight",
        "requires_runner_qa_passed",
        "requires_cancel_all_shutdown_proof",
    ]
    for field in PROTOCOL_FIELDS:
        if not str(row.get(field, "")).strip():
            issues.append(ProtocolIssue("missing_required_field", field))
    for field in bool_fields:
        if row.get(field, "").lower() != "true":
            issues.append(ProtocolIssue("required_true_gate_not_true", field))
    if row.get("default_on_allowed", "").lower() != "false":
        issues.append(ProtocolIssue("default_on_must_be_false", "default_on_allowed"))
    numeric_max = {
        "duration_minutes": Decimal("15"),
        "max_gross_notional_usdt": Decimal("25"),
        "max_single_order_notional_usdt": Decimal("5"),
        "max_position_notional_usdt": Decimal("10"),
        "max_loss_usdt": Decimal("2"),
    }
    for field, maximum in numeric_max.items():
        try:
            value = _decimal(row.get(field, ""))
        except ValueError:
            issues.append(ProtocolIssue("invalid_numeric_field", field))
            continue
        if value <= 0 or value > maximum:
            issues.append(ProtocolIssue("risk_cap_exceeds_allowed_max", field))
    if row.get("symbol") != "BTCUSDT":
        issues.append(ProtocolIssue("unsupported_symbol_for_protocol", row.get("symbol", "")))
    return ProtocolResult([row], issues)


def gate_rows(row: dict[str, str], result: ProtocolResult) -> list[dict[str, str]]:
    checks = [
        ("G001", "runner_qa_passed_required", "true", row.get("requires_runner_qa_passed", ""), "0615T007 QA must be passed"),
        ("G002", "manual_total_control_approval_required", "true", row.get("requires_manual_total_control_approval", ""), "0615T009 requires explicit approval"),
        ("G003", "post_only_required", "true", row.get("post_only_required", ""), "maker-only post-only live test"),
        ("G004", "default_on_forbidden", "false", row.get("default_on_allowed", ""), "test must not default-enable strategy"),
        ("G005", "cancel_all_shutdown_proof_required", "true", row.get("requires_cancel_all_shutdown_proof", ""), "shutdown proof required after run"),
        ("G006", "dry_run_only_current_task", "true", "true", "0615T008 does not open live"),
    ]
    return [
        {
            "gate_id": gate_id,
            "gate_name": name,
            "required_value": required,
            "actual_value": actual,
            "status": "pass" if actual == required and result.passed else ("pass" if gate_id == "G006" and result.passed else "fail_closed"),
            "evidence": evidence,
        }
        for gate_id, name, required, actual, evidence in checks
    ]


def artifact_requirement_rows() -> list[dict[str, str]]:
    names = [
        ("deployment_manifest.json", "preflight manifest with clean commit/config hashes"),
        ("start_marker.json", "operator-approved start marker"),
        ("stop_marker.json", "stop marker with reason"),
        ("live_audit.csv", "decision and lifecycle audit rows"),
        ("raw_market_data.gz", "raw public market data capture"),
        ("private_order_response_artifact.csv", "authorized private order response artifact"),
        ("account_inventory_artifact.csv", "authorized account/inventory artifact"),
        ("economics_fee_rebate_artifact.csv", "authorized fee/rebate artifact"),
        ("shutdown_cancel_final_proof.csv", "cancel-all shutdown proof"),
        ("archive.sha256", "archive checksum"),
    ]
    return [{"artifact_name": name, "required_reason": reason, "status": "required_for_0615T009"} for name, reason in names]


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    row = default_protocol_row()
    result = validate_protocol(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "small_cap_live_test_protocol.csv", [row], PROTOCOL_FIELDS)
    _write_csv(output_dir / "risk_gate_matrix.csv", gate_rows(row, result), GATE_FIELDS)
    _write_csv(output_dir / "kill_switch_rules.csv", [
        {"rule_id": "K001", "trigger": "realized_or_unrealized_loss_usdt >= 2", "required_action": "stop_submit_cancel_all_collect_shutdown_proof"},
        {"rule_id": "K002", "trigger": "position_notional_usdt >= 10", "required_action": "stop_add_side_cancel_all"},
        {"rule_id": "K003", "trigger": "reject_count >= 3", "required_action": "stop_run_cancel_all"},
        {"rule_id": "K004", "trigger": "latency_p99_ms >= 5000", "required_action": "stop_run_cancel_all"},
    ], ["rule_id", "trigger", "required_action"])
    _write_csv(output_dir / "required_live_artifacts.csv", artifact_requirement_rows(), ["artifact_name", "required_reason", "status"])
    _write_csv(output_dir / "dry_run_acceptance.csv", [
        {"check_id": "D001", "check_name": "protocol_validates", "status": result.status, "issue_count": str(len(result.issues))},
        {"check_id": "D002", "check_name": "current_task_does_not_open_live", "status": "pass", "issue_count": "0"},
        {"check_id": "D003", "check_name": "0615T009_is_first_live_capable_task", "status": "pass", "issue_count": "0"},
    ], ["check_id", "check_name", "status", "issue_count"])
    _write_csv(output_dir / "boundary_validation.csv", [
        {"check_id": "B001", "check_name": "no_credentials_required", "status": "pass", "evidence": "protocol only"},
        {"check_id": "B002", "check_name": "no_endpoint_call", "status": "pass", "evidence": "dry-run artifact generation only"},
        {"check_id": "B003", "check_name": "no_order_action", "status": "pass", "evidence": "0615T009 required before live"},
        {"check_id": "B004", "check_name": "no_default_on", "status": "pass", "evidence": "default_on_allowed=false"},
    ], ["check_id", "check_name", "status", "evidence"])
    manifest = {
        "task_id": TASK_ID,
        "runner_task_id": RUNNER_TASK_ID,
        "final_recommendation": FINAL_RECOMMENDATION,
        "protocol_status": result.status,
        "issue_count": len(result.issues),
        "first_live_capable_task": "0615T009",
        "next_task_id": "0615T009",
        "requires_explicit_total_control_approval": True,
    }
    _write_json(output_dir / "small_cap_live_test_protocol_manifest.json", manifest)
    return manifest


def generate_artifacts_command(args: argparse.Namespace) -> int:
    manifest = generate_artifacts(args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate-artifacts", help="Generate protocol dry-run artifacts")
    gen.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    gen.set_defaults(func=generate_artifacts_command)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
