#!/usr/bin/env python3
"""Local Hyperliquid tiny-live operator packet builder.

This module is live-capable preparation only. It generates and validates an
operator packet for a future, separately approved tiny-live run on awsserver1.
It does not call endpoints, read credentials, sign requests, manage nonces,
query accounts, place/cancel orders, start a live bot, deploy, or prove PnL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0616T006"
SOURCE_TASK_IDS = ["0616T002", "0616T003", "0616T004", "0616T005"]
SCHEMA_VERSION = "hyperliquid_tiny_live_operator_packet_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_live_capable_preflight_operator_packet_0616T006"
T005_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_protocol_design_0616T005"
FINAL_RECOMMENDATION = "hyperliquid_tiny_live_live_capable_preflight_operator_packet_ready_for_qa"

APPROVAL_FIELDS = [
    "symbol",
    "max_notional",
    "max_order_size",
    "max_position",
    "max_loss",
    "duration",
    "host_machine",
    "account_scope",
    "real_orders_allowed",
]

BOUNDARY_FLAGS = {
    "operator_packet_only": True,
    "live_capable_preflight_only": True,
    "intended_host_is_awsserver1": True,
    "pullback_local_validation_defined": True,
    "no_private_endpoint_calls": True,
    "no_credentials": True,
    "no_signing": True,
    "no_nonce": True,
    "no_user_stream": True,
    "no_account_query": True,
    "no_order_placement": True,
    "no_order_cancellation": True,
    "no_order_amendment": True,
    "no_live_bot_start": True,
    "no_deployment": True,
    "no_promotion": True,
    "no_pnl_proof": True,
    "no_maker_viability_proof": True,
}


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


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_t005_manifest() -> Dict[str, Any]:
    path = T005_DIR / "tiny_live_protocol_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def approval_rows() -> List[Dict[str, str]]:
    proposed_values = {
        "host_machine": "awsserver1",
        "real_orders_allowed": "false_until_separate_controller_approval",
    }
    return [
        {
            "field": field,
            "proposed_value": proposed_values.get(field, ""),
            "approval_status": "pending_controller_approval",
            "required_before_live": "true",
            "source_task_id": "0616T005",
        }
        for field in APPROVAL_FIELDS
    ]


def host_preflight_rows() -> List[Dict[str, str]]:
    return [
        {"check_id": "H01", "check": "ssh_alias_resolves", "target": "awsserver1", "required": "true", "status": "operator_must_verify"},
        {"check_id": "H02", "check": "repo_path_present", "target": "/home/admin/hftbacktest_or_operator_selected_path", "required": "true", "status": "operator_must_verify"},
        {"check_id": "H03", "check": "git_branch_cross_exchange", "target": "cross-exchange", "required": "true", "status": "operator_must_verify"},
        {"check_id": "H04", "check": "conda_env_available", "target": "hft-py38_or_operator_selected_env", "required": "true", "status": "operator_must_verify"},
        {"check_id": "H05", "check": "clock_and_timezone_recorded", "target": "host_time_utc_and_local", "required": "true", "status": "operator_must_verify"},
        {"check_id": "H06", "check": "disk_space_recorded", "target": "artifact_volume_path", "required": "true", "status": "operator_must_verify"},
        {"check_id": "H07", "check": "public_market_network_reachability", "target": "binance_public_and_hyperliquid_public", "required": "true", "status": "operator_must_verify"},
        {"check_id": "H08", "check": "private_endpoint_not_called_by_preflight", "target": "hyperliquid_private", "required": "true", "status": "pass_by_boundary"},
        {"check_id": "H09", "check": "artifact_archive_and_checksums", "target": "sha256_manifest", "required": "true", "status": "operator_must_verify"},
    ]


def artifact_contract_rows() -> List[Dict[str, str]]:
    return [
        {"artifact": "operator_packet_manifest.json", "phase": "local_preflight", "required": "true", "validation": "json_schema_and_boundary_fields"},
        {"artifact": "approval_fields.csv", "phase": "local_preflight", "required": "true", "validation": "all_required_fields_present_pending_or_approved"},
        {"artifact": "awsserver1_host_preflight.csv", "phase": "awsserver1_preflight", "required": "true", "validation": "operator_verified_no_private_calls"},
        {"artifact": "future_run_intent_marker.json", "phase": "future_live_window", "required": "true", "validation": "explicit_approved_window_required"},
        {"artifact": "public_market_data_manifest.json", "phase": "future_live_window", "required": "true", "validation": "binance_lead_and_hyperliquid_public_inputs"},
        {"artifact": "shutdown_evidence_placeholder.json", "phase": "future_shutdown", "required": "true", "validation": "cancel_all_shutdown_proof_or_fail_closed"},
        {"artifact": "artifact_pullback_manifest.json", "phase": "local_pullback", "required": "true", "validation": "source_host_paths_local_paths_checksums"},
        {"artifact": "sha256sums.txt", "phase": "local_pullback", "required": "true", "validation": "checksum_recalculation"},
    ]


def inert_command_rows() -> List[Dict[str, str]]:
    return [
        {
            "step": "connectivity_check",
            "command": "ssh awsserver1 'hostname && date -u'",
            "real_order_capability": "none",
            "operator_note": "host metadata only",
        },
        {
            "step": "create_remote_artifact_dir",
            "command": "ssh awsserver1 'mkdir -p ~/hftbacktest_live_artifacts/0616T006_preflight'",
            "real_order_capability": "none",
            "operator_note": "directory creation only",
        },
        {
            "step": "pullback_artifacts",
            "command": "rsync -av awsserver1:~/hftbacktest_live_artifacts/0616T006_preflight/ local_live_analysis/hyperliquid_tiny_live_live_capable_preflight_operator_packet_0616T006/pulled_back/",
            "real_order_capability": "none",
            "operator_note": "data copy only",
        },
        {
            "step": "validate_pulled_back",
            "command": "/home/liushuai/workspace/hftbacktest/.conda-envs/hft-py38/bin/python examples/hyperliquid/hyperliquid_tiny_live_operator_packet.py validate-artifacts --input-dir local_live_analysis/hyperliquid_tiny_live_live_capable_preflight_operator_packet_0616T006",
            "real_order_capability": "none",
            "operator_note": "local validation only",
        },
    ]


def placeholder_payloads() -> Dict[str, Dict[str, Any]]:
    return {
        "future_run_intent_marker.json": {
            "task_id": TASK_ID,
            "status": "not_started",
            "requires_separate_controller_approval": True,
            "real_orders_allowed": "pending_controller_approval",
            "run_window_authorized": False,
        },
        "public_market_data_manifest.json": {
            "task_id": TASK_ID,
            "binance_lead_public_data": "required_future_artifact",
            "hyperliquid_lag_public_data": "required_future_artifact",
            "private_endpoint_called": False,
        },
        "shutdown_evidence_placeholder.json": {
            "task_id": TASK_ID,
            "shutdown_evidence_status": "placeholder_until_future_approved_run",
            "cancel_all_real_call_authorized": False,
            "fail_closed_if_missing": True,
        },
        "artifact_pullback_manifest.json": {
            "task_id": TASK_ID,
            "source_host": "awsserver1",
            "local_validation_required": True,
            "checksum_required": True,
            "pulled_back": False,
        },
    }


def boundary_rows() -> List[Dict[str, str]]:
    return [{"check": key, "status": "pass" if value else "fail"} for key, value in BOUNDARY_FLAGS.items()]


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> Dict[str, Any]:
    output_dir = output_dir.resolve()
    t005_manifest = _load_t005_manifest()
    approvals = approval_rows()

    _write_csv(
        output_dir / "approval_fields.csv",
        approvals,
        ["field", "proposed_value", "approval_status", "required_before_live", "source_task_id"],
    )
    _write_csv(
        output_dir / "awsserver1_host_preflight.csv",
        host_preflight_rows(),
        ["check_id", "check", "target", "required", "status"],
    )
    _write_csv(
        output_dir / "artifact_contract.csv",
        artifact_contract_rows(),
        ["artifact", "phase", "required", "validation"],
    )
    _write_csv(
        output_dir / "inert_operator_commands.csv",
        inert_command_rows(),
        ["step", "command", "real_order_capability", "operator_note"],
    )
    _write_csv(output_dir / "boundary_validation.csv", boundary_rows(), ["check", "status"])
    for filename, payload in placeholder_payloads().items():
        _write_json(output_dir / filename, payload)

    _write_text(
        output_dir / "operator_packet_readme.md",
        "\n".join(
            [
                "# Hyperliquid Tiny-Live Operator Packet",
                "",
                "This packet is live-capable preparation only.",
                "It targets future execution on `awsserver1` and local artifact validation.",
                "It does not authorize real orders, cancellation, private endpoint calls, account queries, credentials, signing, nonce handling, live bot startup, deployment, promotion, PnL proof, or maker viability proof.",
                "",
                "All live approval fields remain `pending_controller_approval` until a later separately dispatched task approves them.",
                "",
            ]
        ),
    )

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_task_ids": SOURCE_TASK_IDS,
        "source_t005_final_recommendation": t005_manifest.get("final_recommendation", "missing"),
        "final_recommendation": FINAL_RECOMMENDATION,
        "host_machine": "awsserver1",
        "live_authorized": False,
        "run_window_authorized": False,
        "real_orders_allowed": "pending_controller_approval",
        "approval_fields_pending": [row["field"] for row in approvals if row["approval_status"] == "pending_controller_approval"],
        "boundary_flags": BOUNDARY_FLAGS,
        "git_commit": _git_commit(),
        "artifacts": {
            "approval_fields": str(output_dir / "approval_fields.csv"),
            "awsserver1_host_preflight": str(output_dir / "awsserver1_host_preflight.csv"),
            "artifact_contract": str(output_dir / "artifact_contract.csv"),
            "inert_operator_commands": str(output_dir / "inert_operator_commands.csv"),
            "future_run_intent_marker": str(output_dir / "future_run_intent_marker.json"),
            "public_market_data_manifest": str(output_dir / "public_market_data_manifest.json"),
            "shutdown_evidence_placeholder": str(output_dir / "shutdown_evidence_placeholder.json"),
            "artifact_pullback_manifest": str(output_dir / "artifact_pullback_manifest.json"),
            "boundary_validation": str(output_dir / "boundary_validation.csv"),
            "operator_packet_readme": str(output_dir / "operator_packet_readme.md"),
            "sha256sums": str(output_dir / "sha256sums.txt"),
        },
    }
    _write_json(output_dir / "operator_packet_manifest.json", manifest)
    checksum_targets = [
        output_dir / "operator_packet_manifest.json",
        output_dir / "approval_fields.csv",
        output_dir / "awsserver1_host_preflight.csv",
        output_dir / "artifact_contract.csv",
        output_dir / "inert_operator_commands.csv",
        output_dir / "future_run_intent_marker.json",
        output_dir / "public_market_data_manifest.json",
        output_dir / "shutdown_evidence_placeholder.json",
        output_dir / "artifact_pullback_manifest.json",
        output_dir / "boundary_validation.csv",
        output_dir / "operator_packet_readme.md",
    ]
    checksum_lines = [f"{_sha256(path)}  {path.name}" for path in checksum_targets]
    _write_text(output_dir / "sha256sums.txt", "\n".join(checksum_lines) + "\n")
    return manifest


def validate_artifacts(input_dir: Path) -> Dict[str, Any]:
    input_dir = input_dir.resolve()
    issues: List[Dict[str, str]] = []

    manifest_path = input_dir / "operator_packet_manifest.json"
    if not manifest_path.exists():
        issues.append({"artifact": "operator_packet_manifest.json", "reason": "missing"})
        manifest: Dict[str, Any] = {}
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("task_id") != TASK_ID:
        issues.append({"artifact": "operator_packet_manifest.json", "reason": "unexpected_task_id"})
    if manifest.get("host_machine") != "awsserver1":
        issues.append({"artifact": "operator_packet_manifest.json", "reason": "host_machine_not_awsserver1"})
    if manifest.get("live_authorized") is not False:
        issues.append({"artifact": "operator_packet_manifest.json", "reason": "live_authorized_not_false"})
    if manifest.get("real_orders_allowed") != "pending_controller_approval":
        issues.append({"artifact": "operator_packet_manifest.json", "reason": "real_orders_not_pending"})
    if manifest.get("final_recommendation") != FINAL_RECOMMENDATION:
        issues.append({"artifact": "operator_packet_manifest.json", "reason": "unexpected_final_recommendation"})

    approval_path = input_dir / "approval_fields.csv"
    if not approval_path.exists():
        issues.append({"artifact": "approval_fields.csv", "reason": "missing"})
    else:
        rows = _read_csv(approval_path)
        fields = {row.get("field", "") for row in rows}
        if fields != set(APPROVAL_FIELDS):
            issues.append({"artifact": "approval_fields.csv", "reason": "approval_field_set_mismatch"})
        for row in rows:
            if row.get("approval_status") != "pending_controller_approval":
                issues.append({"artifact": "approval_fields.csv", "reason": "approval_not_pending", "field": row.get("field", "")})

    boundary_path = input_dir / "boundary_validation.csv"
    if not boundary_path.exists():
        issues.append({"artifact": "boundary_validation.csv", "reason": "missing"})
    else:
        for row in _read_csv(boundary_path):
            if row.get("status") != "pass":
                issues.append({"artifact": "boundary_validation.csv", "reason": "boundary_check_failed", "check": row.get("check", "")})

    for required in ["awsserver1_host_preflight.csv", "artifact_contract.csv", "inert_operator_commands.csv", "sha256sums.txt"]:
        if not (input_dir / required).exists():
            issues.append({"artifact": required, "reason": "missing"})

    return {
        "status": "pass" if not issues else "fail_closed",
        "issue_count": len(issues),
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or validate local Hyperliquid tiny-live operator packet artifacts.")
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser("generate-artifacts", help="write local operator packet artifacts")
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    validate = subparsers.add_parser("validate-artifacts", help="validate local operator packet artifacts")
    validate.add_argument("--input-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.command == "generate-artifacts":
        print(json.dumps(generate_artifacts(args.output_dir), indent=2, sort_keys=True))
    elif args.command == "validate-artifacts":
        result = validate_artifacts(args.input_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        if result["status"] != "pass":
            raise SystemExit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
