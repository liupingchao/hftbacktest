#!/usr/bin/env python3
"""Local Hyperliquid shutdown dry-run proof gate.

This runner uses fake local orders only. It does not call endpoints, query open
orders, use credentials, sign requests, manage nonce values, cancel real orders,
or run live.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0616T004"
SOURCE_TASK_ID = "0616T003"
SCHEMA_VERSION = "hyperliquid_shutdown_dry_run_proof_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_shutdown_dry_run_proof_0616T004"
FINAL_RECOMMENDATION = "hyperliquid_shutdown_dry_run_proof_ready_for_qa"

BOUNDARY_FLAGS = {
    "local_fake_dry_run_only": True,
    "no_private_endpoint_calls": True,
    "no_credentials": True,
    "no_signing": True,
    "no_nonce": True,
    "no_user_stream": True,
    "no_account_query": True,
    "no_real_cancel": True,
    "no_order_placement": True,
    "no_live_process": True,
    "no_strategy_implementation": True,
    "no_deployment": True,
    "no_promotion": True,
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


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fake_orders() -> list[dict[str, str]]:
    return [
        {"opaque_order_ref": "oid_sha256_fake_1", "initial_local_state": "open_local_fake", "cancel_intent": "yes"},
        {"opaque_order_ref": "oid_sha256_fake_2", "initial_local_state": "open_local_fake", "cancel_intent": "yes"},
    ]


def build_shutdown_rows(*, missing_terminal: bool = False) -> dict[str, list[dict[str, str]]]:
    orders = fake_orders()
    cancel_intents = [
        {
            "opaque_order_ref": row["opaque_order_ref"],
            "cancel_scope": "local_fake_cancel_all",
            "cancel_request_time": "2026-06-16T14:40:00Z",
            "cancel_request_status": "dry_run_intent_recorded",
            "real_endpoint_called": "false",
        }
        for row in orders
    ]
    terminal_rows = []
    for idx, row in enumerate(orders, start=1):
        if missing_terminal and idx == 2:
            terminal_state = "missing_terminal_fail_closed"
            proof_level = "insufficient_proof"
        else:
            terminal_state = "local_fake_terminal_canceled"
            proof_level = "local_fake_proof_only"
        terminal_rows.append(
            {
                "opaque_order_ref": row["opaque_order_ref"],
                "terminal_state": terminal_state,
                "proof_level": proof_level,
                "exchange_side_no_open_order_proven": "false",
                "fail_closed_reason": "missing_terminal_state" if proof_level == "insufficient_proof" else "",
            }
        )
    return {"orders": orders, "cancel_intents": cancel_intents, "terminal_rows": terminal_rows}


def classify_proof(terminal_rows: list[dict[str, str]]) -> str:
    if any(row["proof_level"] == "insufficient_proof" for row in terminal_rows):
        return "insufficient_proof"
    if all(row["proof_level"] == "local_fake_proof_only" for row in terminal_rows):
        return "local_fake_proof_only"
    return "unknown_fail_closed"


def generate_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    clean = build_shutdown_rows(missing_terminal=False)
    failed = build_shutdown_rows(missing_terminal=True)
    clean_proof = classify_proof(clean["terminal_rows"])
    failed_proof = classify_proof(failed["terminal_rows"])

    startup_manifest = {
        "task_id": TASK_ID,
        "source_task_id": SOURCE_TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "mode": "local_fake_dry_run",
        "real_endpoint_called": False,
        "git_commit": _git_commit(),
        "boundary_flags": BOUNDARY_FLAGS,
    }
    _write_json(output_dir / "startup_manifest.json", startup_manifest)
    _write_csv(output_dir / "fake_open_orders.csv", clean["orders"], ["opaque_order_ref", "initial_local_state", "cancel_intent"])
    _write_csv(output_dir / "cancel_intent_log.csv", clean["cancel_intents"], ["opaque_order_ref", "cancel_scope", "cancel_request_time", "cancel_request_status", "real_endpoint_called"])
    _write_csv(output_dir / "local_terminal_proof.csv", clean["terminal_rows"], ["opaque_order_ref", "terminal_state", "proof_level", "exchange_side_no_open_order_proven", "fail_closed_reason"])
    _write_csv(output_dir / "fail_closed_scenarios.csv", failed["terminal_rows"], ["opaque_order_ref", "terminal_state", "proof_level", "exchange_side_no_open_order_proven", "fail_closed_reason"])
    proof_summary = [
        {"scenario": "clean_local_fake_shutdown", "proof_level": clean_proof, "exchange_side_no_open_order_proven": "false"},
        {"scenario": "missing_terminal_state", "proof_level": failed_proof, "exchange_side_no_open_order_proven": "false"},
    ]
    _write_csv(output_dir / "proof_level_summary.csv", proof_summary, ["scenario", "proof_level", "exchange_side_no_open_order_proven"])
    _write_csv(output_dir / "boundary_validation.csv", [{"check": key, "status": "pass" if value else "fail"} for key, value in BOUNDARY_FLAGS.items()], ["check", "status"])
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_task_id": SOURCE_TASK_ID,
        "final_recommendation": FINAL_RECOMMENDATION,
        "clean_shutdown_proof_level": clean_proof,
        "fail_closed_scenario_proof_level": failed_proof,
        "exchange_side_no_open_order_proven": False,
        "boundary_flags": BOUNDARY_FLAGS,
        "artifacts": {
            "startup_manifest": str(output_dir / "startup_manifest.json"),
            "fake_open_orders": str(output_dir / "fake_open_orders.csv"),
            "cancel_intent_log": str(output_dir / "cancel_intent_log.csv"),
            "local_terminal_proof": str(output_dir / "local_terminal_proof.csv"),
            "fail_closed_scenarios": str(output_dir / "fail_closed_scenarios.csv"),
            "proof_level_summary": str(output_dir / "proof_level_summary.csv"),
            "boundary_validation": str(output_dir / "boundary_validation.csv"),
        },
    }
    _write_json(output_dir / "shutdown_dry_run_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local Hyperliquid shutdown dry-run proof artifacts.")
    subparsers = parser.add_subparsers(dest="command")
    generate = subparsers.add_parser("generate-artifacts", help="write local dry-run proof artifacts")
    generate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if args.command == "generate-artifacts":
        print(json.dumps(generate_artifacts(args.output_dir), indent=2, sort_keys=True))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
