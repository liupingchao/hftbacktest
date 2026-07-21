from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from examples.hyperliquid import cross_exchange_t024_same_window_acceptance as acceptance
from examples.hyperliquid import cross_exchange_online_estimators as online_estimators
from examples.hyperliquid import cross_exchange_live_remote_orchestrator as orchestrator
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


SOURCE_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=acceptance.PROJECT_ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
TASK_ID = "0719T001"
REMOTE_RUN_ROOT = "/remote/principal-task12-artifacts/run"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def live_artifact_dir(input_root: Path) -> Path:
    return (
        input_root
        / "run"
        / "window_01"
        / "window_01"
        / "pulled_back_awsserver1"
    )


def run_task12_acceptance(
    *,
    input_root: Path,
    output_dir: Path,
    expected_task_id: str = TASK_ID,
    expected_source_commit: str = SOURCE_COMMIT,
    expected_window_seconds: float = acceptance.DEFAULT_EXPECTED_WINDOW_SECONDS,
) -> dict:
    return acceptance.run_acceptance(
        input_root=input_root,
        output_dir=output_dir,
        expected_task_id=expected_task_id,
        expected_source_commit=expected_source_commit,
        expected_remote_run_root=REMOTE_RUN_ROOT,
        expected_window_seconds=expected_window_seconds,
    )


def seal_run(input_root: Path) -> None:
    run_root = input_root / "run"
    orchestrator.write_sha256_manifest(run_root)
    verification = orchestrator.verify_sha256_manifest(run_root)
    assert verification["status"] == "pass"


def assert_acceptance_blocked(input_root: Path, output_dir: Path) -> None:
    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=output_dir,
    )
    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def make_artifact(
    root: Path,
    *,
    window_seconds: float = acceptance.DEFAULT_EXPECTED_WINDOW_SECONDS,
) -> Path:
    run = root / "run"
    window = run / "window_01"
    live = window / "window_01" / "pulled_back_awsserver1"
    raw_tracked_refs = [
        {"attempt": 1, "oid": 101, "cloid": "cloid-buy"},
        {"attempt": 2, "oid": 102, "cloid": "cloid-sell"},
    ]
    raw_cancel_results = [
        {
            "method": "cancel",
            "attempt": 1,
            "oid": 101,
            "cloid": "cloid-buy",
            "result": {
                "status": "ok",
                "response": {"data": {"statuses": ["success"]}},
            },
        },
        {
            "method": "cancel",
            "attempt": 2,
            "oid": 102,
            "cloid": "cloid-sell",
            "result": {
                "status": "ok",
                "response": {"data": {"statuses": ["success"]}},
            },
        },
    ]
    cancel_reference_reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=raw_tracked_refs,
        cancel_results=raw_cancel_results,
    )
    persisted_tracked_refs = executor.redact(
        fill_window.persisted_reference_identity_rows(raw_tracked_refs)
    )
    persisted_cancel_results = executor.redact(
        fill_window.persisted_reference_identity_rows(raw_cancel_results)
    )
    assert acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=persisted_tracked_refs,
        cancel_results=persisted_cancel_results,
    ) == cancel_reference_reconciliation
    command = [
        acceptance.EXPECTED_REMOTE_PYTHON,
        acceptance.EXPECTED_WATCHER_SCRIPT,
        "--event-driven-edge-gate-live",
        "--watcher-seconds",
        str(float(window_seconds)),
        "--max-order-size",
        "0.005",
        "--max-loss-usdc",
        "1.0",
        "--max-position-btc",
        "0.01",
        "--max-real-order-submissions",
        "2",
        "--requote-attempts",
        "2",
        "--quote-hold-seconds",
        "3",
        "--wait-seconds",
        "10",
        "--env-file",
        str(root / ".env"),
        "--artifact-task-id",
        TASK_ID,
        "--artifact-window-id",
        "1",
        "--run-id",
        f"{TASK_ID}:window_01",
        "--output-dir",
        f"{REMOTE_RUN_ROOT}/window_01",
        "--hyperliquid-l2book-fast",
        "--exchange-reconciled-manager",
    ]
    fixture_attempt_rows = [
        {
            "attempt": 1,
            "attempt_id": 1,
            "attempt_key": f"{TASK_ID}:window_01:attempt_1",
            "event_sequence": 1,
            "guard_status": "pass",
            "guard_reason": "",
            "edge_gate_status": "pass",
            "edge_gate_reason": "",
            "skip_reason": "",
            "side": "buy",
            "limit_px": "64000.0",
            "size_btc": "0.002",
            "order_status_types": "resting",
            "order_endpoint_called": True,
            "cancel_endpoint_called": True,
        },
        {
            "attempt": 2,
            "attempt_id": 2,
            "attempt_key": f"{TASK_ID}:window_01:attempt_2",
            "event_sequence": 1,
            "guard_status": "pass",
            "guard_reason": "",
            "edge_gate_status": "pass",
            "edge_gate_reason": "",
            "skip_reason": "",
            "side": "sell",
            "limit_px": "66000.0",
            "size_btc": "0.002",
            "order_status_types": "resting",
            "order_endpoint_called": True,
            "cancel_endpoint_called": True,
        },
    ]
    trigger_rows = [
        {
            "event_sequence": 1,
            "source_channel": "l2Book",
            "source_event_exchange_time_ms": 1_000,
            "fresh_touch_allowed": True,
            "trigger_found": True,
            "guard_status": "pass",
            "guard_reason": "",
            "event_to_guard_start_seconds": 0.01,
            "target_event_to_guard_seconds": 0.5,
            "live_window_called": True,
            "private_or_order_endpoint_called_before_trigger": True,
            "private_read_endpoint_called_before_decision": True,
            "order_endpoint_called_before_decision": False,
            "cancel_endpoint_called_before_decision": False,
        }
    ]
    immediate_guard_rows = [
        {
            "attempt": 1,
            "event_sequence": 1,
            "status": "pass",
            "reason": "",
        }
    ]
    edge_gate_rows = [
        {
            "attempt": 1,
            "event_sequence": 1,
            "edge_gate_status": "pass",
            "edge_gate_reason": "",
        }
    ]
    inline_manifest = {
        "private_endpoint_called": True,
        "private_read_endpoint_called": True,
        "real_order_endpoint_called": True,
        "real_cancel_endpoint_called": True,
        "requote_attempts_requested": 2,
        "requote_attempts_completed": 2,
        "candidate_attempt_evidence_row_count": 2,
        "manager_attempt_identity_count": 2,
    }
    decision_evidence_summary = {
        "schema_version": (
            "event_driven_decision_evidence_summary_v1"
        ),
        "candidate_evaluation_row_count": 1,
        "trigger_row_count": 1,
        "anti_drift_block_count": 0,
        "anti_drift_block_reason_counts": {},
        "anti_drift_gate_evaluation_count": 0,
        "anti_drift_gate_pass_count": 0,
        "anti_drift_gate_block_count": 0,
        "immediate_guard_evaluation_count": 1,
        "immediate_guard_pass_count": 1,
        "immediate_guard_fail_count": 0,
        "immediate_guard_failure_reason_counts": {},
        "immediate_guard_failure_reason_atom_counts": {},
        "edge_gate_evaluation_count": 1,
        "edge_gate_pass_count": 1,
        "edge_gate_block_count": 0,
        "edge_gate_block_reason_counts": {},
        "no_submit_stage_counts": {
            "anti_drift_block": 0,
            "immediate_guard_fail": 0,
            "edge_gate_block": 0,
        },
        "no_submit_reason_counts": {},
        "order_authorized_row_count": 1,
        "candidate_attempt_evidence_row_count": 2,
        "manager_attempt_identity_count": 2,
        "submitted_manager_attempt_identity_count": 2,
        "submitted_attempt_count": 2,
        "cancelled_attempt_count": 2,
        "decision_rows_with_private_read_before_count": 1,
        "decision_rows_with_order_before_count": 0,
        "decision_rows_with_cancel_before_count": 0,
        "private_read_endpoint_called": True,
        "real_order_endpoint_called": True,
        "real_cancel_endpoint_called": True,
        "validation_reasons": [],
    }
    inline_manifest["decision_evidence_summary"] = (
        decision_evidence_summary
    )
    source_digests, source_error = acceptance.expected_git_source_snapshot(SOURCE_COMMIT)
    assert source_error == ""
    write_json(
        run / acceptance.RUNTIME_SOURCE_PROVENANCE_NAME,
        {
            "schema_version": (
                "cross_exchange_runtime_source_provenance_v2"
            ),
            "status": "pass",
            "task_id": TASK_ID,
            "source_commit": SOURCE_COMMIT,
            "source_commit_source": "source_commit.txt",
            "sealed_before_watcher_start": True,
            "run_root": REMOTE_RUN_ROOT,
            "python_executable": acceptance.EXPECTED_REMOTE_PYTHON,
            "watcher_command_script": acceptance.EXPECTED_WATCHER_SCRIPT,
            "watcher_commands": [command],
            "file_count": len(source_digests),
            "files": [
                {"path": path, "sha256": digest, "bytes": 1}
                for path, digest in sorted(source_digests.items())
            ],
        },
    )
    write_json(
        run / acceptance.RUNTIME_SOURCE_START_VERIFICATION_NAME,
        {
            "status": "pass",
            "task_id": TASK_ID,
            "phase": "pre_watcher_start",
            "watcher_process_started": False,
        },
    )
    write_json(
        run / acceptance.RUNTIME_SOURCE_POSTRUN_VERIFICATION_NAME,
        {"status": "pass", "task_id": TASK_ID, "phase": "postrun"},
    )
    (run / "source_commit.txt").write_text(SOURCE_COMMIT + "\n", encoding="utf-8")
    write_json(
        root / "preflight" / "orchestrator_preflight.json",
        {
            "status": "pass",
            "preflight_only": True,
            "task_id": TASK_ID,
            "source_commit": SOURCE_COMMIT,
            "remote_repo": "/remote/t025-source",
            "run_root": REMOTE_RUN_ROOT,
            "watcher_commands": [command],
            "envelope": {
                "exact_envelope_profile": "two-sided-manager",
                "mode": "event-driven-edge-gate-live",
                "window_seconds": float(window_seconds),
                "max_order_size_btc": 0.005,
                "max_loss_usdc": 1.0,
                "max_position_btc": 0.01,
                "max_real_order_submissions": 2,
                "quote_hold_seconds": 3,
                "wait_seconds": 10,
                "exchange_reconciled_manager": True,
                "requote_attempts": 2,
                "private_proof_mode": "live_open_orders",
                "hyperliquid_l2book_fast": True,
                "lead_source": "binance_public_book_ticker",
            },
            "strategy_activation": {
                "dynamic_spread_activation_enabled": False,
                "fill_feedback_activation_enabled": False,
                "inventory_skew_activation_enabled": False,
                "multi_level_activation_enabled": False,
                "actual_quote_behavior_changed": False,
            },
        },
    )
    write_json(
        run / "run_complete.json",
        {
            "task_id": TASK_ID,
            "state": "complete",
            "run_root": REMOTE_RUN_ROOT,
        },
    )
    write_json(
        run / "run_status.json",
        {
            "task_id": TASK_ID,
            "state": "complete",
            "remote_repo": "/remote/t025-source",
            "run_root": REMOTE_RUN_ROOT,
        },
    )
    write_json(
        run / "remote_sha256_verification.json",
        {"status": "pass", "missing_count": 0, "mismatch_count": 0},
    )
    write_json(window / "runner_command.json", {"command": command})
    write_json(
        window / "window_status.json",
        {
            "task_id": TASK_ID,
            "state": "complete",
            "window_dir": f"{REMOTE_RUN_ROOT}/window_01",
            "child_returncode": 0,
            "child_reaped": True,
            "termination_escalated_to_sigkill": False,
            "open_orders_proof_after_child_exit": True,
        },
    )
    write_json(
        window / "independent_remote_open_orders_check.json",
        {"task_id": TASK_ID, "final_open_orders_count": 0},
    )
    write_json(
        window / "event_driven_watcher_manifest.json",
        {
            "task_id": TASK_ID,
            "artifact_window_id": 1,
            "watcher_seconds_requested": float(window_seconds),
            "event_driven_evaluation_count": 1,
            "current_candidate_count": 1,
            "trigger_found": True,
            "trigger_count": 1,
            "event_driven_guard_status": "pass",
            "selected_candidate": {"fresh_touch_decision": {"allowed": True}},
            "live_submissions_count": 2,
            "fill_count": 0,
            "maker_fill_count": 0,
            "dynamic_spread_activation_enabled": False,
            "actual_quote_behavior_changed": False,
            "task7_exchange_reconciled_manager_enabled": True,
            "task7_run_id": f"{TASK_ID}:window_01",
            "hyperliquid_l2book_fast": True,
            "edge_gate_enabled": True,
            "edge_gate_live_compatible_source_available": True,
            "edge_gate_source_status": "decision_time_public_fair_mid_provider",
            "edge_gate_pass_count": 1,
            "edge_gate_block_count": 0,
            "anti_drift_pass_count": 0,
            "anti_drift_block_count": 0,
            "candidate_attempt_evidence_row_count": 2,
            "manager_attempt_identity_count": 2,
            "submitted_attempt_count": 2,
            "decision_evidence_summary": decision_evidence_summary,
            "public_waiting_phase_private_or_order_endpoint_called": True,
            "public_waiting_phase_private_read_endpoint_called": True,
            "public_waiting_phase_order_endpoint_called": True,
            "public_waiting_phase_cancel_endpoint_called": True,
            "max_real_order_submissions": 2,
            "max_order_size_btc": 0.005,
        },
    )
    write_json(window / "inline_reprice_manifest.json", inline_manifest)
    write_json(
        window / "event_driven_decision_evidence_summary.json",
        decision_evidence_summary,
    )
    write_csv(
        window / "event_driven_trigger_decision_matrix.csv",
        trigger_rows,
        watcher.trigger_decision_fieldnames(),
    )
    write_csv(
        window / "immediate_pre_submit_guard_matrix.csv",
        immediate_guard_rows,
        ["attempt", "event_sequence", "status", "reason"],
    )
    write_csv(
        window / "anti_drift_gate_matrix.csv",
        [],
        ["status", "reason"],
    )
    write_csv(
        window / "edge_gate_matrix.csv",
        edge_gate_rows,
        [
            "attempt",
            "event_sequence",
            "edge_gate_status",
            "edge_gate_reason",
        ],
    )
    write_csv(
        window / "quote_attempt_matrix.csv",
        fixture_attempt_rows,
        [
            "attempt",
            "attempt_id",
            "attempt_key",
            "event_sequence",
            "guard_status",
            "guard_reason",
            "edge_gate_status",
            "edge_gate_reason",
            "skip_reason",
            "side",
            "limit_px",
            "size_btc",
            "order_status_types",
            "order_endpoint_called",
            "cancel_endpoint_called",
        ],
    )
    write_json(
        window / "online_estimator_snapshot.json",
        {"activation_enabled": False, "actual_quote_behavior_changed": False},
    )
    write_json(
        window / "fill_feedback_snapshot.json",
        {
            "dynamic_spread_activation_enabled": False,
            "fill_feedback_activation_enabled": False,
            "actual_quote_behavior_changed": False,
        },
    )
    write_json(
        window / "live_status.json",
        {
            "risk": {"multi_level_activation_enabled": False},
            "writer_health": {"status": "healthy", "failure_count": 0},
            "kill_switch": {"status": "clear"},
        },
    )
    write_json(
        live / "approved_config_snapshot.json",
        {
            "task_id": TASK_ID,
            "artifact_window_id": 1,
            "symbol": "BTC",
            "time_in_force": "Alo",
            "reduce_only": False,
            "live_mode": True,
            "max_order_size_btc": 0.005,
            "max_loss_usdc": 1.0,
            "max_position_btc": 0.01,
            "max_real_order_submissions": 2,
        },
    )
    write_json(live / "run_intent_marker.json", {"task_id": TASK_ID, "artifact_window_id": 1})
    write_json(
        live / "m2_fill_window_manifest.json",
        {
            "task_id": TASK_ID,
            "artifact_window_id": 1,
            "real_order_endpoint_called": True,
            "real_cancel_endpoint_called": True,
            "order_status_types": ["resting", "resting"],
            "shutdown_proof_status": "pass",
            "final_open_orders_count": 0,
            "fill_count": 0,
            "ledger_fill_rows": 0,
            "final_recommendation": "hyperliquid_tiny_live_m2_fill_window_blocked",
            "blocking_reasons": ["no_fill_observed"],
            "blocking_reason_classification": {"no_fill_observed": "economics_only"},
            "fill_reconciliation": {
                "status": "no_fill_reconciled",
                "mechanism_status": "pass",
                "economics_status": "no_fill_observed",
                "reasons": [],
                "cancel_reference_reconciliation": cancel_reference_reconciliation,
            },
            "cancel_reference_reconciliation": cancel_reference_reconciliation,
        },
    )
    write_json(live / "executor_manifest.json", {"task_id": TASK_ID, "artifact_window_id": 1})
    persisted_order_results = [
        watcher.persisted_order_result(
            {
                "status": "ok",
                "response": {
                    "data": {
                        "statuses": [
                            {
                                "resting": {
                                    "oid": raw_tracked_refs[index]["oid"],
                                    "cloid": raw_tracked_refs[index]["cloid"],
                                }
                            }
                        ]
                    }
                },
            }
        )
        for index in range(2)
    ]
    write_json(
        live / "private_order_response_audit.json",
        {
            "order_submission_attempted": True,
            "order_status_rows": [
                {"attempt": 1, "side": "buy", "status_type": "resting"},
                {"attempt": 2, "side": "sell", "status_type": "resting"},
            ],
            "order_response_rows": [
                {
                    "attempt": 1,
                    "attempt_id": 1,
                    "attempt_key": f"{TASK_ID}:window_01:attempt_1",
                    "side": "buy",
                    "intent_cloid_token": fill_window.reference_identity_token(
                        "cloid",
                        "cloid-buy",
                    ),
                    "result": persisted_order_results[0],
                },
                {
                    "attempt": 2,
                    "attempt_id": 2,
                    "attempt_key": f"{TASK_ID}:window_01:attempt_2",
                    "side": "sell",
                    "intent_cloid_token": fill_window.reference_identity_token(
                        "cloid",
                        "cloid-sell",
                    ),
                    "result": persisted_order_results[1],
                },
            ],
            "order_results": persisted_order_results,
        },
    )
    fill_manifest = json.loads(
        (live / "m2_fill_window_manifest.json").read_text(encoding="utf-8")
    )
    write_json(
        live / "cancel_shutdown_proof.json",
        {
            "proof_status": "pass",
            "tracked_refs": persisted_tracked_refs,
            "cancel_results": persisted_cancel_results,
            "cancel_reference_reconciliation": cancel_reference_reconciliation,
            "fill_reconciliation": fill_manifest["fill_reconciliation"],
        },
    )
    write_json(live / "account_inventory_snapshots.json", {"post_state": {"assetPositions": []}})
    write_json(live / "max_loss_monitor_summary.json", {"status": "pass", "estimated_loss_usdc": 0.0})
    write_json(
        live / "user_fills_pullback_audit.json",
        {
            "schema_version": (
                "redaction_safe_user_fill_pullback_audit_v1"
            ),
            "pullbacks": [
                {
                    "phase": "finalize",
                    "attempt": 2,
                    "start_ms": 900,
                    "end_ms": 2_000,
                    "observed_end_ms": 2_000,
                    "mark_px": 65_000.0,
                    "user_add_rate": 0.0,
                    "fill_count": 0,
                    "fills": [],
                    "raw_fill_evidence_schema_version": (
                        "redaction_safe_user_fill_pullback_v1"
                    ),
                }
            ],
            "pullback_count": 1,
            "raw_payload_redacted": True,
            "fill_attribution_summary": {
                "attributed_fill_count": 0,
                "unattributed_fill_count": 0,
                "attributed_qty_btc": 0,
                "attributed_fee_usdc": 0,
                "fail_closed_reasons": [],
            },
        },
    )
    write_csv(
        live / "quote_attempt_matrix.csv",
        fixture_attempt_rows,
        [
            "attempt",
            "attempt_id",
            "attempt_key",
            "event_sequence",
            "guard_status",
            "guard_reason",
            "edge_gate_status",
            "edge_gate_reason",
            "skip_reason",
            "side",
            "limit_px",
            "size_btc",
            "order_status_types",
            "order_endpoint_called",
            "cancel_endpoint_called",
        ],
    )
    write_csv(
        live / "order_intent_audit.csv",
        [
            {
                "attempt_id": 1,
                "attempt_key": f"{TASK_ID}:window_01:attempt_1",
                "side": "buy",
                "limit_px": "64000.0",
                "size_btc": "0.002",
                "time_in_force": "Alo",
                "cloid_token": fill_window.reference_identity_token(
                    "cloid",
                    "cloid-buy",
                ),
            },
            {
                "attempt_id": 2,
                "attempt_key": f"{TASK_ID}:window_01:attempt_2",
                "side": "sell",
                "limit_px": "66000.0",
                "size_btc": "0.002",
                "time_in_force": "Alo",
                "cloid_token": fill_window.reference_identity_token(
                    "cloid",
                    "cloid-sell",
                ),
            },
        ],
        [
            "attempt_id",
            "attempt_key",
            "side",
            "limit_px",
            "size_btc",
            "time_in_force",
            "cloid_token",
        ],
    )
    write_csv(live / "live_fill_ledger.csv", [], ["fill_id"])
    write_csv(live / "fill_attribution_evidence.csv", [], ["fill_id"])
    write_csv(live / "fill_liquidity_role_evidence.csv", [], ["fill_id"])
    seal_run(root)
    return root


def install_manager_resting_exposure_contract(
    input_root: Path,
) -> Path:
    window = input_root / "run" / "window_01"
    live = live_artifact_dir(input_root)
    base_ms = 1_783_600_000_000
    response_path = live / "private_order_response_audit.json"
    response = json.loads(response_path.read_text(encoding="utf-8"))
    for index, row in enumerate(response["order_response_rows"]):
        row["result"]["manager_actions"] = [
            {
                "action": "submitted",
                "state": "resting",
                "query_status": "resting",
                "order_endpoint_called": True,
                "side": row["side"],
                "submit_end_ms": base_ms + index * 10,
            }
        ]
    response["order_results"] = [
        row["result"] for row in response["order_response_rows"]
    ]
    write_json(response_path, response)

    proof_path = live / "cancel_shutdown_proof.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    for index, row in enumerate(proof["cancel_results"]):
        row["cancel_request_time_ms"] = base_ms + 2_000 + index * 10
    write_json(proof_path, proof)

    intents_by_side = {
        row["side"]: row
        for row in read_csv(live / "order_intent_audit.csv")
    }
    hold_observation = {
        "status": "pass",
        "reason": "",
        "deadline_overrun_seconds": 0.0,
        "reconnect_count_start": 0,
        "reconnect_count_end": 0,
        "disconnect_count_start": 0,
        "disconnect_count_end": 0,
    }
    interval_rows, rebuild_reasons = (
        acceptance.rebuild_manager_resting_interval_contract(
            order_response_rows=response["order_response_rows"],
            intents_by_side=intents_by_side,
            cancel_results=proof["cancel_results"],
            hold_observation=hold_observation,
        )
    )
    assert rebuild_reasons == []

    event_rows = [
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 100,
            "local_receive_time_ms": base_ms + 100,
            "bid_px": 65_000,
            "ask_px": 65_001,
            "bid_depth_btc": 1.0,
            "ask_depth_btc": 1.0,
        },
        {
            "event_kind": "trade",
            "event_time_ms": base_ms + 500,
            "local_receive_time_ms": base_ms + 500,
            "trade_px": 64_000,
            "trade_size_btc": 0.001,
            "aggressor_side": "sell",
            "trade_id": "manager-resting-fixture",
        },
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 1_100,
            "local_receive_time_ms": base_ms + 1_100,
            "bid_px": 65_000,
            "ask_px": 65_001,
            "bid_depth_btc": 1.0,
            "ask_depth_btc": 1.0,
        },
    ]
    exposure_rows, quarantine_rows, censor_rows = (
        online_estimators.build_confirmed_resting_exposure_rows(
            event_rows=event_rows,
            interval_rows=interval_rows,
        )
    )
    assert quarantine_rows == []

    write_csv(
        window / "online_estimator_event_rows.csv",
        event_rows,
        online_estimators.estimator_event_fieldnames(),
    )
    write_csv(
        window / "quote_exposure_intervals.csv",
        exposure_rows,
        online_estimators.quote_exposure_fieldnames(),
    )
    write_csv(
        window / "confirmed_resting_interval_contract.csv",
        interval_rows,
        watcher.manager_resting_interval_fieldnames(),
    )
    quarantine_path = (
        window / "confirmed_resting_exposure_quarantine.csv"
    )
    write_csv(
        quarantine_path,
        quarantine_rows,
        online_estimators.resting_exposure_quarantine_fieldnames(),
    )
    write_csv(
        window / "confirmed_resting_exposure_censor.csv",
        censor_rows,
        online_estimators.resting_exposure_censor_fieldnames(),
    )
    estimator_path = window / "online_estimator_snapshot.json"
    estimator = json.loads(estimator_path.read_text(encoding="utf-8"))
    estimator.update(
        {
            "bucket_ms": 1_000,
            "tick_size": 1.0,
            "max_future_skew_ms": 5_000,
            "manager_resting_exposure": {
                "interval_row_count": len(interval_rows),
                "confirmed_exposure_row_count": len(exposure_rows),
                "quarantine_row_count": len(quarantine_rows),
                "censor_row_count": len(censor_rows),
                "hold_observation": hold_observation,
            },
        }
    )
    write_json(estimator_path, estimator)
    return quarantine_path


def install_v3_terminal_query_proof(input_root: Path) -> None:
    live = live_artifact_dir(input_root)
    proof_path = live / "cancel_shutdown_proof.json"
    fill_path = live / "m2_fill_window_manifest.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    fill_manifest = json.loads(fill_path.read_text(encoding="utf-8"))
    cancel_results = list(proof["cancel_results"])
    cancel_results[1]["result"] = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {
                        "error": (
                            "Order was never placed, already canceled, "
                            "or filled. asset=0"
                        )
                    }
                ]
            }
        },
    }
    terminal_query_results = [
        {
            "attempt": 2,
            "method": "query_order_by_cloid",
            "cloid_token": proof["tracked_refs"][1]["cloid_token"],
            "query_status": "cancel_confirmed",
            "result": {"status": "canceled"},
        }
    ]
    final_open_orders: list[dict] = []
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=cancel_results,
        terminal_query_results=terminal_query_results,
        final_open_orders=final_open_orders,
    )
    assert reconciliation["status"] == "pass"
    assert acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=cancel_results,
        terminal_query_results=terminal_query_results,
        final_open_orders=final_open_orders,
    ) == reconciliation

    fill_reconciliation = dict(fill_manifest["fill_reconciliation"])
    fill_reconciliation[
        "cancel_reference_reconciliation"
    ] = reconciliation
    fill_manifest["fill_reconciliation"] = fill_reconciliation
    fill_manifest["cancel_reference_reconciliation"] = reconciliation
    proof.update(
        {
            "cancel_results": cancel_results,
            "terminal_query_results": terminal_query_results,
            "final_open_orders": final_open_orders,
            "cancel_reference_reconciliation": reconciliation,
            "fill_reconciliation": fill_reconciliation,
        }
    )
    write_json(fill_path, fill_manifest)
    write_json(proof_path, proof)


def install_v4_terminal_history_proof(
    input_root: Path,
    *,
    terminal_status: str = "canceled",
) -> None:
    live = live_artifact_dir(input_root)
    proof_path = live / "cancel_shutdown_proof.json"
    fill_path = live / "m2_fill_window_manifest.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    fill_manifest = json.loads(fill_path.read_text(encoding="utf-8"))
    cancel_results = list(proof["cancel_results"])
    cancel_results[1]["result"] = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {
                        "error": (
                            "Order was never placed, already canceled, "
                            "or filled. asset=0"
                        )
                    }
                ]
            }
        },
    }
    reference = proof["tracked_refs"][1]
    target = {
        "oid_token": reference["oid_token"],
        "cloid_token": reference["cloid_token"],
    }
    exact_order = {
        "oid": "<redacted>",
        "oid_token": target["oid_token"],
        "oid_alias_tokens": {"oid": target["oid_token"]},
        "cloid": "<redacted>",
        "cloid_token": target["cloid_token"],
        "cloid_alias_tokens": {"cloid": target["cloid_token"]},
    }
    foreign_oid_token = fill_window.reference_identity_token(
        "oid",
        999,
    )
    foreign_cloid_token = fill_window.reference_identity_token(
        "cloid",
        "foreign",
    )
    foreign_order = {
        "oid": "<redacted>",
        "oid_token": foreign_oid_token,
        "oid_alias_tokens": {"oid": foreign_oid_token},
        "cloid": "<redacted>",
        "cloid_token": foreign_cloid_token,
        "cloid_alias_tokens": {"cloid": foreign_cloid_token},
    }
    terminal_query_attempts: list[dict] = []
    for direct_round in range(1, 6):
        for method in (
            "query_order_by_oid",
            "query_order_by_cloid",
        ):
            sequence = len(terminal_query_attempts) + 1
            terminal_query_attempts.append(
                {
                    "attempt": 2,
                    "method": method,
                    **target,
                    "query_sequence": sequence,
                    "direct_round": direct_round,
                    "query_started_ms": 1_000 + sequence * 2,
                    "query_ended_ms": 1_001 + sequence * 2,
                    "query_status": "unknown",
                    "result": {"status": "unknownOid"},
                }
            )
    history = {
        "attempt": 2,
        "method": "historical_orders",
        **target,
        "query_sequence": 11,
        "query_started_ms": 1_022,
        "query_ended_ms": 1_023,
        "history_not_before_monotonic": 104.0,
        "query_started_monotonic": 104.1,
        "query_ended_monotonic": 104.2,
        "propagation_delay_satisfied": True,
        "query_status": (
            "rejected"
            if terminal_status == "badAloPxRejected"
            else "cancel_confirmed"
        ),
        "result": {
            "status": "historical_orders",
            "orders": [
                {
                    "order": foreign_order,
                    "status": "canceled",
                },
                {
                    "order": exact_order,
                    "status": terminal_status,
                },
            ],
        },
    }
    terminal_query_attempts.append(history)
    terminal_query_results = [
        {**history, "source_query_sequence": 11}
    ]
    terminal_query_budget = {
        "budget_seconds": 5.0,
        "retry_seconds": 0.25,
        "started_monotonic": 100.0,
        "ended_monotonic": 104.45,
        "elapsed_seconds": 4.45,
        "max_direct_rounds": 5,
        "direct_rounds_used": 5,
        "direct_query_attempt_count": 10,
        "historical_fallback_attempt_count": 1,
        "historical_fallback_max_calls_per_reference": 1,
        "historical_fallback_protocol_version": (
            acceptance.DELAYED_HISTORY_PROTOCOL_VERSION
        ),
        "historical_fallback_propagation_delay_seconds": 4.0,
        "historical_fallback_final_snapshot_reserve_seconds": 0.5,
        "historical_fallback_not_before_monotonic": 104.0,
        "historical_fallback_query_deadline_monotonic": 104.5,
        "historical_fallback_wait_started_monotonic": 101.0,
        "historical_fallback_wait_ended_monotonic": 104.0,
        "historical_fallback_planned_wait_seconds": 3.0,
        "historical_fallback_actual_wait_seconds": 3.0,
        "historical_fallback_deadline_remaining_before_calls_seconds": 1.0,
        "historical_fallback_call_started_after_not_before": True,
        "post_history_final_snapshot_started_monotonic": 104.3,
        "post_history_final_snapshot_ended_monotonic": 104.4,
        "post_history_final_snapshot_complete": True,
    }
    final_open_orders: list[dict] = []
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=cancel_results,
        terminal_query_results=terminal_query_results,
        terminal_query_attempts=terminal_query_attempts,
        terminal_query_budget=terminal_query_budget,
        terminal_query_contract_version="v4",
        final_open_orders=final_open_orders,
    )
    assert reconciliation["status"] == "pass"
    assert acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=cancel_results,
        terminal_query_results=terminal_query_results,
        terminal_query_attempts=terminal_query_attempts,
        terminal_query_budget=terminal_query_budget,
        terminal_query_contract_version="v4",
        final_open_orders=final_open_orders,
    ) == reconciliation

    fill_reconciliation = dict(fill_manifest["fill_reconciliation"])
    fill_reconciliation[
        "cancel_reference_reconciliation"
    ] = reconciliation
    fill_manifest["fill_reconciliation"] = fill_reconciliation
    fill_manifest["cancel_reference_reconciliation"] = reconciliation
    proof.update(
        {
            "cancel_results": cancel_results,
            "terminal_query_results": terminal_query_results,
            "terminal_query_attempts": terminal_query_attempts,
            "terminal_query_budget": terminal_query_budget,
            "terminal_query_contract_version": "v4",
            "final_open_orders": final_open_orders,
            "cancel_reference_reconciliation": reconciliation,
            "fill_reconciliation": fill_reconciliation,
        }
    )
    write_json(fill_path, fill_manifest)
    write_json(proof_path, proof)


def install_delayed_v4_terminal_history_proof(
    input_root: Path,
) -> None:
    install_v4_terminal_history_proof(input_root)
    live = live_artifact_dir(input_root)
    proof_path = live / "cancel_shutdown_proof.json"
    fill_path = live / "m2_fill_window_manifest.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    fill_manifest = json.loads(fill_path.read_text(encoding="utf-8"))
    terminal_query_attempts = proof["terminal_query_attempts"]
    terminal_query_attempts[-1].update(
        {
            "history_not_before_monotonic": 104.0,
            "query_started_monotonic": 104.1,
            "query_ended_monotonic": 104.2,
            "propagation_delay_satisfied": True,
        }
    )
    terminal_query_results = [
        {
            **terminal_query_attempts[-1],
            "source_query_sequence": 11,
        }
    ]
    terminal_query_budget = proof["terminal_query_budget"]
    terminal_query_budget.update(
        {
            "historical_fallback_protocol_version": (
                acceptance.DELAYED_HISTORY_PROTOCOL_VERSION
            ),
            "historical_fallback_propagation_delay_seconds": (
                acceptance.DELAYED_HISTORY_PROPAGATION_DELAY_SECONDS
            ),
            "historical_fallback_final_snapshot_reserve_seconds": (
                acceptance.DELAYED_HISTORY_FINAL_SNAPSHOT_RESERVE_SECONDS
            ),
            "historical_fallback_not_before_monotonic": 104.0,
            "historical_fallback_query_deadline_monotonic": 104.5,
            "historical_fallback_wait_started_monotonic": 101.0,
            "historical_fallback_wait_ended_monotonic": 104.0,
            "historical_fallback_planned_wait_seconds": 3.0,
            "historical_fallback_actual_wait_seconds": 3.0,
            "historical_fallback_deadline_remaining_before_calls_seconds": 1.0,
            "historical_fallback_call_started_after_not_before": True,
            "post_history_final_snapshot_started_monotonic": 104.3,
            "post_history_final_snapshot_ended_monotonic": 104.4,
            "ended_monotonic": 104.45,
            "elapsed_seconds": 4.45,
        }
    )
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
        terminal_query_results=terminal_query_results,
        terminal_query_attempts=terminal_query_attempts,
        terminal_query_budget=terminal_query_budget,
        terminal_query_contract_version="v4",
        final_open_orders=proof["final_open_orders"],
    )
    assert reconciliation["status"] == "pass"
    assert acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
        terminal_query_results=terminal_query_results,
        terminal_query_attempts=terminal_query_attempts,
        terminal_query_budget=terminal_query_budget,
        terminal_query_contract_version="v4",
        final_open_orders=proof["final_open_orders"],
    ) == reconciliation

    fill_reconciliation = dict(fill_manifest["fill_reconciliation"])
    fill_reconciliation[
        "cancel_reference_reconciliation"
    ] = reconciliation
    fill_manifest["fill_reconciliation"] = fill_reconciliation
    fill_manifest["cancel_reference_reconciliation"] = reconciliation
    proof.update(
        {
            "terminal_query_results": terminal_query_results,
            "terminal_query_attempts": terminal_query_attempts,
            "terminal_query_budget": terminal_query_budget,
            "cancel_reference_reconciliation": reconciliation,
            "fill_reconciliation": fill_reconciliation,
        }
    )
    write_json(fill_path, fill_manifest)
    write_json(proof_path, proof)


def install_v4_filled_terminal_query_proof(input_root: Path) -> None:
    live = live_artifact_dir(input_root)
    proof_path = live / "cancel_shutdown_proof.json"
    fill_path = live / "m2_fill_window_manifest.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    fill_manifest = json.loads(fill_path.read_text(encoding="utf-8"))
    terminal_query_attempts: list[dict] = []
    terminal_query_results: list[dict] = []
    for sequence, reference in enumerate(
        proof["tracked_refs"],
        start=1,
    ):
        target = {
            "oid_token": reference["oid_token"],
            "cloid_token": reference["cloid_token"],
        }
        attempt = {
            "attempt": sequence,
            "method": "query_order_by_oid",
            **target,
            "query_sequence": sequence,
            "direct_round": 1,
            "query_started_ms": 1_000 + sequence * 2,
            "query_ended_ms": 1_001 + sequence * 2,
            "query_status": "filled",
            "result": {
                "status": "order",
                "order": {
                    "order": dict(target),
                    "status": "filled",
                },
            },
        }
        terminal_query_attempts.append(attempt)
        terminal_query_results.append(
            {**attempt, "source_query_sequence": sequence}
        )
    terminal_query_budget = {
        "budget_seconds": 5.0,
        "retry_seconds": 0.25,
        "started_monotonic": 100.0,
        "ended_monotonic": 100.5,
        "elapsed_seconds": 0.5,
        "max_direct_rounds": 5,
        "direct_rounds_used": 1,
        "direct_query_attempt_count": 2,
        "historical_fallback_attempt_count": 0,
        "historical_fallback_max_calls_per_reference": 1,
    }
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
        terminal_query_results=terminal_query_results,
        terminal_query_attempts=terminal_query_attempts,
        terminal_query_budget=terminal_query_budget,
        terminal_query_contract_version="v4",
        final_open_orders=[],
    )
    assert reconciliation["status"] == "fail_closed"
    assert "terminal_query_filled_requires_complete_fill_proof" in (
        reconciliation["reasons"]
    )
    assert acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
        terminal_query_results=terminal_query_results,
        terminal_query_attempts=terminal_query_attempts,
        terminal_query_budget=terminal_query_budget,
        terminal_query_contract_version="v4",
        final_open_orders=[],
    ) == reconciliation

    proof.update(
        {
            "terminal_query_results": terminal_query_results,
            "terminal_query_attempts": terminal_query_attempts,
            "terminal_query_budget": terminal_query_budget,
            "terminal_query_contract_version": "v4",
            "final_open_orders": [],
            "cancel_reference_reconciliation": reconciliation,
        }
    )
    fill_manifest["cancel_reference_reconciliation"] = reconciliation
    write_json(proof_path, proof)
    write_json(fill_path, fill_manifest)


def synchronize_v3_cancel_reconciliation(input_root: Path) -> dict:
    live = live_artifact_dir(input_root)
    proof_path = live / "cancel_shutdown_proof.json"
    fill_path = live / "m2_fill_window_manifest.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    fill_manifest = json.loads(fill_path.read_text(encoding="utf-8"))
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
        terminal_query_results=proof["terminal_query_results"],
        final_open_orders=proof["final_open_orders"],
    )
    fill_reconciliation = dict(proof["fill_reconciliation"])
    fill_reconciliation[
        "cancel_reference_reconciliation"
    ] = reconciliation
    proof["cancel_reference_reconciliation"] = reconciliation
    proof["fill_reconciliation"] = fill_reconciliation
    fill_manifest["cancel_reference_reconciliation"] = reconciliation
    fill_manifest["fill_reconciliation"] = fill_reconciliation
    write_json(proof_path, proof)
    write_json(fill_path, fill_manifest)
    return reconciliation


def install_submit_rejected_terminal_proof(input_root: Path) -> None:
    live = live_artifact_dir(input_root)
    private_path = live / "private_order_response_audit.json"
    proof_path = live / "cancel_shutdown_proof.json"
    fill_path = live / "m2_fill_window_manifest.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    fill_manifest = json.loads(fill_path.read_text(encoding="utf-8"))

    buy_response = private["order_response_rows"][0]
    buy_cloid_token = buy_response["intent_cloid_token"]
    buy_result = {
        "status": "ok",
        "side": "buy",
        "response": {
            "type": "order",
            "data": {
                "statuses": [
                    {
                        "error": (
                            "Post only order would have immediately matched"
                        )
                    }
                ]
            },
        },
        "manager_actions": [
            {
                "action": "rejected",
                "state": "rejected",
                "query_status": "rejected",
                "order_endpoint_called": True,
                "side": "buy",
            }
        ],
    }
    buy_response["result"] = buy_result
    private["order_results"][0] = buy_result
    private["order_status_rows"][0] = {
        "attempt": 1,
        "side": "buy",
        "status_type": "rejected",
        "payload": "Post only order would have immediately matched",
    }

    proof["tracked_refs"][0] = {
        "attempt": 1,
        "cloid": "<redacted>",
        "cloid_token": buy_cloid_token,
    }
    proof["cancel_results"] = [
        row
        for row in proof["cancel_results"]
        if row.get("attempt") == 2
    ]
    reconciliation = fill_window.cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
        submit_terminal_results=private["order_response_rows"],
    )
    assert reconciliation["status"] == "pass"
    assert reconciliation == (
        acceptance.rebuild_raw_cancel_reference_reconciliation(
            tracked_refs=proof["tracked_refs"],
            cancel_results=proof["cancel_results"],
            submit_terminal_results=private["order_response_rows"],
        )
    )
    fill_reconciliation = dict(fill_manifest["fill_reconciliation"])
    fill_reconciliation.update(
        {
            "status": "no_fill_reconciled",
            "mechanism_status": "pass",
            "economics_status": "no_fill_observed",
            "reasons": [],
            "cancel_reference_reconciliation": reconciliation,
        }
    )
    fill_manifest.update(
        {
            "order_status_types": ["rejected", "resting"],
            "post_only_reject_count": 1,
            "blocking_reasons": ["no_fill_observed"],
            "blocking_reason_classification": {
                "no_fill_observed": "economics_only"
            },
            "fill_reconciliation": fill_reconciliation,
            "cancel_reference_reconciliation": reconciliation,
        }
    )
    proof["fill_reconciliation"] = fill_reconciliation
    proof["cancel_reference_reconciliation"] = reconciliation
    write_json(private_path, private)
    write_json(proof_path, proof)
    write_json(fill_path, fill_manifest)


def sync_producer_decision_evidence(input_root: Path) -> dict:
    window = input_root / "run" / "window_01"
    live = live_artifact_dir(input_root)
    trigger_rows = read_csv(
        window / "event_driven_trigger_decision_matrix.csv"
    )
    guard_rows = read_csv(
        window / "immediate_pre_submit_guard_matrix.csv"
    )
    anti_drift_rows = read_csv(
        window / "anti_drift_gate_matrix.csv"
    )
    edge_rows = read_csv(window / "edge_gate_matrix.csv")
    attempt_rows = read_csv(live / "quote_attempt_matrix.csv")
    inline_path = window / "inline_reprice_manifest.json"
    inline_manifest = json.loads(
        inline_path.read_text(encoding="utf-8")
    )
    summary = watcher.build_event_driven_decision_evidence_summary(
        trigger_rows=trigger_rows,
        guard_rows=guard_rows,
        anti_drift_rows=anti_drift_rows,
        edge_gate_rows=edge_rows,
        attempt_rows=attempt_rows,
        endpoint_flags=inline_manifest,
    )
    watcher_path = window / "event_driven_watcher_manifest.json"
    watcher_manifest = json.loads(
        watcher_path.read_text(encoding="utf-8")
    )
    watcher_manifest.update(
        {
            "event_driven_evaluation_count": len(trigger_rows),
            "trigger_found": summary["trigger_row_count"] > 0,
            "trigger_count": summary["trigger_row_count"],
            "anti_drift_pass_count": summary[
                "anti_drift_gate_pass_count"
            ],
            "anti_drift_block_count": summary[
                "anti_drift_gate_block_count"
            ],
            "edge_gate_pass_count": summary["edge_gate_pass_count"],
            "edge_gate_block_count": summary[
                "edge_gate_block_count"
            ],
            "live_submissions_count": summary[
                "submitted_attempt_count"
            ],
            "candidate_attempt_evidence_row_count": summary[
                "candidate_attempt_evidence_row_count"
            ],
            "manager_attempt_identity_count": summary[
                "manager_attempt_identity_count"
            ],
            "submitted_attempt_count": summary[
                "submitted_attempt_count"
            ],
            "decision_evidence_summary": summary,
        }
    )
    inline_manifest.update(
        {
            "requote_attempts_completed": summary[
                "submitted_attempt_count"
            ],
            "candidate_attempt_evidence_row_count": summary[
                "candidate_attempt_evidence_row_count"
            ],
            "manager_attempt_identity_count": summary[
                "manager_attempt_identity_count"
            ],
            "decision_evidence_summary": summary,
        }
    )
    write_json(watcher_path, watcher_manifest)
    write_json(inline_path, inline_manifest)
    write_json(
        window / "event_driven_decision_evidence_summary.json",
        summary,
    )
    return summary


def write_sealed_command(input_root: Path, command: list[str]) -> None:
    write_json(
        input_root / "preflight" / "orchestrator_preflight.json",
        {
            **json.loads(
                (
                    input_root
                    / "preflight"
                    / "orchestrator_preflight.json"
                ).read_text(encoding="utf-8")
            ),
            "watcher_commands": [command],
        },
    )
    write_json(
        input_root / "run" / "window_01" / "runner_command.json",
        {"command": command},
    )


def write_fully_sealed_command(
    input_root: Path,
    command: list[str],
) -> None:
    write_sealed_command(input_root, command)
    provenance_path = (
        input_root
        / "run"
        / acceptance.RUNTIME_SOURCE_PROVENANCE_NAME
    )
    provenance = json.loads(
        provenance_path.read_text(encoding="utf-8")
    )
    provenance["watcher_commands"] = [command]
    provenance["python_executable"] = command[0]
    provenance["watcher_command_script"] = command[1]
    write_json(provenance_path, provenance)


def set_filled_lifecycle(
    input_root: Path,
    *,
    filled_attempts: tuple[int, ...],
    cancel_success: bool,
    unrelated_cancel: bool = False,
) -> None:
    live = live_artifact_dir(input_root)
    private_path = live / "private_order_response_audit.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    intents = {
        int(row["attempt_id"]): row
        for row in read_csv(live / "order_intent_audit.csv")
    }
    responses = {
        int(row["attempt_id"]): row
        for row in private["order_response_rows"]
    }
    fill_ledger = fill_window.LiveFillLedger(
        task_id=TASK_ID,
        window_id=1,
    )
    raw_fills: list[dict[str, object]] = []
    for attempt in filled_attempts:
        intent = intents[attempt]
        cloid = "cloid-buy" if attempt == 1 else "cloid-sell"
        fill_ledger.register_attempt(
            attempt_id=attempt,
            intent=executor.OrderIntent(
                symbol="BTC",
                is_buy=intent["side"] == "buy",
                size_btc=float(intent["size_btc"]),
                limit_px=float(intent["limit_px"]),
                time_in_force="Alo",
                reduce_only=False,
                cloid=cloid,
            ),
            submit_start_ms=900,
            submit_end_ms=1_000,
            tracked_refs=[
                {
                    "oid": 100 + attempt,
                    "cloid": cloid,
                }
            ],
            terminal_end_ms=2_000,
        )
        raw_fills.append(
            {
                "fillId": f"fill-{attempt}",
                "coin": "BTC",
                "oid": 100 + attempt,
                "cloid": cloid,
                "side": "B" if intent["side"] == "buy" else "A",
                "sz": intent["size_btc"],
                "px": intent["limit_px"],
                "fee": "0",
                "time": 1_000 + attempt,
                "crossed": False,
            }
        )
    fill_ledger.ingest(
        fills=raw_fills,
        mark_px=65_000.0,
        user_add_rate=0.0,
        pullback_phase="finalize",
        observed_end_ms=2_000,
    )
    fill_rows = fill_ledger.attributed_rows()
    attribution_rows = fill_ledger.evidence_rows()
    role_rows = fill_window.fill_liquidity_role_evidence_rows(
        fill_rows
    )
    write_csv(
        live / "live_fill_ledger.csv",
        fill_rows,
        fill_window.live_fill_ledger_fieldnames(),
    )
    write_csv(
        live / "fill_attribution_evidence.csv",
        attribution_rows,
        fill_window.fill_attribution_evidence_fieldnames(),
    )
    write_csv(
        live / "fill_liquidity_role_evidence.csv",
        role_rows,
        fill_window.fill_liquidity_role_evidence_fieldnames(),
    )
    write_json(
        live / "user_fills_pullback_audit.json",
        {
            "schema_version": (
                "redaction_safe_user_fill_pullback_audit_v1"
            ),
            "pullbacks": fill_window.persisted_user_fill_pullbacks(
                [
                    {
                        "phase": "finalize",
                        "attempt": 2,
                        "start_ms": 900,
                        "end_ms": 2_000,
                        "observed_end_ms": 2_000,
                        "mark_px": 65_000.0,
                        "user_add_rate": 0.0,
                        "fill_count": len(raw_fills),
                        "fills": raw_fills,
                    }
                ]
            ),
            "pullback_count": 1,
            "raw_payload_redacted": True,
            "fill_attribution_summary": fill_ledger.summary(),
        },
    )

    proof_path = live / "cancel_shutdown_proof.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if not cancel_success:
        for row in proof["cancel_results"]:
            row["result"] = {
                "status": "ok",
                "response": {
                    "data": {
                        "statuses": [
                            {"error": "already_filled"}
                        ]
                    }
                },
            }
    if unrelated_cancel:
        proof["cancel_results"][0]["oid_token"] = (
            fill_window.reference_identity_token("oid", 999)
        )
        proof["cancel_results"][0]["cloid_token"] = (
            fill_window.reference_identity_token(
                "cloid",
                "unrelated",
            )
        )
    terminal = fill_window.cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
    )
    fill_reconciliation = {
        "status": "not_applicable_fill_observed",
        "mechanism_status": "not_applicable",
        "economics_status": "fill_observed",
        "reasons": [],
    }
    proof["cancel_reference_reconciliation"] = terminal
    proof["fill_reconciliation"] = fill_reconciliation
    write_json(proof_path, proof)

    fill_manifest_path = live / "m2_fill_window_manifest.json"
    fill_manifest = json.loads(
        fill_manifest_path.read_text(encoding="utf-8")
    )
    fill_manifest.update(
        {
            "fill_count": len(fill_rows),
            "maker_fill_count": len(fill_rows),
            "ledger_fill_rows": len(fill_rows),
            "blocking_reasons": [],
            "blocking_reason_classification": {},
            "final_recommendation": (
                "hyperliquid_tiny_live_m2_fill_window_ready_for_qa"
            ),
            "fill_reconciliation": fill_reconciliation,
            "cancel_reference_reconciliation": terminal,
        }
    )
    write_json(fill_manifest_path, fill_manifest)

    watcher_path = (
        input_root
        / "run"
        / "window_01"
        / "event_driven_watcher_manifest.json"
    )
    watcher_manifest = json.loads(
        watcher_path.read_text(encoding="utf-8")
    )
    watcher_manifest["fill_count"] = len(fill_rows)
    watcher_manifest["maker_fill_count"] = len(fill_rows)
    write_json(watcher_path, watcher_manifest)
    seal_run(input_root)


def set_raw_fill_direction(
    input_root: Path,
    *,
    attempt: int,
    direction: str,
    explicit_side: str | None = None,
) -> None:
    live = live_artifact_dir(input_root)
    pullback_path = live / "user_fills_pullback_audit.json"
    pullback = json.loads(
        pullback_path.read_text(encoding="utf-8")
    )
    raw_fill = next(
        fill
        for fill in pullback["pullbacks"][0]["fills"]
        if fill.get("fillId") == f"fill-{attempt}"
    )
    if explicit_side is None:
        raw_fill.pop("side", None)
    else:
        raw_fill["side"] = explicit_side
    raw_fill["dir"] = direction
    fingerprint = fill_window.fill_payload_fingerprint(raw_fill)
    write_json(pullback_path, pullback)

    for filename in (
        "live_fill_ledger.csv",
        "fill_attribution_evidence.csv",
    ):
        path = live / filename
        rows = read_csv(path)
        row = next(
            row
            for row in rows
            if int(row["attempt_id"]) == attempt
        )
        row["fill_payload_fingerprint"] = fingerprint
        write_csv(
            path,
            rows,
            (
                fill_window.live_fill_ledger_fieldnames()
                if filename == "live_fill_ledger.csv"
                else fill_window.fill_attribution_evidence_fieldnames()
            ),
        )
    seal_run(input_root)


class _ManagerWatcherClient:
    def __init__(self) -> None:
        self.order_intents: list[executor.OrderIntent] = []
        self.cancel_calls: list[dict[str, object]] = []
        self.account_address = (
            "0x0000000000000000000000000000000000000000"
        )

    def open_orders(self, address: str | None = None) -> list[dict]:
        return []

    def order(self, intent: executor.OrderIntent) -> dict:
        self.order_intents.append(intent)
        oid = 300 + len(self.order_intents)
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "resting": {
                                "oid": oid,
                                "cloid": intent.cloid,
                            }
                        }
                    ]
                }
            },
        }

    def cancel_tracked(
        self,
        symbol: str,
        oid: int | None = None,
        cloid: str | None = None,
    ) -> dict:
        self.cancel_calls.append(
            {"symbol": symbol, "oid": oid, "cloid": cloid}
        )
        return {
            "status": "ok",
            "response": {
                "data": {"statuses": [{"success": str(oid or cloid)}]}
            },
        }

    def user_fills_by_time(
        self,
        account: str | None,
        start_ms: int,
        end_ms: int,
        aggregate_by_time: bool = False,
    ) -> list[dict]:
        return []

    def user_fees(self, account: str | None = None) -> dict:
        return {"userAddRate": 0.0}

    def user_state(self, address: str | None = None) -> dict:
        return {"assetPositions": []}


def _manager_l2(ts_ms: int) -> dict:
    return {
        "channel": "l2Book",
        "data": {
            "coin": "BTC",
            "time": ts_ms,
            "levels": [
                [{"px": "65000", "sz": "0.02", "n": 4}],
                [{"px": "65001", "sz": "1.0", "n": 8}],
            ],
        },
    }


def _manager_trade(ts_ms: int) -> dict:
    return {
        "channel": "trades",
        "data": [
            {
                "coin": "BTC",
                "time": ts_ms,
                "px": "64999",
                "sz": "0.04",
                "side": "A",
                "tid": ts_ms,
            }
        ],
    }


def _manager_source(messages: list[dict]):
    for message in messages:
        yield time.time_ns(), message


def write_actual_two_sided_live_artifacts(
    input_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _ManagerWatcherClient:
    window = input_root / "run" / "window_01"
    control_dir = input_root / "control"
    executor.initialize_control_state(control_dir)
    client = _ManagerWatcherClient()
    now_ms = int(time.time() * 1000)
    real_sleep = time.sleep

    def manager_source_with_hold_events():
        yield from _manager_source(
            [
                _manager_l2(now_ms),
                _manager_l2(now_ms + 1),
                _manager_trade(now_ms + 2),
                _manager_l2(now_ms + 3),
            ]
        )
        for index in range(100):
            real_sleep(0.05)
            message = _manager_l2(int(time.time() * 1000))
            message["data"]["levels"][0][0]["sz"] = str(
                0.02 + (index % 2) * 0.001
            )
            yield time.time_ns(), message

    monkeypatch.setattr(watcher.time, "sleep", lambda _: None)

    watcher.run_event_driven_inline_reprice_live(
        output_dir=window,
        watcher_seconds=900,
        env_file=str(input_root / ".env"),
        wait_seconds=10,
        quote_hold_seconds=3,
        requote_attempts=2,
        max_order_size_btc=0.005,
        max_real_order_submissions=2,
        artifact_task_id=TASK_ID,
        artifact_window_id=1,
        run_id=f"{TASK_ID}:window_01",
        use_exchange_reconciled_manager=True,
        hyperliquid_l2book_fast=True,
        edge_gate=True,
        binance_public_state_provider=lambda: {
            "symbol": "BTCUSDT",
            "binance_bid_px": 65020.0,
            "binance_ask_px": 65021.0,
            "signal_ts_ms": int(time.time() * 1000),
            "lead_move_ticks": 10.5,
            "tick_size": 1.0,
            "public_state_seq": 42,
            "source": "t007_actual_manager_watcher",
        },
        event_source_fn=manager_source_with_hold_events,
        live_client_factory=lambda: client,
        control_state_dir=control_dir,
        max_loss_usdc=1.0,
        max_position_btc=0.01,
    )
    seal_run(input_root)
    return client


def test_acceptance_passes_exact_no_fill_lifecycle(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"
    assert manifest["economics_boundary_acceptance"] == "pass"
    assert manifest["live_summary"]["fill_count"] == 0
    assert manifest["multi_level_activation_unlocked"] is False


def test_acceptance_passes_submit_reject_and_resting_cancel_lifecycle(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_submit_rejected_terminal_proof(input_root)
    seal_run(input_root)

    output_dir = tmp_path / "out"
    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=output_dir,
    )
    lifecycle_rows = read_csv(
        output_dir / "lifecycle_evidence_comparison.csv"
    )

    assert manifest["final_recommendation"] == (
        acceptance.PASSED_RECOMMENDATION
    )
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"
    assert all(row["acceptance"] == "pass" for row in lifecycle_rows)
    assert next(
        row
        for row in lifecycle_rows
        if row["check"] == "post_only_reject_count"
    )["observed"] == "1"


def test_acceptance_comparison_csvs_are_hash_seed_deterministic(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    output_dirs = [tmp_path / "seed1", tmp_path / "seed7"]
    for seed, output_dir in zip(("1", "7"), output_dirs, strict=True):
        command = [
            sys.executable,
            str(Path(acceptance.__file__)),
            "--input-root",
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--expected-task-id",
            TASK_ID,
            "--expected-source-commit",
            SOURCE_COMMIT,
            "--expected-remote-run-root",
            REMOTE_RUN_ROOT,
            "--expected-window-seconds",
            str(acceptance.DEFAULT_EXPECTED_WINDOW_SECONDS),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert completed.returncode == 0, (
            completed.stdout + completed.stderr
        )

    comparison_names = [
        "provenance_identity_comparison.csv",
        "config_control_comparison.csv",
        "decision_replay_comparison.csv",
        "lifecycle_evidence_comparison.csv",
        "economics_boundary_matrix.csv",
        "optimism_check_matrix.csv",
    ]
    for name in comparison_names:
        assert (output_dirs[0] / name).read_bytes() == (
            output_dirs[1] / name
        ).read_bytes()


def test_acceptance_passes_reference_bound_terminal_query_contract(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v3_terminal_query_proof(input_root)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.PASSED_RECOMMENDATION
    )
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"


def test_acceptance_passes_exact_v4_terminal_history_contract(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v4_terminal_history_proof(input_root)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.PASSED_RECOMMENDATION
    )
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"


def test_acceptance_passes_exact_v4_terminal_rejection_contract(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v4_terminal_history_proof(
        input_root,
        terminal_status="badAloPxRejected",
    )
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.PASSED_RECOMMENDATION
    )
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"


def test_acceptance_rejects_partial_v4_contract_after_reseal(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v4_terminal_history_proof(input_root)
    proof_path = (
        live_artifact_dir(input_root) / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.pop("terminal_query_budget")
    write_json(proof_path, proof)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_rejects_history_downgrade_without_v4_audit(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v4_terminal_history_proof(input_root)
    proof_path = (
        live_artifact_dir(input_root) / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.pop("terminal_query_attempts")
    proof.pop("terminal_query_budget")
    write_json(proof_path, proof)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_rollout_task_forces_v4_after_direct_only_fields_are_removed() -> None:
    target = {
        "oid_token": fill_window.reference_identity_token("oid", 101),
        "cloid_token": fill_window.reference_identity_token("cloid", "a"),
    }
    direct_result = {
        "attempt": 1,
        "method": "query_order_by_oid",
        **target,
        "query_status": "cancel_confirmed",
        "result": {
            "status": "order",
            "order": {
                "order": dict(target),
                "status": "canceled",
            },
        },
    }

    assert acceptance.bounded_terminal_query_required("0720T022") is False
    assert acceptance.bounded_terminal_query_required("0720T023") is True
    assert acceptance.delayed_history_required("0720T032") is False
    assert acceptance.delayed_history_required("0720T033") is True
    assert acceptance.manager_resting_evidence_required("0720T026") is False
    assert acceptance.manager_resting_evidence_required("0720T031") is True
    reconciliation = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, **target}],
        cancel_results=[
            {
                "attempt": 1,
                **target,
                "result": {
                    "status": "ok",
                    "response": {
                        "data": {
                            "statuses": [
                                {
                                    "error": (
                                        "Order was never placed, already "
                                        "canceled, or filled. asset=0"
                                    )
                                }
                            ]
                        }
                    },
                },
            }
        ],
        terminal_query_results=[direct_result],
        final_open_orders=[],
        require_bounded_contract=True,
    )

    assert reconciliation["schema_version"] == (
        acceptance.RAW_CANCEL_BOUNDED_TERMINAL_QUERY_RECONCILIATION_SCHEMA_VERSION
    )
    assert reconciliation["status"] == "fail_closed"
    assert "terminal_query_v4_contract_incomplete" in (
        reconciliation["reasons"]
    )


def test_task12_rejects_zero_delay_relabel_for_new_tasks(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_delayed_v4_terminal_history_proof(input_root)
    seal_run(input_root)

    run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "valid",
        expected_task_id="0721T033",
    )
    valid_rows = read_csv(
        tmp_path / "valid" / "lifecycle_evidence_comparison.csv"
    )
    valid_input_check = next(
        row
        for row in valid_rows
        if row["check"] == "raw_cancel_proof_inputs_valid"
    )
    assert valid_input_check["acceptance"] == "pass"

    proof_path = (
        live_artifact_dir(input_root) / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["terminal_query_budget"][
        "historical_fallback_propagation_delay_seconds"
    ] = 0.0
    write_json(proof_path, proof)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "zero-delay",
        expected_task_id="0721T033",
    )
    invalid_rows = read_csv(
        tmp_path / "zero-delay" / "lifecycle_evidence_comparison.csv"
    )
    invalid_input_check = next(
        row
        for row in invalid_rows
        if row["check"] == "raw_cancel_proof_inputs_valid"
    )

    assert invalid_input_check["acceptance"] == "fail"
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )


@pytest.mark.parametrize("marker", [None, "", "legacy", True])
def test_task12_rejects_missing_or_wrong_delayed_protocol_marker(
    tmp_path: Path,
    marker: object,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v4_terminal_history_proof(input_root)
    proof_path = (
        live_artifact_dir(input_root) / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if marker is None:
        proof["terminal_query_budget"].pop(
            "historical_fallback_protocol_version"
        )
    else:
        proof["terminal_query_budget"][
            "historical_fallback_protocol_version"
        ] = marker
    write_json(proof_path, proof)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id="0721T034",
    )
    lifecycle_rows = read_csv(
        tmp_path / "out" / "lifecycle_evidence_comparison.csv"
    )
    raw_input_check = next(
        row
        for row in lifecycle_rows
        if row["check"] == "raw_cancel_proof_inputs_valid"
    )

    assert raw_input_check["acceptance"] == "fail"
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )


def test_task12_rejects_direct_method_with_nested_history(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v4_terminal_history_proof(input_root)
    proof_path = (
        live_artifact_dir(input_root) / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["terminal_query_attempts"][-1]["method"] = (
        "query_order_by_cloid"
    )
    proof["terminal_query_results"][0]["method"] = (
        "query_order_by_cloid"
    )
    audit = acceptance.rebuild_raw_terminal_query_attempt_audit(
        tracked_refs=proof["tracked_refs"],
        terminal_query_results=proof["terminal_query_results"],
        terminal_query_attempts=proof["terminal_query_attempts"],
        terminal_query_budget=proof["terminal_query_budget"],
    )
    assert "terminal_audit_query_method_result_mismatch" in (
        audit["reasons"]
    )
    write_json(proof_path, proof)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id="0721T035",
    )
    lifecycle_rows = read_csv(
        tmp_path / "out" / "lifecycle_evidence_comparison.csv"
    )
    raw_input_check = next(
        row
        for row in lifecycle_rows
        if row["check"] == "raw_cancel_proof_inputs_valid"
    )
    independent_rebuild_check = next(
        row
        for row in lifecycle_rows
        if row["check"]
        == "producer_reconciliation_matches_independent_raw_proof"
    )

    assert raw_input_check["acceptance"] == "pass"
    assert independent_rebuild_check["acceptance"] == "fail"
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )


@pytest.mark.parametrize("method_value", [[], {}])
def test_task12_container_method_returns_blocked_manifest(
    tmp_path: Path,
    method_value: object,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v4_terminal_history_proof(input_root)
    proof_path = (
        live_artifact_dir(input_root) / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["terminal_query_attempts"][-1]["method"] = method_value
    proof["terminal_query_results"][0]["method"] = method_value
    write_json(proof_path, proof)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id="0721T036",
    )

    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )


def test_task12_rejects_malformed_historical_result_envelope(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v4_terminal_history_proof(input_root)
    proof_path = (
        live_artifact_dir(input_root) / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    malformed_result = {"status": "historical_orders"}
    proof["terminal_query_attempts"][-1]["result"] = malformed_result
    proof["terminal_query_attempts"][-1]["query_status"] = "unknown"
    proof["terminal_query_results"][0]["result"] = malformed_result
    proof["terminal_query_results"][0]["query_status"] = "unknown"
    audit = acceptance.rebuild_raw_terminal_query_attempt_audit(
        tracked_refs=proof["tracked_refs"],
        terminal_query_results=proof["terminal_query_results"],
        terminal_query_attempts=proof["terminal_query_attempts"],
        terminal_query_budget=proof["terminal_query_budget"],
    )
    assert "terminal_audit_query_method_result_mismatch" in (
        audit["reasons"]
    )
    write_json(proof_path, proof)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id="0721T036",
    )

    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )


def test_validation_report_title_uses_expected_task_id(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    output_dir = tmp_path / "out"

    run_task12_acceptance(
        input_root=input_root,
        output_dir=output_dir,
        expected_task_id="0720T023",
    )

    report = (output_dir / "validation_report.md").read_text(
        encoding="utf-8"
    )
    assert report.splitlines()[0] == "# 0720T023 Same-Window Acceptance"


def test_acceptance_rejects_keyword_forged_terminal_query_after_reseal(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v3_terminal_query_proof(input_root)
    proof_path = (
        live_artifact_dir(input_root) / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["terminal_query_results"][0]["result"] = {
        "status": "ok",
        "note": "order canceled",
    }
    write_json(proof_path, proof)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_blocks_non_string_terminal_status_without_raising(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    install_v3_terminal_query_proof(input_root)
    proof_path = (
        live_artifact_dir(input_root) / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["terminal_query_results"][0].update(
        {
            "query_status": "unknown",
            "result": {"status": []},
        }
    )
    write_json(proof_path, proof)
    reconciliation = synchronize_v3_cancel_reconciliation(input_root)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert reconciliation["status"] == "fail_closed"
    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_passes_externally_authorized_1800_second_duration(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(
        tmp_path / "input",
        window_seconds=1800.0,
    )

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_window_seconds=1800.0,
    )

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"
    assert manifest["expected_window_seconds"] == 1800.0


def test_acceptance_rejects_artifact_duration_different_from_external_authority(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(
        tmp_path / "input",
        window_seconds=900.0,
    )

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_window_seconds=1800.0,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
    rows = read_csv(tmp_path / "out" / "config_control_comparison.csv")
    assert any(
        row["check"] == "preflight_window_seconds"
        and row["acceptance"] == "fail"
        for row in rows
    )


def test_acceptance_rejects_external_duration_above_standing_cap(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(
        tmp_path / "input",
        window_seconds=1800.001,
    )

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_window_seconds=1800.001,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
    rows = read_csv(tmp_path / "out" / "config_control_comparison.csv")
    assert any(
        row["check"] == "expected_window_seconds_within_standing_cap"
        and row["acceptance"] == "fail"
        for row in rows
    )
    assert any(
        row["check"] == "watcher_seconds_within_cap"
        and row["acceptance"] == "fail"
        for row in rows
    )


def test_acceptance_keeps_remote_and_local_run_roots_distinct(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    output_dir = tmp_path / "out"

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=output_dir,
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )
    rows = {
        row["check"]: row
        for row in read_csv(
            output_dir / "provenance_identity_comparison.csv"
        )
    }

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert rows["expected_remote_run_root_canonical"]["acceptance"] == "pass"
    assert rows["preflight_remote_run_root"]["acceptance"] == "pass"
    assert rows["runtime_source_remote_run_root"]["acceptance"] == "pass"
    assert rows["local_pullback_run_root_present"]["acceptance"] == "pass"
    assert rows["runtime_source_remote_run_root"]["observed"].startswith(
        "/remote/"
    )
    assert rows["local_pullback_run_root_present"]["observed"] == str(
        input_root / "run"
    )


def test_acceptance_rejects_remote_run_root_relationship_tamper(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    provenance_path = (
        input_root
        / "run"
        / acceptance.RUNTIME_SOURCE_PROVENANCE_NAME
    )
    provenance = json.loads(
        provenance_path.read_text(encoding="utf-8")
    )
    provenance["run_root"] = "/remote/other-artifacts/run"
    write_json(provenance_path, provenance)
    seal_run(input_root)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_synchronized_remote_root_rewrite(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    run = input_root / "run"
    window = run / "window_01"
    rewritten_root = "/remote/rewritten-principal-task12/run"
    rewritten_window = f"{rewritten_root}/window_01"
    preflight_path = (
        input_root / "preflight" / "orchestrator_preflight.json"
    )
    preflight = json.loads(
        preflight_path.read_text(encoding="utf-8")
    )
    command = list(preflight["watcher_commands"][0])
    command[command.index("--output-dir") + 1] = rewritten_window
    preflight["run_root"] = rewritten_root
    preflight["watcher_commands"] = [command]
    write_json(preflight_path, preflight)
    provenance_path = (
        run / acceptance.RUNTIME_SOURCE_PROVENANCE_NAME
    )
    provenance = json.loads(
        provenance_path.read_text(encoding="utf-8")
    )
    provenance["run_root"] = rewritten_root
    provenance["watcher_commands"] = [command]
    write_json(provenance_path, provenance)
    write_json(window / "runner_command.json", {"command": command})
    for name in ("run_status.json", "run_complete.json"):
        path = run / name
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["run_root"] = rewritten_root
        write_json(path, payload)
    window_status_path = window / "window_status.json"
    window_status = json.loads(
        window_status_path.read_text(encoding="utf-8")
    )
    window_status["window_dir"] = rewritten_window
    write_json(window_status_path, window_status)
    seal_run(input_root)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_noncanonical_expected_remote_root(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")

    manifest = acceptance.run_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
        expected_remote_run_root="/remote/../rewritten/run",
    )

    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )


def test_decision_summary_reconstructs_t011_stage_counts() -> None:
    guard_failure_reason = (
        "outside_quality_a_b_queue_bands;"
        "missing_intent_limit_px;"
        "missing_or_nonpositive_intent_size;"
        "missing_quality_bucket"
    )
    trigger_rows: list[dict[str, object]] = []
    guard_rows: list[dict[str, object]] = []
    anti_drift_rows: list[dict[str, object]] = []
    edge_rows: list[dict[str, object]] = []
    attempt_rows: list[dict[str, object]] = []
    for index in range(5):
        event_sequence = index + 1
        source_time = 1_000 + event_sequence
        trigger_rows.append(
            {
                "event_sequence": event_sequence,
                "source_event_exchange_time_ms": source_time,
                "fresh_touch_allowed": True,
                "trigger_found": True,
                "guard_status": "anti_drift_block",
                "guard_reason": (
                    "adverse_trade_pressure_with_recent_adverse_bbo"
                ),
                "live_window_called": False,
                "private_read_endpoint_called_before_decision": False,
                "order_endpoint_called_before_decision": False,
                "cancel_endpoint_called_before_decision": False,
            }
        )
        anti_drift_rows.append(
            {
                "attempt": 1,
                "event_sequence": event_sequence,
                "phase": "pre_open_orders_public_gate",
                "status": "block",
                "reason": (
                    "adverse_trade_pressure_with_recent_adverse_bbo"
                ),
            }
        )
    for index in range(11):
        event_sequence = index + 6
        source_time = 1_000 + event_sequence
        trigger_rows.append(
            {
                "event_sequence": event_sequence,
                "source_event_exchange_time_ms": source_time,
                "fresh_touch_allowed": True,
                "trigger_found": True,
                "guard_status": "fail_closed",
                "guard_reason": guard_failure_reason,
                "live_window_called": False,
                "private_read_endpoint_called_before_decision": True,
                "order_endpoint_called_before_decision": False,
                "cancel_endpoint_called_before_decision": False,
            }
        )
        guard_rows.append(
            {
                "attempt": 1,
                "candidate_source_exchange_time_ms": source_time,
                "trigger_candidate_source_exchange_time_ms": source_time,
                "status": "fail_closed",
                "reason": guard_failure_reason,
            }
        )
        attempt_rows.append(
            {
                "attempt": 1,
                "attempt_id": 1,
                "attempt_key": f"{TASK_ID}:window_01:attempt_1",
                "event_sequence": event_sequence,
                "guard_status": "fail_closed",
                "guard_reason": guard_failure_reason,
                "edge_gate_status": "",
                "edge_gate_reason": "",
                "skip_reason": guard_failure_reason,
                "side": "",
                "order_endpoint_called": False,
                "cancel_endpoint_called": False,
            }
        )
    for index in range(10):
        event_sequence = index + 17
        source_time = 1_000 + event_sequence
        reason = (
            "fair_mid_source_stale"
            if index < 7
            else "edge_below_required_buffer"
        )
        trigger_rows.append(
            {
                "event_sequence": event_sequence,
                "source_event_exchange_time_ms": source_time,
                "fresh_touch_allowed": True,
                "trigger_found": True,
                "guard_status": "edge_gate_block",
                "guard_reason": reason,
                "live_window_called": False,
                "private_read_endpoint_called_before_decision": True,
                "order_endpoint_called_before_decision": False,
                "cancel_endpoint_called_before_decision": False,
            }
        )
        guard_rows.append(
            {
                "attempt": 1,
                "candidate_source_exchange_time_ms": source_time,
                "trigger_candidate_source_exchange_time_ms": source_time,
                "status": "pass",
                "reason": "",
            }
        )
        edge_rows.append(
            {
                "attempt": 1,
                "event_sequence": event_sequence,
                "edge_gate_status": "block",
                "edge_gate_reason": reason,
            }
        )
        attempt_rows.append(
            {
                "attempt": 1,
                "attempt_id": 1,
                "attempt_key": f"{TASK_ID}:window_01:attempt_1",
                "event_sequence": event_sequence,
                "guard_status": "edge_gate_block",
                "guard_reason": reason,
                "edge_gate_status": "block",
                "edge_gate_reason": reason,
                "skip_reason": reason,
                "side": "",
                "order_endpoint_called": False,
                "cancel_endpoint_called": False,
            }
        )
    for row in trigger_rows:
        row[
            "private_or_order_endpoint_called_before_trigger"
        ] = bool(
            row["private_read_endpoint_called_before_decision"]
            or row["order_endpoint_called_before_decision"]
        )
    endpoint_flags = {
        "private_endpoint_called": True,
        "real_order_endpoint_called": False,
        "real_cancel_endpoint_called": False,
    }
    producer = watcher.build_event_driven_decision_evidence_summary(
        trigger_rows=trigger_rows,
        guard_rows=guard_rows,
        anti_drift_rows=anti_drift_rows,
        edge_gate_rows=edge_rows,
        attempt_rows=attempt_rows,
        endpoint_flags=endpoint_flags,
    )
    independent = (
        acceptance.rebuild_event_driven_decision_evidence_summary(
            trigger_rows=trigger_rows,
            guard_rows=guard_rows,
            anti_drift_rows=anti_drift_rows,
            edge_gate_rows=edge_rows,
            attempt_rows=attempt_rows,
            inline_manifest=endpoint_flags,
            allow_legacy_guard_identity_bridge=True,
        )
    )

    assert independent == producer
    assert independent["trigger_row_count"] == 26
    assert independent["anti_drift_block_count"] == 5
    assert independent["immediate_guard_pass_count"] == 10
    assert independent["immediate_guard_fail_count"] == 11
    assert independent[
        "immediate_guard_failure_reason_atom_counts"
    ] == {
        "missing_intent_limit_px": 11,
        "missing_or_nonpositive_intent_size": 11,
        "missing_quality_bucket": 11,
        "outside_quality_a_b_queue_bands": 11,
    }
    assert independent["edge_gate_block_count"] == 10
    assert independent["edge_gate_block_reason_counts"] == {
        "edge_below_required_buffer": 3,
        "fair_mid_source_stale": 7,
    }
    assert independent["candidate_attempt_evidence_row_count"] == 21
    assert independent["manager_attempt_identity_count"] == 1
    assert independent["submitted_attempt_count"] == 0


def test_acceptance_rejects_synchronized_stale_decision_summary(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    window = input_root / "run" / "window_01"
    watcher_path = window / "event_driven_watcher_manifest.json"
    inline_path = window / "inline_reprice_manifest.json"
    summary_path = (
        window / "event_driven_decision_evidence_summary.json"
    )
    watcher_manifest = json.loads(
        watcher_path.read_text(encoding="utf-8")
    )
    inline_manifest = json.loads(
        inline_path.read_text(encoding="utf-8")
    )
    stale_summary = dict(
        watcher_manifest["decision_evidence_summary"]
    )
    stale_summary["trigger_row_count"] = 0
    watcher_manifest["decision_evidence_summary"] = stale_summary
    inline_manifest["decision_evidence_summary"] = stale_summary
    write_json(watcher_path, watcher_manifest)
    write_json(inline_path, inline_manifest)
    write_json(summary_path, stale_summary)
    seal_run(input_root)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_private_read_summary_conflict(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    watcher_path = (
        input_root
        / "run"
        / "window_01"
        / "event_driven_watcher_manifest.json"
    )
    watcher_manifest = json.loads(
        watcher_path.read_text(encoding="utf-8")
    )
    watcher_manifest[
        "public_waiting_phase_private_read_endpoint_called"
    ] = False
    write_json(watcher_path, watcher_manifest)
    seal_run(input_root)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_top_level_quote_attempt_copy_drift(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    top_level_path = (
        input_root / "run" / "window_01" / "quote_attempt_matrix.csv"
    )
    rows = read_csv(top_level_path)
    rows.append(dict(rows[0]))
    write_csv(top_level_path, rows, list(rows[0]))
    seal_run(input_root)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_malformed_decision_boolean(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    matrix_path = (
        input_root
        / "run"
        / "window_01"
        / "event_driven_trigger_decision_matrix.csv"
    )
    rows = read_csv(matrix_path)
    rows[0]["trigger_found"] = "yes"
    write_csv(matrix_path, rows, list(rows[0]))
    seal_run(input_root)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_synchronized_malformed_decision_identities(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    window = input_root / "run" / "window_01"
    live = live_artifact_dir(input_root)

    guard_path = window / "immediate_pre_submit_guard_matrix.csv"
    guard_rows = read_csv(guard_path)
    guard_rows[0]["event_sequence"] = "not-an-event"
    guard_rows[0]["attempt"] = "not-an-attempt"
    write_csv(guard_path, guard_rows, list(guard_rows[0]))

    edge_path = window / "edge_gate_matrix.csv"
    edge_rows = read_csv(edge_path)
    edge_rows[0]["event_sequence"] = "not-an-event"
    edge_rows[0]["attempt"] = "not-an-attempt"
    write_csv(edge_path, edge_rows, list(edge_rows[0]))

    for attempt_path in (
        window / "quote_attempt_matrix.csv",
        live / "quote_attempt_matrix.csv",
    ):
        attempt_rows = read_csv(attempt_path)
        for row in attempt_rows:
            row["event_sequence"] = "not-an-event"
            row["attempt"] = "not-an-attempt"
        write_csv(attempt_path, attempt_rows, list(attempt_rows[0]))

    sync_producer_decision_evidence(input_root)
    seal_run(input_root)
    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    validation_reasons = manifest[
        "independent_decision_evidence_summary"
    ]["validation_reasons"]
    assert any(
        "immediate_guard_event_sequence" in reason
        for reason in validation_reasons
    )
    assert any(
        "attempt_event_sequence" in reason
        for reason in validation_reasons
    )


def test_acceptance_rejects_legacy_guard_schema_downgrade(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    guard_path = (
        input_root
        / "run"
        / "window_01"
        / "immediate_pre_submit_guard_matrix.csv"
    )
    guard_rows = read_csv(guard_path)
    for row in guard_rows:
        row.pop("event_sequence")
        row["candidate_source_exchange_time_ms"] = "1000"
        row["trigger_candidate_source_exchange_time_ms"] = "1000"
    write_csv(guard_path, guard_rows, list(guard_rows[0]))
    sync_producer_decision_evidence(input_root)
    seal_run(input_root)

    manifest = acceptance.run_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
        expected_remote_run_root=REMOTE_RUN_ROOT,
        allow_legacy_guard_identity_bridge=True,
    )

    assert manifest["legacy_guard_identity_bridge_authorized"] is False
    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    assert any(
        reason.startswith(
            "immediate_guard_event_sequence_legacy_bridge_not_authorized:"
        )
        for reason in manifest[
            "independent_decision_evidence_summary"
        ]["validation_reasons"]
    )


def test_acceptance_rejects_synchronized_noncanonical_attempt_key(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    window = input_root / "run" / "window_01"
    live = live_artifact_dir(input_root)
    for attempt_path in (
        window / "quote_attempt_matrix.csv",
        live / "quote_attempt_matrix.csv",
    ):
        attempt_rows = read_csv(attempt_path)
        for row in attempt_rows:
            row["attempt_key"] = f"forged:attempt_{row['attempt_id']}"
        write_csv(attempt_path, attempt_rows, list(attempt_rows[0]))
    sync_producer_decision_evidence(input_root)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    assert any(
        reason.startswith("attempt_key_mismatch:")
        for reason in manifest[
            "independent_decision_evidence_summary"
        ]["validation_reasons"]
    )


def valid_manager_hold_shutdown_observation(
    *,
    source_close_required: bool = True,
) -> dict[str, object]:
    return {
        "status": "pass",
        "reason": "",
        "deadline_overrun_seconds": 0.05,
        "reconnect_count_start": 0,
        "reconnect_count_end": 0,
        "disconnect_count_start": 0,
        "disconnect_count_end": 0,
        "pump_shutdown_contract_version": (
            acceptance.MANAGER_HOLD_PUMP_SHUTDOWN_CONTRACT_VERSION
        ),
        "pump_stop_requested_monotonic": 100.0,
        "pump_stop_acknowledged": True,
        "pump_stop_acknowledged_monotonic": 100.051,
        "pump_read_inflight_at_stop": True,
        "pump_read_inflight_after_stop_wait": False,
        "pump_source_close_required": source_close_required,
        "pump_source_closed": source_close_required,
        "pump_source_closed_monotonic": (
            100.03 if source_close_required else 0.0
        ),
        "pump_source_close_error": "",
        "pump_thread_exited_monotonic": 100.04,
        "pump_shutdown_wait_started_monotonic": 100.001,
        "pump_shutdown_wait_ended_monotonic": 100.05,
        "pump_shutdown_wait_timeout_seconds": 0.25,
        "pump_shutdown_wait_seconds": 0.049,
        "hold_observer_returned_monotonic": 100.052,
        "manager_cancel_batch_started_monotonic": 100.053,
        "manager_cancel_batch_ended_monotonic": 100.054,
    }


def valid_manager_cancel_timeline_rows() -> list[dict[str, object]]:
    return [
        {
            "manager_cancel_batch_started_monotonic": 100.053,
            "manager_cancel_batch_ended_monotonic": 100.054,
        }
    ]


@pytest.mark.parametrize(
    "attempt_key_suffix",
    [
        "9" * 5_000,
        str(acceptance.RAW_MAX_CANCEL_REFERENCE_ATTEMPT + 1),
    ],
)
def test_acceptance_blocks_oversized_attempt_key_without_raising(
    tmp_path: Path,
    attempt_key_suffix: str,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    window = input_root / "run" / "window_01"
    live = live_artifact_dir(input_root)
    for attempt_path in (
        window / "quote_attempt_matrix.csv",
        live / "quote_attempt_matrix.csv",
    ):
        attempt_rows = read_csv(attempt_path)
        for row in attempt_rows:
            row["attempt_key"] = (
                f"{TASK_ID}:window_01:attempt_{attempt_key_suffix}"
            )
        write_csv(attempt_path, attempt_rows, list(attempt_rows[0]))
    sync_producer_decision_evidence(input_root)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    assert any(
        reason.startswith("attempt_key_mismatch:")
        for reason in manifest[
            "independent_decision_evidence_summary"
        ]["validation_reasons"]
    )


def test_acceptance_rejects_submissions_without_authorized_trigger(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    matrix_path = (
        input_root
        / "run"
        / "window_01"
        / "event_driven_trigger_decision_matrix.csv"
    )
    rows = read_csv(matrix_path)
    rows[0]["live_window_called"] = "False"
    write_csv(matrix_path, rows, list(rows[0]))
    summary = sync_producer_decision_evidence(input_root)
    assert summary["order_authorized_row_count"] == 0
    assert summary["submitted_attempt_count"] == 2
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    validation_reasons = manifest[
        "independent_decision_evidence_summary"
    ]["validation_reasons"]
    assert "submitted_attempts_without_authorized_trigger" in (
        validation_reasons
    )
    assert any(
        reason.startswith("submitted_attempt_not_authorized:")
        for reason in validation_reasons
    )


@pytest.mark.parametrize(
    ("matrix_name", "updates"),
    [
        (
            "anti_drift_gate_matrix.csv",
            {
                "attempt": "1",
                "event_sequence": "1",
                "phase": "pre_open_orders_public_gate",
                "status": "block",
                "reason": "anti-drift-reason-drift",
            },
        ),
        (
            "immediate_pre_submit_guard_matrix.csv",
            {
                "status": "fail_closed",
                "reason": "guard-reason-drift",
            },
        ),
        (
            "edge_gate_matrix.csv",
            {
                "edge_gate_status": "block",
                "edge_gate_reason": "edge-reason-drift",
            },
        ),
    ],
)
def test_acceptance_rejects_synchronized_cross_matrix_reason_drift(
    tmp_path: Path,
    matrix_name: str,
    updates: dict[str, str],
) -> None:
    input_root = make_artifact(tmp_path / "input")
    matrix_path = input_root / "run" / "window_01" / matrix_name
    rows = read_csv(matrix_path)
    if rows:
        rows[0].update(updates)
        fieldnames = list(rows[0])
    else:
        rows = [dict(updates)]
        fieldnames = list(updates)
    write_csv(matrix_path, rows, fieldnames)
    sync_producer_decision_evidence(input_root)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )
    validation_reasons = manifest[
        "independent_decision_evidence_summary"
    ]["validation_reasons"]
    assert any(
        "join_mismatch" in reason
        or "pass_reason_not_empty" in reason
        for reason in validation_reasons
    )


def test_acceptance_rejects_anti_drift_matrix_trigger_join_drift(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    matrix_path = (
        input_root
        / "run"
        / "window_01"
        / "anti_drift_gate_matrix.csv"
    )
    write_csv(
        matrix_path,
        [
            {
                "event_sequence": 1,
                "status": "block",
                "reason": (
                    "adverse_trade_pressure_with_recent_adverse_bbo"
                ),
            }
        ],
        ["event_sequence", "status", "reason"],
    )
    seal_run(input_root)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_independent_terminal_manifest_matches_sealed_fixture(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")

    verification = (
        acceptance.independently_verify_terminal_sha256_manifest(
            input_root / "run"
        )
    )

    assert verification["status"] == "pass"
    assert verification["reasons"] == []
    assert verification["unexpected_count"] == 0


def test_acceptance_rejects_post_seal_mutation_with_stale_summary(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    run_complete = input_root / "run" / "run_complete.json"
    run_complete.write_text(
        run_complete.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    independent = (
        acceptance.independently_verify_terminal_sha256_manifest(
            input_root / "run"
        )
    )
    assert independent["status"] == "fail"
    assert independent["mismatched_files"] == ["run_complete.json"]
    assert_acceptance_blocked(input_root, tmp_path / "out")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "traversal",
        "malformed",
        "blank",
        "whitespace",
        "missing",
        "unexpected",
        "mismatch",
    ],
)
def test_independent_terminal_manifest_rejects_adversarial_layout(
    tmp_path: Path,
    mutation: str,
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    artifact = run_root / "artifact.json"
    artifact.write_text('{"status":"pass"}\n', encoding="utf-8")
    orchestrator.write_sha256_manifest(run_root)
    orchestrator.verify_sha256_manifest(run_root)
    manifest_path = (
        run_root / acceptance.TERMINAL_SHA256_MANIFEST_NAME
    )
    manifest_text = manifest_path.read_text(encoding="utf-8")
    if mutation == "duplicate":
        manifest_path.write_text(
            manifest_text + manifest_text,
            encoding="utf-8",
        )
    elif mutation == "traversal":
        manifest_path.write_text(
            manifest_text + f"{'0' * 64}  ../escape.json\n",
            encoding="utf-8",
        )
    elif mutation == "malformed":
        manifest_path.write_text(
            manifest_text + "not-a-manifest-line\n",
            encoding="utf-8",
        )
    elif mutation == "blank":
        manifest_path.write_text(
            manifest_text + "\n",
            encoding="utf-8",
        )
    elif mutation == "whitespace":
        manifest_path.write_text(
            manifest_text + " \t \n",
            encoding="utf-8",
        )
    elif mutation == "missing":
        artifact.unlink()
    elif mutation == "unexpected":
        (run_root / "unexpected.json").write_text(
            "{}\n",
            encoding="utf-8",
        )
    else:
        artifact.write_text(
            '{"status":"mutated"}\n',
            encoding="utf-8",
        )

    verification = (
        acceptance.independently_verify_terminal_sha256_manifest(
            run_root
        )
    )

    assert verification["status"] == "fail"
    assert verification["reasons"]


@pytest.mark.parametrize("record", ["\n", " \t \n"])
def test_acceptance_rejects_blank_terminal_manifest_record(
    tmp_path: Path,
    record: str,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    manifest_path = (
        input_root
        / "run"
        / acceptance.TERMINAL_SHA256_MANIFEST_NAME
    )
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8") + record,
        encoding="utf-8",
    )

    independent = (
        acceptance.independently_verify_terminal_sha256_manifest(
            input_root / "run"
        )
    )
    assert independent["status"] == "fail"
    assert any(
        reason.startswith("terminal_sha256_line_blank:")
        for reason in independent["reasons"]
    )
    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_passes_actual_two_sided_writer_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    client = write_actual_two_sided_live_artifacts(
        input_root,
        monkeypatch,
    )

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"
    assert manifest["live_summary"]["submissions"] == 2
    assert len(client.order_intents) == 2
    assert len(client.cancel_calls) == 2
    response_audit = json.loads(
        (
            live_artifact_dir(input_root)
            / "private_order_response_audit.json"
        ).read_text(encoding="utf-8")
    )
    assert response_audit["order_response_rows"][0]["result"]["response"][
        "data"
    ]["statuses"][0]["resting"]["oid_token"] == (
        fill_window.reference_identity_token("oid", 301)
    )


def test_acceptance_rejects_one_sided_evidence(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    live = live_artifact_dir(input_root)
    attempt_path = live / "quote_attempt_matrix.csv"
    intent_path = live / "order_intent_audit.csv"
    attempts = read_csv(attempt_path)[:1]
    intents = read_csv(intent_path)[:1]
    write_csv(attempt_path, attempts, list(attempts[0]))
    write_csv(intent_path, intents, list(intents[0]))

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_exact_two_uses_submitted_rows_not_all_candidates(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    window_attempt_path = (
        input_root / "run" / "window_01" / "quote_attempt_matrix.csv"
    )
    live_attempt_path = (
        live_artifact_dir(input_root) / "quote_attempt_matrix.csv"
    )
    attempts = read_csv(live_attempt_path)
    for attempt_id in (3, 4):
        candidate = dict(attempts[0])
        candidate.update(
            {
                "attempt": str(attempt_id),
                "attempt_id": str(attempt_id),
                "attempt_key": (
                    f"{TASK_ID}:window_01:attempt_{attempt_id}"
                ),
                "side": "",
                "limit_px": "",
                "size_btc": "",
                "order_status_types": "skipped",
                "order_endpoint_called": "False",
                "cancel_endpoint_called": "False",
            }
        )
        attempts.append(candidate)
    write_csv(live_attempt_path, attempts, list(attempts[0]))
    write_csv(window_attempt_path, attempts, list(attempts[0]))
    summary = sync_producer_decision_evidence(input_root)
    seal_run(input_root)

    output_dir = tmp_path / "out"
    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=output_dir,
    )
    decision_rows = read_csv(
        output_dir / "decision_replay_comparison.csv"
    )
    exact_two = next(
        row for row in decision_rows if row["check"] == "attempt_row_count"
    )

    assert summary["candidate_attempt_evidence_row_count"] == 4
    assert summary["submitted_attempt_count"] == 2
    assert exact_two["observed"] == "2"
    assert exact_two["acceptance"] == "pass"
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"


def test_acceptance_rejects_three_submitted_rows(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    window_attempt_path = (
        input_root / "run" / "window_01" / "quote_attempt_matrix.csv"
    )
    live_attempt_path = (
        live_artifact_dir(input_root) / "quote_attempt_matrix.csv"
    )
    attempts = read_csv(live_attempt_path)
    third = dict(attempts[0])
    third.update(
        {
            "attempt": "3",
            "attempt_id": "3",
            "attempt_key": f"{TASK_ID}:window_01:attempt_3",
        }
    )
    attempts.append(third)
    write_csv(live_attempt_path, attempts, list(attempts[0]))
    write_csv(window_attempt_path, attempts, list(attempts[0]))
    sync_producer_decision_evidence(input_root)
    seal_run(input_root)

    output_dir = tmp_path / "out"
    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=output_dir,
    )
    decision_rows = read_csv(
        output_dir / "decision_replay_comparison.csv"
    )
    exact_two = next(
        row for row in decision_rows if row["check"] == "attempt_row_count"
    )

    assert exact_two["observed"] == "3"
    assert exact_two["acceptance"] == "fail"
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_rejects_duplicate_side_evidence(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    live = live_artifact_dir(input_root)
    attempt_path = live / "quote_attempt_matrix.csv"
    intent_path = live / "order_intent_audit.csv"
    private_path = live / "private_order_response_audit.json"
    attempts = read_csv(attempt_path)
    intents = read_csv(intent_path)
    private = json.loads(private_path.read_text(encoding="utf-8"))
    attempts[1]["side"] = "buy"
    intents[1]["side"] = "buy"
    private["order_status_rows"][1]["side"] = "buy"
    write_csv(attempt_path, attempts, list(attempts[0]))
    write_csv(intent_path, intents, list(intents[0]))
    write_json(private_path, private)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_aggregate_side_evidence(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    attempt_path = live_artifact_dir(input_root) / "quote_attempt_matrix.csv"
    attempts = read_csv(attempt_path)
    attempts[1]["side"] = "buy+sell"
    write_csv(attempt_path, attempts, list(attempts[0]))

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_stale_attempt_key(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    attempt_path = live_artifact_dir(input_root) / "quote_attempt_matrix.csv"
    attempts = read_csv(attempt_path)
    attempts[1]["attempt_key"] = "0718T999:window_01:attempt_2"
    write_csv(attempt_path, attempts, list(attempts[0]))

    assert_acceptance_blocked(input_root, tmp_path / "out")


@pytest.mark.parametrize(
    ("flag", "replacement"),
    [
        ("--exchange-reconciled-manager", None),
        ("--event-driven-edge-gate-live", "--event-driven-live"),
        ("--requote-attempts", "1"),
        ("--max-real-order-submissions", "1"),
    ],
)
def test_acceptance_rejects_wrong_two_sided_command_contract(
    tmp_path: Path,
    flag: str,
    replacement: str | None,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    command_path = input_root / "run" / "window_01" / "runner_command.json"
    payload = json.loads(command_path.read_text(encoding="utf-8"))
    command = payload["command"]
    index = command.index(flag)
    if replacement is None:
        command.pop(index)
    elif flag.startswith("--event-driven"):
        command[index] = replacement
    else:
        command[index + 1] = replacement
    write_json(command_path, payload)

    assert_acceptance_blocked(input_root, tmp_path / "out")


@pytest.mark.parametrize(
    "duplicate_tokens",
    [
        ["--max-real-order-submissions", "3"],
        ["--requote-attempts", "3"],
        ["--watcher-seconds", "1800"],
        ["--quote-hold-seconds", "4"],
        ["--wait-seconds", "20"],
        ["--artifact-task-id", "FORGED"],
        ["--artifact-window-id", "2"],
        ["--run-id", "forged"],
        ["--exchange-reconciled-manager"],
        ["--hyperliquid-l2book-fast"],
        ["--event-driven-live"],
        ["--max-real-order-submissions=3"],
    ],
)
def test_acceptance_rejects_duplicate_or_noncanonical_sealed_command(
    tmp_path: Path,
    duplicate_tokens: list[str],
) -> None:
    input_root = make_artifact(tmp_path / "input")
    command_path = (
        input_root / "run" / "window_01" / "runner_command.json"
    )
    command = list(
        json.loads(command_path.read_text(encoding="utf-8"))["command"]
    )
    command.extend(duplicate_tokens)
    write_sealed_command(input_root, command)

    assert_acceptance_blocked(input_root, tmp_path / "out")


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_rows",
        "empty_result",
        "duplicate_status",
        "duplicate_response_row",
        "unrelated_reference",
        "mismatched_attempt_fields",
        "wrong_attempt_key",
        "empty_order_results",
    ],
)
def test_acceptance_rejects_forged_raw_order_response(
    tmp_path: Path,
    mutation: str,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    private_path = (
        live_artifact_dir(input_root)
        / "private_order_response_audit.json"
    )
    private = json.loads(private_path.read_text(encoding="utf-8"))
    if mutation == "missing_rows":
        private["order_response_rows"] = []
    elif mutation == "empty_result":
        private["order_response_rows"][0]["result"] = {}
    elif mutation == "duplicate_status":
        statuses = private["order_response_rows"][0]["result"][
            "response"
        ]["data"]["statuses"]
        statuses.append(dict(statuses[0]))
        private["order_results"][0] = private["order_response_rows"][0][
            "result"
        ]
    elif mutation == "duplicate_response_row":
        private["order_response_rows"].append(
            dict(private["order_response_rows"][0])
        )
    elif mutation == "unrelated_reference":
        resting = private["order_response_rows"][0]["result"]["response"][
            "data"
        ]["statuses"][0]["resting"]
        resting.update(
            {
                "oid": 999,
                "oid_token": fill_window.reference_identity_token(
                    "oid",
                    999,
                ),
                "cloid": "unrelated",
                "cloid_token": fill_window.reference_identity_token(
                    "cloid",
                    "unrelated",
                ),
            }
        )
        private["order_results"][0] = private["order_response_rows"][0][
            "result"
        ]
    elif mutation == "wrong_attempt_key":
        private["order_response_rows"][0]["attempt_key"] = (
            f"{TASK_ID}:window_01:attempt_2"
        )
    elif mutation == "mismatched_attempt_fields":
        private["order_response_rows"][0]["attempt"] = 2
    elif mutation == "empty_order_results":
        private["order_results"] = []
    write_json(private_path, private)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_allows_full_reference_bound_fill_terminal_state(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
    )

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"


def test_acceptance_allows_full_raw_fills_after_v4_filled_query(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
    )
    install_v4_filled_terminal_query_proof(input_root)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"


@pytest.mark.parametrize(
    ("attempt", "direction", "explicit_side", "expected_side"),
    [
        (1, "Open Long", None, "buy"),
        (1, "Close Short", None, "buy"),
        (2, "Open Short", None, "sell"),
        (2, "Close Long", None, "sell"),
        (1, "Open Long", "B", "buy"),
        (2, "Close Long", "A", "sell"),
    ],
)
def test_acceptance_allows_exact_hyperliquid_fill_direction(
    tmp_path: Path,
    attempt: int,
    direction: str,
    explicit_side: str | None,
    expected_side: str,
) -> None:
    side_fields = {"dir": direction}
    if explicit_side is not None:
        side_fields["side"] = explicit_side
    assert acceptance.raw_fill_side(side_fields) == expected_side
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
    )
    set_raw_fill_direction(
        input_root,
        attempt=attempt,
        direction=direction,
        explicit_side=explicit_side,
    )

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"


@pytest.mark.parametrize(
    ("attempt", "explicit_side", "direction"),
    [
        (1, "B", "Close Long"),
        (2, "A", "Close Short"),
        (1, "B", "Increase Long"),
    ],
)
def test_acceptance_rejects_synchronized_conflicting_fill_direction(
    tmp_path: Path,
    attempt: int,
    explicit_side: str,
    direction: str,
) -> None:
    assert acceptance.raw_fill_side(
        {"side": explicit_side, "dir": direction}
    ) == "unknown"
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
    )
    set_raw_fill_direction(
        input_root,
        attempt=attempt,
        explicit_side=explicit_side,
        direction=direction,
    )

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_forged_fill_csv_without_raw_pullbacks(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
    )
    pullback_path = (
        live_artifact_dir(input_root)
        / "user_fills_pullback_audit.json"
    )
    pullback = json.loads(
        pullback_path.read_text(encoding="utf-8")
    )
    pullback["pullbacks"] = []
    pullback["pullback_count"] = 0
    write_json(pullback_path, pullback)

    assert_acceptance_blocked(input_root, tmp_path / "out")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("side", "A"),
        ("sz", "0.001"),
        ("crossed", True),
        ("fillId", "forged-fill-id"),
        ("coin", "ETH"),
    ],
)
def test_acceptance_rejects_raw_fill_csv_disagreement(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
    )
    pullback_path = (
        live_artifact_dir(input_root)
        / "user_fills_pullback_audit.json"
    )
    pullback = json.loads(
        pullback_path.read_text(encoding="utf-8")
    )
    pullback["pullbacks"][0]["fills"][0][field] = value
    write_json(pullback_path, pullback)

    assert_acceptance_blocked(input_root, tmp_path / "out")


@pytest.mark.parametrize(
    ("side", "impossible_price"),
    [
        ("buy", "70000"),
        ("sell", "60000"),
    ],
)
def test_acceptance_rejects_synchronized_impossible_fill_price(
    tmp_path: Path,
    side: str,
    impossible_price: str,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
    )
    live = live_artifact_dir(input_root)
    pullback_path = live / "user_fills_pullback_audit.json"
    pullback = json.loads(
        pullback_path.read_text(encoding="utf-8")
    )
    raw_fill = next(
        fill
        for fill in pullback["pullbacks"][0]["fills"]
        if (
            fill.get("side") == "B"
            if side == "buy"
            else fill.get("side") == "A"
        )
    )
    raw_fill["px"] = impossible_price
    fingerprint = fill_window.fill_payload_fingerprint(raw_fill)
    write_json(pullback_path, pullback)

    for filename in (
        "live_fill_ledger.csv",
        "fill_attribution_evidence.csv",
    ):
        path = live / filename
        rows = read_csv(path)
        row = next(row for row in rows if row["side"] == side)
        row["price_usdc"] = impossible_price
        row["fill_payload_fingerprint"] = fingerprint
        write_csv(
            path,
            rows,
            (
                fill_window.live_fill_ledger_fieldnames()
                if filename == "live_fill_ledger.csv"
                else fill_window.fill_attribution_evidence_fieldnames()
            ),
        )
    seal_run(input_root)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_correct_oid_with_unrelated_cloid(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
    )
    pullback_path = (
        live_artifact_dir(input_root)
        / "user_fills_pullback_audit.json"
    )
    pullback = json.loads(
        pullback_path.read_text(encoding="utf-8")
    )
    pullback["pullbacks"][0]["fills"][0]["cloid_token"] = (
        fill_window.reference_identity_token(
            "cloid",
            "cloid-sell",
        )
    )
    write_json(pullback_path, pullback)

    assert_acceptance_blocked(input_root, tmp_path / "out")


@pytest.mark.parametrize(
    ("mutation", "replacement"),
    [
        ("python", "/tmp/forged-python"),
        ("script", "/tmp/forged-watcher.py"),
        ("output", "/tmp/forged-output"),
        ("value_abbreviation", "--max-real-order-sub"),
        ("boolean_abbreviation", "--exchange-reconciled-man"),
        ("path_abbreviation", "--out"),
    ],
)
def test_acceptance_rejects_fully_forged_command_binding(
    tmp_path: Path,
    mutation: str,
    replacement: str,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    command_path = (
        input_root
        / "run"
        / "window_01"
        / "runner_command.json"
    )
    command = list(
        json.loads(command_path.read_text(encoding="utf-8"))[
            "command"
        ]
    )
    if mutation == "python":
        command[0] = replacement
    elif mutation == "script":
        command[1] = replacement
    elif mutation == "output":
        command[command.index("--output-dir") + 1] = replacement
    elif mutation == "value_abbreviation":
        command[command.index("--max-real-order-submissions")] = (
            replacement
        )
    elif mutation == "boolean_abbreviation":
        command[command.index("--exchange-reconciled-manager")] = (
            replacement
        )
    else:
        command[command.index("--output-dir")] = replacement
    write_fully_sealed_command(input_root, command)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_partial_fill_with_failed_cancel_terminal_state(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1,),
        cancel_success=False,
    )

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_full_fill_with_unrelated_cancel_evidence(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
        unrelated_cancel=True,
    )

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_unbound_extra_fill_row(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    set_filled_lifecycle(
        input_root,
        filled_attempts=(1, 2),
        cancel_success=False,
    )
    live = live_artifact_dir(input_root)
    fill_path = live / "live_fill_ledger.csv"
    attribution_path = live / "fill_attribution_evidence.csv"
    role_path = live / "fill_liquidity_role_evidence.csv"
    fill_rows = read_csv(fill_path)
    extra_fill = dict(fill_rows[0])
    extra_fill.update(
        {
            "fill_id": "unbound-fill",
            "attempt_id": "3",
            "attempt_key": f"{TASK_ID}:window_01:attempt_3",
        }
    )
    fill_rows.append(extra_fill)
    write_csv(
        fill_path,
        fill_rows,
        fill_window.live_fill_ledger_fieldnames(),
    )
    write_csv(
        attribution_path,
        fill_rows,
        fill_window.fill_attribution_evidence_fieldnames(),
    )
    role_rows = read_csv(role_path)
    extra_role = dict(role_rows[0])
    extra_role.update(
        {
            "fill_id": "unbound-fill",
            "attempt_id": "3",
            "attempt_key": f"{TASK_ID}:window_01:attempt_3",
        }
    )
    role_rows.append(extra_role)
    write_csv(
        role_path,
        role_rows,
        fill_window.fill_liquidity_role_evidence_fieldnames(),
    )
    fill_manifest_path = live / "m2_fill_window_manifest.json"
    fill_manifest = json.loads(
        fill_manifest_path.read_text(encoding="utf-8")
    )
    fill_manifest["fill_count"] = 3
    fill_manifest["maker_fill_count"] = 3
    fill_manifest["ledger_fill_rows"] = 3
    write_json(fill_manifest_path, fill_manifest)
    watcher_path = (
        input_root
        / "run"
        / "window_01"
        / "event_driven_watcher_manifest.json"
    )
    watcher_manifest = json.loads(
        watcher_path.read_text(encoding="utf-8")
    )
    watcher_manifest["fill_count"] = 3
    watcher_manifest["maker_fill_count"] = 3
    write_json(watcher_path, watcher_manifest)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_acceptance_rejects_forged_second_side_lifecycle(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    live = live_artifact_dir(input_root)
    private_path = live / "private_order_response_audit.json"
    fill_path = live / "m2_fill_window_manifest.json"
    private = json.loads(private_path.read_text(encoding="utf-8"))
    fill_manifest = json.loads(fill_path.read_text(encoding="utf-8"))
    private["order_status_rows"][1]["status_type"] = "rejected"
    fill_manifest["order_status_types"][1] = "rejected"
    write_json(private_path, private)
    write_json(fill_path, fill_manifest)

    assert_acceptance_blocked(input_root, tmp_path / "out")


def test_independent_reconciliation_matches_producer_on_valid_raw_proof() -> None:
    tracked_refs = [{"attempt": 1, "oid": 101, "cloid": "cloid-a"}]
    cancel_results = [
        {
            "attempt": 1,
            "oid": 101,
            "cloid": "cloid-a",
            "result": {
                "status": "ok",
                "response": {"data": {"statuses": ["success"]}},
            },
        }
    ]

    assert acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=tracked_refs,
        cancel_results=cancel_results,
    ) == fill_window.cancel_reference_reconciliation(
        tracked_refs=tracked_refs,
        cancel_results=cancel_results,
    )


@pytest.mark.parametrize("status", [[], {}, True, 1, 1.0, None])
def test_independent_terminal_query_classifier_rejects_non_string_status(
    status: object,
) -> None:
    assert acceptance.raw_terminal_query_status_from_result(
        {"status": status}
    ) == "unknown"


def test_independent_legacy_direct_status_checks_supplied_identity() -> None:
    expected = {
        "oid": fill_window.reference_identity_token("oid", 101),
        "cloid": fill_window.reference_identity_token(
            "cloid",
            "cloid-a",
        ),
    }
    exact = executor.redact_with_reference_tokens(
        {
            "status": "canceled",
            "order": {"oid": 101, "cloid": "cloid-a"},
        }
    )
    foreign = executor.redact_with_reference_tokens(
        {
            "status": "canceled",
            "order": {"oid": 999, "cloid": "foreign"},
        }
    )

    assert acceptance.raw_terminal_query_status_from_result(
        exact,
        method="query_order_by_oid",
        expected_tokens=expected,
    ) == "cancel_confirmed"
    assert acceptance.raw_terminal_query_status_from_result(
        foreign,
        method="query_order_by_oid",
        expected_tokens=expected,
    ) == "unknown"


@pytest.mark.parametrize(
    "conflicting_row",
    [
        {
            "order": {"oid": 101, "cloid": "other"},
            "status": "filled",
        },
        {
            "order": {"oid": 202, "cloid": "cloid-a"},
            "status": "filled",
        },
        {
            "order": {"oid": 101},
            "status": "filled",
        },
        {
            "order": {"cloid": "cloid-a"},
            "status": "filled",
        },
        {
            "order": {"oid": "0101", "cloid": "cloid-a"},
            "status": "filled",
        },
        {
            "order": {"oid": "²", "cloid": "cloid-a"},
            "status": "filled",
        },
        {
            "order": {"oid": "1" * 5000, "cloid": "cloid-a"},
            "status": "filled",
        },
        {
            "order": {
                "oid": str(executor.MAX_REFERENCE_OID + 1),
                "cloid": "cloid-a",
            },
            "status": "filled",
        },
        {
            "order": {
                "oid": 101,
                "orderId": 102,
                "cloid": "cloid-a",
            },
            "status": "filled",
        },
        {
            "order": {
                "oid": 101,
                "cloid": "cloid-a",
                "clientOrderId": "other",
            },
            "status": "filled",
        },
        {"order": [], "status": "filled"},
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": [],
        },
        {
            "order": {"oid": 999},
            "status": "canceled",
        },
        {
            "order": {"cloid": "foreign"},
            "status": "canceled",
        },
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": "",
        },
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": " ",
        },
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": "unknownOid",
        },
        {
            "order": {"oid": 101, "cloid": "cloid-a"},
            "status": "unknownOid",
        },
    ],
)
def test_independent_history_rejects_conflicting_or_malformed_rows(
    conflicting_row: dict,
) -> None:
    expected = {
        "oid": fill_window.reference_identity_token("oid", 101),
        "cloid": fill_window.reference_identity_token(
            "cloid",
            "cloid-a",
        ),
    }
    history = executor.redact_with_reference_tokens(
        {
            "status": "historical_orders",
            "orders": [
                conflicting_row,
                {
                    "order": {"oid": 101, "cloid": "cloid-a"},
                    "status": "canceled",
                },
            ],
        }
    )

    assert acceptance.raw_terminal_query_status_from_result(
        history,
        method="historical_orders",
        expected_tokens=expected,
    ) == "unknown"


def test_independent_history_coverage_tracks_expected_token_kinds() -> None:
    oid = fill_window.reference_identity_token("oid", 101)
    cloid = fill_window.reference_identity_token(
        "cloid",
        "cloid-a",
    )

    assert acceptance.raw_historical_reference_row_classification(
        {
            "order": {"oid": 999},
            "status": "canceled",
        },
        expected_tokens={"oid": oid},
    ) == "foreign"
    assert acceptance.raw_historical_reference_row_classification(
        {
            "order": {"oid": 101, "cloid": "extra"},
            "status": "canceled",
        },
        expected_tokens={"oid": oid},
    ) == "conflicting"
    assert acceptance.raw_historical_reference_row_classification(
        {
            "order": {"oid": 999, "cloid": "extra"},
            "status": "canceled",
        },
        expected_tokens={"oid": oid},
    ) == "malformed"
    assert acceptance.raw_historical_reference_row_classification(
        {
            "order": {"cloid": "foreign"},
            "status": "canceled",
        },
        expected_tokens={"cloid": cloid},
    ) == "foreign"
    assert acceptance.raw_historical_reference_row_classification(
        {
            "order": {"oid": 999, "cloid": "cloid-a"},
            "status": "canceled",
        },
        expected_tokens={"cloid": cloid},
    ) == "conflicting"
    assert acceptance.raw_historical_reference_row_classification(
        {
            "order": {"oid": 999, "cloid": "foreign"},
            "status": "canceled",
        },
        expected_tokens={"cloid": cloid},
    ) == "malformed"


@pytest.mark.parametrize(
    "mutation",
    [
        "bogus_redaction_marker",
        "missing_alias_map_entry",
        "string_conflict_marker",
        "numeric_invalid_marker",
        "aggregate_token_only",
    ],
)
def test_independent_history_rejects_malformed_redaction_schema(
    mutation: str,
) -> None:
    expected = {
        "oid": fill_window.reference_identity_token("oid", 101),
        "cloid": fill_window.reference_identity_token(
            "cloid",
            "cloid-a",
        ),
    }
    order = executor.redact_with_reference_tokens(
        {
            "oid": 101,
            "orderId": "101",
            "cloid": "cloid-a",
            "clientOrderId": "cloid-a",
        }
    )
    if mutation == "bogus_redaction_marker":
        order["oid"] = "<redacted_bogus>"
    elif mutation == "missing_alias_map_entry":
        del order["oid_alias_tokens"]["orderId"]
    elif mutation == "string_conflict_marker":
        order["oid_alias_conflict"] = "true"
    elif mutation == "numeric_invalid_marker":
        order["cloid_alias_invalid"] = 1
    elif mutation == "aggregate_token_only":
        for key in (
            "oid",
            "orderId",
            "cloid",
            "clientOrderId",
            "oid_alias_tokens",
            "cloid_alias_tokens",
        ):
            order.pop(key)
    history = {
        "status": "historical_orders",
        "orders": [{"order": order, "status": "canceled"}],
    }

    assert acceptance.raw_terminal_query_status_from_result(
        history,
        method="historical_orders",
        expected_tokens=expected,
    ) == "unknown"


@pytest.mark.parametrize(
    "malformed_attempt",
    [
        True,
        False,
        1.0,
        1.1,
        1.9,
        0,
        -1,
        float("nan"),
        float("inf"),
        "1.0",
        "1e0",
        " 1",
        "1 ",
        "01",
        "+1",
        "",
        "1" * 5_000,
        acceptance.RAW_MAX_CANCEL_REFERENCE_ATTEMPT + 1,
        str(acceptance.RAW_MAX_CANCEL_REFERENCE_ATTEMPT + 1),
    ],
)
def test_independent_reconciliation_rejects_malformed_attempt_identity(
    malformed_attempt: object,
) -> None:
    reconciliation = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=[{"attempt": malformed_attempt, "oid": 101}],
        cancel_results=[
            {
                "attempt": malformed_attempt,
                "oid": 101,
                "result": {
                    "status": "ok",
                    "response": {"data": {"statuses": ["success"]}},
                },
            }
        ],
    )

    assert reconciliation["status"] == "fail_closed"
    assert "tracked_reference_attempt_missing" in reconciliation["reasons"]
    assert "cancel_result_attempt_missing" in reconciliation["reasons"]


def test_independent_reconciliation_rejects_fractional_cross_attempt_alias() -> None:
    reconciliation = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1.1, "oid": 101}],
        cancel_results=[
            {
                "attempt": 1.9,
                "oid": 101,
                "result": {
                    "status": "ok",
                    "response": {"data": {"statuses": ["success"]}},
                },
            }
        ],
    )

    assert reconciliation["status"] == "fail_closed"
    assert reconciliation["proven_reference_count"] == 0


@pytest.mark.parametrize(
    "attempt",
    [
        1,
        "1",
        acceptance.RAW_MAX_CANCEL_REFERENCE_ATTEMPT,
        str(acceptance.RAW_MAX_CANCEL_REFERENCE_ATTEMPT),
    ],
)
def test_independent_reconciliation_accepts_canonical_attempt_identity(
    attempt: object,
) -> None:
    reconciliation = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=[{"attempt": attempt, "oid": 101}],
        cancel_results=[
            {
                "attempt": attempt,
                "oid": 101,
                "result": {
                    "status": "ok",
                    "response": {"data": {"statuses": ["success"]}},
                },
            }
        ],
    )

    assert reconciliation["status"] == "pass"
    assert reconciliation["reference_rows"][0]["attempt"] == int(attempt)


@pytest.mark.parametrize(
    "statuses",
    [
        [{"success": False}],
        [{"success": None}],
        [{"success": 0}],
        [{"success": -1}],
        [{"success": ""}],
        [{"success": " "}],
        [{"success": 0.0}],
        [{"success": 1.0}],
        [{"success": {}}],
        [{"success": []}],
        [{"success": "oid-101", "extra": True}],
        ["SUCCESS"],
        ["success", "success"],
    ],
)
def test_independent_reconciliation_rejects_malformed_success_status(
    statuses: list[object],
) -> None:
    reconciliation = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101}],
        cancel_results=[
            {
                "attempt": 1,
                "oid": 101,
                "result": {
                    "status": "ok",
                    "response": {"data": {"statuses": statuses}},
                },
            }
        ],
    )

    assert reconciliation["status"] == "fail_closed"
    assert reconciliation["authoritative_success_count"] == 0


@pytest.mark.parametrize(
    "status",
    [
        "success",
        {"success": "oid-101"},
        {"success": 101},
    ],
)
def test_independent_reconciliation_accepts_explicit_success_status(
    status: object,
) -> None:
    reconciliation = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=[{"attempt": 1, "oid": 101}],
        cancel_results=[
            {
                "attempt": 1,
                "oid": 101,
                "result": {
                    "status": "ok",
                    "response": {"data": {"statuses": [status]}},
                },
            }
        ],
    )

    assert reconciliation["status"] == "pass"


def test_independent_reconciliation_rejects_token_conflicting_with_raw_identity() -> None:
    reconciliation = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=[
            {
                "attempt": 1,
                "oid": 101,
                "oid_token": acceptance.raw_reference_identity_token("oid", 999),
            }
        ],
        cancel_results=[
            {
                "attempt": 1,
                "oid": 101,
                "result": {
                    "status": "ok",
                    "response": {"data": {"statuses": ["success"]}},
                },
            }
        ],
    )

    assert reconciliation["status"] == "fail_closed"
    assert (
        "tracked_reference_oid_token_conflicts_with_raw_identity"
        in reconciliation["reasons"]
    )


def test_independent_reconciliation_rejects_invalid_persisted_token_format() -> None:
    reconciliation = acceptance.rebuild_raw_cancel_reference_reconciliation(
        tracked_refs=[
            {
                "attempt": 1,
                "oid": "<redacted>",
                "oid_token": "oid_sha256_not-a-digest",
            }
        ],
        cancel_results=[
            {
                "attempt": 1,
                "oid": "<redacted>",
                "oid_token": "oid_sha256_not-a-digest",
                "result": {
                    "status": "ok",
                    "response": {"data": {"statuses": ["success"]}},
                },
            }
        ],
    )

    assert reconciliation["status"] == "fail_closed"
    assert "tracked_reference_oid_token_invalid" in reconciliation["reasons"]
    assert "cancel_result_oid_token_invalid" in reconciliation["reasons"]


def test_acceptance_fails_without_per_reference_cancel_proof(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    manifest_path = (
        input_root
            / "run"
            / "window_01"
            / "window_01"
            / "pulled_back_awsserver1"
        / "m2_fill_window_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fill_reconciliation"].pop("cancel_reference_reconciliation")
    write_json(manifest_path, payload)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_forged_cancel_reference_summary(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    manifest_path = (
        input_root
            / "run"
            / "window_01"
            / "window_01"
            / "pulled_back_awsserver1"
        / "m2_fill_window_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fill_reconciliation"]["cancel_reference_reconciliation"][
        "cancel_evidence_rows"
    ] = []
    write_json(manifest_path, payload)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_copied_pass_summaries_with_unrelated_raw_target(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    proof_path = (
        input_root
            / "run"
            / "window_01"
            / "window_01"
            / "pulled_back_awsserver1"
        / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["cancel_results"][0]["oid"] = 999
    proof["cancel_results"][0]["cloid"] = "forged-target"
    write_json(proof_path, proof)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_copied_pass_summaries_with_ambiguous_raw_response(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    proof_path = (
        input_root
            / "run"
            / "window_01"
            / "window_01"
            / "pulled_back_awsserver1"
        / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["cancel_results"][0]["result"] = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {
                        "error": (
                            "Order was never placed, already canceled, or filled. "
                            "asset=0"
                        )
                    }
                ]
            }
        },
    }
    write_json(proof_path, proof)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_synchronized_false_success_summaries(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    live_dir = (
        input_root
            / "run"
            / "window_01"
            / "window_01"
            / "pulled_back_awsserver1"
    )
    proof_path = live_dir / "cancel_shutdown_proof.json"
    manifest_path = live_dir / "m2_fill_window_manifest.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["cancel_results"][0]["result"] = {
        "status": "ok",
        "response": {"data": {"statuses": [{"success": False}]}},
    }
    rebuilt = fill_window.cancel_reference_reconciliation(
        tracked_refs=proof["tracked_refs"],
        cancel_results=proof["cancel_results"],
    )
    assert rebuilt["status"] == "fail_closed"
    proof["fill_reconciliation"]["cancel_reference_reconciliation"] = rebuilt
    write_json(proof_path, proof)
    fill_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fill_manifest["fill_reconciliation"]["cancel_reference_reconciliation"] = rebuilt
    write_json(manifest_path, fill_manifest)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_missing_raw_cancel_proof_inputs(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    proof_path = (
        input_root
            / "run"
            / "window_01"
            / "window_01"
            / "pulled_back_awsserver1"
        / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.pop("tracked_refs")
    proof.pop("cancel_results")
    write_json(proof_path, proof)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_stale_inner_identity(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    config = input_root / "run" / "window_01" / "window_01" / "pulled_back_awsserver1" / "approved_config_snapshot.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["task_id"] = "0622T004"
    write_json(config, payload)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_runtime_envelope_mismatch(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    config = input_root / "run" / "window_01" / "window_01" / "pulled_back_awsserver1" / "approved_config_snapshot.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["max_loss_usdc"] = 30.0
    write_json(config, payload)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_if_run_repo_is_not_preflight_repo(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    status = input_root / "run" / "run_status.json"
    payload = json.loads(status.read_text(encoding="utf-8"))
    payload["remote_repo"] = "/remote/other-source"
    write_json(status, payload)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_runtime_source_digest_mismatch(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    provenance_path = input_root / "run" / acceptance.RUNTIME_SOURCE_PROVENANCE_NAME
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    write_json(provenance_path, payload)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_unclassified_producer_blocker(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    manifest_path = (
        input_root
            / "run"
            / "window_01"
            / "window_01"
            / "pulled_back_awsserver1"
        / "m2_fill_window_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["blocking_reasons"].append("unknown_order_state")
    write_json(manifest_path, payload)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_independently_rebuilds_confirmed_resting_exposure() -> None:
    base_ms = 1_783_600_000_000
    event_rows = [
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 100,
            "local_receive_time_ms": base_ms + 110,
            "bid_px": 65000,
            "ask_px": 65001,
            "bid_depth_btc": 0.02,
            "ask_depth_btc": 1.0,
        },
        {
            "event_kind": "trade",
            "event_time_ms": base_ms + 500,
            "local_receive_time_ms": base_ms + 510,
            "trade_px": 65000,
            "trade_size_btc": 0.004,
            "aggressor_side": "sell",
            "trade_id": "touch",
        },
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 1_100,
            "local_receive_time_ms": base_ms + 1_110,
            "bid_px": 64999,
            "ask_px": 65000,
            "bid_depth_btc": 0.03,
            "ask_depth_btc": 0.8,
        },
        {
            "event_kind": "trade",
            "event_time_ms": base_ms + 1_400,
            "local_receive_time_ms": base_ms + 1_410,
            "trade_px": 65000,
            "trade_size_btc": 0.004,
            "aggressor_side": "sell",
            "trade_id": "touch",
        },
        {
            "event_kind": "trade",
            "event_time_ms": base_ms + 1_500,
            "local_receive_time_ms": base_ms + 1_510,
            "trade_px": 64999,
            "trade_size_btc": 0.003,
            "aggressor_side": "sell",
            "trade_id": "through",
        },
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 2_100,
            "local_receive_time_ms": base_ms + 2_110,
            "bid_px": 65000,
            "ask_px": 65001,
            "bid_depth_btc": 0.025,
            "ask_depth_btc": 0.9,
        },
    ]
    interval_rows = [
        {
            "attempt_key": "0720T028:window_01:attempt_1",
            "attempt": 1,
            "side": "buy",
            "quote_px": 65000,
            "start_local_receive_time_ms": base_ms,
            "end_local_receive_time_ms": base_ms + 2_500,
            "resting_confirmed": True,
            "interval_status": "pass",
            "reconnect_count_start": 0,
            "reconnect_count_end": 0,
            "disconnect_count_start": 0,
            "disconnect_count_end": 0,
        }
    ]

    rows, quarantine, censors = (
        acceptance.rebuild_confirmed_resting_exposure_rows(
            event_rows=event_rows,
            interval_rows=interval_rows,
        )
    )

    assert quarantine == []
    assert censors == []
    assert len(rows) == 3
    assert sum(row["arrival_count"] for row in rows) == 2
    assert rows[0]["pre_trade_side_depth_btc"] == 0.02
    assert rows[1]["pre_trade_side_depth_btc"] == 0.03
    assert all(row["resting_confirmed"] is True for row in rows)


def test_acceptance_and_producer_rebuild_same_confirmed_exposure() -> None:
    base_ms = 1_783_600_000_000
    event_rows = [
        {
            "event_kind": "trade",
            "event_time_ms": base_ms + 50,
            "local_receive_time_ms": base_ms + 60,
            "trade_px": 65000.5,
            "trade_size_btc": 0.001,
            "aggressor_side": "sell",
            "trade_id": "leading-before-first-book",
        },
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 100,
            "local_receive_time_ms": base_ms + 110,
            "bid_px": 65000,
            "ask_px": 65001,
            "bid_depth_btc": 0.02,
            "ask_depth_btc": 1.0,
        },
        {
            "event_kind": "trade",
            "event_time_ms": base_ms + 500,
            "local_receive_time_ms": base_ms + 510,
            "trade_px": 65000,
            "trade_size_btc": 0.004,
            "aggressor_side": "sell",
            "trade_id": "touch",
        },
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 1_100,
            "local_receive_time_ms": base_ms + 1_110,
            "bid_px": 64999,
            "ask_px": 65000,
            "bid_depth_btc": 0.03,
            "ask_depth_btc": 0.8,
        },
        {
            "event_kind": "trade",
            "event_time_ms": base_ms + 1_500,
            "local_receive_time_ms": base_ms + 1_510,
            "trade_px": 65000,
            "trade_size_btc": 0.003,
            "aggressor_side": "buy",
            "trade_id": "sell-touch",
        },
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 2_100,
            "local_receive_time_ms": base_ms + 2_110,
            "bid_px": 65000,
            "ask_px": 65001,
            "bid_depth_btc": 0.025,
            "ask_depth_btc": 0.9,
        },
    ]
    interval_rows = [
        {
            "attempt_key": "0720T028:window_01:attempt_1",
            "attempt": 1,
            "side": "buy",
            "quote_px": 65000,
            "start_local_receive_time_ms": base_ms,
            "end_local_receive_time_ms": base_ms + 2_500,
            "resting_confirmed": True,
            "response_status_types": "resting",
            "interval_status": "pass",
            "interval_reason": "",
            "reconnect_count_start": 0,
            "reconnect_count_end": 0,
            "disconnect_count_start": 0,
            "disconnect_count_end": 0,
        },
        {
            "attempt_key": "0720T028:window_01:attempt_2",
            "attempt": 2,
            "side": "sell",
            "quote_px": 65000,
            "start_local_receive_time_ms": base_ms,
            "end_local_receive_time_ms": base_ms + 2_500,
            "resting_confirmed": True,
            "response_status_types": "resting",
            "interval_status": "pass",
            "interval_reason": "",
            "reconnect_count_start": 0,
            "reconnect_count_end": 0,
            "disconnect_count_start": 0,
            "disconnect_count_end": 0,
        },
    ]

    producer_rows, producer_quarantine, producer_censors = (
        online_estimators.build_confirmed_resting_exposure_rows(
            event_rows=event_rows,
            interval_rows=interval_rows,
        )
    )
    rebuilt_rows, rebuilt_quarantine, rebuilt_censors = (
        acceptance.rebuild_confirmed_resting_exposure_rows(
            event_rows=event_rows,
            interval_rows=interval_rows,
        )
    )

    assert [
        acceptance.resting_exposure_projection(row)
        for row in producer_rows
    ] == [
        acceptance.resting_exposure_projection(row)
        for row in rebuilt_rows
    ]
    assert [row["reason"] for row in producer_quarantine] == [
        row["reason"] for row in rebuilt_quarantine
    ]
    assert [
        acceptance.resting_censor_projection(row)
        for row in producer_censors
    ] == [
        acceptance.resting_censor_projection(row)
        for row in rebuilt_censors
    ]
    assert len(rebuilt_censors) == 2
    assert sum(
        row["arrival_count"]
        for row in rebuilt_rows
        if row["side"] == "buy"
    ) == 1


def test_acceptance_quarantines_unconfirmed_or_discontinuous_exposure() -> None:
    base_ms = 1_783_600_000_000
    event_rows = [
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 100,
            "local_receive_time_ms": base_ms + 110,
            "bid_px": 65000,
            "ask_px": 65001,
            "bid_depth_btc": 0.02,
            "ask_depth_btc": 1.0,
        },
        {
            "event_kind": "book",
            "event_time_ms": base_ms + 1_100,
            "local_receive_time_ms": base_ms + 1_110,
            "bid_px": 65000,
            "ask_px": 65001,
            "bid_depth_btc": 0.02,
            "ask_depth_btc": 1.0,
        },
    ]
    interval_rows = [
        {
            "attempt_key": "rejected",
            "attempt": 1,
            "side": "buy",
            "quote_px": 65000,
            "start_local_receive_time_ms": base_ms,
            "end_local_receive_time_ms": base_ms + 1_500,
            "resting_confirmed": False,
            "interval_status": "fail_closed",
            "reconnect_count_start": 0,
            "reconnect_count_end": 0,
            "disconnect_count_start": 0,
            "disconnect_count_end": 0,
        },
        {
            "attempt_key": "reconnected",
            "attempt": 2,
            "side": "sell",
            "quote_px": 65001,
            "start_local_receive_time_ms": base_ms,
            "end_local_receive_time_ms": base_ms + 1_500,
            "resting_confirmed": True,
            "interval_status": "pass",
            "reconnect_count_start": 0,
            "reconnect_count_end": 1,
            "disconnect_count_start": 0,
            "disconnect_count_end": 0,
        },
    ]

    rows, quarantine, _censors = (
        acceptance.rebuild_confirmed_resting_exposure_rows(
            event_rows=event_rows,
            interval_rows=interval_rows,
        )
    )

    assert rows == []
    assert sorted(row["reason"] for row in quarantine) == [
        "interval_not_confirmed_resting",
        "public_stream_continuity_changed",
    ]


def test_acceptance_quarantines_invalid_resting_exposure_config() -> None:
    rows, quarantine, censors = (
        acceptance.rebuild_confirmed_resting_exposure_rows(
            event_rows=[],
            interval_rows=[],
            bucket_ms=0,
        )
    )

    assert rows == []
    assert censors == []
    assert [row["reason"] for row in quarantine] == [
        "invalid_estimator_exposure_config"
    ]


def test_acceptance_trade_only_interval_is_not_censor_clean() -> None:
    base_ms = 1_783_600_000_000
    rows, quarantine, censors = (
        acceptance.rebuild_confirmed_resting_exposure_rows(
            event_rows=[
                {
                    "event_kind": "trade",
                    "event_time_ms": base_ms + 100,
                    "local_receive_time_ms": base_ms + 110,
                    "trade_px": 65000,
                    "trade_size_btc": 0.001,
                    "aggressor_side": "sell",
                    "trade_id": "trade-only-1",
                },
                {
                    "event_kind": "trade",
                    "event_time_ms": base_ms + 900,
                    "local_receive_time_ms": base_ms + 910,
                    "trade_px": 65000,
                    "trade_size_btc": 0.001,
                    "aggressor_side": "sell",
                    "trade_id": "trade-only-2",
                },
            ],
            interval_rows=[
                {
                    "attempt_key": "trade-only",
                    "attempt": 1,
                    "side": "buy",
                    "quote_px": 65000,
                    "start_local_receive_time_ms": base_ms,
                    "end_local_receive_time_ms": base_ms + 1_000,
                    "resting_confirmed": True,
                    "interval_status": "pass",
                    "reconnect_count_start": 0,
                    "reconnect_count_end": 0,
                    "disconnect_count_start": 0,
                    "disconnect_count_end": 0,
                }
            ],
        )
    )

    assert rows == []
    assert censors == []
    assert [row["reason"] for row in quarantine] == [
        "interval_reference_book_missing"
    ]


def test_acceptance_rejects_missing_duplicate_and_forged_censors() -> None:
    expected = {
        "schema_version": (
            acceptance.CONFIRMED_RESTING_CENSOR_SCHEMA_VERSION
        ),
        "row_kind": "leading_left_censor",
        "row_index": 0,
        "attempt_key": "attempt-1",
        "attempt": 1,
        "side": "buy",
        "start_exchange_time_ms": 100,
        "end_exchange_time_ms": 200,
        "duration_ms": 100,
        "reason": "leading_reference_book_left_censored",
        "inference_scope": (
            "manager_confirmed_resting_exposure_leading_"
            "event_time_left_censor"
        ),
    }

    assert acceptance.validate_confirmed_resting_censor_rows(
        persisted_rows=[],
        expected_rows=[expected],
    ) == ["missing_confirmed_resting_censor_row"]

    duplicate_reasons = (
        acceptance.validate_confirmed_resting_censor_rows(
            persisted_rows=[expected, dict(expected)],
            expected_rows=[expected],
        )
    )
    assert "duplicate_confirmed_resting_censor_row" in duplicate_reasons
    assert (
        "overlapping_confirmed_resting_censor_rows"
        in duplicate_reasons
    )

    forged = {
        **expected,
        "start_exchange_time_ms": 101,
        "duration_ms": 99,
    }
    assert acceptance.validate_confirmed_resting_censor_rows(
        persisted_rows=[forged],
        expected_rows=[expected],
    ) == ["confirmed_resting_censor_bounds_mismatch"]

    malformed = {
        **expected,
        "schema_version": "forged",
    }
    assert acceptance.validate_confirmed_resting_censor_rows(
        persisted_rows=[malformed],
        expected_rows=[expected],
    ) == ["malformed_confirmed_resting_censor_row"]


def test_acceptance_accepts_exact_header_only_quarantine_artifact(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    quarantine_path = install_manager_resting_exposure_contract(
        input_root
    )
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )

    assert quarantine_path.is_file()
    assert acceptance.read_csv_fieldnames(quarantine_path) == (
        acceptance.CONFIRMED_RESTING_QUARANTINE_FIELDS
    )
    assert read_csv(quarantine_path) == []
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"
    assert manifest["final_recommendation"] == (
        acceptance.PASSED_RECOMMENDATION
    )


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (
            "missing",
            "confirmed_resting_exposure_quarantine_artifact_present",
        ),
        (
            "malformed_header",
            "confirmed_resting_exposure_quarantine_schema",
        ),
        (
            "forged_row",
            "confirmed_resting_exposure_quarantine_exact_match",
        ),
        (
            "extra_cell",
            "confirmed_resting_exposure_quarantine_validation_reasons",
        ),
        (
            "invalid_integer",
            "confirmed_resting_exposure_quarantine_validation_reasons",
        ),
    ],
)
def test_acceptance_rejects_incomplete_or_forged_quarantine_artifact(
    tmp_path: Path,
    mutation: str,
    failed_check: str,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    quarantine_path = install_manager_resting_exposure_contract(
        input_root
    )
    if mutation == "missing":
        quarantine_path.unlink()
    elif mutation == "malformed_header":
        write_csv(
            quarantine_path,
            [],
            list(acceptance.CONFIRMED_RESTING_QUARANTINE_FIELDS[:-1]),
        )
    elif mutation == "forged_row":
        write_csv(
            quarantine_path,
            [
                {
                    "row_kind": "interval",
                    "row_index": 0,
                    "attempt_key": "forged-attempt",
                    "side": "buy",
                    "event_kind": "",
                    "event_time_ms": "",
                    "local_receive_time_ms": "",
                    "reason": "interval_not_confirmed_resting",
                    "inference_scope": (
                        "manager_confirmed_resting_exposure_"
                        "fail_closed_quarantine"
                    ),
                }
            ],
            list(acceptance.CONFIRMED_RESTING_QUARANTINE_FIELDS),
        )
    elif mutation == "extra_cell":
        quarantine_path.write_text(
            ",".join(acceptance.CONFIRMED_RESTING_QUARANTINE_FIELDS)
            + "\n"
            + (
                "interval,0,forged-attempt,buy,,,,"
                "interval_not_confirmed_resting,"
                "manager_confirmed_resting_exposure_"
                "fail_closed_quarantine,INJECTED\n"
            ),
            encoding="utf-8",
        )
    else:
        write_csv(
            quarantine_path,
            [
                {
                    "row_kind": "interval",
                    "row_index": 0,
                    "attempt_key": "forged-attempt",
                    "side": "buy",
                    "event_kind": "",
                    "event_time_ms": "not-an-int",
                    "local_receive_time_ms": "NaN",
                    "reason": "interval_not_confirmed_resting",
                    "inference_scope": (
                        "manager_confirmed_resting_exposure_"
                        "fail_closed_quarantine"
                    ),
                }
            ],
            list(acceptance.CONFIRMED_RESTING_QUARANTINE_FIELDS),
        )
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
    )
    lifecycle_rows = read_csv(
        tmp_path / "out" / "lifecycle_evidence_comparison.csv"
    )
    check = next(
        row for row in lifecycle_rows if row["check"] == failed_check
    )

    assert check["acceptance"] == "fail"
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )


def test_quarantine_canonical_comparison_uses_complete_row() -> None:
    expected = {
        "row_kind": "interval",
        "row_index": 0,
        "attempt_key": "attempt-1",
        "side": "buy",
        "event_kind": "",
        "event_time_ms": "",
        "local_receive_time_ms": "",
        "reason": "interval_not_confirmed_resting",
        "inference_scope": (
            "manager_confirmed_resting_exposure_"
            "fail_closed_quarantine"
        ),
    }
    forged = {**expected, "attempt_key": "attempt-2"}
    malformed_integer = {**expected, "event_time_ms": "not-an-int"}
    extra_cell = {**expected, None: ["INJECTED"]}

    assert [expected["reason"]] == [forged["reason"]]
    assert acceptance.canonical_resting_quarantine_rows(
        [expected]
    ) != acceptance.canonical_resting_quarantine_rows([forged])
    assert acceptance.canonical_resting_quarantine_rows(
        [expected]
    ) != acceptance.canonical_resting_quarantine_rows(
        [malformed_integer]
    )
    assert acceptance.validate_resting_quarantine_rows(
        [extra_cell]
    ) == ["confirmed_resting_quarantine_row_keys_invalid:0"]


def test_new_task_requires_manager_contract_after_all_derived_evidence_deleted(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    quarantine_path = install_manager_resting_exposure_contract(
        input_root
    )
    window = input_root / "run" / "window_01"
    quarantine_path.unlink()
    (window / "confirmed_resting_exposure_censor.csv").unlink()
    write_csv(
        window / "confirmed_resting_interval_contract.csv",
        [],
        watcher.manager_resting_interval_fieldnames(),
    )
    write_csv(
        window / "quote_exposure_intervals.csv",
        [],
        online_estimators.quote_exposure_fieldnames(),
    )
    estimator_path = window / "online_estimator_snapshot.json"
    estimator = json.loads(estimator_path.read_text(encoding="utf-8"))
    estimator.pop("manager_resting_exposure")
    write_json(estimator_path, estimator)
    seal_run(input_root)

    manifest = run_task12_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id="0721T033",
    )
    lifecycle_rows = read_csv(
        tmp_path / "out" / "lifecycle_evidence_comparison.csv"
    )
    checks = {row["check"]: row for row in lifecycle_rows}

    assert checks[
        "confirmed_resting_censor_artifact_present"
    ]["acceptance"] == "fail"
    assert checks[
        "confirmed_resting_exposure_quarantine_artifact_present"
    ]["acceptance"] == "fail"
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
    assert manifest["final_recommendation"] == (
        acceptance.BLOCKED_RECOMMENDATION
    )


def test_acceptance_independently_rebuilds_manager_resting_interval_contract() -> None:
    intent = executor.OrderIntent(
        symbol="BTC",
        is_buy=True,
        size_btc=0.005,
        limit_px=65000,
        cloid="cloid-buy",
    )
    order_result = {
        "status": "ok",
        "response": {
            "data": {
                "statuses": [
                    {
                        "resting": {
                            "oid": 101,
                            "cloid": intent.cloid,
                        }
                    }
                ]
            }
        },
    }
    manager_action = {
        "action": "submitted",
        "state": "resting",
        "query_status": "resting",
        "order_endpoint_called": True,
        "side": "buy",
        "cloid": intent.cloid,
        "submit_start_ms": 1_000,
        "submit_end_ms": 1_100,
        "order_result": order_result,
    }
    hold_observation = {
        "status": "pass",
        "reason": "",
        "deadline_overrun_seconds": 0.0,
        "reconnect_count_start": 0,
        "reconnect_count_end": 0,
        "disconnect_count_start": 0,
        "disconnect_count_end": 0,
    }
    manager_cycle = {
        "hold_observation": hold_observation,
        "reconcile_result": {"actions": [manager_action]},
        "cancel_actions": [
            {
                "attempt": 1,
                "cloid": intent.cloid,
                "cancel_request_time_ms": 4_100,
            }
        ],
        "intents": [intent],
    }
    producer_rows = watcher.build_manager_resting_interval_rows(
        task_id="0720T028",
        window_id=1,
        first_attempt_id=1,
        manager_cycle=manager_cycle,
    )
    persisted_result = {
        **order_result,
        "manager_actions": [
            {
                key: value
                for key, value in manager_action.items()
                if key != "order_result"
            }
        ],
        "side": "buy",
    }
    _, order_response_rows, _ = watcher.build_inline_order_evidence(
        order_intents=[intent],
        attempt_rows=[
            {
                "attempt_id": 1,
                "attempt_key": "0720T028:window_01:attempt_1",
                "side": "buy",
                "order_endpoint_called": True,
            }
        ],
        order_results=[persisted_result],
    )

    rebuilt_rows, reasons = (
        acceptance.rebuild_manager_resting_interval_contract(
            order_response_rows=order_response_rows,
            intents_by_side={"buy": {"limit_px": 65000}},
            cancel_results=[
                {
                    "attempt": 1,
                    "cancel_request_time_ms": 4_100,
                }
            ],
            hold_observation=hold_observation,
        )
    )

    assert reasons == []
    assert [
        acceptance.manager_resting_interval_projection(row)
        for row in producer_rows
    ] == [
        acceptance.manager_resting_interval_projection(row)
        for row in rebuilt_rows
    ]

    tampered_rows = [dict(producer_rows[0])]
    tampered_rows[0]["end_local_receive_time_ms"] = 4_101
    assert [
        acceptance.manager_resting_interval_projection(row)
        for row in tampered_rows
    ] != [
        acceptance.manager_resting_interval_projection(row)
        for row in rebuilt_rows
    ]


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        (
            "pump_shutdown_contract_version",
            "",
            "manager_hold_pump_shutdown_contract_invalid",
        ),
        (
            "pump_stop_acknowledged",
            "true",
            "manager_hold_pump_stop_acknowledged_invalid",
        ),
        (
            "pump_stop_acknowledged",
            False,
            "manager_hold_public_pump_stop_unacknowledged",
        ),
        (
            "pump_read_inflight_after_stop_wait",
            True,
            "manager_hold_public_pump_read_still_inflight",
        ),
        (
            "pump_source_close_required",
            "true",
            "manager_hold_pump_source_close_required_invalid",
        ),
        (
            "pump_source_closed",
            "true",
            "manager_hold_pump_source_closed_invalid",
        ),
        (
            "pump_source_closed",
            False,
            "manager_hold_public_source_not_closed",
        ),
        (
            "pump_source_close_error",
            "close failed",
            "manager_hold_public_source_close_failed",
        ),
        (
            "pump_shutdown_wait_seconds",
            0.251,
            "manager_hold_pump_shutdown_wait_invalid",
        ),
        (
            "pump_shutdown_wait_timeout_seconds",
            0.5,
            "manager_hold_pump_shutdown_wait_timeout_invalid",
        ),
        (
            "manager_cancel_batch_started_monotonic",
            100.01,
            "manager_hold_pump_cancel_timeline_unordered",
        ),
    ],
)
def test_manager_hold_shutdown_proof_fails_closed(
    field: str,
    value: object,
    expected_reason: str,
) -> None:
    hold_observation = valid_manager_hold_shutdown_observation()
    hold_observation[field] = value

    rows, reasons = (
        acceptance.rebuild_manager_resting_interval_contract(
            order_response_rows=[],
            intents_by_side={},
            cancel_results=valid_manager_cancel_timeline_rows(),
            hold_observation=hold_observation,
            require_pump_shutdown_proof=True,
        )
    )

    assert rows == []
    assert reasons == [expected_reason]


def test_manager_hold_shutdown_proof_accepts_nonclosable_source() -> None:
    hold_observation = valid_manager_hold_shutdown_observation(
        source_close_required=False,
    )

    rows, reasons = (
        acceptance.rebuild_manager_resting_interval_contract(
            order_response_rows=[],
            intents_by_side={},
            cancel_results=valid_manager_cancel_timeline_rows(),
            hold_observation=hold_observation,
            require_pump_shutdown_proof=True,
        )
    )

    assert rows == []
    assert reasons == []


def test_manager_hold_shutdown_proof_requires_closable_live_source() -> None:
    hold_observation = valid_manager_hold_shutdown_observation(
        source_close_required=False,
    )

    rows, reasons = (
        acceptance.rebuild_manager_resting_interval_contract(
            order_response_rows=[],
            intents_by_side={},
            cancel_results=valid_manager_cancel_timeline_rows(),
            hold_observation=hold_observation,
            require_pump_shutdown_proof=True,
            require_pump_source_close=True,
        )
    )

    assert rows == []
    assert reasons == [
        "manager_hold_pump_source_close_required_false"
    ]


def test_manager_hold_shutdown_rejects_cross_artifact_timeline_forgery() -> None:
    hold_observation = valid_manager_hold_shutdown_observation()
    forged_cancel_rows = valid_manager_cancel_timeline_rows()
    forged_cancel_rows[0][
        "manager_cancel_batch_started_monotonic"
    ] = 100.06

    rows, reasons = (
        acceptance.rebuild_manager_resting_interval_contract(
            order_response_rows=[],
            intents_by_side={},
            cancel_results=forged_cancel_rows,
            hold_observation=hold_observation,
            require_pump_shutdown_proof=True,
            require_pump_source_close=True,
        )
    )

    assert rows == []
    assert reasons == [
        "manager_hold_cancel_timeline_cross_artifact_mismatch"
    ]


def test_t038_t039_rollout_predicates_preserve_history_boundaries() -> None:
    assert acceptance.canonical_primary_outcome_required(
        "0721T037"
    ) is False
    assert acceptance.manager_hold_pump_shutdown_required(
        "0721T037"
    ) is False
    assert acceptance.canonical_primary_outcome_required(
        "0721T038"
    ) is True
    assert acceptance.manager_hold_pump_shutdown_required(
        "0721T038"
    ) is True
    assert acceptance.raw_stage_evidence_required(
        "0721T038"
    ) is False
    assert acceptance.raw_stage_evidence_required(
        "0721T039"
    ) is True


def test_manager_batch_attempt_bridge_requires_exact_two_sided_identity() -> None:
    rows = [
        {
            "attempt": "1",
            "attempt_id": "1",
            "attempt_key": "0721T038:window_01:attempt_1",
            "event_sequence": "1896",
            "window_id": "window_01",
            "side": "buy",
            "order_endpoint_called": "True",
        },
        {
            "attempt": "2",
            "attempt_id": "2",
            "attempt_key": "0721T038:window_01:attempt_2",
            "event_sequence": "1896",
            "window_id": "window_01",
            "side": "sell",
            "order_endpoint_called": "True",
        },
    ]

    canonical = (
        acceptance.canonical_manager_batch_attempt_rows_by_event(
            attempt_rows=rows,
            expected_task_id="0721T038",
            expected_window_id="window_01",
        )
    )
    assert [row["attempt_id"] for row in canonical[1896]] == ["1", "2"]

    hostile_cases = [
        [
            *rows,
            {**rows[1], "attempt_key": "0721T038:window_01:attempt_3"},
        ],
        [{**rows[0], "order_endpoint_called": "False"}, rows[1]],
        [{**rows[0], "side": "sell"}, rows[1]],
        [{**rows[0], "attempt_key": "0721T039:window_01:attempt_1"}, rows[1]],
        [{**rows[0], "window_id": "window_02"}, rows[1]],
    ]
    for hostile in hostile_cases:
        assert (
            acceptance.canonical_manager_batch_attempt_rows_by_event(
                attempt_rows=hostile,
                expected_task_id="0721T038",
                expected_window_id="window_01",
            )
            == {}
        )


def valid_t039_anti_drift_row() -> dict[str, object]:
    return {
        "attempt": 1,
        "event_sequence": 1,
        "phase": "post_open_orders_pre_submit_gate",
        "source_channel": "trades",
        "source_event_exchange_time_ms": 1_000,
        "side": "buy",
        "limit_px": 100.0,
        "current_bid": 100.0,
        "current_ask": 101.0,
        "status": "block",
        "reason": (
            "adverse_trade_pressure_with_recent_adverse_bbo"
        ),
        "touch_stability_ms": 300,
        "min_stable_ms": 250,
        "last_adverse_bbo_ms": 700,
        "elapsed_since_adverse_bbo_ms": 300,
        "adverse_trade_qty_btc": "0.04",
        "favorable_trade_qty_btc": "0",
        "fill_support_touch_qty_btc": "0",
        "fill_support_visible_queue_depletion_qty_btc": "0",
        "adverse_strict_through_qty_btc": "0.04",
        "adverse_bbo_move": True,
        "neutral_or_opposite_flow_qty_btc": "0",
        "adverse_flow_ratio": "inf",
        "min_pressure_qty_btc": "0.01",
        "pressure_ratio_threshold": 2.0,
        "adverse_flow_status": "block",
        "current_cross_risk": False,
    }


def test_raw_anti_drift_rebuild_rejects_synchronized_derived_pass() -> None:
    row = valid_t039_anti_drift_row()
    row.update(
        {
            "limit_px": 101.0,
            "status": "pass",
            "reason": "",
            "adverse_flow_status": "pass",
            "current_cross_risk": False,
        }
    )
    validation_reasons: list[str] = []

    rebuilt = acceptance.rebuild_anti_drift_gate_outcome(
        row,
        row_index=0,
        validation_reasons=validation_reasons,
    )

    assert rebuilt == (
        "block",
        (
            "current_touch_would_cross_post_only;"
            "adverse_trade_pressure_with_recent_adverse_bbo"
        ),
    )
    assert "anti_drift_cross_risk_mismatch:0" in validation_reasons
    assert "anti_drift_flow_status_mismatch:0" in validation_reasons


@pytest.mark.parametrize(
    ("field", "value", "reason_prefix"),
    [
        ("side", "hold", "anti_drift_side_limit_bbo_invalid:"),
        ("limit_px", "", "anti_drift_side_limit_bbo_invalid:"),
        ("current_ask", 99.0, "anti_drift_side_limit_bbo_invalid:"),
        (
            "min_pressure_qty_btc",
            "0.02",
            "anti_drift_policy_threshold_mismatch:",
        ),
        (
            "pressure_ratio_threshold",
            "3.0",
            "anti_drift_policy_threshold_mismatch:",
        ),
        (
            "min_stable_ms",
            "100",
            "anti_drift_policy_threshold_mismatch:",
        ),
        (
            "adverse_bbo_move",
            "not-a-bool",
            "invalid_boolean:anti_drift_adverse_bbo_move:",
        ),
        (
            "adverse_trade_qty_btc",
            "0.03",
            "anti_drift_quantity_projection_mismatch:",
        ),
        (
            "adverse_flow_ratio",
            "2",
            "anti_drift_flow_ratio_mismatch:",
        ),
        (
            "adverse_flow_status",
            "pass",
            "anti_drift_flow_status_mismatch:",
        ),
        (
            "current_cross_risk",
            True,
            "anti_drift_cross_risk_mismatch:",
        ),
    ],
)
def test_raw_anti_drift_rebuild_rejects_field_forgery(
    field: str,
    value: object,
    reason_prefix: str,
) -> None:
    row = valid_t039_anti_drift_row()
    row[field] = value
    validation_reasons: list[str] = []

    acceptance.rebuild_anti_drift_gate_outcome(
        row,
        row_index=0,
        validation_reasons=validation_reasons,
    )

    assert any(
        reason.startswith(reason_prefix)
        for reason in validation_reasons
    )
