from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from examples.hyperliquid import cross_exchange_t024_same_window_acceptance as acceptance
from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


SOURCE_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=acceptance.PROJECT_ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
TASK_ID = "0719T001"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_artifact(root: Path) -> Path:
    run = root / "run"
    window = run / "window_01"
    live = window / "window_1" / "pulled_back_awsserver1"
    raw_tracked_refs = [{"attempt": 1, "oid": 101, "cloid": "cloid-a"}]
    raw_cancel_results = [
        {
            "method": "cancel",
            "attempt": 1,
            "oid": 101,
            "cloid": "cloid-a",
            "result": {
                "status": "ok",
                "response": {"data": {"statuses": ["success"]}},
            },
        }
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
        "python",
        "watcher.py",
        "--event-driven-live",
        "--watcher-seconds",
        "900.0",
        "--max-order-size",
        "0.005",
        "--max-loss-usdc",
        "1.0",
        "--max-position-btc",
        "0.01",
        "--max-real-order-submissions",
        "2",
        "--artifact-task-id",
        TASK_ID,
        "--artifact-window-id",
        "1",
    ]
    source_digests, source_error = acceptance.expected_git_source_snapshot(SOURCE_COMMIT)
    assert source_error == ""
    write_json(
        run / acceptance.RUNTIME_SOURCE_PROVENANCE_NAME,
        {
            "status": "pass",
            "task_id": TASK_ID,
            "source_commit": SOURCE_COMMIT,
            "source_commit_source": "source_commit.txt",
            "sealed_before_watcher_start": True,
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
            "strategy_activation": {
                "dynamic_spread_activation_enabled": False,
                "fill_feedback_activation_enabled": False,
                "inventory_skew_activation_enabled": False,
                "multi_level_activation_enabled": False,
                "actual_quote_behavior_changed": False,
            },
        },
    )
    write_json(run / "run_complete.json", {"task_id": TASK_ID, "state": "complete"})
    write_json(
        run / "run_status.json",
        {"task_id": TASK_ID, "state": "complete", "remote_repo": "/remote/t025-source"},
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
            "trigger_found": True,
            "event_driven_guard_status": "pass",
            "selected_candidate": {"fresh_touch_decision": {"allowed": True}},
            "live_submissions_count": 1,
            "fill_count": 0,
            "maker_fill_count": 0,
            "dynamic_spread_activation_enabled": False,
            "actual_quote_behavior_changed": False,
        },
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
            "order_status_types": ["resting"],
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
        },
    )
    write_json(live / "executor_manifest.json", {"task_id": TASK_ID, "artifact_window_id": 1})
    write_json(live / "private_order_response_audit.json", {"order_submission_attempted": True})
    fill_manifest = json.loads(
        (live / "m2_fill_window_manifest.json").read_text(encoding="utf-8")
    )
    write_json(
        live / "cancel_shutdown_proof.json",
        {
            "proof_status": "pass",
            "tracked_refs": persisted_tracked_refs,
            "cancel_results": persisted_cancel_results,
            "fill_reconciliation": fill_manifest["fill_reconciliation"],
        },
    )
    write_json(live / "account_inventory_snapshots.json", {"post_state": {"assetPositions": []}})
    write_json(live / "max_loss_monitor_summary.json", {"status": "pass", "estimated_loss_usdc": 0.0})
    write_json(
        live / "user_fills_pullback_audit.json",
        {"fill_attribution_summary": {"unattributed_fill_count": 0}},
    )
    write_csv(
        live / "quote_attempt_matrix.csv",
        [
            {
                "attempt_key": f"{TASK_ID}:window_01:attempt_1",
                "side": "buy",
                "limit_px": "64000.0",
                "size_btc": "0.002",
                "order_status_types": "resting",
            }
        ],
        ["attempt_key", "side", "limit_px", "size_btc", "order_status_types"],
    )
    write_csv(
        live / "order_intent_audit.csv",
        [{"side": "buy", "limit_px": "64000.0", "size_btc": "0.002", "time_in_force": "Alo"}],
        ["side", "limit_px", "size_btc", "time_in_force"],
    )
    write_csv(live / "live_fill_ledger.csv", [], ["fill_id"])
    write_csv(live / "fill_attribution_evidence.csv", [], ["fill_id"])
    write_csv(live / "fill_liquidity_role_evidence.csv", [], ["fill_id"])
    return root


def test_acceptance_passes_exact_no_fill_lifecycle(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")

    manifest = acceptance.run_acceptance(
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


@pytest.mark.parametrize("attempt", [1, "1"])
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
    assert reconciliation["reference_rows"][0]["attempt"] == 1


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
        / "window_1"
        / "pulled_back_awsserver1"
        / "m2_fill_window_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fill_reconciliation"].pop("cancel_reference_reconciliation")
    write_json(manifest_path, payload)

    manifest = acceptance.run_acceptance(
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
        / "window_1"
        / "pulled_back_awsserver1"
        / "m2_fill_window_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["fill_reconciliation"]["cancel_reference_reconciliation"][
        "cancel_evidence_rows"
    ] = []
    write_json(manifest_path, payload)

    manifest = acceptance.run_acceptance(
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
        / "window_1"
        / "pulled_back_awsserver1"
        / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["cancel_results"][0]["oid"] = 999
    proof["cancel_results"][0]["cloid"] = "forged-target"
    write_json(proof_path, proof)

    manifest = acceptance.run_acceptance(
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
        / "window_1"
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

    manifest = acceptance.run_acceptance(
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
        / "window_1"
        / "pulled_back_awsserver1"
        / "cancel_shutdown_proof.json"
    )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof.pop("tracked_refs")
    proof.pop("cancel_results")
    write_json(proof_path, proof)

    manifest = acceptance.run_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_stale_inner_identity(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    config = input_root / "run" / "window_01" / "window_1" / "pulled_back_awsserver1" / "approved_config_snapshot.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["task_id"] = "0622T004"
    write_json(config, payload)

    manifest = acceptance.run_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def test_acceptance_fails_runtime_envelope_mismatch(tmp_path: Path) -> None:
    input_root = make_artifact(tmp_path / "input")
    config = input_root / "run" / "window_01" / "window_1" / "pulled_back_awsserver1" / "approved_config_snapshot.json"
    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["max_loss_usdc"] = 30.0
    write_json(config, payload)

    manifest = acceptance.run_acceptance(
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

    manifest = acceptance.run_acceptance(
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

    manifest = acceptance.run_acceptance(
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
        / "window_1"
        / "pulled_back_awsserver1"
        / "m2_fill_window_manifest.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["blocking_reasons"].append("unknown_order_state")
    write_json(manifest_path, payload)

    manifest = acceptance.run_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"
