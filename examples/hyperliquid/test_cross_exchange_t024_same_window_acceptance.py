from __future__ import annotations

import csv
import json
from pathlib import Path

from examples.hyperliquid import cross_exchange_t024_same_window_acceptance as acceptance


SOURCE_COMMIT = "a" * 40
TASK_ID = "0718T024"


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
    write_json(
        root / "preflight" / "orchestrator_preflight.json",
        {
            "status": "pass",
            "preflight_only": True,
            "task_id": TASK_ID,
            "source_commit": SOURCE_COMMIT,
            "strategy_activation": {
                "dynamic_spread_activation_enabled": False,
                "fill_feedback_activation_enabled": False,
                "inventory_skew_activation_enabled": False,
                "multi_level_activation_enabled": False,
                "actual_quote_behavior_changed": False,
            },
        },
    )
    (run / "source_commit.txt").parent.mkdir(parents=True, exist_ok=True)
    (run / "source_commit.txt").write_text(SOURCE_COMMIT + "\n", encoding="utf-8")
    write_json(run / "run_complete.json", {"task_id": TASK_ID, "state": "complete"})
    write_json(run / "run_status.json", {"task_id": TASK_ID, "state": "complete"})
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
        },
    )
    write_json(live / "executor_manifest.json", {"task_id": TASK_ID, "artifact_window_id": 1})
    write_json(live / "private_order_response_audit.json", {"order_submission_attempted": True})
    write_json(live / "cancel_shutdown_proof.json", {"proof_status": "pass"})
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
