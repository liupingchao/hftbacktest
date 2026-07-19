from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest

from examples.hyperliquid import cross_exchange_t024_same_window_acceptance as acceptance
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
        / "window_1"
        / "pulled_back_awsserver1"
    )


def assert_acceptance_blocked(input_root: Path, output_dir: Path) -> None:
    manifest = acceptance.run_acceptance(
        input_root=input_root,
        output_dir=output_dir,
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )
    assert manifest["final_recommendation"] == acceptance.BLOCKED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "fail"


def make_artifact(root: Path) -> Path:
    run = root / "run"
    window = run / "window_01"
    live = window / "window_1" / "pulled_back_awsserver1"
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
        "python",
        "watcher.py",
        "--event-driven-edge-gate-live",
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
        "--requote-attempts",
        "2",
        "--exchange-reconciled-manager",
        "--hyperliquid-l2book-fast",
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
            "envelope": {
                "exact_envelope_profile": "two-sided-manager",
                "mode": "event-driven-edge-gate-live",
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
            "artifact_window_id": 1,
            "trigger_found": True,
            "event_driven_guard_status": "pass",
            "selected_candidate": {"fresh_touch_decision": {"allowed": True}},
            "live_submissions_count": 2,
            "fill_count": 0,
            "maker_fill_count": 0,
            "dynamic_spread_activation_enabled": False,
            "actual_quote_behavior_changed": False,
            "task7_exchange_reconciled_manager_enabled": True,
            "edge_gate_enabled": True,
            "edge_gate_live_compatible_source_available": True,
            "edge_gate_source_status": "decision_time_public_fair_mid_provider",
            "edge_gate_pass_count": 1,
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
        },
    )
    write_json(live / "executor_manifest.json", {"task_id": TASK_ID, "artifact_window_id": 1})
    write_json(
        live / "private_order_response_audit.json",
        {
            "order_submission_attempted": True,
            "order_status_rows": [
                {"attempt": 1, "side": "buy", "status_type": "resting"},
                {"attempt": 2, "side": "sell", "status_type": "resting"},
            ],
            "order_results": [
                {
                    "status": "ok",
                    "response": {
                        "data": {
                            "statuses": [
                                {"resting": {"oid": 101, "cloid": "cloid-buy"}}
                            ]
                        }
                    },
                },
                {
                    "status": "ok",
                    "response": {
                        "data": {
                            "statuses": [
                                {"resting": {"oid": 102, "cloid": "cloid-sell"}}
                            ]
                        }
                    },
                },
            ],
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
                "attempt": 1,
                "attempt_id": 1,
                "attempt_key": f"{TASK_ID}:window_01:attempt_1",
                "side": "buy",
                "limit_px": "64000.0",
                "size_btc": "0.002",
                "order_status_types": "resting",
                "order_endpoint_called": True,
            },
            {
                "attempt": 2,
                "attempt_id": 2,
                "attempt_key": f"{TASK_ID}:window_01:attempt_2",
                "side": "sell",
                "limit_px": "66000.0",
                "size_btc": "0.002",
                "order_status_types": "resting",
                "order_endpoint_called": True,
            },
        ],
        [
            "attempt",
            "attempt_id",
            "attempt_key",
            "side",
            "limit_px",
            "size_btc",
            "order_status_types",
            "order_endpoint_called",
        ],
    )
    write_csv(
        live / "order_intent_audit.csv",
        [
            {
                "side": "buy",
                "limit_px": "64000.0",
                "size_btc": "0.002",
                "time_in_force": "Alo",
            },
            {
                "side": "sell",
                "limit_px": "66000.0",
                "size_btc": "0.002",
                "time_in_force": "Alo",
            },
        ],
        ["side", "limit_px", "size_btc", "time_in_force"],
    )
    write_csv(live / "live_fill_ledger.csv", [], ["fill_id"])
    write_csv(live / "fill_attribution_evidence.csv", [], ["fill_id"])
    write_csv(live / "fill_liquidity_role_evidence.csv", [], ["fill_id"])
    return root


def write_actual_two_sided_live_artifacts(input_root: Path) -> None:
    live = (
        input_root
        / "run"
        / "window_01"
        / "window_1"
        / "pulled_back_awsserver1"
    )
    intents = [
        executor.OrderIntent(
            symbol="BTC",
            is_buy=True,
            size_btc=0.002,
            limit_px=64000.0,
            cloid="actual-writer-buy",
        ),
        executor.OrderIntent(
            symbol="BTC",
            is_buy=False,
            size_btc=0.002,
            limit_px=66000.0,
            cloid="actual-writer-sell",
        ),
    ]
    tracked_refs = [
        {"attempt": 1, "oid": 201, "cloid": intents[0].cloid},
        {"attempt": 2, "oid": 202, "cloid": intents[1].cloid},
    ]
    cancel_results = [
        {
            "attempt": row["attempt"],
            "oid": row["oid"],
            "cloid": row["cloid"],
            "result": {
                "status": "ok",
                "response": {
                    "data": {"statuses": [{"success": str(row["oid"])}]}
                },
            },
        }
        for row in tracked_refs
    ]
    order_results = [
        {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {
                            "resting": {
                                "oid": tracked_refs[index]["oid"],
                                "cloid": intent.cloid,
                            }
                        }
                    ]
                }
            },
            "side": "buy" if intent.is_buy else "sell",
        }
        for index, intent in enumerate(intents)
    ]
    attempt_rows = [
        {
            "attempt": index,
            "side": "buy" if intent.is_buy else "sell",
            "limit_px": intent.limit_px,
            "size_btc": intent.size_btc,
            "notional_usdc": intent.notional_usdc,
            "post_only_tif": "Alo",
            "order_endpoint_called": True,
            "order_status_types": "resting",
            "tracked_ref_count": 1,
            "cancel_endpoint_called": True,
            "shutdown_proof_status": "pass",
        }
        for index, intent in enumerate(intents, start=1)
    ]
    watcher.write_inline_order_artifacts(
        output_dir=live,
        env_file=str(live / ".env"),
        env_load={"loaded_keys": []},
        config=executor.TinyLiveConfig(
            artifact_dir=live,
            live_mode=True,
            operator_ack=fill_window.OPERATOR_ACK,
            use_schedule_cancel=False,
            max_order_size_btc=0.005,
            max_position_btc=0.01,
            max_real_order_submissions=2,
            max_loss_usdc=1.0,
        ),
        precision=executor.mock_precision(),
        endpoint_flags={
            "private_endpoint_called": True,
            "real_order_endpoint_called": True,
            "real_cancel_endpoint_called": True,
        },
        order_intents=intents,
        attempt_rows=attempt_rows,
        guard_rows=[],
        latency_rows=[],
        reject_rows=[],
        quote_guard_rows=[],
        order_status_rows=[
            {"attempt": 1, "side": "buy", "status_type": "resting"},
            {"attempt": 2, "side": "sell", "status_type": "resting"},
        ],
        order_results=order_results,
        cancel_results=cancel_results,
        tracked_refs=tracked_refs,
        final_open_orders=[],
        fill_rows=[],
        fill_attribution_rows=[],
        fill_attribution_summary={
            "attributed_fill_count": 0,
            "unattributed_fill_count": 0,
            "fail_closed_reasons": [],
        },
        pre_open_orders=[],
        post_state={"assetPositions": []},
        user_fees={},
        market_markout={},
        blocking_reasons=[],
        max_order_size_btc=0.005,
        requote_attempts_requested=2,
        artifact_task_id=TASK_ID,
        artifact_window_id=1,
        user_fills_pullbacks=[{"fill_count": 0, "fills": []}],
    )


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


def test_acceptance_passes_actual_two_sided_writer_artifacts(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    write_actual_two_sided_live_artifacts(input_root)

    manifest = acceptance.run_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"
    assert manifest["live_summary"]["submissions"] == 2


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


def test_acceptance_fails_synchronized_false_success_summaries(
    tmp_path: Path,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    live_dir = (
        input_root
        / "run"
        / "window_01"
        / "window_1"
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
