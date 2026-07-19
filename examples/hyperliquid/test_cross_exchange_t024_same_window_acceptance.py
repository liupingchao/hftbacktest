from __future__ import annotations

import csv
import json
import subprocess
import time
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
        / "window_01"
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
        "--quote-hold-seconds",
        "3",
        "--wait-seconds",
        "10",
        "--env-file",
        str(root / ".env"),
        "--exchange-reconciled-manager",
        "--hyperliquid-l2book-fast",
        "--artifact-task-id",
        TASK_ID,
        "--artifact-window-id",
        "1",
        "--run-id",
        f"{TASK_ID}:window_01",
        "--output-dir",
        str(window),
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
            "watcher_commands": [command],
            "envelope": {
                "exact_envelope_profile": "two-sided-manager",
                "mode": "event-driven-edge-gate-live",
                "window_seconds": 900.0,
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
            "watcher_seconds_requested": 900.0,
            "trigger_found": True,
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
            "max_real_order_submissions": 2,
            "max_order_size_btc": 0.005,
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
    return root


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
    fill_rows: list[dict[str, object]] = []
    role_rows: list[dict[str, object]] = []
    for attempt in filled_attempts:
        intent = intents[attempt]
        response = responses[attempt]
        resting = response["result"]["response"]["data"]["statuses"][0][
            "resting"
        ]
        oid_token = str(resting.get("oid_token") or "")
        if not oid_token:
            oid_token = fill_window.reference_identity_token(
                "oid",
                resting.get("oid"),
            )
        cloid_token = str(resting.get("cloid_token") or "")
        if not cloid_token:
            cloid_token = fill_window.reference_identity_token(
                "cloid",
                resting.get("cloid"),
            )
        fill_id = f"fill-{attempt}"
        fill_rows.append(
            {
                "source_window": "window_01",
                "window_id": "window_01",
                "attempt_id": attempt,
                "attempt_key": intent["attempt_key"],
                "fill_id": fill_id,
                "side": intent["side"],
                "qty_btc": intent["size_btc"],
                "price_usdc": intent["limit_px"],
                "intent_price_usdc": intent["limit_px"],
                "mark_price_usdc": intent["limit_px"],
                "fee_usdc": "0",
                "rebate_usdc": "0",
                "liquidity": "maker",
                "attribution_status": "matched_tracked_oid",
                "attribution_source": "user_fills_by_time_oid",
                "source_oid_present": True,
                "source_oid_token": oid_token,
                "source_cloid_token": cloid_token,
                "source_has_liquidity_role": True,
                "fill_time_ms": 1_000 + attempt,
                "attribution_interval_start_ms": 900,
                "attribution_interval_end_ms": 2_000,
                "duplicate_pullback_count": 0,
                "ambiguity_reason": "",
                "pullback_phases": "finalize",
                "fill_payload_fingerprint": f"fingerprint-{attempt}",
            }
        )
        role_rows.append(
            {
                "source_window": "window_01",
                "window_id": "window_01",
                "attempt_id": attempt,
                "attempt_key": intent["attempt_key"],
                "fill_id": fill_id,
                "liquidity": "maker",
                "liquidity_role_status": "confirmed_maker",
                "liquidity_role_source": "exchange_fill_crossed_field",
                "source_has_liquidity_role": True,
                "source_oid_present": True,
                "attribution_status": "matched_tracked_oid",
                "fee_pnl_role_gate": "pass_role_known",
            }
        )
    write_csv(
        live / "live_fill_ledger.csv",
        fill_rows,
        fill_window.live_fill_ledger_fieldnames(),
    )
    write_csv(
        live / "fill_attribution_evidence.csv",
        fill_rows,
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
            "fill_attribution_summary": {
                "attributed_fill_count": len(fill_rows),
                "unattributed_fill_count": 0,
                "fail_closed_reasons": [],
            }
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
        event_source_fn=lambda: _manager_source(
            [
                _manager_l2(now_ms),
                _manager_l2(now_ms + 300),
                _manager_trade(now_ms + 301),
                _manager_l2(now_ms + 302),
            ]
        ),
        live_client_factory=lambda: client,
        control_state_dir=control_dir,
        max_loss_usdc=1.0,
        max_position_btc=0.01,
    )
    return client


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_root = make_artifact(tmp_path / "input")
    client = write_actual_two_sided_live_artifacts(
        input_root,
        monkeypatch,
    )

    manifest = acceptance.run_acceptance(
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

    manifest = acceptance.run_acceptance(
        input_root=input_root,
        output_dir=tmp_path / "out",
        expected_task_id=TASK_ID,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert manifest["final_recommendation"] == acceptance.PASSED_RECOMMENDATION
    assert manifest["mechanism_and_evidence_integrity_acceptance"] == "pass"


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
            / "window_01"
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
            / "window_01"
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
            / "window_01"
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
    config = input_root / "run" / "window_01" / "window_01" / "pulled_back_awsserver1" / "approved_config_snapshot.json"
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
    config = input_root / "run" / "window_01" / "window_01" / "pulled_back_awsserver1" / "approved_config_snapshot.json"
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
            / "window_01"
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
