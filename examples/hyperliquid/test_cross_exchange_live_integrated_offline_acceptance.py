from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from examples.hyperliquid import hyperliquid_tiny_live_m2_fill_window as fill_window
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor
from examples.hyperliquid.test_cross_exchange_live_remote_orchestrator import (
    PROJECT_ROOT,
    assert_pid_gone,
    orchestrator_command,
    read_json,
)


def write_integrated_fake_watcher(path: Path) -> None:
    fixtures = {
        "1": {
            "window_id": 1,
            "attempts": [
                {
                    "attempt_id": 1,
                    "submit_start_ms": 1_000,
                    "submit_end_ms": 1_100,
                    "terminal_end_ms": 1_500,
                    "oid": 101,
                    "cloid": "w1-a1",
                }
            ],
            "pullbacks": [
                {
                    "phase": "after_response",
                    "mark_px": 65335.5,
                    "observed_end_ms": 1_300,
                    "fills": [
                        {
                            "fillId": "window-1-fill",
                            "coin": "BTC",
                            "oid": 101,
                            "side": "B",
                            "sz": "0.005",
                            "px": "65335",
                            "fee": "0.01",
                            "time": 1_200,
                            "crossed": False,
                        }
                    ],
                },
                {
                    "phase": "finalize",
                    "mark_px": 65336.5,
                    "observed_end_ms": 1_600,
                    "fills": [
                        {
                            "fillId": "window-1-fill",
                            "coin": "BTC",
                            "oid": 101,
                            "side": "B",
                            "sz": "0.005",
                            "px": "65335",
                            "fee": "0.01",
                            "time": 1_200,
                            "crossed": False,
                        }
                    ],
                },
            ],
        },
        "2": {
            "window_id": 2,
            "attempts": [
                {
                    "attempt_id": 1,
                    "submit_start_ms": 1_000,
                    "submit_end_ms": 1_100,
                    "terminal_end_ms": 1_300,
                    "cloid": "w2-a1",
                },
                {
                    "attempt_id": 2,
                    "submit_start_ms": 1_200,
                    "submit_end_ms": 1_250,
                    "terminal_end_ms": 1_600,
                    "cloid": "w2-a2",
                },
            ],
            "pullbacks": [
                {
                    "phase": "after_response",
                    "mark_px": 65335.5,
                    "observed_end_ms": 1_700,
                    "fills": [
                        {
                            "fillId": "window-2-unique",
                            "coin": "BTC",
                            "side": "B",
                            "sz": "0.002",
                            "px": "65335",
                            "fee": "0.004",
                            "time": 1_500,
                        },
                        {
                            "coin": "BTC",
                            "side": "B",
                            "sz": "0.001",
                            "px": "65335",
                            "fee": "0.002",
                            "time": 1_300,
                        },
                    ],
                },
                {
                    "phase": "finalize",
                    "mark_px": 65336.0,
                    "observed_end_ms": 1_800,
                    "fills": [
                        {
                            "fillId": "window-2-unique",
                            "coin": "BTC",
                            "side": "B",
                            "sz": "0.002",
                            "px": "65335",
                            "fee": "0.004",
                            "time": 1_500,
                        },
                        {
                            "coin": "BTC",
                            "side": "B",
                            "sz": "0.001",
                            "px": "65335",
                            "fee": "0.002",
                            "time": 1_300,
                        },
                    ],
                },
            ],
        },
    }
    fixture_json = json.dumps(fixtures, sort_keys=True)
    path.write_text(
        "\n".join(
            [
                "import argparse, json, os, time",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--event-driven-edge-gate-live', action='store_true')",
                "parser.add_argument('--watcher-seconds')",
                "parser.add_argument('--max-order-size')",
                "parser.add_argument('--max-real-order-submissions')",
                "parser.add_argument('--quote-hold-seconds')",
                "parser.add_argument('--wait-seconds')",
                "parser.add_argument('--env-file')",
                "parser.add_argument('--artifact-task-id')",
                "parser.add_argument('--artifact-window-id', type=int)",
                "parser.add_argument('--output-dir')",
                "parser.add_argument('--hyperliquid-l2book-fast', action='store_true')",
                "args = parser.parse_args()",
                "out = Path(args.output_dir)",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out / 'integrated_watcher_pid.txt').write_text(str(os.getpid()))",
                f"fixtures = json.loads({fixture_json!r})",
                "window = args.artifact_window_id",
                "if window == 3:",
                "    time.sleep(30)",
                "else:",
                "    (out / 'offline_fill_fixture.json').write_text(json.dumps(fixtures[str(window)], sort_keys=True) + '\\n')",
                "    (out / 'integrated_watcher_manifest.json').write_text(json.dumps({'window': window, 'task_id': args.artifact_task_id}, sort_keys=True) + '\\n')",
                "",
            ]
        ),
        encoding="utf-8",
    )


def replay_fill_fixture(fixture: dict) -> fill_window.LiveFillLedger:
    ledger = fill_window.LiveFillLedger(
        task_id="0717T011",
        window_id=int(fixture["window_id"]),
        pullback_grace_ms=100,
    )
    for attempt in fixture["attempts"]:
        intent = executor.OrderIntent(
            symbol="BTC",
            is_buy=True,
            size_btc=0.005,
            limit_px=65335.0,
            cloid=attempt["cloid"],
        )
        refs = [{"oid": attempt["oid"]}] if "oid" in attempt else []
        ledger.register_attempt(
            attempt_id=int(attempt["attempt_id"]),
            intent=intent,
            submit_start_ms=int(attempt["submit_start_ms"]),
            submit_end_ms=int(attempt["submit_end_ms"]),
            tracked_refs=refs,
            terminal_end_ms=int(attempt["terminal_end_ms"]),
        )
    for pullback in fixture["pullbacks"]:
        ledger.ingest(
            fills=list(pullback["fills"]),
            mark_px=float(pullback["mark_px"]),
            user_add_rate=0.0,
            pullback_phase=str(pullback["phase"]),
            observed_end_ms=int(pullback["observed_end_ms"]),
        )
    return ledger


def test_three_window_repair_contracts_work_together_offline(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "integrated_fake_watcher.py"
    write_integrated_fake_watcher(fake_watcher)
    command = orchestrator_command(
        tmp_path,
        fake_watcher,
        windows=4,
        extra_args=[
            "--window-seconds",
            "0.05",
            "--window-timeout-grace-seconds",
            "0.05",
            "--termination-grace-seconds",
            "0.05",
        ],
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2, result.stderr
    run_root = tmp_path / "run"
    status = read_json(run_root / "run_status.json")
    abort = read_json(run_root / "abort_manifest.json")
    assert status["state"] == "failed"
    assert status["phase"] == "failed"
    assert abort["root_open_orders_proof"]["final_open_orders_empty"] is True

    window_01 = run_root / "window_01"
    window_02 = run_root / "window_02"
    window_03 = run_root / "window_03"
    assert window_01.exists()
    assert window_02.exists()
    assert window_03.exists()
    assert not (run_root / "window_04").exists()

    window_03_status = read_json(window_03 / "window_status.json")
    assert window_03_status["termination_reason"] == "watcher_timeout"
    assert window_03_status["child_reaped"] is True
    assert window_03_status["open_orders_proof_after_child_exit"] is True
    assert_pid_gone(int((window_03 / "integrated_watcher_pid.txt").read_text(encoding="utf-8")))

    ledger_01 = replay_fill_fixture(read_json(window_01 / "offline_fill_fixture.json"))
    ledger_02 = replay_fill_fixture(read_json(window_02 / "offline_fill_fixture.json"))
    attributed_01 = ledger_01.attributed_rows()
    attributed_02 = ledger_02.attributed_rows()
    evidence_02 = ledger_02.evidence_rows()

    assert len(ledger_01.evidence_rows()) == 1
    assert len(attributed_01) == 1
    assert attributed_01[0]["qty_btc"] == 0.005
    assert attributed_01[0]["duplicate_pullback_count"] == 1
    assert attributed_01[0]["attempt_key"] == "0717T011:window_01:attempt_1"

    assert len(evidence_02) == 2
    assert len(attributed_02) == 1
    assert attributed_02[0]["qty_btc"] == 0.002
    assert attributed_02[0]["attempt_key"] == "0717T011:window_02:attempt_2"
    ambiguous = [row for row in evidence_02 if row["attribution_status"] == "ambiguous_unattributed_fill"]
    assert len(ambiguous) == 1
    assert ambiguous[0]["ambiguity_reason"] == "multiple_candidate_attempts"
    assert {attributed_01[0]["attempt_key"], attributed_02[0]["attempt_key"]} == {
        "0717T011:window_01:attempt_1",
        "0717T011:window_02:attempt_2",
    }

    verification = read_json(run_root / "remote_sha256_verification.json")
    assert verification["status"] == "pass"
    assert verification["missing_count"] == 0
    assert verification["mismatch_count"] == 0
    checksum = subprocess.run(
        ["sha256sum", "-c", "remote_sha256_manifest.txt"],
        cwd=run_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert checksum.returncode == 0, checksum.stdout + checksum.stderr
