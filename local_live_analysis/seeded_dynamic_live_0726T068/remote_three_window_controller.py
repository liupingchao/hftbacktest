#!/usr/bin/env python3
"""Detached SSM-first controller for T068's three exact live windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_ID = "0726T068"
SOURCE_COMMIT = "a0bc92898ecea43cbdc4219efc1acbcd69580969"
EARLIEST_START = "2026-07-26T09:00:00Z"
SEED_SHA256 = "e35c7fd8f3ec8268e5d50c7963889b73409470c9f01f92ca3a0d598a96562be9"
AUTHORIZED_IDENTITY_SHA256 = "6087b1e43001d770c534cf5e37d3c86fbe11937fc08a02a7fe498d10610b05f9"
EXPECTED_ACCOUNT_SCOPE_SHA256 = "59153858d04cb15ef51660a62b2598b785468306e4d8ab36bdbbfa18ac134c6d"
EXPECTED_SIGNER_SHA256 = "9dd9fcfdb4e3b3e077ba25b41ccc3514d824b952a570fd5a98c3188ec72de422"
PYTHON = "/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python"
ENV_FILE = "/home/admin/XEMM_rust_latest/.env"
LOCK_FILE = "/tmp/hftbacktest_live_test.lock"


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.repo = Path(args.repo).resolve()
        self.control_root = Path(args.control_root).resolve()
        self.guard_script = Path(args.guard_script).resolve()
        self.identity_file = Path(args.identity_file).resolve()
        self.status_path = self.control_root / "controller_status.json"
        self.heartbeat_path = self.control_root / "heartbeat.json"
        self.abort_path = self.control_root / "abort_manifest.json"
        self.final_path = self.control_root / "controller_final.json"
        self.current_child: subprocess.Popen[str] | None = None
        self.stop_requested = False
        self.stop_signal = ""
        self.phase = "created"
        self.current_window = 0
        self.completed_windows: list[int] = []
        identity_bytes = self.identity_file.read_bytes()
        if hashlib.sha256(identity_bytes).hexdigest() != AUTHORIZED_IDENTITY_SHA256:
            raise RuntimeError("authorized_identity_file_sha256_mismatch")
        self.identity = json.loads(identity_bytes)
        if (
            self.identity.get("status") != "pass"
            or self.identity.get("source_commit") != SOURCE_COMMIT
            or self.identity.get("account_scope_sha256")
            != EXPECTED_ACCOUNT_SCOPE_SHA256
            or self.identity.get("signer_sha256") != EXPECTED_SIGNER_SHA256
        ):
            raise RuntimeError("authorized_identity_contract_mismatch")
        self.heartbeat_stop = threading.Event()

    def write_status(self, state: str, **extra: Any) -> None:
        write_json_atomic(
            self.status_path,
            {
                "schema_version": "t068_three_window_controller_status_v1",
                "task_id": TASK_ID,
                "state": state,
                "phase": self.phase,
                "current_window": self.current_window,
                "completed_windows": self.completed_windows,
                "updated_at_utc": utc_now(),
                "source_commit": SOURCE_COMMIT,
                **extra,
            },
        )

    def heartbeat_loop(self) -> None:
        while not self.heartbeat_stop.wait(15.0):
            write_json_atomic(
                self.heartbeat_path,
                {
                    "task_id": TASK_ID,
                    "updated_at_utc": utc_now(),
                    "phase": self.phase,
                    "current_window": self.current_window,
                    "completed_windows": self.completed_windows,
                    "controller_pid": os.getpid(),
                    "child_pid": (
                        self.current_child.pid
                        if self.current_child is not None
                        else None
                    ),
                },
            )

    def request_stop(self, signum: int, _frame: Any) -> None:
        self.stop_requested = True
        self.stop_signal = signal.Signals(signum).name
        if self.current_child is not None and self.current_child.poll() is None:
            try:
                os.killpg(os.getpgid(self.current_child.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    def check_command(self, command: list[str], reason: str) -> None:
        result = subprocess.run(
            command,
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"{reason}:{detail}")

    def require_xemm_inactive(self) -> None:
        if utc_now() < EARLIEST_START:
            raise RuntimeError("controller_started_before_authorized_time")
        if (
            (self.repo / "source_commit.txt")
            .read_text(encoding="utf-8")
            .strip()
            != SOURCE_COMMIT
        ):
            raise RuntimeError("source_commit_mismatch")
        result = subprocess.run(
            ["/bin/systemctl", "is-active", "xemm.service"],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.stdout.strip() != "inactive":
            raise RuntimeError(
                f"xemm_service_not_inactive:{result.stdout.strip()}"
            )
        self.check_command(
            ["/usr/bin/flock", "-n", LOCK_FILE, "-c", "true"],
            "live_lock_held",
        )

    def account_guard(
        self,
        *,
        phase: str,
        output: Path,
        max_abs_position: float,
    ) -> dict[str, Any]:
        command = [
            PYTHON,
            str(self.guard_script),
            "--repo",
            str(self.repo),
            "--env-file",
            ENV_FILE,
            "--phase",
            phase,
            "--output",
            str(output),
            "--expected-source-commit",
            SOURCE_COMMIT,
            "--expected-account-scope-sha256",
            EXPECTED_ACCOUNT_SCOPE_SHA256,
            "--expected-signer-sha256",
            EXPECTED_SIGNER_SHA256,
            "--max-abs-btc-position",
            str(max_abs_position),
            "--require-open-orders-empty",
        ]
        result = subprocess.run(
            command,
            cwd=self.repo,
            text=True,
            capture_output=True,
            check=False,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        if result.returncode != 0 or payload.get("status") != "pass":
            raise RuntimeError(
                f"account_guard_failed:{phase}:{payload.get('failure_reason')}"
            )
        return payload

    def orchestrator_command(self, window_id: int) -> list[str]:
        offset = window_id - 1
        run_root = (
            f"/home/admin/hftbacktest-cross-exchange-artifacts/"
            f"0726T068_window_{window_id:02d}_R1"
        )
        return [
            PYTHON,
            "examples/hyperliquid/cross_exchange_live_remote_orchestrator.py",
            "--task-id",
            TASK_ID,
            "--remote-repo",
            str(self.repo),
            "--python",
            PYTHON,
            "--env-file",
            ENV_FILE,
            "--run-root",
            run_root,
            "--mode",
            "event-driven-edge-gate-live",
            "--exact-envelope-profile",
            "two-sided-seeded-dynamic-manager",
            "--windows",
            "1",
            "--window-id-offset",
            str(offset),
            "--window-seconds",
            "1800",
            "--max-order-size",
            "0.005",
            "--max-loss-usdc",
            "1",
            "--max-position-btc",
            "0.01",
            "--max-submissions",
            "2",
            "--requote-attempts",
            "2",
            "--quote-hold-seconds",
            "3",
            "--wait-seconds",
            "10",
            "--exchange-reconciled-manager",
            "--enable-dynamic-spread",
            "--dynamic-spread-seed-contract",
            str(
                self.repo
                / "local_live_analysis/public_multi_distance_dynamic_seed_0722T066/dynamic_spread_seed_contract.json"
            ),
            "--dynamic-spread-seed-exposures",
            str(
                self.repo
                / "local_live_analysis/public_multi_distance_dynamic_seed_0722T066/quote_exposure_intervals.csv"
            ),
            "--expected-dynamic-spread-seed-sha256",
            SEED_SHA256,
            "--require-strict-seeded-dynamic-submit",
            "--hyperliquid-l2book-fast",
            "--private-proof-mode",
            "live_open_orders",
            "--require-exact-envelope",
        ]

    def run_window(self, window_id: int) -> None:
        self.current_window = window_id
        self.phase = f"window_{window_id:02d}_pre_guard"
        self.write_status("running")
        self.require_xemm_inactive()
        self.account_guard(
            phase=self.phase,
            output=self.control_root
            / f"window_{window_id:02d}_pre_account_guard.json",
            max_abs_position=0.01,
        )
        run_root = Path(
            f"/home/admin/hftbacktest-cross-exchange-artifacts/"
            f"0726T068_window_{window_id:02d}_R1"
        )
        if run_root.exists():
            raise RuntimeError(f"run_root_already_exists:{run_root}")

        self.phase = f"window_{window_id:02d}_running"
        self.write_status("running")
        stdout_path = self.control_root / f"window_{window_id:02d}_stdout.log"
        stderr_path = self.control_root / f"window_{window_id:02d}_stderr.log"
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            self.current_child = subprocess.Popen(
                self.orchestrator_command(window_id),
                cwd=self.repo,
                stdout=stdout,
                stderr=stderr,
                text=True,
                start_new_session=True,
            )
            returncode = self.current_child.wait()
        self.current_child = None

        self.phase = f"window_{window_id:02d}_post_guard"
        post_guard = self.account_guard(
            phase=self.phase,
            output=self.control_root
            / f"window_{window_id:02d}_post_account_guard.json",
            max_abs_position=0.01,
        )
        if returncode != 0:
            raise RuntimeError(
                f"orchestrator_failed:window_{window_id:02d}:rc={returncode}"
            )
        if post_guard.get("open_orders_empty") is not True:
            raise RuntimeError(
                f"post_open_orders_not_empty:window_{window_id:02d}"
            )
        self.completed_windows.append(window_id)
        self.write_status("running")

    def run(self) -> int:
        self.control_root.mkdir(parents=True, exist_ok=True)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, self.request_stop)
        heartbeat = threading.Thread(target=self.heartbeat_loop, daemon=True)
        heartbeat.start()
        try:
            self.phase = "prestart_gate"
            self.write_status("running")
            self.require_xemm_inactive()
            self.account_guard(
                phase="controller_prestart",
                output=self.control_root / "controller_prestart_account_guard.json",
                max_abs_position=0.0,
            )
            for window_id in (1, 2, 3):
                if self.stop_requested:
                    raise RuntimeError(
                        f"controller_stop_requested:{self.stop_signal}"
                    )
                self.run_window(window_id)
            self.phase = "final_account_guard"
            final_guard = self.account_guard(
                phase=self.phase,
                output=self.control_root / "controller_final_account_guard.json",
                max_abs_position=0.01,
            )
            final_payload = {
                "schema_version": "t068_three_window_controller_final_v1",
                "task_id": TASK_ID,
                "state": "complete",
                "completed_at_utc": utc_now(),
                "source_commit": SOURCE_COMMIT,
                "completed_windows": self.completed_windows,
                "account_scope_sha256": final_guard["account_scope_sha256"],
                "signer_sha256": final_guard["signer_sha256"],
                "final_open_orders_empty": final_guard["open_orders_empty"],
                "final_btc_position": final_guard["btc_position"],
                "raw_credentials_written": False,
            }
            write_json_atomic(self.final_path, final_payload)
            self.write_status("complete")
            return 0
        except Exception as exc:
            write_json_atomic(
                self.abort_path,
                {
                    "schema_version": "t068_three_window_controller_abort_v1",
                    "task_id": TASK_ID,
                    "state": "failed",
                    "failed_at_utc": utc_now(),
                    "phase": self.phase,
                    "current_window": self.current_window,
                    "completed_windows": self.completed_windows,
                    "error": str(exc),
                    "stop_signal": self.stop_signal,
                    "raw_credentials_written": False,
                },
            )
            self.write_status("failed", error=str(exc))
            return 2
        finally:
            self.heartbeat_stop.set()
            heartbeat.join(timeout=2.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--guard-script", required=True)
    parser.add_argument("--identity-file", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(Controller(build_parser().parse_args()).run())
