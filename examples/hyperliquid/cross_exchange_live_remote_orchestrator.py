#!/usr/bin/env python3
"""SSM-friendly remote live evidence orchestrator.

This runner is designed to execute on awsserver1.  It intentionally keeps the
live strategy process detached from the caller's SSH session: a caller can
start it through SSM RunCommand, SSH, tmux, or systemd, then recover status and
artifacts later from the output root.

The script does not change strategy behavior.  It wraps an existing watcher
command with a live lock, heartbeat/status files, per-window open-orders proof,
abort manifests, and a final sha256 manifest.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


DEFAULT_REMOTE_REPO = "/home/admin/hftbacktest-cross-exchange"
DEFAULT_REMOTE_PYTHON = "/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python"
DEFAULT_ENV_FILE = "/home/admin/XEMM_rust_latest/.env"
DEFAULT_OUTPUT_ROOT = "/home/admin/hftbacktest-cross-exchange-artifacts"
DEFAULT_WATCHER_SCRIPT = "examples/hyperliquid/hyperliquid_tiny_live_m2_public_watcher.py"
DEFAULT_MODE = "event-driven-edge-gate-live"
DEFAULT_LOCK_FILE = "/tmp/hftbacktest_live_test.lock"
POST_ONLY_TIF = "Alo"


class RemoteOrchestratorError(RuntimeError):
    """Raised when the remote orchestration must fail closed."""


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(executor.redact(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(executor.redact(payload), sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_sha256_manifest(root: Path) -> Path:
    manifest = root / "remote_sha256_manifest.txt"
    rows: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if path == manifest:
            continue
        rows.append(f"{sha256_file(path)}  {path}\n")
    manifest.write_text("".join(rows), encoding="utf-8")
    return manifest


def mode_flag(mode: str) -> str:
    normalized = mode.strip().replace("_", "-")
    allowed = {
        "event-driven-edge-gate-live",
        "event-driven-anti-drift-live",
        "event-driven-inline-reprice-live",
        "event-driven-live",
        "same-process-live",
    }
    if normalized not in allowed:
        raise RemoteOrchestratorError(f"unsupported_live_mode:{mode}")
    return f"--{normalized}"


def load_existing_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


class LiveLock:
    def __init__(self, path: Path, *, task_id: str, output_root: Path) -> None:
        self.path = path
        self.task_id = task_id
        self.output_root = output_root
        self._fh: Any | None = None

    def __enter__(self) -> "LiveLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RemoteOrchestratorError(f"live_lock_already_held:{self.path}") from exc
        payload = {
            "task_id": self.task_id,
            "pid": os.getpid(),
            "acquired_at_utc": utc_now(),
            "output_root": str(self.output_root),
        }
        self._fh.write(json.dumps(payload, sort_keys=True) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._fh is None:
            return
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        finally:
            self._fh.close()
            self._fh = None


class RemoteLiveOrchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.task_id = args.task_id
        self.remote_repo = Path(args.remote_repo).resolve()
        self.output_root = Path(args.output_root).resolve()
        self.run_root = Path(args.run_root).resolve() if args.run_root else self.output_root / f"{self.task_id}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
        self.lock_file = Path(args.lock_file)
        self.status_path = self.run_root / "run_status.json"
        self.heartbeat_path = self.run_root / "heartbeat.json"
        self.event_log_path = self.run_root / "orchestrator_events.jsonl"
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._abort_requested = False
        self._signal_name = ""
        self._current_window = ""
        self._completed_windows: list[str] = []

    def status_payload(self, *, state: str, phase: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "state": state,
            "phase": phase,
            "pid": os.getpid(),
            "updated_at_utc": utc_now(),
            "run_root": str(self.run_root),
            "remote_repo": str(self.remote_repo),
            "current_window": self._current_window,
            "completed_windows": list(self._completed_windows),
            "abort_requested": self._abort_requested,
            "signal_name": self._signal_name,
        }
        existing = load_existing_json(self.status_path)
        if "started_at_utc" in existing:
            payload["started_at_utc"] = existing["started_at_utc"]
        else:
            payload["started_at_utc"] = utc_now()
        if extra:
            payload.update(extra)
        return payload

    def write_status(self, *, state: str, phase: str, extra: dict[str, Any] | None = None) -> None:
        payload = self.status_payload(state=state, phase=phase, extra=extra)
        write_json(self.status_path, payload)
        append_jsonl(self.event_log_path, payload)

    def heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(max(1.0, float(self.args.heartbeat_interval_seconds))):
            write_json(
                self.heartbeat_path,
                {
                    "task_id": self.task_id,
                    "pid": os.getpid(),
                    "heartbeat_at_utc": utc_now(),
                    "run_root": str(self.run_root),
                    "current_window": self._current_window,
                    "completed_windows": list(self._completed_windows),
                    "abort_requested": self._abort_requested,
                    "signal_name": self._signal_name,
                },
            )

    def start_heartbeat(self) -> None:
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(target=self.heartbeat_loop, name="live-orchestrator-heartbeat", daemon=True)
        self._heartbeat_thread.start()
        write_json(
            self.heartbeat_path,
            {
                "task_id": self.task_id,
                "pid": os.getpid(),
                "heartbeat_at_utc": utc_now(),
                "run_root": str(self.run_root),
                "current_window": self._current_window,
                "completed_windows": list(self._completed_windows),
                "abort_requested": False,
                "signal_name": "",
            },
        )

    def stop_heartbeat(self) -> None:
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5)

    def request_abort(self, signum: int, _frame: Any) -> None:
        self._abort_requested = True
        self._signal_name = signal.Signals(signum).name
        self.write_status(state="aborting", phase="signal_received", extra={"signal_number": signum})

    def watcher_command(self, window_dir: Path) -> list[str]:
        command = [
            self.args.python,
            self.args.watcher_script,
            mode_flag(self.args.mode),
            "--watcher-seconds",
            str(self.args.window_seconds),
            "--max-order-size",
            str(self.args.max_order_size),
            "--max-real-order-submissions",
            str(self.args.max_submissions),
            "--quote-hold-seconds",
            str(self.args.quote_hold_seconds),
            "--wait-seconds",
            str(self.args.wait_seconds),
            "--env-file",
            self.args.env_file,
            "--artifact-task-id",
            self.task_id,
            "--output-dir",
            str(window_dir),
        ]
        if self.args.hyperliquid_l2book_fast:
            command.append("--hyperliquid-l2book-fast")
        return command

    def write_open_orders_proof(self, output: Path, *, window: str, phase: str) -> dict[str, Any]:
        if self.args.private_proof_mode == "skipped_for_test":
            payload = {
                "task_id": self.task_id,
                "window": window,
                "phase": phase,
                "private_read_only": True,
                "proof_mode": "skipped_for_test",
                "final_open_orders": [],
                "final_open_orders_count": 0,
                "final_open_orders_empty": True,
                "order_endpoint_called": False,
                "cancel_endpoint_called": False,
                "credentials_written": False,
                "secret_values_written": False,
                "raw_signatures_written": False,
            }
            write_json(output, payload)
            return payload

        executor.load_env_file(Path(self.args.env_file))
        client = executor.build_live_client_from_env()
        orders = client.open_orders()
        payload = {
            "task_id": self.task_id,
            "window": window,
            "phase": phase,
            "checked_at_utc": utc_now(),
            "final_open_orders": executor.redact(orders),
            "final_open_orders_count": len(orders),
            "final_open_orders_empty": len(orders) == 0,
            "private_read_only": True,
            "proof_mode": "live_open_orders",
            "order_endpoint_called": False,
            "cancel_endpoint_called": False,
            "credentials_written": False,
            "secret_values_written": False,
            "raw_signatures_written": False,
        }
        write_json(output, payload)
        return payload

    def run_window(self, index: int) -> dict[str, Any]:
        window = f"{index:02d}"
        self._current_window = window
        window_dir = self.run_root / f"window_{window}"
        window_dir.mkdir(parents=True, exist_ok=True)
        status_file = window_dir / "window_status.json"
        started = utc_now()
        write_json(
            status_file,
            {
                "task_id": self.task_id,
                "window": window,
                "state": "running",
                "started_at_utc": started,
                "window_dir": str(window_dir),
            },
        )
        self.write_status(state="running", phase="window_running", extra={"window": window})
        command = self.watcher_command(window_dir)
        (window_dir / "runner_command.json").write_text(
            json.dumps({"command": command, "redacted": True}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stdout_path = window_dir / "runner_stdout.log"
        stderr_path = window_dir / "runner_stderr.log"
        rc = 0
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=self.remote_repo, stdout=stdout, stderr=stderr, check=False, text=True)
            rc = completed.returncode
        proof_payload: dict[str, Any] = {}
        proof_error = ""
        try:
            proof_payload = self.write_open_orders_proof(window_dir / "independent_remote_open_orders_check.json", window=window, phase="after_window")
        except Exception as exc:
            proof_error = executor._redacted_error(exc)
            write_json(
                window_dir / "independent_remote_open_orders_check.json",
                {
                    "task_id": self.task_id,
                    "window": window,
                    "phase": "after_window",
                    "private_read_only": True,
                    "proof_status": "error",
                    "error": proof_error,
                    "order_endpoint_called": False,
                    "cancel_endpoint_called": False,
                    "credentials_written": False,
                    "secret_values_written": False,
                    "raw_signatures_written": False,
                },
            )
        ended = utc_now()
        state = "complete" if rc == 0 and proof_payload.get("final_open_orders_empty") is True else "failed"
        payload = {
            "task_id": self.task_id,
            "window": window,
            "state": state,
            "started_at_utc": started,
            "ended_at_utc": ended,
            "runner_returncode": rc,
            "window_dir": str(window_dir),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "independent_open_orders_count": proof_payload.get("final_open_orders_count", ""),
            "independent_open_orders_empty": proof_payload.get("final_open_orders_empty", False),
            "independent_open_orders_error": proof_error,
        }
        write_json(status_file, payload)
        if state == "complete":
            self._completed_windows.append(window)
        return payload

    def run(self) -> int:
        self.run_root.mkdir(parents=True, exist_ok=True)
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.request_abort)

        with LiveLock(self.lock_file, task_id=self.task_id, output_root=self.run_root):
            self.start_heartbeat()
            self.write_status(
                state="running",
                phase="started",
                extra={
                    "host": os.uname().nodename,
                    "windows_requested": self.args.windows,
                    "window_seconds": self.args.window_seconds,
                    "mode": self.args.mode,
                    "post_only": POST_ONLY_TIF,
                    "max_order_size": self.args.max_order_size,
                    "max_submissions": self.args.max_submissions,
                    "private_proof_mode": self.args.private_proof_mode,
                },
            )
            window_results: list[dict[str, Any]] = []
            try:
                for index in range(1, int(self.args.windows) + 1):
                    if self._abort_requested:
                        raise RemoteOrchestratorError(f"abort_requested:{self._signal_name}")
                    result = self.run_window(index)
                    window_results.append(result)
                    if result.get("state") != "complete":
                        raise RemoteOrchestratorError(f"window_failed:{result.get('window')}:rc={result.get('runner_returncode')}")
                completed = {
                    "task_id": self.task_id,
                    "state": "complete",
                    "completed_at_utc": utc_now(),
                    "run_root": str(self.run_root),
                    "windows_completed": list(self._completed_windows),
                    "window_results": window_results,
                    "remote_sha256_manifest": str(self.run_root / "remote_sha256_manifest.txt"),
                    "post_only": POST_ONLY_TIF,
                    "order_endpoint_called_by_orchestrator": False,
                    "cancel_endpoint_called_by_orchestrator": False,
                    "private_proof_mode": self.args.private_proof_mode,
                }
                write_json(self.run_root / "run_complete.json", completed)
                self.write_status(state="complete", phase="complete", extra={"windows_completed": list(self._completed_windows)})
                manifest = write_sha256_manifest(self.run_root)
                completed["remote_sha256_manifest"] = str(manifest)
                write_json(self.run_root / "run_complete.json", completed)
                write_sha256_manifest(self.run_root)
                return 0
            except Exception as exc:
                abort = {
                    "task_id": self.task_id,
                    "state": "failed",
                    "failed_at_utc": utc_now(),
                    "run_root": str(self.run_root),
                    "current_window": self._current_window,
                    "completed_windows": list(self._completed_windows),
                    "error": executor._redacted_error(exc),
                    "traceback_redacted": executor.redact(traceback.format_exc()),
                    "abort_requested": self._abort_requested,
                    "signal_name": self._signal_name,
                    "window_results": window_results,
                }
                try:
                    abort["root_open_orders_proof"] = self.write_open_orders_proof(
                        self.run_root / "independent_abort_open_orders_check.json",
                        window=self._current_window or "root",
                        phase="abort",
                    )
                except Exception as proof_exc:
                    abort["root_open_orders_error"] = executor._redacted_error(proof_exc)
                write_json(self.run_root / "abort_manifest.json", abort)
                write_sha256_manifest(self.run_root)
                self.write_status(state="failed", phase="failed", extra={"error": abort["error"]})
                return 2
            finally:
                self.stop_heartbeat()
                self._current_window = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-root", default="")
    parser.add_argument("--watcher-script", default=DEFAULT_WATCHER_SCRIPT)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--window-seconds", type=float, default=1800.0)
    parser.add_argument("--max-order-size", type=float, default=0.005)
    parser.add_argument("--max-submissions", type=int, default=2)
    parser.add_argument("--quote-hold-seconds", type=int, default=3)
    parser.add_argument("--wait-seconds", type=int, default=10)
    parser.add_argument("--hyperliquid-l2book-fast", action="store_true")
    parser.add_argument("--lock-file", default=DEFAULT_LOCK_FILE)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=15.0)
    parser.add_argument(
        "--private-proof-mode",
        choices=("live_open_orders", "skipped_for_test"),
        default="live_open_orders",
        help="Use skipped_for_test only in offline tests. Live runs must keep the default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.windows <= 0:
        raise RemoteOrchestratorError("windows_must_be_positive")
    if args.max_submissions <= 0:
        raise RemoteOrchestratorError("max_submissions_must_be_positive")
    if args.max_order_size <= 0:
        raise RemoteOrchestratorError("max_order_size_must_be_positive")
    orchestrator = RemoteLiveOrchestrator(args)
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())
