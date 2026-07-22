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
import re
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
DEFAULT_CHILD_POLL_SECONDS = 0.25
DEFAULT_TERMINATION_GRACE_SECONDS = 10.0
DEFAULT_WINDOW_TIMEOUT_GRACE_SECONDS = 60.0
SHA256_MANIFEST_NAME = "remote_sha256_manifest.txt"
SHA256_VERIFICATION_NAME = "remote_sha256_verification.json"
EXACT_ENVELOPE_MAX_ORDER_SIZE_BTC = 0.005
EXACT_ENVELOPE_MAX_LOSS_USDC = 1.0
EXACT_ENVELOPE_MAX_POSITION_BTC = 0.01
EXACT_ENVELOPE_MAX_SUBMISSIONS = 2
EXACT_ENVELOPE_MAX_WINDOW_SECONDS = 1800.0
RUNTIME_SOURCE_PROVENANCE_NAME = "runtime_source_provenance.json"
RUNTIME_SOURCE_START_VERIFICATION_NAME = "runtime_source_start_verification.json"
RUNTIME_SOURCE_POSTRUN_VERIFICATION_NAME = "runtime_source_postrun_verification.json"
RUNTIME_SOURCE_SCHEMA_VERSION = "cross_exchange_runtime_source_provenance_v2"
EXACT_PROFILE_LEGACY_SINGLE_ORDER = "legacy-single-order"
EXACT_PROFILE_TWO_SIDED_MANAGER = "two-sided-manager"
EXACT_PROFILE_TWO_SIDED_DYNAMIC_MANAGER = "two-sided-dynamic-manager"
EXACT_PROFILE_TWO_SIDED_FILL_FEEDBACK_MANAGER = "two-sided-fill-feedback-manager"
EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY = (
    "delayed-history-observe-only"
)
DELAYED_HISTORY_OBSERVE_ONLY_MODE = (
    "delayed-history-observe-only-probe"
)
DELAYED_HISTORY_OBSERVE_ONLY_WINDOW_SECONDS = 30.0
DELAYED_HISTORY_OBSERVE_ONLY_DIRECT_ROUNDS = 5
DELAYED_HISTORY_OBSERVE_ONLY_HISTORY_CALLS = 1
DELAYED_HISTORY_OBSERVE_ONLY_PROPAGATION_DELAY_SECONDS = 4.0
DELAYED_HISTORY_OBSERVE_ONLY_FINAL_RESERVE_SECONDS = 0.5
DELAYED_HISTORY_OBSERVE_ONLY_TOTAL_BUDGET_SECONDS = 5.0


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
    root = root.resolve()
    manifest = root / SHA256_MANIFEST_NAME
    rows: list[str] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {SHA256_MANIFEST_NAME, SHA256_VERIFICATION_NAME}:
            continue
        rows.append(f"{sha256_file(path)}  {relative}\n")
    manifest.write_text("".join(rows), encoding="utf-8")
    return manifest


def verify_sha256_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = root / SHA256_MANIFEST_NAME
    verification_path = root / SHA256_VERIFICATION_NAME
    manifest_entry_count = 0
    verified_count = 0
    missing_count = 0
    mismatch_count = 0
    if not manifest.exists():
        missing_count = 1
    else:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            manifest_entry_count += 1
            parts = line.split("  ", 1)
            if len(parts) != 2:
                mismatch_count += 1
                continue
            expected_digest, relative_name = parts
            relative_path = Path(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                mismatch_count += 1
                continue
            candidate = (root / relative_path).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                mismatch_count += 1
                continue
            if not candidate.is_file():
                missing_count += 1
                continue
            if sha256_file(candidate) != expected_digest:
                mismatch_count += 1
                continue
            verified_count += 1
    summary = {
        "manifest_entry_count": manifest_entry_count,
        "verified_count": verified_count,
        "missing_count": missing_count,
        "mismatch_count": mismatch_count,
        "status": "pass" if missing_count == 0 and mismatch_count == 0 else "fail",
    }
    write_json(verification_path, summary)
    return summary


def mode_flag(mode: str) -> str:
    normalized = mode.strip().replace("_", "-")
    allowed = {
        "event-driven-edge-gate-live",
        "event-driven-anti-drift-live",
        "event-driven-inline-reprice-live",
        "event-driven-live",
        "same-process-live",
        DELAYED_HISTORY_OBSERVE_ONLY_MODE,
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


def source_commit_marker(remote_repo: Path) -> str:
    marker = remote_repo / "source_commit.txt"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=remote_repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return ""


def valid_full_commit(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", value.strip()))


def runtime_source_files(remote_repo: Path) -> list[Path]:
    source_root = remote_repo / "examples" / "hyperliquid"
    if not source_root.is_dir():
        raise RemoteOrchestratorError("runtime_source_root_missing:examples/hyperliquid")
    candidates = [
        path
        for path in source_root.rglob("*.py")
        if path.is_file() and not path.name.startswith("test_")
    ]
    if not candidates:
        raise RemoteOrchestratorError("runtime_source_scope_empty")
    return sorted(set(candidates))


def resolve_watcher_script(remote_repo: Path, watcher_script: str) -> Path:
    candidate = Path(watcher_script)
    if not candidate.is_absolute():
        candidate = remote_repo / candidate
    return candidate.resolve()


def validate_args(args: argparse.Namespace) -> None:
    delayed_history_probe_profile = (
        args.require_exact_envelope
        and args.exact_envelope_profile
        == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
    )
    if args.windows <= 0:
        raise RemoteOrchestratorError("windows_must_be_positive")
    if (
        args.max_submissions < 0
        or (
            args.max_submissions == 0
            and not delayed_history_probe_profile
        )
    ):
        raise RemoteOrchestratorError("max_submissions_must_be_positive")
    if args.requote_attempts <= 0:
        raise RemoteOrchestratorError("requote_attempts_must_be_positive")
    if (
        args.max_order_size < 0
        or (
            args.max_order_size == 0
            and not delayed_history_probe_profile
        )
    ):
        raise RemoteOrchestratorError("max_order_size_must_be_positive")
    if args.max_loss_usdc <= 0:
        raise RemoteOrchestratorError("max_loss_usdc_must_be_positive")
    if args.max_position_btc <= 0:
        raise RemoteOrchestratorError("max_position_btc_must_be_positive")
    if args.fill_feedback_target_ratio is not None and (
        not isinstance(args.fill_feedback_target_ratio, (int, float))
        or not float(args.fill_feedback_target_ratio) == float(args.fill_feedback_target_ratio)
        or args.fill_feedback_target_ratio < 0
        or args.fill_feedback_target_ratio > 1
    ):
        raise RemoteOrchestratorError("fill_feedback_target_ratio_outside_zero_one")
    if args.window_seconds <= 0:
        raise RemoteOrchestratorError("window_seconds_must_be_positive")
    if args.child_poll_seconds <= 0:
        raise RemoteOrchestratorError("child_poll_seconds_must_be_positive")
    if args.termination_grace_seconds < 0:
        raise RemoteOrchestratorError("termination_grace_seconds_must_be_nonnegative")
    if args.window_timeout_grace_seconds < 0:
        raise RemoteOrchestratorError("window_timeout_grace_seconds_must_be_nonnegative")
    if (
        not args.require_exact_envelope
        and args.exact_envelope_profile is not None
    ):
        raise RemoteOrchestratorError("exact_envelope_profile_requires_exact_envelope")
    if not args.require_exact_envelope:
        return
    if args.exact_envelope_profile is None:
        raise RemoteOrchestratorError("exact_envelope_profile_required")
    exact_checks = {
        "single_window": args.windows == 1,
        "max_loss_usdc": args.max_loss_usdc == EXACT_ENVELOPE_MAX_LOSS_USDC,
        "max_position_btc": args.max_position_btc == EXACT_ENVELOPE_MAX_POSITION_BTC,
        "private_proof_mode": args.private_proof_mode == "live_open_orders",
    }
    if args.exact_envelope_profile == EXACT_PROFILE_LEGACY_SINGLE_ORDER:
        exact_checks.update(
            {
                "max_order_size_btc": args.max_order_size == EXACT_ENVELOPE_MAX_ORDER_SIZE_BTC,
                "max_submissions": args.max_submissions == EXACT_ENVELOPE_MAX_SUBMISSIONS,
                "window_seconds": args.window_seconds <= EXACT_ENVELOPE_MAX_WINDOW_SECONDS,
                "quote_hold_seconds": args.quote_hold_seconds == 3,
                "wait_seconds": args.wait_seconds == 10,
                "hyperliquid_l2book_fast": args.hyperliquid_l2book_fast is True,
                "mode": args.mode.strip().replace("_", "-") == "event-driven-live",
                "exchange_reconciled_manager": args.exchange_reconciled_manager is False,
                "requote_attempts": args.requote_attempts == 1,
                "fill_feedback_activation": args.enable_fill_feedback is False,
                "fill_feedback_target_absent": (
                    args.fill_feedback_target_ratio is None
                ),
            }
        )
    elif args.exact_envelope_profile in {
        EXACT_PROFILE_TWO_SIDED_MANAGER,
        EXACT_PROFILE_TWO_SIDED_DYNAMIC_MANAGER,
        EXACT_PROFILE_TWO_SIDED_FILL_FEEDBACK_MANAGER,
    }:
        exact_checks.update(
            {
                "max_order_size_btc": args.max_order_size == EXACT_ENVELOPE_MAX_ORDER_SIZE_BTC,
                "max_submissions": args.max_submissions == EXACT_ENVELOPE_MAX_SUBMISSIONS,
                "window_seconds": args.window_seconds <= EXACT_ENVELOPE_MAX_WINDOW_SECONDS,
                "quote_hold_seconds": args.quote_hold_seconds == 3,
                "wait_seconds": args.wait_seconds == 10,
                "hyperliquid_l2book_fast": args.hyperliquid_l2book_fast is True,
                "mode": (
                    args.mode.strip().replace("_", "-")
                    == "event-driven-edge-gate-live"
                ),
                "exchange_reconciled_manager": args.exchange_reconciled_manager is True,
                "requote_attempts": args.requote_attempts == 2,
                "dynamic_spread_activation": args.enable_dynamic_spread
                == (
                    args.exact_envelope_profile
                    == EXACT_PROFILE_TWO_SIDED_DYNAMIC_MANAGER
                ),
                "fill_feedback_activation": args.enable_fill_feedback
                == (
                    args.exact_envelope_profile
                    == EXACT_PROFILE_TWO_SIDED_FILL_FEEDBACK_MANAGER
                ),
                "dynamic_and_fill_mutually_exclusive": not (
                    args.enable_dynamic_spread
                    and args.enable_fill_feedback
                ),
                "fill_feedback_target_absent_for_other_profiles": (
                    args.fill_feedback_target_ratio is None
                    or args.exact_envelope_profile
                    == EXACT_PROFILE_TWO_SIDED_FILL_FEEDBACK_MANAGER
                ),
            }
        )
    elif (
        args.exact_envelope_profile
        == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
    ):
        exact_checks.update(
            {
                "max_order_size_btc": args.max_order_size == 0.0,
                "max_submissions": args.max_submissions == 0,
                "window_seconds": (
                    args.window_seconds
                    == DELAYED_HISTORY_OBSERVE_ONLY_WINDOW_SECONDS
                ),
                "quote_hold_seconds": args.quote_hold_seconds == 0,
                "wait_seconds": args.wait_seconds == 0,
                "hyperliquid_l2book_fast": (
                    args.hyperliquid_l2book_fast is False
                ),
                "mode": (
                    args.mode.strip().replace("_", "-")
                    == DELAYED_HISTORY_OBSERVE_ONLY_MODE
                ),
                "exchange_reconciled_manager": (
                    args.exchange_reconciled_manager is False
                ),
                "requote_attempts": args.requote_attempts == 1,
                "fill_feedback_activation": args.enable_fill_feedback is False,
                "fill_feedback_target_absent": (
                    args.fill_feedback_target_ratio is None
                ),
                "python_executable": (
                    args.python == DEFAULT_REMOTE_PYTHON
                ),
                "watcher_script": (
                    args.watcher_script == DEFAULT_WATCHER_SCRIPT
                ),
                "env_file": args.env_file == DEFAULT_ENV_FILE,
                "lock_file": args.lock_file == DEFAULT_LOCK_FILE,
            }
        )
    else:
        exact_checks["exact_envelope_profile"] = False
    failed = [name for name, passed in exact_checks.items() if not passed]
    if failed:
        raise RemoteOrchestratorError(f"exact_envelope_mismatch:{','.join(failed)}")


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
        self._signal_number: int | None = None
        self._abort_reason = ""
        self._current_window = ""
        self._completed_windows: list[str] = []
        self._active_child: subprocess.Popen[str] | None = None
        self._last_child_lifecycle: dict[str, Any] = {}
        self._runtime_source_provenance: dict[str, Any] = {}

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
            "signal_number": self._signal_number,
            "abort_reason": self._abort_reason,
            "child_lifecycle": dict(self._last_child_lifecycle),
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
        thread = self._heartbeat_thread
        if thread is None:
            return
        thread.join(timeout=5)
        if thread.is_alive():
            raise RemoteOrchestratorError("heartbeat_thread_did_not_stop")
        self._heartbeat_thread = None

    def seal_terminal_artifacts(self) -> dict[str, Any]:
        self.stop_heartbeat()
        write_sha256_manifest(self.run_root)
        return verify_sha256_manifest(self.run_root)

    def write_runtime_source_provenance(self) -> dict[str, Any]:
        marker_path = self.remote_repo / "source_commit.txt"
        source_commit = source_commit_marker(self.remote_repo)
        if not valid_full_commit(source_commit):
            raise RemoteOrchestratorError("runtime_source_commit_missing_or_invalid")
        if self.args.require_exact_envelope and not marker_path.is_file():
            raise RemoteOrchestratorError("runtime_source_commit_marker_missing")
        watcher_path = resolve_watcher_script(self.remote_repo, self.args.watcher_script)
        try:
            watcher_relative = watcher_path.relative_to(self.remote_repo).as_posix()
            watcher_in_remote_source = True
        except ValueError:
            if self.args.require_exact_envelope:
                raise RemoteOrchestratorError("watcher_script_outside_remote_source")
            watcher_relative = str(watcher_path)
            watcher_in_remote_source = False
        if not watcher_path.is_file():
            raise RemoteOrchestratorError("watcher_script_missing")
        if self.args.require_exact_envelope and watcher_relative != DEFAULT_WATCHER_SCRIPT:
            raise RemoteOrchestratorError("exact_envelope_watcher_script_mismatch")

        files = []
        for path in runtime_source_files(self.remote_repo):
            relative = path.relative_to(self.remote_repo).as_posix()
            files.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
        file_paths = {row["path"] for row in files}
        if watcher_in_remote_source and watcher_relative not in file_paths:
            raise RemoteOrchestratorError("watcher_script_not_in_runtime_source_scope")
        payload = {
            "schema_version": RUNTIME_SOURCE_SCHEMA_VERSION,
            "status": "pass",
            "task_id": self.task_id,
            "sealed_at_utc": utc_now(),
            "sealed_before_watcher_start": True,
            "source_commit": source_commit,
            "source_commit_source": "source_commit.txt" if marker_path.is_file() else "git_rev_parse",
            "remote_repo": str(self.remote_repo),
            "run_root": str(self.run_root),
            "python_executable": str(self.args.python),
            "watcher_script": watcher_relative,
            "watcher_command_script": str(self.args.watcher_script),
            "watcher_script_in_remote_source": watcher_in_remote_source,
            "watcher_commands": [
                self.watcher_command(
                    self.run_root / f"window_{index:02d}",
                    window_id=index,
                )
                for index in range(1, int(self.args.windows) + 1)
            ],
            "source_scope": "non-test examples/hyperliquid/**/*.py",
            "file_count": len(files),
            "files": files,
            "watcher_process_started": False,
            "private_endpoint_called": False,
            "order_endpoint_called": False,
            "cancel_endpoint_called": False,
        }
        write_json(self.run_root / RUNTIME_SOURCE_PROVENANCE_NAME, payload)
        source_marker_output = self.run_root / "source_commit.txt"
        source_marker_tmp = source_marker_output.with_suffix(".txt.tmp")
        source_marker_tmp.write_text(source_commit + "\n", encoding="utf-8")
        source_marker_tmp.replace(source_marker_output)
        self._runtime_source_provenance = payload
        return payload

    def verify_runtime_source_provenance(self, *, output_name: str, phase: str) -> dict[str, Any]:
        provenance = self._runtime_source_provenance or load_existing_json(
            self.run_root / RUNTIME_SOURCE_PROVENANCE_NAME
        )
        expected_rows = provenance.get("files")
        if not isinstance(expected_rows, list) or not expected_rows:
            raise RemoteOrchestratorError("runtime_source_provenance_files_missing")
        expected = {
            str(row.get("path", "")): str(row.get("sha256", ""))
            for row in expected_rows
            if isinstance(row, dict) and row.get("path")
        }
        actual_paths = runtime_source_files(self.remote_repo)
        actual = {
            path.relative_to(self.remote_repo).as_posix(): sha256_file(path)
            for path in actual_paths
        }
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        mismatched = sorted(
            path
            for path in set(expected) & set(actual)
            if expected[path] != actual[path]
        )
        source_commit = source_commit_marker(self.remote_repo)
        commit_matches = source_commit == provenance.get("source_commit")
        status = "pass" if not missing and not unexpected and not mismatched and commit_matches else "fail"
        payload = {
            "schema_version": "cross_exchange_runtime_source_verification_v1",
            "status": status,
            "task_id": self.task_id,
            "phase": phase,
            "verified_at_utc": utc_now(),
            "source_commit": source_commit,
            "expected_source_commit": provenance.get("source_commit", ""),
            "source_commit_matches": commit_matches,
            "expected_file_count": len(expected),
            "actual_file_count": len(actual),
            "missing_files": missing,
            "unexpected_files": unexpected,
            "mismatched_files": mismatched,
            "watcher_process_started": self._active_child is not None,
            "private_endpoint_called_by_orchestrator": False,
            "order_endpoint_called_by_orchestrator": False,
            "cancel_endpoint_called_by_orchestrator": False,
        }
        write_json(self.run_root / output_name, payload)
        if status != "pass":
            raise RemoteOrchestratorError(f"runtime_source_{phase}_verification_failed")
        return payload

    def request_abort(self, signum: int, _frame: Any) -> None:
        self._abort_requested = True
        self._signal_name = signal.Signals(signum).name
        self._signal_number = signum
        self._abort_reason = "signal_received"

    def watcher_command(self, window_dir: Path, *, window_id: int) -> list[str]:
        command = [
            self.args.python,
            self.args.watcher_script,
            mode_flag(self.args.mode),
            "--watcher-seconds",
            str(self.args.window_seconds),
            "--max-order-size",
            str(self.args.max_order_size),
            "--max-loss-usdc",
            str(self.args.max_loss_usdc),
            "--max-position-btc",
            str(self.args.max_position_btc),
            "--max-real-order-submissions",
            str(self.args.max_submissions),
            "--requote-attempts",
            str(self.args.requote_attempts),
            "--quote-hold-seconds",
            str(self.args.quote_hold_seconds),
            "--wait-seconds",
            str(self.args.wait_seconds),
            "--env-file",
            self.args.env_file,
            "--artifact-task-id",
            self.task_id,
            "--artifact-window-id",
            str(window_id),
            "--run-id",
            f"{self.task_id}:window_{window_id:02d}",
            "--output-dir",
            str(window_dir),
        ]
        if self.args.hyperliquid_l2book_fast:
            command.append("--hyperliquid-l2book-fast")
        if self.args.exchange_reconciled_manager:
            command.append("--exchange-reconciled-manager")
        if self.args.enable_dynamic_spread:
            command.append("--enable-dynamic-spread")
        if self.args.enable_fill_feedback:
            command.append("--enable-fill-feedback")
            if self.args.fill_feedback_target_ratio is not None:
                command.extend(
                    [
                        "--fill-feedback-target-ratio",
                        str(self.args.fill_feedback_target_ratio),
                    ]
                )
        return command

    def write_preflight(self, output: Path) -> dict[str, Any]:
        commands = [
            self.watcher_command(self.run_root / f"window_{index:02d}", window_id=index)
            for index in range(1, int(self.args.windows) + 1)
        ]
        payload = {
            "schema_version": "cross_exchange_live_orchestrator_preflight_v1",
            "status": "pass",
            "preflight_only": True,
            "task_id": self.task_id,
            "source_commit": source_commit_marker(self.remote_repo),
            "remote_repo": str(self.remote_repo),
            "run_root": str(self.run_root),
            "watcher_commands": commands,
            "envelope": {
                "exact_envelope_profile": self.args.exact_envelope_profile,
                "mode": self.args.mode.strip().replace("_", "-"),
                "symbol": executor.SYMBOL,
                "windows": self.args.windows,
                "window_seconds": self.args.window_seconds,
                "max_order_size_btc": self.args.max_order_size,
                "max_loss_usdc": self.args.max_loss_usdc,
                "max_position_btc": self.args.max_position_btc,
                "max_real_order_submissions": self.args.max_submissions,
                "quote_hold_seconds": self.args.quote_hold_seconds,
                "wait_seconds": self.args.wait_seconds,
                "requote_attempts": self.args.requote_attempts,
                "exchange_reconciled_manager": self.args.exchange_reconciled_manager,
                "dynamic_spread_activation_enabled": bool(
                    self.args.enable_dynamic_spread
                ),
                "fill_feedback_activation_enabled": bool(
                    self.args.enable_fill_feedback
                ),
                "fill_feedback_target_fill_ratio": (
                    ""
                    if self.args.fill_feedback_target_ratio is None
                    else self.args.fill_feedback_target_ratio
                ),
                "hyperliquid_l2book_fast": self.args.hyperliquid_l2book_fast,
                "private_proof_mode": self.args.private_proof_mode,
                "lead_source": (
                    "binance_public_book_ticker"
                    if self.args.exact_envelope_profile
                    in {
                        EXACT_PROFILE_TWO_SIDED_MANAGER,
                        EXACT_PROFILE_TWO_SIDED_DYNAMIC_MANAGER,
                    }
                    else (
                        "legacy_public_trigger"
                        if self.args.exact_envelope_profile
                        == EXACT_PROFILE_LEGACY_SINGLE_ORDER
                        else (
                            "none"
                            if self.args.exact_envelope_profile
                            == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                            else "unspecified"
                        )
                    )
                ),
                "post_only_tif": (
                    "not_applicable"
                    if self.args.exact_envelope_profile
                    == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                    else POST_ONLY_TIF
                ),
            },
            "artifact_identity": {
                "task_id": self.task_id,
                "window_ids": list(range(1, int(self.args.windows) + 1)),
            },
            "strategy_activation": {
                "dynamic_spread_activation_enabled": bool(
                    self.args.enable_dynamic_spread
                ),
                "fill_feedback_activation_enabled": bool(
                    self.args.enable_fill_feedback
                ),
                "inventory_skew_activation_enabled": False,
                "multi_level_activation_enabled": False,
                "actual_quote_behavior_changed": False,
            },
            "probe_contract": {
                "enabled": (
                    self.args.exact_envelope_profile
                    == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                ),
                "probe_kind": (
                    "synthetic_cloid"
                    if self.args.exact_envelope_profile
                    == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                    else ""
                ),
                "direct_query_rounds": (
                    DELAYED_HISTORY_OBSERVE_ONLY_DIRECT_ROUNDS
                    if self.args.exact_envelope_profile
                    == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                    else 0
                ),
                "historical_calls": (
                    DELAYED_HISTORY_OBSERVE_ONLY_HISTORY_CALLS
                    if self.args.exact_envelope_profile
                    == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                    else 0
                ),
                "propagation_delay_seconds": (
                    DELAYED_HISTORY_OBSERVE_ONLY_PROPAGATION_DELAY_SECONDS
                    if self.args.exact_envelope_profile
                    == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                    else 0.0
                ),
                "final_snapshot_reserve_seconds": (
                    DELAYED_HISTORY_OBSERVE_ONLY_FINAL_RESERVE_SECONDS
                    if self.args.exact_envelope_profile
                    == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                    else 0.0
                ),
                "total_query_budget_seconds": (
                    DELAYED_HISTORY_OBSERVE_ONLY_TOTAL_BUDGET_SECONDS
                    if self.args.exact_envelope_profile
                    == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                    else 0.0
                ),
                "terminal_participation": False,
                "public_market_data_connected": False,
            },
            "execution_boundary": {
                "exact_envelope_required": self.args.require_exact_envelope,
                "watcher_process_started": False,
                "credential_file_read": False,
                "private_endpoint_called": False,
                "account_endpoint_called": False,
                "order_endpoint_called": False,
                "cancel_endpoint_called": False,
            },
        }
        write_json(output, payload)
        return payload

    def _terminate_watcher(
        self,
        child: subprocess.Popen[str],
        lifecycle: dict[str, Any],
        *,
        reason: str,
    ) -> tuple[int, dict[str, Any]]:
        lifecycle["termination_requested"] = True
        lifecycle["termination_reason"] = reason
        if child.poll() is not None:
            child.wait()
            lifecycle["child_returncode"] = child.returncode
            lifecycle["child_reaped"] = True
            return child.returncode, lifecycle

        pgid = lifecycle.get("child_process_group_id")
        lifecycle["termination_signal"] = "SIGTERM"
        try:
            if pgid is not None:
                os.killpg(int(pgid), signal.SIGTERM)
            else:
                child.terminate()
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + float(self.args.termination_grace_seconds)
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(min(float(self.args.child_poll_seconds), max(0.0, deadline - time.monotonic())))

        if child.poll() is None:
            lifecycle["termination_escalated_to_sigkill"] = True
            lifecycle["termination_signal"] = "SIGKILL"
            try:
                if pgid is not None:
                    os.killpg(int(pgid), signal.SIGKILL)
                else:
                    child.kill()
            except ProcessLookupError:
                pass
        child.wait()
        lifecycle["child_returncode"] = child.returncode
        lifecycle["child_reaped"] = True
        return child.returncode, lifecycle

    def _run_watcher(
        self,
        command: list[str],
        *,
        stdout: Any,
        stderr: Any,
    ) -> tuple[int, dict[str, Any]]:
        timeout_seconds = float(self.args.window_seconds) + float(self.args.window_timeout_grace_seconds)
        self.verify_runtime_source_provenance(
            output_name=RUNTIME_SOURCE_START_VERIFICATION_NAME,
            phase="pre_watcher_start",
        )
        child = subprocess.Popen(
            command,
            cwd=self.remote_repo,
            stdout=stdout,
            stderr=stderr,
            text=True,
            start_new_session=True,
        )
        self._active_child = child
        try:
            lifecycle: dict[str, Any] = {
                "child_pid": child.pid,
                "child_process_group_id": os.getpgid(child.pid),
                "termination_requested": False,
                "termination_reason": "",
                "termination_signal": "",
                "termination_escalated_to_sigkill": False,
                "child_returncode": None,
                "child_reaped": False,
                "watcher_timeout_seconds": timeout_seconds,
                "open_orders_proof_after_child_exit": False,
            }
        except Exception:
            lifecycle = {
                "child_pid": child.pid,
                "child_process_group_id": None,
                "termination_requested": False,
                "termination_reason": "",
                "termination_signal": "",
                "termination_escalated_to_sigkill": False,
                "child_returncode": None,
                "child_reaped": False,
                "watcher_timeout_seconds": timeout_seconds,
                "open_orders_proof_after_child_exit": False,
            }
        self._last_child_lifecycle = dict(lifecycle)
        started = time.monotonic()
        try:
            while True:
                returncode = child.poll()
                if returncode is not None:
                    child.wait()
                    lifecycle["child_returncode"] = child.returncode
                    lifecycle["child_reaped"] = True
                    self._last_child_lifecycle = dict(lifecycle)
                    return child.returncode, lifecycle
                if self._abort_requested:
                    self.write_status(
                        state="aborting",
                        phase="termination_requested",
                        extra={
                            "termination_reason": self._abort_reason or "signal_received",
                            "child_lifecycle": dict(lifecycle),
                        },
                    )
                    return self._terminate_watcher(
                        child,
                        lifecycle,
                        reason=self._abort_reason or "signal_received",
                    )
                if time.monotonic() - started >= timeout_seconds:
                    self._abort_requested = True
                    self._abort_reason = "watcher_timeout"
                    self.write_status(
                        state="aborting",
                        phase="watcher_timeout",
                        extra={
                            "termination_reason": self._abort_reason,
                            "child_lifecycle": dict(lifecycle),
                        },
                    )
                    return self._terminate_watcher(child, lifecycle, reason=self._abort_reason)
                time.sleep(float(self.args.child_poll_seconds))
        finally:
            self._last_child_lifecycle = dict(lifecycle)
            self._active_child = None

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
        self._last_child_lifecycle = {}
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
        command = self.watcher_command(window_dir, window_id=index)
        (window_dir / "runner_command.json").write_text(
            json.dumps({"command": command, "redacted": True}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stdout_path = window_dir / "runner_stdout.log"
        stderr_path = window_dir / "runner_stderr.log"
        rc = 0
        child_lifecycle: dict[str, Any] = {}
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            rc, child_lifecycle = self._run_watcher(command, stdout=stdout, stderr=stderr)
        proof_payload: dict[str, Any] = {}
        proof_error = ""
        child_lifecycle["open_orders_proof_after_child_exit"] = bool(child_lifecycle.get("child_reaped"))
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
        self._last_child_lifecycle = dict(child_lifecycle)
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
            **child_lifecycle,
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
            try:
                self.write_runtime_source_provenance()
            except Exception as exc:
                failure = {
                    "schema_version": RUNTIME_SOURCE_SCHEMA_VERSION,
                    "status": "fail",
                    "task_id": self.task_id,
                    "sealed_before_watcher_start": False,
                    "error": executor._redacted_error(exc),
                    "watcher_process_started": False,
                    "private_endpoint_called": False,
                    "order_endpoint_called": False,
                    "cancel_endpoint_called": False,
                }
                write_json(self.run_root / RUNTIME_SOURCE_PROVENANCE_NAME, failure)
                write_json(
                    self.run_root / "abort_manifest.json",
                    {
                        "task_id": self.task_id,
                        "state": "failed",
                        "phase": "runtime_source_provenance",
                        "error": failure["error"],
                        "watcher_process_started": False,
                        "private_endpoint_called": False,
                        "order_endpoint_called": False,
                        "cancel_endpoint_called": False,
                    },
                )
                write_sha256_manifest(self.run_root)
                verify_sha256_manifest(self.run_root)
                return 2
            self.start_heartbeat()
            self.write_status(
                state="running",
                phase="started",
                extra={
                    "host": os.uname().nodename,
                    "windows_requested": self.args.windows,
                    "window_seconds": self.args.window_seconds,
                    "mode": self.args.mode,
                    "exact_envelope_profile": self.args.exact_envelope_profile,
                    "exchange_reconciled_manager": self.args.exchange_reconciled_manager,
                    "requote_attempts": self.args.requote_attempts,
                    "post_only": (
                        "not_applicable"
                        if self.args.exact_envelope_profile
                        == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                        else POST_ONLY_TIF
                    ),
                    "max_order_size": self.args.max_order_size,
                    "max_submissions": self.args.max_submissions,
                    "private_proof_mode": self.args.private_proof_mode,
                    "child_poll_seconds": self.args.child_poll_seconds,
                    "termination_grace_seconds": self.args.termination_grace_seconds,
                    "window_timeout_grace_seconds": self.args.window_timeout_grace_seconds,
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
                self.verify_runtime_source_provenance(
                    output_name=RUNTIME_SOURCE_POSTRUN_VERIFICATION_NAME,
                    phase="postrun",
                )
                completed = {
                    "task_id": self.task_id,
                    "state": "complete",
                    "completed_at_utc": utc_now(),
                    "run_root": str(self.run_root),
                    "windows_completed": list(self._completed_windows),
                    "window_results": window_results,
                    "remote_sha256_manifest": SHA256_MANIFEST_NAME,
                    "remote_sha256_verification": SHA256_VERIFICATION_NAME,
                    "post_only": (
                        "not_applicable"
                        if self.args.exact_envelope_profile
                        == EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY
                        else POST_ONLY_TIF
                    ),
                    "order_endpoint_called_by_orchestrator": False,
                    "cancel_endpoint_called_by_orchestrator": False,
                    "private_proof_mode": self.args.private_proof_mode,
                }
                write_json(self.run_root / "run_complete.json", completed)
                self.write_status(state="complete", phase="complete", extra={"windows_completed": list(self._completed_windows)})
                verification = self.seal_terminal_artifacts()
                if verification["status"] != "pass":
                    return 2
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
                    "signal_number": self._signal_number,
                    "abort_reason": self._abort_reason,
                    "remote_sha256_manifest": SHA256_MANIFEST_NAME,
                    "remote_sha256_verification": SHA256_VERIFICATION_NAME,
                    **dict(self._last_child_lifecycle),
                    "child_lifecycle": dict(self._last_child_lifecycle),
                    "window_results": window_results,
                }
                try:
                    if not self._last_child_lifecycle:
                        raise RemoteOrchestratorError("private_abort_proof_skipped_before_child_start")
                    abort["root_open_orders_proof"] = self.write_open_orders_proof(
                        self.run_root / "independent_abort_open_orders_check.json",
                        window=self._current_window or "root",
                        phase="abort",
                    )
                except Exception as proof_exc:
                    abort["root_open_orders_error"] = executor._redacted_error(proof_exc)
                write_json(self.run_root / "abort_manifest.json", abort)
                self.write_status(state="failed", phase="failed", extra={"error": abort["error"]})
                self.seal_terminal_artifacts()
                return 2
            finally:
                self.stop_heartbeat()
                self._current_window = ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
    )
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--remote-repo", default=DEFAULT_REMOTE_REPO)
    parser.add_argument("--python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-root", default="")
    parser.add_argument("--watcher-script", default=DEFAULT_WATCHER_SCRIPT)
    parser.add_argument("--mode", default=DEFAULT_MODE)
    parser.add_argument(
        "--exact-envelope-profile",
        choices=(
            EXACT_PROFILE_LEGACY_SINGLE_ORDER,
            EXACT_PROFILE_TWO_SIDED_MANAGER,
            EXACT_PROFILE_TWO_SIDED_DYNAMIC_MANAGER,
            EXACT_PROFILE_TWO_SIDED_FILL_FEEDBACK_MANAGER,
            EXACT_PROFILE_DELAYED_HISTORY_OBSERVE_ONLY,
        ),
        default=None,
    )
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--window-seconds", type=float, default=1800.0)
    parser.add_argument("--max-order-size", type=float, default=0.005)
    parser.add_argument("--max-loss-usdc", type=float, default=1.0)
    parser.add_argument("--max-position-btc", type=float, default=0.01)
    parser.add_argument("--max-submissions", type=int, default=2)
    parser.add_argument("--requote-attempts", type=int, default=1)
    parser.add_argument("--exchange-reconciled-manager", action="store_true")
    parser.add_argument(
        "--enable-dynamic-spread",
        action="store_true",
        help="Enable only the bounded event-time dynamic half-spread candidate.",
    )
    parser.add_argument(
        "--enable-fill-feedback",
        action="store_true",
        help="Enable only the bounded exposure-weighted fill-feedback quote candidate.",
    )
    parser.add_argument(
        "--fill-feedback-target-ratio",
        type=float,
        default=None,
        help="Optional fill-feedback target ratio in [0, 1]; omission is an explicit neutral fallback.",
    )
    parser.add_argument("--quote-hold-seconds", type=int, default=3)
    parser.add_argument("--wait-seconds", type=int, default=10)
    parser.add_argument("--hyperliquid-l2book-fast", action="store_true")
    parser.add_argument(
        "--require-exact-envelope",
        action="store_true",
        help="Fail before any watcher/private/order work unless the current conservative one-window envelope matches exactly.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Render the exact watcher command and envelope without starting a watcher or reading credentials.",
    )
    parser.add_argument("--preflight-output", default="")
    parser.add_argument("--lock-file", default=DEFAULT_LOCK_FILE)
    parser.add_argument("--heartbeat-interval-seconds", type=float, default=15.0)
    parser.add_argument("--child-poll-seconds", type=float, default=DEFAULT_CHILD_POLL_SECONDS)
    parser.add_argument("--termination-grace-seconds", type=float, default=DEFAULT_TERMINATION_GRACE_SECONDS)
    parser.add_argument("--window-timeout-grace-seconds", type=float, default=DEFAULT_WINDOW_TIMEOUT_GRACE_SECONDS)
    parser.add_argument(
        "--private-proof-mode",
        choices=("live_open_orders", "skipped_for_test"),
        default="live_open_orders",
        help="Use skipped_for_test only in offline tests. Live runs must keep the default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    validate_args(args)
    orchestrator = RemoteLiveOrchestrator(args)
    if args.preflight_only:
        output = (
            Path(args.preflight_output).resolve()
            if args.preflight_output
            else orchestrator.run_root / "orchestrator_preflight.json"
        )
        orchestrator.write_preflight(output)
        return 0
    return orchestrator.run()


if __name__ == "__main__":
    raise SystemExit(main())
