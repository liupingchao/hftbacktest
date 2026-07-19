from __future__ import annotations

import json
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from examples.hyperliquid import cross_exchange_live_remote_orchestrator as orchestrator_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "examples" / "hyperliquid" / "cross_exchange_live_remote_orchestrator.py"


def write_fake_watcher(
    path: Path,
    *,
    returncode: int = 0,
    sleep_seconds: float = 0.0,
    ignore_sigterm: bool = False,
    mutate_source: Path | None = None,
) -> None:
    path.write_text(
        "\n".join(
            [
                "import argparse, json, os, signal, time",
                "from pathlib import Path",
                "parser = argparse.ArgumentParser()",
                "parser.add_argument('--event-driven-edge-gate-live', action='store_true')",
                "parser.add_argument('--watcher-seconds')",
                "parser.add_argument('--max-order-size')",
                "parser.add_argument('--max-loss-usdc')",
                "parser.add_argument('--max-position-btc')",
                "parser.add_argument('--max-real-order-submissions')",
                "parser.add_argument('--requote-attempts')",
                "parser.add_argument('--quote-hold-seconds')",
                "parser.add_argument('--wait-seconds')",
                "parser.add_argument('--env-file')",
                "parser.add_argument('--artifact-task-id')",
                "parser.add_argument('--artifact-window-id', type=int)",
                "parser.add_argument('--run-id')",
                "parser.add_argument('--output-dir')",
                "parser.add_argument('--hyperliquid-l2book-fast', action='store_true')",
                "parser.add_argument('--exchange-reconciled-manager', action='store_true')",
                "args = parser.parse_args()",
                "out = Path(args.output_dir)",
                "out.mkdir(parents=True, exist_ok=True)",
                "(out / 'fake_watcher_pid.txt').write_text(str(os.getpid()))",
                "run_root = out.parent",
                f"{'signal.signal(signal.SIGTERM, signal.SIG_IGN)' if ignore_sigterm else ''}",
                f"time.sleep({sleep_seconds})",
                "payload = {",
                "    'task_id': args.artifact_task_id,",
                "    'artifact_window_id': args.artifact_window_id,",
                "    'watcher_seconds': args.watcher_seconds,",
                "    'max_order_size': args.max_order_size,",
                "    'max_loss_usdc': args.max_loss_usdc,",
                "    'max_position_btc': args.max_position_btc,",
                "    'max_submissions': args.max_real_order_submissions,",
                "    'requote_attempts': args.requote_attempts,",
                "    'exchange_reconciled_manager': args.exchange_reconciled_manager,",
                "    'runtime_source_provenance_present_at_start': (run_root / 'runtime_source_provenance.json').is_file(),",
                "    'runtime_source_start_verification_passed': json.loads((run_root / 'runtime_source_start_verification.json').read_text()).get('status') == 'pass',",
                f"    'returncode': {returncode},",
                "}",
                "(out / 'fake_watcher_manifest.json').write_text(json.dumps(payload, sort_keys=True) + '\\n')",
                (
                    f"Path({str(mutate_source)!r}).write_text("
                    f"Path({str(mutate_source)!r}).read_text() + '\\n# runtime mutation\\n')"
                    if mutate_source is not None
                    else ""
                ),
                "raise SystemExit(payload['returncode'])",
                "",
            ]
        ),
        encoding="utf-8",
    )


def orchestrator_command(
    tmp_path: Path,
    fake_watcher: Path,
    *,
    windows: int = 2,
    extra_args: list[str] | None = None,
    remote_repo: Path = PROJECT_ROOT,
) -> list[str]:
    run_root = tmp_path / "run"
    lock_file = tmp_path / "live.lock"
    command = [
        sys.executable,
        str(SCRIPT),
        "--task-id",
        "TESTT001",
        "--remote-repo",
        str(remote_repo),
        "--python",
        sys.executable,
        "--watcher-script",
        str(fake_watcher),
        "--run-root",
        str(run_root),
        "--output-root",
        str(tmp_path),
        "--lock-file",
        str(lock_file),
        "--windows",
        str(windows),
        "--window-seconds",
        "0.01",
        "--max-order-size",
        "0.005",
        "--max-submissions",
        "2",
        "--private-proof-mode",
        "skipped_for_test",
        "--heartbeat-interval-seconds",
        "1",
        "--child-poll-seconds",
        "0.02",
        "--termination-grace-seconds",
        "0.1",
        "--window-timeout-grace-seconds",
        "0.1",
    ]
    command.extend(extra_args or [])
    return command


def run_orchestrator(
    tmp_path: Path,
    fake_watcher: Path,
    *,
    windows: int = 2,
    extra_args: list[str] | None = None,
    remote_repo: Path = PROJECT_ROOT,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        orchestrator_command(
            tmp_path,
            fake_watcher,
            windows=windows,
            extra_args=extra_args,
            remote_repo=remote_repo,
        ),
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def start_orchestrator(
    tmp_path: Path,
    fake_watcher: Path,
    *,
    windows: int = 2,
    extra_args: list[str] | None = None,
    remote_repo: Path = PROJECT_ROOT,
) -> subprocess.Popen[str]:
    return subprocess.Popen(
        orchestrator_command(
            tmp_path,
            fake_watcher,
            windows=windows,
            extra_args=extra_args,
            remote_repo=remote_repo,
        ),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def wait_for_file(path: Path, *, timeout_seconds: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def assert_pid_gone(pid: int, *, timeout_seconds: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.02)
    raise AssertionError(f"pid {pid} is still alive")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_remote_source(tmp_path: Path, *, with_marker: bool) -> Path:
    remote_repo = tmp_path / "remote_source"
    source = PROJECT_ROOT / "examples" / "hyperliquid"
    destination = remote_repo / "examples" / "hyperliquid"
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("test_*.py", "__pycache__", "*.pyc"),
    )
    if with_marker:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        (remote_repo / "source_commit.txt").write_text(commit + "\n", encoding="utf-8")
    return remote_repo


def manifest_entries(manifest: Path) -> list[tuple[str, str]]:
    entries = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative_name = line.split("  ", 1)
        entries.append((digest, relative_name))
    return entries


def assert_manifest_verifies(run_root: Path) -> None:
    manifest = run_root / "remote_sha256_manifest.txt"
    verification = read_json(run_root / "remote_sha256_verification.json")
    entries = manifest_entries(manifest)
    assert verification["status"] == "pass"
    assert verification["manifest_entry_count"] == len(entries)
    assert verification["verified_count"] == len(entries)
    assert verification["missing_count"] == 0
    assert verification["mismatch_count"] == 0
    for expected_digest, relative_name in entries:
        relative_path = Path(relative_name)
        assert not relative_path.is_absolute()
        assert ".." not in relative_path.parts
        actual = run_root / relative_path
        assert actual.is_file()
        assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected_digest


def test_remote_orchestrator_complete_contract(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)

    result = run_orchestrator(tmp_path, fake_watcher, windows=2)

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "run"
    status = read_json(run_root / "run_status.json")
    complete = read_json(run_root / "run_complete.json")
    heartbeat = read_json(run_root / "heartbeat.json")
    manifest = run_root / "remote_sha256_manifest.txt"

    assert status["state"] == "complete"
    assert status["phase"] == "complete"
    assert complete["windows_completed"] == ["01", "02"]
    assert heartbeat["task_id"] == "TESTT001"
    assert manifest.exists()
    assert "run_complete.json" in manifest.read_text(encoding="utf-8")

    for window in ("01", "02"):
        window_dir = run_root / f"window_{window}"
        window_status = read_json(window_dir / "window_status.json")
        proof = read_json(window_dir / "independent_remote_open_orders_check.json")
        watcher_manifest = read_json(window_dir / "fake_watcher_manifest.json")
        assert window_status["state"] == "complete"
        assert window_status["runner_returncode"] == 0
        assert window_status["child_reaped"] is True
        assert window_status["termination_requested"] is False
        assert window_status["open_orders_proof_after_child_exit"] is True
        assert proof["proof_mode"] == "skipped_for_test"
        assert proof["final_open_orders_empty"] is True
        assert watcher_manifest["task_id"] == "TESTT001"
        assert watcher_manifest["max_loss_usdc"] == "1.0"
        assert watcher_manifest["max_position_btc"] == "0.01"
        assert watcher_manifest["runtime_source_provenance_present_at_start"] is True
        assert watcher_manifest["runtime_source_start_verification_passed"] is True

    provenance = read_json(run_root / "runtime_source_provenance.json")
    start_verification = read_json(run_root / "runtime_source_start_verification.json")
    postrun_verification = read_json(run_root / "runtime_source_postrun_verification.json")
    assert provenance["status"] == "pass"
    assert provenance["sealed_before_watcher_start"] is True
    assert provenance["file_count"] > 0
    assert start_verification["status"] == "pass"
    assert start_verification["watcher_process_started"] is False
    assert postrun_verification["status"] == "pass"


def test_preflight_only_renders_exact_envelope_without_starting_watcher(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)
    preflight = tmp_path / "preflight.json"
    command = orchestrator_command(
        tmp_path,
        fake_watcher,
        windows=1,
        extra_args=[
            "--mode",
            "event-driven-live",
            "--exact-envelope-profile",
            "legacy-single-order",
            "--window-seconds",
            "900",
            "--quote-hold-seconds",
            "3",
            "--wait-seconds",
            "10",
            "--hyperliquid-l2book-fast",
            "--private-proof-mode",
            "live_open_orders",
            "--require-exact-envelope",
            "--preflight-only",
            "--preflight-output",
            str(preflight),
        ],
    )

    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stderr
    payload = read_json(preflight)
    assert payload["status"] == "pass"
    assert payload["task_id"] == "TESTT001"
    assert payload["envelope"]["max_order_size_btc"] == 0.005
    assert payload["envelope"]["max_loss_usdc"] == 1.0
    assert payload["envelope"]["max_position_btc"] == 0.01
    assert payload["envelope"]["max_real_order_submissions"] == 2
    assert payload["artifact_identity"]["window_ids"] == [1]
    assert all(value is False for value in payload["strategy_activation"].values())
    assert payload["execution_boundary"]["watcher_process_started"] is False
    assert payload["execution_boundary"]["credential_file_read"] is False
    assert payload["execution_boundary"]["private_endpoint_called"] is False
    assert payload["execution_boundary"]["order_endpoint_called"] is False
    assert payload["execution_boundary"]["cancel_endpoint_called"] is False
    command_row = payload["watcher_commands"][0]
    assert command_row[command_row.index("--artifact-task-id") + 1] == "TESTT001"
    assert command_row[command_row.index("--artifact-window-id") + 1] == "1"
    assert command_row[command_row.index("--run-id") + 1] == "TESTT001:window_01"
    assert command_row[command_row.index("--max-loss-usdc") + 1] == "1.0"
    assert command_row[command_row.index("--max-position-btc") + 1] == "0.01"
    assert not (tmp_path / "run" / "window_01").exists()


def test_exact_envelope_preflight_rejects_mismatch_before_output(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)
    preflight = tmp_path / "preflight.json"
    command = orchestrator_command(
        tmp_path,
        fake_watcher,
        windows=1,
        extra_args=[
            "--mode",
            "event-driven-live",
            "--exact-envelope-profile",
            "legacy-single-order",
            "--window-seconds",
            "900",
            "--max-loss-usdc",
            "2",
            "--quote-hold-seconds",
            "3",
            "--wait-seconds",
            "10",
            "--hyperliquid-l2book-fast",
            "--private-proof-mode",
            "live_open_orders",
            "--require-exact-envelope",
            "--preflight-only",
            "--preflight-output",
            str(preflight),
        ],
    )

    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)

    assert result.returncode != 0
    assert "exact_envelope_mismatch:max_loss_usdc" in result.stderr
    assert not preflight.exists()


def test_exact_envelope_requires_explicit_profile(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)
    command = orchestrator_command(
        tmp_path,
        fake_watcher,
        windows=1,
        extra_args=[
            "--mode",
            "event-driven-live",
            "--window-seconds",
            "900",
            "--quote-hold-seconds",
            "3",
            "--wait-seconds",
            "10",
            "--hyperliquid-l2book-fast",
            "--private-proof-mode",
            "live_open_orders",
            "--require-exact-envelope",
            "--preflight-only",
        ],
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "exact_envelope_profile_required" in result.stderr


def test_preflight_renders_exact_two_sided_manager_profile(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)
    preflight = tmp_path / "preflight.json"
    command = orchestrator_command(
        tmp_path,
        fake_watcher,
        windows=1,
        extra_args=[
            "--mode",
            "event-driven-edge-gate-live",
            "--exact-envelope-profile",
            "two-sided-manager",
            "--window-seconds",
            "900",
            "--quote-hold-seconds",
            "3",
            "--wait-seconds",
            "10",
            "--requote-attempts",
            "2",
            "--exchange-reconciled-manager",
            "--hyperliquid-l2book-fast",
            "--private-proof-mode",
            "live_open_orders",
            "--require-exact-envelope",
            "--preflight-only",
            "--preflight-output",
            str(preflight),
        ],
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = read_json(preflight)
    assert payload["envelope"]["exact_envelope_profile"] == "two-sided-manager"
    assert payload["envelope"]["mode"] == "event-driven-edge-gate-live"
    assert payload["envelope"]["requote_attempts"] == 2
    assert payload["envelope"]["exchange_reconciled_manager"] is True
    assert payload["envelope"]["hyperliquid_l2book_fast"] is True
    assert payload["envelope"]["private_proof_mode"] == "live_open_orders"
    assert payload["envelope"]["lead_source"] == "binance_public_book_ticker"
    command_row = payload["watcher_commands"][0]
    assert "--event-driven-edge-gate-live" in command_row
    assert "--exchange-reconciled-manager" in command_row
    assert command_row[command_row.index("--requote-attempts") + 1] == "2"
    assert command_row[
        command_row.index("--max-real-order-submissions") + 1
    ] == "2"


def test_exact_two_sided_profile_rejects_missing_manager_flag(
    tmp_path: Path,
) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)
    command = orchestrator_command(
        tmp_path,
        fake_watcher,
        windows=1,
        extra_args=[
            "--mode",
            "event-driven-edge-gate-live",
            "--exact-envelope-profile",
            "two-sided-manager",
            "--window-seconds",
            "900",
            "--quote-hold-seconds",
            "3",
            "--wait-seconds",
            "10",
            "--requote-attempts",
            "2",
            "--hyperliquid-l2book-fast",
            "--private-proof-mode",
            "live_open_orders",
            "--require-exact-envelope",
            "--preflight-only",
        ],
    )

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "exact_envelope_mismatch:exchange_reconciled_manager" in result.stderr


def test_success_manifest_verifies_all_entries(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)

    result = run_orchestrator(tmp_path, fake_watcher, windows=2)

    assert result.returncode == 0, result.stderr
    assert_manifest_verifies(tmp_path / "run")


def test_orchestrator_passes_distinct_artifact_window_ids(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)

    result = run_orchestrator(tmp_path, fake_watcher, windows=2)

    assert result.returncode == 0, result.stderr
    first = read_json(tmp_path / "run" / "window_01" / "fake_watcher_manifest.json")
    second = read_json(tmp_path / "run" / "window_02" / "fake_watcher_manifest.json")
    assert first["artifact_window_id"] == 1
    assert second["artifact_window_id"] == 2


def test_remote_orchestrator_failed_window_writes_abort_manifest(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher_fail.py"
    write_fake_watcher(fake_watcher, returncode=7)

    result = run_orchestrator(tmp_path, fake_watcher, windows=2)

    assert result.returncode == 2
    run_root = tmp_path / "run"
    status = read_json(run_root / "run_status.json")
    abort = read_json(run_root / "abort_manifest.json")
    window_status = read_json(run_root / "window_01" / "window_status.json")
    proof = read_json(run_root / "window_01" / "independent_remote_open_orders_check.json")

    assert status["state"] == "failed"
    assert status["phase"] == "failed"
    assert abort["state"] == "failed"
    assert "window_failed:01:rc=7" in abort["error"]
    assert abort["root_open_orders_proof"]["proof_mode"] == "skipped_for_test"
    assert window_status["state"] == "failed"
    assert window_status["runner_returncode"] == 7
    assert proof["final_open_orders_empty"] is True
    assert (run_root / "remote_sha256_manifest.txt").exists()


def test_failed_manifest_verifies_all_entries(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher_fail.py"
    write_fake_watcher(fake_watcher, returncode=7)

    result = run_orchestrator(tmp_path, fake_watcher, windows=2)

    assert result.returncode == 2, result.stderr
    assert_manifest_verifies(tmp_path / "run")


def test_heartbeat_is_stopped_before_manifest(tmp_path: Path, monkeypatch) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)
    command = orchestrator_command(tmp_path, fake_watcher, windows=1)
    args = orchestrator_module.build_parser().parse_args(command[2:])
    orchestrator = orchestrator_module.RemoteLiveOrchestrator(args)
    observed: dict[str, object] = {}
    original = orchestrator_module.write_sha256_manifest

    def observe(root: Path) -> Path:
        observed["manifest_calls"] = int(observed.get("manifest_calls", 0)) + 1
        observed["heartbeat_alive"] = bool(
            orchestrator._heartbeat_thread is not None and orchestrator._heartbeat_thread.is_alive()
        )
        observed["status"] = read_json(root / "run_status.json")
        events = [
            json.loads(line)
            for line in (root / "orchestrator_events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        observed["last_event"] = events[-1]
        return original(root)

    monkeypatch.setattr(orchestrator_module, "write_sha256_manifest", observe)
    assert orchestrator.run() == 0
    assert observed["manifest_calls"] == 1
    assert observed["heartbeat_alive"] is False


def test_status_and_event_log_are_final_before_manifest(tmp_path: Path, monkeypatch) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)
    command = orchestrator_command(tmp_path, fake_watcher, windows=1)
    args = orchestrator_module.build_parser().parse_args(command[2:])
    orchestrator = orchestrator_module.RemoteLiveOrchestrator(args)
    observed: dict[str, object] = {}
    original = orchestrator_module.write_sha256_manifest

    def observe(root: Path) -> Path:
        observed["status"] = read_json(root / "run_status.json")
        events = [
            json.loads(line)
            for line in (root / "orchestrator_events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        observed["last_event"] = events[-1]
        return original(root)

    monkeypatch.setattr(orchestrator_module, "write_sha256_manifest", observe)
    assert orchestrator.run() == 0
    assert observed["status"]["state"] == "complete"
    assert observed["status"]["phase"] == "complete"
    assert observed["last_event"]["state"] == "complete"
    assert observed["last_event"]["phase"] == "complete"


def test_post_seal_mutation_is_detected(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher.py"
    write_fake_watcher(fake_watcher, returncode=0)

    result = run_orchestrator(tmp_path, fake_watcher, windows=1)

    assert result.returncode == 0, result.stderr
    run_root = tmp_path / "run"
    complete_path = run_root / "run_complete.json"
    complete_path.write_text(complete_path.read_text(encoding="utf-8") + "mutation\n", encoding="utf-8")
    verification = orchestrator_module.verify_sha256_manifest(run_root)
    assert verification["status"] == "fail"
    assert verification["mismatch_count"] == 1


def test_exact_run_missing_source_marker_fails_before_watcher(tmp_path: Path) -> None:
    remote_repo = make_remote_source(tmp_path, with_marker=False)
    fake_watcher = remote_repo / orchestrator_module.DEFAULT_WATCHER_SCRIPT
    write_fake_watcher(fake_watcher)
    command = orchestrator_command(
        tmp_path,
        fake_watcher,
        windows=1,
        extra_args=["--require-exact-envelope"],
        remote_repo=remote_repo,
    )
    args = orchestrator_module.build_parser().parse_args(command[2:])
    orchestrator = orchestrator_module.RemoteLiveOrchestrator(args)

    assert orchestrator.run() == 2
    provenance = read_json(tmp_path / "run" / "runtime_source_provenance.json")
    assert provenance["status"] == "fail"
    assert "runtime_source_commit_missing_or_invalid" in provenance["error"]
    assert provenance["watcher_process_started"] is False
    assert not (tmp_path / "run" / "window_01" / "fake_watcher_pid.txt").exists()
    assert not (tmp_path / "run" / "independent_abort_open_orders_check.json").exists()


def test_runtime_source_mutation_during_watcher_fails_postrun(tmp_path: Path) -> None:
    remote_repo = make_remote_source(tmp_path, with_marker=True)
    fake_watcher = remote_repo / orchestrator_module.DEFAULT_WATCHER_SCRIPT
    mutate_source = remote_repo / "examples" / "hyperliquid" / "cross_exchange_price_math.py"
    write_fake_watcher(fake_watcher, mutate_source=mutate_source)

    result = run_orchestrator(
        tmp_path,
        fake_watcher,
        windows=1,
        remote_repo=remote_repo,
    )

    assert result.returncode == 2, result.stderr
    assert (tmp_path / "run" / "window_01" / "fake_watcher_pid.txt").exists()
    start = read_json(tmp_path / "run" / "runtime_source_start_verification.json")
    postrun = read_json(tmp_path / "run" / "runtime_source_postrun_verification.json")
    abort = read_json(tmp_path / "run" / "abort_manifest.json")
    assert start["status"] == "pass"
    assert postrun["status"] == "fail"
    assert "examples/hyperliquid/cross_exchange_price_math.py" in postrun["mismatched_files"]
    assert "runtime_source_postrun_verification_failed" in abort["error"]
    assert not (tmp_path / "run" / "run_complete.json").exists()


def test_manifest_uses_relative_paths(tmp_path: Path) -> None:
    (tmp_path / "window_01").mkdir()
    (tmp_path / "window_01" / "window_status.json").write_text("{}\n", encoding="utf-8")

    manifest = orchestrator_module.write_sha256_manifest(tmp_path)
    entries = manifest_entries(manifest)

    assert entries == [
        (
            hashlib.sha256(b"{}\n").hexdigest(),
            "window_01/window_status.json",
        )
    ]
    assert str(tmp_path) not in manifest.read_text(encoding="utf-8")


def test_sigterm_terminates_child_and_writes_abort_manifest(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher_sleep.py"
    write_fake_watcher(fake_watcher, sleep_seconds=30)
    process = start_orchestrator(tmp_path, fake_watcher, windows=2)
    pid_file = tmp_path / "run" / "window_01" / "fake_watcher_pid.txt"
    wait_for_file(pid_file)
    watcher_pid = int(pid_file.read_text(encoding="utf-8"))

    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 2, stderr
    assert stdout == ""
    abort = read_json(tmp_path / "run" / "abort_manifest.json")
    window_status = read_json(tmp_path / "run" / "window_01" / "window_status.json")
    assert abort["signal_name"] == "SIGTERM"
    assert abort["child_reaped"] is True
    assert abort["open_orders_proof_after_child_exit"] is True
    assert abort["child_lifecycle"]["termination_reason"] == "signal_received"
    assert window_status["termination_signal"] == "SIGTERM"
    assert window_status["child_reaped"] is True
    assert window_status["open_orders_proof_after_child_exit"] is True
    assert not (tmp_path / "run" / "window_02").exists()
    assert_pid_gone(watcher_pid)


def test_sigint_terminates_child_and_writes_abort_manifest(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher_sigint.py"
    write_fake_watcher(fake_watcher, sleep_seconds=30)
    process = start_orchestrator(tmp_path, fake_watcher, windows=2)
    pid_file = tmp_path / "run" / "window_01" / "fake_watcher_pid.txt"
    wait_for_file(pid_file)
    watcher_pid = int(pid_file.read_text(encoding="utf-8"))

    process.send_signal(signal.SIGINT)
    _stdout, stderr = process.communicate(timeout=5)

    assert process.returncode == 2, stderr
    abort = read_json(tmp_path / "run" / "abort_manifest.json")
    window_status = read_json(tmp_path / "run" / "window_01" / "window_status.json")
    assert abort["signal_name"] == "SIGINT"
    assert abort["child_reaped"] is True
    assert abort["open_orders_proof_after_child_exit"] is True
    assert window_status["termination_reason"] == "signal_received"
    assert window_status["child_reaped"] is True
    assert not (tmp_path / "run" / "window_02").exists()
    assert_pid_gone(watcher_pid)


def test_timeout_terminates_hung_child(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher_timeout.py"
    write_fake_watcher(fake_watcher, sleep_seconds=30)
    result = run_orchestrator(
        tmp_path,
        fake_watcher,
        windows=2,
        extra_args=[
            "--window-seconds",
            "0.05",
            "--window-timeout-grace-seconds",
            "0.05",
        ],
    )

    assert result.returncode == 2, result.stderr
    abort = read_json(tmp_path / "run" / "abort_manifest.json")
    window_status = read_json(tmp_path / "run" / "window_01" / "window_status.json")
    watcher_pid = int((tmp_path / "run" / "window_01" / "fake_watcher_pid.txt").read_text(encoding="utf-8"))
    assert abort["abort_reason"] == "watcher_timeout"
    assert window_status["termination_reason"] == "watcher_timeout"
    assert window_status["watcher_timeout_seconds"] == 0.1
    assert window_status["child_reaped"] is True
    assert not (tmp_path / "run" / "window_02").exists()
    assert_pid_gone(watcher_pid)


def test_sigkill_escalation_when_child_ignores_sigterm(tmp_path: Path) -> None:
    fake_watcher = tmp_path / "fake_watcher_ignore_term.py"
    write_fake_watcher(fake_watcher, sleep_seconds=30, ignore_sigterm=True)
    result = run_orchestrator(
        tmp_path,
        fake_watcher,
        windows=1,
        extra_args=[
            "--window-seconds",
            "0.05",
            "--window-timeout-grace-seconds",
            "0.05",
            "--termination-grace-seconds",
            "0.05",
        ],
    )

    assert result.returncode == 2, result.stderr
    window_status = read_json(tmp_path / "run" / "window_01" / "window_status.json")
    watcher_pid = int((tmp_path / "run" / "window_01" / "fake_watcher_pid.txt").read_text(encoding="utf-8"))
    assert window_status["termination_reason"] == "watcher_timeout"
    assert window_status["termination_signal"] == "SIGKILL"
    assert window_status["termination_escalated_to_sigkill"] is True
    assert window_status["child_reaped"] is True
    assert_pid_gone(watcher_pid)
