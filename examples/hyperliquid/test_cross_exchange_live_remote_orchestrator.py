from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "examples" / "hyperliquid" / "cross_exchange_live_remote_orchestrator.py"


def write_fake_watcher(path: Path, *, returncode: int = 0) -> None:
    path.write_text(
        "\n".join(
            [
                "import argparse, json",
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
                "payload = {",
                "    'task_id': args.artifact_task_id,",
                "    'artifact_window_id': args.artifact_window_id,",
                "    'watcher_seconds': args.watcher_seconds,",
                "    'max_order_size': args.max_order_size,",
                "    'max_submissions': args.max_real_order_submissions,",
                f"    'returncode': {returncode},",
                "}",
                "(out / 'fake_watcher_manifest.json').write_text(json.dumps(payload, sort_keys=True) + '\\n')",
                "raise SystemExit(payload['returncode'])",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_orchestrator(tmp_path: Path, fake_watcher: Path, *, windows: int = 2) -> subprocess.CompletedProcess[str]:
    run_root = tmp_path / "run"
    lock_file = tmp_path / "live.lock"
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task-id",
            "TESTT001",
            "--remote-repo",
            str(PROJECT_ROOT),
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
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        assert proof["proof_mode"] == "skipped_for_test"
        assert proof["final_open_orders_empty"] is True
        assert watcher_manifest["task_id"] == "TESTT001"


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
