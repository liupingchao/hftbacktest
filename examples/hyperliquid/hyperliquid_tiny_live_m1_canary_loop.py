#!/usr/bin/env python3
"""M1 repeated tiny-live canary loop orchestrator.

The loop is deliberately narrow: git-safe remote refresh, read-only final gate,
three independent canary windows, pullback, and aggregate validation. Real order
placement remains delegated to hyperliquid_tiny_live_real_order_executor.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0618T007"
REMOTE_HOST = "awsserver1"
REMOTE_PATH = "/home/admin/hftbacktest-cross-exchange"
REMOTE_ARTIFACT_ROOT = "/home/admin/hftbacktest_live_artifacts"
REMOTE_PYTHON = "/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python"
DEFAULT_ENV_FILE = "/home/admin/XEMM_rust_latest/.env"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m1_canary_loop_0618T007"
FINAL_GATE_SCRIPT = "examples/hyperliquid/hyperliquid_tiny_live_final_go_no_go_gate.py"
EXECUTOR_SCRIPT = "examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py"
SELF_TEST_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_real_order_canary_0618T004_selftest" / "executor_manifest.json"
OPERATOR_ACK = "I_UNDERSTAND_THIS_CAN_PLACE_REAL_HYPERLIQUID_ORDERS"
PASS_RECOMMENDATION = "hyperliquid_tiny_live_m1_repeated_canary_ready_for_qa"
BLOCKED_RECOMMENDATION = "hyperliquid_tiny_live_m1_repeated_canary_blocked"


class LoopError(RuntimeError):
    """Raised when the loop must stop fail-closed."""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def run_command(command: list[str], *, cwd: Path = PROJECT_ROOT, check: bool = True, timeout: int | None = None) -> CommandResult:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    result = CommandResult(command=command, returncode=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)
    if check and completed.returncode != 0:
        raise LoopError(f"command_failed:{' '.join(command)}:{completed.stderr.strip() or completed.stdout.strip()}")
    return result


def ssh(remote_command: str, *, check: bool = True, timeout: int | None = None) -> CommandResult:
    return run_command(["ssh", REMOTE_HOST, remote_command], check=check, timeout=timeout)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def git_short_head() -> str:
    return run_command(["git", "rev-parse", "--short", "HEAD"]).stdout.strip()


def git_full_head() -> str:
    return run_command(["git", "rev-parse", "HEAD"]).stdout.strip()


def git_branch() -> str:
    return run_command(["git", "branch", "--show-current"]).stdout.strip()


def git_dirty_count() -> int:
    return len(run_command(["git", "status", "--short"]).stdout.splitlines())


def collect_remote_facts() -> dict[str, Any]:
    script = (
        f"cd {REMOTE_PATH} && "
        "printf 'branch=%s\\n' \"$(git branch --show-current)\" && "
        "printf 'full_commit=%s\\n' \"$(git rev-parse HEAD)\" && "
        "printf 'commit=%s\\n' \"$(git rev-parse --short HEAD)\" && "
        "printf 'dirty_count=%s\\n' \"$(git status --short | wc -l)\" && "
        f"printf 'python={REMOTE_PYTHON}\\n' && "
        f"printf 'python_version=%s\\n' \"$({REMOTE_PYTHON} --version 2>&1)\" && "
        "printf 'hyperliquid_sdk_available=%s\\n' "
        f"\"$({REMOTE_PYTHON} -c 'import importlib.util; print(str(importlib.util.find_spec(\"hyperliquid\") is not None).lower())')\""
    )
    result = ssh(script, timeout=30)
    facts: dict[str, Any] = {
        "remote_path": REMOTE_PATH,
    }
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        facts[key] = value.strip()
    return facts


def write_remote_facts(path: Path, facts: dict[str, Any]) -> None:
    write_json(
        path,
        {
            "remote_path": REMOTE_PATH,
            "branch": facts.get("branch", ""),
            "full_commit": facts.get("full_commit", ""),
            "commit": facts.get("commit", ""),
            "dirty_count": facts.get("dirty_count", ""),
            "hyperliquid_sdk_available": facts.get("hyperliquid_sdk_available", ""),
            "python": REMOTE_PYTHON,
            "python_version": facts.get("python_version", ""),
        },
    )


def refresh_remote_checkout(output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    local_branch = git_branch()
    local_full_commit = git_full_head()
    local_short_commit = git_short_head()
    if local_branch != "cross-exchange":
        raise LoopError(f"local_branch_not_cross_exchange:{local_branch}")

    before = collect_remote_facts()
    rows.append({"step": "remote_before", "status": "observed", "detail": json.dumps(before, sort_keys=True)})
    if before.get("branch") != "cross-exchange":
        raise LoopError(f"remote_branch_not_cross_exchange:{before.get('branch')}")
    if str(before.get("dirty_count")) != "0":
        raise LoopError(f"remote_dirty_count_nonzero:{before.get('dirty_count')}")
    if before.get("full_commit") == local_full_commit:
        rows.append({"step": "remote_sync", "status": "skipped", "detail": "already_at_local_head"})
        return rows

    bundle = Path("/tmp") / f"hftbacktest-m1-{local_short_commit}.bundle"
    remote_bundle = f"/home/admin/hftbacktest-m1-{local_short_commit}.bundle"
    run_command(["git", "bundle", "create", str(bundle), "HEAD", local_branch])
    run_command(["scp", str(bundle), f"{REMOTE_HOST}:{remote_bundle}"], timeout=120)
    rows.append({"step": "bundle_upload", "status": "pass", "detail": str(bundle)})

    ssh(
        f"cd {REMOTE_PATH} && "
        f"git fetch {remote_bundle} {local_branch} && "
        "test -z \"$(git status --short)\" && "
        "git merge --ff-only FETCH_HEAD && "
        "test -z \"$(git status --short)\"",
        timeout=120,
    )
    after = collect_remote_facts()
    rows.append({"step": "remote_after", "status": "observed", "detail": json.dumps(after, sort_keys=True)})
    if after.get("full_commit") != local_full_commit:
        raise LoopError(f"remote_commit_not_local_after_refresh:{after.get('full_commit')}!={local_full_commit}")
    if str(after.get("dirty_count")) != "0":
        raise LoopError(f"remote_dirty_after_refresh:{after.get('dirty_count')}")
    return rows


def run_final_gate(output_dir: Path) -> dict[str, Any]:
    facts = collect_remote_facts()
    remote_facts_path = output_dir / "final_gate" / "remote_state_input.json"
    write_remote_facts(remote_facts_path, facts)
    command = [
        sys.executable,
        FINAL_GATE_SCRIPT,
        "--output-dir",
        str(output_dir / "final_gate"),
        "--remote-facts",
        str(remote_facts_path),
        "--executor-manifest",
        str(SELF_TEST_MANIFEST),
    ]
    run_command(command, timeout=60)
    manifest = json.loads((output_dir / "final_gate" / "final_go_no_go_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("allow_create_0617T008") is not True:
        raise LoopError(f"final_gate_no_go:{manifest.get('blocking_reasons')}")
    return manifest


def pullback(remote_dir: str, local_dir: Path) -> None:
    if local_dir.exists():
        shutil.rmtree(local_dir)
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    run_command(["scp", "-r", f"{REMOTE_HOST}:{remote_dir}", str(local_dir)], timeout=180)


def validate_window(local_dir: Path, window_id: int) -> dict[str, Any]:
    manifest_path = local_dir / "executor_manifest.json"
    if not manifest_path.exists():
        raise LoopError(f"window_{window_id}_missing_executor_manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cancel_path = local_dir / "cancel_shutdown_proof.json"
    if not cancel_path.exists():
        raise LoopError(f"window_{window_id}_missing_cancel_shutdown_proof")
    cancel_proof = json.loads(cancel_path.read_text(encoding="utf-8"))
    final_open_orders = cancel_proof.get("final_open_orders", [])
    checks = {
        "window": window_id,
        "final_recommendation": manifest.get("final_recommendation", ""),
        "order_submission_attempted": bool(manifest.get("order_submission_attempted")),
        "order_status_types": ",".join(str(item) for item in manifest.get("order_status_types", [])),
        "private_endpoint_called": bool(manifest.get("private_endpoint_called")),
        "real_order_endpoint_called": bool(manifest.get("real_order_endpoint_called")),
        "real_cancel_endpoint_called": bool(manifest.get("real_cancel_endpoint_called")),
        "schedule_cancel_endpoint_called": bool(manifest.get("schedule_cancel_endpoint_called")),
        "schedule_cancel_required": bool(manifest.get("schedule_cancel_required")),
        "use_schedule_cancel": bool(manifest.get("use_schedule_cancel")),
        "shutdown_proof_status": manifest.get("shutdown_proof_status", ""),
        "final_open_orders_count": len(final_open_orders),
        "credentials_written": bool(manifest.get("credentials_written")),
        "secret_values_written": bool(manifest.get("secret_values_written")),
        "raw_signatures_written": bool(manifest.get("raw_signatures_written")),
        "max_loss_status": manifest.get("max_loss_status", ""),
        "git_commit": manifest.get("git_commit", ""),
        "artifact_dir": str(local_dir),
    }
    failures = []
    if checks["final_recommendation"] != "hyperliquid_tiny_live_real_order_canary_ready_for_qa":
        failures.append("recommendation_not_ready")
    if not checks["order_submission_attempted"]:
        failures.append("order_not_attempted")
    if "resting" not in checks["order_status_types"]:
        failures.append("order_did_not_reach_resting")
    if not checks["private_endpoint_called"] or not checks["real_order_endpoint_called"] or not checks["real_cancel_endpoint_called"]:
        failures.append("required_endpoint_missing")
    if checks["schedule_cancel_endpoint_called"] or checks["use_schedule_cancel"]:
        failures.append("schedule_cancel_was_used")
    if checks["schedule_cancel_required"]:
        failures.append("schedule_cancel_marked_required")
    if checks["shutdown_proof_status"] != "pass":
        failures.append("shutdown_not_pass")
    if checks["final_open_orders_count"] != 0:
        failures.append("final_open_orders_not_empty")
    if checks["credentials_written"] or checks["secret_values_written"] or checks["raw_signatures_written"]:
        failures.append("secret_boundary_failed")
    if checks["max_loss_status"] != "pass":
        failures.append("max_loss_not_pass")
    checks["gate_status"] = "pass" if not failures else "fail"
    checks["failures"] = ",".join(failures)
    if failures:
        raise LoopError(f"window_{window_id}_failed:{checks['failures']}")
    return checks


def run_window(window_id: int, output_dir: Path, *, env_file: str, price_offset_bps: float) -> dict[str, Any]:
    remote_dir = f"{REMOTE_ARTIFACT_ROOT}/{TASK_ID}_window_{window_id}"
    local_dir = output_dir / f"window_{window_id}" / "pulled_back_awsserver1"
    ssh(f"rm -rf {remote_dir} && mkdir -p {remote_dir}", timeout=30)
    command = (
        f"cd {REMOTE_PATH} && "
        f"{REMOTE_PYTHON} {EXECUTOR_SCRIPT} "
        "--real-order-canary "
        f"--env-file {env_file} "
        f"--output-dir {remote_dir} "
        f"--canary-task-id {TASK_ID} "
        "--disable-schedule-cancel "
        f"--canary-price-offset-bps {price_offset_bps} "
        f"--operator-ack {OPERATOR_ACK}"
    )
    ssh(command, timeout=180)
    pullback(remote_dir, local_dir)
    return validate_window(local_dir, window_id)


def artifact_nonempty_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(output_dir)), "size_bytes": path.stat().st_size, "status": "pass" if path.stat().st_size > 0 else "fail"})
    return rows


def run_loop(*, output_dir: Path, windows: int, env_file: str, price_offset_bps: float) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    blocking_reasons: list[str] = []
    git_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    final_gate_manifest: dict[str, Any] = {}
    try:
        if windows < 1:
            raise LoopError("windows_must_be_positive")
        git_rows = refresh_remote_checkout(output_dir)
        final_gate_manifest = run_final_gate(output_dir)
        for window_id in range(1, windows + 1):
            window_rows.append(run_window(window_id, output_dir, env_file=env_file, price_offset_bps=price_offset_bps))
            time.sleep(2)
    except Exception as exc:
        blocking_reasons.append(str(exc))

    all_windows_pass = len(window_rows) == windows and all(row.get("gate_status") == "pass" for row in window_rows)
    final_recommendation = PASS_RECOMMENDATION if not blocking_reasons and all_windows_pass else BLOCKED_RECOMMENDATION
    write_csv(output_dir / "git_safety_gate.csv", git_rows, ["step", "status", "detail"])
    write_csv(
        output_dir / "window_gate_matrix.csv",
        window_rows,
        [
            "window",
            "gate_status",
            "failures",
            "final_recommendation",
            "order_submission_attempted",
            "order_status_types",
            "private_endpoint_called",
            "real_order_endpoint_called",
            "real_cancel_endpoint_called",
            "schedule_cancel_endpoint_called",
            "schedule_cancel_required",
            "use_schedule_cancel",
            "shutdown_proof_status",
            "final_open_orders_count",
            "credentials_written",
            "secret_values_written",
            "raw_signatures_written",
            "max_loss_status",
            "git_commit",
            "artifact_dir",
        ],
    )
    artifact_rows = artifact_nonempty_rows(output_dir)
    write_csv(output_dir / "artifact_nonempty_check.csv", artifact_rows, ["path", "size_bytes", "status"])
    manifest = {
        "task_id": TASK_ID,
        "final_recommendation": final_recommendation,
        "blocking_reasons": blocking_reasons,
        "windows_requested": windows,
        "windows_completed": len(window_rows),
        "windows_passed": sum(1 for row in window_rows if row.get("gate_status") == "pass"),
        "schedule_cancel_required": False,
        "tracked_cancel_required": True,
        "final_open_orders_empty_required": True,
        "git_safe_refresh_only": True,
        "destructive_git_operations_allowed": False,
        "local_commit": git_short_head(),
        "local_full_commit": git_full_head(),
        "local_branch": git_branch(),
        "remote_facts_after": collect_remote_facts() if not blocking_reasons or git_rows else {},
        "final_gate": {
            "allow_create_0617T008": final_gate_manifest.get("allow_create_0617T008"),
            "final_recommendation": final_gate_manifest.get("final_recommendation", ""),
            "blocking_reasons": final_gate_manifest.get("blocking_reasons", []),
        },
        "output_files": {
            "artifact_nonempty_check": str(output_dir / "artifact_nonempty_check.csv"),
            "git_safety_gate": str(output_dir / "git_safety_gate.csv"),
            "window_gate_matrix": str(output_dir / "window_gate_matrix.csv"),
        },
    }
    write_json(output_dir / "m1_canary_loop_manifest.json", manifest)
    readme = [
        "# Hyperliquid M1 Repeated Tiny-Live Canary Loop",
        "",
        f"Final recommendation: `{final_recommendation}`",
        "",
        "The loop uses git-safe checkout refresh, final gate validation, three independent real-order canary windows, tracked cancel, final open-orders proof, and redacted pullback artifacts.",
        "",
        "It does not rely on scheduled-cancel, does not run a continuous strategy, does not claim PnL, and does not authorize scale-up or default-on behavior.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    if blocking_reasons:
        raise LoopError(";".join(blocking_reasons))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--canary-price-offset-bps", type=float, default=200.0)
    args = parser.parse_args()
    manifest = run_loop(
        output_dir=args.output_dir,
        windows=args.windows,
        env_file=args.env_file,
        price_offset_bps=args.canary_price_offset_bps,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
