#!/usr/bin/env python3
"""M2B controlled Hyperliquid tiny-live fill loop.

The loop is narrow by design: git-safe refresh, final gate, maker-only Alo
windows, tracked cancel, live private/economics pullback, and T008 ledger
reconciliation. No fill or incomplete economics means M2 remains blocked.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import hyperliquid_tiny_live_m2_pnl_ledger as m2_ledger

TASK_ID = "0618T009"
REMOTE_HOST = "awsserver1"
REMOTE_PATH = "/home/admin/hftbacktest-cross-exchange"
REMOTE_ARTIFACT_ROOT = "/home/admin/hftbacktest_live_artifacts"
REMOTE_PYTHON = "/home/admin/.venvs/hyperliquid-sdk-0618T002/bin/python"
DEFAULT_ENV_FILE = "/home/admin/XEMM_rust_latest/.env"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_m2_fill_loop_0618T009"
FINAL_GATE_SCRIPT = "examples/hyperliquid/hyperliquid_tiny_live_final_go_no_go_gate.py"
REMOTE_WINDOW_SCRIPT = "examples/hyperliquid/hyperliquid_tiny_live_m2_fill_window.py"
SELF_TEST_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_real_order_canary_0618T004_selftest" / "executor_manifest.json"
OPERATOR_ACK = "I_UNDERSTAND_THIS_CAN_PLACE_REAL_HYPERLIQUID_ORDERS"
READY_RECOMMENDATION = "hyperliquid_tiny_live_m2_fill_loop_ready_for_qa"
BLOCKED_RECOMMENDATION = "hyperliquid_tiny_live_m2_fill_loop_blocked"


class LoopError(RuntimeError):
    """Raised when the loop must stop fail-closed."""


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def run_command(command: list[str], *, cwd: Path = PROJECT_ROOT, check: bool = True, timeout: int | None = None) -> CommandResult:
    completed = subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True, timeout=timeout)
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
    facts: dict[str, Any] = {"remote_path": REMOTE_PATH}
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


def refresh_remote_checkout() -> list[dict[str, Any]]:
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
    bundle = Path("/tmp") / f"hftbacktest-m2-{local_short_commit}.bundle"
    remote_bundle = f"/home/admin/hftbacktest-m2-{local_short_commit}.bundle"
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


def run_window(window_id: int, output_dir: Path, *, env_file: str, wait_seconds: int, quote_offset_ticks: int) -> dict[str, Any]:
    remote_dir = f"{REMOTE_ARTIFACT_ROOT}/{TASK_ID}_window_{window_id}"
    local_dir = output_dir / f"window_{window_id}" / "pulled_back_awsserver1"
    ssh(f"rm -rf {remote_dir} && mkdir -p {remote_dir}", timeout=30)
    command = (
        f"cd {REMOTE_PATH} && "
        f"{REMOTE_PYTHON} {REMOTE_WINDOW_SCRIPT} "
        f"--env-file {env_file} "
        f"--output-dir {remote_dir} "
        f"--window-id {window_id} "
        f"--wait-seconds {wait_seconds} "
        f"--quote-offset-ticks {quote_offset_ticks} "
        f"--operator-ack {OPERATOR_ACK}"
    )
    ssh(command, timeout=max(180, wait_seconds + 90))
    pullback(remote_dir, local_dir)
    manifest = json.loads((local_dir / "m2_fill_window_manifest.json").read_text(encoding="utf-8"))
    return {
        "window": window_id,
        "artifact_dir": str(local_dir),
        "final_recommendation": manifest.get("final_recommendation", ""),
        "blocking_reasons": ",".join(manifest.get("blocking_reasons", [])),
        "order_status_types": ",".join(manifest.get("order_status_types", [])),
        "fill_count": manifest.get("fill_count", 0),
        "maker_fill_count": manifest.get("maker_fill_count", 0),
        "ledger_fill_rows": manifest.get("ledger_fill_rows", 0),
        "real_order_endpoint_called": manifest.get("real_order_endpoint_called", False),
        "real_cancel_endpoint_called": manifest.get("real_cancel_endpoint_called", False),
        "final_open_orders_count": manifest.get("final_open_orders_count", 0),
        "shutdown_proof_status": manifest.get("shutdown_proof_status", ""),
        "post_only_tif": manifest.get("post_only_tif", ""),
        "crossing_guard_status": manifest.get("crossing_guard_status", ""),
        "credentials_written": manifest.get("credentials_written", False),
        "raw_signatures_written": manifest.get("raw_signatures_written", False),
    }


def aggregate_live_fills(output_dir: Path) -> Path:
    rows: list[dict[str, str]] = []
    fieldnames = [
        "source_window",
        "fill_id",
        "side",
        "qty_btc",
        "price_usdc",
        "intent_price_usdc",
        "mark_price_usdc",
        "fee_usdc",
        "rebate_usdc",
        "liquidity",
    ]
    for path in sorted(output_dir.glob("window_*/pulled_back_awsserver1/live_fill_ledger.csv")):
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append({field: row.get(field, "") for field in fieldnames})
    aggregate = output_dir / "aggregate_live_fill_ledger.csv"
    write_csv(aggregate, rows, fieldnames)
    return aggregate


def artifact_nonempty_rows(output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            rows.append({"path": str(path.relative_to(output_dir)), "size_bytes": path.stat().st_size, "status": "pass" if path.stat().st_size > 0 else "fail"})
    return rows


def run_loop(*, output_dir: Path, windows: int, env_file: str, wait_seconds: int, quote_offset_ticks: int) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    blocking_reasons: list[str] = []
    git_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    final_gate_manifest: dict[str, Any] = {}
    ledger_manifest: dict[str, Any] = {}
    try:
        if windows < 1 or windows > 3:
            raise LoopError("windows_must_be_1_to_3")
        git_rows = refresh_remote_checkout()
        final_gate_manifest = run_final_gate(output_dir)
        for window_id in range(1, windows + 1):
            row = run_window(window_id, output_dir, env_file=env_file, wait_seconds=wait_seconds, quote_offset_ticks=quote_offset_ticks)
            window_rows.append(row)
            if int(row.get("fill_count") or 0) > 0:
                break
            time.sleep(2)
        aggregate_fills = aggregate_live_fills(output_dir)
        ledger_output = output_dir / "ledger_reconciliation"
        ledger_manifest = m2_ledger.run_ledger(
            input_root=output_dir,
            output_dir=ledger_output,
            fill_ledger=aggregate_fills,
            fill_source_kind="live_pulled_back",
        )
    except Exception as exc:
        blocking_reasons.append(str(exc))

    fill_count = sum(int(row.get("fill_count") or 0) for row in window_rows)
    maker_fill_count = sum(int(row.get("maker_fill_count") or 0) for row in window_rows)
    ledger_pass = ledger_manifest.get("live_realized_pnl_proof") is True and ledger_manifest.get("realized_pnl_proof_status") == "pass"
    if not ledger_pass and "ledger_no_live_realized_pnl_proof" not in blocking_reasons:
        blocking_reasons.append("ledger_no_live_realized_pnl_proof")
    final_recommendation = READY_RECOMMENDATION if fill_count > 0 and maker_fill_count > 0 and ledger_pass and not blocking_reasons else BLOCKED_RECOMMENDATION

    write_csv(output_dir / "git_safety_gate.csv", git_rows, ["step", "status", "detail"])
    write_csv(
        output_dir / "window_result_matrix.csv",
        window_rows,
        [
            "window",
            "artifact_dir",
            "final_recommendation",
            "blocking_reasons",
            "order_status_types",
            "fill_count",
            "maker_fill_count",
            "ledger_fill_rows",
            "real_order_endpoint_called",
            "real_cancel_endpoint_called",
            "final_open_orders_count",
            "shutdown_proof_status",
            "post_only_tif",
            "crossing_guard_status",
            "credentials_written",
            "raw_signatures_written",
        ],
    )
    write_csv(output_dir / "artifact_nonempty_check.csv", artifact_nonempty_rows(output_dir), ["path", "size_bytes", "status"])
    manifest = {
        "task_id": TASK_ID,
        "final_recommendation": final_recommendation,
        "blocking_reasons": blocking_reasons,
        "windows_requested": windows,
        "windows_completed": len(window_rows),
        "fill_count": fill_count,
        "maker_fill_count": maker_fill_count,
        "ledger_pass": ledger_pass,
        "ledger_manifest": ledger_manifest,
        "git_safe_refresh_only": True,
        "local_commit": git_short_head(),
        "local_full_commit": git_full_head(),
        "local_branch": git_branch(),
        "remote_facts_after": collect_remote_facts() if git_rows else {},
        "final_gate": {
            "allow_create_0617T008": final_gate_manifest.get("allow_create_0617T008"),
            "final_recommendation": final_gate_manifest.get("final_recommendation", ""),
            "blocking_reasons": final_gate_manifest.get("blocking_reasons", []),
        },
        "output_files": {
            "window_result_matrix": str(output_dir / "window_result_matrix.csv"),
            "aggregate_live_fill_ledger": str(output_dir / "aggregate_live_fill_ledger.csv"),
            "ledger_reconciliation": str(output_dir / "ledger_reconciliation"),
        },
    }
    write_json(output_dir / "m2_fill_loop_manifest.json", manifest)
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# Hyperliquid M2B Controlled Tiny-Live Fill Loop",
                "",
                f"Final recommendation: `{final_recommendation}`",
                "",
                "This loop may place real post-only Alo orders under the approved caps. It does not permit taker/crossing orders or scale-up.",
                "",
                "No fill or incomplete ledger reconciliation keeps M2 blocked.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if final_recommendation != READY_RECOMMENDATION:
        raise LoopError(";".join(blocking_reasons))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--env-file", default=DEFAULT_ENV_FILE)
    parser.add_argument("--wait-seconds", type=int, default=45)
    parser.add_argument("--quote-offset-ticks", type=int, default=1)
    args = parser.parse_args()
    manifest = run_loop(
        output_dir=args.output_dir,
        windows=args.windows,
        env_file=args.env_file,
        wait_seconds=args.wait_seconds,
        quote_offset_ticks=args.quote_offset_ticks,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
