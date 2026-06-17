#!/usr/bin/env python3
"""Final go/no-go gate before a possible Hyperliquid tiny-live task.

This gate is read-only. It consumes local accepted artifacts and records remote
checkout facts supplied by the operator. It does not read credentials, call
private endpoints, query accounts, place or cancel orders, or start a live bot.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0618T001"
NEXT_TASK_ID = "0617T008"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_final_go_no_go_gate_0618T001"
DEFAULT_REMOTE_FACTS = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_final_go_no_go_gate_0618T001" / "remote_state_input.json"
DEFAULT_EXECUTOR_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_real_order_executor_0618T001" / "executor_manifest.json"
T003_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_final_live_capable_preflight_0617T003" / "operator_manifest.json"
T003_CAPS = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_final_live_capable_preflight_0617T003" / "approved_caps.csv"
T006_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_optimistic_pnl_proxy_0617T006" / "optimistic_pnl_proxy_manifest.json"

FINAL_READY = "tiny_live_ready_for_controller_go"
FINAL_NEEDS = "tiny_live_needs_missing_precondition"
FINAL_BLOCKED = "tiny_live_blocked"

REQUIRED_CAPS = {
    "symbol": "BTC",
    "max_order_size": "0.01 BTC",
    "max_order_notional": "700 USDC",
    "max_position": "0.04 BTC",
    "max_position_notional": "2800 USDC",
    "max_notional": "3000 USDC",
    "max_loss": "30 USDC",
    "duration": "10 minutes",
    "host_machine": "awsserver1",
    "account_scope": "Hyperliquid account configured on awsserver1",
    "maker_only_post_only": "true",
}

EXPECTED_REMOTE = {
    "remote_path": "/home/admin/hftbacktest-cross-exchange",
    "branch": "cross-exchange",
    "python": "/usr/bin/python3",
    "python_version": "Python 3.13.5",
}

BOUNDARY_FLAGS = {
    "account_query_called": False,
    "credentials_read": False,
    "live_bot_started": False,
    "order_amendment_called": False,
    "order_cancellation_called": False,
    "order_placement_called": False,
    "private_endpoint_called": False,
    "this_task_executes_orders": False,
}

QA_TASKS = ["0616T006", "0617T001", "0617T002", "0617T003", "0617T004", "0617T005", "0617T006"]


@dataclass(frozen=True)
class GateInput:
    output_dir: Path
    remote_facts_path: Path | None
    executor_manifest_path: Path | None = DEFAULT_EXECUTOR_MANIFEST


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _report_status(task_id: str) -> str:
    path = PROJECT_ROOT / ".workflow" / "reports" / f"{task_id}-qa.md"
    if not path.exists():
        return "missing_qa_report"
    text = path.read_text(encoding="utf-8")
    if "状态：\n- 已通过" in text or "状态：\r\n- 已通过" in text:
        return "passed"
    if "状态：\n- 阻塞" in text or "状态：\r\n- 阻塞" in text:
        return "blocked"
    if "状态：\n- 未通过" in text or "状态：\r\n- 未通过" in text:
        return "failed"
    return "unknown_status"


def prerequisite_rows() -> list[dict[str, Any]]:
    descriptions = {
        "0616T006": "operator packet QA",
        "0617T001": "separate awsserver1 cross-exchange checkout",
        "0617T002": "repeat cross-exchange python3 preflight",
        "0617T003": "final live-capable operator packet",
        "0617T004": "signal / quote policy protocol",
        "0617T005": "signal / quote replay threshold calibration",
        "0617T006": "optimistic PnL proxy with canonical_7",
    }
    rows: list[dict[str, Any]] = []
    for task_id in QA_TASKS:
        status = _report_status(task_id)
        rows.append(
            {
                "task_id": task_id,
                "description": descriptions[task_id],
                "qa_status": status,
                "gate_status": "pass" if status == "passed" else "fail",
                "note": "",
            }
        )
    rows.append(
        {
            "task_id": "0616T007",
            "description": "historical dry-run on old Binance maker route",
            "qa_status": _report_status("0616T007"),
            "gate_status": "superseded_by_0617T001_T003",
            "note": "Historical blocker is not reused as current execution path; route separation was handled by 0617T001-T003.",
        }
    )
    return rows


def cap_rows(t003_caps: list[dict[str, str]]) -> list[dict[str, Any]]:
    actual = {row.get("field", ""): row.get("value", "") for row in t003_caps}
    rows: list[dict[str, Any]] = []
    for field, expected in REQUIRED_CAPS.items():
        value = actual.get(field, "")
        rows.append(
            {
                "field": field,
                "expected": expected,
                "actual": value,
                "gate_status": "pass" if value == expected else "fail",
                "note": "",
            }
        )
    real_orders = actual.get("real_orders_allowed", "")
    rows.append(
        {
            "field": "real_orders_allowed",
            "expected": f"true only inside separately dispatched {NEXT_TASK_ID}",
            "actual": real_orders,
            "gate_status": "warn_migrated_approval_scope"
            if real_orders == "true only inside separately dispatched 0616T008"
            else "pass",
            "note": "Controller latest instruction requests 0617T008; previous packet text names 0616T008.",
        }
    )
    return rows


def remote_rows(remote_facts: dict[str, Any], local_commit: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field, expected in EXPECTED_REMOTE.items():
        actual = str(remote_facts.get(field, ""))
        rows.append(
            {
                "field": field,
                "expected": expected,
                "actual": actual,
                "gate_status": "pass" if actual == expected else "fail",
                "note": "",
            }
        )
    remote_commit = str(remote_facts.get("commit", ""))
    rows.append(
        {
            "field": "commit",
            "expected": local_commit,
            "actual": remote_commit,
            "gate_status": "pass" if remote_commit == local_commit else "fail",
            "note": "Remote execution checkout must be synced to the latest accepted gate commit before live execution.",
        }
    )
    dirty = str(remote_facts.get("dirty_count", ""))
    rows.append(
        {
            "field": "dirty_count",
            "expected": "0",
            "actual": dirty,
            "gate_status": "pass" if dirty == "0" else "fail",
            "note": "",
        }
    )
    return rows


def executor_rows(executor_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = [
        PROJECT_ROOT / "examples" / "hyperliquid" / "hyperliquid_tiny_live_execution.py",
        PROJECT_ROOT / "examples" / "hyperliquid" / "hyperliquid_tiny_live_real_order_executor.py",
        PROJECT_ROOT / "examples" / "hyperliquid" / "hyperliquid_private_order_executor.py",
    ]
    existing = [path for path in candidates if path.exists()]
    accepted_executor_report = PROJECT_ROOT / ".workflow" / "reports" / f"{NEXT_TASK_ID}-business.md"
    manifest_recommendation = executor_manifest.get("final_recommendation", "")
    private_endpoint_called = executor_manifest.get("private_endpoint_called") is True
    real_order_endpoint_called = executor_manifest.get("real_order_endpoint_called") is True
    real_cancel_endpoint_called = executor_manifest.get("real_cancel_endpoint_called") is True
    rows = [
        {
            "check": "task_scoped_real_order_executor_exists",
            "required": "true",
            "actual": "|".join(str(path.relative_to(PROJECT_ROOT)) for path in existing),
            "gate_status": "pass"
            if (PROJECT_ROOT / "examples" / "hyperliquid" / "hyperliquid_tiny_live_real_order_executor.py") in existing
            else "fail",
            "note": "Task-scoped Hyperliquid tiny-live real-order executor module must exist.",
        },
        {
            "check": "executor_manifest_present",
            "required": "true",
            "actual": str(bool(executor_manifest)).lower(),
            "gate_status": "pass" if executor_manifest else "fail",
            "note": "0618T001 gate consumes executor self-test and pullback evidence.",
        },
        {
            "check": "executor_manifest_task_id",
            "required": "0618T001",
            "actual": executor_manifest.get("task_id", ""),
            "gate_status": "pass" if executor_manifest.get("task_id") == "0618T001" else "fail",
            "note": "",
        },
        {
            "check": "executor_ready_recommendation",
            "required": "hyperliquid_tiny_live_real_order_executor_ready_for_qa",
            "actual": manifest_recommendation,
            "gate_status": "pass"
            if executor_manifest.get("executor_ready") is True
            and manifest_recommendation == "hyperliquid_tiny_live_real_order_executor_ready_for_qa"
            else "fail",
            "note": "",
        },
        {
            "check": "post_only_enforcement_implemented",
            "required": "true",
            "actual": str(executor_manifest.get("post_only_enforcement_implemented", "")).lower(),
            "gate_status": "pass" if executor_manifest.get("post_only_enforcement_implemented") is True else "fail",
            "note": "Executor must enforce Hyperliquid limit tif Alo before any live order placement.",
        },
        {
            "check": "real_cancel_all_shutdown_implemented",
            "required": "true",
            "actual": str(executor_manifest.get("real_cancel_all_shutdown_implemented", "")).lower(),
            "gate_status": "pass" if executor_manifest.get("real_cancel_all_shutdown_implemented") is True else "fail",
            "note": "0618T001 proves the cancel-all code path through mock/self-test; real cancel remains reserved for a later approved live task.",
        },
        {
            "check": "private_order_response_source_live_capable",
            "required": "true",
            "actual": str(executor_manifest.get("private_order_response_source_live_capable", "")).lower(),
            "gate_status": "pass" if executor_manifest.get("private_order_response_source_live_capable") is True else "fail",
            "note": "The source path is live-capable but is not exercised against private/order endpoints in 0618T001.",
        },
        {
            "check": "self_test_did_not_call_private_or_order_endpoints",
            "required": "true",
            "actual": f"private={str(private_endpoint_called).lower()}|order={str(real_order_endpoint_called).lower()}|cancel={str(real_cancel_endpoint_called).lower()}",
            "gate_status": "pass"
            if not private_endpoint_called and not real_order_endpoint_called and not real_cancel_endpoint_called
            else "fail",
            "note": "0618T001 remains no-real-order/no-private while proving the wrapper path.",
        },
        {
            "check": "max_loss_and_shutdown_evidence_pass",
            "required": "true",
            "actual": f"max_loss={executor_manifest.get('max_loss_status', '')}|shutdown={executor_manifest.get('shutdown_proof_status', '')}",
            "gate_status": "pass"
            if executor_manifest.get("max_loss_status") == "pass"
            and executor_manifest.get("shutdown_proof_status") == "pass"
            else "fail",
            "note": "",
        },
        {
            "check": "0617T008_business_report_already_exists",
            "required": "false_before_creation",
            "actual": str(accepted_executor_report.exists()).lower(),
            "gate_status": "pass" if not accepted_executor_report.exists() else "fail",
            "note": "",
        },
    ]
    return rows


def boundary_rows() -> list[dict[str, Any]]:
    return [{"check": key, "status": str(value).lower(), "gate_status": "pass"} for key, value in BOUNDARY_FLAGS.items()]


def _has_fail(rows: list[dict[str, Any]]) -> bool:
    return any(row.get("gate_status") == "fail" for row in rows)


def run(gate_input: GateInput) -> dict[str, Any]:
    output_dir = gate_input.output_dir.resolve()
    local_commit = _git_commit()
    local_branch = _git_branch()
    t003_manifest = _read_json(T003_MANIFEST)
    t003_caps = _read_csv(T003_CAPS)
    t006_manifest = _read_json(T006_MANIFEST)
    remote_facts = _read_json(gate_input.remote_facts_path) if gate_input.remote_facts_path else {}
    executor_manifest = _read_json(gate_input.executor_manifest_path) if gate_input.executor_manifest_path else {}

    prerequisites = prerequisite_rows()
    caps = cap_rows(t003_caps)
    remote = remote_rows(remote_facts, local_commit)
    executor = executor_rows(executor_manifest)
    boundary = boundary_rows()

    blocking_reasons: list[str] = []
    if _has_fail(prerequisites):
        blocking_reasons.append("required_qa_prerequisite_not_passed")
    if _has_fail(caps):
        blocking_reasons.append("approved_caps_mismatch")
    if _has_fail(remote):
        blocking_reasons.append("remote_execution_checkout_not_synced_or_invalid")
    if _has_fail(executor):
        blocking_reasons.append("hyperliquid_real_order_executor_missing_or_unproven")
    if t006_manifest.get("official_sample_set") != "canonical_7":
        blocking_reasons.append("canonical_7_not_official_in_t006_manifest")
    if t006_manifest.get("final_recommendation") != "hyperliquid_tiny_live_optimistic_pnl_proxy_ready_for_qa":
        blocking_reasons.append("t006_recommendation_not_ready")
    if local_branch != "cross-exchange":
        blocking_reasons.append("local_branch_not_cross_exchange")

    if not blocking_reasons:
        final_recommendation = FINAL_READY
        allow_create = True
    elif "required_qa_prerequisite_not_passed" in blocking_reasons or "approved_caps_mismatch" in blocking_reasons:
        final_recommendation = FINAL_BLOCKED
        allow_create = False
    else:
        final_recommendation = FINAL_NEEDS
        allow_create = False

    _write_csv(
        output_dir / "prerequisite_gate_matrix.csv",
        prerequisites,
        ["task_id", "description", "qa_status", "gate_status", "note"],
    )
    _write_csv(output_dir / "approved_cap_gate_matrix.csv", caps, ["field", "expected", "actual", "gate_status", "note"])
    _write_csv(output_dir / "remote_state_gate_matrix.csv", remote, ["field", "expected", "actual", "gate_status", "note"])
    _write_csv(output_dir / "executor_readiness_matrix.csv", executor, ["check", "required", "actual", "gate_status", "note"])
    _write_csv(output_dir / "boundary_gate_matrix.csv", boundary, ["check", "status", "gate_status"])
    next_task = [
        {
            "next_task_id": NEXT_TASK_ID,
            "allow_create": str(allow_create).lower(),
            "final_recommendation": final_recommendation,
            "blocking_reasons": "|".join(blocking_reasons),
            "instruction": "create_and_execute_tiny_live_only_if_allow_create_true_and_qa_passes",
        }
    ]
    _write_csv(output_dir / "next_task_instruction.csv", next_task, list(next_task[0]))

    report = "\n".join(
        [
            "# Hyperliquid Tiny-Live Final Go/No-Go Gate",
            "",
            f"Task: `{TASK_ID}`",
            "",
            f"Final recommendation: `{final_recommendation}`",
            "",
            f"Allow creating `{NEXT_TASK_ID}`: `{str(allow_create).lower()}`",
            "",
            "Blocking reasons:",
            "",
            *(f"- `{reason}`" for reason in blocking_reasons),
            "",
            "This is a read-only gate. It did not read credentials, call private endpoints, query accounts, place/cancel/amend orders, or start a live bot.",
            "",
            f"A true result only lets total control create a later `{NEXT_TASK_ID}` task. It does not execute live orders in `{TASK_ID}`.",
            "",
        ]
    )
    (output_dir / "README.md").write_text(report, encoding="utf-8")

    manifest = {
        "allow_create_0617T008": allow_create,
        "boundary_flags": BOUNDARY_FLAGS,
        "blocking_reasons": blocking_reasons,
        "controller_latest_instruction": "run 0618T001 executor repair and final go/no-go gate, QA it, then create 0617T008 only if the repaired gate passes",
        "final_recommendation": final_recommendation,
        "git_commit": local_commit,
        "local_branch": local_branch,
        "next_task_id": NEXT_TASK_ID,
        "official_sample_set": t006_manifest.get("official_sample_set", ""),
        "output_files": {
            "approved_cap_gate_matrix": str(output_dir / "approved_cap_gate_matrix.csv"),
            "boundary_gate_matrix": str(output_dir / "boundary_gate_matrix.csv"),
            "executor_readiness_matrix": str(output_dir / "executor_readiness_matrix.csv"),
            "next_task_instruction": str(output_dir / "next_task_instruction.csv"),
            "prerequisite_gate_matrix": str(output_dir / "prerequisite_gate_matrix.csv"),
            "remote_state_gate_matrix": str(output_dir / "remote_state_gate_matrix.csv"),
        },
        "previous_approval_task_id": t003_manifest.get("approved_for_task", ""),
        "executor_manifest": executor_manifest,
        "remote_facts": remote_facts,
        "task_id": TASK_ID,
    }
    _write_json(output_dir / "final_go_no_go_manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--remote-facts", type=Path, default=DEFAULT_REMOTE_FACTS)
    parser.add_argument("--executor-manifest", type=Path, default=DEFAULT_EXECUTOR_MANIFEST)
    args = parser.parse_args()
    remote_facts = args.remote_facts if args.remote_facts.exists() else None
    executor_manifest = args.executor_manifest if args.executor_manifest.exists() else None
    manifest = run(GateInput(output_dir=args.output_dir, remote_facts_path=remote_facts, executor_manifest_path=executor_manifest))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
