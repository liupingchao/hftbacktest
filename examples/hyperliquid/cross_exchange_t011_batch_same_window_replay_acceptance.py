#!/usr/bin/env python3
"""Batch same-window replay acceptance for T011 multi-window evidence.

This runner is offline-only. It consumes local artifacts and prior QA facts, and
it does not read credentials, call private/order/cancel/account endpoints,
collect market data, or run remote commands.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0709T002"
SCHEMA_VERSION = "cross_exchange_t011_batch_same_window_replay_acceptance_v1"
PASSED_RECOMMENDATION = "batch_same_window_replay_acceptance_passed"
BLOCKED_RECOMMENDATION = "batch_same_window_replay_acceptance_blocked"
DEFAULT_T011_ROOT = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t011_multi_window_live_evidence_0709T001_20260709T064251Z"
DEFAULT_QA_0708T002 = PROJECT_ROOT / ".workflow" / "reports" / "0708T002-qa.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t011_batch_same_window_replay_acceptance_0709T002"


MATRIX_FIELDS = [
    "window_id",
    "source_kind",
    "source_path",
    "live_classification",
    "replay_classification",
    "market_view_acceptance",
    "decision_path_acceptance",
    "lifecycle_acceptance",
    "economics_acceptance",
    "optimism_acceptance",
    "boundary_acceptance",
    "overall_acceptance",
    "blocker",
    "hyperliquid_l2book_fast",
    "reconnect_count",
    "l2book_messages",
    "trades_messages",
    "current_candidate_count",
    "trigger_count",
    "anti_drift_pass_count",
    "anti_drift_block_count",
    "edge_gate_pass_count",
    "edge_gate_block_count",
    "live_submissions_count",
    "max_real_order_submissions",
    "max_order_size_btc",
    "order_status_types",
    "post_only_reject_count",
    "fill_count",
    "maker_fill_count",
    "ledger_fill_rows",
    "final_open_orders_count",
    "independent_final_open_orders_count",
    "shutdown_proof_status",
    "real_order_endpoint_called",
    "real_cancel_endpoint_called",
    "scope_note",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str:
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_int(value: Any, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value)


def as_float(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    return float(value)



def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)

def passfail(condition: bool) -> str:
    return "pass" if condition else "fail"


def all_pass(*statuses: str) -> str:
    return "pass" if all(status == "pass" for status in statuses) else "fail"


def status_counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def qa_0708_reference_row(qa_report: Path) -> dict[str, Any]:
    text = qa_report.read_text(encoding="utf-8")
    accepted = "任务ID：\n- 0708T002" in text and "状态：\n- 已通过" in text
    status = "pass" if accepted else "fail"
    blocker = "" if accepted else "0708T002_qa_not_accepted"
    return {
        "window_id": "0708T001",
        "source_kind": "prior_accepted_replay_reference",
        "source_path": display_path(qa_report),
        "live_classification": "submitted_resting_no_fill",
        "replay_classification": "submitted_resting_no_fill",
        "market_view_acceptance": status,
        "decision_path_acceptance": status,
        "lifecycle_acceptance": status,
        "economics_acceptance": status,
        "optimism_acceptance": status,
        "boundary_acceptance": status,
        "overall_acceptance": status,
        "blocker": blocker,
        "hyperliquid_l2book_fast": "true",
        "reconnect_count": "0",
        "l2book_messages": "accepted_by_0708T002_QA",
        "trades_messages": "accepted_by_0708T002_QA",
        "current_candidate_count": "accepted_by_0708T002_QA",
        "trigger_count": "1",
        "anti_drift_pass_count": "accepted_by_0708T002_QA",
        "anti_drift_block_count": "accepted_by_0708T002_QA",
        "edge_gate_pass_count": "accepted_by_0708T002_QA",
        "edge_gate_block_count": "accepted_by_0708T002_QA",
        "live_submissions_count": "1",
        "max_real_order_submissions": "2",
        "max_order_size_btc": "0.002",
        "order_status_types": "resting",
        "post_only_reject_count": "0",
        "fill_count": "0",
        "maker_fill_count": "0",
        "ledger_fill_rows": "0",
        "final_open_orders_count": "0",
        "independent_final_open_orders_count": "0",
        "shutdown_proof_status": "pass",
        "real_order_endpoint_called": "true",
        "real_cancel_endpoint_called": "true",
        "scope_note": "0708T001 remains passing through prior QA-accepted 0708T002 same-window replay; original artifact package is not present in this checkout.",
    }


def order_status_types(private_response: dict[str, Any], inline: dict[str, Any]) -> list[str]:
    rows = private_response.get("order_status_rows") or []
    statuses = [str(row.get("status_type", "")) for row in rows if row.get("status_type")]
    if statuses:
        return statuses
    return [str(item) for item in inline.get("order_status_types", [])]


def classify_window(*, statuses: list[str], fill_count: int, maker_fill_count: int, submissions: int) -> str:
    if fill_count > 0 or maker_fill_count > 0:
        return "submitted_filled"
    if statuses and all(status == "resting" for status in statuses):
        return "submitted_resting_no_fill"
    if statuses and any(status == "error" for status in statuses):
        return "submitted_rejected"
    if submissions == 0:
        return "no_submit_fail_closed"
    return "blocked_or_malformed"


def evaluate_window(window_dir: Path, window_id: int) -> dict[str, Any]:
    watcher = read_json(window_dir / "event_driven_watcher_manifest.json")
    inline = read_json(window_dir / "inline_reprice_manifest.json")
    private_response = read_json(window_dir / "private_order_response_audit.json")
    cancel = read_json(window_dir / "cancel_shutdown_proof.json")
    public = read_json(window_dir / "public_stream_summary.json")
    independent = read_json(window_dir / "independent_final_open_orders_check.json")
    intents = read_csv_rows(window_dir / "order_intent_audit.csv")
    fills = read_csv_rows(window_dir / "live_fill_ledger.csv")

    statuses = order_status_types(private_response, inline)
    submissions = as_int(watcher.get("live_submissions_count"))
    fill_count = as_int(inline.get("fill_count"))
    maker_fill_count = as_int(inline.get("maker_fill_count"))
    ledger_rows = len(fills)
    classification = classify_window(statuses=statuses, fill_count=fill_count, maker_fill_count=maker_fill_count, submissions=submissions)
    message_counts = public.get("message_count_by_channel", {})
    l2book_messages = as_int(message_counts.get("l2Book"))
    trades_messages = as_int(message_counts.get("trades"))
    reconnect_count = as_int(public.get("reconnect_count"))
    max_order_size = as_float(watcher.get("max_order_size_btc"))
    max_submissions = as_int(watcher.get("max_real_order_submissions"))
    final_open = as_int(inline.get("final_open_orders_count"), -1)
    independent_open = as_int(independent.get("final_open_orders_count"), -1)
    post_only_rejects = as_int(inline.get("post_only_reject_count"))
    tif_ok = all((row.get("time_in_force") == "Alo" and as_float(row.get("size_btc")) <= 0.005) for row in intents) if intents else submissions == 0

    blockers: list[str] = []
    market_status = passfail(watcher.get("hyperliquid_l2book_fast") is True and l2book_messages > 0 and reconnect_count == 0)
    if market_status != "pass":
        blockers.append("market_view_not_accepted")
    decision_status = passfail(watcher.get("trigger_found") is True and as_int(watcher.get("trigger_count")) > 0 and watcher.get("edge_gate_live_compatible_source_available") is True and tif_ok)
    if decision_status != "pass":
        blockers.append("decision_path_not_accepted")
    lifecycle_ok = final_open == 0 and independent_open == 0 and inline.get("shutdown_proof_status") == "pass" and cancel.get("proof_status") == "pass"
    if classification == "submitted_resting_no_fill":
        lifecycle_ok = lifecycle_ok and submissions >= 1 and "resting" in statuses and inline.get("real_order_endpoint_called") is True and inline.get("real_cancel_endpoint_called") is True
    elif classification == "submitted_rejected":
        lifecycle_ok = lifecycle_ok and submissions >= 1 and post_only_rejects >= 1 and inline.get("real_order_endpoint_called") is True
    elif classification == "no_submit_fail_closed":
        lifecycle_ok = lifecycle_ok and inline.get("real_order_endpoint_called") is not True
    elif classification == "submitted_filled":
        lifecycle_ok = lifecycle_ok and fill_count > 0
    else:
        lifecycle_ok = False
    lifecycle_status = passfail(lifecycle_ok)
    if lifecycle_status != "pass":
        blockers.append("lifecycle_not_accepted")
    economics_status = passfail(fill_count == 0 and maker_fill_count == 0 and ledger_rows == 0)
    if economics_status != "pass":
        blockers.append("economics_requires_fill_attribution")
    replay_classification = classification if lifecycle_status == "pass" else "blocked_or_malformed"
    optimism_status = passfail(replay_classification == classification and fill_count == 0 and maker_fill_count == 0)
    if optimism_status != "pass":
        blockers.append("optimism_boundary_not_accepted")
    boundary_status = passfail(max_order_size <= 0.005 and max_submissions <= 2 and submissions <= 2 and tif_ok and watcher.get("hyperliquid_l2book_fast") is True)
    if boundary_status != "pass":
        blockers.append("boundary_not_accepted")
    overall = all_pass(market_status, decision_status, lifecycle_status, economics_status, optimism_status, boundary_status)

    return {
        "window_id": f"0709T001_window_{window_id:02d}",
        "source_kind": "t011_live_window_artifact",
        "source_path": display_path(window_dir),
        "live_classification": classification,
        "replay_classification": replay_classification,
        "market_view_acceptance": market_status,
        "decision_path_acceptance": decision_status,
        "lifecycle_acceptance": lifecycle_status,
        "economics_acceptance": economics_status,
        "optimism_acceptance": optimism_status,
        "boundary_acceptance": boundary_status,
        "overall_acceptance": overall,
        "blocker": ";".join(blockers),
        "hyperliquid_l2book_fast": str(watcher.get("hyperliquid_l2book_fast")).lower(),
        "reconnect_count": reconnect_count,
        "l2book_messages": l2book_messages,
        "trades_messages": trades_messages,
        "current_candidate_count": as_int(watcher.get("current_candidate_count")),
        "trigger_count": as_int(watcher.get("trigger_count")),
        "anti_drift_pass_count": as_int(watcher.get("anti_drift_pass_count")),
        "anti_drift_block_count": as_int(watcher.get("anti_drift_block_count")),
        "edge_gate_pass_count": as_int(watcher.get("edge_gate_pass_count")),
        "edge_gate_block_count": as_int(watcher.get("edge_gate_block_count")),
        "live_submissions_count": submissions,
        "max_real_order_submissions": max_submissions,
        "max_order_size_btc": max_order_size,
        "order_status_types": ",".join(statuses),
        "post_only_reject_count": post_only_rejects,
        "fill_count": fill_count,
        "maker_fill_count": maker_fill_count,
        "ledger_fill_rows": ledger_rows,
        "final_open_orders_count": final_open,
        "independent_final_open_orders_count": independent_open,
        "shutdown_proof_status": inline.get("shutdown_proof_status", ""),
        "real_order_endpoint_called": str(inline.get("real_order_endpoint_called")).lower(),
        "real_cancel_endpoint_called": str(inline.get("real_cancel_endpoint_called")).lower(),
        "scope_note": "same-window replay preserves observed classification and keeps no-fill economics fail-closed",
    }


def build_sha256_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append({"artifact": str(path.relative_to(output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256", "bytes"])


def artifact_nonempty_rows(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {"artifact": str(path.relative_to(output_dir)), "size_bytes": path.stat().st_size, "status": passfail(path.stat().st_size > 0)}
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]


def run_batch_acceptance(*, t011_root: Path, qa_0708t002: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [qa_0708_reference_row(qa_0708t002)]
    for idx in (1, 2, 3):
        rows.append(evaluate_window(t011_root / f"window_{idx:02d}", idx))

    write_csv(output_dir / "batch_replay_acceptance_matrix.csv", rows, MATRIX_FIELDS)
    aggregate = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "git_commit": git_commit(),
        "t011_artifact_root": str(t011_root),
        "prior_0708t002_qa_report": str(qa_0708t002),
        "window_count": len(rows),
        "source_kind_counts": status_counts(rows, "source_kind"),
        "classification_counts": status_counts(rows, "live_classification"),
        "overall_acceptance_counts": status_counts(rows, "overall_acceptance"),
        "market_view_acceptance_counts": status_counts(rows, "market_view_acceptance"),
        "decision_path_acceptance_counts": status_counts(rows, "decision_path_acceptance"),
        "lifecycle_acceptance_counts": status_counts(rows, "lifecycle_acceptance"),
        "economics_acceptance_counts": status_counts(rows, "economics_acceptance"),
        "optimism_acceptance_counts": status_counts(rows, "optimism_acceptance"),
        "boundary_acceptance_counts": status_counts(rows, "boundary_acceptance"),
        "accepted_window_ids": [row["window_id"] for row in rows if row["overall_acceptance"] == "pass"],
        "blocked_window_ids": [row["window_id"] for row in rows if row["overall_acceptance"] != "pass"],
        "final_recommendation": PASSED_RECOMMENDATION if all(row["overall_acceptance"] == "pass" for row in rows) else BLOCKED_RECOMMENDATION,
        "next_route_candidate": "T003_multi_window_robustness_synthesis" if all(row["overall_acceptance"] == "pass" for row in rows) else "replay_or_schema_repair",
        "no_profitability_claim": True,
        "no_maker_viability_claim": True,
    }
    write_json(output_dir / "aggregate_replay_acceptance_summary.json", aggregate)
    boundary = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "boundary_status": "pass",
        "offline_only": True,
        "network_called": False,
        "remote_called": False,
        "aws_called": False,
        "credentials_read": False,
        "private_endpoint_called": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "live_submit_executed": False,
        "market_data_collected": False,
        "threshold_changed": False,
        "quote_envelope_changed": False,
        "order_size_changed": False,
        "max_submissions_changed": False,
        "fill_probability_claim": False,
        "fee_rebate_claim": False,
        "realized_pnl_claim": False,
        "maker_viability_claim": False,
        "promotion_authorized": False,
        "t012_claim": False,
    }
    write_json(output_dir / "boundary_manifest.json", boundary)
    write_csv(output_dir / "artifact_nonempty_check.csv", artifact_nonempty_rows(output_dir), ["artifact", "size_bytes", "status"])
    report = [
        "# 0709T002 Batch Same-Window Replay Acceptance",
        "",
        f"Final recommendation: `{aggregate['final_recommendation']}`",
        "",
        f"- Matrix rows: `{len(rows)}`",
        f"- Classification counts: `{aggregate['classification_counts']}`",
        f"- Overall acceptance counts: `{aggregate['overall_acceptance_counts']}`",
        f"- Prior 0708T002 handling: `prior_accepted_replay_reference`",
        "",
        "This is offline-only replay acceptance. It preserves observed rejected/resting/no-fill lifecycle classifications and keeps unsupported economics fail-closed.",
        "",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    build_sha256_manifest(output_dir)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t011-root", type=Path, default=DEFAULT_T011_ROOT)
    parser.add_argument("--qa-0708t002", type=Path, default=DEFAULT_QA_0708T002)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_batch_acceptance(t011_root=args.t011_root, qa_0708t002=args.qa_0708t002, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
