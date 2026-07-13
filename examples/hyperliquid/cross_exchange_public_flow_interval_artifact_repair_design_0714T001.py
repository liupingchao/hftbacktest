#!/usr/bin/env python3
"""Offline public-flow interval artifact repair/design for 0714T001.

This runner consumes accepted 0713T003 local artifacts only. It produces a
design package for the next repair task and does not call live, remote,
credential, private, account, order, cancel, or market-data APIs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0714T001"
SOURCE_TASK_ID = "0713T003"
SCHEMA_VERSION = "cross_exchange_public_flow_interval_artifact_repair_design_0714T001_v1"
CONTRACT_VERSION = "cross_exchange_resting_interval_public_flow_capture_contract_v2"
FINAL_ROUTE = "route_to_resting_interval_capture_contract_repair"
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_quote_fill_probability_evidence_0713T003"
DEFAULT_QA_REPORT = PROJECT_ROOT / ".workflow" / "reports" / "0713T003-qa.md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_public_flow_interval_artifact_repair_design_0714T001"

GAP_FIELDS = [
    "gap_id",
    "window_id",
    "evaluation_id",
    "order_attempt_id",
    "gap_scope",
    "current_status",
    "current_source",
    "why_gap_matters",
    "required_future_artifact_field",
    "blocks_route",
    "supported_interpretation",
    "unsupported_interpretation",
]

INSTRUMENTATION_FIELDS = [
    "component",
    "artifact",
    "capture_timing",
    "required_fields",
    "required_status_fields",
    "acceptance_check",
    "failure_mode_if_missing",
]

GATE_FIELDS = [
    "gate_id",
    "gate_name",
    "current_status",
    "required_evidence",
    "result",
    "blocks_route",
    "reason",
]


def git_commit() -> str:
    try:
        runner_path = Path(__file__).resolve().relative_to(PROJECT_ROOT)
        value = subprocess.run(
            ["git", "log", "-n", "1", "--format=%h", "--", str(runner_path)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if value:
            return value
    except Exception:
        pass
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


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


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


def build_sha256_manifest(output_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append({"artifact": str(path.relative_to(output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256", "bytes"])


def counts(rows: list[dict[str, str]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "")
        out[value] = out.get(value, 0) + 1
    return out


def public_trade_zero_interpretation(capture_status: str, trade_through_status: str) -> str:
    complete_statuses = {
        "complete_interval_coverage_zero_trades",
        "exact_interval_public_flow_coverage_zero_trades",
        "complete_attempt_keyed_interval_public_trade_coverage_zero_trades",
    }
    if capture_status in complete_statuses:
        return "zero_public_trades_observed_with_complete_interval_coverage"
    if trade_through_status in {"exact_interval_public_trades_present_no_trade_through", "complete_interval_no_trade_through"}:
        return "no_trade_through_observed_with_complete_interval_coverage"
    return "artifact_gap_not_no_exchange_trades"


def source_files(source_dir: Path) -> dict[str, str]:
    return {
        "manifest": display_path(source_dir / "quote_fill_probability_manifest.json"),
        "final_route": display_path(source_dir / "final_route.json"),
        "attempt_level_matrix": display_path(source_dir / "attempt_level_quote_fill_evidence_matrix.csv"),
        "public_trades_depletion_summary": display_path(source_dir / "resting_interval_public_trades_depletion_summary.csv"),
        "censoring_horizon_matrix": display_path(source_dir / "censoring_horizon_matrix.csv"),
        "same_side_depth_proxy_matrix": display_path(source_dir / "same_side_depth_proxy_matrix.csv"),
        "boundary_manifest": display_path(source_dir / "boundary_manifest.json"),
        "input_source_manifest": display_path(source_dir / "input_source_manifest.json"),
    }


def find_resting_attempts(attempt_rows: list[dict[str, str]], public_flow_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    public_by_key = {
        (row.get("window_id", ""), row.get("evaluation_id", ""), row.get("order_attempt_id", "")): row
        for row in public_flow_rows
    }
    out: list[dict[str, str]] = []
    for row in attempt_rows:
        if row.get("order_status_type") != "resting":
            continue
        key = (row.get("window_id", ""), row.get("evaluation_id", ""), row.get("order_attempt_id", ""))
        merged = dict(row)
        merged.update({f"public_{key}": value for key, value in public_by_key.get(key, {}).items()})
        out.append(merged)
    return out


def gap_rows(resting_attempts: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for attempt in resting_attempts:
        base = {
            "window_id": attempt.get("window_id", ""),
            "evaluation_id": attempt.get("evaluation_id", ""),
            "order_attempt_id": attempt.get("order_attempt_id", ""),
        }
        lifecycle = attempt.get("public_lifecycle_status", attempt.get("exact_timestamp_caveat", ""))
        capture = attempt.get("public_capture_status", "")
        depth = attempt.get("public_depth_status", attempt.get("depth_proxy_status", ""))
        trade_status = attempt.get("trade_through_status", "")
        zero_interpretation = public_trade_zero_interpretation(capture, trade_status)
        rows.extend(
            [
                {
                    **base,
                    "gap_id": "G01",
                    "gap_scope": "resting_interval_start_end",
                    "current_status": lifecycle or "proxy_or_missing",
                    "current_source": "0713T003 resting_interval_public_trades_depletion_summary.csv",
                    "why_gap_matters": "Fill/depletion evidence needs a bounded exchange-side interval, not only a local/proxy lifecycle label.",
                    "required_future_artifact_field": "order_resting_exchange_time_ms, order_resting_local_receive_ts_ns, cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms",
                    "blocks_route": "quote_fill_probability, quote_policy_design, fee_inventory_pnl",
                    "supported_interpretation": "current interval is proxy-censored",
                    "unsupported_interpretation": "exact exchange resting interval",
                },
                {
                    **base,
                    "gap_id": "G02",
                    "gap_scope": "attempt_keyed_public_trades",
                    "current_status": capture or "missing",
                    "current_source": "0713T003 resting_interval_public_trades_depletion_summary.csv",
                    "why_gap_matters": "Zero matching artifact rows is not proof that the exchange had zero public trades during the resting interval.",
                    "required_future_artifact_field": "resting_interval_public_trades.csv: attempt_key, exchange_time_ms, local_receive_ts_ns, px, size_btc, side/aggressor, source_sequence",
                    "blocks_route": "quote_fill_probability, quote_policy_design",
                    "supported_interpretation": zero_interpretation,
                    "unsupported_interpretation": "no_exchange_public_trades_occurred",
                },
                {
                    **base,
                    "gap_id": "G03",
                    "gap_scope": "resting_start_depth",
                    "current_status": depth or "proxy_or_missing",
                    "current_source": "0713T003 same_side_depth_proxy_matrix.csv",
                    "why_gap_matters": "Queue/depletion reasoning needs depth at or after the order is resting, not an inline reprice/pre-submit proxy.",
                    "required_future_artifact_field": "resting_start_l2_snapshot.csv: attempt_key, exchange_time_ms, local_receive_ts_ns, side, quote_px, same_side_levels_at_or_ahead_of_quote",
                    "blocks_route": "queue_depletion, quote_policy_design",
                    "supported_interpretation": "depth proxy exists but is not exact resting-start depth",
                    "unsupported_interpretation": "queue position or queue priority",
                },
                {
                    **base,
                    "gap_id": "G04",
                    "gap_scope": "interval_coverage_proof",
                    "current_status": "coverage_not_proven_complete",
                    "current_source": "0713T003 source_validation_summary + public-flow summary",
                    "why_gap_matters": "A future zero-trade interval is useful only if stream coverage spans the full resting interval with bounded gaps.",
                    "required_future_artifact_field": "public_stream_coverage.csv: attempt_key, stream, start_cursor, end_cursor, first_event_ms, last_event_ms, gap_count, coverage_status",
                    "blocks_route": "no_trade_claim, fill_probability",
                    "supported_interpretation": "public-flow observability is incomplete or unproven",
                    "unsupported_interpretation": "complete interval observation",
                },
                {
                    **base,
                    "gap_id": "G05",
                    "gap_scope": "short_horizon_censoring",
                    "current_status": attempt.get("censoring_status", ""),
                    "current_source": "0713T003 censoring_horizon_matrix.csv",
                    "why_gap_matters": "The accepted resting sample lasted only a short horizon; non-fill over this horizon cannot identify steady-state fill probability.",
                    "required_future_artifact_field": "hold_elapsed_seconds, cancel_reason, observation_horizon_status, right_censoring_status",
                    "blocks_route": "fill_probability_estimate, maker_viability",
                    "supported_interpretation": "short_hold_censored_no_fill",
                    "unsupported_interpretation": "low_fill_probability",
                },
            ]
        )
    return rows


def instrumentation_rows() -> list[dict[str, str]]:
    return [
        {
            "component": "order_lifecycle",
            "artifact": "resting_interval_lifecycle.csv",
            "capture_timing": "on order response, first open-order/resting confirmation, cancel request, cancel ack, shutdown proof",
            "required_fields": "attempt_key, cloid_or_order_ref_redacted, side, limit_px, size_btc, order_resting_exchange_time_ms, order_resting_local_receive_ts_ns, cancel_request_local_ts_ns, cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms",
            "required_status_fields": "resting_start_source_status, interval_end_source_status, lifecycle_completeness_status",
            "acceptance_check": "each submitted/resting attempt has one bounded interval row or explicit fail-closed missing reason",
            "failure_mode_if_missing": "cannot define actual interval for public-flow reconstruction",
        },
        {
            "component": "public_trades",
            "artifact": "resting_interval_public_trades.csv",
            "capture_timing": "continuously during [resting_start, cancel_ack_or_shutdown]",
            "required_fields": "attempt_key, exchange_time_ms, local_receive_ts_ns, source_sequence, px, size_btc, side_or_aggressor, at_quote, through_quote",
            "required_status_fields": "interval_trade_capture_status, public_stream_gap_count, public_stream_coverage_status",
            "acceptance_check": "zero rows are accepted only when public_stream_coverage_status proves full interval coverage",
            "failure_mode_if_missing": "zero captured rows remains artifact gap, not no-trade evidence",
        },
        {
            "component": "resting_depth",
            "artifact": "resting_interval_l2_snapshots.csv",
            "capture_timing": "first L2 snapshot at or after resting confirmation, plus optional pre-cancel snapshot",
            "required_fields": "attempt_key, snapshot_role, exchange_time_ms, local_receive_ts_ns, best_bid, best_ask, quote_px, same_side_visible_qty_at_or_ahead_of_quote_btc, same_side_visible_order_count_at_or_ahead_of_quote",
            "required_status_fields": "snapshot_source_status, quote_in_book_status, depth_reconstruction_status",
            "acceptance_check": "depth row timestamp must be bounded at or after resting start and before interval end",
            "failure_mode_if_missing": "cannot reason about visible depletion or queue proxy",
        },
        {
            "component": "coverage_index",
            "artifact": "public_stream_coverage.csv",
            "capture_timing": "before interval start through after interval end",
            "required_fields": "attempt_key, stream, start_cursor, end_cursor, first_event_exchange_time_ms, last_event_exchange_time_ms, local_receive_min_ns, local_receive_max_ns",
            "required_status_fields": "coverage_status, reconnect_count, gap_count, clock_skew_status",
            "acceptance_check": "complete or explicitly bounded coverage required before no-trade/no-through claims",
            "failure_mode_if_missing": "cannot distinguish missing capture from quiet market interval",
        },
    ]


def acceptance_gate_rows(manifest: dict[str, Any], gap_count: int) -> list[dict[str, str]]:
    source_route = manifest.get("final_route", "")
    return [
        {
            "gate_id": "A01",
            "gate_name": "accepted_source_route",
            "current_status": source_route,
            "required_evidence": "0713T003 QA accepted route_to_public_flow_artifact_repair",
            "result": "pass" if source_route == "route_to_public_flow_artifact_repair" else "fail",
            "blocks_route": "",
            "reason": "Source package is the accepted artifact-repair input.",
        },
        {
            "gate_id": "A02",
            "gate_name": "artifact_gap_explicit",
            "current_status": f"{gap_count} gaps emitted",
            "required_evidence": "gaps must identify proxy/missing interval fields without inventing fill probability",
            "result": "pass" if gap_count else "fail",
            "blocks_route": "controlled_live_evidence",
            "reason": "Next work needs the exact capture contract before a controlled rerun.",
        },
        {
            "gate_id": "A03",
            "gate_name": "quote_policy_design",
            "current_status": "blocked",
            "required_evidence": "attempt-keyed interval trades, exact/bounded lifecycle, and resting-start depth",
            "result": "fail",
            "blocks_route": "quote_policy_design",
            "reason": "Current 0713T003 evidence has one short no-fill interval with zero matching trade rows and proxy depth/lifecycle.",
        },
        {
            "gate_id": "A04",
            "gate_name": "fill_probability_or_maker_viability",
            "current_status": "blocked",
            "required_evidence": "multiple accepted resting intervals with complete interval public-flow/depth and censoring model",
            "result": "fail",
            "blocks_route": "fill_probability, maker_viability, T012, final_mvp",
            "reason": "A single short no-fill sample with artifact gaps cannot estimate fill probability.",
        },
        {
            "gate_id": "A05",
            "gate_name": "next_task_route",
            "current_status": FINAL_ROUTE,
            "required_evidence": "capture contract repair still needed before separately authorized live evidence",
            "result": "pass",
            "blocks_route": "live_retry_until_repair_done",
            "reason": "The correct next step is code/schema instrumentation repair, not threshold tuning or live retry.",
        },
    ]


def contract_payload() -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "purpose": "Make each submitted/resting maker attempt reconstructable from exchange-side resting acknowledgement through cancel/shutdown, with explicit public-flow observability status.",
        "attempt_identity": {
            "required_fields": ["window_id", "evaluation_id", "order_attempt_id", "attempt_key", "side", "limit_px", "size_btc", "post_only_tif"],
            "invariant": "all lifecycle, public-trade, coverage, and depth rows must join on the same attempt_key",
        },
        "lifecycle": {
            "required_fields": [
                "order_resting_exchange_time_ms",
                "order_resting_local_receive_ts_ns",
                "cancel_request_local_ts_ns",
                "cancel_ack_exchange_time_ms_or_shutdown_proof_time_ms",
                "interval_start_source_status",
                "interval_end_source_status",
            ],
            "required_statuses": ["exact_exchange_resting_timestamp", "bounded_exchange_resting_timestamp", "exact_cancel_ack_or_shutdown", "missing_fail_closed"],
        },
        "public_flow": {
            "required_trade_fields": ["attempt_key", "exchange_time_ms", "local_receive_ts_ns", "source_sequence", "px", "size_btc", "side_or_aggressor", "at_quote", "through_quote"],
            "required_coverage_fields": ["attempt_key", "stream", "start_cursor", "end_cursor", "first_event_exchange_time_ms", "last_event_exchange_time_ms", "gap_count", "coverage_status"],
            "zero_row_rule": "A zero-row interval means no captured rows only unless coverage_status proves complete interval coverage.",
        },
        "depth": {
            "required_fields": [
                "attempt_key",
                "snapshot_role",
                "exchange_time_ms",
                "local_receive_ts_ns",
                "best_bid",
                "best_ask",
                "quote_px",
                "same_side_visible_qty_at_or_ahead_of_quote_btc",
                "same_side_visible_order_count_at_or_ahead_of_quote",
                "depth_reconstruction_status",
            ],
            "required_statuses": ["exact_resting_start_l2_depth", "bounded_resting_start_l2_depth", "missing_fail_closed"],
        },
        "blocked_until_contract_satisfied": [
            "quote_fill_probability_estimate",
            "quote_policy_design",
            "queue_priority_claim",
            "fee_inventory_pnl_calibration",
            "maker_viability",
            "T012",
            "final_mvp_pass",
        ],
    }


def boundary_manifest(source_dir: Path) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_dir": display_path(source_dir),
        "offline_only": True,
        "local_source_only": True,
        "live_submit": False,
        "live_retry": False,
        "remote_or_aws_execution": False,
        "credential_reads": False,
        "private_account_order_cancel_endpoints": False,
        "new_market_data_collection": False,
        "threshold_change": False,
        "quote_envelope_change": False,
        "order_size_or_max_submission_change": False,
        "strategy_behavior_change": False,
        "quote_policy_design": False,
        "fill_probability_claim": False,
        "queue_priority_claim": False,
        "fee_rebate_or_realized_pnl_claim": False,
        "maker_viability_claim": False,
        "t012_or_final_mvp_claim": False,
    }


def write_validation_report(output_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# 0714T001 Public-Flow Interval Artifact Repair/Design",
        "",
        f"- final route: `{manifest['final_route']}`",
        f"- source task: `{manifest['source_task_id']}`",
        f"- source route: `{manifest['source_final_route']}`",
        f"- resting attempts: `{manifest['resting_attempt_count']}`",
        f"- artifact gaps: `{manifest['artifact_gap_count']}`",
        "",
        "## Conclusion",
        "",
        "The accepted 0713T003 evidence can support only an artifact-repair route. The next work should implement the capture contract before any controlled live rerun.",
        "",
        "Zero matching attempt-keyed interval public-trade rows remain an artifact observability gap unless future coverage metadata proves the entire resting interval was observed.",
        "",
        "## Unsupported",
        "",
        "- fill probability estimate",
        "- quote policy design",
        "- queue priority",
        "- fee/rebate or realized PnL",
        "- maker viability, T012, promotion, or final MVP pass",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(source_dir: Path = DEFAULT_SOURCE_DIR, qa_report: Path = DEFAULT_QA_REPORT, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    manifest = read_json(source_dir / "quote_fill_probability_manifest.json")
    attempt_rows = read_csv_rows(source_dir / "attempt_level_quote_fill_evidence_matrix.csv")
    public_rows = read_csv_rows(source_dir / "resting_interval_public_trades_depletion_summary.csv")
    resting_attempts = find_resting_attempts(attempt_rows, public_rows)
    gaps = gap_rows(resting_attempts)
    instrumentation = instrumentation_rows()
    gates = acceptance_gate_rows(manifest, len(gaps))
    output_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_task_id": SOURCE_TASK_ID,
        "source_dir": display_path(source_dir),
        "source_files": source_files(source_dir),
        "source_qa_report": display_path(qa_report),
        "source_final_route": manifest.get("final_route"),
        "source_attempt_count": manifest.get("attempt_count"),
        "source_resting_attempt_count": manifest.get("resting_attempt_count"),
        "source_public_trade_summary_row_count": manifest.get("public_trade_summary_row_count"),
        "source_supported_no_fill_reasons": manifest.get("supported_no_fill_reasons", []),
        "source_unsupported_claims": manifest.get("unsupported_claims", []),
        "source_qa_report_present": qa_report.exists(),
    }
    write_json(output_dir / "source_package_manifest.json", source_manifest)
    write_json(output_dir / "required_artifact_contract.json", contract_payload())
    write_json(output_dir / "boundary_manifest.json", boundary_manifest(source_dir))
    write_json(
        output_dir / "final_route.json",
        {
            "task_id": TASK_ID,
            "final_route": FINAL_ROUTE,
            "route_rationale": "0713T003 accepted an artifact-repair route; code/schema capture contract repair is still needed before live retry or quote policy design.",
        },
    )
    write_csv(output_dir / "artifact_gap_matrix.csv", gaps, GAP_FIELDS)
    write_csv(output_dir / "instrumentation_design_matrix.csv", instrumentation, INSTRUMENTATION_FIELDS)
    write_csv(output_dir / "acceptance_gate_matrix.csv", gates, GATE_FIELDS)

    result_manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "git_commit": git_commit(),
        "source_task_id": SOURCE_TASK_ID,
        "source_final_route": manifest.get("final_route"),
        "source_attempt_count": len(attempt_rows),
        "source_order_status_counts": counts(attempt_rows, "order_status_type"),
        "resting_attempt_count": len(resting_attempts),
        "public_trade_summary_row_count": len(public_rows),
        "artifact_gap_count": len(gaps),
        "instrumentation_design_row_count": len(instrumentation),
        "acceptance_gate_row_count": len(gates),
        "final_route": FINAL_ROUTE,
        "next_task_recommendation": "implement_resting_interval_capture_contract_repair_before_any_controlled_live_evidence_rerun",
        "zero_public_trade_interpretation": [
            public_trade_zero_interpretation(row.get("public_capture_status", ""), row.get("trade_through_status", ""))
            for row in resting_attempts
        ],
        "unsupported_claims": [
            "fill_probability_estimate",
            "quote_policy_design",
            "queue_priority",
            "fee_rebate_or_realized_pnl",
            "maker_viability",
            "T012",
            "final_mvp_pass",
        ],
        "output_dir": display_path(output_dir),
        "output_files": {
            "artifact_gap_matrix": display_path(output_dir / "artifact_gap_matrix.csv"),
            "required_artifact_contract": display_path(output_dir / "required_artifact_contract.json"),
            "instrumentation_design_matrix": display_path(output_dir / "instrumentation_design_matrix.csv"),
            "acceptance_gate_matrix": display_path(output_dir / "acceptance_gate_matrix.csv"),
            "source_package_manifest": display_path(output_dir / "source_package_manifest.json"),
            "boundary_manifest": display_path(output_dir / "boundary_manifest.json"),
            "final_route": display_path(output_dir / "final_route.json"),
            "validation_report": display_path(output_dir / "validation_report.md"),
        },
    }
    write_json(output_dir / "repair_design_manifest.json", result_manifest)
    write_validation_report(output_dir, result_manifest)
    build_sha256_manifest(output_dir)
    return result_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--qa-report", type=Path, default=DEFAULT_QA_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_analysis(source_dir=args.source_dir, qa_report=args.qa_report, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
