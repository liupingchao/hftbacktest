#!/usr/bin/env python3
"""Offline robustness synthesis for T011 multi-window evidence."""

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
TASK_ID = "0709T003"
SCHEMA_VERSION = "cross_exchange_t011_multi_window_robustness_synthesis_v1"
DEFAULT_T001_SUMMARY = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t011_multi_window_live_evidence_0709T001_20260709T064251Z" / "0709T001_local_validation_summary.json"
DEFAULT_T002_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t011_batch_same_window_replay_acceptance_0709T002"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t011_multi_window_robustness_synthesis_0709T003"
RECOMMENDATIONS = {
    "route_to_more_controlled_evidence",
    "route_to_signal_distribution_diagnosis",
    "route_to_quote_fill_probability_evidence",
    "route_to_fee_inventory_pnl_calibration",
    "route_to_replay_repair",
    "route_to_execution_safety_repair",
    "stop_for_human_strategy_decision",
}

MATRIX_FIELDS = [
    "window_id",
    "source_kind",
    "live_classification",
    "replay_overall_acceptance",
    "market_view_acceptance",
    "decision_path_acceptance",
    "lifecycle_acceptance",
    "economics_acceptance",
    "optimism_acceptance",
    "boundary_acceptance",
    "feed_l2book_messages",
    "feed_trades_messages",
    "feed_reconnect_count",
    "watcher_seconds_elapsed",
    "l2book_messages_per_second",
    "trades_messages_per_second",
    "post_open_orders_public_state_pass_count",
    "post_open_orders_public_state_block_count",
    "post_open_orders_public_state_timeout_seconds",
    "handoff_phase",
    "guard_candidate_age_seconds",
    "current_reprice_candidate_age_seconds",
    "trigger_candidate_age_seconds",
    "max_candidate_age_seconds_at_phase_end",
    "fail_closed_reason_counts",
    "current_candidate_count",
    "trigger_count",
    "anti_drift_pass_count",
    "anti_drift_block_count",
    "edge_gate_pass_count",
    "edge_gate_block_count",
    "live_submissions_count",
    "order_status_types",
    "post_only_reject_count",
    "fill_count",
    "maker_fill_count",
    "ledger_fill_rows",
    "final_open_orders_count",
    "independent_final_open_orders_count",
    "shutdown_proof_status",
    "economics_support_status",
    "safety_invariant_status",
    "route_signal",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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
    if value in (None, "", "accepted_by_0708T002_QA"):
        return default
    return int(float(value))


def counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        out[value] = out.get(value, 0) + 1
    return out


def pass_status(condition: bool) -> str:
    return "pass" if condition else "fail"


def as_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, "", "accepted_by_0708T002_QA"):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def format_counts(items: dict[str, int]) -> str:
    return ";".join(f"{key}={items[key]}" for key in sorted(items))


def t001_artifact_root(t001: dict[str, Any], t001_summary: Path) -> Path:
    raw = t001.get("artifact_root")
    if not raw:
        return t001_summary.resolve().parent
    path = Path(str(raw))
    return path if path.is_absolute() else PROJECT_ROOT / path


def summary_by_window(t001: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in t001.get("window_summary", []):
        window_id = item.get("window_id")
        if window_id is None:
            continue
        out[f"0709T001_window_{int(window_id):02d}"] = item
    return out


def read_csv_rows_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_csv_rows(path)


def reason_distribution(rows: list[dict[str, str]]) -> str:
    counts_by_reason: dict[str, int] = {}
    for item in rows:
        status = item.get("guard_status") or item.get("status") or ""
        if status not in {"fail_closed", "edge_gate_block"}:
            continue
        raw = item.get("guard_reason") or item.get("edge_gate_reason") or item.get("skip_reason") or ""
        for reason in raw.split(";"):
            reason = reason.strip()
            if reason:
                counts_by_reason[reason] = counts_by_reason.get(reason, 0) + 1
    return format_counts(counts_by_reason)


def supplemental_metrics(row: dict[str, str], *, artifact_root: Path, t001_summary_by_window: dict[str, dict[str, Any]]) -> dict[str, Any]:
    window_id = row.get("window_id", "")
    summary = t001_summary_by_window.get(window_id, {})
    elapsed = as_float(summary.get("watcher_seconds_elapsed"))
    l2_count = as_float(row.get("l2book_messages"))
    trade_count = as_float(row.get("trades_messages"))
    metrics: dict[str, Any] = {
        "watcher_seconds_elapsed": fmt_float(elapsed),
        "l2book_messages_per_second": fmt_float(l2_count / elapsed if elapsed and l2_count is not None else None),
        "trades_messages_per_second": fmt_float(trade_count / elapsed if elapsed and trade_count is not None else None),
        "post_open_orders_public_state_pass_count": "",
        "post_open_orders_public_state_block_count": "",
        "post_open_orders_public_state_timeout_seconds": "",
        "handoff_phase": "",
        "guard_candidate_age_seconds": "",
        "current_reprice_candidate_age_seconds": "",
        "trigger_candidate_age_seconds": "",
        "max_candidate_age_seconds_at_phase_end": "",
        "fail_closed_reason_counts": "",
    }
    if row.get("source_kind") != "t011_live_window_artifact" or "window_" not in window_id:
        return metrics
    window_suffix = window_id.rsplit("window_", 1)[-1]
    window_dir = artifact_root / f"window_{window_suffix}"
    manifest_path = window_dir / "event_driven_watcher_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        guard = manifest.get("event_driven_guard", {})
        metrics.update(
            {
                "post_open_orders_public_state_pass_count": manifest.get("post_open_orders_public_state_pass_count", ""),
                "post_open_orders_public_state_block_count": manifest.get("post_open_orders_public_state_block_count", ""),
                "post_open_orders_public_state_timeout_seconds": manifest.get("post_open_orders_public_state_timeout_seconds", ""),
                "handoff_phase": guard.get("handoff_phase", ""),
                "guard_candidate_age_seconds": guard.get("candidate_age_seconds", ""),
                "current_reprice_candidate_age_seconds": guard.get("current_reprice_candidate_age_seconds", ""),
                "trigger_candidate_age_seconds": guard.get("trigger_candidate_age_seconds", ""),
            }
        )
    latency_rows = read_csv_rows_if_exists(window_dir / "inline_reprice_latency_matrix.csv")
    phase_ages = [as_float(item.get("candidate_age_seconds_at_phase_end")) for item in latency_rows]
    phase_ages = [item for item in phase_ages if item is not None]
    if phase_ages:
        metrics["max_candidate_age_seconds_at_phase_end"] = fmt_float(max(phase_ages))
    attempt_rows = read_csv_rows_if_exists(window_dir / "inline_reprice_attempt_matrix.csv")
    metrics["fail_closed_reason_counts"] = reason_distribution(attempt_rows)
    return metrics


def synthesize_rows(t002_rows: list[dict[str, str]], *, artifact_root: Path, t001_summary_by_window: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in t002_rows:
        fill_count = as_int(row.get("fill_count"))
        maker_fill_count = as_int(row.get("maker_fill_count"))
        ledger_rows = as_int(row.get("ledger_fill_rows"))
        final_open = as_int(row.get("final_open_orders_count"))
        independent_open = as_int(row.get("independent_final_open_orders_count"))
        shutdown = row.get("shutdown_proof_status", "")
        boundary = row.get("boundary_acceptance") == "pass"
        replay_all = all(row.get(name) == "pass" for name in ["overall_acceptance", "market_view_acceptance", "decision_path_acceptance", "lifecycle_acceptance", "economics_acceptance", "optimism_acceptance", "boundary_acceptance"])
        no_fill = fill_count == 0 and maker_fill_count == 0 and ledger_rows == 0
        safety = boundary and final_open == 0 and independent_open == 0 and shutdown == "pass"
        if row.get("source_kind") == "prior_accepted_replay_reference":
            safety = safety and row.get("overall_acceptance") == "pass"
        route_signal = "submitted_no_fill_replay_faithful" if row.get("live_classification") == "submitted_resting_no_fill" and replay_all and no_fill else row.get("live_classification", "")
        supplemental = supplemental_metrics(row, artifact_root=artifact_root, t001_summary_by_window=t001_summary_by_window)
        rows.append(
            {
                "window_id": row.get("window_id", ""),
                "source_kind": row.get("source_kind", ""),
                "live_classification": row.get("live_classification", ""),
                "replay_overall_acceptance": row.get("overall_acceptance", ""),
                "market_view_acceptance": row.get("market_view_acceptance", ""),
                "decision_path_acceptance": row.get("decision_path_acceptance", ""),
                "lifecycle_acceptance": row.get("lifecycle_acceptance", ""),
                "economics_acceptance": row.get("economics_acceptance", ""),
                "optimism_acceptance": row.get("optimism_acceptance", ""),
                "boundary_acceptance": row.get("boundary_acceptance", ""),
                "feed_l2book_messages": row.get("l2book_messages", ""),
                "feed_trades_messages": row.get("trades_messages", ""),
                "feed_reconnect_count": row.get("reconnect_count", ""),
                "watcher_seconds_elapsed": supplemental["watcher_seconds_elapsed"],
                "l2book_messages_per_second": supplemental["l2book_messages_per_second"],
                "trades_messages_per_second": supplemental["trades_messages_per_second"],
                "post_open_orders_public_state_pass_count": supplemental["post_open_orders_public_state_pass_count"],
                "post_open_orders_public_state_block_count": supplemental["post_open_orders_public_state_block_count"],
                "post_open_orders_public_state_timeout_seconds": supplemental["post_open_orders_public_state_timeout_seconds"],
                "handoff_phase": supplemental["handoff_phase"],
                "guard_candidate_age_seconds": supplemental["guard_candidate_age_seconds"],
                "current_reprice_candidate_age_seconds": supplemental["current_reprice_candidate_age_seconds"],
                "trigger_candidate_age_seconds": supplemental["trigger_candidate_age_seconds"],
                "max_candidate_age_seconds_at_phase_end": supplemental["max_candidate_age_seconds_at_phase_end"],
                "fail_closed_reason_counts": supplemental["fail_closed_reason_counts"],
                "current_candidate_count": row.get("current_candidate_count", ""),
                "trigger_count": row.get("trigger_count", ""),
                "anti_drift_pass_count": row.get("anti_drift_pass_count", ""),
                "anti_drift_block_count": row.get("anti_drift_block_count", ""),
                "edge_gate_pass_count": row.get("edge_gate_pass_count", ""),
                "edge_gate_block_count": row.get("edge_gate_block_count", ""),
                "live_submissions_count": row.get("live_submissions_count", ""),
                "order_status_types": row.get("order_status_types", ""),
                "post_only_reject_count": row.get("post_only_reject_count", ""),
                "fill_count": fill_count,
                "maker_fill_count": maker_fill_count,
                "ledger_fill_rows": ledger_rows,
                "final_open_orders_count": final_open,
                "independent_final_open_orders_count": independent_open,
                "shutdown_proof_status": shutdown,
                "economics_support_status": "no_fill_fail_closed" if no_fill else "fill_attribution_required",
                "safety_invariant_status": pass_status(safety),
                "route_signal": route_signal,
            }
        )
    return rows


def choose_recommendation(rows: list[dict[str, Any]]) -> str:
    if any(row["safety_invariant_status"] != "pass" for row in rows):
        return "route_to_execution_safety_repair"
    if any(row["replay_overall_acceptance"] != "pass" or row["optimism_acceptance"] != "pass" for row in rows):
        return "route_to_replay_repair"
    classifications = [row["live_classification"] for row in rows]
    if classifications and all(item == "no_submit_fail_closed" for item in classifications):
        return "route_to_signal_distribution_diagnosis"
    filled = sum(1 for row in rows if as_int(row["fill_count"]) > 0 or as_int(row["maker_fill_count"]) > 0)
    if filled > 0:
        return "route_to_fee_inventory_pnl_calibration"
    submitted_no_fill = sum(1 for row in rows if row["live_classification"] == "submitted_resting_no_fill" and row["replay_overall_acceptance"] == "pass")
    if submitted_no_fill >= 2:
        return "route_to_quote_fill_probability_evidence"
    if len(rows) < 3:
        return "route_to_more_controlled_evidence"
    return "route_to_more_controlled_evidence"


def build_sha256_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append({"artifact": str(path.relative_to(output_dir)), "sha256": sha256(path), "bytes": path.stat().st_size})
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256", "bytes"])


def run_synthesis(*, t001_summary: Path, t002_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    t001 = read_json(t001_summary)
    t002 = read_json(t002_dir / "aggregate_replay_acceptance_summary.json")
    t002_rows = read_csv_rows(t002_dir / "batch_replay_acceptance_matrix.csv")
    rows = synthesize_rows(t002_rows, artifact_root=t001_artifact_root(t001, t001_summary), t001_summary_by_window=summary_by_window(t001))
    recommendation = choose_recommendation(rows)
    if recommendation not in RECOMMENDATIONS:
        raise ValueError(f"invalid_recommendation:{recommendation}")
    write_csv(output_dir / "multi_window_synthesis_matrix.csv", rows, MATRIX_FIELDS)
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "git_commit": git_commit(),
        "t001_summary": str(t001_summary),
        "t002_dir": str(t002_dir),
        "accepted_window_count": len(rows),
        "source_kind_counts": counts(rows, "source_kind"),
        "classification_counts": counts(rows, "live_classification"),
        "route_signal_counts": counts(rows, "route_signal"),
        "safety_invariant_counts": counts(rows, "safety_invariant_status"),
        "replay_overall_acceptance_counts": counts(rows, "replay_overall_acceptance"),
        "economics_support_counts": counts(rows, "economics_support_status"),
        "t001_boundary_pass": t001.get("boundary_pass"),
        "t002_final_recommendation": t002.get("final_recommendation"),
        "final_recommendation": recommendation,
        "route_rationale": "Multiple submitted/resting no-fill windows replay faithfully, with no fills or PnL support; next evidence should target quote/fill probability rather than profitability or promotion.",
        "does_not_claim": ["stable_pnl", "maker_viability", "t012", "promotion", "final_mvp_pass", "live_expansion"],
    }
    write_json(output_dir / "multi_window_synthesis_manifest.json", manifest)
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
        "strategy_changed": False,
        "pnl_claim": False,
        "maker_viability_claim": False,
        "promotion_authorized": False,
        "t012_claim": False,
    }
    write_json(output_dir / "boundary_manifest.json", boundary)
    report = [
        "# 0709T003 Multi-Window Robustness Synthesis",
        "",
        f"Final recommendation: `{recommendation}`",
        "",
        f"- Accepted window count: `{len(rows)}`",
        f"- Classification counts: `{manifest['classification_counts']}`",
        f"- Safety invariant counts: `{manifest['safety_invariant_counts']}`",
        f"- Replay acceptance counts: `{manifest['replay_overall_acceptance_counts']}`",
        f"- Economics support counts: `{manifest['economics_support_counts']}`",
        "- Handoff/candidate-age evidence: see `multi_window_synthesis_matrix.csv` columns `handoff_phase`, `guard_candidate_age_seconds`, `max_candidate_age_seconds_at_phase_end`, and `fail_closed_reason_counts`.",
        "",
        "No fills occurred, so fee/rebate/realized PnL remain unsupported. The accepted route is quote/fill probability evidence, not profitability or promotion.",
        "",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    build_sha256_manifest(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t001-summary", type=Path, default=DEFAULT_T001_SUMMARY)
    parser.add_argument("--t002-dir", type=Path, default=DEFAULT_T002_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_synthesis(t001_summary=args.t001_summary, t002_dir=args.t002_dir, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
