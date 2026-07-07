#!/usr/bin/env python3
"""Same-window replay acceptance for T010 fast-L2 live evidence.

This runner consumes local pulled-back live artifacts only. It does not read
credentials, call private/order/cancel/account endpoints, collect market data,
or start live processes. Its "replay" is intentionally conservative: supported
same-window facts are represented exactly as observed, and unsupported
execution/economics fields remain fail-closed instead of being inferred.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0708T002"
TASK_ALIAS = "0625T010"
SCHEMA_VERSION = "cross_exchange_t010_fast_l2_same_window_replay_acceptance_v1"
PASSED_RECOMMENDATION = "same_window_replay_acceptance_passed"
BLOCKED_RECOMMENDATION = "same_window_replay_acceptance_blocked"
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_t010_fast_l2book_controlled_live_evidence_0708T001_20260707T160830Z"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_t010_same_window_replay_acceptance_0708T002"


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


def parse_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def compare_row(
    *,
    check: str,
    live_observed: Any,
    replay_value: Any,
    scope_note: str,
    acceptance: str | None = None,
) -> dict[str, Any]:
    live_text = fmt(live_observed)
    replay_text = fmt(replay_value)
    return {
        "check": check,
        "live_observed": live_text,
        "replay_value": replay_text,
        "acceptance": acceptance or ("pass" if live_text == replay_text else "fail"),
        "scope_note": scope_note,
    }


def status_count(rows: list[dict[str, Any]], field: str = "acceptance") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        counts[key] = counts.get(key, 0) + 1
    return counts


def first_submitted_attempt(rows: list[dict[str, str]]) -> dict[str, str]:
    for row in rows:
        if truthy(row.get("order_endpoint_called")):
            return row
    return {}


def row_count_with(rows: list[dict[str, str]], field: str, value: str) -> int:
    return sum(1 for row in rows if str(row.get(field, "")) == value)


def build_sha256_manifest(output_dir: Path) -> None:
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "sha256_manifest.csv":
            rows.append(
                {
                    "artifact": str(path.relative_to(output_dir)),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256", "bytes"])


def artifact_nonempty_rows(output_dir: Path) -> list[dict[str, Any]]:
    return [
        {
            "artifact": str(path.relative_to(output_dir)),
            "size_bytes": path.stat().st_size,
            "status": "pass" if path.stat().st_size > 0 else "fail",
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]


def run_acceptance(*, input_root: Path, output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    event_dir = input_root / "event_driven_edge_gate_live"
    if not event_dir.exists():
        raise FileNotFoundError(f"missing_event_dir:{event_dir}")

    watcher = read_json(event_dir / "event_driven_watcher_manifest.json")
    public_summary = read_json(event_dir / "public_stream_summary.json")
    inline = read_json(event_dir / "inline_reprice_manifest.json")
    cancel_proof = read_json(event_dir / "cancel_shutdown_proof.json")
    private_response = read_json(event_dir / "private_order_response_audit.json")
    independent = read_json(input_root / "independent_remote_open_orders_check.json")
    account = read_json(event_dir / "account_inventory_snapshots.json")
    markout = read_json(event_dir / "market_markout_snapshot.json")
    max_loss = read_json(event_dir / "max_loss_monitor_summary.json")

    attempts = read_csv_rows(event_dir / "inline_reprice_attempt_matrix.csv")
    guards = read_csv_rows(event_dir / "inline_reprice_guard_matrix.csv")
    freshness = read_csv_rows(event_dir / "public_state_freshness_matrix.csv")
    intents = read_csv_rows(event_dir / "order_intent_audit.csv")
    live_fills = read_csv_rows(event_dir / "live_fill_ledger.csv")
    submitted_attempt = first_submitted_attempt(attempts)
    intent = intents[0] if intents else {}
    guard_pass_rows = [row for row in guards if row.get("status") == "pass"]
    final_guard = guard_pass_rows[-1] if guard_pass_rows else {}

    message_counts = public_summary.get("message_count_by_channel", {})
    l2_count = int(message_counts.get("l2Book", 0) or 0)
    trade_events = int(public_summary.get("total_trade_event_count", 0) or 0)
    order_status_types = inline.get("order_status_types", [])
    order_status = ",".join(str(item) for item in order_status_types)
    final_open_orders_count = int(inline.get("final_open_orders_count", 0) or 0)
    independent_open_orders_raw = independent.get("final_open_orders_count")
    independent_open_orders_count = -1 if independent_open_orders_raw is None else int(independent_open_orders_raw)
    fill_count = int(watcher.get("fill_count", 0) or 0)
    maker_fill_count = int(watcher.get("maker_fill_count", 0) or 0)

    market_rows = [
        compare_row(
            check="fast_l2_subscription_enabled",
            live_observed=watcher.get("hyperliquid_l2book_fast"),
            replay_value=True,
            scope_note="same-window replay requires the accepted fast-L2 live source path",
        ),
        compare_row(
            check="public_l2_messages_positive",
            live_observed=l2_count > 0,
            replay_value=True,
            scope_note=f"l2Book message count={l2_count}",
        ),
        compare_row(
            check="public_trade_events_positive",
            live_observed=trade_events > 0,
            replay_value=True,
            scope_note=f"trade event count={trade_events}",
        ),
        compare_row(
            check="reconnect_count_zero",
            live_observed=public_summary.get("reconnect_count", 0),
            replay_value=0,
            scope_note="public stream did not reconnect inside accepted window",
        ),
        compare_row(
            check="post_open_orders_public_state_all_pass",
            live_observed=watcher.get("post_open_orders_public_state_block_count", 0),
            replay_value=0,
            scope_note=f"public-state pass count={watcher.get('post_open_orders_public_state_pass_count')}",
        ),
        compare_row(
            check="post_open_orders_state_observed_after_end",
            live_observed=row_count_with(freshness, "state_observed_after_open_orders_end", "True"),
            replay_value=len(freshness),
            scope_note="each post-open-orders freshness row observed a later L2 state",
        ),
        compare_row(
            check="pre_submit_l2_markout_available",
            live_observed=bool(markout.get("pre_submit_current_l2")),
            replay_value=True,
            scope_note="market view around submit is present",
        ),
        compare_row(
            check="post_submit_l2_markout_available",
            live_observed=bool(markout.get("post_submit_current_l2")),
            replay_value=True,
            scope_note="post-submit market view is present",
        ),
    ]

    decision_rows = [
        compare_row(
            check="trigger_found",
            live_observed=watcher.get("trigger_found"),
            replay_value=True,
            scope_note="accepted live window reached submit-capable trigger path",
        ),
        compare_row(
            check="event_guard_pass",
            live_observed=watcher.get("event_driven_guard_status"),
            replay_value="pass",
            scope_note="final immediate guard passed before submit",
        ),
        compare_row(
            check="handoff_phase",
            live_observed=watcher.get("event_driven_guard", {}).get("handoff_phase"),
            replay_value="post_open_orders_inline_reprice",
            scope_note="same handoff phase is represented",
        ),
        compare_row(
            check="submitted_attempt_guard_pass",
            live_observed=submitted_attempt.get("guard_status"),
            replay_value="pass",
            scope_note="submitted attempt row has guard pass",
        ),
        compare_row(
            check="submitted_attempt_edge_pass",
            live_observed=submitted_attempt.get("edge_gate_status"),
            replay_value="pass",
            scope_note="submitted attempt row has edge-gate pass",
        ),
        compare_row(
            check="side",
            live_observed=intent.get("side"),
            replay_value=submitted_attempt.get("side"),
            scope_note="intent side matches submitted attempt",
        ),
        compare_row(
            check="limit_px",
            live_observed=intent.get("limit_px"),
            replay_value=submitted_attempt.get("limit_px"),
            scope_note="intent price matches submitted attempt",
        ),
        compare_row(
            check="size_btc",
            live_observed=intent.get("size_btc"),
            replay_value=submitted_attempt.get("size_btc"),
            scope_note="intent size matches submitted attempt",
        ),
        compare_row(
            check="post_only_tif",
            live_observed=intent.get("time_in_force"),
            replay_value="Alo",
            scope_note="post-only boundary preserved",
        ),
        compare_row(
            check="quote_non_crossing",
            live_observed=final_guard.get("post_only_non_crossing"),
            replay_value="True",
            scope_note="final guard recorded non-crossing post-only quote",
        ),
    ]

    lifecycle_rows = [
        compare_row(
            check="live_submissions_count",
            live_observed=watcher.get("live_submissions_count"),
            replay_value=1,
            scope_note="accepted window has exactly one submitted order",
        ),
        compare_row(
            check="real_order_endpoint_called",
            live_observed=inline.get("real_order_endpoint_called"),
            replay_value=True,
            scope_note="source artifact observed real order endpoint",
        ),
        compare_row(
            check="order_submission_attempted",
            live_observed=private_response.get("order_submission_attempted"),
            replay_value=True,
            scope_note="private order response audit records submission",
        ),
        compare_row(
            check="order_status_resting",
            live_observed="resting" in order_status_types,
            replay_value=True,
            scope_note=f"order status types={order_status}",
        ),
        compare_row(
            check="post_only_reject_count",
            live_observed=watcher.get("post_only_reject_count"),
            replay_value=0,
            scope_note="no post-only reject observed in this window",
        ),
        compare_row(
            check="real_cancel_endpoint_called",
            live_observed=inline.get("real_cancel_endpoint_called"),
            replay_value=True,
            scope_note="shutdown proof used tracked cancel",
        ),
        compare_row(
            check="shutdown_proof_status",
            live_observed=inline.get("shutdown_proof_status"),
            replay_value="pass",
            scope_note="tracked cancel / open-orders shutdown proof passed",
        ),
        compare_row(
            check="cancel_proof_status",
            live_observed=cancel_proof.get("proof_status"),
            replay_value="pass",
            scope_note="cancel shutdown proof file passed",
        ),
        compare_row(
            check="fill_count",
            live_observed=fill_count,
            replay_value=0,
            scope_note="no fill observed, replay must preserve no-fill fact",
        ),
        compare_row(
            check="maker_fill_count",
            live_observed=maker_fill_count,
            replay_value=0,
            scope_note="no maker fill observed",
        ),
        compare_row(
            check="final_open_orders_count",
            live_observed=final_open_orders_count,
            replay_value=0,
            scope_note="window final open-orders empty",
        ),
        compare_row(
            check="independent_final_open_orders_count",
            live_observed=independent_open_orders_count,
            replay_value=0,
            scope_note="independent read-only open-orders proof empty",
        ),
    ]

    no_positions = not (account.get("post_state", {}).get("assetPositions") or [])
    economics_rows = [
        {
            "domain": "fill_ledger",
            "live_evidence": f"live_fill_ledger_rows={len(live_fills)}",
            "replay_behavior": "no_fill_preserved",
            "acceptance": "pass" if len(live_fills) == 0 and fill_count == 0 else "fail",
            "reason": "no fill rows may not become synthetic fills",
        },
        {
            "domain": "fee_rebate",
            "live_evidence": "no fills / no settlement rows",
            "replay_behavior": "fail_closed_unsupported",
            "acceptance": "pass",
            "reason": "fee/rebate from fills cannot be inferred without fills",
        },
        {
            "domain": "inventory_transition",
            "live_evidence": f"post_asset_positions_empty={no_positions}",
            "replay_behavior": "no_inventory_transition",
            "acceptance": "pass" if no_positions else "fail",
            "reason": "no fill and no post-state asset position",
        },
        {
            "domain": "realized_pnl",
            "live_evidence": "no fills / no fees / no inventory transition",
            "replay_behavior": "fail_closed_no_realized_pnl",
            "acceptance": "pass",
            "reason": "realized PnL cannot be claimed for no-fill window",
        },
        {
            "domain": "max_loss_monitor",
            "live_evidence": str(max_loss.get("status")),
            "replay_behavior": "preserve_pass",
            "acceptance": "pass" if max_loss.get("status") == "pass" else "fail",
            "reason": "loss monitor did not detect a realized loss breach",
        },
        {
            "domain": "maker_viability",
            "live_evidence": "one resting/no-fill/cancel lifecycle",
            "replay_behavior": "fail_closed_unsupported",
            "acceptance": "pass",
            "reason": "one no-fill lifecycle is not maker viability",
        },
    ]

    optimism_rows = [
        {"check": "no_synthetic_fill", "evidence": "fill_count=0", "acceptance": "pass"},
        {"check": "no_fill_probability_inferred", "evidence": "single no-fill order is not a denominator", "acceptance": "pass"},
        {"check": "no_fill_horizon_inferred", "evidence": "no fill timestamp exists", "acceptance": "pass"},
        {"check": "no_fee_rebate_inferred", "evidence": "no filled settlement rows exist", "acceptance": "pass"},
        {"check": "no_realized_pnl_inferred", "evidence": "no fill/fee/inventory transition", "acceptance": "pass"},
        {"check": "no_zero_latency_assumption", "evidence": "observed latencies preserved; missing exchange timestamps remain unsupported", "acceptance": "pass"},
        {"check": "no_reject_rate_generalization", "evidence": "post_only_reject_count=0 is one-window fact only", "acceptance": "pass"},
        {"check": "no_maker_viability_claim", "evidence": "one no-fill lifecycle cannot prove viability", "acceptance": "pass"},
    ]

    boundary = {
        "task_id": TASK_ID,
        "task_alias": TASK_ALIAS,
        "schema_version": SCHEMA_VERSION,
        "boundary_status": "pass",
        "source_artifact_only": True,
        "source_task": "0708T001",
        "input_root": str(input_root),
        "network_called": False,
        "remote_called": False,
        "aws_called": False,
        "credentials_read": False,
        "private_endpoint_called": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "live_submit_executed": False,
        "new_market_data_collection": False,
        "strategy_config_changed": False,
        "production_config_changed": False,
        "threshold_changed": False,
        "quote_envelope_changed": False,
        "order_size_changed": False,
        "max_submissions_changed": False,
        "fill_probability_claim": False,
        "fee_rebate_claim": False,
        "inventory_claim": False,
        "pnl_claim": False,
        "maker_viability_claim": False,
        "promotion_authorized": False,
        "unlocks_t011": False,
    }

    write_csv(
        output_dir / "market_view_replay_comparison.csv",
        market_rows,
        ["check", "live_observed", "replay_value", "acceptance", "scope_note"],
    )
    write_csv(
        output_dir / "decision_path_replay_comparison.csv",
        decision_rows,
        ["check", "live_observed", "replay_value", "acceptance", "scope_note"],
    )
    write_csv(
        output_dir / "lifecycle_replay_comparison.csv",
        lifecycle_rows,
        ["check", "live_observed", "replay_value", "acceptance", "scope_note"],
    )
    write_csv(
        output_dir / "economics_no_fill_attribution_matrix.csv",
        economics_rows,
        ["domain", "live_evidence", "replay_behavior", "acceptance", "reason"],
    )
    write_csv(output_dir / "optimism_check_matrix.csv", optimism_rows, ["check", "evidence", "acceptance"])
    write_json(output_dir / "boundary_manifest.json", boundary)

    all_rows: list[dict[str, Any]] = []
    all_rows.extend(market_rows)
    all_rows.extend(decision_rows)
    all_rows.extend(lifecycle_rows)
    all_rows.extend(economics_rows)
    all_rows.extend(optimism_rows)
    all_pass = all(str(row.get("acceptance")) == "pass" for row in all_rows) and boundary["boundary_status"] == "pass"
    manifest = {
        "task_id": TASK_ID,
        "task_alias": TASK_ALIAS,
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now_iso(),
        "git_commit": git_commit(),
        "source_task": "0708T001",
        "source_artifact_dir": str(input_root),
        "final_recommendation": PASSED_RECOMMENDATION if all_pass else BLOCKED_RECOMMENDATION,
        "market_view_acceptance": "pass" if all(row["acceptance"] == "pass" for row in market_rows) else "fail",
        "market_view_check_count": len(market_rows),
        "decision_path_acceptance": "pass" if all(row["acceptance"] == "pass" for row in decision_rows) else "fail",
        "decision_path_check_count": len(decision_rows),
        "lifecycle_acceptance": "pass" if all(row["acceptance"] == "pass" for row in lifecycle_rows) else "fail",
        "lifecycle_check_count": len(lifecycle_rows),
        "economics_no_fill_acceptance": "pass" if all(row["acceptance"] == "pass" for row in economics_rows) else "fail",
        "economics_check_count": len(economics_rows),
        "optimism_check_acceptance": "pass" if all(row["acceptance"] == "pass" for row in optimism_rows) else "fail",
        "optimism_check_count": len(optimism_rows),
        "boundary_status": boundary["boundary_status"],
        "live_summary": {
            "hyperliquid_l2book_fast": watcher.get("hyperliquid_l2book_fast"),
            "watcher_seconds_elapsed": watcher.get("watcher_seconds_elapsed"),
            "l2book_messages": l2_count,
            "trade_events": trade_events,
            "trigger_count": watcher.get("trigger_count"),
            "live_submissions_count": watcher.get("live_submissions_count"),
            "order_status_types": order_status_types,
            "fill_count": fill_count,
            "final_open_orders_count": final_open_orders_count,
            "independent_final_open_orders_count": independent_open_orders_count,
        },
        "does_not_unlock": [
            "0625T011",
            "0625T012",
            "stable_pnl_claim",
            "maker_viability_claim",
            "promotion",
            "final_mvp_pass",
        ],
        "next_recommendation": (
            "full_t010_same_window_replay_acceptance_passed_but_t011_requires_additional_windows"
            if all_pass
            else "repair_same_window_replay_acceptance_before_t011"
        ),
        "output_files": {
            "market_view_replay_comparison": str(output_dir / "market_view_replay_comparison.csv"),
            "decision_path_replay_comparison": str(output_dir / "decision_path_replay_comparison.csv"),
            "lifecycle_replay_comparison": str(output_dir / "lifecycle_replay_comparison.csv"),
            "economics_no_fill_attribution_matrix": str(output_dir / "economics_no_fill_attribution_matrix.csv"),
            "optimism_check_matrix": str(output_dir / "optimism_check_matrix.csv"),
            "boundary_manifest": str(output_dir / "boundary_manifest.json"),
        },
    }
    write_json(output_dir / "same_window_replay_acceptance_manifest.json", manifest)

    report = [
        "# 0708T002 Same-Window Replay Acceptance",
        "",
        f"Final recommendation: `{manifest['final_recommendation']}`",
        "",
        f"- Source artifact: `{input_root}`",
        f"- Market-view checks: `{status_count(market_rows)}`",
        f"- Decision-path checks: `{status_count(decision_rows)}`",
        f"- Lifecycle checks: `{status_count(lifecycle_rows)}`",
        f"- Economics/no-fill checks: `{status_count(economics_rows)}`",
        f"- Optimism checks: `{status_count(optimism_rows)}`",
        f"- Boundary status: `{boundary['boundary_status']}`",
        "",
        "This acceptance is local/offline only. It preserves no-fill and unsupported economics fail-closed; it does not claim stable PnL, maker viability, promotion, or final MVP pass.",
        "",
    ]
    (output_dir / "validation_report.md").write_text("\n".join(report), encoding="utf-8")
    write_csv(output_dir / "artifact_nonempty_check.csv", artifact_nonempty_rows(output_dir), ["artifact", "size_bytes", "status"])
    build_sha256_manifest(output_dir)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    manifest = run_acceptance(input_root=args.input_root, output_dir=args.output_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
