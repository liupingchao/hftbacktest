#!/usr/bin/env python3
"""Read-only replay lifecycle mismatch diagnosis runner."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from execution_outcome_calibration import (
    DEFAULT_HORIZONS_MS,
    DEFAULT_MAKER_FEE_BPS,
    DEFAULT_MAX_FUTURE_GAP_MS,
    DEFAULT_TICK_SIZE,
    _bucketize,
    _domain_bundle,
    _expand,
    _finite,
    _float,
    _generated_at,
    _hash_file,
    _load_json,
    _mean,
    _quantile,
    _rate,
    _safe_div,
    _write_csv,
    _write_json,
    build_submit_key_coverage,
    load_joined_decisions,
)


TASK_ID = "0515T001"


def _live_audit_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("audit_live_*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one audit_live_*.csv under {run_dir}, found {len(candidates)}")
    return candidates[0]


def _replay_audit_csv(run_dir: Path) -> Path:
    path = run_dir / "out" / "backtest_audit_replay" / "audit_bt_audit_replay.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing audit replay csv: {path}")
    return path


def _time_to_fill_ms(row: dict[str, Any]) -> float:
    submit_ts = row.get("submit_ts_local")
    fill_ts = row.get("first_fill_ts_local")
    if submit_ts in {"", None} or fill_ts in {"", None}:
        return math.nan
    try:
        submit_ns = int(submit_ts)
        fill_ns = int(fill_ts)
    except (TypeError, ValueError):
        return math.nan
    if fill_ns < submit_ns:
        return math.nan
    return (fill_ns - submit_ns) / 1_000_000.0


def _case_label(pair: dict[str, Any]) -> str:
    live = pair["live"]
    replay = pair["replay"]
    live_fill = int(live.get("fill_count", 0)) > 0
    replay_fill = int(replay.get("fill_count", 0)) > 0
    live_state = str(live.get("final_order_state") or "")
    replay_state = str(replay.get("final_order_state") or "")
    if not live_fill and replay_fill:
        if live_state == "canceled":
            return "live_canceled_replay_filled"
        if live_state == "open_or_missing":
            return "live_open_replay_filled"
        return "live_nofill_replay_filled"
    if live_fill and not replay_fill:
        if replay_state == "canceled":
            return "live_filled_replay_canceled"
        return "live_filled_replay_nofill"
    if live_state != replay_state:
        return f"state_diff:{live_state}->{replay_state}"
    return "aligned"


def build_matched_submit_state_diff(matched_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in matched_pairs:
        live = pair["live"]
        replay = pair["replay"]
        live_ttf = _time_to_fill_ms(live)
        replay_ttf = _time_to_fill_ms(replay)
        rows.append(
            {
                "submit_key": pair["submit_key"],
                "submit_strategy_seq": pair["submit_strategy_seq"],
                "order_side": pair["order_side"],
                "case_label": _case_label(pair),
                "placement_bucket": live.get("placement_bucket", ""),
                "distance_to_bbo_ticks": live.get("distance_to_bbo_ticks", ""),
                "edge_vs_fair_ticks": live.get("edge_vs_fair_ticks", ""),
                "inventory_score": live.get("inventory_score", ""),
                "latency_signal_ms": live.get("latency_signal_ms", ""),
                "top5_join_age_ms": live.get("top5_join_age_ms", ""),
                "join_stale": live.get("join_stale", ""),
                "live_final_state": live.get("final_order_state", ""),
                "replay_final_state": replay.get("final_order_state", ""),
                "live_fill_count": live.get("fill_count", ""),
                "replay_fill_count": replay.get("fill_count", ""),
                "live_fill_by_5000ms": live.get("fill_by_5000ms", ""),
                "replay_fill_by_5000ms": replay.get("fill_by_5000ms", ""),
                "live_fill_after_cancel_request": live.get("fill_after_cancel_request", ""),
                "replay_fill_after_cancel_request": replay.get("fill_after_cancel_request", ""),
                "live_cancel_to_fill_delay_ms": live.get("cancel_to_fill_delay_ms", ""),
                "replay_cancel_to_fill_delay_ms": replay.get("cancel_to_fill_delay_ms", ""),
                "live_time_to_fill_ms": live_ttf,
                "replay_time_to_fill_ms": replay_ttf,
                "time_to_fill_gap_ms": abs(live_ttf - replay_ttf) if _finite(live_ttf) and _finite(replay_ttf) else math.nan,
                "live_submit_ts_local": live.get("submit_ts_local", ""),
                "replay_submit_ts_local": replay.get("submit_ts_local", ""),
                "live_order_id": live.get("order_id", ""),
                "replay_order_id": replay.get("order_id", ""),
            }
        )
    return rows


def build_replay_only_fill_cases(state_diff_rows: list[dict[str, Any]], horizons_ms: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in state_diff_rows:
        if int(row["live_fill_count"] or 0) > 0 or int(row["replay_fill_count"] or 0) <= 0:
            continue
        output = dict(row)
        output["first_replay_fill_horizon_ms"] = ""
        for horizon_ms in horizons_ms:
            replay_field = f"replay_fill_by_{horizon_ms}ms"
            live_field = f"live_fill_by_{horizon_ms}ms"
            output[replay_field] = ""
            output[live_field] = ""
        rows.append(output)
    return rows


def build_live_cancel_replay_fill_cases(state_diff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in state_diff_rows if row["case_label"] == "live_canceled_replay_filled"]


def build_cancel_fill_timeline_diff(matched_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in matched_pairs:
        live = pair["live"]
        replay = pair["replay"]
        if int(live.get("fill_after_cancel_request", 0)) == 0 and int(replay.get("fill_after_cancel_request", 0)) == 0:
            continue
        live_delay = _float(live.get("cancel_to_fill_delay_ms"))
        replay_delay = _float(replay.get("cancel_to_fill_delay_ms"))
        rows.append(
            {
                "submit_key": pair["submit_key"],
                "submit_strategy_seq": pair["submit_strategy_seq"],
                "order_side": pair["order_side"],
                "placement_bucket": live.get("placement_bucket", ""),
                "inventory_score": live.get("inventory_score", ""),
                "latency_signal_ms": live.get("latency_signal_ms", ""),
                "live_fill_after_cancel_request": live.get("fill_after_cancel_request", ""),
                "replay_fill_after_cancel_request": replay.get("fill_after_cancel_request", ""),
                "live_cancel_request_ts": live.get("cancel_request_ts_local", ""),
                "replay_cancel_request_ts": replay.get("cancel_request_ts_local", ""),
                "live_cancel_ack_ts": live.get("cancel_ack_ts_local", ""),
                "replay_cancel_ack_ts": replay.get("cancel_ack_ts_local", ""),
                "live_first_fill_ts": live.get("first_fill_ts_local", ""),
                "replay_first_fill_ts": replay.get("first_fill_ts_local", ""),
                "live_terminal_ts": live.get("terminal_ts_local", ""),
                "replay_terminal_ts": replay.get("terminal_ts_local", ""),
                "live_cancel_to_fill_delay_ms": live_delay,
                "replay_cancel_to_fill_delay_ms": replay_delay,
                "cancel_to_fill_delay_gap_ms": abs(live_delay - replay_delay) if _finite(live_delay) and _finite(replay_delay) else math.nan,
                "case_label": _case_label(pair),
            }
        )
    return rows


def build_terminal_state_transition_diff(state_diff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in state_diff_rows if row["live_final_state"] != row["replay_final_state"]]


def build_cancel_ack_delay_diff(matched_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in matched_pairs:
        live = pair["live"]
        replay = pair["replay"]
        live_submit = _float(live.get("submit_ts_local"))
        replay_submit = _float(replay.get("submit_ts_local"))
        live_cancel_req = _float(live.get("cancel_request_ts_local"))
        replay_cancel_req = _float(replay.get("cancel_request_ts_local"))
        live_cancel_ack = _float(live.get("cancel_ack_ts_local"))
        replay_cancel_ack = _float(replay.get("cancel_ack_ts_local"))
        if not any(_finite(value) for value in [live_cancel_req, replay_cancel_req, live_cancel_ack, replay_cancel_ack]):
            continue
        live_req_delay = (live_cancel_req - live_submit) / 1_000_000.0 if _finite(live_cancel_req) and _finite(live_submit) else math.nan
        replay_req_delay = (replay_cancel_req - replay_submit) / 1_000_000.0 if _finite(replay_cancel_req) and _finite(replay_submit) else math.nan
        live_ack_delay = (live_cancel_ack - live_cancel_req) / 1_000_000.0 if _finite(live_cancel_ack) and _finite(live_cancel_req) else math.nan
        replay_ack_delay = (replay_cancel_ack - replay_cancel_req) / 1_000_000.0 if _finite(replay_cancel_ack) and _finite(replay_cancel_req) else math.nan
        rows.append(
            {
                "submit_key": pair["submit_key"],
                "submit_strategy_seq": pair["submit_strategy_seq"],
                "order_side": pair["order_side"],
                "placement_bucket": live.get("placement_bucket", ""),
                "live_cancel_request_after_submit_ms": live_req_delay,
                "replay_cancel_request_after_submit_ms": replay_req_delay,
                "cancel_request_after_submit_gap_ms": abs(live_req_delay - replay_req_delay) if _finite(live_req_delay) and _finite(replay_req_delay) else math.nan,
                "live_cancel_ack_after_request_ms": live_ack_delay,
                "replay_cancel_ack_after_request_ms": replay_ack_delay,
                "cancel_ack_after_request_gap_ms": abs(live_ack_delay - replay_ack_delay) if _finite(live_ack_delay) and _finite(replay_ack_delay) else math.nan,
                "live_final_state": live.get("final_order_state", ""),
                "replay_final_state": replay.get("final_order_state", ""),
            }
        )
    return rows


def _group_rate(rows: list[dict[str, Any]], field: str) -> float:
    values = [int(row.get(field, 0)) for row in rows]
    return _safe_div(sum(values), len(values))


def _group_mean(rows: list[dict[str, Any]], field: str) -> float:
    return _mean(_float(row.get(field)) for row in rows)


def _state_diff_group_row(group_name: str, group_label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    replay_only_fill = [row for row in rows if str(row["case_label"]).startswith("live_") and str(row["case_label"]).endswith("_replay_filled")]
    terminal_diff = [row for row in rows if row["live_final_state"] != row["replay_final_state"]]
    return {
        "group_name": group_name,
        "group_label": group_label,
        "rows": len(rows),
        "replay_only_fill_rows": len(replay_only_fill),
        "replay_only_fill_rate": _safe_div(len(replay_only_fill), len(rows)),
        "terminal_state_diff_rows": len(terminal_diff),
        "terminal_state_diff_rate": _safe_div(len(terminal_diff), len(rows)),
        "live_fill_after_cancel_rate": _group_rate(rows, "live_fill_after_cancel_request"),
        "replay_fill_after_cancel_rate": _group_rate(rows, "replay_fill_after_cancel_request"),
        "fill_after_cancel_gap": abs(_group_rate(rows, "live_fill_after_cancel_request") - _group_rate(rows, "replay_fill_after_cancel_request")),
        "live_time_to_fill_mean_ms": _group_mean(rows, "live_time_to_fill_ms"),
        "replay_time_to_fill_mean_ms": _group_mean(rows, "replay_time_to_fill_ms"),
        "time_to_fill_gap_mean_ms": abs(_group_mean(rows, "live_time_to_fill_ms") - _group_mean(rows, "replay_time_to_fill_ms")) if _finite(_group_mean(rows, "live_time_to_fill_ms")) and _finite(_group_mean(rows, "replay_time_to_fill_ms")) else math.nan,
    }


def _numeric_bucket_labels(values: list[Any]) -> list[str]:
    numeric = [_float(value) for value in values]
    labels = ["missing" for _ in numeric]
    finite_values = [value for value in numeric if _finite(value)]
    if not finite_values:
        return labels
    if len(set(round(value, 12) for value in finite_values)) == 1:
        for idx, value in enumerate(numeric):
            if _finite(value):
                labels[idx] = "all"
        return labels
    _, bucket_ids = _bucketize(finite_values, buckets=5)
    it = iter(bucket_ids)
    for idx, value in enumerate(numeric):
        if _finite(value):
            labels[idx] = f"q{int(next(it))}"
    return labels


def build_grouped_state_diff(rows: list[dict[str, Any]], field: str, group_name: str, *, categorical: bool) -> list[dict[str, Any]]:
    values = [row.get(field) for row in rows]
    labels = [str(value) if str(value) else "missing" for value in values] if categorical else _numeric_bucket_labels(values)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, label in zip(rows, labels, strict=False):
        grouped[label].append(row)
    return [_state_diff_group_row(group_name, label, group_rows) for label, group_rows in sorted(grouped.items())]


def build_replay_only_fill_by_horizon(state_diff_rows: list[dict[str, Any]], horizons_ms: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    replay_only_rows = [row for row in state_diff_rows if int(row["live_fill_count"] or 0) == 0 and int(row["replay_fill_count"] or 0) > 0]
    for horizon_ms in horizons_ms:
        count = sum(1 for row in replay_only_rows if int(row["replay_fill_by_5000ms"] or 0) == 1 and _finite(row.get("replay_time_to_fill_ms")) and float(row["replay_time_to_fill_ms"]) <= horizon_ms)
        rows.append(
            {
                "horizon_ms": horizon_ms,
                "replay_only_fill_rows": count,
                "replay_only_fill_rate_vs_all_replay_only": _safe_div(count, len(replay_only_rows)),
            }
        )
    return rows


def build_cancel_race_gap_by_bucket(state_diff_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped = build_grouped_state_diff(state_diff_rows, "placement_bucket", "placement_bucket", categorical=True)
    grouped.extend(build_grouped_state_diff(state_diff_rows, "inventory_score", "inventory_score_bucket", categorical=False))
    grouped.extend(build_grouped_state_diff(state_diff_rows, "latency_signal_ms", "latency_signal_ms_bucket", categorical=False))
    return grouped


def write_summary_markdown(
    path: Path,
    *,
    run_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    state_diff_rows: list[dict[str, Any]],
    cancel_timeline_rows: list[dict[str, Any]],
    placement_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
) -> None:
    replay_only_fill_rows = [row for row in state_diff_rows if int(row["live_fill_count"] or 0) == 0 and int(row["replay_fill_count"] or 0) > 0]
    live_cancel_replay_fill_rows = [row for row in state_diff_rows if row["case_label"] == "live_canceled_replay_filled"]
    top_placement = sorted(placement_rows, key=lambda row: float(row["replay_only_fill_rate"]) if _finite(row["replay_only_fill_rate"]) else -1, reverse=True)[:3]
    top_latency = sorted(latency_rows, key=lambda row: float(row["fill_after_cancel_gap"]) if _finite(row["fill_after_cancel_gap"]) else -1, reverse=True)[:3]
    placement_lines = [
        f"- `{row['group_label']}`: replay_only_fill_rate={row['replay_only_fill_rate']}, fill_after_cancel_gap={row['fill_after_cancel_gap']}"
        for row in top_placement
    ] or ["- none"]
    latency_lines = [
        f"- `{row['group_label']}`: fill_after_cancel_gap={row['fill_after_cancel_gap']}, time_to_fill_gap_mean_ms={row['time_to_fill_gap_mean_ms']}"
        for row in top_latency
    ] or ["- none"]

    lines = [
        f"# {TASK_ID} replay lifecycle mismatch diagnosis",
        "",
        "## Dataset",
        f"- run_dir: `{run_dir}`",
        f"- output_dir: `{output_dir}`",
        f"- matched_submit_rows: `{manifest['row_counts']['matched_submit_rows']}`",
        f"- replay_only_fill_rows: `{manifest['row_counts']['replay_only_fill_rows']}`",
        f"- live_cancel_replay_fill_rows: `{manifest['row_counts']['live_cancel_replay_fill_rows']}`",
        "",
        "## Main Takeaways",
        f"- replay-only fills: `{len(replay_only_fill_rows)}`",
        f"- live-canceled / replay-filled cases: `{len(live_cancel_replay_fill_rows)}`",
        f"- cancel timeline diff rows: `{len(cancel_timeline_rows)}`",
        "",
        "## Priority Hypotheses",
        "- replay long-horizon persistence is likely too optimistic for a meaningful subset of matched submits",
        "- replay cancel-request / cancel-ack / terminal timing likely leaves orders fill-eligible too long after cancel request",
        "- final-state mismatch is concentrated in the replay side rather than in submit matching",
        "",
        "## Placement Hot Spots",
        *placement_lines,
        "",
        "## Latency Hot Spots",
        *latency_lines,
        "",
        "## Boundaries",
        "- This task is diagnosis-only. It does not repair replay fill/cancel logic.",
        "- The diagnosis uses the same matched submit opportunity comparison unit as Stage 6B.",
        "- Results identify repair candidates; they are not quote-adjustment promotion evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_replay_lifecycle_mismatch_diagnosis(
    *,
    run_dir: Path,
    output_dir: Path,
    horizons_ms: Iterable[int] = DEFAULT_HORIZONS_MS,
    tick_size: float = DEFAULT_TICK_SIZE,
    max_future_gap_ms: float = DEFAULT_MAX_FUTURE_GAP_MS,
    maker_fee_bps: float = DEFAULT_MAKER_FEE_BPS,
) -> dict[str, Any]:
    run_dir = _expand(run_dir)
    output_dir = _expand(output_dir)
    live_audit_csv = _live_audit_csv(run_dir)
    replay_audit_csv = _replay_audit_csv(run_dir)
    joined_csv = run_dir / "t009_fixed_sidecar" / "joined_decisions.csv"
    stage3_json = run_dir / "maker_acceptance_stage3.json"
    sidecar_metrics_json = run_dir / "t009_fixed_sidecar" / "metrics.json"
    join_metrics_json = run_dir / "t009_fixed_sidecar" / "joined_decisions.metrics.json"

    joined_rows = load_joined_decisions(joined_csv)
    live_bundle = _domain_bundle(
        audit_csv=live_audit_csv,
        joined_rows=joined_rows,
        tick_size=tick_size,
        horizons_ms=horizons_ms,
        max_future_gap_ms=max_future_gap_ms,
        maker_fee_bps=maker_fee_bps,
    )
    replay_bundle = _domain_bundle(
        audit_csv=replay_audit_csv,
        joined_rows=joined_rows,
        tick_size=tick_size,
        horizons_ms=horizons_ms,
        max_future_gap_ms=max_future_gap_ms,
        maker_fee_bps=maker_fee_bps,
    )

    submit_key_rows, matched_pairs, coverage_summary = build_submit_key_coverage(live_bundle, replay_bundle)
    state_diff_rows = build_matched_submit_state_diff(matched_pairs)
    replay_only_fill_rows = build_replay_only_fill_cases(state_diff_rows, list(horizons_ms))
    live_cancel_replay_fill_rows = build_live_cancel_replay_fill_cases(state_diff_rows)
    cancel_timeline_rows = build_cancel_fill_timeline_diff(matched_pairs)
    terminal_state_rows = build_terminal_state_transition_diff(state_diff_rows)
    cancel_ack_delay_rows = build_cancel_ack_delay_diff(matched_pairs)
    placement_rows = build_grouped_state_diff(state_diff_rows, "placement_bucket", "placement_bucket", categorical=True)
    inventory_rows = build_grouped_state_diff(state_diff_rows, "inventory_score", "inventory_score_bucket", categorical=False)
    latency_rows = build_grouped_state_diff(state_diff_rows, "latency_signal_ms", "latency_signal_ms_bucket", categorical=False)
    replay_only_fill_horizon_rows = build_replay_only_fill_by_horizon(state_diff_rows, list(horizons_ms))
    cancel_race_gap_by_bucket_rows = build_cancel_race_gap_by_bucket(state_diff_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "matched_submit_state_diff.csv", state_diff_rows, fieldnames=list(state_diff_rows[0].keys()) if state_diff_rows else [])
    _write_csv(output_dir / "replay_only_fill_cases.csv", replay_only_fill_rows, fieldnames=list(replay_only_fill_rows[0].keys()) if replay_only_fill_rows else list(state_diff_rows[0].keys()) if state_diff_rows else [])
    _write_csv(output_dir / "live_cancel_replay_fill_cases.csv", live_cancel_replay_fill_rows, fieldnames=list(live_cancel_replay_fill_rows[0].keys()) if live_cancel_replay_fill_rows else list(state_diff_rows[0].keys()) if state_diff_rows else [])
    _write_csv(output_dir / "cancel_fill_timeline_diff.csv", cancel_timeline_rows, fieldnames=list(cancel_timeline_rows[0].keys()) if cancel_timeline_rows else [])
    _write_csv(output_dir / "terminal_state_transition_diff.csv", terminal_state_rows, fieldnames=list(terminal_state_rows[0].keys()) if terminal_state_rows else list(state_diff_rows[0].keys()) if state_diff_rows else [])
    _write_csv(output_dir / "cancel_ack_delay_diff.csv", cancel_ack_delay_rows, fieldnames=list(cancel_ack_delay_rows[0].keys()) if cancel_ack_delay_rows else [])
    _write_csv(output_dir / "state_diff_by_placement.csv", placement_rows, fieldnames=list(placement_rows[0].keys()) if placement_rows else [])
    _write_csv(output_dir / "state_diff_by_inventory.csv", inventory_rows, fieldnames=list(inventory_rows[0].keys()) if inventory_rows else [])
    _write_csv(output_dir / "state_diff_by_latency.csv", latency_rows, fieldnames=list(latency_rows[0].keys()) if latency_rows else [])
    _write_csv(output_dir / "replay_only_fill_by_horizon.csv", replay_only_fill_horizon_rows, fieldnames=list(replay_only_fill_horizon_rows[0].keys()) if replay_only_fill_horizon_rows else [])
    _write_csv(output_dir / "cancel_race_gap_by_bucket.csv", cancel_race_gap_by_bucket_rows, fieldnames=list(cancel_race_gap_by_bucket_rows[0].keys()) if cancel_race_gap_by_bucket_rows else [])

    stage3_payload = _load_json(stage3_json) if stage3_json.exists() else {}
    sidecar_metrics = _load_json(sidecar_metrics_json) if sidecar_metrics_json.exists() else {}
    join_metrics = _load_json(join_metrics_json) if join_metrics_json.exists() else {}
    manifest = {
        "task_id": TASK_ID,
        "generated_at": _generated_at(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "input_hashes": {
            "live_audit_csv": _hash_file(live_audit_csv),
            "replay_audit_csv": _hash_file(replay_audit_csv),
            "joined_decisions_csv": _hash_file(joined_csv),
            "maker_acceptance_stage3_json": _hash_file(stage3_json) if stage3_json.exists() else "",
            "sidecar_metrics_json": _hash_file(sidecar_metrics_json) if sidecar_metrics_json.exists() else "",
            "joined_decisions_metrics_json": _hash_file(join_metrics_json) if join_metrics_json.exists() else "",
        },
        "stage3_classification": (
            stage3_payload.get("market_view", {}).get("classification")
            or stage3_payload.get("classification")
            or "unknown"
        ),
        "row_counts": {
            "matched_submit_rows": coverage_summary["matched_submit_rows"],
            "replay_only_fill_rows": len(replay_only_fill_rows),
            "live_cancel_replay_fill_rows": len(live_cancel_replay_fill_rows),
            "cancel_fill_timeline_rows": len(cancel_timeline_rows),
            "terminal_state_diff_rows": len(terminal_state_rows),
        },
        "sidecar_metrics": sidecar_metrics,
        "joined_decisions_metrics": join_metrics,
        "artifacts": [
            "REPLAY_LIFECYCLE_MISMATCH_DIAGNOSIS_SUMMARY.md",
            "matched_submit_state_diff.csv",
            "replay_only_fill_cases.csv",
            "live_cancel_replay_fill_cases.csv",
            "cancel_fill_timeline_diff.csv",
            "terminal_state_transition_diff.csv",
            "cancel_ack_delay_diff.csv",
            "state_diff_by_placement.csv",
            "state_diff_by_inventory.csv",
            "state_diff_by_latency.csv",
            "replay_only_fill_by_horizon.csv",
            "cancel_race_gap_by_bucket.csv",
            "run_manifest.json",
        ],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    write_summary_markdown(
        output_dir / "REPLAY_LIFECYCLE_MISMATCH_DIAGNOSIS_SUMMARY.md",
        run_dir=run_dir,
        output_dir=output_dir,
        manifest=manifest,
        state_diff_rows=state_diff_rows,
        cancel_timeline_rows=cancel_timeline_rows,
        placement_rows=placement_rows,
        inventory_rows=inventory_rows,
        latency_rows=latency_rows,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory containing Stage 6B and audit artifacts.")
    parser.add_argument(
        "--output-dir",
        help=f"Output directory. Default: <run-dir>/stage6c_replay_lifecycle_mismatch_{TASK_ID}",
    )
    parser.add_argument("--tick-size", type=float, default=DEFAULT_TICK_SIZE)
    parser.add_argument("--maker-fee-bps", type=float, default=DEFAULT_MAKER_FEE_BPS)
    parser.add_argument("--max-future-gap-ms", type=float, default=DEFAULT_MAX_FUTURE_GAP_MS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _expand(args.run_dir)
    output_dir = _expand(args.output_dir) if args.output_dir else run_dir / f"stage6c_replay_lifecycle_mismatch_{TASK_ID}"
    manifest = run_replay_lifecycle_mismatch_diagnosis(
        run_dir=run_dir,
        output_dir=output_dir,
        tick_size=float(args.tick_size),
        max_future_gap_ms=float(args.max_future_gap_ms),
        maker_fee_bps=float(args.maker_fee_bps),
    )
    print(json.dumps({"task_id": TASK_ID, "output_dir": str(output_dir), "row_counts": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
