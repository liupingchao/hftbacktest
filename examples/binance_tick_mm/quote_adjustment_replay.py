#!/usr/bin/env python3
"""Default-off Step 9B quote-adjustment offline replay diagnostics.

This runner does not change strategy behavior. It evaluates the Step 9A
candidate matrix against existing audit and label artifacts, then emits a
diagnostic classification for whether a later replay experiment is ready.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable


TASK_ID = "0519T008"
RUNNER_MODE = "default_off_offline_diagnostic"
DEFAULT_MAX_SAMPLES_PER_CANDIDATE = 50
DEFAULT_TICK_SIZE = 0.1
DEFAULT_MULTI_SAMPLE_OUTPUT_DIR = "stage9c_multi_sample_validation"
DEFAULT_MULTI_SAMPLE_RUN_OUTPUT_PREFIX = "stage9c_quote_adjustment_replay"
REQUIRED_T006_FIELDS = (
    "quote_update_intent",
    "quote_update_action",
    "quote_update_reason",
    "min_move_passed",
    "quote_age_ms",
    "join_age_ms",
    "anchor_age_ms",
    "latency_bucket",
    "throttle_state",
    "token_bucket_state",
    "cancel_readd_bucket",
    "reject_throttle_drop_cause",
    "post_only_pre_check",
    "post_only_post_check",
    "inventory_request_id",
)

DECISIONABLE_BUCKET_VERDICTS = (
    "baseline_comparator",
    "observed_and_not_worse",
    "observed_and_mixed",
    "observed_and_worse",
    "suppressed_by_guard",
    "needs_more_data",
    "needs_runner_or_artifact_change",
    "blocked_by_data_quality",
)

CANDIDATE_CLASSIFICATIONS = (
    "reject",
    "keep_for_research",
    "ready_for_tiny_live_design",
    "blocked_by_data_quality",
    "needs_runner_or_artifact_change",
)

RESEARCH_COMPARISON_THRESHOLDS = {
    "sample_count": 4,
    "duration_min": 120.0,
    "submit_orders": 10000,
    "filled_orders": 250,
}

TINY_LIVE_DESIGN_THRESHOLDS = {
    "sample_count": 5,
    "duration_min": 180.0,
    "submit_orders": 15000,
    "filled_orders": 500,
}


@dataclass(frozen=True)
class CandidateDefinition:
    candidate_id: str
    family: str
    parameters: dict[str, Any]
    decision_inputs: tuple[str, ...]
    disallowed_inputs_check: str
    expected_effect: str
    risk: str
    related_t006_fields: tuple[str, ...]
    simulated_changes: tuple[str, ...]
    proxy_metric_status: str = "diagnostic_proxy"


@dataclass
class CandidateRuntime:
    definition: CandidateDefinition
    decision_seqs: set[int]


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _int(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if text == "":
        return default
    try:
        return int(float(text))
    except ValueError:
        return default


def _bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return math.nan
    return sum(finite) / len(finite)


def _min(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return min(finite) if finite else math.nan


def _max(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    return max(finite) if finite else math.nan


def _rate(count: int, total: int) -> float:
    return float(count) / float(total) if total else math.nan


def _safe_num(value: float) -> float | str:
    return float(value) if _finite(value) else ""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_csv_with_header(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader), list(reader.fieldnames or [])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _find_first(run_dir: Path, *patterns: str) -> Path:
    for pattern in patterns:
        matches = sorted(run_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"none of {patterns!r} found under {run_dir}")


def _parse_utc_timestamp(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_minutes(start: str, stop: str) -> float:
    start_ts = _parse_utc_timestamp(start)
    stop_ts = _parse_utc_timestamp(stop)
    if start_ts is None or stop_ts is None:
        return math.nan
    return max(0.0, (stop_ts - start_ts).total_seconds() / 60.0)


def _row_by_candidate(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row.get("candidate_id", ""): row for row in rows if row.get("candidate_id")}


def _num_delta(row: dict[str, Any], baseline: dict[str, Any], key: str) -> float:
    value = _float(row.get(key))
    base = _float(baseline.get(key))
    if not _finite(value) or not _finite(base):
        return math.nan
    return value - base


def _boolish_text(value: Any) -> str:
    return "true" if _bool(value) else "false"


def _artifact_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "audit_csv": _find_first(run_dir, "audit_live*.csv"),
        "maker_acceptance": run_dir / "maker_acceptance.json",
        "stage8b_decision": run_dir / "stage8b_quote_update_diagnostic_0519T005" / "implementation_planning_decision.json",
        "execution_labels": run_dir
        / "stage5_execution_outcome_labels_0514T005"
        / "execution_outcome_labels.csv",
        "fill_horizon": run_dir / "stage5_execution_outcome_labels_0514T005" / "fill_horizon_labels.csv",
        "fill_markout": run_dir / "stage5_execution_outcome_labels_0514T005" / "fill_markout_labels.csv",
        "stage5c_safety": run_dir / "stage5c_quote_anchor_safety_0518T004" / "quote_anchor_safety_rows.csv",
        "stage6_summary": run_dir / "stage6_final_calibration_0519T001" / "execution_calibration_summary.md",
    }


def candidate_definitions() -> list[CandidateDefinition]:
    return [
        CandidateDefinition(
            candidate_id="baseline_control",
            family="baseline_control",
            parameters={"mode": "no_change"},
            decision_inputs=("action", "planned_action", "reject_reason", "throttle_reason"),
            disallowed_inputs_check="no future labels used for decision",
            expected_effect="reproduce baseline action and diagnostic metrics",
            risk="none; baseline comparator only",
            related_t006_fields=("quote_update_intent", "quote_update_action", "quote_update_reason"),
            simulated_changes=(),
            proxy_metric_status="baseline",
        ),
        CandidateDefinition(
            candidate_id="fair_reservation_shift_edge_25",
            family="fair_reservation_shift",
            parameters={"min_abs_edge_ticks": 25.0, "shift_mode": "reservation/fair diagnostic"},
            decision_inputs=("mid", "fair", "reservation", "target_bid_tick", "target_ask_tick"),
            disallowed_inputs_check="no future markout/fill/PnL feedback",
            expected_effect="reduce adverse markout by shifting quotes away from stale or weak fair edge",
            risk="directional bias or lower fill rate",
            related_t006_fields=("quote_update_intent", "quote_update_reason"),
            simulated_changes=("simulated_quote_intent", "simulated_spread_or_shift"),
        ),
        CandidateDefinition(
            candidate_id="inventory_reservation_shift_band",
            family="inventory_reservation_shift",
            parameters={"position_abs_min": 0.001, "inventory_score_max": 0.5},
            decision_inputs=("position", "inventory_score", "working order state"),
            disallowed_inputs_check="no future inventory recovery outcome",
            expected_effect="reduce max inventory excursion and improve recovery-side preference",
            risk="missed spread capture or overly passive inventory recovery",
            related_t006_fields=("inventory_request_id", "quote_update_reason"),
            simulated_changes=("simulated_quote_intent", "simulated_inventory_request"),
        ),
        CandidateDefinition(
            candidate_id="spread_widening_stale_latency",
            family="spread_widening",
            parameters={"latency_signal_ms_min": 5.0, "book_view_stale_ms_min": 50.0},
            decision_inputs=("latency_signal_ms", "book_view_stale_ms", "anchor_age_ms"),
            disallowed_inputs_check="no future adverse-selection label",
            expected_effect="reduce toxic fills and bad-price exposure in stale/latency regimes",
            risk="lower fill probability and lower participation",
            related_t006_fields=("latency_bucket", "join_age_ms", "anchor_age_ms", "quote_update_reason"),
            simulated_changes=("simulated_spread_widening",),
        ),
        CandidateDefinition(
            candidate_id="size_reduction_or_add_side_suppression_pressure",
            family="size_reduction_or_add_side_suppression",
            parameters={"inventory_score_max": 0.5, "recent_reject_or_throttle_min": 1},
            decision_inputs=("inventory_score", "recent reject/throttle", "position"),
            disallowed_inputs_check="no future fill or markout label",
            expected_effect="reduce inventory-worsening fills and API/churn pressure",
            risk="lower participation and lower spread capture",
            related_t006_fields=("quote_update_reason", "reject_throttle_drop_cause", "inventory_request_id"),
            simulated_changes=("simulated_size_reduction", "simulated_add_side_suppression"),
        ),
        CandidateDefinition(
            candidate_id="stale_latency_no_fresh_add",
            family="stale_latency_no_fresh_add",
            parameters={"latency_signal_ms_min": 5.0, "book_view_stale_ms_min": 50.0},
            decision_inputs=("latency_signal_ms", "book_view_stale_ms", "join_age_ms", "anchor_age_ms"),
            disallowed_inputs_check="no future bad-price outcome",
            expected_effect="avoid adding fresh quotes when market view is stale or latency is high",
            risk="too much inactivity and missed fills",
            related_t006_fields=("latency_bucket", "reject_throttle_drop_cause"),
            simulated_changes=("simulated_add_side_suppression",),
        ),
        CandidateDefinition(
            candidate_id="min_move_quote_age_churn_guard",
            family="min_move_quote_age_churn_guard",
            parameters={"min_target_move_ticks": 2, "min_quote_age_ms": 100.0},
            decision_inputs=("target_move_since_last_quote_or_cancel", "reject_reason", "throttle_reason"),
            disallowed_inputs_check="no same-sample PnL feedback",
            expected_effect="reduce API use and cancel/re-add churn",
            risk="stale quote persistence",
            related_t006_fields=("min_move_passed", "quote_age_ms", "token_bucket_state", "cancel_readd_bucket"),
            simulated_changes=("simulated_hold",),
        ),
        CandidateDefinition(
            candidate_id="post_only_safety_interaction",
            family="post_only_safety_interaction",
            parameters={"uses_stage5c_safety": True},
            decision_inputs=("stage5c clamp/suppress/recheck", "target ticks", "anchor age"),
            disallowed_inputs_check="no future post-only reject label",
            expected_effect="keep post-only risk observable and clamp/suppress bad-price rows",
            risk="over-clamping or too passive quoting",
            related_t006_fields=("post_only_pre_check", "post_only_post_check", "anchor_age_ms"),
            simulated_changes=("simulated_safety_clamp", "simulated_add_side_suppression"),
        ),
    ]


def _decision_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in audit_rows if row.get("event_type") == "decision"]


def _seq(row: dict[str, str], *keys: str) -> int | None:
    for key in keys:
        value = _int(row.get(key))
        if value is not None:
            return value
    return None


def _stage5c_by_seq(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for row in rows:
        seq = _seq(row, "strategy_seq")
        if seq is not None:
            out[seq] = row
    return out


def _edge_ticks(row: dict[str, str], tick_size: float = DEFAULT_TICK_SIZE) -> float:
    mid = _float(row.get("mid"))
    fair = _float(row.get("fair"))
    reservation = _float(row.get("reservation"))
    if _finite(fair) and _finite(mid) and tick_size > 0.0:
        return (fair - mid) / tick_size
    if _finite(reservation) and _finite(mid) and tick_size > 0.0:
        return (reservation - mid) / tick_size
    return math.nan


def _submit_like(row: dict[str, str]) -> bool:
    return "submit" in str(row.get("action", "")).lower() or "submit" in str(row.get("planned_action", "")).lower()


def _recent_pressure(row: dict[str, str]) -> int:
    return int(_float(row.get("recent_reject_count_500ms"), 0.0)) + int(_float(row.get("recent_throttle_count_500ms"), 0.0))


def _candidate_trigger(
    candidate: CandidateDefinition,
    row: dict[str, str],
    *,
    safety_row: dict[str, str] | None,
) -> bool:
    if candidate.family == "baseline_control":
        return True
    if candidate.family == "fair_reservation_shift":
        return abs(_edge_ticks(row)) >= float(candidate.parameters["min_abs_edge_ticks"])
    if candidate.family == "inventory_reservation_shift":
        return abs(_float(row.get("position"), 0.0)) >= float(candidate.parameters["position_abs_min"]) or _float(
            row.get("inventory_score"), 1.0
        ) <= float(candidate.parameters["inventory_score_max"])
    if candidate.family == "spread_widening":
        return _float(row.get("latency_signal_ms"), 0.0) >= float(
            candidate.parameters["latency_signal_ms_min"]
        ) or _float(row.get("book_view_stale_ms"), 0.0) >= float(candidate.parameters["book_view_stale_ms_min"])
    if candidate.family == "size_reduction_or_add_side_suppression":
        return (
            _float(row.get("inventory_score"), 1.0) <= float(candidate.parameters["inventory_score_max"])
            or bool(row.get("reject_reason"))
            or bool(row.get("throttle_reason"))
        )
    if candidate.family == "stale_latency_no_fresh_add":
        return _submit_like(row) and (
            _float(row.get("latency_signal_ms"), 0.0) >= float(candidate.parameters["latency_signal_ms_min"])
            or _float(row.get("book_view_stale_ms"), 0.0) >= float(candidate.parameters["book_view_stale_ms_min"])
        )
    if candidate.family == "min_move_quote_age_churn_guard":
        buy_move = _float(row.get("target_move_since_last_quote_or_cancel_buy"), math.nan)
        sell_move = _float(row.get("target_move_since_last_quote_or_cancel_sell"), math.nan)
        move_values = [value for value in (buy_move, sell_move) if _finite(value)]
        min_move = min(move_values) if move_values else math.inf
        return (
            min_move < float(candidate.parameters["min_target_move_ticks"])
            or row.get("reject_reason") in {"quote_throttle", "api_interval_guard", "token_bucket", "api_limit"}
            or row.get("throttle_reason") != ""
        )
    if candidate.family == "post_only_safety_interaction":
        if not safety_row:
            return False
        return any(
            _bool(safety_row.get(key))
            for key in (
                "bid_clamped",
                "ask_clamped",
                "suppress_buy",
                "suppress_sell",
                "missing_anchor",
                "stale_anchor",
                "post_only_risk_after_recheck",
                "depth_fallback_used",
            )
        )
    return False


def build_candidate_runtimes(
    *,
    decisions: list[dict[str, str]],
    safety_by_seq: dict[int, dict[str, str]],
    candidates: list[CandidateDefinition],
) -> list[CandidateRuntime]:
    runtimes: list[CandidateRuntime] = []
    for candidate in candidates:
        seqs: set[int] = set()
        for row in decisions:
            seq = _seq(row, "strategy_seq")
            if seq is None:
                continue
            if _candidate_trigger(candidate, row, safety_row=safety_by_seq.get(seq)):
                seqs.add(seq)
        runtimes.append(CandidateRuntime(definition=candidate, decision_seqs=seqs))
    return runtimes


def _order_seq(row: dict[str, str]) -> int | None:
    return _seq(row, "decision_context_strategy_seq", "submit_strategy_seq", "linked_strategy_seq")


def _order_id(row: dict[str, str]) -> str:
    return str(row.get("order_id", "")).strip()


def _rows_for_candidate(rows: list[dict[str, str]], candidate: CandidateRuntime) -> list[dict[str, str]]:
    if candidate.definition.family == "baseline_control":
        return list(rows)
    return [row for row in rows if (_order_seq(row) in candidate.decision_seqs)]


def _rows_for_order_ids(rows: list[dict[str, str]], order_ids: set[str]) -> list[dict[str, str]]:
    if not order_ids:
        return []
    return [row for row in rows if _order_id(row) in order_ids]


def _metric_row(
    candidate: CandidateRuntime,
    *,
    total_decisions: int,
    label_rows: list[dict[str, str]],
    horizon_rows: list[dict[str, str]],
    markout_rows: list[dict[str, str]],
) -> dict[str, Any]:
    rows = _rows_for_candidate(label_rows, candidate)
    order_ids = {_order_id(row) for row in rows if _order_id(row)}
    horizons = horizon_rows if candidate.definition.family == "baseline_control" else _rows_for_order_ids(horizon_rows, order_ids)
    markouts = markout_rows if candidate.definition.family == "baseline_control" else _rows_for_order_ids(markout_rows, order_ids)
    filled = [row for row in rows if _float(row.get("fill_count"), 0.0) > 0.0 or _bool(row.get("full_fill"))]
    fill_after_cancel = [row for row in rows if _bool(row.get("fill_after_cancel_request"))]
    markout_net_ev_values = [_float(row.get("net_ev_proxy_ticks")) for row in markouts]
    markout_realized_values = [_float(row.get("realized_spread_proxy_ticks")) for row in markouts]
    label_realized_values = [_float(row.get("realized_spread_proxy_ticks")) for row in rows]
    fee_adjusted_values = [_float(row.get("fee_adjusted_realized_spread_ticks")) for row in rows]
    fee_proxy_values = [
        gross - net
        for gross, net in zip(label_realized_values, fee_adjusted_values, strict=False)
        if _finite(gross) and _finite(net)
    ]
    return {
        "candidate_id": candidate.definition.candidate_id,
        "family": candidate.definition.family,
        "metric_status": "diagnostic_proxy",
        "decision_rows": len(candidate.decision_seqs),
        "decision_rate": _safe_num(_rate(len(candidate.decision_seqs), total_decisions)),
        "submit_orders": len(rows),
        "filled_orders": len(filled),
        "fill_rate": _safe_num(_rate(len(filled), len(rows))),
        "fill_after_cancel_orders": len(fill_after_cancel),
        "fill_after_cancel_rate": _safe_num(_rate(len(fill_after_cancel), len(rows))),
        "fill_by_100ms_rate": _safe_num(_horizon_rate(horizons, 100)),
        "fill_by_500ms_rate": _safe_num(_horizon_rate(horizons, 500)),
        "fill_by_1000ms_rate": _safe_num(_horizon_rate(horizons, 1000)),
        "fill_by_5000ms_rate": _safe_num(_horizon_rate(horizons, 5000)),
        "mean_time_to_fill_ms": _safe_num(_mean(_float(row.get("time_to_fill_ms")) for row in rows)),
        "cancel_to_fill_delay_ms_mean": _safe_num(_mean(_float(row.get("cancel_to_fill_delay_ms")) for row in rows)),
        "mean_markout_100ms_ticks": _safe_num(_markout_mean(markouts, 100)),
        "mean_markout_500ms_ticks": _safe_num(_markout_mean(markouts, 500)),
        "mean_markout_1000ms_ticks": _safe_num(_markout_mean(markouts, 1000)),
        "mean_markout_5000ms_ticks": _safe_num(_markout_mean(markouts, 5000)),
        "gross_pnl_proxy_ticks": _safe_num(sum(value for value in label_realized_values if _finite(value))),
        "net_pnl_proxy_ticks": _safe_num(sum(value for value in fee_adjusted_values if _finite(value))),
        "markout_net_ev_proxy_ticks": _safe_num(sum(value for value in markout_net_ev_values if _finite(value))),
        "fee_proxy_ticks": _safe_num(sum(fee_proxy_values)),
        "fee_proxy_ticks_mean": _safe_num(_mean(fee_proxy_values)),
        "spread_capture_proxy_ticks_mean": _safe_num(_mean(label_realized_values)),
        "realized_spread_proxy_ticks_mean": _safe_num(_mean(markout_realized_values)),
        "fee_adjusted_realized_spread_ticks_mean": _safe_num(_mean(fee_adjusted_values)),
        "sample_limit_note": "single_sample_runner_validation_only",
    }


def _horizon_rate(rows: list[dict[str, str]], horizon_ms: int) -> float:
    matching = [row for row in rows if _int(row.get("horizon_ms")) == horizon_ms and _bool(row.get("horizon_observable"))]
    return _rate(sum(1 for row in matching if _bool(row.get("fill_by_horizon"))), len(matching))


def _markout_mean(rows: list[dict[str, str]], horizon_ms: int) -> float:
    return _mean(
        _float(row.get("side_adjusted_markout_ticks"))
        for row in rows
        if _int(row.get("horizon_ms")) == horizon_ms and _bool(row.get("horizon_observable"))
    )


def _fill_quality_rows(
    candidates: list[CandidateRuntime],
    label_rows: list[dict[str, str]],
    horizon_rows: list[dict[str, str]],
    markout_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        labels = _rows_for_candidate(label_rows, candidate)
        order_ids = {_order_id(row) for row in labels if _order_id(row)}
        horizons = horizon_rows if candidate.definition.family == "baseline_control" else _rows_for_order_ids(horizon_rows, order_ids)
        markouts = markout_rows if candidate.definition.family == "baseline_control" else _rows_for_order_ids(markout_rows, order_ids)
        for horizon in (100, 500, 1000, 5000):
            h_rows = [row for row in horizons if _int(row.get("horizon_ms")) == horizon and _bool(row.get("horizon_observable"))]
            m_rows = [row for row in markouts if _int(row.get("horizon_ms")) == horizon and _bool(row.get("horizon_observable"))]
            out.append(
                {
                    "candidate_id": candidate.definition.candidate_id,
                    "family": candidate.definition.family,
                    "horizon_ms": horizon,
                    "rows": len(h_rows),
                    "fill_rate": _safe_num(_rate(sum(1 for row in h_rows if _bool(row.get("fill_by_horizon"))), len(h_rows))),
                    "time_to_fill_ms_mean": _safe_num(_mean(_float(row.get("time_to_fill_ms")) for row in h_rows)),
                    "side_adjusted_markout_ticks_mean": _safe_num(
                        _mean(_float(row.get("side_adjusted_markout_ticks")) for row in m_rows)
                    ),
                    "spread_capture_ticks_mean": _safe_num(_mean(_float(row.get("realized_spread_proxy_ticks")) for row in m_rows)),
                    "metric_status": "diagnostic_proxy",
                }
            )
    return out


def _inventory_rows(
    candidates: list[CandidateRuntime],
    decisions: list[dict[str, str]],
    label_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    decisions_by_seq = {_seq(row, "strategy_seq"): row for row in decisions if _seq(row, "strategy_seq") is not None}
    for candidate in candidates:
        rows = [decisions_by_seq[seq] for seq in sorted(candidate.decision_seqs) if seq in decisions_by_seq]
        labels = _rows_for_candidate(label_rows, candidate)
        filled_labels = [row for row in labels if _float(row.get("fill_count"), 0.0) > 0.0 or _bool(row.get("full_fill"))]
        positions = [_float(row.get("position"), 0.0) for row in rows]
        zero_crossings = 0
        prev_sign = 0
        for pos in positions:
            sign = 1 if pos > 0 else -1 if pos < 0 else 0
            if prev_sign and sign and sign != prev_sign:
                zero_crossings += 1
            if sign:
                prev_sign = sign
        out.append(
            {
                "candidate_id": candidate.definition.candidate_id,
                "family": candidate.definition.family,
                "decision_rows": len(rows),
                "position_mean": _safe_num(_mean(positions)),
                "max_abs_position": _safe_num(_max(abs(value) for value in positions)),
                "time_in_one_order_band_rate": _safe_num(_rate(sum(1 for value in positions if abs(value) <= 0.001), len(rows))),
                "zero_crossing_count_proxy": zero_crossings,
                "inventory_score_mean": _safe_num(_mean(_float(row.get("inventory_score")) for row in rows)),
                "filled_order_rows": len(filled_labels),
                "inventory_increasing_fill_rate": _safe_num(
                    _rate(sum(1 for row in filled_labels if _bool(row.get("inventory_increasing_fill"))), len(filled_labels))
                ),
                "inventory_reducing_fill_rate": _safe_num(
                    _rate(sum(1 for row in filled_labels if _bool(row.get("inventory_reducing_fill"))), len(filled_labels))
                ),
                "cycle_flat_observed_rate": _safe_num(
                    _rate(sum(1 for row in filled_labels if _bool(row.get("inventory_cycle_flat_observed"))), len(filled_labels))
                ),
                "time_to_flat_ms_mean": _safe_num(_mean(_float(row.get("time_to_flat_ms")) for row in filled_labels)),
                "max_abs_position_until_flat_mean": _safe_num(
                    _mean(_float(row.get("max_abs_position_until_flat")) for row in filled_labels)
                ),
                "recovery_quality_proxy_rate": _safe_num(
                    _rate(
                        sum(
                            1
                            for row in filled_labels
                            if _bool(row.get("inventory_cycle_flat_observed")) or _bool(row.get("crossed_zero_after_fill"))
                        ),
                        len(filled_labels),
                    )
                ),
                "metric_status": "decision_proxy_plus_observed_fill_labels",
            }
        )
    return out


def _api_churn_rows(
    candidates: list[CandidateRuntime],
    decisions: list[dict[str, str]],
    label_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    decisions_by_seq = {_seq(row, "strategy_seq"): row for row in decisions if _seq(row, "strategy_seq") is not None}
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = [decisions_by_seq[seq] for seq in sorted(candidate.decision_seqs) if seq in decisions_by_seq]
        labels = _rows_for_candidate(label_rows, candidate)
        planned = [row for row in rows if str(row.get("planned_action", "")).lower() != "keep"]
        actual = [row for row in rows if str(row.get("action", "")).lower() != "keep"]
        rejects = [row for row in rows if row.get("reject_reason")]
        throttles = [row for row in rows if row.get("throttle_reason")]
        out.append(
            {
                "candidate_id": candidate.definition.candidate_id,
                "family": candidate.definition.family,
                "decision_rows": len(rows),
                "planned_action_rows": len(planned),
                "actual_action_rows": len(actual),
                "planned_actual_mismatch_rows": sum(1 for row in rows if row.get("planned_action") != row.get("action")),
                "reject_rows": len(rejects),
                "throttle_rows": len(throttles),
                "api_drop_rows": sum(1 for row in rows if _bool(row.get("dropped_by_api_limit"))),
                "latency_drop_rows": sum(1 for row in rows if _bool(row.get("dropped_by_latency"))),
                "min_target_move_buy_mean": _safe_num(
                    _mean(_float(row.get("target_move_since_last_quote_or_cancel_buy")) for row in rows)
                ),
                "min_target_move_sell_mean": _safe_num(
                    _mean(_float(row.get("target_move_since_last_quote_or_cancel_sell")) for row in rows)
                ),
                "quote_age_ms_mean": _safe_num(_mean(_float(row.get("quote_age_ms")) for row in rows)),
                "join_age_ms_mean": _safe_num(_mean(_float(row.get("join_age_ms")) for row in rows)),
                "anchor_age_ms_mean": _safe_num(_mean(_float(row.get("anchor_age_ms")) for row in rows)),
                "recent_reject_count_500ms_mean": _safe_num(
                    _mean(_float(row.get("recent_reject_count_500ms")) for row in labels)
                ),
                "recent_throttle_count_500ms_mean": _safe_num(
                    _mean(_float(row.get("recent_throttle_count_500ms")) for row in labels)
                ),
                "fast_cancel_churn_rate": _safe_num(
                    _rate(sum(1 for row in labels if _bool(row.get("fast_cancel_churn"))), len(labels))
                ),
                "cancel_readd_bucket_rows": sum(1 for row in rows if row.get("cancel_readd_bucket")),
                "token_bucket_state_rows": sum(1 for row in rows if row.get("token_bucket_state")),
                "quote_age_metric_status": "missing_t006_field_proxy_only",
            }
        )
    return out


def _post_only_rows(
    candidates: list[CandidateRuntime],
    decisions: list[dict[str, str]],
    safety_by_seq: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    decisions_by_seq = {_seq(row, "strategy_seq"): row for row in decisions if _seq(row, "strategy_seq") is not None}
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = [safety_by_seq[seq] for seq in sorted(candidate.decision_seqs) if seq in safety_by_seq]
        decision_rows = [decisions_by_seq[seq] for seq in sorted(candidate.decision_seqs) if seq in decisions_by_seq]
        out.append(
            {
                "candidate_id": candidate.definition.candidate_id,
                "family": candidate.definition.family,
                "safety_rows": len(rows),
                "decision_rows": len(decision_rows),
                "post_only_pre_check_rows": sum(1 for row in decision_rows if _bool(row.get("post_only_pre_check"))),
                "post_only_post_check_rows": sum(1 for row in decision_rows if _bool(row.get("post_only_post_check"))),
                "bid_clamped_rows": sum(1 for row in rows if _bool(row.get("bid_clamped"))),
                "ask_clamped_rows": sum(1 for row in rows if _bool(row.get("ask_clamped"))),
                "suppress_buy_rows": sum(1 for row in rows if _bool(row.get("suppress_buy"))),
                "suppress_sell_rows": sum(1 for row in rows if _bool(row.get("suppress_sell"))),
                "stale_anchor_rows": sum(1 for row in rows if _bool(row.get("stale_anchor"))),
                "missing_anchor_rows": sum(1 for row in rows if _bool(row.get("missing_anchor"))),
                "post_only_risk_after_recheck_rows": sum(1 for row in rows if _bool(row.get("post_only_risk_after_recheck"))),
                "metric_status": "stage5c_diagnostic_proxy",
            }
        )
    return out


def _candidate_bucket_evidence_rows(
    *,
    candidates: list[CandidateRuntime],
    metrics: list[dict[str, Any]],
    api_rows: list[dict[str, Any]],
    inventory_rows: list[dict[str, Any]],
    post_only_rows: list[dict[str, Any]],
    action_path_rows: list[dict[str, Any]],
    maker_acceptance: dict[str, Any],
) -> list[dict[str, Any]]:
    metrics_by_id = _row_by_candidate(metrics)
    api_by_id = _row_by_candidate(api_rows)
    inventory_by_id = _row_by_candidate(inventory_rows)
    post_only_by_id = _row_by_candidate(post_only_rows)
    coverage_by_id = _row_by_candidate(action_path_rows)
    baseline_metrics = metrics_by_id.get("baseline_control", {})
    baseline_api = api_by_id.get("baseline_control", {})
    baseline_inventory = inventory_by_id.get("baseline_control", {})
    baseline_post_only = post_only_by_id.get("baseline_control", {})
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate.definition.candidate_id
        metric = metrics_by_id.get(candidate_id, {})
        api = api_by_id.get(candidate_id, {})
        inventory = inventory_by_id.get(candidate_id, {})
        post_only = post_only_by_id.get(candidate_id, {})
        coverage = coverage_by_id.get(candidate_id, {})
        common = {
            "candidate_id": candidate_id,
            "family": candidate.definition.family,
            "decision_rows": int(_float(metric.get("decision_rows"), 0.0)),
            "decision_rate": _safe_num(_float(metric.get("decision_rate"))),
            "submit_orders": int(_float(metric.get("submit_orders"), 0.0)),
            "filled_orders": int(_float(metric.get("filled_orders"), 0.0)),
            "fill_rate": _safe_num(_float(metric.get("fill_rate"))),
            "fill_rate_delta_vs_baseline": _safe_num(_num_delta(metric, baseline_metrics, "fill_rate")),
            "fill_after_cancel_rate": _safe_num(_float(metric.get("fill_after_cancel_rate"))),
            "fill_after_cancel_rate_delta_vs_baseline": _safe_num(
                _num_delta(metric, baseline_metrics, "fill_after_cancel_rate")
            ),
            "markout_5000ms_ticks": _safe_num(_float(metric.get("mean_markout_5000ms_ticks"))),
            "markout_5000ms_delta_vs_baseline": _safe_num(
                _num_delta(metric, baseline_metrics, "mean_markout_5000ms_ticks")
            ),
            "net_pnl_proxy_ticks": _safe_num(_float(metric.get("net_pnl_proxy_ticks"))),
            "net_pnl_proxy_delta_vs_baseline": _safe_num(_num_delta(metric, baseline_metrics, "net_pnl_proxy_ticks")),
            "spread_capture_ticks_mean": _safe_num(_float(metric.get("spread_capture_proxy_ticks_mean"))),
            "spread_capture_delta_vs_baseline": _safe_num(
                _num_delta(metric, baseline_metrics, "spread_capture_proxy_ticks_mean")
            ),
            "reject_rows": int(_float(api.get("reject_rows"), 0.0)),
            "throttle_rows": int(_float(api.get("throttle_rows"), 0.0)),
            "api_drop_rows": int(_float(api.get("api_drop_rows"), 0.0)),
            "latency_drop_rows": int(_float(api.get("latency_drop_rows"), 0.0)),
            "planned_action_rows": int(_float(api.get("planned_action_rows"), 0.0)),
            "actual_action_rows": int(_float(api.get("actual_action_rows"), 0.0)),
            "fast_cancel_churn_rate": _safe_num(_float(api.get("fast_cancel_churn_rate"))),
            "fast_cancel_churn_delta_vs_baseline": _safe_num(
                _num_delta(api, baseline_api, "fast_cancel_churn_rate")
            ),
            "inventory_score_mean": _safe_num(_float(inventory.get("inventory_score_mean"))),
            "inventory_score_delta_vs_baseline": _safe_num(
                _num_delta(inventory, baseline_inventory, "inventory_score_mean")
            ),
            "inventory_increasing_fill_rate": _safe_num(_float(inventory.get("inventory_increasing_fill_rate"))),
            "inventory_reducing_fill_rate": _safe_num(_float(inventory.get("inventory_reducing_fill_rate"))),
            "recovery_quality_proxy_rate": _safe_num(_float(inventory.get("recovery_quality_proxy_rate"))),
            "post_only_risk_after_recheck_rows": int(_float(post_only.get("post_only_risk_after_recheck_rows"), 0.0)),
            "missing_anchor_rows": int(_float(post_only.get("missing_anchor_rows"), 0.0)),
            "stale_anchor_rows": int(_float(post_only.get("stale_anchor_rows"), 0.0)),
            "bid_clamped_rows": int(_float(post_only.get("bid_clamped_rows"), 0.0)),
            "ask_clamped_rows": int(_float(post_only.get("ask_clamped_rows"), 0.0)),
            "submit_order_coverage_rate": _safe_num(_float(coverage.get("submit_order_coverage_rate"))),
            "maker_acceptance_passed": int(bool(maker_acceptance.get("passed", False))),
        }
        planned = int(common["planned_action_rows"])
        actual = int(common["actual_action_rows"])
        submit = int(common["submit_orders"])
        decision = int(common["decision_rows"])
        suppressed = (
            int(common["latency_drop_rows"])
            + int(common["api_drop_rows"])
            + int(common["missing_anchor_rows"])
            + int(common["stale_anchor_rows"])
        )
        suppressed_rate = _rate(suppressed, decision)
        submit_realization_rate = _rate(actual, planned)
        scenario_rows = [
            {
                **common,
                "scenario_family": "intended_candidate_regime",
                "scenario_bucket": candidate.definition.family,
                "bucket_rows": decision,
                "bucket_coverage_rate": common["decision_rate"],
                "primary_metric": "decision_coverage",
                "primary_metric_value": common["decision_rate"],
                "metric_status": "decision_time_visible_candidate_trigger",
                "zero_submit_reason": "no_candidate_decision_coverage" if decision == 0 else "",
                "guard_suppressed_rows": suppressed,
                "guard_suppressed_rate": _safe_num(suppressed_rate),
                "submit_realization_rate": _safe_num(submit_realization_rate),
            },
            {
                **common,
                "scenario_family": "execution_observability",
                "scenario_bucket": "observed_submit_fill_quality" if submit > 0 else "no_observed_submit_fill",
                "bucket_rows": submit,
                "bucket_coverage_rate": common["submit_order_coverage_rate"],
                "primary_metric": "filled_orders",
                "primary_metric_value": common["filled_orders"],
                "metric_status": "observed_execution_label_proxy" if submit > 0 else "zero_submit_execution_not_identifiable",
                "zero_submit_reason": (
                    "suppressed_by_guard"
                    if decision > 0 and submit == 0 and suppressed > 0
                    else "no_candidate_decision_coverage" if decision == 0 else ""
                ),
                "guard_suppressed_rows": suppressed,
                "guard_suppressed_rate": _safe_num(suppressed_rate),
                "submit_realization_rate": _safe_num(submit_realization_rate),
            },
            {
                **common,
                "scenario_family": "api_churn_guard_path",
                "scenario_bucket": "guard_suppressed" if suppressed > 0 else "not_guard_suppressed",
                "bucket_rows": decision,
                "bucket_coverage_rate": common["decision_rate"],
                "primary_metric": "guard_suppressed_rows",
                "primary_metric_value": suppressed,
                "metric_status": "t006_audit_and_stage5c_proxy",
                "zero_submit_reason": "suppressed_by_guard" if decision > 0 and submit == 0 and suppressed > 0 else "",
                "guard_suppressed_rows": suppressed,
                "guard_suppressed_rate": _safe_num(suppressed_rate),
                "submit_realization_rate": _safe_num(submit_realization_rate),
            },
            {
                **common,
                "scenario_family": "post_only_safety",
                "scenario_bucket": "post_only_recheck",
                "bucket_rows": int(_float(post_only.get("safety_rows"), decision)),
                "bucket_coverage_rate": "",
                "primary_metric": "post_only_risk_after_recheck_rows",
                "primary_metric_value": common["post_only_risk_after_recheck_rows"],
                "metric_status": "stage5c_diagnostic_proxy",
                "zero_submit_reason": "",
                "guard_suppressed_rows": suppressed,
                "guard_suppressed_rate": _safe_num(suppressed_rate),
                "submit_realization_rate": _safe_num(submit_realization_rate),
            },
            {
                **common,
                "scenario_family": "inventory",
                "scenario_bucket": "inventory_proxy",
                "bucket_rows": decision,
                "bucket_coverage_rate": common["decision_rate"],
                "primary_metric": "inventory_score_mean",
                "primary_metric_value": common["inventory_score_mean"],
                "metric_status": "decision_proxy_plus_observed_fill_labels",
                "zero_submit_reason": "",
                "guard_suppressed_rows": suppressed,
                "guard_suppressed_rate": _safe_num(suppressed_rate),
                "submit_realization_rate": _safe_num(submit_realization_rate),
            },
        ]
        rows.extend(scenario_rows)
    return rows


def _bucket_verdict(row: dict[str, Any]) -> tuple[str, str]:
    candidate_id = row.get("candidate_id", "")
    if candidate_id == "baseline_control":
        return "baseline_comparator", "baseline comparator only"
    if int(_float(row.get("post_only_risk_after_recheck_rows"), 0.0)) > 0:
        return "observed_and_worse", "post-only risk after recheck is nonzero"
    decision_rows = int(_float(row.get("decision_rows"), 0.0))
    submit_orders = int(_float(row.get("submit_orders"), 0.0))
    filled_orders = int(_float(row.get("filled_orders"), 0.0))
    guard_rows = int(_float(row.get("guard_suppressed_rows"), 0.0))
    guard_rate = _float(row.get("guard_suppressed_rate"))
    markout_delta = _float(row.get("markout_5000ms_delta_vs_baseline"))
    fill_delta = _float(row.get("fill_rate_delta_vs_baseline"))
    churn_delta = _float(row.get("fast_cancel_churn_delta_vs_baseline"))
    fac_delta = _float(row.get("fill_after_cancel_rate_delta_vs_baseline"))
    if decision_rows == 0:
        return "needs_more_data", "zero intended-regime decision coverage"
    if submit_orders == 0 and guard_rows > 0:
        return "suppressed_by_guard", "candidate has decisions but all submit/fill observations are guard-suppressed"
    if submit_orders == 0:
        return "needs_runner_or_artifact_change", "candidate has decisions but no submit/fill observations to judge execution"
    if filled_orders == 0:
        return "needs_more_data", "candidate has submits but no fills in this bucket"
    adverse_markout = _finite(markout_delta) and markout_delta < -2.0
    worse_fill = _finite(fill_delta) and fill_delta < -0.001
    worse_churn = _finite(churn_delta) and churn_delta > 0.02
    worse_cancel_fill = _finite(fac_delta) and fac_delta > 0.01
    if adverse_markout and worse_fill:
        return "observed_and_worse", "5s markout worsens while fill rate also drops versus baseline"
    if adverse_markout or worse_fill or worse_churn or worse_cancel_fill:
        return "observed_and_mixed", "one or more execution-quality metrics worsens versus baseline"
    if _finite(guard_rate) and guard_rate > 0.95 and submit_orders < max(5, int(0.001 * decision_rows)):
        return "suppressed_by_guard", "guard suppression dominates the candidate regime"
    return "observed_and_not_worse", "observed submit/fill metrics do not breach current worse thresholds"


def _candidate_bucket_verdict_rows(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in evidence_rows:
        verdict, reason = _bucket_verdict(row)
        out.append(
            {
                **row,
                "bucket_verdict": verdict,
                "bucket_verdict_reason": reason,
                "decisionable": int(verdict in DECISIONABLE_BUCKET_VERDICTS),
            }
        )
    return out


def _coverage_rows(
    *,
    candidates: list[CandidateRuntime],
    decisions: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    maker_acceptance: dict[str, Any],
) -> list[dict[str, Any]]:
    total = len(decisions)
    order_total = len(label_rows)
    acceptance_passed = bool(maker_acceptance.get("passed", False))
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        orders = _rows_for_candidate(label_rows, candidate)
        rows.append(
            {
                "candidate_id": candidate.definition.candidate_id,
                "family": candidate.definition.family,
                "decision_rows": len(candidate.decision_seqs),
                "decision_coverage_rate": _safe_num(_rate(len(candidate.decision_seqs), total)),
                "submit_order_rows": len(orders),
                "submit_order_coverage_rate": _safe_num(_rate(len(orders), order_total)),
                "maker_acceptance_passed": int(acceptance_passed),
                "market_view_gate_status": "passed" if acceptance_passed else "unknown_or_failed",
                "lifecycle_boundary": "step6_closed_for_roadmap_progression_not_exact_queue_or_live_proof",
            }
        )
    return rows


def _audit_field_coverage_rows(headers: list[str], extra_paths: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for field in REQUIRED_T006_FIELDS:
        rows.append(
            {
                "field": field,
                "source": "audit_csv",
                "status": "available" if field in headers else "missing_in_existing_sample",
                "metric_status": "required_t006_field",
            }
        )
    for name, path in extra_paths.items():
        rows.append(
            {
                "field": name,
                "source": str(path),
                "status": "available" if path.exists() else "missing",
                "metric_status": "supporting_artifact",
            }
        )
    return rows


def _candidate_matrix_rows(candidates: list[CandidateDefinition]) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "parameters_json": json.dumps(candidate.parameters, sort_keys=True),
                "decision_time_visible_inputs": "|".join(candidate.decision_inputs),
                "disallowed_inputs_check": candidate.disallowed_inputs_check,
                "expected_effect": candidate.expected_effect,
                "risk": candidate.risk,
                "related_t006_fields": "|".join(candidate.related_t006_fields),
                "simulated_changes": "|".join(candidate.simulated_changes),
                "default_state": "off",
                "runner_mode": RUNNER_MODE,
                "proxy_metric_status": candidate.proxy_metric_status,
            }
        )
    return rows


def _candidate_decision_samples(
    candidates: list[CandidateRuntime],
    decisions: list[dict[str, str]],
    safety_by_seq: dict[int, dict[str, str]],
    max_samples: int,
) -> list[dict[str, Any]]:
    decisions_by_seq = {_seq(row, "strategy_seq"): row for row in decisions if _seq(row, "strategy_seq") is not None}
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        for seq in sorted(candidate.decision_seqs)[:max_samples]:
            row = decisions_by_seq.get(seq)
            if row is None:
                continue
            safety = safety_by_seq.get(seq, {})
            out.append(
                {
                    "candidate_id": candidate.definition.candidate_id,
                    "family": candidate.definition.family,
                    "strategy_seq": seq,
                    "ts_local": row.get("ts_local", ""),
                    "action": row.get("action", ""),
                    "planned_action": row.get("planned_action", ""),
                    "reject_reason": row.get("reject_reason", ""),
                    "throttle_reason": row.get("throttle_reason", ""),
                    "position": row.get("position", ""),
                    "inventory_score": row.get("inventory_score", ""),
                    "latency_signal_ms": row.get("latency_signal_ms", ""),
                    "book_view_stale_ms": row.get("book_view_stale_ms", ""),
                    "stage5c_anchor_source": safety.get("anchor_source", ""),
                    "stage5c_suppress_buy": safety.get("suppress_buy", ""),
                    "stage5c_suppress_sell": safety.get("suppress_sell", ""),
                    "diagnostic_note": "candidate_trigger_sample_only",
                }
            )
    return out


def _classify(
    *,
    missing_t006_count: int,
    metrics: list[dict[str, Any]],
    maker_acceptance: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not maker_acceptance.get("passed", False):
        reasons.append("maker acceptance / market-view gate is not passed in input artifacts")
        return "blocked_by_replay_or_market_view", reasons
    if missing_t006_count:
        reasons.append(
            f"{missing_t006_count} T006 quote-update audit fields are missing in the existing sample; "
            "runner used proxy fields for validation"
        )
        return "needs_more_instrumentation", reasons
    active = [row for row in metrics if row["family"] != "baseline_control" and int(row["decision_rows"]) > 0]
    if not active:
        reasons.append("candidate matrix produced no non-baseline decision coverage")
        return "no_effect", reasons
    worse = [
        row
        for row in active
        if _finite(row.get("fill_after_cancel_rate")) and float(row["fill_after_cancel_rate"]) > 0.25
    ]
    if worse:
        reasons.append("one or more active candidates retain high fill-after-cancel/churn proxy rates")
        return "worse_due_to_churn_or_fill_quality", reasons
    reasons.append("active candidates have nonzero coverage, but only one current-format sample is available")
    return "promising_but_single_sample", reasons


def _write_acceptance_decision(
    path: Path,
    *,
    classification: str,
    reasons: list[str],
    metrics: list[dict[str, Any]],
    missing_t006_count: int,
) -> None:
    key_metrics = "\n".join(
        f"- `{row['candidate_id']}`: decision rows `{row['decision_rows']}`, submit orders `{row['submit_orders']}`, "
        f"fill rate `{row['fill_rate']}`, fill-after-cancel rate `{row['fill_after_cancel_rate']}`"
        for row in metrics
    )
    reason_text = "\n".join(f"- {reason}" for reason in reasons) or "- none"
    text = f"""# 0519T008 Step 9B Quote-Adjustment Replay Decision

## Boundary

- Mode: `{RUNNER_MODE}`
- Dataset: `5-13-day-control-30min`
- Default-off only: yes
- Offline only: yes
- Live / promotion authorized: no
- Production strategy behavior changed: no

## Classification

- result: `{classification}`
- missing T006 audit fields in existing sample: `{missing_t006_count}`

## Reasons

{reason_text}

## Candidate Metrics

{key_metrics}

## Next-Step Interpretation

This output validates runner, metric, artifact, and diagnostic-classification mechanics. A single sample cannot establish live readiness or generalized profitability. If the result is `needs_more_instrumentation`, collect or replay with T006 audit fields before promotion-style claims. If later candidates become `promising_but_single_sample`, open a separate multi-sample validation planning task.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _load_sample_quality(run_dir: Path) -> dict[str, Any]:
    sidecar_metrics_path = run_dir / "t009_fixed_sidecar" / "metrics.json"
    joined_metrics_path = run_dir / "t009_fixed_sidecar" / "joined_decisions.metrics.json"
    sidecar = _read_json(sidecar_metrics_path) if sidecar_metrics_path.exists() else {}
    joined = _read_json(joined_metrics_path) if joined_metrics_path.exists() else {}
    gap_crossed = int(_float(joined.get("gap_crossed_join_count"), 0.0))
    first_aligned = _bool(sidecar.get("first_valid_update_aligned"))
    strict = first_aligned and gap_crossed == 0
    return {
        "first_valid_update_aligned": _boolish_text(first_aligned),
        "depth_pu_mismatch_count": int(_float(sidecar.get("depth_pu_mismatch_count"), 0.0)),
        "decision_join_coverage": _safe_num(_float(joined.get("decision_join_coverage"))),
        "future_join_count": int(_float(joined.get("future_join_count"), 0.0)),
        "gap_crossed_join_count": gap_crossed,
        "stale_join_count": int(_float(joined.get("stale_join_count"), 0.0)),
        "market_view_quality_bucket": "strict_clean_market_view" if strict else "caveated_market_view",
        "strict_market_view_gate_passed": int(strict),
        "accepted_market_view_caveat": int(not strict),
    }


def _sample_manifest_row(run_dir: Path, runner_output_dir: Path, *, caveated_sample_ids: set[str]) -> dict[str, Any]:
    start_marker_path = run_dir / "start_marker.json"
    stop_marker_path = run_dir / "stop_marker.json"
    start_marker = _read_json(start_marker_path) if start_marker_path.exists() else {}
    stop_marker = _read_json(stop_marker_path) if stop_marker_path.exists() else {}
    summary = _read_json(runner_output_dir / "candidate_summary.json")
    metrics = _row_by_candidate(_read_csv(runner_output_dir / "candidate_metrics.csv"))
    baseline = metrics.get("baseline_control", {})
    quality = _load_sample_quality(run_dir)
    sample_id = run_dir.name
    duration_min = _duration_minutes(start_marker.get("generated_at_utc", ""), stop_marker.get("generated_at_utc", ""))
    row = {
        "sample_id": sample_id,
        "run_dir": str(run_dir),
        "runner_output_dir": str(runner_output_dir),
        "start_utc": start_marker.get("generated_at_utc", ""),
        "stop_utc": stop_marker.get("generated_at_utc", ""),
        "duration_min": _safe_num(duration_min),
        "git_short_commit": (start_marker.get("git") or {}).get("short_commit", ""),
        "git_dirty": int(bool((start_marker.get("git") or {}).get("dirty", False))),
        "runner_classification": summary.get("classification", ""),
        "maker_acceptance_passed": int(bool(summary.get("maker_acceptance_passed", False))),
        "baseline_submit_orders": int(_float(baseline.get("submit_orders"), 0.0)),
        "baseline_filled_orders": int(_float(baseline.get("filled_orders"), 0.0)),
        "caveated_sample": int(sample_id in caveated_sample_ids),
    }
    row.update(quality)
    return row


def _sample_set_mass(manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    duration = sum(_float(row.get("duration_min"), 0.0) for row in manifest_rows)
    submits = sum(int(_float(row.get("baseline_submit_orders"), 0.0)) for row in manifest_rows)
    fills = sum(int(_float(row.get("baseline_filled_orders"), 0.0)) for row in manifest_rows)
    return {
        "sample_count": len(manifest_rows),
        "duration_min": round(duration, 2),
        "submit_orders": submits,
        "filled_orders": fills,
        "meets_research_comparison_threshold": (
            len(manifest_rows) >= RESEARCH_COMPARISON_THRESHOLDS["sample_count"]
            and duration >= RESEARCH_COMPARISON_THRESHOLDS["duration_min"]
            and submits >= RESEARCH_COMPARISON_THRESHOLDS["submit_orders"]
            and fills >= RESEARCH_COMPARISON_THRESHOLDS["filled_orders"]
        ),
        "meets_tiny_live_design_threshold": (
            len(manifest_rows) >= TINY_LIVE_DESIGN_THRESHOLDS["sample_count"]
            and duration >= TINY_LIVE_DESIGN_THRESHOLDS["duration_min"]
            and submits >= TINY_LIVE_DESIGN_THRESHOLDS["submit_orders"]
            and fills >= TINY_LIVE_DESIGN_THRESHOLDS["filled_orders"]
        ),
    }


def _load_sample_outputs(sample_id: str, output_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = _read_csv(output_dir / "candidate_metrics.csv")
    api = _row_by_candidate(_read_csv(output_dir / "api_churn_metrics.csv"))
    inventory = _row_by_candidate(_read_csv(output_dir / "inventory_cycle_metrics.csv"))
    post_only = _row_by_candidate(_read_csv(output_dir / "post_only_safety_metrics.csv"))
    bucket_verdicts = _read_csv(output_dir / "candidate_bucket_verdicts.csv")
    baseline = _row_by_candidate(metrics).get("baseline_control", {})
    rows: list[dict[str, Any]] = []
    for row in metrics:
        candidate_id = row.get("candidate_id", "")
        api_row = api.get(candidate_id, {})
        inventory_row = inventory.get(candidate_id, {})
        post_only_row = post_only.get(candidate_id, {})
        out = {
            "sample_id": sample_id,
            "candidate_id": candidate_id,
            "family": row.get("family", ""),
            "metric_status": row.get("metric_status", ""),
            "decision_rows": int(_float(row.get("decision_rows"), 0.0)),
            "decision_rate": _safe_num(_float(row.get("decision_rate"))),
            "submit_orders": int(_float(row.get("submit_orders"), 0.0)),
            "filled_orders": int(_float(row.get("filled_orders"), 0.0)),
            "fill_rate": _safe_num(_float(row.get("fill_rate"))),
            "fill_rate_delta_vs_baseline": _safe_num(_num_delta(row, baseline, "fill_rate")),
            "fill_after_cancel_orders": int(_float(row.get("fill_after_cancel_orders"), 0.0)),
            "fill_after_cancel_rate": _safe_num(_float(row.get("fill_after_cancel_rate"))),
            "fill_after_cancel_rate_delta_vs_baseline": _safe_num(
                _num_delta(row, baseline, "fill_after_cancel_rate")
            ),
            "mean_time_to_fill_ms": _safe_num(_float(row.get("mean_time_to_fill_ms"))),
            "markout_5000ms_ticks": _safe_num(_float(row.get("mean_markout_5000ms_ticks"))),
            "markout_5000ms_delta_vs_baseline": _safe_num(
                _num_delta(row, baseline, "mean_markout_5000ms_ticks")
            ),
            "net_pnl_proxy_ticks": _safe_num(_float(row.get("net_pnl_proxy_ticks"))),
            "net_pnl_proxy_delta_vs_baseline": _safe_num(_num_delta(row, baseline, "net_pnl_proxy_ticks")),
            "spread_capture_proxy_ticks_mean": _safe_num(_float(row.get("spread_capture_proxy_ticks_mean"))),
            "spread_capture_delta_vs_baseline": _safe_num(
                _num_delta(row, baseline, "spread_capture_proxy_ticks_mean")
            ),
            "reject_rows": int(_float(api_row.get("reject_rows"), 0.0)),
            "throttle_rows": int(_float(api_row.get("throttle_rows"), 0.0)),
            "api_drop_rows": int(_float(api_row.get("api_drop_rows"), 0.0)),
            "latency_drop_rows": int(_float(api_row.get("latency_drop_rows"), 0.0)),
            "fast_cancel_churn_rate": _safe_num(_float(api_row.get("fast_cancel_churn_rate"))),
            "inventory_score_mean": _safe_num(_float(inventory_row.get("inventory_score_mean"))),
            "post_only_risk_after_recheck_rows": int(
                _float(post_only_row.get("post_only_risk_after_recheck_rows"), 0.0)
            ),
            "missing_anchor_rows": int(_float(post_only_row.get("missing_anchor_rows"), 0.0)),
            "stale_anchor_rows": int(_float(post_only_row.get("stale_anchor_rows"), 0.0)),
        }
        rows.append(out)
    enriched_verdicts = []
    for row in bucket_verdicts:
        enriched_verdicts.append({"sample_id": sample_id, **row})
    return rows, enriched_verdicts


def _classify_candidate_set(rows: list[dict[str, Any]], verdict_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not rows:
        return "blocked_by_data_quality", "candidate has no rows in sample set"
    candidate_id = rows[0].get("candidate_id", "")
    if candidate_id == "baseline_control":
        return "keep_for_research", "baseline comparator only; not a promotion candidate"
    samples = len({row.get("sample_id", "") for row in rows})
    decision_coverage_samples = sum(1 for row in rows if int(_float(row.get("decision_rows"), 0.0)) > 0)
    submit_coverage_samples = sum(1 for row in rows if int(_float(row.get("submit_orders"), 0.0)) > 0)
    total_submit = sum(int(_float(row.get("submit_orders"), 0.0)) for row in rows)
    total_fill = sum(int(_float(row.get("filled_orders"), 0.0)) for row in rows)
    post_only_risk = sum(int(_float(row.get("post_only_risk_after_recheck_rows"), 0.0)) for row in rows)
    bucket_verdicts = {row.get("bucket_verdict", "") for row in verdict_rows}
    intended_decision_verdicts = {
        row.get("bucket_verdict", "")
        for row in verdict_rows
        if row.get("scenario_family") in {"intended_candidate_regime", "execution_observability", "api_churn_guard_path"}
    }
    markout_deltas = [_float(row.get("markout_5000ms_delta_vs_baseline")) for row in rows]
    fill_deltas = [_float(row.get("fill_rate_delta_vs_baseline")) for row in rows]
    worse_markout_count = sum(1 for value in markout_deltas if _finite(value) and value < -2.0)
    worse_fill_count = sum(1 for value in fill_deltas if _finite(value) and value < -0.001)
    if post_only_risk > 0:
        return "reject", "post-only risk after recheck is nonzero in accepted samples"
    if decision_coverage_samples == 0:
        return "reject", "zero intended-regime decision coverage across accepted samples"
    if "needs_runner_or_artifact_change" in bucket_verdicts:
        return "needs_runner_or_artifact_change", "at least one bucket cannot be judged from current artifacts"
    if "suppressed_by_guard" in intended_decision_verdicts and submit_coverage_samples == 0:
        return "keep_for_research", "candidate regime is decisionable but fully guard-suppressed in current samples"
    if submit_coverage_samples == 0 or total_submit == 0:
        return "keep_for_research", "candidate has no submit observations after current gates"
    if total_fill == 0:
        return "keep_for_research", "candidate has submit observations but no fills"
    if worse_markout_count >= 2 and worse_fill_count >= 2:
        return "reject", "multiple samples show worse 5s markout while fill rate also drops"
    if worse_markout_count >= max(2, math.ceil(samples / 2)):
        return "reject", "5s markout worsens in multiple samples"
    return "keep_for_research", "decisionable with no hard safety breach, but evidence is still offline/proxy"


def _stability_rows(
    *,
    candidate_rows: list[dict[str, Any]],
    verdict_rows: list[dict[str, Any]],
    accepted_sample_ids: set[str],
    clean_only_sample_ids: set[str],
) -> list[dict[str, Any]]:
    candidates = sorted({row.get("candidate_id", "") for row in candidate_rows})
    out: list[dict[str, Any]] = []
    for candidate_id in candidates:
        accepted_rows = [
            row for row in candidate_rows if row.get("candidate_id") == candidate_id and row.get("sample_id") in accepted_sample_ids
        ]
        clean_rows = [
            row for row in candidate_rows if row.get("candidate_id") == candidate_id and row.get("sample_id") in clean_only_sample_ids
        ]
        accepted_verdicts = [
            row for row in verdict_rows if row.get("candidate_id") == candidate_id and row.get("sample_id") in accepted_sample_ids
        ]
        clean_verdicts = [
            row for row in verdict_rows if row.get("candidate_id") == candidate_id and row.get("sample_id") in clean_only_sample_ids
        ]
        accepted_class, accepted_reason = _classify_candidate_set(accepted_rows, accepted_verdicts)
        clean_class, clean_reason = _classify_candidate_set(clean_rows, clean_verdicts)
        markout_deltas = [_float(row.get("markout_5000ms_delta_vs_baseline")) for row in accepted_rows]
        fill_deltas = [_float(row.get("fill_rate_delta_vs_baseline")) for row in accepted_rows]
        out.append(
            {
                "candidate_id": candidate_id,
                "family": accepted_rows[0].get("family", "") if accepted_rows else "",
                "accepted_classification": accepted_class,
                "accepted_reason": accepted_reason,
                "clean_only_classification": clean_class,
                "clean_only_reason": clean_reason,
                "classification_changed_by_clean_only": int(accepted_class != clean_class),
                "accepted_sample_count": len(accepted_rows),
                "accepted_decision_coverage_sample_count": sum(
                    1 for row in accepted_rows if int(_float(row.get("decision_rows"), 0.0)) > 0
                ),
                "accepted_submit_coverage_sample_count": sum(
                    1 for row in accepted_rows if int(_float(row.get("submit_orders"), 0.0)) > 0
                ),
                "accepted_total_submit_orders": sum(int(_float(row.get("submit_orders"), 0.0)) for row in accepted_rows),
                "accepted_total_filled_orders": sum(int(_float(row.get("filled_orders"), 0.0)) for row in accepted_rows),
                "accepted_median_fill_rate_delta_vs_baseline": _safe_num(median([v for v in fill_deltas if _finite(v)]))
                if any(_finite(v) for v in fill_deltas)
                else "",
                "accepted_median_markout_5000ms_delta_vs_baseline": _safe_num(
                    median([v for v in markout_deltas if _finite(v)])
                )
                if any(_finite(v) for v in markout_deltas)
                else "",
                "accepted_worse_markout_sample_count": sum(
                    1 for value in markout_deltas if _finite(value) and value < -2.0
                ),
                "accepted_bucket_verdicts": "|".join(sorted({str(row.get("bucket_verdict", "")) for row in accepted_verdicts})),
                "clean_only_sample_count": len(clean_rows),
                "clean_only_total_submit_orders": sum(int(_float(row.get("submit_orders"), 0.0)) for row in clean_rows),
                "clean_only_total_filled_orders": sum(int(_float(row.get("filled_orders"), 0.0)) for row in clean_rows),
                "ready_for_tiny_live_design": int(accepted_class == "ready_for_tiny_live_design"),
            }
        )
    return out


def _write_multi_sample_report(
    path: Path,
    *,
    manifest_rows: list[dict[str, Any]],
    accepted_mass: dict[str, Any],
    clean_only_mass: dict[str, Any],
    stability_rows: list[dict[str, Any]],
    caveated_sample_ids: set[str],
) -> None:
    def table_cell(value: Any) -> str:
        return str(value).replace("|", "<br>")

    rows = "\n".join(
        "|{candidate_id}|{accepted_classification}|{clean_only_classification}|{accepted_total_submit_orders}|{accepted_total_filled_orders}|{accepted_bucket_verdicts}|{accepted_reason}|".format(
            candidate_id=table_cell(row["candidate_id"]),
            accepted_classification=table_cell(row["accepted_classification"]),
            clean_only_classification=table_cell(row["clean_only_classification"]),
            accepted_total_submit_orders=table_cell(row["accepted_total_submit_orders"]),
            accepted_total_filled_orders=table_cell(row["accepted_total_filled_orders"]),
            accepted_bucket_verdicts=table_cell(row["accepted_bucket_verdicts"]),
            accepted_reason=table_cell(row["accepted_reason"]),
        )
        for row in stability_rows
    )
    samples = "\n".join(
        f"- `{row['sample_id']}`: duration `{row['duration_min']}` min, submits `{row['baseline_submit_orders']}`, "
        f"fills `{row['baseline_filled_orders']}`, market view `{row['market_view_quality_bucket']}`"
        for row in manifest_rows
    )
    caveat_text = ", ".join(sorted(caveated_sample_ids)) or "none"
    text = f"""# 0520T002 Step 9C Candidate-Bucket Decisionability Validation

## Boundary

- Mode: default-off offline validator / artifact hardening.
- No strategy behavior, live behavior, production default, sample collection, or promotion change was performed.
- Caveated samples excluded from clean-only sensitivity: `{caveat_text}`.

## Sample Set

{samples}

| Set | Samples | Duration Min | Submit Orders | Filled Orders | Research Threshold | Tiny-Live Threshold |
|---|---:|---:|---:|---:|---|---|
| accepted-set | {accepted_mass['sample_count']} | {accepted_mass['duration_min']} | {accepted_mass['submit_orders']} | {accepted_mass['filled_orders']} | {accepted_mass['meets_research_comparison_threshold']} | {accepted_mass['meets_tiny_live_design_threshold']} |
| clean-only | {clean_only_mass['sample_count']} | {clean_only_mass['duration_min']} | {clean_only_mass['submit_orders']} | {clean_only_mass['filled_orders']} | {clean_only_mass['meets_research_comparison_threshold']} | {clean_only_mass['meets_tiny_live_design_threshold']} |

## Candidate Verdicts

|candidate_id|accepted|clean_only|accepted_submit|accepted_filled|bucket_verdicts|reason|
|---|---|---|---:|---:|---|---|
{rows}

## Interpretation

- The current 8 Step 9 families now have explicit bucket evidence and bucket verdict rows.
- `spread_widening_stale_latency` is decisionable as a guard-suppressed regime when its decisions exist but submit/fill observations are blocked by current latency / anchor guards.
- No candidate is promoted to `ready_for_tiny_live_design` by this task.
- Remaining tiny-live blockers are sample/event mass and offline/proxy evidence, not live authorization.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_multi_sample_validation(
    *,
    run_dirs: list[Path],
    output_dir: Path,
    task_id: str,
    caveated_sample_ids: set[str] | None = None,
) -> dict[str, Any]:
    if not run_dirs:
        raise ValueError("at least one --multi-sample-run-dir is required")
    caveated_sample_ids = set(caveated_sample_ids or set())
    output_dir = _expand(output_dir)
    manifest_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    bucket_verdicts: list[dict[str, Any]] = []
    run_outputs: list[dict[str, str]] = []
    for run_dir_arg in run_dirs:
        run_dir = _expand(run_dir_arg)
        sample_output = run_dir / f"{DEFAULT_MULTI_SAMPLE_RUN_OUTPUT_PREFIX}_{task_id}"
        run_quote_adjustment_replay(run_dir=run_dir, output_dir=sample_output)
        manifest_rows.append(_sample_manifest_row(run_dir, sample_output, caveated_sample_ids=caveated_sample_ids))
        sample_metrics, sample_bucket_verdicts = _load_sample_outputs(run_dir.name, sample_output)
        candidate_rows.extend(sample_metrics)
        bucket_verdicts.extend(sample_bucket_verdicts)
        run_outputs.append({"sample_id": run_dir.name, "output_dir": str(sample_output)})

    accepted_sample_ids = {row["sample_id"] for row in manifest_rows}
    clean_only_sample_ids = accepted_sample_ids - caveated_sample_ids
    accepted_manifest = [row for row in manifest_rows if row["sample_id"] in accepted_sample_ids]
    clean_manifest = [row for row in manifest_rows if row["sample_id"] in clean_only_sample_ids]
    accepted_mass = _sample_set_mass(accepted_manifest)
    clean_only_mass = _sample_set_mass(clean_manifest)
    stability = _stability_rows(
        candidate_rows=candidate_rows,
        verdict_rows=bucket_verdicts,
        accepted_sample_ids=accepted_sample_ids,
        clean_only_sample_ids=clean_only_sample_ids,
    )
    summary = {
        "task_id": task_id,
        "runner_mode": "multi_sample_candidate_bucket_decisionability",
        "generated_at": _generated_at(),
        "accepted_set_mass": accepted_mass,
        "clean_only_mass": clean_only_mass,
        "caveated_sample_ids": sorted(caveated_sample_ids),
        "run_outputs": run_outputs,
        "candidate_classifications": {
            row["candidate_id"]: {
                "accepted": row["accepted_classification"],
                "clean_only": row["clean_only_classification"],
                "reason": row["accepted_reason"],
            }
            for row in stability
        },
        "not_authorized": [
            "live",
            "default-on",
            "promotion",
            "sample expansion",
            "production behavior change",
            "strategy behavior change",
        ],
    }
    _write_csv(output_dir / "sample_manifest.csv", manifest_rows)
    _write_json(output_dir / "sample_manifest.json", manifest_rows)
    _write_json(output_dir / "sample_set_summary.json", {"accepted_set": accepted_mass, "clean_only": clean_only_mass})
    _write_csv(output_dir / "candidate_sample_metrics.csv", candidate_rows)
    _write_csv(output_dir / "candidate_bucket_verdicts.csv", bucket_verdicts)
    _write_csv(output_dir / "candidate_stability_summary.csv", stability)
    _write_json(output_dir / "candidate_stability_summary.json", {row["candidate_id"]: row for row in stability})
    _write_json(output_dir / "decisionability_summary.json", summary)
    _write_multi_sample_report(
        output_dir / "validation_report.md",
        manifest_rows=manifest_rows,
        accepted_mass=accepted_mass,
        clean_only_mass=clean_only_mass,
        stability_rows=stability,
        caveated_sample_ids=caveated_sample_ids,
    )
    return summary


def run_quote_adjustment_replay(
    *,
    run_dir: Path,
    output_dir: Path,
    max_samples_per_candidate: int = DEFAULT_MAX_SAMPLES_PER_CANDIDATE,
) -> dict[str, Any]:
    run_dir = _expand(run_dir)
    output_dir = _expand(output_dir)
    paths = _artifact_paths(run_dir)
    for name, path in paths.items():
        if name == "stage6_summary":
            continue
        if not path.exists():
            raise FileNotFoundError(f"required input {name} not found: {path}")

    audit_rows, audit_headers = _read_csv_with_header(paths["audit_csv"])
    decisions = _decision_rows(audit_rows)
    label_rows = _read_csv(paths["execution_labels"])
    horizon_rows = _read_csv(paths["fill_horizon"])
    markout_rows = _read_csv(paths["fill_markout"])
    safety_rows = _read_csv(paths["stage5c_safety"])
    safety_by_seq = _stage5c_by_seq(safety_rows)
    maker_acceptance = _read_json(paths["maker_acceptance"]) if paths["maker_acceptance"].exists() else {"passed": False}
    stage8b_decision = _read_json(paths["stage8b_decision"]) if paths["stage8b_decision"].exists() else {}

    definitions = candidate_definitions()
    runtimes = build_candidate_runtimes(decisions=decisions, safety_by_seq=safety_by_seq, candidates=definitions)
    matrix_rows = _candidate_matrix_rows(definitions)
    metrics = [
        _metric_row(
            candidate,
            total_decisions=len(decisions),
            label_rows=label_rows,
            horizon_rows=horizon_rows,
            markout_rows=markout_rows,
        )
        for candidate in runtimes
    ]
    fill_quality_rows = _fill_quality_rows(runtimes, label_rows, horizon_rows, markout_rows)
    inventory_rows = _inventory_rows(runtimes, decisions, label_rows)
    api_churn_rows = _api_churn_rows(runtimes, decisions, label_rows)
    post_only_rows = _post_only_rows(runtimes, decisions, safety_by_seq)
    action_path_rows = _coverage_rows(
        candidates=runtimes,
        decisions=decisions,
        label_rows=label_rows,
        maker_acceptance=maker_acceptance,
    )
    bucket_evidence_rows = _candidate_bucket_evidence_rows(
        candidates=runtimes,
        metrics=metrics,
        api_rows=api_churn_rows,
        inventory_rows=inventory_rows,
        post_only_rows=post_only_rows,
        action_path_rows=action_path_rows,
        maker_acceptance=maker_acceptance,
    )
    bucket_verdict_rows = _candidate_bucket_verdict_rows(bucket_evidence_rows)
    audit_field_rows = _audit_field_coverage_rows(
        audit_headers,
        {
            "stage5_execution_outcome_labels": paths["execution_labels"],
            "stage5c_quote_anchor_safety": paths["stage5c_safety"],
            "stage8b_planning_decision": paths["stage8b_decision"],
            "stage6_final_calibration_summary": paths["stage6_summary"],
        },
    )
    samples = _candidate_decision_samples(runtimes, decisions, safety_by_seq, max_samples_per_candidate)
    missing_t006 = [row for row in audit_field_rows if row["metric_status"] == "required_t006_field" and row["status"] != "available"]
    classification, reasons = _classify(
        missing_t006_count=len(missing_t006),
        metrics=metrics,
        maker_acceptance=maker_acceptance,
    )

    summary = {
        "task_id": TASK_ID,
        "runner_mode": RUNNER_MODE,
        "classification": classification,
        "classification_reasons": reasons,
        "dataset": run_dir.name,
        "decision_rows": len(decisions),
        "submit_orders": len(label_rows),
        "candidate_count": len(definitions),
        "missing_t006_field_count": len(missing_t006),
        "maker_acceptance_passed": bool(maker_acceptance.get("passed", False)),
        "stage8b_conclusion": stage8b_decision.get("conclusion", ""),
        "bucket_verdict_count": len(bucket_verdict_rows),
        "bucket_verdicts": sorted({str(row["bucket_verdict"]) for row in bucket_verdict_rows}),
        "not_authorized": [
            "live",
            "default-on",
            "promotion",
            "sample expansion",
            "production behavior change",
            "Step5C promotion",
            "inventory-control implementation",
            "queue/touch repair",
        ],
    }

    _write_csv(output_dir / "candidate_matrix.csv", matrix_rows)
    _write_json(output_dir / "candidate_matrix.json", matrix_rows)
    _write_json(output_dir / "candidate_summary.json", summary)
    _write_csv(output_dir / "candidate_metrics.csv", metrics)
    _write_csv(output_dir / "fill_quality_by_candidate.csv", fill_quality_rows)
    _write_csv(output_dir / "inventory_cycle_metrics.csv", inventory_rows)
    _write_csv(output_dir / "api_churn_metrics.csv", api_churn_rows)
    _write_csv(output_dir / "post_only_safety_metrics.csv", post_only_rows)
    _write_csv(output_dir / "action_path_coverage.csv", action_path_rows)
    _write_csv(output_dir / "candidate_bucket_evidence.csv", bucket_evidence_rows)
    _write_csv(output_dir / "candidate_bucket_verdicts.csv", bucket_verdict_rows)
    _write_csv(output_dir / "audit_field_coverage.csv", audit_field_rows)
    _write_csv(output_dir / "candidate_decision_samples.csv", samples)
    _write_acceptance_decision(
        output_dir / "acceptance_decision.md",
        classification=classification,
        reasons=reasons,
        metrics=metrics,
        missing_t006_count=len(missing_t006),
    )

    manifest = {
        "task_id": TASK_ID,
        "runner_mode": RUNNER_MODE,
        "generated_at": _generated_at(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "classification": classification,
        "classification_reasons": reasons,
        "inputs": {
            name: {
                "path": str(path),
                "sha256": _hash_file(path) if path.exists() and path.is_file() else "",
            }
            for name, path in paths.items()
        },
        "outputs": [
            "run_manifest.json",
            "candidate_matrix.csv",
            "candidate_matrix.json",
            "candidate_summary.json",
            "candidate_metrics.csv",
            "fill_quality_by_candidate.csv",
            "inventory_cycle_metrics.csv",
            "api_churn_metrics.csv",
            "post_only_safety_metrics.csv",
            "action_path_coverage.csv",
            "candidate_bucket_evidence.csv",
            "candidate_bucket_verdicts.csv",
            "audit_field_coverage.csv",
            "candidate_decision_samples.csv",
            "acceptance_decision.md",
        ],
        "not_authorized": summary["not_authorized"],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Path to local_live_analysis/<run_id>")
    parser.add_argument("--output-dir", type=Path, help="Output directory for Stage 9B/9C artifacts")
    parser.add_argument(
        "--multi-sample-run-dir",
        type=Path,
        action="append",
        default=[],
        help="Path to a local_live_analysis/<run_id> directory to include in multi-sample validation",
    )
    parser.add_argument(
        "--task-id",
        default=TASK_ID,
        help="Task id suffix for task-scoped outputs in multi-sample mode",
    )
    parser.add_argument(
        "--caveated-sample-id",
        action="append",
        default=[],
        help="Sample id to include in accepted-set but exclude from clean-only sensitivity",
    )
    parser.add_argument(
        "--max-samples-per-candidate",
        type=int,
        default=DEFAULT_MAX_SAMPLES_PER_CANDIDATE,
        help="Maximum candidate decision samples to emit per candidate",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.multi_sample_run_dir:
        if args.output_dir is None:
            raise SystemExit("--output-dir is required in multi-sample mode")
        summary = run_multi_sample_validation(
            run_dirs=list(args.multi_sample_run_dir),
            output_dir=args.output_dir,
            task_id=str(args.task_id),
            caveated_sample_ids=set(args.caveated_sample_id or []),
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "classification": "multi_sample_decisionability_complete",
                    "candidate_count": len(summary["candidate_classifications"]),
                    "output_dir": str(_expand(args.output_dir)),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.run_dir is None or args.output_dir is None:
        raise SystemExit("--run-dir and --output-dir are required in single-sample mode")
    manifest = run_quote_adjustment_replay(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        max_samples_per_candidate=max(0, int(args.max_samples_per_candidate)),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "classification": manifest["classification"],
                "output_dir": manifest["output_dir"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
