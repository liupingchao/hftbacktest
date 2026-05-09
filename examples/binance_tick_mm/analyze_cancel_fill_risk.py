#!/usr/bin/env python3
"""Diagnose cancel-requested fill risk across live audit windows."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


MARKOUT_HORIZONS_NS = {
    "1s": 1_000_000_000,
    "5s": 5_000_000_000,
    "30s": 30_000_000_000,
}

FILL_EVENT_TYPES = {"fill", "partial_fill"}
TERMINAL_EVENT_TYPES = {"cancel_ack", "fill", "expired", "rejected"}
TERMINAL_STATUSES = {"canceled", "cancelled", "filled", "expired", "rejected"}
SOURCE_PATHS = [
    "same_side_readd_inventory_worsening",
    "same_side_readd_other",
    "inventory_worsening_no_readd",
    "inventory_reducing_cancel_race",
    "flat_or_unknown_cancel_race",
]


@dataclass
class DecisionPoint:
    ts_local: int
    mid: float
    position: float


@dataclass
class PendingCancel:
    order_id: str
    side: str
    cancel_ts: int
    qty: float
    had_same_side_readd: bool = False
    had_inventory_worsening_readd: bool = False


def _safe_int(raw: Any, default: int = 0) -> int:
    try:
        if raw is None or raw == "":
            return default
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        if raw is None or raw == "":
            return default
        return float(raw)
    except (TypeError, ValueError):
        return default


def _truthy(raw: Any) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "y"}


def _side_direction(side: str) -> int:
    side = side.strip().lower()
    if side == "buy":
        return 1
    if side == "sell":
        return -1
    return 0


def _row_side(row: dict[str, str]) -> str:
    return str(row.get("order_side") or "").strip().lower()


def _row_price(row: dict[str, str]) -> float:
    return _safe_float(row.get("fill_price")) or _safe_float(row.get("order_price"))


def _row_qty(row: dict[str, str]) -> float:
    return (
        _safe_float(row.get("fill_qty"))
        or _safe_float(row.get("order_executed_qty"))
        or _safe_float(row.get("order_qty"))
    )


def _is_fill(row: dict[str, str]) -> bool:
    return str(row.get("event_type") or "").strip().lower() in FILL_EVENT_TYPES


def _is_fill_after_cancel_request(row: dict[str, str]) -> bool:
    return _is_fill(row) and (_truthy(row.get("fill_after_cancel_request")) or _safe_int(row.get("cancel_request_ts")) > 0)


def _is_terminal(row: dict[str, str]) -> bool:
    event_type = str(row.get("event_type") or "").strip().lower()
    status = str(row.get("order_status") or "").strip().lower()
    return event_type in TERMINAL_EVENT_TYPES or status in TERMINAL_STATUSES


def _is_decision(row: dict[str, str]) -> bool:
    return str(row.get("event_type") or "").strip().lower() == "decision"


def _is_cancel_request(row: dict[str, str]) -> bool:
    event_type = str(row.get("event_type") or "").strip().lower()
    return event_type == "cancel_sent" or _truthy(row.get("cancel_requested")) or _safe_int(row.get("cancel_request_ts")) > 0


def _is_order_submit(row: dict[str, str]) -> bool:
    event_type = str(row.get("event_type") or "").strip().lower()
    action = str(row.get("action") or "").strip().lower()
    return event_type == "order_submit_sent" or action in {"submit_buy", "submit_sell"}


def _add_side_readd(side: str, position: float) -> bool:
    return (side == "buy" and position >= 0.0) or (side == "sell" and position <= 0.0)


def _previous_decision(decisions: list[DecisionPoint], ts_local: int) -> DecisionPoint | None:
    idx = bisect.bisect_right([point.ts_local for point in decisions], ts_local) - 1
    if idx < 0:
        return None
    return decisions[idx]


def _future_mid(
    decision_ts: list[int],
    decisions: list[DecisionPoint],
    ts_local: int,
    horizon_ns: int,
) -> float | None:
    idx = bisect.bisect_left(decision_ts, ts_local + horizon_ns)
    if idx >= len(decisions):
        return None
    return decisions[idx].mid


def _load_rows(audit_csv: Path) -> list[dict[str, str]]:
    with audit_csv.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _decision_points(rows: list[dict[str, str]]) -> list[DecisionPoint]:
    decisions: list[DecisionPoint] = []
    for row in rows:
        if not _is_decision(row):
            continue
        ts_local = _safe_int(row.get("ts_local"))
        mid = _safe_float(row.get("mid"))
        if ts_local <= 0 or mid <= 0.0:
            continue
        decisions.append(
            DecisionPoint(
                ts_local=ts_local,
                mid=mid,
                position=_safe_float(row.get("position")),
            )
        )
    decisions.sort(key=lambda point: point.ts_local)
    return decisions


def _maker_acceptance_status(run_dir: Path) -> tuple[bool | None, str]:
    candidates = [
        run_dir / "maker_acceptance_stage6i.json",
        run_dir / "maker_acceptance.json",
        *sorted(run_dir.glob("maker_acceptance*.json")),
    ]
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        passed = data.get("passed")
        if isinstance(passed, bool):
            return passed, str(path)
    return None, ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * pct))))
    return float(ordered[idx])


def _cancel_latency_bucket(cancel_to_fill_ms: float) -> str:
    if cancel_to_fill_ms <= 0.0:
        return "unknown"
    if cancel_to_fill_ms <= 10.0:
        return "le_10ms"
    if cancel_to_fill_ms <= 50.0:
        return "10_50ms"
    return "gt_50ms"


def _source_path(
    *,
    same_side_readd: bool,
    inventory_worsening_readd: bool,
    worsened_inventory: bool,
    abs_position_before: float,
    abs_position_after: float,
) -> str:
    if same_side_readd and (inventory_worsening_readd or worsened_inventory):
        return "same_side_readd_inventory_worsening"
    if same_side_readd:
        return "same_side_readd_other"
    if worsened_inventory:
        return "inventory_worsening_no_readd"
    if abs_position_after < abs_position_before:
        return "inventory_reducing_cancel_race"
    return "flat_or_unknown_cancel_race"


def _guard_candidate_path(source_path: str) -> str:
    if source_path in {
        "same_side_readd_inventory_worsening",
        "same_side_readd_other",
        "inventory_worsening_no_readd",
    }:
        return "add_side_guard_candidate"
    if source_path == "inventory_reducing_cancel_race":
        return "adverse_selection_candidate"
    return "unknown_candidate"


def _new_path_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {"count": 0, "notional": 0.0, "worsening_count": 0}
    for horizon_name in MARKOUT_HORIZONS_NS:
        metrics[f"adverse_markout_{horizon_name}_count"] = 0
        metrics[f"weighted_markout_{horizon_name}_sum"] = 0.0
        metrics[f"adverse_weighted_markout_{horizon_name}_sum"] = 0.0
    return metrics


def _update_path_metrics(metrics: dict[str, Any], event: dict[str, Any]) -> None:
    metrics["count"] += 1
    metrics["notional"] += float(event.get("fill_notional") or 0.0)
    metrics["worsening_count"] += int(event.get("worsened_inventory") or 0)
    qty = float(event.get("fill_qty") or 0.0)
    for horizon_name in MARKOUT_HORIZONS_NS:
        raw_markout = event.get(f"markout_{horizon_name}")
        if raw_markout in {"", None}:
            continue
        markout = float(raw_markout)
        weighted = markout * qty
        metrics[f"weighted_markout_{horizon_name}_sum"] += weighted
        if markout < 0.0:
            metrics[f"adverse_markout_{horizon_name}_count"] += 1
            metrics[f"adverse_weighted_markout_{horizon_name}_sum"] += weighted


def analyze_audit_csv(
    audit_csv: Path,
    *,
    run_id: str | None = None,
    sample_class: str | None = None,
) -> dict[str, Any]:
    rows = _load_rows(audit_csv)
    decisions = _decision_points(rows)
    decision_ts = [point.ts_local for point in decisions]
    pending: dict[str, PendingCancel] = {}
    events: list[dict[str, Any]] = []
    same_side_readd_count = 0
    same_side_readd_qty = 0.0
    inventory_worsening_readd_count = 0
    same_side_readd_then_cancel_fill_count = 0
    total_fill_count = 0
    total_fill_qty = 0.0
    total_fill_notional = 0.0
    fill_after_cancel_qty = 0.0
    fill_after_cancel_notional = 0.0
    worsening_fill_count = 0
    worsening_fill_notional = 0.0
    cancel_to_fill_ms: list[float] = []
    cancel_fill_abs_position_after: list[float] = []
    markouts: dict[str, list[float]] = {name: [] for name in MARKOUT_HORIZONS_NS}
    weighted_markouts: dict[str, float] = {name: 0.0 for name in MARKOUT_HORIZONS_NS}
    by_side: dict[str, dict[str, float]] = {
        "buy": {"count": 0, "qty": 0.0, "notional": 0.0},
        "sell": {"count": 0, "qty": 0.0, "notional": 0.0},
    }
    source_path_metrics = {path: _new_path_metrics() for path in SOURCE_PATHS}
    latency_bucket_metrics: dict[str, dict[str, Any]] = {
        "le_10ms": _new_path_metrics(),
        "10_50ms": _new_path_metrics(),
        "gt_50ms": _new_path_metrics(),
        "unknown": _new_path_metrics(),
    }
    last_position = 0.0

    resolved_run_id = run_id or audit_csv.parent.name
    acceptance_source = ""
    if sample_class is None:
        acceptance, acceptance_source = _maker_acceptance_status(audit_csv.parent)
        sample_class = "current_format" if acceptance is True else "historical_or_failed_gate"

    for row in rows:
        ts_local = _safe_int(row.get("ts_local"))
        event_type = str(row.get("event_type") or "").strip().lower()
        order_id = str(row.get("order_id") or row.get("linked_order_id") or "").strip()
        side = _row_side(row)
        qty = _row_qty(row)
        price = _row_price(row)

        if _is_decision(row):
            last_position = _safe_float(row.get("position"), last_position)

        if order_id and side in {"buy", "sell"} and _is_order_submit(row):
            matching_pending = [order for order in pending.values() if order.side == side]
            if matching_pending:
                same_side_readd_count += 1
                same_side_readd_qty += qty
                worsening_readd = _add_side_readd(side, last_position)
                if worsening_readd:
                    inventory_worsening_readd_count += 1
                for order in matching_pending:
                    order.had_same_side_readd = True
                    order.had_inventory_worsening_readd = order.had_inventory_worsening_readd or worsening_readd

        if order_id and side in {"buy", "sell"} and _is_cancel_request(row):
            cancel_ts = _safe_int(row.get("cancel_request_ts")) or ts_local
            pending.setdefault(
                order_id,
                PendingCancel(order_id=order_id, side=side, cancel_ts=cancel_ts, qty=qty),
            )

        if _is_fill(row):
            total_fill_count += 1
            total_fill_qty += qty
            total_fill_notional += qty * price

        if _is_fill_after_cancel_request(row):
            direction = _side_direction(side)
            signed_qty = direction * qty
            position_after = _safe_float(row.get("position"), last_position + signed_qty)
            position_before = position_after - signed_qty
            fill_notional = qty * price
            cancel_request_ts = _safe_int(row.get("cancel_request_ts"))
            pending_order = pending.get(order_id)
            if cancel_request_ts <= 0 and pending_order is not None:
                cancel_request_ts = pending_order.cancel_ts
            same_side_readd_before_fill = bool(pending_order.had_same_side_readd if pending_order else False)
            inventory_worsening_readd_before_fill = bool(
                pending_order.had_inventory_worsening_readd if pending_order else False
            )
            if same_side_readd_before_fill:
                same_side_readd_then_cancel_fill_count += 1
            worsened_inventory = abs(position_after) > abs(position_before)
            if worsened_inventory:
                worsening_fill_count += 1
                worsening_fill_notional += fill_notional
            cancel_to_fill_value_ms = (
                (ts_local - cancel_request_ts) / 1_000_000.0 if cancel_request_ts > 0 else 0.0
            )
            if cancel_request_ts > 0 and ts_local > 0:
                cancel_to_fill_ms.append(cancel_to_fill_value_ms)
            fill_after_cancel_qty += qty
            fill_after_cancel_notional += fill_notional
            cancel_fill_abs_position_after.append(abs(position_after))
            if side in by_side:
                by_side[side]["count"] += 1
                by_side[side]["qty"] += qty
                by_side[side]["notional"] += fill_notional

            prior_decision = _previous_decision(decisions, ts_local)
            row_mid = _safe_float(row.get("mid")) or (prior_decision.mid if prior_decision is not None else 0.0)
            source_path = _source_path(
                same_side_readd=same_side_readd_before_fill,
                inventory_worsening_readd=inventory_worsening_readd_before_fill,
                worsened_inventory=worsened_inventory,
                abs_position_before=abs(position_before),
                abs_position_after=abs(position_after),
            )
            latency_bucket = _cancel_latency_bucket(cancel_to_fill_value_ms)
            event: dict[str, Any] = {
                "run_id": resolved_run_id,
                "sample_class": sample_class,
                "ts_local": ts_local,
                "order_id": order_id,
                "order_side": side,
                "event_type": event_type,
                "order_status": row.get("order_status", ""),
                "fill_price": price,
                "fill_qty": qty,
                "fill_notional": fill_notional,
                "cancel_request_ts": cancel_request_ts,
                "cancel_to_fill_ms": cancel_to_fill_value_ms,
                "cancel_latency_bucket": latency_bucket,
                "source_path": source_path,
                "guard_candidate_path": _guard_candidate_path(source_path),
                "position_before": position_before,
                "position_after": position_after,
                "abs_position_before": abs(position_before),
                "abs_position_after": abs(position_after),
                "worsened_inventory": int(worsened_inventory),
                "same_side_readd_before_fill": int(same_side_readd_before_fill),
                "inventory_worsening_readd_before_fill": int(inventory_worsening_readd_before_fill),
                "mid_at_fill": row_mid,
            }
            for horizon_name, horizon_ns in MARKOUT_HORIZONS_NS.items():
                future_mid = _future_mid(decision_ts, decisions, ts_local, horizon_ns)
                if future_mid is None or price <= 0.0 or direction == 0:
                    event[f"future_mid_{horizon_name}"] = ""
                    event[f"markout_{horizon_name}"] = ""
                    event[f"weighted_markout_{horizon_name}"] = ""
                    continue
                markout = direction * (future_mid - price)
                weighted = markout * qty
                markouts[horizon_name].append(markout)
                weighted_markouts[horizon_name] += weighted
                event[f"future_mid_{horizon_name}"] = future_mid
                event[f"markout_{horizon_name}"] = markout
                event[f"weighted_markout_{horizon_name}"] = weighted
            _update_path_metrics(source_path_metrics[source_path], event)
            _update_path_metrics(latency_bucket_metrics[latency_bucket], event)
            events.append(event)

        if order_id and _is_terminal(row):
            pending.pop(order_id, None)

    fill_after_cancel_count = len(events)
    summary: dict[str, Any] = {
        "run_id": resolved_run_id,
        "audit_csv": str(audit_csv),
        "sample_class": sample_class,
        "acceptance_source": acceptance_source,
        "row_count": len(rows),
        "decision_count": len(decisions),
        "total_fill_count": total_fill_count,
        "total_fill_qty": total_fill_qty,
        "total_fill_notional": total_fill_notional,
        "fill_after_cancel_request_count": fill_after_cancel_count,
        "fill_after_cancel_request_qty": fill_after_cancel_qty,
        "fill_after_cancel_request_notional": fill_after_cancel_notional,
        "fill_after_cancel_request_count_rate": fill_after_cancel_count / total_fill_count if total_fill_count else 0.0,
        "fill_after_cancel_request_notional_rate": fill_after_cancel_notional / total_fill_notional if total_fill_notional else 0.0,
        "fill_after_cancel_request_buy_count": int(by_side["buy"]["count"]),
        "fill_after_cancel_request_sell_count": int(by_side["sell"]["count"]),
        "fill_after_cancel_request_buy_notional": by_side["buy"]["notional"],
        "fill_after_cancel_request_sell_notional": by_side["sell"]["notional"],
        "same_side_readd_while_cancel_requested_count": same_side_readd_count,
        "same_side_readd_while_cancel_requested_qty": same_side_readd_qty,
        "inventory_worsening_readd_while_cancel_requested_count": inventory_worsening_readd_count,
        "same_side_readd_then_cancel_fill_count": same_side_readd_then_cancel_fill_count,
        "worsening_fill_after_cancel_request_count": worsening_fill_count,
        "worsening_fill_after_cancel_request_notional": worsening_fill_notional,
        "cancel_to_fill_latency_ms_p50": float(median(cancel_to_fill_ms)) if cancel_to_fill_ms else 0.0,
        "cancel_to_fill_latency_ms_p90": _percentile(cancel_to_fill_ms, 0.90),
        "cancel_to_fill_latency_ms_max": max(cancel_to_fill_ms) if cancel_to_fill_ms else 0.0,
        "max_abs_position_after_cancel_requested_fill": max(cancel_fill_abs_position_after)
        if cancel_fill_abs_position_after
        else 0.0,
    }
    for horizon_name in MARKOUT_HORIZONS_NS:
        values = markouts[horizon_name]
        summary[f"markout_{horizon_name}_count"] = len(values)
        summary[f"markout_{horizon_name}_median"] = float(median(values)) if values else 0.0
        summary[f"markout_{horizon_name}_min"] = min(values) if values else 0.0
        summary[f"weighted_markout_{horizon_name}_sum"] = weighted_markouts[horizon_name]
    for source_path, metrics in source_path_metrics.items():
        prefix = f"source_path_{source_path}"
        summary[f"{prefix}_count"] = int(metrics["count"])
        summary[f"{prefix}_notional"] = float(metrics["notional"])
        summary[f"{prefix}_worsening_count"] = int(metrics["worsening_count"])
        for horizon_name in MARKOUT_HORIZONS_NS:
            summary[f"{prefix}_adverse_markout_{horizon_name}_count"] = int(
                metrics[f"adverse_markout_{horizon_name}_count"]
            )
            summary[f"{prefix}_weighted_markout_{horizon_name}_sum"] = float(
                metrics[f"weighted_markout_{horizon_name}_sum"]
            )
            summary[f"{prefix}_adverse_weighted_markout_{horizon_name}_sum"] = float(
                metrics[f"adverse_weighted_markout_{horizon_name}_sum"]
            )
    add_side_candidate_count = sum(
        int(source_path_metrics[path]["count"])
        for path in [
            "same_side_readd_inventory_worsening",
            "same_side_readd_other",
            "inventory_worsening_no_readd",
        ]
    )
    adverse_selection_candidate_count = int(source_path_metrics["inventory_reducing_cancel_race"]["count"])
    summary["guard_candidate_add_side_count"] = add_side_candidate_count
    summary["guard_candidate_adverse_selection_count"] = adverse_selection_candidate_count
    for horizon_name in MARKOUT_HORIZONS_NS:
        summary[f"guard_candidate_add_side_weighted_markout_{horizon_name}_sum"] = sum(
            float(source_path_metrics[path][f"weighted_markout_{horizon_name}_sum"])
            for path in [
                "same_side_readd_inventory_worsening",
                "same_side_readd_other",
                "inventory_worsening_no_readd",
            ]
        )
        summary[f"guard_candidate_adverse_selection_weighted_markout_{horizon_name}_sum"] = float(
            source_path_metrics["inventory_reducing_cancel_race"][f"weighted_markout_{horizon_name}_sum"]
        )
    for bucket, metrics in latency_bucket_metrics.items():
        prefix = f"cancel_latency_bucket_{bucket}"
        summary[f"{prefix}_count"] = int(metrics["count"])
        summary[f"{prefix}_notional"] = float(metrics["notional"])
        for horizon_name in MARKOUT_HORIZONS_NS:
            summary[f"{prefix}_adverse_markout_{horizon_name}_count"] = int(
                metrics[f"adverse_markout_{horizon_name}_count"]
            )
            summary[f"{prefix}_weighted_markout_{horizon_name}_sum"] = float(
                metrics[f"weighted_markout_{horizon_name}_sum"]
            )

    return {"summary": summary, "events": events}


def _find_audit_csv(local_root: Path, run_id: str) -> Path:
    run_dir = local_root / run_id
    exact = run_dir / f"audit_live_{run_id}.csv"
    if exact.exists():
        return exact
    matches = sorted(run_dir.glob("audit_live*.csv"))
    if not matches:
        raise FileNotFoundError(f"no audit_live*.csv found under {run_dir}")
    return matches[0]


def _cross_window_decision(summaries: list[dict[str, Any]]) -> str:
    current = [row for row in summaries if row.get("sample_class") == "current_format"]
    cancel_fill_material = [
        row
        for row in current
        if float(row.get("fill_after_cancel_request_notional_rate", 0.0)) >= 0.10
        or int(row.get("worsening_fill_after_cancel_request_count", 0)) > 0
    ]
    same_side_overlap = [
        row
        for row in current
        if int(row.get("same_side_readd_then_cancel_fill_count", 0)) > 0
    ]
    if len(current) < 2:
        return "collect_more_current_format_data"
    if len(cancel_fill_material) >= 2 and len(same_side_overlap) >= 2:
        return "proceed_to_stage6j_narrow_rule"
    if len(cancel_fill_material) >= 2:
        return "diagnose_cancel_fill_risk_before_strategy_rule"
    return "do_not_implement_rule_continue_stage6k"


def _write_markdown_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    decision = _cross_window_decision(summaries)
    current_count = sum(1 for row in summaries if row.get("sample_class") == "current_format")
    lines = [
        "# Stage 6I Cancel-Requested Fill-Risk Summary",
        "",
        f"- Decision: `{decision}`",
        f"- Current-format sample count: `{current_count}`",
        f"- Total processed samples: `{len(summaries)}`",
        "",
        "## Per-Run Metrics",
        "",
        "| run | class | fills | cancel-fill count | cancel-fill notional rate | same-side readd | readd then cancel-fill | worsening cancel-fill | p90 cancel-fill ms | max abs pos after |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            "| {run_id} | {sample_class} | {total_fill_count} | {fill_after_cancel_request_count} | "
            "{fill_after_cancel_request_notional_rate:.6f} | {same_side_readd_while_cancel_requested_count} | "
            "{same_side_readd_then_cancel_fill_count} | {worsening_fill_after_cancel_request_count} | "
            "{cancel_to_fill_latency_ms_p90:.3f} | {max_abs_position_after_cancel_requested_fill:.6f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Guard Candidate Split",
            "",
            "| run | class | add-side candidate count | add-side weighted 1s | adverse-selection count | adverse-selection weighted 1s |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        lines.append(
            "| {run_id} | {sample_class} | {add_count} | {add_w1:.6f} | {adv_count} | {adv_w1:.6f} |".format(
                run_id=row["run_id"],
                sample_class=row["sample_class"],
                add_count=int(row.get("guard_candidate_add_side_count") or 0),
                add_w1=float(row.get("guard_candidate_add_side_weighted_markout_1s_sum") or 0.0),
                adv_count=int(row.get("guard_candidate_adverse_selection_count") or 0),
                adv_w1=float(row.get("guard_candidate_adverse_selection_weighted_markout_1s_sum") or 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## Source-Path Attribution",
            "",
            "| run | class | path | count | adverse 1s | weighted 1s | adverse 5s | weighted 5s | adverse 30s | weighted 30s |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        for source_path in SOURCE_PATHS:
            count = int(row.get(f"source_path_{source_path}_count") or 0)
            if count <= 0:
                continue
            lines.append(
                "| {run_id} | {sample_class} | {source_path} | {count} | {adv1} | {w1:.6f} | {adv5} | {w5:.6f} | {adv30} | {w30:.6f} |".format(
                    run_id=row["run_id"],
                    sample_class=row["sample_class"],
                    source_path=source_path,
                    count=count,
                    adv1=int(row.get(f"source_path_{source_path}_adverse_markout_1s_count") or 0),
                    w1=float(row.get(f"source_path_{source_path}_weighted_markout_1s_sum") or 0.0),
                    adv5=int(row.get(f"source_path_{source_path}_adverse_markout_5s_count") or 0),
                    w5=float(row.get(f"source_path_{source_path}_weighted_markout_5s_sum") or 0.0),
                    adv30=int(row.get(f"source_path_{source_path}_adverse_markout_30s_count") or 0),
                    w30=float(row.get(f"source_path_{source_path}_weighted_markout_30s_sum") or 0.0),
                )
            )

    lines.extend(
        [
            "",
            "## Cancel-Latency Buckets",
            "",
            "| run | class | bucket | count | adverse 1s | weighted 1s | adverse 5s | weighted 5s |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summaries:
        for bucket in ["le_10ms", "10_50ms", "gt_50ms", "unknown"]:
            count = int(row.get(f"cancel_latency_bucket_{bucket}_count") or 0)
            if count <= 0:
                continue
            lines.append(
                "| {run_id} | {sample_class} | {bucket} | {count} | {adv1} | {w1:.6f} | {adv5} | {w5:.6f} |".format(
                    run_id=row["run_id"],
                    sample_class=row["sample_class"],
                    bucket=bucket,
                    count=count,
                    adv1=int(row.get(f"cancel_latency_bucket_{bucket}_adverse_markout_1s_count") or 0),
                    w1=float(row.get(f"cancel_latency_bucket_{bucket}_weighted_markout_1s_sum") or 0.0),
                    adv5=int(row.get(f"cancel_latency_bucket_{bucket}_adverse_markout_5s_count") or 0),
                    w5=float(row.get(f"cancel_latency_bucket_{bucket}_weighted_markout_5s_sum") or 0.0),
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `current_format` samples have a passing `maker_acceptance.json` beside the audit CSV.",
            "- `historical_or_failed_gate` samples are diagnostic/control evidence only.",
            "- Stage 6J should only proceed when the issue repeats across at least two current-format windows.",
            "- A narrow same-side rule additionally requires repeated same-side re-add overlap, not just cancel-fill frequency.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def run_analysis(
    *,
    run_ids: list[str],
    audit_csvs: list[Path],
    local_root: Path,
    out_dir: Path,
) -> dict[str, Any]:
    inputs: list[tuple[str, Path]] = []
    for run_id in run_ids:
        inputs.append((run_id, _find_audit_csv(local_root, run_id)))
    for path in audit_csvs:
        inputs.append((path.parent.name, path))

    summaries: list[dict[str, Any]] = []
    for run_id, audit_csv in inputs:
        result = analyze_audit_csv(audit_csv, run_id=run_id)
        run_out = out_dir / run_id
        _write_csv(run_out / "cancel_fill_events.csv", result["events"])
        (run_out / "cancel_fill_summary.json").write_text(
            json.dumps(result["summary"], indent=2, ensure_ascii=True) + "\n"
        )
        summaries.append(result["summary"])

    _write_csv(out_dir / "stage6i_cancel_fill_summary.csv", summaries)
    (out_dir / "stage6i_cancel_fill_summary.json").write_text(
        json.dumps(
            {
                "decision": _cross_window_decision(summaries),
                "summaries": summaries,
            },
            indent=2,
            ensure_ascii=True,
        )
        + "\n"
    )
    _write_markdown_report(out_dir / "STAGE6I_CANCEL_FILL_RISK_SUMMARY.md", summaries)
    return {"decision": _cross_window_decision(summaries), "summaries": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze cancel-requested fill risk from live audit CSVs")
    parser.add_argument("--run-id", action="append", default=[], help="Run id under --local-root")
    parser.add_argument("--audit-csv", action="append", default=[], help="Direct audit_live*.csv path")
    parser.add_argument("--local-root", default="local_live_analysis")
    parser.add_argument("--out-dir", default="local_live_analysis/stage6i_cancel_fill_risk")
    args = parser.parse_args()

    audit_csvs = [Path(path).expanduser().resolve() for path in args.audit_csv]
    if not args.run_id and not audit_csvs:
        parser.error("provide at least one --run-id or --audit-csv")

    result = run_analysis(
        run_ids=list(args.run_id),
        audit_csvs=audit_csvs,
        local_root=Path(args.local_root).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
    )
    print(json.dumps({"decision": result["decision"], "runs": len(result["summaries"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
