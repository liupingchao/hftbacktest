#!/usr/bin/env python3
"""Read-only replay lifecycle mismatch diagnosis runner."""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
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
RESIDUAL_TASK_ID = "0515T004"
DEFAULT_RESIDUAL_WINDOW_MS = 50.0
RESIDUAL_SUPPORT_WINDOWS_MS = (10, 25, 50)


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


def _raw_market_gzip(run_dir: Path) -> Path:
    candidates = sorted((run_dir / "raw_market_data").glob("*.gz"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one raw market gzip under {run_dir / 'raw_market_data'}, found {len(candidates)}")
    return candidates[0]


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_top5_sidecar(path: Path) -> list[dict[str, str]]:
    rows = _load_csv_rows(path)
    filtered: list[dict[str, str]] = []
    for row in rows:
        if str(row.get("event_type", "")).strip() != "depthUpdate":
            continue
        if str(row.get("sync_aligned", "")).strip().lower() != "true":
            continue
        try:
            row["_local_ts_int"] = int(row.get("local_ts", "0") or "0")
        except ValueError:
            continue
        filtered.append(row)
    filtered.sort(key=lambda row: int(row["_local_ts_int"]))
    return filtered


def _asof_sidecar_row(rows: list[dict[str, str]], ts_local: int) -> dict[str, str] | None:
    ts_values = [int(row["_local_ts_int"]) for row in rows]
    idx = bisect.bisect_right(ts_values, int(ts_local)) - 1
    return rows[idx] if idx >= 0 else None


def _load_raw_market_events(
    raw_gzip: Path,
    *,
    start_ts_local: int,
    end_ts_local: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with gzip.open(raw_gzip, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw_ts_text, payload = line.split(" ", 1)
            raw_ts = int(raw_ts_text)
            if raw_ts < start_ts_local:
                continue
            if raw_ts > end_ts_local:
                break
            message = json.loads(payload)
            stream = str(message.get("stream", ""))
            data = message.get("data", {})
            event_type = str(data.get("e", ""))
            events.append(
                {
                    "raw_local_ts": raw_ts,
                    "stream": stream,
                    "event_type": event_type,
                    "data": data,
                }
            )
    return events


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


def _supportive_trade_for_case(event: dict[str, Any], *, side: str, order_price: float) -> bool:
    if event.get("event_type") != "trade":
        return False
    data = event.get("data", {})
    try:
        trade_price = float(data.get("p", "nan"))
    except ValueError:
        return False
    buyer_is_maker = bool(data.get("m"))
    if side == "buy":
        return buyer_is_maker and trade_price <= float(order_price)
    if side == "sell":
        return (not buyer_is_maker) and trade_price >= float(order_price)
    return False


def _best_supportive_trade_stats(
    supportive_trades: list[dict[str, Any]],
    *,
    anchor_ts: int,
) -> tuple[int, float, float, int]:
    if not supportive_trades:
        return 0, math.nan, math.nan, 0
    before_or_at = [row for row in supportive_trades if int(row["raw_local_ts"]) <= anchor_ts]
    if not before_or_at:
        return 0, math.nan, math.nan, 0
    nearest = max(before_or_at, key=lambda row: int(row["raw_local_ts"]))
    delay_ms = (anchor_ts - int(nearest["raw_local_ts"])) / 1_000_000.0
    return (
        len(before_or_at),
        float(nearest["trade_price"]),
        delay_ms,
        int(nearest["raw_local_ts"]),
    )


def _window_support_counts(
    supportive_trades: list[dict[str, Any]],
    *,
    anchor_ts: int,
    windows_ms: tuple[int, ...] = RESIDUAL_SUPPORT_WINDOWS_MS,
) -> dict[str, int]:
    out: dict[str, int] = {}
    for window_ms in windows_ms:
        lower = int(anchor_ts) - int(window_ms) * 1_000_000
        out[f"supportive_trade_count_{window_ms}ms"] = sum(
            1 for row in supportive_trades if lower <= int(row["raw_local_ts"]) <= int(anchor_ts)
        )
    return out


def _classify_residual_trigger(case_row: dict[str, Any]) -> tuple[str, str]:
    case_label = str(case_row.get("case_label", ""))
    if case_label == "live_filled_replay_canceled":
        cancel_delay_ms = _float(case_row.get("live_cancel_to_fill_delay_ms"))
        supportive_10ms = int(case_row.get("live_supportive_trade_count_10ms", 0) or 0)
        if _finite(cancel_delay_ms) and cancel_delay_ms <= 25.0 and supportive_10ms > 0:
            return (
                "cancel_race_window_too_short",
                "live filled shortly after cancel request with dense supportive trades near the fill, while replay terminalized to cancel",
            )
        return (
            "residual_cancel_race_miss_uncertain",
            "live filled after cancel request but the current evidence is not yet enough to distinguish race-window miss from another replay-side omission",
        )
    if case_label == "live_canceled_replay_filled":
        replay_supportive_10ms = int(case_row.get("replay_supportive_trade_count_10ms", 0) or 0)
        replay_supportive_50ms = int(case_row.get("replay_supportive_trade_count_50ms", 0) or 0)
        cancel_lead_ms = _float(case_row.get("live_cancel_after_replay_fill_ms"))
        if replay_supportive_10ms > 0 and _finite(cancel_lead_ms) and cancel_lead_ms >= 100.0:
            return (
                "touch_fill_assumption_too_optimistic",
                "replay filled on supportive trades while live remained working for much longer and later canceled, which points more to optimistic touch-fill / queue proxy than cancel timing",
            )
        if replay_supportive_50ms > 0:
            return (
                "queue_exposure_proxy_bias_possible",
                "replay saw supportive trades before live canceled, but current evidence does not isolate whether the issue is touch optimism or submit-after-queue exposure approximation bias",
            )
        return (
            "residual_replay_fill_trigger_uncertain",
            "replay filled without enough nearby supportive-trade evidence to identify whether the issue is queue exposure approximation or another replay-side trigger",
        )
    return ("not_residual_case", "case is not part of the residual mismatch set")


def build_residual_case_diagnosis(
    *,
    state_diff_rows: list[dict[str, Any]],
    live_audit_rows: list[dict[str, str]],
    replay_audit_rows: list[dict[str, str]],
    top5_rows: list[dict[str, str]],
    raw_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    residual_rows = [
        row for row in state_diff_rows
        if row["case_label"] in {"live_filled_replay_canceled", "live_canceled_replay_filled"}
    ]
    live_by_order_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    replay_by_order_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in live_audit_rows:
        oid = str(row.get("order_id", "") or "").strip()
        if oid:
            live_by_order_id[oid].append(row)
    for row in replay_audit_rows:
        oid = str(row.get("order_id", "") or "").strip()
        if oid:
            replay_by_order_id[oid].append(row)

    results: list[dict[str, Any]] = []
    for row in residual_rows:
        order_side = str(row.get("order_side", "") or "")
        live_order_id = str(row.get("live_order_id", "") or row.get("replay_order_id", "") or "")
        replay_order_id = str(row.get("replay_order_id", "") or row.get("live_order_id", "") or "")
        order_price = math.nan
        live_submit_ts = _float(row.get("live_submit_ts_local"))
        replay_submit_ts = _float(row.get("replay_submit_ts_local"))
        live_cancel_req_ts = math.nan
        live_fill_ts = math.nan
        live_terminal_ts = math.nan
        replay_fill_ts = math.nan
        replay_cancel_req_ts = math.nan
        replay_terminal_ts = math.nan

        for domain_rows, is_live in ((live_by_order_id.get(live_order_id, []), True), (replay_by_order_id.get(replay_order_id, []), False)):
            for event_row in domain_rows:
                evt = str(event_row.get("event_type", "") or "")
                if not _finite(order_price):
                    order_price = _float(event_row.get("order_price"))
                if evt == "cancel_sent":
                    if is_live:
                        live_cancel_req_ts = _float(event_row.get("cancel_request_ts")) or _float(event_row.get("ts_local"))
                    else:
                        replay_cancel_req_ts = _float(event_row.get("cancel_request_ts")) or _float(event_row.get("ts_local"))
                if evt == "fill":
                    if is_live:
                        live_fill_ts = _float(event_row.get("ts_local"))
                    else:
                        replay_fill_ts = _float(event_row.get("ts_local"))
                if evt in {"fill", "cancel_ack", "expired", "rejected"}:
                    if is_live:
                        live_terminal_ts = max(_float(event_row.get("ts_local")), live_terminal_ts)
                    else:
                        replay_terminal_ts = max(_float(event_row.get("ts_local")), replay_terminal_ts)

        anchor_points = [
            value for value in (
                live_submit_ts,
                replay_submit_ts,
                live_cancel_req_ts,
                live_fill_ts,
                replay_fill_ts,
                live_terminal_ts,
                replay_terminal_ts,
            ) if _finite(value)
        ]
        if not anchor_points:
            continue
        window_start = int(min(anchor_points)) - int(DEFAULT_RESIDUAL_WINDOW_MS * 1_000_000)
        window_end = int(max(anchor_points)) + int(DEFAULT_RESIDUAL_WINDOW_MS * 1_000_000)
        raw_window = [event for event in raw_events if window_start <= int(event["raw_local_ts"]) <= window_end]
        supportive_trades = []
        for event in raw_window:
            if _supportive_trade_for_case(event, side=order_side, order_price=order_price):
                data = event["data"]
                supportive_trades.append(
                    {
                        "raw_local_ts": int(event["raw_local_ts"]),
                        "trade_price": float(data.get("p")),
                        "trade_qty": float(data.get("q")),
                        "buyer_is_maker": int(bool(data.get("m"))),
                    }
                )

        live_support_count, live_nearest_px, live_nearest_delay_ms, live_nearest_ts = _best_supportive_trade_stats(
            supportive_trades,
            anchor_ts=int(live_fill_ts) if _finite(live_fill_ts) else int(live_terminal_ts),
        )
        replay_support_count, replay_nearest_px, replay_nearest_delay_ms, replay_nearest_ts = _best_supportive_trade_stats(
            supportive_trades,
            anchor_ts=int(replay_fill_ts) if _finite(replay_fill_ts) else int(replay_terminal_ts),
        )
        sidecar_at_submit = _asof_sidecar_row(top5_rows, int(live_submit_ts))
        sidecar_at_live_terminal = _asof_sidecar_row(top5_rows, int(live_terminal_ts)) if _finite(live_terminal_ts) else None
        sidecar_at_replay_fill = _asof_sidecar_row(top5_rows, int(replay_fill_ts)) if _finite(replay_fill_ts) else None

        result = {
            "submit_key": row["submit_key"],
            "case_label": row["case_label"],
            "order_side": order_side,
            "live_order_id": live_order_id,
            "replay_order_id": replay_order_id,
            "order_price": order_price,
            "live_submit_ts_local": int(live_submit_ts) if _finite(live_submit_ts) else "",
            "replay_submit_ts_local": int(replay_submit_ts) if _finite(replay_submit_ts) else "",
            "live_cancel_request_ts_local": int(live_cancel_req_ts) if _finite(live_cancel_req_ts) else "",
            "replay_cancel_request_ts_local": int(replay_cancel_req_ts) if _finite(replay_cancel_req_ts) else "",
            "live_fill_ts_local": int(live_fill_ts) if _finite(live_fill_ts) else "",
            "replay_fill_ts_local": int(replay_fill_ts) if _finite(replay_fill_ts) else "",
            "live_terminal_ts_local": int(live_terminal_ts) if _finite(live_terminal_ts) else "",
            "replay_terminal_ts_local": int(replay_terminal_ts) if _finite(replay_terminal_ts) else "",
            "live_cancel_to_fill_delay_ms": row.get("live_cancel_to_fill_delay_ms", ""),
            "replay_time_to_fill_ms": row.get("replay_time_to_fill_ms", ""),
            "live_supportive_trade_count_before_anchor": live_support_count,
            "live_nearest_supportive_trade_price": live_nearest_px,
            "live_nearest_supportive_trade_delay_ms": live_nearest_delay_ms,
            "live_nearest_supportive_trade_ts_local": live_nearest_ts or "",
            "replay_supportive_trade_count_before_anchor": replay_support_count,
            "replay_nearest_supportive_trade_price": replay_nearest_px,
            "replay_nearest_supportive_trade_delay_ms": replay_nearest_delay_ms,
            "replay_nearest_supportive_trade_ts_local": replay_nearest_ts or "",
            "live_cancel_after_replay_fill_ms": (
                (live_cancel_req_ts - replay_fill_ts) / 1_000_000.0
                if _finite(live_cancel_req_ts) and _finite(replay_fill_ts)
                else math.nan
            ),
            "sidecar_submit_bid_top1_px": sidecar_at_submit.get("bid_top5_px", "").split("|")[0] if sidecar_at_submit else "",
            "sidecar_submit_ask_top1_px": sidecar_at_submit.get("ask_top5_px", "").split("|")[0] if sidecar_at_submit else "",
            "sidecar_submit_bid_top1_qty": sidecar_at_submit.get("bid_top5_qtys", "").split("|")[0] if sidecar_at_submit else "",
            "sidecar_submit_ask_top1_qty": sidecar_at_submit.get("ask_top5_qtys", "").split("|")[0] if sidecar_at_submit else "",
            "sidecar_submit_depth_age_ms": sidecar_at_submit.get("bookticker_depth_age_ms", "") if sidecar_at_submit else "",
            "sidecar_live_terminal_bid_top1_px": sidecar_at_live_terminal.get("bid_top5_px", "").split("|")[0] if sidecar_at_live_terminal else "",
            "sidecar_live_terminal_ask_top1_px": sidecar_at_live_terminal.get("ask_top5_px", "").split("|")[0] if sidecar_at_live_terminal else "",
            "sidecar_replay_fill_bid_top1_px": sidecar_at_replay_fill.get("bid_top5_px", "").split("|")[0] if sidecar_at_replay_fill else "",
            "sidecar_replay_fill_ask_top1_px": sidecar_at_replay_fill.get("ask_top5_px", "").split("|")[0] if sidecar_at_replay_fill else "",
            "raw_window_event_count": len(raw_window),
            "raw_window_trade_count": sum(1 for event in raw_window if event["event_type"] == "trade"),
            "raw_window_depth_count": sum(1 for event in raw_window if event["event_type"] == "depthUpdate"),
            "raw_window_bookticker_count": sum(1 for event in raw_window if event["event_type"] == "bookTicker"),
        }
        result.update({f"live_{k}": v for k, v in _window_support_counts(supportive_trades, anchor_ts=int(live_fill_ts) if _finite(live_fill_ts) else int(live_terminal_ts)).items()})
        result.update({f"replay_{k}": v for k, v in _window_support_counts(supportive_trades, anchor_ts=int(replay_fill_ts) if _finite(replay_fill_ts) else int(replay_terminal_ts)).items()})
        trigger_class, trigger_note = _classify_residual_trigger(result)
        result["residual_trigger_class"] = trigger_class
        result["residual_trigger_note"] = trigger_note
        results.append(result)
    return results


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


def write_residual_summary_markdown(
    path: Path,
    *,
    run_dir: Path,
    output_dir: Path,
    residual_rows: list[dict[str, Any]],
) -> None:
    lines = [
        f"# {RESIDUAL_TASK_ID} residual replay fill mismatch diagnosis",
        "",
        "## Dataset",
        f"- run_dir: `{run_dir}`",
        f"- output_dir: `{output_dir}`",
        f"- residual_case_count: `{len(residual_rows)}`",
        "",
        "## Residual Cases",
    ]
    if not residual_rows:
        lines.extend(["- none", ""])
    for row in residual_rows:
        lines.extend(
            [
                f"### {row['submit_key']}",
                f"- case_label: `{row['case_label']}`",
                f"- trigger_class: `{row['residual_trigger_class']}`",
                f"- note: {row['residual_trigger_note']}",
                f"- live timeline: submit `{row['live_submit_ts_local']}`, cancel_req `{row['live_cancel_request_ts_local']}`, fill `{row['live_fill_ts_local']}`, terminal `{row['live_terminal_ts_local']}`",
                f"- replay timeline: submit `{row['replay_submit_ts_local']}`, cancel_req `{row['replay_cancel_request_ts_local']}`, fill `{row['replay_fill_ts_local']}`, terminal `{row['replay_terminal_ts_local']}`",
                f"- raw supportive trades before live anchor: `{row['live_supportive_trade_count_before_anchor']}`",
                f"- raw supportive trades before replay anchor: `{row['replay_supportive_trade_count_before_anchor']}`",
                f"- sidecar submit top1: bid `{row['sidecar_submit_bid_top1_px']}` / ask `{row['sidecar_submit_ask_top1_px']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "- This task remains diagnosis-only. It does not modify replay behavior.",
            "- Use these residual classifications to decide whether a separate `0515T005` repair is justified.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_residual_replay_fill_diagnosis(
    *,
    run_dir: Path,
    output_dir: Path,
    tick_size: float = DEFAULT_TICK_SIZE,
    max_future_gap_ms: float = DEFAULT_MAX_FUTURE_GAP_MS,
    maker_fee_bps: float = DEFAULT_MAKER_FEE_BPS,
) -> dict[str, Any]:
    run_dir = _expand(run_dir)
    output_dir = _expand(output_dir)
    live_audit_csv = _live_audit_csv(run_dir)
    replay_audit_csv = _replay_audit_csv(run_dir)
    joined_csv = run_dir / "t009_fixed_sidecar" / "joined_decisions.csv"
    top5_csv = run_dir / "t009_fixed_sidecar" / "top5_sidecar.csv"
    raw_gzip = _raw_market_gzip(run_dir)

    joined_rows = load_joined_decisions(joined_csv)
    live_bundle = _domain_bundle(
        audit_csv=live_audit_csv,
        joined_rows=joined_rows,
        tick_size=tick_size,
        horizons_ms=DEFAULT_HORIZONS_MS,
        max_future_gap_ms=max_future_gap_ms,
        maker_fee_bps=maker_fee_bps,
    )
    replay_bundle = _domain_bundle(
        audit_csv=replay_audit_csv,
        joined_rows=joined_rows,
        tick_size=tick_size,
        horizons_ms=DEFAULT_HORIZONS_MS,
        max_future_gap_ms=max_future_gap_ms,
        maker_fee_bps=maker_fee_bps,
    )
    _, matched_pairs, _ = build_submit_key_coverage(live_bundle, replay_bundle)
    state_diff_rows = build_matched_submit_state_diff(matched_pairs)
    residual_state_rows = [
        row for row in state_diff_rows
        if row["case_label"] in {"live_filled_replay_canceled", "live_canceled_replay_filled"}
    ]
    if residual_state_rows:
        anchor_points = []
        for row in residual_state_rows:
            for field in (
                "live_submit_ts_local",
                "replay_submit_ts_local",
                "live_time_to_fill_ms",
            ):
                value = _float(row.get(field))
                if _finite(value):
                    anchor_points.append(int(value))
        ts_candidates = []
        for row in residual_state_rows:
            for field in ("live_submit_ts_local", "replay_submit_ts_local"):
                value = _float(row.get(field))
                if _finite(value):
                    ts_candidates.append(int(value))
        window_start = min(ts_candidates) - 10_000_000_000
        window_end = max(ts_candidates) + 10_000_000_000
    else:
        window_start = 0
        window_end = 0
    raw_events = _load_raw_market_events(raw_gzip, start_ts_local=window_start, end_ts_local=window_end) if residual_state_rows else []
    residual_rows = build_residual_case_diagnosis(
        state_diff_rows=state_diff_rows,
        live_audit_rows=_load_csv_rows(live_audit_csv),
        replay_audit_rows=_load_csv_rows(replay_audit_csv),
        top5_rows=_load_top5_sidecar(top5_csv),
        raw_events=raw_events,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "residual_case_diagnosis.csv",
        residual_rows,
        fieldnames=list(residual_rows[0].keys()) if residual_rows else [],
    )
    manifest = {
        "task_id": RESIDUAL_TASK_ID,
        "generated_at": _generated_at(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "input_hashes": {
            "live_audit_csv": _hash_file(live_audit_csv),
            "replay_audit_csv": _hash_file(replay_audit_csv),
            "joined_decisions_csv": _hash_file(joined_csv),
            "top5_sidecar_csv": _hash_file(top5_csv),
            "raw_market_gzip": _hash_file(raw_gzip),
        },
        "row_counts": {
            "residual_case_rows": len(residual_rows),
            "cancel_race_window_too_short_rows": sum(1 for row in residual_rows if row["residual_trigger_class"] == "cancel_race_window_too_short"),
            "touch_fill_assumption_too_optimistic_rows": sum(1 for row in residual_rows if row["residual_trigger_class"] == "touch_fill_assumption_too_optimistic"),
            "queue_exposure_proxy_bias_possible_rows": sum(1 for row in residual_rows if row["residual_trigger_class"] == "queue_exposure_proxy_bias_possible"),
        },
        "artifacts": [
            "residual_case_diagnosis.csv",
            "RESIDUAL_REPLAY_FILL_DIAGNOSIS_SUMMARY.md",
            "run_manifest.json",
        ],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    write_residual_summary_markdown(
        output_dir / "RESIDUAL_REPLAY_FILL_DIAGNOSIS_SUMMARY.md",
        run_dir=run_dir,
        output_dir=output_dir,
        residual_rows=residual_rows,
    )
    return manifest


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
    parser.add_argument(
        "--residual-only",
        action="store_true",
        help=f"Run the {RESIDUAL_TASK_ID} residual-case diagnosis instead of the default {TASK_ID} mismatch runner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _expand(args.run_dir)
    if args.residual_only:
        output_dir = _expand(args.output_dir) if args.output_dir else run_dir / f"stage6e_residual_replay_fill_diagnosis_{RESIDUAL_TASK_ID}"
        manifest = run_residual_replay_fill_diagnosis(
            run_dir=run_dir,
            output_dir=output_dir,
            tick_size=float(args.tick_size),
            max_future_gap_ms=float(args.max_future_gap_ms),
            maker_fee_bps=float(args.maker_fee_bps),
        )
        task_id = RESIDUAL_TASK_ID
    else:
        output_dir = _expand(args.output_dir) if args.output_dir else run_dir / f"stage6c_replay_lifecycle_mismatch_{TASK_ID}"
        manifest = run_replay_lifecycle_mismatch_diagnosis(
            run_dir=run_dir,
            output_dir=output_dir,
            tick_size=float(args.tick_size),
            max_future_gap_ms=float(args.max_future_gap_ms),
            maker_fee_bps=float(args.maker_fee_bps),
        )
        task_id = TASK_ID
    print(json.dumps({"task_id": task_id, "output_dir": str(output_dir), "row_counts": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
