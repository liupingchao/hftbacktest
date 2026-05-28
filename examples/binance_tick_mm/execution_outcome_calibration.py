#!/usr/bin/env python3
"""Read-only Stage 6B replay/live execution outcome calibration runner."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from execution_outcome_labels import (
    DEFAULT_HORIZONS_MS,
    DEFAULT_MAKER_FEE_BPS,
    DEFAULT_MAX_FUTURE_GAP_MS,
    DEFAULT_TICK_SIZE,
    _bucketize,
    _expand,
    _finite,
    _float,
    _generated_at,
    _hash_file,
    _load_json,
    _mean,
    _quantile,
    _safe_div,
    _write_csv,
    _write_json,
    build_decision_index,
    build_execution_labels,
    load_audit_rows,
    load_joined_decisions,
)


TASK_ID = "0514T007"


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


def _replay_audit_csv(run_dir: Path) -> Path:
    compact_path = run_dir / "out" / "backtest_audit_replay" / "audit_bt_audit_replay.compact_lifecycle.csv"
    if compact_path.exists():
        return compact_path
    path = run_dir / "out" / "backtest_audit_replay" / "audit_bt_audit_replay.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing audit replay csv: {path}")
    return path


def _live_audit_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("audit_live_*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one audit_live_*.csv under {run_dir}, found {len(candidates)}")
    return candidates[0]


def _submit_key(row: dict[str, Any]) -> str:
    strategy_seq = row.get("submit_strategy_seq")
    side = str(row.get("order_side") or "").strip().lower()
    if strategy_seq in {"", None} or side not in {"buy", "sell"}:
        raise ValueError(f"Cannot build submit key from row: submit_strategy_seq={strategy_seq!r}, side={side!r}")
    return f"{int(strategy_seq)}|{side}"


def _rate(rows: Iterable[dict[str, Any]], field: str) -> float:
    values = [int(row.get(field, 0)) for row in rows]
    if not values:
        return math.nan
    return float(sum(values)) / float(len(values))


def _values(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        value = _float(row.get(field))
        if _finite(value):
            out.append(float(value))
    return out


def _subset_pairs(pairs: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> list[dict[str, Any]]:
    return [pair for pair in pairs if predicate(pair)]


def _state_counts(rows: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(str(row.get("final_order_state") or "") for row in rows)


def _fill_gap_threshold(horizon_ms: int) -> float:
    if horizon_ms <= 100:
        return 0.01
    if horizon_ms <= 500:
        return 0.015
    if horizon_ms <= 1_000:
        return 0.02
    return 0.03


def _domain_bundle(
    *,
    audit_csv: Path,
    joined_rows: dict[int, dict[str, str]],
    tick_size: float,
    horizons_ms: Iterable[int],
    max_future_gap_ms: float,
    maker_fee_bps: float,
) -> dict[str, Any]:
    audit_rows = load_audit_rows(audit_csv)
    decision_index = build_decision_index(audit_rows, joined_rows, tick_size=tick_size)
    execution_rows, fill_horizon_rows, fill_markout_rows, row_counts = build_execution_labels(
        audit_rows=audit_rows,
        decision_index=decision_index,
        horizons_ms=horizons_ms,
        tick_size=tick_size,
        max_future_gap_ms=max_future_gap_ms,
        maker_fee_bps=maker_fee_bps,
    )
    by_key: dict[str, dict[str, Any]] = {}
    order_id_to_key: dict[str, str] = {}
    for row in execution_rows:
        submit_key = _submit_key(row)
        if submit_key in by_key:
            raise ValueError(f"Duplicate submit key within one domain: {submit_key}")
        row["submit_key"] = submit_key
        by_key[submit_key] = row
        order_id_to_key[str(row["order_id"])] = submit_key

    fill_horizon_by_key_h: dict[tuple[str, int], dict[str, Any]] = {}
    for row in fill_horizon_rows:
        submit_key = order_id_to_key.get(str(row["order_id"]))
        if submit_key is None:
            continue
        row["submit_key"] = submit_key
        fill_horizon_by_key_h[(submit_key, int(row["horizon_ms"]))] = row

    fill_markout_by_key_h: dict[tuple[str, int], dict[str, Any]] = {}
    for row in fill_markout_rows:
        submit_key = order_id_to_key.get(str(row["order_id"]))
        if submit_key is None:
            continue
        row["submit_key"] = submit_key
        fill_markout_by_key_h[(submit_key, int(row["horizon_ms"]))] = row

    return {
        "audit_csv": audit_csv,
        "execution_rows": execution_rows,
        "fill_horizon_rows": fill_horizon_rows,
        "fill_markout_rows": fill_markout_rows,
        "row_counts": row_counts,
        "by_key": by_key,
        "fill_horizon_by_key_h": fill_horizon_by_key_h,
        "fill_markout_by_key_h": fill_markout_by_key_h,
    }


def build_submit_key_coverage(
    live_bundle: dict[str, Any],
    replay_bundle: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    live_by_key = live_bundle["by_key"]
    replay_by_key = replay_bundle["by_key"]
    keys = sorted(set(live_by_key) | set(replay_by_key), key=lambda item: (int(item.split("|", 1)[0]), item.split("|", 1)[1]))
    rows: list[dict[str, Any]] = []
    matched_pairs: list[dict[str, Any]] = []
    for submit_key in keys:
        strategy_seq_text, order_side = submit_key.split("|", 1)
        live_row = live_by_key.get(submit_key)
        replay_row = replay_by_key.get(submit_key)
        matched = int(live_row is not None and replay_row is not None)
        price_equal = ""
        qty_equal = ""
        submit_ts_equal = ""
        if matched:
            price_equal = int(_float(live_row.get("order_price_tick")) == _float(replay_row.get("order_price_tick")))
            qty_equal = int(_float(live_row.get("order_qty")) == _float(replay_row.get("order_qty")))
            submit_ts_equal = int(int(live_row["submit_ts_local"]) == int(replay_row["submit_ts_local"]))
            matched_pairs.append(
                {
                    "submit_key": submit_key,
                    "submit_strategy_seq": int(strategy_seq_text),
                    "order_side": order_side,
                    "live": live_row,
                    "replay": replay_row,
                }
            )
        rows.append(
            {
                "submit_key": submit_key,
                "submit_strategy_seq": int(strategy_seq_text),
                "order_side": order_side,
                "live_present": int(live_row is not None),
                "replay_present": int(replay_row is not None),
                "matched": matched,
                "live_order_id": live_row.get("order_id") if live_row else "",
                "replay_order_id": replay_row.get("order_id") if replay_row else "",
                "live_order_price_tick": live_row.get("order_price_tick") if live_row else "",
                "replay_order_price_tick": replay_row.get("order_price_tick") if replay_row else "",
                "price_tick_equal": price_equal,
                "live_order_qty": live_row.get("order_qty") if live_row else "",
                "replay_order_qty": replay_row.get("order_qty") if replay_row else "",
                "qty_equal": qty_equal,
                "live_submit_ts_local": live_row.get("submit_ts_local") if live_row else "",
                "replay_submit_ts_local": replay_row.get("submit_ts_local") if replay_row else "",
                "submit_ts_equal": submit_ts_equal,
                "live_placement_bucket": live_row.get("placement_bucket") if live_row else "",
                "replay_placement_bucket": replay_row.get("placement_bucket") if replay_row else "",
            }
        )
    summary = {
        "live_submit_rows": len(live_by_key),
        "replay_submit_rows": len(replay_by_key),
        "matched_submit_rows": len(matched_pairs),
        "unmatched_live_rows": sum(1 for row in rows if row["live_present"] and not row["replay_present"]),
        "unmatched_replay_rows": sum(1 for row in rows if row["replay_present"] and not row["live_present"]),
        "matched_rate_vs_live": _safe_div(len(matched_pairs), len(live_by_key)),
        "matched_rate_vs_replay": _safe_div(len(matched_pairs), len(replay_by_key)),
        "price_tick_equal_rows": sum(int(row["price_tick_equal"] or 0) for row in rows if row["matched"]),
        "qty_equal_rows": sum(int(row["qty_equal"] or 0) for row in rows if row["matched"]),
    }
    return rows, matched_pairs, summary


def build_coverage_gap(
    *,
    live_bundle: dict[str, Any],
    replay_bundle: dict[str, Any],
    matched_pairs: list[dict[str, Any]],
    horizons_ms: Iterable[int],
    coverage_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = [
        {
            "label_family": "submit_matching",
            "horizon_ms": "",
            "live_total_rows": coverage_summary["live_submit_rows"],
            "replay_total_rows": coverage_summary["replay_submit_rows"],
            "matched_rows": coverage_summary["matched_submit_rows"],
            "live_observable_rows": coverage_summary["live_submit_rows"],
            "replay_observable_rows": coverage_summary["replay_submit_rows"],
            "both_observable_rows": coverage_summary["matched_submit_rows"],
            "live_positive_rows": "",
            "replay_positive_rows": "",
            "absolute_gap": abs(coverage_summary["matched_rate_vs_live"] - coverage_summary["matched_rate_vs_replay"]),
            "notes": "coverage measured on normalized submit opportunities",
        }
    ]
    for horizon_ms in horizons_ms:
        live_obs = sum(int(pair["live"].get(f"horizon_observable_{horizon_ms}ms", 0)) for pair in matched_pairs)
        replay_obs = sum(int(pair["replay"].get(f"horizon_observable_{horizon_ms}ms", 0)) for pair in matched_pairs)
        both_obs_pairs = [
            pair
            for pair in matched_pairs
            if int(pair["live"].get(f"horizon_observable_{horizon_ms}ms", 0)) == 1
            and int(pair["replay"].get(f"horizon_observable_{horizon_ms}ms", 0)) == 1
        ]
        rows.append(
            {
                "label_family": "fill_probability",
                "horizon_ms": horizon_ms,
                "live_total_rows": len(live_bundle["execution_rows"]),
                "replay_total_rows": len(replay_bundle["execution_rows"]),
                "matched_rows": len(matched_pairs),
                "live_observable_rows": live_obs,
                "replay_observable_rows": replay_obs,
                "both_observable_rows": len(both_obs_pairs),
                "live_positive_rows": sum(int(pair["live"].get(f"fill_by_{horizon_ms}ms", 0)) for pair in matched_pairs),
                "replay_positive_rows": sum(int(pair["replay"].get(f"fill_by_{horizon_ms}ms", 0)) for pair in matched_pairs),
                "absolute_gap": abs(_safe_div(live_obs, len(matched_pairs)) - _safe_div(replay_obs, len(matched_pairs))),
                "notes": "observable coverage on matched submit universe",
            }
        )
        live_markout_obs = sum(int(pair["live"].get(f"markout_observable_{horizon_ms}ms", 0)) for pair in matched_pairs)
        replay_markout_obs = sum(int(pair["replay"].get(f"markout_observable_{horizon_ms}ms", 0)) for pair in matched_pairs)
        rows.append(
            {
                "label_family": "fill_markout",
                "horizon_ms": horizon_ms,
                "live_total_rows": len(live_bundle["execution_rows"]),
                "replay_total_rows": len(replay_bundle["execution_rows"]),
                "matched_rows": len(matched_pairs),
                "live_observable_rows": live_markout_obs,
                "replay_observable_rows": replay_markout_obs,
                "both_observable_rows": sum(
                    1
                    for pair in matched_pairs
                    if int(pair["live"].get(f"markout_observable_{horizon_ms}ms", 0)) == 1
                    and int(pair["replay"].get(f"markout_observable_{horizon_ms}ms", 0)) == 1
                ),
                "live_positive_rows": "",
                "replay_positive_rows": "",
                "absolute_gap": abs(_safe_div(live_markout_obs, len(matched_pairs)) - _safe_div(replay_markout_obs, len(matched_pairs))),
                "notes": "markout coverage on matched submit universe",
            }
        )
    return rows


def build_fill_horizon_gap(matched_pairs: list[dict[str, Any]], horizons_ms: Iterable[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_ms in horizons_ms:
        both_observable = _subset_pairs(
            matched_pairs,
            lambda pair: int(pair["live"].get(f"horizon_observable_{horizon_ms}ms", 0)) == 1
            and int(pair["replay"].get(f"horizon_observable_{horizon_ms}ms", 0)) == 1,
        )
        live_rate_all = _rate((pair["live"] for pair in matched_pairs), f"fill_by_{horizon_ms}ms")
        replay_rate_all = _rate((pair["replay"] for pair in matched_pairs), f"fill_by_{horizon_ms}ms")
        live_rate_both = _rate((pair["live"] for pair in both_observable), f"fill_by_{horizon_ms}ms")
        replay_rate_both = _rate((pair["replay"] for pair in both_observable), f"fill_by_{horizon_ms}ms")
        abs_gap = abs(live_rate_all - replay_rate_all) if _finite(live_rate_all) and _finite(replay_rate_all) else math.nan
        rows.append(
            {
                "horizon_ms": horizon_ms,
                "matched_rows": len(matched_pairs),
                "both_observable_rows": len(both_observable),
                "live_positive_rows_all": sum(int(pair["live"].get(f"fill_by_{horizon_ms}ms", 0)) for pair in matched_pairs),
                "replay_positive_rows_all": sum(int(pair["replay"].get(f"fill_by_{horizon_ms}ms", 0)) for pair in matched_pairs),
                "live_fill_rate_all": live_rate_all,
                "replay_fill_rate_all": replay_rate_all,
                "absolute_gap_all": abs_gap,
                "relative_gap_all_vs_live": _safe_div(abs_gap, live_rate_all),
                "live_fill_rate_both_observable": live_rate_both,
                "replay_fill_rate_both_observable": replay_rate_both,
                "absolute_gap_both_observable": abs(live_rate_both - replay_rate_both) if _finite(live_rate_both) and _finite(replay_rate_both) else math.nan,
                "aligned": int(_finite(abs_gap) and abs_gap <= _fill_gap_threshold(int(horizon_ms))),
            }
        )
    return rows


def build_time_to_fill_gap(matched_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    live_all = [_time_to_fill_ms(pair["live"]) for pair in matched_pairs if _finite(_time_to_fill_ms(pair["live"]))]
    replay_all = [_time_to_fill_ms(pair["replay"]) for pair in matched_pairs if _finite(_time_to_fill_ms(pair["replay"]))]
    both = [
        pair
        for pair in matched_pairs
        if _finite(_time_to_fill_ms(pair["live"])) and _finite(_time_to_fill_ms(pair["replay"]))
    ]
    live_both = [_time_to_fill_ms(pair["live"]) for pair in both]
    replay_both = [_time_to_fill_ms(pair["replay"]) for pair in both]
    rows = []
    for scope, live_values, replay_values in [
        ("matched_any_filled", live_all, replay_all),
        ("matched_both_filled", live_both, replay_both),
    ]:
        rows.append(
            {
                "scope": scope,
                "live_rows": len(live_values),
                "replay_rows": len(replay_values),
                "paired_rows": len(both) if scope == "matched_both_filled" else len(matched_pairs),
                "live_mean_ms": _mean(live_values),
                "replay_mean_ms": _mean(replay_values),
                "absolute_gap_mean_ms": abs(_mean(live_values) - _mean(replay_values)) if _finite(_mean(live_values)) and _finite(_mean(replay_values)) else math.nan,
                "live_p50_ms": _quantile(live_values, 0.5),
                "replay_p50_ms": _quantile(replay_values, 0.5),
                "absolute_gap_p50_ms": abs(_quantile(live_values, 0.5) - _quantile(replay_values, 0.5)) if _finite(_quantile(live_values, 0.5)) and _finite(_quantile(replay_values, 0.5)) else math.nan,
                "live_p90_ms": _quantile(live_values, 0.9),
                "replay_p90_ms": _quantile(replay_values, 0.9),
                "absolute_gap_p90_ms": abs(_quantile(live_values, 0.9) - _quantile(replay_values, 0.9)) if _finite(_quantile(live_values, 0.9)) and _finite(_quantile(replay_values, 0.9)) else math.nan,
            }
        )
    return rows


def build_final_state_gap(matched_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(matched_pairs)
    live_counts = _state_counts(pair["live"] for pair in matched_pairs)
    replay_counts = _state_counts(pair["replay"] for pair in matched_pairs)
    rows: list[dict[str, Any]] = []
    for state in sorted(set(live_counts) | set(replay_counts)):
        live_rate = _safe_div(live_counts[state], total)
        replay_rate = _safe_div(replay_counts[state], total)
        abs_gap = abs(live_rate - replay_rate) if _finite(live_rate) and _finite(replay_rate) else math.nan
        rows.append(
            {
                "final_order_state": state,
                "matched_rows": total,
                "live_count": live_counts[state],
                "replay_count": replay_counts[state],
                "live_rate": live_rate,
                "replay_rate": replay_rate,
                "absolute_gap": abs_gap,
                "aligned": int(_finite(abs_gap) and abs_gap <= 0.02),
            }
        )
    return rows


def build_cancel_race_gap(matched_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric_name, field, threshold in [
        ("fill_after_cancel_request_rate", "fill_after_cancel_request", 0.01),
        ("fast_cancel_churn_rate", "fast_cancel_churn", 0.10),
    ]:
        live_value = _rate((pair["live"] for pair in matched_pairs), field)
        replay_value = _rate((pair["replay"] for pair in matched_pairs), field)
        abs_gap = abs(live_value - replay_value) if _finite(live_value) and _finite(replay_value) else math.nan
        rows.append(
            {
                "metric_name": metric_name,
                "scope": "matched_submit_universe",
                "live_rows": len(matched_pairs),
                "replay_rows": len(matched_pairs),
                "paired_rows": len(matched_pairs),
                "live_value": live_value,
                "replay_value": replay_value,
                "absolute_gap": abs_gap,
                "aligned": int(_finite(abs_gap) and abs_gap <= threshold),
                "notes": "",
            }
        )

    for scope, live_values, replay_values in [
        (
            "delay_all_observed",
            _values((pair["live"] for pair in matched_pairs), "cancel_to_fill_delay_ms"),
            _values((pair["replay"] for pair in matched_pairs), "cancel_to_fill_delay_ms"),
        ),
        (
            "delay_both_observed",
            [_float(pair["live"].get("cancel_to_fill_delay_ms")) for pair in matched_pairs if _finite(pair["live"].get("cancel_to_fill_delay_ms")) and _finite(pair["replay"].get("cancel_to_fill_delay_ms"))],
            [_float(pair["replay"].get("cancel_to_fill_delay_ms")) for pair in matched_pairs if _finite(pair["live"].get("cancel_to_fill_delay_ms")) and _finite(pair["replay"].get("cancel_to_fill_delay_ms"))],
        ),
    ]:
        live_p50 = _quantile(live_values, 0.5)
        replay_p50 = _quantile(replay_values, 0.5)
        live_p90 = _quantile(live_values, 0.9)
        replay_p90 = _quantile(replay_values, 0.9)
        rows.append(
            {
                "metric_name": f"cancel_to_fill_delay_ms_{scope}",
                "scope": scope,
                "live_rows": len(live_values),
                "replay_rows": len(replay_values),
                "paired_rows": min(len(live_values), len(replay_values)),
                "live_value": _mean(live_values),
                "replay_value": _mean(replay_values),
                "absolute_gap": abs(_mean(live_values) - _mean(replay_values)) if _finite(_mean(live_values)) and _finite(_mean(replay_values)) else math.nan,
                "aligned": "",
                "notes": f"p50_gap_ms={abs(live_p50 - replay_p50) if _finite(live_p50) and _finite(replay_p50) else math.nan}, p90_gap_ms={abs(live_p90 - replay_p90) if _finite(live_p90) and _finite(replay_p90) else math.nan}",
            }
        )
    return rows


def build_markout_gap(matched_pairs: list[dict[str, Any]], horizons_ms: Iterable[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon_ms in horizons_ms:
        both_observable = [
            pair
            for pair in matched_pairs
            if int(pair["live"].get(f"markout_observable_{horizon_ms}ms", 0)) == 1
            and int(pair["replay"].get(f"markout_observable_{horizon_ms}ms", 0)) == 1
            and _finite(pair["live"].get(f"fill_markout_{horizon_ms}ms_ticks"))
            and _finite(pair["replay"].get(f"fill_markout_{horizon_ms}ms_ticks"))
        ]
        live_markouts = [_float(pair["live"].get(f"fill_markout_{horizon_ms}ms_ticks")) for pair in both_observable]
        replay_markouts = [_float(pair["replay"].get(f"fill_markout_{horizon_ms}ms_ticks")) for pair in both_observable]
        live_spread = [_float(pair["live"].get("realized_spread_proxy_ticks")) for pair in both_observable if _finite(pair["live"].get("realized_spread_proxy_ticks"))]
        replay_spread = [_float(pair["replay"].get("realized_spread_proxy_ticks")) for pair in both_observable if _finite(pair["replay"].get("realized_spread_proxy_ticks"))]
        live_net = [_float(pair["live"].get(f"net_ev_proxy_{horizon_ms}ms_ticks")) for pair in both_observable if _finite(pair["live"].get(f"net_ev_proxy_{horizon_ms}ms_ticks"))]
        replay_net = [_float(pair["replay"].get(f"net_ev_proxy_{horizon_ms}ms_ticks")) for pair in both_observable if _finite(pair["replay"].get(f"net_ev_proxy_{horizon_ms}ms_ticks"))]
        rows.append(
            {
                "horizon_ms": horizon_ms,
                "both_observable_rows": len(both_observable),
                "live_markout_mean_ticks": _mean(live_markouts),
                "replay_markout_mean_ticks": _mean(replay_markouts),
                "markout_mean_absolute_gap_ticks": abs(_mean(live_markouts) - _mean(replay_markouts)) if _finite(_mean(live_markouts)) and _finite(_mean(replay_markouts)) else math.nan,
                "live_markout_median_ticks": _quantile(live_markouts, 0.5),
                "replay_markout_median_ticks": _quantile(replay_markouts, 0.5),
                "live_positive_markout_rate": _safe_div(sum(1 for value in live_markouts if value > 0.0), len(live_markouts)),
                "replay_positive_markout_rate": _safe_div(sum(1 for value in replay_markouts if value > 0.0), len(replay_markouts)),
                "live_realized_spread_mean_ticks": _mean(live_spread),
                "replay_realized_spread_mean_ticks": _mean(replay_spread),
                "spread_mean_absolute_gap_ticks": abs(_mean(live_spread) - _mean(replay_spread)) if _finite(_mean(live_spread)) and _finite(_mean(replay_spread)) else math.nan,
                "live_net_ev_mean_ticks": _mean(live_net),
                "replay_net_ev_mean_ticks": _mean(replay_net),
                "net_ev_mean_absolute_gap_ticks": abs(_mean(live_net) - _mean(replay_net)) if _finite(_mean(live_net)) and _finite(_mean(replay_net)) else math.nan,
            }
        )
    return rows


def _numeric_quantile_labels(values: list[float]) -> list[str]:
    finite_values = [value for value in values if _finite(value)]
    labels = ["missing" for _ in values]
    if not finite_values:
        return labels
    if len(set(round(value, 12) for value in finite_values)) == 1:
        for idx, value in enumerate(values):
            if _finite(value):
                labels[idx] = "all"
        return labels
    _, bucket_ids = _bucketize(finite_values, buckets=5)
    finite_iter = iter(bucket_ids)
    for idx, value in enumerate(values):
        if _finite(value):
            labels[idx] = f"q{int(next(finite_iter))}"
    return labels


def _strata_metric_row(strata_family: str, strata_label: str, group: list[dict[str, Any]]) -> dict[str, Any]:
    both_markout = [
        pair
        for pair in group
        if int(pair["live"].get("markout_observable_500ms", 0)) == 1
        and int(pair["replay"].get("markout_observable_500ms", 0)) == 1
        and _finite(pair["live"].get("fill_markout_500ms_ticks"))
        and _finite(pair["replay"].get("fill_markout_500ms_ticks"))
    ]
    both_filled = [
        pair
        for pair in group
        if _finite(_time_to_fill_ms(pair["live"])) and _finite(_time_to_fill_ms(pair["replay"]))
    ]
    live_markout = [_float(pair["live"].get("fill_markout_500ms_ticks")) for pair in both_markout]
    replay_markout = [_float(pair["replay"].get("fill_markout_500ms_ticks")) for pair in both_markout]
    live_ttf = [_time_to_fill_ms(pair["live"]) for pair in both_filled]
    replay_ttf = [_time_to_fill_ms(pair["replay"]) for pair in both_filled]
    live_fill_500 = _rate((pair["live"] for pair in group), "fill_by_500ms")
    replay_fill_500 = _rate((pair["replay"] for pair in group), "fill_by_500ms")
    live_fill_5000 = _rate((pair["live"] for pair in group), "fill_by_5000ms")
    replay_fill_5000 = _rate((pair["replay"] for pair in group), "fill_by_5000ms")
    live_cancel = _rate((pair["live"] for pair in group), "fill_after_cancel_request")
    replay_cancel = _rate((pair["replay"] for pair in group), "fill_after_cancel_request")
    return {
        "strata_family": strata_family,
        "strata_label": strata_label,
        "matched_rows": len(group),
        "live_fill_500ms_rate": live_fill_500,
        "replay_fill_500ms_rate": replay_fill_500,
        "fill_500ms_absolute_gap": abs(live_fill_500 - replay_fill_500) if _finite(live_fill_500) and _finite(replay_fill_500) else math.nan,
        "live_fill_5000ms_rate": live_fill_5000,
        "replay_fill_5000ms_rate": replay_fill_5000,
        "fill_5000ms_absolute_gap": abs(live_fill_5000 - replay_fill_5000) if _finite(live_fill_5000) and _finite(replay_fill_5000) else math.nan,
        "live_fill_after_cancel_rate": live_cancel,
        "replay_fill_after_cancel_rate": replay_cancel,
        "fill_after_cancel_absolute_gap": abs(live_cancel - replay_cancel) if _finite(live_cancel) and _finite(replay_cancel) else math.nan,
        "markout_500ms_rows": len(both_markout),
        "live_markout_500ms_mean_ticks": _mean(live_markout),
        "replay_markout_500ms_mean_ticks": _mean(replay_markout),
        "markout_500ms_mean_absolute_gap_ticks": abs(_mean(live_markout) - _mean(replay_markout)) if _finite(_mean(live_markout)) and _finite(_mean(replay_markout)) else math.nan,
        "time_to_fill_rows": len(both_filled),
        "live_time_to_fill_mean_ms": _mean(live_ttf),
        "replay_time_to_fill_mean_ms": _mean(replay_ttf),
        "time_to_fill_mean_absolute_gap_ms": abs(_mean(live_ttf) - _mean(replay_ttf)) if _finite(_mean(live_ttf)) and _finite(_mean(replay_ttf)) else math.nan,
    }


def build_strata_gap_rows(
    matched_pairs: list[dict[str, Any]],
    *,
    family_name: str,
    extractor: Callable[[dict[str, Any]], Any],
    categorical: bool,
) -> list[dict[str, Any]]:
    live_values = [extractor(pair["live"]) for pair in matched_pairs]
    if categorical:
        labels = [str(value) if str(value) else "missing" for value in live_values]
    else:
        numeric_values = [_float(value) for value in live_values]
        labels = _numeric_quantile_labels(numeric_values)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair, label in zip(matched_pairs, labels, strict=False):
        grouped[label].append(pair)
    return [_strata_metric_row(family_name, label, group) for label, group in sorted(grouped.items())]


def determine_decision_state(
    *,
    coverage_summary: dict[str, Any],
    fill_horizon_rows: list[dict[str, Any]],
    final_state_rows: list[dict[str, Any]],
    cancel_race_rows: list[dict[str, Any]],
    live_filled_orders: int,
) -> str:
    if min(coverage_summary["matched_rate_vs_live"], coverage_summary["matched_rate_vs_replay"]) < 0.95:
        return "diagnostic_only_gap_too_large"
    major_gap = 0
    for row in fill_horizon_rows:
        if _finite(row["absolute_gap_all"]) and float(row["absolute_gap_all"]) > _fill_gap_threshold(int(row["horizon_ms"])):
            major_gap += 1
    for row in final_state_rows:
        if row["final_order_state"] in {"filled", "canceled"} and _finite(row["absolute_gap"]) and float(row["absolute_gap"]) > 0.02:
            major_gap += 1
    for row in cancel_race_rows:
        if row["metric_name"] == "fill_after_cancel_request_rate" and _finite(row["absolute_gap"]) and float(row["absolute_gap"]) > 0.01:
            major_gap += 1
        if row["metric_name"] == "fast_cancel_churn_rate" and _finite(row["absolute_gap"]) and float(row["absolute_gap"]) > 0.10:
            major_gap += 1
    if major_gap >= 2:
        return "diagnostic_only_gap_too_large"
    if live_filled_orders < 75:
        return "requires_more_current_format_samples"
    return "methodology_valid_single_sample"


def write_summary_markdown(
    path: Path,
    *,
    run_dir: Path,
    output_dir: Path,
    manifest: dict[str, Any],
    coverage_summary: dict[str, Any],
    fill_horizon_rows: list[dict[str, Any]],
    final_state_rows: list[dict[str, Any]],
    cancel_race_rows: list[dict[str, Any]],
) -> None:
    aligned: list[str] = []
    not_aligned: list[str] = []
    if min(coverage_summary["matched_rate_vs_live"], coverage_summary["matched_rate_vs_replay"]) >= 0.99:
        aligned.append(
            f"submit-key coverage {coverage_summary['matched_submit_rows']}/{coverage_summary['live_submit_rows']} matched"
        )
    else:
        not_aligned.append(
            f"submit-key coverage live={coverage_summary['matched_rate_vs_live']:.4f}, replay={coverage_summary['matched_rate_vs_replay']:.4f}"
        )
    for row in fill_horizon_rows:
        label = f"fill@{row['horizon_ms']}ms gap={row['absolute_gap_all']:.4f}"
        if int(row["aligned"]) == 1:
            aligned.append(label)
        else:
            not_aligned.append(label)
    for row in final_state_rows:
        if row["final_order_state"] in {"filled", "canceled"}:
            label = f"final_state {row['final_order_state']} gap={row['absolute_gap']:.4f}"
            if int(row["aligned"]) == 1:
                aligned.append(label)
            else:
                not_aligned.append(label)
    for row in cancel_race_rows:
        if row["metric_name"] in {"fill_after_cancel_request_rate", "fast_cancel_churn_rate"}:
            label = f"{row['metric_name']} gap={row['absolute_gap']:.4f}"
            if int(row["aligned"]) == 1:
                aligned.append(label)
            else:
                not_aligned.append(label)

    aligned_lines = [f"- {item}" for item in aligned[:8]] if aligned else ["- none"]
    not_aligned_lines = [f"- {item}" for item in not_aligned[:12]] if not_aligned else ["- none"]

    lines = [
        f"# {TASK_ID} execution calibration",
        "",
        "## Dataset",
        f"- run_dir: `{run_dir}`",
        f"- output_dir: `{output_dir}`",
        f"- stage3 classification: `{manifest['stage3_classification']}`",
        f"- decision_state: `{manifest['decision_state']}`",
        f"- live_submit_orders: `{manifest['row_counts']['live_submit_orders']}`",
        f"- replay_submit_orders: `{manifest['row_counts']['replay_submit_orders']}`",
        f"- matched_submit_orders: `{manifest['row_counts']['matched_submit_orders']}`",
        f"- live_filled_orders: `{manifest['row_counts']['live_filled_orders']}`",
        f"- replay_filled_orders: `{manifest['row_counts']['replay_filled_orders']}`",
        "",
        "## Coverage",
        f"- matched submit coverage vs live: `{coverage_summary['matched_rate_vs_live']}`",
        f"- matched submit coverage vs replay: `{coverage_summary['matched_rate_vs_replay']}`",
        f"- price tick equality on matched submits: `{coverage_summary['price_tick_equal_rows']}/{coverage_summary['matched_submit_rows']}`",
        f"- qty equality on matched submits: `{coverage_summary['qty_equal_rows']}/{coverage_summary['matched_submit_rows']}`",
        "",
        "## Aligned Enough",
        *aligned_lines,
        "",
        "## Not Aligned",
        *not_aligned_lines,
        "",
        "## Boundaries",
        "- This runner compares observed replay/live lifecycle proxies only.",
        "- Matched submit opportunities are the primary comparison unit; unmatched coverage is reported separately.",
        "- Queue priority, missed opportunity, and realized PnL decomposition remain observed-only proxy concepts; no exact queue or counterfactual fill proof is claimed.",
        "- A single current-format sample can validate methodology, but it cannot alone authorize quote-adjustment promotion.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_execution_outcome_calibration(
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
    coverage_gap_rows = build_coverage_gap(
        live_bundle=live_bundle,
        replay_bundle=replay_bundle,
        matched_pairs=matched_pairs,
        horizons_ms=horizons_ms,
        coverage_summary=coverage_summary,
    )
    fill_horizon_gap_rows = build_fill_horizon_gap(matched_pairs, horizons_ms)
    time_to_fill_gap_rows = build_time_to_fill_gap(matched_pairs)
    final_state_gap_rows = build_final_state_gap(matched_pairs)
    cancel_race_gap_rows = build_cancel_race_gap(matched_pairs)
    markout_gap_rows = build_markout_gap(matched_pairs, horizons_ms)

    placement_rows: list[dict[str, Any]] = []
    placement_rows.extend(build_strata_gap_rows(matched_pairs, family_name="placement_bucket", extractor=lambda row: row.get("placement_bucket"), categorical=True))
    placement_rows.extend(build_strata_gap_rows(matched_pairs, family_name="distance_to_bbo_ticks_bucket", extractor=lambda row: row.get("distance_to_bbo_ticks"), categorical=False))
    placement_rows.extend(build_strata_gap_rows(matched_pairs, family_name="edge_vs_fair_ticks_bucket", extractor=lambda row: row.get("edge_vs_fair_ticks"), categorical=False))

    inventory_rows: list[dict[str, Any]] = []
    inventory_rows.extend(build_strata_gap_rows(matched_pairs, family_name="inventory_score_bucket", extractor=lambda row: row.get("inventory_score"), categorical=False))
    inventory_rows.extend(build_strata_gap_rows(matched_pairs, family_name="same_side_top1_qty_bucket", extractor=lambda row: row.get("same_side_top1_qty"), categorical=False))
    inventory_rows.extend(build_strata_gap_rows(matched_pairs, family_name="same_side_top5_qty_bucket", extractor=lambda row: row.get("same_side_top5_qty"), categorical=False))

    latency_rows: list[dict[str, Any]] = []
    latency_rows.extend(build_strata_gap_rows(matched_pairs, family_name="latency_signal_ms_bucket", extractor=lambda row: row.get("latency_signal_ms"), categorical=False))
    latency_rows.extend(build_strata_gap_rows(matched_pairs, family_name="top5_join_age_ms_bucket", extractor=lambda row: row.get("top5_join_age_ms"), categorical=False))
    latency_rows.extend(build_strata_gap_rows(matched_pairs, family_name="join_stale", extractor=lambda row: row.get("join_stale"), categorical=True))

    decision_state = determine_decision_state(
        coverage_summary=coverage_summary,
        fill_horizon_rows=fill_horizon_gap_rows,
        final_state_rows=final_state_gap_rows,
        cancel_race_rows=cancel_race_gap_rows,
        live_filled_orders=int(live_bundle["row_counts"]["filled_orders"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "submit_key_coverage.csv", submit_key_rows, fieldnames=list(submit_key_rows[0].keys()) if submit_key_rows else [])
    _write_csv(output_dir / "coverage_gap.csv", coverage_gap_rows, fieldnames=list(coverage_gap_rows[0].keys()) if coverage_gap_rows else [])
    _write_csv(output_dir / "fill_horizon_gap.csv", fill_horizon_gap_rows, fieldnames=list(fill_horizon_gap_rows[0].keys()) if fill_horizon_gap_rows else [])
    _write_csv(output_dir / "time_to_fill_gap.csv", time_to_fill_gap_rows, fieldnames=list(time_to_fill_gap_rows[0].keys()) if time_to_fill_gap_rows else [])
    _write_csv(output_dir / "final_state_gap.csv", final_state_gap_rows, fieldnames=list(final_state_gap_rows[0].keys()) if final_state_gap_rows else [])
    _write_csv(output_dir / "cancel_race_gap.csv", cancel_race_gap_rows, fieldnames=list(cancel_race_gap_rows[0].keys()) if cancel_race_gap_rows else [])
    _write_csv(output_dir / "markout_gap.csv", markout_gap_rows, fieldnames=list(markout_gap_rows[0].keys()) if markout_gap_rows else [])
    _write_csv(output_dir / "placement_strata_gap.csv", placement_rows, fieldnames=list(placement_rows[0].keys()) if placement_rows else [])
    _write_csv(output_dir / "inventory_strata_gap.csv", inventory_rows, fieldnames=list(inventory_rows[0].keys()) if inventory_rows else [])
    _write_csv(output_dir / "latency_strata_gap.csv", latency_rows, fieldnames=list(latency_rows[0].keys()) if latency_rows else [])

    stage3_payload = _load_json(stage3_json) if stage3_json.exists() else {}
    sidecar_metrics = _load_json(sidecar_metrics_json) if sidecar_metrics_json.exists() else {}
    join_metrics = _load_json(join_metrics_json) if join_metrics_json.exists() else {}
    manifest = {
        "task_id": TASK_ID,
        "generated_at": _generated_at(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "decision_state": decision_state,
        "input_hashes": {
            "live_audit_csv": _hash_file(live_audit_csv),
            "replay_audit_csv": _hash_file(replay_audit_csv),
            "joined_decisions_csv": _hash_file(joined_csv),
            "maker_acceptance_stage3_json": _hash_file(stage3_json) if stage3_json.exists() else "",
            "sidecar_metrics_json": _hash_file(sidecar_metrics_json) if sidecar_metrics_json.exists() else "",
            "joined_decisions_metrics_json": _hash_file(join_metrics_json) if join_metrics_json.exists() else "",
        },
        "parameters": {
            "horizons_ms": [int(value) for value in horizons_ms],
            "tick_size": tick_size,
            "max_future_gap_ms": max_future_gap_ms,
            "maker_fee_bps": maker_fee_bps,
        },
        "stage3_classification": (
            stage3_payload.get("market_view", {}).get("classification")
            or stage3_payload.get("classification")
            or "unknown"
        ),
        "sidecar_metrics": sidecar_metrics,
        "joined_decisions_metrics": join_metrics,
        "row_counts": {
            "live_submit_orders": int(live_bundle["row_counts"]["submit_orders"]),
            "replay_submit_orders": int(replay_bundle["row_counts"]["submit_orders"]),
            "matched_submit_orders": int(coverage_summary["matched_submit_rows"]),
            "live_filled_orders": int(live_bundle["row_counts"]["filled_orders"]),
            "replay_filled_orders": int(replay_bundle["row_counts"]["filled_orders"]),
            "live_fill_after_cancel_orders": int(live_bundle["row_counts"]["fill_after_cancel_orders"]),
            "replay_fill_after_cancel_orders": int(replay_bundle["row_counts"]["fill_after_cancel_orders"]),
        },
        "artifacts": [
            "execution_calibration_summary.md",
            "submit_key_coverage.csv",
            "fill_horizon_gap.csv",
            "time_to_fill_gap.csv",
            "final_state_gap.csv",
            "cancel_race_gap.csv",
            "markout_gap.csv",
            "placement_strata_gap.csv",
            "inventory_strata_gap.csv",
            "latency_strata_gap.csv",
            "coverage_gap.csv",
            "run_manifest.json",
        ],
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    write_summary_markdown(
        output_dir / "execution_calibration_summary.md",
        run_dir=run_dir,
        output_dir=output_dir,
        manifest=manifest,
        coverage_summary=coverage_summary,
        fill_horizon_rows=fill_horizon_gap_rows,
        final_state_rows=final_state_gap_rows,
        cancel_race_rows=cancel_race_gap_rows,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Run directory containing live audit, audit replay, and T009 artifacts.")
    parser.add_argument(
        "--output-dir",
        help=f"Output directory. Default: <run-dir>/stage6_execution_calibration_{TASK_ID}",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        default=DEFAULT_TICK_SIZE,
        help=f"Tick size used for tick-normalized labels. Default: {DEFAULT_TICK_SIZE}",
    )
    parser.add_argument(
        "--maker-fee-bps",
        type=float,
        default=DEFAULT_MAKER_FEE_BPS,
        help=f"Maker fee assumption in bps for fee-adjusted proxy labels. Default: {DEFAULT_MAKER_FEE_BPS}",
    )
    parser.add_argument(
        "--max-future-gap-ms",
        type=float,
        default=DEFAULT_MAX_FUTURE_GAP_MS,
        help=f"Maximum allowed future-decision gap for markout labels. Default: {DEFAULT_MAX_FUTURE_GAP_MS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _expand(args.run_dir)
    output_dir = _expand(args.output_dir) if args.output_dir else run_dir / f"stage6_execution_calibration_{TASK_ID}"
    manifest = run_execution_outcome_calibration(
        run_dir=run_dir,
        output_dir=output_dir,
        tick_size=float(args.tick_size),
        max_future_gap_ms=float(args.max_future_gap_ms),
        maker_fee_bps=float(args.maker_fee_bps),
    )
    print(json.dumps({"task_id": TASK_ID, "output_dir": str(output_dir), "decision_state": manifest["decision_state"], "row_counts": manifest["row_counts"]}, indent=2))


if __name__ == "__main__":
    main()
