#!/usr/bin/env python3
"""Read-only Step 5B quote-anchor / post-only diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TASK_ID = "0518T003"
DEFAULT_TICK_SIZE = 0.1
DEFAULT_LATENCY_GUARD_MS = 5.0
DEFAULT_JOIN_STALE_MS = 250.0


@dataclass(frozen=True)
class Top5Snapshot:
    raw_seq: int
    local_ts: int
    bid_tick: int | None
    ask_tick: int | None
    bookticker_bid_tick: int | None
    bookticker_ask_tick: int | None
    bookticker_depth_age_ms: float
    startup_excluded: bool
    sync_gap: bool
    sync_waiting_snapshot: bool


@dataclass(frozen=True)
class DecisionRow:
    strategy_seq: int
    ts_local: int
    market_view_source: str
    top5_source: str
    audit_bid_tick: int | None
    audit_ask_tick: int | None
    top5_bid_tick: int | None
    top5_ask_tick: int | None
    bookticker_bid_tick: int | None
    bookticker_ask_tick: int | None
    target_bid_tick: int | None
    target_ask_tick: int | None
    fair: float
    reservation: float
    half_spread: float
    feed_latency_ms: float
    latency_signal_ms: float
    book_view_stale_ms: float
    join_stale: bool
    join_gap_crossed: bool
    join_missing: bool
    join_used_future: bool
    top5_join_age_ms: float
    depth_join_age_ms: float
    bookticker_join_age_ms: float
    max_join_age_ms: float
    action: str
    planned_action: str
    reject_reason: str
    throttle_reason: str


def _expand(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _parse_bool(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


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


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_div(num: float, denom: float) -> float:
    if not _finite(num) or not _finite(denom) or float(denom) == 0.0:
        return math.nan
    return float(num) / float(denom)


def _mean(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return math.nan
    return float(np.mean(np.asarray(finite, dtype=np.float64)))


def _quantile(values: Iterable[float], q: float) -> float:
    finite = [float(value) for value in values if _finite(value)]
    if not finite:
        return math.nan
    return float(np.quantile(np.asarray(finite, dtype=np.float64), q))


def _rate(count: int, total: int) -> float:
    return _safe_div(float(count), float(total))


def _price_to_tick(price: Any, tick_size: float) -> int | None:
    px = _float(price)
    if not _finite(px) or tick_size <= 0.0:
        return None
    return int(round(px / tick_size))


def _floor_tick(price: float, tick_size: float) -> int | None:
    if not _finite(price) or tick_size <= 0.0:
        return None
    return int(math.floor(price / tick_size))


def _ceil_tick(price: float, tick_size: float) -> int | None:
    if not _finite(price) or tick_size <= 0.0:
        return None
    return int(math.ceil(price / tick_size))


def _first_pipe_int(value: Any) -> int | None:
    for part in str(value or "").split("|"):
        parsed = _int(part)
        if parsed is not None:
            return parsed
    return None


def _first_pipe_price_tick(value: Any, tick_size: float) -> int | None:
    for part in str(value or "").split("|"):
        parsed = _price_to_tick(part, tick_size)
        if parsed is not None:
            return parsed
    return None


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generated_at() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _find_audit_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("audit_live*.csv"))
    if not candidates:
        raise FileNotFoundError(f"no audit_live*.csv found in {run_dir}")
    return candidates[0]


def _read_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config_live.toml"
    if not config_path.exists():
        return {}
    return tomllib.loads(config_path.read_text(encoding="utf-8"))


def _config_float(config: dict[str, Any], section: str, key: str, default: float) -> float:
    value = config.get(section, {}).get(key, default) if config else default
    parsed = _float(value, default)
    return float(parsed) if _finite(parsed) else float(default)


def load_top5_snapshots(path: Path, tick_size: float) -> dict[int, Top5Snapshot]:
    snapshots: dict[int, Top5Snapshot] = {}
    for row in _load_csv(path):
        raw_seq = _int(row.get("raw_seq"))
        if raw_seq is None:
            continue
        snapshots[raw_seq] = Top5Snapshot(
            raw_seq=raw_seq,
            local_ts=_int(row.get("local_ts"), 0) or 0,
            bid_tick=_first_pipe_int(row.get("bid_top5_ticks"))
            or _first_pipe_price_tick(row.get("bid_top5_px"), tick_size),
            ask_tick=_first_pipe_int(row.get("ask_top5_ticks"))
            or _first_pipe_price_tick(row.get("ask_top5_px"), tick_size),
            bookticker_bid_tick=_price_to_tick(row.get("bookticker_bid_px"), tick_size),
            bookticker_ask_tick=_price_to_tick(row.get("bookticker_ask_px"), tick_size),
            bookticker_depth_age_ms=_float(row.get("bookticker_depth_age_ms")),
            startup_excluded=_parse_bool(row.get("startup_excluded")),
            sync_gap=_parse_bool(row.get("sync_gap")),
            sync_waiting_snapshot=_parse_bool(row.get("sync_waiting_snapshot")),
        )
    return snapshots


def load_joined_decisions(path: Path) -> dict[int, dict[str, str]]:
    joined: dict[int, dict[str, str]] = {}
    for row in _load_csv(path):
        seq = _int(row.get("strategy_seq"))
        if seq is not None:
            joined[seq] = row
    return joined


def load_decision_rows(
    *,
    audit_csv: Path,
    joined_decisions_csv: Path,
    top5_sidecar_csv: Path,
    tick_size: float,
) -> list[DecisionRow]:
    joined = load_joined_decisions(joined_decisions_csv)
    snapshots = load_top5_snapshots(top5_sidecar_csv, tick_size)
    rows: list[DecisionRow] = []
    for audit in _load_csv(audit_csv):
        if audit.get("event_type") != "decision":
            continue
        seq = _int(audit.get("strategy_seq"))
        ts_local = _int(audit.get("ts_local"))
        if seq is None or ts_local is None:
            continue
        join = joined.get(seq, {})
        snapshot = snapshots.get(_int(join.get("joined_raw_seq"), -1) or -1)
        rows.append(
            DecisionRow(
                strategy_seq=seq,
                ts_local=ts_local,
                market_view_source=str(audit.get("market_view_source", "")),
                top5_source=str(audit.get("top5_source", "")),
                audit_bid_tick=_price_to_tick(audit.get("best_bid"), tick_size),
                audit_ask_tick=_price_to_tick(audit.get("best_ask"), tick_size),
                top5_bid_tick=(snapshot.bid_tick if snapshot else None)
                or _int(audit.get("top5_depth_best_bid_tick")),
                top5_ask_tick=(snapshot.ask_tick if snapshot else None)
                or _int(audit.get("top5_depth_best_ask_tick")),
                bookticker_bid_tick=snapshot.bookticker_bid_tick if snapshot else None,
                bookticker_ask_tick=snapshot.bookticker_ask_tick if snapshot else None,
                target_bid_tick=_int(audit.get("target_bid_tick")),
                target_ask_tick=_int(audit.get("target_ask_tick")),
                fair=_float(audit.get("fair")),
                reservation=_float(audit.get("reservation")),
                half_spread=_float(audit.get("half_spread")),
                feed_latency_ms=_safe_div(_float(audit.get("feed_latency_ns")), 1_000_000.0),
                latency_signal_ms=_float(audit.get("latency_signal_ms")),
                book_view_stale_ms=_float(audit.get("book_view_stale_ms")),
                join_stale=_parse_bool(join.get("join_stale")),
                join_gap_crossed=_parse_bool(join.get("join_gap_crossed")),
                join_missing=_parse_bool(join.get("join_missing")) or not bool(join),
                join_used_future=_parse_bool(join.get("join_used_future")),
                top5_join_age_ms=_float(join.get("top5_join_age_ms")),
                depth_join_age_ms=_float(join.get("depth_join_age_ms")),
                bookticker_join_age_ms=_float(join.get("bookticker_join_age_ms")),
                max_join_age_ms=_float(join.get("max_join_age_ms")),
                action=str(audit.get("action", "")),
                planned_action=str(audit.get("planned_action", "")),
                reject_reason=str(audit.get("reject_reason", "")),
                throttle_reason=str(audit.get("throttle_reason", "")),
            )
        )
    return rows


def _source_ticks(row: DecisionRow, source: str) -> tuple[int | None, int | None]:
    if source == "audit_depth":
        return row.audit_bid_tick, row.audit_ask_tick
    if source == "top5_depth":
        return row.top5_bid_tick, row.top5_ask_tick
    if source == "bookticker":
        return row.bookticker_bid_tick, row.bookticker_ask_tick
    raise ValueError(f"unknown source {source}")


def _valid_bbo(bid_tick: int | None, ask_tick: int | None) -> bool:
    return bid_tick is not None and ask_tick is not None and ask_tick > bid_tick


def _post_only_risk(
    *,
    bid_tick: int | None,
    ask_tick: int | None,
    anchor_bid_tick: int | None,
    anchor_ask_tick: int | None,
) -> bool:
    if not _valid_bbo(anchor_bid_tick, anchor_ask_tick):
        return True
    bid_risk = bid_tick is not None and bid_tick > int(anchor_bid_tick)
    ask_risk = ask_tick is not None and ask_tick < int(anchor_ask_tick)
    cross_risk = (
        bid_tick is not None
        and ask_tick is not None
        and (bid_tick >= int(anchor_ask_tick) or ask_tick <= int(anchor_bid_tick))
    )
    return bool(bid_risk or ask_risk or cross_risk)


def build_bbo_source_drift(decisions: list[DecisionRow]) -> list[dict[str, Any]]:
    source_pairs = (
        ("audit_depth", "top5_depth"),
        ("audit_depth", "bookticker"),
        ("bookticker", "top5_depth"),
    )
    rows: list[dict[str, Any]] = []
    for left, right in source_pairs:
        for side in ("bid", "ask"):
            diffs: list[float] = []
            stale_diffs: list[float] = []
            missing = 0
            stale_rows = 0
            for decision in decisions:
                left_bid, left_ask = _source_ticks(decision, left)
                right_bid, right_ask = _source_ticks(decision, right)
                lhs = left_bid if side == "bid" else left_ask
                rhs = right_bid if side == "bid" else right_ask
                if lhs is None or rhs is None:
                    missing += 1
                    continue
                diff = float(lhs - rhs)
                diffs.append(diff)
                if decision.join_stale:
                    stale_rows += 1
                    stale_diffs.append(diff)
            rows.append(
                {
                    "source_pair": f"{left}_vs_{right}",
                    "side": side,
                    "rows": len(diffs),
                    "missing_rows": missing,
                    "exact_match_rows": sum(1 for value in diffs if value == 0.0),
                    "mismatch_rows": sum(1 for value in diffs if value != 0.0),
                    "mismatch_rate": _rate(sum(1 for value in diffs if value != 0.0), len(diffs)),
                    "mean_signed_drift_ticks": _mean(diffs),
                    "mean_abs_drift_ticks": _mean(abs(value) for value in diffs),
                    "p99_abs_drift_ticks": _quantile([abs(value) for value in diffs], 0.99),
                    "max_abs_drift_ticks": max([abs(value) for value in diffs], default=math.nan),
                    "stale_rows_with_pair": stale_rows,
                    "stale_mean_abs_drift_ticks": _mean(abs(value) for value in stale_diffs),
                }
            )
    return rows


def _bucket_from_distance(distance: float) -> str:
    if not _finite(distance):
        return "unknown"
    if distance < 0:
        return "inside_or_crossed"
    if distance == 0:
        return "touch"
    if distance <= 1:
        return "step_back_1"
    return "step_back_gt1"


def _group_key(row: dict[str, str], names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(row.get(name, "") or "unknown") for name in names)


def build_quote_distance_summary(execution_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in execution_rows:
        bucket = str(row.get("placement_bucket") or _bucket_from_distance(_float(row.get("distance_to_bbo_ticks"))))
        side = str(row.get("order_side") or "unknown")
        groups[(bucket, side)].append(row)
    out: list[dict[str, Any]] = []
    for (bucket, side), rows in sorted(groups.items()):
        out.append(
            {
                "placement_bucket": bucket,
                "order_side": side,
                "submit_rows": len(rows),
                "distance_to_bbo_ticks_mean": _mean(_float(row.get("distance_to_bbo_ticks")) for row in rows),
                "edge_vs_fair_ticks_mean": _mean(_float(row.get("edge_vs_fair_ticks")) for row in rows),
                "post_only_risk_rows": sum(1 for row in rows if _parse_bool(row.get("post_only_risk"))),
                "fill_500ms_rate": _mean(_float(row.get("fill_by_500ms")) for row in rows),
                "fill_5000ms_rate": _mean(_float(row.get("fill_by_5000ms")) for row in rows),
                "time_to_fill_ms_mean": _mean(_float(row.get("time_to_fill_ms")) for row in rows),
                "markout_500ms_ticks_mean": _mean(_float(row.get("fill_markout_500ms_ticks")) for row in rows),
                "fill_after_cancel_request_rows": sum(
                    1 for row in rows if _parse_bool(row.get("fill_after_cancel_request"))
                ),
                "fast_cancel_churn_rows": sum(1 for row in rows if _parse_bool(row.get("fast_cancel_churn"))),
            }
        )
    return out


def build_reject_throttle_churn_summary(
    decisions: list[DecisionRow],
    execution_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reject_counts = Counter(row.reject_reason or "none" for row in decisions)
    throttle_counts = Counter(row.throttle_reason or "none" for row in decisions)
    for reason, count in sorted(reject_counts.items()):
        rows.append({"category": "decision_reject_reason", "bucket": reason, "rows": count})
    for reason, count in sorted(throttle_counts.items()):
        rows.append({"category": "decision_throttle_reason", "bucket": reason, "rows": count})
    rows.extend(
        [
            {
                "category": "submit_post_only_risk",
                "bucket": "true",
                "rows": sum(1 for row in execution_rows if _parse_bool(row.get("post_only_risk"))),
            },
            {
                "category": "submit_post_only_risk",
                "bucket": "false",
                "rows": sum(1 for row in execution_rows if not _parse_bool(row.get("post_only_risk"))),
            },
            {
                "category": "submit_fast_cancel_churn",
                "bucket": "true",
                "rows": sum(1 for row in execution_rows if _parse_bool(row.get("fast_cancel_churn"))),
            },
            {
                "category": "submit_fast_cancel_churn",
                "bucket": "false",
                "rows": sum(1 for row in execution_rows if not _parse_bool(row.get("fast_cancel_churn"))),
            },
        ]
    )
    return rows


def _latency_bucket(value: float, guard_ms: float) -> str:
    if not _finite(value):
        return "missing"
    if value <= guard_ms:
        return "within_guard"
    return "above_guard"


def _join_bucket(row: DecisionRow, stale_ms: float) -> str:
    if row.join_missing:
        return "join_missing"
    if row.join_gap_crossed:
        return "join_gap_crossed"
    if row.join_used_future:
        return "join_used_future"
    if row.join_stale or (_finite(row.max_join_age_ms) and row.max_join_age_ms > stale_ms):
        return "join_stale"
    return "accepted_non_stale"


def build_stale_latency_summary(
    decisions: list[DecisionRow],
    *,
    latency_guard_ms: float,
    join_stale_ms: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[DecisionRow]] = defaultdict(list)
    for row in decisions:
        groups[(_join_bucket(row, join_stale_ms), _latency_bucket(row.feed_latency_ms, latency_guard_ms))].append(row)
    out: list[dict[str, Any]] = []
    for (join_bucket, latency_bucket), rows in sorted(groups.items()):
        submit_like = [row for row in rows if "submit" in row.action or "submit" in row.planned_action]
        reject_like = [row for row in rows if row.reject_reason]
        throttle_like = [row for row in rows if row.throttle_reason]
        out.append(
            {
                "join_quality_bucket": join_bucket,
                "feed_latency_bucket": latency_bucket,
                "decision_rows": len(rows),
                "submit_or_planned_submit_rows": len(submit_like),
                "reject_reason_rows": len(reject_like),
                "throttle_reason_rows": len(throttle_like),
                "feed_latency_ms_mean": _mean(row.feed_latency_ms for row in rows),
                "feed_latency_ms_p99": _quantile([row.feed_latency_ms for row in rows], 0.99),
                "max_join_age_ms_mean": _mean(row.max_join_age_ms for row in rows),
                "max_join_age_ms_p99": _quantile([row.max_join_age_ms for row in rows], 0.99),
                "bookticker_join_age_ms_p99": _quantile([row.bookticker_join_age_ms for row in rows], 0.99),
            }
        )
    return out


def _anchor_state(row: dict[str, str]) -> str:
    if _parse_bool(row.get("join_gap_crossed")):
        return "join_gap_crossed"
    if _parse_bool(row.get("join_stale")):
        return "join_stale"
    source = row.get("market_view_source") or "unknown_market_view"
    top5 = row.get("top5_source") or "unknown_top5"
    return f"{source}|{top5}"


def build_anchor_fill_markout_summary(execution_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in execution_rows:
        groups[(_anchor_state(row), str(row.get("placement_bucket") or "unknown"))].append(row)
    out: list[dict[str, Any]] = []
    for (anchor_state, placement), rows in sorted(groups.items()):
        out.append(
            {
                "anchor_state": anchor_state,
                "placement_bucket": placement,
                "submit_rows": len(rows),
                "fill_rate": _mean(1.0 - _float(row.get("no_fill"), 1.0) for row in rows),
                "fill_500ms_rate": _mean(_float(row.get("fill_by_500ms")) for row in rows),
                "fill_5000ms_rate": _mean(_float(row.get("fill_by_5000ms")) for row in rows),
                "time_to_fill_ms_mean": _mean(_float(row.get("time_to_fill_ms")) for row in rows),
                "markout_500ms_ticks_mean": _mean(_float(row.get("fill_markout_500ms_ticks")) for row in rows),
                "markout_5000ms_ticks_mean": _mean(_float(row.get("fill_markout_5000ms_ticks")) for row in rows),
                "realized_spread_proxy_ticks_mean": _mean(
                    _float(row.get("realized_spread_proxy_ticks")) for row in rows
                ),
                "fill_after_cancel_request_rows": sum(
                    1 for row in rows if _parse_bool(row.get("fill_after_cancel_request"))
                ),
                "observed_only_note": "fill/markout labels are observed-only and not counterfactual fill proof",
            }
        )
    return out


def build_enforcement_gap_matrix(
    *,
    config: dict[str, Any],
    decisions: list[DecisionRow],
    execution_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    market_sources = sorted({row.market_view_source or "missing" for row in decisions})
    top5_sources = sorted({row.top5_source or "missing" for row in decisions})
    reject_reasons = sorted({row.reject_reason for row in decisions if row.reject_reason})
    quote_throttle_enabled = bool(config.get("strategy", {}).get("quote_throttle_enabled", False))
    two_phase_replace = bool(config.get("strategy", {}).get("two_phase_replace_enabled", False))
    latency_guard_ms = _config_float(config, "latency", "latency_guard_ms", DEFAULT_LATENCY_GUARD_MS)
    api_limit_enabled = bool(config.get("api_limit", {}).get("enabled", False)) if config else False
    post_only_risk_rows = sum(1 for row in execution_rows if _parse_bool(row.get("post_only_risk")))

    return [
        {
            "constraint": "fast_bbo_bookticker_hard_anchor",
            "current_status": "design_gap",
            "current_mechanism": "decision path uses market_view_source from depth; bookTicker appears only in sidecar diagnostics",
            "evidence": f"market_view_sources={market_sources}; top5_sources={top5_sources}",
            "next_requirement": "implement explicit fast BBO/bookTicker anchor arbitration before claiming hard anchor",
        },
        {
            "constraint": "depth_bbo_guarded_fallback_only",
            "current_status": "design_gap",
            "current_mechanism": "depth BBO is still the primary live/replay MarketView input",
            "evidence": f"market_view_sources={market_sources}",
            "next_requirement": "make depth BBO secondary or guarded fallback only after Step 5B thresholds are accepted",
        },
        {
            "constraint": "top5_not_final_hard_anchor",
            "current_status": "currently_satisfied",
            "current_mechanism": "top5 is present as pricing/risk/audit fields and labels; submit path is not keyed directly to top5 reconstructed BBO",
            "evidence": f"top5_sources={top5_sources}; submit_rows={len(execution_rows)}",
            "next_requirement": "keep top5 as pricing/risk/diagnostic input unless later diagnostics justify promotion",
        },
        {
            "constraint": "side_conservative_rounding_and_post_round_recheck",
            "current_status": "design_gap",
            "current_mechanism": "current strategy records target ticks after clamp->round_to_tick; no explicit bid=floor/ask=ceil helper or post-round recheck is present",
            "evidence": f"post_only_risk_rows_in_stage5_labels={post_only_risk_rows}",
            "next_requirement": "use rounding_clamp_counterfactual output to scope a default-off implementation",
        },
        {
            "constraint": "stale_latency_join_age_submit_suppression",
            "current_status": "partial",
            "current_mechanism": "feed latency guard exists; join-age guard is sidecar/diagnostic-only and not in live quote control",
            "evidence": f"latency_guard_ms={latency_guard_ms}",
            "next_requirement": "define accepted anchor age/join-age thresholds before strategy implementation",
        },
        {
            "constraint": "reject_throttle_drop_cooldown_path",
            "current_status": "partial",
            "current_mechanism": "GTX, api interval, token bucket, quote throttle and two-phase replace exist; no post-only reject specific cooldown/backoff is proven here",
            "evidence": (
                f"api_limit_enabled={api_limit_enabled}; quote_throttle_enabled={quote_throttle_enabled}; "
                f"two_phase_replace_enabled={two_phase_replace}; reject_reasons={reject_reasons}"
            ),
            "next_requirement": "quantify reject/throttle/churn and add explicit reject response only in later implementation task",
        },
    ]


def _design_ticks(row: DecisionRow, anchor_source: str, tick_size: float) -> tuple[int | None, int | None]:
    anchor_bid, anchor_ask = _source_ticks(row, anchor_source)
    raw_bid = row.reservation - row.half_spread
    raw_ask = row.reservation + row.half_spread
    bid = _floor_tick(raw_bid, tick_size)
    ask = _ceil_tick(raw_ask, tick_size)
    if bid is not None and anchor_bid is not None:
        bid = min(bid, anchor_bid)
    if ask is not None and anchor_ask is not None:
        ask = max(ask, anchor_ask)
    return bid, ask


def build_rounding_clamp_counterfactual(decisions: list[DecisionRow], *, tick_size: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in ("audit_depth", "bookticker", "top5_depth"):
        valid = 0
        current_risk = 0
        design_risk = 0
        current_bid_risk = 0
        current_ask_risk = 0
        current_cross_risk = 0
        design_changed_bid = 0
        design_changed_ask = 0
        missing_anchor = 0
        for decision in decisions:
            anchor_bid, anchor_ask = _source_ticks(decision, source)
            if not _valid_bbo(anchor_bid, anchor_ask):
                missing_anchor += 1
                continue
            valid += 1
            current_bid = decision.target_bid_tick
            current_ask = decision.target_ask_tick
            current_bid_bad = current_bid is not None and current_bid > int(anchor_bid)
            current_ask_bad = current_ask is not None and current_ask < int(anchor_ask)
            current_cross_bad = (
                current_bid is not None
                and current_ask is not None
                and (current_bid >= int(anchor_ask) or current_ask <= int(anchor_bid))
            )
            if current_bid_bad:
                current_bid_risk += 1
            if current_ask_bad:
                current_ask_risk += 1
            if current_cross_bad:
                current_cross_risk += 1
            if current_bid_bad or current_ask_bad or current_cross_bad:
                current_risk += 1
            design_bid, design_ask = _design_ticks(decision, source, tick_size)
            if _post_only_risk(
                bid_tick=design_bid,
                ask_tick=design_ask,
                anchor_bid_tick=anchor_bid,
                anchor_ask_tick=anchor_ask,
            ):
                design_risk += 1
            if current_bid is not None and design_bid is not None and current_bid != design_bid:
                design_changed_bid += 1
            if current_ask is not None and design_ask is not None and current_ask != design_ask:
                design_changed_ask += 1
        rows.append(
            {
                "anchor_source": source,
                "decision_rows_with_valid_anchor": valid,
                "missing_or_invalid_anchor_rows": missing_anchor,
                "current_post_round_risk_rows": current_risk,
                "current_post_round_risk_rate": _rate(current_risk, valid),
                "current_bid_above_anchor_rows": current_bid_risk,
                "current_ask_below_anchor_rows": current_ask_risk,
                "current_crossed_bbo_rows": current_cross_risk,
                "design_post_round_risk_rows": design_risk,
                "design_post_round_risk_rate": _rate(design_risk, valid),
                "design_changed_bid_rows": design_changed_bid,
                "design_changed_ask_rows": design_changed_ask,
                "counterfactual_note": "read-only comparison; no strategy behavior changed",
            }
        )
    return rows


def _summary_value(rows: list[dict[str, Any]], key: str, match_key: str, match_value: str) -> Any:
    for row in rows:
        if str(row.get(match_key)) == match_value:
            return row.get(key)
    return ""


def write_summary(
    path: Path,
    *,
    row_counts: dict[str, Any],
    bbo_rows: list[dict[str, Any]],
    enforcement_rows: list[dict[str, Any]],
    counterfactual_rows: list[dict[str, Any]],
) -> None:
    audit_vs_book_bid = next(
        (
            row
            for row in bbo_rows
            if row["source_pair"] == "audit_depth_vs_bookticker" and row["side"] == "bid"
        ),
        {},
    )
    audit_vs_book_ask = next(
        (
            row
            for row in bbo_rows
            if row["source_pair"] == "audit_depth_vs_bookticker" and row["side"] == "ask"
        ),
        {},
    )
    audit_counter = next((row for row in counterfactual_rows if row["anchor_source"] == "audit_depth"), {})
    book_counter = next((row for row in counterfactual_rows if row["anchor_source"] == "bookticker"), {})
    design_gaps = [row["constraint"] for row in enforcement_rows if row["current_status"] == "design_gap"]
    partials = [row["constraint"] for row in enforcement_rows if row["current_status"] == "partial"]
    readiness = "diagnostic_only_not_ready_for_direct_implementation" if design_gaps else "candidate_for_default_off_design_review"

    text = f"""# Step 5B Quote Anchor Diagnostic Summary

- task_id: `{TASK_ID}`
- generated_at: `{_generated_at()}`
- decision_rows: `{row_counts.get('decision_rows', 0)}`
- submit_label_rows: `{row_counts.get('submit_label_rows', 0)}`
- readiness: `{readiness}`

## BBO Drift

- audit_depth vs bookTicker bid mismatch rate: `{audit_vs_book_bid.get('mismatch_rate')}`
- audit_depth vs bookTicker ask mismatch rate: `{audit_vs_book_ask.get('mismatch_rate')}`
- audit_depth vs bookTicker bid p99 abs drift ticks: `{audit_vs_book_bid.get('p99_abs_drift_ticks')}`
- audit_depth vs bookTicker ask p99 abs drift ticks: `{audit_vs_book_ask.get('p99_abs_drift_ticks')}`

## Rounding / Clamp Counterfactual

- current path vs audit_depth post-round risk rows: `{audit_counter.get('current_post_round_risk_rows')}`
- T002 design path vs audit_depth post-round risk rows: `{audit_counter.get('design_post_round_risk_rows')}`
- current path vs bookTicker post-round risk rows: `{book_counter.get('current_post_round_risk_rows')}`
- T002 design path vs bookTicker post-round risk rows: `{book_counter.get('design_post_round_risk_rows')}`

## Enforcement Gaps

- design gaps: `{', '.join(design_gaps) if design_gaps else 'none'}`
- partial coverage: `{', '.join(partials) if partials else 'none'}`

## Boundary

- This runner is read-only.
- It does not modify quote placement, fair/reservation formulas, risk guards, live scripts, replay lifecycle, or generated standard schemas.
- Fill, markout, opportunity-cost, and PnL outputs remain observed-only and are not counterfactual fill proof.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def run_quote_anchor_diagnostic(
    *,
    run_dir: Path,
    output_dir: Path,
    tick_size: float | None = None,
) -> dict[str, Any]:
    run_dir = _expand(run_dir)
    output_dir = _expand(output_dir)
    config = _read_config(run_dir)
    tick = float(tick_size or _config_float(config, "market", "tick_size", DEFAULT_TICK_SIZE))
    latency_guard_ms = _config_float(config, "latency", "latency_guard_ms", DEFAULT_LATENCY_GUARD_MS)
    join_stale_ms = DEFAULT_JOIN_STALE_MS

    audit_csv = _find_audit_csv(run_dir)
    sidecar_dir = run_dir / "t009_fixed_sidecar"
    joined_decisions_csv = sidecar_dir / "joined_decisions.csv"
    top5_sidecar_csv = sidecar_dir / "top5_sidecar.csv"
    stage5_dir = run_dir / "stage5_execution_outcome_labels_0514T005"
    execution_labels_csv = stage5_dir / "execution_outcome_labels.csv"

    decisions = load_decision_rows(
        audit_csv=audit_csv,
        joined_decisions_csv=joined_decisions_csv,
        top5_sidecar_csv=top5_sidecar_csv,
        tick_size=tick,
    )
    execution_rows = _load_csv(execution_labels_csv)

    bbo_rows = build_bbo_source_drift(decisions)
    quote_distance_rows = build_quote_distance_summary(execution_rows)
    reject_rows = build_reject_throttle_churn_summary(decisions, execution_rows)
    stale_latency_rows = build_stale_latency_summary(
        decisions,
        latency_guard_ms=latency_guard_ms,
        join_stale_ms=join_stale_ms,
    )
    anchor_fill_rows = build_anchor_fill_markout_summary(execution_rows)
    enforcement_rows = build_enforcement_gap_matrix(
        config=config,
        decisions=decisions,
        execution_rows=execution_rows,
    )
    counterfactual_rows = build_rounding_clamp_counterfactual(decisions, tick_size=tick)

    _write_csv(
        output_dir / "bbo_source_drift.csv",
        bbo_rows,
        [
            "source_pair",
            "side",
            "rows",
            "missing_rows",
            "exact_match_rows",
            "mismatch_rows",
            "mismatch_rate",
            "mean_signed_drift_ticks",
            "mean_abs_drift_ticks",
            "p99_abs_drift_ticks",
            "max_abs_drift_ticks",
            "stale_rows_with_pair",
            "stale_mean_abs_drift_ticks",
        ],
    )
    _write_csv(
        output_dir / "quote_distance_bucket_summary.csv",
        quote_distance_rows,
        [
            "placement_bucket",
            "order_side",
            "submit_rows",
            "distance_to_bbo_ticks_mean",
            "edge_vs_fair_ticks_mean",
            "post_only_risk_rows",
            "fill_500ms_rate",
            "fill_5000ms_rate",
            "time_to_fill_ms_mean",
            "markout_500ms_ticks_mean",
            "fill_after_cancel_request_rows",
            "fast_cancel_churn_rows",
        ],
    )
    _write_csv(output_dir / "post_only_reject_throttle_churn_summary.csv", reject_rows, ["category", "bucket", "rows"])
    _write_csv(
        output_dir / "stale_latency_guard_summary.csv",
        stale_latency_rows,
        [
            "join_quality_bucket",
            "feed_latency_bucket",
            "decision_rows",
            "submit_or_planned_submit_rows",
            "reject_reason_rows",
            "throttle_reason_rows",
            "feed_latency_ms_mean",
            "feed_latency_ms_p99",
            "max_join_age_ms_mean",
            "max_join_age_ms_p99",
            "bookticker_join_age_ms_p99",
        ],
    )
    _write_csv(
        output_dir / "anchor_source_x_fill_markout.csv",
        anchor_fill_rows,
        [
            "anchor_state",
            "placement_bucket",
            "submit_rows",
            "fill_rate",
            "fill_500ms_rate",
            "fill_5000ms_rate",
            "time_to_fill_ms_mean",
            "markout_500ms_ticks_mean",
            "markout_5000ms_ticks_mean",
            "realized_spread_proxy_ticks_mean",
            "fill_after_cancel_request_rows",
            "observed_only_note",
        ],
    )
    _write_csv(
        output_dir / "current_enforcement_gap_matrix.csv",
        enforcement_rows,
        ["constraint", "current_status", "current_mechanism", "evidence", "next_requirement"],
    )
    _write_csv(
        output_dir / "rounding_clamp_counterfactual.csv",
        counterfactual_rows,
        [
            "anchor_source",
            "decision_rows_with_valid_anchor",
            "missing_or_invalid_anchor_rows",
            "current_post_round_risk_rows",
            "current_post_round_risk_rate",
            "current_bid_above_anchor_rows",
            "current_ask_below_anchor_rows",
            "current_crossed_bbo_rows",
            "design_post_round_risk_rows",
            "design_post_round_risk_rate",
            "design_changed_bid_rows",
            "design_changed_ask_rows",
            "counterfactual_note",
        ],
    )

    row_counts = {
        "decision_rows": len(decisions),
        "submit_label_rows": len(execution_rows),
        "join_stale_decision_rows": sum(1 for row in decisions if row.join_stale),
        "join_gap_crossed_decision_rows": sum(1 for row in decisions if row.join_gap_crossed),
        "join_missing_decision_rows": sum(1 for row in decisions if row.join_missing),
        "bookticker_anchor_available_rows": sum(1 for row in decisions if _valid_bbo(row.bookticker_bid_tick, row.bookticker_ask_tick)),
        "top5_anchor_available_rows": sum(1 for row in decisions if _valid_bbo(row.top5_bid_tick, row.top5_ask_tick)),
    }
    write_summary(
        output_dir / "quote_anchor_diagnostic_summary.md",
        row_counts=row_counts,
        bbo_rows=bbo_rows,
        enforcement_rows=enforcement_rows,
        counterfactual_rows=counterfactual_rows,
    )

    manifest = {
        "task_id": TASK_ID,
        "mode": "read_only_diagnostic",
        "generated_at": _generated_at(),
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "tick_size": tick,
        "latency_guard_ms": latency_guard_ms,
        "join_stale_ms": join_stale_ms,
        "row_counts": row_counts,
        "inputs": {
            "audit_csv": str(audit_csv),
            "audit_csv_sha256": _hash_file(audit_csv),
            "joined_decisions_csv": str(joined_decisions_csv),
            "joined_decisions_csv_sha256": _hash_file(joined_decisions_csv),
            "top5_sidecar_csv": str(top5_sidecar_csv),
            "top5_sidecar_csv_sha256": _hash_file(top5_sidecar_csv),
            "execution_labels_csv": str(execution_labels_csv),
            "execution_labels_csv_sha256": _hash_file(execution_labels_csv),
        },
        "outputs": [
            "quote_anchor_diagnostic_summary.md",
            "bbo_source_drift.csv",
            "quote_distance_bucket_summary.csv",
            "post_only_reject_throttle_churn_summary.csv",
            "stale_latency_guard_summary.csv",
            "anchor_source_x_fill_markout.csv",
            "current_enforcement_gap_matrix.csv",
            "rounding_clamp_counterfactual.csv",
            "run_manifest.json",
        ],
        "boundary": {
            "strategy_behavior_changed": False,
            "quote_placement_changed": False,
            "live_started": False,
            "replay_experiment_started": False,
            "counterfactual_fill_proof": False,
        },
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path, help="Local live-analysis run directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory for read-only diagnostics")
    parser.add_argument("--tick-size", type=float, default=None, help="Override tick size; defaults to config_live.toml or 0.1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    manifest = run_quote_anchor_diagnostic(
        run_dir=args.run_dir,
        output_dir=args.output_dir,
        tick_size=args.tick_size,
    )
    print(json.dumps({"status": "ok", "output_dir": manifest["output_dir"], "row_counts": manifest["row_counts"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
