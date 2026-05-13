#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - task runner should still report schema gaps.
    np = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / "local_live_analysis"
STAGE6J = LOCAL / "stage6j_cross_sample_0512T005"
OUT = LOCAL / "t008_market_data_view_quality"

SAMPLES = [
    "5-11-night-active",
    "5-10-day-control-1h-06",
    "5-9-noon",
    "5-9-small",
]

FORCED_MARKET_FIELDS = [
    "best_bid",
    "best_ask",
    "mid",
    "bid_size",
    "ask_size",
    "fair",
    "reservation",
    "half_spread",
]
TARGET_FIELDS = ["target_bid_tick", "target_ask_tick"]
TOP5_FIELDS = ["bid_top5_ticks", "bid_top5_qtys", "ask_top5_ticks", "ask_top5_qtys"]
TIMESTAMP_FIELDS = ["ts_local", "ts_exch", "feed_latency_ns", "replay_lag_abs_ns"]
COMPARE_FIELDS = FORCED_MARKET_FIELDS + TARGET_FIELDS + TOP5_FIELDS

FULL_DEPTH_OR_SEQUENCE_FIELDS = [
    "depth_update_U",
    "depth_update_u",
    "depth_update_pu",
    "lastUpdateId",
    "bookTicker_bid",
    "bookTicker_ask",
    "top_n_book_json",
    "depth_update_exchange_time",
    "depth_update_local_receive_ts",
]


@dataclass
class FieldStats:
    count: int = 0
    mismatch: int = 0
    deltas: list[float] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def _is_decision(row: dict[str, str]) -> bool:
    event_type = str(row.get("event_type", "")).strip()
    return event_type in {"", "decision"}


def _parse_int(raw: Any, default: int = 0) -> int:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return default


def _safe_float(raw: Any) -> float | None:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _dist(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    vals = sorted(values)
    return {
        "count": float(len(vals)),
        "mean": float(sum(vals) / len(vals)),
        "p50": float(_quantile(vals, 0.50)),
        "p90": float(_quantile(vals, 0.90)),
        "p99": float(_quantile(vals, 0.99)),
        "min": float(vals[0]),
        "max": float(vals[-1]),
    }


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


def _decision_row_count(path: Path) -> tuple[int, int]:
    total = 0
    duplicates = 0
    seen: set[int] = set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _is_decision(row):
                continue
            ts = _parse_int(row.get("ts_local"))
            if ts <= 0:
                continue
            total += 1
            if ts in seen:
                duplicates += 1
            seen.add(ts)
    return total, duplicates


def _load_live_map(path: Path, fields: list[str]) -> tuple[dict[int, dict[str, str]], int, int]:
    rows: dict[int, dict[str, str]] = {}
    total = 0
    duplicates = 0
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _is_decision(row):
                continue
            ts = _parse_int(row.get("ts_local"))
            if ts <= 0:
                continue
            total += 1
            if ts in rows:
                duplicates += 1
                continue
            rows[ts] = {field_name: str(row.get(field_name, "")) for field_name in fields}
    return rows, total, duplicates


def _field_group(field_name: str) -> str:
    if field_name in FORCED_MARKET_FIELDS:
        return "compressed_market"
    if field_name in TARGET_FIELDS:
        return "target_ticks"
    if field_name in TOP5_FIELDS:
        return "top5_not_overlayed"
    return "other"


def _numeric_field(field_name: str) -> bool:
    return field_name not in {"bid_top5_ticks", "bid_top5_qtys", "ask_top5_ticks", "ask_top5_qtys"}


def _values_match(field_name: str, left: str, right: str) -> tuple[bool, float | None]:
    if _numeric_field(field_name):
        a = _safe_float(left)
        b = _safe_float(right)
        if a is None or b is None:
            return str(left).strip() == str(right).strip(), None
        delta = abs(a - b)
        tolerance = 1e-9 if field_name not in TARGET_FIELDS else 0.0
        return delta <= tolerance, delta
    return str(left).strip() == str(right).strip(), None


def _compare_audit_views(
    *,
    sample: str,
    mode: str,
    live_rows: dict[int, dict[str, str]],
    bt_csv: Path,
    lag_filter_ns: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    field_stats: dict[str, FieldStats] = {field_name: FieldStats() for field_name in COMPARE_FIELDS}
    group_total: dict[str, int] = defaultdict(int)
    group_exact: dict[str, int] = defaultdict(int)
    total_bt_decisions = 0
    common_rows = 0
    lag_filtered_out = 0
    missing_live = 0
    duplicate_bt = 0
    seen_bt: set[int] = set()

    with bt_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not _is_decision(row):
                continue
            ts = _parse_int(row.get("ts_local"))
            if ts <= 0:
                continue
            total_bt_decisions += 1
            if ts in seen_bt:
                duplicate_bt += 1
            seen_bt.add(ts)
            if lag_filter_ns is not None:
                lag = _parse_int(row.get("replay_lag_abs_ns"), default=0)
                if lag > lag_filter_ns:
                    lag_filtered_out += 1
                    continue
            live = live_rows.get(ts)
            if live is None:
                missing_live += 1
                continue
            common_rows += 1
            group_mismatch: dict[str, bool] = defaultdict(bool)
            for field_name in COMPARE_FIELDS:
                stats = field_stats[field_name]
                stats.count += 1
                ok, delta = _values_match(field_name, live.get(field_name, ""), str(row.get(field_name, "")))
                if delta is not None:
                    stats.deltas.append(delta)
                if not ok:
                    stats.mismatch += 1
                    group_mismatch[_field_group(field_name)] = True
                    if len(stats.examples) < 3:
                        stats.examples.append(
                            f"ts={ts} live={live.get(field_name, '')} replay={row.get(field_name, '')}"
                        )
            for group_name in {"compressed_market", "target_ticks", "top5_not_overlayed"}:
                group_fields = [f for f in COMPARE_FIELDS if _field_group(f) == group_name]
                if not group_fields:
                    continue
                group_total[group_name] += 1
                if not any(
                    field_stats[field_name].mismatch > 0
                    and field_stats[field_name].examples
                    and field_stats[field_name].examples[-1].startswith(f"ts={ts} ")
                    for field_name in group_fields
                ):
                    # The example-based condition above avoids storing per-row vectors.
                    # Recompute exactly for this row to keep the row-level group metric correct.
                    exact = True
                    for field_name in group_fields:
                        ok, _ = _values_match(field_name, live.get(field_name, ""), str(row.get(field_name, "")))
                        if not ok:
                            exact = False
                            break
                    if exact:
                        group_exact[group_name] += 1
                else:
                    exact = True
                    for field_name in group_fields:
                        ok, _ = _values_match(field_name, live.get(field_name, ""), str(row.get(field_name, "")))
                        if not ok:
                            exact = False
                            break
                    if exact:
                        group_exact[group_name] += 1

    field_rows: list[dict[str, Any]] = []
    for field_name, stats in field_stats.items():
        dist = _dist(stats.deltas)
        field_rows.append(
            {
                "sample": sample,
                "mode": mode,
                "field_group": _field_group(field_name),
                "field": field_name,
                "common_decision_rows": stats.count,
                "mismatch_rows": stats.mismatch,
                "mismatch_rate": stats.mismatch / stats.count if stats.count else 0.0,
                "delta_mean": dist["mean"],
                "delta_p50": dist["p50"],
                "delta_p90": dist["p90"],
                "delta_p99": dist["p99"],
                "delta_max": dist["max"],
                "examples": " || ".join(stats.examples),
            }
        )

    group_rows: list[dict[str, Any]] = []
    for group_name in ["compressed_market", "target_ticks", "top5_not_overlayed"]:
        total = group_total[group_name]
        exact = group_exact[group_name]
        group_rows.append(
            {
                "sample": sample,
                "mode": mode,
                "field_group": group_name,
                "common_decision_rows": total,
                "exact_rows": exact,
                "exact_rate": exact / total if total else 0.0,
                "mismatch_rows": total - exact,
                "mismatch_rate": (total - exact) / total if total else 0.0,
            }
        )

    meta = {
        "sample": sample,
        "mode": mode,
        "bt_decision_rows": total_bt_decisions,
        "common_decision_rows": common_rows,
        "missing_live_rows": missing_live,
        "duplicate_bt_ts": duplicate_bt,
        "lag_filter_ns": lag_filter_ns if lag_filter_ns is not None else "",
        "lag_filtered_out": lag_filtered_out,
    }
    return field_rows, group_rows, meta


def _audit_schema_rows(sample: str, source: str, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return [
            {
                "sample": sample,
                "source": source,
                "path": str(path),
                "field": "__file_exists__",
                "present": 0,
            }
        ]
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
    required = (
        FORCED_MARKET_FIELDS
        + TARGET_FIELDS
        + TOP5_FIELDS
        + ["ts_local", "ts_exch", "feed_latency_ns", "bt_feed_ts_local", "bt_feed_ts_exch"]
        + FULL_DEPTH_OR_SEQUENCE_FIELDS
    )
    return [
        {
            "sample": sample,
            "source": source,
            "path": str(path),
            "field": field_name,
            "present": int(field_name in fields),
        }
        for field_name in required
    ]


def _npz_schema_row(sample: str, npz_path: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample": sample,
        "source": "converted_npz",
        "path": str(npz_path),
        "exists": int(npz_path.exists()),
        "dtype_names": "",
        "row_count": 0,
        "has_update_id_fields": 0,
    }
    if not npz_path.exists() or np is None:
        return row
    data = np.load(npz_path)
    key = data.files[0] if data.files else ""
    arr = data[key] if key else []
    dtype_names = list(getattr(arr, "dtype", None).names or [])
    row["dtype_names"] = "|".join(dtype_names)
    row["row_count"] = int(len(arr))
    row["has_update_id_fields"] = int(any(name in dtype_names for name in ["U", "u", "pu", "lastUpdateId"]))
    return row


def _find_raw_gzip(sample: str) -> Path:
    matches = sorted((LOCAL / sample / "raw_market_data").glob("*.gz"))
    return matches[0] if matches else LOCAL / sample / "raw_market_data" / "__missing__.gz"


def _find_npz(sample: str) -> Path:
    matches = sorted((LOCAL / sample / "out" / "live_raw").glob("*/*.npz"))
    return matches[0] if matches else LOCAL / sample / "out" / "live_raw" / "__missing__.npz"


def _raw_depth_sequence(sample: str, raw_gz: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample": sample,
        "path": str(raw_gz),
        "exists": int(raw_gz.exists()),
        "raw_lines": 0,
        "depth_update_events": 0,
        "snapshot_events": 0,
        "bookticker_events": 0,
        "depth_streams": "",
        "bookticker_streams": "",
        "first_snapshot_last_update_id": "",
        "first_depth_U": "",
        "first_depth_u": "",
        "last_depth_u": "",
        "pu_mismatch_count": 0,
        "non_monotonic_u_count": 0,
        "missing_sequence_fields_count": 0,
        "max_local_latency_ms": 0.0,
        "latency_ms_p50": 0.0,
        "latency_ms_p90": 0.0,
        "latency_ms_p99": 0.0,
        "gzip_eof_error": 0,
    }
    if not raw_gz.exists():
        return row

    depth_streams: set[str] = set()
    bookticker_streams: set[str] = set()
    prev_u: int | None = None
    latencies_ms: list[float] = []

    f = None
    try:
        f = gzip.open(raw_gz, "rt")
        while True:
            try:
                line = next(f)
            except StopIteration:
                break
            except EOFError:
                row["gzip_eof_error"] = 1
                break
            line = line.strip()
            if not line:
                continue
            row["raw_lines"] += 1
            try:
                ts_text, payload_text = line.split(" ", 1)
                local_ts = int(ts_text)
                payload = json.loads(payload_text)
            except Exception:
                continue
            stream = str(payload.get("stream", ""))
            data = payload.get("data", payload)
            if not isinstance(data, dict):
                continue
            event_type = str(data.get("e", ""))
            if "bookTicker" in stream or event_type == "bookTicker":
                row["bookticker_events"] += 1
                if stream:
                    bookticker_streams.add(stream)
                event_ms = _parse_int(data.get("E"), default=0)
                if event_ms > 0:
                    latencies_ms.append((local_ts - event_ms * 1_000_000) / 1_000_000.0)
                continue
            if "lastUpdateId" in data:
                row["snapshot_events"] += 1
                if not row["first_snapshot_last_update_id"]:
                    row["first_snapshot_last_update_id"] = str(data.get("lastUpdateId", ""))
                event_ms = _parse_int(data.get("E") or data.get("T"), default=0)
                if event_ms > 0:
                    latencies_ms.append((local_ts - event_ms * 1_000_000) / 1_000_000.0)
                continue
            if "depth" not in stream and event_type != "depthUpdate":
                continue
            row["depth_update_events"] += 1
            if stream:
                depth_streams.add(stream)
            U = _parse_int(data.get("U"), default=0)
            u = _parse_int(data.get("u"), default=0)
            pu = _parse_int(data.get("pu"), default=0)
            if U <= 0 or u <= 0:
                row["missing_sequence_fields_count"] += 1
            else:
                if not row["first_depth_U"]:
                    row["first_depth_U"] = str(U)
                    row["first_depth_u"] = str(u)
                if prev_u is not None:
                    if pu > 0 and pu != prev_u:
                        row["pu_mismatch_count"] += 1
                    if u <= prev_u:
                        row["non_monotonic_u_count"] += 1
                prev_u = u
                row["last_depth_u"] = str(u)
            event_ms = _parse_int(data.get("E") or data.get("T"), default=0)
            if event_ms > 0:
                latencies_ms.append((local_ts - event_ms * 1_000_000) / 1_000_000.0)
    finally:
        if f is not None:
            try:
                f.close()
            except Exception:
                pass

    dist = _dist(latencies_ms)
    row["depth_streams"] = "|".join(sorted(depth_streams))
    row["bookticker_streams"] = "|".join(sorted(bookticker_streams))
    row["max_local_latency_ms"] = dist["max"]
    row["latency_ms_p50"] = dist["p50"]
    row["latency_ms_p90"] = dist["p90"]
    row["latency_ms_p99"] = dist["p99"]
    return row


def _top_mismatches(view_rows: list[dict[str, Any]], sample: str, field: str, mode: str) -> dict[str, Any]:
    matching = [r for r in view_rows if r["sample"] == sample and r["mode"] == mode and r["field"] == field]
    if not matching:
        return {"mismatch_rate": 0.0, "delta_p50": 0.0, "delta_p99": 0.0, "delta_max": 0.0}
    row = matching[0]
    return {
        "mismatch_rate": float(row.get("mismatch_rate", 0.0)),
        "delta_p50": float(row.get("delta_p50", 0.0)),
        "delta_p99": float(row.get("delta_p99", 0.0)),
        "delta_max": float(row.get("delta_max", 0.0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sample_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    npz_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    compare_meta_rows: list[dict[str, Any]] = []

    for sample in SAMPLES:
        run_dir = LOCAL / sample
        live_csv = run_dir / f"audit_live_{sample}.csv"
        overlay_csv = run_dir / "out" / "backtest_audit_replay" / "audit_bt_audit_replay.csv"
        overlay_result = _read_json(run_dir / "backtest_audit_replay_result.json")
        stage6j_csv = STAGE6J / sample / "01_baseline_inflight_only" / "audit_bt_stage6j.csv"
        stage6j_result = _read_json(STAGE6J / sample / "01_baseline_inflight_only" / "summary.json")
        raw_gz = _find_raw_gzip(sample)
        npz_path = _find_npz(sample)

        live_map, live_count, live_dups = _load_live_map(live_csv, COMPARE_FIELDS)
        overlay_count, overlay_dups = _decision_row_count(overlay_csv) if overlay_csv.exists() else (0, 0)
        stage6j_count, stage6j_dups = _decision_row_count(stage6j_csv) if stage6j_csv.exists() else (0, 0)

        for source, path in [
            ("live_audit", live_csv),
            ("audit_replay_overlay", overlay_csv),
            ("stage6j_no_overlay_baseline", stage6j_csv),
        ]:
            schema_rows.extend(_audit_schema_rows(sample, source, path))

        npz_rows.append(_npz_schema_row(sample, npz_path))
        raw_rows.append(_raw_depth_sequence(sample, raw_gz))

        if overlay_csv.exists():
            fields, groups, meta = _compare_audit_views(
                sample=sample,
                mode="audit_replay_overlay_all",
                live_rows=live_map,
                bt_csv=overlay_csv,
                lag_filter_ns=None,
            )
            view_rows.extend(fields)
            group_rows.extend(groups)
            compare_meta_rows.append(meta)
        if stage6j_csv.exists():
            for mode, lag_filter in [
                ("stage6j_no_overlay_all", None),
                ("stage6j_no_overlay_lag_le_250ms", 250_000_000),
            ]:
                fields, groups, meta = _compare_audit_views(
                    sample=sample,
                    mode=mode,
                    live_rows=live_map,
                    bt_csv=stage6j_csv,
                    lag_filter_ns=lag_filter,
                )
                view_rows.extend(fields)
                group_rows.extend(groups)
                compare_meta_rows.append(meta)

        bb = _top_mismatches(view_rows, sample, "best_bid", "stage6j_no_overlay_lag_le_250ms")
        ba = _top_mismatches(view_rows, sample, "best_ask", "stage6j_no_overlay_lag_le_250ms")
        tbt = _top_mismatches(view_rows, sample, "target_bid_tick", "stage6j_no_overlay_lag_le_250ms")
        tat = _top_mismatches(view_rows, sample, "target_ask_tick", "stage6j_no_overlay_lag_le_250ms")
        sample_rows.append(
            {
                "sample": sample,
                "live_decision_rows": live_count,
                "live_duplicate_ts": live_dups,
                "audit_replay_decision_rows": overlay_count,
                "audit_replay_duplicate_ts": overlay_dups,
                "audit_replay_market_state_overlay_mode": overlay_result.get(
                    "audit_replay_market_state_overlay_mode", ""
                ),
                "audit_replay_market_state_overlay_count": overlay_result.get(
                    "audit_replay_market_state_overlay_count", ""
                ),
                "audit_replay_market_state_loaded_count": overlay_result.get(
                    "audit_replay_market_state_loaded_count", ""
                ),
                "stage6j_decision_rows": stage6j_count,
                "stage6j_duplicate_ts": stage6j_dups,
                "stage6j_market_state_overlay_mode": stage6j_result.get(
                    "audit_replay_market_state_overlay_mode", ""
                ),
                "stage6j_market_state_overlay_count": stage6j_result.get(
                    "audit_replay_market_state_overlay_count", ""
                ),
                "stage6j_market_state_loaded_count": stage6j_result.get(
                    "audit_replay_market_state_loaded_count", ""
                ),
                "stage6j_best_bid_mismatch_rate_lag_le_250ms": bb["mismatch_rate"],
                "stage6j_best_bid_delta_p99_lag_le_250ms": bb["delta_p99"],
                "stage6j_best_ask_mismatch_rate_lag_le_250ms": ba["mismatch_rate"],
                "stage6j_best_ask_delta_p99_lag_le_250ms": ba["delta_p99"],
                "stage6j_target_bid_tick_mismatch_rate_lag_le_250ms": tbt["mismatch_rate"],
                "stage6j_target_ask_tick_mismatch_rate_lag_le_250ms": tat["mismatch_rate"],
            }
        )

    _write_csv(OUT / "t008_sample_summary.csv", sample_rows)
    _write_csv(OUT / "t008_audit_schema_coverage.csv", schema_rows)
    _write_csv(OUT / "t008_converted_npz_schema.csv", npz_rows)
    _write_csv(OUT / "t008_raw_depth_sequence_quality.csv", raw_rows)
    _write_csv(OUT / "t008_view_field_comparison.csv", view_rows)
    _write_csv(OUT / "t008_view_group_comparison.csv", group_rows)
    _write_csv(OUT / "t008_view_compare_meta.csv", compare_meta_rows)

    summary_metrics = {
        "samples": SAMPLES,
        "sample_summary": sample_rows,
        "raw_depth_sequence": raw_rows,
        "converted_npz_schema": npz_rows,
        "view_group_comparison": group_rows,
    }
    (OUT / "t008_summary_metrics.json").write_text(json.dumps(summary_metrics, indent=2, sort_keys=True))

    main_sample = next((row for row in sample_rows if row["sample"] == "5-11-night-active"), sample_rows[0])
    main_raw = next((row for row in raw_rows if row["sample"] == "5-11-night-active"), raw_rows[0])
    overlay_groups = [
        row
        for row in group_rows
        if row["sample"] == "5-11-night-active" and row["mode"] == "audit_replay_overlay_all"
    ]
    stage6j_groups = [
        row
        for row in group_rows
        if row["sample"] == "5-11-night-active" and row["mode"] == "stage6j_no_overlay_lag_le_250ms"
    ]

    def group_rate(rows: list[dict[str, Any]], group_name: str) -> float:
        for row in rows:
            if row["field_group"] == group_name:
                return float(row["mismatch_rate"])
        return 0.0

    summary_lines = [
        "# T008 Market Data View Quality Summary",
        "",
        "## Scope",
        "",
        "- Read-only analysis over existing local live samples, audit replay outputs, Stage 6J baseline no-overlay audits, raw Binance gzip files, and converted npz files.",
        "- No strategy code changes, no new replay, no live run.",
        "",
        "## Core Conclusions",
        "",
        "- `strategy_core.decide_actions()` does not receive full depth directly; live/backtest outer loops compress `hbt.depth(0)` into best bid/ask, mid, top5 size strings, fair/reservation/half_spread, and target ticks before calling the shared action core.",
        "- Both live and backtest call `hbt.depth(0)`, but shared API is not shared market state: live reads the connector-maintained book at decision time, while Stage 6J no-overlay reads the replay-reconstructed book.",
        "- Audit replay with `market_state_overlay=audit` forces only compressed market/fair/target fields from live audit. Top5 tick/qty strings are still generated from replay depth and are not overlayed.",
        "- Current audit fields are enough for the existing simple strategy's compressed action-path alignment, but not enough to prove full L2 / queue / OFI / microprice equivalence.",
        "- Current converted npz files do not retain Binance update ids (`U/u/pu` or `lastUpdateId`). Raw gzip has them, but later backtest/audit artifacts do not expose them at decision rows.",
        "- Recommendation: create T009 before microprice / OFI / queue strategy work to extend the data layer/audit schema with top-N book, update ids, exchange timestamps, local receipt timestamps, and bookTicker-vs-depth consistency fields.",
        "",
        "## Main Sample Metrics",
        "",
        f"- Main sample: `{main_sample['sample']}`.",
        f"- Live decision rows: `{main_sample['live_decision_rows']}`.",
        f"- Audit replay market overlay: mode `{main_sample['audit_replay_market_state_overlay_mode']}`, overlay count `{main_sample['audit_replay_market_state_overlay_count']}`.",
        f"- Stage 6J baseline market overlay: mode `{main_sample['stage6j_market_state_overlay_mode']}`, overlay count `{main_sample['stage6j_market_state_overlay_count']}`.",
        f"- Stage 6J no-overlay best bid mismatch rate after lag filter: `{float(main_sample['stage6j_best_bid_mismatch_rate_lag_le_250ms']):.6f}`.",
        f"- Stage 6J no-overlay best ask mismatch rate after lag filter: `{float(main_sample['stage6j_best_ask_mismatch_rate_lag_le_250ms']):.6f}`.",
        f"- Stage 6J no-overlay target bid tick mismatch rate after lag filter: `{float(main_sample['stage6j_target_bid_tick_mismatch_rate_lag_le_250ms']):.6f}`.",
        f"- Stage 6J no-overlay target ask tick mismatch rate after lag filter: `{float(main_sample['stage6j_target_ask_tick_mismatch_rate_lag_le_250ms']):.6f}`.",
        f"- Raw depth events: `{main_raw['depth_update_events']}`, snapshots `{main_raw['snapshot_events']}`, bookTicker events `{main_raw['bookticker_events']}`, `pu` mismatches `{main_raw['pu_mismatch_count']}`.",
        "",
        "## View Comparison Rates For Main Sample",
        "",
        "| mode | compressed_market mismatch | target_ticks mismatch | top5 mismatch |",
        "|---|---:|---:|---:|",
        f"| audit_replay_overlay_all | {group_rate(overlay_groups, 'compressed_market'):.6f} | {group_rate(overlay_groups, 'target_ticks'):.6f} | {group_rate(overlay_groups, 'top5_not_overlayed'):.6f} |",
        f"| stage6j_no_overlay_lag_le_250ms | {group_rate(stage6j_groups, 'compressed_market'):.6f} | {group_rate(stage6j_groups, 'target_ticks'):.6f} | {group_rate(stage6j_groups, 'top5_not_overlayed'):.6f} |",
        "",
        "## Output Files",
        "",
        "- `t008_sample_summary.csv`",
        "- `t008_audit_schema_coverage.csv`",
        "- `t008_converted_npz_schema.csv`",
        "- `t008_raw_depth_sequence_quality.csv`",
        "- `t008_view_field_comparison.csv`",
        "- `t008_view_group_comparison.csv`",
        "- `t008_view_compare_meta.csv`",
        "- `t008_summary_metrics.json`",
    ]
    (OUT / "T008_MARKET_DATA_VIEW_SUMMARY.md").write_text("\n".join(summary_lines) + "\n")

    print(json.dumps({"out_dir": str(OUT), "samples": len(SAMPLES)}, indent=2))


if __name__ == "__main__":
    main()
