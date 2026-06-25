#!/usr/bin/env python3
"""Validate synchronized public samples for the cross-exchange MVP."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0625T002"
SCHEMA_VERSION = "cross_exchange_sample_expansion_v1"
DEFAULT_ANALYSIS_ROOT = PROJECT_ROOT / "local_live_analysis"
DEFAULT_OUTPUT_DIR = (
    DEFAULT_ANALYSIS_ROOT / "cross_exchange_mvp_sample_expansion_0625T002"
)
DEFAULT_SAMPLE_IDS = [
    "xemm_0625_t002_utc15_a",
    "xemm_0625_t002_utc15_b",
    "xemm_0625_t002_utc16_c",
]
CONTEXT_HORIZON_MS = 1000
TICK_SIZE = 0.1
BOUNDARY_FLAGS = {
    "offline_local_processing_only": True,
    "public_market_data_only": True,
    "no_credentials": True,
    "no_private_account_order_cancel_endpoints": True,
    "no_live_client_initialization": True,
    "no_live_orders": True,
    "no_strategy_or_watcher_change": True,
    "no_signal_threshold_tuning": True,
    "no_side_mapping_freeze": True,
    "both_hyperliquid_touch_alternatives_preserved": True,
    "future_labels_not_decision_inputs": True,
    "no_canary_or_promotion_authorization": True,
}
CONTEXT_FIELDS = [
    "sample_id",
    "observed_regime",
    "source_row_index",
    "future_row_index",
    "hyperliquid_decision_ts",
    "hyperliquid_l2book_local_ts",
    "hyperliquid_l2book_event_ts",
    "binance_local_ts",
    "binance_exch_ts",
    "binance_source_age_ms",
    "hyperliquid_join_age_ms",
    "nominal_horizon_ms",
    "effective_future_age_ms",
    "future_hyperliquid_decision_ts",
    "hyperliquid_current_bid_px",
    "hyperliquid_current_ask_px",
    "hyperliquid_buy_touch_quote_px",
    "hyperliquid_sell_touch_quote_px",
    "tick_size",
    "hyperliquid_mid_px",
    "hyperliquid_top5_microprice_px",
    "hyperliquid_spread_ticks",
    "hyperliquid_bid_top5_px",
    "hyperliquid_ask_top5_px",
    "hyperliquid_bid_top5_qtys",
    "hyperliquid_ask_top5_qtys",
    "binance_mid_px",
    "binance_top5_microprice_px",
    "binance_bid_top5_px",
    "binance_ask_top5_px",
    "binance_bid_top5_qtys",
    "binance_ask_top5_qtys",
    "input_binance_top5_imbalance",
    "input_binance_microprice_minus_mid_ticks",
    "input_binance_mid_move_ticks_from_prev",
    "input_binance_top5_bid_qty",
    "basis_mid_ticks",
    "hyperliquid_top5_imbalance",
    "hyperliquid_microprice_minus_mid_ticks",
    "hyperliquid_context_quality",
    "future_hyperliquid_mid_px",
    "future_hyperliquid_top5_microprice_px",
    "hyperliquid_future_mid_move_ticks",
    "hyperliquid_future_microprice_minus_mid_change_ticks",
    "label_row_quality",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _fmt(value: float | None, places: int = 8) -> str:
    if value is None:
        return ""
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _primary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("joined_row_quality") == "primary_usable"
        and row.get("cross_exchange_future_join") in {"0", "false", "False", ""}
        and row.get("cross_exchange_missing_binance_join") in {"0", "false", "False", ""}
    ]


def _sample_paths(root: Path, sample_id: str) -> dict[str, Path]:
    public_dir = root / f"cross_exchange_public_sample_{sample_id}"
    join_dir = root / f"cross_exchange_lead_lag_join_{sample_id}"
    pricing_dir = root / f"binance_led_hyperliquid_pricing_signal_{sample_id}"
    return {
        "public_dir": public_dir,
        "sample_manifest": public_dir / "sample_manifest.json",
        "binance_collection": public_dir / "binance_public_raw" / "collection_manifest.json",
        "hyperliquid_collection": public_dir
        / "hyperliquid_public_sample"
        / "collection_manifest.json",
        "binance_raw": public_dir / "binance_public_raw" / "raw.gz",
        "hyperliquid_raw": public_dir / "hyperliquid_public_sample" / "raw.gz",
        "binance_alignment": public_dir / "binance_alignment" / "metrics.json",
        "hyperliquid_alignment": public_dir
        / "hyperliquid_public_sample"
        / "alignment"
        / "metrics.json",
        "join_features": join_dir / "cross_exchange_joined_features.csv",
        "join_quality": join_dir / "join_quality_summary.json",
        "pricing_rows": pricing_dir / "pricing_signal_rows.csv",
        "pricing_manifest": pricing_dir / "run_manifest.json",
    }


def _load_decision_time_sample(root: Path, sample_id: str) -> dict[str, Any]:
    paths = _sample_paths(root, sample_id)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{sample_id} missing required artifacts: {missing}")
    sample_manifest = _read_json(paths["sample_manifest"])
    binance_collection = _read_json(paths["binance_collection"])
    hyperliquid_collection = _read_json(paths["hyperliquid_collection"])
    binance_alignment = _read_json(paths["binance_alignment"])
    hyperliquid_alignment = _read_json(paths["hyperliquid_alignment"])
    join_quality = _read_json(paths["join_quality"])
    joined_rows = _read_csv(paths["join_features"])
    primary_rows = _primary_rows(joined_rows)
    rv_values = [
        value
        for value in (_float(row.get("binance_rolling_rv_ticks_20")) for row in primary_rows)
        if value is not None
    ]
    hl_spreads = [
        value
        for value in (_float(row.get("hyperliquid_spread_ticks")) for row in primary_rows)
        if value is not None
    ]
    hl_depth = [
        bid + ask
        for row in primary_rows
        if (bid := _float(row.get("hyperliquid_top5_bid_qty"))) is not None
        and (ask := _float(row.get("hyperliquid_top5_ask_qty"))) is not None
    ]
    duration = min(
        float(binance_collection["actual_duration_seconds"]),
        float(hyperliquid_collection["actual_duration_seconds"]),
    )
    binance_trade_count = int(
        binance_collection.get("message_count_by_event_type", {}).get("trade", 0)
    )
    hl_trade_count = int(hyperliquid_alignment.get("trade_event_count", 0))
    return {
        "sample_id": sample_id,
        "paths": paths,
        "sample_manifest": sample_manifest,
        "binance_collection": binance_collection,
        "hyperliquid_collection": hyperliquid_collection,
        "binance_alignment": binance_alignment,
        "hyperliquid_alignment": hyperliquid_alignment,
        "join_quality": join_quality,
        "joined_rows": joined_rows,
        "primary_rows": primary_rows,
        "regime_inputs": {
            "binance_rolling_rv_ticks_20_mean": (
                statistics.fmean(rv_values) if rv_values else 0.0
            ),
            "combined_trade_events_per_second": (
                (binance_trade_count + hl_trade_count) / duration if duration else 0.0
            ),
            "hyperliquid_spread_ticks_median": (
                statistics.median(hl_spreads) if hl_spreads else 0.0
            ),
            "hyperliquid_top5_total_qty_median": (
                statistics.median(hl_depth) if hl_depth else 0.0
            ),
        },
    }


def _assign_regimes(samples: list[dict[str, Any]]) -> None:
    """Assign relative regimes before any future-label file is read."""
    ordered = sorted(
        samples,
        key=lambda sample: (
            sample["regime_inputs"]["binance_rolling_rv_ticks_20_mean"],
            sample["regime_inputs"]["combined_trade_events_per_second"],
            -sample["regime_inputs"]["hyperliquid_top5_total_qty_median"],
            sample["sample_id"],
        ),
    )
    labels = [
        "low_activity_liquidity",
        "normal_activity_liquidity",
        "high_activity_liquidity",
    ]
    for index, sample in enumerate(ordered):
        bucket_index = min(len(labels) - 1, (index * len(labels)) // len(ordered))
        sample["observed_regime"] = labels[bucket_index]
        sample["regime_rank"] = index + 1


def _build_context_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    pricing_rows = _read_csv(sample["paths"]["pricing_rows"])
    primary = sample["primary_rows"]
    rows: list[dict[str, Any]] = []
    for pricing in pricing_rows:
        if pricing.get("horizon_ms") != str(CONTEXT_HORIZON_MS):
            continue
        source_index = int(pricing["source_row_index"])
        future_index = int(pricing["future_row_index"])
        if source_index >= len(primary) or future_index >= len(primary):
            continue
        source = primary[source_index]
        future = primary[future_index]
        row = {
            "sample_id": sample["sample_id"],
            "observed_regime": sample["observed_regime"],
            "source_row_index": source_index,
            "future_row_index": future_index,
            "hyperliquid_decision_ts": source.get("hyperliquid_decision_ts", ""),
            "hyperliquid_l2book_local_ts": source.get("hyperliquid_l2book_local_ts", ""),
            "hyperliquid_l2book_event_ts": source.get("hyperliquid_l2book_event_ts", ""),
            "binance_local_ts": source.get("binance_local_ts", ""),
            "binance_exch_ts": source.get("binance_exch_ts", ""),
            "binance_source_age_ms": source.get("binance_source_age_ms", ""),
            "hyperliquid_join_age_ms": source.get("hyperliquid_join_age_ms", ""),
            "nominal_horizon_ms": CONTEXT_HORIZON_MS,
            "effective_future_age_ms": pricing.get("effective_future_age_ms", ""),
            "future_hyperliquid_decision_ts": pricing.get(
                "future_hyperliquid_decision_ts", ""
            ),
            "hyperliquid_current_bid_px": source.get("hyperliquid_best_bid_px", ""),
            "hyperliquid_current_ask_px": source.get("hyperliquid_best_ask_px", ""),
            "hyperliquid_buy_touch_quote_px": source.get("hyperliquid_best_bid_px", ""),
            "hyperliquid_sell_touch_quote_px": source.get("hyperliquid_best_ask_px", ""),
            "tick_size": TICK_SIZE,
            "hyperliquid_mid_px": source.get("hyperliquid_mid_px", ""),
            "hyperliquid_top5_microprice_px": source.get(
                "hyperliquid_top5_microprice_px", ""
            ),
            "hyperliquid_spread_ticks": source.get("hyperliquid_spread_ticks", ""),
            "hyperliquid_bid_top5_px": source.get("hyperliquid_bid_topn_px", ""),
            "hyperliquid_ask_top5_px": source.get("hyperliquid_ask_topn_px", ""),
            "hyperliquid_bid_top5_qtys": source.get("hyperliquid_bid_topn_qtys", ""),
            "hyperliquid_ask_top5_qtys": source.get("hyperliquid_ask_topn_qtys", ""),
            "binance_mid_px": source.get("binance_mid_px", ""),
            "binance_top5_microprice_px": source.get("binance_top5_microprice_px", ""),
            "binance_bid_top5_px": source.get("binance_bid_top5_px", ""),
            "binance_ask_top5_px": source.get("binance_ask_top5_px", ""),
            "binance_bid_top5_qtys": source.get("binance_bid_top5_qtys", ""),
            "binance_ask_top5_qtys": source.get("binance_ask_top5_qtys", ""),
            "input_binance_top5_imbalance": pricing.get(
                "input_binance_top5_imbalance", ""
            ),
            "input_binance_microprice_minus_mid_ticks": pricing.get(
                "input_binance_microprice_minus_mid_ticks", ""
            ),
            "input_binance_mid_move_ticks_from_prev": pricing.get(
                "input_binance_mid_move_ticks_from_prev", ""
            ),
            "input_binance_top5_bid_qty": pricing.get(
                "input_binance_top5_bid_qty", ""
            ),
            "basis_mid_ticks": source.get("basis_mid_ticks", ""),
            "hyperliquid_top5_imbalance": source.get("hyperliquid_top5_imbalance", ""),
            "hyperliquid_microprice_minus_mid_ticks": source.get(
                "hyperliquid_microprice_minus_mid_ticks", ""
            ),
            "hyperliquid_context_quality": source.get(
                "hyperliquid_context_quality", ""
            ),
            "future_hyperliquid_mid_px": future.get("hyperliquid_mid_px", ""),
            "future_hyperliquid_top5_microprice_px": future.get(
                "hyperliquid_top5_microprice_px", ""
            ),
            "hyperliquid_future_mid_move_ticks": pricing.get(
                "hyperliquid_future_mid_move_ticks", ""
            ),
            "hyperliquid_future_microprice_minus_mid_change_ticks": pricing.get(
                "hyperliquid_future_microprice_minus_mid_change_ticks", ""
            ),
            "label_row_quality": pricing.get("label_row_quality", ""),
        }
        row["complete_context"] = all(str(row.get(field, "")).strip() for field in CONTEXT_FIELDS)
        rows.append(row)
    return rows


def _sample_quality_row(sample: dict[str, Any], context_rows: list[dict[str, Any]]) -> dict[str, Any]:
    binance = sample["binance_collection"]
    hyperliquid = sample["hyperliquid_collection"]
    sample_manifest = sample["sample_manifest"]
    join = sample["join_quality"]["cross_exchange_join"]
    binance_sha = _sha256(sample["paths"]["binance_raw"])
    hyperliquid_sha = _sha256(sample["paths"]["hyperliquid_raw"])
    source_ages = [
        value
        for value in (_float(row.get("binance_source_age_ms")) for row in sample["joined_rows"])
        if value is not None
    ]
    complete_count = sum(bool(row["complete_context"]) for row in context_rows)
    return {
        "sample_id": sample["sample_id"],
        "requested_duration_seconds": sample_manifest["requested_duration_seconds"],
        "actual_duration_seconds": _fmt(
            min(
                float(binance["actual_duration_seconds"]),
                float(hyperliquid["actual_duration_seconds"]),
            ),
            6,
        ),
        "synchronized_overlap_seconds": _fmt(
            float(sample_manifest["overlap"]["overlap_seconds"]), 6
        ),
        "start_time_utc": binance["local_start_time"],
        "observed_regime": sample["observed_regime"],
        "host": "awsserver1",
        "interpreter": "/home/admin/hft_live/venv/bin/python",
        "collection_commit": sample_manifest.get("git_commit", ""),
        "binance_bookticker_count": binance["message_count_by_event_type"].get(
            "bookTicker", 0
        ),
        "binance_depth_count": binance["message_count_by_event_type"].get(
            "depthUpdate", 0
        ),
        "binance_trade_count": binance["message_count_by_event_type"].get("trade", 0),
        "hyperliquid_l2book_count": hyperliquid["message_count_by_channel"].get(
            "l2Book", 0
        ),
        "hyperliquid_trade_message_count": hyperliquid[
            "message_count_by_channel"
        ].get("trades", 0),
        "hyperliquid_trade_event_count": sample["hyperliquid_alignment"].get(
            "trade_event_count", 0
        ),
        "binance_reconnect_count": binance["reconnect_count"],
        "hyperliquid_reconnect_count": hyperliquid["reconnect_count"],
        "binance_checksum_match": binance_sha == binance["raw_sha256"],
        "hyperliquid_checksum_match": hyperliquid_sha == hyperliquid["raw_sha256"],
        "binance_top5_rows": sample["binance_alignment"].get("top5_row_count", 0),
        "hyperliquid_top5_rows": sample["hyperliquid_alignment"].get(
            "topn_row_count", 0
        ),
        "hyperliquid_decision_rows": sample["hyperliquid_alignment"].get(
            "decision_row_count", 0
        ),
        "join_rows": len(sample["joined_rows"]),
        "primary_rows": len(sample["primary_rows"]),
        "excluded_rows": len(sample["joined_rows"]) - len(sample["primary_rows"]),
        "future_join_count": join["future_join_count"],
        "missing_join_count": join["missing_binance_join_count"],
        "stale_join_count": join["stale_binance_source_count"],
        "source_age_ms_p50": _fmt(_percentile(source_ages, 0.50), 6),
        "source_age_ms_p90": _fmt(_percentile(source_ages, 0.90), 6),
        "source_age_ms_p99": _fmt(_percentile(source_ages, 0.99), 6),
        "context_rows_1000ms": len(context_rows),
        "complete_symmetric_context_rows_1000ms": complete_count,
        "sample_valid": (
            float(sample_manifest["overlap"]["overlap_seconds"]) >= 1500
            and binance_sha == binance["raw_sha256"]
            and hyperliquid_sha == hyperliquid["raw_sha256"]
            and join["future_join_count"] == 0
            and complete_count >= 20
        ),
    }


def _field_coverage_rows(
    sample_id: str, context_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for field in CONTEXT_FIELDS:
        non_empty = sum(bool(str(row.get(field, "")).strip()) for row in context_rows)
        rows.append(
            {
                "sample_id": sample_id,
                "field": field,
                "required_for_complete_context": True,
                "row_count": len(context_rows),
                "non_empty_count": non_empty,
                "coverage_rate": _fmt(non_empty / len(context_rows) if context_rows else 0),
            }
        )
    return rows


def _effective_horizon_rows(sample: dict[str, Any]) -> list[dict[str, Any]]:
    pricing_rows = _read_csv(sample["paths"]["pricing_rows"])
    by_horizon: dict[int, list[float]] = {}
    for row in pricing_rows:
        horizon = int(row["horizon_ms"])
        age = _float(row.get("effective_future_age_ms"))
        if age is not None:
            by_horizon.setdefault(horizon, []).append(age)
    return [
        {
            "sample_id": sample["sample_id"],
            "horizon_ms": horizon,
            "label_row_count": len(ages),
            "effective_future_age_ms_min": _fmt(min(ages), 6),
            "effective_future_age_ms_p50": _fmt(_percentile(ages, 0.50), 6),
            "effective_future_age_ms_mean": _fmt(statistics.fmean(ages), 6),
            "effective_future_age_ms_p90": _fmt(_percentile(ages, 0.90), 6),
            "effective_future_age_ms_max": _fmt(max(ages), 6),
        }
        for horizon, ages in sorted(by_horizon.items())
    ]


def build_artifacts(
    *,
    analysis_root: Path,
    sample_ids: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    samples = [_load_decision_time_sample(analysis_root, sample_id) for sample_id in sample_ids]
    _assign_regimes(samples)

    all_context_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for sample in samples:
        context_rows = _build_context_rows(sample)
        all_context_rows.extend(context_rows)
        quality_rows.append(_sample_quality_row(sample, context_rows))
        field_rows.extend(_field_coverage_rows(sample["sample_id"], context_rows))
        horizon_rows.extend(_effective_horizon_rows(sample))
        regime_rows.append(
            {
                "sample_id": sample["sample_id"],
                "regime_assignment_order": sample["regime_rank"],
                "observed_regime": sample["observed_regime"],
                "binance_rolling_rv_ticks_20_mean": _fmt(
                    sample["regime_inputs"]["binance_rolling_rv_ticks_20_mean"]
                ),
                "combined_trade_events_per_second": _fmt(
                    sample["regime_inputs"]["combined_trade_events_per_second"]
                ),
                "hyperliquid_spread_ticks_median": _fmt(
                    sample["regime_inputs"]["hyperliquid_spread_ticks_median"]
                ),
                "hyperliquid_top5_total_qty_median": _fmt(
                    sample["regime_inputs"]["hyperliquid_top5_total_qty_median"]
                ),
                "classification_input_policy": (
                    "decision_time_public_fields_only_before_future_label_read"
                ),
            }
        )

    starts = sorted(
        datetime.fromisoformat(row["start_time_utc"]).astimezone(timezone.utc)
        for row in quality_rows
    )
    start_separations = [
        (current - previous).total_seconds()
        for previous, current in zip(starts, starts[1:])
    ]
    complete_count = sum(bool(row["complete_context"]) for row in all_context_rows)
    regime_count = len({row["observed_regime"] for row in regime_rows})
    samples_valid = all(bool(row["sample_valid"]) for row in quality_rows)
    start_spacing_valid = all(value >= 1800 for value in start_separations)
    if not samples_valid or len(samples) != 3 or not start_spacing_valid:
        recommendation = "sample_collection_invalid"
    elif regime_count < 2 or complete_count < 100:
        recommendation = "needs_more_public_samples"
    else:
        recommendation = "sample_contract_ready_for_signal_acceptance"

    output_dir.mkdir(parents=True, exist_ok=True)
    quality_fields = list(quality_rows[0]) if quality_rows else []
    regime_fields = list(regime_rows[0]) if regime_rows else []
    horizon_fields = list(horizon_rows[0]) if horizon_rows else []
    _write_csv(output_dir / "sample_quality_matrix.csv", quality_rows, quality_fields)
    _write_csv(
        output_dir / "field_coverage_matrix.csv",
        field_rows,
        [
            "sample_id",
            "field",
            "required_for_complete_context",
            "row_count",
            "non_empty_count",
            "coverage_rate",
        ],
    )
    _write_csv(output_dir / "regime_summary.csv", regime_rows, regime_fields)
    _write_csv(
        output_dir / "effective_horizon_coverage.csv", horizon_rows, horizon_fields
    )
    _write_csv(
        output_dir / "symmetric_edge_context_coverage.csv",
        all_context_rows,
        CONTEXT_FIELDS + ["complete_context"],
    )
    boundary_manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "boundary_flags": BOUNDARY_FLAGS,
        "context_policy": {
            "horizon_ms": CONTEXT_HORIZON_MS,
            "quote_side_selected": False,
            "buy_touch_definition": "current_hyperliquid_best_bid",
            "sell_touch_definition": "current_hyperliquid_best_ask",
            "future_values_role": "labels_only",
        },
    }
    _write_json(output_dir / "boundary_manifest.json", boundary_manifest)
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_root": str(analysis_root),
        "output_dir": str(output_dir),
        "sample_ids": sample_ids,
        "sample_count": len(samples),
        "new_sample_count": len(samples),
        "requested_duration_seconds": 1800,
        "minimum_overlap_seconds": 1500,
        "minimum_start_separation_seconds": 1800,
        "start_separation_seconds": start_separations,
        "start_spacing_valid": start_spacing_valid,
        "regime_policy": (
            "relative ordering by decision-time Binance rolling RV, combined public "
            "trade-event rate, and Hyperliquid public top5 liquidity; assigned before "
            "pricing/future-label files are read"
        ),
        "observed_regime_count": regime_count,
        "observed_regimes": sorted({row["observed_regime"] for row in regime_rows}),
        "context_horizon_ms": CONTEXT_HORIZON_MS,
        "symmetric_context_row_count": len(all_context_rows),
        "complete_symmetric_context_row_count": complete_count,
        "complete_context_rows_by_sample": {
            row["sample_id"]: row["complete_symmetric_context_rows_1000ms"]
            for row in quality_rows
        },
        "all_samples_valid": samples_valid,
        "recommendation": recommendation,
        "t003_creation_unlocked": (
            recommendation == "sample_contract_ready_for_signal_acceptance"
        ),
        "future_labels_are_decision_inputs": False,
        "quote_side_selected": False,
        "artifacts": {
            name: str(output_dir / name)
            for name in [
                "sample_expansion_manifest.json",
                "sample_quality_matrix.csv",
                "field_coverage_matrix.csv",
                "regime_summary.csv",
                "effective_horizon_coverage.csv",
                "symmetric_edge_context_coverage.csv",
                "boundary_manifest.json",
                "recommendation.md",
            ]
        },
    }
    _write_json(output_dir / "sample_expansion_manifest.json", manifest)
    recommendation_text = (
        f"# T002 Recommendation\n\n"
        f"`{recommendation}`\n\n"
        f"- New synchronized public windows: `{len(samples)}`\n"
        f"- Valid windows: `{sum(bool(row['sample_valid']) for row in quality_rows)}`\n"
        f"- Observed public regimes: `{regime_count}`\n"
        f"- Complete symmetric 1000ms contexts: `{complete_count}`\n"
        f"- Per-window complete contexts: "
        + ", ".join(
            f"`{row['sample_id']}={row['complete_symmetric_context_rows_1000ms']}`"
            for row in quality_rows
        )
        + "\n"
        f"- T003 creation unlocked: "
        f"`{str(manifest['t003_creation_unlocked']).lower()}`\n\n"
        "Both Hyperliquid touch alternatives are retained on every complete row. "
        "No side or executable-edge contract is selected by T002.\n"
    )
    (output_dir / "recommendation.md").write_text(recommendation_text, encoding="utf-8")
    return {
        "manifest": manifest,
        "quality_rows": quality_rows,
        "regime_rows": regime_rows,
        "horizon_rows": horizon_rows,
        "context_rows": all_context_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the T002 synchronized public sample expansion package."
    )
    parser.add_argument(
        "--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT
    )
    parser.add_argument(
        "--sample-id", action="append", dest="sample_ids", default=None
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build_artifacts(
        analysis_root=args.analysis_root.expanduser().resolve(),
        sample_ids=args.sample_ids or DEFAULT_SAMPLE_IDS,
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
