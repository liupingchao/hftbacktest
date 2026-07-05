#!/usr/bin/env python3
"""Offline public market-view replay alignment for the cross-exchange MVP.

This runner consumes QA-accepted local public artifacts only. It reconstructs
replay market-view rows from the accepted aligned public context package,
replays the shared T004 decision kernel, and compares the result with T005
production-shadow decisions. It does not collect data, read credentials, call
private/order endpoints, initialize a live client, or authorize live orders.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import cross_exchange_shared_signal_kernel as shared_kernel


TASK_ID = "0625T007"
SCHEMA_VERSION = "cross_exchange_public_replay_alignment_v1"
FINAL_RECOMMENDATION = "public_market_view_replay_alignment_ready_for_qa"
BLOCKED_RECOMMENDATION = "public_market_view_replay_alignment_blocked"

DEFAULT_INPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_hl_fast_sample_expansion_0627T001"
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_signal_acceptance_0625T003" / "accepted_signal_contract.json"
DEFAULT_KERNEL_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_shared_kernel_0625T004" / "shared_kernel_manifest.json"
DEFAULT_KERNEL_BOUNDARY = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_shared_kernel_0625T004" / "boundary_manifest.json"
DEFAULT_SHADOW_MANIFEST = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_production_shadow_0625T005" / "production_shadow_manifest.json"
DEFAULT_REFERENCE_DECISIONS = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_production_shadow_0625T005" / "shadow_decision_rows.csv"
DEFAULT_REPLAY_CONTRACT = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_audit_replay_contract_0625T006" / "replay_input_contract.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "cross_exchange_mvp_public_replay_alignment_0625T007"

HORIZON_MS = 1000
NEAR_TARGET_LOWER_MS = 1000.0
NEAR_TARGET_UPPER_MS = 1250.0
NUMERIC_COMPARE_TOLERANCE = 1e-8
MAX_UNEXPLAINED_MISMATCHES = 0

NUMERIC_FIELDS = {
    "nominal_horizon_ms",
    "effective_future_age_ms",
    "hyperliquid_decision_ts",
    "hyperliquid_l2book_local_ts",
    "hyperliquid_l2book_event_ts",
    "binance_local_ts",
    "binance_exch_ts",
    "binance_source_age_ms",
    "hyperliquid_join_age_ms",
    "future_hyperliquid_decision_ts",
    "hyperliquid_current_bid_px",
    "hyperliquid_current_ask_px",
    "hyperliquid_buy_touch_quote_px",
    "hyperliquid_sell_touch_quote_px",
    "tick_size",
    "hyperliquid_mid_px",
    "basis_mid_ticks",
    "input_binance_top5_imbalance",
    "input_binance_microprice_minus_mid_ticks",
    "input_binance_mid_move_ticks_from_prev",
}
REPLAY_MARKET_VIEW_FIELDS = [
    "sample_id",
    "row_id",
    "decision_id",
    "observed_regime",
    "hyperliquid_decision_ts",
    "hyperliquid_l2book_local_ts",
    "hyperliquid_l2book_event_ts",
    "binance_local_ts",
    "binance_exch_ts",
    "binance_source_age_ms",
    "hyperliquid_join_age_ms",
    "hyperliquid_current_bid_px",
    "hyperliquid_current_ask_px",
    "hyperliquid_mid_px",
    "tick_size",
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
    "source_age_bucket",
    "basis_bucket",
    "warning_bucket",
    "market_view_status",
    "market_view_issue",
]
REPLAY_DECISION_FIELDS = [
    "row_id",
    "sample_id",
    "observed_regime",
    "decision_id",
    "action",
    "block_reason",
    "signal_status",
    "signal_score",
    "signal_abs_z",
    "side",
    "fair_mid_px",
    "quote_px",
    "edge_ticks",
    "required_edge_ticks",
    "quote_type",
    "time_in_force",
    "post_only",
    "source_age_bucket",
    "basis_bucket",
    "warning_bucket",
    "order_endpoint_called",
    "private_endpoint_called",
    "credential_read",
]
COMPARE_FIELDS = [
    "action",
    "block_reason",
    "signal_status",
    "signal_score",
    "signal_abs_z",
    "side",
    "fair_mid_px",
    "quote_px",
    "edge_ticks",
    "required_edge_ticks",
    "quote_type",
    "time_in_force",
    "post_only",
    "source_age_bucket",
    "basis_bucket",
    "warning_bucket",
    "order_endpoint_called",
    "private_endpoint_called",
    "credential_read",
]
NUMERIC_COMPARE_FIELDS = {
    "signal_score",
    "signal_abs_z",
    "fair_mid_px",
    "quote_px",
    "edge_ticks",
    "required_edge_ticks",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh)]


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _fmt(value: Any, places: int = 8) -> str:
    parsed = _float(value)
    if parsed is None:
        return ""
    text = f"{parsed:.{places}f}".rstrip("0").rstrip(".")
    return text or "0"


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _bucket_by_quantiles(value: float | None, low: float, high: float, prefix: str) -> str:
    if value is None:
        return f"{prefix}_missing"
    if value <= low:
        return f"{prefix}_low"
    if value >= high:
        return f"{prefix}_high"
    return f"{prefix}_mid"


def parse_public_rows(input_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    raw_rows = read_csv(input_dir / "symmetric_edge_context_coverage.csv")
    parsed_rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw_rows, start=1):
        parsed: dict[str, Any] = dict(row)
        parsed["row_id"] = index
        for field in NUMERIC_FIELDS:
            parsed[field] = _float(row.get(field))
        parsed_rows.append(parsed)

    valid_rows = [
        row
        for row in parsed_rows
        if _bool(row.get("valid_for_1000ms_signal_acceptance"))
        and row.get("nominal_horizon_ms") == float(HORIZON_MS)
        and row.get("effective_future_age_ms") is not None
        and NEAR_TARGET_LOWER_MS <= row["effective_future_age_ms"] <= NEAR_TARGET_UPPER_MS
    ]
    source_ages = [row["binance_source_age_ms"] for row in valid_rows if row.get("binance_source_age_ms") is not None]
    bases = [row["basis_mid_ticks"] for row in valid_rows if row.get("basis_mid_ticks") is not None]
    source_low = _percentile(source_ages, 1 / 3) or 0.0
    source_high = _percentile(source_ages, 2 / 3) or 0.0
    basis_low = _percentile(bases, 1 / 3) or 0.0
    basis_high = _percentile(bases, 2 / 3) or 0.0
    for row in valid_rows:
        row["source_age_bucket"] = _bucket_by_quantiles(row.get("binance_source_age_ms"), source_low, source_high, "binance_source_age")
        row["basis_bucket"] = _bucket_by_quantiles(row.get("basis_mid_ticks"), basis_low, basis_high, "basis")
        row["warning_bucket"] = (
            row.get("sample_id") == "xemm_0627_t001_hlfast_utc17_b"
            and row["source_age_bucket"] == "binance_source_age_mid"
        )
        row["decision_id"] = f"{row.get('sample_id')}:{row.get('row_id')}"
    return valid_rows, raw_rows


def _pipe_count(value: Any) -> int:
    text = str(value or "")
    if not text:
        return 0
    return len(text.split("|"))


def _market_view_issue(row: dict[str, Any]) -> str:
    issues: list[str] = []
    bid = _float(row.get("hyperliquid_current_bid_px"))
    ask = _float(row.get("hyperliquid_current_ask_px"))
    mid = _float(row.get("hyperliquid_mid_px"))
    tick_size = _float(row.get("tick_size"))
    if bid is None or ask is None or ask <= bid:
        issues.append("invalid_hyperliquid_bbo")
    if mid is None:
        issues.append("missing_hyperliquid_mid")
    if tick_size is None or tick_size <= 0:
        issues.append("invalid_tick_size")
    for field in [
        "hyperliquid_bid_top5_px",
        "hyperliquid_ask_top5_px",
        "hyperliquid_bid_top5_qtys",
        "hyperliquid_ask_top5_qtys",
        "binance_bid_top5_px",
        "binance_ask_top5_px",
        "binance_bid_top5_qtys",
        "binance_ask_top5_qtys",
    ]:
        if _pipe_count(row.get(field)) < 5:
            issues.append(f"missing_top5:{field}")
    for field in [
        "hyperliquid_decision_ts",
        "hyperliquid_l2book_local_ts",
        "hyperliquid_l2book_event_ts",
        "binance_local_ts",
        "binance_exch_ts",
    ]:
        if _float(row.get(field)) is None:
            issues.append(f"missing_timestamp:{field}")
    for field in ["binance_source_age_ms", "hyperliquid_join_age_ms"]:
        value = _float(row.get(field))
        if value is None or value < 0:
            issues.append(f"invalid_source_age:{field}")
    return "|".join(issues)


def replay_market_view_rows(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in valid_rows:
        issue = _market_view_issue(row)
        output = {field: row.get(field, "") for field in REPLAY_MARKET_VIEW_FIELDS if field not in {"market_view_status", "market_view_issue"}}
        output["market_view_status"] = "pass" if not issue else "fail_closed"
        output["market_view_issue"] = issue
        rows.append(output)
    return rows


def _market_view_for_kernel(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": row.get("decision_id", ""),
        "sample_id": row.get("sample_id", ""),
        "observed_regime": row.get("observed_regime", ""),
        "hyperliquid_bid_px": row.get("hyperliquid_current_bid_px"),
        "hyperliquid_ask_px": row.get("hyperliquid_current_ask_px"),
        "hyperliquid_mid_px": row.get("hyperliquid_mid_px"),
        "tick_size": row.get("tick_size"),
        "input_binance_top5_imbalance": row.get("input_binance_top5_imbalance"),
        "input_binance_microprice_minus_mid_ticks": row.get("input_binance_microprice_minus_mid_ticks"),
        "input_binance_mid_move_ticks_from_prev": row.get("input_binance_mid_move_ticks_from_prev"),
        "source_age_bucket": row.get("source_age_bucket", ""),
        "basis_bucket": row.get("basis_bucket", ""),
        "warning_bucket": row.get("warning_bucket", False),
    }


def _decision_row_from_kernel(row: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_id": row.get("row_id", ""),
        "sample_id": row.get("sample_id", ""),
        "observed_regime": row.get("observed_regime", ""),
        "decision_id": decision.get("decision_id", ""),
        "action": decision.get("action", ""),
        "block_reason": decision.get("block_reason", ""),
        "signal_status": decision.get("signal_status", ""),
        "signal_score": _fmt(decision.get("signal_score")),
        "signal_abs_z": _fmt(decision.get("signal_abs_z")),
        "side": decision.get("side", ""),
        "fair_mid_px": _fmt(decision.get("fair_mid_px")),
        "quote_px": _fmt(decision.get("quote_px")),
        "edge_ticks": _fmt(decision.get("edge_ticks")),
        "required_edge_ticks": _fmt(decision.get("required_edge_ticks")),
        "quote_type": (decision.get("quote_intent") or {}).get("quote_type", ""),
        "time_in_force": (decision.get("quote_intent") or {}).get("time_in_force", ""),
        "post_only": (decision.get("quote_intent") or {}).get("post_only", ""),
        "source_age_bucket": row.get("source_age_bucket", ""),
        "basis_bucket": row.get("basis_bucket", ""),
        "warning_bucket": bool(row.get("warning_bucket", False)),
        "order_endpoint_called": False,
        "private_endpoint_called": False,
        "credential_read": False,
    }


def replay_decision_rows(
    valid_rows: list[dict[str, Any]],
    *,
    contract: dict[str, Any],
    normalization_stats: dict[str, dict[str, Any]],
    kernel_parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    expected_move = float(kernel_parameters.get("expected_move_ticks_per_signal_z", shared_kernel.DEFAULT_EXPECTED_MOVE_TICKS_PER_SIGNAL_Z))
    required_edge = float(kernel_parameters.get("required_edge_ticks", shared_kernel.DEFAULT_REQUIRED_EDGE_TICKS))
    rows: list[dict[str, Any]] = []
    for row in valid_rows:
        decision = shared_kernel.evaluate_shared_kernel(
            _market_view_for_kernel(row),
            contract=contract,
            normalization_stats=normalization_stats,
            expected_move_ticks_per_signal_z=expected_move,
            required_edge_ticks=required_edge,
        )
        rows.append(_decision_row_from_kernel(row, decision))
    return rows


def _comparable(value: Any) -> str:
    return str(value).strip()


def _field_matches(field: str, left: Any, right: Any) -> bool:
    if field in NUMERIC_COMPARE_FIELDS:
        left_value = _float(left)
        right_value = _float(right)
        if left_value is None and right_value is None:
            return True
        if left_value is None or right_value is None:
            return False
        return abs(left_value - right_value) <= NUMERIC_COMPARE_TOLERANCE
    if field in {"post_only", "warning_bucket", "order_endpoint_called", "private_endpoint_called", "credential_read"}:
        return _bool(left) == _bool(right)
    return _comparable(left) == _comparable(right)


def compare_decisions(replay_rows: list[dict[str, Any]], reference_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_by_id = {row["decision_id"]: row for row in reference_rows}
    replay_by_id = {str(row["decision_id"]): row for row in replay_rows}
    comparison_rows: list[dict[str, Any]] = []
    mismatch_rows: list[dict[str, Any]] = []
    all_ids = sorted(set(reference_by_id) | set(replay_by_id))
    for decision_id in all_ids:
        replay = replay_by_id.get(decision_id)
        reference = reference_by_id.get(decision_id)
        if replay is None:
            comparison_rows.append(
                {
                    "decision_id": decision_id,
                    "sample_id": reference.get("sample_id", "") if reference else "",
                    "status": "missing_replay",
                    "mismatch_count": 1,
                    "mismatch_fields": "decision_id",
                    "action_match": False,
                }
            )
            mismatch_rows.append({"decision_id": decision_id, "sample_id": reference.get("sample_id", "") if reference else "", "field": "decision_id", "reference_value": "present", "replay_value": "missing", "attribution": "missing_replay_decision"})
            continue
        if reference is None:
            comparison_rows.append(
                {
                    "decision_id": decision_id,
                    "sample_id": replay.get("sample_id", ""),
                    "status": "missing_reference",
                    "mismatch_count": 1,
                    "mismatch_fields": "decision_id",
                    "action_match": False,
                }
            )
            mismatch_rows.append({"decision_id": decision_id, "sample_id": replay.get("sample_id", ""), "field": "decision_id", "reference_value": "missing", "replay_value": "present", "attribution": "extra_replay_decision"})
            continue
        mismatches: list[str] = []
        for field in COMPARE_FIELDS:
            if not _field_matches(field, replay.get(field, ""), reference.get(field, "")):
                mismatches.append(field)
                mismatch_rows.append(
                    {
                        "decision_id": decision_id,
                        "sample_id": replay.get("sample_id", ""),
                        "field": field,
                        "reference_value": reference.get(field, ""),
                        "replay_value": replay.get(field, ""),
                        "attribution": "unexplained_action_path_mismatch",
                    }
                )
        comparison_rows.append(
            {
                "decision_id": decision_id,
                "sample_id": replay.get("sample_id", ""),
                "status": "match" if not mismatches else "mismatch",
                "mismatch_count": len(mismatches),
                "mismatch_fields": "|".join(mismatches),
                "action_match": "action" not in mismatches,
            }
        )
    return comparison_rows, mismatch_rows


def mismatch_attribution_rows(mismatch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not mismatch_rows:
        return [{"attribution": "none", "field": "none", "mismatch_count": 0, "status": "pass"}]
    counts = Counter((row["attribution"], row["field"]) for row in mismatch_rows)
    return [
        {"attribution": attribution, "field": field, "mismatch_count": count, "status": "fail_closed"}
        for (attribution, field), count in sorted(counts.items())
    ]


def cadence_source_age_rows(market_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in market_rows:
        grouped[str(row.get("sample_id", ""))].append(row)
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(grouped):
        group = sorted(grouped[sample_id], key=lambda item: _float(item.get("hyperliquid_decision_ts")) or -1)
        decision_ts = [_float(row.get("hyperliquid_decision_ts")) for row in group]
        decision_ts_values = [value for value in decision_ts if value is not None]
        deltas_ms = [
            (right - left) / 1_000_000.0
            for left, right in zip(decision_ts_values, decision_ts_values[1:])
            if right >= left
        ]
        non_monotonic = sum(
            1
            for left, right in zip(decision_ts_values, decision_ts_values[1:])
            if right < left
        )
        binance_ages = [_float(row.get("binance_source_age_ms")) for row in group]
        hl_ages = [_float(row.get("hyperliquid_join_age_ms")) for row in group]
        rows.append(
            {
                "sample_id": sample_id,
                "replay_rows": len(group),
                "market_view_fail_closed_count": sum(1 for row in group if row.get("market_view_status") != "pass"),
                "non_monotonic_decision_ts_count": non_monotonic,
                "median_decision_delta_ms": _fmt(_median(deltas_ms)),
                "max_decision_delta_ms": _fmt(max(deltas_ms) if deltas_ms else None),
                "median_binance_source_age_ms": _fmt(_median([value for value in binance_ages if value is not None])),
                "max_binance_source_age_ms": _fmt(max([value for value in binance_ages if value is not None], default=math.nan)),
                "median_hyperliquid_join_age_ms": _fmt(_median([value for value in hl_ages if value is not None])),
                "max_hyperliquid_join_age_ms": _fmt(max([value for value in hl_ages if value is not None], default=math.nan)),
                "cadence_source_age_gate": "pass" if non_monotonic == 0 and all(row.get("market_view_status") == "pass" for row in group) else "fail_closed",
            }
        )
    return rows


def future_join_report_rows(valid_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    replay_future_fields = [field for field in REPLAY_MARKET_VIEW_FIELDS if "future" in field.lower()]
    future_timestamp_not_after_decision = 0
    for row in valid_rows:
        future_ts = _float(row.get("future_hyperliquid_decision_ts"))
        decision_ts = _float(row.get("hyperliquid_decision_ts"))
        if future_ts is None or decision_ts is None or future_ts <= decision_ts:
            future_timestamp_not_after_decision += 1
    return [
        {
            "check_id": "decision_input_future_field_count",
            "value": len(replay_future_fields),
            "status": "pass" if len(replay_future_fields) == 0 else "fail_closed",
            "detail": "|".join(replay_future_fields),
        },
        {
            "check_id": "future_label_rows_present_but_excluded",
            "value": len(valid_rows),
            "status": "pass",
            "detail": "future labels are present in source package but not replay decision inputs",
        },
        {
            "check_id": "future_timestamp_not_after_decision_count",
            "value": future_timestamp_not_after_decision,
            "status": "pass" if future_timestamp_not_after_decision == 0 else "fail_closed",
            "detail": "future_hyperliquid_decision_ts must be after hyperliquid_decision_ts",
        },
    ]


def boundary_manifest() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "public_market_data_only": True,
        "offline_local_processing_only": True,
        "no_new_public_collection": True,
        "no_submit": True,
        "no_network_collection": True,
        "no_aws_execution": True,
        "no_remote_alignment": True,
        "no_credentials": True,
        "no_private_account_order_cancel_endpoints": True,
        "no_user_stream": True,
        "no_live_client_initialization": True,
        "no_live_orders": True,
        "no_watcher_strategy_change": True,
        "no_production_config_change": True,
        "no_signal_feature_search": True,
        "no_threshold_tuning": True,
        "no_side_mapping_change": True,
        "no_horizon_change": True,
        "no_canary_or_promotion_authorization": True,
        "raw_websocket_files_not_present_in_local_repo": True,
        "replay_source_is_qa_accepted_aligned_public_context": True,
        "future_labels_used_as_decision_inputs": False,
    }


def validation_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# 0625T007 Public Replay Alignment Report",
        "",
        f"- final_recommendation: `{manifest['final_recommendation']}`",
        f"- replay_rows: `{manifest['replay_row_count']}`",
        f"- reference_decision_rows: `{manifest['reference_decision_row_count']}`",
        f"- matched_decision_rows: `{manifest['matched_decision_row_count']}`",
        f"- mismatched_decision_rows: `{manifest['mismatched_decision_row_count']}`",
        f"- action_mismatch_count: `{manifest['action_mismatch_count']}`",
        f"- future_join_count: `{manifest['future_join_count']}`",
        f"- market_view_fail_closed_count: `{manifest['market_view_fail_closed_count']}`",
        "",
        "The replay source is the QA-accepted aligned public context package. Raw WebSocket files are not present in this local repository and no new public collection was performed.",
        "",
        "No submit/private/live/order behavior is authorized by this artifact.",
    ]
    return "\n".join(lines) + "\n"


def build_artifacts(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    kernel_manifest_path: Path = DEFAULT_KERNEL_MANIFEST,
    kernel_boundary_path: Path = DEFAULT_KERNEL_BOUNDARY,
    shadow_manifest_path: Path = DEFAULT_SHADOW_MANIFEST,
    reference_decision_path: Path = DEFAULT_REFERENCE_DECISIONS,
    replay_contract_path: Path = DEFAULT_REPLAY_CONTRACT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in [
        input_dir / "symmetric_edge_context_coverage.csv",
        contract_path,
        kernel_manifest_path,
        kernel_boundary_path,
        shadow_manifest_path,
        reference_decision_path,
        replay_contract_path,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    valid_rows, raw_rows = parse_public_rows(input_dir)
    market_rows = replay_market_view_rows(valid_rows)
    contract = shared_kernel.load_signal_contract(contract_path)
    kernel_manifest = read_json(kernel_manifest_path)
    kernel_boundary = read_json(kernel_boundary_path)
    shadow_manifest = read_json(shadow_manifest_path)
    replay_contract = read_json(replay_contract_path)
    normalization_stats = dict(shadow_manifest.get("normalization_stats") or {})
    kernel_parameters = dict(shadow_manifest.get("kernel_parameters") or kernel_manifest.get("kernel_parameters") or {})
    replay_rows = replay_decision_rows(valid_rows, contract=contract, normalization_stats=normalization_stats, kernel_parameters=kernel_parameters)
    reference_rows = read_csv(reference_decision_path)
    comparison_rows, mismatch_rows = compare_decisions(replay_rows, reference_rows)
    attribution_rows = mismatch_attribution_rows(mismatch_rows)
    cadence_rows = cadence_source_age_rows(market_rows)
    future_rows = future_join_report_rows(valid_rows)
    boundary = boundary_manifest()

    mismatched_decisions = sum(1 for row in comparison_rows if row.get("status") != "match")
    action_mismatch_count = sum(1 for row in mismatch_rows if row.get("field") == "action")
    market_view_fail_closed_count = sum(1 for row in market_rows if row.get("market_view_status") != "pass")
    future_join_count = sum(int(row["value"]) for row in future_rows if row["status"] != "pass")
    cadence_gate_fail_count = sum(1 for row in cadence_rows if row.get("cadence_source_age_gate") != "pass")
    reference_missing_count = sum(1 for row in comparison_rows if row.get("status") == "missing_reference")
    replay_missing_count = sum(1 for row in comparison_rows if row.get("status") == "missing_replay")
    matched_count = sum(1 for row in comparison_rows if row.get("status") == "match")
    blocking_reasons: list[str] = []
    if mismatched_decisions > MAX_UNEXPLAINED_MISMATCHES:
        blocking_reasons.append("unexplained_action_path_mismatch")
    if action_mismatch_count:
        blocking_reasons.append("action_mismatch")
    if market_view_fail_closed_count:
        blocking_reasons.append("market_view_gate_failed")
    if future_join_count:
        blocking_reasons.append("future_join_gate_failed")
    if cadence_gate_fail_count:
        blocking_reasons.append("cadence_source_age_gate_failed")
    if kernel_boundary.get("no_live_orders") is not True:
        blocking_reasons.append("kernel_boundary_no_live_orders_not_true")
    if replay_contract.get("schema_hash") != "0a899c61d63cf5326e16fa8b2d95ae7dc965b04ada72f3ba99811abfca0b9ab5":
        blocking_reasons.append("unexpected_t006_schema_hash")
    final_recommendation = FINAL_RECOMMENDATION if not blocking_reasons else BLOCKED_RECOMMENDATION

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_task_ids": ["0627T001", "0625T004", "0625T005", "0625T006"],
        "input_dir": str(input_dir),
        "contract_path": str(contract_path),
        "kernel_manifest_path": str(kernel_manifest_path),
        "shadow_manifest_path": str(shadow_manifest_path),
        "reference_decision_path": str(reference_decision_path),
        "replay_contract_path": str(replay_contract_path),
        "raw_websocket_files_present": False,
        "replay_source": "qa_accepted_0627T001_aligned_public_context_rows",
        "raw_row_count": len(raw_rows),
        "replay_row_count": len(replay_rows),
        "reference_decision_row_count": len(reference_rows),
        "comparison_row_count": len(comparison_rows),
        "matched_decision_row_count": matched_count,
        "mismatched_decision_row_count": mismatched_decisions,
        "mismatch_row_count": len(mismatch_rows),
        "action_mismatch_count": action_mismatch_count,
        "missing_reference_count": reference_missing_count,
        "missing_replay_count": replay_missing_count,
        "market_view_fail_closed_count": market_view_fail_closed_count,
        "cadence_source_age_gate_fail_count": cadence_gate_fail_count,
        "future_join_count": future_join_count,
        "kernel_parameters": kernel_parameters,
        "normalization_stats_source": shadow_manifest.get("normalization_stats_source", ""),
        "t006_schema_hash": replay_contract.get("schema_hash", ""),
        "t005_caveat_preserved": "median_adjusted_counterfactual_edge_ticks=-1.5",
        "final_recommendation": final_recommendation,
        "blocking_reasons": blocking_reasons,
        "output_files": {
            "replay_alignment_manifest": str(output_dir / "replay_alignment_manifest.json"),
            "replay_market_view_rows": str(output_dir / "replay_market_view_rows.csv"),
            "replay_decision_rows": str(output_dir / "replay_decision_rows.csv"),
            "action_path_comparison": str(output_dir / "action_path_comparison.csv"),
            "mismatch_attribution": str(output_dir / "mismatch_attribution.csv"),
            "cadence_source_age_report": str(output_dir / "cadence_source_age_report.csv"),
            "future_join_report": str(output_dir / "future_join_report.csv"),
            "boundary_manifest": str(output_dir / "boundary_manifest.json"),
            "validation_report": str(output_dir / "validation_report.md"),
        },
    }

    write_csv(output_dir / "replay_market_view_rows.csv", market_rows, REPLAY_MARKET_VIEW_FIELDS)
    write_csv(output_dir / "replay_decision_rows.csv", replay_rows, REPLAY_DECISION_FIELDS)
    write_csv(
        output_dir / "action_path_comparison.csv",
        comparison_rows,
        ["decision_id", "sample_id", "status", "mismatch_count", "mismatch_fields", "action_match"],
    )
    write_csv(
        output_dir / "mismatch_attribution.csv",
        attribution_rows,
        ["attribution", "field", "mismatch_count", "status"],
    )
    write_csv(
        output_dir / "cadence_source_age_report.csv",
        cadence_rows,
        [
            "sample_id",
            "replay_rows",
            "market_view_fail_closed_count",
            "non_monotonic_decision_ts_count",
            "median_decision_delta_ms",
            "max_decision_delta_ms",
            "median_binance_source_age_ms",
            "max_binance_source_age_ms",
            "median_hyperliquid_join_age_ms",
            "max_hyperliquid_join_age_ms",
            "cadence_source_age_gate",
        ],
    )
    write_csv(output_dir / "future_join_report.csv", future_rows, ["check_id", "value", "status", "detail"])
    write_json(output_dir / "boundary_manifest.json", boundary)
    write_json(output_dir / "replay_alignment_manifest.json", manifest)
    (output_dir / "validation_report.md").write_text(validation_report(manifest), encoding="utf-8")
    return {
        "manifest": manifest,
        "market_rows": market_rows,
        "replay_rows": replay_rows,
        "comparison_rows": comparison_rows,
        "mismatch_rows": mismatch_rows,
        "cadence_rows": cadence_rows,
        "future_rows": future_rows,
        "boundary_manifest": boundary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline cross-exchange public market-view replay alignment")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--contract-path", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--kernel-manifest-path", type=Path, default=DEFAULT_KERNEL_MANIFEST)
    parser.add_argument("--kernel-boundary-path", type=Path, default=DEFAULT_KERNEL_BOUNDARY)
    parser.add_argument("--shadow-manifest-path", type=Path, default=DEFAULT_SHADOW_MANIFEST)
    parser.add_argument("--reference-decision-path", type=Path, default=DEFAULT_REFERENCE_DECISIONS)
    parser.add_argument("--replay-contract-path", type=Path, default=DEFAULT_REPLAY_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = build_artifacts(
        input_dir=args.input_dir,
        contract_path=args.contract_path,
        kernel_manifest_path=args.kernel_manifest_path,
        kernel_boundary_path=args.kernel_boundary_path,
        shadow_manifest_path=args.shadow_manifest_path,
        reference_decision_path=args.reference_decision_path,
        replay_contract_path=args.replay_contract_path,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
