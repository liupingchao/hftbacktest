#!/usr/bin/env python3
"""Replay an exact dynamic seed through the production quote builder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher
from examples.hyperliquid import hyperliquid_tiny_live_real_order_executor as executor


TASK_ID = "0722T067"
SCHEMA_VERSION = "cross_exchange_exact_seeded_dynamic_shadow_v1"
DEFAULT_SOURCE_ROOT = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "public_multi_distance_dynamic_seed_0722T066"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "exact_seeded_dynamic_shadow_0722T067"
)
DEFAULT_EXPECTED_SEED_SHA256 = (
    "e35c7fd8f3ec8268e5d50c7963889b73409470c9f01f92ca3a0d598a96562be9"
)


class SeededDynamicShadowError(ValueError):
    """Raised when strict shadow acceptance cannot be established."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def _ingest_event(
    estimator: Any,
    row: dict[str, str],
) -> None:
    local_receive = row.get("local_receive_time_ms")
    local_receive_ms = (
        None if local_receive in {"", None} else int(local_receive)
    )
    if row["event_kind"] == "book":
        estimator.ingest_book(
            event_time_ms=int(row["event_time_ms"]),
            local_receive_time_ms=local_receive_ms,
            bid_px=float(row["bid_px"]),
            ask_px=float(row["ask_px"]),
            bid_depth_btc=float(row["bid_depth_btc"]),
            ask_depth_btc=float(row["ask_depth_btc"]),
        )
        return
    if row["event_kind"] == "trade":
        estimator.ingest_trade(
            event_time_ms=int(row["event_time_ms"]),
            local_receive_time_ms=local_receive_ms,
            trade_px=float(row["trade_px"]),
            trade_size_btc=float(row["trade_size_btc"]),
            aggressor_side=str(row["aggressor_side"]),
            trade_id=str(row.get("trade_id") or ""),
        )
        return
    raise SeededDynamicShadowError("unsupported_event_kind")


def shadow_fieldnames() -> list[str]:
    return [
        "event_index",
        "event_kind",
        "event_time_ms",
        "current_bid_px",
        "current_ask_px",
        "candidate_status",
        "candidate_reason",
        "candidate_bounded",
        "candidate_half_spread_ticks",
        "fixed_half_spread_ticks",
        "authoritative_half_spread_ticks",
        "dynamic_overlay_changed",
        "fixed_bid_px",
        "fixed_ask_px",
        "dynamic_bid_px",
        "dynamic_ask_px",
        "final_quote_behavior_changed",
        "strict_gate_status",
        "strict_gate_allowed",
        "strict_gate_reason",
        "order_endpoint_called",
        "inference_scope",
    ]


def run_shadow(
    *,
    event_rows_path: Path,
    contract_path: Path,
    exposure_path: Path,
    expected_seed_contract_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state = watcher.EventDrivenPublicState(
        max_order_size_btc=watcher.DEFAULT_MAX_ORDER_SIZE_BTC
    )
    seed_load = watcher.load_exact_dynamic_seed(
        state=state,
        contract_path=contract_path,
        exposure_path=exposure_path,
        expected_seed_contract_sha256=expected_seed_contract_sha256,
    )
    if (
        seed_load["current_market_event_count_after"] != 0
        or seed_load["current_market_bucket_count_after"] != 0
    ):
        raise SeededDynamicShadowError(
            "seed_populated_current_market_state"
        )

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    precision = executor.PrecisionFacts(
        symbol=executor.SYMBOL,
        sz_decimals=5,
        tick_size=float(contract["tick_size"]),
        lot_size=0.00001,
        mid_px=1.0,
        source="t067_committed_public_shadow",
    )
    event_rows = _read_csv(event_rows_path)
    current_bid: float | None = None
    current_ask: float | None = None
    rows: list[dict[str, Any]] = []
    for event_index, event in enumerate(event_rows, start=1):
        _ingest_event(state.online_estimator, event)
        if event["event_kind"] == "book":
            current_bid = float(event["bid_px"])
            current_ask = float(event["ask_px"])
        if current_bid is None or current_ask is None:
            continue
        candidate = (
            state.online_estimator.snapshot(inventory_ratio=0.0).get(
                "dynamic_half_spread_candidate"
            )
            or {}
        )
        dynamic_quote = watcher.build_task7_desired_quotes(
            best_bid=current_bid,
            best_ask=current_ask,
            forecast_mid_px=(current_bid + current_ask) / 2.0,
            position_btc=0.0,
            size_btc=watcher.DEFAULT_MAX_ORDER_SIZE_BTC,
            precision=precision,
            task_id=TASK_ID,
            run_id="t067-public-shadow",
            window_id=1,
            dynamic_spread_activation_enabled=True,
            dynamic_spread_candidate=dict(candidate),
        )
        fixed_quote = watcher.build_task7_desired_quotes(
            best_bid=current_bid,
            best_ask=current_ask,
            forecast_mid_px=(current_bid + current_ask) / 2.0,
            position_btc=0.0,
            size_btc=watcher.DEFAULT_MAX_ORDER_SIZE_BTC,
            precision=precision,
            task_id=TASK_ID,
            run_id="t067-public-shadow-fixed",
            window_id=1,
        )
        gate = watcher.strict_seeded_dynamic_submit_gate(
            required=True,
            seed_load_result=seed_load,
            expected_seed_contract_sha256=(
                expected_seed_contract_sha256
            ),
            quote_result=dynamic_quote,
        )
        overlay = dynamic_quote["dynamic_spread_overlay"]
        rows.append(
            {
                "event_index": event_index,
                "event_kind": event["event_kind"],
                "event_time_ms": int(event["event_time_ms"]),
                "current_bid_px": current_bid,
                "current_ask_px": current_ask,
                "candidate_status": overlay["candidate_status"],
                "candidate_reason": candidate.get("reason", ""),
                "candidate_bounded": overlay["candidate_bounded"],
                "candidate_half_spread_ticks": overlay[
                    "dynamic_candidate_half_spread_ticks"
                ],
                "fixed_half_spread_ticks": overlay[
                    "fixed_half_spread_ticks"
                ],
                "authoritative_half_spread_ticks": overlay[
                    "authoritative_half_spread_ticks"
                ],
                "dynamic_overlay_changed": overlay[
                    "quote_behavior_changed"
                ],
                "fixed_bid_px": fixed_quote["bid_px"],
                "fixed_ask_px": fixed_quote["ask_px"],
                "dynamic_bid_px": dynamic_quote["bid_px"],
                "dynamic_ask_px": dynamic_quote["ask_px"],
                "final_quote_behavior_changed": dynamic_quote[
                    "actual_quote_behavior_changed"
                ],
                "strict_gate_status": gate["status"],
                "strict_gate_allowed": gate["allowed"],
                "strict_gate_reason": gate["reason"],
                "order_endpoint_called": False,
                "inference_scope": (
                    "same_sample_production_quote_mechanism_shadow_not_fill_or_economics"
                ),
            }
        )

    pass_rows = [
        row for row in rows if row["candidate_status"] == "pass"
    ]
    strict_rows = [
        row for row in rows if row["strict_gate_allowed"] is True
    ]
    fallback_rows = [
        row for row in rows if row["candidate_status"] != "pass"
    ]
    final_changed_rows = [
        row for row in rows if row["final_quote_behavior_changed"] is True
    ]
    if not strict_rows:
        raise SeededDynamicShadowError(
            "no_strict_seeded_dynamic_quote_reachable"
        )
    if any(
        row["strict_gate_allowed"] is True for row in fallback_rows
    ):
        raise SeededDynamicShadowError(
            "fallback_candidate_bypassed_strict_gate"
        )
    if len(strict_rows) != len(final_changed_rows):
        raise SeededDynamicShadowError(
            "strict_gate_and_final_quote_change_mismatch"
        )

    matrix_path = output_dir / "seeded_dynamic_quote_shadow.csv"
    _write_csv(matrix_path, rows, shadow_fieldnames())
    summary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "status": "pass",
        "seed_contract_sha256": expected_seed_contract_sha256,
        "seed_loaded_row_count": seed_load["loaded_row_count"],
        "seed_current_market_state_contaminated": seed_load[
            "current_market_state_contaminated"
        ],
        "source_event_row_count": len(event_rows),
        "evaluated_quote_row_count": len(rows),
        "candidate_pass_count": len(pass_rows),
        "candidate_fallback_count": len(fallback_rows),
        "final_quote_behavior_changed_count": len(final_changed_rows),
        "strict_gate_pass_count": len(strict_rows),
        "strict_gate_block_count": len(rows) - len(strict_rows),
        "strict_gate_fallback_allowed_count": sum(
            row["strict_gate_allowed"] is True for row in fallback_rows
        ),
        "live_submissions_count": 0,
        "credential_file_read": False,
        "private_endpoint_called": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "same_sample_mechanism_shadow": True,
        "fill_evidence": False,
        "economics_evidence": False,
        "final_recommendation": (
            "accept_exact_seeded_dynamic_production_quote_wiring_for_fresh_authorized_live"
        ),
        "inference_scope": (
            "production_quote_mechanism_reachability_not_oos_fill_or_economics"
        ),
        "inputs": {
            "event_rows": _portable(event_rows_path),
            "event_rows_sha256": _sha256(event_rows_path),
            "seed_contract": _portable(contract_path),
            "seed_contract_file_sha256": _sha256(contract_path),
            "seed_exposures": _portable(exposure_path),
            "seed_exposures_sha256": _sha256(exposure_path),
        },
        "output_files": {
            "quote_shadow_matrix": _portable(matrix_path),
            "summary": _portable(
                output_dir / "seeded_dynamic_shadow_summary.json"
            ),
            "boundary": _portable(output_dir / "boundary_manifest.json"),
            "recommendation": _portable(output_dir / "recommendation.md"),
        },
    }
    boundary = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "public_event_rows_only": True,
        "counterfactual_seed_exposures_only": True,
        "production_quote_builder_used": True,
        "strict_pre_submit_gate_used": True,
        "live_client_initialized": False,
        "credential_file_read": False,
        "private_endpoint_called": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "service_or_orchestrator_started": False,
        "same_sample_mechanism_shadow": True,
        "fill_evidence": False,
        "economics_evidence": False,
    }
    _write_json(output_dir / "seeded_dynamic_shadow_summary.json", summary)
    _write_json(output_dir / "boundary_manifest.json", boundary)
    (output_dir / "recommendation.md").write_text(
        "# Recommendation\n\n"
        "- Accept exact seeded dynamic production-quote wiring for a later "
        "fresh-authorized bounded live task.\n"
        "- Do not treat this same-sample no-submit shadow as fill, OOS, or "
        "economics evidence.\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--event-rows",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "online_estimator_event_rows.csv",
    )
    parser.add_argument(
        "--seed-contract",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "dynamic_spread_seed_contract.json",
    )
    parser.add_argument(
        "--seed-exposures",
        type=Path,
        default=DEFAULT_SOURCE_ROOT / "quote_exposure_intervals.csv",
    )
    parser.add_argument(
        "--expected-seed-sha256",
        default=DEFAULT_EXPECTED_SEED_SHA256,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = run_shadow(
        event_rows_path=args.event_rows,
        contract_path=args.seed_contract,
        exposure_path=args.seed_exposures,
        expected_seed_contract_sha256=args.expected_seed_sha256,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
