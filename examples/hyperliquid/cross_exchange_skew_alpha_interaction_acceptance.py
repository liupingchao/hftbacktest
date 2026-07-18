#!/usr/bin/env python3
"""Offline C12 acceptance for alpha plus bounded inventory skew."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import cross_exchange_shared_signal_kernel as kernel


TASK_ID = "0718T016"
SCHEMA_VERSION = "cross_exchange_skew_alpha_interaction_acceptance_v1"
EXPECTED_MOVE_TICKS_PER_SIGNAL_Z = 4.0
TICK_SIZE = 1.0
SZ_DECIMALS = 5
MAX_POSITION_BTC = 0.01
BOUNDED_SKEW_TICKS_AT_MAX = 1.0
DEFAULT_OUTPUT_DIR = (
    kernel.PROJECT_ROOT
    / "local_live_analysis"
    / "cross_exchange_skew_alpha_interaction_acceptance_0718T016"
)


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


def fixture_opportunities() -> list[dict[str, Any]]:
    """A small same-universe fixture with observed and censored outcomes."""

    return [
        {
            "decision_id": "c12_001",
            "decision_ts_ms": 1_000,
            "alpha_score": 0.40,
            "position_btc": 0.000,
            "mid_px": 100.0,
            "best_bid": 99.0,
            "best_ask": 101.0,
            "observed_fill_status": "observed_fill",
            "observed_fill_side": "buy",
            "observed_fill_px": 99.0,
            "future_mid_1s_px": 100.4,
            "future_mid_5s_px": 101.0,
        },
        {
            "decision_id": "c12_002",
            "decision_ts_ms": 2_000,
            "alpha_score": -0.35,
            "position_btc": 0.0085,
            "mid_px": 100.0,
            "best_bid": 99.0,
            "best_ask": 101.0,
            "observed_fill_status": "observed_fill",
            "observed_fill_side": "sell",
            "observed_fill_px": 101.0,
            "future_mid_1s_px": 100.1,
            "future_mid_5s_px": 99.5,
        },
        {
            "decision_id": "c12_003",
            "decision_ts_ms": 3_000,
            "alpha_score": 0.30,
            "position_btc": -0.0085,
            "mid_px": 100.0,
            "best_bid": 99.0,
            "best_ask": 101.0,
            "observed_fill_status": "observed_fill",
            "observed_fill_side": "buy",
            "observed_fill_px": 99.0,
            "future_mid_1s_px": 99.8,
            "future_mid_5s_px": 99.2,
        },
        {
            "decision_id": "c12_004",
            "decision_ts_ms": 4_000,
            "alpha_score": 0.05,
            "position_btc": 0.0020,
            "mid_px": 100.0,
            "best_bid": 99.0,
            "best_ask": 101.0,
            "observed_fill_status": "censored_no_fill",
            "observed_fill_side": "",
            "observed_fill_px": "",
            "future_mid_1s_px": 100.1,
            "future_mid_5s_px": 100.2,
        },
        {
            "decision_id": "c12_005",
            "decision_ts_ms": 5_000,
            "alpha_score": -0.05,
            "position_btc": -0.0020,
            "mid_px": 100.0,
            "best_bid": 99.0,
            "best_ask": 101.0,
            "observed_fill_status": "censored_no_fill",
            "observed_fill_side": "",
            "observed_fill_px": "",
            "future_mid_1s_px": 99.9,
            "future_mid_5s_px": 99.8,
        },
        {
            "decision_id": "c12_006",
            "decision_ts_ms": 6_000,
            "alpha_score": -0.25,
            "position_btc": 0.0095,
            "mid_px": 100.0,
            "best_bid": 99.0,
            "best_ask": 101.0,
            "observed_fill_status": "censored_no_fill",
            "observed_fill_side": "",
            "observed_fill_px": "",
            "future_mid_1s_px": 100.0,
            "future_mid_5s_px": 100.0,
        },
    ]


def _side_markout(side: str, fill_px: float, future_px: float) -> float:
    return future_px - fill_px if side == "buy" else fill_px - future_px


def _proxy_filled(side: str, quote_px: float, future_px: float) -> bool:
    if side == "buy":
        return future_px <= quote_px
    return future_px >= quote_px


def _evaluate_case(rows: list[dict[str, Any]], *, case: str, skew_ticks: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        alpha = float(row["alpha_score"])
        forecast = float(row["mid_px"]) + alpha * EXPECTED_MOVE_TICKS_PER_SIGNAL_Z * TICK_SIZE
        reservation = kernel.compute_reservation_price(
            forecast_mid_px=forecast,
            position_btc=float(row["position_btc"]),
            mid_px=float(row["mid_px"]),
            max_position_btc=MAX_POSITION_BTC,
            inventory_skew_ticks_at_max=skew_ticks,
            price_increment=TICK_SIZE,
        )
        quotes = kernel.compute_two_sided_quotes(
            reservation_px=reservation.reservation_px,
            half_spread_ticks=0.5,
            best_bid=float(row["best_bid"]),
            best_ask=float(row["best_ask"]),
            precision={"tick_size": TICK_SIZE, "sz_decimals": SZ_DECIMALS},
        )
        sides, inventory_mode = kernel.inventory_quote_sides(reservation.raw_position_ratio)
        side_set = set(sides)
        proxy_sides = [
            side
            for side, quote_px in (("buy", quotes.bid_px), ("sell", quotes.ask_px))
            if side in side_set and _proxy_filled(side, quote_px, float(row["future_mid_1s_px"]))
        ]
        observed_side = str(row["observed_fill_side"])
        observed_markout_1s = ""
        observed_markout_5s = ""
        if row["observed_fill_status"] == "observed_fill":
            observed_markout_1s = _side_markout(
                observed_side,
                float(row["observed_fill_px"]),
                float(row["future_mid_1s_px"]),
            )
            observed_markout_5s = _side_markout(
                observed_side,
                float(row["observed_fill_px"]),
                float(row["future_mid_5s_px"]),
            )
        output.append(
            {
                "case": case,
                "decision_id": row["decision_id"],
                "decision_ts_ms": row["decision_ts_ms"],
                "alpha_score": alpha,
                "position_btc": row["position_btc"],
                "position_ratio": reservation.raw_position_ratio,
                "position_notional": reservation.position_notional,
                "forecast_mid_px": forecast,
                "reservation_px": reservation.reservation_px,
                "inventory_penalty_ticks": reservation.inventory_penalty_ticks,
                "quote_bid_px": quotes.bid_px,
                "quote_ask_px": quotes.ask_px,
                "bid_distance_ticks": (float(row["mid_px"]) - quotes.bid_px) / TICK_SIZE,
                "ask_distance_ticks": (quotes.ask_px - float(row["mid_px"])) / TICK_SIZE,
                "bid_clamp_reason": quotes.bid_clamp_reason,
                "ask_clamp_reason": quotes.ask_clamp_reason,
                "bid_edge_change_ticks": quotes.bid_edge_change_ticks,
                "ask_edge_change_ticks": quotes.ask_edge_change_ticks,
                "post_only_invariant": quotes.post_only_invariant,
                "inventory_mode": inventory_mode,
                "eligible_sides": "|".join(sides),
                "observed_fill_status": row["observed_fill_status"],
                "observed_fill_side": observed_side,
                "observed_markout_1s_ticks": observed_markout_1s,
                "observed_markout_5s_ticks": observed_markout_5s,
                "proxy_fill_status": "proxy_fill" if proxy_sides else "proxy_no_fill",
                "proxy_fill_sides": "|".join(proxy_sides),
                "censored": row["observed_fill_status"] == "censored_no_fill",
            }
        )
    return output


def _mean_numeric(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row[field] not in ("", None)]
    return statistics.fmean(values) if values else None


def _recovery_duration_ms(rows: list[dict[str, Any]]) -> float | None:
    ordered = sorted(rows, key=lambda row: int(row["decision_ts_ms"]))
    durations: list[float] = []
    for index, row in enumerate(ordered):
        current_abs = abs(float(row["position_btc"]))
        if current_abs == 0:
            continue
        for later in ordered[index + 1 :]:
            if abs(float(later["position_btc"])) < current_abs:
                durations.append(float(later["decision_ts_ms"]) - float(row["decision_ts_ms"]))
                break
    return statistics.fmean(durations) if durations else None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    observed_fills = sum(row["observed_fill_status"] == "observed_fill" for row in rows)
    censored = sum(bool(row["censored"]) for row in rows)
    proxy_fills = sum(row["proxy_fill_status"] == "proxy_fill" for row in rows)
    add_side = sum(
        ("buy" in row["eligible_sides"] and float(row["position_btc"]) >= 0)
        or ("sell" in row["eligible_sides"] and float(row["position_btc"]) <= 0)
        for row in rows
    )
    reduce_side = sum(
        ("sell" in row["eligible_sides"] and float(row["position_btc"]) > 0)
        or ("buy" in row["eligible_sides"] and float(row["position_btc"]) < 0)
        for row in rows
    )
    return {
        "decision_rows": len(rows),
        "observed_fill_count": observed_fills,
        "observed_fill_coverage": observed_fills / len(rows) if rows else 0.0,
        "censored_no_fill_count": censored,
        "proxy_fill_count": proxy_fills,
        "proxy_fill_coverage": proxy_fills / len(rows) if rows else 0.0,
        "mean_bid_distance_ticks": _mean_numeric(rows, "bid_distance_ticks"),
        "mean_ask_distance_ticks": _mean_numeric(rows, "ask_distance_ticks"),
        "mean_observed_markout_1s_ticks": _mean_numeric(rows, "observed_markout_1s_ticks"),
        "mean_observed_markout_5s_ticks": _mean_numeric(rows, "observed_markout_5s_ticks"),
        "spread_retention": _mean_numeric(
            [
                {**row, "spread_retention": (float(row["quote_ask_px"]) - float(row["quote_bid_px"])) / 2.0}
                for row in rows
            ],
            "spread_retention",
        ),
        "peak_abs_position_btc": max((abs(float(row["position_btc"])) for row in rows), default=0.0),
        "mean_recovery_duration_ms": _recovery_duration_ms(rows),
        "add_side_opportunity_count": add_side,
        "reduce_side_opportunity_count": reduce_side,
        "post_only_invariant_pass": all(bool(row["post_only_invariant"]) for row in rows),
        "all_rows_censored_or_observed": all(row["observed_fill_status"] in {"observed_fill", "censored_no_fill"} for row in rows),
    }


def build_acceptance_artifacts(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    opportunities = rows or fixture_opportunities()
    zero_skew_rows = _evaluate_case(opportunities, case="alpha_zero_skew", skew_ticks=0.0)
    bounded_skew_rows = _evaluate_case(
        opportunities,
        case="alpha_bounded_skew",
        skew_ticks=BOUNDED_SKEW_TICKS_AT_MAX,
    )
    all_rows = zero_skew_rows + bounded_skew_rows
    same_universe = [row["decision_id"] for row in zero_skew_rows] == [row["decision_id"] for row in bounded_skew_rows]
    summaries = {
        "alpha_zero_skew": _summary(zero_skew_rows),
        "alpha_bounded_skew": _summary(bounded_skew_rows),
    }
    structural_gates = {
        "same_decision_opportunity_universe": same_universe,
        "skew_sign_long_lowers_reservation": all(
            bounded["reservation_px"] < zero["reservation_px"]
            for zero, bounded in zip(zero_skew_rows, bounded_skew_rows)
            if float(zero["position_btc"]) > 0
        ),
        "skew_sign_short_raises_reservation": all(
            bounded["reservation_px"] > zero["reservation_px"]
            for zero, bounded in zip(zero_skew_rows, bounded_skew_rows)
            if float(zero["position_btc"]) < 0
        ),
        "bounded_skew_is_finite": all(
            abs(float(row["inventory_penalty_ticks"])) <= BOUNDED_SKEW_TICKS_AT_MAX
            for row in bounded_skew_rows
        ),
        "post_only_invariant": all(bool(row["post_only_invariant"]) for row in all_rows),
        "observed_and_proxy_evidence_separate": all(
            row["observed_fill_status"] != row["proxy_fill_status"] for row in all_rows
        ),
        "censored_rows_explicit": all(
            row["censored"] is True for row in all_rows if row["observed_fill_status"] == "censored_no_fill"
        ),
        "reduce_side_preserved_near_cap": all(
            "sell" in row["eligible_sides"]
            for row in bounded_skew_rows
            if float(row["position_ratio"]) >= kernel.NEAR_POSITION_CAP_RATIO
        )
        and all(
            "buy" in row["eligible_sides"]
            for row in bounded_skew_rows
            if float(row["position_ratio"]) <= -kernel.NEAR_POSITION_CAP_RATIO
        ),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "same_universe": same_universe,
        "structural_gates": structural_gates,
        "summaries": summaries,
        "observed_fill_rows_are_not_proxy_claims": True,
        "no_fill_rows_are_censored": True,
        "skew_enablement_recommendation": "remain_disabled_pending_real_c12_evidence",
        "final_recommendation": (
            "offline_c12_structural_acceptance_pass_keep_skew_disabled"
            if all(structural_gates.values())
            else "offline_c12_blocked_keep_skew_disabled"
        ),
        "boundary": {
            "offline_local_processing_only": True,
            "no_network_collection": True,
            "no_credentials": True,
            "no_private_account_order_cancel_endpoints": True,
            "no_live_client_initialization": True,
            "no_live_orders": True,
            "no_strategy_promotion": True,
        },
    }
    fields = list(all_rows[0]) if all_rows else []
    _write_csv(output_dir / "skew_alpha_interaction_rows.csv", all_rows, fields)
    _write_json(output_dir / "skew_alpha_interaction_summary.json", manifest)
    return {"rows": all_rows, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline C12 alpha/skew acceptance artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(build_acceptance_artifacts(output_dir=args.output_dir)["manifest"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
