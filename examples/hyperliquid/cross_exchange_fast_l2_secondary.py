#!/usr/bin/env python3
"""Build the frozen Hyperliquid fast-L2 secondary liquidity family."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
from numba import njit

try:
    from cross_exchange_three_session_commonality import (
        BLOCK_LENGTH_MS,
        HORIZONS_MS,
        MAX_ABS_OUTCOME_BPS,
        MAX_ABS_STANDARDIZED_VALUE,
        PRIMARY_BOOTSTRAP_DRAWS,
        RUN_SEED,
        SESSION_SPECS,
        CommonalityError,
        _design_matrix,
        _file_record,
        _load_masks,
        _ols_from_sufficient,
        _read_json,
        _refresh_commonality_files,
        _write_csv,
        add_horizon_outcomes,
        benjamini_hochberg,
        build_segment_bbo,
        canonical_hash,
        unbiased_index,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid.cross_exchange_three_session_commonality import (
        BLOCK_LENGTH_MS,
        HORIZONS_MS,
        MAX_ABS_OUTCOME_BPS,
        MAX_ABS_STANDARDIZED_VALUE,
        PRIMARY_BOOTSTRAP_DRAWS,
        RUN_SEED,
        SESSION_SPECS,
        CommonalityError,
        _design_matrix,
        _file_record,
        _load_masks,
        _ols_from_sufficient,
        _read_json,
        _refresh_commonality_files,
        _write_csv,
        add_horizon_outcomes,
        benjamini_hochberg,
        build_segment_bbo,
        canonical_hash,
        unbiased_index,
    )


SCHEMA_VERSION = "hyperliquid_fast_l2_secondary_family_v1"
PRICE_SCALE = 100_000_000
FAST_L2_MAX_AGE_NS = 2_000_000_000
MIN_REPLENISHMENT_AT_RISK = 100

CAMPAIGN_DIRS = {
    "jul30": "local_live_analysis/cross_exchange_collection_campaign_0730T011_skhynix_4h_8x30m",
    "aug03": "local_live_analysis/cross_exchange_collection_campaign_0802T001_skhynix_5h_10x30m",
    "aug04": "local_live_analysis/cross_exchange_collection_campaign_0804T001_skhynix_2h_continuous",
}


def _price_units(values: np.ndarray) -> np.ndarray:
    return np.rint(values.astype(np.float64) * PRICE_SCALE).astype(np.int64)


def read_fast_market_raw(path: Path) -> dict[str, np.ndarray]:
    snapshot_ts: list[int] = []
    bid_px: list[list[int]] = []
    bid_qty: list[list[float]] = []
    ask_px: list[list[int]] = []
    ask_qty: list[list[float]] = []
    trade_ts: list[int] = []
    trade_px: list[int] = []
    trade_qty: list[float] = []
    trade_side: list[int] = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            try:
                local_ts_text, payload_text = line.split(" ", 1)
                local_ts = int(local_ts_text)
                payload = json.loads(payload_text)
            except (ValueError, json.JSONDecodeError) as exc:
                raise CommonalityError(f"{path}:{line_number}: invalid raw row") from exc
            channel = payload.get("channel")
            if channel == "l2Book":
                data = payload.get("data", {})
                levels = data.get("levels", [])
                if len(levels) != 2 or len(levels[0]) != 5 or len(levels[1]) != 5:
                    raise CommonalityError(
                        f"{path}:{line_number}: expected fast-L2 5x5 snapshot"
                    )
                snapshot_ts.append(local_ts)
                bid_px.append(
                    [int(round(float(level["px"]) * PRICE_SCALE)) for level in levels[0]]
                )
                bid_qty.append([float(level["sz"]) for level in levels[0]])
                ask_px.append(
                    [int(round(float(level["px"]) * PRICE_SCALE)) for level in levels[1]]
                )
                ask_qty.append([float(level["sz"]) for level in levels[1]])
            elif channel == "trades":
                for trade in payload.get("data", []):
                    side = trade.get("side")
                    if side not in {"A", "B"}:
                        raise CommonalityError(
                            f"{path}:{line_number}: missing aggressor side"
                        )
                    trade_ts.append(local_ts)
                    trade_px.append(
                        int(round(float(trade["px"]) * PRICE_SCALE))
                    )
                    trade_qty.append(float(trade["sz"]))
                    trade_side.append(1 if side == "B" else -1)
    if not snapshot_ts:
        raise CommonalityError(f"{path}: no fast-L2 snapshots")
    snapshot_order = np.argsort(np.asarray(snapshot_ts), kind="stable")
    trade_order = np.argsort(np.asarray(trade_ts), kind="stable")
    return {
        "snapshot_ts": np.asarray(snapshot_ts, dtype=np.int64)[snapshot_order],
        "bid_px": np.asarray(bid_px, dtype=np.int64)[snapshot_order],
        "bid_qty": np.asarray(bid_qty, dtype=np.float64)[snapshot_order],
        "ask_px": np.asarray(ask_px, dtype=np.int64)[snapshot_order],
        "ask_qty": np.asarray(ask_qty, dtype=np.float64)[snapshot_order],
        "trade_ts": np.asarray(trade_ts, dtype=np.int64)[trade_order],
        "trade_px": np.asarray(trade_px, dtype=np.int64)[trade_order],
        "trade_qty": np.asarray(trade_qty, dtype=np.float64)[trade_order],
        "trade_side": np.asarray(trade_side, dtype=np.int8)[trade_order],
    }


@njit
def _observed_anchor_qty(
    anchor_px: int,
    prices: np.ndarray,
    quantities: np.ndarray,
    is_ask: bool,
) -> tuple[float, bool]:
    best = prices[0]
    worst = prices[-1]
    if is_ask and best > anchor_px:
        return 0.0, True
    if not is_ask and best < anchor_px:
        return 0.0, True
    for index in range(prices.shape[0]):
        if prices[index] == anchor_px:
            return quantities[index], True
    if is_ask and anchor_px > worst:
        return math.nan, False
    if not is_ask and anchor_px < worst:
        return math.nan, False
    return 0.0, True


@njit
def _depth_and_replenishment(
    decision_ts: np.ndarray,
    anchor_px: np.ndarray,
    snapshot_ts: np.ndarray,
    prices: np.ndarray,
    quantities: np.ndarray,
    horizon_ns: int,
    is_ask: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = decision_ts.shape[0]
    depth = np.full(size, np.nan)
    replenishment = np.full(size, np.nan)
    anchor_eligible = np.zeros(size, dtype=np.bool_)
    depth_eligible = np.zeros(size, dtype=np.bool_)
    replenishment_eligible = np.zeros(size, dtype=np.bool_)
    for row in range(size):
        start = np.searchsorted(snapshot_ts, decision_ts[row], side="right") - 1
        if start < 0 or decision_ts[row] - snapshot_ts[start] > FAST_L2_MAX_AGE_NS:
            continue
        initial_qty, observed = _observed_anchor_qty(
            anchor_px[row], prices[start], quantities[start], is_ask
        )
        if not observed or not math.isfinite(initial_qty) or initial_qty <= 0:
            continue
        anchor_eligible[row] = True
        target_ts = decision_ts[row] + horizon_ns
        end = np.searchsorted(snapshot_ts, target_ts, side="right") - 1
        if end < start or target_ts - snapshot_ts[end] > FAST_L2_MAX_AGE_NS:
            continue
        target_qty, target_observed = _observed_anchor_qty(
            anchor_px[row], prices[end], quantities[end], is_ask
        )
        if target_observed:
            depth[row] = max(0.0, 1.0 - target_qty / initial_qty)
            depth_eligible[row] = True
        triggered = False
        recovered = False
        path_observed = True
        for snapshot_index in range(start + 1, end + 1):
            current_qty, current_observed = _observed_anchor_qty(
                anchor_px[row],
                prices[snapshot_index],
                quantities[snapshot_index],
                is_ask,
            )
            if not current_observed:
                path_observed = False
                break
            if not triggered and current_qty <= 0.50 * initial_qty:
                triggered = True
            elif triggered and current_qty >= 0.80 * initial_qty:
                recovered = True
        if triggered and path_observed:
            replenishment[row] = 0.0 if recovered else 1.0
            replenishment_eligible[row] = True
    return (
        depth,
        replenishment,
        anchor_eligible,
        depth_eligible,
        replenishment_eligible,
    )


@njit
def _forward_window_extreme(
    query_ts: np.ndarray,
    event_ts: np.ndarray,
    event_values: np.ndarray,
    horizon_ns: int,
    want_maximum: bool,
) -> np.ndarray:
    output = np.full(query_ts.shape[0], np.nan)
    if event_ts.shape[0] == 0:
        return output
    deque_indices = np.empty(event_ts.shape[0], dtype=np.int64)
    head = 0
    tail = 0
    right = 0
    for query_index in range(query_ts.shape[0]):
        target = query_ts[query_index] + horizon_ns
        while right < event_ts.shape[0] and event_ts[right] <= target:
            while tail > head:
                prior_value = event_values[deque_indices[tail - 1]]
                if want_maximum:
                    if prior_value > event_values[right]:
                        break
                elif prior_value < event_values[right]:
                    break
                tail -= 1
            deque_indices[tail] = right
            tail += 1
            right += 1
        while head < tail and event_ts[deque_indices[head]] <= query_ts[query_index]:
            head += 1
        if head < tail:
            output[query_index] = event_values[deque_indices[head]]
    return output


def attach_secondary_labels(
    frame: pl.DataFrame,
    fast: dict[str, np.ndarray],
) -> pl.DataFrame:
    decision_ts = frame["decision_ts_ns"].to_numpy()
    ask_anchor = _price_units(frame["hyperliquid_ask_q"].to_numpy())
    bid_anchor = _price_units(frame["hyperliquid_bid_q"].to_numpy())
    buyer_mask = fast["trade_side"] == 1
    seller_mask = fast["trade_side"] == -1
    buyer_ts = fast["trade_ts"][buyer_mask]
    buyer_px = fast["trade_px"][buyer_mask]
    seller_ts = fast["trade_ts"][seller_mask]
    seller_px = fast["trade_px"][seller_mask]
    columns: list[pl.Series] = []
    for horizon in HORIZONS_MS:
        horizon_ns = horizon * 1_000_000
        max_buyer_px = _forward_window_extreme(
            decision_ts, buyer_ts, buyer_px, horizon_ns, True
        )
        min_seller_px = _forward_window_extreme(
            decision_ts, seller_ts, seller_px, horizon_ns, False
        )
        for direction, anchor, prices, quantities, is_ask in (
            (
                "d_bh",
                ask_anchor,
                fast["ask_px"],
                fast["ask_qty"],
                True,
            ),
            (
                "d_hb",
                bid_anchor,
                fast["bid_px"],
                fast["bid_qty"],
                False,
            ),
        ):
            (
                depth,
                replenishment,
                anchor_eligible,
                depth_eligible,
                replenishment_eligible,
            ) = _depth_and_replenishment(
                decision_ts,
                anchor,
                fast["snapshot_ts"],
                prices,
                quantities,
                horizon_ns,
                is_ask,
            )
            if direction == "d_bh":
                trade_arrival = (
                    np.isfinite(max_buyer_px) & (max_buyer_px >= anchor)
                ).astype(float)
            else:
                trade_arrival = (
                    np.isfinite(min_seller_px) & (min_seller_px <= anchor)
                ).astype(float)
            trade_arrival[~anchor_eligible] = np.nan
            columns.extend(
                [
                    pl.Series(
                        f"{direction}_h{horizon}_depth_depletion_fraction",
                        depth,
                    ),
                    pl.Series(
                        f"{direction}_h{horizon}_depth_depletion_eligible",
                        depth_eligible,
                    ),
                    pl.Series(
                        f"{direction}_h{horizon}_trade_arrival", trade_arrival
                    ),
                    pl.Series(
                        f"{direction}_h{horizon}_trade_arrival_eligible",
                        anchor_eligible,
                    ),
                    pl.Series(
                        f"{direction}_h{horizon}_replenishment_failure",
                        replenishment,
                    ),
                    pl.Series(
                        f"{direction}_h{horizon}_replenishment_failure_eligible",
                        replenishment_eligible,
                    ),
                ]
            )
    return frame.with_columns(columns)


def _secondary_fit_key(direction: str, horizon: int, outcome: str) -> str:
    return canonical_hash(
        ["bbo-secondary-fit-v1", direction, str(horizon), outcome]
    )


def _secondary_hypothesis_key(fit_key: str, predictor: str) -> str:
    return canonical_hash(["bbo-secondary-hypothesis-v1", fit_key, predictor])


def _secondary_eligible_column(
    direction: str, horizon: int, outcome: str
) -> str:
    eligibility_name = (
        "depth_depletion" if outcome == "depth_depletion_fraction" else outcome
    )
    return f"{direction}_h{horizon}_{eligibility_name}_eligible"


def build_secondary_block_statistics(
    session_id: str,
    frames: Sequence[pl.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    segment_ids = sorted({str(frame["segment_id"][0]) for frame in frames})
    block_records: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for direction in ("d_bh", "d_hb"):
        for horizon in HORIZONS_MS:
            for outcome in (
                "depth_depletion_fraction",
                "trade_arrival",
                "replenishment_failure",
            ):
                outcome_column = f"{direction}_h{horizon}_{outcome}"
                eligible_column = _secondary_eligible_column(
                    direction, horizon, outcome
                )
                fit_key = _secondary_fit_key(direction, horizon, outcome)
                total_xtx = None
                total_xty = None
                total_unadjusted_xtx = None
                total_unadjusted_xty = None
                total_rows = 0
                eligible_denominator = 0
                for frame in frames:
                    eligible_denominator += frame.filter(
                        pl.col("quality_eligible")
                    ).height
                    selected = frame.filter(
                        pl.col("quality_eligible")
                        & pl.col(eligible_column)
                        & pl.col(outcome_column).is_finite()
                        & pl.col(f"{direction}_level_z").is_finite()
                        & pl.col(f"{direction}_change_z_100ms").is_finite()
                        & (
                            pl.col(f"{direction}_level_z").abs()
                            <= MAX_ABS_STANDARDIZED_VALUE
                        )
                        & (
                            pl.col(f"{direction}_change_z_100ms").abs()
                            <= MAX_ABS_STANDARDIZED_VALUE
                        )
                        & pl.col("basis_residual_bps").is_finite()
                        & (
                            pl.col("basis_residual_bps").abs()
                            <= MAX_ABS_STANDARDIZED_VALUE
                        )
                        & pl.col("binance_volatility_60s_bps").is_finite()
                    )
                    if not selected.height:
                        continue
                    x = _design_matrix(
                        selected, direction, segment_ids, adjusted=True
                    )
                    unadjusted_x = _design_matrix(
                        selected, direction, segment_ids, adjusted=False
                    )
                    y = selected[outcome_column].to_numpy()
                    finite = (
                        np.isfinite(x).all(axis=1)
                        & np.isfinite(unadjusted_x).all(axis=1)
                        & np.isfinite(y)
                        & (np.max(np.abs(x), axis=1) <= MAX_ABS_STANDARDIZED_VALUE)
                        & (
                            np.max(np.abs(unadjusted_x), axis=1)
                            <= MAX_ABS_STANDARDIZED_VALUE
                        )
                        & (np.abs(y) <= MAX_ABS_OUTCOME_BPS)
                    )
                    x = x[finite]
                    unadjusted_x = unadjusted_x[finite]
                    y = y[finite]
                    ts = selected["decision_ts_ns"].to_numpy()[finite]
                    if not len(y):
                        continue
                    segment_id = str(selected["segment_id"][0])
                    eligible_start = int(
                        frame.filter(pl.col("quality_eligible"))[
                            "decision_ts_ns"
                        ].min()
                    )
                    block_ids = (
                        (ts - eligible_start)
                        // (BLOCK_LENGTH_MS * 1_000_000)
                    ).astype(np.int64)
                    for block_id in np.unique(block_ids):
                        mask = block_ids == block_id
                        bx = x[mask]
                        ubx = unadjusted_x[mask]
                        by = y[mask]
                        with np.errstate(all="ignore"):
                            xtx = bx.T @ bx
                            xty = bx.T @ by
                            unadjusted_xtx = ubx.T @ ubx
                            unadjusted_xty = ubx.T @ by
                        if (
                            not np.isfinite(xtx).all()
                            or not np.isfinite(xty).all()
                            or not np.isfinite(unadjusted_xtx).all()
                            or not np.isfinite(unadjusted_xty).all()
                        ):
                            raise CommonalityError(
                                f"{session_id}/{segment_id}/{fit_key}/"
                                f"{block_id}: non-finite sufficient statistics"
                            )
                        block_records.append(
                            {
                                "session_id": session_id,
                                "segment_id": segment_id,
                                "direction": direction,
                                "horizon_ms": horizon,
                                "outcome": outcome,
                                "fit_key": fit_key,
                                "primary_fit_key": fit_key,
                                "block_id": int(block_id),
                                "row_count": int(mask.sum()),
                                "xtx": xtx,
                                "xty": xty,
                                "unadjusted_xtx": unadjusted_xtx,
                                "unadjusted_xty": unadjusted_xty,
                            }
                        )
                        total_xtx = (
                            xtx.copy() if total_xtx is None else total_xtx + xtx
                        )
                        total_xty = (
                            xty.copy() if total_xty is None else total_xty + xty
                        )
                        total_unadjusted_xtx = (
                            unadjusted_xtx.copy()
                            if total_unadjusted_xtx is None
                            else total_unadjusted_xtx + unadjusted_xtx
                        )
                        total_unadjusted_xty = (
                            unadjusted_xty.copy()
                            if total_unadjusted_xty is None
                            else total_unadjusted_xty + unadjusted_xty
                        )
                        total_rows += int(mask.sum())
                coverage = (
                    total_rows / eligible_denominator if eligible_denominator else 0.0
                )
                quality_fail = (
                    total_rows < MIN_REPLENISHMENT_AT_RISK
                    if outcome == "replenishment_failure"
                    else total_rows == 0
                )
                if total_xtx is None or quality_fail:
                    adjusted_beta = np.full(2, np.nan)
                    unadjusted_beta = np.full(2, np.nan)
                    fit_ok = False
                else:
                    adjusted_full, adjusted_ok = _ols_from_sufficient(
                        total_xtx, total_xty
                    )
                    unadjusted_full, unadjusted_ok = _ols_from_sufficient(
                        total_unadjusted_xtx, total_unadjusted_xty
                    )
                    fit_ok = adjusted_ok and unadjusted_ok
                    adjusted_beta = (
                        adjusted_full[:2] if fit_ok else np.full(2, np.nan)
                    )
                    unadjusted_beta = (
                        unadjusted_full[:2] if fit_ok else np.full(2, np.nan)
                    )
                quality_rows.append(
                    {
                        "session_id": session_id,
                        "direction": direction,
                        "horizon_ms": horizon,
                        "outcome": outcome,
                        "eligible_denominator": eligible_denominator,
                        "observed_or_at_risk_count": total_rows,
                        "coverage": coverage,
                        "quality_fail": quality_fail,
                    }
                )
                for predictor_index, predictor in enumerate(("level", "change")):
                    point_rows.append(
                        {
                            "session_id": session_id,
                            "direction": direction,
                            "horizon_ms": horizon,
                            "outcome": outcome,
                            "fit_key": fit_key,
                            "hypothesis_key": _secondary_hypothesis_key(
                                fit_key, predictor
                            ),
                            "predictor": predictor,
                            "beta": adjusted_beta[predictor_index],
                            "unadjusted_beta": unadjusted_beta[predictor_index],
                            "row_count": total_rows,
                            "coverage": coverage,
                            "quality_fail": quality_fail,
                            "fit_ok": fit_ok,
                        }
                    )
    return block_records, point_rows, quality_rows


def bootstrap_secondary(
    block_records: Sequence[dict[str, Any]],
    point_rows: list[dict[str, Any]],
    *,
    draws: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in block_records:
        grouped[(record["session_id"], record["fit_key"])][
            record["segment_id"]
        ].append(record)
    point_by_fit = defaultdict(dict)
    for row in point_rows:
        point_by_fit[(row["session_id"], row["fit_key"])][row["predictor"]] = row
    output: list[dict[str, Any]] = []
    for (session_id, fit_key), by_segment in grouped.items():
        if any(len(blocks) < 2 for blocks in by_segment.values()):
            continue
        beta_draws = np.full((draws, 2), np.nan)
        for draw_id in range(draws):
            xtx = None
            xty = None
            for segment_id in sorted(by_segment):
                blocks = sorted(
                    by_segment[segment_id], key=lambda row: row["block_id"]
                )
                for position in range(len(blocks)):
                    index = unbiased_index(
                        [
                            "bootstrap-v1",
                            RUN_SEED,
                            session_id,
                            segment_id,
                            str(BLOCK_LENGTH_MS),
                            str(draw_id),
                            str(position),
                        ],
                        len(blocks),
                    )
                    selected = blocks[index]
                    xtx = (
                        selected["xtx"].copy()
                        if xtx is None
                        else xtx + selected["xtx"]
                    )
                    xty = (
                        selected["xty"].copy()
                        if xty is None
                        else xty + selected["xty"]
                    )
            beta, ok = _ols_from_sufficient(xtx, xty)
            if ok:
                beta_draws[draw_id] = beta[:2]
        for predictor_index, predictor in enumerate(("level", "change")):
            point = point_by_fit[(session_id, fit_key)][predictor]
            finite = np.sort(
                beta_draws[:, predictor_index][
                    np.isfinite(beta_draws[:, predictor_index])
                ]
            )
            if len(finite) != draws:
                lower = upper = sign_stability = p_value = math.nan
            else:
                lower = float(finite[math.ceil(0.025 * draws) - 1])
                upper = float(finite[math.ceil(0.975 * draws) - 1])
                sign_stability = float(np.mean(finite > 0))
                p_value = float((1 + np.sum(finite <= 0)) / (1 + draws))
            output.append(
                {
                    **point,
                    "bootstrap_lower": lower,
                    "bootstrap_upper": upper,
                    "bootstrap_sign_stability": sign_stability,
                    "bootstrap_one_sided_p": (
                        1.0 if point["quality_fail"] else p_value
                    ),
                    "bootstrap_draws": draws,
                }
            )
    return output


def build_secondary_family(
    repo_root: Path,
    output_root: Path,
    *,
    bootstrap_draws: int,
) -> dict[str, Any]:
    all_rows: list[dict[str, Any]] = []
    all_quality: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    for session_id, spec in SESSION_SPECS.items():
        research_dir = repo_root / spec["research_dir"]
        campaign_dir = repo_root / CAMPAIGN_DIRS[session_id]
        epochs, masks = _load_masks(research_dir)
        frames: list[pl.DataFrame] = []
        for segment_id in sorted(epochs):
            frame, _ = build_segment_bbo(
                session_id=session_id,
                segment_id=segment_id,
                research_dir=research_dir,
                epoch=epochs[segment_id],
                masks=masks.get(segment_id, []),
            )
            raw_path = (
                campaign_dir
                / "segments"
                / segment_id
                / "skhynix/sample/hyperliquid_public_sample/raw.gz"
            )
            fast = read_fast_market_raw(raw_path)
            frames.append(attach_secondary_labels(frame, fast))
            input_records.append(_file_record(raw_path, repo_root))
        blocks, points, quality = build_secondary_block_statistics(
            session_id, frames
        )
        rows = bootstrap_secondary(blocks, points, draws=bootstrap_draws)
        benjamini_hochberg(rows, "bootstrap_one_sided_p", "bootstrap_q")
        all_rows.extend(rows)
        all_quality.extend(quality)
        del frames
    expected_rows = 3 * 2 * len(HORIZONS_MS) * 3 * 2
    if len(all_rows) != expected_rows:
        raise CommonalityError(
            f"secondary family expected {expected_rows} rows, found {len(all_rows)}"
        )
    output_path = output_root / "mechanism/bbo_dislocation_secondary_liquidity.csv.gz"
    quality_path = output_root / "mechanism/bbo_secondary_liquidity_quality.csv"
    _write_csv(output_path, all_rows, list(all_rows[0]))
    _write_csv(quality_path, all_quality, list(all_quality[0]))
    quality_failures = sum(bool(row["quality_fail"]) for row in all_quality)
    fit_failures = sum(not bool(row["fit_ok"]) for row in all_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "passes": quality_failures == 0 and fit_failures == 0,
        "row_count": len(all_rows),
        "quality_row_count": len(all_quality),
        "quality_failure_count": quality_failures,
        "fit_failure_count": fit_failures,
        "bootstrap_draws": bootstrap_draws,
        "input_count": len(input_records),
        "inputs": input_records,
        "secondary_output": _file_record(output_path, output_root),
        "quality_output": _file_record(quality_path, output_root),
        "contract": {
            "fast_l2_shape": "5x5",
            "fast_l2_max_age_ms": FAST_L2_MAX_AGE_NS / 1_000_000,
            "anchor_price_observability": "zero_only_when_inside_observed_depth_or_crossed",
            "trade_side_required": True,
            "replenishment_trigger_fraction": 0.50,
            "replenishment_recovery_fraction": 0.80,
            "minimum_replenishment_at_risk_per_session": MIN_REPLENISHMENT_AT_RISK,
            "depth_and_trade_minimum_coverage": None,
            "depth_and_trade_coverage_is_published": True,
            "separate_bh_family": True,
            "cannot_promote_primary_tier": True,
        },
    }
    manifest_path = (
        output_root / "mechanism/bbo_secondary_liquidity_manifest.json"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    commonality_path = output_root / "commonality_manifest.json"
    commonality = _read_json(commonality_path)
    commonality["secondary_fast_l2_family_complete"] = manifest["passes"]
    commonality["counts"]["secondary_liquidity_hypothesis_rows"] = len(all_rows)
    commonality_path.write_text(
        json.dumps(commonality, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_commonality_files(output_root)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output-dir",
        default="local_live_analysis/skhynix_three_session_commonality_0804T008",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=PRIMARY_BOOTSTRAP_DRAWS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output_root = (repo_root / args.output_dir).resolve()
    try:
        manifest = build_secondary_family(
            repo_root,
            output_root,
            bootstrap_draws=args.bootstrap_draws,
        )
    except (
        CommonalityError,
        OSError,
        ValueError,
        KeyError,
        pl.exceptions.PolarsError,
    ) as exc:
        print(json.dumps({"passes": False, "error": str(exc)}, indent=2))
        return 4
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["passes"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
