from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.hyperliquid.skhynix_obi_reversal_track_a0 import CaptureBinding
from examples.hyperliquid.skhynix_obi_reversal_track_a1_a2 import (
    CaptureData,
    extract_decision_features,
    first_passage,
    pre_detection_transition,
)


def test_first_passage_orients_follow_and_fail_by_side() -> None:
    valid = np.ones(5, dtype=bool)
    upward = np.asarray([0.0, 0.5, 0.5, 0.0, 0.0])

    follow = first_passage(upward, valid, 0, side=1, tau_steps=4)
    fail = first_passage(upward, valid, 0, side=-1, tau_steps=4)

    assert follow["event_type"] == "follow"
    assert follow["event_time_ms"] == 200
    assert fail["event_type"] == "fail"
    assert fail["event_time_ms"] == 200


def test_first_passage_censoring_reasons() -> None:
    deltas = np.zeros(5)

    tau = first_passage(deltas, np.ones(5, dtype=bool), 0, 1, tau_steps=2)
    capture = first_passage(deltas, np.ones(5, dtype=bool), 3, 1, tau_steps=2)
    validity = np.ones(5, dtype=bool)
    validity[2] = False
    quality = first_passage(deltas, validity, 0, 1, tau_steps=4)

    assert tau["censor_reason"] == "tau_max"
    assert tau["event_time_ms"] == 200
    assert capture["censor_reason"] == "capture_end"
    assert capture["event_time_ms"] == 100
    assert quality["censor_reason"] == "quality_failure"
    assert quality["event_time_ms"] == 200


def test_first_passage_ignores_path_after_first_hit() -> None:
    first = np.asarray([0.0, 0.5, 0.5, 100.0, -100.0])
    second = np.asarray([0.0, 0.5, 0.5, -100.0, 100.0])
    valid = np.ones(5, dtype=bool)

    first_result = first_passage(first, valid, 0, side=1, tau_steps=4)
    second_result = first_passage(second, valid, 0, side=1, tau_steps=4)

    assert first_result == second_result
    assert first_result["event_type"] == "follow"
    assert first_result["event_time_ms"] == 200


def test_pre_detection_transition_is_separate() -> None:
    deltas = np.asarray([0.0, 0.4, 0.6, 0.5])
    result = pre_detection_transition(
        deltas,
        np.ones(4, dtype=bool),
        cross_index=0,
        detect_index=3,
        side=1,
    )
    assert result["pre_detection_transition"] == "follow"
    assert result["pre_detection_transition_time_ms"] == 200
    assert (
        result["pre_detection_oriented_displacement_at_detection_ticks"]
        == 1.5
    )


def _capture(features: np.ndarray, names: tuple[str, ...]) -> CaptureData:
    binding = CaptureBinding(
        capture_id="fixture",
        research_date="2026-08-01",
        role="historical_method_development",
        start_utc="",
        end_utc="",
        duration_seconds=len(features) / 10,
        raw_path=Path("raw"),
        raw_size_bytes=0,
        raw_sha256="",
        raw_size_verified=True,
        raw_hash_verified=True,
        depth_gap_count=0,
        cache_path=Path("cache"),
        cache_sha256="",
        cache_size_bytes=0,
    )
    return CaptureData(
        binding=binding,
        ts_ns=np.arange(len(features), dtype=np.int64) * 100_000_000,
        valid=np.ones(len(features), dtype=bool),
        feature_names=names,
        base_features=features,
    )


def test_decision_features_ignore_post_decision_midpoint() -> None:
    names = (
        *(f"bid_qty_log_l{level}" for level in range(1, 6)),
        *(f"ask_qty_log_l{level}" for level in range(1, 6)),
        *(f"bid_add_log_l{level}" for level in range(1, 6)),
        *(f"bid_cancel_log_l{level}" for level in range(1, 6)),
        *(f"ask_add_log_l{level}" for level in range(1, 6)),
        *(f"ask_cancel_log_l{level}" for level in range(1, 6)),
        "trade_buy_qty_log",
        "trade_sell_qty_log",
        "spread_ticks",
        "midpoint_delta_ticks",
        "bid_depth_concentration",
        "ask_depth_concentration",
        "source_age_ms_log",
        "no_new_information",
        "depth_update_count_log",
    )
    mapping = {name: index for index, name in enumerate(names)}
    base = np.zeros((60, len(names)), dtype=np.float64)
    for level in range(1, 6):
        base[:, mapping[f"bid_qty_log_l{level}"]] = np.log1p(2.0)
        base[:, mapping[f"ask_qty_log_l{level}"]] = np.log1p(1.0)
    base[:, mapping["spread_ticks"]] = 1.0
    base[:, mapping["bid_depth_concentration"]] = 0.2
    base[:, mapping["ask_depth_concentration"]] = 0.2

    changed = base.copy()
    changed[31:, mapping["midpoint_delta_ticks"]] = 999.0

    before = extract_decision_features(_capture(base, names), 30)
    after = extract_decision_features(_capture(changed, names), 30)

    assert before == after
