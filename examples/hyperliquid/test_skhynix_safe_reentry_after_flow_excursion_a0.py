from __future__ import annotations

from pathlib import Path

import numpy as np

from examples.hyperliquid.skhynix_safe_reentry_after_flow_excursion_a0 import (
    CHECKPOINT_NS,
    Capture,
    classify_a0,
    match_controls,
    run_state_machine,
)


PLUS = np.asarray([4.0, 0.0, 4.0, 0.0, 0.0, 0.0])
MINUS = np.asarray([0.0, 4.0, 0.0, 4.0, 0.0, 0.0])
QUIET = np.zeros(6)


def _capture() -> Capture:
    return Capture(
        capture_id="fixture",
        research_date="2026-08-01",
        role="historical_method_development",
        start_utc="2026-08-01T00:00:00+00:00",
        end_utc="2026-08-01T00:00:04+00:00",
        duration_seconds=4.0,
        session_id="fixture",
        manifest_path=Path("fixture"),
        raw_path=Path("fixture"),
        raw_size_bytes=0,
        raw_sha256="",
        bookticker_count=0,
        depth_count=0,
        trade_count=0,
        connection_epoch_count=1,
        depth_gap_count=0,
    )


def _arrays(
    length: int = 160,
    *,
    spread_ticks: float = 2.0,
) -> dict[str, np.ndarray]:
    z = np.zeros((length, 6), dtype=np.float64)
    x = np.zeros((length, 6), dtype=np.float64)
    return {
        "ts_ns": np.arange(length, dtype=np.int64) * CHECKPOINT_NS,
        "segment_id": np.zeros(length, dtype=np.int64),
        "valid": np.ones(length, dtype=bool),
        "x": x,
        "z": z,
        "local_scale": np.full((length, 6), 2.0),
        "bid_depth_current": np.full(length, 10.0),
        "ask_depth_current": np.full(length, 10.0),
        "obi_current": np.zeros(length),
        "spread_ticks": np.full(length, spread_ticks),
        "activity_count": np.ones(length, dtype=np.int64),
    }


def _set_pressure(
    arrays: dict[str, np.ndarray], indices: range | list[int], direction: int
) -> None:
    pattern = PLUS if direction == 1 else MINUS
    for index in indices:
        arrays["z"][index] = pattern
        arrays["x"][index] = pattern


def _detect(arrays: dict[str, np.ndarray]):
    event_seq = np.arange(len(arrays["ts_ns"]), dtype=np.int64)
    return run_state_machine(
        _capture(),
        arrays,
        event_seq,
        np.ones(6),
    )


def test_full_prequiet_and_persistence_confirm_without_backdating() -> None:
    arrays = _arrays()
    _set_pressure(arrays, range(50, 56), 1)
    result = _detect(arrays)
    assert len(result.episodes) == 1
    episode = result.episodes[0]
    assert episode["candidate_ts_ns"] == 50 * CHECKPOINT_NS
    assert episode["confirmation_ts_ns"] == 55 * CHECKPOINT_NS
    assert episode["confirmation_delay_ms"] == 100.0
    assert episode["qualifying_exposure_ms"] == 100.0
    assert episode["prequiet_checkpoint_count"] == 50
    assert result.anchors[0]["anchor_ts_ns"] == 106 * CHECKPOINT_NS


def test_transient_candidate_is_rejected() -> None:
    arrays = _arrays()
    _set_pressure(arrays, [50], 1)
    result = _detect(arrays)
    statuses = [row["candidate_status"] for row in result.candidates]
    assert "transient_rejected" in statuses
    assert result.episodes == []
    assert result.anchors == []


def test_preconfirmation_direction_switch_rejects_candidate() -> None:
    arrays = _arrays()
    _set_pressure(arrays, [50, 51], 1)
    _set_pressure(arrays, [52], -1)
    result = _detect(arrays)
    statuses = [row["candidate_status"] for row in result.candidates]
    assert "pre_confirmation_direction_switch" in statuses
    assert result.episodes == []


def test_opposite_pressure_after_confirmation_stays_in_one_episode() -> None:
    arrays = _arrays()
    _set_pressure(arrays, range(50, 56), 1)
    _set_pressure(arrays, [56], -1)
    result = _detect(arrays)
    assert len(result.episodes) == 1
    assert result.episodes[0]["direction_switch_count"] == 1
    assert len({row["episode_id"] for row in result.episodes}) == 1


def test_refractory_pressure_reset_keeps_episode_identity() -> None:
    arrays = _arrays(length=190)
    _set_pressure(arrays, range(50, 56), 1)
    _set_pressure(arrays, [80], 1)
    result = _detect(arrays)
    assert len(result.episodes) == 1
    assert result.episodes[0]["refractory_reset_count"] == 1
    assert len(result.anchors) == 1
    assert result.anchors[0]["episode_id"] == result.episodes[0]["episode_id"]
    assert result.anchors[0]["anchor_ts_ns"] == 131 * CHECKPOINT_NS


def test_recovered_without_wide_spread_does_not_wait() -> None:
    arrays = _arrays()
    _set_pressure(arrays, range(50, 56), 1)
    arrays["spread_ticks"][106] = 1.0
    arrays["spread_ticks"][107:] = 2.0
    result = _detect(arrays)
    assert len(result.episodes) == 1
    assert (
        result.episodes[0]["terminal_status"]
        == "recovered_without_wide_spread"
    )
    assert result.anchors == []


def test_classification_calls_dense_excursions_near_continuous() -> None:
    gates = {
        "A0_0_source_closure": True,
        "A0_1_zero_outcome_boundary": True,
        "A0_2_normalization_support": True,
        "A0_3_excursion_support": False,
        "A0_4_novelty_persistence_compression": True,
        "A0_5_safe_reentry_support": True,
        "A0_6_control_common_support": True,
        "A0_7_followup_geometry": True,
    }
    assert (
        classify_a0(
            gates,
            excursion_rate=101,
            median_inter_excursion_ms=3_000,
            violation_counts={},
            anchor_state_invalid=False,
        )
        == "A0_excursion_still_near_continuous"
    )


def test_anchor_without_same_date_controls_is_unmatched() -> None:
    anchor = {
        "episode_id": "episode",
        "capture_id": "fixture",
        "research_date": "2026-08-01",
        "anchor_ts_ns": 1,
        "direction": 1,
        "spread_ticks": 2.0,
        "obi_current": 0.0,
        "bid_depth_current": 10.0,
        "ask_depth_current": 10.0,
        "activity_count": 1,
    }
    matched, unmatched = match_controls([anchor], [])
    assert matched == []
    assert unmatched[0]["reason"] == "no_same_date_control_reference"


def test_controls_on_dates_without_anchors_are_still_binned() -> None:
    control = {
        "control_id": 1,
        "capture_id": "fixture",
        "research_date": "2026-08-02",
        "ts_ns": 1,
        "direction": 1,
        "spread_ticks": 2.0,
        "obi_current": 0.0,
        "bid_depth_current": 10.0,
        "ask_depth_current": 10.0,
        "activity_count": 1,
    }
    matched, unmatched = match_controls([], [control])
    assert matched == []
    assert unmatched == []
    assert control["spread_bin"] == 2
