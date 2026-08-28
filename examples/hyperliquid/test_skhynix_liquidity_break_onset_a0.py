from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from examples.hyperliquid.skhynix_liquidity_break_onset_a0 import (
    BASELINE_MIN_CHECKPOINTS,
    BASELINE_SHIFT_CHECKPOINTS,
    BASELINE_WINDOW_CHECKPOINTS,
    A0Error,
    Capture,
    CaptureCache,
    Detector,
    ReplayEngine,
    ReplaySnapshot,
    _component_pair,
    _orientation_indices,
    _rolling_normalization,
    _write_contracts,
    build_control_candidates,
    classify_a0,
    choose_direction,
    match_controls,
    structural_predicate,
)


def test_structural_predicate_requires_two_channels_and_positive_depletion() -> None:
    assert structural_predicate(
        np.asarray([0.1, 0.2, 0.0]),
        np.asarray([3.1, 3.2, 0.1]),
    )
    assert not structural_predicate(
        np.asarray([-0.1, 0.2, 0.0]),
        np.asarray([3.1, 3.2, 0.1]),
    )
    assert not structural_predicate(
        np.asarray([0.1, 0.2, 0.0]),
        np.asarray([3.1, 2.9, 0.1]),
    )


def test_direction_conflict_uses_frozen_gap() -> None:
    x = np.asarray([0.1, 0.1, 0.1])
    direction, ambiguous = choose_direction(
        x,
        np.asarray([3.0, 3.0, 0.0]),
        x,
        np.asarray([3.1, 3.1, 0.0]),
    )
    assert direction == 0
    assert ambiguous
    direction, ambiguous = choose_direction(
        x,
        np.asarray([4.0, 4.0, 0.0]),
        x,
        np.asarray([3.0, 3.0, 0.0]),
    )
    assert direction == 1
    assert not ambiguous


def test_orientation_indices_are_mirrored() -> None:
    assert _orientation_indices(1) == (0, 2, 4)
    assert _orientation_indices(-1) == (1, 3, 5)


def test_component_pair_classification() -> None:
    assert _component_pair(np.asarray([3.1, 3.2, 0.0])) == "dep_trade"
    assert _component_pair(np.asarray([3.1, 0.0, 3.2])) == "dep_ofi"
    assert _component_pair(np.asarray([0.0, 3.1, 3.2])) == "trade_ofi"
    assert _component_pair(np.asarray([3.1, 3.2, 3.3])) == "dep_trade_ofi"


def test_rolling_normalization_excludes_guard_and_current_value() -> None:
    length = (
        BASELINE_SHIFT_CHECKPOINTS
        + BASELINE_WINDOW_CHECKPOINTS
        + 5
    )
    values = np.arange(length, dtype=float)
    frame = pd.DataFrame({"x": values})
    segments = np.zeros(length, dtype=int)
    center, scale, q25, q75 = _rolling_normalization(frame, segments)
    index = BASELINE_SHIFT_CHECKPOINTS + BASELINE_WINDOW_CHECKPOINTS - 1
    admitted = values[
        index
        - BASELINE_SHIFT_CHECKPOINTS
        - BASELINE_WINDOW_CHECKPOINTS
        + 1 : index - BASELINE_SHIFT_CHECKPOINTS + 1
    ]
    assert len(admitted) == BASELINE_WINDOW_CHECKPOINTS
    assert center[index, 0] == np.median(admitted)
    assert q25[index, 0] == np.quantile(admitted, 0.25)
    assert q75[index, 0] == np.quantile(admitted, 0.75)
    assert np.isclose(
        scale[index, 0],
        (np.quantile(admitted, 0.75) - np.quantile(admitted, 0.25)) / 1.349,
    )
    assert values[index] not in admitted


def test_rolling_normalization_resets_by_segment() -> None:
    length = BASELINE_MIN_CHECKPOINTS + BASELINE_SHIFT_CHECKPOINTS + 10
    frame = pd.DataFrame(
        {"x": np.concatenate((np.ones(length), np.ones(length) * 100))}
    )
    segments = np.concatenate(
        (np.zeros(length, dtype=int), np.ones(length, dtype=int))
    )
    center, _, _, _ = _rolling_normalization(frame, segments)
    assert np.isnan(center[length, 0])
    assert center[-1, 0] == 100


def test_dense_anchor_failure_is_classified_as_near_continuous() -> None:
    gates = {
        "A0_0_source_closure": True,
        "A0_1_zero_outcome_boundary": True,
        "A0_2_anchor_support": False,
        "A0_3_detection_geometry": True,
        "A0_4_component_diversity": True,
        "A0_5_control_common_support": True,
        "A0_6_followup_geometry": True,
    }
    assert (
        classify_a0(
            gates,
            anchor_rate=301,
            precursor_p90_ms=10,
            inter_anchor_p50_ms=500,
            active_occupancy=0.5,
        )
        == "A0_anchor_near_continuous"
    )


def _capture(raw_path: Path) -> Capture:
    return Capture(
        capture_id="fixture",
        research_date="2026-08-01",
        role="historical_method_development",
        start_utc="2026-08-01T00:00:00+00:00",
        end_utc="2026-08-01T00:00:01+00:00",
        duration_seconds=1.0,
        session_id="fixture",
        manifest_path=raw_path,
        raw_path=raw_path,
        raw_size_bytes=raw_path.stat().st_size if raw_path.exists() else 0,
        raw_sha256=(
            hashlib.sha256(raw_path.read_bytes()).hexdigest()
            if raw_path.exists()
            else ""
        ),
        bookticker_count=0,
        depth_count=0,
        trade_count=0,
        connection_epoch_count=1,
        depth_gap_count=0,
    )


def _write_raw(path: Path, messages: list[tuple[int, dict[str, object]]]) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for ts_ns, message in messages:
                handle.write(
                    f"{ts_ns} ".encode()
                    + json.dumps(message, sort_keys=True).encode()
                    + b"\n"
                )


def _snapshot_message() -> dict[str, object]:
    return {
        "lastUpdateId": 100,
        "bids": [[f"{100 - i * 0.01:.2f}", "1.0"] for i in range(5)],
        "asks": [[f"{100.01 + i * 0.01:.2f}", "1.0"] for i in range(5)],
    }


def test_equal_timestamp_events_preserve_file_order(tmp_path: Path) -> None:
    base = 1_800_000_000_000_000_000
    raw_path = tmp_path / "order.gz"
    _write_raw(
        raw_path,
        [
            (base, _snapshot_message()),
            (base + 60_000_000, {"e": "trade", "q": "1.0", "m": False}),
            (base + 60_000_000, {"e": "trade", "q": "2.0", "m": True}),
        ],
    )
    rows: list[ReplaySnapshot] = []
    ReplayEngine(_capture(raw_path)).run(on_event=rows.append)
    assert [row.event_seq for row in rows] == [2, 3]
    assert rows[0].trade_signed == 1.0
    assert rows[1].trade_signed == -1.0


def test_depth_sequence_gap_fails_closed(tmp_path: Path) -> None:
    base = 1_800_000_000_000_000_000
    raw_path = tmp_path / "gap.gz"
    _write_raw(
        raw_path,
        [
            (base, _snapshot_message()),
            (
                base + 60_000_000,
                {
                    "e": "depthUpdate",
                    "U": 101,
                    "u": 101,
                    "pu": 100,
                    "b": [["100.00", "0.5"]],
                    "a": [],
                },
            ),
            (
                base + 70_000_000,
                {
                    "e": "depthUpdate",
                    "U": 102,
                    "u": 102,
                    "pu": 999,
                    "b": [["100.00", "0.4"]],
                    "a": [],
                },
            ),
        ],
    )
    with pytest.raises(A0Error, match="depth_sequence_gap"):
        ReplayEngine(_capture(raw_path)).run()


def _detector(tmp_path: Path) -> Detector:
    final_path = tmp_path / "final.npz"
    np.savez(
        final_path,
        ts_ns=np.asarray([0], dtype=np.int64),
        center=np.zeros((1, 6)),
        local_scale=np.full((1, 6), 2.0),
        applied_scale=np.ones((1, 6)),
        trade_scale=np.ones(1),
        segment_end_ids=np.asarray([0], dtype=np.int64),
        segment_end_ts=np.asarray([1_000_000_000], dtype=np.int64),
    )
    capture = _capture(tmp_path / "unused.gz")
    return Detector(
        capture,
        final_path,
        {"depth_scale_floor": 1.0, "trade_scale_floor": 1.0},
        np.ones(6),
    )


def _event(
    ts_ns: int,
    *,
    dep_up: float = 0.0,
    dep_down: float = 0.0,
    trade_signed: float = 0.0,
) -> ReplaySnapshot:
    return ReplaySnapshot(
        ts_ns=ts_ns,
        event_seq=1,
        segment_id=0,
        valid=True,
        dep_up_num=dep_up,
        dep_down_num=dep_down,
        trade_signed=trade_signed,
        trade_total=abs(trade_signed),
        ofi_up_num=0.0,
        bid_depth_start=1.0,
        ask_depth_start=1.0,
        total_depth_start=2.0,
        bid_depth_current=1.0,
        ask_depth_current=1.0,
        obi_current=0.0,
        spread_ticks=1.0,
        activity_count=1,
    )


def test_detector_active_lock_and_causal_release(tmp_path: Path) -> None:
    detector = _detector(tmp_path)
    detector.on_event(_event(1, dep_up=4.0, trade_signed=4.0))
    detector.on_event(_event(2, dep_up=4.0, trade_signed=4.0))
    detector.on_event(_event(10))
    detector.on_event(_event(100_000_010))
    assert len(detector.anchors) == 1
    assert detector.state == 0
    assert detector.intervals[0]["exit_reason"] == "causal_release"
    assert detector.intervals[0]["end_ts_ns"] == 100_000_010


def test_detector_opposite_onset_switches_at_same_event(tmp_path: Path) -> None:
    detector = _detector(tmp_path)
    detector.on_event(_event(1, dep_up=4.0, trade_signed=4.0))
    detector.on_event(_event(2, dep_down=4.0, trade_signed=-4.0))
    assert [row["direction"] for row in detector.anchors] == [1, -1]
    assert detector.intervals[0]["exit_reason"] == "opposite_onset"
    assert detector.intervals[0]["end_ts_ns"] == 2
    assert detector.state == -1


def test_detector_reset_censors_active_interval(tmp_path: Path) -> None:
    detector = _detector(tmp_path)
    detector.on_event(_event(1, dep_up=4.0, trade_signed=4.0))
    detector.on_reset(0, 25)
    assert detector.state == 0
    assert detector.intervals[0]["exit_reason"] == "reset_or_capture_end"
    assert detector.intervals[0]["censored"] == "true"


def test_control_candidate_is_not_excluded_by_future_anchor(tmp_path: Path) -> None:
    final_path = tmp_path / "control.npz"
    ts = np.asarray([0, 250_000_000, 500_000_000], dtype=np.int64)
    np.savez(
        final_path,
        ts_ns=ts,
        valid=np.ones(3, dtype=bool),
        x=np.zeros((3, 6)),
        z=np.zeros((3, 6)),
        obi_current=np.zeros(3),
        spread_ticks=np.ones(3),
        bid_depth_current=np.ones(3),
        ask_depth_current=np.ones(3),
        activity_count=np.ones(3),
    )
    capture = _capture(tmp_path / "unused.gz")
    cache = CaptureCache(
        capture=capture,
        raw_path=capture.raw_path,
        final_path=final_path,
        segment_end_by_id={0: int(ts[-1])},
        first_ts_ns=int(ts[0]),
        last_ts_ns=int(ts[-1]),
    )
    controls = build_control_candidates(
        [cache],
        [
            {
                "capture_id": capture.capture_id,
                "start_ts_ns": 400_000_000,
                "end_ts_ns": 600_000_000,
            }
        ],
    )
    assert {row["ts_ns"] for row in controls} == {250_000_000}


def test_control_matching_never_reuses_a_control() -> None:
    anchors = [
        {
            "capture_id": f"a{index}",
            "research_date": "2026-08-01",
            "direction": 1,
            "start_ts_ns": index,
            "obi_current": 0.21,
            "spread_ticks": 1.0,
            "total_depth_current": 2.0,
            "activity_count": 1,
        }
        for index in range(2)
    ]
    controls = [
        {
            "control_id": 7,
            "capture_id": "c",
            "research_date": "2026-08-01",
            "direction": 1,
            "ts_ns": 10,
            "obi_current": 0.21,
            "spread_ticks": 1.0,
            "total_depth_current": 2.0,
            "activity_count": 1,
            "time_block": 0,
        }
    ]
    matched, unmatched = match_controls(anchors, controls)
    assert len(matched) == 1
    assert len(unmatched) == 1
    assert len({row["control_id"] for row in matched}) == len(matched)


def test_outcome_access_contract_remains_zero_target(tmp_path: Path) -> None:
    _write_contracts(
        tmp_path,
        {"depth_scale_floor": 1.0, "trade_scale_floor": 1.0},
        np.ones(6),
    )
    ledger = json.loads(
        (tmp_path / "contracts" / "outcome_access_ledger.json").read_text()
    )
    assert ledger["future_midpoint_fields_read"] == []
    assert ledger["future_best_quote_fields_read"] == []
    assert ledger["targets_materialized"] is False
    assert ledger["H0_H1_fitted"] is False
