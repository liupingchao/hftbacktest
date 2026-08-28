from __future__ import annotations

import gzip
import json
import math

import numpy as np

from examples.hyperliquid import skhynix_flow_internal_directional_alpha_a0 as a0


def _step(
    machine: a0.DominanceStateMachine,
    ts_ms: int,
    *,
    q_up: bool = False,
    q_down: bool = False,
    mixed: bool = False,
    release: bool = False,
    active: bool = True,
) -> a0.StateStep:
    return machine.step(
        ts_ms * 1_000_000,
        quality_ok=True,
        active=active,
        q={-1: q_down, 1: q_up},
        q_both=q_up and q_down,
        q_release=release,
        q_mixed=mixed,
    )


def _ready_machine() -> tuple[a0.DominanceStateMachine, int]:
    machine = a0.DominanceStateMachine()
    _step(machine, 0, mixed=True)
    for ts_ms in range(20, 520, 20):
        _step(machine, ts_ms, mixed=True)
    assert machine.state == a0.MIXED_READY
    return machine, 520


def _confirmed_machine() -> tuple[a0.DominanceStateMachine, int]:
    machine, ts_ms = _ready_machine()
    opened = _step(machine, ts_ms, q_up=True)
    assert opened.anchor_type is None
    for offset in (20, 40, 60, 80, 100):
        assert _step(machine, ts_ms + offset, q_up=True).anchor_type is None
    confirmed = _step(machine, ts_ms + 120, q_up=True)
    assert confirmed.anchor_type == "mixed_onset"
    return machine, ts_ms + 120


def test_safe_div_zero_is_unavailable() -> None:
    assert math.isnan(a0._safe_div(1.0, 0.0))
    assert a0._safe_div(1.0, 2.0) == 0.5


def test_rolling_sum_is_segment_local_and_right_aligned() -> None:
    values = np.asarray([1, 2, 3, 10, 20, 30], dtype=float)
    segments = np.asarray([0, 0, 0, 1, 1, 1])
    result = a0._rolling_sum(values, segments, 2)
    np.testing.assert_allclose(
        result,
        np.asarray([np.nan, 3, 5, np.nan, 30, 50]),
        equal_nan=True,
    )


def test_replay_emits_checkpoint_before_same_timestamp_messages(tmp_path) -> None:
    raw_path = tmp_path / "raw.gz"
    snapshot = {
        "lastUpdateId": 100,
        "bids": [[str(100 - index), "10"] for index in range(5)],
        "asks": [[str(101 + index), "10"] for index in range(5)],
    }
    depth = {
        "e": "depthUpdate",
        "U": 101,
        "u": 101,
        "pu": 100,
        "b": [["100", "9"]],
        "a": [],
    }
    trade = {"e": "trade", "q": "2", "m": False}
    unknown = {"e": "bookTicker", "b": "100", "a": "101"}
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        for ts_ns, payload in (
            (1, snapshot),
            (20_000_000, depth),
            (20_000_000, trade),
            (40_000_000, unknown),
        ):
            handle.write(f"{ts_ns} {json.dumps(payload)}\n")
    capture = a0.Capture(
        capture_id="fixture",
        research_date="2026-08-27",
        role="historical_no_refit_replay",
        start_utc="",
        end_utc="",
        duration_seconds=1,
        raw_path=raw_path,
        raw_size_bytes=raw_path.stat().st_size,
        raw_sha256="",
        depth_gap_count=0,
    )
    rows = []
    a0.FlowReplay(capture).run(rows.append)
    assert [row.ts_ns for row in rows] == [20_000_000, 40_000_000]
    assert rows[0].event_seq == 1
    assert rows[0].activity == 0
    assert rows[1].event_seq == 3
    assert rows[1].activity == 2
    assert rows[1].bid_depletion == 1
    assert rows[1].trade_signed == 2


def test_ratio_predicates_preserve_component_agreement() -> None:
    q, both, release, payload = a0.ratio_predicates(
        np.asarray([0.8, 0.7, -0.1]),
        composite_500=0.4,
        available_100=3,
        available_500=3,
    )
    assert q == {-1: False, 1: True}
    assert not both
    assert not release
    assert payload[1] == 2
    assert payload[-1] == 0


def test_candidate_checkpoint_contributes_zero_and_confirms_at_120ms() -> None:
    machine, ts_ms = _ready_machine()
    opened = _step(machine, ts_ms, q_up=True)
    assert machine.state == a0.DOM_CANDIDATE
    assert machine.candidate_exposure_ns == 0
    assert opened.anchor_type is None
    for offset in (20, 40, 60, 80, 100):
        assert _step(machine, ts_ms + offset, q_up=True).anchor_type is None
    confirmed = _step(machine, ts_ms + 120, q_up=True)
    assert confirmed.anchor_type == "mixed_onset"
    assert confirmed.anchor_direction == 1
    assert machine.confirmed_at == (ts_ms + 120) * 1_000_000


def test_transient_candidate_rejects_at_300ms() -> None:
    machine, ts_ms = _ready_machine()
    _step(machine, ts_ms, q_up=True)
    for offset in range(20, 300, 20):
        result = _step(machine, ts_ms + offset)
        assert result.candidate_status is None
    result = _step(machine, ts_ms + 300)
    assert result.candidate_status == "transient_rejected"
    assert machine.state == a0.MIXED_BUILDING


def test_active_support_loss_clears_candidate_counters() -> None:
    machine, ts_ms = _ready_machine()
    _step(machine, ts_ms, q_up=True)
    _step(machine, ts_ms + 20, q_up=True)
    assert machine.candidate_exposure_ns == a0.CHECKPOINT_NS
    result = _step(machine, ts_ms + 40, active=False)
    assert result.candidate_status == "flow_support_lost"
    assert machine.state == a0.INACTIVE
    assert machine.candidate_exposure_ns == 0


def test_preconfirmation_direction_switch_rejects() -> None:
    machine, ts_ms = _ready_machine()
    _step(machine, ts_ms, q_up=True)
    result = _step(machine, ts_ms + 20, q_down=True)
    assert result.candidate_status == "pre_confirmation_direction_switch"
    assert machine.state == a0.MIXED_BUILDING


def test_refractory_prevents_flip_anchor_then_persistent_flip_confirms() -> None:
    machine, confirmed_ms = _confirmed_machine()
    for offset in (20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280):
        result = _step(machine, confirmed_ms + offset, q_down=True)
        assert result.anchor_type is None
        assert machine.state == a0.DOMINANT
    opened = _step(machine, confirmed_ms + 300, q_down=True)
    assert opened.anchor_type is None
    assert machine.state == a0.FLIP_CANDIDATE
    for offset in (320, 340, 360, 380, 400):
        assert _step(machine, confirmed_ms + offset, q_down=True).anchor_type is None
    flipped = _step(machine, confirmed_ms + 420, q_down=True)
    assert flipped.anchor_type == "persistent_flip"
    assert flipped.anchor_direction == -1


def test_flip_reassertion_returns_to_original_direction() -> None:
    machine, confirmed_ms = _confirmed_machine()
    _step(machine, confirmed_ms + 300, q_down=True)
    result = _step(machine, confirmed_ms + 320, q_up=True)
    assert result.candidate_status == "flip_rejected_original_reasserted"
    assert machine.state == a0.DOMINANT
    assert machine.direction == 1


def test_release_requires_continuous_200ms() -> None:
    machine, confirmed_ms = _confirmed_machine()
    opened = _step(machine, confirmed_ms + 20, release=True)
    assert opened.anchor_type is None
    assert machine.release_exposure_ns == 0
    for offset in range(40, 220, 20):
        result = _step(machine, confirmed_ms + offset, release=True)
        assert result.state_closed is None
    result = _step(machine, confirmed_ms + 220, release=True)
    assert result.state_closed == "release_to_mixed"
    assert machine.state == a0.MIXED_BUILDING


def test_release_interruption_returns_to_dominant() -> None:
    machine, confirmed_ms = _confirmed_machine()
    _step(machine, confirmed_ms + 20, release=True)
    _step(machine, confirmed_ms + 40, release=True)
    result = _step(machine, confirmed_ms + 60)
    assert result.candidate_status == "release_interrupted"
    assert machine.state == a0.DOMINANT


def test_canonical_sha_is_key_order_independent() -> None:
    assert a0._canonical_sha({"b": 2, "a": 1}) == a0._canonical_sha(
        {"a": 1, "b": 2}
    )


def test_control_matching_never_reuses_underlying_checkpoint() -> None:
    anchors = [
        {
            "anchor_id": f"a{index}",
            "research_date": "2026-08-27",
            "direction": 1,
            "spread_ticks": 1,
            "obi_bin": 11,
            "bid_depth_quintile": 2,
            "ask_depth_quintile": 2,
            "activity_quintile": 2,
            "return_500_quintile": 2,
            "vol_2000_quintile": 2,
            "anchor_ts_ns": 1_000_000_000_000 + index,
            "anchor_event_seq": index,
            "capture_id": "anchor",
            "segment_id": 0,
            "midpoint": 100.0,
        }
        for index in range(2)
    ]
    controls = [
        {
            "control_id": f"c{direction}",
            "research_date": "2026-08-27",
            "control_direction": direction,
            "spread_ticks": 1,
            "obi_bin": 11,
            "bid_depth_quintile": 2,
            "ask_depth_quintile": 2,
            "activity_quintile": 2,
            "return_500_quintile": 2,
            "vol_2000_quintile": 2,
            "checkpoint_ts_ns": 1_000_000_000_100,
            "checkpoint_event_seq": 5,
            "capture_id": "control",
            "segment_id": 0,
            "midpoint": 100.0,
        }
        for direction in (-1, 1)
    ]
    matched, unmatched = a0.match_controls(anchors, controls)
    assert len(matched) == 1
    assert len(unmatched) == 1
