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
        "U": 100,
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
    assert a0.FlowReplay(capture).initial_bridge_failure_count == 0


def test_initial_bridge_accepts_snapshot_id_equal_to_final_update(tmp_path) -> None:
    raw_path = tmp_path / "bridge_equal.gz"
    snapshot = {
        "lastUpdateId": 100,
        "bids": [[str(100 - index), "10"] for index in range(5)],
        "asks": [[str(101 + index), "10"] for index in range(5)],
    }
    depth = {
        "e": "depthUpdate",
        "U": 99,
        "u": 100,
        "pu": 98,
        "b": [["100", "8"]],
        "a": [],
    }
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        handle.write(f"1 {json.dumps(snapshot)}\n")
        handle.write(f"10000000 {json.dumps(depth)}\n")
        handle.write(f"20000000 {json.dumps({'e': 'bookTicker'})}\n")
    capture = a0.Capture(
        capture_id="bridge_equal",
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
    engine = a0.FlowReplay(capture)
    engine.run(rows.append)
    assert engine.initial_bridge_failure_count == 0
    assert rows[0].bid_depletion == 2


def test_type7_quantile_matches_linear_interpolation() -> None:
    assert a0._type7_quantile([0, 10, 20, 30], 0.60) == 18


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


def _capture(raw_path, capture_id: str = "fixture", role: str = "historical_no_refit_replay"):
    return a0.Capture(
        capture_id=capture_id,
        research_date="2026-08-27",
        role=role,
        start_utc="",
        end_utc="",
        duration_seconds=1,
        raw_path=raw_path,
        raw_size_bytes=raw_path.stat().st_size if raw_path.exists() else 0,
        raw_sha256="",
        depth_gap_count=0,
    )


def _initialized_replay(tmp_path, name: str) -> a0.FlowReplay:
    raw_path = tmp_path / f"{name}.gz"
    raw_path.touch()
    engine = a0.FlowReplay(_capture(raw_path, name))
    engine.bid_book.reset([[str(100 - index), "10"] for index in range(5)])
    engine.ask_book.reset([[str(101 + index), "10"] for index in range(5)])
    engine.initialized = True
    engine.sequence_ready = True
    engine.snapshot_id = 100
    engine.next_checkpoint = 20_000_000
    engine.segment_id = 0
    return engine


def test_depth_rank_fallback_and_mirrored_atomic_signs(tmp_path) -> None:
    ask_engine = _initialized_replay(tmp_path, "ask")
    assert ask_engine._apply_depth(
        1,
        {
            "U": 100,
            "u": 100,
            "pu": 0,
            "b": [],
            "a": [["100.5", "3"]],
        },
    )
    assert ask_engine.ofi == -3
    assert ask_engine.ofi_abs == 3
    assert ask_engine._apply_depth(
        2,
        {
            "U": 101,
            "u": 101,
            "pu": 100,
            "b": [],
            "a": [["100.5", "1"]],
        },
    )
    assert ask_engine.ask_depletion == 2
    assert ask_engine.ofi == -1
    assert ask_engine.ofi_abs == 5

    bid_engine = _initialized_replay(tmp_path, "bid")
    assert bid_engine._apply_depth(
        1,
        {
            "U": 100,
            "u": 100,
            "pu": 0,
            "b": [["100.5", "3"]],
            "a": [],
        },
    )
    assert bid_engine.ofi == 3
    assert bid_engine.ofi_abs == 3
    bid_engine._apply_trade(2, {"q": "2", "m": False})
    bid_engine._apply_trade(3, {"q": "5", "m": True})
    assert bid_engine.trade_signed == -3
    assert bid_engine.trade_total == 7


def test_non_overlapping_bins_are_additive_and_snapshot_is_neutral(tmp_path) -> None:
    raw_path = tmp_path / "bins.gz"
    snapshot = {
        "lastUpdateId": 100,
        "bids": [[str(100 - index), "10"] for index in range(5)],
        "asks": [[str(101 + index), "10"] for index in range(5)],
    }
    with gzip.open(raw_path, "wt", encoding="utf-8") as handle:
        for ts_ns, payload in (
            (0, snapshot),
            (1_000_000, {"e": "trade", "q": "1", "m": False}),
            (19_000_000, {"e": "trade", "q": "2", "m": False}),
            (20_000_000, {"e": "trade", "q": "3", "m": False}),
            (39_000_000, {"e": "trade", "q": "4", "m": False}),
            (40_000_000, {"e": "bookTicker"}),
        ):
            handle.write(f"{ts_ns} {json.dumps(payload)}\n")
    rows = []
    engine = a0.FlowReplay(_capture(raw_path, "bins"))
    engine.run(rows.append)
    assert [row.activity for row in rows] == [2, 2]
    assert [row.trade_signed for row in rows] == [3, 7]
    assert sum(row.trade_signed for row in rows) == 10
    assert engine.non_admitted_contributions == 0


def test_non_admitted_contribution_audit_is_fault_sensitive(tmp_path) -> None:
    engine = _initialized_replay(tmp_path, "audit")
    before = engine._bin_totals()
    engine.trade_total = 1
    engine._audit_message_contribution(before, admitted=False)
    assert engine.non_admitted_contributions == 1


def test_ratio_runtime_audit_detects_floor_and_bound_faults() -> None:
    ratios = np.asarray([0.5, 2.0, np.nan, 0.1])
    denominators = np.asarray([1.0, 1.0, 2.0, 0.0])
    assert a0._ratio_audit_counts(ratios, denominators) == (1, 1, 1, 1)


def _write_cache(path, *, activity: int) -> None:
    count = 125
    payload = {
        "ts_ns": np.arange(count, dtype=np.int64) * a0.CHECKPOINT_NS,
        "event_seq": np.arange(count, dtype=np.int64),
        "segment_id": np.zeros(count, dtype=np.int32),
        "valid_book": np.ones(count, dtype=np.int8),
        "ready": np.ones(count, dtype=np.int8),
        "trade_signed": np.ones(count),
        "trade_total": np.ones(count),
        "ask_depletion": np.ones(count),
        "bid_depletion": np.zeros(count),
        "ofi": np.ones(count),
        "ofi_abs": np.ones(count),
        "activity": np.full(count, activity),
        "bid_depth": np.full(count, 10.0),
        "ask_depth": np.full(count, 10.0),
        "obi": np.zeros(count),
        "spread_ticks": np.ones(count),
        "midpoint": np.full(count, 100.0),
        "tick_size": np.full(count, 1.0),
    }
    np.savez(path, **payload)


def test_calibration_uses_only_frozen_role_without_per_date_refit(tmp_path) -> None:
    calibration_path = tmp_path / "calibration.npz"
    research_path = tmp_path / "research.npz"
    _write_cache(calibration_path, activity=1)
    _write_cache(research_path, activity=100)
    captures = [
        _capture(
            calibration_path,
            "calibration",
            "historical_normalization_calibration",
        ),
        _capture(research_path, "research"),
    ]
    contract = a0.calibration_contract(
        captures,
        {
            "calibration": calibration_path,
            "research": research_path,
        },
    )
    assert contract["activity_q60"] == 25
    assert contract["roles_used"] == ["historical_normalization_calibration"]
    assert contract["per_date_refits"] == 0


def test_mixed_history_and_same_direction_renewal_are_state_derived() -> None:
    machine, ts_ms = _ready_machine()
    assert machine.mixed_exposure_ns == a0.MIXED_HISTORY_NS
    _step(machine, ts_ms, q_up=True)
    assert machine.candidate_origin_mixed_history_ns == a0.MIXED_HISTORY_NS
    for offset in (20, 40, 60, 80, 100):
        _step(machine, ts_ms + offset, q_up=True)
    confirmed = _step(machine, ts_ms + 120, q_up=True)
    assert confirmed.anchor_type == "mixed_onset"
    assert machine.renewal_count == 0
    assert _step(machine, ts_ms + 140).anchor_type is None
    renewal = _step(machine, ts_ms + 160, q_up=True)
    assert renewal.anchor_type is None
    assert machine.renewal_count == 1


def test_release_to_flip_precedence_and_quality_reset_censoring() -> None:
    machine, confirmed_ms = _confirmed_machine()
    _step(machine, confirmed_ms + 300, release=True)
    assert machine.state == a0.RELEASE_CANDIDATE
    superseded = _step(
        machine,
        confirmed_ms + 320,
        q_down=True,
        release=True,
    )
    assert superseded.candidate_status == "superseded_by_flip"
    assert machine.state == a0.FLIP_CANDIDATE
    reset = machine.step(
        (confirmed_ms + 340) * 1_000_000,
        quality_ok=False,
        active=True,
        q={-1: True, 1: False},
        q_both=False,
        q_release=False,
        q_mixed=False,
    )
    assert reset.state_closed == "quality_reset_censored"
    assert machine.state == a0.INACTIVE
    assert not any(
        (
            machine.candidate_at,
            machine.candidate_exposure_ns,
            machine.release_exposure_ns,
            machine.renewal_count,
            machine.flip_attempt_count,
        )
    )


def test_transition_precedence_active_loss_and_ambiguity() -> None:
    machine, ts_ms = _ready_machine()
    lost = machine.step(
        ts_ms * 1_000_000,
        quality_ok=True,
        active=False,
        q={-1: False, 1: True},
        q_both=False,
        q_release=False,
        q_mixed=False,
    )
    assert lost.transition.endswith("flow_support_lost")
    assert lost.anchor_type is None

    machine, ts_ms = _ready_machine()
    ambiguous = _step(machine, ts_ms, q_up=True, q_down=True)
    assert ambiguous.anchor_type is None
    assert machine.state == a0.MIXED_BUILDING


def test_quality_reset_has_precedence_in_every_state() -> None:
    states = (
        a0.MIXED_BUILDING,
        a0.MIXED_READY,
        a0.DOM_CANDIDATE,
        a0.DOMINANT,
        a0.FLIP_CANDIDATE,
        a0.RELEASE_CANDIDATE,
    )
    for state in states:
        machine = a0.DominanceStateMachine()
        machine.state = state
        machine.direction = 1
        machine.candidate_at = 1
        machine.candidate_exposure_ns = a0.CHECKPOINT_NS
        machine.release_exposure_ns = a0.CHECKPOINT_NS
        result = machine.step(
            1_000_000_000,
            quality_ok=False,
            active=False,
            q={-1: True, 1: True},
            q_both=True,
            q_release=True,
            q_mixed=True,
        )
        assert result.transition.endswith("quality_reset")
        assert result.anchor_type is None
        assert machine.state == a0.INACTIVE


def test_control_calendar_maps_first_valid_checkpoint_and_keeps_earlier_grid() -> None:
    grid, next_grid = a0._control_grid_assignment(
        None,
        10_000_000,
        segment_changed=False,
        checkpoint_valid=False,
    )
    assert grid is None
    assert next_grid == 250_000_000
    grid, next_grid = a0._control_grid_assignment(
        next_grid,
        760_000_000,
        segment_changed=False,
        checkpoint_valid=True,
    )
    assert grid == 250_000_000
    assert next_grid == 1_000_000_000
    grid, next_grid = a0._control_grid_assignment(
        next_grid,
        1_010_000_000,
        segment_changed=True,
        checkpoint_valid=True,
    )
    assert grid is None
    assert next_grid == 1_250_000_000


def test_dependence_clusters_and_pair_geometry_preserve_complete_pairs() -> None:
    anchors = [
        {
            "anchor_id": f"a{index}",
            "capture_id": "capture",
            "segment_id": 0,
            "research_date": "2026-08-27",
            "anchor_ts_ns": index * 50_000_000,
        }
        for index in range(2)
    ]
    controls = [
        {
            "control_id": f"c{index}",
            "capture_id": "capture",
            "segment_id": 0,
            "research_date": "2026-08-27",
            "checkpoint_ts_ns": index * 50_000_000 + 10_000_000,
            "control_direction": 1,
        }
        for index in range(2)
    ]
    pairs = [
        {
            "pair_id": f"p{index}",
            "anchor_id": f"a{index}",
            "anchor_capture_id": "capture",
            "anchor_segment_id": 0,
            "anchor_ts_ns": index * 50_000_000,
            "control_capture_id": "capture",
            "control_segment_id": 0,
            "control_checkpoint_ts_ns": index * 50_000_000 + 10_000_000,
            "control_direction": 1,
            "research_date": "2026-08-27",
        }
        for index in range(2)
    ]
    anchor_clusters, anchor_map = a0._cluster_rows(
        anchors,
        capture_field="capture_id",
        ts_field="anchor_ts_ns",
        id_field="anchor_id",
    )
    control_clusters, control_map = a0._cluster_rows(
        controls,
        capture_field="capture_id",
        ts_field="checkpoint_ts_ns",
        id_field="control_id",
    )
    assert len(anchor_clusters) == len(control_clusters) == 1
    checkpoint_map = {
        (
            row["capture_id"],
            row["segment_id"],
            row["checkpoint_ts_ns"],
        ): control_map[row["control_id"]]
        for row in controls
    }
    pair_clusters, pair_map = a0._pair_clusters(
        pairs,
        anchor_map,
        checkpoint_map,
    )
    assert len(pair_clusters) == 1
    assert len(set(pair_map.values())) == 1
    geometry, _, pair_components, _ = a0.geometry_audit(
        anchors,
        controls,
        pairs,
        {("capture", 0): 10_000_000_000},
    )
    assert all(row["pair_dependence_component_count"] == 1 for row in geometry)
    assert all(
        sum(row["pair_count"] for row in pair_components if row["tau_ms"] == tau)
        == 2
        for tau in a0.TAU_CANDIDATES_MS
    )


def test_a0_7_conditions_are_jointly_evaluated_on_one_horizon() -> None:
    base = {
        "overall_complete_coverage": 1.0,
        "minimum_date_complete_coverage": 1.0,
        "overlap_component_count": 100,
        "max_overlap_component_entry_share": 0.04,
        "max_date_overlap_component_share": 0.09,
        "pair_dependence_component_count": 100,
        "max_pair_dependence_pair_share": 0.04,
        "max_date_pair_dependence_share": 0.09,
        "passed": False,
    }
    rows = [
        {**base, "tau_ms": 100, "overlap_component_count": 99},
        {
            **base,
            "tau_ms": 500,
            "max_pair_dependence_pair_share": 0.06,
        },
    ]
    selected = a0._select_geometry_evaluation_row(rows, primary_tau=None)
    assert selected["tau_ms"] == 500
    conditions = a0._geometry_conditions_for_row(selected)
    assert len(conditions) == 8
    assert not all(passed for _, passed in conditions)
    assert dict(conditions)["overlap_component_count_ge_100"]
    assert not dict(conditions)["max_pair_dependence_pair_share_le_0_05"]
