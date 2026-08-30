from __future__ import annotations

import ast
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from examples.hyperliquid import (
    skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1 as AUDIT,
)
from examples.hyperliquid import (
    skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1_verifier as VERIFIER,
)


def synthetic_features(
    *,
    epochs: int = 1,
    segment_start_ns: int = 0,
) -> dict[str, np.ndarray]:
    length = epochs * AUDIT.EXPECTED_EPOCH_CHECKPOINTS
    ts = np.arange(length, dtype=np.int64) * AUDIT.CHECKPOINT_NS
    zeros = np.zeros(length, dtype=np.float64)
    ones = np.ones(length, dtype=np.float64)
    return {
        "ts_ns": ts,
        "event_seq": np.arange(length, dtype=np.int64),
        "segment_id": np.zeros(length, dtype=np.int32),
        "segment_start_ts": np.full(length, segment_start_ns, dtype=np.int64),
        "ready": np.ones(length, dtype=bool),
        "valid_book": np.ones(length, dtype=bool),
        "activity_500": np.full(length, 100.0),
        "ratios_100": np.zeros((length, 3), dtype=np.float64),
        "ratios_500": np.zeros((length, 3), dtype=np.float64),
        "trade_total": ones.copy(),
        "trade_signed": zeros.copy(),
        "bid_depletion": ones.copy(),
        "ask_depletion": ones.copy(),
        "ofi": zeros.copy(),
        "ofi_abs": ones.copy(),
    }


def set_direction(
    features: dict[str, np.ndarray],
    index: int,
    channel: int,
    direction: int,
    *,
    fast: float = 0.50,
    medium: float = 0.25,
) -> None:
    features["ratios_100"][index, channel] = direction * fast
    features["ratios_500"][index, channel] = direction * medium


def confirmed_trade_fixture(index: int = 1510) -> dict[str, np.ndarray]:
    features = synthetic_features()
    set_direction(features, index, 0, 1)
    set_direction(features, index + 1, 0, 1)
    return features


def analyze(features: dict[str, np.ndarray]) -> dict[str, object]:
    return AUDIT.analyze_features(
        capture_id="fixture",
        research_date="2026-08-30",
        features=features,
    )


def primary_trigger(result: dict[str, object]) -> dict[str, object]:
    rows = [row for row in result["trigger_rows"] if row["variant"] == "TRADE_LED"]
    assert len(rows) == 1
    return rows[0]


def passing_gate_values() -> dict[str, object]:
    return {
        "baseline_authority_verified": True,
        "frozen_successor_identities_verified": True,
        "direct_callable_bindings_verified": True,
        "claim_and_lock_valid_before_cache": True,
        "canonical_source_closure_exact": True,
        "source_preflight_violation_count": 0,
        "raw_a_b_difference_count": 0,
        "poison_cache_count": 29,
        "poison_unconsumed_field_count": 15,
        "poison_changed_field_instance_count": 435,
        "poison_consumed_field_mismatch_count": 0,
        "raw_a_p_difference_count": 0,
        "action_partition_violation_count": 0,
        "unauthorized_ttl_refresh_count": 0,
        "cross_segment_memory_carry_count": 0,
        "conservation_violation_count": 0,
        "fixed_epoch_violation_count": 0,
        "slice_mismatch_count": 0,
        "cross_segment_compared_checkpoint_count": 0,
        "represented_slice_date_count": 4,
        "distinct_comparable_epoch_count": 30,
        "compared_support_checkpoint_count": 1,
        "schema_violation_count": 0,
        "numeric_violation_count": 0,
        "trade_led_confirmed_cluster_count": 30,
        "trade_led_represented_date_count": 4,
        "trade_led_maximum_single_date_share": 0.50,
    }


def raw_cache_payload(length: int = 80) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    row_fields = AUDIT.epoch_authority.ROW_ALIGNED_CACHE_FIELDS
    for index, name in enumerate(sorted(AUDIT.feature_authority.ALLOWED_CACHE_FIELDS)):
        if name in row_fields:
            if name == "ts_ns":
                value = np.arange(length, dtype=np.int64) * AUDIT.CHECKPOINT_NS
            elif name == "event_seq":
                value = np.arange(length, dtype=np.int64)
            elif name == "segment_id":
                value = np.zeros(length, dtype=np.int32)
            elif name in {"ready", "valid_book"}:
                value = np.ones(length, dtype=bool)
            else:
                value = np.full(length, index + 1, dtype=np.float64)
        elif name == "cache_schema_version":
            value = np.asarray([4], dtype=np.int32)
        elif name in {"segment_end_ids", "segment_end_ts"}:
            value = np.asarray([0], dtype=np.int64)
        else:
            value = np.asarray([index + 1], dtype=np.float64)
        result[name] = value
    return result


def write_raw_cache(path: Path, length: int = 80) -> dict[str, np.ndarray]:
    payload = raw_cache_payload(length)
    np.savez_compressed(path, **payload)
    return payload


def test_frozen_identity_and_exact_output_domains() -> None:
    assert AUDIT.IDEA_SHA256 == (
        "a916717f21e1714298520e69f8e2702920f4cd54308f5d554691d4364a1cc997"
    )
    assert AUDIT.PLAN_SHA256 == (
        "8171a7bed7b216527fed468dbcff8f31ab1f48ec871bb2eee9b0ae7ad123f79c"
    )
    assert len(AUDIT.RAW_11) == 11
    assert len(AUDIT.SEALED_15) == 15
    assert len(AUDIT.EVIDENCED_16) == 16
    assert len(AUDIT.FINAL_17) == 17
    assert len(set(AUDIT.FINAL_17)) == 17


def test_authority_files_and_all_eight_callables_are_frozen() -> None:
    repo = Path(__file__).resolve().parents[2]
    rows = AUDIT.verify_authority_bindings(repo)
    assert [row["callable_name"] for row in rows] == [
        "build_features",
        "source_preflight",
        "base_eligibility",
        "channel_actions",
        "channel_memories",
        "epoch_support_ledger",
        "materialize_poisoned_cache_set",
        "verify_poison_attestation",
    ]


def test_analyze_features_directly_calls_six_scientific_authorities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[str] = []
    names = (
        "source_preflight",
        "base_eligibility",
        "channel_actions",
        "channel_memories",
        "epoch_support_ledger",
    )
    for name in names:
        original = getattr(AUDIT.epoch_authority, name)

        def wrapper(*args: object, _name=name, _original=original, **kwargs: object):
            called.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(AUDIT.epoch_authority, name, wrapper)
    analyze(confirmed_trade_fixture())
    assert called == list(names)


@pytest.mark.parametrize(
    ("fast", "medium", "qualifies"),
    [
        (0.50, 0.25, True),
        (-0.50, -0.25, True),
        (0.499999, 0.25, False),
        (0.50, 0.249999, False),
    ],
)
def test_threshold_boundaries_are_inclusive(
    fast: float, medium: float, qualifies: bool
) -> None:
    features = synthetic_features()
    index = 1510
    direction = 1 if fast > 0 else -1
    set_direction(
        features,
        index,
        0,
        direction,
        fast=abs(fast),
        medium=abs(medium),
    )
    actions = AUDIT.epoch_authority.channel_actions(
        features=features,
        base_eligible=np.ones(len(features["ts_ns"]), dtype=bool),
        event_masks={
            "trade": np.ones(len(features["ts_ns"]), dtype=bool),
            "depletion": np.ones(len(features["ts_ns"]), dtype=bool),
            "ofi": np.ones(len(features["ts_ns"]), dtype=bool),
        },
        margin=0.0,
    )
    expected = (
        AUDIT.epoch_authority.NEW_POS
        if direction == 1
        else AUDIT.epoch_authority.NEW_NEG
    )
    assert bool(actions[index, 0] == expected) is qualifies


def test_ttl_100ms_is_fresh_and_120ms_is_stale() -> None:
    actions = np.full((8, 3), AUDIT.NO_UPDATE, dtype=np.int8)
    actions[0, 0] = AUDIT.NEW_POS
    ts = np.arange(8, dtype=np.int64) * AUDIT.CHECKPOINT_NS
    states, ages, _ = AUDIT.epoch_authority.channel_memories(
        actions=actions,
        ts_ns=ts,
        segments=np.zeros(8, dtype=np.int32),
        ttl_ms=100,
    )
    assert states[5, 0] == 1
    assert ages[5, 0] == 100
    assert states[6, 0] == 9
    assert ages[6, 0] == -1


def test_six_point_prestate_excludes_current_checkpoint() -> None:
    features = confirmed_trade_fixture()
    result = analyze(features)
    assert primary_trigger(result)["candidate_event_seq"] == 1510
    features = confirmed_trade_fixture()
    set_direction(features, 1509, 0, 1)
    assert 1510 not in [
        row["candidate_event_seq"]
        for row in analyze(features)["trigger_rows"]
        if row["variant"] == "TRADE_LED"
    ]


@pytest.mark.parametrize(
    ("index", "expected_raw", "expected_omitted"),
    [(750, 1, 0), (2250, 1, 1)],
)
def test_core_is_half_open(
    index: int, expected_raw: int, expected_omitted: int
) -> None:
    features = synthetic_features(segment_start_ns=-30_000_000_000)
    set_direction(features, index, 0, 1)
    set_direction(features, min(index + 1, 2999), 0, 1)
    result = analyze(features)
    row = next(
        row
        for row in result["counter_rows"]
        if row["variant"] == "TRADE_LED" and row["direction"] == 1
    )
    assert row["raw_onset_count"] == expected_raw
    assert row["epoch_core_omitted_count"] == expected_omitted


@pytest.mark.parametrize(
    ("index", "retained", "edge_omitted"),
    [(2240, 1, 0), (2241, 0, 1)],
)
def test_confirmation_close_equality_and_plus_20ms(
    index: int, retained: int, edge_omitted: int
) -> None:
    features = synthetic_features(segment_start_ns=-30_000_000_000)
    set_direction(features, index, 0, 1)
    if index + 1 < len(features["ts_ns"]):
        set_direction(features, index + 1, 0, 1)
    result = analyze(features)
    row = next(
        row
        for row in result["counter_rows"]
        if row["variant"] == "TRADE_LED" and row["direction"] == 1
    )
    assert row["retained_count"] == retained
    assert row["confirmation_edge_omitted_count"] == edge_omitted


def test_same_checkpoint_neutral_clears_old_opposite_before_veto() -> None:
    features = confirmed_trade_fixture()
    set_direction(features, 1509, 1, -1)
    features["ratios_100"][1510, 1] = 0.0
    features["ratios_500"][1510, 1] = 0.0
    trigger = primary_trigger(analyze(features))
    assert trigger["secondary_opposite_count"] == 0


def test_same_checkpoint_new_opposite_is_visible_to_veto() -> None:
    features = confirmed_trade_fixture()
    set_direction(features, 1510, 1, -1)
    result = analyze(features)
    row = next(
        row
        for row in result["counter_rows"]
        if row["variant"] == "TRADE_LED" and row["direction"] == 1
    )
    assert row["anchor_vetoed_count"] == 1
    assert row["retained_count"] == 0


def test_vetoed_onset_does_not_occupy_thinning_key() -> None:
    features = synthetic_features()
    first, second = 1510, 1525
    set_direction(features, first, 0, 1)
    set_direction(features, first, 1, -1)
    set_direction(features, second, 0, 1)
    set_direction(features, second + 1, 0, 1)
    trigger = primary_trigger(analyze(features))
    assert trigger["candidate_event_seq"] == second


def test_earliest_retained_failure_suppresses_later_confirmable_trigger() -> None:
    features = synthetic_features()
    first, second = 1510, 1525
    set_direction(features, first, 0, 1)
    set_direction(features, second, 0, 1)
    set_direction(features, second + 1, 0, 1)
    result = analyze(features)
    trigger = primary_trigger(result)
    assert trigger["candidate_event_seq"] == first
    assert trigger["confirmation_status"] == "CANCELLED"
    row = next(
        row
        for row in result["counter_rows"]
        if row["variant"] == "TRADE_LED" and row["direction"] == 1
    )
    assert row["same_key_suppressed_count"] == 1
    assert row["retained_count"] == 1


def test_opposite_directions_and_variants_share_epoch_cluster() -> None:
    features = synthetic_features()
    set_direction(features, 1510, 0, 1)
    set_direction(features, 1511, 0, 1)
    set_direction(features, 1530, 0, -1)
    set_direction(features, 1531, 0, -1)
    set_direction(features, 1550, 1, 1)
    set_direction(features, 1551, 1, 1)
    rows = analyze(features)["trigger_rows"]
    clusters = {
        row["dependence_cluster_id"]
        for row in rows
        if row["variant"] in {"TRADE_LED", "DEPLETION_LED"}
    }
    assert clusters == {"fixture:0"}


def test_confirmation_records_first_update_but_waits_until_close() -> None:
    trigger = primary_trigger(analyze(confirmed_trade_fixture()))
    assert trigger["first_additional_same_update_ts_ns"] == 30_220_000_000
    assert trigger["confirmation_window_close_ts_ns"] == 30_400_000_000
    assert (
        trigger["first_additional_same_update_ts_ns"]
        < trigger["confirmation_window_close_ts_ns"]
    )


@pytest.mark.parametrize(
    ("mutator", "reason"),
    [
        (
            lambda features: set_direction(features, 1512, 2, -1),
            "explicit_opposite_update",
        ),
        (
            lambda features: None,
            "no_additional_same_leader_update",
        ),
    ],
)
def test_cancellation_atoms_and_reason_precedence(mutator: object, reason: str) -> None:
    features = synthetic_features()
    set_direction(features, 1510, 0, 1)
    mutator(features)
    trigger = primary_trigger(analyze(features))
    assert trigger["cancel_reason"] == reason
    assert trigger[reason] is True


def test_confirmation_segment_boundary_atom_is_independent() -> None:
    features = synthetic_features()
    set_direction(features, 1510, 0, 1)
    features["segment_id"][1511:1521] = 1
    invalid, masks = AUDIT.epoch_authority.source_preflight(features)
    assert invalid == 0
    _, _, base = AUDIT.epoch_authority.base_eligibility(
        features, AUDIT.feature_authority
    )
    actions = AUDIT.epoch_authority.channel_actions(
        features=features,
        base_eligible=base,
        event_masks=masks,
        margin=0.0,
    )
    status = AUDIT._confirmation_status(
        trigger_index=1510,
        direction=1,
        leader_index=0,
        features=features,
        actions=actions,
    )
    assert status["confirmation_segment_boundary"] is True
    assert status["cancel_reason"] == "confirmation_segment_boundary"


def test_conservation_holds_for_every_epoch_variant_direction() -> None:
    result = analyze(confirmed_trade_fixture())
    assert result["conservation_violation_count"] == 0
    for row in result["counter_rows"]:
        assert row["raw_onset_count"] == (
            row["epoch_core_omitted_count"]
            + row["confirmation_edge_omitted_count"]
            + row["anchor_vetoed_count"]
            + row["veto_admitted_count"]
        )
        assert row["veto_admitted_count"] == (
            row["retained_count"] + row["same_key_suppressed_count"]
        )
        assert row["retained_count"] == (
            row["confirmed_count"] + row["cancelled_count"]
        )


def test_source_invalid_fails_before_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = confirmed_trade_fixture()
    features["trade_signed"][0] = np.nan
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError

    monkeypatch.setattr(AUDIT.epoch_authority, "channel_actions", forbidden)
    with pytest.raises(AUDIT.SourcePreflightError):
        analyze(features)
    assert called is False


def test_feature_frame_roundtrip_and_typed_arithmetic() -> None:
    features = {
        "scalar": np.asarray(7, dtype=np.int16),
        "empty": np.empty((2, 0), dtype=np.float64),
        "vector": np.asarray([1, 2, 3], dtype=np.int32),
    }
    frame, sender, rows = AUDIT.pack_feature_frame(
        call_index=0, features=features, field_access_rows=[]
    )
    restored, header, receiver = AUDIT.unpack_feature_frame(frame)
    assert sender == receiver
    assert header["feature_key_count"] == 3
    assert rows[0]["offset_bytes"] == 0
    assert rows[-1]["offset_bytes"] + rows[-1]["length_bytes"] == len(
        frame
    ) - 8 - int.from_bytes(frame[:8], "big")
    for name in features:
        assert np.array_equal(restored[name], features[name])


@pytest.mark.parametrize(
    "mutation",
    ["trailing", "payload", "offset", "dtype", "negative_shape"],
)
def test_feature_frame_hostile_mutations_fail(mutation: str) -> None:
    features = {"x": np.asarray([1, 2], dtype=np.int32)}
    frame, _, _ = AUDIT.pack_feature_frame(
        call_index=0, features=features, field_access_rows=[]
    )
    header_len = int.from_bytes(frame[:8], "big")
    header = json.loads(frame[8 : 8 + header_len])
    payload = frame[8 + header_len :]
    if mutation == "trailing":
        bad = frame + b"x"
    elif mutation == "payload":
        bad = frame[:-1] + bytes([frame[-1] ^ 1])
    else:
        if mutation == "offset":
            header["arrays"][0]["offset_bytes"] = 1
        elif mutation == "dtype":
            header["arrays"][0]["dtype_str"] = "<i8"
        else:
            header["arrays"][0]["shape"] = [-1]
        encoded = AUDIT.canonical_bytes(header)
        bad = len(encoded).to_bytes(8, "big") + encoded + payload
    with pytest.raises(AUDIT.AuditError):
        AUDIT.unpack_feature_frame(bad)


def test_instrumented_builder_reads_exact_consumed_fields(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "fixture.npz"
    write_raw_cache(cache)
    features, accesses, schema_count, forbidden_count = (
        AUDIT.build_features_instrumented(
            cache_path=cache,
            call_index=0,
            build_label="A",
            input_sha256=AUDIT.sha256_file(cache),
        )
    )
    assert schema_count == 1
    assert forbidden_count == 0
    assert [row["field"] for row in accesses] == sorted(
        AUDIT.feature_authority.CONSUMED_CACHE_FIELDS
    )
    assert AUDIT.feature_sha256(features)


def test_full_loader_detector_subprocess_boundary(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "fixture.npz"
    write_raw_cache(cache)
    analysis, call, accesses, events = AUDIT.execute_feature_call(
        cache_path=cache,
        call_index=0,
        build_label="A",
        unit_kind="FULL",
        capture_id="fixture",
        research_date="2026-08-30",
        slice_ordinal=None,
        input_authority="CANONICAL",
    )
    assert call["feature_output_sha256"] == call["consumer_input_sha256"]
    assert call["consumer_input_sha256"] == call["detector_exit_sha256"]
    assert call["sender_ipc"]["sent_frame_count"] == 1
    assert call["receiver_ipc"]["received_frame_count"] == 1
    assert call["receiver_ipc"]["eof_observed"] is True
    assert len(accesses) == 12
    assert [row["phase"] for row in events] == ["HASHER", "LOADER"]
    assert "_slice_specs" in analysis
    assert not any(isinstance(value, np.ndarray) for value in analysis.values())


def test_slice_materializer_runs_in_separate_process_and_ledgers_io(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    write_raw_cache(source, 100)
    target = tmp_path / "work" / "slice_000000.npz"
    result, events = AUDIT.materialize_slice_subprocess(
        input_path=source,
        output_path=target,
        segment_id=0,
        nominal_start_ts_ns=20 * AUDIT.CHECKPOINT_NS,
        call_index=1,
        build_label="A",
    )
    assert result["actual_start_ts_ns"] == 20 * AUDIT.CHECKPOINT_NS
    assert [(row["operation"], row["caller_name"]) for row in events] == [
        ("READ_INPUT", "materialize_slice"),
        ("WRITE_SLICE", "materialize_slice"),
    ]


def test_instrumentation_requires_exact_raw_event_matrix(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "fixture.npz"
    write_raw_cache(cache)
    _, call, accesses, events = AUDIT.execute_feature_call(
        cache_path=cache,
        call_index=0,
        build_label="A",
        unit_kind="FULL",
        capture_id="fixture",
        research_date="2026-08-30",
        slice_ordinal=None,
        input_authority="CANONICAL",
    )
    payload = AUDIT.instrumentation_evidence(
        attempt_id="attempt",
        feature_calls=[call],
        field_accesses=accesses,
        raw_open_events=events,
    )
    assert [row["event_index"] for row in payload["raw_open_events"]] == [0, 1]
    assert all(row["allowed"] for row in payload["raw_open_events"])
    with pytest.raises(AUDIT.AuditError, match="raw_open_event_duplicate"):
        AUDIT.instrumentation_evidence(
            attempt_id="attempt",
            feature_calls=[call],
            field_accesses=accesses,
            raw_open_events=[*events, events[0]],
        )


def test_runner_has_zero_np_load_attribute_call_sites() -> None:
    tree = ast.parse(Path(AUDIT.__file__).read_text(encoding="ascii"))
    callsites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "np"
        and node.attr == "load"
    ]
    assert callsites == []


def test_materialized_slice_rebuilds_raw_and_is_no_replace(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.npz"
    raw = write_raw_cache(source, 100)
    target = tmp_path / "work" / "slice_000000.npz"
    result = AUDIT.materialize_slice(
        input_path=source,
        output_path=target,
        segment_id=0,
        nominal_start_ts_ns=20 * AUDIT.CHECKPOINT_NS,
    )
    assert result["actual_start_ts_ns"] == 20 * AUDIT.CHECKPOINT_NS
    with np.load(target, allow_pickle=False) as handle:
        assert np.array_equal(handle["ts_ns"], raw["ts_ns"][20:])
    with pytest.raises(AUDIT.AuditError, match="slice_exists"):
        AUDIT.materialize_slice(
            input_path=source,
            output_path=target,
            segment_id=0,
            nominal_start_ts_ns=0,
        )


def test_slice_identity_mutations_fail_each_surface() -> None:
    full = analyze(confirmed_trade_fixture())
    sliced = copy.deepcopy(full)
    full["_features"] = confirmed_trade_fixture()
    sliced["_features"] = confirmed_trade_fixture()
    base = AUDIT.slice_invariance_row(
        full_analysis=full,
        sliced_analysis=sliced,
        segment_id=0,
        nominal_start_ts_ns=0,
        actual_start_ts_ns=-AUDIT.SLICE_GUARD_NS,
        slice_source_sha256="a" * 64,
    )
    assert base["mismatch_reason"] == "none"
    surfaces = (
        ("epoch_rows", "disposition", "counter"),
        ("counter_rows", "raw_onset_count", "counter"),
        ("trigger_rows", "candidate_event_seq", "retained"),
        ("trigger_rows", "cancel_reason", "status"),
        ("support_rows", "memory_int", "support"),
    )
    for collection, field, expected in surfaces:
        mutated = copy.deepcopy(sliced)
        if mutated[collection]:
            value = mutated[collection][0][field]
            mutated[collection][0][field] = (
                value + 1 if isinstance(value, int) else f"{value}_mutated"
            )
        row = AUDIT.slice_invariance_row(
            full_analysis=full,
            sliced_analysis=mutated,
            segment_id=0,
            nominal_start_ts_ns=0,
            actual_start_ts_ns=-AUDIT.SLICE_GUARD_NS,
            slice_source_sha256="a" * 64,
        )
        if mutated[collection]:
            assert row["mismatch_reason"] in {expected, "epoch_disposition"}


def test_slice_comparable_epochs_stay_in_artificial_start_segment() -> None:
    features = synthetic_features(epochs=2, segment_start_ns=-30_000_000_000)
    features["segment_id"][3000:] = 1
    features["segment_start_ts"][3000:] = features["ts_ns"][3000]
    full = analyze(features)
    sliced = copy.deepcopy(full)
    comparable = AUDIT.comparable_epoch_ids(
        full_analysis=full,
        sliced_analysis=sliced,
        actual_start_ts_ns=-AUDIT.SLICE_GUARD_NS,
        segment_id=0,
    )
    assert comparable == {0}


def test_synthetic_raw_build_and_exact_17_path_seal(
    tmp_path: Path,
) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    cache_name = "2026-08-30_fixture.npz"
    cache = cache_root / cache_name
    write_raw_cache(cache)
    inventory = [
        {
            "cache_name": cache_name,
            "size_bytes": cache.stat().st_size,
            "row_count": 80,
            "cache_schema_version": 4,
            "cache_sha256": AUDIT.sha256_file(cache),
            "source_authority_verified": True,
            "cache_field_schema_verified": True,
        }
    ]
    results = {}
    call_index = 0
    roots = {}
    for label in ("A", "B", "P"):
        root = tmp_path / label
        roots[label] = root
        results[label] = AUDIT.build_raw_output(
            repo_root=Path(__file__).resolve().parents[2],
            build_label=label,
            input_cache_root=cache_root,
            canonical_inventory=inventory,
            output_root=root,
            work_root=tmp_path / "work",
            start_call_index=call_index,
            input_authority="CANONICAL",
            authority_binding_factory=lambda count: {
                "schema_version": 1,
                "direct_call_count": count,
            },
        )
        call_index = results[label]["next_call_index"]
        assert {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        } == set(AUDIT.RAW_11)
    aggregate = results["A"]["aggregate"]
    slice_rows = results["A"]["slice_rows"]
    integrity = {
        "source_preflight_violation_count": 0,
        "action_partition_violation_count": 0,
        "unauthorized_ttl_refresh_count": 0,
        "cross_segment_memory_carry_count": 0,
        "conservation_violation_count": 0,
        "fixed_epoch_violation_count": 0,
        "slice_mismatch_count": sum(
            row["mismatch_reason"] != "none" for row in slice_rows
        ),
        "cross_segment_compared_checkpoint_count": 0,
        "represented_slice_date_count": 1,
        "distinct_comparable_epoch_count": 0,
        "compared_support_checkpoint_count": 0,
        "schema_violation_count": 0,
        "numeric_violation_count": 0,
    }
    sealed, comparisons = AUDIT.seal_roots(
        roots=roots,
        aggregate=aggregate,
        authority_state={
            "baseline_authority_verified": True,
            "frozen_successor_identities_verified": True,
            "direct_callable_bindings_verified": True,
            "claim_and_lock_valid_before_cache": True,
            "canonical_source_closure_exact": False,
        },
        poison_evidence={
            "cache_count": 29,
            "unconsumed_field_count": 15,
            "nonempty_unconsumed_field_instance_count": 435,
            "changed_unconsumed_field_instance_count": 435,
            "consumed_field_mismatch_count": 0,
            "attestation_sha256": "a" * 64,
        },
        integrity=integrity,
        attempt_id="synthetic",
        implementation_head="b" * 40,
    )
    assert sealed["classification"] == "Aminus1_authority_or_source_failed"
    assert comparisons["final_a_b"]["difference_count"] == 0
    assert comparisons["final_a_p"]["difference_count"] == 0
    for root in roots.values():
        assert {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
        } == set(AUDIT.FINAL_17)
        manifest = json.loads((root / "run_manifest.json").read_text(encoding="ascii"))
        assert manifest["artifact_count"] == 16
        assert all(row["path"] != "run_manifest.json" for row in manifest["artifacts"])


def test_gate_success_and_exact_classification() -> None:
    gates = AUDIT.build_gates(passing_gate_values())
    assert [row["status"] for row in gates] == ["PASS"] * 4
    assert AUDIT.classify(gates) == "Aminus1_trade_led_recurrent_structural_candidate"


@pytest.mark.parametrize(
    ("field", "classification"),
    [
        ("source_preflight_violation_count", "Aminus1_authority_or_source_failed"),
        ("raw_a_p_difference_count", "Aminus1_outcome_boundary_violated"),
        ("slice_mismatch_count", "Aminus1_detector_integrity_failed"),
        (
            "trade_led_confirmed_cluster_count",
            "Aminus1_trade_led_structural_support_not_estimable",
        ),
        (
            "trade_led_maximum_single_date_share",
            "Aminus1_trade_led_structure_date_concentrated",
        ),
    ],
)
def test_gate_failure_precedence(field: str, classification: str) -> None:
    values = passing_gate_values()
    values[field] = (
        1
        if field
        not in {
            "trade_led_confirmed_cluster_count",
            "trade_led_maximum_single_date_share",
        }
        else 0
        if field == "trade_led_confirmed_cluster_count"
        else 0.51
    )
    gates = AUDIT.build_gates(values)
    assert AUDIT.classify(gates) == classification
    failed_index = next(
        index for index, row in enumerate(gates) if row["status"] == "FAIL"
    )
    assert all(row["status"] == "NOT_EVALUATED" for row in gates[failed_index + 1 :])


def test_a_minus1_3_count_and_date_short_circuit() -> None:
    values = passing_gate_values()
    values["trade_led_confirmed_cluster_count"] = 0
    gate = AUDIT.build_gates(values)[3]
    assert [row["status"] for row in gate["conditions"]] == [
        "FAIL",
        "NOT_EVALUATED",
        "NOT_EVALUATED",
    ]
    assert all(
        row["actual"] is None and row["passed"] is None
        for row in gate["conditions"][1:]
    )
    values = passing_gate_values()
    values["trade_led_represented_date_count"] = 3
    gate = AUDIT.build_gates(values)[3]
    assert [row["status"] for row in gate["conditions"]] == [
        "PASS",
        "FAIL",
        "NOT_EVALUATED",
    ]


def test_sensitivity_cannot_rescue_primary() -> None:
    values = passing_gate_values()
    values["trade_led_confirmed_cluster_count"] = 2
    assert (
        AUDIT.classify(AUDIT.build_gates(values))
        == "Aminus1_trade_led_structural_support_not_estimable"
    )
    assert not any(
        "depletion" in condition["condition"] or "ofi" in condition["condition"]
        for gate in AUDIT.build_gates(values)
        for condition in gate["conditions"]
    )


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), True])
def test_numeric_gate_mutations_fail_closed(bad: object) -> None:
    values = passing_gate_values()
    values["slice_mismatch_count"] = bad
    gates = AUDIT.build_gates(values)
    assert gates[2]["status"] == "FAIL"


def test_manifest_self_exclusion_and_comparison_missing_extra(
    tmp_path: Path,
) -> None:
    left, right = tmp_path / "a", tmp_path / "b"
    for root in (left, right):
        for relative in AUDIT.RAW_11:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(relative + "\n", encoding="ascii")
    exact = AUDIT.comparison("RAW_11:A_vs_B", left, right, AUDIT.RAW_11)
    assert exact["difference_count"] == 0
    (right / AUDIT.RAW_11[0]).unlink()
    missing = AUDIT.comparison("RAW_11:A_vs_B", left, right, AUDIT.RAW_11)
    assert missing["difference_count"] == 1
    rows = AUDIT.manifest_rows(left, AUDIT.RAW_11)
    assert [row["path"] for row in rows] == sorted(AUDIT.RAW_11)


def test_work_manifest_rejects_duplicate_path_and_key() -> None:
    row = {
        "build_label": "A",
        "cache_name": "x.npz",
        "slice_ordinal": 0,
        "path": "work/A/x.npz/slice_000000.npz",
        "size_bytes": 1,
        "sha256": "a" * 64,
    }
    with pytest.raises(AUDIT.AuditError, match="work_duplicate_path"):
        AUDIT.work_manifest_payload(attempt_id="attempt", rows=[row, dict(row)])


def test_no_replace_publication_rejects_second_write(
    tmp_path: Path,
) -> None:
    path = tmp_path / "artifact.json"
    AUDIT.write_json_no_replace(path, {"schema_version": 1})
    with pytest.raises(AUDIT.AuditError, match="no_replace_exists"):
        AUDIT.write_json_no_replace(path, {"schema_version": 1})


def test_push_once_writes_exact_receipt_and_rejects_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    (attempt / "push-ledger").mkdir(parents=True)
    old = None
    new = "a" * 40

    def fake_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(AUDIT.subprocess, "run", fake_run)
    monkeypatch.setattr(
        AUDIT,
        "ls_remote_observation",
        lambda repo, observation_id: {
            "observation_id": observation_id,
            "command": ["git", "ls-remote", "--heads", "origin", AUDIT.CONTROLLER_REF],
            "exit_code": 0,
            "stdout": f"{new}\t{AUDIT.CONTROLLER_REF}\n",
            "stderr": "",
            "observed_head": new,
        },
    )
    row, _ = AUDIT.push_once(
        repo_root=tmp_path,
        attempt_root=attempt,
        ordinal=0,
        phase="CONSUMPTION",
        expected_old_head=old,
        expected_new_head=new,
        pre_observation_id="PRE_CONSUMPTION",
        post_observation_id="POST_CONSUMPTION",
    )
    assert row["ordinal"] == 0
    assert row["retry_allowed"] is False
    with pytest.raises(AUDIT.AuditError, match="push_receipt_exists"):
        AUDIT.push_once(
            repo_root=tmp_path,
            attempt_root=attempt,
            ordinal=0,
            phase="CONSUMPTION",
            expected_old_head=old,
            expected_new_head=new,
            pre_observation_id="PRE_CONSUMPTION",
            post_observation_id="POST_CONSUMPTION",
        )


def test_failed_push_is_terminal_and_cannot_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "attempt"
    (attempt / "push-ledger").mkdir(parents=True)

    def failed_run(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="", stderr="failed")

    monkeypatch.setattr(AUDIT.subprocess, "run", failed_run)
    kwargs = {
        "repo_root": tmp_path,
        "attempt_root": attempt,
        "ordinal": 0,
        "phase": "CONSUMPTION",
        "expected_old_head": None,
        "expected_new_head": "a" * 40,
        "pre_observation_id": "PRE_CONSUMPTION",
        "post_observation_id": "POST_CONSUMPTION",
    }
    with pytest.raises(AUDIT.AuditError, match="push_failed"):
        AUDIT.push_once(**kwargs)
    receipt = attempt / "push-ledger/000-consumption.json"
    assert json.loads(receipt.read_text(encoding="ascii"))["exit_code"] == 1
    with pytest.raises(AUDIT.AuditError, match="push_receipt_exists"):
        AUDIT.push_once(**kwargs)


def test_verifier_check_ids_and_exit_row_encoding() -> None:
    assert len(VERIFIER.CHECK_IDS) == 13
    assert len(set(VERIFIER.CHECK_IDS)) == 13
    assert VERIFIER.CHECK_IDS[0] == "V00_CLI_AND_ROOTS"
    assert VERIFIER.CHECK_IDS[-1] == "V12_POST_SEAL_DRIFT"


def test_verifier_comparison_validation_detects_mutation(
    tmp_path: Path,
) -> None:
    left, right = tmp_path / "a", tmp_path / "b"
    left.mkdir()
    right.mkdir()
    (left / "x").write_text("same", encoding="ascii")
    (right / "x").write_text("same", encoding="ascii")
    payload = VERIFIER.comparison("RAW_11:A_vs_B", left, right, ("x",))
    VERIFIER.validate_comparison(payload)
    payload["difference_count"] = 1
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.validate_comparison(payload)


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("sender_ipc", "sent_frame_count", 2),
        ("receiver_ipc", "eof_observed", False),
        ("receiver_ipc", "frame_sha256", "b" * 64),
    ],
)
def test_verifier_rejects_ipc_endpoint_mutations(
    target: str,
    field: str,
    value: object,
) -> None:
    common = {
        "header_sha256": "a" * 64,
        "payload_sha256": "b" * 64,
        "payload_size_bytes": 4,
        "frame_sha256": "c" * 64,
        "frame_size_bytes": 24,
    }
    call = {
        "sender_ipc": {
            **common,
            "sent_frame_count": 1,
            "send_end_closed": True,
        },
        "receiver_ipc": {
            **common,
            "received_frame_count": 1,
            "eof_observed": True,
            "unused_byte_count": 0,
        },
    }
    VERIFIER.validate_ipc_endpoints(call)
    call[target][field] = value
    with pytest.raises(VERIFIER.VerificationError):
        VERIFIER.validate_ipc_endpoints(call)


def test_verifier_rejects_manifest_self_inclusion(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    for relative in AUDIT.EVIDENCED_16:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="ascii")
    rows = VERIFIER.manifest_rows(root, AUDIT.EVIDENCED_16)
    assert len(rows) == 16
    assert all(row["path"] != "run_manifest.json" for row in rows)


def init_git_repo(path: Path) -> str:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Fixture"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "fixture@example.com"],
        cwd=path,
        check=True,
    )
    (path / "seed.txt").write_text("seed\n", encoding="ascii")
    subprocess.run(["git", "add", "seed.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "seed"], cwd=path, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()


def commit_claim_transition(
    path: Path, *, extra_delta: bool = False
) -> tuple[str, str, str]:
    claim = path / AUDIT.CLAIM_ARMED_PATH
    claim.parent.mkdir(parents=True, exist_ok=True)
    claim.write_text(
        json.dumps(
            {
                "task_id": AUDIT.TASK_ID,
                "status": "ARMED_FOR_SINGLE_USE",
                "implementation_tag": AUDIT.IMPLEMENTATION_TAG,
                "formal_argv": ["formal"],
                "controller_ref": AUDIT.CONTROLLER_REF,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    subprocess.run(["git", "add", str(AUDIT.CLAIM_ARMED_PATH)], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "armed"], cwd=path, check=True)
    implementation = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    blob = subprocess.check_output(
        [
            "git",
            "rev-parse",
            f"{implementation}:{AUDIT.CLAIM_ARMED_PATH.as_posix()}",
        ],
        cwd=path,
        text=True,
    ).strip()
    claimed = path / AUDIT.CLAIMED_PATH
    os.link(claim, claimed)
    claim.unlink()
    if extra_delta:
        (path / "extra.txt").write_text("extra\n", encoding="ascii")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", AUDIT.CONSUMPTION_MESSAGE],
        cwd=path,
        check=True,
    )
    consumption = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    return implementation, consumption, blob


def test_consumption_transition_is_exact_same_blob_only_delta(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    implementation, consumption, blob = commit_claim_transition(tmp_path)
    AUDIT.verify_consumption_transition(
        repo_root=tmp_path,
        implementation_head=implementation,
        consumption_head=consumption,
        expected_claim_blob=blob,
    )
    VERIFIER.verify_consumption_transition(
        repo_root=tmp_path,
        implementation_head=implementation,
        consumption_head=consumption,
        expected_claim_blob=blob,
    )


def test_consumption_transition_rejects_extra_tree_delta(tmp_path: Path) -> None:
    init_git_repo(tmp_path)
    implementation, consumption, blob = commit_claim_transition(
        tmp_path, extra_delta=True
    )
    with pytest.raises(AUDIT.AuditError, match="consumption_exact_rename_delta"):
        AUDIT.verify_consumption_transition(
            repo_root=tmp_path,
            implementation_head=implementation,
            consumption_head=consumption,
            expected_claim_blob=blob,
        )
    with pytest.raises(
        VERIFIER.VerificationError, match="consumption_exact_rename_delta"
    ):
        VERIFIER.verify_consumption_transition(
            repo_root=tmp_path,
            implementation_head=implementation,
            consumption_head=consumption,
            expected_claim_blob=blob,
        )


@pytest.mark.parametrize("object_kind", ["blob", "tree"])
def test_historical_scan_rejects_dangling_attempt_objects(
    tmp_path: Path, object_kind: str
) -> None:
    init_git_repo(tmp_path)
    if object_kind == "blob":
        payload = json.dumps(
            {
                "task_id": AUDIT.TASK_ID,
                "status": "ARMED_FOR_SINGLE_USE",
                "implementation_tag": AUDIT.IMPLEMENTATION_TAG,
                "formal_argv": ["formal"],
                "controller_ref": AUDIT.CONTROLLER_REF,
            }
        ).encode()
        subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=tmp_path,
            input=payload,
            check=True,
        )
        expected = "historical_attempt_claim_blob"
    else:
        blob = (
            subprocess.check_output(
                ["git", "hash-object", "-w", "--stdin"],
                cwd=tmp_path,
                input=b"plain\n",
            )
            .decode()
            .strip()
        )
        subprocess.run(
            ["git", "mktree"],
            cwd=tmp_path,
            input=f"100644 blob {blob}\t{AUDIT.CLAIMED_PATH.name}\n".encode(),
            check=True,
        )
        expected = "historical_attempt_tree"
    with pytest.raises(AUDIT.AuditError, match=expected):
        AUDIT.verify_no_historical_attempt(tmp_path)


def test_detector_child_rejects_inherited_regular_file_fd(tmp_path: Path) -> None:
    target = tmp_path / "raw-capability.npz"
    target.write_bytes(b"fixture")
    descriptor = os.open(target, os.O_RDONLY)
    code = (
        "from examples.hyperliquid import "
        "skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1 as a;"
        "import sys;"
        "\ntry:\n a.enforce_detector_fd_boundary({0,1,2})"
        "\nexcept a.AuditError as e:\n"
        " sys.exit(0 if str(e).startswith('detector_inherited_fd:') else 3)"
        "\nsys.exit(4)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            pass_fds=(descriptor,),
            check=False,
        )
    finally:
        os.close(descriptor)
    assert result.returncode == 0


def test_detector_child_rejects_unregistered_control_pipe_fd() -> None:
    read_fd, write_fd = os.pipe()
    code = (
        "from examples.hyperliquid import "
        "skhynix_fixed_epoch_leader_trigger_opposition_veto_a_minus1 as a;"
        "import sys;"
        "\ntry:\n a.enforce_detector_fd_boundary({0,1,2})"
        "\nexcept a.AuditError as e:\n"
        " sys.exit(0 if str(e).startswith('detector_control_fd_domain:') else 3)"
        "\nsys.exit(4)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[2],
            pass_fds=(read_fd,),
            check=False,
        )
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert result.returncode == 0


def test_verifier_rejects_work_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(VERIFIER.VerificationError, match="relative_path_domain"):
        VERIFIER.safe_relative_child(tmp_path, "work/../../escape.npz", "work/")


def test_verifier_v00_rejects_tag_alias_before_resolution(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    context = {
        "repo_root": Path.cwd().resolve(),
        "attempt_root": attempt.resolve(),
        "result_out": Path.cwd().resolve()
        / ".workflow/reports/0830T002-terminal-verifier.json",
        "implementation_tag": "alias",
        "consumption_tag": VERIFIER.CONSUMPTION_TAG,
        "terminal_tag": VERIFIER.TERMINAL_TAG,
    }
    with pytest.raises(VERIFIER.VerificationError, match="verifier_frozen_tags"):
        VERIFIER.check_v00(context)


def write_typed_csv_row(
    path: Path,
    relative: str,
    overrides: dict[str, str],
) -> None:
    header = VERIFIER.CSV_HEADERS[relative]
    row: dict[str, str] = {}
    for field in header:
        if field in VERIFIER.CSV_BOOL_FIELDS:
            row[field] = "False"
        elif field in VERIFIER.CSV_JSON_FIELDS:
            row[field] = "[]"
        elif field in VERIFIER.CSV_OPTIONAL_FIELDS:
            row[field] = ""
        elif field.endswith("_sha256") or field in {
            "candidate_id",
            "retained_candidate_id",
        }:
            row[field] = "a" * 64
        elif field in VERIFIER.CSV_TEXT_FIELDS:
            row[field] = "x"
        else:
            row[field] = "0"
    row.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def canonical_trigger_row(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    candidate = VERIFIER.canonical_sha(
        ["fixture", "TRADE_LED", 0, 0, -1, 30_200_000_000, 1510]
    )
    trigger_path = tmp_path / "trigger.csv"
    write_typed_csv_row(
        trigger_path,
        "support/trigger_ledger.csv",
        {
            "research_date": "2026-08-30",
            "capture_id": "fixture",
            "variant": "TRADE_LED",
            "epoch_id": "0",
            "segment_id": "0",
            "direction": "-1",
            "candidate_id": candidate,
            "candidate_ts_ns": "30200000000",
            "candidate_event_seq": "1510",
            "dependence_cluster_id": "fixture:0",
            "leader_channel": "trade",
            "confirmation_status": "CONFIRMED",
            "cancel_reason": "none",
        },
    )
    counter_path = tmp_path / "counter.csv"
    write_typed_csv_row(
        counter_path,
        "support/epoch_variant_counters.csv",
        {
            "research_date": "2026-08-30",
            "capture_id": "fixture",
            "variant": "TRADE_LED",
            "direction": "-1",
            "retained_count": "1",
            "confirmed_count": "1",
            "retained_candidate_id": candidate,
        },
    )
    trigger = VERIFIER.typed_csv_rows(trigger_path, "support/trigger_ledger.csv")[0]
    counter = VERIFIER.typed_csv_rows(
        counter_path, "support/epoch_variant_counters.csv"
    )[0]
    return trigger, counter


def test_verifier_accepts_negative_direction_and_canonical_candidate(
    tmp_path: Path,
) -> None:
    trigger, counter = canonical_trigger_row(tmp_path)
    assert trigger["direction"] == -1
    assert counter["direction"] == -1
    VERIFIER.validate_candidate_links(
        {
            "support/trigger_ledger.csv": [trigger],
            "support/epoch_variant_counters.csv": [counter],
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_id", "0" * 64),
        ("dependence_cluster_id", "wrong:0"),
    ],
)
def test_verifier_rejects_noncanonical_trigger_identity(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    trigger, _ = canonical_trigger_row(tmp_path)
    trigger[field] = value
    with pytest.raises(VERIFIER.VerificationError, match="trigger_rows_domain"):
        VERIFIER.validate_csv_semantics("support/trigger_ledger.csv", [trigger])


def test_verifier_rejects_counter_candidate_link_mutation(tmp_path: Path) -> None:
    trigger, counter = canonical_trigger_row(tmp_path)
    counter["retained_candidate_id"] = "0" * 64
    with pytest.raises(
        VERIFIER.VerificationError,
        match="counter_retained_candidate_identity",
    ):
        VERIFIER.validate_candidate_links(
            {
                "support/trigger_ledger.csv": [trigger],
                "support/epoch_variant_counters.csv": [counter],
            }
        )


def write_ascii(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="ascii")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(VERIFIER.pretty_json_bytes(payload))


def test_production_v01_rejects_runner_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    init_git_repo(tmp_path)
    idea = "idea\n"
    plan = "plan\n"
    runner = "import numpy as np\ndef raw_onset_indices():\n    return np.load('x')\n"
    for path, content in (
        (VERIFIER.IDEA_PATH, idea),
        (VERIFIER.PLAN_PATH, plan),
        (VERIFIER.TASK_PATH, "task\n"),
        (VERIFIER.RUNNER_PATH, runner),
        (VERIFIER.VERIFIER_PATH, "verifier\n"),
        (VERIFIER.TEST_PATH, "tests\n"),
    ):
        write_ascii(tmp_path / path, content)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "implementation"], cwd=tmp_path, check=True
    )
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    monkeypatch.setattr(
        VERIFIER, "IDEA_SHA256", VERIFIER.sha256_file(tmp_path / VERIFIER.IDEA_PATH)
    )
    monkeypatch.setattr(
        VERIFIER, "PLAN_SHA256", VERIFIER.sha256_file(tmp_path / VERIFIER.PLAN_PATH)
    )
    with pytest.raises(
        VERIFIER.VerificationError,
        match="successor_np_load_callsite",
    ):
        VERIFIER.check_v01({"repo_root": tmp_path, "implementation_head": head})


def test_production_v02_rejects_terminal_git_artifact_mutation(
    tmp_path: Path,
) -> None:
    init_git_repo(tmp_path)
    implementation, consumption, _ = commit_claim_transition(tmp_path)
    write_json(tmp_path / VERIFIER.TERMINAL_RECEIPT_PATH, {"mutated": True})
    subprocess.run(
        ["git", "add", VERIFIER.TERMINAL_RECEIPT_PATH.as_posix()],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "wrong terminal message"],
        cwd=tmp_path,
        check=True,
    )
    terminal = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True
    ).strip()
    for tag, head in (
        (VERIFIER.IMPLEMENTATION_TAG, implementation),
        (VERIFIER.CONSUMPTION_TAG, consumption),
        (VERIFIER.TERMINAL_TAG, terminal),
    ):
        subprocess.run(
            ["git", "tag", "-a", tag, "-m", tag, head],
            cwd=tmp_path,
            check=True,
        )
    with pytest.raises(
        VERIFIER.VerificationError,
        match="terminal_commit_message",
    ):
        VERIFIER.check_v02(
            {
                "repo_root": tmp_path,
                "implementation_head": implementation,
                "consumption_head": consumption,
                "terminal_head": terminal,
                "implementation_tag": VERIFIER.IMPLEMENTATION_TAG,
                "consumption_tag": VERIFIER.CONSUMPTION_TAG,
                "terminal_tag": VERIFIER.TERMINAL_TAG,
            }
        )


def test_production_v03_rejects_claim_artifact_mutation(tmp_path: Path) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    identity_sha = "a" * 64
    claim = {
        "schema_version": 1,
        "task_id": VERIFIER.TASK_ID,
        "attempt_id": "attempt",
        "implementation_tag": VERIFIER.IMPLEMENTATION_TAG,
        "formal_argv": ["mutated"],
        "repo_root": str(tmp_path),
        "source_cache_root": str(tmp_path / "source"),
        "attempt_root": str(attempt),
        "idea_sha256": VERIFIER.IDEA_SHA256,
        "plan_sha256": VERIFIER.PLAN_SHA256,
        "task_sha256": identity_sha,
        "runner_sha256": identity_sha,
        "verifier_sha256": identity_sha,
        "tests_sha256": identity_sha,
        "controller_remote": VERIFIER.CONTROLLER_REMOTE,
        "controller_url": VERIFIER.CONTROLLER_URL,
        "controller_ref": VERIFIER.CONTROLLER_REF,
        "status": "ARMED_FOR_SINGLE_USE",
    }
    write_json(tmp_path / VERIFIER.CLAIMED_PATH, claim)
    identities = {
        path.as_posix(): {"sha256": identity_sha}
        for path in (
            VERIFIER.TASK_PATH,
            VERIFIER.RUNNER_PATH,
            VERIFIER.VERIFIER_PATH,
            VERIFIER.TEST_PATH,
        )
    }
    with pytest.raises(VERIFIER.VerificationError, match="claim_formal_argv"):
        VERIFIER.check_v03(
            {
                "repo_root": tmp_path,
                "attempt_root": attempt,
                "consumption_head": "b" * 40,
                "implementation_identities": identities,
            }
        )


def test_production_v04_rejects_attempt_child_artifact_mutation(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    write_ascii(attempt / "unexpected", "mutation\n")
    with pytest.raises(VERIFIER.VerificationError, match="attempt_children"):
        VERIFIER.check_v04({"attempt_root": attempt})


def test_production_v05_rejects_work_manifest_artifact_mutation(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    write_json(
        attempt / "work-manifest.json",
        {
            "schema_version": 2,
            "attempt_id": "attempt",
            "row_count": 0,
            "per_build_slice_count": 0,
            "rows": [],
            "tree_sha256": VERIFIER.canonical_sha([]),
        },
    )
    with pytest.raises(
        VERIFIER.VerificationError,
        match="work_manifest_values",
    ):
        VERIFIER.check_v05(
            {"attempt_root": attempt, "claim": {"attempt_id": "attempt"}}
        )


def test_production_v06_rejects_poison_attestation_artifact_mutation(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    write_json(
        attempt / "poison-attestation.json",
        {
            "task_id": "mutated",
            "hypothesis_id": "FIXED_CAUSAL_EPOCH_MSTATE_V2",
            "poison_output_root": str((attempt / "poison_p").resolve()),
            "source_inventory_sha256": "a" * 64,
            "cache_count": 29,
            "unconsumed_fields": list(VERIFIER.UNCONSUMED_FIELDS),
            "unconsumed_field_count": 15,
            "nonempty_unconsumed_field_instance_count": 435,
            "changed_unconsumed_field_instance_count": 435,
            "consumed_field_mismatch_count": 0,
            "caches": [],
        },
    )
    with pytest.raises(VERIFIER.VerificationError, match="poison_values"):
        VERIFIER.check_v06({"attempt_root": attempt, "inventory": []})


def test_production_v07_rejects_final17_artifact_mutation(tmp_path: Path) -> None:
    roots = {}
    for label in VERIFIER.ROOT_LABELS:
        root = tmp_path / label
        root.mkdir()
        roots[label] = root
    with pytest.raises(VERIFIER.VerificationError, match="final17_path_set"):
        VERIFIER.check_v07({"roots": roots})


def test_production_v08_rejects_manifest_artifact_mutation(tmp_path: Path) -> None:
    root = tmp_path / "A"
    for relative in VERIFIER.FINAL_PATHS:
        write_ascii(root / relative, "artifact\n")
    rows = VERIFIER.manifest_rows(
        root,
        tuple(path for path in VERIFIER.FINAL_PATHS if path != "run_manifest.json"),
    )
    write_json(
        root / "run_manifest.json",
        {"schema_version": 1, "artifact_count": 15, "artifacts": rows},
    )
    with pytest.raises(VERIFIER.VerificationError, match="manifest:A"):
        VERIFIER.check_v08({"roots": {"A": root}})


def test_production_v09_rejects_execution_evidence_artifact_mutation(
    tmp_path: Path,
) -> None:
    roots = {label: tmp_path / label for label in VERIFIER.ROOT_LABELS}
    for relative in VERIFIER.SEALED_PATHS:
        for root in roots.values():
            write_ascii(root / relative, f"{relative}\n")
    evidence = {
        "raw_a_b": VERIFIER.comparison(
            "RAW_11:A_vs_B", roots["A"], roots["B"], VERIFIER.RAW_PATHS
        ),
        "raw_a_p": VERIFIER.comparison(
            "RAW_11:A_vs_P", roots["A"], roots["P"], VERIFIER.RAW_PATHS
        ),
        "sealed_a_b": VERIFIER.comparison(
            "SEALED_15:A_vs_B", roots["A"], roots["B"], VERIFIER.SEALED_PATHS
        ),
        "sealed_a_p": VERIFIER.comparison(
            "SEALED_15:A_vs_P", roots["A"], roots["P"], VERIFIER.SEALED_PATHS
        ),
        "implementation_head": "mutated",
    }
    write_json(roots["A"] / "contracts/execution_evidence.json", evidence)
    with pytest.raises(
        VERIFIER.VerificationError,
        match="execution_implementation",
    ):
        VERIFIER.check_v09({"roots": roots, "implementation_head": "a" * 40})


def test_production_v10_rejects_attempt_result_artifact_mutation(
    tmp_path: Path,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    write_json(
        attempt / "attempt-result.json",
        {
            "schema_version": 1,
            "task_id": "mutated",
            "attempt_id": "attempt",
            "status": "COMPLETED",
            "phase": "FINAL_17_CLOSED",
            "exit_code": 0,
            "finished_at_utc": "2026-08-30T00:00:00.000000Z",
            "consumption_head": "a" * 40,
            "controller_ref": VERIFIER.CONTROLLER_REF,
            "attempt_lock_sha256": "a" * 64,
            "claimed_sha256": "a" * 64,
            "poison_attestation_sha256": "a" * 64,
            "instrumentation_evidence_sha256": "a" * 64,
            "work_manifest_sha256": "a" * 64,
            "work_tree_sha256": "a" * 64,
            "final_a_b": {},
            "final_a_p": {},
            "root_rows": [],
        },
    )
    with pytest.raises(VERIFIER.VerificationError, match="attempt_result_values"):
        VERIFIER.check_v10(
            {
                "attempt_root": attempt,
                "claim": {"attempt_id": "attempt"},
            }
        )


def test_production_v11_rejects_terminal_receipt_artifact_mutation(
    tmp_path: Path,
) -> None:
    write_json(
        tmp_path / VERIFIER.TERMINAL_RECEIPT_PATH,
        {
            "schema_version": 1,
            "task_id": "mutated",
            "attempt_id": "attempt",
            "status": "COMPLETED",
            "implementation_head": "a" * 40,
            "consumption_head": "b" * 40,
            "controller_remote": VERIFIER.CONTROLLER_REMOTE,
            "controller_ref": VERIFIER.CONTROLLER_REF,
            "attempt_result_sha256": "a" * 64,
            "attempt_lock_sha256": "a" * 64,
            "poison_attestation_sha256": "a" * 64,
            "instrumentation_evidence_sha256": "a" * 64,
            "work_manifest_sha256": "a" * 64,
            "work_tree_sha256": "a" * 64,
            "root_rows": [],
            "sealed_at_utc": "2026-08-30T00:00:00.000000Z",
        },
    )
    with pytest.raises(
        VERIFIER.VerificationError,
        match="terminal_receipt_values",
    ):
        VERIFIER.check_v11(
            {
                "repo_root": tmp_path,
                "attempt_result": {"attempt_id": "attempt"},
            }
        )


def test_production_v12_rejects_post_seal_artifact_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    write_ascii(attempt / "unexpected", "mutation\n")
    verifier_path = tmp_path / VERIFIER.VERIFIER_PATH
    write_ascii(verifier_path, "verifier\n")
    monkeypatch.setattr(VERIFIER, "git_text", lambda *args, **kwargs: "")
    with pytest.raises(
        VERIFIER.VerificationError,
        match="post_seal_attempt_children",
    ):
        VERIFIER.check_v12(
            {
                "repo_root": tmp_path,
                "attempt_root": attempt,
                "verifier_sha256": VERIFIER.sha256_file(verifier_path),
                "roots": {},
                "root_tree_sha": {},
            }
        )


def test_verifier_recomputes_gate_truth_and_precedence() -> None:
    gates = AUDIT.build_gates(passing_gate_values())
    payload = {
        "schema_version": 1,
        "gate_order": list(AUDIT.GATE_ORDER),
        "gates": gates,
        "first_failed_gate_id": None,
        "classification": AUDIT.classify(gates),
    }
    VERIFIER.validate_gate_contract(payload)
    mutated = copy.deepcopy(payload)
    mutated["gates"][0]["conditions"][0]["actual"] = False
    with pytest.raises(VERIFIER.VerificationError, match="gate_condition_truth"):
        VERIFIER.validate_gate_contract(mutated)
    mutated = copy.deepcopy(payload)
    mutated["gates"][0]["conditions"][0].update(
        {"status": "NOT_EVALUATED", "passed": None, "actual": None}
    )
    with pytest.raises(VERIFIER.VerificationError, match="gate_condition_precedence"):
        VERIFIER.validate_gate_contract(mutated)


def test_verifier_comparison_rejects_false_equal_flag(tmp_path: Path) -> None:
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "x").write_text("same\n", encoding="ascii")
    (right / "x").write_text("same\n", encoding="ascii")
    payload = VERIFIER.comparison("FINAL_17:A_vs_B", left, right, ("x",))
    payload["rows"][0]["equal"] = False
    payload["difference_count"] = 1
    with pytest.raises(VERIFIER.VerificationError, match="comparison_equal_value"):
        VERIFIER.validate_comparison(
            payload,
            expected_domain="FINAL_17:A_vs_B",
            expected_paths=("x",),
        )


@pytest.mark.parametrize("failure_index", range(13))
def test_verifier_v00_v12_mutation_short_circuit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_index: int,
) -> None:
    functions = []
    for index in range(13):
        if index == failure_index:
            functions.append(
                lambda context, index=index: (_ for _ in ()).throw(
                    VERIFIER.VerificationError(f"mutation:{index}")
                )
            )
        else:
            functions.append(lambda context: None)
    monkeypatch.setattr(VERIFIER, "CHECK_FUNCTIONS", tuple(functions))
    args = SimpleNamespace(
        repo_root=tmp_path,
        attempt_root=tmp_path / "attempt",
        result_out=tmp_path / "result.json",
        implementation_tag=VERIFIER.IMPLEMENTATION_TAG,
        consumption_tag=VERIFIER.CONSUMPTION_TAG,
        terminal_tag=VERIFIER.TERMINAL_TAG,
    )
    exit_code, payload = VERIFIER.verify_terminal(args)
    assert exit_code == 2
    assert payload["first_failure_code"] == VERIFIER.CHECK_IDS[failure_index]
    assert [row["status"] for row in payload["checks"][:failure_index]] == [
        "PASS"
    ] * failure_index
    assert payload["checks"][failure_index]["status"] == "FAIL"
    assert all(
        row["status"] == "NOT_EVALUATED"
        for row in payload["checks"][failure_index + 1 :]
    )


def test_parse_args_rejects_nonformal_modes() -> None:
    with pytest.raises(SystemExit):
        AUDIT.parse_args([])
    args = AUDIT.parse_args(
        [
            "--formal-attempt",
            "--repo-root",
            "/repo",
            "--source-cache-root",
            "/source",
            "--attempt-root",
            "/attempt",
        ]
    )
    assert args.formal_attempt is True


@pytest.mark.parametrize(
    "script",
    [
        AUDIT.RUNNER_PATH,
        AUDIT.VERIFIER_PATH,
    ],
)
def test_frozen_cli_entrypoints_import_from_repo_root(script: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo_root / script), "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
