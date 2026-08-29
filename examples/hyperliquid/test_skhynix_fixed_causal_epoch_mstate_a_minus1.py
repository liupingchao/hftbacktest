from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parent
    / "skhynix_fixed_causal_epoch_mstate_a_minus1.py"
)
SPEC = importlib.util.spec_from_file_location("mstate_audit", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
REPO_ROOT = Path(__file__).resolve().parents[2]
PREDECESSOR = AUDIT.load_bound_predecessor(REPO_ROOT)


def synthetic_features(length: int = 400) -> dict[str, np.ndarray]:
    ts = np.arange(length, dtype=np.int64) * AUDIT.CHECKPOINT_NS
    segments = np.zeros(length, dtype=np.int32)
    fast = np.full((length, 3), 0.9, dtype=np.float64)
    medium = np.full((length, 3), 0.7, dtype=np.float64)
    one = np.ones(length, dtype=np.float64)
    return {
        "ts_ns": ts,
        "event_seq": np.arange(length, dtype=np.int64),
        "segment_id": segments,
        "segment_start_ts": np.zeros(length, dtype=np.int64),
        "ready": np.ones(length, dtype=bool),
        "valid_book": np.ones(length, dtype=bool),
        "activity_500": np.full(length, 100.0),
        "ratios_100": fast,
        "ratios_500": medium,
        "trade_total": one.copy(),
        "trade_signed": one.copy(),
        "bid_depletion": one.copy(),
        "ask_depletion": one.copy(),
        "ofi": one.copy(),
        "ofi_abs": one.copy(),
    }


def test_filter_family_is_exact_and_ordered() -> None:
    assert len(AUDIT.FILTERS) == 27
    assert AUDIT.FILTERS[0] == AUDIT.PrecisionFilter(
        "F000", 100, 200, 0.0
    )
    assert AUDIT.FILTERS[-1] == AUDIT.PrecisionFilter(
        "F222", 40, 800, 0.2
    )


def test_bound_predecessor_matches_all_frozen_callables() -> None:
    for name in AUDIT.PREDECESSOR_AST_SHA256:
        value = getattr(PREDECESSOR, name)
        assert callable(value)
        assert value.__name__ == name


def test_source_preflight_and_channel_specific_event_masks() -> None:
    features = synthetic_features(10)
    features["trade_total"][:] = 0
    features["bid_depletion"][:] = 0
    features["ask_depletion"][:] = 0
    features["ofi_abs"][:] = 0
    features["trade_total"][1] = 1
    features["ask_depletion"][2] = 1
    features["ofi_abs"][3] = 1
    invalid, masks = AUDIT.source_preflight(features)
    assert invalid == 0
    assert np.flatnonzero(masks["trade"]).tolist() == [1]
    assert np.flatnonzero(masks["depletion"]).tolist() == [2]
    assert np.flatnonzero(masks["ofi"]).tolist() == [3]


def test_source_preflight_rejects_all_registered_invalid_types() -> None:
    for field, value in (
        ("trade_total", -1.0),
        ("bid_depletion", np.nan),
        ("ask_depletion", np.inf),
        ("ofi_abs", -np.inf),
        ("trade_signed", np.nan),
        ("ofi", np.inf),
    ):
        features = synthetic_features(10)
        features[field][4] = value
        invalid, _ = AUDIT.source_preflight(features)
        assert invalid == 1


def test_no_new_evidence_is_no_update_despite_finite_ratios() -> None:
    features = synthetic_features(10)
    event_masks = {name: np.zeros(10, dtype=bool) for name in AUDIT.CHANNELS}
    actions = AUDIT.channel_actions(
        features=features,
        base_eligible=np.ones(10, dtype=bool),
        event_masks=event_masks,
        margin=0.0,
    )
    assert np.all(actions == AUDIT.NO_UPDATE)


def test_strong_threshold_boundaries_are_inclusive() -> None:
    features = synthetic_features(2)
    features["ratios_100"][:, :] = (0.5, -0.5, 0.49)
    features["ratios_500"][:, :] = (0.25, -0.25, 0.25)
    masks = {name: np.ones(2, dtype=bool) for name in AUDIT.CHANNELS}
    actions = AUDIT.channel_actions(
        features=features,
        base_eligible=np.ones(2, dtype=bool),
        event_masks=masks,
        margin=0.0,
    )
    assert np.all(actions[:, 0] == AUDIT.NEW_POS)
    assert np.all(actions[:, 1] == AUDIT.NEW_NEG)
    assert np.all(actions[:, 2] == AUDIT.NEW_NEUTRAL)


def test_new_invalid_clears_channel_memory() -> None:
    actions = np.full((5, 3), AUDIT.NO_UPDATE, dtype=np.int8)
    actions[0] = AUDIT.NEW_POS
    actions[2, 0] = AUDIT.NEW_INVALID
    states, _, _ = AUDIT.channel_memories(
        actions=actions,
        ts_ns=np.arange(5, dtype=np.int64) * AUDIT.CHECKPOINT_NS,
        segments=np.zeros(5, dtype=np.int32),
        ttl_ms=100,
    )
    assert states[1, 0] == 1
    assert states[2, 0] == 9
    assert states[3, 0] == 9


def test_neutral_overwrites_direction_immediately() -> None:
    actions = np.full((5, 3), AUDIT.NO_UPDATE, dtype=np.int8)
    actions[0] = AUDIT.NEW_POS
    actions[2, 0] = AUDIT.NEW_NEUTRAL
    states, _, diagnostics = AUDIT.channel_memories(
        actions=actions,
        ts_ns=np.arange(5, dtype=np.int64) * AUDIT.CHECKPOINT_NS,
        segments=np.zeros(5, dtype=np.int32),
        ttl_ms=100,
    )
    assert states[1, 0] == 1
    assert states[2, 0] == 0
    assert diagnostics["neutral_overwrite"][0] == 1


def test_no_update_does_not_refresh_ttl_and_expiry_is_strictly_after_ttl() -> None:
    actions = np.full((8, 3), AUDIT.NO_UPDATE, dtype=np.int8)
    actions[0] = AUDIT.NEW_POS
    states, ages, diagnostics = AUDIT.channel_memories(
        actions=actions,
        ts_ns=np.arange(8, dtype=np.int64) * AUDIT.CHECKPOINT_NS,
        segments=np.zeros(8, dtype=np.int32),
        ttl_ms=100,
    )
    assert states[5, 0] == 1
    assert ages[5, 0] == 100
    assert states[6, 0] == 9
    assert diagnostics["expiry"][0] == 1
    assert diagnostics["unauthorized_refresh"][0] == 0


def test_segment_boundary_and_global_invalid_clear_all_memories() -> None:
    actions = np.full((6, 3), AUDIT.NO_UPDATE, dtype=np.int8)
    actions[0] = AUDIT.NEW_POS
    segments = np.zeros(6, dtype=np.int32)
    segments[3:] = 1
    states, _, diagnostics = AUDIT.channel_memories(
        actions=actions,
        ts_ns=np.arange(6, dtype=np.int64) * AUDIT.CHECKPOINT_NS,
        segments=segments,
        ttl_ms=100,
    )
    assert np.all(states[2] == 1)
    assert np.all(states[3] == 9)
    assert np.sum(diagnostics["cross_segment_carry"]) == 0


def test_aggregate_mstate_requires_all_three_fresh_channels() -> None:
    memories = np.asarray(
        [[1, 1, 1], [-1, -1, -1], [1, 0, 1], [1, 1, 9]],
        dtype=np.int8,
    )
    assert AUDIT.aggregate_mstate(memories).tolist() == [
        AUDIT.M_SIGNAL_POS,
        AUDIT.M_SIGNAL_NEG,
        AUDIT.M_BACKGROUND,
        AUDIT.M_ABSTAIN,
    ]


def test_abstain_to_signal_cannot_form_anchor() -> None:
    state = np.full(20, AUDIT.M_BACKGROUND, dtype=np.int8)
    state[5] = AUDIT.M_ABSTAIN
    state[6] = AUDIT.M_SIGNAL_POS
    state[14] = AUDIT.M_SIGNAL_POS
    mask = AUDIT.natural_onset_mask(
        state, np.zeros(20, dtype=np.int32), 1
    )
    assert not mask[6]
    assert mask[14]


def test_filter_family_is_delete_only_from_common_candidate() -> None:
    features = synthetic_features(100)
    states_by_key = {}
    for ttl in AUDIT.TTLS_MS:
        for margin in AUDIT.MARGINS:
            states_by_key[(ttl, margin)] = np.full(
                100, AUDIT.M_BACKGROUND, dtype=np.int8
            )
    for state in states_by_key.values():
        state[20:31] = AUDIT.M_SIGNAL_POS
    states_by_key[(40, 0.2)][25] = AUDIT.M_BACKGROUND
    candidate = {"candidate_index": 20, "direction": 1}
    AUDIT.evaluate_filter_family(
        candidates=[candidate],
        features=features,
        family={"states_by_key": states_by_key},
    )
    assert "F000" in candidate["admitted_filter_ids"]
    assert "F202" not in candidate["admitted_filter_ids"]
    assert candidate["filter_cancel_reasons"]["F202"] == "consensus_lost"


def test_external_exposure_uses_additive_left_influence() -> None:
    features = synthetic_features(200)
    analysis = {
        "ts_ns": (
            AUDIT.CORE_OPEN_NS
            + np.arange(200, dtype=np.int64) * AUDIT.CHECKPOINT_NS
        ),
        "eligible_epoch_mask": np.ones(200, dtype=bool),
        "states_by_key": {
            (ttl, margin): np.full(
                200, AUDIT.M_BACKGROUND, dtype=np.int8
            )
            for ttl in AUDIT.TTLS_MS
            for margin in AUDIT.MARGINS
        }
    }
    comparison = np.ones(200, dtype=bool)
    candidate = 100
    item = AUDIT.FILTERS[0]
    earliest = candidate - (
        AUDIT.PRESTATE_MS + AUDIT.RATIO_HISTORY_MS + item.ttl_ms
    ) // AUDIT.CHECKPOINT_MS
    comparison[earliest] = False
    masks = AUDIT.exposure_masks(
        analysis=analysis,
        comparison=comparison,
        segments=features["segment_id"],
    )
    assert not masks[item.filter_id][candidate]


def test_rng_stream_banks_are_disjoint_and_deterministic() -> None:
    assert AUDIT.stream_root(5, 30_000, 0, 0) != AUDIT.stream_root(
        6, 30_000, 0, 0
    )
    first = AUDIT.rng_for(5, 30_000, 7, 3).integers(0, 2, 100)
    second = AUDIT.rng_for(5, 30_000, 7, 3).integers(0, 2, 100)
    assert np.array_equal(first, second)


def test_monotonicity_detects_stricter_only_candidate() -> None:
    candidate_sets = {item.filter_id: set() for item in AUDIT.FILTERS}
    cluster_sets = {item.filter_id: set() for item in AUDIT.FILTERS}
    candidate_sets["F000"] = {"shared"}
    candidate_sets["F100"] = {"shared", "violation"}
    cluster_sets["F000"] = {"cluster-shared"}
    cluster_sets["F100"] = {"cluster-shared", "cluster-violation"}
    rows = AUDIT.monotonicity_rows(candidate_sets, cluster_sets)
    target = next(
        row
        for row in rows
        if row["looser_filter_id"] == "F000"
        and row["stricter_filter_id"] == "F100"
    )
    assert target["candidate_subset_violations"] == 1
    assert target["cluster_subset_violations"] == 1


def test_zero_selection_exposure_produces_meta_abstain() -> None:
    dates = [f"2026-08-{day:02d}" for day in range(1, 10)]
    counts = np.zeros(
        (AUDIT.NULL_REPLICATES, len(dates), len(AUDIT.FILTERS)),
        dtype=np.int64,
    )
    exposure = np.zeros((len(dates), len(AUDIT.FILTERS)), dtype=np.int64)
    rows = AUDIT.select_filters(
        dates=dates,
        selection_counts=counts,
        selection_exposure_checkpoint_counts=exposure,
    )
    assert {row["fold_state"] for row in rows} == {"META_ABSTAIN"}


def test_required_non_cache_artifact_set_matches_frozen_plan() -> None:
    assert len(AUDIT.REQUIRED_NON_CACHE_ARTIFACTS) == 25
    assert "contracts/mstate_detector_contract.json" in (
        AUDIT.REQUIRED_NON_CACHE_ARTIFACTS
    )
    assert "support/orphan_strict_onset_by_date.csv" in (
        AUDIT.REQUIRED_NON_CACHE_ARTIFACTS
    )
    assert "support/epoch_support_by_date.csv" in (
        AUDIT.REQUIRED_NON_CACHE_ARTIFACTS
    )


def test_a_minus1_2_classification_matches_frozen_plan() -> None:
    gates = [
        {"gate_id": "A-1-0", "passed": True},
        {"gate_id": "A-1-1", "passed": True},
        {"gate_id": "A-1-2", "passed": False},
    ]
    assert AUDIT.classify(gates) == "Aminus1_mstate_integrity_failed"


def test_slice_actual_identity_excludes_later_segments() -> None:
    candidates = [
        {
            "capture_id": "capture",
            "epoch_id": 1,
            "candidate_ts_ns": 100,
            "candidate_event_seq": 10,
            "direction": 1,
            "segment_id": 0,
            "dependence_cluster_id": "s0",
            "admitted_filter_ids": ["F000"],
        },
        {
            "capture_id": "capture",
            "epoch_id": 2,
            "candidate_ts_ns": 120,
            "candidate_event_seq": 12,
            "direction": 1,
            "segment_id": 1,
            "dependence_cluster_id": "s1",
            "admitted_filter_ids": ["F000"],
        },
    ]
    actual = AUDIT.admitted_identity_set(
        candidates, epoch_ids={1, 2}, segment_id=0
    )
    assert actual == {("capture", 1, 100, 10, 1, "F000", "s0")}


def test_complete_epoch_and_segment_boundary_dispositions() -> None:
    features = synthetic_features(AUDIT.EXPECTED_EPOCH_CHECKPOINTS)
    rows, _, mask = AUDIT.epoch_support_ledger(
        capture_id="capture",
        research_date="2026-08-29",
        features=features,
    )
    assert len(rows) == 1
    assert rows[0]["disposition"] == "eligible"
    assert np.all(mask)

    features["segment_id"][1500:] = 1
    rows, _, mask = AUDIT.epoch_support_ledger(
        capture_id="capture",
        research_date="2026-08-29",
        features=features,
    )
    assert rows[0]["disposition"] == "segment_boundary"
    assert not np.any(mask)


def test_fixed_epoch_thinning_is_earliest_per_direction_and_core_is_half_open() -> None:
    features = synthetic_features(AUDIT.EXPECTED_EPOCH_CHECKPOINTS)
    state = np.full(len(features["ts_ns"]), AUDIT.M_BACKGROUND, dtype=np.int8)
    core_open = AUDIT.CORE_OPEN_NS // AUDIT.CHECKPOINT_NS
    core_close = AUDIT.CORE_CLOSE_NS // AUDIT.CHECKPOINT_NS
    state[100] = AUDIT.M_SIGNAL_POS
    state[core_open] = AUDIT.M_SIGNAL_POS
    state[core_open + 10] = AUDIT.M_SIGNAL_POS
    state[core_open + 20] = AUDIT.M_SIGNAL_NEG
    state[core_close] = AUDIT.M_SIGNAL_NEG
    family = {
        "states_by_key": {(100, 0.0): state},
        "ages_by_key": {
            (100, 0.0): np.zeros((len(state), 3), dtype=np.int64)
        },
    }
    candidates, counts, epoch_rows, _ = AUDIT.common_candidate_ledger(
        capture_id="capture",
        research_date="2026-08-29",
        features=features,
        family=family,
        predecessor=PREDECESSOR,
    )
    assert [
        (row["candidate_index"], row["direction"]) for row in candidates
    ] == [(core_open, 1), (core_open + 20, -1)]
    assert counts["same_epoch_direction_suppressed"] == 1
    assert {row["dependence_cluster_id"] for row in candidates} == {
        "capture:0"
    }
    assert epoch_rows[0]["edge_guard_omitted_pos_count"] == 1
    assert epoch_rows[0]["edge_guard_omitted_neg_count"] == 1


def candidate_diagnostic_fixture() -> dict[str, object]:
    epoch_id = 7
    start = epoch_id * AUDIT.EPOCH_NS
    return {
        "capture_id": "capture",
        "research_date": "2026-08-29",
        "epoch_id": epoch_id,
        "epoch_start_ns": start,
        "core_open_ns": start + AUDIT.CORE_OPEN_NS,
        "core_close_ns": start + AUDIT.CORE_CLOSE_NS,
        "segment_id": 0,
        "direction": 1,
        "candidate_id": "candidate",
        "candidate_ts_ns": start + AUDIT.CORE_OPEN_NS,
        "candidate_event_seq": 99,
        "dependence_cluster_id": f"capture:{epoch_id}",
        "channel_last_observation_ts_ns": {
            channel: start for channel in AUDIT.CHANNELS
        },
        "channel_last_observation_ages_ms": {
            channel: 0 for channel in AUDIT.CHANNELS
        },
        "common_prestate_background_count": AUDIT.PRESTATE_COUNT,
        "common_prestate_abstain_count": 0,
        "common_prestate_signal_count": 0,
        "admitted_filter_ids": [],
        "confirmations": {},
        "filter_cancel_reasons": {
            item.filter_id: "consensus_lost" for item in AUDIT.FILTERS
        },
    }


def test_candidate_diagnostics_populates_frozen_epoch_schema() -> None:
    _, rows, count, _ = AUDIT.candidate_diagnostics(
        [[candidate_diagnostic_fixture()]]
    )
    assert count == 1
    assert set(rows[0]) == set(AUDIT.CANDIDATE_LEDGER_FIELDS)
    assert rows[0]["epoch_id"] == 7
    assert rows[0]["epoch_start_ns"] == 7 * AUDIT.EPOCH_NS
    assert rows[0]["core_open_ns"] == (
        7 * AUDIT.EPOCH_NS + AUDIT.CORE_OPEN_NS
    )
    assert rows[0]["core_close_ns"] == (
        7 * AUDIT.EPOCH_NS + AUDIT.CORE_CLOSE_NS
    )
    AUDIT.validate_candidate_ledger_rows(rows)


def test_candidate_ledger_schema_fails_closed_on_missing_epoch_value() -> None:
    _, rows, _, _ = AUDIT.candidate_diagnostics(
        [[candidate_diagnostic_fixture()]]
    )
    rows[0]["epoch_id"] = ""
    with pytest.raises(
        AUDIT.AuditError, match="candidate_ledger_integer_type"
    ):
        AUDIT.validate_candidate_ledger_rows(rows)


def test_global_timestamp_order_fails_before_epoch_enumeration() -> None:
    features = synthetic_features(AUDIT.EXPECTED_EPOCH_CHECKPOINTS * 2)
    boundary = AUDIT.EXPECTED_EPOCH_CHECKPOINTS
    features["ts_ns"][boundary - 1], features["ts_ns"][boundary] = (
        features["ts_ns"][boundary],
        features["ts_ns"][boundary - 1],
    )
    with np.testing.assert_raises_regex(
        AUDIT.AuditError, "raw_timestamp_not_strictly_increasing"
    ):
        AUDIT.epoch_support_ledger(
            capture_id="capture",
            research_date="2026-08-29",
            features=features,
        )


def test_slice_source_hash_ignores_unconsumed_values() -> None:
    raw = {
        name: np.asarray([index], dtype=np.float64)
        for index, name in enumerate(sorted(AUDIT.ROW_ALIGNED_CACHE_FIELDS))
    }
    raw["ts_ns"] = np.asarray([0], dtype=np.int64)
    raw["event_seq"] = np.asarray([0], dtype=np.int64)
    raw["segment_id"] = np.asarray([0], dtype=np.int32)
    for name in AUDIT.METADATA_CACHE_FIELDS:
        raw[name] = np.asarray([1], dtype=np.int64)
    first = AUDIT.slice_source_sha256(
        raw, PREDECESSOR.CONSUMED_CACHE_FIELDS
    )
    raw["obi"][0] = 999.0
    raw["midpoint"][0] = -999.0
    second = AUDIT.slice_source_sha256(
        raw, PREDECESSOR.CONSUMED_CACHE_FIELDS
    )
    assert first == second
    raw["trade_total"][0] += 1
    assert first != AUDIT.slice_source_sha256(
        raw, PREDECESSOR.CONSUMED_CACHE_FIELDS
    )


def test_outcome_poison_changes_every_unconsumed_value_only(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    poisoned = tmp_path / "poison-cache"
    output = tmp_path / "poison-output"
    canonical.mkdir()
    cache_name = "2026-08-29_fixture.npz"
    raw = {
        name: np.asarray([index + 1, index + 2], dtype=np.int64)
        for index, name in enumerate(
            sorted(PREDECESSOR.ALLOWED_CACHE_FIELDS)
        )
    }
    np.savez_compressed(canonical / cache_name, **raw)
    inventory = [
        {
            "cache_name": cache_name,
            "size_bytes": 1,
            "row_count": 2,
            "cache_schema_version": 4,
            "cache_sha256": "source",
            "paired_determinism_verified": True,
            "cache_field_schema_verified": True,
        }
    ]
    attestation = AUDIT.materialize_poisoned_cache_set(
        canonical_cache_root=canonical,
        poisoned_cache_root=poisoned,
        poison_output_root=output,
        cache_inventory=inventory,
        allowed_fields=PREDECESSOR.ALLOWED_CACHE_FIELDS,
        consumed_fields=PREDECESSOR.CONSUMED_CACHE_FIELDS,
    )
    with np.load(poisoned / cache_name, allow_pickle=False) as handle:
        for name, original in raw.items():
            actual = handle[name]
            assert actual.dtype == original.dtype
            assert actual.shape == original.shape
            if name in PREDECESSOR.CONSUMED_CACHE_FIELDS:
                assert np.array_equal(actual, original)
            else:
                assert actual.tobytes() != original.tobytes()
    assert attestation["consumed_field_mismatch_count"] == 0
    assert (
        attestation["changed_unconsumed_field_instance_count"]
        == len(
            set(PREDECESSOR.ALLOWED_CACHE_FIELDS)
            - set(PREDECESSOR.CONSUMED_CACHE_FIELDS)
        )
    )
    evidence = AUDIT.verify_poison_attestation(
        poison_output_root=output,
        cache_inventory=inventory,
        allowed_fields=PREDECESSOR.ALLOWED_CACHE_FIELDS,
        consumed_fields=PREDECESSOR.CONSUMED_CACHE_FIELDS,
    )
    assert evidence["executed"] is True
    assert evidence["consumed_field_mismatch_count"] == 0


def test_outcome_poison_attestation_mutation_fails_closed(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical"
    poisoned = tmp_path / "poison-cache"
    output = tmp_path / "poison-output"
    canonical.mkdir()
    cache_name = "2026-08-29_fixture.npz"
    raw = {
        name: np.asarray([1], dtype=np.int64)
        for name in PREDECESSOR.ALLOWED_CACHE_FIELDS
    }
    np.savez_compressed(canonical / cache_name, **raw)
    inventory = [
        {
            "cache_name": cache_name,
            "size_bytes": 1,
            "row_count": 1,
            "cache_schema_version": 4,
            "cache_sha256": "source",
            "paired_determinism_verified": True,
            "cache_field_schema_verified": True,
        }
    ]
    AUDIT.materialize_poisoned_cache_set(
        canonical_cache_root=canonical,
        poisoned_cache_root=poisoned,
        poison_output_root=output,
        cache_inventory=inventory,
        allowed_fields=PREDECESSOR.ALLOWED_CACHE_FIELDS,
        consumed_fields=PREDECESSOR.CONSUMED_CACHE_FIELDS,
    )
    path = AUDIT.poison_attestation_path(output)
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["changed_unconsumed_field_instance_count"] -= 1
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    with pytest.raises(AUDIT.AuditError, match="poison_attestation_invalid"):
        AUDIT.verify_poison_attestation(
            poison_output_root=output,
            cache_inventory=inventory,
            allowed_fields=PREDECESSOR.ALLOWED_CACHE_FIELDS,
            consumed_fields=PREDECESSOR.CONSUMED_CACHE_FIELDS,
        )


def test_triad_comparison_detects_poison_artifact_mutation(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / name for name in ("a", "b", "poison")]
    for root in roots:
        target = root / "reports" / "result.json"
        target.parent.mkdir(parents=True)
        target.write_text('{"same":true}\n', encoding="ascii")
    AUDIT.compare_triad(PREDECESSOR, *roots, stage="preseal")
    (roots[2] / "reports" / "result.json").write_text(
        '{"same":false}\n', encoding="ascii"
    )
    with pytest.raises(AUDIT.AuditError, match="preseal_poison_output_mismatch"):
        AUDIT.compare_triad(PREDECESSOR, *roots, stage="preseal")
