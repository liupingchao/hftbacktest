from __future__ import annotations

import copy
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


def passing_gate_summary() -> dict[str, object]:
    estimator = {
        **AUDIT.exposure_units(180_000),
        "observed_cluster_count": 30,
        "null_cluster_count_p95": 1.0,
        "null_false_cluster_rate_p95_per_hour": 0.05,
        "structural_null_burden_ratio_p95": 0.05,
        "count_tail_p": 0.001,
        "represented_date_count": 4,
        "maximum_single_date_share": 0.25,
        "dates_above_date_null_p90": 4,
        "estimable": True,
    }
    return {
        "plan_sha_verified": True,
        "predecessor_binding_verified": True,
        "source_cache_closure": True,
        "zero_outcome_boundary": True,
        "unexpected_field_count": 0,
        "determinism_evidence": {
            "preseal_difference_count": 0,
            "pending_difference_count": 0,
            "final_difference_count": 0,
        },
        "null_admissibility": {
            "replicate_count_exact": True,
            "minimum_distinct_fingerprints": 199,
            "stream_identity_overlap": 0,
            "invariant_mismatches": 0,
            "minimum_date_pair_count": 4,
            "maximum_date_p95_joint_distance": 0.5,
        },
        "integrity": {
            "mstate_partition_violations": 0,
            "action_partition_violations": 0,
            "new_invalid_action_count": 0,
            "ttl_refresh_violations": 0,
            "cross_segment_memory_carry": 0,
            "anchor_contract_violations": 0,
            "feature_boundary_violations": 0,
            "monotonicity_violations": 0,
            "slice_invariance_mismatches": 0,
            "represented_slice_date_count": 4,
            "distinct_comparable_epoch_count": 30,
            "compared_support_checkpoint_count": 1,
            "common_cluster_maximum_5s_burst": 1,
            "fold_count": 9,
            "observed_selection_access": 0,
            "null_bank_overlap": 0,
            "numeric_integrity_violations": 0,
        },
        "estimators": {
            duration: copy.deepcopy(estimator)
            for duration in ("10000", "30000", "60000")
        },
        "raw": {
            **AUDIT.exposure_units(180_000),
            "observed_cluster_count": 1,
            "cluster_rate_per_hour": 1.0,
            "raw_supported_epoch_count": 100,
            "occupied_epoch_count": 10,
            "occupied_supported_epoch_share": 0.1,
            "structurally_eligible_epoch_count": 100,
            "structurally_occupied_epoch_count": 10,
            "structurally_occupied_epoch_share": 0.1,
            "occupied_subset_violation_count": 0,
            "structurally_occupied_subset_violation_count": 0,
            "raw_supported_epoch_identity_sha256": "a" * 64,
            "occupied_epoch_identity_sha256": "b" * 64,
            "structurally_eligible_epoch_identity_sha256": "c" * 64,
        },
    }


def gates_by_id(summary: dict[str, object]) -> dict[str, dict[str, object]]:
    return {
        row["gate_id"]: row
        for row in AUDIT.build_gates(
            summary, deterministic_build=True
        )
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


@pytest.mark.parametrize(
    ("mutation", "expected"),
    (
        ("partial_start", "partial_capture_start"),
        ("partial_end", "partial_capture_end"),
        ("missing", "missing_checkpoint"),
        ("off_grid", "irregular_checkpoint"),
        ("empty_intermediate", "missing_checkpoint"),
    ),
)
def test_epoch_disposition_mutations_fail_closed(
    mutation: str, expected: str
) -> None:
    features = synthetic_features(AUDIT.EXPECTED_EPOCH_CHECKPOINTS)
    if mutation == "partial_start":
        features = {
            name: value[1:].copy() for name, value in features.items()
        }
    elif mutation == "partial_end":
        features = {
            name: value[:-1].copy() for name, value in features.items()
        }
    elif mutation == "missing":
        features = {
            name: np.delete(value, 1000) for name, value in features.items()
        }
    elif mutation == "off_grid":
        features["ts_ns"][1000] += 1
    elif mutation == "empty_intermediate":
        features = synthetic_features(
            AUDIT.EXPECTED_EPOCH_CHECKPOINTS * 3
        )
        keep = (features["ts_ns"] // AUDIT.EPOCH_NS) != 1
        features = {
            name: value[keep].copy() for name, value in features.items()
        }
    rows, _, mask = AUDIT.epoch_support_ledger(
        capture_id="capture",
        research_date="2026-08-29",
        features=features,
    )
    matching = (
        next(row for row in rows if row["epoch_id"] == 1)
        if mutation == "empty_intermediate"
        else rows[0]
    )
    assert matching["disposition"] == expected
    assert matching["segment_id"] == ""
    assert not np.any(
        mask[
            (features["ts_ns"] // AUDIT.EPOCH_NS)
            == int(matching["epoch_id"])
        ]
    )


def test_duplicate_timestamp_fails_before_disposition_enumeration() -> None:
    features = synthetic_features(AUDIT.EXPECTED_EPOCH_CHECKPOINTS)
    features["ts_ns"][1000] = features["ts_ns"][999]
    with pytest.raises(
        AUDIT.AuditError, match="raw_timestamp_not_strictly_increasing"
    ):
        AUDIT.epoch_support_ledger(
            capture_id="capture",
            research_date="2026-08-29",
            features=features,
        )


@pytest.mark.parametrize("second_direction", (1, -1))
def test_core_reset_isolated_from_same_or_opposite_direction_onsets(
    second_direction: int,
) -> None:
    features = synthetic_features(AUDIT.EXPECTED_EPOCH_CHECKPOINTS)
    core_open = AUDIT.CORE_OPEN_NS // AUDIT.CHECKPOINT_NS
    reset_index = core_open + 300
    features["segment_id"][reset_index:] = 1
    state = np.full(len(features["ts_ns"]), AUDIT.M_BACKGROUND, dtype=np.int8)
    state[core_open + 100] = AUDIT.M_SIGNAL_POS
    state[reset_index + 100] = second_direction
    family = {
        "states_by_key": {(100, 0.0): state},
        "ages_by_key": {
            (100, 0.0): np.zeros((len(state), 3), dtype=np.int64)
        },
    }
    candidates, counts, epoch_rows, eligible = (
        AUDIT.common_candidate_ledger(
            capture_id="capture",
            research_date="2026-08-29",
            features=features,
            family=family,
            predecessor=PREDECESSOR,
        )
    )
    assert candidates == []
    assert counts["fixed_epoch_admitted"] == 0
    assert epoch_rows[0]["disposition"] == "segment_boundary"
    assert epoch_rows[0]["dependence_cluster_id"] == ""
    assert not np.any(eligible)


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


def test_support_identity_hash_locks_numeric_epoch_order_and_state() -> None:
    epoch_ids = {2, 10}
    ts = np.asarray(
        [
            epoch_id * AUDIT.EPOCH_NS + AUDIT.CORE_OPEN_NS
            for epoch_id in sorted(epoch_ids)
        ],
        dtype=np.int64,
    )
    analysis = {
        "ts_ns": ts,
        "states_by_key": {
            (item.ttl_ms, item.margin): np.asarray(
                [AUDIT.M_BACKGROUND, AUDIT.M_SIGNAL_POS], dtype=np.int8
            )
            for item in AUDIT.FILTERS
        },
    }
    count, checkpoints, actual_sha = AUDIT.support_identity(
        analysis, epoch_ids, "capture"
    )
    expected = [
        (
            "capture",
            epoch_id,
            int(ts[index]),
            filter_index,
            int(
                analysis["states_by_key"][(item.ttl_ms, item.margin)][
                    index
                ]
            ),
        )
        for index, epoch_id in enumerate(sorted(epoch_ids))
        for filter_index, item in enumerate(AUDIT.FILTERS)
    ]
    assert count == 2 * len(AUDIT.FILTERS)
    assert checkpoints == 2
    assert actual_sha == AUDIT.canonical_sha(expected)
    assert actual_sha != AUDIT.canonical_sha(list(reversed(expected)))

    analysis["states_by_key"][(100, 0.0)][1] = AUDIT.M_SIGNAL_NEG
    assert AUDIT.support_identity(analysis, epoch_ids, "capture")[2] != (
        actual_sha
    )


@pytest.mark.parametrize(
    ("mutation", "expected_gate"),
    (
        ("mstate", "A-1-2"),
        ("null_nonfinite", "A-1-3"),
        ("spoofed_occupancy_share", "A-1-4"),
        ("malformed_occupancy_hash", "A-1-4"),
        ("structural_subset", "A-1-4"),
    ),
)
def test_gate_mutations_fail_at_frozen_precedence(
    mutation: str, expected_gate: str
) -> None:
    summary = passing_gate_summary()
    if mutation == "mstate":
        summary["integrity"]["mstate_partition_violations"] = 1
    elif mutation == "null_nonfinite":
        summary["null_admissibility"][
            "maximum_date_p95_joint_distance"
        ] = np.nan
    elif mutation == "spoofed_occupancy_share":
        summary["raw"]["occupied_supported_epoch_share"] = 0.09
    elif mutation == "malformed_occupancy_hash":
        summary["raw"]["occupied_epoch_identity_sha256"] = "not-a-hash"
    elif mutation == "structural_subset":
        summary["raw"]["structurally_eligible_epoch_count"] = 9
        summary["raw"]["structurally_occupied_epoch_count"] = 10
        summary["raw"]["structurally_occupied_epoch_share"] = 10 / 9
    gates = gates_by_id(summary)
    assert gates[expected_gate]["status"] == "FAIL"
    failed = False
    for gate_id in (
        "A-1-0",
        "A-1-1",
        "A-1-2",
        "A-1-3",
        "A-1-4",
        "A-1-5",
        "A-1-6",
        "A-1-7",
    ):
        row = gates[gate_id]
        if failed:
            assert row["status"] == "NOT_EVALUATED"
            assert row["passed"] is None
            assert all(
                condition["status"] == "NOT_EVALUATED"
                and condition["passed"] is None
                and condition["actual"] is None
                and condition["required"]
                for condition in row["conditions"]
            )
        elif gate_id == expected_gate:
            failed = True


@pytest.mark.parametrize(
    ("failed_gate", "field", "value"),
    (
        ("A-1-5", "observed_cluster_count", 29),
        ("A-1-6", "null_false_cluster_rate_p95_per_hour", 0.11),
    ),
)
def test_late_gate_failure_keeps_a_minus1_7_not_evaluated(
    failed_gate: str, field: str, value: float
) -> None:
    summary = passing_gate_summary()
    summary["estimators"]["30000"][field] = value
    gates = gates_by_id(summary)
    assert gates[failed_gate]["status"] == "FAIL"
    assert gates["A-1-7"]["status"] == "NOT_EVALUATED"
    assert all(
        condition["passed"] is None and condition["actual"] is None
        for condition in gates["A-1-7"]["conditions"]
    )


def test_zero_occupancy_with_positive_denominator_is_numeric_zero() -> None:
    summary = passing_gate_summary()
    summary["raw"]["occupied_epoch_count"] = 0
    summary["raw"]["occupied_supported_epoch_share"] = 0.0
    summary["raw"]["structurally_occupied_epoch_count"] = 0
    summary["raw"]["structurally_occupied_epoch_share"] = 0.0
    assert AUDIT.numeric_integrity_violations(summary) == 0


def test_manifest_self_exclusion_and_exact_25_path_mutations(
    tmp_path: Path,
) -> None:
    for relative in AUDIT.REQUIRED_NON_CACHE_ARTIFACTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="ascii")
    produced = AUDIT.non_cache_artifact_paths(tmp_path)
    assert produced == AUDIT.REQUIRED_NON_CACHE_ARTIFACTS
    manifest = PREDECESSOR.artifact_manifest(tmp_path)
    manifest_paths = {row["path"] for row in manifest["artifacts"]}
    assert "run_manifest.json" not in manifest_paths
    assert manifest_paths == (
        AUDIT.REQUIRED_NON_CACHE_ARTIFACTS - {"run_manifest.json"}
    )
    assert manifest["artifact_count"] == 24

    extra = tmp_path / "unexpected.json"
    extra.write_text("{}\n", encoding="ascii")
    assert AUDIT.non_cache_artifact_paths(tmp_path) != (
        AUDIT.REQUIRED_NON_CACHE_ARTIFACTS
    )
    extra.unlink()
    (tmp_path / next(iter(AUDIT.REQUIRED_NON_CACHE_ARTIFACTS))).unlink()
    assert AUDIT.non_cache_artifact_paths(tmp_path) != (
        AUDIT.REQUIRED_NON_CACHE_ARTIFACTS
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


def test_outcome_access_payload_tracks_pending_and_final_poison_state() -> None:
    pending_evidence = {"stage": "pending", "executed": False}
    pending = AUDIT.outcome_access_payload(
        summary={
            "zero_outcome_boundary": False,
            "outcome_poison_evidence": pending_evidence,
        },
        consumed_cache_fields={"ts_ns", "event_seq"},
    )
    assert pending["poisoned_unconsumed_fields_change_output"] is None
    assert pending["poison_protocol"] == pending_evidence

    final_evidence = {
        "stage": "final",
        "executed": True,
        "final_difference_count": 0,
    }
    final = AUDIT.outcome_access_payload(
        summary={
            "zero_outcome_boundary": True,
            "outcome_poison_evidence": final_evidence,
        },
        consumed_cache_fields={"ts_ns", "event_seq"},
    )
    assert final["poisoned_unconsumed_fields_change_output"] is False
    assert final["poison_protocol"] == final_evidence
