from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parent
    / "skhynix_fresh_channel_consensus_mstate_a_minus1.py"
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
    assert AUDIT.stream_root(3, 30_000, 0, 0) != AUDIT.stream_root(
        4, 30_000, 0, 0
    )
    first = AUDIT.rng_for(3, 30_000, 7, 3).integers(0, 2, 100)
    second = AUDIT.rng_for(3, 30_000, 7, 3).integers(0, 2, 100)
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
    assert len(AUDIT.REQUIRED_NON_CACHE_ARTIFACTS) == 23
    assert "contracts/mstate_detector_contract.json" in (
        AUDIT.REQUIRED_NON_CACHE_ARTIFACTS
    )
    assert "support/orphan_strict_onset_by_date.csv" in (
        AUDIT.REQUIRED_NON_CACHE_ARTIFACTS
    )


def test_a_minus1_2_classification_matches_frozen_plan() -> None:
    gates = [
        {"gate_id": "A-1-0", "passed": True},
        {"gate_id": "A-1-1", "passed": True},
        {"gate_id": "A-1-2", "passed": False},
    ]
    assert AUDIT.classify(gates) == "Aminus1_mstate_integrity_failed"
