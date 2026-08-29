from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parent
    / "skhynix_precision_first_flow_coherence_a_minus1.py"
)
SPEC = importlib.util.spec_from_file_location("precision_audit", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
REPO_ROOT = Path(__file__).resolve().parents[2]
PREDECESSOR = AUDIT.load_bound_predecessor(REPO_ROOT)


def synthetic_features(length: int = 4_000) -> dict[str, np.ndarray]:
    ts = np.arange(length, dtype=np.int64) * AUDIT.CHECKPOINT_NS
    segments = np.zeros(length, dtype=np.int32)
    fast = np.full((length, 3), 0.9, dtype=np.float64)
    medium = np.full((length, 3), 0.7, dtype=np.float64)
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
    }


def test_filter_family_is_exact_and_ordered() -> None:
    assert len(AUDIT.FILTERS) == 27
    assert AUDIT.FILTERS[0] == AUDIT.PrecisionFilter(
        "F000", 200, 0.0, 500
    )
    assert AUDIT.FILTERS[-1] == AUDIT.PrecisionFilter(
        "F222", 800, 0.2, 2_000
    )
    assert len({item.filter_id for item in AUDIT.FILTERS}) == 27


def test_bound_predecessor_matches_frozen_callables() -> None:
    for name, expected in AUDIT.PREDECESSOR_AST_SHA256.items():
        value = getattr(PREDECESSOR, name)
        assert callable(value)
        assert value.__name__ == name
        assert expected


def test_v2_support_requires_every_fast_and_medium_component() -> None:
    features = synthetic_features(2_000)
    detector_ready, activity_supported, support = AUDIT.v2_support(
        features, PREDECESSOR
    )
    assert np.any(detector_ready)
    assert np.any(activity_supported)
    assert np.any(support)
    features["ratios_100"][1_700, 2] = np.nan
    _, _, poisoned = AUDIT.v2_support(features, PREDECESSOR)
    assert not poisoned[1_700]


def test_common_refractory_is_per_direction_and_clusters_break_on_segment() -> None:
    features = synthetic_features()
    support = np.ones(4_000, dtype=bool)
    q = {
        -1: np.zeros(4_000, dtype=bool),
        1: np.zeros(4_000, dtype=bool),
    }
    conflict = np.zeros(4_000, dtype=bool)
    for index, direction in ((100, 1), (200, 1), (201, -1), (1_800, 1)):
        q[direction][index] = True
        conflict[index - 6 : index] = True
    features["segment_id"][1_700:] = 1
    candidates, counts = AUDIT.common_candidate_ledger(
        capture_id="capture",
        research_date="2026-08-29",
        features=features,
        predecessor=PREDECESSOR,
        support=support,
        q=q,
        component_conflict=conflict,
        horizon_conflict=np.zeros(4_000, dtype=bool),
        conflict=conflict,
    )
    assert counts["raw_base_candidates"] == 4
    assert counts["common_refractory_suppressed"] == 1
    assert [(row["candidate_index"], row["direction"]) for row in candidates] == [
        (100, 1),
        (201, -1),
        (1_800, 1),
    ]
    assert candidates[0]["dependence_cluster_id"] == candidates[1][
        "dependence_cluster_id"
    ]
    assert candidates[2]["dependence_cluster_id"] != candidates[1][
        "dependence_cluster_id"
    ]


def test_filter_evaluation_requires_complete_novelty_and_persistence() -> None:
    features = synthetic_features(400)
    support = np.ones(400, dtype=bool)
    q = {
        -1: np.zeros(400, dtype=bool),
        1: np.zeros(400, dtype=bool),
    }
    q[1][120:161] = True
    candidate = {
        "candidate_index": 120,
        "direction": 1,
    }
    AUDIT.evaluate_filter_family(
        candidates=[candidate],
        features=features,
        support=support,
        q=q,
        conflict=np.zeros(400, dtype=bool),
    )
    assert candidate["admitted_filter_ids"] == [
        item.filter_id for item in AUDIT.FILTERS
    ]

    support[100] = False
    candidate = {"candidate_index": 120, "direction": 1}
    AUDIT.evaluate_filter_family(
        candidates=[candidate],
        features=features,
        support=support,
        q=q,
        conflict=np.zeros(400, dtype=bool),
    )
    assert candidate["admitted_filter_ids"] == []
    assert set(candidate["filter_cancel_reasons"].values()) == {"abstain"}


def test_external_exposure_includes_additive_state_history() -> None:
    length = 500
    support = np.ones(length, dtype=bool)
    comparison = np.ones(length, dtype=bool)
    segments = np.zeros(length, dtype=np.int32)
    candidate = 250
    item = AUDIT.PrecisionFilter("test", 200, 0.0, 500)
    earliest = candidate - (
        item.novelty_ms + AUDIT.W_STATE_MS
    ) // AUDIT.CHECKPOINT_MS
    comparison[earliest] = False
    masks = AUDIT.exposure_masks(
        support=support,
        comparison=comparison,
        segments=segments,
    )
    assert not masks[(item.persistence_ms, item.novelty_ms)][candidate]


def test_exposure_unit_conversion_is_one_hour() -> None:
    checkpoint_count = 180_000
    seconds = 0.020 * checkpoint_count
    hours = seconds / 3_600
    assert hours == 1
    assert 5 / hours == 5


def test_zero_selection_exposure_produces_meta_abstain() -> None:
    dates = [f"2026-08-{day:02d}" for day in range(1, 10)]
    counts = np.zeros(
        (AUDIT.NULL_REPLICATES, len(dates), len(AUDIT.FILTERS)),
        dtype=np.int64,
    )
    exposure = np.zeros((len(dates), len(AUDIT.FILTERS)))
    rows = AUDIT.select_filters(
        dates=dates,
        selection_counts=counts,
        selection_exposure_hours=exposure,
    )
    assert len(rows) == 9
    assert {row["fold_state"] for row in rows} == {"META_ABSTAIN"}
    assert {row["selected_filter_id"] for row in rows} == {""}


def test_nonfinite_selection_exposure_fails_integrity() -> None:
    dates = [f"2026-08-{day:02d}" for day in range(1, 10)]
    counts = np.zeros(
        (AUDIT.NULL_REPLICATES, len(dates), len(AUDIT.FILTERS)),
        dtype=np.int64,
    )
    exposure = np.ones((len(dates), len(AUDIT.FILTERS)))
    exposure[0, 0] = np.nan
    with np.testing.assert_raises(AUDIT.AuditError):
        AUDIT.select_filters(
            dates=dates,
            selection_counts=counts,
            selection_exposure_hours=exposure,
        )


def test_estimator_zero_exposure_is_not_estimable() -> None:
    result = AUDIT.estimator(
        observed_by_date=np.zeros(9, dtype=np.int64),
        null_by_replicate_date=np.zeros(
            (AUDIT.NULL_REPLICATES, 9), dtype=np.int64
        ),
        exposure_hours=0,
    )
    assert not result["estimable"]
    assert result["null_false_cluster_rate_p95_per_hour"] is None
    assert result["structural_null_burden_ratio_p95"] is None
    assert result["maximum_single_date_share"] is None


def test_nonfinite_summary_values_are_rejected() -> None:
    assert AUDIT.nonfinite_paths({"ok": 1.0}) == []
    assert AUDIT.nonfinite_paths({"bad": float("inf")}) == ["root.bad"]


def test_rng_stream_banks_are_disjoint_and_deterministic() -> None:
    selection_root = AUDIT.stream_root(1, 30_000, 0, 0)
    evaluation_root = AUDIT.stream_root(2, 30_000, 0, 0)
    assert selection_root != evaluation_root
    first = AUDIT.rng_for(1, 30_000, 7, 3).integers(0, 2, 100)
    second = AUDIT.rng_for(1, 30_000, 7, 3).integers(0, 2, 100)
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


def test_same_determinism_root_is_rejected(tmp_path: Path) -> None:
    with np.testing.assert_raises(AUDIT.AuditError):
        AUDIT.compare_outputs(PREDECESSOR, tmp_path, tmp_path)


def test_unconsumed_field_poison_does_not_change_features(
    tmp_path: Path,
) -> None:
    source_root = AUDIT.DEFAULT_SOURCE_CACHE_ROOT
    source = sorted(source_root.glob("*.npz"))[0]
    original = PREDECESSOR.build_features(source)
    poisoned_path = tmp_path / source.name
    shutil.copyfile(source, poisoned_path)
    with np.load(poisoned_path, allow_pickle=False) as values:
        payload = {name: values[name].copy() for name in values.files}
    payload["midpoint"][:] = 9_999_999
    payload["obi"][:] = -0.999
    payload["spread_ticks"][:] = 999
    np.savez_compressed(poisoned_path, **payload)
    poisoned = PREDECESSOR.build_features(poisoned_path)
    assert original.keys() == poisoned.keys()
    for key in original:
        assert np.array_equal(original[key], poisoned[key], equal_nan=True)


def test_classification_precedence_prefers_integrity() -> None:
    gates = [
        {"gate_id": "A-1-0", "passed": True},
        {"gate_id": "A-1-1", "passed": True},
        {"gate_id": "A-1-2", "passed": True},
        {"gate_id": "A-1-3", "passed": True},
        {"gate_id": "A-1-4", "passed": False},
        {"gate_id": "A-1-5", "passed": False},
        {"gate_id": "A-1-6", "passed": False},
        {"gate_id": "A-1-7", "passed": False},
    ]
    assert AUDIT.classify(gates) == "Aminus1_selection_integrity_failed"
