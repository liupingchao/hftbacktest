from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parent / "skhynix_flow_coherence_a_minus1_audit.py"
)
SPEC = importlib.util.spec_from_file_location("coherence_audit", MODULE_PATH)
assert SPEC and SPEC.loader
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)


def synthetic_features(length: int = 120) -> dict[str, np.ndarray]:
    ts = np.arange(length, dtype=np.int64) * AUDIT.CHECKPOINT_NS
    segment = np.zeros(length, dtype=np.int32)
    ratios = np.zeros((length, 3), dtype=np.float64)
    features = {
        "ts_ns": ts,
        "event_seq": np.arange(length, dtype=np.int32),
        "segment_id": segment,
        "segment_start_ts": np.zeros(length, dtype=np.int64),
        "ready": np.ones(length, dtype=bool),
        "valid_book": np.ones(length, dtype=bool),
        "activity_500": np.full(length, 100.0),
        "available_500": np.full(length, 3, dtype=np.int32),
        "trade_signed": np.zeros(length, dtype=np.float64),
    }
    for window in AUDIT.WINDOWS_MS:
        features[f"ratios_{window}"] = ratios.copy()
        features[f"available_{window}"] = np.full(
            length, 3, dtype=np.int32
        )
        features[f"composite_{window}"] = np.zeros(length)
        features[f"denominators_{window}"] = np.ones((length, 3))
    return features


def test_trade_is_required_for_coherence() -> None:
    features = synthetic_features()
    active = np.ones(len(features["ts_ns"]), dtype=bool)
    features["ratios_100"][:, 1:] = 0.9
    features["ratios_500"][:, 1:] = 0.9
    q = AUDIT.coherence_predicates(features, active, AUDIT.VARIANTS[0])
    assert not np.any(q[-1])
    assert not np.any(q[1])


def test_depth_split_is_ambiguous() -> None:
    features = synthetic_features()
    active = np.ones(len(features["ts_ns"]), dtype=bool)
    features["ratios_100"][:] = np.array([0.9, 0.9, -0.9])
    features["ratios_500"][:] = np.array([0.9, 0.9, 0.9])
    q = AUDIT.coherence_predicates(features, active, AUDIT.VARIANTS[0])
    assert not np.any(q[1])


def test_candidate_checkpoint_contributes_zero_exposure() -> None:
    features = synthetic_features()
    active = np.ones(len(features["ts_ns"]), dtype=bool)
    component = np.zeros(len(active), dtype=bool)
    horizon = np.zeros(len(active), dtype=bool)
    conflict = np.zeros(len(active), dtype=bool)
    component[:25] = True
    conflict[:25] = True
    q = {-1: np.zeros(len(active), dtype=bool), 1: np.zeros(len(active), dtype=bool)}
    q[1][25:32] = True
    anchors, _ = AUDIT.detect_provisional(
        capture_id="test",
        research_date="2026-08-28",
        features=features,
        active=active,
        q=q,
        component_conflict=component,
        horizon_conflict=horizon,
        conflict=conflict,
        variant=AUDIT.VARIANTS[0],
    )
    assert len(anchors) == 1
    assert anchors[0]["candidate_ts_ns"] == int(features["ts_ns"][25])
    assert anchors[0]["confirmation_ts_ns"] == int(features["ts_ns"][31])
    assert anchors[0]["persistence_exposure_ms"] == 120


def test_opposite_coherence_cancels_candidate() -> None:
    features = synthetic_features()
    active = np.ones(len(features["ts_ns"]), dtype=bool)
    conflict = np.zeros(len(active), dtype=bool)
    conflict[:25] = True
    q = {-1: np.zeros(len(active), dtype=bool), 1: np.zeros(len(active), dtype=bool)}
    q[1][25:28] = True
    q[-1][28] = True
    anchors, counts = AUDIT.detect_provisional(
        capture_id="test",
        research_date="2026-08-28",
        features=features,
        active=active,
        q=q,
        component_conflict=conflict,
        horizon_conflict=np.zeros(len(active), dtype=bool),
        conflict=conflict,
        variant=AUDIT.VARIANTS[0],
    )
    assert anchors == []
    assert counts["opposite_coherence"] == 1


def test_compact_null_detector_matches_full_detector() -> None:
    features = synthetic_features()
    active = np.ones(len(features["ts_ns"]), dtype=bool)
    conflict = np.zeros(len(active), dtype=bool)
    conflict[:25] = True
    q = {-1: np.zeros(len(active), dtype=bool), 1: np.zeros(len(active), dtype=bool)}
    q[1][25:32] = True
    full, _ = AUDIT.detect_provisional(
        capture_id="test",
        research_date="2026-08-28",
        features=features,
        active=active,
        q=q,
        component_conflict=conflict,
        horizon_conflict=np.zeros(len(active), dtype=bool),
        conflict=conflict,
        variant=AUDIT.VARIANTS[0],
        comparison_mask=np.ones(len(active), dtype=bool),
    )
    count, dwells = AUDIT.detect_compact_null(
        features=features,
        active=active,
        q=q,
        conflict=conflict,
        comparison_mask=np.ones(len(active), dtype=bool),
        variant=AUDIT.VARIANTS[0],
    )
    assert count == len(full)
    assert dwells == [float(anchor["coherence_dwell_ms"]) for anchor in full]


def test_type7_quantile() -> None:
    assert AUDIT.finite_quantile([0, 10], 0.95) == 9.5


def test_neutral_gap_prevents_direct_transition() -> None:
    features = synthetic_features()
    active = np.ones(len(features["ts_ns"]), dtype=bool)
    conflict = np.zeros(len(active), dtype=bool)
    conflict[10:24] = True
    q = {-1: np.zeros(len(active), dtype=bool), 1: np.zeros(len(active), dtype=bool)}
    q[1][25:32] = True
    anchors, _ = AUDIT.detect_provisional(
        capture_id="test",
        research_date="2026-08-28",
        features=features,
        active=active,
        q=q,
        component_conflict=conflict,
        horizon_conflict=np.zeros(len(active), dtype=bool),
        conflict=conflict,
        variant=AUDIT.VARIANTS[0],
    )
    assert anchors == []


def test_direction_path_permutation_is_deterministic_and_preserves_magnitude() -> None:
    length = AUDIT.NULL_BLOCK_NS // AUDIT.CHECKPOINT_NS
    features = synthetic_features(length)
    microblock_count = AUDIT.NULL_MICROBLOCK_NS // AUDIT.CHECKPOINT_NS
    pattern = np.concatenate(
        [
            np.full(microblock_count, 1.0 if block % 2 == 0 else -1.0)
            for block in range(length // microblock_count)
        ]
    )
    features["trade_signed"] = pattern.copy()
    for window in AUDIT.WINDOWS_MS:
        features[f"ratios_{window}"][:, 0] = pattern * 0.7
        features[f"denominators_{window}"][:, 0] = 10.0
    first, first_mask, first_diagnostics = AUDIT.permute_trade_direction_paths(
        features,
        np.ones(length, dtype=bool),
        np.random.Generator(np.random.PCG64(7)),
        AUDIT.NULL_MICROBLOCK_NS,
    )
    second, second_mask, second_diagnostics = AUDIT.permute_trade_direction_paths(
        features,
        np.ones(length, dtype=bool),
        np.random.Generator(np.random.PCG64(7)),
        AUDIT.NULL_MICROBLOCK_NS,
    )
    assert np.array_equal(first_mask, second_mask)
    assert first_diagnostics == second_diagnostics
    assert first_diagnostics["maximum_pair_label_count_difference"] == 0
    assert first_diagnostics["matched_pair_count"] == 5
    for window in AUDIT.WINDOWS_MS:
        first_trade = first[f"ratios_{window}"][:, 0]
        second_trade = second[f"ratios_{window}"][:, 0]
        assert np.array_equal(first_trade, second_trade)
        assert np.array_equal(
            np.abs(first_trade),
            np.abs(features[f"ratios_{window}"][:, 0]),
        )


def test_classification_prefers_nuisance_failure() -> None:
    gates = [
        {"gate_id": "A-1-0", "passed": True, "conditions": []},
        {"gate_id": "A-1-1", "passed": True, "conditions": []},
        {
            "gate_id": "A-1-2",
            "passed": False,
            "conditions": [
                {"condition": "cooldown_zone_share_le_0_10", "passed": False}
            ],
        },
    ]
    assert AUDIT.classify(gates) == "Aminus1_nuisance_dominated"


def test_cache_field_schema_fails_closed_on_extra_field() -> None:
    AUDIT.validate_cache_field_names(
        sorted(AUDIT.ALLOWED_CACHE_FIELDS), "accepted.npz"
    )
    with np.testing.assert_raises(AUDIT.AuditError):
        AUDIT.validate_cache_field_names(
            sorted((*AUDIT.ALLOWED_CACHE_FIELDS, "future_return_500ms")),
            "hostile.npz",
        )


def test_feature_window_boundary_audit_detects_cross_segment_history() -> None:
    features = synthetic_features(120)
    features["ready"][:] = False
    features["ready"][99:] = True
    assert AUDIT.feature_window_boundary_violations(features) == 0
    features["segment_id"][105:] = 1
    assert AUDIT.feature_window_boundary_violations(features) == 15


def test_slice_invariance_records_variant_identity_and_metrics() -> None:
    features = synthetic_features(32_000)
    active = np.zeros(len(features["ts_ns"]), dtype=bool)
    q = {
        -1: np.zeros(len(active), dtype=bool),
        1: np.zeros(len(active), dtype=bool),
    }
    conflict = np.zeros(len(active), dtype=bool)
    rows = AUDIT.slice_invariance_rows(
        capture_id="test",
        research_date="2026-08-28",
        features=features,
        active=active,
        q=q,
        component_conflict=conflict,
        horizon_conflict=conflict,
        conflict=conflict,
        full_anchors=[],
        variant=AUDIT.VARIANTS[-1],
    )
    assert len(rows) == 1
    assert rows[0]["variant_id"] == "V8"
    assert rows[0]["identity_exact"]
    assert rows[0]["metrics_exact"]
    assert rows[0]["exact_match"]


def test_determinism_pair_requires_distinct_roots(tmp_path: Path) -> None:
    with np.testing.assert_raises(AUDIT.AuditError):
        AUDIT.finalize_existing_pair(tmp_path, tmp_path, tmp_path)
