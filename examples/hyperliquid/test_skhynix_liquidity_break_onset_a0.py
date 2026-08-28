from __future__ import annotations

import numpy as np
import pandas as pd

from examples.hyperliquid.skhynix_liquidity_break_onset_a0 import (
    BASELINE_MIN_CHECKPOINTS,
    BASELINE_SHIFT_CHECKPOINTS,
    BASELINE_WINDOW_CHECKPOINTS,
    _component_pair,
    _orientation_indices,
    _rolling_normalization,
    classify_a0,
    choose_direction,
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
