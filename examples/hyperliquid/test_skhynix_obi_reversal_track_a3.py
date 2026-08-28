from __future__ import annotations

import numpy as np

from examples.hyperliquid.skhynix_obi_reversal_track_a3 import (
    Entry,
    HazardModel,
    Standardizer,
    build_risk_design,
    cumulative_incidence,
    date_block_bootstrap,
    elapsed_time_bin,
    fit_hazard_model,
    multinomial_objective,
)


def _entry(
    entry_id: str,
    event_time_ms: int,
    cause_code: int,
    reversal: int,
) -> Entry:
    return Entry(
        entry_id=entry_id,
        entry_type="reversal" if reversal else "control",
        reversal_indicator=reversal,
        capture_id="fixture",
        research_date="2026-08-01",
        role="historical_method_development",
        side=1,
        event_type="follow" if cause_code == 1 else "fail",
        cause_code=cause_code,
        event_time_ms=event_time_ms,
        features=np.linspace(0.1, 1.9, 19),
    )


def test_elapsed_time_bins_are_frozen() -> None:
    assert [elapsed_time_bin(value) for value in (100, 200, 500)] == [0, 1, 1]
    assert [elapsed_time_bin(value) for value in (600, 1000, 1100)] == [2, 2, 3]
    assert [elapsed_time_bin(value) for value in (2000, 2100, 5000)] == [3, 4, 4]
    assert elapsed_time_bin(5100) == 5


def test_risk_expansion_and_H1_only_R_column() -> None:
    entries = [_entry("a", 100, 1, 0), _entry("b", 300, 2, 1)]
    standardizer = Standardizer(np.zeros(19), np.ones(19))
    h0, y0, index0 = build_risk_design(
        entries,
        standardizer,
        include_reversal=False,
    )
    h1, y1, index1 = build_risk_design(
        entries,
        standardizer,
        include_reversal=True,
    )
    assert h0.shape == (4, 25)
    assert h1.shape == (4, 26)
    assert np.array_equal(h0, h1[:, :-1])
    assert np.array_equal(y0, np.asarray([1, 0, 0, 2]))
    assert np.array_equal(y0, y1)
    assert np.array_equal(index0, np.asarray([0, 1, 1, 1]))
    assert np.array_equal(index0, index1)
    assert np.array_equal(h1[:, -1], np.asarray([0, 1, 1, 1]))


def test_multinomial_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(7)
    design = rng.normal(size=(12, 5))
    targets = rng.integers(0, 3, size=12, dtype=np.int8)
    values = rng.normal(scale=0.1, size=10)
    mask = np.asarray([0, 1, 1, 1, 1], dtype=float)
    _, gradient = multinomial_objective(
        values,
        design,
        targets,
        0.01,
        mask,
    )
    numerical = np.zeros_like(values)
    epsilon = 1e-6
    for index in range(len(values)):
        upper = values.copy()
        lower = values.copy()
        upper[index] += epsilon
        lower[index] -= epsilon
        upper_value, _ = multinomial_objective(
            upper,
            design,
            targets,
            0.01,
            mask,
        )
        lower_value, _ = multinomial_objective(
            lower,
            design,
            targets,
            0.01,
            mask,
        )
        numerical[index] = (upper_value - lower_value) / (2 * epsilon)
    assert np.allclose(gradient, numerical, atol=1e-6)


def test_cumulative_incidence_conserves_probability() -> None:
    entries = [_entry("a", 100, 1, 0), _entry("b", 300, 2, 1)]
    standardizer = Standardizer(np.zeros(19), np.ones(19))
    feature_names = tuple(f"x{index}" for index in range(26))
    coefficients = np.zeros((2, 26))
    coefficients[:, :6] = -2.0
    model = HazardModel(
        feature_names=feature_names,
        coefficients=coefficients,
        ridge=0.01,
        converged=True,
        iterations=1,
        objective=0.0,
        gradient_max_abs=0.0,
    )
    predictions = cumulative_incidence(
        model,
        entries,
        standardizer,
        include_reversal=True,
    )
    for values in predictions.values():
        assert np.allclose(np.sum(values, axis=1), 1.0)
        assert np.all(values >= 0)
        assert np.all(values <= 1)


def test_date_block_bootstrap_is_deterministic() -> None:
    deltas = {"a": 0.01, "b": -0.02, "c": 0.03}
    first = date_block_bootstrap(deltas, replicates=100, seed=42)
    second = date_block_bootstrap(deltas, replicates=100, seed=42)
    assert first == second
    assert first["date_count"] == 3


def test_fit_recovers_reversal_follow_direction() -> None:
    rng = np.random.default_rng(11)
    row_count = 2_000
    reversal = rng.integers(0, 2, size=row_count)
    design = np.column_stack((np.ones(row_count), reversal))
    follow_logit = -2.0 + 1.2 * reversal
    fail_logit = -2.0 - 0.8 * reversal
    denominator = 1 + np.exp(follow_logit) + np.exp(fail_logit)
    probabilities = np.column_stack(
        (
            1 / denominator,
            np.exp(follow_logit) / denominator,
            np.exp(fail_logit) / denominator,
        )
    )
    uniform = rng.random(row_count)
    cumulative = np.cumsum(probabilities, axis=1)
    targets = np.sum(uniform[:, None] > cumulative, axis=1).astype(np.int8)
    model = fit_hazard_model(
        design,
        targets,
        ("intercept", "reversal_indicator_R"),
        ridge=1e-4,
    )
    assert model.converged
    assert model.coefficients[0, 1] > 0
    assert model.coefficients[1, 1] < 0
