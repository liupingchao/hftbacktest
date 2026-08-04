from __future__ import annotations

import numpy as np

from examples.tutorial_reproduction import notebook_support
from examples.tutorial_reproduction.advanced_experiments import (
    _apt_fair_data,
    _basis_fair_data,
    _causal_standardize_components,
    _fit_glft_prefix,
    _rolling_mean,
    _rolling_zscore,
)


def test_advanced_notebooks_are_assigned_to_task_0804t004() -> None:
    advanced_slugs = [slug for slug, _, _ in notebook_support.EXPERIMENTS[9:]]

    assert len(advanced_slugs) == 7
    assert {
        notebook_support._TASK_BY_SLUG[slug] for slug in advanced_slugs
    } == {"0804T004"}


def test_rolling_helpers_are_point_in_time() -> None:
    values = np.asarray([1.0, 2.0, 100.0, 4.0])

    mean = _rolling_mean(values, 2)
    zscore = _rolling_zscore(values, 2)

    np.testing.assert_allclose(mean, [1.0, 1.5, 51.0, 52.0])
    assert zscore[0] == 0.0
    assert zscore[1] > 0
    assert zscore[2] > 0
    assert zscore[3] < 0


def test_basis_and_apt_fair_data_preserve_timestamps() -> None:
    timestamps = np.arange(1, 5, dtype=np.int64)
    series = {
        "timestamp": timestamps,
        "mid": np.asarray([100.0, 101.0, 102.0, 103.0]),
        "index": np.asarray([99.0, 100.0, 101.0, 102.0]),
    }

    basis_fair, basis = _basis_fair_data(series)
    apt_fair, index_return = _apt_fair_data(series)

    np.testing.assert_array_equal(basis_fair[:, 0], timestamps)
    np.testing.assert_array_equal(apt_fair[:, 0], timestamps)
    np.testing.assert_allclose(basis, 1.0)
    assert np.isfinite(basis_fair[:, 1]).all()
    assert np.isfinite(apt_fair[:, 1]).all()
    assert np.isfinite(index_return).all()


def test_future_components_do_not_change_historical_standardization() -> None:
    components = np.arange(80, dtype=np.float64).reshape(20, 4)
    changed = components.copy()
    changed[10:] *= 1000

    original = _causal_standardize_components(components, 5)
    modified = _causal_standardize_components(changed, 5)

    np.testing.assert_allclose(original[:10], modified[:10])


def test_glft_prefix_fit_ignores_future_observations() -> None:
    arrival = np.tile(np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]), 40)
    changes = np.sin(np.arange(len(arrival)) / 10)
    changed_arrival = arrival.copy()
    changed_changes = changes.copy()
    changed_arrival[100:] = 1000
    changed_changes[100:] = 1000

    original = _fit_glft_prefix(arrival, changes, 100, 100, 100_000_000)
    modified = _fit_glft_prefix(
        changed_arrival,
        changed_changes,
        100,
        100,
        100_000_000,
    )

    np.testing.assert_allclose(original, modified, equal_nan=True)
