from __future__ import annotations

import gzip
import io
import math

import numpy as np

import skhynix_stage_h0b_contracts as contracts
import skhynix_stage_h0b as h0b


def test_nearest_rank_is_one_based() -> None:
    values = np.asarray([4.0, 1.0, 3.0, 2.0])
    assert contracts.nearest_rank(values, 0.25) == 1.0
    assert contracts.nearest_rank(values, 0.50) == 2.0
    assert contracts.nearest_rank(values, 0.95) == 4.0


def test_exact_survival_respects_bin_boundaries() -> None:
    q = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float64)
    observed = contracts.exact_survival(
        np.repeat(q, 5, axis=0),
        np.asarray([0, 5_000_000, 10_000_000, 15_000_000, 50_000_000]),
    )
    assert observed[0] == 1.0
    assert math.isclose(observed[1], 0.9**0.5)
    assert math.isclose(observed[2], 0.9)
    assert math.isclose(observed[3], 0.9 * 0.8**0.5)
    assert math.isclose(observed[4], 0.9 * 0.8 * 0.7 * 0.6 * 0.5)


def test_interval_likelihood_branches() -> None:
    q = np.full((3, 5), 0.1, dtype=np.float64)
    branch = np.asarray([1, 2, 3], dtype=np.int8)
    lower = np.asarray([9_000_000, 0, 40_000_000], dtype=np.float64)
    upper = np.asarray(
        [11_000_000, contracts.HORIZON_NS, 55_000_000],
        dtype=np.float64,
    )
    likelihood, loss = contracts.likelihood_and_loss(
        q, branch, lower, upper
    )
    expected_event = contracts.exact_survival(
        q[:1], lower[:1]
    )[0] - contracts.exact_survival(q[:1], upper[:1])[0]
    assert math.isclose(likelihood[0], expected_event)
    assert math.isclose(likelihood[1], 0.9**5)
    assert math.isclose(
        likelihood[2],
        contracts.exact_survival(q[2:], lower[2:])[0],
    )
    assert np.isfinite(loss).all()


def test_interval_eta_gradient_matches_finite_difference() -> None:
    eta = np.asarray(
        [
            [-2.1, -1.8, -2.4, -1.5, -2.0],
            [-1.9, -2.2, -1.7, -2.3, -1.6],
            [-2.0, -2.0, -2.0, -2.0, -2.0],
        ],
        dtype=np.float64,
    )
    branch = np.asarray([1, 2, 3], dtype=np.int8)
    lower = np.asarray([9_000_000, 0, 40_000_000], dtype=np.float64)
    upper = np.asarray(
        [11_000_000, contracts.HORIZON_NS, 60_000_000],
        dtype=np.float64,
    )
    q = contracts.sigmoid(eta)
    _, loss, gradient = contracts.interval_objective_eta_gradient(
        q, branch, lower, upper
    )
    epsilon = 1e-6
    for row in range(3):
        for column in range(5):
            changed = eta.copy()
            changed[row, column] += epsilon
            _, plus = contracts.likelihood_and_loss(
                contracts.sigmoid(changed), branch, lower, upper
            )
            changed[row, column] -= 2 * epsilon
            _, minus = contracts.likelihood_and_loss(
                contracts.sigmoid(changed), branch, lower, upper
            )
            numerical = (plus[row] - minus[row]) / (2 * epsilon)
            assert math.isclose(
                gradient[row, column],
                numerical,
                rel_tol=2e-6,
                abs_tol=2e-6,
            )
    assert np.isfinite(loss).all()


def test_design_matrix_order_and_missing_policy() -> None:
    raw = np.asarray(
        [
            [0.1, 0.01, 0.2, 2.0, 0.8],
            [0.2, 0.04, 0.3, np.nan, 0.7],
            [0.3, 0.09, 0.4, 4.0, 0.6],
        ],
        dtype=np.float64,
    )
    scales = contracts.fit_feature_scales(raw, contracts.H0_RAW_FEATURES)
    design = contracts.transform_design(
        raw,
        np.asarray([1.0, 0.0, 1.0]),
        scales,
        model="H0",
    )
    assert design.shape == (3, len(contracts.H0_DESIGN_COLUMNS))
    assert design[:, 0].tolist() == [1.0, 0.0, 1.0]
    missing_offset = 1 + len(contracts.H0_RAW_FEATURES)
    assert design[:, missing_offset + 3].tolist() == [0.0, 1.0, 0.0]


def test_walk_forward_exact_fold_counts() -> None:
    jul30 = contracts.build_walk_forward_folds(
        [index * contracts.BLOCK_NS for index in range(232)]
    )
    aug04 = contracts.build_walk_forward_folds(
        [index * contracts.BLOCK_NS for index in range(119)]
    )
    assert [len(fold.test_blocks) for fold in jul30] == [
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        20,
        12,
    ]
    assert [len(fold.test_blocks) for fold in aug04] == [20, 20, 19]


def test_km_processes_events_before_censors_at_tie() -> None:
    durations = np.asarray([10.0, 10.0, 20.0])
    censored = np.asarray([False, True, False])
    assert contracts.kaplan_meier_median(durations, censored) == 20.0
    assert (
        contracts.kaplan_meier_median(
            np.asarray([10.0, 10.0, 10.0]),
            np.asarray([True, True, False]),
        )
        is None
    )


def test_classification_precedence_and_non_rescue() -> None:
    facts = {
        "jul30": {
            "data_quality": True,
            "rq1": True,
            "rq2": True,
            "rq3": False,
        },
        "aug04": {
            "data_quality": True,
            "rq1": True,
            "rq2": True,
            "rq3": False,
        },
    }
    classification, path, reasons = contracts.classify_primary(facts)
    assert classification == "predictable_but_not_latency_actionable"
    assert path[-1] == "rq3_6600ms"
    assert reasons == ["both_formal_sessions_fail_rq3_6600ms"]


def test_deterministic_gzip_has_zero_mtime_and_empty_filename() -> None:
    raw = b"a,b\n1,2\n"
    first = contracts.deterministic_gzip(raw)
    second = contracts.deterministic_gzip(raw)
    assert first == second
    with gzip.GzipFile(fileobj=io.BytesIO(first), mode="rb") as handle:
        assert handle.read() == raw


def test_event_geometry_uses_side_direction_and_strict_timestamp_lower() -> None:
    events = h0b.QuoteEvents(
        local_ts_ns=np.asarray(
            [100_000_000, 120_000_000, 120_000_000, 140_000_000],
            dtype=np.int64,
        ),
        event_seq=np.asarray([1, 2, 3, 4], dtype=np.int64),
        bid_px=np.asarray([99.0, 99.5, 101.0, 98.0]),
        ask_px=np.asarray([101.0, 100.5, 102.0, 98.5]),
        valid=np.asarray([True, True, True, True]),
    )
    grid = np.asarray([100_000_000], dtype=np.int64)
    reference = np.asarray([0], dtype=np.int64)
    ask_branch, ask_lower, ask_upper, ask_event = h0b.event_geometry(
        grid=grid,
        reference_indexes=reference,
        events=events,
        side="maker_ask_risk",
    )
    bid_branch, bid_lower, bid_upper, bid_event = h0b.event_geometry(
        grid=grid,
        reference_indexes=reference,
        events=events,
        side="maker_bid_risk",
    )
    assert ask_branch.tolist() == [1]
    assert ask_lower.tolist() == [0]
    assert ask_upper.tolist() == [20_000_000]
    assert ask_event.tolist() == [True]
    assert bid_branch.tolist() == [1]
    assert bid_lower.tolist() == [20_000_000]
    assert bid_upper.tolist() == [40_000_000]
    assert bid_event.tolist() == [True]


def test_basis_residual_is_prior_only() -> None:
    residual = h0b.basis_residual(
        np.asarray([10.0, 20.0, np.nan, 30.0], dtype=np.float64)
    )
    alpha = 1.0 - math.exp(
        -math.log(2.0) * contracts.GRID_NS / 60_000_000_000
    )
    assert math.isnan(residual[0])
    assert residual[1] == 10.0
    assert math.isnan(residual[2])
    assert math.isclose(residual[3], 30.0 - (10.0 + alpha * 10.0))


def test_regime_debounce_and_boundary_censor() -> None:
    count = 15
    dataset = h0b.SessionDataset(
        session="jul30",
        segment_ids=("segment_0001",),
        grid_ts_ns=np.arange(count, dtype=np.int64) * contracts.GRID_NS,
        segment_code=np.zeros(count, dtype=np.int16),
        block_start_ns=np.zeros(count, dtype=np.int64),
        h0_raw=np.zeros((count, 5)),
        h1_added_raw=np.zeros((count, 2, 5)),
        dose=np.zeros((count, 2)),
        branch=np.full((count, 2), 2, dtype=np.int8),
        lower_elapsed_ns=np.zeros((count, 2), dtype=np.int64),
        upper_elapsed_ns=np.full(
            (count, 2),
            contracts.HORIZON_NS,
            dtype=np.int64,
        ),
        binary_identified=np.ones((count, 2), dtype=bool),
        binary_event=np.zeros((count, 2), dtype=bool),
        flow_unit=np.full(count, "background", dtype=object),
        projection_rows=[],
        censoring_rows=[],
        exclusion_rows=[],
    )
    risk = np.asarray(
        [0.1, 0.9, 0.9, 0.9, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2]
        + [0.9] * 5
    )
    rows = h0b.detect_regimes(
        dataset=dataset,
        fold_id=1,
        anchor_indexes=np.arange(count),
        side_index=0,
        risk=risk,
        entry_threshold=0.9,
        exit_threshold=0.2,
    )
    assert len(rows) == 2
    assert rows[0]["t_detect_ns"] == 30_000_000
    assert rows[0]["t_exit_ns"] == 90_000_000
    assert rows[0]["censored"] is False
    assert rows[1]["censored"] is True
