from __future__ import annotations

import gzip
import io
import math
from collections import Counter

import numpy as np
import pytest

import skhynix_stage_h0b_contracts as contracts
import skhynix_stage_h0b as h0b


NEGATIVE_CASES = tuple(
    (
        surface["surface_id"],
        mutation["mutation_id"],
        mutation["expected_error_code"],
    )
    for surface in h0b.read_json(h0b.MATRIX_PATH)["surfaces"]
    for mutation in surface["negative_mutations"]
)


@pytest.mark.parametrize(
    ("surface_id", "mutation_id", "expected_code"),
    NEGATIVE_CASES,
)
def test_each_hostile_case_uses_production_error_code(
    surface_id: str,
    mutation_id: str,
    expected_code: str,
) -> None:
    with pytest.raises(contracts.H0BError) as captured:
        h0b.negative_case(surface_id, mutation_id, expected_code)
    assert captured.value.code == expected_code


def test_matrix_expected_code_cannot_issue_production_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = h0b.read_json(h0b.MATRIX_PATH)
    matrix["surfaces"][0]["negative_mutations"][0][
        "expected_error_code"
    ] = "H0B_BUILD_MISMATCH"
    monkeypatch.setattr(h0b, "read_json", lambda _: matrix)
    with pytest.raises(contracts.H0BError) as captured:
        h0b.negative_case(
            "kernel_pin",
            "mutate_kernel_pin",
            "H0B_BUILD_MISMATCH",
        )
    assert captured.value.code == "H0B_KERNEL_PIN_MISMATCH"


def test_noop_production_guard_cannot_pass_hostile_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        h0b,
        "validate_external_action_boundary",
        lambda _: None,
    )
    with pytest.raises(contracts.H0BError) as captured:
        h0b.negative_case(
            "zero_external_action",
            "mutate_zero_external_action",
            "H0B_EXTERNAL_ACTION_FORBIDDEN",
        )
    expected_codes = {expected for _, _, expected in NEGATIVE_CASES}
    assert captured.value.code == h0b.HOSTILE_FAIL_OPEN_SENTINEL
    assert captured.value.code not in expected_codes


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


@pytest.mark.parametrize(
    ("support_class", "expected"),
    [
        (
            "interval_likelihood_only_supported",
            "diagnostic_censored_interval_likelihood_only",
        ),
        (
            "right_censored_segment",
            "diagnostic_censored_right_censored_segment",
        ),
        (
            "right_censored_source_end",
            "diagnostic_censored_right_censored_source_end",
        ),
        ("epoch_censored", "diagnostic_censored_epoch_censored"),
        (
            "core_quality_censored",
            "diagnostic_censored_core_quality_censored",
        ),
        (
            "source_gap_censored",
            "diagnostic_censored_source_gap_censored",
        ),
        (
            "reference_quote_unavailable",
            "diagnostic_censored_reference_quote_unavailable",
        ),
        (
            "invalid_quote_state",
            "diagnostic_censored_invalid_quote_state",
        ),
    ],
)
def test_landmark_support_classes_are_exhaustive(
    support_class: str,
    expected: str,
) -> None:
    assert (
        h0b.landmark_status_from_support_class(support_class) == expected
    )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {
                "reference_available": True,
                "reference_valid": True,
                "target_inside_segment": True,
                "same_epoch": True,
                "core_quality_eligible": True,
                "source_gap": False,
                "source_ended": False,
                "endpoint_closed": True,
                "interval_bounds_supported": True,
            },
            "binary_identification_supported",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": True,
                "target_inside_segment": True,
                "same_epoch": True,
                "core_quality_eligible": True,
                "source_gap": False,
                "source_ended": False,
                "endpoint_closed": False,
                "interval_bounds_supported": True,
            },
            "interval_likelihood_only_supported",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": True,
                "target_inside_segment": False,
                "same_epoch": True,
                "core_quality_eligible": True,
                "source_gap": False,
                "source_ended": False,
                "endpoint_closed": True,
                "interval_bounds_supported": True,
            },
            "right_censored_segment",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": True,
                "target_inside_segment": True,
                "same_epoch": True,
                "core_quality_eligible": True,
                "source_gap": False,
                "source_ended": True,
                "endpoint_closed": True,
                "interval_bounds_supported": True,
            },
            "right_censored_source_end",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": True,
                "target_inside_segment": True,
                "same_epoch": False,
                "core_quality_eligible": True,
                "source_gap": False,
                "source_ended": False,
                "endpoint_closed": True,
                "interval_bounds_supported": True,
            },
            "epoch_censored",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": True,
                "target_inside_segment": True,
                "same_epoch": True,
                "core_quality_eligible": False,
                "source_gap": False,
                "source_ended": False,
                "endpoint_closed": True,
                "interval_bounds_supported": True,
            },
            "core_quality_censored",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": True,
                "target_inside_segment": True,
                "same_epoch": True,
                "core_quality_eligible": True,
                "source_gap": True,
                "source_ended": False,
                "endpoint_closed": True,
                "interval_bounds_supported": True,
            },
            "source_gap_censored",
        ),
        (
            {
                "reference_available": False,
                "reference_valid": False,
                "target_inside_segment": True,
                "same_epoch": True,
                "core_quality_eligible": True,
                "source_gap": False,
                "source_ended": False,
                "endpoint_closed": True,
                "interval_bounds_supported": True,
            },
            "reference_quote_unavailable",
        ),
        (
            {
                "reference_available": True,
                "reference_valid": False,
                "target_inside_segment": True,
                "same_epoch": True,
                "core_quality_eligible": True,
                "source_gap": False,
                "source_ended": False,
                "endpoint_closed": True,
                "interval_bounds_supported": True,
            },
            "invalid_quote_state",
        ),
    ],
)
def test_production_support_classifier_reaches_all_nine_classes(
    kwargs: dict[str, bool],
    expected: str,
) -> None:
    state = h0b.AcceptedH0ASupportState(**kwargs)
    assert h0b.accepted_h0a_support_class_from_state(state) == expected


def test_landmark_binary_status_and_grid_first_precedence() -> None:
    assert h0b.landmark_status_from_support_class(
        "binary_identification_supported",
        event=True,
    ) == "identified_event"
    assert h0b.landmark_status_from_support_class(
        "binary_identification_supported",
        event=False,
    ) == "identified_no_event"
    assert h0b.landmark_status_with_precedence(
        outside_grid=True,
        support_class="right_censored_segment",
    ) == "diagnostic_censored_grid_boundary"


def test_stage4_endpoint_has_three_states() -> None:
    base = {
        "time_to_first_adverse_target_bbo_event_status": "",
        "time_to_first_adverse_target_bbo_event_interval_upper_ns": "",
        "time_to_first_adverse_target_bbo_event_censor_time_ns": "",
    }
    event = {
        **base,
        "time_to_first_adverse_target_bbo_event_status": (
            "interval_censored"
        ),
        "time_to_first_adverse_target_bbo_event_interval_upper_ns": "150",
    }
    no_event = {
        **base,
        "time_to_first_adverse_target_bbo_event_status": "right_censored",
        "time_to_first_adverse_target_bbo_event_censor_time_ns": "150",
    }
    assert h0b.stage4_endpoint_status(event, endpoint_ns=150) == (
        "identified_event"
    )
    assert h0b.stage4_endpoint_status(no_event, endpoint_ns=150) == (
        "identified_no_event"
    )
    assert h0b.stage4_endpoint_status(event, endpoint_ns=149) == (
        "diagnostic_censored"
    )


def test_stage4_crosscheck_three_by_three_conservation() -> None:
    counter: Counter[str] = Counter()
    statuses = (
        "identified_event",
        "identified_no_event",
        "diagnostic_censored_grid_boundary",
    )
    stage4_statuses = (
        "identified_event",
        "identified_no_event",
        "diagnostic_censored",
    )
    for h0b_status in statuses:
        for stage4_status in stage4_statuses:
            h0b.update_stage4_crosscheck_counter(
                counter,
                h0b_status=h0b_status,
                stage4_status=stage4_status,
            )
    row = h0b.stage4_crosscheck_row(
        scope="segment",
        segment_id="segment_0001",
        side="maker_ask_risk",
        counter=counter,
        direction_match=True,
        naming_match=True,
    )
    assert row["joined_count"] == 9
    assert row["eligible_count"] == 4
    assert row["censored_count"] == 5
    assert row["both_event_count"] == 1
    assert row["h0b_only_count"] == 1
    assert row["stage4_only_count"] == 1
    assert row["neither_count"] == 1
    assert row["agreement_fraction"] == 0.5


def test_stage4_crosscheck_empty_ratios_are_blank_cells() -> None:
    counter: Counter[str] = Counter()
    h0b.update_stage4_crosscheck_counter(
        counter,
        h0b_status="diagnostic_censored_grid_boundary",
        stage4_status="diagnostic_censored",
    )
    row = h0b.stage4_crosscheck_row(
        scope="segment",
        segment_id="segment_0001",
        side="maker_bid_risk",
        counter=counter,
        direction_match=True,
        naming_match=True,
    )
    assert row["eligible_count"] == 0
    assert row["h0b_event_rate"] is None
    assert row["stage4_event_rate"] is None
    assert row["agreement_fraction"] is None


def test_stage4_access_ledger_requires_permit_before_exact_paths() -> None:
    permit_sha = "a" * 64
    prefix = (
        "local_live_analysis/"
        "skhynix_trigger_aligned_episode_research_v1_stage04_"
        "jul30_episode_v3/"
    )
    events = [
        {
            "sequence": 1,
            "process_role": "H0B1_DIAGNOSTIC_PERMIT",
            "phase": "post_primary_seal_permit",
            "relative_path": "stage4_diagnostic_permit.json",
            "access_kind": "fsync_write",
            "bytes_read": 0,
            "permit_sha256": permit_sha,
            "admitted": True,
        }
    ]
    for sequence, relative in enumerate(
        sorted(h0b.STAGE4_OUTCOMES),
        start=2,
    ):
        events.append(
            {
                "sequence": sequence,
                "process_role": "H0B1_DIAGNOSTIC",
                "phase": "post_primary_seal_stage4",
                "relative_path": prefix + relative,
                "access_kind": "exact_11_field_projection",
                "bytes_read": 1,
                "permit_sha256": permit_sha,
                "admitted": True,
            }
        )
    ledger = {
        "schema_version": "skhynix_stage_h0b_outcome_access_ledger_v1",
        "task_id": contracts.TASK_ID,
        "build_label": "A",
        "events": events,
    }
    h0b.validate_stage4_access_ledger(
        ledger,
        build_label="A",
        diagnostic_permit_sha256=permit_sha,
    )
    ledger["events"][1]["sequence"] = 1
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_stage4_access_ledger(
            ledger,
            build_label="A",
            diagnostic_permit_sha256=permit_sha,
        )
    assert captured.value.code == "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL"
    ledger["events"][1]["sequence"] = 2
    shadow = {
        **ledger["events"][0],
        "sequence": 2,
        "phase": "diagnostic_shadow",
        "relative_path": "shadow_permit.json",
    }
    ledger["events"].insert(1, shadow)
    for sequence, event in enumerate(ledger["events"], start=1):
        event["sequence"] = sequence
    with pytest.raises(contracts.H0BError) as captured:
        h0b.validate_stage4_access_ledger(
            ledger,
            build_label="A",
            diagnostic_permit_sha256=permit_sha,
        )
    assert captured.value.code == "H0B_STAGE4_OPEN_BEFORE_PRIMARY_SEAL"


def test_hostile_generic_surface_executes_contract_mutation() -> None:
    with pytest.raises(contracts.H0BError) as captured:
        h0b.negative_case(
            "source_schema",
            "mutate_source_schema",
            "H0B_SOURCE_SCHEMA_MISMATCH",
        )
    assert captured.value.code == "H0B_SOURCE_SCHEMA_MISMATCH"


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
