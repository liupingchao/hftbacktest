from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parent / "cross_exchange_trigger_density_core.py"
)
SPEC = importlib.util.spec_from_file_location("trigger_density_core", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def _row(
    candidate_seq: int,
    *,
    segment_id: str = "segment_0001",
    shock_ts_ns: int = 1_000_000_000,
    decision_ts_ns: int | None = None,
    direction_sign: int = 1,
    aggressor_side: str | None = None,
    impact_ratio: float = 0.60,
    primary_episode: bool = True,
    rejection_reason: str = "",
) -> dict[str, str]:
    if decision_ts_ns is None:
        decision_ts_ns = shock_ts_ns + 50_000_000 if primary_episode else shock_ts_ns + 30_000_000
    if aggressor_side is None:
        aggressor_side = "buy" if direction_sign > 0 else "sell"
    values = {
        "campaign_id": "campaign-1",
        "segment_id": segment_id,
        "profile_id": "skhynix",
        "candidate_seq": str(candidate_seq),
        "aggressor_side": aggressor_side,
        "direction_sign": str(direction_sign),
        "burst_start_ts_ns": str(shock_ts_ns - 10_000_000),
        "burst_end_ts_ns": str(shock_ts_ns),
        "burst_duration_ms": "10.0",
        "burst_trade_count": "3",
        "burst_trade_qty": "1.5",
        "touch_trade_qty": "1.5",
        "touch_trade_qty_at_shock": "0.5",
        "touch_trade_qty_through_decision": "1.5",
        "post_decision_burst_trade_count": "0",
        "pre_state_ts_ns": str(shock_ts_ns - 20_000_000),
        "pre_best_px": "950.0",
        "pre_best_qty": "1.0",
        "shock_ts_ns": str(shock_ts_ns),
        "impact_ratio": str(impact_ratio),
        "shock_impact_ratio": "0.4",
        "decision_ts_ns": str(decision_ts_ns),
        "confirmation_lag_ms": "50.0",
        "confirmed_best_px": "951.0",
        "confirmed_best_qty": "0.5",
        "price_level_depleted": "true",
        "queue_drop_ratio": "1.0",
        "confirmed_removed_qty": "1.0",
        "trade_explained_ratio": "1.0",
        "attribution": "trade_driven",
        "pre_hl_bbo_ts_ns": str(shock_ts_ns - 40_000_000),
        "pre_hl_bbo_age_ms": "40.0",
        "pre_hl_fast_source_ts_ns": str(shock_ts_ns - 45_000_000),
        "pre_hl_fast_age_ms": "45.0",
        "primary_episode": "true" if primary_episode else "false",
        "rejection_reason": rejection_reason,
    }
    return {field: values[field] for field in core.TRIGGER_AUDIT_SCHEMA}


def _spans() -> list[core.StructuralSpan]:
    return [
        core.StructuralSpan(
            session_id="jul30",
            segment_id="segment_0001",
            start_ts_ns=0,
            end_ts_ns=180_000_000_000,
        ),
        core.StructuralSpan(
            session_id="jul30",
            segment_id="segment_0002",
            start_ts_ns=200_000_000_000,
            end_ts_ns=260_000_000_000,
        ),
    ]


def test_parse_trigger_audit_rows_accepts_exact_schema_and_builds_ids() -> None:
    rows = [
        _row(1, shock_ts_ns=1_000_000_000),
        _row(2, shock_ts_ns=2_000_000_000, direction_sign=-1),
        _row(
            1,
            segment_id="segment_0002",
            shock_ts_ns=210_000_000_000,
            primary_episode=False,
            rejection_reason="same_direction_dedup",
        ),
    ]
    parsed = core.parse_trigger_audit_rows("jul30", rows)
    assert [candidate.candidate_id for candidate in parsed] == [
        "jul30:segment_0001:1",
        "jul30:segment_0001:2",
        "jul30:segment_0002:1",
    ]
    assert core.validate_family_population(
        parsed, expected_candidate_count=3, expected_confirmed_count=2
    ) == {"candidate_count": 3, "confirmed_count": 2}


def test_parse_trigger_audit_rows_rejects_outcome_like_schema_drift() -> None:
    row = _row(1)
    drifted = copy.deepcopy(row)
    drifted["outcome_markout_250ms"] = "1.25"
    with pytest.raises(core.DensityContractError, match="outcome-like fields"):
        core.parse_trigger_audit_rows("jul30", [drifted])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([_row(1), _row(3, shock_ts_ns=2_000_000_000)], "missing candidate_seq"),
        (
            [_row(1), _row(2, shock_ts_ns=3_000_000_000), _row(1, segment_id="segment_0001", shock_ts_ns=4_000_000_000)],
            "duplicate or reordered candidate_seq",
        ),
        (
            [_row(1, segment_id="segment_0001"), _row(1, segment_id="segment_0002", shock_ts_ns=210_000_000_000), _row(2, segment_id="segment_0001", shock_ts_ns=2_000_000_000)],
            "segment rows are reordered",
        ),
        (
            [_row(1, shock_ts_ns=2_000_000_000), _row(2, shock_ts_ns=1_000_000_000)],
            "reordered shock_ts_ns",
        ),
    ],
)
def test_parse_trigger_audit_rows_rejects_missing_duplicate_and_reordered_candidates(
    rows: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(core.DensityContractError, match=message):
        core.parse_trigger_audit_rows("jul30", rows)


def test_parse_trigger_audit_rows_rejects_bad_confirmed_timestamp_and_flag() -> None:
    bad_timestamp = _row(1, shock_ts_ns=1_000_000_000, decision_ts_ns=1_000_000_000)
    with pytest.raises(core.DensityContractError, match="valid decision_ts_ns"):
        core.parse_trigger_audit_rows("jul30", [bad_timestamp])

    bad_flag = _row(
        1,
        primary_episode=False,
        rejection_reason="",
    )
    with pytest.raises(core.DensityContractError, match="rejected candidate requires"):
        core.parse_trigger_audit_rows("jul30", [bad_flag])


def test_count_rates_and_inter_trigger_quantiles_match_structural_population() -> None:
    rows = [
        _row(1, shock_ts_ns=1_000_000_000, direction_sign=1),
        _row(2, shock_ts_ns=4_000_000_000, direction_sign=1),
        _row(3, shock_ts_ns=10_000_000_000, direction_sign=-1),
        _row(
            1,
            segment_id="segment_0002",
            shock_ts_ns=220_000_000_000,
            direction_sign=-1,
            primary_episode=False,
            rejection_reason="same_direction_dedup",
        ),
    ]
    candidates = core.parse_trigger_audit_rows("jul30", rows)
    summary = core.build_count_rate_summary(candidates, _spans())
    assert summary[0]["count"] == 4
    assert summary[1]["count"] == 3
    assert summary[0]["duration_seconds"] == 240.0
    assert summary[0]["rate_per_minute"] == pytest.approx(1.0)
    assert summary[1]["rate_per_hour"] == pytest.approx(45.0)

    distributions = core.build_inter_trigger_distribution(candidates)
    by_key = {
        (row["population"], row["side_relation"]): row for row in distributions
    }
    assert by_key[(core.FAMILY_A, "all")]["pair_count"] == 2
    assert by_key[(core.FAMILY_A, "all")]["p50_ms"] == pytest.approx(4_500.0)
    assert by_key[(core.FAMILY_A, "same_side")]["pair_count"] == 1
    assert by_key[(core.FAMILY_A, "same_side")]["p50_ms"] == pytest.approx(3_000.0)
    assert by_key[(core.FAMILY_A, "opposite_side")]["pair_count"] == 1
    assert by_key[(core.FAMILY_A, "opposite_side")]["p50_ms"] == pytest.approx(6_000.0)
    assert by_key[(core.FAMILY_B, "all")]["pair_count"] == 2


def test_family_b_time_statistics_use_decision_landmark() -> None:
    rows = [
        _row(
            1,
            shock_ts_ns=1_000_000_000,
            decision_ts_ns=1_010_000_000,
        ),
        _row(
            2,
            shock_ts_ns=1_100_000_000,
            decision_ts_ns=1_300_000_000,
        ),
    ]
    candidates = core.parse_trigger_audit_rows("jul30", rows)
    distributions = core.build_inter_trigger_distribution(candidates)
    by_population = {
        row["population"]: row
        for row in distributions
        if row["side_relation"] == "all"
    }
    assert by_population[core.FAMILY_A]["p50_ms"] == pytest.approx(100.0)
    assert by_population[core.FAMILY_B]["p50_ms"] == pytest.approx(290.0)

    blocks = core.build_time_block_membership(candidates, _spans())
    family_b = [
        row for row in blocks if row["population"] == core.FAMILY_B
    ]
    assert [row["landmark_ts_ns"] for row in family_b] == [
        1_010_000_000,
        1_300_000_000,
    ]


def test_window_union_summary_reports_coverage_blocks_and_longest_run() -> None:
    rows = [
        _row(1, shock_ts_ns=1_000_000_000),
        _row(2, shock_ts_ns=2_500_000_000),
        _row(3, shock_ts_ns=20_000_000_000, primary_episode=False, rejection_reason="mixed_attribution"),
    ]
    candidates = core.parse_trigger_audit_rows("jul30", rows)
    summary = core.build_window_union_summary(candidates, _spans())
    by_population = {row["population"]: row for row in summary}
    assert by_population[core.FAMILY_A]["overlap_block_count_2000ms"] == 2
    assert by_population[core.FAMILY_A]["window_union_coverage_ms"] == pytest.approx(
        5_500.0
    )
    assert by_population[core.FAMILY_A][
        "window_union_coverage_fraction"
    ] == pytest.approx(5_500.0 / 240_000.0)
    assert by_population[core.FAMILY_A][
        "longest_continuous_trigger_run_ms"
    ] == pytest.approx(3_500.0)
    assert by_population[core.FAMILY_B]["overlap_block_count_2000ms"] == 1


def test_sensitivity_membership_freezes_impact_refractory_and_first_per_flow() -> None:
    rows = [
        _row(1, shock_ts_ns=1_000_000_000, direction_sign=1, impact_ratio=0.55),
        _row(2, shock_ts_ns=1_090_000_000, direction_sign=1, impact_ratio=0.72),
        _row(3, shock_ts_ns=1_400_000_000, direction_sign=-1, impact_ratio=0.40),
    ]
    candidates = core.parse_trigger_audit_rows("jul30", rows)
    memberships = core.build_sensitivity_membership(
        candidates,
        primary_flow_first_candidate_ids={"jul30:segment_0001:1", "jul30:segment_0001:3"},
    )
    second = memberships[1]
    assert second["impact_ge_050"] is True
    assert second["impact_ge_070"] is True
    assert second["same_side_refractory_100ms"] is False
    assert second["same_side_refractory_250ms"] is False
    assert second["same_side_refractory_500ms"] is False
    assert second["population"] == core.FAMILY_A
    assert second["family_a_eligible"] is True
    assert second["family_b_eligible"] is True
    assert memberships[0]["first_per_primary_flow_episode"] is True
    assert memberships[2]["first_per_primary_flow_episode"] is True

    with pytest.raises(core.DensityContractError, match="unknown first-per-flow"):
        core.build_sensitivity_membership(
            candidates, primary_flow_first_candidate_ids={"jul30:segment_9999:1"}
        )


def test_refractory_resets_at_segment_boundary() -> None:
    rows = [
        _row(1, segment_id="segment_0001", shock_ts_ns=59_990_000_000),
        _row(
            1,
            segment_id="segment_0002",
            shock_ts_ns=200_010_000_000,
        ),
    ]
    candidates = core.parse_trigger_audit_rows("jul30", rows)
    memberships = core.build_sensitivity_membership(candidates, set())
    assert memberships[0]["same_side_refractory_500ms"] is True
    assert memberships[1]["same_side_refractory_500ms"] is True


def test_refractory_is_measured_from_last_retained_same_side_candidate() -> None:
    rows = [
        _row(1, shock_ts_ns=1_000_000_000),
        _row(2, shock_ts_ns=1_090_000_000),
        _row(3, shock_ts_ns=1_180_000_000),
    ]
    candidates = core.parse_trigger_audit_rows("jul30", rows)
    memberships = core.build_sensitivity_membership(candidates, set())
    assert [
        row["same_side_refractory_100ms"] for row in memberships
    ] == [True, False, True]
    assert [
        row["same_side_refractory_250ms"] for row in memberships
    ] == [True, False, False]


def test_time_block_membership_and_summary_anchor_blocks_at_segment_start() -> None:
    rows = [
        _row(1, shock_ts_ns=1_000_000_000),
        _row(2, shock_ts_ns=61_000_000_000),
        _row(
            1,
            segment_id="segment_0002",
            shock_ts_ns=240_000_000_000,
            primary_episode=False,
            rejection_reason="same_direction_dedup",
        ),
    ]
    candidates = core.parse_trigger_audit_rows("jul30", rows)
    memberships = core.build_time_block_membership(candidates, _spans())
    assert [row["block_id"] for row in memberships] == [
        "segment_0001:block_0000",
        "segment_0001:block_0000",
        "segment_0001:block_0001",
        "segment_0001:block_0001",
        "segment_0002:block_0000",
    ]
    summary = core.summarize_time_blocks(memberships, _spans())
    by_population = {row["population"]: row for row in summary}
    assert by_population[core.FAMILY_A]["time_block_count_60s"] == 4
    assert by_population[core.FAMILY_B]["time_block_count_60s"] == 4
    assert by_population[core.FAMILY_A]["count_semantics"] == (
        "all_segment_contained_time_blocks"
    )

    catalog = core.build_time_block_catalog(_spans())
    assert len(catalog) == 4
    assert catalog[-1]["segment_id"] == "segment_0002"
    assert catalog[-1]["partial_block"] is False


def test_effective_sample_size_reports_available_and_unavailable_segments() -> None:
    rows = [
        _row(1, shock_ts_ns=1_000_000_000),
        _row(2, shock_ts_ns=2_100_000_000),
        _row(3, shock_ts_ns=4_000_000_000),
        _row(4, shock_ts_ns=7_500_000_000),
        _row(
            1,
            segment_id="segment_0002",
            shock_ts_ns=201_000_000_000,
            primary_episode=False,
            rejection_reason="same_direction_dedup",
        ),
    ]
    candidates = core.parse_trigger_audit_rows("jul30", rows)
    rows_out = core.build_effective_sample_size_rows(candidates, _spans())
    first_available = next(
        row
        for row in rows_out
        if row["population"] == core.FAMILY_A
        and row["segment_id"] == "segment_0001"
    )
    assert first_available["available"] is True
    assert first_available["effective_sample_size"] <= first_available["sample_size_n"]

    unavailable = next(
        row
        for row in rows_out
        if row["population"] == core.FAMILY_B
        and row["segment_id"] == "segment_0002"
    )
    assert unavailable["available"] is False
    assert unavailable["unavailable_reason"] == "constant_series"

    session_total = next(
        row
        for row in rows_out
        if row["population"] == core.FAMILY_B
        and row["segment_id"] == "session_total"
    )
    assert session_total["available"] is False
    assert "segment_0002:constant_series" in session_total["unavailable_reason"]


def test_validate_effective_sample_size_rows_rejects_row_count_as_neff() -> None:
    row = {
        "population": core.FAMILY_A,
        "segment_id": "segment_0001",
        "estimator_name": "bartlett_ess_1s_geyer_ipps",
        "available": True,
        "unavailable_reason": "",
        "sample_size_n": 10,
        "bartlett_tau": None,
        "effective_sample_size": 10.0,
    }
    with pytest.raises(core.DensityContractError, match="row_count-as-N_eff"):
        core.validate_effective_sample_size_rows([row])
