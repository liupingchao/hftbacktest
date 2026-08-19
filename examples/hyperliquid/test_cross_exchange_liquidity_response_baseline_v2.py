from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_liquidity_response_baseline_v2 as baseline  # noqa: E402
import cross_exchange_liquidity_response_case_hierarchy as hierarchy  # noqa: E402
from test_cross_exchange_liquidity_response_case_hierarchy import (  # noqa: E402
    _episode_row,
    _m1_package,
    _timeline_row,
)


def _response_row(segment_id: str, index: int) -> dict[str, str]:
    row = _episode_row(
        episode_id=f"{segment_id}-000001",
        segment_id=segment_id,
        shock_ts_ns=1_000_000_000 + index * 10_000_000_000,
        decision_ts_ns=1_010_000_000 + index * 10_000_000_000,
        h2000_source_ts_ns=3_010_000_000 + index * 10_000_000_000,
    )
    decision_ts = int(row["decision_ts_ns"])
    row.update(
        {
            "confirmation_lag_ms": "10",
            "hl_first_bbo_response_ts_ns": str(decision_ts + 20_000_000),
            "hl_first_bbo_response_latency_ms": "20",
            "hl_withdrawal_observed": "true",
            "hl_withdrawal_latency_ms": "20",
            "hl_retreat_observed": "true",
            "hl_retreat_latency_ms": "25",
            "hl_replenishment_observed": "true",
            "hl_replenishment_latency_ms": "30",
            "hl_follow_observed": "true",
            "hl_follow_latency_ms": "35",
            "h1000_adverse_markout_ticks": str(index + 10),
            "h2000_adverse_markout_ticks": str(index + 20),
        }
    )
    for horizon in baseline.RESPONSE_HORIZONS_MS:
        row.update(
            {
                f"h{horizon}_covered": "true",
                f"h{horizon}_target_inside_segment": "true",
                f"h{horizon}_source_ts_ns": str(
                    decision_ts + horizon * 1_000_000
                ),
                f"h{horizon}_no_new_information": "false",
                f"h{horizon}_spread_ticks": str(30 - index),
                f"h{horizon}_impacted_qty_ratio": str(0.7 + index * 0.01),
                f"h{horizon}_opposite_qty_ratio": str(0.8 + index * 0.01),
                f"h{horizon}_fast_top5_impacted_qty_ratio": str(
                    0.9 + index * 0.01
                ),
                f"h{horizon}_fast_top5_opposite_qty_ratio": str(
                    1.0 + index * 0.01
                ),
                f"h{horizon}_fast_top5_imbalance": str(0.1 + index * 0.01),
                f"h{horizon}_directional_mid_markout_ticks": str(index + 1),
                f"h{horizon}_fast_age_ms": "10",
            }
        )
    return row


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    segments = [
        *hierarchy.DISCOVERY_SEGMENTS,
        *hierarchy.HELDOUT_SEGMENTS,
    ]
    rows = [
        _response_row(segment_id, index)
        for index, segment_id in enumerate(segments)
    ]
    timelines = {
        segment_id: [
            _timeline_row(index * 10_000_000_000 + 900_000_000),
            _timeline_row(index * 10_000_000_000 + 1_100_000_000),
            _timeline_row(index * 10_000_000_000 + 1_600_000_000),
        ]
        for index, segment_id in enumerate(segments)
    }
    m1 = _m1_package(tmp_path, rows, timelines)
    hierarchy_root = tmp_path / "case_hierarchy"
    hierarchy.build_shock_atom_catalog(
        m1_dir=m1,
        output_dir=hierarchy_root,
        expected_atom_count=len(rows),
    )
    hierarchy.build_shock_cluster_episodes_v2(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )
    return m1, hierarchy_root


def _gzip_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_feature_allowlist_excludes_adverse_and_pnl_fields() -> None:
    for target in baseline.TARGET_SPECS:
        fields = baseline._target_feature_fields(target)
        assert not any(
            token in field.lower()
            for field in fields
            for token in baseline.ADVERSE_FIELD_TOKENS
        )
    assert all("adverse" not in target for target in baseline.TARGET_SPECS)


def test_segment_file_guard_rejects_forbidden_segment(tmp_path: Path) -> None:
    guard = baseline.SegmentFileGuard(hierarchy.DISCOVERY_SEGMENTS)

    with pytest.raises(baseline.BaselineV2BuildError, match="forbidden"):
        guard.open_gzip_csv(
            tmp_path / "segment_0004.csv.gz",
            segment_id="segment_0004",
            role="m1_episode",
        )


def test_embargo_rejects_overlapping_evidence_intervals() -> None:
    query = {
        "episode_start_ts_ns": 100_000_000_000,
        "episode_end_ts_ns": 100_050_000_000,
        "h100_outcome_known_from_ts_ns": 100_100_000_000,
    }
    candidate = {
        "episode_start_ts_ns": 20_000_000_000,
        "episode_end_ts_ns": 20_050_000_000,
        "h100_outcome_known_from_ts_ns": 99_000_000_000,
    }

    assert not baseline._disjoint_with_embargo(
        query, candidate, "h100_spread_change_ticks", 60
    )
    candidate["h100_outcome_known_from_ts_ns"] = 30_000_000_000
    assert baseline._disjoint_with_embargo(
        query, candidate, "h100_spread_change_ticks", 60
    )
    long_query = {
        "episode_start_ts_ns": 100_000_000_000,
        "episode_end_ts_ns": 180_000_000_000,
        "h100_outcome_known_from_ts_ns": 100_100_000_000,
    }
    later_candidate = {
        "episode_start_ts_ns": 170_000_000_000,
        "episode_end_ts_ns": 171_000_000_000,
        "h100_outcome_known_from_ts_ns": 170_100_000_000,
    }
    assert not baseline._disjoint_with_embargo(
        long_query, later_candidate, "h100_spread_change_ticks", 60
    )


def test_scale_floor_is_ten_percent_of_discovery_mad() -> None:
    rows = [
        {"h100_spread_change_ticks": value}
        for value in (0.0, 2.0, 4.0, 6.0, 100.0)
    ]

    assert baseline._target_mad(rows, "h100_spread_change_ticks") == pytest.approx(
        0.2
    )


def test_acf_uses_absolute_wall_clock_seconds_without_gap_compression() -> None:
    bins = {
        ("segment_0001", second): float(index % 2)
        for index, second in enumerate(range(0, 62, 2))
    }

    lag_one_acf, lag_one_pairs = baseline._wall_clock_acf(bins, 1)
    lag_two_acf, lag_two_pairs = baseline._wall_clock_acf(bins, 2)

    assert lag_one_acf is None
    assert lag_one_pairs == 0
    assert lag_two_acf is not None
    assert lag_two_pairs == baseline.ACF_MIN_PAIR_COUNT


def test_input_stability_gate_rejects_late_mutation(tmp_path: Path) -> None:
    path = tmp_path / "input.csv"
    path.write_text("before\n", encoding="utf-8")
    expected_sha = hierarchy.sha256_file(path)
    path.write_text("after\n", encoding="utf-8")

    with pytest.raises(baseline.BaselineV2BuildError, match="input data SHA"):
        baseline._assert_input_provenance_stable(
            manifest_paths=[],
            input_provenance=[
                {
                    "role": "m1_episode",
                    "path": str(path),
                    "sha256": expected_sha,
                }
            ],
        )


def test_portable_hgb_matches_sklearn_and_serializes_deterministically(
    tmp_path: Path,
) -> None:
    x = np.column_stack(
        [
            np.arange(240, dtype=float),
            np.sin(np.arange(240, dtype=float) / 7.0),
            np.cos(np.arange(240, dtype=float) / 11.0),
        ]
    )
    x[::17, 1] = np.nan
    y = 0.25 * x[:, 0] + np.nan_to_num(x[:, 1], nan=0.0)
    first = baseline._hgb_model(0.50).fit(x, y)
    second = baseline._hgb_model(0.50).fit(x, y)
    first_export = baseline._export_hgb_model(first)
    second_export = baseline._export_hgb_model(second)

    np.testing.assert_array_equal(
        baseline._portable_hgb_predict(x, first_export),
        first.predict(x),
    )
    assert first_export == second_export

    first_path = tmp_path / "first.json.gz"
    second_path = tmp_path / "second.json.gz"
    baseline._write_model_bundle(first_path, first_export)
    baseline._write_model_bundle(second_path, second_export)
    assert hierarchy.sha256_file(first_path) == hierarchy.sha256_file(second_path)
    signed_zero_path = tmp_path / "signed_zero.json.gz"
    baseline._write_model_bundle(
        signed_zero_path, {"positive": 0.0, "negative": -0.0}
    )
    with gzip.open(signed_zero_path, "rt", encoding="utf-8") as fh:
        assert "-0.0" not in fh.read()


def test_discovery_freeze_never_opens_post_selection_files(tmp_path: Path) -> None:
    m1, hierarchy_root = _build_fixture(tmp_path)

    manifest = baseline.build_discovery_freeze(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )

    assert manifest["passes"] is True
    assert manifest["boundary"]["opened_heldout_data_file_count"] == 0
    contract = json.loads(
        (hierarchy_root / "baseline_v2" / "frozen_research_contract.json").read_text()
    )
    assert {item["segment_id"] for item in contract["opened_files"]} == set(
        hierarchy.DISCOVERY_SEGMENTS
    )
    assert contract["adverse_evaluation_only"] == [
        "h1000_adverse_markout_ticks",
        "h2000_adverse_markout_ticks",
    ]
    assert (
        contract["acf_diagnostics"]["binning"]
        == "per-segment absolute 1-second wall-clock bins; missing seconds are not compressed"
    )
    assert manifest["neighbor_summary"] == baseline._matched_neighbor_summary(
        _gzip_rows(
            hierarchy_root
            / "baseline_v2"
            / "discovery_cv_predictions.csv.gz"
        )
    )
    baseline_dir = hierarchy_root / "baseline_v2"
    baseline._validate_discovery_package(baseline_dir)
    repeated = baseline.build_discovery_freeze(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )
    assert repeated == manifest

    manifest_path = baseline_dir / "baseline_manifest.json"
    original_manifest = json.loads(manifest_path.read_text())
    contract_path = baseline_dir / "frozen_research_contract.json"
    original_contract = json.loads(contract_path.read_text())
    drifted_manifest = json.loads(manifest_path.read_text())
    drifted_manifest["split_label"] = "held_out"
    hierarchy._write_json(manifest_path, drifted_manifest)
    with pytest.raises(baseline.BaselineV2BuildError, match="split/status"):
        baseline._validate_discovery_package(baseline_dir)

    hierarchy._write_json(manifest_path, original_manifest)
    drifted_contract = json.loads(json.dumps(original_contract))
    drifted_contract["official_baseline"] = "matched_neighbor_median"
    hierarchy._write_json(contract_path, drifted_contract)
    drifted_manifest = json.loads(manifest_path.read_text())
    drifted_manifest["frozen_contract_sha256"] = hierarchy.sha256_file(
        contract_path
    )
    hierarchy._write_json(manifest_path, drifted_manifest)
    with pytest.raises(
        baseline.BaselineV2BuildError, match="model selection drift"
    ):
        baseline._validate_discovery_package(baseline_dir)

    hierarchy._write_json(contract_path, original_contract)
    hierarchy._write_json(manifest_path, original_manifest)
    drifted_contract = json.loads(json.dumps(original_contract))
    drifted_contract["neighbor_contract"]["minimum_neighbors"] = 49
    hierarchy._write_json(contract_path, drifted_contract)
    drifted_manifest = json.loads(json.dumps(original_manifest))
    drifted_manifest["frozen_contract_sha256"] = hierarchy.sha256_file(
        contract_path
    )
    hierarchy._write_json(manifest_path, drifted_manifest)
    with pytest.raises(baseline.BaselineV2BuildError, match="neighbor contract"):
        baseline.build_post_selection_evaluation(
            hierarchy_dir=hierarchy_root,
            m1_dir=m1,
        )
    assert not (baseline_dir / "heldout_consumption_manifest.json").exists()

    hierarchy._write_json(contract_path, original_contract)
    hierarchy._write_json(manifest_path, original_manifest)
    drifted_contract = json.loads(json.dumps(original_contract))
    drifted_contract["unknown_contract_field"] = "forbidden"
    hierarchy._write_json(contract_path, drifted_contract)
    drifted_manifest = json.loads(json.dumps(original_manifest))
    drifted_manifest["frozen_contract_sha256"] = hierarchy.sha256_file(
        contract_path
    )
    hierarchy._write_json(manifest_path, drifted_manifest)
    with pytest.raises(baseline.BaselineV2BuildError, match="contract schema"):
        baseline._validate_discovery_package(baseline_dir)

    hierarchy._write_json(contract_path, original_contract)
    hierarchy._write_json(manifest_path, original_manifest)
    drifted_contract = json.loads(json.dumps(original_contract))
    drifted_contract["input_provenance"][0]["sha256"] = "0" * 64
    hierarchy._write_json(contract_path, drifted_contract)
    drifted_manifest = json.loads(json.dumps(original_manifest))
    drifted_manifest["frozen_contract_sha256"] = hierarchy.sha256_file(
        contract_path
    )
    hierarchy._write_json(manifest_path, drifted_manifest)
    with pytest.raises(
        baseline.BaselineV2BuildError, match="provenance data drift"
    ):
        baseline.build_post_selection_evaluation(
            hierarchy_dir=hierarchy_root,
            m1_dir=m1,
        )
    assert not (baseline_dir / "heldout_consumption_manifest.json").exists()

    hierarchy._write_json(contract_path, original_contract)
    hierarchy._write_json(manifest_path, original_manifest)
    drifted_manifest = json.loads(json.dumps(original_manifest))
    drifted_manifest["neighbor_summary"]["unavailable_query_count"] += 1
    hierarchy._write_json(manifest_path, drifted_manifest)
    with pytest.raises(baseline.BaselineV2BuildError, match="neighbor summary"):
        baseline._validate_discovery_package(baseline_dir)

    hierarchy._write_json(manifest_path, original_manifest)
    drifted_manifest = json.loads(json.dumps(original_manifest))
    drifted_manifest["outputs"]["discovery_episode_features"]["row_count"] += 1
    drifted_manifest["counts"]["episode_count"] += 1
    hierarchy._write_json(manifest_path, drifted_manifest)
    with pytest.raises(baseline.BaselineV2BuildError, match="row count drift"):
        baseline._validate_discovery_package(baseline_dir)

    hierarchy._write_json(contract_path, original_contract)
    hierarchy._write_json(manifest_path, original_manifest)


def test_post_selection_evaluation_never_emits_heldout_label(tmp_path: Path) -> None:
    m1, hierarchy_root = _build_fixture(tmp_path)
    baseline.build_discovery_freeze(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )
    baseline_dir = hierarchy_root / "baseline_v2"
    frozen_manifest_sha = hierarchy.sha256_file(
        baseline_dir / "baseline_manifest.json"
    )
    frozen_contract_sha = hierarchy.sha256_file(
        baseline_dir / "frozen_research_contract.json"
    )

    manifest = baseline.build_post_selection_evaluation(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )

    assert manifest["passes"] is True
    assert manifest["split_label"] == "post_selection"
    assert manifest["formal_heldout_authorized"] is False
    predictions = _gzip_rows(
        hierarchy_root / "baseline_v2" / "post_selection_predictions.csv.gz"
    )
    assert predictions
    assert {row["split_label"] for row in predictions} == {"post_selection"}
    consumption = json.loads(
        (
            hierarchy_root
            / "baseline_v2"
            / "heldout_consumption_manifest.json"
        ).read_text()
    )
    assert consumption["label"] == "post_selection"
    assert consumption["formal_heldout_authorized"] is False
    assert {item["role"] for item in consumption["input_sha256"]} == {
        "m1_manifest",
        "m1_episode",
        "timeline",
    }
    assert len(consumption["input_sha256"]) == 11
    assert all(
        isinstance(item["row_count"], int) and item["row_count"] >= 1
        for item in consumption["input_sha256"]
    )
    assert len(manifest["input_provenance"]) == 10
    assert len(manifest["opened_files"]) == 10
    assert (
        hierarchy.sha256_file(baseline_dir / "baseline_manifest.json")
        == frozen_manifest_sha
    )
    assert (
        hierarchy.sha256_file(baseline_dir / "frozen_research_contract.json")
        == frozen_contract_sha
    )

    repeated = baseline.build_post_selection_evaluation(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )
    assert repeated == manifest
    consumption_path = baseline_dir / "heldout_consumption_manifest.json"
    post_manifest_path = baseline_dir / "post_selection_manifest.json"
    original_consumption = json.loads(consumption_path.read_text())
    original_post_manifest = json.loads(post_manifest_path.read_text())
    drifted_consumption = json.loads(json.dumps(original_consumption))
    drifted_consumption["input_sha256"] = drifted_consumption["input_sha256"][:1]
    drifted_consumption["unknown_field"] = "forbidden"
    hierarchy._write_json(consumption_path, drifted_consumption)
    drifted_post_manifest = json.loads(json.dumps(original_post_manifest))
    drifted_post_manifest["outputs"]["heldout_consumption_manifest"][
        "sha256"
    ] = hierarchy.sha256_file(consumption_path)
    hierarchy._write_json(post_manifest_path, drifted_post_manifest)
    with pytest.raises(
        baseline.BaselineV2BuildError, match="consumption manifest schema"
    ):
        baseline.build_post_selection_evaluation(
            hierarchy_dir=hierarchy_root,
            m1_dir=m1,
        )

    hierarchy._write_json(consumption_path, original_consumption)
    hierarchy._write_json(post_manifest_path, original_post_manifest)
    drifted_consumption = json.loads(json.dumps(original_consumption))
    drifted_consumption["input_sha256"] = drifted_consumption["input_sha256"][:1]
    hierarchy._write_json(consumption_path, drifted_consumption)
    drifted_post_manifest = json.loads(json.dumps(original_post_manifest))
    drifted_post_manifest["outputs"]["heldout_consumption_manifest"][
        "sha256"
    ] = hierarchy.sha256_file(consumption_path)
    hierarchy._write_json(post_manifest_path, drifted_post_manifest)
    with pytest.raises(
        baseline.BaselineV2BuildError, match="input cardinality"
    ):
        baseline.build_post_selection_evaluation(
            hierarchy_dir=hierarchy_root,
            m1_dir=m1,
        )

    hierarchy._write_json(consumption_path, original_consumption)
    hierarchy._write_json(post_manifest_path, original_post_manifest)
    drifted_consumption = json.loads(json.dumps(original_consumption))
    drifted_consumption["input_sha256"][1]["row_count"] += 1
    hierarchy._write_json(consumption_path, drifted_consumption)
    drifted_post_manifest = json.loads(json.dumps(original_post_manifest))
    drifted_post_manifest["outputs"]["heldout_consumption_manifest"][
        "sha256"
    ] = hierarchy.sha256_file(consumption_path)
    hierarchy._write_json(post_manifest_path, drifted_post_manifest)
    with pytest.raises(
        baseline.BaselineV2BuildError, match="input row count drift"
    ):
        baseline.build_post_selection_evaluation(
            hierarchy_dir=hierarchy_root,
            m1_dir=m1,
        )

    hierarchy._write_json(consumption_path, original_consumption)
    hierarchy._write_json(post_manifest_path, original_post_manifest)
    with pytest.raises(
        baseline.BaselineV2BuildError, match="post-selection data was consumed"
    ):
        baseline.build_discovery_freeze(
            hierarchy_dir=hierarchy_root,
            m1_dir=m1,
        )


def test_post_selection_consumption_closes_coordinated_m1_row_count_drift(
    tmp_path: Path,
) -> None:
    m1, hierarchy_root = _build_fixture(tmp_path)
    baseline.build_discovery_freeze(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )
    baseline_dir = hierarchy_root / "baseline_v2"
    m1_manifest_path = m1 / "motif_episode_manifest.json"
    m1_manifest = json.loads(m1_manifest_path.read_text())
    m1_manifest["outputs"]["episodes"]["segment_0004"]["row_count"] += 1
    hierarchy._write_json(m1_manifest_path, m1_manifest)

    contract_path = baseline_dir / "frozen_research_contract.json"
    contract = json.loads(contract_path.read_text())
    contract["m1_manifest_sha256"] = hierarchy.sha256_file(m1_manifest_path)
    hierarchy._write_json(contract_path, contract)
    discovery_manifest_path = baseline_dir / "baseline_manifest.json"
    discovery_manifest = json.loads(discovery_manifest_path.read_text())
    discovery_manifest["frozen_contract_sha256"] = hierarchy.sha256_file(
        contract_path
    )
    hierarchy._write_json(discovery_manifest_path, discovery_manifest)

    with pytest.raises(
        baseline.BaselineV2BuildError, match="consumption input row count drift"
    ):
        baseline.build_post_selection_evaluation(
            hierarchy_dir=hierarchy_root,
            m1_dir=m1,
        )
    assert (baseline_dir / "heldout_consumption_manifest.json").is_file()
    assert not (baseline_dir / "post_selection_manifest.json").exists()
