from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_liquidity_response_case_hierarchy as hierarchy  # noqa: E402
import cross_exchange_liquidity_response_regime_v2 as regime  # noqa: E402


def _context_rows(window_count: int = 30) -> list[dict[str, object]]:
    rows = []
    for segment_index, segment_id in enumerate(
        [*regime.DISCOVERY_SEGMENTS, *regime.POST_SELECTION_SEGMENTS]
    ):
        split = (
            "discovery"
            if segment_id in regime.DISCOVERY_SEGMENTS
            else "post_selection"
        )
        for window_index in range(window_count):
            row: dict[str, object] = {
                "segment_id": segment_id,
                "split_label": split,
                "window_seq": window_index + 1,
                "window_start_ts_ns": (
                    segment_index * 10**15
                    + window_index * regime.WINDOW_NS
                ),
                "window_end_ts_ns": (
                    segment_index * 10**15
                    + (window_index + 1) * regime.WINDOW_NS
                ),
            }
            for field_index, field in enumerate(regime.CONTEXT_FEATURE_FIELDS):
                row[field] = (
                    segment_index * 0.1
                    + window_index * 0.01
                    + field_index * 0.001
                )
            rows.append(row)
    return rows


def test_transform_is_fit_on_discovery_only() -> None:
    rows = _context_rows()
    medians, scales, transform_sha = regime._fit_context_transform(rows)
    for row in rows:
        if row["split_label"] == "post_selection":
            for field in regime.CONTEXT_FEATURE_FIELDS:
                row[field] = float(row[field]) + 1_000_000
    changed_medians, changed_scales, changed_sha = (
        regime._fit_context_transform(rows)
    )
    assert changed_medians == medians
    assert changed_scales == scales
    assert changed_sha == transform_sha


def test_real_scores_use_frozen_component_scales() -> None:
    rows = _context_rows(window_count=8)
    for row in rows:
        for field in regime.CONTEXT_FEATURE_FIELDS:
            row[field] = 0.0
    segment_one = [
        row for row in rows if row["segment_id"] == "segment_0001"
    ]
    for row in segment_one[4:]:
        row[regime.CONTEXT_FEATURE_FIELDS[0]] = 10.0
    medians = {field: 0.0 for field in regime.CONTEXT_FEATURE_FIELDS}
    scales = {field: 1.0 for field in regime.CONTEXT_FEATURE_FIELDS}
    scores = regime._candidate_scores(rows, medians, scales)
    score = next(
        row
        for row in scores
        if row["segment_id"] == "segment_0001"
        and row["boundary_window_seq"] == 5
    )
    assert score["change_score"] == pytest.approx(10.0)


def test_block_shuffle_preserves_contiguous_blocks() -> None:
    rows = _context_rows(window_count=10)
    field = regime.CONTEXT_FEATURE_FIELDS[0]
    segment = [row for row in rows if row["segment_id"] == "segment_0001"]
    for index, row in enumerate(segment):
        row[field] = float(index)
    shuffled = regime._shuffle_context_blocks(
        rows, block_minutes=5, rng=np.random.default_rng(0)
    )
    values = [
        int(float(row[field]))
        for row in shuffled
        if row["segment_id"] == "segment_0001"
    ]
    assert values in [list(range(10)), list(range(5, 10)) + list(range(5))]
    assert [
        int(row["window_seq"])
        for row in shuffled
        if row["segment_id"] == "segment_0001"
    ] == list(range(1, 11))


def test_short_edge_interval_removes_internal_boundary() -> None:
    rows = _context_rows(window_count=7)
    segment = [
        row for row in rows if row["segment_id"] == "segment_0001"
    ]
    segment[-1]["window_end_ts_ns"] = (
        int(segment[-1]["window_end_ts_ns"]) - regime.WINDOW_NS // 2
    )
    candidate = {
        "segment_id": "segment_0001",
        "boundary_window_seq": 5,
        "boundary_ts_ns": segment[4]["window_start_ts_ns"],
        "change_score": 10.0,
    }
    accepted = regime._merge_short_intervals([candidate], rows)
    assert accepted == []


def test_surrogate_publication_uses_same_transform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _context_rows()
    for row in rows:
        for field in regime.CONTEXT_FEATURE_FIELDS:
            row[field] = 0.0
    for row in rows:
        if (
            row["segment_id"] == "segment_0001"
            and int(row["window_seq"]) >= 16
        ):
            row[regime.CONTEXT_FEATURE_FIELDS[0]] = 100.0
    medians = {field: 0.0 for field in regime.CONTEXT_FEATURE_FIELDS}
    scales = {field: 1.0 for field in regime.CONTEXT_FEATURE_FIELDS}
    transform_sha = "a" * 64

    def zero_null(
        context_rows: list[dict[str, object]],
        context_medians: dict[str, float],
        context_scales: dict[str, float],
        *,
        surrogate_count: int,
        block_minutes: int,
        seed: int,
    ) -> dict[str, dict[str, list[float]]]:
        del context_rows, context_medians, context_scales, block_minutes, seed
        return {
            segment_id: {
                "max_scores": [0.0] * surrogate_count,
                "boundary_counts": [0.0] * surrogate_count,
            }
            for segment_id in [
                *regime.DISCOVERY_SEGMENTS,
                *regime.POST_SELECTION_SEGMENTS,
                "__all__",
            ]
        }

    monkeypatch.setattr(regime, "_surrogate_null", zero_null)
    audit, boundaries, summary, _ = regime._calibrate_boundaries(
        rows,
        medians,
        scales,
        transform_sha,
        surrogate_count=100,
    )
    assert any(row["boundary_origin"] == "data_driven" for row in boundaries)
    assert all(
        row["real_score_transform_sha256"]
        == row["surrogate_score_transform_sha256"]
        == transform_sha
        for row in [*audit, *summary]
    )
    assert all(
        float(row["empirical_max_score_p_value"]) <= regime.EMPIRICAL_P_LIMIT
        for row in audit
        if row["publication_status"] == "published"
    )


def test_motif_linkage_cannot_upgrade_not_supported() -> None:
    intervals = [
        {
            "regime_id": "segment_0001-RV2-0001",
            "segment_id": "segment_0001",
            "split_label": "discovery",
            "start_ts_ns": 0,
            "end_ts_ns": 10,
            "boundary_start_origin": "segment",
        }
    ]
    episodes = [
        {
            "flow_episode_id": "episode-1",
            "segment_id": "segment_0001",
            "split_label": "discovery",
            "episode_decision_ts_ns": "5",
        }
    ]
    membership = [
        {
            "flow_episode_id": "episode-1",
            "motif_id": "M2-0001",
            "prototype_distance": "1.25",
        }
    ]
    prototypes = [{"motif_id": "M2-0001", "classification": "not_supported"}]
    _, motif_rows, _ = regime._link_episodes_and_motifs(
        episodes, intervals, membership, prototypes
    )
    assert motif_rows[0]["source_motif_classification"] == "not_supported"
    assert motif_rows[0]["classification"] == "not_supported"

    prototypes[0]["classification"] = "needs_fresh_holdout"
    with pytest.raises(regime.RegimeV2BuildError, match="cannot upgrade"):
        regime._link_episodes_and_motifs(
            episodes, intervals, membership, prototypes
        )


def test_build_rejects_too_few_surrogates(tmp_path: Path) -> None:
    with pytest.raises(regime.RegimeV2BuildError, match="at least 100"):
        regime.build_regime_v2(
            hierarchy_dir=tmp_path,
            surrogate_count=99,
        )


def test_source_output_drift_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "source.csv"
    path.write_text("value\n1\n", encoding="utf-8")
    owner = {
        "outputs": {
            "source": {
                "path": "source.csv",
                "row_count": 1,
                "sha256": hierarchy.sha256_file(path),
            }
        }
    }
    item = regime._validate_output_entry(
        tmp_path, "source_role", owner, "source"
    )
    assert item["row_count"] == 1
    path.write_text("value\n1\n2\n", encoding="utf-8")
    with pytest.raises(regime.RegimeV2BuildError, match="source output drift"):
        regime._validate_output_entry(
            tmp_path, "source_role", owner, "source"
        )


def test_algorithm_contract_is_exact_and_source_derived() -> None:
    rows = _context_rows()
    medians, scales, transform_sha = regime._fit_context_transform(rows)
    scores = regime._candidate_scores(rows, medians, scales)
    threshold = regime._quantile(
        [
            float(row["change_score"])
            for row in scores
            if row["segment_id"] in regime.DISCOVERY_SEGMENTS
        ],
        regime.CHANGE_THRESHOLD_QUANTILE,
    )
    contract = {
        "context_medians": medians,
        "context_scales": scales,
        "context_transform_sha256": transform_sha,
        "score_metric": regime.SCORE_METRIC,
        "detector_contract": regime._detector_contract(threshold),
        "surrogate_contract": regime._surrogate_contract(
            regime.SURROGATE_COUNT
        ),
        "label_thresholds": regime._label_thresholds(rows),
        "motif_linkage_contract": regime._motif_linkage_contract(),
    }
    regime._validate_algorithm_contract(contract, rows)

    detector_drift = json.loads(json.dumps(contract))
    detector_drift["detector_contract"]["short_interval_merge"] = "disabled"
    with pytest.raises(regime.RegimeV2BuildError, match="detector contract"):
        regime._validate_algorithm_contract(detector_drift, rows)

    surrogate_drift = json.loads(json.dumps(contract))
    surrogate_drift["surrogate_contract"].update(
        {
            "primary_block_minutes": 4,
            "diagnostic_block_minutes": [4],
            "seed": 7,
            "threshold_estimator_replayed_per_surrogate": False,
        }
    )
    with pytest.raises(regime.RegimeV2BuildError, match="surrogate contract"):
        regime._validate_algorithm_contract(surrogate_drift, rows)

    transform_drift = json.loads(json.dumps(contract))
    first_field = regime.CONTEXT_FEATURE_FIELDS[0]
    transform_drift["context_medians"][first_field] += 1.0
    transform_drift["context_transform_sha256"] = regime._canonical_sha(
        {
            "feature_fields": regime.CONTEXT_FEATURE_FIELDS,
            "medians": transform_drift["context_medians"],
            "scales": transform_drift["context_scales"],
            "score_metric": regime.SCORE_METRIC,
        }
    )
    with pytest.raises(
        regime.RegimeV2BuildError, match="source-derived context transform"
    ):
        regime._validate_algorithm_contract(transform_drift, rows)


def test_validate_existing_rejects_unknown_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "regime_v2"
    output_dir.mkdir()
    contract = {
        key: None for key in regime.CONTRACT_KEYS
    }
    medians = {field: 0.0 for field in regime.CONTEXT_FEATURE_FIELDS}
    scales = {field: 1.0 for field in regime.CONTEXT_FEATURE_FIELDS}
    transform_sha = regime._canonical_sha(
        {
            "feature_fields": regime.CONTEXT_FEATURE_FIELDS,
            "medians": medians,
            "scales": scales,
            "score_metric": (
                "euclidean_norm_of_pre3_post3_component_medians"
            ),
        }
    )
    contract.update(
        {
            "task_id": regime.TASK_ID,
            "schema_version": regime.SCHEMA_VERSION,
            "input_provenance": [],
            "source_episode_schema_version": (
                hierarchy.EPISODE_V2_SCHEMA_VERSION
            ),
            "source_baseline_schema_version": regime.baseline.SCHEMA_VERSION,
            "source_motif_schema_version": regime.motif.SCHEMA_VERSION,
            "discovery_segments": regime.DISCOVERY_SEGMENTS,
            "post_selection_segments": regime.POST_SELECTION_SEGMENTS,
            "context_feature_fields": regime.CONTEXT_FEATURE_FIELDS,
            "context_medians": medians,
            "context_scales": scales,
            "context_transform_sha256": transform_sha,
            "score_metric": (
                "euclidean_norm_of_pre3_post3_component_medians"
            ),
            "surrogate_contract": {
                "same_frozen_transform_as_real": True,
                "primary_count": 100,
            },
            "legacy_regime_file_sha256": {},
        }
    )
    hierarchy._write_json(output_dir / "frozen_regime_contract.json", contract)
    manifest = {key: None for key in regime.MANIFEST_KEYS}
    manifest.update(
        {
            "task_id": regime.TASK_ID,
            "schema_version": regime.SCHEMA_VERSION,
            "passes": True,
            "formal_heldout_authorized": False,
            "fresh_holdout_available": False,
            "frozen_contract_sha256": hierarchy.sha256_file(
                output_dir / "frozen_regime_contract.json"
            ),
            "outputs": {"unknown": {}},
        }
    )
    hierarchy._write_json(output_dir / "regime_manifest.json", manifest)
    monkeypatch.setattr(
        regime,
        "_validate_source",
        lambda _: ({}, {}, {}, {}, []),
    )
    with pytest.raises(regime.RegimeV2BuildError, match="output key drift"):
        regime._validate_existing(output_dir)
