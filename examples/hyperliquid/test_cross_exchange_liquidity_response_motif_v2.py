from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_liquidity_response_motif_v2 as motif  # noqa: E402
import cross_exchange_liquidity_response_baseline_v2 as baseline  # noqa: E402
import cross_exchange_liquidity_response_case_hierarchy as hierarchy  # noqa: E402
from test_cross_exchange_liquidity_response_baseline_v2 import (  # noqa: E402
    _build_fixture,
)


def _synthetic_rows(count: int = 360) -> list[dict[str, object]]:
    rows = []
    for index in range(count):
        segment = index % 3
        group = (index // 3) % 3
        row: dict[str, object] = {
            "flow_episode_id": f"segment_000{segment + 1}-E{index:06d}",
            "segment_id": f"segment_000{segment + 1}",
            "split_label": "discovery",
            "motif_discovery_eligible": "true",
        }
        for field_index, field in enumerate(motif.STRUCTURAL_FEATURE_FIELDS):
            row[field] = group * 20 + field_index + (index % 7) * 0.01
        for target_index, target in enumerate(motif.TARGETS):
            row[f"response_residual__{target}"] = (
                group * 10 + target_index * 0.01 + (index % 5) * 0.001
            )
            row[f"response_observed__{target}"] = 1
        rows.append(row)
    return rows


def test_feature_allowlist_has_no_adverse_or_pnl_fields() -> None:
    assert not [
        field
        for field in motif.MOTIF_FEATURE_FIELDS
        if any(
            token in field.lower()
            for token in motif.FORBIDDEN_FEATURE_TOKENS
        )
    ]


def test_deterministic_npz_bytes(tmp_path: Path) -> None:
    arrays = {
        "a": np.array([1.0, 2.0]),
        "b": np.array(["x", "y"]),
    }
    left = tmp_path / "left.npz"
    right = tmp_path / "right.npz"
    motif._write_deterministic_npz(left, arrays)
    motif._write_deterministic_npz(right, arrays)
    assert left.read_bytes() == right.read_bytes()


def test_discovery_pipeline_limits_graph_and_uses_real_medoids() -> None:
    rows = _synthetic_rows()
    (
        membership,
        prototypes,
        _,
        _,
        surrogate,
        _,
        diagnostics,
        _,
    ) = motif._motif_artifacts(rows, [], surrogate_count=3)

    assert diagnostics["graph_max_degree"] <= 10
    episode_ids = {row["flow_episode_id"] for row in rows}
    assert all(row["prototype_episode_id"] in episode_ids for row in prototypes)
    assert {
        row["classification"] for row in prototypes
    } <= {"needs_fresh_holdout", "not_supported"}
    assert all(row["split_label"] == "discovery" for row in membership)
    assert all(row["screening_surrogate_count"] == 3 for row in surrogate)


def test_medoid_is_an_observed_vector() -> None:
    vectors = np.array([[0.0], [1.0], [10.0]])
    assert motif._medoid_local_index(vectors) == 1


def test_neighbor_sets_preserve_distance_order_after_removing_self() -> None:
    indices = np.array(
        [
            [3, 0, 2, 1],
            [1, 3, 2, 0],
            [2, 1, 0, 3],
            [0, 3, 1, 2],
        ]
    )
    neighbors = motif._ordered_neighbor_sets(indices, limit=2)
    assert neighbors[0] == {3, 2}
    assert neighbors[1] == {3, 2}


def test_post_selection_assignment_never_emits_heldout() -> None:
    discovery = _synthetic_rows()
    post = [dict(row) for row in discovery[:12]]
    for index, row in enumerate(post):
        row["flow_episode_id"] = f"segment_0004-E{index:06d}"
        row["segment_id"] = "segment_0004"
        row["split_label"] = "post_selection"
    membership, prototypes, *rest = motif._motif_artifacts(
        discovery, post, surrogate_count=2
    )
    del prototypes, rest
    assert {
        row["split_label"] for row in membership
    } <= {"discovery", "post_selection"}
    assert not any("held" + "_out" in str(row) for row in membership)


def test_bh_q_values_are_monotone_in_rank() -> None:
    p_values = [0.04, 0.001, 0.02]
    q_values = motif._bh_q_values(p_values)
    ordered = sorted(zip(p_values, q_values))
    assert ordered[0][1] <= ordered[1][1] <= ordered[2][1]


def test_validate_existing_rejects_unknown_output(tmp_path: Path) -> None:
    output = tmp_path / "motif_v2"
    output.mkdir()
    (output / "motif_manifest.json").write_text(
        json.dumps(
            {
                "task_id": motif.TASK_ID,
                "schema_version": motif.SCHEMA_VERSION,
                "passes": True,
                "formal_heldout_authorized": False,
                "frozen_contract_sha256": "0" * 64,
                "outputs": {"unknown": {}},
            }
        )
    )
    (output / "frozen_motif_contract.json").write_text("{}")
    with pytest.raises(motif.MotifV2BuildError):
        motif._validate_existing(output)


def test_end_to_end_uses_baseline_v2_and_rejects_source_drift(
    tmp_path: Path,
) -> None:
    m1, hierarchy_root = _build_fixture(tmp_path)
    baseline.build_discovery_freeze(hierarchy_dir=hierarchy_root, m1_dir=m1)
    baseline.build_post_selection_evaluation(
        hierarchy_dir=hierarchy_root, m1_dir=m1
    )
    legacy_motif = hierarchy_root / "motif"
    legacy_motif.mkdir()
    legacy_marker = legacy_motif / "immutable.txt"
    legacy_marker.write_text("legacy\n")

    manifest = motif.build_motif_v2(
        hierarchy_dir=hierarchy_root, surrogate_count=2
    )
    assert manifest["passes"] is True
    assert manifest["formal_heldout_authorized"] is False
    assert manifest["boundary"]["adverse_pnl_feature_count"] == 0
    assert legacy_marker.read_text() == "legacy\n"
    assert (
        motif.build_motif_v2(
            hierarchy_dir=hierarchy_root, surrogate_count=2
        )
        == manifest
    )

    motif_dir = hierarchy_root / "motif_v2"
    motif_manifest_path = motif_dir / "motif_manifest.json"
    motif_contract_path = motif_dir / "frozen_motif_contract.json"
    original_motif_manifest = json.loads(motif_manifest_path.read_text())
    original_motif_contract = json.loads(motif_contract_path.read_text())
    drifted_contract = json.loads(json.dumps(original_motif_contract))
    drifted_contract["input_provenance"][0]["role"] = drifted_contract[
        "input_provenance"
    ][1]["role"]
    hierarchy._write_json(motif_contract_path, drifted_contract)
    drifted_manifest = json.loads(json.dumps(original_motif_manifest))
    contract_sha = hierarchy.sha256_file(motif_contract_path)
    drifted_manifest["frozen_contract_sha256"] = contract_sha
    drifted_manifest["outputs"]["frozen_motif_contract"]["sha256"] = contract_sha
    hierarchy._write_json(motif_manifest_path, drifted_manifest)
    with pytest.raises(motif.MotifV2BuildError, match="input identity drift"):
        motif._validate_existing(motif_dir)

    hierarchy._write_json(motif_contract_path, original_motif_contract)
    hierarchy._write_json(motif_manifest_path, original_motif_manifest)
    drifted_manifest = json.loads(json.dumps(original_motif_manifest))
    drifted_manifest["outputs"]["frozen_motif_contract"]["unknown"] = "forbidden"
    hierarchy._write_json(motif_manifest_path, drifted_manifest)
    with pytest.raises(motif.MotifV2BuildError, match="output schema drift"):
        motif._validate_existing(motif_dir)

    hierarchy._write_json(motif_manifest_path, original_motif_manifest)
    drifted_manifest = json.loads(json.dumps(original_motif_manifest))
    drifted_manifest["outputs"]["frozen_motif_contract"]["row_count"] = 999
    hierarchy._write_json(motif_manifest_path, drifted_manifest)
    with pytest.raises(motif.MotifV2BuildError, match="JSON row count drift"):
        motif._validate_existing(motif_dir)

    hierarchy._write_json(motif_manifest_path, original_motif_manifest)
    baseline_manifest_path = (
        hierarchy_root / "baseline_v2" / "baseline_manifest.json"
    )
    baseline_manifest = json.loads(baseline_manifest_path.read_text())
    baseline_manifest["split_label"] = "post_selection"
    hierarchy._write_json(baseline_manifest_path, baseline_manifest)
    with pytest.raises(
        baseline.BaselineV2BuildError, match="split/status drift"
    ):
        motif.build_motif_v2(
            hierarchy_dir=hierarchy_root, surrogate_count=2
        )
