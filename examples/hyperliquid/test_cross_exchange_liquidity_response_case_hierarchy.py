from __future__ import annotations

import csv
import gzip
import io
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_liquidity_response_case_hierarchy as hierarchy  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_episode_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as gz:
        wrapper = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        writer = csv.DictWriter(wrapper, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        wrapper.detach()


def _episode_row(
    *,
    episode_id: str,
    segment_id: str = "segment_0001",
    shock_ts_ns: int = 980_000_000,
    decision_ts_ns: int = 1_000_000_000,
    direction_sign: int = 1,
    aggressor_side: str = "buy",
    binance_mid_px: float = 99.995,
    impacted_top5_qty: float = 15.0,
    opposite_top5_qty: float = 17.0,
    shock_impact_ratio: float = 0.4,
    covered_2000: bool = True,
    h2000_source_ts_ns: int = 3_000_000_000,
) -> dict[str, str]:
    row = {
        "episode_id": episode_id,
        "campaign_id": "campaign",
        "profile_id": "skhynix",
        "segment_id": segment_id,
        "shock_ts_ns": str(shock_ts_ns),
        "decision_ts_ns": str(decision_ts_ns),
        "pre_state_ts_ns": str(decision_ts_ns - 30_000_000),
        "aggressor_side": aggressor_side,
        "direction_sign": str(direction_sign),
        "pre_best_px": "100.0",
        "pre_best_qty": "5.0",
        "shock_impact_ratio": str(shock_impact_ratio),
        "impact_ratio": "1.2",
        "queue_drop_ratio": "0.8",
        "confirmed_removed_qty": "4.0",
        "trade_explained_ratio": "1.0",
        "attribution": "trade_driven",
        "binance_pre_bid_px": "99.99",
        "binance_pre_ask_px": "100.0",
        "binance_pre_mid_px": str(binance_mid_px),
        "binance_pre_spread_px": "0.01",
        "binance_pre_impacted_qty": "5.0",
        "binance_pre_opposite_qty": "7.0",
        "binance_pre_top5_impacted_qty": str(impacted_top5_qty),
        "binance_pre_top5_opposite_qty": str(opposite_top5_qty),
        "binance_pre_top5_imbalance": "0.1",
        "pre_hl_bbo_ts_ns": str(decision_ts_ns - 50_000_000),
        "pre_hl_bbo_age_ms": "50",
        "pre_hl_fast_source_ts_ns": str(decision_ts_ns - 60_000_000),
        "pre_hl_fast_age_ms": "60",
        "hl_tick_size": "0.01",
        "hl_pre_bid_px": "99.5",
        "hl_pre_bid_qty": "10.0",
        "hl_pre_ask_px": "99.8",
        "hl_pre_ask_qty": "11.0",
        "hl_pre_mid_px": "99.65",
        "hl_pre_spread_ticks": "30",
        "hl_pre_impacted_qty": "11.0",
        "hl_pre_opposite_qty": "10.0",
        "hl_pre_fast_top5_impacted_qty": "21.0",
        "hl_pre_fast_top5_opposite_qty": "20.0",
        "hl_pre_fast_top5_imbalance": "-0.02",
        "basis_mid_bps": "34.0",
        "h1000_covered": "true",
        "h1000_target_inside_segment": "true",
        "h1000_source_ts_ns": str(decision_ts_ns + 1_000_000_000),
        "h1000_no_new_information": "false",
        "h2000_covered": str(covered_2000).lower(),
        "h2000_target_inside_segment": str(covered_2000).lower(),
        "h2000_source_ts_ns": str(h2000_source_ts_ns) if covered_2000 else "",
        "h2000_no_new_information": "false",
    }
    return row


def _write_timeline_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(filename="", mode="wb", fileobj=path.open("wb"), mtime=0) as gz:
        wrapper = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        writer = csv.DictWriter(wrapper, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        wrapper.detach()


def _timeline_row(ts_ns: int, *, bid_px: float = 99.99, ask_px: float = 100.0, qty: float = 4.0) -> dict[str, str]:
    row = {
        "common_ts_ns": str(ts_ns),
        "binance_bid_1_px": str(bid_px),
        "binance_ask_1_px": str(ask_px),
    }
    for level in range(1, 6):
        row[f"binance_bid_{level}_qty"] = str(qty)
        row[f"binance_ask_{level}_qty"] = str(qty)
        if level > 1:
            row[f"binance_bid_{level}_px"] = str(bid_px - (level - 1) * 0.01)
            row[f"binance_ask_{level}_px"] = str(ask_px + (level - 1) * 0.01)
    return row


def _m1_package(
    tmp_path: Path,
    rows: list[dict],
    timeline_rows: list[dict] | dict[str, list[dict]] | None = None,
) -> Path:
    root = tmp_path / "m1"
    rows_by_segment: dict[str, list[dict]] = {}
    for row in rows:
        rows_by_segment.setdefault(row["segment_id"], []).append(row)
    episode_outputs = {}
    for segment_id, segment_rows in sorted(rows_by_segment.items()):
        episode_path = root / "episodes" / f"{segment_id}.csv.gz"
        _write_episode_csv(episode_path, segment_rows)
        episode_outputs[segment_id] = {
            "path": f"episodes/{segment_id}.csv.gz",
            "row_count": len(segment_rows),
            "sha256": hierarchy.sha256_file(episode_path),
        }
    input_provenance = []
    if timeline_rows is not None:
        timelines_by_segment = (
            timeline_rows
            if isinstance(timeline_rows, dict)
            else {"segment_0001": timeline_rows}
        )
        for segment_id, segment_timeline_rows in sorted(timelines_by_segment.items()):
            timeline_path = root / "timeline" / f"{segment_id}.csv.gz"
            _write_timeline_csv(timeline_path, segment_timeline_rows)
            input_provenance.append(
                {
                    "role": "timeline",
                    "segment_id": segment_id,
                    "path": str(timeline_path),
                    "row_count": len(segment_timeline_rows),
                    "sha256": hierarchy.sha256_file(timeline_path),
                }
            )
    manifest = {
        "task_id": "0801T001",
        "schema_version": "hyperliquid_liquidity_response_motif_v2",
        "passes": True,
        "counts": {"primary_episode_count": len(rows)},
        "horizons": {
            "primary_ms": [1000, 2000],
            "primary_tolerance_ms": {"1000": 250, "2000": 250},
        },
        "outputs": {"episodes": episode_outputs},
        "input_provenance": input_provenance,
    }
    _write_json(root / "motif_episode_manifest.json", manifest)
    return root


def _read_atoms(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _as_atom_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{**row, "atom_id": row["episode_id"]} for row in rows]


def _write_valid_episode_v2_freeze(
    episode_v2_dir: Path,
    *,
    parameters: dict | None = None,
) -> dict:
    outputs = {}
    for name, spec in hierarchy.EPISODE_V2_OUTPUT_SPECS.items():
        path = episode_v2_dir.parent / spec["path"]
        if path.suffix == ".gz":
            hierarchy._write_gzip_csv(path, [], spec["fields"])
        else:
            hierarchy._write_csv(path, [], spec["fields"])
        outputs[name] = {
            "path": spec["path"],
            "row_count": 0,
            "sha256": hierarchy.sha256_file(path),
        }
    manifest = {
        "schema_version": hierarchy.EPISODE_V2_SCHEMA_VERSION,
        "boundary_version": hierarchy.EPISODE_V2_BOUNDARY_VERSION,
        "boundary_parameters": parameters
        if parameters is not None
        else hierarchy._episode_v2_boundary_parameters(),
        "discovery_segments": hierarchy.DISCOVERY_SEGMENTS,
        "calibration_contract": {
            "scope": "discovery_only",
            "discovery_atom_count": 0,
            "discovery_atom_rows_sha256": hierarchy.hashlib.sha256(b"").hexdigest(),
            "outcomes_read": False,
            "heldout_structural_calibration_used": False,
        },
        "outputs": outputs,
    }
    _write_json(episode_v2_dir / "episode_manifest.json", manifest)
    return manifest


def test_builds_one_to_one_shock_atom_catalog(tmp_path: Path) -> None:
    rows = [
        _episode_row(episode_id="segment_0001-000001"),
        _episode_row(
            episode_id="segment_0001-000002",
            covered_2000=False,
            h2000_source_ts_ns=2_500_000_000,
        ),
    ]
    m1 = _m1_package(tmp_path, rows)
    output = tmp_path / "case_hierarchy"

    manifest = hierarchy.build_shock_atom_catalog(
        m1_dir=m1,
        output_dir=output,
        expected_atom_count=2,
    )

    assert manifest["passes"] is True
    assert manifest["counts"]["atom_count"] == 2
    atoms = _read_atoms(output / "atom" / "shock_atom_catalog.csv.gz")
    assert [row["atom_id"] for row in atoms] == [
        "segment_0001-000001",
        "segment_0001-000002",
    ]
    assert atoms[0]["visible_from_ts_ns"] == rows[0]["decision_ts_ns"]
    assert atoms[0]["outcome_known_from_ts_ns"] == rows[0]["h2000_source_ts_ns"]
    assert atoms[1]["outcome_known_from_ts_ns"] == ""
    assert atoms[1]["h2000_source_ts_ns"] == ""
    assert (output / "atom" / "shock_atom_manifest.json").is_file()
    assert (output / "case_hierarchy_manifest.json").is_file()


def test_rejects_m1_episode_sha_drift(tmp_path: Path) -> None:
    m1 = _m1_package(tmp_path, [_episode_row(episode_id="segment_0001-000001")])
    episode_path = m1 / "episodes" / "segment_0001.csv.gz"
    _write_episode_csv(
        episode_path,
        [_episode_row(episode_id="segment_0001-000001", h2000_source_ts_ns=3_001_000_000)],
    )

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="SHA drift"):
        hierarchy.build_shock_atom_catalog(
            m1_dir=m1,
            output_dir=tmp_path / "out",
            expected_atom_count=1,
        )


def test_rejects_duplicate_atom_ids(tmp_path: Path) -> None:
    row = _episode_row(episode_id="segment_0001-000001")
    m1 = _m1_package(tmp_path, [row, row])

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="duplicate atom id"):
        hierarchy.build_shock_atom_catalog(
            m1_dir=m1,
            output_dir=tmp_path / "out",
            expected_atom_count=2,
        )


def test_rejects_cross_segment_identity(tmp_path: Path) -> None:
    row = _episode_row(episode_id="segment_0002-000001", segment_id="segment_0001")
    m1 = _m1_package(tmp_path, [row])

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="atom/segment identity"):
        hierarchy.build_shock_atom_catalog(
            m1_dir=m1,
            output_dir=tmp_path / "out",
            expected_atom_count=1,
        )


def test_rejects_future_response_outside_tolerance(tmp_path: Path) -> None:
    row = _episode_row(
        episode_id="segment_0001-000001",
        h2000_source_ts_ns=3_251_000_001,
    )
    m1 = _m1_package(tmp_path, [row])

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="exceeds tolerance"):
        hierarchy.build_shock_atom_catalog(
            m1_dir=m1,
            output_dir=tmp_path / "out",
            expected_atom_count=1,
        )


def _build_atom_fixture(
    tmp_path: Path, rows: list[dict], timeline_rows: list[dict]
) -> tuple[Path, Path]:
    m1 = _m1_package(tmp_path, rows, timeline_rows)
    hierarchy_root = tmp_path / "case_hierarchy"
    hierarchy.build_shock_atom_catalog(
        m1_dir=m1,
        output_dir=hierarchy_root,
        expected_atom_count=len(rows),
    )
    return m1, hierarchy_root


def _read_gzip_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_episode_builder_merges_dense_clusters_without_recovery(tmp_path: Path) -> None:
    rows = [
        _episode_row(
            episode_id="segment_0001-000001",
            shock_ts_ns=1_000_000_000,
            decision_ts_ns=1_010_000_000,
        ),
        _episode_row(
            episode_id="segment_0001-000002",
            shock_ts_ns=1_050_000_000,
            decision_ts_ns=1_060_000_000,
        ),
        _episode_row(
            episode_id="segment_0001-000003",
            shock_ts_ns=1_180_000_000,
            decision_ts_ns=1_190_000_000,
        ),
    ]
    timeline_rows = [
        _timeline_row(1_070_000_000, ask_px=100.08, qty=1.0),
        _timeline_row(1_130_000_000, ask_px=100.08, qty=1.0),
    ]
    m1, hierarchy_root = _build_atom_fixture(tmp_path, rows, timeline_rows)

    manifest = hierarchy.build_shock_cluster_episodes(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )

    assert manifest["passes"] is True
    assert manifest["counts"]["atom_count"] == 3
    assert manifest["counts"]["cluster_count"] == 2
    assert manifest["counts"]["continuous_flow_episode_count"] == 1
    membership = _read_gzip_rows(hierarchy_root / "episode" / "shock_atom_membership.csv.gz")
    assert len({row["atom_id"] for row in membership}) == 3
    assert len({row["flow_episode_id"] for row in membership}) == 1


def test_episode_builder_stops_at_recovery_checkpoint(tmp_path: Path) -> None:
    rows = [
        _episode_row(
            episode_id="segment_0001-000001",
            shock_ts_ns=1_000_000_000,
            decision_ts_ns=1_010_000_000,
        ),
        _episode_row(
            episode_id="segment_0001-000002",
            shock_ts_ns=1_130_000_000,
            decision_ts_ns=1_140_000_000,
        ),
    ]
    timeline_rows = [
        _timeline_row(1_050_000_000, ask_px=100.0, qty=4.0),
        _timeline_row(1_110_000_000, ask_px=100.0, qty=4.0),
    ]
    m1, hierarchy_root = _build_atom_fixture(tmp_path, rows, timeline_rows)

    manifest = hierarchy.build_shock_cluster_episodes(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )

    assert manifest["counts"]["cluster_count"] == 2
    assert manifest["counts"]["continuous_flow_episode_count"] == 2
    audit = _read_gzip_rows(hierarchy_root / "episode" / "episode_boundary_audit.csv.gz")
    assert audit[0]["recovery_status"] == "recovery_checkpoint"
    assert audit[0]["merged"] == "false"


def test_episode_builder_fails_closed_on_missing_recovery_evidence(tmp_path: Path) -> None:
    rows = [
        _episode_row(
            episode_id="segment_0001-000001",
            shock_ts_ns=1_000_000_000,
            decision_ts_ns=1_010_000_000,
        ),
        _episode_row(
            episode_id="segment_0001-000002",
            shock_ts_ns=1_130_000_000,
            decision_ts_ns=1_140_000_000,
        ),
    ]
    timeline_rows = [_timeline_row(1_090_000_000, ask_px=100.0, qty=4.0)]
    m1, hierarchy_root = _build_atom_fixture(tmp_path, rows, timeline_rows)

    hierarchy.build_shock_cluster_episodes(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )

    audit = _read_gzip_rows(hierarchy_root / "episode" / "episode_boundary_audit.csv.gz")
    membership = _read_gzip_rows(hierarchy_root / "episode" / "shock_atom_membership.csv.gz")
    assert audit[0]["recovery_status"] == "missing_recovery_evidence"
    assert audit[0]["merged"] == "false"
    assert len({row["flow_episode_id"] for row in membership}) == 2


def test_episode_builder_marks_long_flow_and_reversal_phase(tmp_path: Path) -> None:
    rows = []
    timeline_rows = []
    shock_times = [1_000_000_000 + index * 200_000_000 for index in range(27)]
    for index, shock_ts in enumerate(shock_times, start=1):
        direction = 1 if index == 1 else -1
        rows.append(
            _episode_row(
                episode_id=f"segment_0001-{index:06d}",
                shock_ts_ns=shock_ts,
                decision_ts_ns=shock_ts + 10_000_000,
                direction_sign=direction,
                aggressor_side="buy" if direction == 1 else "sell",
            )
        )
    for left, right in zip(shock_times, shock_times[1:]):
        timeline_rows.append(_timeline_row(left + 40_000_000, ask_px=100.08, qty=1.0))
        timeline_rows.append(_timeline_row(right - 40_000_000, ask_px=100.08, qty=1.0))
    m1, hierarchy_root = _build_atom_fixture(tmp_path, rows, timeline_rows)

    hierarchy.build_shock_cluster_episodes(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )

    episodes = _read_gzip_rows(
        hierarchy_root / "episode" / "continuous_flow_episode_catalog.csv.gz"
    )
    phases = _read_gzip_rows(hierarchy_root / "episode" / "flow_episode_phases.csv.gz")
    assert any(row["long_flow_case"] == "true" for row in episodes)
    assert any(row["phase_type"] == "reversal" for row in phases)


def test_baseline_stage_publishes_contract_without_supported_claim(tmp_path: Path) -> None:
    rows = [
        _episode_row(
            episode_id=f"segment_0001-{index:06d}",
            shock_ts_ns=1_000_000_000 + index * 200_000_000,
            decision_ts_ns=1_010_000_000 + index * 200_000_000,
        )
        for index in range(1, 4)
    ]
    timeline_rows = [
        _timeline_row(1_050_000_000, ask_px=100.08, qty=1.0),
        _timeline_row(1_110_000_000, ask_px=100.08, qty=1.0),
    ]
    m1, hierarchy_root = _build_atom_fixture(tmp_path, rows, timeline_rows)
    hierarchy.build_shock_cluster_episodes(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )

    manifest = hierarchy.build_baseline_motif_prototypes(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )

    assert manifest["passes"] is True
    baseline_manifest = json.loads(
        (hierarchy_root / "baseline" / "baseline_manifest.json").read_text()
    )
    motif_manifest = json.loads(
        (hierarchy_root / "motif" / "motif_manifest.json").read_text()
    )
    assert baseline_manifest["official_baseline"] == "matched_neighbor_median"
    assert motif_manifest["diagnostics"]["reason"] == "insufficient_discovery_rows"
    assert (hierarchy_root / "baseline" / "frozen_research_contract.json").is_file()
    assert (hierarchy_root / "baseline" / "heldout_consumption_manifest.json").is_file()


def test_v2_phase_requires_rolling_sign_reversal(tmp_path: Path) -> None:
    del tmp_path
    atoms = [
        _episode_row(
            episode_id="segment_0001-000001",
            shock_ts_ns=1_000_000_000,
            direction_sign=1,
            shock_impact_ratio=1.0,
        ),
        _episode_row(
            episode_id="segment_0001-000002",
            shock_ts_ns=1_040_000_000,
            direction_sign=-1,
            aggressor_side="sell",
            shock_impact_ratio=0.1,
        ),
        _episode_row(
            episode_id="segment_0001-000003",
            shock_ts_ns=1_060_000_000,
            direction_sign=-1,
            aggressor_side="sell",
            shock_impact_ratio=0.1,
        ),
        _episode_row(
            episode_id="segment_0001-000004",
            shock_ts_ns=1_200_000_000,
            direction_sign=-1,
            aggressor_side="sell",
            shock_impact_ratio=0.4,
        ),
        _episode_row(
            episode_id="segment_0001-000005",
            shock_ts_ns=1_220_000_000,
            direction_sign=-1,
            aggressor_side="sell",
            shock_impact_ratio=0.4,
        ),
    ]

    phases = hierarchy._phase_rows_v2(
        "segment_0001-E000001", _as_atom_rows(atoms)
    )

    assert len(phases) == 2
    assert phases[0]["start_atom_id"] == "segment_0001-000001"
    assert phases[0]["end_atom_id"] == "segment_0001-000003"
    assert phases[1]["start_atom_id"] == "segment_0001-000004"
    assert phases[1]["phase_type"] == "reversal"
    assert phases[1]["direction_sign"] == -1
    assert phases[1]["phase_algorithm"] == "rolling_signed_impact_v1"
    assert phases[1]["reversal_confirmation_atoms"] == 2


def test_v2_phase_rejects_single_atom_reversal_perturbation() -> None:
    atoms = [
        _episode_row(
            episode_id="segment_0001-000001",
            shock_ts_ns=1_000_000_000,
            direction_sign=1,
            shock_impact_ratio=0.4,
        ),
        _episode_row(
            episode_id="segment_0001-000002",
            shock_ts_ns=1_200_000_000,
            direction_sign=-1,
            aggressor_side="sell",
            shock_impact_ratio=0.4,
        ),
        _episode_row(
            episode_id="segment_0001-000003",
            shock_ts_ns=1_220_000_000,
            direction_sign=1,
            shock_impact_ratio=0.4,
        ),
    ]

    phases = hierarchy._phase_rows_v2(
        "segment_0001-E000001", _as_atom_rows(atoms)
    )

    assert len(phases) == 1
    assert phases[0]["phase_type"] == "onset"


def test_episode_v2_freeze_rejects_parameter_drift(tmp_path: Path) -> None:
    episode_v2_dir = tmp_path / "episode_v2"
    parameters = hierarchy._episode_v2_boundary_parameters()
    _write_valid_episode_v2_freeze(
        episode_v2_dir,
        parameters={**parameters, "rolling_phase_window_ms": 125},
    )

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="parameter or phase"):
        hierarchy._validate_existing_episode_v2_freeze(
            episode_v2_dir, parameters
        )


def test_episode_v2_freeze_accepts_exact_contract(tmp_path: Path) -> None:
    episode_v2_dir = tmp_path / "episode_v2"
    parameters = hierarchy._episode_v2_boundary_parameters()
    manifest = _write_valid_episode_v2_freeze(episode_v2_dir)

    assert hierarchy._validate_existing_episode_v2_freeze(
        episode_v2_dir, parameters
    ) == (manifest["outputs"], manifest["calibration_contract"])


def test_episode_v2_freeze_rejects_missing_required_output(tmp_path: Path) -> None:
    episode_v2_dir = tmp_path / "episode_v2"
    manifest = _write_valid_episode_v2_freeze(episode_v2_dir)
    manifest["outputs"].pop("flow_episode_phases")
    _write_json(episode_v2_dir / "episode_manifest.json", manifest)

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="key set"):
        hierarchy._validate_existing_episode_v2_freeze(
            episode_v2_dir, hierarchy._episode_v2_boundary_parameters()
        )


def test_episode_v2_freeze_rejects_extra_output(tmp_path: Path) -> None:
    episode_v2_dir = tmp_path / "episode_v2"
    manifest = _write_valid_episode_v2_freeze(episode_v2_dir)
    manifest["outputs"]["unexpected"] = dict(
        manifest["outputs"]["flow_episode_phases"]
    )
    _write_json(episode_v2_dir / "episode_manifest.json", manifest)

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="key set"):
        hierarchy._validate_existing_episode_v2_freeze(
            episode_v2_dir, hierarchy._episode_v2_boundary_parameters()
        )


def test_episode_v2_candidate_drift_is_rejected() -> None:
    frozen = {
        name: {
            "path": spec["path"],
            "row_count": 1,
            "sha256": f"{index:064x}",
        }
        for index, (name, spec) in enumerate(
            hierarchy.EPISODE_V2_OUTPUT_SPECS.items(), start=1
        )
    }
    candidate = {name: dict(output) for name, output in frozen.items()}
    candidate["flow_episode_phases"]["sha256"] = "f" * 64

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="structural output"):
        hierarchy._assert_episode_v2_candidate_matches_frozen(frozen, candidate)


def test_episode_v2_discovery_input_drift_is_rejected() -> None:
    frozen = {
        "discovery_atom_count": 3,
        "discovery_atom_rows_sha256": "a" * 64,
    }

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="input SHA drift"):
        hierarchy._assert_episode_v2_discovery_input_matches_frozen(
            frozen,
            discovery_atom_count=3,
            discovery_input_sha="b" * 64,
        )


def test_v2_phase_conservation_rejects_missing_atom() -> None:
    membership = [
        {
            "flow_episode_id": "segment_0001-E000001",
            "episode_atom_seq": 1,
            "atom_id": "segment_0001-000001",
        },
        {
            "flow_episode_id": "segment_0001-E000001",
            "episode_atom_seq": 2,
            "atom_id": "segment_0001-000002",
        },
    ]
    episodes = [
        {
            "flow_episode_id": "segment_0001-E000001",
            "atom_count": 2,
        }
    ]
    phases = [
        {
            "flow_episode_id": "segment_0001-E000001",
            "phase_seq": 1,
            "phase_type": "onset",
            "atom_count": 1,
            "start_atom_id": "segment_0001-000001",
            "end_atom_id": "segment_0001-000001",
        }
    ]

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="do not conserve"):
        hierarchy._validate_phase_membership_conservation(
            membership=membership,
            episodes=episodes,
            phases=phases,
        )


def test_episode_v2_builder_rejects_algorithm_result_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        _episode_row(
            episode_id=f"{segment_id}-000001",
            segment_id=segment_id,
            shock_ts_ns=1_000_000_000,
            decision_ts_ns=1_010_000_000,
        )
        for segment_id in hierarchy.DISCOVERY_SEGMENTS
    ]
    timelines = {
        segment_id: [
            _timeline_row(900_000_000),
            _timeline_row(1_100_000_000),
        ]
        for segment_id in hierarchy.DISCOVERY_SEGMENTS
    }
    m1 = _m1_package(tmp_path, rows, timelines)
    hierarchy_root = tmp_path / "case_hierarchy"
    hierarchy.build_shock_atom_catalog(
        m1_dir=m1,
        output_dir=hierarchy_root,
        expected_atom_count=3,
    )
    hierarchy.build_shock_cluster_episodes_v2(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )
    phase_path = hierarchy_root / "episode_v2" / "flow_episode_phases.csv.gz"
    frozen_phase_sha = hierarchy.sha256_file(phase_path)
    original_phase_rows = hierarchy._phase_rows_v2

    def drifted_phase_rows(*args, **kwargs):
        result = original_phase_rows(*args, **kwargs)
        result[0]["rolling_signed_impact_end"] = (
            float(result[0]["rolling_signed_impact_end"]) + 1.0
        )
        return result

    monkeypatch.setattr(hierarchy, "_phase_rows_v2", drifted_phase_rows)

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="structural output"):
        hierarchy.build_shock_cluster_episodes_v2(
            hierarchy_dir=hierarchy_root,
            m1_dir=m1,
        )
    assert hierarchy.sha256_file(phase_path) == frozen_phase_sha


def test_episode_v2_builder_rejects_discovery_atom_input_drift(
    tmp_path: Path,
) -> None:
    rows = [
        _episode_row(
            episode_id=f"{segment_id}-000001",
            segment_id=segment_id,
            shock_ts_ns=1_000_000_000,
            decision_ts_ns=1_010_000_000,
        )
        for segment_id in hierarchy.DISCOVERY_SEGMENTS
    ]
    timelines = {
        segment_id: [
            _timeline_row(900_000_000),
            _timeline_row(1_100_000_000),
        ]
        for segment_id in hierarchy.DISCOVERY_SEGMENTS
    }
    m1 = _m1_package(tmp_path, rows, timelines)
    hierarchy_root = tmp_path / "case_hierarchy"
    hierarchy.build_shock_atom_catalog(
        m1_dir=m1,
        output_dir=hierarchy_root,
        expected_atom_count=3,
    )
    hierarchy.build_shock_cluster_episodes_v2(
        hierarchy_dir=hierarchy_root,
        m1_dir=m1,
    )
    episode_manifest_path = hierarchy_root / "episode_v2" / "episode_manifest.json"
    frozen_episode_manifest_sha = hierarchy.sha256_file(episode_manifest_path)
    atom_path = hierarchy_root / "atom" / "shock_atom_catalog.csv.gz"
    atom_rows = _read_gzip_rows(atom_path)
    atom_rows[0]["basis_mid_bps"] = "999"
    hierarchy._write_gzip_csv(atom_path, atom_rows, hierarchy.ATOM_FIELDS)
    atom_manifest_path = hierarchy_root / "atom" / "shock_atom_manifest.json"
    atom_manifest = json.loads(atom_manifest_path.read_text())
    atom_manifest["outputs"]["shock_atom_catalog"]["sha256"] = hierarchy.sha256_file(
        atom_path
    )
    _write_json(atom_manifest_path, atom_manifest)

    with pytest.raises(hierarchy.CaseHierarchyBuildError, match="input SHA drift"):
        hierarchy.build_shock_cluster_episodes_v2(
            hierarchy_dir=hierarchy_root,
            m1_dir=m1,
        )
    assert hierarchy.sha256_file(episode_manifest_path) == frozen_episode_manifest_sha


def test_discovery_atom_hash_ignores_heldout_rows() -> None:
    discovery = {
        segment_id: [
            _episode_row(
                episode_id=f"{segment_id}-000001",
                segment_id=segment_id,
            )
        ]
        for segment_id in hierarchy.DISCOVERY_SEGMENTS
    }
    all_rows = {
        **discovery,
        "segment_0004": [
            _episode_row(
                episode_id="segment_0004-000001",
                segment_id="segment_0004",
                shock_impact_ratio=0.4,
            )
        ],
    }
    selected = hierarchy._discovery_atoms(all_rows)
    first_hash = hierarchy._atom_subset_sha256(selected)
    all_rows["segment_0004"][0]["shock_impact_ratio"] = "999"

    assert hierarchy._atom_subset_sha256(hierarchy._discovery_atoms(all_rows)) == first_hash
