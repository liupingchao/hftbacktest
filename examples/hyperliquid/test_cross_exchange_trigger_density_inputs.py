from __future__ import annotations

import csv
import gzip
import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cross_exchange_trigger_density_inputs as inputs  # noqa: E402
from cross_exchange_trigger_density_core import TRIGGER_AUDIT_SCHEMA  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _timeline_fields() -> list[str]:
    return list(inputs.TIMELINE_REQUIRED_FIELDS)


def _timeline_rows() -> list[dict[str, str]]:
    rows = []
    for seq, ts, bid, ask, bid_qty, ask_qty in (
        (1, 110, 99.0, 101.0, 1.0, 2.0),
        (2, 110, 100.0, 102.0, 3.0, 4.0),
        (3, 130, 100.0, 103.0, 5.0, 6.0),
        (4, 150, 101.0, 103.0, 7.0, 8.0),
    ):
        row = {
            "campaign_id": "campaign",
            "segment_id": "segment_0001",
            "profile_id": "skhynix",
            "common_seq": str(seq),
            "common_ts_ns": str(ts),
            "binance_bid_1_px": str(bid),
            "binance_ask_1_px": str(ask),
        }
        for level in range(1, 6):
            row[f"binance_bid_{level}_qty"] = str(bid_qty + level - 1)
            row[f"binance_ask_{level}_qty"] = str(ask_qty + level - 1)
        rows.append(row)
    return rows


def _trigger_row(**overrides: str) -> dict[str, str]:
    row = {field: "0" for field in TRIGGER_AUDIT_SCHEMA}
    row.update(
        {
            "campaign_id": "campaign",
            "segment_id": "segment_0001",
            "profile_id": "skhynix",
            "candidate_seq": "1",
            "aggressor_side": "buy",
            "direction_sign": "1",
            "pre_state_ts_ns": "110",
            "pre_best_px": "102.0",
            "pre_best_qty": "4.0",
            "shock_ts_ns": "140",
            "impact_ratio": "0.5",
            "decision_ts_ns": "145",
            "primary_episode": "true",
            "rejection_reason": "",
        }
    )
    row.update(overrides)
    return row


def _make_stage1(root: Path) -> Path:
    stage1 = root / "stage1"
    artifact = stage1 / "frozen_research_contract.json"
    _write_json(artifact, {"contract": "fixture"})
    artifacts = [
        {
            "path": "frozen_research_contract.json",
            "bytes": artifact.stat().st_size,
            "sha256": _sha(artifact),
        }
    ]
    core = inputs._canonical_json_sha256(artifacts)
    _write_json(
        stage1 / "research_manifest.json",
        {"artifacts": artifacts, "core_package_sha256": core},
    )
    return stage1


def _make_fixture(root: Path) -> tuple[inputs.FrozenSessionSpec, Path]:
    trigger_dir = root / "trigger"
    campaign_dir = root / "campaign"
    trigger_dir.mkdir(parents=True)
    timeline_dir = campaign_dir / "segments/segment_0001/skhynix"
    sample_dir = timeline_dir / "sample"
    sample_dir.mkdir(parents=True)

    trigger_path = trigger_dir / "trigger_audit.csv.gz"
    with gzip.open(trigger_path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIGGER_AUDIT_SCHEMA)
        writer.writeheader()
        writer.writerow(_trigger_row())
    trigger_sha = _sha(trigger_path)
    _write_json(
        trigger_dir / "motif_episode_manifest.json",
        {
            "schema_version": inputs.TRIGGER_MANIFEST_SCHEMA_VERSION,
            "campaign_id": "campaign",
            "counts": {
                "candidate_count": 1,
                "primary_episode_count": 1,
                "segment_count": 1,
            },
            "outputs": {
                "trigger_audit": {
                    "path": "trigger_audit.csv.gz",
                    "row_count": 1,
                    "sha256": trigger_sha,
                }
            },
        },
    )
    timeline_path = timeline_dir / "common_l2_timeline.csv.gz"
    with gzip.open(timeline_path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_timeline_fields())
        writer.writeheader()
        writer.writerows(_timeline_rows())
    timeline_sha = _sha(timeline_path)
    timeline_manifest = {
        "schema_version": inputs.TIMELINE_MANIFEST_SCHEMA_VERSION,
        "campaign_id": "campaign",
        "segment_id": "segment_0001",
        "profile_id": "skhynix",
        "passes": True,
        "failures": [],
        "timestamp_regression_count": 0,
        "future_join_count": 0,
        "timeline_row_count": 4,
        "timeline_sha256": timeline_sha,
        "first_common_ts_ns": 110,
        "last_common_ts_ns": 150,
        "timeline_file": str(timeline_path),
    }
    _write_json(
        timeline_dir / "common_l2_timeline_manifest.json",
        timeline_manifest,
    )
    trigger_manifest_path = trigger_dir / "motif_episode_manifest.json"
    trigger_manifest = json.loads(
        trigger_manifest_path.read_text(encoding="utf-8")
    )
    trigger_manifest["passes"] = True
    trigger_manifest["input_provenance"] = [
        {
            "path": str(timeline_path),
            "role": "timeline",
            "segment_id": "segment_0001",
            "row_count": 4,
            "sha256": timeline_sha,
        }
    ]
    _write_json(trigger_manifest_path, trigger_manifest)
    _write_json(
        campaign_dir / "campaign_manifest.json",
        {
            "schema_version": inputs.CAMPAIGN_MANIFEST_SCHEMA_VERSION,
            "campaign_id": "campaign",
            "passes": True,
            "cross_segment_continuity_claimed": False,
            "degraded_intervals": [],
            "segments": [
                {
                    "segment_id": "segment_0001",
                    "manifest": str(
                        campaign_dir / "segments/segment_0001/segment_manifest.json"
                    ),
                }
            ],
        },
    )
    quality = {
        "passes": True,
        "failures": [],
        "degraded_intervals": [],
        "reconnect_policy": {
            "hard_fail_tracks": ["binance", "fast_market", "standard_l2"]
        },
    }
    _write_json(
        campaign_dir / "segments/segment_0001/segment_manifest.json",
        {
            "schema_version": inputs.CAMPAIGN_MANIFEST_SCHEMA_VERSION,
            "campaign_id": "campaign",
            "segment_id": "segment_0001",
            "segment_index": 1,
            "passes": True,
            "fresh_snapshots": True,
            "cross_segment_continuity_claimed": False,
            "profiles": {
                "skhynix": {
                    "child": {"profile_id": "skhynix"},
                    "symbols": {
                        "binance": inputs.BINANCE_SYMBOL,
                        "hyperliquid": inputs.HYPERLIQUID_COIN,
                    },
                    "timeline": timeline_manifest,
                    "timeline_manifest": str(
                        timeline_dir / "common_l2_timeline_manifest.json"
                    ),
                    "strict_quality": quality,
                }
            },
        },
    )
    _write_json(
        sample_dir / "binance_public_raw/collection_manifest.json",
        {
            "schema_version": inputs.BINANCE_COLLECTOR_SCHEMA_VERSION,
            "exchange": "binance_usdm_futures",
            "symbol": inputs.BINANCE_SYMBOL,
            "local_start_ts": 100,
            "local_end_ts": 200,
            "reconnect_count": 0,
            "connection_attempt_count": 1,
            "disconnect_events": [],
            "depth_snapshot_bridge_count": 1,
            "depth_snapshot_bridge_valid": True,
            "depth_continuity_gap_count": 0,
        },
    )
    _write_json(
        sample_dir / "hyperliquid_public_sample/collection_manifest.json",
        {
            "schema_version": inputs.HYPERLIQUID_COLLECTOR_SCHEMA_VERSION,
            "exchange": "hyperliquid",
            "coin": inputs.HYPERLIQUID_COIN,
            "track_id": "fast_market",
            "reconnect_count": 0,
            "connection_attempt_count": 1,
            "disconnect_events": [],
        },
    )
    _write_json(
        sample_dir
        / "hyperliquid_public_sample/research_tracks/standard_l2"
        / "collection_manifest.json",
        {
            "schema_version": inputs.HYPERLIQUID_COLLECTOR_SCHEMA_VERSION,
            "exchange": "hyperliquid",
            "coin": inputs.HYPERLIQUID_COIN,
            "track_id": "standard_l2",
            "reconnect_count": 0,
            "connection_attempt_count": 1,
            "disconnect_events": [],
        },
    )
    spec = inputs.FrozenSessionSpec(
        session_id="fixture",
        trigger_path=trigger_dir,
        timeline_campaign_path=campaign_dir,
        expected_campaign_id="campaign",
        expected_segment_count=1,
        expected_candidate_count=1,
        expected_confirmed_count=1,
        expected_trigger_sha256=trigger_sha,
    )
    return spec, campaign_dir


def _load_fixture(tmp_path: Path) -> inputs.BoundSession:
    spec, _ = _make_fixture(tmp_path)
    stage1 = _make_stage1(tmp_path)
    original = inputs.EXPECTED_STAGE1_CORE_PACKAGE_SHA256
    fixture_core = json.loads(
        (stage1 / "research_manifest.json").read_text(encoding="utf-8")
    )["core_package_sha256"]
    inputs.EXPECTED_STAGE1_CORE_PACKAGE_SHA256 = fixture_core
    try:
        bound = inputs.load_bound_sessions(tmp_path, stage1, session_specs=(spec,))
        return bound.load_session("fixture")
    finally:
        inputs.EXPECTED_STAGE1_CORE_PACKAGE_SHA256 = original


def _rewrite_trigger(
    tmp_path: Path, row: dict[str, str], fields=None
) -> inputs.FrozenSessionSpec:
    path = tmp_path / "trigger/trigger_audit.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields or TRIGGER_AUDIT_SCHEMA)
        writer.writeheader()
        writer.writerow(row)
    sha = _sha(path)
    manifest_path = tmp_path / "trigger/motif_episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["trigger_audit"]["sha256"] = sha
    _write_json(manifest_path, manifest)
    return inputs.FrozenSessionSpec(
        "fixture",
        tmp_path / "trigger",
        tmp_path / "campaign",
        "campaign",
        1,
        1,
        1,
        sha,
    )


def _rewrite_timeline(tmp_path: Path, rows: list[dict[str, str]], fields=None) -> None:
    path = tmp_path / "campaign/segments/segment_0001/skhynix/common_l2_timeline.csv.gz"
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=fields or _timeline_fields(),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = path.with_name("common_l2_timeline_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "timeline_sha256": _sha(path),
            "timeline_row_count": len(rows),
            "first_common_ts_ns": int(rows[0]["common_ts_ns"]),
            "last_common_ts_ns": int(rows[-1]["common_ts_ns"]),
        }
    )
    _write_json(manifest_path, manifest)
    segment_manifest_path = (
        tmp_path / "campaign/segments/segment_0001/segment_manifest.json"
    )
    segment_manifest = json.loads(segment_manifest_path.read_text(encoding="utf-8"))
    segment_manifest["profiles"]["skhynix"]["timeline"] = manifest
    _write_json(segment_manifest_path, segment_manifest)
    trigger_manifest_path = tmp_path / "trigger/motif_episode_manifest.json"
    trigger_manifest = json.loads(
        trigger_manifest_path.read_text(encoding="utf-8")
    )
    timeline_provenance = next(
        row
        for row in trigger_manifest["input_provenance"]
        if row["role"] == "timeline"
    )
    timeline_provenance.update(
        {
            "path": str(path),
            "row_count": len(rows),
            "sha256": _sha(path),
        }
    )
    _write_json(trigger_manifest_path, trigger_manifest)


def _load_custom(tmp_path: Path, spec: inputs.FrozenSessionSpec) -> inputs.BoundSession:
    stage1 = _make_stage1(tmp_path)
    old = inputs.EXPECTED_STAGE1_CORE_PACKAGE_SHA256
    fixture_core = json.loads(
        (stage1 / "research_manifest.json").read_text(encoding="utf-8")
    )["core_package_sha256"]
    inputs.EXPECTED_STAGE1_CORE_PACKAGE_SHA256 = fixture_core
    try:
        return inputs.load_bound_sessions(
            tmp_path, stage1, session_specs=(spec,)
        ).load_session("fixture")
    finally:
        inputs.EXPECTED_STAGE1_CORE_PACKAGE_SHA256 = old


def test_fixture_success_uses_requested_single_session_api(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, _ = _make_fixture(tmp_path)
    stage1 = _make_stage1(tmp_path)
    fixture_core = json.loads(
        (stage1 / "research_manifest.json").read_text(encoding="utf-8")
    )["core_package_sha256"]
    monkeypatch.setattr(
        inputs,
        "EXPECTED_STAGE1_CORE_PACKAGE_SHA256",
        fixture_core,
    )
    session = inputs.load_bound_session(
        spec,
        stage1,
        source_root=tmp_path,
    )
    assert session.session_id == "fixture"
    assert len(session.candidates) == 1


def test_legal_duplicate_uses_highest_common_seq_and_not_shock_asof(
    tmp_path: Path,
) -> None:
    session = _load_fixture(tmp_path)
    row = session.merging_candidates[0]
    assert row["pre_state_common_seq"] == 2
    assert row["binance_pre_mid_px"] == 101.0
    assert row["binance_pre_mid_px"] != 101.5
    assert session.timelines_by_segment["segment_0001"]["ts_ns"] == [110, 110, 130, 150]
    assert session.segment_bindings[0].duplicate_timestamp_group_count == 1


def test_legal_equal_timestamps_retain_common_seq_order(tmp_path: Path) -> None:
    spec, _ = _make_fixture(tmp_path)
    rows = _timeline_rows()
    duplicate = dict(rows[2])
    duplicate["common_seq"] = "4"
    rows[3]["common_seq"] = "5"
    rows.insert(3, duplicate)
    _rewrite_timeline(tmp_path, rows)
    session = _load_custom(tmp_path, spec)
    timeline = session.timelines_by_segment["segment_0001"]
    assert timeline["ts_ns"] == [110, 110, 130, 130, 150]
    assert session.segment_bindings[0].duplicate_timestamp_group_count == 2


def test_hash_count_and_schema_drift_fail_closed(tmp_path: Path) -> None:
    spec, _ = _make_fixture(tmp_path)
    (tmp_path / "trigger/trigger_audit.csv.gz").write_bytes(b"drift")
    with pytest.raises(inputs.InputBindingError, match="sha256"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    spec, _ = _make_fixture(tmp_path)
    manifest_path = tmp_path / "trigger/motif_episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["trigger_audit"]["row_count"] = 2
    _write_json(manifest_path, manifest)
    with pytest.raises(inputs.InputBindingError, match="row_count"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    spec, _ = _make_fixture(tmp_path)
    fields = list(TRIGGER_AUDIT_SCHEMA) + ["future_outcome"]
    row = _trigger_row()
    row["future_outcome"] = "1"
    spec = _rewrite_trigger(tmp_path, row, fields)
    with pytest.raises(inputs.InputBindingError, match="outcome-like"):
        _load_custom(tmp_path, spec)


@pytest.mark.parametrize(
    ("relative_path", "field", "value", "message"),
    [
        ("campaign_manifest.json", "campaign_id", "other", "campaign_id"),
        (
            "segments/segment_0001/segment_manifest.json",
            "segment_id",
            "segment_0002",
            "segment identity",
        ),
        (
            "segments/segment_0001/skhynix/sample/binance_public_raw/"
            "collection_manifest.json",
            "symbol",
            "OTHERUSDT",
            "collector symbol",
        ),
        (
            "segments/segment_0001/skhynix/common_l2_timeline_manifest.json",
            "profile_id",
            "other",
            "timeline profile",
        ),
    ],
)
def test_manifest_identity_drift_fails_closed(
    tmp_path: Path,
    relative_path: str,
    field: str,
    value: object,
    message: str,
) -> None:
    spec, campaign = _make_fixture(tmp_path)
    path = campaign / relative_path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(path, manifest)
    with pytest.raises(inputs.InputBindingError, match=message):
        _load_custom(tmp_path, spec)


def test_timeline_sha_count_and_schema_drift_fail_closed(tmp_path: Path) -> None:
    spec, campaign = _make_fixture(tmp_path)
    path = campaign / "segments/segment_0001/skhynix/common_l2_timeline_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["timeline_sha256"] = "0" * 64
    _write_json(path, manifest)
    with pytest.raises(inputs.InputBindingError, match="timeline sha256"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    spec, campaign = _make_fixture(tmp_path)
    path = campaign / "segments/segment_0001/skhynix/common_l2_timeline_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["timeline_row_count"] = 5
    _write_json(path, manifest)
    segment_path = campaign / "segments/segment_0001/segment_manifest.json"
    segment = json.loads(segment_path.read_text(encoding="utf-8"))
    segment["profiles"]["skhynix"]["timeline"]["timeline_row_count"] = 5
    _write_json(segment_path, segment)
    with pytest.raises(inputs.InputBindingError, match="timeline row count"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    spec, _ = _make_fixture(tmp_path)
    fields = _timeline_fields()
    fields.remove("binance_ask_5_qty")
    _rewrite_timeline(tmp_path, _timeline_rows(), fields)
    with pytest.raises(inputs.InputBindingError, match="missing required fields"):
        _load_custom(tmp_path, spec)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reconnect_count", 1, "Binance reconnect"),
        ("depth_snapshot_bridge_count", 0, "snapshot bridge count"),
        ("depth_snapshot_bridge_count", 2, "snapshot bridge count"),
    ],
)
def test_core_reconnect_and_snapshot_contract(
    tmp_path: Path, field: str, value: int, message: str
) -> None:
    spec, campaign = _make_fixture(tmp_path)
    path = (
        campaign
        / "segments/segment_0001/skhynix/sample/binance_public_raw"
        / "collection_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[field] = value
    _write_json(path, manifest)
    with pytest.raises(inputs.InputBindingError, match=message):
        _load_custom(tmp_path, spec)


def test_aux_reconnect_does_not_create_binance_epoch(tmp_path: Path) -> None:
    spec, campaign = _make_fixture(tmp_path)
    path = campaign / "campaign_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["degraded_intervals"] = [
        {"track_id": "asset_context", "reason": "websocket_reconnect"}
    ]
    _write_json(path, manifest)
    session = _load_custom(tmp_path, spec)
    assert session.merging_candidates[0]["connection_epoch_id"] == "0"


@pytest.mark.parametrize(
    ("relative_path", "track"),
    [
        ("hyperliquid_public_sample/collection_manifest.json", "fast_market"),
        (
            "hyperliquid_public_sample/research_tracks/standard_l2/"
            "collection_manifest.json",
            "standard_l2",
        ),
    ],
)
def test_hyperliquid_core_reconnect_fails(
    tmp_path: Path, relative_path: str, track: str
) -> None:
    spec, campaign = _make_fixture(tmp_path)
    path = campaign / "segments/segment_0001/skhynix/sample" / relative_path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["reconnect_count"] = 1
    _write_json(path, manifest)
    with pytest.raises(inputs.InputBindingError, match=f"{track} reconnect"):
        _load_custom(tmp_path, spec)


def test_structural_end_is_not_timeline_end(tmp_path: Path) -> None:
    session = _load_fixture(tmp_path)
    row = session.merging_candidates[0]
    assert row["segment_end_ts_ns"] == 200
    assert session.segment_bindings[0].observable_last_ts_ns == 150


def test_trigger_timeline_provenance_drift_fails_closed(tmp_path: Path) -> None:
    spec, _ = _make_fixture(tmp_path)
    manifest_path = tmp_path / "trigger/motif_episode_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["input_provenance"][0]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(inputs.InputBindingError, match="provenance sha256"):
        _load_custom(tmp_path, spec)


def test_missing_pre_state_and_future_landmarks_fail(tmp_path: Path) -> None:
    spec, _ = _make_fixture(tmp_path)
    spec = _rewrite_trigger(
        tmp_path,
        _trigger_row(pre_state_ts_ns="120", pre_best_px="102.0", pre_best_qty="4.0"),
    )
    with pytest.raises(inputs.InputBindingError, match="absent from timeline"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    _make_fixture(tmp_path)
    spec = _rewrite_trigger(
        tmp_path, _trigger_row(shock_ts_ns="210", decision_ts_ns="220")
    )
    with pytest.raises(inputs.InputBindingError, match="outside structural span"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    _make_fixture(tmp_path)
    spec = _rewrite_trigger(
        tmp_path,
        _trigger_row(pre_state_ts_ns="150", shock_ts_ns="140"),
    )
    with pytest.raises(inputs.InputBindingError, match="pre-state/shock"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    _make_fixture(tmp_path)
    spec = _rewrite_trigger(
        tmp_path,
        _trigger_row(decision_ts_ns="201"),
    )
    with pytest.raises(inputs.InputBindingError, match="confirmed landmark"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    _make_fixture(tmp_path)
    spec = _rewrite_trigger(
        tmp_path,
        _trigger_row(shock_ts_ns="200", decision_ts_ns="201"),
    )
    with pytest.raises(inputs.InputBindingError, match="outside structural span"):
        _load_custom(tmp_path, spec)


def test_cross_segment_identity_fails(tmp_path: Path) -> None:
    _make_fixture(tmp_path)
    spec = _rewrite_trigger(tmp_path, _trigger_row(segment_id="segment_0002"))
    with pytest.raises(inputs.InputBindingError, match="trigger segments"):
        _load_custom(tmp_path, spec)


def test_outcome_like_path_and_timeline_field_fail(tmp_path: Path) -> None:
    role = inputs.InputRole(tmp_path / "future_outcome.csv", "common_l2_timeline")
    with pytest.raises(inputs.InputBindingError, match="forbidden input path"):
        inputs.inventory_files((role,))

    spec, _ = _make_fixture(tmp_path)
    fields = _timeline_fields() + ["markout_500ms"]
    rows = _timeline_rows()
    for row in rows:
        row["markout_500ms"] = "0"
    _rewrite_timeline(tmp_path, rows, fields)
    with pytest.raises(inputs.InputBindingError, match="outcome-like"):
        _load_custom(tmp_path, spec)


def test_timeline_order_and_common_seq_fail_closed(tmp_path: Path) -> None:
    spec, _ = _make_fixture(tmp_path)
    rows = _timeline_rows()
    rows[1]["common_seq"] = "1"
    _rewrite_timeline(tmp_path, rows)
    with pytest.raises(inputs.InputBindingError, match="must increase"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    spec, _ = _make_fixture(tmp_path)
    rows = _timeline_rows()
    rows[2]["common_seq"] = "1"
    _rewrite_timeline(tmp_path, rows)
    with pytest.raises(inputs.InputBindingError, match="common_seq"):
        _load_custom(tmp_path, spec)

    shutil.rmtree(tmp_path)
    tmp_path.mkdir()
    spec, _ = _make_fixture(tmp_path)
    rows = _timeline_rows()
    rows[2]["common_ts_ns"] = "109"
    _rewrite_timeline(tmp_path, rows)
    with pytest.raises(inputs.InputBindingError, match="must increase"):
        _load_custom(tmp_path, spec)


def test_stage1_sha_drift_and_full_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage1 = _make_stage1(tmp_path)
    inventory = inputs.inventory_stage1_package(stage1)
    assert {row.path for row in inventory} == {
        "frozen_research_contract.json",
        "research_manifest.json",
    }
    fixture_core = json.loads(
        (stage1 / "research_manifest.json").read_text(encoding="utf-8")
    )["core_package_sha256"]
    monkeypatch.setattr(
        inputs,
        "EXPECTED_STAGE1_CORE_PACKAGE_SHA256",
        fixture_core,
    )
    assert inputs.verify_stage1_package(stage1).core_package_sha256 == fixture_core
    monkeypatch.setattr(
        inputs,
        "EXPECTED_STAGE1_CORE_PACKAGE_SHA256",
        "0" * 64,
    )
    with pytest.raises(inputs.InputBindingError, match="accepted stage1"):
        inputs.verify_stage1_package(stage1)


def test_inventory_before_after_helper_fails_on_drift(tmp_path: Path) -> None:
    path = tmp_path / "common_l2_timeline.csv.gz"
    path.write_bytes(b"before")
    role = inputs.InputRole(path, "common_l2_timeline")
    before = inputs.inventory_files((role,))
    path.write_bytes(b"after")
    after = inputs.inventory_files((role,))
    with pytest.raises(inputs.InputBindingError, match="inventory changed"):
        inputs.assert_inventory_unchanged(before, after)


def test_real_input_smoke() -> None:
    source_root = Path("/Users/liu/Documents/hftbacktest")
    stage1 = Path(
        "/Users/liu/Documents/hftbacktest-0814t001-skhynix-episode-research/"
        "local_live_analysis/skhynix_trigger_aligned_episode_research_v1"
    )
    if not source_root.is_dir() or not stage1.is_dir():
        pytest.skip("frozen local research inputs are unavailable")
    bound = inputs.load_all_bound_sessions(stage1, source_root=source_root)
    session = bound.load_session("aug04")
    assert len(session.candidates) == 67468
    assert sum(candidate.primary_episode for candidate in session.candidates) == 43253
    assert len(session.merging_candidates) == 67468
    assert session.segment_bindings[0].duplicate_timestamp_group_count >= 1
