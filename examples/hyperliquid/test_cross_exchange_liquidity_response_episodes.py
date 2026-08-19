from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import pytest

import cross_exchange_liquidity_response_episodes as motif


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_gzip_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _timeline_row(ts_ns: int, *, bid_qty: float, ask_qty: float) -> dict[str, str]:
    row = {
        "campaign_id": "campaign",
        "segment_id": "segment_0001",
        "profile_id": "skhynix",
        "common_ts_ns": str(ts_ns),
        "hyperliquid_fast_local_ts_ns": str(ts_ns - 1_000_000),
        "hyperliquid_fast_age_ms": "1",
    }
    for level in range(1, 6):
        row[f"binance_bid_{level}_px"] = str(99.0 - (level - 1) * 0.01)
        row[f"binance_ask_{level}_px"] = str(100.0 + (level - 1) * 0.01)
        row[f"binance_bid_{level}_qty"] = str(bid_qty if level == 1 else 2.0)
        row[f"binance_ask_{level}_qty"] = str(ask_qty if level == 1 else 2.0)
        row[f"hyperliquid_fast_bid_{level}_px"] = str(98.0 - (level - 1) * 0.01)
        row[f"hyperliquid_fast_ask_{level}_px"] = str(98.01 + (level - 1) * 0.01)
        row[f"hyperliquid_fast_bid_{level}_qty"] = str(5.0 + level)
        row[f"hyperliquid_fast_ask_{level}_qty"] = str(4.0 + level)
    return row


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    event_store = tmp_path / "r0"
    alignment = event_store / "alignment"
    segment_dir = event_store / "segments" / "segment_0001"
    timeline = tmp_path / "common_l2_timeline.csv.gz"
    base = 1_000_000_000

    timeline_rows = [
        _timeline_row(base, bid_qty=10.0, ask_qty=10.0),
        _timeline_row(base + 20_000_000, bid_qty=10.0, ask_qty=6.0),
        _timeline_row(base + 100_000_000, bid_qty=10.0, ask_qty=10.0),
        _timeline_row(base + 200_000_000, bid_qty=10.0, ask_qty=10.0),
        _timeline_row(base + 220_000_000, bid_qty=6.0, ask_qty=10.0),
        _timeline_row(base + 3_000_000_000, bid_qty=10.0, ask_qty=10.0),
    ]
    timeline_fields = list(timeline_rows[0])
    _write_gzip_csv(timeline, timeline_fields, timeline_rows)

    binance_fields = [
        "segment_id",
        "event_seq",
        "source_raw_seq",
        "local_ts_ns",
        "exchange_ts_ns",
        "event_type",
        "symbol",
        "update_id",
        "bid_px",
        "bid_qty",
        "ask_px",
        "ask_qty",
        "buyer_is_maker",
        "aggressor_side",
        "trade_px",
        "trade_qty",
        "trade_id",
    ]
    trade_specs = [
        (base + 10_000_000, "buy", 100.0, 2.0),
        (base + 11_000_000, "buy", 100.0, 2.0),
        (base + 12_000_000, "buy", 100.0, 4.0),
        (base + 14_000_000, "buy", 100.0, 4.0),
        (base + 210_000_000, "sell", 99.0, 4.0),
    ]
    binance_rows = []
    for index, (ts_ns, side, px, qty) in enumerate(trade_specs, 1):
        binance_rows.append(
            {
                "segment_id": "segment_0001",
                "event_seq": index,
                "source_raw_seq": index,
                "local_ts_ns": ts_ns,
                "exchange_ts_ns": ts_ns,
                "event_type": "trade",
                "symbol": "SKHYNIXUSDT",
                "update_id": "",
                "bid_px": "",
                "bid_qty": "",
                "ask_px": "",
                "ask_qty": "",
                "buyer_is_maker": side == "sell",
                "aggressor_side": side,
                "trade_px": px,
                "trade_qty": qty,
                "trade_id": index,
            }
        )
    binance_path = segment_dir / "binance_hot_events.csv.gz"
    _write_gzip_csv(binance_path, binance_fields, binance_rows)

    hl_fields = [
        "segment_id",
        "event_seq",
        "source_raw_seq",
        "source_item_index",
        "local_ts_ns",
        "exchange_ts_ns",
        "event_type",
        "coin",
        "bid_px",
        "bid_qty",
        "bid_n",
        "ask_px",
        "ask_qty",
        "ask_n",
        "trade_side",
        "trade_px",
        "trade_qty",
        "trade_id",
        "trade_hash",
        "trade_users_json",
    ]
    bbo_specs = [
        (base - 10_000_000, 98.00, 8.0, 98.01, 10.0),
        (base + 30_000_000, 98.00, 8.0, 98.01, 6.0),
        (base + 120_000_000, 98.00, 8.0, 98.01, 9.0),
        (base + 230_000_000, 97.99, 5.0, 98.01, 9.0),
        (base + 1_100_000_000, 98.01, 8.0, 98.02, 9.0),
        (base + 1_300_000_000, 97.99, 8.0, 98.00, 9.0),
        (base + 2_100_000_000, 98.02, 8.0, 98.03, 9.0),
        (base + 2_300_000_000, 97.99, 8.0, 98.00, 9.0),
    ]
    hl_rows = []
    for index, (ts_ns, bid_px, bid_qty, ask_px, ask_qty) in enumerate(bbo_specs, 1):
        hl_rows.append(
            {
                "segment_id": "segment_0001",
                "event_seq": index,
                "source_raw_seq": index,
                "source_item_index": 0,
                "local_ts_ns": ts_ns,
                "exchange_ts_ns": ts_ns,
                "event_type": "bbo",
                "coin": "xyz:SKHX",
                "bid_px": bid_px,
                "bid_qty": bid_qty,
                "bid_n": 1,
                "ask_px": ask_px,
                "ask_qty": ask_qty,
                "ask_n": 1,
                "trade_side": "",
                "trade_px": "",
                "trade_qty": "",
                "trade_id": "",
                "trade_hash": "",
                "trade_users_json": "",
            }
        )
    hl_path = segment_dir / "hyperliquid_hot_events.csv.gz"
    _write_gzip_csv(hl_path, hl_fields, hl_rows)

    normalized_outputs = {
        "binance_hot_events": {
            "path": "segments/segment_0001/binance_hot_events.csv.gz",
            "row_count": len(binance_rows),
            "sha256": motif.sha256_file(binance_path),
        },
        "hyperliquid_hot_events": {
            "path": "segments/segment_0001/hyperliquid_hot_events.csv.gz",
            "row_count": len(hl_rows),
            "sha256": motif.sha256_file(hl_path),
        },
    }
    segment_manifest = {
        "task_id": "0730T014",
        "passes": True,
        "campaign_id": "campaign",
        "profile_id": "skhynix",
        "segment_id": "segment_0001",
        "symbols": {
            "binance": "SKHYNIXUSDT",
            "hyperliquid": "xyz:SKHX",
        },
        "outputs": normalized_outputs,
        "segment_boundary": {
            "last_common_ts_ns": base + 3_000_000_000,
        },
        "source_files": {
            "timeline": {
                "path": str(timeline),
                "row_count": len(timeline_rows),
                "sha256": motif.sha256_file(timeline),
            }
        },
    }
    segment_manifest_path = segment_dir / "segment_event_store_manifest.json"
    _write_json(segment_manifest_path, segment_manifest)
    descriptor = {
        "segment_id": "segment_0001",
        "manifest_path": "segments/segment_0001/segment_event_store_manifest.json",
        "manifest_sha256": motif.sha256_file(segment_manifest_path),
        "outputs": normalized_outputs,
    }
    r0 = {
        "task_id": "0730T014",
        "passes": True,
        "campaign_id": "campaign",
        "profile_id": "skhynix",
        "symbols": {
            "binance": "SKHYNIXUSDT",
            "hyperliquid": "xyz:SKHX",
        },
        "segment_count": 1,
        "segments": [descriptor],
        "boundary": {
            "local_existing_data_only": True,
            "new_collection_performed": False,
        },
    }
    r0_path = event_store / "research_input_manifest.json"
    _write_json(r0_path, r0)
    alignment_manifest = {
        "task_id": "0730T016",
        "passes": True,
        "accepted_primary_horizons_ms": [1000, 2000],
        "diagnostic_horizons_ms": [10, 25, 50, 100, 250, 500],
        "horizon_tolerance_ms": {
            "10": 50,
            "25": 50,
            "50": 50,
            "100": 100,
            "250": 100,
            "500": 100,
            "1000": 250,
            "2000": 250,
        },
        "join_clock": "same_host_local_receipt_time_time_ns",
        "provenance_pass": True,
        "exact_masks_pass": True,
        "labels_pass": True,
        "reconciliation_pass": True,
        "source_hashes_unchanged": True,
        "r0_output_hashes_unchanged": True,
        "input_hashes_unchanged": True,
        "timestamp_regression_count": 0,
        "future_decision_join_count": 0,
        "cross_segment_label_count": 0,
        "source_manifest": {
            "path": str(r0_path.resolve()),
            "sha256": motif.sha256_file(r0_path),
        },
    }
    _write_json(alignment / "alignment_manifest.json", alignment_manifest)
    return event_store, alignment


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _refresh_input_manifests(event_store: Path, alignment: Path) -> None:
    r0_path = event_store / "research_input_manifest.json"
    r0 = json.loads(r0_path.read_text())
    descriptor = r0["segments"][0]
    segment_manifest_path = event_store / descriptor["manifest_path"]
    segment_manifest = json.loads(segment_manifest_path.read_text())
    for role in ("binance_hot_events", "hyperliquid_hot_events"):
        output = descriptor["outputs"][role]
        path = event_store / output["path"]
        rows = _read_gzip_csv(path)
        refreshed = {"row_count": len(rows), "sha256": motif.sha256_file(path)}
        output.update(refreshed)
        segment_manifest["outputs"][role].update(refreshed)
    timeline = Path(segment_manifest["source_files"]["timeline"]["path"])
    segment_manifest["source_files"]["timeline"].update(
        {
            "row_count": len(_read_gzip_csv(timeline)),
            "sha256": motif.sha256_file(timeline),
        }
    )
    _write_json(segment_manifest_path, segment_manifest)
    descriptor["manifest_sha256"] = motif.sha256_file(segment_manifest_path)
    _write_json(r0_path, r0)
    alignment_manifest_path = alignment / "alignment_manifest.json"
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    alignment_manifest["source_manifest"]["sha256"] = motif.sha256_file(r0_path)
    _write_json(alignment_manifest_path, alignment_manifest)


def test_builder_creates_directional_primary_episodes(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    output = tmp_path / "episodes"

    manifest = motif.build_liquidity_response_episodes(
        event_store_dir=event_store,
        alignment_dir=alignment,
        output_dir=output,
    )

    assert manifest["passes"] is True
    assert manifest["counts"]["primary_episode_count"] == 2
    assert manifest["counts"]["buy_episode_count"] == 1
    assert manifest["counts"]["sell_episode_count"] == 1
    audits = _read_gzip_csv(output / "trigger_audit.csv.gz")
    assert len(audits) == 2
    assert all(row["attribution"] == "trade_driven" for row in audits)
    assert all(row["primary_episode"] == "true" for row in audits)

    episodes = _read_gzip_csv(output / "episodes" / "segment_0001.csv.gz")
    buy = next(row for row in episodes if row["aggressor_side"] == "buy")
    sell = next(row for row in episodes if row["aggressor_side"] == "sell")
    assert float(buy["impact_ratio"]) == pytest.approx(1.2)
    assert buy["hl_withdrawal_observed"] == "true"
    assert buy["hl_replenishment_observed"] == "true"
    assert buy["h100_mode"] == "diagnostic_wall"
    assert buy["h1000_mode"] == "primary_response"
    assert buy["h1000_covered"] == "true"
    assert float(buy["h1000_effective_horizon_ms"]) >= 1000
    assert float(buy["h1000_adverse_markout_ticks"]) > 0
    assert float(sell["h1000_adverse_markout_ticks"]) > 0


def test_threshold_and_missing_confirmation_are_audited(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    binance = event_store / "segments" / "segment_0001" / "binance_hot_events.csv.gz"
    rows = _read_gzip_csv(binance)
    fields = list(rows[0])
    rows.append(
        {
            **rows[0],
            "event_seq": "99",
            "source_raw_seq": "99",
            "local_ts_ns": str(3_500_000_000),
            "exchange_ts_ns": str(3_500_000_000),
            "trade_qty": "4",
            "trade_id": "99",
        }
    )
    _write_gzip_csv(binance, fields, rows)
    _refresh_input_manifests(event_store, alignment)

    output = tmp_path / "episodes"
    manifest = motif.build_liquidity_response_episodes(
        event_store_dir=event_store,
        alignment_dir=alignment,
        output_dir=output,
    )

    assert manifest["counts"]["candidate_count"] == 3
    rejected = [
        row
        for row in _read_gzip_csv(output / "trigger_audit.csv.gz")
        if row["primary_episode"] == "false"
    ]
    assert rejected[0]["rejection_reason"] == "no_depth_confirmation_within_100ms"


def test_confirmation_event_can_publish_only_one_primary_episode(
    tmp_path: Path,
) -> None:
    event_store, alignment = _fixture(tmp_path)
    segment_dir = event_store / "segments" / "segment_0001"
    binance = segment_dir / "binance_hot_events.csv.gz"
    binance_rows = _read_gzip_csv(binance)
    fields = list(binance_rows[0])
    binance_rows.insert(
        4,
        {
            **binance_rows[0],
            "event_seq": "50",
            "source_raw_seq": "50",
            "local_ts_ns": str(1_070_000_000),
            "exchange_ts_ns": str(1_070_000_000),
            "trade_qty": "4",
            "trade_id": "50",
        },
    )
    binance_rows.sort(key=lambda row: int(row["local_ts_ns"]))
    _write_gzip_csv(binance, fields, binance_rows)

    segment_manifest = json.loads(
        (segment_dir / "segment_event_store_manifest.json").read_text()
    )
    timeline = Path(segment_manifest["source_files"]["timeline"]["path"])
    timeline_rows = _read_gzip_csv(timeline)
    timeline_fields = list(timeline_rows[0])
    timeline_rows[1]["binance_ask_1_qty"] = "10"
    timeline_rows[2]["binance_ask_1_qty"] = "6"
    _write_gzip_csv(timeline, timeline_fields, timeline_rows)
    _refresh_input_manifests(event_store, alignment)

    output = tmp_path / "episodes"
    motif.build_liquidity_response_episodes(
        event_store_dir=event_store,
        alignment_dir=alignment,
        output_dir=output,
    )

    audits = _read_gzip_csv(output / "trigger_audit.csv.gz")
    reuse = [row for row in audits if row["rejection_reason"] == "confirmation_reuse_excluded"]
    assert len(reuse) == 1


def test_attribution_excludes_post_decision_trade() -> None:
    base = 1_000_000_000
    pre = motif.TimelineState(
        ts_ns=base,
        binance_bid_px=(99.0, 98.99, 98.98, 98.97, 98.96),
        binance_bid_qty=(10.0, 2.0, 2.0, 2.0, 2.0),
        binance_ask_px=(100.0, 100.01, 100.02, 100.03, 100.04),
        binance_ask_qty=(10.0, 2.0, 2.0, 2.0, 2.0),
        fast_source_ts_ns=base - 1,
        fast_age_ms=0.0,
        fast_bid_px=(98.0, 97.99, 97.98, 97.97, 97.96),
        fast_bid_qty=(5.0, 5.0, 5.0, 5.0, 5.0),
        fast_ask_px=(98.01, 98.02, 98.03, 98.04, 98.05),
        fast_ask_qty=(5.0, 5.0, 5.0, 5.0, 5.0),
    )
    decision = motif.TimelineState(
        **{**pre.__dict__, "ts_ns": base + 5_000_000, "binance_ask_qty": (6.0, 2.0, 2.0, 2.0, 2.0)}
    )
    pre_bbo = motif.BboState(
        ts_ns=base - 1, bid_px=98.0, bid_qty=8.0, ask_px=98.01, ask_qty=10.0
    )
    burst = [
        {"ts_ns": base + 1_000_000, "side": "buy", "px": 100.0, "qty": 4.0},
        {"ts_ns": base + 8_000_000, "side": "buy", "px": 100.0, "qty": 6.0},
    ]

    audit, _, _ = motif._candidate_from_burst(
        burst,
        timeline=[pre, decision],
        timeline_ts=[pre.ts_ns, decision.ts_ns],
        bbo=[pre_bbo],
        bbo_ts=[pre_bbo.ts_ns],
        boundary_end_ns=base + 3_000_000_000,
        candidate_seq=1,
        campaign_id="campaign",
        segment_id="segment_0001",
        profile_id="skhynix",
    )

    assert audit is not None
    assert audit["touch_trade_qty"] == pytest.approx(10.0)
    assert audit["touch_trade_qty_through_decision"] == pytest.approx(4.0)
    assert audit["post_decision_burst_trade_count"] == 1
    assert audit["trade_explained_ratio"] == pytest.approx(1.0)


def test_bad_r1_source_sha_fails_closed_and_preserves_output(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    output = tmp_path / "episodes"
    output.mkdir()
    marker = output / "accepted.txt"
    marker.write_text("keep", encoding="utf-8")
    alignment_manifest_path = alignment / "alignment_manifest.json"
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    alignment_manifest["source_manifest"]["sha256"] = "0" * 64
    _write_json(alignment_manifest_path, alignment_manifest)

    with pytest.raises(motif.MotifBuildError, match="source manifest SHA"):
        motif.build_liquidity_response_episodes(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=output,
            clean_output=True,
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_primary_horizon_contract_must_match_r1(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    alignment_manifest_path = alignment / "alignment_manifest.json"
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    alignment_manifest["accepted_primary_horizons_ms"] = [500, 1000]
    _write_json(alignment_manifest_path, alignment_manifest)

    with pytest.raises(motif.MotifBuildError, match="primary horizons changed"):
        motif.build_liquidity_response_episodes(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "episodes",
        )


def test_r1_primary_tolerance_drift_fails_closed(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    alignment_manifest_path = alignment / "alignment_manifest.json"
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    alignment_manifest["horizon_tolerance_ms"]["1000"] = 999
    _write_json(alignment_manifest_path, alignment_manifest)

    with pytest.raises(motif.MotifBuildError, match="h1000 tolerance changed"):
        motif.build_liquidity_response_episodes(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "episodes",
        )


def test_r1_acceptance_gate_regression_fails_closed(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    alignment_manifest_path = alignment / "alignment_manifest.json"
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    alignment_manifest["reconciliation_pass"] = False
    _write_json(alignment_manifest_path, alignment_manifest)

    with pytest.raises(motif.MotifBuildError, match="acceptance gates"):
        motif.build_liquidity_response_episodes(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "episodes",
        )


def test_diagnostic_alignment_preserves_failure_without_blocking_structure(
    tmp_path: Path,
) -> None:
    event_store, alignment = _fixture(tmp_path)
    alignment_manifest_path = alignment / "alignment_manifest.json"
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    alignment_manifest["task_id"] = "0803T001"
    alignment_manifest["passes"] = False
    alignment_manifest["reconciliation_pass"] = False
    alignment_manifest["accepted_primary_horizons_ms"] = []
    alignment_manifest["diagnostic_horizons_ms"].extend([1000, 2000])
    _write_json(alignment_manifest_path, alignment_manifest)

    manifest = motif.build_liquidity_response_episodes(
        event_store_dir=event_store,
        alignment_dir=alignment,
        output_dir=tmp_path / "diagnostic-output",
        task_id="0803T002",
        expected_alignment_task_id="0803T001",
        diagnostic_alignment=True,
        schema_version=motif.DIAGNOSTIC_SCHEMA_VERSION,
    )

    assert manifest["passes"] is True
    assert manifest["diagnostic_mode"] is True
    assert manifest["formal_eligible"] is False
    assert manifest["source_alignment_passes"] is False
    assert manifest["source_alignment_reconciliation_pass"] is False


def test_segment_profile_identity_fails_closed(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    segment_manifest_path = (
        event_store / "segments" / "segment_0001" / "segment_event_store_manifest.json"
    )
    segment_manifest = json.loads(segment_manifest_path.read_text())
    segment_manifest["profile_id"] = "wrong-profile"
    _write_json(segment_manifest_path, segment_manifest)
    _refresh_input_manifests(event_store, alignment)

    with pytest.raises(motif.MotifBuildError, match="profile identity mismatch"):
        motif.build_liquidity_response_episodes(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "episodes",
        )


def test_segment_descriptor_output_drift_fails_closed(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    segment_manifest_path = (
        event_store / "segments" / "segment_0001" / "segment_event_store_manifest.json"
    )
    segment_manifest = json.loads(segment_manifest_path.read_text())
    segment_manifest["outputs"] = {
        "binance_hot_events": {"path": "wrong", "row_count": 0, "sha256": "0" * 64}
    }
    _write_json(segment_manifest_path, segment_manifest)
    r0_path = event_store / "research_input_manifest.json"
    r0 = json.loads(r0_path.read_text())
    r0["segments"][0]["manifest_sha256"] = motif.sha256_file(segment_manifest_path)
    _write_json(r0_path, r0)
    alignment_manifest_path = alignment / "alignment_manifest.json"
    alignment_manifest = json.loads(alignment_manifest_path.read_text())
    alignment_manifest["source_manifest"]["sha256"] = motif.sha256_file(r0_path)
    _write_json(alignment_manifest_path, alignment_manifest)

    with pytest.raises(motif.MotifBuildError, match="descriptor output contract"):
        motif.build_liquidity_response_episodes(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "episodes",
        )


def test_timeline_row_identity_fails_closed(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    segment_manifest_path = (
        event_store / "segments" / "segment_0001" / "segment_event_store_manifest.json"
    )
    segment_manifest = json.loads(segment_manifest_path.read_text())
    timeline_path = Path(segment_manifest["source_files"]["timeline"]["path"])
    rows = _read_gzip_csv(timeline_path)
    rows[2]["profile_id"] = "wrong-profile"
    _write_gzip_csv(timeline_path, list(rows[0]), rows)
    _refresh_input_manifests(event_store, alignment)

    with pytest.raises(motif.MotifBuildError, match="timeline identity mismatch"):
        motif.build_liquidity_response_episodes(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "episodes",
        )


def test_segment_manifest_change_during_build_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event_store, alignment = _fixture(tmp_path)
    original = motif._process_segment

    def mutate_manifest_after_processing(**kwargs):
        result = original(**kwargs)
        manifest_path = Path(kwargs["segment"]["_manifest_path"])
        manifest_path.write_text(
            manifest_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(motif, "_process_segment", mutate_manifest_after_processing)
    with pytest.raises(motif.MotifBuildError, match="input changed during build"):
        motif.build_liquidity_response_episodes(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "episodes",
        )


def test_atomic_directory_exchange_swaps_without_missing_path(tmp_path: Path) -> None:
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    (old / "value.txt").write_text("old", encoding="utf-8")
    (new / "value.txt").write_text("new", encoding="utf-8")

    motif._atomic_exchange_directories(old, new)

    assert old.is_dir()
    assert new.is_dir()
    assert (old / "value.txt").read_text(encoding="utf-8") == "new"
    assert (new / "value.txt").read_text(encoding="utf-8") == "old"


def test_clean_rebuild_atomically_replaces_existing_output(tmp_path: Path) -> None:
    event_store, alignment = _fixture(tmp_path)
    output = tmp_path / "episodes"
    motif.build_liquidity_response_episodes(
        event_store_dir=event_store,
        alignment_dir=alignment,
        output_dir=output,
    )
    marker = output / "old-publication.txt"
    marker.write_text("old", encoding="utf-8")

    manifest = motif.build_liquidity_response_episodes(
        event_store_dir=event_store,
        alignment_dir=alignment,
        output_dir=output,
        clean_output=True,
    )

    assert manifest["passes"] is True
    assert output.is_dir()
    assert not marker.exists()
    assert (output / "motif_episode_manifest.json").is_file()
