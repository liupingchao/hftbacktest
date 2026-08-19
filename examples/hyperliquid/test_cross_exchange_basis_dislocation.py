from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import polars as pl
import pytest

from examples.hyperliquid.cross_exchange_postprocess import (
    basis_dislocation as module,
)
from examples.hyperliquid.cross_exchange_postprocess.basis_dislocation import (
    BasisDislocationError,
    build_basis_dislocation,
    build_segment_state,
)
from examples.hyperliquid.cross_exchange_postprocess.contracts import sha256_file


BINANCE_FIELDS = [
    "segment_id",
    "event_seq",
    "source_raw_seq",
    "local_ts_ns",
    "exchange_ts_ns",
    "event_type",
    "symbol",
    "connection_epoch_id",
    "degraded",
    "degraded_interval_ids",
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
HYPERLIQUID_FIELDS = [
    "segment_id",
    "event_seq",
    "source_raw_seq",
    "source_item_index",
    "local_ts_ns",
    "exchange_ts_ns",
    "event_type",
    "coin",
    "connection_epoch_id",
    "degraded",
    "degraded_interval_ids",
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
BASE_TS = 1_780_000_000_000_000_000


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_gzip_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _bbo_rows(
    *,
    count: int,
    crossed_hyperliquid: bool = False,
) -> tuple[list[dict], list[dict]]:
    binance = []
    hyperliquid = []
    for index in range(count):
        ts = BASE_TS + index * 1_000_000_000
        common_mid = 100.0 + index * 0.001
        dislocation = math.sin(index / 17.0) * 0.02
        b_mid = common_mid + dislocation
        h_mid = common_mid
        b_bid, b_ask = b_mid - 0.01, b_mid + 0.01
        h_bid, h_ask = h_mid - 0.01, h_mid + 0.01
        if crossed_hyperliquid and index == count // 2:
            h_ask = h_bid - 0.01
        binance.append(
            {
                "segment_id": "segment_0001",
                "event_seq": index + 1,
                "source_raw_seq": index + 1,
                "local_ts_ns": ts,
                "exchange_ts_ns": ts - 1_000_000,
                "event_type": "bookTicker",
                "symbol": "TESTUSDT",
                "connection_epoch_id": 0,
                "degraded": "false",
                "degraded_interval_ids": "",
                "update_id": index + 1,
                "bid_px": b_bid,
                "bid_qty": 2.0,
                "ask_px": b_ask,
                "ask_qty": 3.0,
            }
        )
        hyperliquid.append(
            {
                "segment_id": "segment_0001",
                "event_seq": index + 1,
                "source_raw_seq": index + 1,
                "source_item_index": 0,
                "local_ts_ns": ts,
                "exchange_ts_ns": ts - 2_000_000,
                "event_type": "bbo",
                "coin": "TEST",
                "connection_epoch_id": 0,
                "degraded": "false",
                "degraded_interval_ids": "",
                "bid_px": h_bid,
                "bid_qty": 4.0,
                "bid_n": 1,
                "ask_px": h_ask,
                "ask_qty": 5.0,
                "ask_n": 1,
            }
        )
    return binance, hyperliquid


def _segment_dir(tmp_path: Path, *, count: int = 1_001) -> Path:
    segment = tmp_path / "segment"
    binance, hyperliquid = _bbo_rows(count=count)
    _write_gzip_csv(
        segment / "binance_hot_events.csv.gz",
        BINANCE_FIELDS,
        binance,
    )
    _write_gzip_csv(
        segment / "hyperliquid_hot_events.csv.gz",
        HYPERLIQUID_FIELDS,
        hyperliquid,
    )
    return segment


def test_build_segment_state_is_trailing_only_and_preserves_identity(
    tmp_path: Path,
) -> None:
    segment = _segment_dir(tmp_path)
    epoch = (BASE_TS, BASE_TS + 1_000 * 1_000_000_000)
    frame, quality, _ = build_segment_state(
        campaign_id="fixture",
        profile_id="test",
        segment_id="segment_0001",
        segment_dir=segment,
        epoch=epoch,
        intervals=[],
    )
    assert quality["passes"] is True
    assert quality["future_join_count"] == 0
    assert quality["spread_invariant_max_abs_error"] <= 1e-8
    assert frame["trigger_track"][0] == "hyperliquid"
    assert frame.filter(pl.col("feature_eligible")).height > 0
    assert (
        frame["d_bh_q"]
        + frame["d_hb_q"]
        + frame["binance_spread_q"]
        + frame["hyperliquid_spread_q"]
    ).abs().max() <= 1e-8

    cutoff = BASE_TS + 950 * 1_000_000_000
    before = frame.filter(pl.col("decision_ts_ns") <= cutoff).select(
        "decision_ts_ns",
        "basis_mid_z",
        "d_bh_level_z",
        "d_hb_level_z",
    )
    hyper_path = segment / "hyperliquid_hot_events.csv.gz"
    with gzip.open(hyper_path, "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[-1]["bid_px"] = "101.0"
    rows[-1]["ask_px"] = "101.02"
    _write_gzip_csv(hyper_path, HYPERLIQUID_FIELDS, rows)
    changed, _, _ = build_segment_state(
        campaign_id="fixture",
        profile_id="test",
        segment_id="segment_0001",
        segment_dir=segment,
        epoch=epoch,
        intervals=[],
    )
    assert before.equals(
        changed.filter(pl.col("decision_ts_ns") <= cutoff).select(
            "decision_ts_ns",
            "basis_mid_z",
            "d_bh_level_z",
            "d_hb_level_z",
        ),
        null_equal=True,
    )


def test_masks_are_inclusive_and_auxiliary_does_not_block_basis(
    tmp_path: Path,
) -> None:
    segment = _segment_dir(tmp_path, count=8)
    intervals = [
        {
            "mask_type": "core_l2_reconnect_interval",
            "track_id": "hyperliquid_standard",
            "start_ns": BASE_TS + 2_000_000_000,
            "end_ns": BASE_TS + 3_000_000_000,
        },
        {
            "mask_type": "auxiliary_degraded_interval",
            "track_id": "asset_context",
            "start_ns": BASE_TS + 4_000_000_000,
            "end_ns": BASE_TS + 4_000_000_000,
        },
    ]
    frame, _, _ = build_segment_state(
        campaign_id="fixture",
        profile_id="test",
        segment_id="segment_0001",
        segment_dir=segment,
        epoch=(BASE_TS, BASE_TS + 7_000_000_000),
        intervals=intervals,
    )
    by_ts = {row["decision_ts_ns"]: row for row in frame.to_dicts()}
    assert by_ts[BASE_TS + 2_000_000_000]["core_masked"] is True
    assert by_ts[BASE_TS + 3_000_000_000]["core_masked"] is True
    assert by_ts[BASE_TS + 4_000_000_000]["core_masked"] is False
    assert by_ts[BASE_TS + 4_000_000_000]["auxiliary_masked"] is True


def test_fast_reconnect_clears_old_hyperliquid_state(tmp_path: Path) -> None:
    segment = tmp_path / "segment"
    binance, hyperliquid = _bbo_rows(count=8)
    for row in hyperliquid:
        if int(row["local_ts_ns"]) >= BASE_TS + 7_000_000_000:
            row["connection_epoch_id"] = 1
    same_timestamp_old_epoch = dict(hyperliquid[-1])
    same_timestamp_old_epoch["event_seq"] = 100
    same_timestamp_old_epoch["source_raw_seq"] = 100
    same_timestamp_old_epoch["connection_epoch_id"] = 0
    hyperliquid.append(same_timestamp_old_epoch)
    _write_gzip_csv(
        segment / "binance_hot_events.csv.gz",
        BINANCE_FIELDS,
        binance,
    )
    _write_gzip_csv(
        segment / "hyperliquid_hot_events.csv.gz",
        HYPERLIQUID_FIELDS,
        hyperliquid,
    )
    frame, quality, _ = build_segment_state(
        campaign_id="fixture",
        profile_id="test",
        segment_id="segment_0001",
        segment_dir=segment,
        epoch=(BASE_TS, BASE_TS + 7_000_000_000),
        intervals=[
            {
                "mask_type": "core_l2_reconnect_interval",
                "track_id": "hyperliquid_fast",
                "start_ns": BASE_TS + 5_000_000_000,
                "end_ns": BASE_TS + 6_000_000_000,
            }
        ],
    )
    timestamps = set(frame["decision_ts_ns"].to_list())
    assert BASE_TS + 5_000_000_000 not in timestamps
    assert BASE_TS + 6_000_000_000 not in timestamps
    assert quality["hyperliquid_old_epoch_bbo_suppressed_count"] == 3
    assert quality["old_hyperliquid_epoch_state_leak_count"] == 0
    recovered = frame.filter(
        pl.col("decision_ts_ns") == BASE_TS + 7_000_000_000
    ).row(0, named=True)
    assert recovered["hyperliquid_connection_epoch_id"] == 1
    assert math.isnan(recovered["d_bh_change_bps_100ms"])
    assert math.isnan(recovered["d_hb_change_bps_100ms"])
    assert recovered["basis_mid_z"] is None
    assert (
        quality["old_hyperliquid_state_forward_filled_across_reconnect"]
        is False
    )


def test_fast_reconnect_without_higher_epoch_bbo_fails_closed(
    tmp_path: Path,
) -> None:
    segment = _segment_dir(tmp_path, count=8)
    with pytest.raises(
        BasisDislocationError,
        match="no higher-epoch Hyperliquid BBO recovery",
    ):
        build_segment_state(
            campaign_id="fixture",
            profile_id="test",
            segment_id="segment_0001",
            segment_dir=segment,
            epoch=(BASE_TS, BASE_TS + 7_000_000_000),
            intervals=[
                {
                    "mask_type": "core_l2_reconnect_interval",
                    "track_id": "hyperliquid_fast",
                    "start_ns": BASE_TS + 5_000_000_000,
                    "end_ns": BASE_TS + 6_000_000_000,
                }
            ],
        )


def test_binance_reconnect_clears_old_state_and_restarts_features(
    tmp_path: Path,
) -> None:
    segment = tmp_path / "segment"
    binance, hyperliquid = _bbo_rows(count=1_102)
    recovery_ts = BASE_TS + 101 * 1_000_000_000
    for row in binance:
        if int(row["local_ts_ns"]) >= recovery_ts:
            row["connection_epoch_id"] = 1
    same_timestamp_old_epoch = dict(binance[101])
    same_timestamp_old_epoch["event_seq"] = 2_000
    same_timestamp_old_epoch["source_raw_seq"] = 2_000
    same_timestamp_old_epoch["connection_epoch_id"] = 0
    binance.append(same_timestamp_old_epoch)
    _write_gzip_csv(
        segment / "binance_hot_events.csv.gz",
        BINANCE_FIELDS,
        binance,
    )
    _write_gzip_csv(
        segment / "hyperliquid_hot_events.csv.gz",
        HYPERLIQUID_FIELDS,
        hyperliquid,
    )
    frame, quality, _ = build_segment_state(
        campaign_id="fixture",
        profile_id="test",
        segment_id="segment_0001",
        segment_dir=segment,
        epoch=(BASE_TS, BASE_TS + 1_101 * 1_000_000_000),
        intervals=[
            {
                "mask_type": "core_l2_reconnect_interval",
                "track_id": "binance",
                "start_ns": BASE_TS + 100 * 1_000_000_000,
                "end_ns": BASE_TS + 100 * 1_000_000_000,
            }
        ],
    )
    recovered = frame.filter(
        pl.col("decision_ts_ns") == recovery_ts
    ).row(0, named=True)
    before_warmup = frame.filter(
        pl.col("decision_ts_ns") == BASE_TS + 1_000 * 1_000_000_000
    ).row(0, named=True)
    at_warmup = frame.filter(
        pl.col("decision_ts_ns") == BASE_TS + 1_001 * 1_000_000_000
    ).row(0, named=True)
    assert recovered["binance_connection_epoch_id"] == 1
    assert math.isnan(recovered["d_bh_change_bps_100ms"])
    assert math.isnan(recovered["binance_volatility_60s_bps"])
    assert recovered["basis_mid_z"] is None
    assert before_warmup["feature_eligible"] is False
    assert at_warmup["feature_eligible"] is True
    assert quality["binance_reset_count"] == 1
    assert quality["binance_old_epoch_bbo_suppressed_count"] == 2
    assert quality["old_binance_epoch_state_leak_count"] == 0
    assert quality["old_binance_state_forward_filled_across_reconnect"] is False


def test_fast_reconnect_restarts_feature_warmup(tmp_path: Path) -> None:
    segment = tmp_path / "segment"
    binance, hyperliquid = _bbo_rows(count=1_102)
    recovery_ts = BASE_TS + 101 * 1_000_000_000
    for row in hyperliquid:
        if int(row["local_ts_ns"]) >= recovery_ts:
            row["connection_epoch_id"] = 1
    _write_gzip_csv(
        segment / "binance_hot_events.csv.gz",
        BINANCE_FIELDS,
        binance,
    )
    _write_gzip_csv(
        segment / "hyperliquid_hot_events.csv.gz",
        HYPERLIQUID_FIELDS,
        hyperliquid,
    )
    frame, _, _ = build_segment_state(
        campaign_id="fixture",
        profile_id="test",
        segment_id="segment_0001",
        segment_dir=segment,
        epoch=(BASE_TS, BASE_TS + 1_101 * 1_000_000_000),
        intervals=[
            {
                "mask_type": "core_l2_reconnect_interval",
                "track_id": "hyperliquid_fast",
                "start_ns": BASE_TS + 100 * 1_000_000_000,
                "end_ns": BASE_TS + 100 * 1_000_000_000,
            }
        ],
    )
    before_warmup = frame.filter(
        pl.col("decision_ts_ns") == BASE_TS + 1_000 * 1_000_000_000
    ).row(0, named=True)
    at_warmup = frame.filter(
        pl.col("decision_ts_ns") == BASE_TS + 1_001 * 1_000_000_000
    ).row(0, named=True)
    assert before_warmup["feature_eligible"] is False
    assert at_warmup["feature_eligible"] is True


def test_crossed_book_fails_closed(tmp_path: Path) -> None:
    segment = tmp_path / "segment"
    binance, hyperliquid = _bbo_rows(
        count=8,
        crossed_hyperliquid=True,
    )
    _write_gzip_csv(
        segment / "binance_hot_events.csv.gz",
        BINANCE_FIELDS,
        binance,
    )
    _write_gzip_csv(
        segment / "hyperliquid_hot_events.csv.gz",
        HYPERLIQUID_FIELDS,
        hyperliquid,
    )
    with pytest.raises(BasisDislocationError, match="crossed venue books"):
        build_segment_state(
            campaign_id="fixture",
            profile_id="test",
            segment_id="segment_0001",
            segment_dir=segment,
            epoch=(BASE_TS, BASE_TS + 7_000_000_000),
            intervals=[],
        )


def _dataset_fixture(tmp_path: Path) -> tuple[Path, Path]:
    event_store = tmp_path / "r0"
    alignment = tmp_path / "r1"
    segment = event_store / "segments" / "segment_0001"
    binance, hyperliquid = _bbo_rows(count=1_001)
    binance_path = segment / "binance_hot_events.csv.gz"
    hyper_path = segment / "hyperliquid_hot_events.csv.gz"
    _write_gzip_csv(binance_path, BINANCE_FIELDS, binance)
    _write_gzip_csv(hyper_path, HYPERLIQUID_FIELDS, hyperliquid)
    mask_path = event_store / "segment_and_mask_index.csv"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    with mask_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "campaign_id",
                "segment_id",
                "profile_id",
                "first_common_ts_ns",
                "last_common_ts_ns",
                "previous_segment_gap_ms",
                "cross_segment_continuity_claimed",
                "mask_type",
                "track_id",
                "mask_start_ts_ns",
                "mask_end_ts_ns",
                "duration_ms",
                "policy",
                "reason",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "campaign_id": "fixture",
                "segment_id": "segment_0001",
                "profile_id": "test",
                "first_common_ts_ns": BASE_TS,
                "last_common_ts_ns": BASE_TS + 1_000 * 1_000_000_000,
                "cross_segment_continuity_claimed": "false",
                "mask_type": "segment_epoch",
                "policy": "never_compute_across_segment_boundary",
                "reason": "fresh_exchange_snapshots",
            }
        )
    r0 = {
        "passes": True,
        "campaign_id": "fixture",
        "profile_id": "test",
        "segment_and_mask_index": {
            "path": "segment_and_mask_index.csv",
            "row_count": 1,
            "sha256": sha256_file(mask_path),
        },
        "segments": [
            {
                "segment_id": "segment_0001",
                "outputs": {
                    "binance_hot_events": {
                        "path": "segments/segment_0001/binance_hot_events.csv.gz",
                        "row_count": len(binance),
                        "sha256": sha256_file(binance_path),
                    },
                    "hyperliquid_hot_events": {
                        "path": "segments/segment_0001/hyperliquid_hot_events.csv.gz",
                        "row_count": len(hyperliquid),
                        "sha256": sha256_file(hyper_path),
                    },
                },
            }
        ],
    }
    r0_path = event_store / "research_input_manifest.json"
    _write_json(r0_path, r0)
    _write_json(
        alignment / "alignment_manifest.json",
        {
            "passes": True,
            "campaign_id": "fixture",
            "profile_id": "test",
            "exact_masks_pass": True,
            "exact_horizon_masks_pass": True,
            "reconciliation_pass": True,
            "source_manifest": {"sha256": sha256_file(r0_path)},
        },
    )
    return event_store, alignment


def _rebind_alignment_manifest(event_store: Path, alignment: Path) -> None:
    r0_path = event_store / "research_input_manifest.json"
    manifest_path = alignment / "alignment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_manifest"]["sha256"] = sha256_file(r0_path)
    _write_json(manifest_path, manifest)


def test_full_build_is_deterministic_and_binds_inputs(tmp_path: Path) -> None:
    event_store, alignment = _dataset_fixture(tmp_path)
    output_a = tmp_path / "output_a"
    output_b = tmp_path / "output_b"
    first = build_basis_dislocation(
        event_store_dir=event_store,
        alignment_dir=alignment,
        output_dir=output_a,
    )
    second = build_basis_dislocation(
        event_store_dir=event_store,
        alignment_dir=alignment,
        output_dir=output_b,
    )
    assert first["passes"] is True
    assert second["passes"] is True
    relative = "segments/segment_0001/basis_dislocation_state.csv.gz"
    assert sha256_file(output_a / relative) == sha256_file(output_b / relative)
    assert (
        sha256_file(output_a / "basis_quality_by_segment.csv")
        == sha256_file(output_b / "basis_quality_by_segment.csv")
    )


def test_full_build_rejects_preexisting_r0_artifact_tamper(
    tmp_path: Path,
) -> None:
    event_store, alignment = _dataset_fixture(tmp_path)
    hyper_path = (
        event_store
        / "segments"
        / "segment_0001"
        / "hyperliquid_hot_events.csv.gz"
    )
    with gzip.open(hyper_path, "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[-1]["bid_px"] = "123.0"
    _write_gzip_csv(hyper_path, HYPERLIQUID_FIELDS, rows)
    with pytest.raises(
        BasisDislocationError,
        match="R0 artifact SHA mismatch for segment_0001:hyperliquid_hot_events",
    ):
        build_basis_dislocation(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "output",
        )


def test_full_build_rejects_incomplete_r0_artifact_descriptor(
    tmp_path: Path,
) -> None:
    event_store, alignment = _dataset_fixture(tmp_path)
    r0_path = event_store / "research_input_manifest.json"
    r0 = json.loads(r0_path.read_text(encoding="utf-8"))
    del r0["segments"][0]["outputs"]["binance_hot_events"]["sha256"]
    _write_json(r0_path, r0)
    _rebind_alignment_manifest(event_store, alignment)
    with pytest.raises(
        BasisDislocationError,
        match="R0 artifact descriptor incomplete",
    ):
        build_basis_dislocation(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "output",
        )


def test_full_build_rejects_r0_artifact_row_count_mismatch(
    tmp_path: Path,
) -> None:
    event_store, alignment = _dataset_fixture(tmp_path)
    r0_path = event_store / "research_input_manifest.json"
    r0 = json.loads(r0_path.read_text(encoding="utf-8"))
    descriptor = r0["segments"][0]["outputs"]["binance_hot_events"]
    descriptor["row_count"] += 1
    _write_json(r0_path, r0)
    _rebind_alignment_manifest(event_store, alignment)
    with pytest.raises(
        BasisDislocationError,
        match="R0 artifact row count mismatch",
    ):
        build_basis_dislocation(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "output",
        )


def test_full_build_detects_input_mutation_during_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_store, alignment = _dataset_fixture(tmp_path)
    original = module.build_segment_state

    def mutating_build_segment_state(**kwargs):
        result = original(**kwargs)
        hyper_path = (
            event_store
            / "segments"
            / "segment_0001"
            / "hyperliquid_hot_events.csv.gz"
        )
        with gzip.open(
            hyper_path,
            "rt",
            encoding="utf-8",
            newline="",
        ) as fh:
            rows = list(csv.DictReader(fh))
        rows[-1]["ask_px"] = "124.0"
        _write_gzip_csv(hyper_path, HYPERLIQUID_FIELDS, rows)
        return result

    monkeypatch.setattr(
        module,
        "build_segment_state",
        mutating_build_segment_state,
    )
    with pytest.raises(
        BasisDislocationError,
        match="R0 artifact SHA mismatch",
    ):
        module.build_basis_dislocation(
            event_store_dir=event_store,
            alignment_dir=alignment,
            output_dir=tmp_path / "output",
        )
