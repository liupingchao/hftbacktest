from __future__ import annotations

import csv
import gzip
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_alignment_acceptance as alignment


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_gzip_lines(path: Path, count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for index in range(count):
            fh.write(f"{index} {{}}\n")


def _write_gzip_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _timeline_row(
    ts_ns: int,
    *,
    trigger_track: str,
    binance_bid: float,
    binance_ask: float,
    fast_bid: float = 100.0,
    fast_ask: float = 101.0,
) -> dict:
    row = {
        "segment_id": "segment_0001",
        "common_ts_ns": ts_ns,
        "trigger_track": trigger_track,
        "binance_local_ts_ns": ts_ns,
        "binance_connection_epoch_id": 0,
        "binance_bid_1_px": binance_bid,
        "binance_ask_1_px": binance_ask,
        "hyperliquid_fast_local_ts_ns": ts_ns,
        "hyperliquid_fast_bid_1_px": fast_bid,
        "hyperliquid_fast_ask_1_px": fast_ask,
        "hyperliquid_fast_age_ms": 0,
        "hyperliquid_standard_local_ts_ns": ts_ns,
        "hyperliquid_standard_bid_1_px": fast_bid,
        "hyperliquid_standard_ask_1_px": fast_ask,
        "hyperliquid_standard_age_ms": 0,
    }
    for level in range(1, 6):
        row[f"hyperliquid_fast_bid_{level}_px"] = fast_bid - level / 100
        row[f"hyperliquid_fast_ask_{level}_px"] = fast_ask + level / 100
        row[f"hyperliquid_standard_bid_{level}_px"] = fast_bid - level / 100
        row[f"hyperliquid_standard_ask_{level}_px"] = fast_ask + level / 100
    return row


def _event_store(tmp_path: Path, *, extreme_fast_price: bool = False) -> Path:
    root = tmp_path / "event-store"
    segment_dir = root / "segments" / "segment_0001"
    campaign_id = "campaign"
    profile_id = "skhynix"
    ms = 1_000_000
    bbo_times = [
        150,
        210,
        225,
        250,
        300,
        310,
        325,
        350,
        400,
        450,
        550,
        700,
        800,
        1200,
        1300,
        2200,
        2300,
    ]
    bbo_times = [value * ms for value in bbo_times]
    fast_bid = 1.0 if extreme_fast_price else 100.0
    fast_ask = 2.0 if extreme_fast_price else 101.0
    timeline_rows = [
        _timeline_row(
            100 * ms,
            trigger_track="hyperliquid_standard",
            binance_bid=100,
            binance_ask=101,
            fast_bid=fast_bid,
            fast_ask=fast_ask,
        )
    ]
    for ts_ns in bbo_times:
        binance_bid, binance_ask = (
            (101, 102) if ts_ns >= 300 * ms else (100, 101)
        )
        timeline_rows.append(
            _timeline_row(
                ts_ns,
                trigger_track="hyperliquid_fast",
                binance_bid=binance_bid,
                binance_ask=binance_ask,
                fast_bid=fast_bid,
                fast_ask=fast_ask,
            )
        )
    timeline_fields = list(timeline_rows[0])
    timeline_path = tmp_path / "sources" / "timeline.csv.gz"
    _write_gzip_csv(timeline_path, timeline_fields, timeline_rows)

    binance_rows = [
        {
            "segment_id": "segment_0001",
            "event_seq": 1,
            "source_raw_seq": 1,
            "local_ts_ns": 50 * ms,
            "exchange_ts_ns": 49 * ms,
            "event_type": "bookTicker",
            "connection_epoch_id": 0,
            "degraded": "false",
            "degraded_interval_ids": "",
            "bid_px": 99,
            "ask_px": 100,
        },
        {
            "segment_id": "segment_0001",
            "event_seq": 2,
            "source_raw_seq": 2,
            "local_ts_ns": 200 * ms,
            "exchange_ts_ns": 199 * ms,
            "event_type": "bookTicker",
            "connection_epoch_id": 0,
            "degraded": "false",
            "degraded_interval_ids": "",
            "bid_px": 100,
            "ask_px": 101,
        },
        {
            "segment_id": "segment_0001",
            "event_seq": 3,
            "source_raw_seq": 3,
            "local_ts_ns": 250 * ms,
            "exchange_ts_ns": 249 * ms,
            "event_type": "bookTicker",
            "connection_epoch_id": 0,
            "degraded": "false",
            "degraded_interval_ids": "",
            "bid_px": 100,
            "ask_px": 101,
        },
        {
            "segment_id": "segment_0001",
            "event_seq": 4,
            "source_raw_seq": 4,
            "local_ts_ns": 300 * ms,
            "exchange_ts_ns": 299 * ms,
            "event_type": "bookTicker",
            "connection_epoch_id": 0,
            "degraded": "false",
            "degraded_interval_ids": "",
            "bid_px": 101,
            "ask_px": 102,
        },
    ]
    binance_fields = list(binance_rows[0])
    binance_output = segment_dir / "binance_hot_events.csv.gz"
    _write_gzip_csv(binance_output, binance_fields, binance_rows)

    hyper_rows = []
    for index, ts_ns in enumerate(bbo_times):
        changed_then_reverted = ts_ns == 210 * ms
        hyper_rows.append(
            {
            "segment_id": "segment_0001",
            "event_seq": index + 1,
            "local_ts_ns": ts_ns,
            "exchange_ts_ns": ts_ns - 1,
            "event_type": "bbo",
            "bid_px": 100.01 if changed_then_reverted else 100,
            "ask_px": 101.01 if changed_then_reverted else 101,
        }
        )
    hyper_fields = list(hyper_rows[0])
    hyper_output = segment_dir / "hyperliquid_hot_events.csv.gz"
    _write_gzip_csv(hyper_output, hyper_fields, hyper_rows)
    aux_output = segment_dir / "hyperliquid_auxiliary_events.csv.gz"
    _write_gzip_csv(aux_output, ["segment_id", "event_type"], [])

    source_specs = {
        "timeline": (timeline_path, len(timeline_rows)),
        "binance": (tmp_path / "sources" / "binance.gz", 4),
        "hyperliquid_fast": (
            tmp_path / "sources" / "hyperliquid_fast.gz",
            len(hyper_rows),
        ),
        "standard_l2": (tmp_path / "sources" / "standard.gz", 1),
        "asset_context": (tmp_path / "sources" / "asset.gz", 1),
        "main_all_mids": (tmp_path / "sources" / "main.gz", 1),
        "target_dex_all_mids": (tmp_path / "sources" / "target.gz", 1),
    }
    for source_id, (path, count) in source_specs.items():
        if source_id != "timeline":
            _write_gzip_lines(path, count)

    outputs = {
        "binance_hot_events": {
            "path": "segments/segment_0001/binance_hot_events.csv.gz",
            "row_count": len(binance_rows),
            "sha256": alignment.sha256_file(binance_output),
        },
        "hyperliquid_hot_events": {
            "path": "segments/segment_0001/hyperliquid_hot_events.csv.gz",
            "row_count": len(hyper_rows),
            "sha256": alignment.sha256_file(hyper_output),
        },
        "hyperliquid_auxiliary_events": {
            "path": "segments/segment_0001/hyperliquid_auxiliary_events.csv.gz",
            "row_count": 0,
            "sha256": alignment.sha256_file(aux_output),
        },
    }
    segment_manifest = {
        "passes": True,
        "campaign_id": campaign_id,
        "profile_id": profile_id,
        "segment_id": "segment_0001",
        "segment_boundary": {
            "first_common_ts_ns": 100 * ms,
            "last_common_ts_ns": 3000 * ms,
            "previous_segment_gap_ms": None,
            "cross_segment_continuity_claimed": False,
        },
        "source_files": {
            source_id: {
                "path": str(path),
                "sha256": alignment.sha256_file(path),
                **({"row_count": count} if source_id == "timeline" else {}),
            }
            for source_id, (path, count) in source_specs.items()
        },
        "outputs": outputs,
        "reconciliation": {
            "binance": {
                "source_message_count_by_event_type": {"bookTicker": 4},
            },
            "hyperliquid_fast": {
                "source_message_count_by_channel": {"bbo": len(hyper_rows)},
            },
            "hyperliquid_standard_l2": {"raw_row_count": 1},
            "hyperliquid_auxiliary": {
                "tracks": {
                    "asset_context": {
                        "source_message_count_by_channel": {"activeAssetCtx": 1}
                    },
                    "main_all_mids": {
                        "source_message_count_by_channel": {"allMids": 1}
                    },
                    "target_dex_all_mids": {
                        "source_message_count_by_channel": {"allMids": 1}
                    },
                }
            },
        },
    }
    segment_manifest_path = segment_dir / "segment_event_store_manifest.json"
    _write_json(segment_manifest_path, segment_manifest)

    mask_path = root / "segment_and_mask_index.csv"
    mask_fields = [
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
    ]
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    with mask_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=mask_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(
            {
                "campaign_id": campaign_id,
                "segment_id": "segment_0001",
                "profile_id": profile_id,
                "first_common_ts_ns": 100 * ms,
                "last_common_ts_ns": 3000 * ms,
                "previous_segment_gap_ms": "",
                "cross_segment_continuity_claimed": "false",
                "mask_type": "segment_epoch",
                "policy": "never_compute_across_segment_boundary",
                "reason": "fresh_exchange_snapshots",
            }
        )
    source_manifest = {
        "passes": True,
        "campaign_id": campaign_id,
        "profile_id": profile_id,
        "segment_count": 1,
        "degraded_interval_count": 0,
        "degraded_intervals": [],
        "source_hash_count": 7,
        "source_hashes_unchanged": True,
        "aggregate_counts": {
            "binance_hot_rows": len(binance_rows),
            "hyperliquid_hot_rows": len(hyper_rows),
            "hyperliquid_auxiliary_rows": 0,
            "timeline_rows": len(timeline_rows),
        },
        "segment_and_mask_index": {
            "path": "segment_and_mask_index.csv",
            "row_count": 1,
            "sha256": alignment.sha256_file(mask_path),
        },
        "segments": [
            {
                "segment_id": "segment_0001",
                "manifest_path": (
                    "segments/segment_0001/segment_event_store_manifest.json"
                ),
                "manifest_sha256": alignment.sha256_file(segment_manifest_path),
                "outputs": outputs,
            }
        ],
        "boundary": {
            "new_collection_performed": False,
            "local_existing_data_only": True,
        },
    }
    _write_json(root / "research_input_manifest.json", source_manifest)
    return root


def test_quantile_price_distance_and_freshness_tiers() -> None:
    assert alignment._quantile([], 0.5) is None
    assert alignment._quantile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert alignment._price_distance_bps(99.0, 101.0) == pytest.approx(200.0)
    assert alignment._tier(250.0, "hyperliquid_bbo") == "primary"
    assert alignment._tier(250.1, "hyperliquid_bbo") == "watch"
    assert alignment._tier(500.1, "hyperliquid_bbo") == "reject"


def test_reconciliation_accepts_async_small_distance_and_rejects_extreme() -> None:
    acceptable = alignment._ReconciliationAccumulator(
        "hyperliquid_fast_vs_bbo"
    )
    for index in range(100):
        acceptable.observe(
            ts_ns=index * 1_000_000,
            exact_match=False,
            distance_bps=2.0,
            source_age_ms=100.0,
        )
    assert acceptable.finalize(100_000_000)["gate_pass"] == "true"

    extreme = alignment._ReconciliationAccumulator(
        "hyperliquid_fast_vs_bbo"
    )
    for index in range(100):
        extreme.observe(
            ts_ns=index * 1_000_000,
            exact_match=False,
            distance_bps=1000.0,
            source_age_ms=100.0,
        )
    assert extreme.finalize(100_000_000)["gate_pass"] == "false"


def test_full_build_freezes_labels_and_complete_warmup(tmp_path: Path) -> None:
    event_store = _event_store(tmp_path)
    output = tmp_path / "alignment"
    manifest = alignment.build_alignment_acceptance(
        event_store_dir=event_store,
        output_dir=output,
    )
    assert manifest["passes"] is True
    assert manifest["provenance_pass"] is True
    assert manifest["exact_masks_pass"] is True
    assert manifest["labels_pass"] is True
    assert manifest["reconciliation_pass"] is True
    runtime = manifest["runtime_source"]["builder"]
    runtime_archive = output / runtime["archive_path"]
    assert runtime_archive.is_file()
    assert runtime["archive_sha256"] == runtime["sha256"]
    quality = list(
        csv.DictReader((output / "alignment_quality_by_segment.csv").open())
    )
    assert quality[0]["decision_count"] == "3"
    assert quality[0]["eligible_decision_count"] == "2"
    assert quality[0]["warmup_excluded_count"] == "1"
    assert "missing_hyperliquid_bbo_asof" in quality[0][
        "warmup_reason_counts_json"
    ]
    with gzip.open(
        output / "decision_labels" / "segment_0001.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as fh:
        labels = list(csv.DictReader(fh))
    assert len(labels) == 3
    assert labels[0]["eligible"] == "false"
    assert labels[0]["h100_target_ts_ns"] == "150000000"
    assert labels[0]["h100_primary_source_ts_ns"] == "150000000"
    assert labels[1]["h10_primary_covered"] == "true"
    assert labels[1]["h10_primary_source_age_ms"] == "0.0"
    assert labels[1]["h10_next_source_ts_ns"] == "210000000"
    assert labels[1]["h10_wall_no_new_information"] == "false"
    assert labels[1]["h100_price_update_occurred_inside_horizon"] == "true"
    assert labels[1]["h100_wall_price_changed"] == "false"
    assert labels[1]["h2000_primary_source_ts_ns"] == "2200000000"

    reconciliation = list(
        csv.DictReader((output / "top_of_book_reconciliation.csv").open())
    )
    binance = next(
        row
        for row in reconciliation
        if row["comparison"] == "binance_depth_vs_book_ticker"
    )
    assert binance["missing_asof_count"] == "0"


@pytest.mark.parametrize("track_id", ["hyperliquid_fast", "binance"])
def test_core_reconnect_mask_excludes_intersecting_horizons_exactly(
    tmp_path: Path,
    track_id: str,
) -> None:
    event_store = _event_store(tmp_path)
    interval = {
        "segment_id": "segment_0001",
        "track_id": track_id,
        "reason": "websocket_reconnect",
        "degraded_start_local_ts_ns": 240_000_000,
        "recovered_local_ts_ns": 260_000_000,
        "duration_ms": 20.0,
        "policy": "split_replay_epoch_and_exclude_intersecting_horizons",
    }
    mask_path = event_store / "segment_and_mask_index.csv"
    with mask_path.open(newline="", encoding="utf-8") as fh:
        mask_rows = list(csv.DictReader(fh))
        mask_fields = list(mask_rows[0])
    mask_rows.append(
        {
            **mask_rows[0],
            "mask_type": "core_l2_reconnect_interval",
            "track_id": track_id,
            "mask_start_ts_ns": "240000000",
            "mask_end_ts_ns": "260000000",
            "duration_ms": "20.0",
            "policy": "split_replay_epoch_and_exclude_intersecting_horizons",
            "reason": (
                f"websocket_reconnect:segment_0001:{track_id}:1"
            ),
        }
    )
    with mask_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=mask_fields, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(mask_rows)

    hyper_path = (
        event_store
        / "segments"
        / "segment_0001"
        / "hyperliquid_hot_events.csv.gz"
    )
    with gzip.open(hyper_path, "rt", encoding="utf-8", newline="") as fh:
        hyper_rows = list(csv.DictReader(fh))
        hyper_fields = list(hyper_rows[0])
    hyper_fields.extend(
        field
        for field in (
            "connection_epoch_id",
            "degraded",
            "degraded_interval_ids",
        )
        if field not in hyper_fields
    )
    for row in hyper_rows:
        row["connection_epoch_id"] = (
            "1" if int(row["local_ts_ns"]) > 260_000_000 else "0"
        )
        row["degraded"] = "false"
        row["degraded_interval_ids"] = ""
    _write_gzip_csv(hyper_path, hyper_fields, hyper_rows)

    binance_path = (
        event_store
        / "segments"
        / "segment_0001"
        / "binance_hot_events.csv.gz"
    )
    with gzip.open(binance_path, "rt", encoding="utf-8", newline="") as fh:
        binance_rows = list(csv.DictReader(fh))
        binance_fields = list(binance_rows[0])
    decision_inside = next(
        row for row in binance_rows if row["local_ts_ns"] == "250000000"
    )
    decision_inside["bid_px"] = "100.5"
    decision_inside["ask_px"] = "101.5"
    if track_id == "binance":
        decision_inside["degraded"] = "true"
        decision_inside["degraded_interval_ids"] = (
            "segment_0001:binance:1"
        )
    _write_gzip_csv(binance_path, binance_fields, binance_rows)

    segment_path = (
        event_store
        / "segments"
        / "segment_0001"
        / "segment_event_store_manifest.json"
    )
    segment = json.loads(segment_path.read_text())
    segment["outputs"]["hyperliquid_hot_events"]["sha256"] = (
        alignment.sha256_file(hyper_path)
    )
    segment["outputs"]["binance_hot_events"]["sha256"] = (
        alignment.sha256_file(binance_path)
    )
    _write_json(segment_path, segment)
    source_path = event_store / "research_input_manifest.json"
    source = json.loads(source_path.read_text())
    source["degraded_interval_count"] = 1
    source["degraded_intervals"] = [interval]
    source["segment_and_mask_index"]["row_count"] = 2
    source["segment_and_mask_index"]["sha256"] = alignment.sha256_file(
        mask_path
    )
    source["segments"][0]["manifest_sha256"] = alignment.sha256_file(
        segment_path
    )
    source["segments"][0]["outputs"]["hyperliquid_hot_events"] = segment[
        "outputs"
    ]["hyperliquid_hot_events"]
    source["segments"][0]["outputs"]["binance_hot_events"] = segment[
        "outputs"
    ]["binance_hot_events"]
    _write_json(source_path, source)

    output = tmp_path / "alignment"
    manifest = alignment.build_alignment_acceptance(
        event_store_dir=event_store,
        output_dir=output,
    )

    assert manifest["exact_horizon_masks_pass"] is True
    assert manifest["cross_epoch_label_count"] == 0
    assert manifest["mask_counts"]["core_l2_reconnect_interval"] == 1
    with gzip.open(
        output / "decision_labels" / "segment_0001.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as fh:
        labels = list(csv.DictReader(fh))
    before = next(
        row for row in labels if row["decision_local_ts_ns"] == "200000000"
    )
    assert before["h25_mask_intersection"] == "false"
    assert before["h25_primary_covered"] == "true"
    assert before["h50_mask_intersection"] == "true"
    assert (
        before["h50_mask_interval_ids"]
        == f"segment_0001:{track_id}:1"
    )
    assert before["h50_primary_covered"] == ""
    inside = next(
        row for row in labels if row["decision_local_ts_ns"] == "250000000"
    )
    assert inside["eligible"] == "false"
    assert inside["warmup_reason"] == "core_l2_reconnect_interval"
    assert inside["binance_connection_epoch_id"] == "0"
    assert inside["binance_degraded"] == (
        "true" if track_id == "binance" else "false"
    )
    assert inside["binance_degraded_interval_ids"] == (
        "segment_0001:binance:1" if track_id == "binance" else ""
    )
    repeat = alignment.build_alignment_acceptance(
        event_store_dir=event_store,
        output_dir=tmp_path / "alignment-repeat",
    )
    assert (
        repeat["decision_label_outputs"]["segment_0001"]["sha256"]
        == manifest["decision_label_outputs"]["segment_0001"]["sha256"]
    )


def test_primary_label_uses_target_asof_state_when_next_event_is_late(
    tmp_path: Path,
) -> None:
    event_store = _event_store(tmp_path)
    hyper_path = (
        event_store
        / "segments"
        / "segment_0001"
        / "hyperliquid_hot_events.csv.gz"
    )
    with gzip.open(hyper_path, "rt", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0])
    rows = [
        row
        for row in rows
        if int(row["local_ts_ns"]) not in {210_000_000, 225_000_000}
    ]
    _write_gzip_csv(hyper_path, fields, rows)

    segment_path = (
        event_store
        / "segments"
        / "segment_0001"
        / "segment_event_store_manifest.json"
    )
    segment = json.loads(segment_path.read_text())
    segment["outputs"]["hyperliquid_hot_events"]["row_count"] = len(rows)
    segment["outputs"]["hyperliquid_hot_events"]["sha256"] = (
        alignment.sha256_file(hyper_path)
    )
    _write_json(segment_path, segment)
    source_path = event_store / "research_input_manifest.json"
    source = json.loads(source_path.read_text())
    source["aggregate_counts"]["hyperliquid_hot_rows"] = len(rows)
    source["segments"][0]["manifest_sha256"] = alignment.sha256_file(
        segment_path
    )
    source["segments"][0]["outputs"]["hyperliquid_hot_events"] = segment[
        "outputs"
    ]["hyperliquid_hot_events"]
    _write_json(source_path, source)

    output = tmp_path / "alignment"
    alignment.build_alignment_acceptance(
        event_store_dir=event_store,
        output_dir=output,
    )
    with gzip.open(
        output / "decision_labels" / "segment_0001.csv.gz",
        "rt",
        encoding="utf-8",
        newline="",
    ) as fh:
        labels = list(csv.DictReader(fh))
    decision = next(
        row for row in labels if row["decision_local_ts_ns"] == "200000000"
    )
    assert decision["h10_primary_source_ts_ns"] == "150000000"
    assert decision["h10_primary_effective_horizon_ms"] == "10.0"
    assert decision["h10_primary_source_age_ms"] == "60.0"
    assert decision["h10_primary_covered"] == "true"
    assert decision["h10_next_source_ts_ns"] == "250000000"
    assert decision["h10_next_inside_tolerance"] == "true"


def test_bad_provenance_and_bad_mask_fail_closed(tmp_path: Path) -> None:
    event_store = _event_store(tmp_path / "provenance")
    segment_path = (
        event_store
        / "segments"
        / "segment_0001"
        / "segment_event_store_manifest.json"
    )
    segment = json.loads(segment_path.read_text())
    segment["outputs"]["binance_hot_events"]["sha256"] = "0" * 64
    _write_json(segment_path, segment)
    source_path = event_store / "research_input_manifest.json"
    source = json.loads(source_path.read_text())
    source["segments"][0]["manifest_sha256"] = alignment.sha256_file(segment_path)
    _write_json(source_path, source)
    with pytest.raises(alignment.AlignmentError, match="R0 output mismatch"):
        alignment.build_alignment_acceptance(
            event_store_dir=event_store,
            output_dir=tmp_path / "bad-provenance-output",
        )

    event_store = _event_store(tmp_path / "mask")
    mask_path = event_store / "segment_and_mask_index.csv"
    rows = list(csv.DictReader(mask_path.open()))
    rows[0]["segment_id"] = "segment_9999"
    with mask_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    source_path = event_store / "research_input_manifest.json"
    source = json.loads(source_path.read_text())
    source["segment_and_mask_index"]["sha256"] = alignment.sha256_file(mask_path)
    _write_json(source_path, source)
    with pytest.raises(alignment.AlignmentError, match="unknown segment"):
        alignment.build_alignment_acceptance(
            event_store_dir=event_store,
            output_dir=tmp_path / "bad-mask-output",
        )


def test_mask_gap_accepts_equivalent_fixed_decimal_serialization(
    tmp_path: Path,
) -> None:
    event_store = _event_store(tmp_path)
    segment_path = (
        event_store
        / "segments"
        / "segment_0001"
        / "segment_event_store_manifest.json"
    )
    segment = json.loads(segment_path.read_text())
    segment["segment_boundary"]["previous_segment_gap_ms"] = 888.0954
    _write_json(segment_path, segment)

    mask_path = event_store / "segment_and_mask_index.csv"
    with mask_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0])
    rows[0]["previous_segment_gap_ms"] = "888.095400"
    with mask_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    source_path = event_store / "research_input_manifest.json"
    source = json.loads(source_path.read_text())
    source["segments"][0]["manifest_sha256"] = alignment.sha256_file(segment_path)
    source["segment_and_mask_index"]["sha256"] = alignment.sha256_file(mask_path)
    _write_json(source_path, source)

    manifest = alignment.build_alignment_acceptance(
        event_store_dir=event_store,
        output_dir=tmp_path / "alignment",
    )
    assert manifest["passes"] is True


def test_extreme_reconciliation_publishes_fail_not_pass(tmp_path: Path) -> None:
    event_store = _event_store(tmp_path, extreme_fast_price=True)
    manifest = alignment.build_alignment_acceptance(
        event_store_dir=event_store,
        output_dir=tmp_path / "alignment",
    )
    assert manifest["reconciliation_pass"] is False
    assert manifest["passes"] is False


def test_label_publication_and_r0_output_stability_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_store = _event_store(tmp_path / "labels")
    real_process = alignment._process_segment

    def truncate_label(**kwargs: object) -> dict:
        result = real_process(**kwargs)
        path = Path(str(kwargs["label_output_path"]))
        alignment_fields = alignment._label_fields()
        _write_gzip_csv(path, alignment_fields, [])
        return result

    monkeypatch.setattr(alignment, "_process_segment", truncate_label)
    with pytest.raises(alignment.AlignmentError, match="frozen label row mismatch"):
        alignment.build_alignment_acceptance(
            event_store_dir=event_store,
            output_dir=tmp_path / "truncated-output",
        )

    monkeypatch.setattr(alignment, "_process_segment", real_process)
    event_store = _event_store(tmp_path / "stability")

    def mutate_normalized_output(**kwargs: object) -> dict:
        result = real_process(**kwargs)
        path = (
            Path(str(kwargs["event_store_dir"]))
            / "segments"
            / "segment_0001"
            / "binance_hot_events.csv.gz"
        )
        with gzip.open(path, "at", encoding="utf-8") as fh:
            fh.write("mutated\n")
        return result

    monkeypatch.setattr(alignment, "_process_segment", mutate_normalized_output)
    with pytest.raises(alignment.AlignmentError, match="R0 input changed"):
        alignment.build_alignment_acceptance(
            event_store_dir=event_store,
            output_dir=tmp_path / "mutated-input-output",
        )


def test_degraded_mask_reason_requires_exact_identity(tmp_path: Path) -> None:
    event_store = _event_store(tmp_path)
    mask_path = event_store / "segment_and_mask_index.csv"
    with mask_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fields = list(rows[0])
    rows.append(
        {
            **rows[0],
            "mask_type": "auxiliary_degraded_interval",
            "track_id": "asset_context",
            "mask_start_ts_ns": "500000000",
            "mask_end_ts_ns": "600000000",
            "duration_ms": "100",
            "policy": "exclude_or_mask_auxiliary_features",
            "reason": (
                "websocket_reconnect:segment_0001:asset_context:1:forged"
            ),
        }
    )
    with mask_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    source_path = event_store / "research_input_manifest.json"
    source = json.loads(source_path.read_text())
    source["degraded_interval_count"] = 1
    source["degraded_intervals"] = [
        {
            "segment_id": "segment_0001",
            "track_id": "asset_context",
            "degraded_start_local_ts_ns": 500000000,
            "recovered_local_ts_ns": 600000000,
            "duration_ms": 100,
            "policy": "exclude_or_mask_auxiliary_features",
            "reason": "websocket_reconnect",
        }
    ]
    source["segment_and_mask_index"]["row_count"] = 2
    source["segment_and_mask_index"]["sha256"] = alignment.sha256_file(mask_path)
    _write_json(source_path, source)
    with pytest.raises(alignment.AlignmentError, match="missing exact degraded"):
        alignment.build_alignment_acceptance(
            event_store_dir=event_store,
            output_dir=tmp_path / "bad-reason-output",
        )


def test_failed_clean_build_preserves_old_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_store = _event_store(tmp_path)
    output = tmp_path / "alignment"
    output.mkdir()
    old = output / "accepted.txt"
    old.write_text("old", encoding="utf-8")

    def fail_process(**_kwargs: object) -> dict:
        raise alignment.AlignmentError("injected failure")

    monkeypatch.setattr(alignment, "_process_segment", fail_process)
    with pytest.raises(alignment.AlignmentError, match="injected failure"):
        alignment.build_alignment_acceptance(
            event_store_dir=event_store,
            output_dir=output,
            clean_output=True,
        )
    assert old.read_text(encoding="utf-8") == "old"
    assert not output.with_name(output.name + ".tmp").exists()
