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

import cross_exchange_research_dataset as dataset


def _write_raw(path: Path, rows: list[tuple[int, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for local_ts, payload in rows:
            fh.write(f"{local_ts} {json.dumps(payload, separators=(',', ':'))}\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _campaign(tmp_path: Path) -> Path:
    root = tmp_path / "campaign"
    profile_dir = root / "segments" / "segment_0001" / "skhynix"
    sample = profile_dir / "sample"
    binance_raw = sample / "binance_public_raw" / "raw.gz"
    fast_raw = sample / "hyperliquid_public_sample" / "raw.gz"
    standard_raw = (
        sample
        / "hyperliquid_public_sample"
        / "research_tracks"
        / "standard_l2"
        / "raw.gz"
    )
    asset_raw = (
        sample
        / "hyperliquid_public_sample"
        / "research_tracks"
        / "asset_context"
        / "raw.gz"
    )
    main_mids_raw = (
        sample
        / "hyperliquid_public_sample"
        / "research_tracks"
        / "main_all_mids"
        / "raw.gz"
    )
    target_mids_raw = (
        sample
        / "hyperliquid_public_sample"
        / "research_tracks"
        / "target_dex_all_mids"
        / "raw.gz"
    )
    _write_raw(
        binance_raw,
        [
            (
                100,
                {
                    "lastUpdateId": 1,
                    "bids": [["100", "1"]],
                    "asks": [["101", "1"]],
                },
            ),
            (
                200,
                {
                    "stream": "skhynixusdt@depth@0ms",
                    "data": {
                        "e": "depthUpdate",
                        "s": "SKHYNIXUSDT",
                        "T": 1,
                        "U": 1,
                        "u": 2,
                        "pu": 1,
                        "b": [],
                        "a": [],
                    },
                },
            ),
            (
                300,
                {
                    "stream": "skhynixusdt@bookTicker",
                    "data": {
                        "e": "bookTicker",
                        "s": "SKHYNIXUSDT",
                        "T": 2,
                        "u": 3,
                        "b": "100",
                        "B": "2",
                        "a": "101",
                        "A": "3",
                    },
                },
            ),
            (
                400,
                {
                    "stream": "skhynixusdt@trade",
                    "data": {
                        "e": "trade",
                        "s": "SKHYNIXUSDT",
                        "T": 3,
                        "p": "101",
                        "q": "0.5",
                        "t": 9,
                        "m": False,
                    },
                },
            ),
        ],
    )
    _write_raw(
        fast_raw,
        [
            (
                150,
                {
                    "channel": "subscriptionResponse",
                    "data": {"method": "subscribe"},
                },
            ),
            (
                250,
                {
                    "channel": "l2Book",
                    "data": {
                        "coin": "xyz:SKHX",
                        "time": 1,
                        "levels": [
                            [{"px": "100", "sz": "1", "n": 1}],
                            [{"px": "101", "sz": "1", "n": 1}],
                        ],
                    },
                },
            ),
            (
                350,
                {
                    "channel": "bbo",
                    "data": {
                        "coin": "xyz:SKHX",
                        "time": 2,
                        "bbo": [
                            {"px": "100", "sz": "2", "n": 1},
                            {"px": "101", "sz": "3", "n": 2},
                        ],
                    },
                },
            ),
            (
                450,
                {
                    "channel": "trades",
                    "data": [
                        {
                            "coin": "xyz:SKHX",
                            "time": 3,
                            "px": "101",
                            "sz": "0.2",
                            "side": "B",
                            "tid": 10,
                            "hash": "0x1",
                            "users": ["a", "b"],
                        },
                        {
                            "coin": "xyz:SKHX",
                            "time": 4,
                            "px": "100",
                            "sz": "0.3",
                            "side": "A",
                            "tid": 11,
                            "hash": "0x2",
                            "users": ["c", "d"],
                        },
                    ],
                },
            ),
        ],
    )
    _write_raw(
        standard_raw,
        [
            (
                160,
                {
                    "channel": "subscriptionResponse",
                    "data": {"method": "subscribe"},
                },
            ),
            (
                260,
                {
                    "channel": "l2Book",
                    "data": {
                        "coin": "xyz:SKHX",
                        "time": 1,
                        "levels": [
                            [{"px": "100", "sz": "2", "n": 1}],
                            [{"px": "101", "sz": "2", "n": 1}],
                        ],
                    },
                },
            ),
        ],
    )
    _write_raw(
        asset_raw,
        [
            (
                170,
                {"channel": "subscriptionResponse", "data": {"method": "subscribe"}},
            ),
            (200, {"channel": "pong"}),
            (
                300,
                {
                    "channel": "activeAssetCtx",
                    "data": {
                        "coin": "xyz:SKHX",
                        "ctx": {
                            "markPx": "100.5",
                            "oraclePx": "100.4",
                            "midPx": "100.5",
                            "funding": "0.001",
                            "premium": "0.002",
                            "openInterest": "5",
                            "impactPxs": ["100.3", "100.7"],
                        },
                    },
                },
            ),
            (
                500,
                {
                    "channel": "candle",
                    "data": {
                        "s": "xyz:SKHX",
                        "t": 1,
                        "T": 2,
                        "o": "100",
                        "h": "102",
                        "l": "99",
                        "c": "101",
                        "v": "5",
                        "n": 3,
                    },
                },
            ),
        ],
    )
    _write_raw(
        main_mids_raw,
        [
            (
                180,
                {"channel": "subscriptionResponse", "data": {"method": "subscribe"}},
            ),
            (320, {"channel": "allMids", "data": {"mids": {"BTC": "1"}}}),
        ],
    )
    _write_raw(
        target_mids_raw,
        [
            (
                190,
                {"channel": "subscriptionResponse", "data": {"method": "subscribe"}},
            ),
            (
                330,
                {
                    "channel": "allMids",
                    "data": {"dex": "xyz", "mids": {"xyz:SKHX": "100.6"}},
                },
            ),
        ],
    )

    timeline = profile_dir / "common_l2_timeline.csv.gz"
    timeline.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(timeline, "wt", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["campaign_id", "segment_id", "profile_id", "common_ts_ns"])
        writer.writerow(["campaign", "segment_0001", "skhynix", 260])
        writer.writerow(["campaign", "segment_0001", "skhynix", 300])
    timeline_sha = dataset.sha256_file(timeline)
    _write_json(
        profile_dir / "common_l2_timeline_manifest.json",
        {
            "passes": True,
            "campaign_id": "campaign",
            "segment_id": "segment_0001",
            "profile_id": "skhynix",
            "binance_symbol": "SKHYNIXUSDT",
            "hyperliquid_coin": "xyz:SKHX",
            "timeline_sha256": timeline_sha,
            "timeline_row_count": 2,
            "first_common_ts_ns": 260,
            "last_common_ts_ns": 300,
            "connection_epoch_count_by_track": {
                "binance": 1,
                "hyperliquid_fast": 1,
                "hyperliquid_standard": 1,
            },
            "segment_boundary": {
                "cross_segment_continuity_claimed": False,
                "fresh_snapshots": True,
            },
            "source_raw": {
                "binance": {"sha256": dataset.sha256_file(binance_raw)},
                "hyperliquid_fast": {"sha256": dataset.sha256_file(fast_raw)},
                "hyperliquid_standard": {
                    "sha256": dataset.sha256_file(standard_raw)
                },
            },
        },
    )
    _write_json(profile_dir / "strict_quality.json", {"passes": True})
    _write_json(
        sample / "binance_public_raw" / "collection_manifest.json",
        {
            "raw_sha256": dataset.sha256_file(binance_raw),
            "depth_snapshot_bridge_count": 1,
            "message_count_by_event_type": {
                "depthUpdate": 1,
                "bookTicker": 1,
                "trade": 1,
            },
        },
    )
    _write_json(
        sample / "hyperliquid_public_sample" / "collection_manifest.json",
        {
            "raw_sha256": dataset.sha256_file(fast_raw),
            "raw_row_count": 4,
            "raw_row_count_reconciled": True,
            "parse_error_count": 0,
            "message_count_by_channel": {
                "subscriptionResponse": 1,
                "l2Book": 1,
                "bbo": 1,
                "trades": 1,
            },
            "control_message_count_by_channel": {},
        },
    )

    track_specs = {
        "fast_market": (
            ".",
            fast_raw,
            {"subscriptionResponse": 1, "l2Book": 1, "bbo": 1, "trades": 1},
            {},
        ),
        "standard_l2": (
            "research_tracks/standard_l2",
            standard_raw,
            {"subscriptionResponse": 1, "l2Book": 1},
            {},
        ),
        "asset_context": (
            "research_tracks/asset_context",
            asset_raw,
            {"subscriptionResponse": 1, "activeAssetCtx": 1, "candle": 1},
            {"pong": 1},
        ),
        "main_all_mids": (
            "research_tracks/main_all_mids",
            main_mids_raw,
            {"subscriptionResponse": 1, "allMids": 1},
            {},
        ),
        "target_dex_all_mids": (
            "research_tracks/target_dex_all_mids",
            target_mids_raw,
            {"subscriptionResponse": 1, "allMids": 1},
            {},
        ),
    }
    tracks = {
        track_id: {
            "relative_output_dir": relative,
            "raw_sha256": dataset.sha256_file(raw),
            "message_count_by_channel": counts,
        }
        for track_id, (relative, raw, counts, _control_counts) in track_specs.items()
    }
    for track_id, (relative, raw, counts, control_counts) in track_specs.items():
        _write_json(
            raw.parent / "collection_manifest.json",
            {
                "raw_sha256": dataset.sha256_file(raw),
                "raw_row_count": sum(counts.values()) + sum(control_counts.values()),
                "raw_row_count_reconciled": True,
                "parse_error_count": 0,
                "message_count_by_channel": counts,
                "control_message_count_by_channel": control_counts,
            },
        )
    _write_json(
        sample / "hyperliquid_public_sample" / "research_bundle_manifest.json",
        {
            "quality": {"all_tracks_pass": True},
            "tracks": tracks,
        },
    )
    _write_json(
        root / "campaign_manifest.json",
        {
            "passes": True,
            "campaign_id": "campaign",
            "profiles": ["skhynix"],
            "segments": [{"segment_id": "segment_0001"}],
            "cross_segment_continuity_claimed": False,
            "degraded_intervals": [
                {
                    "segment_id": "segment_0001",
                    "track_id": "asset_context",
                    "degraded_start_local_ts_ns": 250,
                    "recovered_local_ts_ns": 350,
                    "duration_ms": 0.0001,
                    "policy": "exclude_or_mask_auxiliary_features",
                    "reason": "websocket_reconnect",
                }
            ],
        },
    )
    with (root / "timeline_index.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "campaign_id",
                "segment_id",
                "profile_id",
                "timeline_path",
                "timeline_sha256",
                "row_count",
                "first_common_ts_ns",
                "last_common_ts_ns",
                "previous_segment_gap_ms",
                "cross_segment_continuity_claimed",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "campaign_id": "campaign",
                "segment_id": "segment_0001",
                "profile_id": "skhynix",
                "timeline_path": str(timeline),
                "timeline_sha256": timeline_sha,
                "row_count": 2,
                "first_common_ts_ns": 260,
                "last_common_ts_ns": 300,
                "previous_segment_gap_ms": "",
                "cross_segment_continuity_claimed": "False",
            }
        )
    return root


def _read_gzip_csv(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _refresh_source_sha(campaign: Path, source_id: str) -> Path:
    profile = campaign / "segments" / "segment_0001" / "skhynix"
    sample = profile / "sample"
    source_specs = {
        "binance": (
            sample / "binance_public_raw" / "raw.gz",
            "binance",
            None,
        ),
        "fast_market": (
            sample / "hyperliquid_public_sample" / "raw.gz",
            "hyperliquid_fast",
            "fast_market",
        ),
        "standard_l2": (
            sample
            / "hyperliquid_public_sample"
            / "research_tracks"
            / "standard_l2"
            / "raw.gz",
            "hyperliquid_standard",
            "standard_l2",
        ),
    }
    raw, timeline_source_id, bundle_track_id = source_specs[source_id]
    digest = dataset.sha256_file(raw)
    collection_path = raw.parent / "collection_manifest.json"
    collection = json.loads(collection_path.read_text())
    collection["raw_sha256"] = digest
    _write_json(collection_path, collection)
    if bundle_track_id:
        bundle_path = (
            sample
            / "hyperliquid_public_sample"
            / "research_bundle_manifest.json"
        )
        bundle = json.loads(bundle_path.read_text())
        bundle["tracks"][bundle_track_id]["raw_sha256"] = digest
        _write_json(bundle_path, bundle)
    timeline_path = profile / "common_l2_timeline_manifest.json"
    timeline = json.loads(timeline_path.read_text())
    timeline["source_raw"][timeline_source_id]["sha256"] = digest
    _write_json(timeline_path, timeline)
    return raw


def _rewrite_timeline_index(campaign: Path, **updates: object) -> None:
    path = campaign / "timeline_index.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0])
    rows[0].update({key: str(value) for key, value in updates.items()})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_timeline_row_identity(
    campaign: Path,
    *,
    field: str,
    value: str,
) -> None:
    profile = campaign / "segments" / "segment_0001" / "skhynix"
    timeline_path = profile / "common_l2_timeline.csv.gz"
    with gzip.open(timeline_path, "rt", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0])
    rows[0][field] = value
    with gzip.open(timeline_path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    digest = dataset.sha256_file(timeline_path)
    manifest_path = profile / "common_l2_timeline_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["timeline_sha256"] = digest
    _write_json(manifest_path, manifest)
    _rewrite_timeline_index(campaign, timeline_sha256=digest)


def test_build_research_dataset_reconciles_events_and_masks(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    manifest = dataset.build_research_dataset(
        campaign_dir=campaign,
        output_dir=output,
        profile_id="skhynix",
    )

    assert manifest["passes"] is True
    assert manifest["aggregate_counts"]["timeline_rows"] == 2
    assert manifest["aggregate_counts"]["binance_hot_rows"] == 2
    assert manifest["aggregate_counts"]["hyperliquid_hot_rows"] == 3
    assert manifest["aggregate_counts"]["hyperliquid_trade_items"] == 2
    assert manifest["aggregate_counts"]["hyperliquid_auxiliary_rows"] == 4
    assert manifest["aggregate_counts"]["mask_rows"] == 2
    assert manifest["source_hashes_unchanged"] is True
    assert manifest["boundary"]["new_collection_performed"] is False
    runtime = manifest["runtime_source"]["builder"]
    runtime_archive = output / runtime["archive_path"]
    assert runtime_archive.is_file()
    assert runtime["archive_sha256"] == runtime["sha256"]

    segment = output / "segments" / "segment_0001"
    binance = _read_gzip_csv(segment / "binance_hot_events.csv.gz")
    assert [row["event_type"] for row in binance] == ["bookTicker", "trade"]
    assert binance[1]["aggressor_side"] == "buy"

    hyperliquid = _read_gzip_csv(segment / "hyperliquid_hot_events.csv.gz")
    assert [row["event_type"] for row in hyperliquid] == ["bbo", "trade", "trade"]
    assert hyperliquid[-1]["trade_side"] == "A"

    auxiliary = _read_gzip_csv(segment / "hyperliquid_auxiliary_events.csv.gz")
    auxiliary_timestamps = [int(row["local_ts_ns"]) for row in auxiliary]
    assert auxiliary_timestamps == sorted(auxiliary_timestamps)
    asset_ctx = next(row for row in auxiliary if row["event_type"] == "activeAssetCtx")
    assert asset_ctx["degraded"] == "true"
    assert asset_ctx["degraded_interval_id"] == "segment_0001:asset_context:1"
    target_mid = next(
        row
        for row in auxiliary
        if row["track_id"] == "target_dex_all_mids" and row["event_type"] == "allMids"
    )
    assert target_mid["target_mid_px"] == "100.6"


@pytest.mark.parametrize(
    ("track_id", "source_track_id", "hot_output"),
    [
        (
            "hyperliquid_fast",
            "fast_market",
            "hyperliquid_hot_events.csv.gz",
        ),
        ("binance", "binance", "binance_hot_events.csv.gz"),
    ],
)
def test_core_reconnect_publishes_epoch_aware_events_and_exact_mask(
    tmp_path: Path,
    track_id: str,
    source_track_id: str,
    hot_output: str,
) -> None:
    campaign = _campaign(tmp_path)
    interval = {
        "segment_id": "segment_0001",
        "track_id": track_id,
        "source_track_id": source_track_id,
        "track_class": "core",
        "reason": "websocket_reconnect",
        "connection_attempt": 1,
        "disconnect_local_ts_ns": 240,
        "degraded_start_local_ts_ns": 240,
        "recovered_local_ts_ns": 260,
        "duration_ms": 0.00002,
        "policy": "split_replay_epoch_and_exclude_intersecting_horizons",
    }
    profile = campaign / "segments" / "segment_0001" / "skhynix"
    strict_path = profile / "strict_quality.json"
    _write_json(
        strict_path,
        {
            "passes": True,
            "continuous_exact_replay": False,
            "segmented_replay_eligible": True,
            "degraded_intervals": [interval],
        },
    )
    timeline_manifest_path = profile / "common_l2_timeline_manifest.json"
    timeline_manifest = json.loads(timeline_manifest_path.read_text())
    timeline_manifest["reconnect_intervals"] = [interval]
    timeline_manifest["connection_epoch_count_by_track"] = {
        "binance": 2 if track_id == "binance" else 1,
        "hyperliquid_fast": 2 if track_id == "hyperliquid_fast" else 1,
        "hyperliquid_standard": 1,
    }
    timeline_manifest["capability_boundary"] = {
        "continuous_exact_replay": False,
        "segmented_replay_eligible": True,
        "old_l2_state_forward_filled_across_reconnect": False,
    }
    _write_json(timeline_manifest_path, timeline_manifest)
    campaign_manifest_path = campaign / "campaign_manifest.json"
    campaign_manifest = json.loads(campaign_manifest_path.read_text())
    campaign_manifest["degraded_intervals"].append(interval)
    _write_json(campaign_manifest_path, campaign_manifest)

    output = tmp_path / "output"
    manifest = dataset.build_research_dataset(
        campaign_dir=campaign,
        output_dir=output,
        profile_id="skhynix",
    )

    assert manifest["passes"] is True
    assert manifest["aggregate_counts"]["mask_rows"] == 3
    masks = list(
        csv.DictReader((output / "segment_and_mask_index.csv").open())
    )
    core_mask = next(
        row for row in masks if row["mask_type"] == "core_l2_reconnect_interval"
    )
    assert core_mask["track_id"] == track_id
    assert core_mask["mask_start_ts_ns"] == "240"
    assert core_mask["mask_end_ts_ns"] == "260"
    assert (
        core_mask["policy"]
        == "split_replay_epoch_and_exclude_intersecting_horizons"
    )
    hot_rows = _read_gzip_csv(
        output / "segments" / "segment_0001" / hot_output
    )
    assert {row["connection_epoch_id"] for row in hot_rows} == {"1"}
    segment_manifest = json.loads(
        (
            output
            / "segments"
            / "segment_0001"
            / "segment_event_store_manifest.json"
        ).read_text()
    )
    assert segment_manifest["connection_epochs"][track_id] == 2
    assert (
        segment_manifest["capability_boundary"]["continuous_exact_replay"]
        is False
    )
    repeat = dataset.build_research_dataset(
        campaign_dir=campaign,
        output_dir=tmp_path / "output-repeat",
        profile_id="skhynix",
    )
    assert {
        key: value["sha256"]
        for key, value in manifest["segments"][0]["outputs"].items()
    } == {
        key: value["sha256"]
        for key, value in repeat["segments"][0]["outputs"].items()
    }


def test_connection_epoch_changes_at_disconnect_timestamp() -> None:
    assert dataset._connection_epoch_id(
        intervals=[
            {
                "segment_id": "segment_0001",
                "track_id": "binance",
                "reason": "websocket_reconnect",
                "disconnect_local_ts_ns": 300,
            }
        ],
        segment_id="segment_0001",
        track_id="binance",
        local_ts_ns=300,
    ) == 1


def test_source_hash_mismatch_fails_without_publishing_output(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    manifest_path = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "sample"
        / "binance_public_raw"
        / "collection_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    output = tmp_path / "output"
    with pytest.raises(dataset.ResearchDatasetError, match="SHA mismatch"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=output,
            profile_id="skhynix",
        )
    assert not output.exists()


def test_count_mismatch_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    manifest_path = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "sample"
        / "hyperliquid_public_sample"
        / "collection_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["message_count_by_channel"]["bbo"] = 2
    _write_json(manifest_path, manifest)
    bundle_path = manifest_path.parent / "research_bundle_manifest.json"
    bundle = json.loads(bundle_path.read_text())
    bundle["tracks"]["fast_market"]["message_count_by_channel"]["bbo"] = 2
    _write_json(bundle_path, bundle)
    with pytest.raises(dataset.ResearchDatasetError, match="channel counts mismatch"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "output",
            profile_id="skhynix",
        )


def test_wrong_symbol_and_timestamp_regression_fail_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    raw = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "sample"
        / "binance_public_raw"
        / "raw.gz"
    )
    rows = []
    with gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            local_ts, payload = line.split(" ", 1)
            rows.append((int(local_ts), json.loads(payload)))
    rows[2][1]["data"]["s"] = "BTCUSDT"
    _write_raw(raw, rows)
    _refresh_source_sha(campaign, "binance")
    with pytest.raises(dataset.ResearchDatasetError, match="expected symbol"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "output",
            profile_id="skhynix",
        )

    campaign = _campaign(tmp_path / "regression")
    raw = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "sample"
        / "binance_public_raw"
        / "raw.gz"
    )
    rows = []
    with gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            local_ts, payload = line.split(" ", 1)
            rows.append((int(local_ts), json.loads(payload)))
    rows[-1] = (250, rows[-1][1])
    _write_raw(raw, rows)
    _refresh_source_sha(campaign, "binance")
    with pytest.raises(dataset.ResearchDatasetError, match="timestamp regressed"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "regression-output",
            profile_id="skhynix",
        )


def test_nonempty_output_requires_explicit_clean(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(dataset.ResearchDatasetError, match="nonempty"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=output,
            profile_id="skhynix",
        )
    manifest = dataset.build_research_dataset(
        campaign_dir=campaign,
        output_dir=output,
        profile_id="skhynix",
        clean_output=True,
    )
    assert manifest["passes"] is True
    assert not (output / "stale.txt").exists()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"campaign_id": "wrong"}, "index campaign mismatch"),
        ({"cross_segment_continuity_claimed": "True"}, "index claims continuity"),
        ({"first_common_ts_ns": 261}, "CSV/index boundary mismatch"),
        ({"last_common_ts_ns": 301}, "CSV/index boundary mismatch"),
    ],
)
def test_timeline_index_boundary_mismatches_fail_closed(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    campaign = _campaign(tmp_path)
    _rewrite_timeline_index(campaign, **updates)
    with pytest.raises(dataset.ResearchDatasetError, match=message):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "output",
            profile_id="skhynix",
        )


def test_binance_depth_symbol_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    raw = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "sample"
        / "binance_public_raw"
        / "raw.gz"
    )
    rows = []
    with gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            local_ts, payload = line.split(" ", 1)
            rows.append((int(local_ts), json.loads(payload)))
    rows[1][1]["data"]["s"] = "BTCUSDT"
    _write_raw(raw, rows)
    _refresh_source_sha(campaign, "binance")
    with pytest.raises(dataset.ResearchDatasetError, match="expected symbol"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "output",
            profile_id="skhynix",
        )


def test_standard_l2_coin_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    raw = _refresh_source_sha(campaign, "standard_l2")
    rows = []
    with gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            local_ts, payload = line.split(" ", 1)
            rows.append((int(local_ts), json.loads(payload)))
    rows[1][1]["data"]["coin"] = "BTC"
    _write_raw(raw, rows)
    _refresh_source_sha(campaign, "standard_l2")
    with pytest.raises(dataset.ResearchDatasetError, match="standard-L2 coin"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "output",
            profile_id="skhynix",
        )


def test_hyperliquid_extra_channel_and_raw_row_mismatch_fail_closed(
    tmp_path: Path,
) -> None:
    campaign = _campaign(tmp_path)
    raw = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "sample"
        / "hyperliquid_public_sample"
        / "raw.gz"
    )
    rows = []
    with gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            local_ts, payload = line.split(" ", 1)
            rows.append((int(local_ts), json.loads(payload)))
    rows.append((500, {"channel": "mystery", "data": {}}))
    _write_raw(raw, rows)
    _refresh_source_sha(campaign, "fast_market")
    manifest_path = raw.parent / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_row_count"] = 5
    _write_json(manifest_path, manifest)
    with pytest.raises(dataset.ResearchDatasetError, match="channel counts mismatch"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "extra-output",
            profile_id="skhynix",
        )

    campaign = _campaign(tmp_path / "row-count")
    manifest_path = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "sample"
        / "hyperliquid_public_sample"
        / "collection_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["raw_row_count"] = 5
    _write_json(manifest_path, manifest)
    with pytest.raises(dataset.ResearchDatasetError, match="raw row count mismatch"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "row-count-output",
            profile_id="skhynix",
        )


def test_timeline_source_raw_mismatch_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    manifest_path = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "common_l2_timeline_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["source_raw"]["hyperliquid_standard"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    with pytest.raises(dataset.ResearchDatasetError, match="source-raw SHA mismatch"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "output",
            profile_id="skhynix",
        )


def test_failed_clean_rebuild_preserves_previous_output(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    output = tmp_path / "output"
    dataset.build_research_dataset(
        campaign_dir=campaign,
        output_dir=output,
        profile_id="skhynix",
    )
    old_manifest = (output / "research_input_manifest.json").read_bytes()

    raw = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "sample"
        / "binance_public_raw"
        / "raw.gz"
    )
    rows = []
    with gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            local_ts, payload = line.split(" ", 1)
            rows.append((int(local_ts), json.loads(payload)))
    rows[1][1]["data"]["s"] = "BTCUSDT"
    _write_raw(raw, rows)
    _refresh_source_sha(campaign, "binance")
    with pytest.raises(dataset.ResearchDatasetError, match="expected symbol"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=output,
            profile_id="skhynix",
            clean_output=True,
        )
    assert (output / "research_input_manifest.json").read_bytes() == old_manifest
    assert not output.with_name(output.name + ".tmp").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("campaign_id", "wrong-campaign"),
        ("segment_id", "segment_9999"),
        ("profile_id", "btc"),
    ],
)
def test_timeline_csv_row_identity_fails_closed(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    campaign = _campaign(tmp_path)
    _rewrite_timeline_row_identity(campaign, field=field, value=value)
    with pytest.raises(dataset.ResearchDatasetError, match="row identity mismatch"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "output",
            profile_id="skhynix",
        )


def test_main_all_mids_named_dex_fails_closed(tmp_path: Path) -> None:
    campaign = _campaign(tmp_path)
    raw = (
        campaign
        / "segments"
        / "segment_0001"
        / "skhynix"
        / "sample"
        / "hyperliquid_public_sample"
        / "research_tracks"
        / "main_all_mids"
        / "raw.gz"
    )
    rows = []
    with gzip.open(raw, "rt", encoding="utf-8") as fh:
        for line in fh:
            local_ts, payload = line.split(" ", 1)
            rows.append((int(local_ts), json.loads(payload)))
    rows[1][1]["data"]["dex"] = "xyz"
    _write_raw(raw, rows)
    digest = dataset.sha256_file(raw)
    collection_path = raw.parent / "collection_manifest.json"
    collection = json.loads(collection_path.read_text())
    collection["raw_sha256"] = digest
    _write_json(collection_path, collection)
    bundle_path = (
        raw.parents[2] / "research_bundle_manifest.json"
    )
    bundle = json.loads(bundle_path.read_text())
    bundle["tracks"]["main_all_mids"]["raw_sha256"] = digest
    _write_json(bundle_path, bundle)
    with pytest.raises(dataset.ResearchDatasetError, match="must not carry named dex"):
        dataset.build_research_dataset(
            campaign_dir=campaign,
            output_dir=tmp_path / "output",
            profile_id="skhynix",
        )
