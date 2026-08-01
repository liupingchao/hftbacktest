from __future__ import annotations

import gzip
import json
import os
import csv
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_collection_supervisor as supervisor
from cross_exchange_l2_timeline import sha256_file


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_gzip_raw(path: Path, events: list[tuple[int, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for local_ts_ns, payload in events:
            fh.write(
                f"{local_ts_ns} "
                f"{json.dumps(payload, sort_keys=True, separators=(',', ':'))}\n"
            )


def _configure_auxiliary_reconnect(
    sample_dir: Path,
    *,
    track_id: str = "asset_context",
    include_recovery_ack: bool = True,
    include_resumed_data: bool = True,
    resumed_ts_ns: int = 31_000_000_000,
) -> None:
    if track_id == "asset_context":
        relative = "research_tracks/asset_context"
        channel = "activeAssetCtx"
        subscription = {"coin": "BTC", "type": "activeAssetCtx"}
    elif track_id == "main_all_mids":
        relative = "research_tracks/main_all_mids"
        channel = "allMids"
        subscription = {"type": "allMids"}
    else:
        raise ValueError(track_id)
    raw_path = sample_dir / "hyperliquid_public_sample" / relative / "raw.gz"
    ack = {
        "channel": "subscriptionResponse",
        "data": {"method": "subscribe", "subscription": subscription},
    }
    data = {"channel": channel, "data": {"test": True}}
    events = [
        (1_000_000_000, ack),
        (2_000_000_000, data),
        (29_000_000_000, data),
    ]
    if include_recovery_ack:
        events.append((30_200_000_000, ack))
    if include_resumed_data:
        events.append((resumed_ts_ns, data))
        events.append((59_000_000_000, data))
    events.sort(key=lambda item: item[0])
    _write_gzip_raw(raw_path, events)

    manifest_path = raw_path.parent / "collection_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = json.dumps(subscription, sort_keys=True, separators=(",", ":"))
    manifest.update(
        {
            "local_start_ts": 100,
            "local_end_ts": 60_000_000_100,
            "connection_attempt_count": 2,
            "reconnect_count": 1,
            "disconnect_events": [
                {
                    "connection_attempt": 1,
                    "attempt_started_ns": 100,
                    "disconnect_local_ts": 30_000_000_000,
                    "reason": "connection reset",
                }
            ],
            "expected_subscription_identities": [identity],
            "all_required_subscription_acks_received": include_recovery_ack,
            "raw_row_count": len(events),
            "raw_row_count_reconciled": True,
            "parse_error_count": 0,
            "first_local_ts_by_channel": {channel: 2_000_000_000},
            "last_local_ts_by_channel": {
                channel: 59_000_000_000 if include_resumed_data else 29_000_000_000
            },
            "message_count_by_channel": {
                channel: sum(1 for _, payload in events if payload["channel"] == channel),
                "subscriptionResponse": sum(
                    1 for _, payload in events if payload["channel"] == "subscriptionResponse"
                ),
            },
            "arrival_gap_ms_by_channel": {
                channel: {
                    "count": 3 if include_resumed_data else 1,
                    "max": (
                        (resumed_ts_ns - 29_000_000_000) / 1_000_000.0
                        if include_resumed_data
                        else 27_000.0
                    ),
                }
            },
            "close_reason": "duration_elapsed",
        }
    )
    _write_json(manifest_path, manifest)

    bundle_path = sample_dir / "hyperliquid_public_sample" / "research_bundle_manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["tracks"][track_id]["reconnect_count"] = 1
    bundle["tracks"][track_id]["raw_sha256"] = sha256_file(raw_path)
    _write_json(bundle_path, bundle)


def _valid_sample(tmp_path: Path) -> Path:
    sample_dir = tmp_path / "sample"
    binance_raw = sample_dir / "binance_public_raw" / "raw.gz"
    fast_raw = sample_dir / "hyperliquid_public_sample" / "raw.gz"
    binance_raw.parent.mkdir(parents=True)
    fast_raw.parent.mkdir(parents=True)
    binance_raw.write_bytes(b"binance")
    fast_raw.write_bytes(b"fast")
    tracks = {}
    track_channels = {
        "fast_market": ("l2Book", "bbo", "trades"),
        "standard_l2": ("l2Book",),
        "asset_context": ("activeAssetCtx", "candle"),
        "main_all_mids": ("allMids",),
        "target_dex_all_mids": ("allMids",),
    }
    for track_id, relative in (
        ("fast_market", "."),
        ("standard_l2", "research_tracks/standard_l2"),
        ("asset_context", "research_tracks/asset_context"),
        ("main_all_mids", "research_tracks/main_all_mids"),
        ("target_dex_all_mids", "research_tracks/target_dex_all_mids"),
    ):
        raw_path = (sample_dir / "hyperliquid_public_sample" / relative / "raw.gz").resolve()
        if relative != ".":
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(track_id.encode())
        tracks[track_id] = {
            "relative_output_dir": relative,
            "raw_sha256": sha256_file(raw_path),
            "reconnect_count": 0,
            "quality": {"passes": True},
        }
        channels = track_channels[track_id]
        _write_json(
            raw_path.parent / "collection_manifest.json",
            {
                "local_start_ts": 100,
                "local_end_ts": 60_000_000_100,
                "first_local_ts_by_channel": {channel: 1_000_000_000 for channel in channels},
                "last_local_ts_by_channel": {channel: 59_000_000_000 for channel in channels},
                "message_count_by_channel": {channel: 10 for channel in channels},
                "arrival_gap_ms_by_channel": {
                    channel: {"count": 9, "max": 1_000.0}
                    for channel in channels
                },
                "reconnect_count": 0,
            },
        )
    _write_json(
        sample_dir / "binance_public_raw" / "collection_manifest.json",
        {
            "actual_duration_seconds": 60.1,
            "close_reason": "duration_elapsed",
            "depth_replay_ready": True,
            "depth_snapshot_bridge_valid": True,
            "depth_continuity_gap_count": 0,
            "reconnect_count": 0,
            "message_count_by_event_type": {"depthUpdate": 10, "trade": 1, "bookTicker": 10},
            "first_local_ts_by_event_type": {
                "depthUpdate": 1_000_000_000,
                "trade": 1_000_000_000,
                "bookTicker": 1_000_000_000,
            },
            "last_local_ts_by_event_type": {
                "depthUpdate": 59_000_000_000,
                "trade": 59_000_000_000,
                "bookTicker": 59_000_000_000,
            },
            "arrival_gap_ms_by_event_type": {
                "depthUpdate": {"count": 9, "max": 1_000.0},
                "bookTicker": {"count": 9, "max": 1_000.0},
            },
            "raw_sha256": sha256_file(binance_raw),
            "local_start_ts": 100,
            "local_end_ts": 60_000_000_100,
        },
    )
    _write_json(
        sample_dir / "hyperliquid_public_sample" / "collection_manifest.json",
        {
            "actual_duration_seconds": 60.1,
            "reconnect_count": 0,
            "message_count_by_channel": {"l2Book": 10, "trades": 1, "bbo": 10},
            "first_local_ts_by_channel": {
                "l2Book": 1_000_000_000,
                "trades": 1_000_000_000,
                "bbo": 1_000_000_000,
            },
            "last_local_ts_by_channel": {
                "l2Book": 59_000_000_000,
                "trades": 59_000_000_000,
                "bbo": 59_000_000_000,
            },
            "arrival_gap_ms_by_channel": {
                "l2Book": {"count": 9, "max": 1_000.0},
                "bbo": {"count": 9, "max": 1_000.0},
                "trades": {"count": 0, "max": 0.0},
            },
            "raw_sha256": sha256_file(fast_raw),
            "local_start_ts": 200,
            "local_end_ts": 60_000_000_200,
            "research_bundle": {
                "enabled": True,
                "all_tracks_pass": True,
            },
        },
    )
    _write_json(
        sample_dir / "hyperliquid_public_sample" / "research_bundle_manifest.json",
        {
            "quality": {
                "all_tracks_pass": True,
                "dual_l2_information_quality": {
                    "fast_is_shallow_top5": True,
                    "standard_is_deeper_than_fast": True,
                },
            },
            "track_overlap": {"overlap_seconds": 60.0},
            "tracks": tracks,
        },
    )
    return sample_dir


def _args(tmp_path: Path):
    return supervisor.parse_args(
        [
            "--output-dir",
            str(tmp_path / "campaign"),
            "--campaign-id",
            "test-campaign",
            "--profiles",
            "btc",
            "--total-duration-seconds",
            "1",
            "--segment-duration-seconds",
            "1",
            "--poll-interval-seconds",
            "0.01",
            "--termination-grace-seconds",
            "0.05",
            "--heartbeat-interval-seconds",
            "0.01",
        ]
    )


def _prepare_existing_campaign(args) -> Path:
    campaign_root = Path(args.output_dir)
    segment_dir = campaign_root / "segments" / "segment_0001"
    sample_dir = _valid_sample(segment_dir / "btc")
    _write_json(
        sample_dir / "sample_manifest.json",
        {
            "task_id": "0729T010",
            "requested_duration_seconds": 1.0,
        },
    )
    _write_json(
        segment_dir / "segment_child_results.json",
        {
            "schema_version": supervisor.SCHEMA_VERSION,
            "segment_id": "segment_0001",
            "requested_duration_seconds": 1.0,
            "children": {
                "btc": {
                    "returncode": 0,
                    "group_cleanup": {"group_alive_after_cleanup": False},
                }
            },
            "passes": True,
        },
    )
    _write_json(
        campaign_root / "runtime_source.json",
        {"sealed_at": "2026-07-29T00:00:00+00:00", "files": {"collector": {}}},
    )
    _write_json(
        campaign_root / "abort_manifest.json",
        {"task_id": "0729T010", "passes": False},
    )
    _write_json(
        campaign_root / "run_status.json",
        {"task_id": "0729T010", "state": "failed"},
    )
    _write_json(
        campaign_root / "heartbeat.json",
        {"task_id": "0729T010", "state": "failed"},
    )
    return campaign_root


def test_parse_profiles_and_compute_segments() -> None:
    assert supervisor.parse_profiles("btc,eth,btc,mu") == ["btc", "eth", "mu"]
    assert supervisor.compute_segments(8.0, 3.0) == [3.0, 3.0, 2.0]
    with pytest.raises(ValueError):
        supervisor.compute_segments(0, 1)


def test_collection_command_enables_research_max_and_raw_only(tmp_path: Path) -> None:
    command = supervisor.build_collection_command(
        python_executable=sys.executable,
        profile_id="skhynix",
        sample_dir=tmp_path,
        duration_seconds=30,
        task_id="0729T009",
    )
    assert "--hyperliquid-research-max" in command
    assert "--skip-alignment" in command
    assert "--clean-output" in command
    assert command[command.index("--symbol-profile") + 1] == "skhynix"


def test_strict_quality_accepts_complete_sample(tmp_path: Path) -> None:
    quality = supervisor.validate_profile_sample(
        sample_dir=_valid_sample(tmp_path),
        profile_id="btc",
        requested_duration_seconds=60,
    )
    assert quality["passes"] is True
    assert quality["failures"] == []
    assert quality["cross_exchange_overlap_seconds"] == pytest.approx(59.9999999)


def test_strict_quality_accepts_verified_auxiliary_reconnect(tmp_path: Path) -> None:
    sample_dir = _valid_sample(tmp_path)
    _configure_auxiliary_reconnect(sample_dir)
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="btc",
        requested_duration_seconds=60,
    )
    assert quality["passes"] is True
    assert quality["failures"] == []
    assert quality["warnings"] == [
        "hyperliquid_auxiliary_track_recovered_reconnect:asset_context:1"
    ]
    assert len(quality["degraded_intervals"]) == 1
    interval = quality["degraded_intervals"][0]
    assert interval["track_id"] == "asset_context"
    assert interval["degraded_start_local_ts_ns"] == 29_000_000_000
    assert interval["recovered_local_ts_ns"] == 31_000_000_000
    assert interval["duration_ms"] == pytest.approx(2_000.0)


@pytest.mark.parametrize(
    ("include_recovery_ack", "include_resumed_data", "failure"),
    [
        (
            False,
            True,
            "hyperliquid_auxiliary_subscription_ack_incomplete:asset_context",
        ),
        (
            True,
            False,
            "hyperliquid_auxiliary_resumed_data_missing:asset_context:activeAssetCtx",
        ),
    ],
)
def test_strict_quality_rejects_unverified_auxiliary_reconnect(
    tmp_path: Path,
    include_recovery_ack: bool,
    include_resumed_data: bool,
    failure: str,
) -> None:
    sample_dir = _valid_sample(tmp_path)
    _configure_auxiliary_reconnect(
        sample_dir,
        include_recovery_ack=include_recovery_ack,
        include_resumed_data=include_resumed_data,
    )
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="btc",
        requested_duration_seconds=60,
    )
    assert quality["passes"] is False
    assert failure in quality["failures"]


def test_strict_quality_rejects_auxiliary_degraded_interval_above_gate(
    tmp_path: Path,
) -> None:
    sample_dir = _valid_sample(tmp_path)
    _configure_auxiliary_reconnect(
        sample_dir,
        resumed_ts_ns=50_000_000_000,
    )
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="btc",
        requested_duration_seconds=60,
        max_market_arrival_gap_seconds=15,
    )
    assert quality["passes"] is False
    assert (
        "hyperliquid_auxiliary_degraded_interval_above_gate:asset_context"
        in quality["failures"]
    )


@pytest.mark.parametrize("track_id", ["fast_market", "standard_l2"])
def test_strict_quality_keeps_core_l2_reconnect_as_hard_failure(
    tmp_path: Path,
    track_id: str,
) -> None:
    sample_dir = _valid_sample(tmp_path)
    bundle_path = sample_dir / "hyperliquid_public_sample" / "research_bundle_manifest.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    bundle["tracks"][track_id]["reconnect_count"] = 1
    _write_json(bundle_path, bundle)
    relative = bundle["tracks"][track_id]["relative_output_dir"]
    manifest_path = (
        sample_dir
        / "hyperliquid_public_sample"
        / relative
        / "collection_manifest.json"
    ).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["reconnect_count"] = 1
    _write_json(manifest_path, manifest)
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="btc",
        requested_duration_seconds=60,
    )
    assert quality["passes"] is False
    assert f"hyperliquid_core_track_reconnect_nonzero:{track_id}" in quality["failures"]


def test_main_dex_profiles_do_not_require_target_dex_all_mids(tmp_path: Path) -> None:
    sample_dir = _valid_sample(tmp_path)
    path = sample_dir / "hyperliquid_public_sample" / "research_bundle_manifest.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    del bundle["tracks"]["target_dex_all_mids"]
    _write_json(path, bundle)
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="btc",
        requested_duration_seconds=60,
    )
    assert quality["passes"] is True


def test_named_dex_profiles_require_target_dex_all_mids(tmp_path: Path) -> None:
    sample_dir = _valid_sample(tmp_path)
    path = sample_dir / "hyperliquid_public_sample" / "research_bundle_manifest.json"
    bundle = json.loads(path.read_text(encoding="utf-8"))
    del bundle["tracks"]["target_dex_all_mids"]
    _write_json(path, bundle)
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="mu",
        requested_duration_seconds=60,
    )
    assert "hyperliquid_required_tracks_missing" in quality["failures"]


@pytest.mark.parametrize(
    ("target", "key", "value", "failure"),
    [
        ("binance", "depth_snapshot_bridge_valid", False, "binance_snapshot_bridge_invalid"),
        ("binance", "depth_continuity_gap_count", 1, "binance_depth_continuity_gap"),
        ("binance", "reconnect_count", 1, "binance_reconnect_nonzero"),
        ("hyperliquid", "research_bundle", {"enabled": True, "all_tracks_pass": False}, "hyperliquid_research_bundle_failed"),
    ],
)
def test_strict_quality_fault_injection(
    tmp_path: Path,
    target: str,
    key: str,
    value,
    failure: str,
) -> None:
    sample_dir = _valid_sample(tmp_path)
    path = (
        sample_dir / "binance_public_raw" / "collection_manifest.json"
        if target == "binance"
        else sample_dir / "hyperliquid_public_sample" / "collection_manifest.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[key] = value
    _write_json(path, manifest)
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="btc",
        requested_duration_seconds=60,
    )
    assert quality["passes"] is False
    assert failure in quality["failures"]


def test_strict_quality_rejects_missing_manifest(tmp_path: Path) -> None:
    quality = supervisor.validate_profile_sample(
        sample_dir=tmp_path / "missing",
        profile_id="btc",
        requested_duration_seconds=60,
    )
    assert quality["passes"] is False
    assert any(item.startswith("missing_binance_manifest") for item in quality["failures"])


def test_strict_quality_rejects_raw_sha_mismatch(tmp_path: Path) -> None:
    sample_dir = _valid_sample(tmp_path)
    (sample_dir / "binance_public_raw" / "raw.gz").write_bytes(b"changed")
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="btc",
        requested_duration_seconds=60,
    )
    assert "binance_raw_sha256_mismatch" in quality["failures"]


def test_strict_quality_rejects_stale_market_channels(tmp_path: Path) -> None:
    sample_dir = _valid_sample(tmp_path)
    path = sample_dir / "hyperliquid_public_sample" / "research_tracks" / "standard_l2" / "collection_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["last_local_ts_by_channel"]["l2Book"] = 2_000_000_000
    _write_json(path, manifest)
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="btc",
        requested_duration_seconds=60,
    )
    assert quality["passes"] is False
    assert "hyperliquid_standard_l2_l2Book_coverage_below_gate" in quality["failures"]
    assert "hyperliquid_standard_l2_l2Book_tail_stale" in quality["failures"]


def test_strict_quality_rejects_head_silence_even_when_coverage_passes(tmp_path: Path) -> None:
    sample_dir = _valid_sample(tmp_path)
    path = sample_dir / "hyperliquid_public_sample" / "research_tracks" / "standard_l2" / "collection_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["first_local_ts_by_channel"]["l2Book"] = 11_000_000_000
    _write_json(path, manifest)
    quality = supervisor.validate_profile_sample(
        sample_dir=sample_dir,
        profile_id="btc",
        requested_duration_seconds=60,
    )
    metrics = quality["freshness"]["hyperliquid"]["standard_l2"]["l2Book"]
    assert metrics["coverage_ratio"] >= 0.80
    assert "hyperliquid_standard_l2_l2Book_head_stale" in quality["failures"]


def test_wait_segment_fails_on_nonzero_and_reaps_peer(tmp_path: Path) -> None:
    instance = supervisor.CollectionCampaignSupervisor(_args(tmp_path))
    log_a = tmp_path / "a.log"
    log_b = tmp_path / "b.log"
    handle_a = log_a.open("w", encoding="utf-8")
    handle_b = log_b.open("w", encoding="utf-8")
    failed = subprocess.Popen(
        [sys.executable, "-c", "raise SystemExit(7)"],
        stdout=handle_a,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    peer = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=handle_b,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    instance.active_children = {
        "btc": supervisor.ManagedChild("btc", [], failed, log_a, handle_a, time.monotonic()),
        "eth": supervisor.ManagedChild("eth", [], peer, log_b, handle_b, time.monotonic()),
    }
    with pytest.raises(supervisor.SupervisorError, match="child_nonzero:btc:7"):
        instance.wait_segment(duration_seconds=1)
    assert failed.poll() == 7
    assert peer.poll() is not None
    assert instance.active_children == {}


def test_wait_segment_reaps_descendants_left_by_failed_wrapper(tmp_path: Path) -> None:
    instance = supervisor.CollectionCampaignSupervisor(_args(tmp_path))
    pid_path = tmp_path / "descendant.pid"
    log_path = tmp_path / "failed-wrapper.log"
    handle = log_path.open("w", encoding="utf-8")
    code = (
        "import pathlib,subprocess,sys;"
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']);"
        f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid));"
        "raise SystemExit(7)"
    )
    failed = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    instance.active_children = {
        "btc": supervisor.ManagedChild("btc", [], failed, log_path, handle, time.monotonic())
    }
    with pytest.raises(supervisor.SupervisorError, match="child_nonzero:btc:7"):
        instance.wait_segment(duration_seconds=1)
    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        try:
            os.kill(descendant_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"descendant process {descendant_pid} survived process-group cleanup")
    cleanup = instance.last_child_results["btc"]["group_cleanup"]
    assert cleanup["group_detected"] is True
    assert cleanup["group_alive_after_cleanup"] is False


def test_wait_segment_timeout_terminates_and_reaps_child(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.timeout_grace_seconds = 0.01
    instance = supervisor.CollectionCampaignSupervisor(args)
    log_path = tmp_path / "timeout.log"
    handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    instance.active_children = {
        "btc": supervisor.ManagedChild("btc", [], process, log_path, handle, time.monotonic())
    }
    with pytest.raises(supervisor.SupervisorError, match="segment_timeout"):
        instance.wait_segment(duration_seconds=0.01)
    assert process.poll() is not None
    assert instance.active_children == {}


def test_run_writes_abort_manifest_when_timeline_or_segment_fails(monkeypatch, tmp_path: Path) -> None:
    instance = supervisor.CollectionCampaignSupervisor(_args(tmp_path))

    def fail_segment(**_kwargs):
        raise supervisor.SupervisorError("timeline_quality_failed:segment_0001:btc")

    monkeypatch.setattr(instance, "collect_segment", fail_segment)
    assert instance.run() == 2
    abort = json.loads((instance.campaign_root / "abort_manifest.json").read_text(encoding="utf-8"))
    status = json.loads(instance.status_path.read_text(encoding="utf-8"))
    assert abort["error"] == "timeline_quality_failed:segment_0001:btc"
    assert abort["runtime_source"]["files"]["supervisor"]["sha256"]
    assert status["state"] == "failed"
    assert status["phase"] == "campaign_aborted"
    heartbeat = json.loads(instance.heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["state"] == "failed"
    assert heartbeat["phase"] == "heartbeat_stopped"
    assert heartbeat["terminal"] is True


def test_collect_segment_persists_child_results_on_failure(monkeypatch, tmp_path: Path) -> None:
    instance = supervisor.CollectionCampaignSupervisor(_args(tmp_path))
    instance.campaign_root.mkdir(parents=True)
    instance.write_status(state="running", phase="test")
    def fail_wait(*, duration_seconds: float):
        instance.active_children.clear()
        instance.last_child_results = {
            "btc": {
                "pid": 123,
                "returncode": 7,
                "reason": "exited",
                "group_cleanup": {"group_detected": True, "group_alive_after_cleanup": False},
            }
        }
        raise supervisor.SupervisorError(f"child_nonzero:btc:7:{duration_seconds}")

    monkeypatch.setattr(
        instance,
        "spawn_child",
        lambda **_kwargs: SimpleNamespace(
            process=SimpleNamespace(pid=123),
            command=[],
            log_path=tmp_path / "collector.log",
        ),
    )
    monkeypatch.setattr(instance, "wait_segment", fail_wait)
    with pytest.raises(supervisor.SupervisorError, match="child_nonzero"):
        instance.collect_segment(segment_index=1, duration_seconds=1, profiles=["btc"])
    payload = json.loads(
        (instance.campaign_root / "segments" / "segment_0001" / "segment_child_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["passes"] is False
    assert payload["children"]["btc"]["returncode"] == 7


def test_spawn_failure_resets_prior_child_results_and_records_failure(monkeypatch, tmp_path: Path) -> None:
    instance = supervisor.CollectionCampaignSupervisor(_args(tmp_path))
    instance.campaign_root.mkdir(parents=True)
    instance.write_status(state="running", phase="test")
    instance.last_child_results = {"eth": {"returncode": 0}}

    def fail_spawn(**_kwargs):
        raise RuntimeError("spawn exploded")

    monkeypatch.setattr(instance, "spawn_child", fail_spawn)
    with pytest.raises(RuntimeError, match="spawn exploded"):
        instance.collect_segment(segment_index=2, duration_seconds=1, profiles=["btc"])
    payload = json.loads(
        (instance.campaign_root / "segments" / "segment_0002" / "segment_child_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(payload["children"]) == {"btc"}
    assert payload["children"]["btc"]["reason"] == "spawn_failed"
    assert payload["children"]["btc"]["error"] == "spawn exploded"


def test_run_collects_all_segments_before_postprocessing(monkeypatch, tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.total_duration_seconds = 3
    args.segment_duration_seconds = 1
    instance = supervisor.CollectionCampaignSupervisor(args)
    calls: list[str] = []

    def fake_collect(*, segment_index: int, duration_seconds: float, profiles: list[str]):
        calls.append(f"collect{segment_index}")
        return {
            "segment_id": f"segment_{segment_index:04d}",
            "segment_index": segment_index,
            "segment_dir": instance.campaign_root / "segments" / f"segment_{segment_index:04d}",
            "requested_duration_seconds": duration_seconds,
            "child_results": {},
        }

    def fake_process(collection: dict, *, profiles: list[str]):
        calls.append(f"process{collection['segment_index']}")
        return {
            "segment_id": collection["segment_id"],
            "requested_duration_seconds": collection["requested_duration_seconds"],
            "profiles": {},
        }

    def fake_index(_segments, _profiles):
        path = instance.campaign_root / "timeline_index.csv"
        path.write_text("test\n", encoding="utf-8")
        return path

    monkeypatch.setattr(instance, "collect_segment", fake_collect)
    monkeypatch.setattr(instance, "process_segment", fake_process)
    monkeypatch.setattr(instance, "write_timeline_index", fake_index)
    assert instance.run() == 0
    assert calls == ["collect1", "collect2", "collect3", "process1", "process2", "process3"]


def test_postprocess_only_reuses_existing_collection_without_spawning(
    monkeypatch,
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.postprocess_only = True
    campaign_root = _prepare_existing_campaign(args)

    def fake_timeline_builder(**kwargs):
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        timeline_path = output_dir / "common_l2_timeline.csv.gz"
        timeline_path.write_bytes(b"timeline")
        manifest = {
            "passes": True,
            "failures": [],
            "timeline_file": str(timeline_path),
            "timeline_sha256": sha256_file(timeline_path),
            "timeline_row_count": 10,
            "first_common_ts_ns": 100,
            "last_common_ts_ns": 200,
        }
        _write_json(output_dir / "common_l2_timeline_manifest.json", manifest)
        return manifest

    instance = supervisor.CollectionCampaignSupervisor(
        args,
        timeline_builder=fake_timeline_builder,
    )

    def fail_collect(**_kwargs):
        raise AssertionError("postprocess-only must not spawn collectors")

    monkeypatch.setattr(instance, "collect_segment", fail_collect)
    assert instance.run() == 0
    manifest = json.loads(
        (campaign_root / "campaign_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["passes"] is True
    assert manifest["execution_mode"] == "postprocess_only"
    assert manifest["collection_task_ids"] == ["0729T010"]
    assert manifest["collection_runtime_source"]["sealed_at"] == "2026-07-29T00:00:00+00:00"
    assert manifest["postprocess_runtime_source"]["files"]["supervisor"]["sha256"]
    assert len(manifest["segments"]) == 1
    assert not (campaign_root / "abort_manifest.json").exists()
    history_paths = list((campaign_root / "postprocess_history").glob("*/abort_manifest.json"))
    assert len(history_paths) == 1


def test_postprocess_only_rejects_clean_output(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.postprocess_only = True
    args.clean_output = True
    _prepare_existing_campaign(args)
    instance = supervisor.CollectionCampaignSupervisor(args)
    with pytest.raises(
        supervisor.SupervisorError,
        match="postprocess_only_forbids_clean_output",
    ):
        instance.run()


def test_timeline_index_has_header_and_all_segment_rows(tmp_path: Path) -> None:
    instance = supervisor.CollectionCampaignSupervisor(_args(tmp_path))
    instance.campaign_root.mkdir(parents=True)
    segments = []
    for index, first_ts in ((1, 100), (2, 300)):
        segments.append(
            {
                "segment_id": f"segment_{index:04d}",
                "profiles": {
                    "btc": {
                        "timeline": {
                            "timeline_file": f"segment-{index}.csv.gz",
                            "timeline_sha256": f"sha-{index}",
                            "timeline_row_count": 10 + index,
                            "first_common_ts_ns": first_ts,
                            "last_common_ts_ns": first_ts + 100,
                        }
                    }
                },
            }
        )
    path = instance.write_timeline_index(segments, ["btc"])
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["segment_id"] == "segment_0001"
    assert rows[1]["segment_id"] == "segment_0002"
    assert rows[1]["previous_segment_gap_ms"] == "0.000100"


def test_heartbeat_and_terminal_status_are_atomic_json(tmp_path: Path) -> None:
    instance = supervisor.CollectionCampaignSupervisor(_args(tmp_path))
    instance.campaign_root.mkdir(parents=True)
    instance.start_heartbeat()
    instance.write_status(state="running", phase="test")
    time.sleep(0.03)
    instance.stop_heartbeat()
    assert json.loads(instance.status_path.read_text(encoding="utf-8"))["phase"] == "test"
    heartbeat = json.loads(instance.heartbeat_path.read_text(encoding="utf-8"))
    assert heartbeat["campaign_id"] == "test-campaign"


def test_campaign_rejects_nonempty_output_without_clean_flag(tmp_path: Path) -> None:
    args = _args(tmp_path)
    campaign_root = Path(args.output_dir)
    campaign_root.mkdir(parents=True)
    (campaign_root / "stale.txt").write_text("stale", encoding="utf-8")
    instance = supervisor.CollectionCampaignSupervisor(args)
    with pytest.raises(supervisor.SupervisorError, match="campaign_output_not_empty"):
        instance.run()


def test_supervisor_argument_ranges_fail_closed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.min_overlap_ratio = -1
    with pytest.raises(supervisor.SupervisorError, match="min_overlap_ratio_must_be_in_0_1"):
        supervisor.validate_supervisor_args(args)
    args = _args(tmp_path)
    args.max_hyperliquid_fast_age_ms = 0
    with pytest.raises(supervisor.SupervisorError, match="max_hyperliquid_fast_age_ms_must_be_positive"):
        supervisor.validate_supervisor_args(args)
    args = _args(tmp_path)
    args.max_market_arrival_gap_seconds = float("inf")
    with pytest.raises(supervisor.SupervisorError, match="max_market_arrival_gap_seconds_must_be_positive"):
        supervisor.validate_supervisor_args(args)
