from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import hyperliquid_public_sample as sample


class _FakeResponse:
    ok = True
    status_code = 200

    def json(self) -> dict:
        return {
            "coin": "BTC",
            "time": 1_000,
            "levels": [
                [{"px": "100.0", "sz": "1.0", "n": 1}],
                [{"px": "100.1", "sz": "2.0", "n": 2}],
            ],
        }


class WebSocketTimeoutException(Exception):
    pass


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._messages = [
            {
                "channel": "subscriptionResponse",
                "data": {"method": "subscribe", "subscription": {"type": "l2Book", "coin": "BTC"}},
            },
            {
                "channel": "subscriptionResponse",
                "data": {"method": "subscribe", "subscription": {"type": "trades", "coin": "BTC"}},
            },
            {
                "channel": "l2Book",
                "data": {
                    "coin": "BTC",
                    "time": 1_000,
                    "levels": [
                        [{"px": "100.0", "sz": "1.0", "n": 1}],
                        [{"px": "100.1", "sz": "2.0", "n": 1}],
                    ],
                },
            },
            {
                "channel": "trades",
                "data": [
                    {
                        "coin": "BTC",
                        "side": "B",
                        "px": "100.1",
                        "sz": "0.01",
                        "time": 1_001,
                        "hash": "0x1",
                        "tid": 1,
                        "users": ["0x0", "0x1"],
                    }
                ],
            },
        ]

    def send(self, text: str) -> None:
        self.sent.append(text)

    def recv(self) -> str:
        if self._messages:
            return json.dumps(self._messages.pop(0))
        raise WebSocketTimeoutException("timeout")

    def settimeout(self, _timeout: float) -> None:
        return None

    def close(self) -> None:
        self.closed = True


def test_fetch_l2book_snapshot_extracts_recovery_summary() -> None:
    snapshot = sample.fetch_l2book_snapshot(
        info_url="https://example.invalid/info",
        coin="BTC",
        reason="startup",
        timeout=1.0,
        task_id="0601T001",
        post=lambda *args, **kwargs: _FakeResponse(),
    )

    assert snapshot["status"] == "ok"
    assert snapshot["task_id"] == "0601T001"
    assert snapshot["best_bid_px"] == "100.0"
    assert snapshot["best_ask_px"] == "100.1"
    assert snapshot["bid_level_count"] == 1
    assert snapshot["ask_level_count"] == 1


def test_l2book_fast_subscription_adds_fast_only_to_l2book() -> None:
    messages = sample._subscription_messages(["l2Book", "trades"], "BTC", l2book_fast=True)
    payloads = [json.loads(message) for message in messages]

    assert payloads[0]["subscription"] == {"type": "l2Book", "coin": "BTC", "fast": True}
    assert payloads[1]["subscription"] == {"type": "trades", "coin": "BTC"}


def test_research_max_track_specs_preserve_hot_and_deeper_tracks() -> None:
    tracks = {track["track_id"]: track for track in sample.research_max_track_specs("xyz:SKHX")}

    assert tracks["fast_market"]["relative_output_dir"] == "."
    assert tracks["fast_market"]["subscriptions"] == [
        {"type": "l2Book", "coin": "xyz:SKHX", "fast": True},
        {"type": "trades", "coin": "xyz:SKHX"},
        {"type": "bbo", "coin": "xyz:SKHX"},
    ]
    assert tracks["standard_l2"]["subscriptions"] == [{"type": "l2Book", "coin": "xyz:SKHX"}]
    assert tracks["asset_context"]["subscriptions"] == [
        {"type": "activeAssetCtx", "coin": "xyz:SKHX"},
        {"type": "candle", "coin": "xyz:SKHX", "interval": "1m"},
    ]
    assert tracks["main_all_mids"]["subscriptions"] == [{"type": "allMids"}]
    assert tracks["target_dex_all_mids"]["subscriptions"] == [{"type": "allMids", "dex": "xyz"}]


def test_collection_stats_reports_local_and_exchange_cadence() -> None:
    stats = sample.CollectionStats(session_id="test", channels=["l2Book"])
    stats.observe(1_000_000_000, {"channel": "l2Book", "data": {"time": 1_000, "levels": [[], []]}})
    stats.observe(1_500_000_000, {"channel": "l2Book", "data": {"time": 1_510, "levels": [[], []]}})
    stats.observe(2_100_000_000, {"channel": "l2Book", "data": {"time": 2_100, "levels": [[], []]}})

    assert sample._gap_summary(stats.arrival_gap_ms_by_channel["l2Book"]) == {
        "count": 2,
        "min": 500.0,
        "p50": 550.0,
        "p90": 590.0,
        "p99": 599.0,
        "max": 600.0,
    }
    assert sample._gap_summary(stats.exchange_gap_ms_by_channel["l2Book"])["p50"] == 550.0


def test_empty_frame_is_transport_close_not_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "raw.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        message = sample.write_raw_message(fh, 123, "")

    assert message == {
        "channel": "transport_close",
        "reason": "empty_websocket_frame",
    }
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        local_ts, payload = fh.read().split(" ", 1)
    assert local_ts == "123"
    assert json.loads(payload)["channel"] == "transport_close"


def test_nonempty_malformed_frame_remains_parse_error(tmp_path: Path) -> None:
    path = tmp_path / "raw.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        message = sample.write_raw_message(fh, 456, "{bad")

    assert message == {"channel": "parse_error", "raw_text": "{bad"}


def test_collect_research_bundle_writes_aggregate_manifest(tmp_path: Path) -> None:
    def fake_collector(**kwargs) -> dict:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        subscriptions = kwargs["subscription_specs"]
        message_counts = {str(subscription["type"]): 2 for subscription in subscriptions}
        manifest = {
            "local_start_ts": 100,
            "local_end_ts": 60_000_000_100,
            "raw_file": str(output_dir / "raw.gz"),
            "raw_sha256": f"sha-{kwargs['track_id']}",
            "subscription_requests": subscriptions,
            "message_count_by_channel": message_counts,
            "arrival_gap_ms_by_channel": {"l2Book": {"p50": 500.0}},
            "exchange_gap_ms_by_channel": {"l2Book": {"p50": 500.0}},
            "l2book_level_shape_counts": (
                {"5x5": 2} if kwargs["l2book_fast"] else {"20x20": 2}
            ),
            "connection_attempt_count": 1,
            "reconnect_count": 0,
            "close_reason": "duration_elapsed",
            "all_required_subscription_acks_received": True,
            "parse_error_count": 0,
            "raw_row_count_reconciled": True,
            "subscription_ack_count": len(subscriptions),
        }
        sample._write_json(output_dir / "collection_manifest.json", manifest)
        return manifest

    manifest = sample.collect_research_bundle(
        coin="xyz:SKHX",
        duration_seconds=60.0,
        output_dir=tmp_path,
        network="mainnet",
        ws_url="wss://example.invalid/ws",
        info_url="https://example.invalid/info",
        request_timeout=1.0,
        websocket_timeout=1.0,
        max_reconnects=0,
        task_id="0729T008",
        collector=fake_collector,
    )

    bundle = json.loads((tmp_path / "research_bundle_manifest.json").read_text(encoding="utf-8"))
    assert bundle["collection_priority"] == "information_max"
    assert bundle["track_count"] == 5
    assert bundle["quality"]["all_tracks_pass"] is True
    assert bundle["quality"]["dual_l2_information_quality"]["passes"] is True
    assert bundle["track_overlap"]["overlap_seconds"] == 60.0
    assert bundle["tracks"]["fast_market"]["l2book_level_shape_counts"] == {"5x5": 2}
    assert bundle["tracks"]["standard_l2"]["l2book_level_shape_counts"] == {"20x20": 2}
    assert manifest["research_bundle"]["track_count"] == 5
    assert manifest["research_bundle"]["all_tracks_pass"] is True


def test_collect_research_bundle_propagates_track_failure(tmp_path: Path) -> None:
    observed_stop_events = []

    def failing_collector(**kwargs) -> dict:
        observed_stop_events.append(kwargs["stop_event"])
        if kwargs["track_id"] == "standard_l2":
            raise RuntimeError("standard_l2_failed")
        return {
            "local_start_ts": 100,
            "local_end_ts": 200,
            "raw_file": "",
            "raw_sha256": "",
            "subscription_requests": kwargs["subscription_specs"],
            "message_count_by_channel": {
                str(subscription["type"]): 1 for subscription in kwargs["subscription_specs"]
            },
            "connection_attempt_count": 1,
            "reconnect_count": 0,
            "close_reason": "duration_elapsed",
            "all_required_subscription_acks_received": True,
            "parse_error_count": 0,
            "raw_row_count_reconciled": True,
            "subscription_ack_count": len(kwargs["subscription_specs"]),
        }

    with pytest.raises(RuntimeError, match="standard_l2_failed"):
        sample.collect_research_bundle(
            coin="xyz:SKHX",
            duration_seconds=1.0,
            output_dir=tmp_path,
            network="mainnet",
            ws_url="wss://example.invalid/ws",
            info_url="https://example.invalid/info",
            request_timeout=1.0,
            websocket_timeout=1.0,
            max_reconnects=0,
            collector=failing_collector,
        )
    assert observed_stop_events
    assert len({id(event) for event in observed_stop_events}) == 1
    assert observed_stop_events[0].is_set()


def test_collect_research_bundle_fails_closed_when_standard_l2_is_not_deeper(
    tmp_path: Path,
) -> None:
    def shallow_collector(**kwargs) -> dict:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        subscriptions = kwargs["subscription_specs"]
        has_l2 = any(subscription["type"] == "l2Book" for subscription in subscriptions)
        manifest = {
            "local_start_ts": 100,
            "local_end_ts": 200,
            "raw_file": str(output_dir / "raw.gz"),
            "raw_sha256": "sha",
            "subscription_requests": subscriptions,
            "message_count_by_channel": {
                str(subscription["type"]): 1 for subscription in subscriptions
            },
            "l2book_level_shape_counts": {"5x5": 1} if has_l2 else {},
            "connection_attempt_count": 1,
            "reconnect_count": 0,
            "close_reason": "duration_elapsed",
            "all_required_subscription_acks_received": True,
            "parse_error_count": 0,
            "raw_row_count_reconciled": True,
            "subscription_ack_count": len(subscriptions),
        }
        sample._write_json(output_dir / "collection_manifest.json", manifest)
        return manifest

    with pytest.raises(sample.ResearchBundleQualityError):
        sample.collect_research_bundle(
            coin="xyz:SKHX",
            duration_seconds=1.0,
            output_dir=tmp_path,
            network="mainnet",
            ws_url="wss://example.invalid/ws",
            info_url="https://example.invalid/info",
            request_timeout=1.0,
            websocket_timeout=1.0,
            max_reconnects=0,
            collector=shallow_collector,
        )

    bundle = json.loads((tmp_path / "research_bundle_manifest.json").read_text(encoding="utf-8"))
    assert bundle["quality"]["all_tracks_pass"] is False
    assert bundle["quality"]["dual_l2_information_quality"]["standard_is_deeper_than_fast"] is False


def test_subscription_response_must_be_subscribe_ack() -> None:
    stats = sample.CollectionStats(session_id="test", channels=["l2Book"])
    stats.observe(
        1,
        {
            "channel": "subscriptionResponse",
            "data": {
                "method": "unsubscribe",
                "subscription": {"type": "l2Book", "coin": "BTC", "fast": True},
            },
        },
    )

    assert stats.subscription_ack_count_by_channel == {}
    assert stats.subscription_ack_count_by_identity == {}


def test_collect_sample_writes_public_manifest_and_raw(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sample, "_connect_websocket", lambda *_args, **_kwargs: _FakeWebSocket())
    monkeypatch.setattr(
        sample,
        "fetch_l2book_snapshot",
        lambda **kwargs: {
            "schema_version": sample.SCHEMA_VERSION,
            "task_id": kwargs.get("task_id", sample.TASK_ID),
            "local_ts": 1,
            "local_time": "2026-05-29T00:00:00+00:00",
            "reason": kwargs["reason"],
            "coin": kwargs["coin"],
            "status": "ok",
            "http_status": 200,
            "error": "",
            "best_bid_px": "100.0",
            "best_ask_px": "100.1",
            "bid_level_count": 1,
            "ask_level_count": 1,
            "raw_payload": {"levels": [[{"px": "100.0"}], [{"px": "100.1"}]]},
        },
    )

    manifest = sample.collect_sample(
        coin="BTC",
        channels=["l2Book", "trades"],
        duration_seconds=0.02,
        output_dir=tmp_path,
        network="mainnet",
        ws_url="wss://example.invalid/ws",
        info_url="https://example.invalid/info",
        request_timeout=1.0,
        websocket_timeout=0.01,
        max_reconnects=0,
        l2book_fast=True,
        task_id="0601T001",
    )

    assert manifest["task_id"] == "0601T001"
    assert manifest["subscription_ack_received"] is True
    assert manifest["subscription_ack_count_by_channel"] == {"l2Book": 1, "trades": 1}
    assert manifest["message_count_by_channel"]["l2Book"] == 1
    assert manifest["message_count_by_channel"]["trades"] == 1
    assert manifest["l2book_level_shape_counts"] == {"1x1": 1}
    assert manifest["raw_row_count"] == 4
    assert manifest["raw_row_count_reconciled"] is True
    assert manifest["subscription_options"]["l2book_fast"] is True
    assert manifest["recovery_snapshot_count"] == 1
    assert manifest["no_order_endpoints"] is True
    assert (tmp_path / "collection_manifest.json").exists()
    assert (tmp_path / "raw.sha256").read_text(encoding="utf-8").strip() == manifest["raw_sha256"]

    with gzip.open(tmp_path / "raw.gz", "rt", encoding="utf-8") as fh:
        channels = [json.loads(line.split(" ", 1)[1])["channel"] for line in fh if line.strip()]
    assert channels == ["subscriptionResponse", "subscriptionResponse", "l2Book", "trades"]

    with (tmp_path / "recovery_snapshots.jsonl").open(encoding="utf-8") as fh:
        snapshots = [json.loads(line) for line in fh if line.strip()]
    assert snapshots[0]["task_id"] == "0601T001"
