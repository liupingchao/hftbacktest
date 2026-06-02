from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


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
        task_id="0601T001",
    )

    assert manifest["task_id"] == "0601T001"
    assert manifest["subscription_ack_received"] is True
    assert manifest["subscription_ack_count_by_channel"] == {"l2Book": 1, "trades": 1}
    assert manifest["message_count_by_channel"]["l2Book"] == 1
    assert manifest["message_count_by_channel"]["trades"] == 1
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
