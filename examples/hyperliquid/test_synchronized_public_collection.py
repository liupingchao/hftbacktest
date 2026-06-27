from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import synchronized_public_collection as sync


class WebSocketTimeoutException(Exception):
    pass


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False
        self._messages = [
            {"result": None, "id": "binance-test"},
            {
                "stream": "btcusdt@depth@0ms",
                "data": {
                    "e": "depthUpdate",
                    "E": 1_000_001,
                    "T": 1_000_001,
                    "s": "BTCUSDT",
                    "U": 11,
                    "u": 12,
                    "pu": 10,
                    "b": [["100.0", "1.0"]],
                    "a": [["100.1", "2.0"]],
                },
            },
            {
                "stream": "btcusdt@trade",
                "data": {
                    "e": "trade",
                    "E": 1_000_002,
                    "T": 1_000_002,
                    "s": "BTCUSDT",
                    "t": 123,
                    "p": "100.1",
                    "q": "0.01",
                    "m": False,
                },
            },
            {
                "stream": "btcusdt@bookTicker",
                "data": {
                    "e": "bookTicker",
                    "E": 1_000_003,
                    "T": 1_000_003,
                    "s": "BTCUSDT",
                    "u": 12,
                    "b": "100.0",
                    "B": "1.0",
                    "a": "100.1",
                    "A": "2.0",
                },
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

    def ping(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True


class _DoneProcess:
    pid = 12345
    returncode = 0

    def wait(self) -> int:
        return self.returncode


def test_build_collection_commands_stay_public_only(tmp_path: Path) -> None:
    binance_cmd = sync.build_binance_collection_command(
        output_dir=tmp_path / "binance",
        symbol="BTCUSDT",
        duration_seconds=600,
        streams=["trade", "depth@0ms", "bookTicker"],
        task_id="0602T001",
    )
    hyperliquid_cmd = sync.build_hyperliquid_collection_command(
        output_dir=tmp_path / "hl",
        coin="BTC",
        duration_seconds=600,
        task_id="0602T001",
        l2book_fast=True,
    )

    joined = " ".join(binance_cmd + hyperliquid_cmd)
    assert "run_live.sh" not in joined
    assert "live_tick_mm.py" not in joined
    assert "connector" not in joined
    assert "collect-binance-public" in binance_cmd
    assert "hyperliquid_public_sample.py" in joined
    assert "--l2book-fast" in hyperliquid_cmd


def test_orchestrate_collection_can_skip_remote_alignment(monkeypatch, tmp_path: Path) -> None:
    run_process_commands: list[list[str]] = []

    def fake_run_process(command: list[str], *, cwd: Path, log_path: Path) -> _DoneProcess:
        run_process_commands.append(command)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("ok\n", encoding="utf-8")
        return _DoneProcess()

    def fail_run_command(*_args, **_kwargs):
        raise AssertionError("alignment command must not run when --skip-alignment is set")

    def fake_read_json(path: str | Path) -> dict:
        text = str(path)
        if text.endswith("binance_public_raw/collection_manifest.json"):
            return {
                "local_start_ts": 100,
                "local_end_ts": 1_000_000_000_100,
                "raw_sha256": "binance-sha",
                "message_count_by_event_type": {"depthUpdate": 2, "trade": 1, "bookTicker": 1},
                "depth_snapshot_status": "ok",
            }
        if text.endswith("hyperliquid_public_sample/collection_manifest.json"):
            return {
                "local_start_ts": 200,
                "local_end_ts": 900_000_000_200,
                "raw_sha256": "hl-sha",
                "message_count_by_channel": {"l2Book": 2, "trades": 1},
            }
        raise AssertionError(f"unexpected json read: {path}")

    monkeypatch.setattr(sync, "run_process", fake_run_process)
    monkeypatch.setattr(sync, "run_command", fail_run_command)
    monkeypatch.setattr(sync, "_read_json", fake_read_json)

    args = sync.parse_args(
        [
            "collect",
            "--output-dir",
            str(tmp_path),
            "--duration-seconds",
            "1",
            "--task-id",
            "0627T001",
            "--hyperliquid-l2book-fast",
            "--skip-alignment",
        ]
    )
    assert sync.orchestrate_collection(args) == 0

    assert len(run_process_commands) == 2
    run_manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    sample_manifest = json.loads((tmp_path / "sample_manifest.json").read_text(encoding="utf-8"))
    quality = json.loads((tmp_path / "synchronization_quality_summary.json").read_text(encoding="utf-8"))
    assert run_manifest["alignment_status"] == "skipped"
    assert run_manifest["raw_collection_only"] is True
    assert sample_manifest["raw_collection_only"] is True
    assert run_manifest["binance_alignment"]["required_execution_host"] == "macmini_or_amdserver"
    assert quality["raw_collection_only"] is True
    assert quality["alignment_execution_host"] == "macmini_or_amdserver"


def test_compute_overlap_reports_non_empty_window() -> None:
    overlap = sync.compute_overlap(
        {"local_start_ts": 100, "local_end_ts": 1_100},
        {"local_start_ts": 200, "local_end_ts": 900},
    )

    assert overlap["overlap_start_ts"] == 200
    assert overlap["overlap_end_ts"] == 900
    assert overlap["overlap_seconds"] == 0.0000007
    assert overlap["non_empty"] is True


def test_build_binance_stream_names_prefixes_symbol() -> None:
    assert sync.build_binance_stream_names("BTCUSDT", ["trade", "depth@0ms", "btcusdt@bookTicker"]) == [
        "btcusdt@trade",
        "btcusdt@depth@0ms",
        "btcusdt@bookticker",
    ]


def test_normalize_binance_trade_adds_converter_market_flag() -> None:
    normalized = sync.normalize_binance_message(
        {
            "stream": "btcusdt@trade",
            "data": {
                "e": "trade",
                "E": 1_000,
                "s": "BTCUSDT",
                "t": 1,
                "p": "100.0",
                "q": "0.01",
                "m": True,
            },
        },
        "BTCUSDT",
    )

    assert normalized["data"]["X"] == "MARKET"
    assert normalized["data"]["T"] == 1_000


def test_collect_binance_public_sample_writes_raw_and_manifest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sync, "_connect_websocket", lambda *_args, **_kwargs: _FakeWebSocket())
    monkeypatch.setattr(
        sync,
        "fetch_binance_depth_snapshot",
        lambda **_kwargs: {
            "local_ts": 1_000_000_000_000,
            "local_time": "2026-06-02T00:00:00+00:00",
            "url": "https://example.invalid/fapi/v1/depth",
            "symbol": "BTCUSDT",
            "limit": 1000,
            "http_status": 200,
            "status": "ok",
            "snapshot": {
                "lastUpdateId": 10,
                "T": 1_000_000,
                "bids": [["100.0", "1.0"]],
                "asks": [["100.1", "2.0"]],
            },
        },
    )

    manifest = sync.collect_binance_public_sample(
        symbol="BTCUSDT",
        duration_seconds=0.02,
        output_dir=tmp_path,
        ws_url="wss://example.invalid/ws",
        rest_url="https://example.invalid",
        streams=["trade", "depth@0ms", "bookTicker"],
        request_timeout=1.0,
        websocket_timeout=0.01,
        max_reconnects=0,
        snapshot_limit=1000,
        task_id="0602T001",
    )

    assert manifest["task_id"] == "0602T001"
    assert manifest["depth_snapshot_status"] == "ok"
    assert manifest["message_count_by_event_type"]["depthUpdate"] == 1
    assert manifest["message_count_by_event_type"]["trade"] == 1
    assert manifest["message_count_by_event_type"]["bookTicker"] == 1
    assert manifest["no_order_endpoints"] is True
    assert manifest["no_strategy_process"] is True
    assert (tmp_path / "raw.sha256").read_text(encoding="utf-8").strip() == manifest["raw_sha256"]

    with gzip.open(tmp_path / "raw.gz", "rt", encoding="utf-8") as fh:
        messages = [json.loads(line.split(" ", 1)[1]) for line in fh if line.strip()]
    assert messages[0]["lastUpdateId"] == 10
    trade = next(msg for msg in messages if msg.get("data", {}).get("e") == "trade")
    assert trade["data"]["X"] == "MARKET"


def test_write_synchronized_manifests_records_overlap_and_boundaries(tmp_path: Path) -> None:
    binance_alignment = tmp_path / "binance_alignment"
    hyperliquid_alignment = tmp_path / "hyperliquid_public_sample" / "alignment"
    sync._write_json(binance_alignment / "metrics.json", {"top5_row_count": 2})
    sync._write_json(
        hyperliquid_alignment / "metrics.json",
        {"sample_classification": "passes_pricing_research_market_view"},
    )
    quality = sync.write_synchronized_manifests(
        output_dir=tmp_path,
        binance_manifest={
            "local_start_ts": 100,
            "local_end_ts": 1_000_000_000_100,
            "raw_sha256": "binance-sha",
            "message_count_by_event_type": {"depthUpdate": 2, "trade": 1, "bookTicker": 1},
            "depth_snapshot_status": "ok",
        },
        hyperliquid_manifest={
            "local_start_ts": 200,
            "local_end_ts": 900_000_000_200,
            "raw_sha256": "hl-sha",
            "message_count_by_channel": {"l2Book": 2, "trades": 1},
        },
        binance_alignment={"returncode": 0},
        hyperliquid_alignment={"returncode": 0},
        task_id="0602T001",
        planned_start_time="2026-06-02T00:00:00+00:00",
        binance_symbol="BTCUSDT",
        hyperliquid_coin="BTC",
        requested_duration_seconds=900.0,
        commands={"binance_collection": ["python", "collect-binance-public"]},
    )

    assert quality["overlap"]["non_empty"] is True
    assert quality["binance"]["depth_update_count"] == 2
    assert quality["hyperliquid"]["classification"] == "passes_pricing_research_market_view"
    assert quality["lead_lag_statistical_conclusion"] == "not_calculated_in_0602T001"

    sample_manifest = json.loads((tmp_path / "sample_manifest.json").read_text(encoding="utf-8"))
    assert sample_manifest["no_private_keys"] is True
    assert sample_manifest["venues"]["binance"]["role"] == "lead_public_market_data"
    assert sample_manifest["venues"]["hyperliquid"]["role"] == "lag_public_market_data"
