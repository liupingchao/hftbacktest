from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import cross_exchange_funding_rates as funding


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> Any:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHttp:
    def __init__(self, *, start_ms: int) -> None:
        self.start_ms = start_ms
        self.get_calls: list[dict[str, Any]] = []
        self.post_calls: list[dict[str, Any]] = []

    def get(self, url: str, *, params: dict[str, Any], timeout: float) -> FakeResponse:
        self.get_calls.append({"url": url, "params": params, "timeout": timeout})
        if url.endswith(funding.BINANCE_PREMIUM_INDEX_PATH):
            return FakeResponse(
                {
                    "symbol": "SKHYNIXUSDT",
                    "lastFundingRate": "0.00220000",
                    "nextFundingTime": str(self.start_ms + 8 * 60 * 60 * 1000),
                    "time": str(self.start_ms),
                    "markPrice": "1289.7",
                    "indexPrice": "1288.1",
                    "interestRate": "0.00010000",
                }
            )
        if url.endswith(funding.BINANCE_FUNDING_HISTORY_PATH):
            return FakeResponse(
                [
                    {
                        "symbol": "SKHYNIXUSDT",
                        "fundingTime": self.start_ms + 1,
                        "fundingRate": "0.00100000",
                        "markPrice": "1200",
                    },
                    {
                        "symbol": "SKHYNIXUSDT",
                        "fundingTime": self.start_ms + 8 * 60 * 60 * 1000,
                        "fundingRate": "-0.00050000",
                        "markPrice": "1210",
                    },
                ]
            )
        raise AssertionError(f"unexpected GET {url}")

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        if json["type"] == "metaAndAssetCtxs":
            return FakeResponse(
                [
                    {
                        "universe": [
                            {"name": "xyz:CL", "szDecimals": 3, "maxLeverage": 20},
                            {"name": "xyz:SKHX", "szDecimals": 3, "maxLeverage": 10},
                        ]
                    },
                    [
                        {"funding": "0.00001000", "markPx": "78.0"},
                        {
                            "funding": "0.00070000",
                            "premium": "0.0065",
                            "markPx": "1338.1",
                            "midPx": "1337.9",
                            "oraclePx": "1329.2",
                            "impactPxs": ["1337.7", "1338.1"],
                            "openInterest": "470000",
                        },
                    ],
                ]
            )
        if json["type"] == "fundingHistory":
            return FakeResponse(
                [
                    {
                        "coin": "xyz:SKHX",
                        "time": self.start_ms + 1,
                        "fundingRate": "0.00020000",
                        "premium": "0.001",
                    },
                    {
                        "coin": "xyz:SKHX",
                        "time": self.start_ms + 60 * 60 * 1000,
                        "fundingRate": "0.00040000",
                        "premium": "0.002",
                    },
                ]
            )
        raise AssertionError(f"unexpected POST {json}")


class FakeWs:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.closed = False
        self.sent: list[str] = []

    def settimeout(self, _timeout: float) -> None:
        return None

    def send(self, text: str) -> None:
        self.sent.append(text)

    def recv(self) -> str:
        import json

        return json.dumps(self.payload)

    def close(self) -> None:
        self.closed = True


def test_resolve_symbols_defaults_to_skhynix_profile() -> None:
    symbols = funding.resolve_symbols(symbol_profile="skhynix", binance_symbol="", hyperliquid_coin="")

    assert symbols["binance_symbol"] == "SKHYNIXUSDT"
    assert symbols["hyperliquid_coin"] == "xyz:SKHX"


def test_fetch_cross_exchange_funding_normalizes_current_history_and_net_summary() -> None:
    start_ms = 1_800_000_000_000
    end_ms = start_ms + 24 * 60 * 60 * 1000
    fake = FakeHttp(start_ms=start_ms)

    result = funding.fetch_cross_exchange_funding(
        symbol_profile="skhynix",
        start_ms=start_ms,
        end_ms=end_ms,
        notional=100_000.0,
        binance_current_source="rest",
        get=fake.get,
        post=fake.post,
    )

    assert result["symbols"]["binance_symbol"] == "SKHYNIXUSDT"
    assert result["symbols"]["hyperliquid_coin"] == "xyz:SKHX"
    assert result["venues"]["binance"]["current"]["funding_bps"] == 22.0
    assert result["venues"]["hyperliquid"]["current"]["funding_bps"] == 7.0
    assert result["venues"]["binance"]["summary"]["sum_funding_bps"] == 5.0
    assert round(result["venues"]["hyperliquid"]["summary"]["sum_funding_bps"], 10) == 6.0
    assert round(result["net_summary"]["long_binance_short_hyperliquid_bps"], 10) == 1.0
    assert round(result["net_summary"]["long_binance_short_hyperliquid_pnl"], 10) == 10.0
    assert result["boundary"]["no_private_keys"] is True
    assert result["boundary"]["no_order_endpoints"] is True
    assert fake.post_calls[0]["json"] == {"type": "metaAndAssetCtxs", "dex": "xyz"}
    assert fake.post_calls[1]["json"]["type"] == "fundingHistory"
    assert fake.post_calls[1]["json"]["coin"] == "xyz:SKHX"


def test_fetch_binance_current_funding_ws_uses_mark_price_stream() -> None:
    ws_payload = {
        "e": "markPriceUpdate",
        "E": 1_800_000_000_000,
        "s": "SKHYNIXUSDT",
        "p": "1289.7",
        "i": "1288.1",
        "P": "1288.5",
        "r": "0.00220000",
        "T": 1_800_028_800_000,
    }
    seen: dict[str, Any] = {}

    def fake_connect(url: str, timeout: float) -> FakeWs:
        seen["url"] = url
        seen["timeout"] = timeout
        return FakeWs(ws_payload)

    current, status = funding.fetch_binance_current_funding_ws(
        symbol="SKHYNIXUSDT",
        timeout=3.0,
        ws_connect=fake_connect,
    )

    assert status == "message_received"
    assert seen["url"] == funding.DEFAULT_BINANCE_WS_URL
    assert current["funding_rate"] == "0.00220000"
    assert current["funding_bps"] == 22.0
    assert current["source"] == "binance_mark_price_websocket"
    assert current["stream"] == "skhynixusdt@markPrice@1s"


def test_parse_time_accepts_epoch_ms_and_utc_iso() -> None:
    assert funding.parse_time_ms("1800000000000") == 1_800_000_000_000
    assert funding.parse_time_ms("2027-01-15T08:00:00Z") == 1_800_000_000_000
