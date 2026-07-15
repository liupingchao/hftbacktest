#!/usr/bin/env python3
"""Fetch public Binance/Hyperliquid funding-rate snapshots and history.

The module is intentionally public-data scoped. It never reads credentials and
never calls private, account, or order endpoints. Use
``fetch_cross_exchange_funding`` from other scripts, or run this file directly
for a JSON snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

try:
    from cross_exchange_symbol_registry import available_profile_ids, get_symbol_profile
except ModuleNotFoundError:  # pragma: no cover - package import path
    from examples.hyperliquid.cross_exchange_symbol_registry import (
        available_profile_ids,
        get_symbol_profile,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0714T005"
SCHEMA_VERSION = "cross_exchange_funding_rates_v1"
DEFAULT_SYMBOL_PROFILE = "skhynix"
DEFAULT_LOOKBACK_HOURS = 24.0
DEFAULT_REQUEST_TIMEOUT = 15.0
DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BASE_DELAY = 1.0
DEFAULT_RETRY_MAX_DELAY = 8.0
DEFAULT_BINANCE_CURRENT_SOURCE = "auto"
DEFAULT_WEBSOCKET_TIMEOUT = 5.0
DEFAULT_BINANCE_BASE_URL = "https://fapi.binance.com"
DEFAULT_BINANCE_WS_URL = "wss://fstream.binance.com/ws"
DEFAULT_HYPERLIQUID_INFO_URL = "https://api.hyperliquid.xyz/info"
BINANCE_FUNDING_HISTORY_PATH = "/fapi/v1/fundingRate"
BINANCE_PREMIUM_INDEX_PATH = "/fapi/v1/premiumIndex"
OFFICIAL_REFERENCES = [
    "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History",
    "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Mark-Price",
    "https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Mark-Price-Stream",
    "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint/perpetuals",
]
PUBLIC_BOUNDARY = {
    "no_private_keys": True,
    "no_private_account_endpoints": True,
    "no_order_endpoints": True,
    "no_order_lifecycle": True,
    "no_strategy_process": True,
    "no_live_trading_bot": True,
}


HttpGet = Callable[..., Any]
HttpPost = Callable[..., Any]
WsConnect = Callable[..., Any]
RETRYABLE_HTTP_STATUSES = {418, 429, 500, 502, 503, 504}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_now_ms() -> int:
    return int(time.time() * 1000)


def iso_from_ms(timestamp_ms: int | None) -> str:
    if timestamp_ms is None:
        return ""
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc).isoformat()


def parse_time_ms(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def rate_to_bps(rate: float | str | None) -> float | None:
    value = float_or_none(rate)
    if value is None:
        return None
    return value * 10_000.0


def float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _json_response(response: Any) -> Any:
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    return response.json()


def _http_status(response: Any) -> int:
    return int(getattr(response, "status_code", 0) or 0)


def _retry_after_seconds(response: Any) -> float | None:
    headers = getattr(response, "headers", {}) or {}
    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None
    parsed = float_or_none(value)
    return parsed if parsed is not None and parsed >= 0 else None


def _request_with_retries(
    request: Callable[[], Any],
    *,
    retry_attempts: int,
    retry_base_delay: float,
    retry_max_delay: float,
) -> Any:
    attempts = max(1, retry_attempts)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            response = request()
        except requests.RequestException as exc:
            last_exc = exc
            if attempt >= attempts:
                raise
        else:
            status = _http_status(response)
            if status not in RETRYABLE_HTTP_STATUSES or attempt >= attempts:
                return response
            retry_after = _retry_after_seconds(response)
            delay = retry_after if retry_after is not None else retry_base_delay * (2 ** (attempt - 1))
            time.sleep(min(max(0.0, delay), retry_max_delay))
            continue
        delay = retry_base_delay * (2 ** (attempt - 1))
        time.sleep(min(max(0.0, delay), retry_max_delay))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("request retry loop ended without response")


def _binance_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _dex_from_coin(coin: str) -> str:
    return coin.split(":", 1)[0] if ":" in coin else ""


def _optional_dex_body(body: dict[str, Any], dex: str | None) -> dict[str, Any]:
    if dex:
        return {**body, "dex": dex}
    return body


def resolve_symbols(
    *,
    symbol_profile: str = DEFAULT_SYMBOL_PROFILE,
    binance_symbol: str = "",
    hyperliquid_coin: str = "",
) -> dict[str, str]:
    profile = get_symbol_profile(symbol_profile)
    resolved_binance = (binance_symbol or profile.binance_symbol).strip().upper()
    resolved_hl = (hyperliquid_coin or profile.hyperliquid_coin).strip()
    return {
        "symbol_profile": profile.profile_id,
        "binance_symbol": resolved_binance,
        "hyperliquid_coin": resolved_hl,
        "profile_binance_symbol": profile.binance_symbol,
        "profile_hyperliquid_coin": profile.hyperliquid_coin,
    }


def normalize_binance_current(payload: dict[str, Any], *, symbol: str) -> dict[str, Any]:
    rate = payload.get("lastFundingRate", payload.get("fundingRate", payload.get("r", "")))
    next_value = payload.get("nextFundingTime", payload.get("T", ""))
    event_value = payload.get("time", payload.get("E", ""))
    next_funding_ms = int(next_value) if next_value not in (None, "") else None
    event_ms = int(event_value) if event_value not in (None, "") else None
    return {
        "exchange": "binance_usdm",
        "symbol": str(payload.get("symbol") or payload.get("s") or symbol),
        "funding_rate": str(rate),
        "funding_bps": rate_to_bps(rate),
        "time_ms": event_ms,
        "time_utc": iso_from_ms(event_ms),
        "next_funding_time_ms": next_funding_ms,
        "next_funding_time_utc": iso_from_ms(next_funding_ms),
        "mark_price": str(payload.get("markPrice", payload.get("p", ""))),
        "index_price": str(payload.get("indexPrice", payload.get("i", ""))),
        "estimated_settle_price": str(payload.get("estimatedSettlePrice", payload.get("P", ""))),
        "interest_rate": str(payload.get("interestRate", "")),
    }


def normalize_binance_history_row(row: dict[str, Any], *, symbol: str) -> dict[str, Any]:
    timestamp_ms = int(row["fundingTime"])
    rate = str(row.get("fundingRate", ""))
    return {
        "exchange": "binance_usdm",
        "symbol": str(row.get("symbol") or symbol),
        "time_ms": timestamp_ms,
        "time_utc": iso_from_ms(timestamp_ms),
        "funding_rate": rate,
        "funding_bps": rate_to_bps(rate),
        "mark_price": str(row.get("markPrice", "")),
    }


def normalize_hyperliquid_current(
    meta_item: dict[str, Any],
    ctx: dict[str, Any],
    *,
    coin: str,
    dex: str,
) -> dict[str, Any]:
    rate = str(ctx.get("funding", ""))
    return {
        "exchange": "hyperliquid",
        "coin": str(meta_item.get("name") or coin),
        "dex": dex,
        "funding_rate": rate,
        "funding_bps": rate_to_bps(rate),
        "premium": str(ctx.get("premium", "")),
        "mark_price": str(ctx.get("markPx", "")),
        "mid_price": str(ctx.get("midPx", "")),
        "oracle_price": str(ctx.get("oraclePx", "")),
        "impact_prices": ctx.get("impactPxs", []),
        "open_interest": str(ctx.get("openInterest", "")),
        "day_base_volume": str(ctx.get("dayBaseVlm", "")),
        "day_notional_volume": str(ctx.get("dayNtlVlm", "")),
        "previous_day_price": str(ctx.get("prevDayPx", "")),
        "sz_decimals": meta_item.get("szDecimals"),
        "max_leverage": meta_item.get("maxLeverage"),
    }


def normalize_hyperliquid_history_row(row: dict[str, Any], *, coin: str) -> dict[str, Any]:
    timestamp_ms = int(row["time"])
    rate = str(row.get("fundingRate", ""))
    return {
        "exchange": "hyperliquid",
        "coin": str(row.get("coin") or coin),
        "time_ms": timestamp_ms,
        "time_utc": iso_from_ms(timestamp_ms),
        "funding_rate": rate,
        "funding_bps": rate_to_bps(rate),
        "premium": str(row.get("premium", "")),
    }


def fetch_binance_current_funding_rest(
    *,
    symbol: str,
    base_url: str = DEFAULT_BINANCE_BASE_URL,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    get: HttpGet = requests.get,
) -> tuple[dict[str, Any], int]:
    response = _request_with_retries(
        lambda: get(
            _binance_url(base_url, BINANCE_PREMIUM_INDEX_PATH),
            params={"symbol": symbol},
            timeout=timeout,
        ),
        retry_attempts=retry_attempts,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    payload = _json_response(response)
    return normalize_binance_current(payload, symbol=symbol), _http_status(response)


def default_ws_connect(url: str, timeout: float) -> Any:
    try:
        import websocket
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("Python package 'websocket-client' is required for Binance mark-price WS current funding.") from exc
    return websocket.create_connection(url, timeout=timeout)


def fetch_binance_current_funding_ws(
    *,
    symbol: str,
    ws_url: str = DEFAULT_BINANCE_WS_URL,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ws_connect: WsConnect = default_ws_connect,
) -> tuple[dict[str, Any], str]:
    stream = f"{symbol.lower()}@markPrice@1s"
    stream_url = ws_url.rstrip("/")
    ws = ws_connect(stream_url, timeout=timeout)
    deadline = time.monotonic() + timeout
    try:
        if hasattr(ws, "settimeout"):
            ws.settimeout(timeout)
        if hasattr(ws, "send"):
            ws.send(json.dumps({"method": "SUBSCRIBE", "params": [stream], "id": 1}))
        while time.monotonic() < deadline:
            text = ws.recv()
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            message = json.loads(str(text))
            data = message.get("data", message) if isinstance(message, dict) else {}
            if isinstance(data, dict) and (data.get("e") == "markPriceUpdate" or "r" in data):
                current = normalize_binance_current(data, symbol=symbol)
                current["source"] = "binance_mark_price_websocket"
                current["stream_url"] = stream_url
                current["stream"] = stream
                return current, "message_received"
        raise TimeoutError(f"no Binance mark-price update received for {symbol} within {timeout}s")
    finally:
        if hasattr(ws, "close"):
            ws.close()


def fetch_binance_funding_history(
    *,
    symbol: str,
    start_ms: int,
    end_ms: int,
    base_url: str = DEFAULT_BINANCE_BASE_URL,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    limit: int = 1000,
    max_pages: int = 20,
    get: HttpGet = requests.get,
) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    statuses: list[int] = []
    seen_times: set[int] = set()
    cursor = start_ms
    for _page in range(max_pages):
        response = _request_with_retries(
            lambda: get(
                _binance_url(base_url, BINANCE_FUNDING_HISTORY_PATH),
                params={"symbol": symbol, "startTime": cursor, "endTime": end_ms, "limit": limit},
                timeout=timeout,
            ),
            retry_attempts=retry_attempts,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
        )
        statuses.append(_http_status(response))
        payload = _json_response(response)
        if not isinstance(payload, list) or not payload:
            break
        page_rows = sorted(payload, key=lambda item: int(item["fundingTime"]))
        for row in page_rows:
            timestamp_ms = int(row["fundingTime"])
            if timestamp_ms < start_ms or timestamp_ms > end_ms or timestamp_ms in seen_times:
                continue
            seen_times.add(timestamp_ms)
            rows.append(normalize_binance_history_row(row, symbol=symbol))
        last_ms = int(page_rows[-1]["fundingTime"])
        next_cursor = last_ms + 1
        if len(page_rows) < limit or next_cursor <= cursor or next_cursor > end_ms:
            break
        cursor = next_cursor
    return sorted(rows, key=lambda row: row["time_ms"]), statuses


def fetch_hyperliquid_current_funding(
    *,
    coin: str,
    dex: str | None = None,
    info_url: str = DEFAULT_HYPERLIQUID_INFO_URL,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    post: HttpPost = requests.post,
) -> tuple[dict[str, Any], int]:
    resolved_dex = _dex_from_coin(coin) if dex is None else dex
    body = _optional_dex_body({"type": "metaAndAssetCtxs"}, resolved_dex)
    response = _request_with_retries(
        lambda: post(info_url, json=body, timeout=timeout),
        retry_attempts=retry_attempts,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
    )
    payload = _json_response(response)
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("unexpected Hyperliquid metaAndAssetCtxs payload")
    meta, ctxs = payload
    universe = meta.get("universe", []) if isinstance(meta, dict) else []
    for item, ctx in zip(universe, ctxs):
        if item.get("name") == coin:
            return normalize_hyperliquid_current(item, ctx, coin=coin, dex=resolved_dex), _http_status(response)
    raise ValueError(f"Hyperliquid coin {coin!r} not found in metaAndAssetCtxs dex={resolved_dex!r}")


def fetch_hyperliquid_funding_history(
    *,
    coin: str,
    start_ms: int,
    end_ms: int,
    info_url: str = DEFAULT_HYPERLIQUID_INFO_URL,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    max_pages: int = 20,
    post: HttpPost = requests.post,
) -> tuple[list[dict[str, Any]], list[int]]:
    rows: list[dict[str, Any]] = []
    statuses: list[int] = []
    seen_times: set[int] = set()
    cursor = start_ms
    for _page in range(max_pages):
        response = _request_with_retries(
            lambda: post(
                info_url,
                json={"type": "fundingHistory", "coin": coin, "startTime": cursor, "endTime": end_ms},
                timeout=timeout,
            ),
            retry_attempts=retry_attempts,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
        )
        statuses.append(_http_status(response))
        payload = _json_response(response)
        if not isinstance(payload, list) or not payload:
            break
        page_rows = sorted(payload, key=lambda item: int(item["time"]))
        for row in page_rows:
            timestamp_ms = int(row["time"])
            if timestamp_ms < start_ms or timestamp_ms > end_ms or timestamp_ms in seen_times:
                continue
            seen_times.add(timestamp_ms)
            rows.append(normalize_hyperliquid_history_row(row, coin=coin))
        last_ms = int(page_rows[-1]["time"])
        next_cursor = last_ms + 1
        if len(page_rows) < 500 or next_cursor <= cursor or next_cursor > end_ms:
            break
        cursor = next_cursor
    return sorted(rows, key=lambda row: row["time_ms"]), statuses


def summarize_history(
    rows: list[dict[str, Any]],
    *,
    start_ms: int,
    end_ms: int,
    notional: float | None = None,
) -> dict[str, Any]:
    rates = [float_or_none(row.get("funding_rate")) or 0.0 for row in rows]
    total = sum(rates)
    window_days = max((end_ms - start_ms) / 86_400_000.0, 0.0)
    simple_apr = total * 365.0 / window_days if window_days > 0 else None
    summary = {
        "row_count": len(rows),
        "sum_funding_rate": total,
        "sum_funding_bps": total * 10_000.0,
        "window_days": window_days,
        "simple_apr": simple_apr,
        "simple_apr_percent": simple_apr * 100.0 if simple_apr is not None else None,
        "latest": rows[-1] if rows else None,
    }
    if notional is not None:
        summary["funding_pnl_for_long_position"] = -notional * total
        summary["funding_pnl_for_short_position"] = notional * total
    return summary


def build_net_summary(
    *,
    binance_summary: dict[str, Any],
    hyperliquid_summary: dict[str, Any],
    notional: float | None,
) -> dict[str, Any]:
    binance_total = float(binance_summary.get("sum_funding_rate", 0.0) or 0.0)
    hl_total = float(hyperliquid_summary.get("sum_funding_rate", 0.0) or 0.0)
    long_binance_short_hl = hl_total - binance_total
    short_binance_long_hl = binance_total - hl_total
    result = {
        "long_binance_short_hyperliquid_funding_rate": long_binance_short_hl,
        "long_binance_short_hyperliquid_bps": long_binance_short_hl * 10_000.0,
        "short_binance_long_hyperliquid_funding_rate": short_binance_long_hl,
        "short_binance_long_hyperliquid_bps": short_binance_long_hl * 10_000.0,
        "sign_convention": (
            "Positive funding normally means longs pay shorts. Net fields are funding-only "
            "diagnostics over the selected history window."
        ),
    }
    if notional is not None:
        result["long_binance_short_hyperliquid_pnl"] = notional * long_binance_short_hl
        result["short_binance_long_hyperliquid_pnl"] = notional * short_binance_long_hl
        result["notional"] = notional
    return result


def fetch_cross_exchange_funding(
    *,
    symbol_profile: str = DEFAULT_SYMBOL_PROFILE,
    binance_symbol: str = "",
    hyperliquid_coin: str = "",
    hyperliquid_dex: str | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    lookback_hours: float = DEFAULT_LOOKBACK_HOURS,
    notional: float | None = None,
    binance_base_url: str = DEFAULT_BINANCE_BASE_URL,
    binance_ws_url: str = DEFAULT_BINANCE_WS_URL,
    binance_current_source: str = DEFAULT_BINANCE_CURRENT_SOURCE,
    hyperliquid_info_url: str = DEFAULT_HYPERLIQUID_INFO_URL,
    timeout: float = DEFAULT_REQUEST_TIMEOUT,
    websocket_timeout: float = DEFAULT_WEBSOCKET_TIMEOUT,
    retry_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY,
    retry_max_delay: float = DEFAULT_RETRY_MAX_DELAY,
    binance_limit: int = 1000,
    max_pages: int = 20,
    get: HttpGet = requests.get,
    post: HttpPost = requests.post,
    ws_connect: WsConnect = default_ws_connect,
) -> dict[str, Any]:
    resolved = resolve_symbols(
        symbol_profile=symbol_profile,
        binance_symbol=binance_symbol,
        hyperliquid_coin=hyperliquid_coin,
    )
    now_ms = utc_now_ms()
    resolved_end_ms = end_ms if end_ms is not None else now_ms
    resolved_start_ms = start_ms if start_ms is not None else int(resolved_end_ms - lookback_hours * 60 * 60 * 1000)
    if resolved_start_ms >= resolved_end_ms:
        raise ValueError("start_ms must be earlier than end_ms")

    if binance_current_source == "auto":
        current_attempts: list[dict[str, Any]] = []
        try:
            binance_current, binance_current_status = fetch_binance_current_funding_ws(
                symbol=resolved["binance_symbol"],
                ws_url=binance_ws_url,
                timeout=websocket_timeout,
                ws_connect=ws_connect,
            )
            current_attempts.append(
                {
                    "transport": "websocket",
                    "url": binance_current.get("stream_url", ""),
                    "stream": binance_current.get("stream", ""),
                    "status": binance_current_status,
                }
            )
        except Exception as ws_exc:
            current_attempts.append(
                {
                    "transport": "websocket",
                    "url": binance_ws_url.rstrip("/"),
                    "stream": f"{resolved['binance_symbol'].lower()}@markPrice@1s",
                    "status": "error",
                    "error": str(ws_exc),
                }
            )
            try:
                binance_current, binance_current_status = fetch_binance_current_funding_rest(
                    symbol=resolved["binance_symbol"],
                    base_url=binance_base_url,
                    timeout=timeout,
                    retry_attempts=retry_attempts,
                    retry_base_delay=retry_base_delay,
                    retry_max_delay=retry_max_delay,
                    get=get,
                )
                binance_current["source"] = "binance_premium_index_rest"
                current_attempts.append(
                    {
                        "transport": "rest",
                        "method": "GET",
                        "url": _binance_url(binance_base_url, BINANCE_PREMIUM_INDEX_PATH),
                        "http_status": binance_current_status,
                    }
                )
            except Exception as rest_exc:
                binance_current = {
                    "exchange": "binance_usdm",
                    "symbol": resolved["binance_symbol"],
                    "source": "unavailable",
                    "funding_rate": "",
                    "funding_bps": None,
                    "errors": current_attempts
                    + [
                        {
                            "transport": "rest",
                            "method": "GET",
                            "url": _binance_url(binance_base_url, BINANCE_PREMIUM_INDEX_PATH),
                            "status": "error",
                            "error": str(rest_exc),
                        }
                    ],
                }
                current_attempts = list(binance_current["errors"])
        binance_current_api = {
            "transport": "auto",
            "status": "available" if binance_current.get("source") != "unavailable" else "unavailable",
            "attempts": current_attempts,
        }
    elif binance_current_source == "ws":
        binance_current, binance_current_status = fetch_binance_current_funding_ws(
            symbol=resolved["binance_symbol"],
            ws_url=binance_ws_url,
            timeout=websocket_timeout,
            ws_connect=ws_connect,
        )
        binance_current_api = {
            "transport": "websocket",
            "url": binance_current.get("stream_url", ""),
            "status": binance_current_status,
        }
    elif binance_current_source == "rest":
        binance_current, binance_current_status = fetch_binance_current_funding_rest(
            symbol=resolved["binance_symbol"],
            base_url=binance_base_url,
            timeout=timeout,
            retry_attempts=retry_attempts,
            retry_base_delay=retry_base_delay,
            retry_max_delay=retry_max_delay,
            get=get,
        )
        binance_current["source"] = "binance_premium_index_rest"
        binance_current_api = {
            "transport": "rest",
            "method": "GET",
            "url": _binance_url(binance_base_url, BINANCE_PREMIUM_INDEX_PATH),
            "http_status": binance_current_status,
        }
    elif binance_current_source == "skip":
        binance_current = {
            "exchange": "binance_usdm",
            "symbol": resolved["binance_symbol"],
            "source": "skipped",
            "funding_rate": "",
            "funding_bps": None,
        }
        binance_current_api = {"transport": "none", "status": "skipped"}
    else:
        raise ValueError(f"unsupported binance_current_source {binance_current_source!r}")

    binance_history, binance_history_statuses = fetch_binance_funding_history(
        symbol=resolved["binance_symbol"],
        start_ms=resolved_start_ms,
        end_ms=resolved_end_ms,
        base_url=binance_base_url,
        timeout=timeout,
        retry_attempts=retry_attempts,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
        limit=binance_limit,
        max_pages=max_pages,
        get=get,
    )
    hl_current, hl_current_status = fetch_hyperliquid_current_funding(
        coin=resolved["hyperliquid_coin"],
        dex=hyperliquid_dex,
        info_url=hyperliquid_info_url,
        timeout=timeout,
        retry_attempts=retry_attempts,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
        post=post,
    )
    hl_history, hl_history_statuses = fetch_hyperliquid_funding_history(
        coin=resolved["hyperliquid_coin"],
        start_ms=resolved_start_ms,
        end_ms=resolved_end_ms,
        info_url=hyperliquid_info_url,
        timeout=timeout,
        retry_attempts=retry_attempts,
        retry_base_delay=retry_base_delay,
        retry_max_delay=retry_max_delay,
        max_pages=max_pages,
        post=post,
    )

    binance_summary = summarize_history(
        binance_history,
        start_ms=resolved_start_ms,
        end_ms=resolved_end_ms,
        notional=notional,
    )
    hl_summary = summarize_history(
        hl_history,
        start_ms=resolved_start_ms,
        end_ms=resolved_end_ms,
        notional=notional,
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "git_commit": _git_commit(),
        "queried_at_utc": utc_now(),
        "window": {
            "start_ms": resolved_start_ms,
            "end_ms": resolved_end_ms,
            "start_utc": iso_from_ms(resolved_start_ms),
            "end_utc": iso_from_ms(resolved_end_ms),
            "lookback_hours": (resolved_end_ms - resolved_start_ms) / 3_600_000.0,
        },
        "symbols": resolved,
        "api": {
            "binance_current": {
                **binance_current_api,
            },
            "binance_history": {
                "method": "GET",
                "url": _binance_url(binance_base_url, BINANCE_FUNDING_HISTORY_PATH),
                "http_statuses": binance_history_statuses,
            },
            "hyperliquid_current": {
                "method": "POST",
                "url": hyperliquid_info_url,
                "body_type": "metaAndAssetCtxs",
                "dex": hl_current.get("dex", ""),
                "http_status": hl_current_status,
            },
            "hyperliquid_history": {
                "method": "POST",
                "url": hyperliquid_info_url,
                "body_type": "fundingHistory",
                "http_statuses": hl_history_statuses,
            },
            "official_references": OFFICIAL_REFERENCES,
        },
        "venues": {
            "binance": {
                "exchange": "binance_usdm",
                "symbol": resolved["binance_symbol"],
                "current": binance_current,
                "history": binance_history,
                "summary": binance_summary,
            },
            "hyperliquid": {
                "exchange": "hyperliquid",
                "coin": resolved["hyperliquid_coin"],
                "current": hl_current,
                "history": hl_history,
                "summary": hl_summary,
            },
        },
        "net_summary": build_net_summary(
            binance_summary=binance_summary,
            hyperliquid_summary=hl_summary,
            notional=notional,
        ),
        "boundary": PUBLIC_BOUNDARY,
    }


def write_json(path: Path, payload: dict[str, Any], *, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2 if pretty else None, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch public Binance/Hyperliquid current and historical funding rates."
    )
    parser.add_argument("--symbol-profile", choices=available_profile_ids(), default=DEFAULT_SYMBOL_PROFILE)
    parser.add_argument("--binance-symbol", default="", help="Override Binance USD-M symbol.")
    parser.add_argument("--hyperliquid-coin", default="", help="Override Hyperliquid coin, e.g. xyz:SKHX.")
    parser.add_argument(
        "--hyperliquid-dex",
        default=None,
        help="Override Hyperliquid dex for current metaAndAssetCtxs; defaults to coin prefix before ':'.",
    )
    parser.add_argument("--lookback-hours", type=float, default=DEFAULT_LOOKBACK_HOURS)
    parser.add_argument("--start-time", default="", help="UTC ISO timestamp or epoch ms for history start.")
    parser.add_argument("--end-time", default="", help="UTC ISO timestamp or epoch ms for history end.")
    parser.add_argument("--notional", type=float, default=None, help="Optional notional for funding-only PnL math.")
    parser.add_argument(
        "--binance-current-source",
        choices=["auto", "ws", "rest", "skip"],
        default=DEFAULT_BINANCE_CURRENT_SOURCE,
        help="auto tries Binance mark-price WS first, then REST premiumIndex if the stream is unavailable.",
    )
    parser.add_argument("--binance-base-url", default=DEFAULT_BINANCE_BASE_URL)
    parser.add_argument("--binance-ws-url", default=DEFAULT_BINANCE_WS_URL)
    parser.add_argument("--hyperliquid-info-url", default=DEFAULT_HYPERLIQUID_INFO_URL)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--websocket-timeout", type=float, default=DEFAULT_WEBSOCKET_TIMEOUT)
    parser.add_argument("--retry-attempts", type=int, default=DEFAULT_RETRY_ATTEMPTS)
    parser.add_argument("--retry-base-delay", type=float, default=DEFAULT_RETRY_BASE_DELAY)
    parser.add_argument("--retry-max-delay", type=float, default=DEFAULT_RETRY_MAX_DELAY)
    parser.add_argument("--binance-limit", type=int, default=1000)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = fetch_cross_exchange_funding(
        symbol_profile=args.symbol_profile,
        binance_symbol=args.binance_symbol,
        hyperliquid_coin=args.hyperliquid_coin,
        hyperliquid_dex=args.hyperliquid_dex,
        start_ms=parse_time_ms(args.start_time),
        end_ms=parse_time_ms(args.end_time),
        lookback_hours=args.lookback_hours,
        notional=args.notional,
        binance_base_url=args.binance_base_url,
        binance_ws_url=args.binance_ws_url,
        binance_current_source=args.binance_current_source,
        hyperliquid_info_url=args.hyperliquid_info_url,
        timeout=args.request_timeout,
        websocket_timeout=args.websocket_timeout,
        retry_attempts=args.retry_attempts,
        retry_base_delay=args.retry_base_delay,
        retry_max_delay=args.retry_max_delay,
        binance_limit=args.binance_limit,
        max_pages=args.max_pages,
    )
    text = json.dumps(payload, indent=2 if args.pretty else None, sort_keys=True)
    if args.output_json:
        write_json(Path(args.output_json).expanduser().resolve(), payload, pretty=args.pretty)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
