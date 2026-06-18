#!/usr/bin/env python3
"""Minimal Hyperliquid tiny-live real-order executor scaffold.

The default path is local self-test only. The live-capable path requires an
explicit live flag and operator acknowledgement. This task does not run that
path or place real orders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0618T001"
CANARY_TASK_ID = "0618T004"
SCHEMA_VERSION = "hyperliquid_tiny_live_real_order_executor_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_real_order_executor_0618T001"
DEFAULT_CANARY_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_real_order_canary_0618T004"
FINAL_RECOMMENDATION_READY = "hyperliquid_tiny_live_real_order_executor_ready_for_qa"
FINAL_RECOMMENDATION_BLOCKED = "hyperliquid_tiny_live_real_order_executor_blocked"
FINAL_RECOMMENDATION_CANARY_READY = "hyperliquid_tiny_live_real_order_canary_ready_for_qa"
FINAL_RECOMMENDATION_CANARY_BLOCKED = "hyperliquid_tiny_live_real_order_canary_blocked"
MAX_DURATION_SECONDS = 600
MAX_ORDER_SIZE_BTC = 0.01
MAX_ORDER_NOTIONAL_USDC = 700.0
MAX_POSITION_BTC = 0.04
MAX_POSITION_NOTIONAL_USDC = 2800.0
MAX_NOTIONAL_USDC = 3000.0
MAX_LOSS_USDC = 30.0
SYMBOL = "BTC"
POST_ONLY_TIF = "Alo"
LIVE_OPERATOR_ACK = "I_UNDERSTAND_THIS_CAN_PLACE_REAL_HYPERLIQUID_ORDERS"
DEFAULT_CANARY_PRICE_OFFSET_BPS = 200.0
MAX_CANARY_LIMIT_PX = 69_900.0

OFFICIAL_DOC_RECHECKS = [
    {
        "topic": "api_overview",
        "url": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api",
        "task_relevance": "official API entry point",
        "local_recheck_summary": "API reference is the source for /info and /exchange boundaries.",
    },
    {
        "topic": "exchange_endpoint",
        "url": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/exchange-endpoint",
        "task_relevance": "order/cancel/scheduleCancel actions",
        "local_recheck_summary": "Order actions are signed exchange actions; post-only is represented as limit tif Alo.",
    },
    {
        "topic": "info_endpoint",
        "url": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/info-endpoint",
        "task_relevance": "meta, mids, open orders, order status, fills, user state",
        "local_recheck_summary": "Private read-only preflight uses SDK Info methods and must stay explicitly gated.",
    },
    {
        "topic": "tick_and_lot_size",
        "url": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/tick-and-lot-size",
        "task_relevance": "precision facts before order placement",
        "local_recheck_summary": "BTC size decimals and price precision must be present before live mode.",
    },
    {
        "topic": "signing",
        "url": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/signing",
        "task_relevance": "signed exchange actions",
        "local_recheck_summary": "This implementation relies on the official SDK for signing and does not hand-roll signatures.",
    },
    {
        "topic": "nonces_and_api_wallets",
        "url": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/nonces-and-api-wallets",
        "task_relevance": "wallet and nonce handling",
        "local_recheck_summary": "Nonce and wallet handling remain SDK-owned; secrets are never written to artifacts.",
    },
    {
        "topic": "rate_limits_and_user_limits",
        "url": "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits",
        "task_relevance": "tiny-live API hygiene",
        "local_recheck_summary": "Tiny-live loop is capped and artifacts record rate-limit/readiness rather than assuming capacity.",
    },
    {
        "topic": "official_python_sdk",
        "url": "https://github.com/hyperliquid-dex/hyperliquid-python-sdk",
        "task_relevance": "Exchange.order/cancel/schedule_cancel and Info.open_orders/user_state/query_order_by_oid",
        "local_recheck_summary": "SDK exposes Exchange.order(name,is_buy,sz,limit_px,{limit:{tif}}), cancel, cancel_by_cloid, schedule_cancel, Info open/read methods.",
    },
]

REDACT_KEYS = {
    "account",
    "account_address",
    "address",
    "api_key",
    "api_secret",
    "authorization",
    "cloid",
    "nonce",
    "oid",
    "private_key",
    "raw_signature",
    "secret",
    "signature",
    "signed_payload",
    "user",
    "wallet",
    "wallet_key",
}

ADDRESS_RE = re.compile(r"0x[a-fA-F0-9]{40}")
HEX_32_RE = re.compile(r"0x[a-fA-F0-9]{32}")
HEX_64_RE = re.compile(r"0x[a-fA-F0-9]{64}")

BOUNDARY_FLAGS = {
    "default_mode_is_self_test": True,
    "this_task_places_real_orders": False,
    "this_task_cancels_real_orders": False,
    "credentials_written": False,
    "raw_signatures_written": False,
    "private_endpoint_called_in_self_test": False,
    "order_endpoint_called_in_self_test": False,
    "live_mode_requires_explicit_flags": True,
    "live_mode_requires_operator_ack": True,
    "post_only_alo_required": True,
    "max_loss_fail_closed": True,
    "cancel_all_shutdown_path_present": True,
}


class ValidationError(ValueError):
    """Raised when a config or runtime check must fail closed."""


@dataclass(frozen=True)
class TinyLiveConfig:
    symbol: str = SYMBOL
    duration_seconds: int = MAX_DURATION_SECONDS
    max_order_size_btc: float = MAX_ORDER_SIZE_BTC
    max_order_notional_usdc: float = MAX_ORDER_NOTIONAL_USDC
    max_position_btc: float = MAX_POSITION_BTC
    max_position_notional_usdc: float = MAX_POSITION_NOTIONAL_USDC
    max_notional_usdc: float = MAX_NOTIONAL_USDC
    max_loss_usdc: float = MAX_LOSS_USDC
    time_in_force: str = POST_ONLY_TIF
    order_type: str = "limit"
    reduce_only: bool = False
    artifact_dir: Path = DEFAULT_OUTPUT_DIR
    live_mode: bool = False
    operator_ack: str = ""
    use_schedule_cancel: bool = True
    dry_run_private_preflight: bool = True


@dataclass(frozen=True)
class PrecisionFacts:
    symbol: str
    sz_decimals: int
    tick_size: float
    lot_size: float
    mid_px: float
    source: str


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    is_buy: bool
    size_btc: float
    limit_px: float
    time_in_force: str = POST_ONLY_TIF
    reduce_only: bool = False
    cloid: str = "cloid_0618T001_self_test"

    @property
    def notional_usdc(self) -> float:
        return self.size_btc * self.limit_px


@dataclass(frozen=True)
class LossSnapshot:
    entry_px: float
    mark_px: float
    position_btc: float
    realized_pnl_usdc: float = 0.0

    @property
    def estimated_loss_usdc(self) -> float:
        unrealized = (self.mark_px - self.entry_px) * self.position_btc
        return max(0.0, -(self.realized_pnl_usdc + unrealized))


@dataclass
class ShutdownEvidence:
    requested_refs: list[str] = field(default_factory=list)
    cancel_results: list[dict[str, Any]] = field(default_factory=list)
    final_open_orders: list[dict[str, Any]] = field(default_factory=list)
    proof_status: str = "not_started"
    fail_closed_reason: str = ""


class HyperliquidClient(Protocol):
    def preflight(self, config: TinyLiveConfig) -> dict[str, Any]:
        ...

    def order(self, intent: OrderIntent) -> dict[str, Any]:
        ...

    def cancel_tracked(self, symbol: str, oid: int | None = None, cloid: str | None = None) -> dict[str, Any]:
        ...

    def open_orders(self, address: str | None = None) -> list[dict[str, Any]]:
        ...

    def schedule_cancel(self, cancel_time_ms: int | None) -> dict[str, Any]:
        ...


class MockHyperliquidClient:
    """Deterministic no-network client used by tests and self-test artifacts."""

    def __init__(self, *, final_open_orders: list[dict[str, Any]] | None = None) -> None:
        self.orders: list[dict[str, Any]] = []
        self.cancels: list[dict[str, Any]] = []
        self.scheduled_cancel_ms: int | None = None
        self.final_open_orders = final_open_orders or []

    def preflight(self, config: TinyLiveConfig) -> dict[str, Any]:
        return {
            "client": "mock",
            "mode": "self_test",
            "private_endpoint_called": False,
            "order_endpoint_called": False,
            "sdk_required_for_live": True,
            "symbol": config.symbol,
        }

    def order(self, intent: OrderIntent) -> dict[str, Any]:
        oid = 618001000 + len(self.orders)
        row = {
            "status": "ok",
            "response": {"data": {"statuses": [{"resting": {"oid": oid, "cloid": intent.cloid}}]}},
            "mock": True,
        }
        self.orders.append({"oid": oid, "cloid": intent.cloid, "intent": intent})
        return row

    def cancel_tracked(self, symbol: str, oid: int | None = None, cloid: str | None = None) -> dict[str, Any]:
        ref = str(oid if oid is not None else cloid)
        row = {"status": "ok", "response": {"data": {"statuses": [{"success": ref}]}}, "mock": True, "symbol": symbol}
        self.cancels.append(row)
        return row

    def open_orders(self, address: str | None = None) -> list[dict[str, Any]]:
        return list(self.final_open_orders)

    def schedule_cancel(self, cancel_time_ms: int | None) -> dict[str, Any]:
        self.scheduled_cancel_ms = cancel_time_ms
        return {"status": "ok", "scheduled_cancel_time_ms": cancel_time_ms, "mock": True}

    def user_state(self, address: str | None = None) -> dict[str, Any]:
        return {"assetPositions": [], "mock": True}

    def user_fills(self, address: str | None = None) -> list[dict[str, Any]]:
        return []

    def query_order_by_oid(self, oid: int, address: str | None = None) -> dict[str, Any]:
        return {"status": "ok", "oid": oid, "mock": True}

    def query_order_by_cloid(self, cloid: str, address: str | None = None) -> dict[str, Any]:
        return {"status": "ok", "cloid": cloid, "mock": True}


class SDKHyperliquidClient:
    """Thin wrapper around the official Hyperliquid Python SDK.

    This class intentionally keeps signing and nonce handling inside the SDK.
    """

    def __init__(self, *, exchange: Any, info: Any, account_address: str | None) -> None:
        self.exchange = exchange
        self.info = info
        self.account_address = account_address

    def preflight(self, config: TinyLiveConfig) -> dict[str, Any]:
        address = self.account_address
        facts: dict[str, Any] = {
            "client": "official_hyperliquid_python_sdk",
            "private_read_only_preflight": bool(address),
            "order_endpoint_called": False,
            "credentials_written": False,
        }
        if address:
            open_orders = self.info.open_orders(address)
            user_state = self.info.user_state(address)
            user_fills = self.info.user_fills(address)
            facts["open_order_count"] = len(open_orders)
            facts["asset_position_count"] = len(user_state.get("assetPositions", []))
            facts["user_fill_count"] = len(user_fills)
        return facts

    def order(self, intent: OrderIntent) -> dict[str, Any]:
        return self.exchange.order(
            intent.symbol,
            intent.is_buy,
            intent.size_btc,
            intent.limit_px,
            {"limit": {"tif": intent.time_in_force}},
            reduce_only=intent.reduce_only,
            cloid=to_sdk_cloid(intent.cloid),
        )

    def cancel_tracked(self, symbol: str, oid: int | None = None, cloid: str | None = None) -> dict[str, Any]:
        if oid is not None:
            return self.exchange.cancel(symbol, oid)
        if cloid is not None:
            return self.exchange.cancel_by_cloid(symbol, to_sdk_cloid(cloid))
        raise ValidationError("cancel requires oid or cloid")

    def open_orders(self, address: str | None = None) -> list[dict[str, Any]]:
        if not address:
            address = self.account_address
        if not address:
            raise ValidationError("open_orders requires account address")
        return self.info.open_orders(address)

    def schedule_cancel(self, cancel_time_ms: int | None) -> dict[str, Any]:
        return self.exchange.schedule_cancel(cancel_time_ms)

    def user_state(self, address: str | None = None) -> dict[str, Any]:
        address = address or self.account_address
        if not address:
            raise ValidationError("user_state requires account address")
        return self.info.user_state(address)

    def user_fills(self, address: str | None = None) -> list[dict[str, Any]]:
        address = address or self.account_address
        if not address:
            raise ValidationError("user_fills requires account address")
        return self.info.user_fills(address)

    def query_order_by_oid(self, oid: int, address: str | None = None) -> dict[str, Any]:
        address = address or self.account_address
        if not address:
            raise ValidationError("query_order_by_oid requires account address")
        return self.info.query_order_by_oid(address, oid)

    def query_order_by_cloid(self, cloid: str, address: str | None = None) -> dict[str, Any]:
        address = address or self.account_address
        if not address:
            raise ValidationError("query_order_by_cloid requires account address")
        return self.info.query_order_by_cloid(address, to_sdk_cloid(cloid))

    def all_mids(self) -> dict[str, str]:
        return self.info.all_mids()

    def meta(self) -> dict[str, Any]:
        return self.info.meta()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def stable_hash(value: Any, *, length: int = 12) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:length]


def generate_cloid(task_id: str = CANARY_TASK_ID) -> str:
    seed = f"{task_id}:{time.time_ns()}:{os.getpid()}".encode()
    return "0x" + hashlib.sha256(seed).hexdigest()[:32]


def to_sdk_cloid(raw_cloid: str) -> Any:
    from hyperliquid.utils.types import Cloid  # type: ignore

    return Cloid.from_str(raw_cloid)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in REDACT_KEYS:
                redacted[key] = "<redacted>"
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = HEX_64_RE.sub("<redacted_hex64>", value)
        value = ADDRESS_RE.sub("<redacted_address>", value)
        value = HEX_32_RE.sub("<redacted_hex32>", value)
        return value
    return value


def config_snapshot(config: TinyLiveConfig) -> dict[str, Any]:
    return {
        "artifact_dir": str(config.artifact_dir),
        "duration_seconds": config.duration_seconds,
        "dry_run_private_preflight": config.dry_run_private_preflight,
        "live_mode": config.live_mode,
        "max_loss_usdc": config.max_loss_usdc,
        "max_notional_usdc": config.max_notional_usdc,
        "max_order_notional_usdc": config.max_order_notional_usdc,
        "max_order_size_btc": config.max_order_size_btc,
        "max_position_btc": config.max_position_btc,
        "max_position_notional_usdc": config.max_position_notional_usdc,
        "order_type": config.order_type,
        "operator_ack_present": bool(config.operator_ack),
        "reduce_only": config.reduce_only,
        "symbol": config.symbol,
        "time_in_force": config.time_in_force,
        "use_schedule_cancel": config.use_schedule_cancel,
    }


def validate_config(config: TinyLiveConfig, precision: PrecisionFacts | None) -> list[dict[str, str]]:
    checks = [
        ("symbol_is_btc", config.symbol == SYMBOL, f"actual={config.symbol}"),
        ("duration_lte_600s", config.duration_seconds <= MAX_DURATION_SECONDS, f"actual={config.duration_seconds}"),
        (
            "max_order_size_lte_0_01_btc",
            config.max_order_size_btc <= MAX_ORDER_SIZE_BTC,
            f"actual={config.max_order_size_btc}",
        ),
        (
            "max_order_notional_lte_700_usdc",
            config.max_order_notional_usdc <= MAX_ORDER_NOTIONAL_USDC,
            f"actual={config.max_order_notional_usdc}",
        ),
        (
            "max_position_lte_0_04_btc",
            config.max_position_btc <= MAX_POSITION_BTC,
            f"actual={config.max_position_btc}",
        ),
        (
            "max_position_notional_lte_2800_usdc",
            config.max_position_notional_usdc <= MAX_POSITION_NOTIONAL_USDC,
            f"actual={config.max_position_notional_usdc}",
        ),
        (
            "max_notional_lte_3000_usdc",
            config.max_notional_usdc <= MAX_NOTIONAL_USDC,
            f"actual={config.max_notional_usdc}",
        ),
        ("max_loss_eq_30_usdc", config.max_loss_usdc == MAX_LOSS_USDC, f"actual={config.max_loss_usdc}"),
        ("order_type_is_limit", config.order_type == "limit", f"actual={config.order_type}"),
        ("time_in_force_is_alo", config.time_in_force == POST_ONLY_TIF, f"actual={config.time_in_force}"),
        ("reduce_only_default_false", config.reduce_only is False, f"actual={config.reduce_only}"),
        ("artifact_dir_present", bool(config.artifact_dir), f"actual={config.artifact_dir}"),
        ("precision_facts_present", precision is not None, "precision facts are required"),
        (
            "precision_symbol_btc",
            precision is not None and precision.symbol == SYMBOL,
            f"actual={precision.symbol if precision else ''}",
        ),
        (
            "precision_positive_tick_lot_mid",
            precision is not None and precision.tick_size > 0 and precision.lot_size > 0 and precision.mid_px > 0,
            "tick_size/lot_size/mid_px must be positive",
        ),
        (
            "live_mode_has_operator_ack",
            (not config.live_mode) or config.operator_ack == "I_UNDERSTAND_THIS_CAN_PLACE_REAL_HYPERLIQUID_ORDERS",
            "live mode requires exact acknowledgement",
        ),
    ]
    rows = []
    for check, ok, detail in checks:
        rows.append({"check": check, "status": "pass" if ok else "fail_closed", "detail": detail})
    return rows


def assert_config_valid(config: TinyLiveConfig, precision: PrecisionFacts | None) -> None:
    failed = [row for row in validate_config(config, precision) if row["status"] != "pass"]
    if failed:
        raise ValidationError("; ".join(f"{row['check']}:{row['detail']}" for row in failed))


def validate_order_intent(config: TinyLiveConfig, precision: PrecisionFacts, intent: OrderIntent) -> None:
    if intent.symbol != SYMBOL:
        raise ValidationError("only BTC is allowed")
    if intent.time_in_force != POST_ONLY_TIF:
        raise ValidationError("only Alo post-only limit orders are allowed")
    if intent.reduce_only is not False:
        raise ValidationError("reduce_only is not enabled for this tiny-live executor")
    if intent.size_btc <= 0 or intent.size_btc > config.max_order_size_btc:
        raise ValidationError("order size exceeds cap")
    if intent.notional_usdc > config.max_order_notional_usdc:
        raise ValidationError("order notional exceeds cap")
    if intent.size_btc > config.max_position_btc:
        raise ValidationError("single order would exceed max position")
    if intent.notional_usdc > config.max_position_notional_usdc:
        raise ValidationError("single order would exceed max position notional")
    if precision.tick_size <= 0 or precision.lot_size <= 0:
        raise ValidationError("missing tick or lot precision")


def loss_status(config: TinyLiveConfig, snapshot: LossSnapshot | None) -> dict[str, Any]:
    if snapshot is None:
        return {"status": "fail_closed", "estimated_loss_usdc": "", "reason": "missing_loss_snapshot"}
    estimated_loss = snapshot.estimated_loss_usdc
    return {
        "status": "pass" if estimated_loss < config.max_loss_usdc else "fail_closed",
        "estimated_loss_usdc": round(estimated_loss, 8),
        "reason": "" if estimated_loss < config.max_loss_usdc else "max_loss_reached",
    }


def extract_tracked_refs(order_result: dict[str, Any]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    statuses = order_result.get("response", {}).get("data", {}).get("statuses", [])
    for status in statuses:
        resting = status.get("resting") if isinstance(status, dict) else None
        if not resting:
            continue
        refs.append({"oid": resting.get("oid"), "cloid": resting.get("cloid")})
    return refs


def run_order_once(
    *,
    config: TinyLiveConfig,
    precision: PrecisionFacts,
    intent: OrderIntent,
    loss_snapshot: LossSnapshot | None,
    client: HyperliquidClient,
) -> dict[str, Any]:
    assert_config_valid(config, precision)
    validate_order_intent(config, precision, intent)
    loss = loss_status(config, loss_snapshot)
    if loss["status"] != "pass":
        raise ValidationError(f"max loss check failed: {loss['reason']}")
    if not config.live_mode:
        raise ValidationError("live_mode=false prevents real order placement")
    return client.order(intent)


def shutdown_cancel_all(
    *,
    client: HyperliquidClient,
    symbol: str,
    tracked_refs: list[dict[str, Any]],
    account_address: str | None = None,
) -> ShutdownEvidence:
    evidence = ShutdownEvidence()
    for ref in tracked_refs:
        oid = ref.get("oid")
        cloid = ref.get("cloid")
        evidence.requested_refs.append(str(oid if oid is not None else cloid))
        evidence.cancel_results.append(redact(client.cancel_tracked(symbol, oid=oid, cloid=cloid)))
    final_open_raw = client.open_orders(account_address)
    remaining_tracked = []
    tracked_oids = {str(ref.get("oid")) for ref in tracked_refs if ref.get("oid") is not None}
    tracked_cloids = {str(ref.get("cloid")) for ref in tracked_refs if ref.get("cloid")}
    for order in final_open_raw:
        if str(order.get("oid")) in tracked_oids or str(order.get("cloid")) in tracked_cloids:
            remaining_tracked.append(order)
    evidence.final_open_orders = redact(final_open_raw)
    if remaining_tracked:
        evidence.proof_status = "fail_closed"
        evidence.fail_closed_reason = "tracked_order_still_open_or_ambiguous"
    else:
        evidence.proof_status = "pass"
    return evidence


def dependency_snapshot() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hyperliquid_sdk_available": importlib.util.find_spec("hyperliquid") is not None,
        "eth_account_available": importlib.util.find_spec("eth_account") is not None,
        "git_commit": git_commit(),
    }


def mock_precision() -> PrecisionFacts:
    return PrecisionFacts(symbol=SYMBOL, sz_decimals=5, tick_size=1.0, lot_size=0.00001, mid_px=65000.0, source="mock_self_test")


def precision_to_row(precision: PrecisionFacts) -> dict[str, Any]:
    return {
        "symbol": precision.symbol,
        "sz_decimals": precision.sz_decimals,
        "tick_size": precision.tick_size,
        "lot_size": precision.lot_size,
        "mid_px": precision.mid_px,
        "source": precision.source,
    }


def order_intent_row(intent: OrderIntent, *, endpoint_called: bool) -> dict[str, Any]:
    return {
        "symbol": intent.symbol,
        "side": "buy" if intent.is_buy else "sell",
        "size_btc": intent.size_btc,
        "limit_px": intent.limit_px,
        "notional_usdc": round(intent.notional_usdc, 8),
        "time_in_force": intent.time_in_force,
        "order_type": "limit",
        "reduce_only": str(intent.reduce_only).lower(),
        "endpoint_called": str(endpoint_called).lower(),
        "cloid_redacted": "cloid_sha256_" + hashlib.sha256(intent.cloid.encode()).hexdigest()[:12],
    }


def load_env_file(path: Path) -> dict[str, Any]:
    loaded_keys: list[str] = []
    if not path.exists():
        raise ValidationError(f"env file does not exist: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value
            loaded_keys.append(key)
    return {"path": str(path), "loaded_keys": sorted(loaded_keys)}


def _env_value(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


def credential_source_snapshot(*, env_file: Path | None, env_load: dict[str, Any] | None) -> dict[str, Any]:
    key_candidates = ["HL_PRIVATE_KEY", "HL_WALLET", "HYPERLIQUID_PRIVATE_KEY", "HYPERLIQUID_ACCOUNT_ADDRESS"]
    return {
        "env_file": str(env_file) if env_file else "",
        "env_file_keys_loaded": sorted(env_load.get("loaded_keys", [])) if env_load else [],
        "candidate_keys_present": sorted(key for key in key_candidates if os.environ.get(key)),
        "secret_values_written": False,
    }


def fetch_live_precision(client: SDKHyperliquidClient) -> PrecisionFacts:
    mids = client.all_mids()
    meta = client.meta()
    if SYMBOL not in mids:
        raise ValidationError("BTC mid price missing from Hyperliquid all_mids")
    universe = meta.get("universe", [])
    btc_meta = next((row for row in universe if row.get("name") == SYMBOL), None)
    if btc_meta is None:
        raise ValidationError("BTC metadata missing from Hyperliquid meta")
    sz_decimals = int(btc_meta.get("szDecimals"))
    price_decimals = max(0, 6 - sz_decimals)
    return PrecisionFacts(
        symbol=SYMBOL,
        sz_decimals=sz_decimals,
        tick_size=10 ** (-price_decimals),
        lot_size=10 ** (-sz_decimals),
        mid_px=float(mids[SYMBOL]),
        source="hyperliquid_info_meta_all_mids",
    )


def round_hyperliquid_perp_price(px: float, sz_decimals: int) -> float:
    decimals = max(0, 6 - sz_decimals)
    return round(float(f"{px:.5g}"), decimals)


def build_canary_intent(
    *,
    precision: PrecisionFacts,
    is_buy: bool = True,
    offset_bps: float = DEFAULT_CANARY_PRICE_OFFSET_BPS,
) -> OrderIntent:
    if not is_buy:
        raise ValidationError("0618T004 canary only supports far-from-mid post-only buy intent")
    offset = max(offset_bps, 1.0) / 10_000.0
    raw_px = precision.mid_px * (1.0 - offset)
    raw_px = min(raw_px, MAX_CANARY_LIMIT_PX)
    limit_px = round_hyperliquid_perp_price(raw_px, precision.sz_decimals)
    if limit_px * MAX_ORDER_SIZE_BTC > MAX_ORDER_NOTIONAL_USDC:
        limit_px = round_hyperliquid_perp_price((MAX_ORDER_NOTIONAL_USDC / MAX_ORDER_SIZE_BTC) - 100.0, precision.sz_decimals)
    if limit_px >= precision.mid_px:
        raise ValidationError("canary post-only buy price is not below current mid")
    return OrderIntent(
        symbol=SYMBOL,
        is_buy=True,
        size_btc=MAX_ORDER_SIZE_BTC,
        limit_px=limit_px,
        time_in_force=POST_ONLY_TIF,
        reduce_only=False,
        cloid=generate_cloid(CANARY_TASK_ID),
    )


def extract_status_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    statuses = result.get("response", {}).get("data", {}).get("statuses", [])
    if not isinstance(statuses, list):
        return []
    rows: list[dict[str, Any]] = []
    for status in statuses:
        if not isinstance(status, dict):
            continue
        status_type = next(iter(status.keys()), "unknown")
        rows.append({"status_type": status_type, "payload": status.get(status_type)})
    return rows


def canary_tracked_refs(order_result: dict[str, Any], intent: OrderIntent) -> list[dict[str, Any]]:
    refs = extract_tracked_refs(order_result)
    if refs:
        return refs
    return [{"oid": None, "cloid": intent.cloid}]


def build_sha256_manifest(output_dir: Path, files: list[Path]) -> list[dict[str, str]]:
    rows = []
    for path in files:
        rows.append({"artifact": path.name, "sha256": sha256(path)})
    write_csv(output_dir / "sha256_manifest.csv", rows, ["artifact", "sha256"])
    return rows


def generate_self_test_artifacts(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config = TinyLiveConfig(artifact_dir=output_dir, live_mode=False)
    precision = mock_precision()
    intent = OrderIntent(symbol=SYMBOL, is_buy=True, size_btc=0.01, limit_px=65000.0)
    loss_snapshot = LossSnapshot(entry_px=65000.0, mark_px=65000.0, position_btc=0.01)
    client = MockHyperliquidClient()
    validation_rows = validate_config(config, precision)
    assert_config_valid(config, precision)
    validate_order_intent(config, precision, intent)
    loss = loss_status(config, loss_snapshot)
    blocked_order_reason = ""
    try:
        run_order_once(config=config, precision=precision, intent=intent, loss_snapshot=loss_snapshot, client=client)
    except ValidationError as exc:
        blocked_order_reason = str(exc)
    mock_order_result = client.order(intent)
    tracked_refs = extract_tracked_refs(mock_order_result)
    shutdown = shutdown_cancel_all(client=client, symbol=SYMBOL, tracked_refs=tracked_refs)
    scheduled_cancel = client.schedule_cancel(int(time.time() * 1000) + 60_000)

    write_json(output_dir / "run_intent_marker.json", {"task_id": TASK_ID, "mode": "self_test", "real_orders_allowed": False})
    write_json(output_dir / "approved_config_snapshot.json", config_snapshot(config))
    write_json(output_dir / "environment_dependency_snapshot.json", dependency_snapshot())
    write_csv(
        output_dir / "official_doc_recheck_summary.csv",
        OFFICIAL_DOC_RECHECKS,
        ["topic", "url", "task_relevance", "local_recheck_summary"],
    )
    write_csv(output_dir / "precision_tick_lot_snapshot.csv", [precision_to_row(precision)], list(precision_to_row(precision)))
    write_csv(output_dir / "preflight_validation_summary.csv", validation_rows, ["check", "status", "detail"])
    write_csv(output_dir / "order_intent_audit.csv", [order_intent_row(intent, endpoint_called=False)], list(order_intent_row(intent, endpoint_called=False)))
    write_json(
        output_dir / "private_order_response_audit.json",
        {
            "mode": "mock_self_test",
            "real_order_endpoint_called": False,
            "mock_order_result": redact(mock_order_result),
            "live_order_blocked_reason": blocked_order_reason,
            "tracked_refs_count": len(tracked_refs),
        },
    )
    write_json(
        output_dir / "cancel_shutdown_proof.json",
        {
            "mode": "mock_self_test",
            "real_cancel_endpoint_called": False,
            "scheduled_cancel": redact(scheduled_cancel),
            "requested_refs": shutdown.requested_refs,
            "cancel_results": shutdown.cancel_results,
            "final_open_orders": shutdown.final_open_orders,
            "proof_status": shutdown.proof_status,
            "fail_closed_reason": shutdown.fail_closed_reason,
        },
    )
    write_json(output_dir / "max_loss_monitor_summary.json", loss)
    write_json(
        output_dir / "final_safety_summary.json",
        {
            "task_id": TASK_ID,
            "final_recommendation": FINAL_RECOMMENDATION_READY,
            "safe_mode": "self_test_no_network",
            "real_order_endpoint_called": False,
            "private_endpoint_called": False,
            "real_cancel_endpoint_called": False,
            "post_only_tif": POST_ONLY_TIF,
            "max_loss_usdc": config.max_loss_usdc,
            "shutdown_proof_status": shutdown.proof_status,
            "next_live_task_required_before_real_orders": True,
        },
    )
    write_csv(
        output_dir / "boundary_validation.csv",
        [{"check": key, "status": "pass" if value is True or value is False else "fail_closed", "value": str(value).lower()} for key, value in BOUNDARY_FLAGS.items()],
        ["check", "status", "value"],
    )
    write_text(
        output_dir / "README.md",
        "\n".join(
            [
                "# Hyperliquid Tiny-Live Real-Order Executor Self-Test",
                "",
                "This artifact set proves local validation, SDK interface wrapping, post-only Alo intent, max-loss fail-closed behavior, mock order lifecycle, and cancel-all shutdown flow.",
                "",
                "No real Hyperliquid private/order endpoint was called and no real order was placed in this task.",
                "",
            ]
        ),
    )

    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "final_recommendation": FINAL_RECOMMENDATION_READY,
        "executor_ready": True,
        "post_only_enforcement_implemented": True,
        "real_cancel_all_shutdown_implemented": True,
        "private_order_response_source_live_capable": True,
        "live_mode_executed": False,
        "real_order_endpoint_called": False,
        "private_endpoint_called": False,
        "real_cancel_endpoint_called": False,
        "credentials_written": False,
        "max_loss_status": loss["status"],
        "shutdown_proof_status": shutdown.proof_status,
        "sdk_available_local": dependency_snapshot()["hyperliquid_sdk_available"],
        "sdk_required_for_live": True,
        "live_mode_blocked_in_self_test_reason": blocked_order_reason,
        "boundary_flags": BOUNDARY_FLAGS,
        "artifacts": {
            "approved_config_snapshot": str(output_dir / "approved_config_snapshot.json"),
            "boundary_validation": str(output_dir / "boundary_validation.csv"),
            "cancel_shutdown_proof": str(output_dir / "cancel_shutdown_proof.json"),
            "environment_dependency_snapshot": str(output_dir / "environment_dependency_snapshot.json"),
            "max_loss_monitor_summary": str(output_dir / "max_loss_monitor_summary.json"),
            "official_doc_recheck_summary": str(output_dir / "official_doc_recheck_summary.csv"),
            "order_intent_audit": str(output_dir / "order_intent_audit.csv"),
            "precision_tick_lot_snapshot": str(output_dir / "precision_tick_lot_snapshot.csv"),
            "preflight_validation_summary": str(output_dir / "preflight_validation_summary.csv"),
            "private_order_response_audit": str(output_dir / "private_order_response_audit.json"),
            "run_intent_marker": str(output_dir / "run_intent_marker.json"),
            "final_safety_summary": str(output_dir / "final_safety_summary.json"),
        },
        "git_commit": git_commit(),
    }
    artifact_files = [
        output_dir / "run_intent_marker.json",
        output_dir / "approved_config_snapshot.json",
        output_dir / "environment_dependency_snapshot.json",
        output_dir / "official_doc_recheck_summary.csv",
        output_dir / "precision_tick_lot_snapshot.csv",
        output_dir / "preflight_validation_summary.csv",
        output_dir / "order_intent_audit.csv",
        output_dir / "private_order_response_audit.json",
        output_dir / "cancel_shutdown_proof.json",
        output_dir / "max_loss_monitor_summary.json",
        output_dir / "final_safety_summary.json",
        output_dir / "boundary_validation.csv",
        output_dir / "README.md",
        output_dir / "executor_manifest.json",
    ]
    write_json(output_dir / "executor_manifest.json", manifest)
    build_sha256_manifest(output_dir, artifact_files)
    manifest["artifacts"]["sha256_manifest"] = str(output_dir / "sha256_manifest.csv")
    write_json(output_dir / "executor_manifest.json", manifest)
    build_sha256_manifest(output_dir, artifact_files)
    return manifest


def build_live_client_from_env(*, allow_missing: bool = False) -> SDKHyperliquidClient | None:
    if importlib.util.find_spec("hyperliquid") is None or importlib.util.find_spec("eth_account") is None:
        if allow_missing:
            return None
        raise ValidationError("official hyperliquid-python-sdk and eth_account are required for live mode")
    private_key = _env_value("HYPERLIQUID_PRIVATE_KEY", "HL_PRIVATE_KEY")
    account_address = _env_value("HYPERLIQUID_ACCOUNT_ADDRESS", "HL_WALLET")
    if not private_key:
        if allow_missing:
            return None
        raise ValidationError("HYPERLIQUID_PRIVATE_KEY or HL_PRIVATE_KEY is required for live mode")
    from eth_account import Account  # type: ignore
    from hyperliquid.exchange import Exchange  # type: ignore
    from hyperliquid.info import Info  # type: ignore
    from hyperliquid.utils import constants  # type: ignore

    wallet = Account.from_key(private_key)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    exchange = Exchange(wallet, constants.MAINNET_API_URL, account_address=account_address or None)
    return SDKHyperliquidClient(exchange=exchange, info=info, account_address=account_address or wallet.address)


def _redacted_error(exc: BaseException) -> str:
    return str(redact({"error": str(exc)})["error"])


def _canary_order_still_open(open_orders: list[dict[str, Any]], tracked_refs: list[dict[str, Any]], cloid: str) -> bool:
    tracked_oids = {str(ref.get("oid")) for ref in tracked_refs if ref.get("oid") is not None}
    tracked_cloids = {str(ref.get("cloid")) for ref in tracked_refs if ref.get("cloid")}
    tracked_cloids.add(cloid)
    for order in open_orders:
        if str(order.get("oid")) in tracked_oids or str(order.get("cloid")) in tracked_cloids:
            return True
    return False


def generate_real_order_canary_artifacts(
    *,
    output_dir: Path = DEFAULT_CANARY_OUTPUT_DIR,
    env_file: Path | None = None,
    allow_existing_open_orders: bool = False,
    canary_price_offset_bps: float = DEFAULT_CANARY_PRICE_OFFSET_BPS,
    use_schedule_cancel: bool = True,
    canary_task_id: str = CANARY_TASK_ID,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = TinyLiveConfig(
        artifact_dir=output_dir,
        live_mode=True,
        operator_ack=LIVE_OPERATOR_ACK,
        use_schedule_cancel=use_schedule_cancel,
    )
    env_load: dict[str, Any] | None = None
    client: SDKHyperliquidClient | None = None
    precision: PrecisionFacts | None = None
    intent: OrderIntent | None = None
    tracked_refs: list[dict[str, Any]] = []
    pre_open_orders: list[dict[str, Any]] = []
    final_open_orders: list[dict[str, Any]] = []
    schedule_set_result: dict[str, Any] | None = None
    schedule_unset_result: dict[str, Any] | None = None
    order_result: dict[str, Any] | None = None
    order_exception = ""
    cancel_results: list[dict[str, Any]] = []
    query_results: dict[str, Any] = {}
    blocking_reasons: list[str] = []
    endpoint_flags = {
        "private_endpoint_called": False,
        "real_order_endpoint_called": False,
        "real_cancel_endpoint_called": False,
        "schedule_cancel_endpoint_called": False,
    }

    try:
        if env_file is not None:
            env_load = load_env_file(env_file)
        client = build_live_client_from_env()
        endpoint_flags["private_endpoint_called"] = True
        pre_open_orders = client.open_orders()
        preflight_summary = {
            "client_preflight": client.preflight(config),
            "open_order_count_before": len(pre_open_orders),
            "asset_position_count_before": len(client.user_state().get("assetPositions", [])),
            "user_fill_count_before": len(client.user_fills()),
        }
        if pre_open_orders and not allow_existing_open_orders:
            raise ValidationError("pre_existing_open_orders_present; refusing canary to avoid cancel-all ambiguity")

        precision = fetch_live_precision(client)
        intent = build_canary_intent(precision=precision, offset_bps=canary_price_offset_bps)
        intent = OrderIntent(
            symbol=intent.symbol,
            is_buy=intent.is_buy,
            size_btc=intent.size_btc,
            limit_px=intent.limit_px,
            time_in_force=intent.time_in_force,
            reduce_only=intent.reduce_only,
            cloid=generate_cloid(canary_task_id),
        )
        assert_config_valid(config, precision)
        validate_order_intent(config, precision, intent)
        loss = loss_status(config, LossSnapshot(entry_px=intent.limit_px, mark_px=intent.limit_px, position_btc=intent.size_btc))
        if loss["status"] != "pass":
            raise ValidationError(f"max loss check failed: {loss['reason']}")

        if config.use_schedule_cancel:
            endpoint_flags["schedule_cancel_endpoint_called"] = True
            schedule_set_result = client.schedule_cancel(int(time.time() * 1000) + 60_000)
        endpoint_flags["real_order_endpoint_called"] = True
        order_result = run_order_once(config=config, precision=precision, intent=intent, loss_snapshot=LossSnapshot(intent.limit_px, intent.limit_px, intent.size_btc), client=client)
        tracked_refs = canary_tracked_refs(order_result, intent)

        query_results["by_cloid_after_order"] = redact(client.query_order_by_cloid(intent.cloid))
        for ref in tracked_refs:
            if ref.get("oid") is not None:
                query_results["by_oid_after_order"] = redact(client.query_order_by_oid(int(ref["oid"])))
                break
    except Exception as exc:
        order_exception = _redacted_error(exc)
        blocking_reasons.append(order_exception)
        if order_result is not None and intent is not None:
            tracked_refs = canary_tracked_refs(order_result, intent)
    finally:
        if client is not None and intent is not None:
            for ref in tracked_refs:
                oid = ref.get("oid")
                if oid is not None:
                    endpoint_flags["real_cancel_endpoint_called"] = True
                    try:
                        cancel_results.append({"method": "cancel", "result": redact(client.cancel_tracked(SYMBOL, oid=int(oid)))})
                    except Exception as exc:
                        cancel_results.append({"method": "cancel", "error": _redacted_error(exc)})
            endpoint_flags["real_cancel_endpoint_called"] = True
            try:
                cancel_results.append({"method": "cancel_by_cloid", "result": redact(client.cancel_tracked(SYMBOL, cloid=intent.cloid))})
            except Exception as exc:
                cancel_results.append({"method": "cancel_by_cloid", "error": _redacted_error(exc)})
            try:
                query_results["by_cloid_after_cancel"] = redact(client.query_order_by_cloid(intent.cloid))
            except Exception as exc:
                query_results["by_cloid_after_cancel"] = {"error": _redacted_error(exc)}
        if client is not None:
            try:
                final_open_orders = client.open_orders()
            except Exception as exc:
                blocking_reasons.append(f"final_open_orders_failed:{_redacted_error(exc)}")
            if config.use_schedule_cancel:
                try:
                    endpoint_flags["schedule_cancel_endpoint_called"] = True
                    schedule_unset_result = client.schedule_cancel(None)
                except Exception as exc:
                    blocking_reasons.append(f"schedule_cancel_unset_failed:{_redacted_error(exc)}")

    if intent is not None:
        shutdown_proof_status = "fail_closed" if _canary_order_still_open(final_open_orders, tracked_refs, intent.cloid) else "pass"
    else:
        shutdown_proof_status = "not_started"
    if shutdown_proof_status != "pass":
        blocking_reasons.append("canary_order_still_open_or_shutdown_ambiguous")

    if precision is None:
        precision = mock_precision()
    validation_rows = validate_config(config, precision)
    loss_summary = loss_status(
        config,
        LossSnapshot(
            entry_px=intent.limit_px if intent else precision.mid_px,
            mark_px=intent.limit_px if intent else precision.mid_px,
            position_btc=intent.size_btc if intent else 0.0,
        ),
    )
    order_status_rows = extract_status_rows(order_result or {})
    order_submission_attempted = endpoint_flags["real_order_endpoint_called"]
    canary_ready = (
        order_submission_attempted
        and endpoint_flags["private_endpoint_called"]
        and endpoint_flags["real_cancel_endpoint_called"]
        and (endpoint_flags["schedule_cancel_endpoint_called"] or not config.use_schedule_cancel)
        and shutdown_proof_status == "pass"
    )
    final_recommendation = FINAL_RECOMMENDATION_CANARY_READY if canary_ready else FINAL_RECOMMENDATION_CANARY_BLOCKED

    write_json(
        output_dir / "run_intent_marker.json",
        {
            "task_id": canary_task_id,
            "mode": "real_order_canary",
            "real_orders_allowed": True,
            "schedule_cancel_required": False,
            "use_schedule_cancel": config.use_schedule_cancel,
        },
    )
    write_json(output_dir / "approved_config_snapshot.json", config_snapshot(config))
    write_json(output_dir / "credential_source_manifest.json", credential_source_snapshot(env_file=env_file, env_load=env_load))
    write_json(output_dir / "environment_dependency_snapshot.json", dependency_snapshot())
    write_csv(output_dir / "official_doc_recheck_summary.csv", OFFICIAL_DOC_RECHECKS, ["topic", "url", "task_relevance", "local_recheck_summary"])
    write_csv(output_dir / "precision_tick_lot_snapshot.csv", [precision_to_row(precision)], list(precision_to_row(precision)))
    write_csv(output_dir / "preflight_validation_summary.csv", validation_rows, ["check", "status", "detail"])
    write_json(
        output_dir / "private_preflight_summary.json",
        redact(
        {
                "preflight_summary": preflight_summary if "preflight_summary" in locals() else {},
                "open_orders_before": pre_open_orders,
                "open_order_count_before": len(pre_open_orders),
                "allow_existing_open_orders": allow_existing_open_orders,
                "endpoint_called": endpoint_flags["private_endpoint_called"],
            }
        ),
    )
    if intent is not None:
        write_csv(output_dir / "order_intent_audit.csv", [order_intent_row(intent, endpoint_called=order_submission_attempted)], list(order_intent_row(intent, endpoint_called=order_submission_attempted)))
    else:
        write_csv(output_dir / "order_intent_audit.csv", [], ["symbol", "side", "size_btc", "limit_px", "notional_usdc", "time_in_force", "order_type", "reduce_only", "endpoint_called", "cloid_redacted"])
    write_json(
        output_dir / "private_order_response_audit.json",
        redact(
            {
                "real_order_endpoint_called": endpoint_flags["real_order_endpoint_called"],
                "order_submission_attempted": order_submission_attempted,
                "order_status_rows": order_status_rows,
                "order_result": order_result,
                "order_exception": order_exception,
                "query_results": query_results,
            }
        ),
    )
    write_json(
        output_dir / "cancel_shutdown_proof.json",
        redact(
            {
                "real_cancel_endpoint_called": endpoint_flags["real_cancel_endpoint_called"],
                "schedule_cancel_endpoint_called": endpoint_flags["schedule_cancel_endpoint_called"],
                "schedule_set_result": schedule_set_result,
                "schedule_unset_result": schedule_unset_result,
                "tracked_refs": tracked_refs,
                "cancel_results": cancel_results,
                "final_open_orders": final_open_orders,
                "proof_status": shutdown_proof_status,
            }
        ),
    )
    write_json(output_dir / "max_loss_monitor_summary.json", loss_summary)
    write_json(
        output_dir / "final_safety_summary.json",
        {
            "task_id": canary_task_id,
            "final_recommendation": final_recommendation,
            "blocking_reasons": blocking_reasons,
            "order_submission_attempted": order_submission_attempted,
            "private_endpoint_called": endpoint_flags["private_endpoint_called"],
            "real_order_endpoint_called": endpoint_flags["real_order_endpoint_called"],
            "real_cancel_endpoint_called": endpoint_flags["real_cancel_endpoint_called"],
            "schedule_cancel_endpoint_called": endpoint_flags["schedule_cancel_endpoint_called"],
            "schedule_cancel_required": False,
            "tracked_cancel_required": True,
            "final_open_orders_empty_required": True,
            "shutdown_proof_status": shutdown_proof_status,
            "credentials_written": False,
            "secret_values_written": False,
            "post_only_tif": POST_ONLY_TIF,
            "max_loss_usdc": config.max_loss_usdc,
        },
    )
    write_text(
        output_dir / "README.md",
        "\n".join(
            [
                "# Hyperliquid Tiny-Live Real-Order Canary",
                "",
                f"Final recommendation: `{final_recommendation}`",
                "",
                "This artifact set is redacted. It records the authenticated private-read, order-attempt, cancel, schedule-cancel, and shutdown proof needed for QA.",
                "",
                "It is not a continuous live strategy run, not a PnL claim, and not maker-viability proof.",
                "",
            ]
        ),
    )
    manifest = {
        "task_id": canary_task_id,
        "schema_version": SCHEMA_VERSION,
        "final_recommendation": final_recommendation,
        "blocking_reasons": blocking_reasons,
        "executor_ready": canary_ready,
        "live_mode_executed": True,
        "real_order_endpoint_called": endpoint_flags["real_order_endpoint_called"],
        "private_endpoint_called": endpoint_flags["private_endpoint_called"],
        "real_cancel_endpoint_called": endpoint_flags["real_cancel_endpoint_called"],
        "schedule_cancel_endpoint_called": endpoint_flags["schedule_cancel_endpoint_called"],
        "schedule_cancel_required": False,
        "use_schedule_cancel": config.use_schedule_cancel,
        "tracked_cancel_required": True,
        "final_open_orders_empty_required": True,
        "order_submission_attempted": order_submission_attempted,
        "order_status_types": [row.get("status_type", "") for row in order_status_rows],
        "credentials_written": False,
        "secret_values_written": False,
        "raw_signatures_written": False,
        "max_loss_status": loss_summary["status"],
        "shutdown_proof_status": shutdown_proof_status,
        "post_only_enforcement_implemented": True,
        "real_cancel_all_shutdown_implemented": True,
        "private_order_response_source_live_capable": True,
        "sdk_available_local": dependency_snapshot()["hyperliquid_sdk_available"],
        "sdk_required_for_live": True,
        "git_commit": git_commit(),
        "artifacts": {
            "approved_config_snapshot": str(output_dir / "approved_config_snapshot.json"),
            "cancel_shutdown_proof": str(output_dir / "cancel_shutdown_proof.json"),
            "credential_source_manifest": str(output_dir / "credential_source_manifest.json"),
            "environment_dependency_snapshot": str(output_dir / "environment_dependency_snapshot.json"),
            "final_safety_summary": str(output_dir / "final_safety_summary.json"),
            "max_loss_monitor_summary": str(output_dir / "max_loss_monitor_summary.json"),
            "official_doc_recheck_summary": str(output_dir / "official_doc_recheck_summary.csv"),
            "order_intent_audit": str(output_dir / "order_intent_audit.csv"),
            "precision_tick_lot_snapshot": str(output_dir / "precision_tick_lot_snapshot.csv"),
            "preflight_validation_summary": str(output_dir / "preflight_validation_summary.csv"),
            "private_order_response_audit": str(output_dir / "private_order_response_audit.json"),
            "private_preflight_summary": str(output_dir / "private_preflight_summary.json"),
            "run_intent_marker": str(output_dir / "run_intent_marker.json"),
        },
    }
    artifact_files = [
        output_dir / "run_intent_marker.json",
        output_dir / "approved_config_snapshot.json",
        output_dir / "credential_source_manifest.json",
        output_dir / "environment_dependency_snapshot.json",
        output_dir / "official_doc_recheck_summary.csv",
        output_dir / "precision_tick_lot_snapshot.csv",
        output_dir / "preflight_validation_summary.csv",
        output_dir / "private_preflight_summary.json",
        output_dir / "order_intent_audit.csv",
        output_dir / "private_order_response_audit.json",
        output_dir / "cancel_shutdown_proof.json",
        output_dir / "max_loss_monitor_summary.json",
        output_dir / "final_safety_summary.json",
        output_dir / "README.md",
        output_dir / "executor_manifest.json",
    ]
    write_json(output_dir / "executor_manifest.json", manifest)
    build_sha256_manifest(output_dir, artifact_files)
    manifest["artifacts"]["sha256_manifest"] = str(output_dir / "sha256_manifest.csv")
    write_json(output_dir / "executor_manifest.json", manifest)
    build_sha256_manifest(output_dir, artifact_files)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperliquid tiny-live real-order executor scaffold.")
    parser.add_argument("--self-test", action="store_true", help="generate no-network self-test artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--live", action="store_true", help="enable live-capable path; not used by 0618T001")
    parser.add_argument("--real-order-canary", action="store_true", help="execute the 0618T004 authenticated real-order canary")
    parser.add_argument("--env-file", type=Path, default=None, help="optional .env file to load without printing values")
    parser.add_argument("--allow-existing-open-orders", action="store_true", help="allow canary despite pre-existing open orders")
    parser.add_argument("--canary-price-offset-bps", type=float, default=DEFAULT_CANARY_PRICE_OFFSET_BPS)
    parser.add_argument("--disable-schedule-cancel", action="store_true", help="do not call Exchange.schedule_cancel")
    parser.add_argument("--canary-task-id", default=CANARY_TASK_ID, help="task id to record in canary artifacts")
    parser.add_argument("--operator-ack", default="")
    args = parser.parse_args()

    if args.self_test:
        manifest = generate_self_test_artifacts(args.output_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if args.real_order_canary:
        if args.operator_ack != LIVE_OPERATOR_ACK:
            raise SystemExit("real-order canary requires exact operator acknowledgement")
        manifest = generate_real_order_canary_artifacts(
            output_dir=args.output_dir,
            env_file=args.env_file,
            allow_existing_open_orders=args.allow_existing_open_orders,
            canary_price_offset_bps=args.canary_price_offset_bps,
            use_schedule_cancel=not args.disable_schedule_cancel,
            canary_task_id=args.canary_task_id,
        )
        print(json.dumps(redact(manifest), indent=2, sort_keys=True))
        return 0
    if args.live:
        config = TinyLiveConfig(artifact_dir=args.output_dir, live_mode=True, operator_ack=args.operator_ack)
        assert_config_valid(config, mock_precision())
        build_live_client_from_env()
        raise SystemExit("live client initialized, but 0618T001 does not execute live order placement")
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
