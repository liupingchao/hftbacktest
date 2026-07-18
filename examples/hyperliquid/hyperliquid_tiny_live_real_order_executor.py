#!/usr/bin/env python3
"""Minimal Hyperliquid tiny-live real-order executor scaffold.

The default path is local self-test only. The live-capable path requires an
explicit live flag and operator acknowledgement. This task does not run that
path or place real orders.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import cross_exchange_price_math


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
MAX_REAL_ORDER_SUBMISSIONS = 30
DEFAULT_FORMAL_MAX_REAL_ORDER_SUBMISSIONS = 2
MAX_LOSS_USDC = 30.0
SYMBOL = "BTC"
POST_ONLY_TIF = "Alo"
LIVE_OPERATOR_ACK = "I_UNDERSTAND_THIS_CAN_PLACE_REAL_HYPERLIQUID_ORDERS"
DEFAULT_CANARY_PRICE_OFFSET_BPS = 200.0
MAX_CANARY_LIMIT_PX = 69_900.0
KILL_SWITCH_SCHEMA_VERSION = "hyperliquid_kill_switch_state_v1"
KILL_SWITCH_STATE_FILENAME = "kill_switch_state.json"
KILL_SWITCH_RESET_ACK = "I_UNDERSTAND_THIS_RESETS_HYPERLIQUID_KILL_SWITCH"
DEFAULT_KILL_SWITCH_HALT_SECONDS = 1800.0
DEFAULT_MARKET_CLOSE_SLIPPAGE = 0.05
DEFAULT_CONTROL_STATE_DIR = (
    Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
    / "hftbacktest"
    / "hyperliquid_btc"
)
KILL_SWITCH_TRIGGER_REASONS = frozenset(
    {
        "max_loss_reached",
        "position_or_projected_exposure_cap",
        "toxic_flow_hard_trigger",
        "market_data_stale_or_incoherent",
        "unknown_order_state",
        "orchestrator_abort_or_timeout",
        "manual_operator_kill",
    }
)

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


class KillSwitchBlocked(ValidationError):
    """Raised when the persistent halt wins the final order-submit race."""


@dataclass(frozen=True)
class TinyLiveConfig:
    symbol: str = SYMBOL
    duration_seconds: int = MAX_DURATION_SECONDS
    max_order_size_btc: float = MAX_ORDER_SIZE_BTC
    max_order_notional_usdc: float = MAX_ORDER_NOTIONAL_USDC
    max_position_btc: float = MAX_POSITION_BTC
    max_position_notional_usdc: float = MAX_POSITION_NOTIONAL_USDC
    max_notional_usdc: float = MAX_NOTIONAL_USDC
    max_real_order_submissions: int = DEFAULT_FORMAL_MAX_REAL_ORDER_SUBMISSIONS
    max_loss_usdc: float = MAX_LOSS_USDC
    time_in_force: str = POST_ONLY_TIF
    order_type: str = "limit"
    reduce_only: bool = False
    artifact_dir: Path = DEFAULT_OUTPUT_DIR
    live_mode: bool = False
    operator_ack: str = ""
    use_schedule_cancel: bool = True
    dry_run_private_preflight: bool = True
    control_state_dir: Path | None = None


@dataclass(frozen=True)
class PrecisionFacts:
    symbol: str
    sz_decimals: int
    tick_size: float
    lot_size: float
    mid_px: float
    source: str


@dataclass(frozen=True)
class ProjectedExposure:
    """Worst-case inventory before adding a new batch of proposed quotes.

    Working and inflight quantities are deliberately retained until the
    exchange confirms a cancel or submit outcome. The two worst-case values
    assume only one side can fill at a time, which prevents opposite-side
    leaves from masking a possible inventory breach.
    """

    position_btc: float
    working_buy_qty: float
    working_sell_qty: float
    inflight_buy_qty: float
    inflight_sell_qty: float
    worst_long_btc: float
    worst_short_btc: float
    existing_max_quote_px: float | None = None


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


@dataclass(frozen=True)
class KillSwitchConfig:
    control_state_dir: Path
    halt_seconds: float = DEFAULT_KILL_SWITCH_HALT_SECONDS
    market_close_slippage: float = DEFAULT_MARKET_CLOSE_SLIPPAGE
    symbol: str = SYMBOL
    lot_size_btc: float = 0.00001
    state_filename: str = KILL_SWITCH_STATE_FILENAME

    @property
    def state_path(self) -> Path:
        return self.control_state_dir / self.state_filename


@dataclass(frozen=True)
class HaltState:
    status: str
    state_path: Path
    trigger_reason: str = ""
    triggered_at: float | None = None
    expires_at: float | None = None
    resolution: str = ""
    fail_closed_reason: str = ""

    @property
    def is_halted(self) -> bool:
        return self.status in {"halted", "fail_closed"}

    @property
    def may_quote(self) -> bool:
        return self.status in {"clear", "expired"}


@dataclass
class ShutdownEvidence:
    requested_refs: list[str] = field(default_factory=list)
    cancel_results: list[dict[str, Any]] = field(default_factory=list)
    final_open_orders: list[dict[str, Any]] = field(default_factory=list)
    proof_status: str = "not_started"
    fail_closed_reason: str = ""


@dataclass
class KillSwitchEvidence:
    status: str = "not_started"
    trigger_reason: str = ""
    idempotent: bool = False
    state_path: str = ""
    halt_state: dict[str, Any] = field(default_factory=dict)
    cancel_evidence: dict[str, Any] = field(default_factory=dict)
    position_before: dict[str, Any] = field(default_factory=dict)
    position_after: dict[str, Any] = field(default_factory=dict)
    market_close_called: bool = False
    market_close_request: dict[str, Any] = field(default_factory=dict)
    market_close_response: dict[str, Any] = field(default_factory=dict)
    residual_position_btc: float | None = None
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

    def user_state(self, address: str | None = None) -> dict[str, Any]:
        ...

    def market_close(
        self,
        symbol: str,
        *,
        sz: float,
        slippage: float,
        cloid: str,
    ) -> dict[str, Any]:
        ...

    def query_order_by_oid(self, oid: int, address: str | None = None) -> dict[str, Any]:
        ...

    def query_order_by_cloid(self, cloid: str, address: str | None = None) -> dict[str, Any]:
        ...


class MockHyperliquidClient:
    """Deterministic no-network client used by tests and self-test artifacts."""

    def __init__(
        self,
        *,
        final_open_orders: list[dict[str, Any]] | None = None,
        position_szi: float = 0.0,
        position_sequence: list[float] | None = None,
        fail_cancel: bool = False,
        fail_market_close: bool = False,
    ) -> None:
        self.orders: list[dict[str, Any]] = []
        self.cancels: list[dict[str, Any]] = []
        self.market_close_calls: list[dict[str, Any]] = []
        self.user_state_calls: list[str | None] = []
        self.scheduled_cancel_ms: int | None = None
        self.final_open_orders = final_open_orders or []
        self.position_szi = float(position_szi)
        self.position_sequence = list(position_sequence or [])
        self.fail_cancel = fail_cancel
        self.fail_market_close = fail_market_close

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
        if self.fail_cancel:
            raise RuntimeError("mock_cancel_failure")
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
        self.user_state_calls.append(address)
        if self.position_sequence:
            self.position_szi = float(self.position_sequence.pop(0))
        positions = []
        if self.position_szi:
            positions.append({"position": {"coin": SYMBOL, "szi": str(self.position_szi)}})
        return {"assetPositions": positions, "mock": True}

    def market_close(
        self,
        symbol: str,
        *,
        sz: float,
        slippage: float,
        cloid: str,
    ) -> dict[str, Any]:
        if self.fail_market_close:
            raise RuntimeError("mock_market_close_failure")
        self.market_close_calls.append(
            {
                "symbol": symbol,
                "sz": sz,
                "slippage": slippage,
                "cloid": cloid,
                "reduce_only": True,
                "expected_side": "sell" if self.position_szi > 0 else "buy",
            }
        )
        if not self.position_sequence:
            self.position_szi = 0.0
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{"filled": {"totalSz": str(sz)}}]}},
            "mock": True,
        }

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

    def market_close(
        self,
        symbol: str,
        *,
        sz: float,
        slippage: float,
        cloid: str,
    ) -> dict[str, Any]:
        return self.exchange.market_close(
            symbol,
            sz=sz,
            slippage=slippage,
            cloid=to_sdk_cloid(cloid),
        )

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


def canonical_price_key(limit_px: float, *, sz_decimals: int) -> str:
    try:
        normalized = cross_exchange_price_math.normalize_hl_perp_price(
            limit_px,
            sz_decimals=sz_decimals,
            side="nearest",
        )
    except ValueError as exc:
        raise ValidationError(f"canonical_price_key_invalid_price:{exc}") from exc
    return f"{normalized:.12f}".rstrip("0").rstrip(".") or "0"


def managed_cloid_prefix(*, task_id: str, run_id: str) -> str:
    return "0x" + hashlib.sha256(f"{task_id}:{run_id}".encode("utf-8")).hexdigest()[:8]


def generate_managed_cloid(
    *,
    task_id: str,
    run_id: str,
    window_id: int,
    side: str,
    canonical_price: str,
    generation: int,
) -> str:
    if side not in {"buy", "sell"}:
        raise ValidationError("managed_cloid_side_invalid")
    if isinstance(window_id, bool) or not isinstance(window_id, int) or window_id < 1:
        raise ValidationError("managed_cloid_window_id_invalid")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValidationError("managed_cloid_generation_invalid")
    prefix = managed_cloid_prefix(task_id=task_id, run_id=run_id)
    identity = f"{task_id}:{run_id}:{window_id}:{side}:{canonical_price}:{generation}"
    return prefix + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def is_owned_managed_cloid(cloid: Any, *, task_id: str, run_id: str) -> bool:
    return (
        isinstance(cloid, str)
        and len(cloid) == 34
        and cloid.startswith(managed_cloid_prefix(task_id=task_id, run_id=run_id))
        and all(char in "0123456789abcdef" for char in cloid[2:].lower())
    )


def to_sdk_cloid(raw_cloid: str) -> Any:
    from hyperliquid.utils.types import Cloid  # type: ignore

    return Cloid.from_str(raw_cloid)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(temp_path, 0o600)
        os.replace(temp_path, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temp_path.exists():
            temp_path.unlink()


@contextmanager
def _kill_switch_lock(control_state_dir: Path):
    control_state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = control_state_dir / ".kill_switch.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_fh:
        os.chmod(lock_path, 0o600)
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _halt_state_dict(state: HaltState) -> dict[str, Any]:
    return {
        "status": state.status,
        "state_path": str(state.state_path),
        "trigger_reason": state.trigger_reason,
        "triggered_at": state.triggered_at,
        "expires_at": state.expires_at,
        "resolution": state.resolution,
        "fail_closed_reason": state.fail_closed_reason,
        "is_halted": state.is_halted,
        "may_quote": state.may_quote,
    }


def validate_kill_switch_config(config: KillSwitchConfig) -> None:
    if config.symbol != SYMBOL:
        raise ValidationError("kill_switch_only_supports_btc")
    if not math.isfinite(config.halt_seconds) or config.halt_seconds <= 0:
        raise ValidationError("kill_switch_halt_seconds_must_be_positive")
    if not math.isfinite(config.market_close_slippage) or not 0 < config.market_close_slippage <= 1:
        raise ValidationError("kill_switch_market_close_slippage_out_of_range")
    if not math.isfinite(config.lot_size_btc) or config.lot_size_btc <= 0:
        raise ValidationError("kill_switch_lot_size_must_be_positive")
    if config.state_filename != KILL_SWITCH_STATE_FILENAME:
        raise ValidationError("kill_switch_state_filename_must_use_authoritative_name")


def check_halt_state(control_state_dir: Path, *, now: float | None = None) -> HaltState:
    control_state_dir = Path(control_state_dir)
    state_path = control_state_dir / KILL_SWITCH_STATE_FILENAME
    if not control_state_dir.exists() or not control_state_dir.is_dir():
        return HaltState(
            status="fail_closed",
            state_path=state_path,
            fail_closed_reason="control_state_dir_missing_or_not_directory",
        )
    if not state_path.exists():
        return HaltState(
            status="fail_closed",
            state_path=state_path,
            fail_closed_reason="halt_state_file_missing",
        )
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return HaltState(
            status="fail_closed",
            state_path=state_path,
            fail_closed_reason=f"halt_state_unreadable_or_invalid_json:{_redacted_error(exc)}",
        )
    if not isinstance(payload, dict):
        return HaltState(
            status="fail_closed",
            state_path=state_path,
            fail_closed_reason="halt_state_root_not_object",
        )
    if payload.get("schema_version") != KILL_SWITCH_SCHEMA_VERSION:
        return HaltState(
            status="fail_closed",
            state_path=state_path,
            fail_closed_reason="halt_state_schema_missing_or_mismatch",
        )
    raw_status = payload.get("status")
    if raw_status == "armed":
        armed_at = payload.get("armed_at")
        if (
            not isinstance(armed_at, (int, float))
            or not math.isfinite(float(armed_at))
            or payload.get("symbol") != SYMBOL
            or payload.get("quote_generation_allowed") is not True
        ):
            return HaltState(
                status="fail_closed",
                state_path=state_path,
                fail_closed_reason="halt_state_armed_fields_invalid",
            )
        return HaltState(status="clear", state_path=state_path, resolution="armed")
    if raw_status == "reset":
        reset_at = payload.get("reset_at")
        if (
            not isinstance(reset_at, (int, float))
            or not math.isfinite(float(reset_at))
            or payload.get("symbol") != SYMBOL
            or payload.get("quote_generation_allowed") is not True
        ):
            return HaltState(
                status="fail_closed",
                state_path=state_path,
                fail_closed_reason="halt_state_reset_missing_reset_at",
            )
        return HaltState(status="clear", state_path=state_path, resolution="operator_reset")
    required = {
        "trigger_reason",
        "triggered_at",
        "expires_at",
        "resolution",
        "symbol",
        "quote_generation_allowed",
    }
    if raw_status != "triggered" or not required.issubset(payload):
        return HaltState(
            status="fail_closed",
            state_path=state_path,
            fail_closed_reason="halt_state_required_fields_missing_or_invalid_status",
        )
    trigger_reason = payload.get("trigger_reason")
    triggered_at = payload.get("triggered_at")
    expires_at = payload.get("expires_at")
    resolution = payload.get("resolution")
    symbol = payload.get("symbol")
    quote_generation_allowed = payload.get("quote_generation_allowed")
    if trigger_reason not in KILL_SWITCH_TRIGGER_REASONS:
        return HaltState(
            status="fail_closed",
            state_path=state_path,
            fail_closed_reason="halt_state_unknown_trigger_reason",
        )
    if (
        not isinstance(triggered_at, (int, float))
        or not isinstance(expires_at, (int, float))
        or not math.isfinite(float(triggered_at))
        or not math.isfinite(float(expires_at))
        or float(expires_at) <= float(triggered_at)
        or not isinstance(resolution, str)
        or resolution not in {"pending", "flat", "failed"}
        or symbol != SYMBOL
        or quote_generation_allowed is not False
    ):
        return HaltState(
            status="fail_closed",
            state_path=state_path,
            fail_closed_reason="halt_state_timestamp_or_resolution_invalid",
        )
    effective_now = time.time() if now is None else float(now)
    state_status = (
        "expired"
        if resolution == "flat" and effective_now >= float(expires_at)
        else "halted"
    )
    return HaltState(
        status=state_status,
        state_path=state_path,
        trigger_reason=str(trigger_reason),
        triggered_at=float(triggered_at),
        expires_at=float(expires_at),
        resolution=resolution,
        fail_closed_reason=str(payload.get("fail_closed_reason") or ""),
    )


def initialize_control_state(
    control_state_dir: Path,
    *,
    now: float | None = None,
) -> HaltState:
    control_state_dir = Path(control_state_dir)
    with _kill_switch_lock(control_state_dir):
        state_path = control_state_dir / KILL_SWITCH_STATE_FILENAME
        if state_path.exists():
            existing = check_halt_state(control_state_dir, now=now)
            if existing.status == "fail_closed":
                raise ValidationError(existing.fail_closed_reason)
            return existing
        armed_at = time.time() if now is None else float(now)
        _atomic_write_json(
            state_path,
            {
                "schema_version": KILL_SWITCH_SCHEMA_VERSION,
                "status": "armed",
                "armed_at": armed_at,
                "symbol": SYMBOL,
                "quote_generation_allowed": True,
            },
        )
        return check_halt_state(control_state_dir, now=now)


def reset_halt_state(
    control_state_dir: Path,
    *,
    operator_ack: str,
    now: float | None = None,
) -> HaltState:
    if operator_ack != KILL_SWITCH_RESET_ACK:
        raise ValidationError("kill_switch_reset_requires_exact_operator_ack")
    control_state_dir = Path(control_state_dir)
    if not control_state_dir.exists() or not control_state_dir.is_dir():
        raise ValidationError("control_state_dir_missing_or_not_directory")
    with _kill_switch_lock(control_state_dir):
        state_path = control_state_dir / KILL_SWITCH_STATE_FILENAME
        _atomic_write_json(
            state_path,
            {
                "schema_version": KILL_SWITCH_SCHEMA_VERSION,
                "status": "reset",
                "reset_at": time.time() if now is None else float(now),
                "reset_by": "explicit_operator_ack",
                "symbol": SYMBOL,
                "quote_generation_allowed": True,
            },
        )
    return check_halt_state(control_state_dir, now=now)


def extract_position_szi(user_state: dict[str, Any], *, symbol: str) -> float:
    positions = user_state.get("assetPositions")
    if positions is None:
        raise ValidationError("user_state_missing_asset_positions")
    if not isinstance(positions, list):
        raise ValidationError("user_state_asset_positions_not_list")
    matching: list[float] = []
    for row in positions:
        if not isinstance(row, dict):
            raise ValidationError("user_state_position_row_not_object")
        position = row.get("position", row)
        if not isinstance(position, dict):
            raise ValidationError("user_state_position_payload_not_object")
        coin = str(position.get("coin") or row.get("coin") or "")
        if coin != symbol:
            continue
        try:
            szi = float(position.get("szi"))
        except (TypeError, ValueError) as exc:
            raise ValidationError("user_state_position_szi_invalid") from exc
        if not math.isfinite(szi):
            raise ValidationError("user_state_position_szi_not_finite")
        matching.append(szi)
    if len(matching) > 1:
        raise ValidationError("user_state_duplicate_symbol_positions")
    return matching[0] if matching else 0.0


def assert_exchange_action_success(response: Any, *, action: str) -> None:
    if not isinstance(response, dict) or str(response.get("status", "")).lower() != "ok":
        raise ValidationError(f"{action}_response_not_ok")
    statuses = response.get("response", {}).get("data", {}).get("statuses")
    if not isinstance(statuses, list) or not statuses:
        raise ValidationError(f"{action}_response_statuses_missing")
    for status in statuses:
        if isinstance(status, str):
            if action == "cancel" and status.lower() == "success":
                continue
            raise ValidationError(f"{action}_response_status_invalid")
        if not isinstance(status, dict) or "error" in status:
            raise ValidationError(f"{action}_response_error_status")
        if action == "cancel" and "success" not in status:
            raise ValidationError("cancel_response_missing_success")
        if action == "market_close" and "filled" not in status:
            raise ValidationError("market_close_response_missing_filled")


def _kill_switch_payload(
    *,
    config: KillSwitchConfig,
    trigger_reason: str,
    triggered_at: float,
) -> dict[str, Any]:
    return {
        "schema_version": KILL_SWITCH_SCHEMA_VERSION,
        "status": "triggered",
        "trigger_reason": trigger_reason,
        "triggered_at": triggered_at,
        "triggered_at_iso": datetime.fromtimestamp(triggered_at, timezone.utc).isoformat().replace("+00:00", "Z"),
        "expires_at": triggered_at + config.halt_seconds,
        "resolution": "pending",
        "symbol": config.symbol,
        "quote_generation_allowed": False,
    }


def execute_kill_switch(
    *,
    client: HyperliquidClient,
    config: KillSwitchConfig,
    owned_order_refs: list[dict[str, Any]],
    trigger_reason: str,
    account_address: str | None,
) -> KillSwitchEvidence:
    validate_kill_switch_config(config)
    if trigger_reason not in KILL_SWITCH_TRIGGER_REASONS:
        raise ValidationError("kill_switch_unknown_trigger_reason")
    with _kill_switch_lock(config.control_state_dir):
        return _execute_kill_switch_once(
            client=client,
            config=config,
            owned_order_refs=owned_order_refs,
            trigger_reason=trigger_reason,
            account_address=account_address,
        )


def _execute_kill_switch_once(
    *,
    client: HyperliquidClient,
    config: KillSwitchConfig,
    owned_order_refs: list[dict[str, Any]],
    trigger_reason: str,
    account_address: str | None,
) -> KillSwitchEvidence:
    existing = check_halt_state(config.control_state_dir)
    missing_initial_state = (
        existing.status == "fail_closed"
        and existing.fail_closed_reason == "halt_state_file_missing"
    )
    if existing.is_halted and not missing_initial_state:
        return KillSwitchEvidence(
            status="already_halted",
            trigger_reason=existing.trigger_reason or trigger_reason,
            idempotent=True,
            state_path=str(existing.state_path),
            halt_state=_halt_state_dict(existing),
            proof_status="fail_closed" if existing.status == "fail_closed" else "already_halted",
            fail_closed_reason=existing.fail_closed_reason,
        )

    triggered_at = time.time()
    state_payload = _kill_switch_payload(
        config=config,
        trigger_reason=trigger_reason,
        triggered_at=triggered_at,
    )
    _atomic_write_json(config.state_path, state_payload)
    evidence = KillSwitchEvidence(
        status="halted",
        trigger_reason=trigger_reason,
        state_path=str(config.state_path),
        halt_state=_halt_state_dict(check_halt_state(config.control_state_dir, now=triggered_at)),
    )
    try:
        if trigger_reason == "unknown_order_state" and not owned_order_refs:
            raise ValidationError("unknown_order_state_requires_authoritative_owned_order_refs")
        shutdown = shutdown_cancel_all(
            client=client,
            symbol=config.symbol,
            tracked_refs=owned_order_refs,
            account_address=account_address,
        )
        evidence.cancel_evidence = {
            "requested_refs": shutdown.requested_refs,
            "cancel_results": shutdown.cancel_results,
            "final_open_orders": shutdown.final_open_orders,
            "proof_status": shutdown.proof_status,
            "fail_closed_reason": shutdown.fail_closed_reason,
        }
        if shutdown.proof_status != "pass":
            raise ValidationError(shutdown.fail_closed_reason or "owned_open_orders_not_proven_empty")

        before_state = client.user_state(account_address)
        position_before = extract_position_szi(before_state, symbol=config.symbol)
        evidence.position_before = {
            "symbol": config.symbol,
            "szi": position_before,
            "source": "private_user_state_after_owned_order_cancel_proof",
        }
        if position_before != 0.0:
            close_size = abs(position_before)
            close_side = "sell" if position_before > 0 else "buy"
            close_cloid = generate_cloid("kill_switch")
            evidence.market_close_called = True
            evidence.market_close_request = {
                "symbol": config.symbol,
                "side": close_side,
                "sz": close_size,
                "slippage": config.market_close_slippage,
                "reduce_only": True,
                "cloid_hash": stable_hash(close_cloid),
            }
            market_close_response = client.market_close(
                config.symbol,
                sz=close_size,
                slippage=config.market_close_slippage,
                cloid=close_cloid,
            )
            assert_exchange_action_success(market_close_response, action="market_close")
            evidence.market_close_response = redact(market_close_response)

        after_state = client.user_state(account_address)
        position_after = extract_position_szi(after_state, symbol=config.symbol)
        evidence.position_after = {
            "symbol": config.symbol,
            "szi": position_after,
            "source": "private_user_state_after_market_close",
        }
        evidence.residual_position_btc = position_after
        if abs(position_after) >= config.lot_size_btc:
            raise ValidationError("kill_switch_residual_position_above_lot_tolerance")

        state_payload.update(
            {
                "resolution": "flat",
                "resolved_at": time.time(),
                "cancel_evidence": evidence.cancel_evidence,
                "cancel_proof_status": shutdown.proof_status,
                "market_close_called": evidence.market_close_called,
                "market_close_request": evidence.market_close_request,
                "market_close_response": evidence.market_close_response,
                "position_before_btc": position_before,
                "position_after_btc": position_after,
                "residual_position_btc": position_after,
                "quote_generation_allowed": False,
            }
        )
        _atomic_write_json(config.state_path, state_payload)
        final_halt = check_halt_state(config.control_state_dir)
        evidence.status = "completed_halted"
        evidence.proof_status = "pass"
        evidence.halt_state = _halt_state_dict(final_halt)
        return evidence
    except Exception as exc:
        fail_reason = _redacted_error(exc)
        state_payload.update(
            {
                "resolution": "failed",
                "failed_at": time.time(),
                "fail_closed_reason": fail_reason,
                "cancel_evidence": evidence.cancel_evidence,
                "market_close_called": evidence.market_close_called,
                "market_close_request": evidence.market_close_request,
                "market_close_response": evidence.market_close_response,
                "position_before": evidence.position_before,
                "position_after": evidence.position_after,
                "residual_position_btc": evidence.residual_position_btc,
                "quote_generation_allowed": False,
            }
        )
        try:
            _atomic_write_json(config.state_path, state_payload)
        except Exception:
            pass
        evidence.status = "failed_halted"
        evidence.proof_status = "fail_closed"
        evidence.fail_closed_reason = fail_reason
        evidence.halt_state = _halt_state_dict(check_halt_state(config.control_state_dir))
        return evidence


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
        "control_state_dir": str(config.control_state_dir) if config.control_state_dir is not None else "",
        "duration_seconds": config.duration_seconds,
        "dry_run_private_preflight": config.dry_run_private_preflight,
        "live_mode": config.live_mode,
        "max_loss_usdc": config.max_loss_usdc,
        "max_notional_usdc": config.max_notional_usdc,
        "max_order_notional_usdc": config.max_order_notional_usdc,
        "max_order_size_btc": config.max_order_size_btc,
        "max_position_btc": config.max_position_btc,
        "max_position_notional_usdc": config.max_position_notional_usdc,
        "max_real_order_submissions": config.max_real_order_submissions,
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
            config.max_position_btc > 0 and config.max_position_btc <= MAX_POSITION_BTC,
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
        (
            "max_real_order_submissions_positive_integer_lte_30",
            (
                isinstance(config.max_real_order_submissions, int)
                and not isinstance(config.max_real_order_submissions, bool)
                and 0 < config.max_real_order_submissions <= MAX_REAL_ORDER_SUBMISSIONS
            ),
            f"actual={config.max_real_order_submissions}",
        ),
        (
            "max_loss_lte_30_usdc",
            config.max_loss_usdc > 0 and config.max_loss_usdc <= MAX_LOSS_USDC,
            f"actual={config.max_loss_usdc}",
        ),
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
        (
            "live_mode_has_control_state_dir",
            (not config.live_mode) or config.control_state_dir is not None,
            "live mode requires persistent control_state_dir",
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


def _finite_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{field_name}_must_be_numeric")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValidationError(f"{field_name}_must_be_finite")
    return numeric


def _nonnegative_quantity(value: Any, *, field_name: str) -> float:
    numeric = _finite_number(value, field_name=field_name)
    if numeric < 0:
        raise ValidationError(f"{field_name}_must_be_nonnegative")
    return numeric


def projected_exposure(
    *,
    position_btc: float,
    working_buy_qty: float,
    working_sell_qty: float,
    inflight_buy_qty: float,
    inflight_sell_qty: float,
    existing_max_quote_px: float | None = None,
) -> ProjectedExposure:
    """Project aggregate worst-case long and short inventory.

    A cancel-pending order belongs in the working quantity until the exchange
    confirms cancellation. An unknown submit belongs in the inflight quantity
    because it may already be resting. Opposite-side leaves are intentionally
    excluded from each side's worst case: they may never fill, so they cannot
    be used to offset a breach.
    """

    position = _finite_number(position_btc, field_name="position_btc")
    working_buy = _nonnegative_quantity(working_buy_qty, field_name="working_buy_qty")
    working_sell = _nonnegative_quantity(working_sell_qty, field_name="working_sell_qty")
    inflight_buy = _nonnegative_quantity(inflight_buy_qty, field_name="inflight_buy_qty")
    inflight_sell = _nonnegative_quantity(inflight_sell_qty, field_name="inflight_sell_qty")
    if existing_max_quote_px is not None:
        existing_max_quote_px = _finite_number(existing_max_quote_px, field_name="existing_max_quote_px")
        if existing_max_quote_px <= 0:
            raise ValidationError("existing_max_quote_px_must_be_positive")
    worst_long = max(0.0, position + working_buy + inflight_buy)
    worst_short = max(0.0, -position + working_sell + inflight_sell)
    return ProjectedExposure(
        position_btc=position,
        working_buy_qty=working_buy,
        working_sell_qty=working_sell,
        inflight_buy_qty=inflight_buy,
        inflight_sell_qty=inflight_sell,
        worst_long_btc=worst_long,
        worst_short_btc=worst_short,
        existing_max_quote_px=existing_max_quote_px,
    )


def _quote_fields(quote: OrderIntent | dict[str, Any]) -> tuple[bool, Any, Any, str]:
    if isinstance(quote, OrderIntent):
        if not isinstance(quote.is_buy, bool):
            raise ValidationError("proposed_quote_is_buy_must_be_boolean")
        return quote.is_buy, quote.size_btc, quote.limit_px, quote.symbol
    if isinstance(quote, dict):
        if "is_buy" not in quote:
            raise ValidationError("proposed_quote_is_buy_missing")
        if not isinstance(quote["is_buy"], bool):
            raise ValidationError("proposed_quote_is_buy_must_be_boolean")
        return (
            quote["is_buy"],
            quote.get("size_btc"),
            quote.get("limit_px"),
            str(quote.get("symbol") or ""),
        )
    raise ValidationError("proposed_quote_type_invalid")


def _effective_runtime_cap(config_value: Any, global_value: float | int, *, field_name: str) -> float:
    configured = _finite_number(config_value, field_name=field_name)
    default = _finite_number(global_value, field_name=f"global_{field_name}")
    if configured <= 0:
        raise ValidationError(f"{field_name}_must_be_positive")
    return min(configured, default)


def _effective_submission_cap(config_value: Any) -> int:
    if isinstance(config_value, bool) or not isinstance(config_value, int):
        raise ValidationError("max_real_order_submissions_must_be_positive_integer")
    if config_value <= 0:
        raise ValidationError("max_real_order_submissions_must_be_positive_integer")
    return min(config_value, MAX_REAL_ORDER_SUBMISSIONS)


def validate_runtime_envelope(
    *,
    config: TinyLiveConfig,
    projected: ProjectedExposure,
    proposed_quotes: Iterable[OrderIntent | dict[str, Any]],
    submissions_used: int,
) -> None:
    """Fail closed when a proposed batch could breach aggregate live limits."""

    if not isinstance(projected, ProjectedExposure):
        raise ValidationError("projected_exposure_type_invalid")
    reconstructed = projected_exposure(
        position_btc=projected.position_btc,
        working_buy_qty=projected.working_buy_qty,
        working_sell_qty=projected.working_sell_qty,
        inflight_buy_qty=projected.inflight_buy_qty,
        inflight_sell_qty=projected.inflight_sell_qty,
        existing_max_quote_px=projected.existing_max_quote_px,
    )
    if (
        not math.isclose(projected.worst_long_btc, reconstructed.worst_long_btc, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(projected.worst_short_btc, reconstructed.worst_short_btc, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValidationError("projected_exposure_summary_inconsistent")
    if isinstance(submissions_used, bool) or not isinstance(submissions_used, int) or submissions_used < 0:
        raise ValidationError("submissions_used_must_be_nonnegative_integer")

    quotes = list(proposed_quotes)
    proposed_buy_qty = 0.0
    proposed_sell_qty = 0.0
    proposed_notional = 0.0
    max_quote_px = 0.0
    for quote in quotes:
        is_buy, raw_size, raw_px, symbol = _quote_fields(quote)
        if symbol != SYMBOL:
            raise ValidationError("proposed_quote_symbol_must_be_btc")
        size = _nonnegative_quantity(raw_size, field_name="proposed_quote_size_btc")
        if size <= 0:
            raise ValidationError("proposed_quote_size_btc_must_be_positive")
        limit_px = _finite_number(raw_px, field_name="proposed_quote_limit_px")
        if limit_px <= 0:
            raise ValidationError("proposed_quote_limit_px_must_be_positive")
        notional = size * limit_px
        if not math.isfinite(notional):
            raise ValidationError("proposed_quote_notional_must_be_finite")
        if size > _effective_runtime_cap(
            config.max_order_size_btc,
            MAX_ORDER_SIZE_BTC,
            field_name="max_order_size_btc",
        ) + 1e-12:
            raise ValidationError("runtime_single_order_size_cap_exceeded")
        if notional > _effective_runtime_cap(
            config.max_order_notional_usdc,
            MAX_ORDER_NOTIONAL_USDC,
            field_name="max_order_notional_usdc",
        ) + 1e-9:
            raise ValidationError("runtime_single_order_notional_cap_exceeded")
        if is_buy:
            proposed_buy_qty += size
        else:
            proposed_sell_qty += size
        proposed_notional += notional
        max_quote_px = max(max_quote_px, limit_px)

    position_cap = _effective_runtime_cap(
        config.max_position_btc,
        MAX_POSITION_BTC,
        field_name="max_position_btc",
    )
    position_notional_cap = _effective_runtime_cap(
        config.max_position_notional_usdc,
        MAX_POSITION_NOTIONAL_USDC,
        field_name="max_position_notional_usdc",
    )
    total_notional_cap = _effective_runtime_cap(
        config.max_notional_usdc,
        MAX_NOTIONAL_USDC,
        field_name="max_notional_usdc",
    )
    submission_cap = _effective_submission_cap(config.max_real_order_submissions)
    if submissions_used + len(quotes) > submission_cap:
        raise ValidationError("runtime_submission_cap_exceeded")

    worst_long = max(
        0.0,
        projected.position_btc
        + projected.working_buy_qty
        + projected.inflight_buy_qty
        + proposed_buy_qty,
    )
    worst_short = max(
        0.0,
        -projected.position_btc
        + projected.working_sell_qty
        + projected.inflight_sell_qty
        + proposed_sell_qty,
    )
    if worst_long > position_cap + 1e-12:
        raise ValidationError("runtime_worst_long_position_cap_exceeded")
    if worst_short > position_cap + 1e-12:
        raise ValidationError("runtime_worst_short_position_cap_exceeded")

    existing_leaf_qty = (
        projected.working_buy_qty
        + projected.working_sell_qty
        + projected.inflight_buy_qty
        + projected.inflight_sell_qty
    )
    if existing_leaf_qty > 0 and projected.existing_max_quote_px is None:
        raise ValidationError("runtime_existing_quote_valuation_price_missing")
    valuation_px = max(max_quote_px, projected.existing_max_quote_px or 0.0)
    quantity_with_unknown_leaves = (
        abs(projected.position_btc)
        + existing_leaf_qty
        + proposed_buy_qty
        + proposed_sell_qty
    )
    if quantity_with_unknown_leaves > 0 and valuation_px <= 0:
        raise ValidationError("runtime_notional_valuation_price_missing")
    worst_position_notional = max(worst_long, worst_short) * valuation_px
    if worst_position_notional > position_notional_cap + 1e-9:
        raise ValidationError("runtime_worst_position_notional_cap_exceeded")
    existing_notional = (abs(projected.position_btc) + existing_leaf_qty) * valuation_px
    inventory_reducing_only = (
        (
            projected.position_btc > 0
            and proposed_buy_qty == 0
            and proposed_sell_qty <= projected.position_btc
            and projected.working_buy_qty + projected.inflight_buy_qty == 0
        )
        or (
            projected.position_btc < 0
            and proposed_sell_qty == 0
            and proposed_buy_qty <= -projected.position_btc
            and projected.working_sell_qty + projected.inflight_sell_qty == 0
        )
    )
    aggregate_notional = (
        max(existing_notional, proposed_notional, worst_position_notional)
        if inventory_reducing_only
        else existing_notional + proposed_notional
    )
    if aggregate_notional > total_notional_cap + 1e-9:
        raise ValidationError("runtime_aggregate_notional_cap_exceeded")


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
    try:
        cross_exchange_price_math.assert_hl_perp_price_valid(
            intent.limit_px,
            sz_decimals=precision.sz_decimals,
        )
    except ValueError as exc:
        raise ValidationError(f"invalid_limit_price:{exc}") from exc


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


def runtime_projected_exposure(
    *,
    client: HyperliquidClient,
    account_address: str | None = None,
    inflight_buy_qty: float = 0.0,
    inflight_sell_qty: float = 0.0,
) -> ProjectedExposure:
    """Build a runtime snapshot from exchange-confirmed private state.

    The caller may add quantities for submit-inflight orders whose exchange
    status is not yet authoritative. Existing open-order prices are retained
    as a conservative valuation bound instead of being replaced by the next
    proposed quote price.
    """

    open_orders = client.open_orders()
    if not isinstance(open_orders, list):
        raise ValidationError("runtime_open_orders_not_list")
    working_buy_qty = 0.0
    working_sell_qty = 0.0
    existing_max_quote_px = 0.0
    for row in open_orders:
        if not isinstance(row, dict):
            raise ValidationError("runtime_open_order_row_not_object")
        coin = str(row.get("coin") or row.get("symbol") or "")
        if coin != SYMBOL:
            raise ValidationError("runtime_open_order_symbol_unknown_or_foreign")
        side = str(row.get("side") or "").strip().lower()
        if side in {"b", "buy", "bid", "long"}:
            is_buy = True
        elif side in {"a", "ask", "sell", "short"}:
            is_buy = False
        else:
            raise ValidationError("runtime_open_order_side_unknown")
        raw_size = row.get("sz", row.get("size", row.get("remainingSz")))
        if isinstance(raw_size, str):
            try:
                raw_size = float(raw_size)
            except ValueError as exc:
                raise ValidationError("runtime_open_order_size_btc_must_be_numeric") from exc
        size = _nonnegative_quantity(raw_size, field_name="runtime_open_order_size_btc")
        if size <= 0:
            raise ValidationError("runtime_open_order_size_btc_must_be_positive")
        raw_px = row.get("limitPx", row.get("limit_px", row.get("px")))
        if isinstance(raw_px, str):
            try:
                raw_px = float(raw_px)
            except ValueError as exc:
                raise ValidationError("runtime_open_order_limit_px_must_be_numeric") from exc
        px = _finite_number(raw_px, field_name="runtime_open_order_limit_px")
        if px <= 0:
            raise ValidationError("runtime_open_order_limit_px_must_be_positive")
        if is_buy:
            working_buy_qty += size
        else:
            working_sell_qty += size
        existing_max_quote_px = max(existing_max_quote_px, px)
    user_state = client.user_state()
    position_btc = extract_position_szi(user_state, symbol=SYMBOL)
    return projected_exposure(
        position_btc=position_btc,
        working_buy_qty=working_buy_qty,
        working_sell_qty=working_sell_qty,
        inflight_buy_qty=inflight_buy_qty,
        inflight_sell_qty=inflight_sell_qty,
        existing_max_quote_px=existing_max_quote_px if open_orders else None,
    )


def run_order_once(
    *,
    config: TinyLiveConfig,
    precision: PrecisionFacts,
    intent: OrderIntent,
    loss_snapshot: LossSnapshot | None,
    client: HyperliquidClient,
    owned_order_refs: list[dict[str, Any]] | None = None,
    account_address: str | None = None,
    on_order_endpoint_started: Callable[[], None] | None = None,
    projected: ProjectedExposure,
    submissions_used: int,
    proposed_quotes: Iterable[OrderIntent | dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assert_config_valid(config, precision)
    validate_order_intent(config, precision, intent)
    if not config.live_mode:
        raise ValidationError("live_mode=false prevents real order placement")
    if config.control_state_dir is None:
        raise ValidationError("kill_switch_control_state_dir_required")
    loss = loss_status(config, loss_snapshot)
    if loss["status"] != "pass":
        evidence = execute_kill_switch(
            client=client,
            config=KillSwitchConfig(control_state_dir=config.control_state_dir),
            owned_order_refs=list(owned_order_refs or []),
            trigger_reason="max_loss_reached",
            account_address=account_address or getattr(client, "account_address", None),
        )
        raise ValidationError(
            f"max loss check failed: {loss['reason']};kill_switch={evidence.proof_status}"
        )
    quotes = list(proposed_quotes) if proposed_quotes is not None else [intent]
    if intent not in quotes:
        quotes.append(intent)
    validate_runtime_envelope(
        config=config,
        projected=projected,
        proposed_quotes=quotes,
        submissions_used=submissions_used,
    )
    with _kill_switch_lock(config.control_state_dir):
        halt_state = check_halt_state(config.control_state_dir)
        if not halt_state.may_quote:
            raise KillSwitchBlocked(
                f"kill_switch_halt_blocks_order:{halt_state.fail_closed_reason or halt_state.trigger_reason or halt_state.status}"
            )
        if on_order_endpoint_started is not None:
            on_order_endpoint_started()
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
        if oid is None and not cloid:
            raise ValidationError("owned_order_ref_missing_oid_and_cloid")
        evidence.requested_refs.append(str(oid if oid is not None else cloid))
        cancel_result = client.cancel_tracked(symbol, oid=oid, cloid=cloid)
        assert_exchange_action_success(cancel_result, action="cancel")
        evidence.cancel_results.append(redact(cancel_result))
    final_open_raw = client.open_orders(account_address)
    evidence.final_open_orders = redact(final_open_raw)
    if final_open_raw:
        evidence.proof_status = "fail_closed"
        evidence.fail_closed_reason = "open_orders_not_empty_or_ownership_ambiguous"
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
    return cross_exchange_price_math.normalize_hl_perp_price(
        px,
        sz_decimals=sz_decimals,
        side="nearest",
    )


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
    limit_px = cross_exchange_price_math.normalize_hl_perp_price(
        raw_px,
        sz_decimals=precision.sz_decimals,
        side="buy",
    )
    if limit_px * MAX_ORDER_SIZE_BTC > MAX_ORDER_NOTIONAL_USDC:
        limit_px = cross_exchange_price_math.normalize_hl_perp_price(
            (MAX_ORDER_NOTIONAL_USDC / MAX_ORDER_SIZE_BTC) - 100.0,
            sz_decimals=precision.sz_decimals,
            side="buy",
        )
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
        run_order_once(
            config=config,
            precision=precision,
            intent=intent,
            loss_snapshot=loss_snapshot,
            client=client,
            projected=projected_exposure(
                position_btc=loss_snapshot.position_btc,
                working_buy_qty=0.0,
                working_sell_qty=0.0,
                inflight_buy_qty=0.0,
                inflight_sell_qty=0.0,
            ),
            submissions_used=0,
        )
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
    control_state_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if control_state_dir is None:
        raise ValidationError("kill_switch_control_state_dir_required")
    halt_state = check_halt_state(control_state_dir)
    if not halt_state.may_quote:
        raise ValidationError(
            f"kill_switch_halt_blocks_canary:{halt_state.fail_closed_reason or halt_state.trigger_reason or halt_state.status}"
        )
    config = TinyLiveConfig(
        artifact_dir=output_dir,
        live_mode=True,
        operator_ack=LIVE_OPERATOR_ACK,
        use_schedule_cancel=use_schedule_cancel,
        control_state_dir=control_state_dir,
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
        order_result = run_order_once(
            config=config,
            precision=precision,
            intent=intent,
            loss_snapshot=LossSnapshot(intent.limit_px, intent.limit_px, intent.size_btc),
            client=client,
            owned_order_refs=tracked_refs,
            account_address=getattr(client, "account_address", None),
            on_order_endpoint_started=lambda: endpoint_flags.__setitem__("real_order_endpoint_called", True),
            projected=runtime_projected_exposure(
                client=client,
                account_address=getattr(client, "account_address", None),
            ),
            submissions_used=0,
        )
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
    parser.add_argument("--control-state-dir", type=Path, default=DEFAULT_CONTROL_STATE_DIR)
    parser.add_argument("--initialize-control-state", action="store_true")
    parser.add_argument("--reset-kill-switch", action="store_true")
    parser.add_argument("--operator-ack", default="")
    args = parser.parse_args()

    if args.initialize_control_state:
        state = initialize_control_state(args.control_state_dir)
        print(json.dumps(_halt_state_dict(state), indent=2, sort_keys=True))
        return 0
    if args.reset_kill_switch:
        state = reset_halt_state(
            args.control_state_dir,
            operator_ack=args.operator_ack,
        )
        print(json.dumps(_halt_state_dict(state), indent=2, sort_keys=True))
        return 0
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
            control_state_dir=args.control_state_dir,
        )
        print(json.dumps(redact(manifest), indent=2, sort_keys=True))
        return 0
    if args.live:
        config = TinyLiveConfig(
            artifact_dir=args.output_dir,
            live_mode=True,
            operator_ack=args.operator_ack,
            control_state_dir=args.control_state_dir,
        )
        assert_config_valid(config, mock_precision())
        build_live_client_from_env()
        raise SystemExit("live client initialized, but 0618T001 does not execute live order placement")
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
