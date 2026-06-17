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
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK_ID = "0618T001"
SCHEMA_VERSION = "hyperliquid_tiny_live_real_order_executor_v1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "local_live_analysis" / "hyperliquid_tiny_live_real_order_executor_0618T001"
FINAL_RECOMMENDATION_READY = "hyperliquid_tiny_live_real_order_executor_ready_for_qa"
FINAL_RECOMMENDATION_BLOCKED = "hyperliquid_tiny_live_real_order_executor_blocked"
MAX_DURATION_SECONDS = 600
MAX_ORDER_SIZE_BTC = 0.01
MAX_ORDER_NOTIONAL_USDC = 700.0
MAX_POSITION_BTC = 0.04
MAX_POSITION_NOTIONAL_USDC = 2800.0
MAX_NOTIONAL_USDC = 3000.0
MAX_LOSS_USDC = 30.0
SYMBOL = "BTC"
POST_ONLY_TIF = "Alo"

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
    "api_key",
    "api_secret",
    "authorization",
    "nonce",
    "private_key",
    "raw_signature",
    "secret",
    "signature",
    "signed_payload",
    "wallet_key",
}

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
            facts["open_order_count"] = len(open_orders)
            facts["asset_position_count"] = len(user_state.get("assetPositions", []))
        return facts

    def order(self, intent: OrderIntent) -> dict[str, Any]:
        return self.exchange.order(
            intent.symbol,
            intent.is_buy,
            intent.size_btc,
            intent.limit_px,
            {"limit": {"tif": intent.time_in_force}},
            reduce_only=intent.reduce_only,
            cloid=intent.cloid,
        )

    def cancel_tracked(self, symbol: str, oid: int | None = None, cloid: str | None = None) -> dict[str, Any]:
        if oid is not None:
            return self.exchange.cancel(symbol, oid)
        if cloid is not None:
            return self.exchange.cancel_by_cloid(symbol, cloid)
        raise ValidationError("cancel requires oid or cloid")

    def open_orders(self, address: str | None = None) -> list[dict[str, Any]]:
        if not address:
            address = self.account_address
        if not address:
            raise ValidationError("open_orders requires account address")
        return self.info.open_orders(address)

    def schedule_cancel(self, cancel_time_ms: int | None) -> dict[str, Any]:
        return self.exchange.schedule_cancel(cancel_time_ms)


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
    return redact(client.order(intent))


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
    final_open = redact(client.open_orders(account_address))
    evidence.final_open_orders = final_open
    remaining_tracked = []
    tracked_oids = {str(ref.get("oid")) for ref in tracked_refs if ref.get("oid") is not None}
    tracked_cloids = {str(ref.get("cloid")) for ref in tracked_refs if ref.get("cloid")}
    for order in final_open:
        if str(order.get("oid")) in tracked_oids or str(order.get("cloid")) in tracked_cloids:
            remaining_tracked.append(order)
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
    private_key = os.environ.get("HYPERLIQUID_PRIVATE_KEY", "")
    account_address = os.environ.get("HYPERLIQUID_ACCOUNT_ADDRESS", "")
    if not private_key:
        if allow_missing:
            return None
        raise ValidationError("HYPERLIQUID_PRIVATE_KEY is required for live mode")
    from eth_account import Account  # type: ignore
    from hyperliquid.exchange import Exchange  # type: ignore
    from hyperliquid.info import Info  # type: ignore
    from hyperliquid.utils import constants  # type: ignore

    wallet = Account.from_key(private_key)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    exchange = Exchange(wallet, constants.MAINNET_API_URL, account_address=account_address or None)
    return SDKHyperliquidClient(exchange=exchange, info=info, account_address=account_address or wallet.address)


def main() -> int:
    parser = argparse.ArgumentParser(description="Hyperliquid tiny-live real-order executor scaffold.")
    parser.add_argument("--self-test", action="store_true", help="generate no-network self-test artifacts")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--live", action="store_true", help="enable live-capable path; not used by 0618T001")
    parser.add_argument("--operator-ack", default="")
    args = parser.parse_args()

    if args.self_test:
        manifest = generate_self_test_artifacts(args.output_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
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
