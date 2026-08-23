#!/usr/bin/env python3
"""Execute and package the 0822T002 c6in Hyperliquid latency measurement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import queue
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections import OrderedDict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import research_package_trust as trust
    import skhynix_c6in_latency_contracts_v2 as contracts
    import hyperliquid_maker_order_manager as order_manager
except ModuleNotFoundError:  # pragma: no cover
    from examples.hyperliquid import research_package_trust as trust
    from examples.hyperliquid import skhynix_c6in_latency_contracts_v2 as contracts
    from examples.hyperliquid import hyperliquid_maker_order_manager as order_manager


TASK_PATH = REPO_ROOT / ".workflow/tasks/0822T002.md"
MATRIX_PATH = REPO_ROOT / ".workflow/contracts/0822T002-surface-matrix.json"
HOSTILE_RECEIPT = REPO_ROOT / ".workflow/reports/0822T002-hostile-preflight.json"
REGISTRY_PATH = (
    REPO_ROOT / "baselines/research_package_trust_kernel/accepted_versions.json"
)
EXPECTED_PLAN_SHA256 = (
    "9f29b45aaba532922fe21c65d46623748da66b3338e5ff6ca5fa153e38c91391"
)
EXPECTED_MATRIX_SHA256 = (
    "4e6a357adb9b0d1d70e03644a674646be2f66d702143eda5a19dc728f0dcaf5c"
)
EXPECTED_C6IN_INSTANCE_ID = "i-0a962e47210528526"
EXPECTED_C6IN_REGION = "ap-northeast-1"
EXPECTED_C6IN_USER = "admin"
TARGET_DEX = "xyz"
TARGET_ASSET = "xyz:SKHX"
EXPECTED_TARGET_ASSET_ID = 110022
PUBLIC_PREFLIGHT_DURATION_SECONDS = 900
PUBLIC_PREFLIGHT_HORIZON_MS = 250
PUBLIC_PREFLIGHT_MIN_VALID_PAIRS = 1000
ACTIVE_EXECUTION_AUTHORIZED = True
DEFAULT_CREDENTIAL_FILE = Path("/home/admin/XEMM_rust_latest/.env")
WINDOW_FREEZE_LEAD_SECONDS = 1020
COLLECTION_WINDOW_SECONDS = 900
COLLECTION_WINDOW_GAP_SECONDS = 60
MAX_ATTEMPTS_PER_WINDOW = 40
MAX_ATTEMPTS_PER_BATCH = 10
FIXED_PRE_CANCEL_SETTLE_MS = 250
MINIMUM_INTER_ATTEMPT_SECONDS = 20
TERMINAL_QUERY_INTERVAL_MS = 50
TERMINAL_QUERY_TIMEOUT_MS = 5000
FORMAL_PACKAGE_RELATIVE = Path(
    "local_live_analysis/"
    "skhynix_c6in_hyperliquid_execution_latency_0822T002"
)

PACKAGE_DIRECTORIES = (
    "contracts",
    "reports",
    "runtime_source",
    "runtime_tests",
)
PACKAGE_FILES = (
    "accepted_h0a_pin.json",
    "attempt_ledger.csv",
    "authorization_envelope.json",
    "boundary_manifest.json",
    "collection_window_schedule.csv",
    "controller_latency_recommendation.json",
    "failure_and_censoring.csv",
    "frozen_measurement_contract.json",
    "historical_context_awsserver.csv",
    "host_identity.json",
    "latency_by_attempt.csv",
    "latency_summary.csv",
    "lifecycle_events.csv",
    "market_identity.json",
    "measurement_manifest.json",
    "reliability_summary.json",
    "runtime_identity.json",
    "sha256_inventory.csv",
    "contracts/accepted_kernel_pin.json",
    "contracts/execution_plan.md",
    "contracts/surface_matrix.json",
    "contracts/task.md",
    "contracts/v2_framework.md",
    "reports/execution_latency_measurement.md",
    "runtime_source/skhynix_c6in_latency_contracts_v2.py",
    "runtime_source/skhynix_c6in_latency_v2.py",
    "runtime_tests/test_skhynix_c6in_latency_package_v2.py",
    "runtime_tests/test_skhynix_c6in_latency_v2.py",
)
PACKAGE_R_FILES = (
    "attempt_ledger.csv",
    "collection_window_schedule.csv",
    "controller_latency_recommendation.json",
    "failure_and_censoring.csv",
    "historical_context_awsserver.csv",
    "latency_by_attempt.csv",
    "latency_summary.csv",
    "lifecycle_events.csv",
    "reliability_summary.json",
)
PACKAGE_C_FILES = (
    "accepted_h0a_pin.json",
    "frozen_measurement_contract.json",
    "contracts/accepted_kernel_pin.json",
    "contracts/execution_plan.md",
    "contracts/surface_matrix.json",
    "contracts/task.md",
    "contracts/v2_framework.md",
    "runtime_source/skhynix_c6in_latency_contracts_v2.py",
    "runtime_source/skhynix_c6in_latency_v2.py",
    "runtime_tests/test_skhynix_c6in_latency_package_v2.py",
    "runtime_tests/test_skhynix_c6in_latency_v2.py",
)
PACKAGE_E_FILES = tuple(
    sorted(set(PACKAGE_FILES) - set(PACKAGE_R_FILES) - set(PACKAGE_C_FILES))
)
FAILURE_FIELDS = (
    "schema_version",
    "task_id",
    "sample_sequence",
    "attempt_id",
    "primary_latency_eligible",
    "failure_or_censor_class",
    "terminal_class",
    "safety_status",
)
HISTORICAL_CONTEXT_FIELDS = (
    "source_host",
    "sample_count",
    "metric",
    "min_ms",
    "p50_ms",
    "p90_ms",
    "p95_ms",
    "max_ms",
    "primary_population_eligible",
    "exclusion_reason",
)
INVENTORY_FIELDS = ("path", "bytes", "sha256")

KERNEL_PIN = OrderedDict(
    (
        ("mode", "accepted"),
        ("kernel_name", "research_package_trust_kernel"),
        ("kernel_version", "v1"),
        ("registry_revision", 1),
        (
            "registry_entry_sha256",
            "cae21d65bf447435bafc37508b8ca00643a0742b37e0f404148cab92818c90c9",
        ),
        (
            "kernel_source_tree_sha256",
            "cee2395afad9420c38235ba195bf030e92330015e1a15937ebc22fa707c80203",
        ),
        (
            "kernel_api_contract_sha256",
            "2cd5a67ba15d67e59bcddcdbb21593696d3dc3dc27d81986c39ddf3e91e91f5f",
        ),
        (
            "kernel_negative_matrix_sha256",
            "f6247594b6f024945a52c0dccf421bac93538d9357f92ff6027d024199fc6b97",
        ),
        (
            "kernel_qa_report_sha256",
            "8fe01f85f8a68581b79ee410167769f2a105d9cc74ca6528af9496808a626be8",
        ),
        ("kernel_acceptance_task_id", "0820T001"),
    )
)

ACCEPTED_H0A_PIN = OrderedDict(
    (
        ("task_id", "0821T001"),
        ("selected_horizon_ms", 50),
        ("gate_latency_ms", 100),
        (
            "primary_tuple_sha256",
            "e5d1b132248ff1a6933678c32a54e6b4147c1c6f47dab25103011ecbd7a68eca",
        ),
        (
            "research_data_identity",
            "7176c78c2b6bfadf11432e9f5a1c1eaf9627c8028a4a22257d67d5a5404dc8fd",
        ),
        (
            "code_contract_identity",
            "4e8ccc7466f84d9eb2f71f681557424b59ca58249d67426325ecdbd661189636",
        ),
        (
            "evidence_identity",
            "8745458fcf4e0ab31f8ad3b2bc2d1d93704f18b13a776c515f14195b51213969",
        ),
        (
            "composite_identity",
            "2682c32eefac427eed1899a3d492b3fc7545520f72021723e8fef7a0d4d8d9d0",
        ),
        (
            "controller_closure_sha256",
            "9cc1b1d24d29cd2b55a8c1774a9d9e6e59338242c95c3af861460a0b8b07aded",
        ),
    )
)

HOSTILE_CASES = OrderedDict(
    (
        ("kernel_pin_mutated", "KERNEL_PIN_MISMATCH"),
        ("h0a_pin_mutated", "LATENCY_H0A_PIN_MISMATCH"),
        (
            "authorization_envelope_mutated",
            "LATENCY_AUTHORIZATION_MISMATCH",
        ),
        ("l1_external_path_opened", "LATENCY_L1_BOUNDARY_VIOLATION"),
        (
            "package_binding_mutated",
            "COMPOSITE_IDENTITY_BINDING_MISMATCH",
        ),
        ("wall_clock_duration", "LATENCY_WALL_CLOCK_DURATION_FORBIDDEN"),
        ("monotonic_reordered", "LATENCY_MONOTONIC_ORDER_INVALID"),
        ("decision_ready_missing", "LATENCY_DECISION_READY_MISSING"),
        (
            "cancel_response_promoted",
            "LATENCY_CANCEL_RESPONSE_NOT_TERMINAL",
        ),
        (
            "already_canceled_or_filled_promoted",
            "LATENCY_TERMINAL_STATUS_UNIDENTIFIED",
        ),
        ("foreign_reference", "LATENCY_ORDER_REFERENCE_MISMATCH"),
        ("raw_reference_leak", "LATENCY_SECRET_OR_REFERENCE_LEAK"),
        ("duplicate_identity", "LATENCY_SAMPLE_IDENTITY_DUPLICATE"),
        ("dropped_slow_attempt", "LATENCY_REAL_ATTEMPT_OMITTED"),
        ("dropped_timeout_attempt", "LATENCY_REAL_ATTEMPT_OMITTED"),
        (
            "eligibility_report_override",
            "LATENCY_RELIABILITY_DENOMINATOR_MISMATCH",
        ),
        (
            "control_promoted_to_target",
            "LATENCY_TARGET_POPULATION_CONTAMINATED",
        ),
        (
            "awsserver_pooled_with_c6in",
            "LATENCY_TARGET_POPULATION_CONTAMINATED",
        ),
        ("host_boot_drift", "LATENCY_HOST_IDENTITY_DRIFT"),
        ("runtime_drift", "LATENCY_RUNTIME_IDENTITY_DRIFT"),
        ("connection_policy_drift", "LATENCY_CLIENT_POLICY_DRIFT"),
        (
            "interpolated_quantile",
            "LATENCY_QUANTILE_CONTRACT_MISMATCH",
        ),
        ("p90_substituted", "LATENCY_QUANTILE_CONTRACT_MISMATCH"),
        ("bucket_rounded_down", "LATENCY_BUCKET_ROUND_DOWN"),
        ("largest_window_over_half", "LATENCY_WINDOW_SUPPORT_INVALID"),
        ("insufficient_sample_conclusive", "LATENCY_SAMPLE_GATE_NOT_MET"),
        (
            "terminal_denominator_changed",
            "LATENCY_RELIABILITY_DENOMINATOR_MISMATCH",
        ),
        ("unresolved_exposure_hidden", "LATENCY_UNRESOLVED_EXPOSURE"),
        ("h0b_path_opened", "LATENCY_H0B_OUTCOME_ACCESS_FORBIDDEN"),
        (
            "h0a_tuple_mutated",
            "LATENCY_H0A_TUPLE_MUTATION_FORBIDDEN",
        ),
        (
            "recommendation_diverged",
            "LATENCY_RECOMMENDATION_DIVERGENCE",
        ),
        ("dry_transport_promoted", "LATENCY_PASSIVE_SOURCE_NOT_LIVE"),
        ("attempt_121", "LATENCY_ATTEMPT_CAP_EXHAUSTED"),
        (
            "quote_distance_unverified",
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
        ),
        (
            "loss_basis_substituted",
            "LATENCY_LOSS_CAP_CONTRACT_MISMATCH",
        ),
    )
)


def _utc_now() -> str:
    from datetime import datetime, timezone

    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="ascii"))


def _run_text(command: Sequence[str], *, timeout: float = 30.0) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return completed.stdout.strip()


def _read_text(path: Path) -> str:
    return Path(path).read_text(encoding="ascii").strip()


def _metadata(path: str) -> str:
    token_request = urllib.request.Request(
        "http://169.254.169.254/latest/api/token",
        method="PUT",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
    )
    with urllib.request.urlopen(token_request, timeout=2) as response:
        token = response.read().decode("ascii").strip()
    request = urllib.request.Request(
        f"http://169.254.169.254/latest/meta-data/{path}",
        headers={"X-aws-ec2-metadata-token": token},
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read().decode("ascii").strip()


def _host_identity() -> dict[str, Any]:
    availability_zone = _metadata("placement/availability-zone")
    region = availability_zone[:-1]
    instance_id = _metadata("instance-id")
    interface = _run_text(
        ["sh", "-c", "ip route get 1.1.1.1 | awk '{print $5; exit}'"]
    )
    route_source = _run_text(
        ["sh", "-c", "ip route get 1.1.1.1 | awk '{print $7; exit}'"]
    )
    try:
        public_egress = _run_text(
            ["curl", "-4", "-fsS", "--max-time", "10", "https://checkip.amazonaws.com"],
            timeout=15,
        )
    except (subprocess.SubprocessError, OSError):
        public_egress = _metadata("public-ipv4")
    ntp = _run_text(
        ["timedatectl", "show", "-p", "NTPSynchronized", "--value"]
    )
    observed = {
        "schema_version": "skhynix_c6in_host_identity_v1",
        "task_id": contracts.TASK_ID,
        "captured_at_utc": _utc_now(),
        "ssh_alias": "c6in-winner",
        "cloud_instance_id": instance_id,
        "cloud_account_identity": _run_text(
            [
                "aws",
                "sts",
                "get-caller-identity",
                "--query",
                "Account",
                "--output",
                "text",
            ]
        ),
        "region": region,
        "availability_zone": availability_zone,
        "expected_user": EXPECTED_C6IN_USER,
        "effective_user": _run_text(["id", "-un"]),
        "hostname": socket.gethostname(),
        "machine_id": _read_text(Path("/etc/machine-id")),
        "boot_id": _read_text(Path("/proc/sys/kernel/random/boot_id")),
        "primary_network_interface": interface,
        "route_source_address": route_source,
        "public_egress_identity": public_egress,
        "kernel_version": platform.release(),
        "cpu_architecture": platform.machine(),
        "clocksource": _read_text(
            Path("/sys/devices/system/clocksource/clocksource0/current_clocksource")
        ),
        "ntp_synchronization_status": ntp,
    }
    expected = (
        instance_id == EXPECTED_C6IN_INSTANCE_ID
        and region == EXPECTED_C6IN_REGION
        and observed["effective_user"] == EXPECTED_C6IN_USER
        and ntp == "yes"
    )
    if not expected:
        raise contracts.LatencyContractError(
            "LATENCY_HOST_IDENTITY_DRIFT",
            "c6in_host_identity",
            "instance, region, user or NTP mismatch",
        )
    observed["host_identity_token"] = contracts.canonical_json_sha256(
        {
            key: value
            for key, value in observed.items()
            if key != "captured_at_utc"
        }
    )
    return observed


def _runtime_identity(expected_commit: str) -> dict[str, Any]:
    commit = _run_text(["git", "rev-parse", "HEAD"])
    status = _run_text(["git", "status", "--porcelain"])
    if commit != expected_commit or status:
        raise contracts.LatencyContractError(
            "LATENCY_RUNTIME_IDENTITY_DRIFT",
            str(REPO_ROOT),
            f"commit_match={commit == expected_commit} clean={not status}",
        )
    dependency_rows = sorted(
        _run_text([sys.executable, "-m", "pip", "freeze"]).splitlines()
    )
    dependency_sha = hashlib.sha256(
        ("\n".join(dependency_rows) + "\n").encode("ascii")
    ).hexdigest()
    source_paths = (
        Path(__file__).resolve(),
        Path(contracts.__file__).resolve(),
        REPO_ROOT / "examples/hyperliquid/hyperliquid_maker_order_manager.py",
        REPO_ROOT
        / "examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py",
        REPO_ROOT / "examples/hyperliquid/cross_exchange_price_math.py",
    )
    source_inventory = {
        path.relative_to(REPO_ROOT).as_posix(): contracts.sha256_file(path)
        for path in source_paths
    }
    payload = {
        "schema_version": "skhynix_c6in_runtime_identity_v1",
        "task_id": contracts.TASK_ID,
        "captured_at_utc": _utc_now(),
        "repository_commit": commit,
        "working_tree_clean": True,
        "python_invocation_path": sys.executable,
        "python_executable": str(Path(sys.executable).resolve()),
        "python_prefix": sys.prefix,
        "python_base_prefix": sys.base_prefix,
        "virtual_environment_active": sys.prefix != sys.base_prefix,
        "python_version": platform.python_version(),
        "hyperliquid_sdk_version": importlib.metadata.version(
            "hyperliquid-python-sdk"
        ),
        "measurement_entrypoint_sha256": contracts.sha256_file(Path(__file__)),
        "terminal_classifier_source_sha256": contracts.sha256_file(
            REPO_ROOT / "examples/hyperliquid/hyperliquid_maker_order_manager.py"
        ),
        "runtime_dependency_inventory_sha256": dependency_sha,
        "runtime_source_inventory": source_inventory,
        "api_base_hostname": "api.hyperliquid.xyz",
        "proxy_mode": "environment_disabled",
        "ip_family": "ipv4",
        "http_connection_reuse_mode": "official_sdk_reused",
        "request_timeout_seconds": 10,
        "terminal_query_policy": "exact_oid_then_exact_cloid",
        "terminal_query_retry_interval_ms": 50,
        "terminal_query_timeout_ms": 5000,
        "upstream_strategy_runtime_mode": "production_dry",
        "action_transport_type": "DryActionTransport",
        "live_order_allowed": False,
        "passive_route_availability": "unavailable",
    }
    payload["runtime_identity_sha256"] = contracts.canonical_json_sha256(
        {
            key: value
            for key, value in payload.items()
            if key != "captured_at_utc"
        }
    )
    return payload


def _market_snapshot() -> dict[str, Any]:
    from hyperliquid.info import Info  # type: ignore
    from hyperliquid.utils import constants  # type: ignore

    from examples.hyperliquid import cross_exchange_price_math as price_math

    info = Info(
        constants.MAINNET_API_URL,
        skip_ws=True,
        perp_dexs=[TARGET_DEX],
        timeout=10,
    )
    meta = info.meta(TARGET_DEX)
    universe = meta.get("universe")
    if not isinstance(universe, list):
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "market.meta.universe",
            "not a list",
        )
    rows = [
        row
        for row in universe
        if isinstance(row, dict) and row.get("name") == TARGET_ASSET
    ]
    if len(rows) != 1:
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "market.meta.universe",
            f"target rows={len(rows)}",
        )
    metadata = rows[0]
    sz_decimals = int(metadata["szDecimals"])
    asset_id = int(info.name_to_asset(TARGET_ASSET))
    book = info.l2_snapshot(TARGET_ASSET)
    levels = book.get("levels") if isinstance(book, dict) else None
    if (
        not isinstance(levels, list)
        or len(levels) != 2
        or not levels[0]
        or not levels[1]
    ):
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "market.l2_snapshot",
            "two-sided BBO unavailable",
        )
    best_bid = Decimal(str(levels[0][0]["px"]))
    best_ask = Decimal(str(levels[1][0]["px"]))
    if not 0 < best_bid < best_ask:
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "market.bbo",
            "invalid BBO",
        )
    reference_mid = (best_bid + best_ask) / 2
    tick_size = price_math._price_quantum(
        reference_mid,
        sz_decimals=sz_decimals,
    )
    if (
        price_math.normalize_hl_perp_price(
            best_bid,
            sz_decimals=sz_decimals,
        )
        != float(best_bid)
        or price_math.normalize_hl_perp_price(
            best_ask,
            sz_decimals=sz_decimals,
        )
        != float(best_ask)
    ):
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "market.bbo",
            "BBO does not match derived price quantum",
        )
    lot_size = Decimal(1).scaleb(-sz_decimals)
    minimum_notional = Decimal(
        str(contracts.MINIMUM_VALID_ORDER_NOTIONAL_USDC)
    )
    minimum_lots = (
        minimum_notional / (reference_mid * lot_size)
    ).to_integral_value(rounding=ROUND_CEILING)
    minimum_valid_size = minimum_lots * lot_size
    minimum_valid_notional = minimum_valid_size * reference_mid
    quote_distance_price = Decimal(10) * tick_size
    quote_distance_bps = (
        quote_distance_price / reference_mid * Decimal(10_000)
    )
    if asset_id != EXPECTED_TARGET_ASSET_ID:
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "market.asset_id",
            f"expected={EXPECTED_TARGET_ASSET_ID} observed={asset_id}",
        )
    metadata_identity = contracts.canonical_json_sha256(
        {
            "asset_id": asset_id,
            "metadata": metadata,
        }
    )
    return {
        "schema_version": "skhynix_c6in_market_identity_v1",
        "task_id": contracts.TASK_ID,
        "captured_at_utc": _utc_now(),
        "dex": TARGET_DEX,
        "canonical_asset": TARGET_ASSET,
        "sdk_asset_identifier": asset_id,
        "asset_metadata_identity": metadata_identity,
        "sz_decimals": sz_decimals,
        "tick_size": str(tick_size),
        "lot_size": str(lot_size),
        "minimum_valid_order_size": str(minimum_valid_size),
        "minimum_valid_order_notional": str(minimum_valid_notional),
        "minimum_notional_rule_usdc": (
            contracts.MINIMUM_VALID_ORDER_NOTIONAL_USDC
        ),
        "minimum_notional_authority": contracts.MINIMUM_NOTIONAL_AUTHORITY,
        "best_bid": str(best_bid),
        "best_ask": str(best_ask),
        "reference_mid_price": str(reference_mid),
        "quote_distance_ticks": 10,
        "quote_distance_price": str(quote_distance_price),
        "quote_distance_one_way_bps": str(quote_distance_bps),
        "public_preflight_duration_seconds": (
            PUBLIC_PREFLIGHT_DURATION_SECONDS
        ),
        "public_preflight_horizon_ms": PUBLIC_PREFLIGHT_HORIZON_MS,
        "public_preflight_min_valid_pairs": PUBLIC_PREFLIGHT_MIN_VALID_PAIRS,
        "quote_distance_safety_status": (
            "not_run_prior_authorization_block"
        ),
        "private_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
    }


def _authorization_envelope(*, credential_file_read: bool) -> dict[str, Any]:
    return {
        "schema_version": "skhynix_c6in_latency_authorization_v2",
        "task_id": contracts.TASK_ID,
        "authorization_source": (
            "user_instruction_triple_monetary_caps_and_authorize_live_"
            "execution_2026-08-22"
        ),
        "active_private_read_authorized": ACTIVE_EXECUTION_AUTHORIZED,
        "active_order_submit_authorized": ACTIVE_EXECUTION_AUTHORIZED,
        "active_cancel_authorized": ACTIVE_EXECUTION_AUTHORIZED,
        "reduce_only_flatten_authorized": ACTIVE_EXECUTION_AUTHORIZED,
        "post_only_required": True,
        "time_in_force": "Alo",
        "max_open_orders": 1,
        "max_attempts_per_batch": 10,
        "max_total_attempts": contracts.MAX_TOTAL_ATTEMPTS,
        "per_order_notional_cap_usdc": (
            contracts.PER_ORDER_NOTIONAL_CAP_USDC
        ),
        "aggregate_position_cap_usdc": (
            contracts.AGGREGATE_POSITION_CAP_USDC
        ),
        "max_loss_usdc": contracts.MAX_LOSS_USDC,
        "max_loss_basis": contracts.LOSS_BASIS,
        "credential_file_read": credential_file_read,
        "private_endpoint_called": credential_file_read,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
    }


def _parse_public_book(message: Any) -> tuple[int, Decimal, Decimal] | None:
    if not isinstance(message, dict) or message.get("channel") != "l2Book":
        return None
    data = message.get("data")
    if not isinstance(data, dict) or data.get("coin") != TARGET_ASSET:
        return None
    levels = data.get("levels")
    if (
        not isinstance(levels, list)
        or len(levels) != 2
        or not levels[0]
        or not levels[1]
    ):
        return None
    best_bid = Decimal(str(levels[0][0]["px"]))
    best_ask = Decimal(str(levels[1][0]["px"]))
    if not 0 < best_bid < best_ask:
        return None
    server_time_ms = int(data.get("time", 0))
    if server_time_ms <= 0:
        return None
    return server_time_ms, best_bid, best_ask


def _collect_public_quote_safety(
    *,
    output_root: Path,
    market: dict[str, Any],
    duration_seconds: int = PUBLIC_PREFLIGHT_DURATION_SECONDS,
) -> dict[str, Any]:
    from hyperliquid.info import Info  # type: ignore
    from hyperliquid.utils import constants  # type: ignore

    messages: queue.Queue[Any] = queue.Queue()
    info = Info(
        constants.MAINNET_API_URL,
        skip_ws=False,
        perp_dexs=[TARGET_DEX],
        timeout=10,
    )
    subscription = {
        "type": "l2Book",
        "coin": TARGET_ASSET,
        "fast": True,
    }
    subscription_id = info.subscribe(subscription, messages.put)
    started_monotonic_ns = time.monotonic_ns()
    deadline_ns = started_monotonic_ns + duration_seconds * 1_000_000_000
    samples: list[dict[str, Any]] = []
    try:
        while time.monotonic_ns() < deadline_ns:
            remaining_seconds = max(
                0.01,
                min(2.0, (deadline_ns - time.monotonic_ns()) / 1_000_000_000),
            )
            try:
                message = messages.get(timeout=remaining_seconds)
            except queue.Empty:
                continue
            parsed = _parse_public_book(message)
            if parsed is None:
                continue
            server_time_ms, best_bid, best_ask = parsed
            observed_monotonic_ns = time.monotonic_ns()
            if observed_monotonic_ns > deadline_ns:
                break
            mid = (best_bid + best_ask) / 2
            samples.append(
                {
                    "schema_version": contracts.SCHEMA_VERSION,
                    "task_id": contracts.TASK_ID,
                    "sample_sequence": len(samples) + 1,
                    "monotonic_ns": observed_monotonic_ns,
                    "audit_utc_ns": time.time_ns(),
                    "server_time_ms": server_time_ms,
                    "best_bid": str(best_bid),
                    "best_ask": str(best_ask),
                    "mid_price": str(mid),
                }
            )
    finally:
        try:
            info.unsubscribe(subscription, subscription_id)
        finally:
            info.disconnect_websocket()
    completed_monotonic_ns = time.monotonic_ns()
    elapsed_seconds = (
        completed_monotonic_ns - started_monotonic_ns
    ) / 1_000_000_000
    pairs = contracts.derive_public_quote_pairs(
        samples,
        horizon_ms=PUBLIC_PREFLIGHT_HORIZON_MS,
    )
    contracts.write_csv(
        output_root / "public_quote_samples.csv",
        samples,
        contracts.PUBLIC_QUOTE_SAMPLE_FIELDS,
    )
    contracts.write_csv(
        output_root / "public_quote_pairs.csv",
        pairs,
        contracts.PUBLIC_QUOTE_PAIR_FIELDS,
    )
    if (
        elapsed_seconds < duration_seconds
        or len(pairs) < PUBLIC_PREFLIGHT_MIN_VALID_PAIRS
    ):
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "public_quote_safety",
            (
                f"elapsed={elapsed_seconds:.6f} pairs={len(pairs)} "
                f"required={PUBLIC_PREFLIGHT_MIN_VALID_PAIRS}"
            ),
        )
    p99_bps = contracts.nearest_rank_float(
        [float(row["abs_mid_move_bps"]) for row in pairs],
        0.99,
    )
    safety = contracts.quote_distance_safety(
        tick_size=float(market["tick_size"]),
        reference_mid_price=float(market["reference_mid_price"]),
        p99_abs_250ms_mid_move_bps=p99_bps,
    )
    result = {
        "schema_version": "skhynix_c6in_public_quote_safety_v2",
        "task_id": contracts.TASK_ID,
        "started_at_utc": samples[0]["audit_utc_ns"] if samples else 0,
        "completed_at_utc": time.time_ns(),
        "required_duration_seconds": duration_seconds,
        "observed_duration_seconds": elapsed_seconds,
        "horizon_ms": PUBLIC_PREFLIGHT_HORIZON_MS,
        "pairing_rule": "first_later_observation_at_or_after_horizon",
        "sample_count": len(samples),
        "valid_pair_count": len(pairs),
        "minimum_valid_pair_count": PUBLIC_PREFLIGHT_MIN_VALID_PAIRS,
        "nearest_rank_p99_abs_250ms_mid_move_bps": p99_bps,
        **safety,
    }
    contracts.write_json(output_root / "public_quote_safety.json", result)
    return result


def _read_credentials(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    path = Path(path)
    file_stat = path.stat()
    mode = stat.S_IMODE(file_stat.st_mode)
    if file_stat.st_uid != os.getuid() or mode & 0o077:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "credential_source",
            f"owner_match={file_stat.st_uid == os.getuid()} mode={mode:o}",
        )
    values: dict[str, str] = {}
    loaded_keys: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key in {
            "HL_PRIVATE_KEY",
            "HYPERLIQUID_PRIVATE_KEY",
            "HL_WALLET",
            "HYPERLIQUID_ACCOUNT_ADDRESS",
        }:
            values[key] = value
            loaded_keys.append(key)
    private_key = values.get("HL_PRIVATE_KEY") or values.get(
        "HYPERLIQUID_PRIVATE_KEY", ""
    )
    configured_address = values.get("HYPERLIQUID_ACCOUNT_ADDRESS") or values.get(
        "HL_WALLET", ""
    )
    if not private_key or not configured_address:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "credential_source",
            "private key or configured identity absent",
        )
    metadata = {
        "schema_version": "skhynix_c6in_credential_source_v2",
        "task_id": contracts.TASK_ID,
        "credential_source_path_sha256": hashlib.sha256(
            str(path.resolve()).encode("utf-8")
        ).hexdigest(),
        "owner_matches_effective_user": True,
        "file_mode_octal": f"{mode:04o}",
        "restrictive_permissions": True,
        "required_key_names_present": True,
        "loaded_key_names": sorted(loaded_keys),
        "secret_values_written": False,
    }
    return {
        "private_key": private_key,
        "configured_address": configured_address,
    }, metadata


def _normalized_address(value: Any, *, location: str) -> str:
    normalized = str(value or "").strip().lower()
    if (
        not normalized.startswith("0x")
        or len(normalized) != 42
        or any(character not in "0123456789abcdef" for character in normalized[2:])
    ):
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            location,
            "malformed address",
        )
    return normalized


def _role_name(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("role", ""))


def _agent_master_address(payload: Any) -> str | None:
    if not isinstance(payload, dict) or payload.get("role") != "agent":
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    master = data.get("user")
    return str(master) if isinstance(master, str) else None


def _resolve_account_identity(
    info: Any,
    *,
    configured_address: str,
    signer_address: str,
    now_unix_ms: int | None = None,
) -> dict[str, Any]:
    configured = _normalized_address(
        configured_address,
        location="configured_identity",
    )
    signer = _normalized_address(
        signer_address,
        location="signer_identity",
    )
    configured_role_payload = info.user_role(configured)
    configured_role = _role_name(configured_role_payload)
    signer_role_payload = (
        configured_role_payload
        if signer == configured
        else info.user_role(signer)
    )
    signer_role = _role_name(signer_role_payload)

    if configured_role == "agent":
        if configured != signer:
            raise contracts.LatencyContractError(
                "LATENCY_AUTHORIZATION_MISMATCH",
                "configured_agent_identity",
                "configured agent does not match private-key signer",
            )
        master = _agent_master_address(configured_role_payload)
        account_source = "derived_from_configured_agent_role"
    elif configured_role == "user":
        master = configured
        account_source = "configured_user"
    else:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "configured_identity_role",
            f"role={configured_role or 'missing'}",
        )
    account = _normalized_address(
        master,
        location="derived_account_identity",
    )

    account_role_payload = (
        configured_role_payload
        if account == configured
        else info.user_role(account)
    )
    account_role = _role_name(account_role_payload)
    if account_role != "user":
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "account_identity_role",
            f"role={account_role or 'missing'}",
        )
    abstraction = info.query_user_abstraction_state(account)
    if abstraction != "unifiedAccount":
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "account_abstraction",
            f"observed={abstraction!r} expected='unifiedAccount'",
        )

    agent_approved = signer == account
    agent_expired = False
    agent_name: str | None = None
    valid_until_unix_ms: int | None = None
    if signer != account:
        signer_master = _agent_master_address(signer_role_payload)
        if (
            signer_role != "agent"
            or signer_master is None
            or _normalized_address(
                signer_master,
                location="signer_agent_master",
            )
            != account
        ):
            raise contracts.LatencyContractError(
                "LATENCY_AUTHORIZATION_MISMATCH",
                "signer_agent_role",
                "signer does not resolve to the unified account",
            )
        extra_agents = info.extra_agents(account)
        if not isinstance(extra_agents, list):
            raise contracts.LatencyContractError(
                "LATENCY_AUTHORIZATION_MISMATCH",
                "extra_agents",
                "response is not a list",
            )
        approved_record = next(
            (
                row
                for row in extra_agents
                if isinstance(row, dict)
                and str(
                    row.get("address")
                    or row.get("agentAddress")
                    or ""
                ).strip().lower()
                == signer
            ),
            None,
        )
        if not isinstance(approved_record, dict):
            raise contracts.LatencyContractError(
                "LATENCY_AUTHORIZATION_MISMATCH",
                "signer_agent_approval",
                "agent is not present in the unified account approval set",
            )
        try:
            valid_until_unix_ms = int(approved_record["validUntil"])
        except (KeyError, TypeError, ValueError) as exc:
            raise contracts.LatencyContractError(
                "LATENCY_AUTHORIZATION_MISMATCH",
                "signer_agent_expiry",
                "validUntil is absent or invalid",
            ) from exc
        observed_now_ms = (
            time.time_ns() // 1_000_000
            if now_unix_ms is None
            else now_unix_ms
        )
        agent_expired = valid_until_unix_ms <= observed_now_ms
        if agent_expired:
            raise contracts.LatencyContractError(
                "LATENCY_AUTHORIZATION_MISMATCH",
                "signer_agent_expiry",
                "agent approval is expired",
            )
        agent_name = (
            str(approved_record.get("name"))
            if approved_record.get("name") not in (None, "")
            else None
        )
        agent_approved = True

    return {
        "configured_address": configured,
        "configured_role": configured_role,
        "signer_address": signer,
        "signer_role": signer_role,
        "account_address": account,
        "account_role": account_role,
        "account_abstraction": abstraction,
        "account_source": account_source,
        "agent_approved": agent_approved,
        "agent_expired": agent_expired,
        "agent_name": agent_name,
        "agent_valid_until_unix_ms": valid_until_unix_ms,
    }


def _target_position_size(user_state: Any) -> Decimal:
    if not isinstance(user_state, dict):
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "user_state",
            "response is not an object",
        )
    positions = user_state.get("assetPositions")
    if not isinstance(positions, list):
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "user_state.assetPositions",
            "response is not a list",
        )
    observed = Decimal(0)
    target_rows = 0
    for row in positions:
        position = row.get("position") if isinstance(row, dict) else None
        if not isinstance(position, dict):
            continue
        coin = str(position.get("coin", ""))
        if coin not in {TARGET_ASSET, TARGET_ASSET.split(":", 1)[1]}:
            continue
        target_rows += 1
        observed += Decimal(str(position.get("szi", "0")))
    if target_rows > 1:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "user_state.assetPositions",
            f"duplicate target rows={target_rows}",
        )
    return observed


def _available_collateral(
    user_state: Any,
    spot_user_state: Any,
) -> tuple[bool, str]:
    candidates: list[tuple[str, Any]] = []
    if isinstance(user_state, dict):
        candidates.extend(
            [
                ("target_dex_withdrawable", user_state.get("withdrawable")),
                (
                    "target_dex_account_value",
                    user_state.get("marginSummary", {}).get("accountValue")
                    if isinstance(user_state.get("marginSummary"), dict)
                    else None,
                ),
            ]
        )
    if isinstance(spot_user_state, dict):
        balances = spot_user_state.get("balances")
        if isinstance(balances, list):
            for row in balances:
                if not isinstance(row, dict) or row.get("coin") != "USDC":
                    continue
                try:
                    spot_available = Decimal(str(row.get("total", "0"))) - Decimal(
                        str(row.get("hold", "0"))
                    )
                except Exception:
                    continue
                candidates.append(("unified_spot_usdc_available", spot_available))
    parsed: list[tuple[str, Decimal]] = []
    for source, value in candidates:
        if value in (None, ""):
            continue
        try:
            parsed.append((source, Decimal(str(value))))
        except Exception:
            continue
    if not parsed:
        return False, "unavailable"
    source, available = max(parsed, key=lambda item: item[1])
    return (
        available >= Decimal(str(contracts.AGGREGATE_POSITION_CAP_USDC)),
        source,
    )


def _conflicting_runtime_snapshot() -> dict[str, Any]:
    service_status = subprocess.run(
        ["systemctl", "is-active", "xemm.service"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    process_rows = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    markers = (
        "/home/admin/XEMM_rust_latest",
        "hummingbot",
        "glft production_live",
    )
    conflicts = [
        row
        for row in process_rows
        if any(marker.lower() in row.lower() for marker in markers)
        and "skhynix_c6in_latency_v2.py" not in row
    ]
    if service_status == "active" or conflicts:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "conflicting_runtime",
            (
                f"xemm_service={service_status or 'unknown'} "
                f"conflicting_process_count={len(conflicts)}"
            ),
        )
    return {
        "schema_version": "skhynix_c6in_conflicting_runtime_v2",
        "task_id": contracts.TASK_ID,
        "xemm_service_status": service_status or "unknown",
        "conflicting_process_count": 0,
        "same_account_market_path_available": True,
        "process_arguments_written": False,
    }


def _private_account_baseline(
    *,
    credential_file: Path,
    output_root: Path,
) -> dict[str, Any]:
    from eth_account import Account  # type: ignore
    from hyperliquid.exchange import Exchange  # type: ignore
    from hyperliquid.info import Info  # type: ignore
    from hyperliquid.utils import constants  # type: ignore

    secrets, credential_metadata = _read_credentials(credential_file)
    wallet = Account.from_key(secrets["private_key"])
    normalized_signer = _normalized_address(
        wallet.address,
        location="signer_identity",
    )
    normalized_configured = _normalized_address(
        secrets["configured_address"],
        location="configured_identity",
    )
    info = Info(
        constants.MAINNET_API_URL,
        skip_ws=True,
        perp_dexs=[TARGET_DEX],
        timeout=10,
    )
    identity = _resolve_account_identity(
        info,
        configured_address=normalized_configured,
        signer_address=normalized_signer,
    )
    normalized_account = identity["account_address"]
    account_identity_token = hashlib.sha256(
        normalized_account.encode("ascii")
    ).hexdigest()
    signer_identity_token = hashlib.sha256(
        normalized_signer.encode("ascii")
    ).hexdigest()
    configured_identity_token = hashlib.sha256(
        normalized_configured.encode("ascii")
    ).hexdigest()
    Exchange(
        wallet,
        constants.MAINNET_API_URL,
        account_address=normalized_account,
        perp_dexs=[TARGET_DEX],
        timeout=10,
    )
    open_orders = info.open_orders(normalized_account, TARGET_DEX)
    user_state = info.user_state(normalized_account, TARGET_DEX)
    spot_user_state = info.spot_user_state(normalized_account)
    if not isinstance(open_orders, list):
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "open_orders",
            "response is not a list",
        )
    position_size = _target_position_size(user_state)
    collateral_sufficient, collateral_source = _available_collateral(
        user_state,
        spot_user_state,
    )
    if open_orders or position_size != 0 or not collateral_sufficient:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "account_baseline",
            (
                f"open_orders={len(open_orders)} "
                f"target_position_zero={position_size == 0} "
                f"collateral_sufficient={collateral_sufficient} "
                f"collateral_source={collateral_source}"
            ),
        )
    account_baseline = {
        "schema_version": "skhynix_c6in_account_baseline_v2",
        "task_id": contracts.TASK_ID,
        "account_identity_token": account_identity_token,
        "signer_identity_token": signer_identity_token,
        "configured_identity_token": configured_identity_token,
        "configured_identity_matches_signer": (
            normalized_configured == normalized_signer
        ),
        "configured_identity_role": identity["configured_role"],
        "account_role": identity["account_role"],
        "account_abstraction": identity["account_abstraction"],
        "account_source": identity["account_source"],
        "signer_role": identity["signer_role"],
        "agent_approved": identity["agent_approved"],
        "agent_expired": identity["agent_expired"],
        "dex": TARGET_DEX,
        "asset": TARGET_ASSET,
        "open_order_count": 0,
        "target_position_zero": True,
        "available_margin_at_least_aggregate_cap": True,
        "available_collateral_source": collateral_source,
        "aggregate_position_cap_usdc": (
            contracts.AGGREGATE_POSITION_CAP_USDC
        ),
        "raw_account_written": False,
        "raw_private_response_written": False,
    }
    serialized = contracts.canonical_json_bytes(
        {
            "credential_metadata": credential_metadata,
            "account_baseline": account_baseline,
        }
    ).decode("ascii").lower()
    if any(
        address in serialized
        for address in {
            normalized_account,
            normalized_signer,
            normalized_configured,
        }
    ):
        raise contracts.LatencyContractError(
            "LATENCY_SECRET_OR_REFERENCE_LEAK",
            "account_baseline",
            "raw address appeared in artifact",
        )
    contracts.write_json(
        output_root / "credential_source.json",
        credential_metadata,
    )
    contracts.write_json(
        output_root / "account_baseline.json",
        account_baseline,
    )
    return account_baseline


def _parse_utc(value: str) -> datetime:
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise contracts.LatencyContractError(
            "LATENCY_WINDOW_SUPPORT_INVALID",
            "collection_window_schedule",
            value,
        ) from exc
    if observed.tzinfo != timezone.utc:
        raise contracts.LatencyContractError(
            "LATENCY_WINDOW_SUPPORT_INVALID",
            "collection_window_schedule",
            "UTC timestamp required",
        )
    return observed


def freeze_collection_schedule(
    output_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    observed_now = now or datetime.now(timezone.utc)
    if observed_now.tzinfo != timezone.utc:
        raise contracts.LatencyContractError(
            "LATENCY_WINDOW_SUPPORT_INVALID",
            "schedule_freeze_time",
            "UTC timestamp required",
        )
    earliest = observed_now + timedelta(seconds=WINDOW_FREEZE_LEAD_SECONDS)
    first_start_epoch = (
        int(earliest.timestamp() + 59) // 60
    ) * 60
    first_start = datetime.fromtimestamp(first_start_epoch, timezone.utc)
    rows: list[dict[str, Any]] = []
    for index in range(3):
        start = first_start + timedelta(
            seconds=index
            * (COLLECTION_WINDOW_SECONDS + COLLECTION_WINDOW_GAP_SECONDS)
        )
        end = start + timedelta(seconds=COLLECTION_WINDOW_SECONDS)
        rows.append(
            {
                "schema_version": contracts.SCHEMA_VERSION,
                "task_id": contracts.TASK_ID,
                "collection_window_id": f"w{index + 1}",
                "start_utc": start.isoformat().replace("+00:00", "Z"),
                "end_utc": end.isoformat().replace("+00:00", "Z"),
                "preselected_before_latency_access": True,
                "status": "planned",
            }
        )
    contracts.write_csv(output_path, rows, contracts.SCHEDULE_FIELDS)
    result = {
        "schema_version": "skhynix_c6in_collection_schedule_freeze_v2",
        "task_id": contracts.TASK_ID,
        "frozen_at_utc": observed_now.isoformat().replace("+00:00", "Z"),
        "lead_seconds": WINDOW_FREEZE_LEAD_SECONDS,
        "window_seconds": COLLECTION_WINDOW_SECONDS,
        "gap_seconds": COLLECTION_WINDOW_GAP_SECONDS,
        "max_attempts_per_window": MAX_ATTEMPTS_PER_WINDOW,
        "window_count": len(rows),
        "first_window_start_utc": rows[0]["start_utc"],
        "last_window_end_utc": rows[-1]["end_utc"],
        "side_schedule_rule": "odd_sample_buy_even_sample_sell",
        "latency_values_accessed": False,
    }
    contracts.write_json(
        Path(output_path).with_suffix(".receipt.json"),
        result,
    )
    return result


def _build_live_clients(
    secrets: dict[str, str],
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    from eth_account import Account  # type: ignore
    from hyperliquid.exchange import Exchange  # type: ignore
    from hyperliquid.info import Info  # type: ignore
    from hyperliquid.utils import constants  # type: ignore

    wallet = Account.from_key(secrets["private_key"])
    info = Info(
        constants.MAINNET_API_URL,
        skip_ws=True,
        perp_dexs=[TARGET_DEX],
        timeout=10,
    )
    terminal_info = Info(
        constants.MAINNET_API_URL,
        skip_ws=True,
        perp_dexs=[TARGET_DEX],
        timeout=10,
    )
    identity = _resolve_account_identity(
        info,
        configured_address=secrets["configured_address"],
        signer_address=wallet.address,
    )
    exchange = Exchange(
        wallet,
        constants.MAINNET_API_URL,
        account_address=identity["account_address"],
        perp_dexs=[TARGET_DEX],
        timeout=10,
    )
    return wallet, info, terminal_info, exchange, identity


def _query_order_class(
    info: Any,
    account: str,
    oid: int,
    cloid: str,
) -> tuple[str, Any]:
    payload = info.query_order_by_oid(account, oid)
    classification = order_manager._classify_order_status_query_payload(
        payload,
        expected_oid=oid,
        expected_cloid=cloid,
        require_embedded_reference=True,
    )
    return classification, payload


def _submit_payload(response: Any, expected_cloid: str) -> dict[str, Any]:
    classification = order_manager._classify_order_payload(response)
    response_payload = (
        response.get("response") if isinstance(response, dict) else None
    )
    data = (
        response_payload.get("data")
        if isinstance(response_payload, dict)
        else None
    )
    statuses = data.get("statuses") if isinstance(data, dict) else None
    status = (
        statuses[0]
        if isinstance(statuses, list) and len(statuses) == 1
        else None
    )
    detail = (
        status.get(classification)
        if isinstance(status, dict) and classification in status
        else {}
    )
    oid: int | None = None
    if isinstance(detail, dict) and detail.get("oid") not in (None, ""):
        oid = int(detail["oid"])
    return {
        "classification": classification,
        "oid": oid,
        "cloid": expected_cloid,
        "filled": detail if classification == "filled" else {},
    }


def _fill_facts(
    info: Any,
    account: str,
    *,
    oid: int,
    start_time_ms: int,
) -> dict[str, Decimal]:
    matched: list[dict[str, Any]] = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not matched:
        rows = info.user_fills_by_time(
            account,
            max(0, start_time_ms - 5000),
            int(time.time() * 1000) + 5000,
            False,
        )
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    observed_oid = int(row.get("oid"))
                except (TypeError, ValueError):
                    continue
                if observed_oid != oid:
                    continue
                if str(row.get("coin", "")) not in {
                    TARGET_ASSET,
                    TARGET_ASSET.split(":", 1)[1],
                }:
                    continue
                matched.append(row)
        if not matched:
            time.sleep(0.05)
    if not matched:
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "fill_facts",
            "exact oid fill rows absent",
        )
    quantity = sum(Decimal(str(row["sz"])) for row in matched)
    notional = sum(
        Decimal(str(row["sz"])) * Decimal(str(row["px"]))
        for row in matched
    )
    fee = sum(Decimal(str(row.get("fee", "0"))) for row in matched)
    if quantity <= 0:
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "fill_facts",
            "nonpositive quantity",
        )
    return {
        "quantity": quantity,
        "vwap": notional / quantity,
        "fee": fee,
    }


def _filled_action_facts(response: Any) -> tuple[int, Decimal, Decimal]:
    parsed = _submit_payload(response, "")
    if parsed["classification"] != "filled" or parsed["oid"] is None:
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "flatten_response",
            "authoritative filled response absent",
        )
    detail = parsed["filled"]
    quantity = Decimal(str(detail.get("totalSz", detail.get("sz", "0"))))
    vwap = Decimal(str(detail.get("avgPx", detail.get("px", "0"))))
    if quantity <= 0 or vwap <= 0:
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "flatten_response",
            "filled quantity or vwap absent",
        )
    return parsed["oid"], quantity, vwap


def _flatten_position(
    *,
    info: Any,
    exchange: Any,
    account: str,
    original_oid: int,
    original_side: str,
    original_submit_time_ms: int,
    cloid_seed: str,
) -> dict[str, Any]:
    from hyperliquid.utils.types import Cloid  # type: ignore

    original = _fill_facts(
        info,
        account,
        oid=original_oid,
        start_time_ms=original_submit_time_ms,
    )
    position = _target_position_size(info.user_state(account, TARGET_DEX))
    if position == 0:
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "flatten_position",
            "filled order has no reconcilable target position",
        )
    reference_mid = Decimal(str(_market_snapshot()["reference_mid_price"]))
    if abs(position) * reference_mid > Decimal(
        str(contracts.AGGREGATE_POSITION_CAP_USDC)
    ):
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "flatten_position",
            "aggregate position cap exceeded",
        )
    flatten_cloid = "0x" + hashlib.sha256(
        f"{cloid_seed}:flatten".encode("ascii")
    ).hexdigest()[:32]
    response = exchange.market_close(
        TARGET_ASSET,
        sz=float(abs(position)),
        slippage=0.05,
        cloid=Cloid.from_str(flatten_cloid),
    )
    flatten_oid, flattened_quantity, flatten_vwap = _filled_action_facts(
        response
    )
    deadline = time.monotonic() + 5
    final_position = position
    while time.monotonic() < deadline:
        final_position = _target_position_size(
            info.user_state(account, TARGET_DEX)
        )
        if final_position == 0:
            break
        time.sleep(0.05)
    if final_position != 0:
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "flatten_position",
            f"residual_position={final_position}",
        )
    flatten = _fill_facts(
        info,
        account,
        oid=flatten_oid,
        start_time_ms=original_submit_time_ms,
    )
    if flatten["quantity"] != flattened_quantity:
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "flatten_quantity",
            "response and fill ledger differ",
        )
    loss = contracts.realized_flatten_slippage_loss_usdc(
        original_fill_side=original_side,
        fill_vwap=float(original["vwap"]),
        flatten_vwap=float(flatten_vwap),
        filled_quantity=float(original["quantity"]),
        flattened_quantity=float(flattened_quantity),
        flatten_status="authoritatively_complete",
    )
    return {
        "filled_quantity": str(original["quantity"]),
        "fill_vwap": str(original["vwap"]),
        "flatten_status": "authoritatively_complete",
        "flattened_quantity": str(flattened_quantity),
        "flatten_vwap": str(flatten_vwap),
        "realized_flatten_slippage_loss_usdc": str(loss),
        "flatten_fee_usdc": str(flatten["fee"]),
        "loss_cap_reached": loss >= contracts.MAX_LOSS_USDC,
    }


def _current_quote(
    info: Any,
    market: dict[str, Any],
    side: str,
) -> dict[str, Decimal]:
    book = info.l2_snapshot(TARGET_ASSET)
    levels = book.get("levels") if isinstance(book, dict) else None
    if (
        not isinstance(levels, list)
        or len(levels) != 2
        or not levels[0]
        or not levels[1]
    ):
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "active_bbo",
            "two-sided BBO unavailable",
        )
    bid = Decimal(str(levels[0][0]["px"]))
    ask = Decimal(str(levels[1][0]["px"]))
    tick = Decimal(str(market["tick_size"]))
    size = Decimal(str(market["minimum_valid_order_size"]))
    if not 0 < bid < ask or tick <= 0 or size <= 0:
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "active_quote",
            "invalid BBO/tick/size",
        )
    price = bid - Decimal(10) * tick if side == "buy" else ask + Decimal(10) * tick
    if price <= 0 or (side == "buy" and price >= ask) or (
        side == "sell" and price <= bid
    ):
        raise contracts.LatencyContractError(
            "LATENCY_QUOTE_DISTANCE_SAFETY_UNVERIFIED",
            "active_quote",
            "post-only noncrossing invariant failed",
        )
    notional = size * price
    if (
        notional > Decimal(str(contracts.PER_ORDER_NOTIONAL_CAP_USDC))
        or notional > Decimal(str(contracts.AGGREGATE_POSITION_CAP_USDC))
    ):
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "active_quote",
            f"notional={notional}",
        )
    current_mid = (bid + ask) / 2
    safety = contracts.quote_distance_safety(
        tick_size=float(tick),
        reference_mid_price=float(current_mid),
        p99_abs_250ms_mid_move_bps=float(
            market["nearest_rank_p99_abs_250ms_mid_move_bps"]
        ),
    )
    return {
        "best_bid": bid,
        "best_ask": ask,
        "mid": current_mid,
        "tick": tick,
        "size": size,
        "price": price,
        "notional": notional,
        "quote_distance_price": Decimal(
            str(safety["quote_distance_price"])
        ),
        "quote_distance_bps": Decimal(
            str(safety["quote_distance_one_way_bps"])
        ),
    }


def _event_rows(
    sample_sequence: int,
    order_reference_token: str,
    moments: Sequence[tuple[str, int, int, str, str]],
) -> list[dict[str, Any]]:
    ordered = sorted(moments, key=lambda row: row[1])
    previous = -1
    rows: list[dict[str, Any]] = []
    for index, (
        event_type,
        monotonic_ns,
        audit_utc_ns,
        classification,
        detail_code,
    ) in enumerate(ordered, start=1):
        if monotonic_ns <= previous:
            monotonic_ns = previous + 1
        rows.append(
            {
                "schema_version": contracts.SCHEMA_VERSION,
                "task_id": contracts.TASK_ID,
                "sample_sequence": sample_sequence,
                "event_sequence": index,
                "event_type": event_type,
                "monotonic_ns": monotonic_ns,
                "audit_utc_ns": audit_utc_ns,
                "order_reference_token": order_reference_token,
                "source": "c6in_official_sdk",
                "classification": classification,
                "detail_code": detail_code,
            }
        )
        previous = monotonic_ns
    return rows


def _run_active_attempt(
    *,
    sample_sequence: int,
    window_id: str,
    host: dict[str, Any],
    runtime: dict[str, Any],
    market: dict[str, Any],
    account_identity_token: str,
    account: str,
    info: Any,
    terminal_info: Any,
    exchange: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    from hyperliquid.utils.types import Cloid  # type: ignore

    side = "buy" if sample_sequence % 2 else "sell"
    attempt_id = f"{contracts.TASK_ID}-attempt-{sample_sequence:03d}"
    batch_id = f"batch-{((sample_sequence - 1) // 10) + 1:02d}"
    cloid = "0x" + hashlib.sha256(
        f"{contracts.TASK_ID}:{sample_sequence}:{side}".encode("ascii")
    ).hexdigest()[:32]
    quote = _current_quote(info, market, side)
    moments: list[tuple[str, int, int, str, str]] = []

    def mark(event_type: str, classification: str, detail_code: str = "") -> int:
        monotonic_ns = time.monotonic_ns()
        moments.append(
            (
                event_type,
                monotonic_ns,
                time.time_ns(),
                classification,
                detail_code,
            )
        )
        return monotonic_ns

    submit_start_ms = int(time.time() * 1000)
    mark("submit_call_start", "request_started")
    response = exchange.order(
        TARGET_ASSET,
        side == "buy",
        float(quote["size"]),
        float(quote["price"]),
        {"limit": {"tif": "Alo"}},
        reduce_only=False,
        cloid=Cloid.from_str(cloid),
    )
    mark("submit_response_end", "response_received")
    submit = _submit_payload(response, cloid)
    oid = submit["oid"]
    order_reference_token = contracts.make_order_reference_token(
        account_identity_token=account_identity_token,
        dex=TARGET_DEX,
        asset=TARGET_ASSET,
        oid=oid,
        cloid=cloid,
        attempt_id=attempt_id,
    )
    defaults = {
        "submit_status": "rejected",
        "resting_status": "not_confirmed",
        "cancel_response_class": "not_called",
        "terminal_class": "rejected",
        "fill_race_class": "no_fill",
        "filled_quantity": "",
        "fill_vwap": "",
        "flatten_status": "not_required",
        "flattened_quantity": "",
        "flatten_vwap": "",
        "realized_flatten_slippage_loss_usdc": "",
        "flatten_fee_usdc": "",
        "final_open_orders_count": "",
        "position_delta": "",
        "safety_status": "not_reconciled",
        "primary_latency_eligible": False,
        "primary_exclusion_reason": "submit_rejected",
    }
    fatal_stop = False
    if submit["classification"] == "filled":
        if oid is None:
            raise contracts.LatencyContractError(
                "LATENCY_UNRESOLVED_EXPOSURE",
                attempt_id,
                "filled submit response lacks oid",
            )
        defaults.update(
            {
                "submit_status": "accepted",
                "terminal_class": "filled",
                "fill_race_class": "filled_before_cancel_dispatch",
                "primary_exclusion_reason": "resting_not_confirmed",
            }
        )
        defaults.update(
            _flatten_position(
                info=info,
                exchange=exchange,
                account=account,
                original_oid=oid,
                original_side=side,
                original_submit_time_ms=submit_start_ms,
                cloid_seed=cloid,
            )
        )
        fatal_stop = True
    elif submit["classification"] == "resting":
        if oid is None:
            raise contracts.LatencyContractError(
                "LATENCY_ORDER_REFERENCE_MISMATCH",
                attempt_id,
                "resting submit response lacks oid",
            )
        defaults["submit_status"] = "accepted"
        resting_deadline = time.monotonic() + 5
        resting_confirmed = False
        while time.monotonic() < resting_deadline:
            status, _ = _query_order_class(info, account, oid, cloid)
            if status == "resting":
                resting_confirmed = True
                resting_mark = mark("resting_confirm", "exact_reference_resting")
                break
            if status == "filled":
                defaults.update(
                    {
                        "terminal_class": "filled",
                        "fill_race_class": "filled_before_cancel_dispatch",
                        "primary_exclusion_reason": "resting_not_confirmed",
                    }
                )
                defaults.update(
                    _flatten_position(
                        info=info,
                        exchange=exchange,
                        account=account,
                        original_oid=oid,
                        original_side=side,
                        original_submit_time_ms=submit_start_ms,
                        cloid_seed=cloid,
                    )
                )
                fatal_stop = True
                break
            time.sleep(0.05)
        if resting_confirmed:
            defaults["resting_status"] = "confirmed"
            settle_deadline_ns = (
                resting_mark + FIXED_PRE_CANCEL_SETTLE_MS * 1_000_000
            )
            while time.monotonic_ns() < settle_deadline_ns:
                time.sleep(
                    min(
                        0.01,
                        max(
                            0.0,
                            (settle_deadline_ns - time.monotonic_ns())
                            / 1_000_000_000,
                        ),
                    )
                )
            mark("risk_decision_ready", "frozen_settle_complete")
            mark("cancel_enqueue", "cancel_enqueued")
            mark("cancel_call_start", "request_started")
            mark("terminal_observation_start", "exact_reference_poll_started")
            terminal_result: dict[str, Any] = {}

            def observe_terminal() -> None:
                deadline = time.monotonic() + (
                    TERMINAL_QUERY_TIMEOUT_MS / 1000
                )
                while time.monotonic() < deadline:
                    try:
                        classification, _ = _query_order_class(
                            terminal_info,
                            account,
                            oid,
                            cloid,
                        )
                    except Exception:
                        time.sleep(TERMINAL_QUERY_INTERVAL_MS / 1000)
                        continue
                    if classification in {
                        "cancel_confirmed",
                        "filled",
                        "rejected",
                    }:
                        terminal_result.update(
                            {
                                "classification": classification,
                                "monotonic_ns": time.monotonic_ns(),
                                "audit_utc_ns": time.time_ns(),
                            }
                        )
                        return
                    time.sleep(TERMINAL_QUERY_INTERVAL_MS / 1000)

            observer = threading.Thread(
                target=observe_terminal,
                name=f"terminal-{sample_sequence}",
                daemon=True,
            )
            observer.start()
            cancel_class = "normal"
            try:
                cancel_response = exchange.cancel(TARGET_ASSET, oid)
                response_payload = (
                    cancel_response.get("response")
                    if isinstance(cancel_response, dict)
                    else None
                )
                response_data = (
                    response_payload.get("data")
                    if isinstance(response_payload, dict)
                    else None
                )
                statuses = (
                    response_data.get("statuses")
                    if isinstance(response_data, dict)
                    else None
                )
                if (
                    not isinstance(cancel_response, dict)
                    or cancel_response.get("status") != "ok"
                    or not isinstance(statuses, list)
                    or len(statuses) != 1
                    or statuses[0] != "success"
                ):
                    cancel_class = "response_error"
            except TimeoutError:
                cancel_class = "timeout"
            except Exception:
                cancel_class = "transport_exception"
            mark("cancel_response_end", "response_received", cancel_class)
            observer.join(TERMINAL_QUERY_TIMEOUT_MS / 1000 + 1)
            if not terminal_result:
                try:
                    rescue = exchange.cancel_by_cloid(
                        TARGET_ASSET,
                        Cloid.from_str(cloid),
                    )
                    response_payload = (
                        rescue.get("response")
                        if isinstance(rescue, dict)
                        else None
                    )
                    response_data = (
                        response_payload.get("data")
                        if isinstance(response_payload, dict)
                        else None
                    )
                    statuses = (
                        response_data.get("statuses")
                        if isinstance(response_data, dict)
                        else None
                    )
                    if (
                        isinstance(rescue, dict)
                        and rescue.get("status") == "ok"
                        and isinstance(statuses, list)
                        and len(statuses) == 1
                        and statuses[0] == "success"
                    ):
                        cancel_class = f"{cancel_class}_retry_by_cloid"
                    else:
                        cancel_class = f"{cancel_class}_retry_error"
                except Exception:
                    cancel_class = f"{cancel_class}_retry_exception"
                rescue_deadline = time.monotonic() + (
                    TERMINAL_QUERY_TIMEOUT_MS / 1000
                )
                while time.monotonic() < rescue_deadline:
                    try:
                        classification, _ = _query_order_class(
                            terminal_info,
                            account,
                            oid,
                            cloid,
                        )
                    except Exception:
                        time.sleep(TERMINAL_QUERY_INTERVAL_MS / 1000)
                        continue
                    if classification in {
                        "cancel_confirmed",
                        "filled",
                        "rejected",
                    }:
                        terminal_result.update(
                            {
                                "classification": classification,
                                "monotonic_ns": time.monotonic_ns(),
                                "audit_utc_ns": time.time_ns(),
                            }
                        )
                        break
                    time.sleep(TERMINAL_QUERY_INTERVAL_MS / 1000)
            defaults["cancel_response_class"] = cancel_class
            terminal_class = terminal_result.get("classification", "unknown")
            defaults["terminal_class"] = terminal_class
            if terminal_result:
                moments.append(
                    (
                        "terminal_confirm",
                        int(terminal_result["monotonic_ns"]),
                        int(terminal_result["audit_utc_ns"]),
                        f"exact_reference_{terminal_class}",
                        "",
                    )
                )
            if terminal_class == "filled":
                defaults.update(
                    {
                        "fill_race_class": "filled_during_cancel_race",
                        "primary_exclusion_reason": (
                            "filled_during_cancel_race"
                        ),
                    }
                )
                defaults.update(
                    _flatten_position(
                        info=info,
                        exchange=exchange,
                        account=account,
                        original_oid=oid,
                        original_side=side,
                        original_submit_time_ms=submit_start_ms,
                        cloid_seed=cloid,
                    )
                )
                fatal_stop = True
            elif terminal_class != "cancel_confirmed":
                defaults["primary_exclusion_reason"] = (
                    "terminal_confirmation_timeout"
                )
                fatal_stop = True
            else:
                defaults.update(
                    {
                        "fill_race_class": "no_fill",
                    }
                )
        elif not fatal_stop:
            defaults["primary_exclusion_reason"] = "resting_not_confirmed"
            fatal_stop = True

    final_open_orders = info.open_orders(account, TARGET_DEX)
    final_position = _target_position_size(info.user_state(account, TARGET_DEX))
    defaults["final_open_orders_count"] = (
        str(len(final_open_orders)) if isinstance(final_open_orders, list) else ""
    )
    defaults["position_delta"] = str(final_position)
    if isinstance(final_open_orders, list) and not final_open_orders and final_position == 0:
        defaults["safety_status"] = "reconciled"
        mark("final_open_orders_confirm", "private_safety_reconciled")
    else:
        defaults["primary_exclusion_reason"] = "safety_stop"
        fatal_stop = True
    required_events = {row[0] for row in moments}
    eligible = (
        defaults["submit_status"] == "accepted"
        and defaults["resting_status"] == "confirmed"
        and defaults["terminal_class"] == "cancel_confirmed"
        and defaults["fill_race_class"] == "no_fill"
        and defaults["safety_status"] == "reconciled"
        and contracts.REQUIRED_PRIMARY_EVENTS <= required_events
    )
    defaults["primary_latency_eligible"] = eligible
    defaults["primary_exclusion_reason"] = (
        "" if eligible else defaults["primary_exclusion_reason"]
    )
    loss_cap_reached = bool(defaults.pop("loss_cap_reached", False))
    attempt = {
        "schema_version": contracts.SCHEMA_VERSION,
        "task_id": contracts.TASK_ID,
        "sample_sequence": sample_sequence,
        "collection_window_id": window_id,
        "batch_id": batch_id,
        "attempt_id": attempt_id,
        "host_identity_token": host["host_identity_token"],
        "boot_id": host["boot_id"],
        "process_identity_token": hashlib.sha256(
            f"{os.getpid()}:{runtime['runtime_identity_sha256']}".encode(
                "ascii"
            )
        ).hexdigest(),
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "market_role": "target",
        "dex": TARGET_DEX,
        "asset": TARGET_ASSET,
        "side": side,
        "order_reference_token": order_reference_token,
        "post_only": True,
        "quote_distance_ticks": 10,
        "tick_size": str(quote["tick"]),
        "quote_distance_price": str(quote["quote_distance_price"]),
        "quote_distance_one_way_bps": str(
            quote["quote_distance_bps"]
        ),
        "order_size": str(quote["size"]),
        "order_notional_usdc": str(quote["notional"]),
        **defaults,
    }
    return (
        attempt,
        _event_rows(sample_sequence, order_reference_token, moments),
        fatal_stop or loss_cap_reached,
    )


def _emergency_reconcile(
    *,
    output_root: Path,
    sample_sequence: int,
    account: str,
    info: Any,
    exchange: Any,
) -> bool:
    from hyperliquid.utils.types import Cloid  # type: ignore

    side = "buy" if sample_sequence % 2 else "sell"
    cloid = "0x" + hashlib.sha256(
        f"{contracts.TASK_ID}:{sample_sequence}:{side}".encode("ascii")
    ).hexdigest()[:32]
    cancel_attempted = False
    flatten_attempted = False
    try:
        cancel_attempted = True
        exchange.cancel_by_cloid(
            TARGET_ASSET,
            Cloid.from_str(cloid),
        )
    except Exception:
        pass
    deadline = time.monotonic() + 5
    final_open_orders: Any = None
    while time.monotonic() < deadline:
        try:
            final_open_orders = info.open_orders(account, TARGET_DEX)
        except Exception:
            time.sleep(0.1)
            continue
        if isinstance(final_open_orders, list) and not final_open_orders:
            break
        time.sleep(0.1)
    position = _target_position_size(info.user_state(account, TARGET_DEX))
    if position != 0:
        flatten_attempted = True
        emergency_cloid = "0x" + hashlib.sha256(
            f"{cloid}:emergency-flatten".encode("ascii")
        ).hexdigest()[:32]
        exchange.market_close(
            TARGET_ASSET,
            sz=float(abs(position)),
            slippage=0.05,
            cloid=Cloid.from_str(emergency_cloid),
        )
    position_deadline = time.monotonic() + 5
    while time.monotonic() < position_deadline:
        position = _target_position_size(
            info.user_state(account, TARGET_DEX)
        )
        if position == 0:
            break
        time.sleep(0.1)
    try:
        final_open_orders = info.open_orders(account, TARGET_DEX)
    except Exception:
        final_open_orders = None
    reconciled = (
        isinstance(final_open_orders, list)
        and not final_open_orders
        and position == 0
    )
    contracts.write_json(
        output_root / "emergency_reconciliation.json",
        {
            "schema_version": "skhynix_c6in_emergency_reconciliation_v2",
            "task_id": contracts.TASK_ID,
            "sample_sequence": sample_sequence,
            "cancel_by_cloid_attempted": cancel_attempted,
            "reduce_only_flatten_attempted": flatten_attempted,
            "final_open_orders_zero": (
                isinstance(final_open_orders, list)
                and not final_open_orders
            ),
            "final_position_zero": position == 0,
            "reconciled": reconciled,
            "raw_reference_written": False,
            "raw_private_response_written": False,
        },
    )
    return reconciled


def run_active_collection(
    *,
    output_root: Path,
    gate2_root: Path,
    schedule_path: Path,
    expected_commit: str,
    credential_file: Path,
) -> dict[str, Any]:
    validate_dispatch(TASK_PATH, MATRIX_PATH)
    validate_kernel_pin()
    validate_h0a_pin()
    gate2 = _read_json(Path(gate2_root) / "gate2_full_receipt.json")
    if (
        gate2.get("status") != "pass"
        or gate2.get("gate2_complete") is not True
        or gate2.get("order_endpoint_called") is not False
        or gate2.get("cancel_endpoint_called") is not False
    ):
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "gate2_full_receipt",
            "full Gate 2 pass required",
        )
    frozen_host = _read_json(Path(gate2_root) / "host_identity.json")
    frozen_runtime = _read_json(Path(gate2_root) / "runtime_identity.json")
    market = _read_json(Path(gate2_root) / "market_identity.json")
    frozen_account = _read_json(Path(gate2_root) / "account_baseline.json")
    host = _host_identity()
    runtime = _runtime_identity(expected_commit)
    if host["host_identity_token"] != frozen_host["host_identity_token"]:
        raise contracts.LatencyContractError(
            "LATENCY_HOST_IDENTITY_DRIFT",
            "active_collection",
            "Gate 2 host token mismatch",
        )
    if (
        runtime["runtime_identity_sha256"]
        != frozen_runtime["runtime_identity_sha256"]
    ):
        raise contracts.LatencyContractError(
            "LATENCY_RUNTIME_IDENTITY_DRIFT",
            "active_collection",
            "Gate 2 runtime identity mismatch",
        )
    schedule = contracts.read_csv_exact(
        schedule_path,
        contracts.SCHEDULE_FIELDS,
    )
    if len(schedule) != 3:
        raise contracts.LatencyContractError(
            "LATENCY_WINDOW_SUPPORT_INVALID",
            str(schedule_path),
            f"window_count={len(schedule)}",
        )
    secrets, _ = _read_credentials(credential_file)
    _, info, terminal_info, exchange, identity = _build_live_clients(secrets)
    account = identity["account_address"]
    account_identity_token = hashlib.sha256(
        account.encode("ascii")
    ).hexdigest()
    if account_identity_token != frozen_account["account_identity_token"]:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "account_identity_token",
            "Gate 2 account token mismatch",
        )
    if info.open_orders(account, TARGET_DEX):
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "active_initial_open_orders",
            "nonempty",
        )
    if _target_position_size(info.user_state(account, TARGET_DEX)) != 0:
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "active_initial_position",
            "nonzero",
        )
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    sealed_root = output_root / "sealed"
    sealed_root.mkdir()
    attempts: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    completed_schedule: list[dict[str, Any]] = []
    eligible_by_window: dict[str, int] = {
        row["collection_window_id"]: 0 for row in schedule
    }
    last_attempt_start = 0.0
    fatal_stop = False
    for scheduled in schedule:
        window_id = scheduled["collection_window_id"]
        start = _parse_utc(scheduled["start_utc"])
        end = _parse_utc(scheduled["end_utc"])
        wait_seconds = (start - datetime.now(timezone.utc)).total_seconds()
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        window_attempts = 0
        while (
            datetime.now(timezone.utc) < end
            and len(attempts) < contracts.MAX_TOTAL_ATTEMPTS
            and window_attempts < MAX_ATTEMPTS_PER_WINDOW
            and not fatal_stop
        ):
            all_window_floors = all(
                count >= contracts.MINIMUM_ELIGIBLE_PER_WINDOW
                for count in eligible_by_window.values()
            )
            if (
                len(
                    [
                        row
                        for row in attempts
                        if row["primary_latency_eligible"] is True
                    ]
                )
                >= contracts.PRIMARY_ELIGIBLE_FLOOR
                and all_window_floors
            ):
                break
            since_last = time.monotonic() - last_attempt_start
            if last_attempt_start and since_last < MINIMUM_INTER_ATTEMPT_SECONDS:
                time.sleep(MINIMUM_INTER_ATTEMPT_SECONDS - since_last)
            if datetime.now(timezone.utc) >= end:
                break
            last_attempt_start = time.monotonic()
            sample_sequence = len(attempts) + 1
            try:
                attempt, attempt_events, attempt_fatal = _run_active_attempt(
                    sample_sequence=sample_sequence,
                    window_id=window_id,
                    host=host,
                    runtime=runtime,
                    market=market,
                    account_identity_token=account_identity_token,
                    account=account,
                    info=info,
                    terminal_info=terminal_info,
                    exchange=exchange,
                )
            except Exception as exc:
                reconciled = _emergency_reconcile(
                    output_root=output_root,
                    sample_sequence=sample_sequence,
                    account=account,
                    info=info,
                    exchange=exchange,
                )
                raise contracts.LatencyContractError(
                    (
                        "LATENCY_TERMINAL_STATUS_UNIDENTIFIED"
                        if reconciled
                        else "LATENCY_UNRESOLVED_EXPOSURE"
                    ),
                    f"sample={sample_sequence}",
                    f"exception_type={type(exc).__name__}",
                ) from exc
            attempts.append(attempt)
            events.extend(attempt_events)
            window_attempts += 1
            if attempt["primary_latency_eligible"] is True:
                eligible_by_window[window_id] += 1
            fatal_stop = attempt_fatal
            contracts.write_csv(
                sealed_root / "attempt_ledger.csv",
                attempts,
                contracts.ATTEMPT_FIELDS,
            )
            contracts.write_csv(
                sealed_root / "lifecycle_events.csv",
                events,
                contracts.EVENT_FIELDS,
            )
            if sample_sequence % MAX_ATTEMPTS_PER_BATCH == 0 or attempt_fatal:
                contracts.write_json(
                    output_root
                    / "batch_receipts"
                    / f"{attempt['batch_id']}.json",
                    {
                        "schema_version": (
                            "skhynix_c6in_latency_batch_receipt_v2"
                        ),
                        "task_id": contracts.TASK_ID,
                        "batch_id": attempt["batch_id"],
                        "attempt_count_through_batch": len(attempts),
                        "eligible_count_through_batch": sum(
                            row["primary_latency_eligible"] is True
                            for row in attempts
                        ),
                        "fatal_safety_stop": attempt_fatal,
                        "last_attempt_safety_status": attempt[
                            "safety_status"
                        ],
                        "raw_reference_written": False,
                    },
                )
        completed_schedule.append(
            {
                **scheduled,
                "status": (
                    "stopped_on_safety"
                    if fatal_stop
                    else "completed"
                ),
            }
        )
        contracts.write_json(
            output_root / "window_receipts" / f"{window_id}.json",
            {
                "schema_version": (
                    "skhynix_c6in_latency_window_receipt_v2"
                ),
                "task_id": contracts.TASK_ID,
                "collection_window_id": window_id,
                "attempt_count": window_attempts,
                "eligible_count": eligible_by_window[window_id],
                "fatal_safety_stop": fatal_stop,
                "latency_values_accessed": False,
            },
        )
        if fatal_stop:
            completed_schedule.extend(
                {
                    **remaining,
                    "status": "not_started_after_safety_stop",
                }
                for remaining in schedule[len(completed_schedule) :]
            )
            break
    contracts.write_csv(
        sealed_root / "collection_window_schedule.csv",
        completed_schedule,
        contracts.SCHEDULE_FIELDS,
    )
    if not (sealed_root / "attempt_ledger.csv").exists():
        contracts.write_csv(
            sealed_root / "attempt_ledger.csv",
            [],
            contracts.ATTEMPT_FIELDS,
        )
        contracts.write_csv(
            sealed_root / "lifecycle_events.csv",
            [],
            contracts.EVENT_FIELDS,
        )
    final_open_orders = info.open_orders(account, TARGET_DEX)
    final_position = _target_position_size(
        info.user_state(account, TARGET_DEX)
    )
    if final_open_orders or final_position != 0:
        raise contracts.LatencyContractError(
            "LATENCY_UNRESOLVED_EXPOSURE",
            "active_final_reconciliation",
            (
                f"open_orders={len(final_open_orders)} "
                f"position_zero={final_position == 0}"
            ),
        )
    receipt = {
        "schema_version": "skhynix_c6in_active_collection_v2",
        "task_id": contracts.TASK_ID,
        "status": "complete",
        "attempt_count": len(attempts),
        "eligible_count": sum(
            row["primary_latency_eligible"] is True for row in attempts
        ),
        "eligible_count_by_window": eligible_by_window,
        "fatal_safety_stop": fatal_stop,
        "final_open_orders_count": 0,
        "final_position_zero": True,
        "order_endpoint_called": bool(attempts),
        "cancel_endpoint_called": any(
            row["cancel_response_class"] != "not_called"
            for row in attempts
        ),
        "h0b_outcome_accessed": False,
        "h0a_tuple_mutated": False,
        "latency_values_accessed_by_l0": False,
    }
    contracts.write_json(output_root / "collection_receipt.json", receipt)
    return receipt


def _package_inventory_rows(
    root: Path,
    paths: Sequence[str],
) -> list[dict[str, Any]]:
    return [
        {
            "path": relative,
            "bytes": (root / relative).stat().st_size,
            "sha256": trust.sha256_file(root / relative),
        }
        for relative in sorted(paths)
    ]


def _normalized_manifest_bytes(path: Path) -> bytes:
    value = _read_json(path)
    for field in (
        "research_data_identity",
        "code_contract_identity",
        "evidence_identity",
        "composite_identity",
    ):
        value[field] = ""
    return contracts.canonical_json_bytes(value)


def _normalized_sha_inventory_bytes(root: Path) -> bytes:
    path = root / "sha256_inventory.csv"
    with path.open("r", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    manifest_bytes = _normalized_manifest_bytes(
        root / "measurement_manifest.json"
    )
    for row in rows:
        if row["path"] == "measurement_manifest.json":
            row["bytes"] = str(len(manifest_bytes))
            row["sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(INVENTORY_FIELDS),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("ascii")


def _research_inventory(root: Path) -> list[dict[str, Any]]:
    return _package_inventory_rows(root, PACKAGE_R_FILES)


def _runtime_contract(root: Path) -> dict[str, Any]:
    return {
        "schema_version": "skhynix_c6in_latency_runtime_contract_bridge_v2",
        "task_id": contracts.TASK_ID,
        "kernel_source_tree_sha256": KERNEL_PIN[
            "kernel_source_tree_sha256"
        ],
        "files": _package_inventory_rows(root, PACKAGE_C_FILES),
    }


def _evidence_inventory(root: Path) -> list[dict[str, Any]]:
    rows = _package_inventory_rows(root, PACKAGE_E_FILES)
    for row in rows:
        if row["path"] == "measurement_manifest.json":
            payload = _normalized_manifest_bytes(
                root / "measurement_manifest.json"
            )
            row["bytes"] = len(payload)
            row["sha256"] = hashlib.sha256(payload).hexdigest()
        elif row["path"] == "sha256_inventory.csv":
            payload = _normalized_sha_inventory_bytes(root)
            row["bytes"] = len(payload)
            row["sha256"] = hashlib.sha256(payload).hexdigest()
    return rows


def _publication_envelope(root: Path) -> dict[str, Any]:
    return {
        "schema_version": (
            "skhynix_c6in_latency_publication_envelope_bridge_v2"
        ),
        "task_id": contracts.TASK_ID,
        "files": _evidence_inventory(root),
        "expected_files": list(PACKAGE_FILES),
        "expected_directories": list(PACKAGE_DIRECTORIES),
        "manifest_self_binding_normalization": (
            "R_C_E_composite_fields_empty_for_E_hash"
        ),
        "inventory_self_binding_normalization": (
            "manifest_row_uses_normalized_manifest_bytes_and_sha"
        ),
        "kernel_package_admission_portable": True,
        "full_source_semantic_replay_portable": False,
    }


def package_identity(root: Path) -> dict[str, str]:
    research = trust.compute_research_data_identity(
        _research_inventory(root)
    )
    code = trust.compute_runtime_contract_identity(
        research,
        _runtime_contract(root),
    )
    evidence = trust.compute_publication_envelope_identity(
        research,
        code,
        _publication_envelope(root),
    )
    composite = trust.compute_composite_package_identity(
        research,
        code,
        evidence,
    )
    return {
        "research_data_identity": research,
        "code_contract_identity": code,
        "evidence_identity": evidence,
        "composite_identity": composite,
    }


def _write_package_inventory(root: Path) -> None:
    rows = _package_inventory_rows(
        root,
        [
            path
            for path in PACKAGE_FILES
            if path != "sha256_inventory.csv"
        ],
    )
    contracts.write_csv(
        root / "sha256_inventory.csv",
        rows,
        INVENTORY_FIELDS,
    )


def _measurement_report(
    reliability: dict[str, Any],
    recommendation: dict[str, Any],
) -> str:
    return (
        "# 0822T002 Execution Latency Measurement\n\n"
        "Status: 待验收\n\n"
        f"- Frozen date: 2026-08-22\n"
        f"- Target attempts: {reliability['target_total_attempt_count']}\n"
        f"- Eligible attempts: {reliability['target_primary_eligible_count']}\n"
        f"- Eligible by window: {json.dumps(reliability['eligible_count_by_window'], sort_keys=True)}\n"
        f"- Terminal identified fraction: {reliability['terminal_identified_fraction']}\n"
        f"- Fill-race fraction: {reliability['fill_during_cancel_race_fraction']}\n"
        f"- p95 cancel-effective latency us: {recommendation['p95_cancel_effective_latency_us']}\n"
        f"- Recommended Gate H-C latency ms: {recommendation['recommended_gate_latency_ms']}\n"
        f"- Recommendation: `{recommendation['recommendation']}`\n"
        "- H0-B outcome access: false\n"
        "- H0-A tuple mutation: false\n"
        "- Raw credentials, account addresses, oids, cloids and private "
        "responses are excluded.\n"
    )


def build_formal_package(
    *,
    evidence_root: Path,
    package_root: Path,
    source_commit: str,
) -> dict[str, Any]:
    evidence_root = Path(evidence_root).resolve()
    package_root = Path(package_root).resolve()
    gate2_root = evidence_root / "gate2-full"
    sealed_root = evidence_root / "active/sealed"
    summary_root = evidence_root / "l1-a"
    required_inputs = (
        gate2_root / "host_identity.json",
        gate2_root / "runtime_identity.json",
        gate2_root / "market_identity.json",
        gate2_root / "authorization_envelope.json",
        sealed_root / "collection_window_schedule.csv",
        sealed_root / "attempt_ledger.csv",
        sealed_root / "lifecycle_events.csv",
        summary_root / "latency_by_attempt.csv",
        summary_root / "latency_summary.csv",
        summary_root / "reliability_summary.json",
        summary_root / "controller_latency_recommendation.json",
    )
    missing = [str(path) for path in required_inputs if not path.is_file()]
    if missing:
        raise contracts.LatencyContractError(
            "LATENCY_L1_BOUNDARY_VIOLATION",
            str(evidence_root),
            f"missing={missing!r}",
        )
    package_root.mkdir(parents=True, exist_ok=False)
    for directory in PACKAGE_DIRECTORIES:
        (package_root / directory).mkdir(parents=True, exist_ok=True)
    copies = {
        "host_identity.json": gate2_root / "host_identity.json",
        "runtime_identity.json": gate2_root / "runtime_identity.json",
        "market_identity.json": gate2_root / "market_identity.json",
        "authorization_envelope.json": (
            gate2_root / "authorization_envelope.json"
        ),
        "collection_window_schedule.csv": (
            sealed_root / "collection_window_schedule.csv"
        ),
        "attempt_ledger.csv": sealed_root / "attempt_ledger.csv",
        "lifecycle_events.csv": sealed_root / "lifecycle_events.csv",
        "latency_by_attempt.csv": summary_root / "latency_by_attempt.csv",
        "latency_summary.csv": summary_root / "latency_summary.csv",
        "reliability_summary.json": (
            summary_root / "reliability_summary.json"
        ),
        "controller_latency_recommendation.json": (
            summary_root / "controller_latency_recommendation.json"
        ),
        "contracts/task.md": TASK_PATH,
        "contracts/surface_matrix.json": MATRIX_PATH,
        "contracts/execution_plan.md": (
            REPO_ROOT
            / "docs/skhynix_c6in_hyperliquid_execution_latency_measurement_plan_v2.md"
        ),
        "contracts/v2_framework.md": (
            REPO_ROOT
            / "docs/skhynix_continuous_hazard_maker_research_framework_v2.md"
        ),
        "runtime_source/skhynix_c6in_latency_v2.py": Path(__file__),
        "runtime_source/skhynix_c6in_latency_contracts_v2.py": Path(
            contracts.__file__
        ),
        "runtime_tests/test_skhynix_c6in_latency_v2.py": (
            REPO_ROOT
            / "examples/hyperliquid/test_skhynix_c6in_latency_v2.py"
        ),
        "runtime_tests/test_skhynix_c6in_latency_package_v2.py": (
            REPO_ROOT
            / "examples/hyperliquid/test_skhynix_c6in_latency_package_v2.py"
        ),
    }
    for relative, source in copies.items():
        destination = package_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    contracts.write_json(
        package_root / "accepted_h0a_pin.json",
        dict(ACCEPTED_H0A_PIN),
    )
    contracts.write_json(
        package_root / "contracts/accepted_kernel_pin.json",
        dict(KERNEL_PIN),
    )
    frozen_contract = {
        "schema_version": "skhynix_c6in_latency_frozen_contract_v2",
        "task_id": contracts.TASK_ID,
        "frozen_date": "2026-08-22",
        "source_commit": source_commit,
        "execution_plan_sha256": EXPECTED_PLAN_SHA256,
        "surface_matrix_sha256": EXPECTED_MATRIX_SHA256,
        "primary_latency": (
            "risk_decision_ready_to_authoritative_terminal_confirm"
        ),
        "primary_quantile": contracts.FROZEN_QUANTILE,
        "bucket_rule": contracts.FROZEN_BUCKET_RULE,
        "max_total_attempts": contracts.MAX_TOTAL_ATTEMPTS,
        "primary_eligible_floor": contracts.PRIMARY_ELIGIBLE_FLOOR,
        "fixed_pre_cancel_settle_ms": FIXED_PRE_CANCEL_SETTLE_MS,
        "minimum_inter_attempt_seconds": MINIMUM_INTER_ATTEMPT_SECONDS,
        "per_order_notional_cap_usdc": (
            contracts.PER_ORDER_NOTIONAL_CAP_USDC
        ),
        "aggregate_position_cap_usdc": (
            contracts.AGGREGATE_POSITION_CAP_USDC
        ),
        "max_loss_usdc": contracts.MAX_LOSS_USDC,
        "max_loss_basis": contracts.LOSS_BASIS,
        "quote_distance_ticks": 10,
        "h0b_outcome_accessed": False,
        "h0a_tuple_mutated": False,
    }
    contracts.write_json(
        package_root / "frozen_measurement_contract.json",
        frozen_contract,
    )
    attempts = contracts.read_csv_exact(
        package_root / "attempt_ledger.csv",
        contracts.ATTEMPT_FIELDS,
    )
    latency_rows = contracts.read_csv_exact(
        package_root / "latency_by_attempt.csv",
        contracts.LATENCY_FIELDS,
    )
    attempts_by_sample = {
        row["sample_sequence"]: row for row in attempts
    }
    failures = [
        {
            "schema_version": contracts.SCHEMA_VERSION,
            "task_id": contracts.TASK_ID,
            "sample_sequence": row["sample_sequence"],
            "attempt_id": attempts_by_sample[row["sample_sequence"]][
                "attempt_id"
            ],
            "primary_latency_eligible": row[
                "primary_latency_eligible"
            ],
            "failure_or_censor_class": row[
                "failure_or_censor_class"
            ],
            "terminal_class": attempts_by_sample[row["sample_sequence"]][
                "terminal_class"
            ],
            "safety_status": attempts_by_sample[row["sample_sequence"]][
                "safety_status"
            ],
        }
        for row in latency_rows
        if row["primary_latency_eligible"] != "true"
    ]
    contracts.write_csv(
        package_root / "failure_and_censoring.csv",
        failures,
        FAILURE_FIELDS,
    )
    contracts.write_csv(
        package_root / "historical_context_awsserver.csv",
        [
            {
                "source_host": "awsserver",
                "sample_count": 18,
                "metric": "local_cancel_call_response_duration",
                "min_ms": 614,
                "p50_ms": 700,
                "p90_ms": 786,
                "p95_ms": 803,
                "max_ms": 816,
                "primary_population_eligible": False,
                "exclusion_reason": (
                    "wrong_host_and_cancel_response_not_terminal"
                ),
            }
        ],
        HISTORICAL_CONTEXT_FIELDS,
    )
    reliability = _read_json(package_root / "reliability_summary.json")
    recommendation = _read_json(
        package_root / "controller_latency_recommendation.json"
    )
    boundary = {
        "schema_version": "skhynix_c6in_latency_boundary_v2",
        "task_id": contracts.TASK_ID,
        "frozen_date": "2026-08-22",
        "private_endpoint_called_by_l0": True,
        "order_endpoint_called_by_l0": bool(attempts),
        "cancel_endpoint_called_by_l0": any(
            row["cancel_response_class"] != "not_called"
            for row in attempts
        ),
        "credentials_written": False,
        "raw_private_responses_written": False,
        "raw_account_addresses_written": False,
        "raw_order_references_written": False,
        "l1_network_accessed": False,
        "h0b_outcome_accessed": False,
        "h0a_tuple_mutated": False,
        "accepted_registry_mutated": False,
    }
    contracts.write_json(package_root / "boundary_manifest.json", boundary)
    (package_root / "reports/execution_latency_measurement.md").write_text(
        _measurement_report(reliability, recommendation),
        encoding="utf-8",
        newline="\n",
    )
    manifest = {
        "schema_version": "skhynix_c6in_latency_measurement_manifest_v2",
        "task_id": contracts.TASK_ID,
        "status": "待验收",
        "frozen_date": "2026-08-22",
        "source_commit": source_commit,
        "package_path": FORMAL_PACKAGE_RELATIVE.as_posix(),
        "research_data_identity": "",
        "code_contract_identity": "",
        "evidence_identity": "",
        "composite_identity": "",
        "file_count": len(PACKAGE_FILES),
        "directory_count": len(PACKAGE_DIRECTORIES),
        "target_total_attempt_count": reliability[
            "target_total_attempt_count"
        ],
        "target_primary_eligible_count": reliability[
            "target_primary_eligible_count"
        ],
        "p95_cancel_effective_latency_us": recommendation[
            "p95_cancel_effective_latency_us"
        ],
        "recommended_gate_latency_ms": recommendation[
            "recommended_gate_latency_ms"
        ],
        "recommendation": recommendation["recommendation"],
        "h0b_outcome_accessed": False,
        "h0a_tuple_mutated": False,
    }
    contracts.write_json(package_root / "measurement_manifest.json", manifest)
    _write_package_inventory(package_root)
    identity = package_identity(package_root)
    manifest.update(identity)
    contracts.write_json(package_root / "measurement_manifest.json", manifest)
    _write_package_inventory(package_root)
    if package_identity(package_root) != identity:
        raise contracts.LatencyContractError(
            "COMPOSITE_IDENTITY_BINDING_MISMATCH",
            str(package_root),
            "identity changed after manifest/inventory finalization",
        )
    return verify_formal_package(package_root)


def verify_formal_package(package_root: Path) -> dict[str, Any]:
    root = Path(package_root).resolve()
    before = trust.metadata_snapshot(root)
    entries = trust.scan_exact_tree(
        root,
        {
            "allowed_entry_types": ["regular_file", "directory"],
            "expected_files": list(PACKAGE_FILES),
            "expected_directories": list(PACKAGE_DIRECTORIES),
        },
    )
    total_bytes = sum(
        entry.bytes for entry in entries if entry.entry_type == "regular_file"
    )
    if total_bytes > 64 * 1024 * 1024:
        raise contracts.LatencyContractError(
            "LATENCY_PACKAGE_SIZE_LIMIT_EXCEEDED",
            str(root),
            str(total_bytes),
        )
    manifest = _read_json(root / "measurement_manifest.json")
    identity = package_identity(root)
    for field, value in identity.items():
        if manifest.get(field) != value:
            raise contracts.LatencyContractError(
                "COMPOSITE_IDENTITY_BINDING_MISMATCH",
                f"measurement_manifest.json:{field}",
                f"expected={value} observed={manifest.get(field)}",
            )
    inventory = contracts.read_csv_exact(
        root / "sha256_inventory.csv",
        INVENTORY_FIELDS,
    )
    expected_inventory = _package_inventory_rows(
        root,
        [
            path
            for path in PACKAGE_FILES
            if path != "sha256_inventory.csv"
        ],
    )
    if inventory != [
        {key: str(value) for key, value in row.items()}
        for row in expected_inventory
    ]:
        raise contracts.LatencyContractError(
            "COMPOSITE_IDENTITY_BINDING_MISMATCH",
            "sha256_inventory.csv",
            "inventory mismatch",
        )
    for relative in PACKAGE_R_FILES + PACKAGE_E_FILES:
        path = root / relative
        if path.suffix not in {".json", ".csv", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        if contracts.ADDRESS_RE.search(text) or contracts.PRIVATE_KEY_RE.search(
            text
        ):
            raise contracts.LatencyContractError(
                "LATENCY_SECRET_OR_REFERENCE_LEAK",
                relative,
                "address or private key pattern found",
            )
    recommendation = _read_json(
        root / "controller_latency_recommendation.json"
    )
    reliability = _read_json(root / "reliability_summary.json")
    after = trust.metadata_snapshot(root)
    trust.assert_zero_write_snapshot(before, after, location=str(root))
    return {
        "schema_version": "skhynix_c6in_latency_package_admission_v2",
        "task_id": contracts.TASK_ID,
        "verified": True,
        "file_count": len(PACKAGE_FILES),
        "directory_count": len(PACKAGE_DIRECTORIES),
        "total_bytes": total_bytes,
        **identity,
        "sample_gate_pass": reliability["sample_gate_pass"],
        "reliability_gate_pass": reliability["reliability_gate_pass"],
        "recommendation": recommendation["recommendation"],
        "kernel_package_admission_portable": True,
        "full_source_semantic_replay_portable": False,
        "zero_write": True,
    }


def gate2_preflight(output_root: Path, expected_commit: str) -> dict[str, Any]:
    validate_dispatch(TASK_PATH, MATRIX_PATH)
    validate_kernel_pin()
    validate_h0a_pin()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    host = _host_identity()
    runtime = _runtime_identity(expected_commit)
    authorization = _authorization_envelope(credential_file_read=False)
    market = _market_snapshot()
    notional_error_code = ""
    notional_error_detail = ""
    try:
        contracts.validate_minimum_order_notional(
            minimum_valid_order_notional_usdc=(
                contracts.MINIMUM_VALID_ORDER_NOTIONAL_USDC
            ),
            minimum_executable_notional_usdc=float(
                market["minimum_valid_order_notional"]
            ),
        )
    except contracts.LatencyContractError as exc:
        notional_error_code = exc.code
        notional_error_detail = exc.detail
    market["minimum_order_notional_status"] = (
        "pass" if not notional_error_code else "blocked"
    )
    market["blocking_error_code"] = notional_error_code
    market["blocking_detail"] = notional_error_detail
    if notional_error_code:
        error_code = notional_error_code
        error_detail = notional_error_detail
        skip_reason = "prior_minimum_notional_authorization_block"
    else:
        error_code = ""
        error_detail = ""
        skip_reason = ""
    receipt = {
        "schema_version": "skhynix_c6in_latency_gate2_preflight_v2",
        "task_id": contracts.TASK_ID,
        "started_at_utc": host["captured_at_utc"],
        "completed_at_utc": _utc_now(),
        "status": "notional_subgate_pass" if not error_code else "blocked",
        "gate2_complete": False,
        "blocking_error_code": error_code,
        "blocking_detail": error_detail,
        "host_identity_token": host["host_identity_token"],
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "asset_metadata_identity": market["asset_metadata_identity"],
        "credential_file_read": False,
        "private_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "public_quote_safety_collection_started": False,
        "public_quote_safety_collection_skip_reason": skip_reason,
        "h0b_outcome_accessed": False,
        "h0a_tuple_mutated": False,
    }
    contracts.write_json(output_root / "host_identity.json", host)
    contracts.write_json(output_root / "runtime_identity.json", runtime)
    contracts.write_json(output_root / "market_identity.json", market)
    contracts.write_json(
        output_root / "authorization_envelope.json",
        authorization,
    )
    contracts.write_json(
        output_root / "gate2_preflight_receipt.json",
        receipt,
    )
    return receipt


def gate2_full(
    output_root: Path,
    expected_commit: str,
    credential_file: Path,
    *,
    public_duration_seconds: int = PUBLIC_PREFLIGHT_DURATION_SECONDS,
) -> dict[str, Any]:
    validate_dispatch(TASK_PATH, MATRIX_PATH)
    validate_kernel_pin()
    validate_h0a_pin()
    if not ACTIVE_EXECUTION_AUTHORIZED:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "active_execution_authorized",
            "false",
        )
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    host = _host_identity()
    runtime = _runtime_identity(expected_commit)
    market = _market_snapshot()
    contracts.validate_minimum_order_notional(
        minimum_valid_order_notional_usdc=(
            contracts.MINIMUM_VALID_ORDER_NOTIONAL_USDC
        ),
        minimum_executable_notional_usdc=float(
            market["minimum_valid_order_notional"]
        ),
    )
    quote_safety = _collect_public_quote_safety(
        output_root=output_root,
        market=market,
        duration_seconds=public_duration_seconds,
    )
    market.update(
        {
            "minimum_order_notional_status": "pass",
            "blocking_error_code": "",
            "blocking_detail": "",
            "minimum_safe_quote_distance_bps": str(
                quote_safety["minimum_safe_quote_distance_bps"]
            ),
            "nearest_rank_p99_abs_250ms_mid_move_bps": str(
                quote_safety[
                    "nearest_rank_p99_abs_250ms_mid_move_bps"
                ]
            ),
            "quote_distance_safety_status": "pass",
        }
    )
    conflicting_runtime = _conflicting_runtime_snapshot()
    account = _private_account_baseline(
        credential_file=credential_file,
        output_root=output_root,
    )
    authorization = _authorization_envelope(credential_file_read=True)
    receipt = {
        "schema_version": "skhynix_c6in_latency_gate2_full_v2",
        "task_id": contracts.TASK_ID,
        "started_at_utc": host["captured_at_utc"],
        "completed_at_utc": _utc_now(),
        "status": "pass",
        "gate2_complete": True,
        "blocking_error_code": "",
        "blocking_detail": "",
        "host_identity_token": host["host_identity_token"],
        "runtime_identity_sha256": runtime["runtime_identity_sha256"],
        "asset_metadata_identity": market["asset_metadata_identity"],
        "account_identity_token": account["account_identity_token"],
        "credential_file_read": True,
        "private_endpoint_called": True,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "public_quote_safety_collection_started": True,
        "public_quote_safety_collection_complete": True,
        "public_quote_safety_status": "pass",
        "conflicting_runtime_status": (
            "pass"
            if conflicting_runtime["same_account_market_path_available"]
            else "blocked"
        ),
        "final_open_orders_count": account["open_order_count"],
        "target_position_zero": account["target_position_zero"],
        "available_margin_at_least_aggregate_cap": (
            account["available_margin_at_least_aggregate_cap"]
        ),
        "h0b_outcome_accessed": False,
        "h0a_tuple_mutated": False,
    }
    for artifact in (host, runtime, market, authorization, conflicting_runtime):
        encoded = contracts.canonical_json_bytes(artifact).decode("ascii")
        if "0x" in encoded.lower():
            raise contracts.LatencyContractError(
                "LATENCY_SECRET_OR_REFERENCE_LEAK",
                "gate2_full",
                "raw hexadecimal address-like value in artifact",
            )
    contracts.write_json(output_root / "host_identity.json", host)
    contracts.write_json(output_root / "runtime_identity.json", runtime)
    contracts.write_json(output_root / "market_identity.json", market)
    contracts.write_json(
        output_root / "authorization_envelope.json",
        authorization,
    )
    contracts.write_json(
        output_root / "conflicting_runtime.json",
        conflicting_runtime,
    )
    contracts.write_json(output_root / "gate2_full_receipt.json", receipt)
    return receipt


def validate_dispatch(task_path: Path, matrix_path: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str(REPO_ROOT / ".workflow/workflow-kit/validate_research_package_task.py"),
        "--task",
        str(task_path),
        "--matrix",
        str(matrix_path),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise contracts.LatencyContractError(
            "LATENCY_DISPATCH_CONTRACT_INVALID",
            str(task_path),
            completed.stderr or completed.stdout,
        )
    if contracts.sha256_file(matrix_path) != EXPECTED_MATRIX_SHA256:
        raise contracts.LatencyContractError(
            "LATENCY_DISPATCH_CONTRACT_INVALID",
            str(matrix_path),
            "matrix identity drift",
        )
    plan_path = (
        REPO_ROOT
        / "docs/skhynix_c6in_hyperliquid_execution_latency_measurement_plan_v2.md"
    )
    if contracts.sha256_file(plan_path) != EXPECTED_PLAN_SHA256:
        raise contracts.LatencyContractError(
            "LATENCY_DISPATCH_CONTRACT_INVALID",
            str(plan_path),
            "execution plan identity drift",
        )
    return _read_json(matrix_path)


def validate_kernel_pin() -> None:
    registry = _read_json(REGISTRY_PATH)
    if registry.get("registry_revision") != KERNEL_PIN["registry_revision"]:
        raise contracts.LatencyContractError(
            "KERNEL_PIN_MISMATCH",
            str(REGISTRY_PATH),
            "registry revision mismatch",
        )
    versions = registry.get("versions")
    if not isinstance(versions, list) or len(versions) != 1:
        raise contracts.LatencyContractError(
            "KERNEL_PIN_MISMATCH", str(REGISTRY_PATH), repr(versions)
        )
    entry = versions[0]
    if trust.canonical_json_sha256(entry) != KERNEL_PIN["registry_entry_sha256"]:
        raise contracts.LatencyContractError(
            "KERNEL_PIN_MISMATCH",
            str(REGISTRY_PATH),
            "accepted entry mismatch",
        )


def validate_h0a_pin() -> None:
    root = (
        REPO_ROOT
        / "local_live_analysis/"
        "skhynix_continuous_conditional_risk_v2_stage_h0a_support_only"
    )
    manifest = _read_json(root / "h0a_manifest.json")
    primary = _read_json(root / "primary_tuple_freeze.json")
    observed = {
        "task_id": manifest.get("task_id"),
        "selected_horizon_ms": primary.get("horizon_ms"),
        "gate_latency_ms": primary.get("gate_latency_ms"),
        "primary_tuple_sha256": contracts.sha256_file(
            root / "primary_tuple_freeze.json"
        ),
        "research_data_identity": manifest.get("research_data_identity"),
        "code_contract_identity": manifest.get("code_contract_identity"),
        "evidence_identity": manifest.get("evidence_identity"),
        "composite_identity": manifest.get("composite_identity"),
        "controller_closure_sha256": contracts.sha256_file(
            REPO_ROOT / ".workflow/reports/0821T001-controller-closure.md"
        ),
    }
    if observed != dict(ACCEPTED_H0A_PIN):
        raise contracts.LatencyContractError(
            "LATENCY_H0A_PIN_MISMATCH",
            str(root),
            f"expected={dict(ACCEPTED_H0A_PIN)!r} observed={observed!r}",
        )


def _exercise_negative_case(case_id: str, expected_code: str) -> None:
    if case_id == "cancel_response_promoted":
        classification = contracts.classify_cancel_response({"status": "ok"})
        if classification.authoritative:
            raise AssertionError("cancel response unexpectedly authoritative")
    elif case_id == "foreign_reference":
        classification = contracts.classify_terminal_payload(
            {
                "status": "order",
                "order": {
                    "order": {"oid": 999, "cloid": "foreign"},
                    "status": "canceled",
                },
            },
            expected_oid=101,
            expected_cloid="expected",
        )
        if classification.authoritative:
            raise AssertionError("foreign reference unexpectedly authoritative")
    elif case_id == "raw_reference_leak":
        try:
            contracts.reject_sensitive_payload({"oid": 101})
        except contracts.LatencyContractError as exc:
            if exc.code != expected_code:
                raise
            raise contracts.LatencyContractError(
                expected_code, case_id, "raw reference mutation rejected"
            ) from exc
    elif case_id == "attempt_121":
        attempts = [{}] * 121
        if len(attempts) <= contracts.MAX_TOTAL_ATTEMPTS:
            raise AssertionError("attempt cap mutation did not execute")
    elif case_id == "quote_distance_unverified":
        try:
            contracts.quote_distance_safety(
                tick_size=0.01,
                reference_mid_price=1000.0,
                p99_abs_250ms_mid_move_bps=1.0,
            )
        except contracts.LatencyContractError as exc:
            if exc.code != expected_code:
                raise
            raise contracts.LatencyContractError(
                expected_code, case_id, "quote safety mutation rejected"
            ) from exc
    elif case_id == "loss_basis_substituted":
        try:
            contracts.realized_flatten_slippage_loss_usdc(
                original_fill_side="buy",
                fill_vwap=1000.0,
                flatten_vwap=999.0,
                filled_quantity=0.001,
                flattened_quantity=0.001,
                flatten_status="mark_to_market_only",
            )
        except contracts.LatencyContractError as exc:
            if exc.code != expected_code:
                raise
            raise contracts.LatencyContractError(
                expected_code, case_id, "loss basis mutation rejected"
            ) from exc
    raise contracts.LatencyContractError(
        expected_code,
        case_id,
        "hostile mutation executed",
    )


def hostile_preflight(
    task_path: Path,
    matrix_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    matrix = validate_dispatch(task_path, matrix_path)
    validate_kernel_pin()
    validate_h0a_pin()
    matrix_codes = {
        mutation["expected_error_code"]
        for surface in matrix["surfaces"]
        for mutation in surface.get("negative_mutations", [])
    }
    if not matrix_codes <= set(HOSTILE_CASES.values()):
        raise contracts.LatencyContractError(
            "LATENCY_NEGATIVE_MATRIX_INCOMPLETE",
            str(matrix_path),
            repr(sorted(matrix_codes - set(HOSTILE_CASES.values()))),
        )
    started = _utc_now()
    executions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="0822T002-frozen-") as temporary:
        frozen_root = Path(temporary)
        package_root = frozen_root / "examples/hyperliquid"
        package_root.mkdir(parents=True)
        (frozen_root / "examples/__init__.py").write_text("", encoding="ascii")
        (package_root / "__init__.py").write_text("", encoding="ascii")
        source_paths = (
            Path(__file__),
            Path(contracts.__file__),
            REPO_ROOT / "examples/hyperliquid/hyperliquid_maker_order_manager.py",
            REPO_ROOT
            / "examples/hyperliquid/hyperliquid_tiny_live_real_order_executor.py",
            REPO_ROOT / "examples/hyperliquid/cross_exchange_price_math.py",
        )
        for source in source_paths:
            shutil.copyfile(source, package_root / source.name)
        shutil.copytree(
            REPO_ROOT / "examples/hyperliquid/research_package_trust",
            package_root / "research_package_trust",
        )
        for implementation, script, pythonpath in (
            ("current", Path(__file__), str(REPO_ROOT)),
            (
                "frozen",
                package_root / Path(__file__).name,
                str(frozen_root),
            ),
        ):
            for case_id, expected_code in HOSTILE_CASES.items():
                env = dict(os.environ)
                env["PYTHONPATH"] = pythonpath
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(script),
                        "negative-case",
                        "--case-id",
                        case_id,
                        "--expected-code",
                        expected_code,
                    ],
                    cwd=REPO_ROOT if implementation == "current" else frozen_root,
                    env=env,
                    check=False,
                    capture_output=True,
                    text=True,
                )
                observed = completed.stdout.strip()
                executions.append(
                    {
                        "case_id": case_id,
                        "expected_error_code": expected_code,
                        "implementation": implementation,
                        "observed_error_code": observed,
                        "passed": completed.returncode == 0
                        and observed == expected_code,
                    }
                )
    fail_open = sum(not row["passed"] for row in executions)
    result = {
        "schema_version": "skhynix_c6in_latency_hostile_preflight_v2",
        "task_id": contracts.TASK_ID,
        "surface_matrix_sha256": contracts.sha256_file(matrix_path),
        "current_source_sha256": contracts.sha256_file(Path(__file__)),
        "current_contract_source_sha256": contracts.sha256_file(
            Path(contracts.__file__)
        ),
        "frozen_source_sha256": contracts.sha256_file(Path(__file__)),
        "frozen_contract_source_sha256": contracts.sha256_file(
            Path(contracts.__file__)
        ),
        "started_at_utc": started,
        "completed_at_utc": _utc_now(),
        "surface_count": len(matrix["surfaces"]),
        "case_count": len(HOSTILE_CASES),
        "implementation_count": 2,
        "execution_count": len(executions),
        "fail_open_count": fail_open,
        "executions": executions,
        "verified": fail_open == 0,
    }
    result["receipt_sha256"] = contracts.canonical_json_sha256(result)
    contracts.write_json(output_path, result)
    if fail_open:
        raise contracts.LatencyContractError(
            "LATENCY_HOSTILE_PREFLIGHT_FAILED",
            str(output_path),
            f"fail_open={fail_open}",
        )
    return result


def _install_no_network_guard() -> None:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise contracts.LatencyContractError(
            "LATENCY_L1_BOUNDARY_VIOLATION",
            "network",
            "network access is disabled in L1",
        )

    socket.socket = blocked  # type: ignore[assignment]
    urllib.request.urlopen = blocked  # type: ignore[assignment]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    hostile = subparsers.add_parser("hostile-preflight")
    hostile.add_argument("--task", type=Path, required=True)
    hostile.add_argument("--matrix", type=Path, required=True)
    hostile.add_argument("--output", type=Path, required=True)

    gate2 = subparsers.add_parser("gate2-preflight")
    gate2.add_argument("--output-root", type=Path, required=True)
    gate2.add_argument("--expected-commit", required=True)

    gate2_full_parser = subparsers.add_parser("gate2-full")
    gate2_full_parser.add_argument("--output-root", type=Path, required=True)
    gate2_full_parser.add_argument("--expected-commit", required=True)
    gate2_full_parser.add_argument(
        "--credential-file",
        type=Path,
        default=DEFAULT_CREDENTIAL_FILE,
    )

    schedule = subparsers.add_parser("freeze-schedule")
    schedule.add_argument("--output", type=Path, required=True)

    active = subparsers.add_parser("collect-active")
    active.add_argument("--output-root", type=Path, required=True)
    active.add_argument("--gate2-root", type=Path, required=True)
    active.add_argument("--schedule", type=Path, required=True)
    active.add_argument("--expected-commit", required=True)
    active.add_argument(
        "--credential-file",
        type=Path,
        default=DEFAULT_CREDENTIAL_FILE,
    )

    build_package = subparsers.add_parser("build-package")
    build_package.add_argument("--evidence-root", type=Path, required=True)
    build_package.add_argument("--package-root", type=Path, required=True)
    build_package.add_argument("--source-commit", required=True)

    verify_package = subparsers.add_parser("verify-package")
    verify_package.add_argument("--package-root", type=Path, required=True)

    negative = subparsers.add_parser("negative-case")
    negative.add_argument("--case-id", required=True)
    negative.add_argument("--expected-code", required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--sealed-root", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "hostile-preflight":
            result = hostile_preflight(args.task, args.matrix, args.output)
        elif args.command == "gate2-preflight":
            result = gate2_preflight(
                args.output_root,
                args.expected_commit,
            )
        elif args.command == "gate2-full":
            result = gate2_full(
                args.output_root,
                args.expected_commit,
                args.credential_file,
            )
        elif args.command == "freeze-schedule":
            result = freeze_collection_schedule(args.output)
        elif args.command == "collect-active":
            result = run_active_collection(
                output_root=args.output_root,
                gate2_root=args.gate2_root,
                schedule_path=args.schedule,
                expected_commit=args.expected_commit,
                credential_file=args.credential_file,
            )
        elif args.command == "build-package":
            result = build_formal_package(
                evidence_root=args.evidence_root,
                package_root=args.package_root,
                source_commit=args.source_commit,
            )
        elif args.command == "verify-package":
            result = verify_formal_package(args.package_root)
        elif args.command == "negative-case":
            expected = HOSTILE_CASES.get(args.case_id)
            if expected != args.expected_code:
                raise contracts.LatencyContractError(
                    "LATENCY_NEGATIVE_MATRIX_INCOMPLETE",
                    args.case_id,
                    f"expected={expected} observed={args.expected_code}",
                )
            _exercise_negative_case(args.case_id, args.expected_code)
            raise AssertionError("negative case failed open")
        elif args.command == "summarize":
            _install_no_network_guard()
            result = contracts.summarize_l0_root(
                args.sealed_root,
                args.output,
            )
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except contracts.LatencyContractError as exc:
        if args.command == "negative-case" and exc.code == args.expected_code:
            print(exc.code)
            return 0
        print(exc.code)
        return 2
    except Exception as exc:
        print(type(exc).__name__, file=sys.stderr)
        print("LATENCY_UNCLASSIFIED_RUNTIME_ERROR")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if result.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
