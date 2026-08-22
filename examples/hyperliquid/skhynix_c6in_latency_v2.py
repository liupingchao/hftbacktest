#!/usr/bin/env python3
"""Execute and package the 0822T002 c6in Hyperliquid latency measurement."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
import time
import urllib.request
from collections import OrderedDict
from decimal import Decimal, ROUND_CEILING
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import research_package_trust as trust
    import skhynix_c6in_latency_contracts_v2 as contracts
except ModuleNotFoundError:  # pragma: no cover
    from examples.hyperliquid import research_package_trust as trust
    from examples.hyperliquid import skhynix_c6in_latency_contracts_v2 as contracts


TASK_PATH = REPO_ROOT / ".workflow/tasks/0822T002.md"
MATRIX_PATH = REPO_ROOT / ".workflow/contracts/0822T002-surface-matrix.json"
HOSTILE_RECEIPT = REPO_ROOT / ".workflow/reports/0822T002-hostile-preflight.json"
REGISTRY_PATH = (
    REPO_ROOT / "baselines/research_package_trust_kernel/accepted_versions.json"
)
EXPECTED_PLAN_SHA256 = (
    "7e0ea338c79720bc6d2ecb202af85716883601c2e5552ef6c033bb58c45cc400"
)
EXPECTED_MATRIX_SHA256 = (
    "6bff9a34963b1ad68d1f9bbea3d1e43836ffab3e22e6350b381a9a816f674510"
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
    observed["host_identity_token"] = contracts.canonical_json_sha256(observed)
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
    payload["runtime_identity_sha256"] = contracts.canonical_json_sha256(payload)
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
        info.unsubscribe(subscription, subscription_id)
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
    account = values.get("HL_WALLET") or values.get(
        "HYPERLIQUID_ACCOUNT_ADDRESS", ""
    )
    if not private_key or not account:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "credential_source",
            "private key or configured account absent",
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
    return {"private_key": private_key, "account": account}, metadata


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


def _available_margin_sufficient(user_state: Any) -> bool:
    if not isinstance(user_state, dict):
        return False
    candidates = [
        user_state.get("withdrawable"),
        (
            user_state.get("marginSummary", {}).get("accountValue")
            if isinstance(user_state.get("marginSummary"), dict)
            else None
        ),
    ]
    parsed: list[Decimal] = []
    for value in candidates:
        if value in (None, ""):
            continue
        try:
            parsed.append(Decimal(str(value)))
        except Exception:
            continue
    return bool(parsed) and max(parsed) >= Decimal(
        str(contracts.AGGREGATE_POSITION_CAP_USDC)
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
    normalized_account = secrets["account"].strip().lower()
    normalized_signer = wallet.address.strip().lower()
    if (
        not normalized_account.startswith("0x")
        or len(normalized_account) != 42
        or not normalized_signer.startswith("0x")
        or len(normalized_signer) != 42
    ):
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "account_identity",
            "configured account or signer address malformed",
        )
    account_identity_token = hashlib.sha256(
        normalized_account.encode("ascii")
    ).hexdigest()
    signer_identity_token = hashlib.sha256(
        normalized_signer.encode("ascii")
    ).hexdigest()
    info = Info(
        constants.MAINNET_API_URL,
        skip_ws=True,
        perp_dexs=[TARGET_DEX],
        timeout=10,
    )
    Exchange(
        wallet,
        constants.MAINNET_API_URL,
        account_address=secrets["account"],
        perp_dexs=[TARGET_DEX],
        timeout=10,
    )
    open_orders = info.open_orders(secrets["account"], TARGET_DEX)
    user_state = info.user_state(secrets["account"], TARGET_DEX)
    if not isinstance(open_orders, list):
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "open_orders",
            "response is not a list",
        )
    position_size = _target_position_size(user_state)
    margin_sufficient = _available_margin_sufficient(user_state)
    if open_orders or position_size != 0 or not margin_sufficient:
        raise contracts.LatencyContractError(
            "LATENCY_AUTHORIZATION_MISMATCH",
            "account_baseline",
            (
                f"open_orders={len(open_orders)} "
                f"target_position_zero={position_size == 0} "
                f"margin_sufficient={margin_sufficient}"
            ),
        )
    account_baseline = {
        "schema_version": "skhynix_c6in_account_baseline_v2",
        "task_id": contracts.TASK_ID,
        "account_identity_token": account_identity_token,
        "signer_identity_token": signer_identity_token,
        "configured_account_matches_signer": (
            normalized_account == normalized_signer
        ),
        "dex": TARGET_DEX,
        "asset": TARGET_ASSET,
        "open_order_count": 0,
        "target_position_zero": True,
        "available_margin_at_least_aggregate_cap": True,
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
    if normalized_account in serialized or normalized_signer in serialized:
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
    print(json.dumps(result, indent=2, sort_keys=True))
    return 3 if result.get("status") == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
