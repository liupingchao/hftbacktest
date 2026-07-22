#!/usr/bin/env python3
"""Build a source-pinned dynamic-spread seed from public market data only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.hyperliquid import cross_exchange_online_estimators as estimators
from examples.hyperliquid import hyperliquid_tiny_live_m2_public_watcher as watcher


TASK_ID = "0722T066"
SCHEMA_VERSION = "cross_exchange_public_multi_distance_dynamic_seed_v1"
SEED_SCHEMA_VERSION = "cross_exchange_dynamic_spread_seed_v1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "local_live_analysis"
    / "public_multi_distance_dynamic_seed_0722T066"
)
DEFAULT_DISTANCE_TICKS = (0.5, 2.5, 5.5, 10.5)
DEFAULT_INTERVAL_MS = 5_000
DEFAULT_WATCHER_SECONDS = 180.0
MIN_OBSERVATIONS_PER_SIDE = 3
MIN_DISTANCES_PER_SIDE = 2
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
HEX_COMMIT = re.compile(r"^[0-9a-f]{40}$")
SEED_CONTRACT_FIELDS = {
    "schema_version",
    "task_id",
    "source_commit",
    "runner_sha256",
    "symbol",
    "tick_size",
    "interval_ms",
    "distance_ticks",
    "source_event_rows_sha256",
    "quote_exposure_rows_sha256",
    "intensity_fit_rows_sha256",
    "source_event_row_count",
    "quote_exposure_row_count",
    "per_side",
    "seed_eligible",
    "inference_scope",
    "seed_contract_sha256",
}


class SeedError(ValueError):
    """Raised when public source evidence or a seed contract is invalid."""


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_hash(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _portable_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        raise SeedError("path_outside_project") from None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _validate_distances(values: Iterable[float]) -> tuple[float, ...]:
    distances = tuple(float(value) for value in values)
    if (
        len(distances) < MIN_DISTANCES_PER_SIDE
        or any(not math.isfinite(value) or value <= 0 for value in distances)
        or tuple(sorted(set(distances))) != distances
    ):
        raise SeedError("distance_grid_invalid")
    return distances


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_object_exists(object_id: str) -> bool:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{object_id}^{{commit}}"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def collect_public_event_rows(
    *,
    watcher_seconds: float,
    hyperliquid_l2book_fast: bool,
    event_source: Iterable[tuple[int, dict[str, Any]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if watcher_seconds <= 0:
        raise SeedError("watcher_seconds_must_be_positive")
    state = watcher.EventDrivenPublicState(max_order_size_btc=0.005)
    source = (
        event_source
        if event_source is not None
        else watcher.live_public_event_source(
            watcher_seconds=watcher_seconds,
            websocket_timeout=5.0,
            max_reconnects=3,
            yield_timeouts=True,
            hyperliquid_l2book_fast=hyperliquid_l2book_fast,
        )
    )
    started = time.monotonic()
    close_reason = "source_exhausted"
    for local_ts_ns, message in source:
        if event_source is not None and time.monotonic() - started > watcher_seconds:
            close_reason = "duration_elapsed"
            break
        if not isinstance(message, dict):
            continue
        channel = str(message.get("channel", "unknown"))
        if channel == "disconnect":
            data = message.get("data")
            reason = (
                str(data.get("reason") or "disconnect")
                if isinstance(data, dict)
                else "disconnect"
            )
            state.disconnect_events.append(
                {"local_ts_ns": local_ts_ns, "reason": reason}
            )
            continue
        state.observe(local_ts_ns, message)
    rows = state.online_estimator.event_rows()
    summary = {
        "watcher_seconds_requested": watcher_seconds,
        "watcher_seconds_elapsed": round(time.monotonic() - started, 6),
        "hyperliquid_l2book_fast": hyperliquid_l2book_fast,
        "close_reason": close_reason,
        "message_count_by_channel": state.message_count_by_channel,
        "book_event_count": state.book_event_count,
        "trade_event_count": state.trade_event_count,
        "disconnect_count": len(state.disconnect_events),
        "disconnect_events": state.disconnect_events,
        "source_event_row_count": len(rows),
        "public_market_data_only": True,
    }
    return rows, summary


def _latest_book_before(
    book_rows: list[dict[str, Any]],
    start_ms: int,
) -> dict[str, Any] | None:
    latest = None
    for row in book_rows:
        if int(row["event_time_ms"]) > start_ms:
            break
        latest = row
    return latest


def build_counterfactual_exposures(
    *,
    event_rows: list[dict[str, Any]],
    interval_ms: int,
    distance_ticks: tuple[float, ...],
    tick_size: float = 1.0,
) -> estimators.EventTimeOnlineEstimator:
    if interval_ms <= 0:
        raise SeedError("interval_ms_must_be_positive")
    estimator = estimators.replay_estimator_rows(
        event_rows=event_rows,
        bucket_ms=estimators.DEFAULT_BUCKET_MS,
        tick_size=tick_size,
    )
    books = sorted(
        (
            row
            for row in event_rows
            if str(row.get("event_kind")) == "book"
        ),
        key=lambda row: int(row["event_time_ms"]),
    )
    if not books:
        raise SeedError("public_book_rows_missing")
    last_event_ms = max(int(row["event_time_ms"]) for row in event_rows)
    first_book_ms = int(books[0]["event_time_ms"])
    start_ms = (
        (first_book_ms + interval_ms - 1) // interval_ms
    ) * interval_ms
    interval_index = 0
    while start_ms + interval_ms <= last_event_ms:
        book = _latest_book_before(books, start_ms)
        if book is not None:
            bid = float(book["bid_px"])
            ask = float(book["ask_px"])
            mid = (bid + ask) / 2.0
            for side in ("buy", "sell"):
                for distance in distance_ticks:
                    quote = (
                        mid - distance * tick_size
                        if side == "buy"
                        else mid + distance * tick_size
                    )
                    if quote <= 0:
                        raise SeedError("counterfactual_quote_nonpositive")
                    estimator.observe_quote_exposure_from_public_flow(
                        exposure_id=(
                            f"{TASK_ID}:{interval_index}:{side}:"
                            f"{distance:g}"
                        ),
                        side=side,
                        quote_px=quote,
                        reference_mid_px=mid,
                        start_exchange_time_ms=start_ms,
                        end_exchange_time_ms=start_ms + interval_ms,
                        resting_confirmed=False,
                        source=(
                            "public_multi_distance_counterfactual_"
                            "non_resting_seed"
                        ),
                    )
            interval_index += 1
        start_ms += interval_ms
    if interval_index == 0:
        raise SeedError("no_complete_public_exposure_interval")
    return estimator


def _seed_contract(
    *,
    source_commit: str,
    runner_sha256: str,
    interval_ms: int,
    distance_ticks: tuple[float, ...],
    event_path: Path,
    exposure_path: Path,
    intensity_path: Path,
    event_rows: list[dict[str, Any]],
    exposure_rows: list[dict[str, Any]],
    intensity_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fit_by_side = {str(row["side"]): row for row in intensity_rows}
    per_side: dict[str, Any] = {}
    for side in ("buy", "sell"):
        side_rows = [
            row for row in exposure_rows if str(row["side"]) == side
        ]
        distances = sorted(
            {float(row["distance_ticks"]) for row in side_rows}
        )
        fit = fit_by_side.get(side) or {}
        per_side[side] = {
            "observation_count": len(side_rows),
            "distance_variation_count": len(distances),
            "distance_ticks": distances,
            "fit_status": str(fit.get("status") or ""),
            "fit_reason": str(fit.get("reason") or ""),
            "A": fit.get("A", ""),
            "k": fit.get("k", ""),
            "fit_rmse": fit.get("fit_rmse", ""),
            "confidence": fit.get("confidence", ""),
            "eligible": (
                len(side_rows) >= MIN_OBSERVATIONS_PER_SIDE
                and len(distances) >= MIN_DISTANCES_PER_SIDE
                and fit.get("status") == "pass"
            ),
        }
    payload = {
        "schema_version": SEED_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "source_commit": source_commit,
        "runner_sha256": runner_sha256,
        "symbol": "BTC",
        "tick_size": 1.0,
        "interval_ms": interval_ms,
        "distance_ticks": list(distance_ticks),
        "source_event_rows_sha256": _file_hash(event_path),
        "quote_exposure_rows_sha256": _file_hash(exposure_path),
        "intensity_fit_rows_sha256": _file_hash(intensity_path),
        "source_event_row_count": len(event_rows),
        "quote_exposure_row_count": len(exposure_rows),
        "per_side": per_side,
        "seed_eligible": all(
            per_side[side]["eligible"] for side in ("buy", "sell")
        ),
        "inference_scope": (
            "public_counterfactual_non_resting_intensity_seed_"
            "not_fill_or_live_quote_proof"
        ),
    }
    payload["seed_contract_sha256"] = _sha256_bytes(_canonical(payload))
    return payload


def build_artifacts(
    *,
    event_rows: list[dict[str, Any]],
    output_dir: Path,
    source_commit: str,
    interval_ms: int = DEFAULT_INTERVAL_MS,
    distance_ticks: Iterable[float] = DEFAULT_DISTANCE_TICKS,
    collection_summary: dict[str, Any] | None = None,
    runner_path: Path | None = None,
) -> dict[str, Any]:
    if not HEX_COMMIT.fullmatch(source_commit):
        raise SeedError("source_commit_invalid")
    output_dir = output_dir.resolve()
    _portable_path(output_dir)
    distances = _validate_distances(distance_ticks)
    runner_path = (
        runner_path
        or Path(__file__).resolve()
    )
    runner_sha256 = _file_hash(runner_path)
    estimator = build_counterfactual_exposures(
        event_rows=event_rows,
        interval_ms=interval_ms,
        distance_ticks=distances,
    )
    exposure_rows = estimator.quote_exposure_rows()
    intensity_rows = estimator.intensity_rows()
    snapshot = estimator.snapshot()

    event_path = output_dir / "online_estimator_event_rows.csv"
    exposure_path = output_dir / "quote_exposure_intervals.csv"
    intensity_path = output_dir / "online_intensity_fit.csv"
    _write_csv(
        event_path,
        event_rows,
        estimators.estimator_event_fieldnames(),
    )
    _write_csv(
        output_dir / "online_estimator_bucket_matrix.csv",
        estimator.bucket_rows(),
        estimators.estimator_bucket_fieldnames(),
    )
    _write_csv(
        exposure_path,
        exposure_rows,
        estimators.quote_exposure_fieldnames(),
    )
    _write_csv(
        intensity_path,
        intensity_rows,
        estimators.intensity_fit_fieldnames(),
    )
    _write_json(output_dir / "online_estimator_core_snapshot.json", snapshot)
    contract = _seed_contract(
        source_commit=source_commit,
        runner_sha256=runner_sha256,
        interval_ms=interval_ms,
        distance_ticks=distances,
        event_path=event_path,
        exposure_path=exposure_path,
        intensity_path=intensity_path,
        event_rows=event_rows,
        exposure_rows=exposure_rows,
        intensity_rows=intensity_rows,
    )
    _write_json(output_dir / "dynamic_spread_seed_contract.json", contract)
    boundary = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "public_market_data_only": True,
        "counterfactual_exposure_only": True,
        "resting_confirmed": False,
        "credentials_read": False,
        "live_client_initialized": False,
        "private_endpoint_called": False,
        "account_endpoint_called": False,
        "order_endpoint_called": False,
        "cancel_endpoint_called": False,
        "service_or_orchestrator_started": False,
        "real_orders_allowed": False,
        "actual_quote_behavior_changed": False,
        "live_authorization_consumed": False,
    }
    recommendation = (
        "accept_source_pinned_public_dynamic_seed_for_later_explicit_live"
        if contract["seed_eligible"]
        else "blocked_public_dynamic_seed_threshold_or_fit_not_met"
    )
    manifest = {
        "task_id": TASK_ID,
        "schema_version": SCHEMA_VERSION,
        "source_commit": source_commit,
        "runner_path": _portable_path(runner_path),
        "runner_sha256": runner_sha256,
        "config": {
            "interval_ms": interval_ms,
            "distance_ticks": list(distances),
            "minimum_observations_per_side": MIN_OBSERVATIONS_PER_SIDE,
            "minimum_distances_per_side": MIN_DISTANCES_PER_SIDE,
        },
        "collection_summary": collection_summary or {
            "mode": "offline_rebuild_from_committed_event_rows"
        },
        "seed_contract_sha256": contract["seed_contract_sha256"],
        "seed_eligible": contract["seed_eligible"],
        "per_side": contract["per_side"],
        "final_recommendation": recommendation,
        "boundary": boundary,
        "output_files": {
            "event_rows": _portable_path(event_path),
            "bucket_matrix": _portable_path(
                output_dir / "online_estimator_bucket_matrix.csv"
            ),
            "quote_exposures": _portable_path(exposure_path),
            "intensity_fit": _portable_path(intensity_path),
            "estimator_snapshot": _portable_path(
                output_dir / "online_estimator_core_snapshot.json"
            ),
            "seed_contract": _portable_path(
                output_dir / "dynamic_spread_seed_contract.json"
            ),
            "boundary_manifest": _portable_path(
                output_dir / "boundary_manifest.json"
            ),
        },
    }
    _write_json(output_dir / "boundary_manifest.json", boundary)
    _write_json(output_dir / "public_dynamic_seed_manifest.json", manifest)
    (output_dir / "recommendation.md").write_text(
        "\n".join(
            [
                "# Public Multi-Distance Dynamic Seed",
                "",
                f"`{recommendation}`",
                "",
                f"- Seed eligible: `{contract['seed_eligible']}`.",
                "- Exposure is counterfactual and non-resting.",
                "- No live quote behavior changed and no order endpoint was called.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "manifest": manifest,
        "contract": contract,
        "boundary": boundary,
    }


def load_seed_into_estimator(
    *,
    estimator: estimators.EventTimeOnlineEstimator,
    contract_path: Path,
    exposure_path: Path,
    expected_seed_contract_sha256: str,
) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or set(contract) != SEED_CONTRACT_FIELDS:
        raise SeedError("seed_contract_fields_mismatch")
    if contract.get("schema_version") != SEED_SCHEMA_VERSION:
        raise SeedError("seed_contract_schema_mismatch")
    if contract.get("task_id") != TASK_ID:
        raise SeedError("seed_contract_task_id_mismatch")
    if not HEX_SHA256.fullmatch(expected_seed_contract_sha256):
        raise SeedError("expected_seed_contract_sha256_invalid")
    claimed_hash = str(contract.get("seed_contract_sha256") or "")
    payload = dict(contract)
    payload.pop("seed_contract_sha256", None)
    computed_hash = _sha256_bytes(_canonical(payload))
    if claimed_hash != computed_hash:
        raise SeedError("seed_contract_self_hash_mismatch")
    if claimed_hash != expected_seed_contract_sha256:
        raise SeedError("seed_contract_expected_hash_mismatch")
    if _file_hash(exposure_path) != contract["quote_exposure_rows_sha256"]:
        raise SeedError("seed_exposure_file_hash_mismatch")
    if contract.get("seed_eligible") is not True:
        raise SeedError("seed_contract_not_eligible")
    rows = _read_csv(exposure_path)
    if len(rows) != int(contract["quote_exposure_row_count"]):
        raise SeedError("seed_exposure_row_count_mismatch")
    before = len(estimator.quote_exposure_rows())
    for row in rows:
        if (
            str(row.get("resting_confirmed")).lower()
            not in {"false", "0"}
            or row.get("source")
            != "public_multi_distance_counterfactual_non_resting_seed"
        ):
            raise SeedError("seed_exposure_boundary_mismatch")
        estimator.observe_quote_exposure(
            exposure_id=f"seed:{row['exposure_id']}",
            side=str(row["side"]),
            quote_px=float(row["quote_px"]),
            reference_mid_px=float(row["reference_mid_px"]),
            start_exchange_time_ms=int(row["start_exchange_time_ms"]),
            end_exchange_time_ms=int(row["end_exchange_time_ms"]),
            arrival_count=int(row["arrival_count"]),
            arrival_volume_btc=float(row["arrival_volume_btc"]),
            pre_trade_side_depth_btc=(
                None
                if row.get("pre_trade_side_depth_btc") in {"", None}
                else float(row["pre_trade_side_depth_btc"])
            ),
            max_sweep_depth_penetration=(
                None
                if row.get("max_sweep_depth_penetration") in {"", None}
                else float(row["max_sweep_depth_penetration"])
            ),
            arrival_evidence_source=str(row["arrival_evidence_source"]),
            resting_confirmed=False,
            source=str(row["source"]),
        )
    fits = {
        side: estimator.fit_intensity(side)
        for side in ("buy", "sell")
    }
    if any(fits[side]["status"] != "pass" for side in fits):
        raise SeedError("loaded_seed_intensity_fit_not_pass")
    return {
        "loaded_row_count": len(estimator.quote_exposure_rows()) - before,
        "seed_contract_sha256": claimed_hash,
        "fits": fits,
        "actual_quote_behavior_changed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--watcher-seconds",
        type=float,
        default=DEFAULT_WATCHER_SECONDS,
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
    )
    parser.add_argument(
        "--distance-ticks",
        type=float,
        nargs="+",
        default=list(DEFAULT_DISTANCE_TICKS),
    )
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--hyperliquid-l2book-fast", action="store_true")
    parser.add_argument("--rebuild-from-event-rows", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    current_head = _git_head()
    if (
        args.rebuild_from_event_rows is None
        and args.source_commit != current_head
    ):
        raise SeedError("source_commit_not_current_head")
    if (
        args.rebuild_from_event_rows is not None
        and not _git_object_exists(args.source_commit)
    ):
        raise SeedError("source_commit_not_found")
    if args.rebuild_from_event_rows is not None:
        event_rows = _read_csv(args.rebuild_from_event_rows)
        collection_summary = {
            "mode": "offline_rebuild_from_committed_event_rows",
            "source_event_rows": _portable_path(args.rebuild_from_event_rows),
        }
    else:
        event_rows, collection_summary = collect_public_event_rows(
            watcher_seconds=args.watcher_seconds,
            hyperliquid_l2book_fast=args.hyperliquid_l2book_fast,
        )
    result = build_artifacts(
        event_rows=event_rows,
        output_dir=args.output_dir,
        source_commit=args.source_commit,
        interval_ms=args.interval_ms,
        distance_ticks=args.distance_ticks,
        collection_summary=collection_summary,
    )
    print(json.dumps(result["manifest"], indent=2, sort_keys=True))
    return 0 if result["contract"]["seed_eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
